"""Kolibri chat, reasoning and native JSON tools without model weights or a GPU."""
import io
import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest
import urllib.request

from serve import frontend as F
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, model_card_sampling, serve

TOOLS = [{"name": "weather", "description": "Weather", "parameters": {"type": "object",
          "properties": {"city": {"type": "string"}}}}]
CALL = '<tool_call>\n{"name":"weather","arguments":{"city":"Köln"}}\n</tool_call>'
# Small template fixture with Kolibri's message/tool syntax and effort precedence.
TEMPLATE = '''{% set off = reasoning_effort == 'none' if reasoning_effort is defined else enable_thinking is defined and not enable_thinking %}
<|im_start|>system\n{{ reasoning_effort|default('high') }}
{% if tools %}<tool_call>{"name": NAME, "arguments": OBJECT}</tool_call>
{% for tool in tools %}{{ tool|tojson }}{% endfor %}{% endif %}<|im_end|>
{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}
{{ message.reasoning_content|default('') }}
{% for call in message.tool_calls|default([]) %}<tool_call>{{ call.function|tojson }}</tool_call>{% endfor %}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% if off %}<think>\n\n</think>\n\n{% endif %}{% endif %}'''


def parse(text, *, thinking=False, tools=TOOLS, step=1, reason=None):
    p = F.KolibriOutputParser(thinking=thinking, tools=tools, stream_tools=True)
    events = []
    for i in range(0, len(text), step):
        events += p.feed(text[i:i + step])
    return events + p.finish(reason)


class KolibriParser(unittest.TestCase):
    def test_fragmented_reasoning_and_multiple_calls(self):
        text = "<think>\nÜberlegen.\n</think>\n\n" + CALL + "\n" + CALL
        for step in (1, 7, len(text)):
            ev = parse(text, thinking=True, step=step)
            self.assertEqual("".join(x.text for x in ev if x.kind == "reasoning"), "\nÜberlegen.\n")
            self.assertEqual([x.call.arguments for x in ev if x.kind == "tool_call"], [{"city": "Köln"}] * 2)

    def test_json_strings_can_contain_envelope_markers(self):
        args = {"city": 'quote " \\ </tool_call> { } <think> </think> test'}
        text = "<tool_call>" + json.dumps({"name": "weather", "arguments": args}) + "</tool_call>"
        for step in (1, 11, len(text)):
            calls = [x.call for x in parse(text, step=step) if x.kind == "tool_call"]
            self.assertEqual([c.arguments for c in calls], [args])
        ev = parse("<think>\n" + text + "</think>\n", thinking=True)
        self.assertEqual([x.call.arguments for x in ev if x.kind == "tool_call"], [args])

    def test_invalid_unknown_and_unfinished_calls_are_text(self):
        for text in (CALL.replace("weather", "unknown"), '<tool_call>{bad}</tool_call>', CALL[:-5],
                     '<tool_call><function=weather></function></tool_call>'):
            for step in (1, len(text)):
                ev = parse(text, step=step)
                self.assertFalse(any(x.kind == "tool_call" for x in ev))
                self.assertEqual("".join(x.text for x in ev), text)
        self.assertFalse(any(x.kind == "tool_call" for x in parse(CALL, tools=None)))

    def test_quoted_calls_and_think_tags_stay_text(self):
        text = "```json\n" + CALL + "\n```\nLiteral <think> marker."
        ev = parse(text)
        self.assertFalse(any(x.kind == "tool_call" for x in ev))
        self.assertEqual("".join(x.text for x in ev), text)

    def test_incomplete_reasoning_call_is_not_executed_on_cancel(self):
        ev = parse("<think>\n" + CALL, thinking=True, reason="cancel")
        self.assertFalse(any(x.kind == "tool_call" for x in ev))

    def test_forced_prefix_and_existing_invalid_choice_contract(self):
        name = 'a"b'
        prefix = F.kolibri_forced_call(F.forced_call({"type": "function", "function": {"name": name}}, [{"name": name}]))
        # A trailing standalone space made real Kolibri continue with scalar 7}} instead of the arguments object.
        self.assertEqual(prefix, '<tool_call>\n{"name": "a\\"b", "arguments":')
        calls = [x.call for x in parse(prefix + ' {"city":"Köln"}}</tool_call>', tools=[{"name": name}])
                 if x.kind == "tool_call"]
        self.assertEqual([c.arguments for c in calls], [{"city": "Köln"}])
        self.assertEqual(F.kolibri_forced_call(F.forced_call("required", TOOLS + [{"name": "other"}])),
                         '<tool_call>\n{"name": "')
        self.assertIsNone(F.kolibri_forced_call(F.forced_call("none", TOOLS)))
        self.assertIsNone(F.kolibri_forced_call(F.forced_call("required", [])))


class KolibriEffort(unittest.TestCase):
    def setUp(self):
        F.use_family("kolibri1")
        self.addCleanup(F.use_family, None)

    def test_effort_overrides_boolean_in_both_orders(self):
        for ctk in ({"enable_thinking": False, "reasoning_effort": "low"},
                    {"reasoning_effort": "low", "enable_thinking": False}):
            _, _, kw = F.openai_to_messages({"messages": [], "chat_template_kwargs": ctk})
            self.assertEqual(kw, {"reasoning_effort": "low"})
        _, _, kw = F.openai_to_messages({"messages": [], "reasoning_effort": "medium",
                                         "chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(kw, {"reasoning_effort": "medium"})

    def test_none_minimal_and_anthropic_effort(self):
        self.assertEqual(F.effort_kwargs("none"), {"enable_thinking": False})
        self.assertEqual(F.effort_kwargs("minimal"), {"reasoning_effort": "low"})
        self.assertEqual(F.effort_kwargs("xhigh"), {"reasoning_effort": "high"})
        _, _, kw = F.anthropic_to_messages({"messages": [], "thinking": {"type": "disabled"},
                                           "output_config": {"effort": "medium"}})
        self.assertEqual(kw, {"reasoning_effort": "medium"})


class KolibriEngineProtocol(unittest.TestCase):
    def test_model_top_k_reaches_only_a_capable_engine(self):
        for info, expected in (({}, 64), ({"family": "qwen4exp", "sampling_top_k": 128}, 64),
                               ({"family": "kolibri1", "sampling_top_k": 128}, 128)):
            engine = StrataEngine.__new__(StrataEngine)
            engine.info = info
            engine.proc = SimpleNamespace(stdin=io.StringIO(), poll=lambda: None)
            engine.lines, engine.can_stop, engine.silence_s = queue.Queue(), False, 0
            engine.lines.put("T 72")
            engine.lines.put("DONE 1 1 0 1 stop 0 0 0")
            self.assertEqual(list(engine.generate([1], 1, model_card_sampling("kolibri1", True),
                                                   threading.Event())), [72])
            self.assertIn(f" top_k={expected} ", engine.proc.stdin.getvalue())


PROTOCOL_TEST = Path(os.environ.get("STRATA_KOLIBRI_PROTOCOL_TEST",
                                  Path(__file__).resolve().parents[1] / "build/kolibri-protocol-test"))


class KolibriPrefixTemplate(unittest.TestCase):
    def test_history_stays_an_exact_extension_only_when_opted_in(self):
        # The model template drops old thinking unless preserve_thinking is on.
        source = '''{%- for m in messages -%}
{{- '<|im_start|>' + m.role + '\\n' -}}
{%- if m.role == 'assistant' and preserve_thinking|default(false) -%}
{{- '<think>\\n' + m.reasoning_content|default('') + '\\n</think>\\n\\n' -}}
{%- endif -%}
{{- m.content + '<|im_end|>\\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}
{%- if enable_thinking is defined and not enable_thinking -%}{{- '<think>\\n\\n</think>\\n\\n' -}}{%- endif -%}
{%- endif -%}'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "template.jinja"
            path.write_text(source)
            svc = Service.__new__(Service)
            svc.effort_end = False
            svc.template = F.ChatTemplate(path, "kolibri1")
            user = {"role": "user", "content": "Question"}
            for off, reasoning in ((True, ""), (False, "Thinking.")):
                kwargs = {"enable_thinking": not off}
                initial = svc.template.render([user], **kwargs)
                answer = "Answer" if off else "<think>\nThinking.\n</think>\n\nAnswer"
                history = [user, {"role": "assistant", "content": "Answer", "reasoning_content": reasoning},
                           {"role": "user", "content": "Next"}]
                expected = initial + answer + "<|im_end|>\n"
                for family, info in (("kolibri1", {}), ("qwen4exp", {"prefix_cache": 1}),
                                     ("kolibri1", {"prefix_cache": 1})):
                    svc.family, svc.engine = family, SimpleNamespace(info=info)
                    actual = svc.render_prompt(history, None, kwargs)
                    if family == "kolibri1" and info.get("prefix_cache") == 1:
                        self.assertTrue(actual.startswith(expected))
                        explicit = svc.render_prompt(history, None, {**kwargs, "preserve_thinking": False})
                        self.assertEqual(explicit, svc.template.render(history, **kwargs))
                    else:
                        self.assertEqual(actual, svc.template.render(history, **kwargs))
                self.assertEqual(kwargs, {"enable_thinking": not off})


@unittest.skipUnless(PROTOCOL_TEST.is_file(), "build the kolibri-protocol-test target for CPU protocol checks")
class KolibriPrefixProtocol(unittest.TestCase):
    def setUp(self):
        self.start("--prefix-cache")

    def start(self, *args):
        if hasattr(self, "proc"):
            self.close(self.proc)
        proc = subprocess.Popen([str(PROTOCOL_TEST), *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.proc, self.lines = proc, queue.Queue()
        self.addCleanup(self.close, proc)
        lines = self.lines

        def read():
            for line in proc.stdout:
                lines.put(line.rstrip())
            lines.put(None)

        threading.Thread(target=read, daemon=True).start()
        self.info = self.take()
        self.assertTrue(self.info.startswith("INFO family=kolibri1 "))
        self.assertEqual(self.take(), "READY 256 stop")

    @staticmethod
    def close(proc):
        if proc.poll() is None:
            proc.stdin.close()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        proc.stdout.close()

    def take(self):
        line = self.lines.get(timeout=5)
        self.assertIsNotNone(line, "protocol fixture exited unexpectedly")
        return line

    def send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    @staticmethod
    def expected(ids, count):
        history, result = list(ids), []
        for _ in range(count):
            value = 7
            for token in history:
                value = (value * 33 + token + 1) & 0xffffffff
            result.append(1 + value % 29)
            history.append(result[-1])
        return result

    def request(self, ids, count=3, options="", stop_at=None):
        self.send(f"GEN {count} {options} " + ",".join(map(str, ids)))
        tokens, progress, resumed, stopped = [], [], 0, False
        while True:
            line = self.take()
            self.assertFalse(line.startswith("ERR"), line)
            if line.startswith("RESUME "):
                resumed = int(line.split()[1])
            elif line.startswith("PP "):
                progress.append(line)
            elif line.startswith("T "):
                tokens.append(int(line.split()[1]))
            elif line.startswith("DONE "):
                f = line.split()
                self.assertEqual(len(f), 16)
                self.assertEqual((int(f[1]), int(f[2]), int(f[8])), (len(tokens), len(ids), resumed))
                read = int(f[14])
                self.assertLessEqual(resumed + read, len(ids))
                if f[5] != "cancel":
                    self.assertEqual(resumed + read, len(ids))
                return dict(tokens=tokens, reused=resumed, read=read, finish=f[5], progress=progress)
            if stop_at and not stopped and line.startswith(stop_at):
                self.send("STOP")
                stopped = True

    def test_extension_keeps_only_forwarded_output(self):
        first = list(range(12))
        a = self.request(first, 4)
        self.assertEqual(a["tokens"], self.expected(first, 4))
        second = first + a["tokens"] + [7, 8]
        b = self.request(second)
        self.assertEqual((b["reused"], b["read"]), (len(first) + 3, 3))
        self.assertEqual(b["tokens"], self.expected(second, 3))
        third = second + b["tokens"] + [9]
        c = self.request(third)
        self.assertEqual(c["reused"], len(second) + 2)
        self.assertEqual(c["tokens"], self.expected(third, 3))

    def test_fully_cached_prompt_reuses_logits_without_caching_final_token(self):
        ids = [5, 6, 7]
        a = self.request(ids, 1)
        b = self.request(ids, 1)
        self.assertEqual((b["reused"], b["read"], b["progress"]), (3, 0, []))
        self.assertEqual(a["tokens"], b["tokens"])
        c = self.request(ids + a["tokens"], 1)
        self.assertEqual((c["reused"], c["read"]), (3, 1))

    def test_repeating_a_prompt_after_decode_does_not_rewind(self):
        ids = list(range(16))
        a = self.request(ids)
        b = self.request(ids)
        self.assertEqual(b["reused"], 0)
        self.assertEqual(a["tokens"], b["tokens"])

    def test_edited_shorter_and_diverging_prompts_restart(self):
        original = list(range(16))
        for changed in (original[:-1], [2] + original[1:], original[:-1] + [20]):
            a = self.request(original)
            b = self.request(changed)
            self.assertEqual(b["reused"], 0)
            self.assertEqual(b["tokens"], self.expected(changed, 3))
            # Sharing part of the answer cannot rewind local KV either.
            c = self.request(original + a["tokens"][:1] + [30])
            self.assertEqual(c["reused"], 0)

    def test_checkpoint_disabled_can_reuse_but_does_not_retain(self):
        self.request([1, 2, 3], 1)
        a = self.request([1, 2, 3, 4], 1, "ckpt=0")
        self.assertEqual(a["reused"], 3)
        b = self.request([1, 2, 3, 4], 1)
        self.assertEqual(b["reused"], 0)

    def test_invalid_request_leaves_valid_cache_and_context_is_not_bypassed(self):
        self.request([1, 2, 3], 1)
        for line in ("GEN 1 1,2,32", "GEN 254 1,2,3"):
            self.send(line)
            self.assertTrue(self.take().startswith("ERR "))
        a = self.request([1, 2, 3, 4], 1)
        self.assertEqual(a["reused"], 3)
        full = [1] * 255
        self.request(full, 1)
        b = self.request(full, 1)
        self.assertEqual((b["reused"], b["read"]), (255, 0))

    def test_cancelled_prefill_retains_only_completed_chunks(self):
        ids = [i % 30 for i in range(240)]
        a = self.request(ids, 2, stop_at="PP ")
        self.assertEqual((a["finish"], a["tokens"]), ("cancel", []))
        self.assertGreater(a["read"], 0)
        self.assertLess(a["read"], len(ids))
        b = self.request(ids + [7], 2)
        self.assertEqual(b["reused"], a["read"])
        self.assertEqual(b["tokens"], self.expected(ids + [7], 2))

    def test_cancelled_extension_counts_reused_and_new_tokens_separately(self):
        base = list(range(16))
        self.request(base, 1)
        ids = base + [i % 30 for i in range(224)]
        a = self.request(ids, 2, stop_at="PP ")
        self.assertEqual((a["finish"], a["reused"]), ("cancel", len(base)))
        self.assertGreater(a["read"], 0)
        self.assertLess(a["reused"] + a["read"], len(ids))
        b = self.request(ids + [7], 2)
        self.assertEqual(b["reused"], a["reused"] + a["read"])
        self.assertEqual(b["tokens"], self.expected(ids + [7], 2))

    def test_cancelled_decode_commits_inflight_token_only_after_forward(self):
        ids = list(range(12))
        a = self.request(ids, 200, stop_at="T ")
        self.assertEqual(a["finish"], "cancel")
        self.assertLess(len(a["tokens"]), 200)
        following = ids + a["tokens"] + [7, 8]
        b = self.request(following, 2)
        self.assertIn(b["reused"], (len(ids) + len(a["tokens"]) - 1, len(ids) + len(a["tokens"])))
        self.assertEqual(b["tokens"], self.expected(following, 2))

    def test_eos_is_not_marked_as_forwarded(self):
        ids = [1, 2, 3]
        a = self.request(ids, 5, "logit_bias=31:100")
        self.assertEqual((a["tokens"], a["finish"]), ([31], "stop"))
        b = self.request(ids + [31, 4], 1)
        self.assertEqual((b["reused"], b["read"]), (3, 2))

    def test_fully_cached_logits_are_not_changed_by_cpu_penalties(self):
        ids = list(range(1, 30))
        a = self.request(ids, 1)
        b = self.request(ids, 1, "penalty_last_n=256 penalty_present=100")
        self.assertEqual(b["tokens"], [0])
        c = self.request(ids, 1)
        self.assertEqual(c["tokens"], a["tokens"])
        self.assertEqual((b["reused"], c["reused"]), (29, 29))

    def test_flag_off_retains_existing_request_behavior(self):
        self.start()
        self.assertNotIn("prefix_cache", self.info)
        for _ in range(2):
            a = self.request([1, 2, 3], 1)
            self.assertEqual((a["reused"], a["read"]), (0, 3))
            self.assertEqual(a["tokens"], self.expected([1, 2, 3], 1))


class RecordingEngine(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.sampling = dict(sampling or {})
        prompt = self.tok.decode(ids)
        if prompt.endswith('"arguments":'):
            text = ' {"city":"Köln"}}</tool_call>'
        elif prompt.endswith('{"name": "'):
            text = 'weather", "arguments": {"city":"Köln"}}</tool_call>'
        else:
            text = "<think>\nÜberlegen.\n</think>\n\n" + CALL
        self.script = self.tok.encode(text) + self.tok.encode("<|im_end|>", parse_special=True)
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class KolibriChatAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name) / "template.jinja"
        path.write_text(TEMPLATE)
        cls.tok = ByteTokenizer()
        cls.tok.special_ids = {"tokenizer.ggml.eos_token_id": cls.tok.encode("<|im_end|>", True)[0]}
        cls.engine = RecordingEngine(cls.tok, "ok")
        cls.svc = Service(cls.engine, cls.tok, F.ChatTemplate(path, "kolibri1"))
        cls.svc.model_arch = "kolibri1"
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.tmp.cleanup()

    def setUp(self):
        self.svc.set_family("kolibri1")
        self.addCleanup(F.use_family, None)

    def request(self, **extra):
        body = {"model": "kolibri-1", "messages": [{"role": "user", "content": "Wetter?"}],
                "tools": [{"type": "function", "function": x} for x in TOOLS], "max_tokens": 512, **extra}
        request = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)

    def test_chat_reasoning_and_tools(self):
        result = self.request()
        msg = result["choices"][0]["message"]
        self.assertNotIn("<think>", msg["reasoning_content"])
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "weather")
        self.assertEqual(json.loads(msg["tool_calls"][0]["function"]["arguments"]), {"city": "Köln"})
        self.assertEqual(result["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(self.engine.sampling["top_k"], 128)
        self.assertEqual(self.engine.sampling["top_p"], 0.97)

    def test_required_and_named_with_reasoning_on_and_off(self):
        for choice in ("required", {"type": "function", "function": {"name": "weather"}}):
            for effort in ("none", "low"):
                msg = self.request(tool_choice=choice, reasoning_effort=effort)["choices"][0]["message"]
                self.assertEqual(msg["tool_calls"][0]["function"]["name"], "weather")

    def test_none_does_not_emit_tool_calls(self):
        msg = self.request(tool_choice="none")["choices"][0]["message"]
        self.assertFalse(msg.get("tool_calls"))

    def test_required_with_multiple_tools(self):
        tools = TOOLS + [{"name": "other", "parameters": {"type": "object", "properties": {}}}]
        msg = self.request(tool_choice="required", reasoning_effort="none",
                           tools=[{"type": "function", "function": t} for t in tools])["choices"][0]["message"]
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "weather")

    def test_tool_result_history_is_rendered(self):
        self.request(messages=[{"role": "user", "content": "Wetter?"},
                               {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function",
                                "function": {"name": "weather", "arguments": '{"city":"Köln"}'}}]},
                               {"role": "tool", "tool_call_id": "call_1", "content": "Köln: sonnig"},
                               {"role": "user", "content": "Danke!"}])
        prompt = self.tok.decode(self.engine.last_prompt)
        self.assertIn('"name": "weather"', prompt)
        self.assertIn("Köln: sonnig", prompt)

    def test_streaming_emits_complete_json_arguments(self):
        body = {"model": "kolibri-1", "messages": [{"role": "user", "content": "Wetter?"}],
                "tools": [{"type": "function", "function": x} for x in TOOLS], "max_tokens": 512, "stream": True}
        request = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as response:
            events = [json.loads(line[6:]) for line in response.read().decode().splitlines()
                      if line.startswith("data: ") and line != "data: [DONE]"]
        calls = [c for e in events for c in e["choices"][0].get("delta", {}).get("tool_calls", [])]
        self.assertEqual("".join(c.get("function", {}).get("name", "") for c in calls), "weather")
        self.assertEqual(json.loads("".join(c.get("function", {}).get("arguments", "") for c in calls)), {"city": "Köln"})

    def test_metadata_eos_and_template_capabilities(self):
        self.assertEqual(self.svc.stop_ids, {257})
        self.assertEqual(self.engine.eos_ids, {257})
        for name in ("supports_tools", "supports_tool_calls", "supports_parallel_tool_calls"):
            self.assertTrue(self.svc.template.caps[name], name)

    def test_sampling_request_overrides_and_wire_limit(self):
        self.request(temperature=0, top_p=0.9, top_k=12)
        self.assertEqual(self.engine.sampling["temperature"], 0)
        defaults = model_card_sampling("kolibri1", True)
        self.assertIn(" top_k=128", StrataEngine.sampling_keys(defaults, 128))
        self.assertIn(" top_k=64", StrataEngine.sampling_keys(defaults))


if __name__ == "__main__":
    unittest.main()
