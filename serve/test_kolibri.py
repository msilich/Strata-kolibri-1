"""Kolibri chat, reasoning and native JSON tools without model weights or a GPU."""
import io
import json
from pathlib import Path
import queue
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
