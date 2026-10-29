"""Kolibri's architecture selects its tokenizer even when GGUF calls its pre-tokenizer qwen2."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_tokenizer as ST


def metadata(arch="kolibri1"):
    return {"general.architecture": arch, "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen2",
            "tokenizer.ggml.tokens": [ST.BYTE_TO_UNICODE[b] for b in range(256)] + ["_j", "<|im_end|>"],
            "tokenizer.ggml.token_type": [1] * 257 + [3], "tokenizer.ggml.merges": ["_ j"],
            "tokenizer.ggml.eos_token_id": 257, "tokenizer.chat_template": "{{ messages[0].content }}"}


class KolibriTokenizer(unittest.TestCase):
    def test_combining_marks_use_model_boundaries(self):
        md = metadata()
        with mock.patch("gguf_reader.GGUFFile", return_value=SimpleNamespace(metadata=md)):
            tok = ST.Tokenizer.from_gguf("unused.gguf")
        text = "a\u0308_j"
        expected = [tok.ids[ST.BYTE_TO_UNICODE[b]] for b in text.encode()]
        self.assertEqual(tok.encode(text), expected)
        old = ST.Tokenizer(tok.tokens, ["_ j"], tok.token_types, pre="qwen2")
        self.assertIn(256, old.encode(text))
        for text in ("Grüße!", "a\u0308_j", "Don't stop 123456", "👩‍💻\r\n\t日本語"):
            self.assertEqual(tok.decode(tok.encode(text)), text)

    def test_export_preserves_architecture_eos_and_template(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch("gguf_reader.GGUFFile", return_value=SimpleNamespace(metadata=metadata())):
                cfg = ST.extract("unused.gguf", folder)
            path = Path(folder) / "tokenizer"
            self.assertEqual(cfg["arch"], "kolibri1")
            self.assertEqual(cfg["special_ids"]["tokenizer.ggml.eos_token_id"], 257)
            self.assertEqual(cfg["pre_pattern"], ST.KOLIBRI1_PATTERN)
            self.assertEqual(json.loads((path / "tokenizer.json").read_text()), cfg)
            self.assertEqual((path / "chat_template.jinja").read_text(), metadata()["tokenizer.chat_template"])

    def test_other_export_keeps_original_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch("gguf_reader.GGUFFile", return_value=SimpleNamespace(metadata=metadata("qwen4exp"))):
                cfg = ST.extract("unused.gguf", folder)
            self.assertNotIn("arch", cfg)
            self.assertEqual(cfg["pre_pattern"], ST.QWEN35_PATTERN)


if __name__ == "__main__":
    unittest.main()
