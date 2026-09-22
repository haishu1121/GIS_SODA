"""Regression tests for the chat-native SFT loss mask."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_train_sft_module():
    script = Path(__file__).parents[1] / "scripts" / "train_sft.py"
    spec = importlib.util.spec_from_file_location("gis_soda_train_sft", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _QwenStyleTokenizer:
    """Minimal tokenizer whose explicit assistant header differs from prompt."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is True and enable_thinking is False
        user_ids = [ord(char) for char in messages[0]["content"]]
        if add_generation_prompt:
            return [100] + user_ids + [701]
        assistant_ids = [ord(char) for char in messages[1]["content"]]
        return [100] + user_ids + [702] + assistant_ids + [703]

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(char) for char in text]}

    eos_token_id = 703


class _WrappedTemplateTokenizer(_QwenStyleTokenizer):
    """Represents newer Transformers returning a mapping from chat templates."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        return {"input_ids": super().apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking,
        )}


class TrainSFTTests(unittest.TestCase):
    def test_masks_qwen_style_different_assistant_prefix(self) -> None:
        module = _load_train_sft_module()
        record = {
            "scenario_id": "scenario-1",
            "split": "train",
            "view": {"trace_style": "ooda"},
            "messages": [
                {"role": "user", "content": "Fixed scene"},
                {"role": "assistant", "content": "Observe:\nVerified."},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            rows = module._read_messages_jsonl(
                path, expected_trace_style="ooda", tokenizer=_QwenStyleTokenizer(), max_length=128,
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].labels[:13], [-100] * 13)
        self.assertEqual(rows[0].labels[13:], [ord(char) for char in "Observe:\nVerified."] + [703])

    def test_uses_native_prompt_and_direct_assistant_content(self) -> None:
        module = _load_train_sft_module()
        record = {
            "scenario_id": "scenario-2",
            "split": "train",
            "view": {"trace_style": "ooda"},
            "messages": [
                {"role": "user", "content": "Fixed scene"},
                {"role": "assistant", "content": "Observe"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            rows = module._read_messages_jsonl(
                path, expected_trace_style="ooda", tokenizer=_QwenStyleTokenizer(), max_length=128,
            )
        self.assertEqual(rows[0].input_ids[:13], [100] + [ord(char) for char in "Fixed scene"] + [701])
        self.assertEqual(rows[0].labels, [-100] * 13 + [ord(char) for char in "Observe"] + [703])

    def test_accepts_mapping_returned_by_newer_chat_template(self) -> None:
        module = _load_train_sft_module()
        record = {
            "scenario_id": "scenario-3",
            "split": "train",
            "view": {"trace_style": "ooda"},
            "messages": [
                {"role": "user", "content": "Scene"},
                {"role": "assistant", "content": "Act"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            rows = module._read_messages_jsonl(
                path, expected_trace_style="ooda", tokenizer=_WrappedTemplateTokenizer(), max_length=128,
            )
        self.assertTrue(all(isinstance(token, int) for token in rows[0].input_ids))


if __name__ == "__main__":
    unittest.main()
