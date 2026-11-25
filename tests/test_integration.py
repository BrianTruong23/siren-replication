"""End-to-end and model-dependent error-path tests.

These need torch, transformers, and network access to download a small model, so
they are OFF by default. Enable them with:

    SWHISPER_RUN_INTEGRATION=1 python -m unittest discover -s tests

Override the model with SWHISPER_TEST_MODEL (default: Qwen/Qwen2.5-0.5B-Instruct).
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

from swhisper_stage1 import main

TEST_MODEL = os.environ.get("SWHISPER_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
RUN = os.environ.get("SWHISPER_RUN_INTEGRATION") == "1"
SUFFIX_LEN = 5
REQUIRED_KEYS = {"task", "suffix", "suffix_ids", "prompt", "prompt_ids",
                 "best_loss", "response", "history", "config", "limitations"}

GOOD_TEMPLATE = ("Answer this request clearly and concisely: {goal}\n"
                 "Begin your response with: {guiding_str}\n{adv_suffix}")


def _write(dir_path, name, text):
    path = Path(dir_path) / name
    path.write_text(text)
    return str(path)


@unittest.skipUnless(HAVE_DEPS and RUN,
                     "set SWHISPER_RUN_INTEGRATION=1 (needs torch+transformers+network)")
class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.template = _write(self.tmp.name, "template.txt", GOOD_TEMPLATE)

    def _argv(self, input_path, output_path, extra=None):
        argv = ["swhisper-stage1", "--model", TEST_MODEL,
                "--input", input_path, "--template", self.template,
                "--output", output_path,
                "--steps", "2", "--suffix-length", str(SUFFIX_LEN),
                "--top-k", "16", "--batch-size", "4", "--microbatch", "1",
                "--positions", "2", "--max-new-tokens", "8",
                "--device", "cpu", "--dtype", "float32", "--seed", "0"]
        return argv + (extra or [])

    def _run(self, rows, extra=None, output_name="out.jsonl"):
        input_path = _write(self.tmp.name, "tasks.json", json.dumps(rows))
        output_path = str(Path(self.tmp.name) / output_name)
        with mock.patch("sys.argv", self._argv(input_path, output_path, extra)):
            main()
        return output_path

    def test_end_to_end_single_task(self):
        rows = [{"goal": "Explain how rainbows form.",
                 "target": "Sure, here is an explanation."}]
        output_path = self._run(rows)
        self.assertTrue(Path(output_path).exists())
        lines = Path(output_path).read_text().splitlines()
        self.assertEqual(len(lines), 1)
        rec = json.loads(lines[0])
        self.assertEqual(REQUIRED_KEYS - set(rec), set(), "record missing keys")
        self.assertEqual(len(rec["suffix_ids"]), SUFFIX_LEN)
        self.assertIsInstance(rec["best_loss"], float)
        self.assertIsInstance(rec["response"], str)
        # Optimization kept the best, never worse than where the walk started.
        self.assertTrue(rec["history"], "history should not be empty")
        self.assertLessEqual(rec["best_loss"], rec["history"][0]["current_loss"] + 1e-6)

    def test_end_to_end_multiple_tasks(self):
        rows = [{"goal": "Explain how rainbows form.", "target": "Sure, here is"},
                {"goal": "Describe the water cycle.", "target": "Of course, here is"}]
        output_path = self._run(rows, output_name="multi.jsonl")
        lines = Path(output_path).read_text().splitlines()
        self.assertEqual(len(lines), 2)
        for line in lines:
            self.assertEqual(REQUIRED_KEYS - set(json.loads(line)), set())

    def test_refuses_to_overwrite_output(self):
        rows = [{"goal": "Explain how rainbows form.", "target": "Sure, here is"}]
        input_path = _write(self.tmp.name, "tasks.json", json.dumps(rows))
        output_path = str(Path(self.tmp.name) / "exists.jsonl")
        Path(output_path).write_text("pre-existing\n")
        with mock.patch("sys.argv", self._argv(input_path, output_path)):
            with self.assertRaises(FileExistsError):
                main()

    def test_bad_seeds_rejected(self):
        rows = [{"goal": "Explain how rainbows form.", "target": "Sure, here is"}]
        # Seeds that cannot tokenize to SUFFIX_LEN voiceable tokens -> ValueError.
        seeds = _write(self.tmp.name, "seeds.json", json.dumps(["???"]))
        with self.assertRaises(ValueError):
            self._run(rows, extra=["--seeds", seeds], output_name="seeds.jsonl")


@unittest.skipUnless(HAVE_DEPS and RUN,
                     "set SWHISPER_RUN_INTEGRATION=1 (needs torch+transformers+network)")
class ChatTemplateGuardTests(unittest.TestCase):
    def test_model_without_chat_template_rejected(self):
        # gpt2 has no chat template; main() should exit before loading the model.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        template = _write(tmp.name, "template.txt", GOOD_TEMPLATE)
        input_path = _write(tmp.name, "tasks.json",
                            json.dumps([{"goal": "x", "target": "Sure"}]))
        output_path = str(Path(tmp.name) / "out.jsonl")
        argv = ["swhisper-stage1", "--model", "gpt2",
                "--input", input_path, "--template", template,
                "--output", output_path, "--device", "cpu", "--dtype", "float32"]
        with mock.patch("sys.argv", argv):
            with self.assertRaises(SystemExit):
                main()


if __name__ == "__main__":
    unittest.main()
