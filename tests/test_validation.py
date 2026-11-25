"""Error-path tests that do not require torch/transformers.

These exercise argument and template validation, which happen before the heavy
`import torch` inside main(), so they run on any machine.
"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from swhisper_stage1 import main, validate_rows


@contextlib.contextmanager
def _quiet():
    """Swallow argparse's usage text printed to stderr on parser.error()."""
    with contextlib.redirect_stderr(io.StringIO()):
        yield

GOOD_TEMPLATE = "Request: {goal}\nBegin with: {guiding_str}\n{adv_suffix}"
GOOD_ROWS = [{"goal": "Explain rainbows.", "target": "Sure, here is"}]


def _write(dir_path, name, text):
    path = Path(dir_path) / name
    path.write_text(text)
    return str(path)


class ArgValidationTests(unittest.TestCase):
    """main() calls parser.error() for bad args, which raises SystemExit(2)."""

    def _run(self, extra):
        argv = ["swhisper-stage1", "--model", "dummy/model"] + extra
        with mock.patch("sys.argv", argv), _quiet():
            main()

    def test_nonpositive_steps(self):
        with self.assertRaises(SystemExit):
            self._run(["--steps", "0"])

    def test_nonpositive_suffix_length(self):
        with self.assertRaises(SystemExit):
            self._run(["--suffix-length", "-1"])

    def test_nonpositive_temperature(self):
        with self.assertRaises(SystemExit):
            self._run(["--temperature", "0"])

    def test_negative_semantic_weight(self):
        with self.assertRaises(SystemExit):
            self._run(["--semantic-weight", "-0.5"])


class TemplateValidationTests(unittest.TestCase):
    """Template checks also happen before torch is imported."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.input = _write(self.tmp.name, "tasks.json", json.dumps(GOOD_ROWS))

    def _run(self, template_text):
        template = _write(self.tmp.name, "template.txt", template_text)
        argv = ["swhisper-stage1", "--model", "dummy/model",
                "--input", self.input, "--template", template]
        with mock.patch("sys.argv", argv), _quiet():
            main()

    def test_template_missing_goal(self):
        with self.assertRaises(SystemExit):
            self._run("No goal here, ends right.\n{adv_suffix}")

    def test_template_not_ending_with_suffix(self):
        with self.assertRaises(SystemExit):
            self._run("Request: {goal}\n{adv_suffix} trailing text")

    def test_good_template_passes_validation(self):
        # A valid template should get past validation and fail only later, at the
        # torch import or model load -- never with SystemExit from parser.error().
        try:
            self._run(GOOD_TEMPLATE)
        except SystemExit:
            self.fail("valid template was rejected by argument validation")
        except BaseException:
            pass  # expected: ModuleNotFoundError (no torch) or model-load failure


class InputRowValidationTests(unittest.TestCase):
    """Bad input rows raise ValueError (via validate_rows), not SystemExit."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.template = _write(self.tmp.name, "template.txt", GOOD_TEMPLATE)

    def _run(self, rows_json):
        input_path = _write(self.tmp.name, "tasks.json", rows_json)
        argv = ["swhisper-stage1", "--model", "dummy/model",
                "--input", input_path, "--template", self.template]
        with mock.patch("sys.argv", argv), _quiet():
            main()

    def test_empty_array(self):
        with self.assertRaises(ValueError):
            self._run("[]")

    def test_missing_target(self):
        with self.assertRaises(ValueError):
            self._run(json.dumps([{"goal": "x"}]))

    def test_blank_goal(self):
        with self.assertRaises(ValueError):
            self._run(json.dumps([{"goal": "   ", "target": "Sure"}]))

    def test_validate_rows_direct(self):
        validate_rows([{"goal": "Explain rainbows", "target": "Sure"}])
        for rows in ([], {}, [{"goal": "x"}], [{"goal": "x", "target": ""}]):
            with self.assertRaises(ValueError):
                validate_rows(rows)


if __name__ == "__main__":
    unittest.main()
