import unittest
from swhisper_stage1 import validate_rows, voiceable


class SetupTests(unittest.TestCase):
    def test_vocabulary(self):
        for word in (" hello", "world", "don't", "well-known"):
            self.assertTrue(voiceable(word))
        for word in ("", "!!!", "123", "a b", "<|end|>"):
            self.assertFalse(voiceable(word))

    def test_tasks(self):
        validate_rows([{"goal": "Explain rainbows", "target": "Sure"}])
        for rows in ([], {}, [{"goal": "x"}], [{"goal": "x", "target": ""}]):
            with self.assertRaises(ValueError):
                validate_rows(rows)


if __name__ == "__main__":
    unittest.main()
