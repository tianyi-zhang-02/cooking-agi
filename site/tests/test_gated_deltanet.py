import math
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class GatedDeltaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blocks = {}
        for suffix in (".md", ".en.md"):
            text = (ROOT / "00-foundations/deep-dives" / ("gated-deltanet" + suffix)).read_text()
            cls.blocks[suffix] = re.findall(r"```python\n(.*?)```", text, re.S)
        namespace = {}
        for block in cls.blocks[".md"]:
            exec(compile(block, "gated-deltanet", "exec"), namespace)
        cls.step = staticmethod(namespace["gated_delta_step"])

    def test_bilingual_code_matches(self):
        self.assertEqual(self.blocks[".md"], self.blocks[".en.md"])

    def test_zero_write_and_zero_decay(self):
        self.assertEqual(self.step([[2, 10]], [1, 0], [6], 0.5, 0), [[1, 5]])
        self.assertEqual(self.step([[2, 10]], [1, 0], [6], 0, 0.5), [[3, 0]])

    def test_normalized_key_interpolation(self):
        key = [1 / math.sqrt(2)] * 2
        state = [[2, 10], [-1, 3]]
        values = [6, 2]
        output = self.step(state, key, values, 0.5, 0.25)
        for before, after, target in zip(state, output, values):
            old_read = sum(entry * direction for entry, direction in zip(before, key))
            new_read = sum(entry * direction for entry, direction in zip(after, key))
            self.assertAlmostEqual(new_read, 0.75 * 0.5 * old_read + 0.25 * target)

    def test_resume_and_reset_are_different(self):
        prefix = self.step([[0, 0]], [1, 0], [2], 1, 1)
        resumed = self.step(prefix, [0, 1], [6], 1, 1)
        reset = self.step([[0, 0]], [0, 1], [6], 1, 1)
        self.assertEqual(resumed, [[2, 6]])
        self.assertEqual(reset, [[0, 6]])

    def test_invalid_inputs(self):
        for args in [([], [1], [2], 1, 1), ([[1]], [1, 0], [2], 1, 1), ([[1]], [2], [2], 1, 1), ([[1]], [1], [2], -1, 1), ([[1]], [1], [math.nan], 1, 1)]:
            with self.assertRaises(ValueError):
                self.step(*args)


if __name__ == "__main__":
    unittest.main()
