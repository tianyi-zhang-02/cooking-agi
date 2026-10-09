import math
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class SparseContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blocks = []
        for suffix in (".md", ".en.md"):
            text = (ROOT / "00-foundations/deep-dives" / ("latent-and-sparse-attention" + suffix)).read_text()
            cls.blocks.append(next(block for block in re.findall(r"```python\n(.*?)```", text, re.S) if "def shared_block_choice" in block))
        namespace = {}
        exec(cls.blocks[0], namespace)
        cls.choose = staticmethod(namespace["shared_block_choice"])

    def test_bilingual_code(self):
        self.assertEqual(*self.blocks)

    def test_group_choice_is_not_each_heads_favorite(self):
        heads = [[0.6, 0.3, 0.1], [0.05, 0.45, 0.5]]
        self.assertEqual(self.choose(heads, 1), [1])
        self.assertEqual(self.choose([heads[0]], 1), [0])
        self.assertEqual(self.choose([heads[1]], 1), [2])

    def test_ties_and_full_selection(self):
        self.assertEqual(self.choose([[0.5, 0.5]], 1), [0])
        self.assertEqual(self.choose([[0.2, 0.8]], 2), [1, 0])

    def test_invalid_distributions(self):
        for heads, count in [([], 1), ([[1]], 2), ([[math.nan]], 1), ([[0.2, 0.2]], 1), ([[1], [0.5, 0.5]], 1)]:
            with self.assertRaises(ValueError):
                self.choose(heads, count)

    def test_yarn_amplitude_is_squared(self):
        amplitude = 1 + 0.1 * math.log(4)
        self.assertAlmostEqual(amplitude ** 2, 1.296476992780706, places=6)
        self.assertEqual(sum(value ** 2 for value in [4, 6]), 52)
        self.assertEqual(sum(value ** 2 for value in [2, 6]), 40)


if __name__ == "__main__":
    unittest.main()
