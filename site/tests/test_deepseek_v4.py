import math
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class DeepSeekV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blocks = {}
        for suffix in (".md", ".en.md"):
            text = (ROOT / "00-foundations/model-families" / ("deepseek-v4" + suffix)).read_text()
            cls.blocks[suffix] = re.findall(r"```python\n(.*?)```", text, re.S)
        namespace = {}
        for block in cls.blocks[".md"]:
            exec(compile(block, "deepseek-v4", "exec"), namespace)
        cls.ledger = staticmethod(namespace["attention_ledger"])

    def test_bilingual_code(self):
        self.assertEqual(self.blocks[".md"], self.blocks[".en.md"])

    def test_storage_grows_even_when_reads_are_bounded(self):
        short = self.ledger(1024, 4, 128, 32, 128)
        long = self.ledger(2048, 4, 128, 32, 128)
        self.assertEqual(long["light_stored"], 2 * short["light_stored"])
        self.assertEqual(long["light_main_reads"], short["light_main_reads"])

    def test_no_complete_heavy_block_yet(self):
        result = self.ledger(9, 4, 128, 32, 128)
        self.assertEqual(result["heavy_stored"], 0)
        self.assertEqual(result["heavy_main_reads"], 9)
        self.assertEqual(result["light_stored"], 2)

    def test_invalid_counts(self):
        for args in [(0, 4, 128, 32, 128), (9, 4, 4, 2, 2), (9, True, 4, 2, 2)]:
            with self.assertRaises(ValueError):
                self.ledger(*args)

    def test_fixed_doubly_stochastic_example(self):
        output = [0.8 * 2 + 0.2 * 10, 0.2 * 2 + 0.8 * 10]
        self.assertAlmostEqual(sum(output), 12)
        self.assertAlmostEqual(sum(value ** 2 for value in output), 83.52)
        self.assertLess(sum(value ** 2 for value in output), 104)

    def test_local_kl_example(self):
        divergence = 0.8 * math.log(1.6) + 0.2 * math.log(0.4)
        self.assertAlmostEqual(divergence, 0.19274475702175753)


if __name__ == "__main__":
    unittest.main()
