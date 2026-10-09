from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class VisualDataContractsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blocks = []
        for suffix in (".md", ".en.md"):
            text = (ROOT / "03-multimodal-learning" / ("blip-and-q-former" + suffix)).read_text()
            cls.blocks.append(re.findall(r"```python\n(.*?)```", text, re.S))
        namespace = {}
        for block in cls.blocks[0]:
            exec(block, namespace)
        cls.report = staticmethod(namespace["filter_report"])

    def test_bilingual_examples(self):
        self.assertEqual(*self.blocks)

    def test_precision_retention_tradeoff(self):
        low = self.report([0.9, 0.8, 0.6, 0.4], [True, False, True, True], 0.5)
        high = self.report([0.9, 0.8, 0.6, 0.4], [True, False, True, True], 0.85)
        self.assertAlmostEqual(low["precision"], 2 / 3)
        self.assertEqual(high, {"kept": 1, "precision": 1, "retention": 0.25, "recall": 1 / 3})

    def test_undefined_denominators_not_fake_perfect_scores(self):
        self.assertIsNone(self.report([0.1], [True], 0.5)["precision"])
        self.assertIsNone(self.report([0.9], [False], 0.5)["recall"])

    def test_invalid_records(self):
        for args in [([], [], 0.5), ([0.8], [], 0.5), ([float("nan")], [True], 0.5),
                     ([0.8], [1], 0.5), ([0.8], [True], 2)]:
            with self.assertRaises(ValueError):
                self.report(*args)


if __name__ == "__main__":
    unittest.main()
