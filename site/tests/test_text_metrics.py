import math
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


def snippets(suffix=".md"):
    return re.findall(r"```python\n(.*?)```", (ROOT / ("07-evaluation/text-metrics" + suffix)).read_text(), re.S)


class TextMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = {}
        for snippet in snippets():
            exec(compile(snippet, "text-metrics", "exec"), cls.namespace)

    def test_bilingual_snippets_match(self):
        self.assertEqual(snippets(), snippets(".en.md"))

    def test_bleu_hand_calculations(self):
        score = self.namespace["bleu_example"]
        reference = "the bus leaves at ten".split()
        self.assertAlmostEqual(score("the bus leaves".split(), reference), 0.513417119)
        self.assertAlmostEqual(score("the bus leaves at nine".split(), reference), math.sqrt(0.6))
        self.assertEqual(score(reference, reference), 1)
        self.assertEqual(score([], reference), 0)
        self.assertEqual(score(["unrelated"], reference), 0)
        self.assertEqual(score(["red"], ["red"], 2), 0)

    def test_ngram_counts_clip_repetitions(self):
        counts = self.namespace["ngram_counts"]
        self.assertEqual(counts(["a", "a", "a"], 2), {("a", "a"): 2})
        score = self.namespace["bleu_example"]
        self.assertEqual(score(["red"] * 3, ["red", "blue"], 1), 1 / 3)

    def test_invalid_orders_and_empty_references(self):
        for order in (0, -1, 1.5, True):
            with self.subTest(order=order), self.assertRaises(ValueError):
                self.namespace["bleu_example"](["a"], ["a"], order)
        with self.assertRaises(ValueError):
            self.namespace["bleu_example"](["a"], [])

    def test_lcs_and_edit_distance(self):
        longest = self.namespace["lcs_length"]
        distance = self.namespace["edit_distance"]
        for left, right, expected_lcs, expected_distance in [
            ("abc", "abc", 3, 0), ("abc", "xyz", 0, 3),
            ("", "abc", 0, 3), ("abc", "", 0, 3),
            ("abc", "ac", 2, 1), ("ab", "ba", 1, 2),
            ("1280", "1230", 3, 1), ("", "", 0, 0),
        ]:
            with self.subTest(left=left, right=right):
                self.assertEqual(longest(left, right), expected_lcs)
                self.assertEqual(longest(right, left), expected_lcs)
                self.assertEqual(distance(left, right), expected_distance)
                self.assertEqual(distance(right, left), expected_distance)

    def test_error_rate_can_exceed_one(self):
        self.assertEqual(self.namespace["edit_distance"](["a"], ["a", "b", "c", "d"]), 3)

    def test_perplexity_and_rouge_example_arithmetic(self):
        self.assertAlmostEqual(math.exp(-(math.log(0.5) + math.log(0.25)) / 2), math.sqrt(8))
        precision, recall = 4 / 5, 1
        self.assertAlmostEqual(2 * precision * recall / (precision + recall), 8 / 9)


if __name__ == "__main__":
    unittest.main()
