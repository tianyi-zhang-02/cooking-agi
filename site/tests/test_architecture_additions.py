import math
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
LESSONS = ("gated-attention", "engram", "attention-residuals", "position-and-context", "latent-and-sparse-attention")


def snippets(lesson, suffix=".md"):
    text = (ROOT / "00-foundations/deep-dives" / (lesson + suffix)).read_text()
    return re.findall(r"```python\n(.*?)```", text, re.S)


class ArchitectureAdditionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespaces = {}
        for lesson in LESSONS:
            namespace = {}
            for snippet in snippets(lesson):
                exec(compile(snippet, lesson, "exec"), namespace)
            cls.namespaces[lesson] = namespace

    def test_bilingual_code_matches(self):
        for lesson in LESSONS:
            self.assertEqual(snippets(lesson), snippets(lesson, ".en.md"))

    def test_gates_are_stable_and_not_zero_logit_identity(self):
        sigmoid = self.namespaces["gated-attention"]["sigmoid"]
        self.assertEqual(sigmoid(10000), 1)
        self.assertEqual(sigmoid(-10000), 0)
        self.assertEqual(sigmoid(0), 0.5)
        gated = self.namespaces["gated-attention"]["gated_heads"]
        for heads, logits in [([], []), ([[1]], []), ([[]], [0]), ([[1]], [math.nan])]:
            with self.assertRaises(ValueError):
                gated(heads, logits)

    def test_engram_collisions_and_invalid_ids(self):
        bucket = self.namespaces["engram"]["bigram_bucket"]
        self.assertEqual(bucket([1, 2], 5), bucket([2, 4], 5))
        self.assertNotEqual(bucket([1, 2], 7), bucket([2, 4], 7))
        for tokens, size in [([1], 5), ([1, -1], 5), ([True, 2], 5), ([1, 2], 0)]:
            with self.assertRaises(ValueError):
                bucket(tokens, size)

    def test_depth_mix_and_global_normalization(self):
        mix = self.namespaces["attention-residuals"]["depth_mix"]
        self.assertEqual(mix([0, 0, 0, 0], [0, 4, 4, 4]), 3)
        self.assertEqual((mix([0], [0]) + mix([0, 0, 0], [4, 4, 4])) / 2, 2)
        for scores, values in [([], []), ([1], []), ([1], [math.inf]), ([math.nan], [1])]:
            with self.assertRaises(ValueError):
                mix(scores, values)

    def test_dca_boundaries_and_bounded_distances(self):
        distance = self.namespaces["position-and-context"]["dca_distance"]
        for query in range(100):
            for key in range(query + 1):
                result = distance(query, key)
                self.assertTrue(0 <= result < 8)
                if query // 5 == key // 5:
                    self.assertEqual(result, query - key)
        for args in [(2, 3), (-1, 0), (1, True), (1, 0, 5, 5)]:
            with self.assertRaises(ValueError):
                distance(*args)

    def test_dsa_selection_and_missed_evidence(self):
        namespace = self.namespaces["latent-and-sparse-attention"]
        select = namespace["selected_positions"]
        self.assertEqual(select([1, 1, 1], 2), [0, 1])
        self.assertGreater(namespace["dense_output"], 9)
        self.assertEqual(namespace["sparse_output"], 0)
        for scores, count in [([], 1), ([1], 0), ([1], 2), ([math.nan], 1), ([1], True)]:
            with self.assertRaises(ValueError):
                select(scores, count)


if __name__ == "__main__":
    unittest.main()
