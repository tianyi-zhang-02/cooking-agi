import importlib.util
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "two_tower_reference", ROOT / "practice/recommender-systems/code/two_tower_reference.py")
REFERENCE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REFERENCE
SPEC.loader.exec_module(REFERENCE)


class RetrievalContractTests(unittest.TestCase):
    def setUp(self):
        self.spec = REFERENCE.EmbeddingSpec("space-v1", 2, "cosine")
        self.items = {"A": (0.8, 0.6), "B": (1.0, 1.0), "C": (0.0, 1.0)}

    def test_exact_scores_and_order(self):
        result = REFERENCE.exact_top_k((1, 0), self.spec, self.items, self.spec, 2)
        self.assertEqual([item_id for item_id, score in result], ["A", "B"])
        self.assertAlmostEqual(result[0][1], .8)
        self.assertAlmostEqual(result[1][1], 1 / math.sqrt(2))

    def test_excluded_items_never_enter_result(self):
        result = REFERENCE.exact_top_k((1, 0), self.spec, self.items, self.spec, 2, {"A"})
        self.assertEqual([item_id for item_id, score in result], ["B", "C"])
        self.assertEqual(REFERENCE.exact_top_k((1, 0), self.spec, {}, self.spec, 2), [])
        self.assertEqual(REFERENCE.exact_top_k(
            (1, 0), self.spec, self.items, self.spec, 2, set(self.items)), [])

    def test_same_dimensions_do_not_prove_compatibility(self):
        for incompatible in [
            REFERENCE.EmbeddingSpec("space-v2", 2, "cosine"),
            REFERENCE.EmbeddingSpec("space-v1", 2, "inner_product"),
            REFERENCE.EmbeddingSpec("space-v1", 3, "cosine"),
        ]:
            with self.assertRaisesRegex(ValueError, "incompatible"):
                REFERENCE.exact_top_k((1, 0), incompatible, self.items, self.spec, 2)

    def test_cosine_and_inner_product_are_different_contracts(self):
        items = {"aligned": (2, 0), "large": (100, 100)}
        cosine_result = REFERENCE.exact_top_k((1, 0), self.spec, items, self.spec, 1)
        inner_spec = REFERENCE.EmbeddingSpec("space-v1", 2, "inner_product")
        inner_result = REFERENCE.exact_top_k((1, 0), inner_spec, items, inner_spec, 1)
        self.assertEqual(cosine_result[0][0], "aligned")
        self.assertEqual(inner_result[0][0], "large")

    def test_ties_stable_and_inputs_unchanged(self):
        items = {"Z": [1, 0], "A": [1, 0]}
        query = [4, 0]
        result = REFERENCE.exact_top_k(query, self.spec, items, self.spec, 5)
        self.assertEqual(result, [("A", 1.0), ("Z", 1.0)])
        self.assertEqual(items, {"Z": [1, 0], "A": [1, 0]})
        self.assertEqual(query, [4, 0])

    def test_invalid_vectors_and_limits_are_rejected(self):
        for query in [[], [1], [0, 0], [math.nan, 1], [math.inf, 1]]:
            with self.assertRaises(ValueError):
                REFERENCE.exact_top_k(query, self.spec, self.items, self.spec, 2)
        for vector in [[1], [0, 0], [math.nan, 1]]:
            with self.assertRaises(ValueError):
                REFERENCE.exact_top_k((1, 0), self.spec, {"bad": vector}, self.spec, 2)
        for limit in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                REFERENCE.exact_top_k((1, 0), self.spec, self.items, self.spec, limit)
        invalid = REFERENCE.EmbeddingSpec("", 2, "cosine")
        with self.assertRaises(ValueError):
            REFERENCE.exact_top_k((1, 0), invalid, self.items, invalid, 1)


if __name__ == "__main__":
    unittest.main()
