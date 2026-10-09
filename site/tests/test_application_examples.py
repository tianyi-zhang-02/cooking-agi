import math
from pathlib import Path
import re
import tomllib
import unittest
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[2]
CHAPTERS = ("04-search/vector-indexes", "04-search/rag-evidence", "10-agents/patterns")


def snippets(chapter, suffix=".md"):
    return re.findall(r"```python\n(.*?)```", (ROOT / (chapter + suffix)).read_text(), re.S)


def namespace_for(chapter):
    namespace = {}
    for snippet in snippets(chapter):
        exec(compile(snippet, chapter, "exec"), namespace)
    return namespace


class ApplicationExamplesTests(unittest.TestCase):
    def test_dual_encoder_examples_match_and_execute(self):
        self.assertEqual(snippets("04-search/dual-encoder"), snippets("04-search/dual-encoder", ".en.md"))
        namespace_for("04-search/dual-encoder")

    def test_contrastive_loss_does_not_cancel_after_a_large_shift(self):
        calculate = namespace_for("04-search/dual-encoder")["contrastive_step"]
        for value in (0, 1e20, -1e20, 1e300, -1e300):
            loss, probabilities, gradients = calculate([value, value])
            self.assertAlmostEqual(loss, math.log(2))
            self.assertEqual(probabilities, [0.5, 0.5])
            self.assertEqual(gradients, [-0.5, 0.5])

    def test_contrastive_gradient_matches_finite_differences(self):
        calculate = namespace_for("04-search/dual-encoder")["contrastive_step"]
        scores = [2., 1., -1.]
        step = 1e-6
        for temperature in (0.2, 1., 3.):
            gradients = calculate(scores, temperature)[2]
            for index in range(len(scores)):
                plus = scores.copy()
                minus = scores.copy()
                plus[index] += step
                minus[index] -= step
                numeric = (calculate(plus, temperature)[0] - calculate(minus, temperature)[0]) / (2 * step)
                self.assertAlmostEqual(gradients[index], numeric, places=8)

    def test_contrastive_rejects_nonfinite_inputs(self):
        calculate = namespace_for("04-search/dual-encoder")["contrastive_step"]
        for scores, temperature in (([], 1), ([1, 2], 0), ([1, 2], -1),
                                    ([1, 2], math.nan), ([1, 2], math.inf),
                                    ([math.inf, 1], 1), ([1, math.nan], 1), ([1e300], 1e-300)):
            with self.subTest(scores=scores, temperature=temperature), self.assertRaises(ValueError):
                calculate(scores, temperature)

    def test_bilingual_examples_match_and_execute(self):
        for chapter in CHAPTERS:
            with self.subTest(chapter=chapter):
                self.assertEqual(snippets(chapter), snippets(chapter, ".en.md"))
                namespace_for(chapter)

    def test_links_resolve_and_keep_english(self):
        for chapter in CHAPTERS:
            for suffix in (".md", ".en.md"):
                source = ROOT / (chapter + suffix)
                for target in re.findall(r"\]\(([^)]+)\)", source.read_text()):
                    parsed = urlsplit(target)
                    if parsed.scheme or not parsed.path:
                        continue
                    resolved = (source.parent / unquote(parsed.path)).resolve()
                    self.assertTrue(resolved.exists(), f"{source}: {target}")
                    if suffix == ".en.md" and resolved.suffix == ".md" and resolved != ROOT / (chapter + ".md"):
                        self.assertTrue(resolved.name.endswith(".en.md"), f"{source}: {target}")

    def test_navigation_order_and_labels(self):
        nav = tomllib.loads((ROOT / "site/nav.toml").read_text())
        for chapter in CHAPTERS:
            source = Path(chapter + ".md")
            section = next(item for item in nav["section"] if item["dir"] == str(source.parent))
            self.assertIn(source.name, section["order"])
            self.assertEqual(len(nav["label"][str(source)]), 2)

    def test_adc_matches_reconstructed_squared_distance(self):
        namespace = namespace_for("04-search/vector-indexes")
        calculate = namespace["adc_squared_distance"]
        for first_code in range(3):
            for second_code in range(3):
                codes = [first_code, second_code]
                query = sum(namespace["query_parts"], [])
                reconstructed = sum([book[code] for book, code in zip(namespace["codebooks"], codes)], [])
                expected = sum((value - center) ** 2 for value, center in zip(query, reconstructed))
                self.assertEqual(calculate(namespace["query_parts"], namespace["codebooks"], codes), expected)

    def test_codebook_relabeling_does_not_change_adc(self):
        namespace = namespace_for("04-search/vector-indexes")
        calculate = namespace["adc_squared_distance"]
        books = namespace["codebooks"]
        query = namespace["query_parts"]
        before = calculate(query, books, [0, 2])
        after = calculate(query, [list(reversed(book)) for book in books], [2, 0])
        self.assertEqual(before, after)

    def test_adc_rejects_truncation_and_invalid_codes(self):
        namespace = namespace_for("04-search/vector-indexes")
        calculate = namespace["adc_squared_distance"]
        for query, books, codes in [([[1]], [[[1]]], []), ([[1, 2]], [[[1]]], [0]),
                                    ([[1]], [[[1]]], [-1]), ([[1]], [[[1]]], [1])]:
            with self.subTest(query=query, codes=codes), self.assertRaises(ValueError):
                calculate(query, books, codes)

    def test_metric_denominators_are_distinct(self):
        calculate = namespace_for("04-search/rag-evidence")["retrieval_metrics"]
        actual = calculate(["B", "C", "D"], {"C"}, 3)
        self.assertEqual(actual, {"hit": 1, "precision": 1 / 3, "recall": 1, "reciprocal_rank": 0.5})

    def test_short_returns_use_fixed_precision_denominator(self):
        calculate = namespace_for("04-search/rag-evidence")["retrieval_metrics"]
        self.assertEqual(calculate(["A"], {"A"}, 5)["precision"], 0.2)

    def test_empty_positives_and_empty_returns(self):
        calculate = namespace_for("04-search/rag-evidence")["retrieval_metrics"]
        self.assertIsNone(calculate(["A"], set(), 3)["recall"])
        self.assertEqual(calculate([], {"A"}, 3),
                         {"hit": 0, "precision": 0, "recall": 0, "reciprocal_rank": 0})

    def test_relevance_outside_cutoff_is_not_a_hit(self):
        calculate = namespace_for("04-search/rag-evidence")["retrieval_metrics"]
        actual = calculate(["B", "C", "A"], {"A"}, 2)
        self.assertEqual(actual["hit"], 0)
        self.assertEqual(actual["reciprocal_rank"], 0)

    def test_duplicate_ids_and_invalid_cutoff_are_rejected(self):
        calculate = namespace_for("04-search/rag-evidence")["retrieval_metrics"]
        for results, cutoff in [(["A", "A"], 2), (["A"], 0), (["A"], -1)]:
            with self.subTest(results=results, cutoff=cutoff), self.assertRaises(ValueError):
                calculate(results, {"A"}, cutoff)

    def test_memory_example_excludes_overhead(self):
        self.assertTrue(math.isclose(10_000_000 * 768 * 4 / 2 ** 30, 28.61, abs_tol=0.005))
        self.assertEqual(768 * 4 / 64, 48)


if __name__ == "__main__":
    unittest.main()
