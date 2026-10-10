import importlib.util
import math
from pathlib import Path
import re
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("lexical_interpretation", ROOT / "04-search/code/lexical_retrieval.py")
LEXICAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LEXICAL)


def contrastive_function(language):
    source = ROOT / f"04-search/dual-encoder{language}.md"
    namespace = {}
    for code in re.findall(r"```python\n(.*?)\n```", source.read_text(), re.S):
        exec(compile(code, str(source), "exec"), namespace)
    return namespace["contrastive_step"]


class RetrievalInterpretationTests(unittest.TestCase):
    def test_document_duplication_changes_raw_not_relative_tf(self):
        original, duplicate = [1, 1], [2, 2]
        weights = [math.log(4 / 3), math.log(2)]
        raw_original = sum(count * weight for count, weight in zip(original, weights))
        raw_duplicate = sum(count * weight for count, weight in zip(duplicate, weights))
        self.assertAlmostEqual(raw_duplicate, 2 * raw_original)
        self.assertEqual([count / sum(original) for count in original],
                         [count / sum(duplicate) for count in duplicate])

    def test_l2_normalized_tfidf_preserves_duplicate_direction(self):
        original = [math.log(4 / 3), math.log(2)]
        duplicate = [2 * value for value in original]
        query = [0.6, 0.8]
        cosine_scores = [sum(value * target for value, target in zip(vector, query)) / math.hypot(*vector)
                         for vector in (original, duplicate)]
        self.assertAlmostEqual(*cosine_scores)

    def test_half_collection_idf_annuls_all_term_contributions(self):
        inverse_frequency = math.log((6 - 3 + 0.5) / (3 + 0.5))
        self.assertEqual(inverse_frequency, 0)
        for frequency in (1, 2, 20):
            for length_weight in (0, 0.75, 1):
                self.assertEqual(LEXICAL.bm25_term(frequency, 20, 10, inverse_frequency, 1.2, length_weight), 0)

    def test_positive_idf_restores_frequency_differences(self):
        inverse_frequency = math.log1p((6 - 3 + 0.5) / (3 + 0.5))
        self.assertAlmostEqual(inverse_frequency, math.log(2))
        once = LEXICAL.bm25_term(1, 10, 10, inverse_frequency, 1.2, 0)
        twice = LEXICAL.bm25_term(2, 10, 10, inverse_frequency, 1.2, 0)
        self.assertGreater(twice, once)
        self.assertLess(twice, 2 * once)

    def test_tie_breaking_does_not_establish_relevance(self):
        scores = {"A": 0, "B": 0}
        for order in (["A", "B"], ["B", "A"]):
            self.assertEqual(sorted(order, key=scores.get, reverse=True), order)

    def test_zero_length_weight_really_removes_length_effect(self):
        short = LEXICAL.bm25_term(1, 2, 5, 1, 1.2, 0)
        long = LEXICAL.bm25_term(1, 8, 5, 1, 1.2, 0)
        self.assertEqual(short, long)
        self.assertGreater(LEXICAL.bm25_term(1, 2, 5, 1, 1.2, 0.75),
                           LEXICAL.bm25_term(1, 8, 5, 1, 1.2, 0.75))

    def test_geometry_example_uses_unit_vectors_and_matches_losses(self):
        query = [1, 0]
        candidates = [[1, 0], [0, 1], [-1, 0]]
        self.assertTrue(all(math.isclose(math.hypot(*vector), 1) for vector in candidates))
        scores = [sum(first * second for first, second in zip(query, vector)) for vector in candidates]
        self.assertEqual(scores, [1, 0, -1])
        for language in ("", ".en"):
            step = contrastive_function(language)
            collapsed, probabilities, _ = step([1, 1, 1])
            spread, _, _ = step(scores)
            self.assertAlmostEqual(collapsed, math.log(3))
            self.assertEqual(probabilities, [1 / 3] * 3)
            self.assertAlmostEqual(spread, math.log1p(math.exp(-1) + math.exp(-2)))
            self.assertEqual([f"{value:.3f}" for value in (collapsed, spread)], ["1.099", "0.408"])

    def test_even_document_spacing_does_not_fix_wrong_query_alignment(self):
        documents = [[1, 0], [-0.5, math.sqrt(3) / 2], [-0.5, -math.sqrt(3) / 2]]
        for index, query in enumerate(documents[1:] + documents[:1]):
            scores = [sum(first * second for first, second in zip(query, document)) for document in documents]
            self.assertNotEqual(max(range(3), key=scores.__getitem__), index)
            self.assertAlmostEqual(scores[index], -0.5)
        self.assertTrue(all(math.isclose(math.hypot(*document), 1) for document in documents))

    def test_geometry_figure_has_localized_visible_explanation(self):
        for language, html_language, phrase in (("", "zh-CN", "同样让正例重合"), (".en", "en", "The positive stays in place")):
            note = (ROOT / f"04-search/dual-encoder{language}.md").read_text()
            figure = re.search(r'<figure[^>]*id="geometry-example".*?</figure>', note, re.S).group()
            self.assertIn(f'lang="{html_language}"', figure)
            self.assertIn(phrase, figure)
            self.assertEqual(figure.count("<li>"), 3)
            self.assertNotIn("<details", note[:note.index(figure)])
            self.assertIn("wang20k/wang20k.pdf", note)
            self.assertIn("{#where-labels-come-from}", note)

    def test_three_reference_readings_are_recorded_without_changing_scope(self):
        coverage = tomllib.loads((ROOT / "site/appendix-coverage.toml").read_text())
        self.assertEqual(coverage["scope"], "screenshots-and-browser-directory-not-complete-reference")
        topics = {topic["id"]: topic for chapter in coverage["chapter"] for topic in chapter["topics"]}
        for identifier in ("qwen-bge", "bm25-tfidf", "infonce-cross-entropy"):
            self.assertEqual(topics[identifier]["read"], "body")
            self.assertEqual(topics[identifier]["state"], "article")
        self.assertEqual(topics["softmax-implementations"]["read"], "title")


if __name__ == "__main__":
    unittest.main()
