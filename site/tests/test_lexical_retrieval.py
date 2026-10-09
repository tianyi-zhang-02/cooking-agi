import ast
import importlib.util
import math
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "04-search/code/lexical_retrieval.py"
SPEC = importlib.util.spec_from_file_location("lexical_retrieval", SOURCE)
LEXICAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LEXICAL)


class LexicalRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            "cache cache error".split(),
            "cache error".split(),
            "cache guide setup notes".split(),
            "release notes".split(),
        ]
        self.index = LEXICAL.LexicalIndex(self.documents)

    def test_document_frequency_counts_documents_not_occurrences(self):
        self.assertEqual(self.index.document_frequency["cache"], 3)
        self.assertEqual(self.index.document_frequency["error"], 2)
        self.assertEqual(self.index.lengths, [3, 2, 4, 2])
        self.assertEqual(self.index.average_length, 2.75)

    def test_tfidf_matches_hand_calculation(self):
        scores = self.index.tfidf(["cache", "error"])
        expected = [math.log(32 / 9), math.log(8 / 3), math.log(4 / 3), 0]
        for observed, reference in zip(scores, expected):
            self.assertAlmostEqual(observed, reference)
        self.assertEqual([round(score, 4) for score in scores], [1.2685, 0.9808, 0.2877, 0.0])

    def test_bm25_matches_hand_calculation(self):
        scores = self.index.bm25(["cache", "error"])
        cache_idf, error_idf = math.log(10 / 7), math.log(2)
        expected = [
            cache_idf * 4.4 / (2 + 1.2 * (0.25 + 0.75 * 3 / 2.75))
            + error_idf * 2.2 / (1 + 1.2 * (0.25 + 0.75 * 3 / 2.75)),
            (cache_idf + error_idf) * 2.2 / (1 + 1.2 * (0.25 + 0.75 * 2 / 2.75)),
            cache_idf * 2.2 / (1 + 1.2 * (0.25 + 0.75 * 4 / 2.75)),
            0,
        ]
        for observed, reference in zip(scores, expected):
            self.assertAlmostEqual(observed, reference)
        self.assertEqual([round(score, 4) for score in scores], [1.1465, 1.1817, 0.3008, 0.0])
        self.assertGreater(scores[1], scores[0])

    def test_empty_and_unseen_queries(self):
        for method in (self.index.tfidf, self.index.bm25):
            self.assertEqual(method([]), [0.0] * 4)
            self.assertEqual(method(["unknown"]), [0.0] * 4)
            self.assertEqual(method(["cache", "cache", "error", "unknown"]), method(["cache", "error"]))

    def test_empty_collections_and_documents(self):
        for documents in ([], [[]], [[], []]):
            index = LEXICAL.LexicalIndex(documents)
            self.assertEqual(index.tfidf(["cache"]), [0.0] * len(documents))
            self.assertEqual(index.bm25(["cache"], k1=0), [0.0] * len(documents))
        index = LEXICAL.LexicalIndex([[], ["cache"]])
        self.assertEqual(index.average_length, 0.5)
        self.assertEqual(index.bm25(["cache"])[0], 0)
        self.assertTrue(math.isfinite(index.bm25(["cache"])[1]))

    def test_binary_limit_and_length_control(self):
        scores = self.index.bm25(["cache", "error"], k1=0)
        self.assertEqual(scores[0], scores[1])
        self.assertAlmostEqual(scores[2], math.log(10 / 7))
        no_length = self.index.bm25(["cache", "error"], length_weight=0)
        self.assertGreater(no_length[0], no_length[1])
        index = LEXICAL.LexicalIndex([["cache"], ["cache", "extra", "words"]])
        self.assertEqual(*index.bm25(["cache"], length_weight=0))
        self.assertGreater(*index.bm25(["cache"], length_weight=1))

    def test_saturation_has_diminishing_increments(self):
        scores = [LEXICAL.bm25_term(frequency, 10, 10, 1, 1.2, 0.75) for frequency in range(1, 10)]
        increments = [later - earlier for earlier, later in zip(scores, scores[1:])]
        self.assertTrue(all(0 < score < 2.2 for score in scores))
        self.assertTrue(all(earlier > later for earlier, later in zip(increments, increments[1:])))
        self.assertEqual(LEXICAL.bm25_term(0, 0, 0, 1, 0, 1), 0)

    def test_invalid_inputs(self):
        for documents in ("raw text", ["raw text"], [[""]], [[42]]):
            with self.assertRaises((TypeError, ValueError)):
                LEXICAL.LexicalIndex(documents)
        for query in ("cache", [None], [""]):
            for method in (self.index.tfidf, self.index.bm25):
                with self.assertRaises((TypeError, ValueError)):
                    method(query)
        for arguments in ({"k1": -1}, {"k1": math.inf}, {"k1": math.nan}, {"k1": True},
                          {"length_weight": -0.1}, {"length_weight": 1.1}, {"length_weight": math.nan}, {"length_weight": "0.75"}):
            with self.assertRaises(ValueError):
                self.index.bm25(["cache"], **arguments)

    def test_document_order_only_reorders_scores(self):
        reverse = LEXICAL.LexicalIndex(reversed(self.documents))
        self.assertEqual(reverse.bm25(["cache", "error"]), self.index.bm25(["cache", "error"])[::-1])

    def test_bilingual_example_matches_executable_function(self):
        source = SOURCE.read_text()
        function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "bm25_term")
        expected = ast.get_source_segment(source, function)
        for suffix in (".md", ".en.md"):
            note = (ROOT / "04-search" / ("tfidf-and-bm25" + suffix)).read_text()
            blocks = re.findall(r"```python\n(.*?)\n```", note, re.S)
            self.assertEqual(blocks, [expected])
            self.assertIn("BM25: [1.1465, 1.1817, 0.3008, 0.0]", note)
            self.assertIn("TF-IDF: [1.2685, 0.9808, 0.2877, 0.0]", note)


if __name__ == "__main__":
    unittest.main()
