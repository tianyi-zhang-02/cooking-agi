import ast
import importlib.util
import math
from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "site"))
import build

SPEC = importlib.util.spec_from_file_location("bert_example_loss", ROOT / "00-foundations/code/mlm_contracts.py")
MLM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MLM)


class BertExampleTests(unittest.TestCase):
    def sources(self):
        for suffix in ("", ".en"):
            yield suffix or ".zh", (ROOT / f"00-foundations/core/bert{suffix}.md").read_text()

    def test_embedding_addition_uses_matching_dimensions(self):
        pattern = r"`(\[[\d, ]+\]) \+ (\[[\d, ]+\]) \+ (\[[\d, ]+\]) = (\[[\d, ]+\])`"
        for language, source in self.sources():
            with self.subTest(language=language):
                matched = re.search(pattern, source)
                self.assertIsNotNone(matched)
                vectors = [ast.literal_eval(value) for value in matched.groups()]
                self.assertEqual([len(vector) for vector in vectors], [3, 3, 3, 3])
                self.assertEqual([sum(values) for values in zip(*vectors[:3])], vectors[3])

    def test_classification_example_matches_cross_entropy(self):
        for language, source in self.sources():
            with self.subTest(language=language):
                example = source.split("{#classify-example}", 1)[1].split("{#answer-span}", 1)[0]
                logits = ast.literal_eval(re.search(r"(?:输出是|Suppose they are) `(\[[\d, ]+\])`", example).group(1))
                probability, loss = map(float, re.findall(r"\\approx(\d+(?:\.\d+)?)", example))
                expected = MLM.masked_cross_entropy([logits], [1])
                self.assertAlmostEqual(loss, expected, places=6)
                self.assertAlmostEqual(probability, math.exp(-expected), places=6)
                self.assertIn("[2, 768]", example)

    def test_accuracy_example_does_not_hide_minority_recall(self):
        labels = [0] * 95 + [1] * 5
        predictions = [0] * len(labels)
        correct = sum(predicted == actual for predicted, actual in zip(predictions, labels))
        recovered = sum(predicted == actual == 1 for predicted, actual in zip(predictions, labels))
        accuracy = correct / len(labels)
        recall = recovered / labels.count(1)
        for language, source in self.sources():
            with self.subTest(language=language):
                checks = source.split("{#checks}", 1)[1]
                self.assertRegex(checks, rf"{accuracy:.0%} (?:accuracy|准确率（accuracy）)")
                self.assertIn(f"{recall:.0%}", checks)

    def test_examples_have_matching_direct_links_and_rendered_math(self):
        anchors = ["training-step", "classify-example", "answer-span", "adaptation", "retrieval-use"]
        for language, source in self.sources():
            with self.subTest(language=language):
                body, _ = build.render_markdown(source)
                for anchor in anchors:
                    self.assertEqual(body.count(f'id="{anchor}"'), 1)
                self.assertIn('href="#classify-example"', body)
                self.assertIn("run_classifier.py", body)
                self.assertIn("D19-1410", body)
                self.assertIn("0.126928", body)
                self.assertNotIn("{#classify-example}", body)


if __name__ == "__main__":
    unittest.main()
