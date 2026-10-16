from contextlib import redirect_stdout
from fractions import Fraction
import io
from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "site"))
import build
import paritycheck


class DataSupervisionTests(unittest.TestCase):
    def test_bilingual_structure_anchors_and_examples(self):
        for chapter in ("01-data-and-feedback/data-as-supervision", "practice/post-training/data-pipeline"):
            chinese = (ROOT / f"{chapter}.md").read_text()
            english = (ROOT / f"{chapter}.en.md").read_text()
            self.assertEqual(paritycheck.shape(chinese), paritycheck.shape(english))
            self.assertEqual(re.findall(r"\{#([^}]+)\}", chinese), re.findall(r"\{#([^}]+)\}", english))
            self.assertEqual(re.findall(r"```python\n(.*?)```", chinese, re.S),
                             re.findall(r"```python\n(.*?)```", english, re.S))
            self.assertFalse(paritycheck.source_issues(chinese))
            self.assertFalse(paritycheck.source_issues(english))

    def test_filtering_example_is_executable_and_exact(self):
        for suffix in ("", ".en"):
            article = (ROOT / f"01-data-and-feedback/data-as-supervision{suffix}.md").read_text()
            examples = re.findall(r"```python\n(.*?)```", article, re.S)
            self.assertEqual(len(examples), 1)
            namespace, output = {}, io.StringIO()
            with redirect_stdout(output):
                exec(compile(examples[0], "filtering-example", "exec"), namespace)
            self.assertEqual(output.getvalue(), "Before: 10.0%; after: 2.4%\n")
            self.assertEqual(namespace["before_share"], Fraction(1, 10))
            self.assertEqual(namespace["after_share"], Fraction(2, 83))
            self.assertLess(namespace["after_share"], namespace["before_share"])
            self.assertTrue(all(0 <= count <= namespace["before"][category]
                                for category, count in namespace["after"].items()))

    def test_selection_and_normalization_counterexamples(self):
        proportionate_retention = {"supported": 45, "missing_evidence": 5}
        self.assertEqual(Fraction(proportionate_retention["missing_evidence"],
                                  sum(proportionate_retention.values())), Fraction(1, 10))
        example_mean = (Fraction(1, 5) + 1) / 2
        token_mean = (Fraction(1, 5) + 3) / 4
        self.assertEqual(example_mean, Fraction(3, 5))
        self.assertEqual(token_mean, Fraction(4, 5))
        self.assertNotEqual(example_mean, token_mean)

    def test_diagram_is_localized_and_readable_without_interaction(self):
        for suffix in ("", ".en"):
            article = (ROOT / f"01-data-and-feedback/data-as-supervision{suffix}.md").read_text()
            figures = re.findall(r'<figure class="worked-update".*?</figure>', article, re.S)
            self.assertEqual(len(figures), 1)
            self.assertIn("aria-label=", figures[0])
            self.assertEqual(figures[0].count("<li>"), 3)
            self.assertIn("<figcaption>", figures[0])
            if suffix:
                self.assertNotRegex(figures[0], r"[\u4e00-\u9fff]")
                self.assertIn("Conflicting evidence", figures[0])
            else:
                self.assertIn("资料冲突", figures[0])
                self.assertNotIn("Conflicting evidence", figures[0])
            self.assertNotIn("<button", figures[0])

    def test_primary_sources_and_math_match(self):
        articles = [(ROOT / f"01-data-and-feedback/data-as-supervision{suffix}.md").read_text()
                    for suffix in ("", ".en")]
        expected_sources = {
            "https://arxiv.org/html/2203.02155v1",
            "https://arxiv.org/html/2305.18290v3",
            "https://arxiv.org/html/2501.12948v2",
            "https://arxiv.org/html/2305.11206v1",
            "https://arxiv.org/html/2411.15124v3",
            "https://arxiv.org/html/2503.14476v2",
            "https://arxiv.org/html/2511.19399v3",
            "https://arxiv.org/html/2609.29421v2",
            "https://huggingface.co/docs/trl/v1.16.0/en/sft_trainer",
            "https://huggingface.co/docs/trl/v1.16.0/en/chat_templates",
            "https://docs.pytorch.org/docs/2.14/notes/randomness.html",
        }
        for article in articles:
            self.assertEqual(set(re.findall(r"\]\((https://[^)]+)\)", article)), expected_sources)
            self.assertIn(r"\alpha_i\geq 0", article)
        self.assertEqual(re.findall(r"\$\$(.*?)\$\$", articles[0], re.S),
                         re.findall(r"\$\$(.*?)\$\$", articles[1], re.S))

    def test_new_chapter_is_published_once_and_connected(self):
        previous_nav, previous_sources = build.NAV.copy(), build.BY_SRC.copy()
        try:
            pages, _ = build.discover(build.load_nav())
            for suffix in ("", ".en"):
                relative = f"01-data-and-feedback/data-as-supervision{suffix}.md"
                matches = [page for page in pages if page.src == ROOT / relative]
                self.assertEqual(len(matches), 1)
                self.assertEqual(matches[0].section["group"], "data-retrieval")
                for entry in ("01-data-and-feedback/README", "05-post-training/sft-and-its-ceiling",
                              "practice/post-training/README", "practice/post-training/data-pipeline"):
                    self.assertIn(f"data-as-supervision{suffix}.md", (ROOT / f"{entry}{suffix}.md").read_text())
        finally:
            build.NAV.clear()
            build.NAV.update(previous_nav)
            build.BY_SRC.clear()
            build.BY_SRC.update(previous_sources)


if __name__ == "__main__":
    unittest.main()
