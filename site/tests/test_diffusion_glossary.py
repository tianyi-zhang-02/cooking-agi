from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "site"))
import build


class DiffusionGlossaryTests(unittest.TestCase):
    def setUp(self):
        self.terms = build.load_glossary()

    def test_key_terms_are_explained_once_in_prose(self):
        source = "<p>自回归和扩散语言模型。</p><p>再比较自回归和扩散语言模型。</p>"
        body, used = build.annotate(source, self.terms)
        self.assertEqual(body.count('class="term-en"'), 2)
        self.assertIn("（Autoregressive, AR）", body)
        self.assertIn("（Diffusion Language Model, dLLM）", body)
        self.assertEqual([term["zh"] for term in used], ["自回归", "扩散语言模型"])

    def test_existing_explanations_and_navigation_are_not_annotated_again(self):
        source = (
            '<h2 id="comparison">自回归和扩散语言模型</h2>'
            '<a href="comparison.html">扩散语言模型</a>'
            '<code>自回归</code>'
            '<p>自回归（Autoregressive, AR）与扩散语言模型（Diffusion Language Model, dLLM）。</p>'
            '<p>自回归和扩散语言模型。</p>'
        )
        body, used = build.annotate(source, self.terms)
        self.assertEqual(body, source)
        self.assertEqual(len(used), 2)

    def test_language_model_term_does_not_split_diffusion_term(self):
        body, used = build.annotate("<p>扩散语言模型。</p>", self.terms)
        self.assertEqual([term["zh"] for term in used], ["扩散语言模型"])
        self.assertEqual(body.count('class="term-en"'), 1)

    def test_series_overview_and_comparison_have_full_english_names(self):
        for name in ("README", "ar-and-diffusion", "mdlm-walkthrough"):
            with self.subTest(note=name):
                source = (ROOT / "11-frontier/diffusion-lm" / f"{name}.md").read_text(encoding="utf-8")
                markup, _ = build.render_markdown(source)
                body, used = build.annotate(markup, self.terms)
                self.assertIn("（Autoregressive, AR）", body)
                self.assertIn("（Diffusion Language Model, dLLM）", body)
                self.assertEqual(len(used), len({term["zh"] for term in used}))


if __name__ == "__main__":
    unittest.main()
