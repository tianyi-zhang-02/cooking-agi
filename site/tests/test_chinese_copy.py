from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build


class ChineseCopyTests(unittest.TestCase):
    def setUp(self):
        self.terms = [
            ("多头注意力", "multi-head attention", "多组注意力并行计算"),
            ("注意力", "attention", "按相关程度组合信息"),
            ("监督微调", "SFT", "从示范学习"),
            ("强化学习", "RL", "根据回报学习"),
            ("预训练", "pre-training", "学习通用能力"),
        ]

    def test_every_occurrence_uses_longest_term_and_glossary_is_unique(self):
        output, used = build.annotate("<p>多头注意力。多头注意力。</p>", self.terms)
        self.assertEqual(output.count('class="term"'), 2)
        self.assertIn("（multi-head attention）", output)
        self.assertEqual([term["zh"] for term in used], ["多头注意力"])

    def test_compound_word_does_not_consume_variance_annotation(self):
        terms = [("方差", "variance", "偏离均值的平方的期望")]
        output, used = build.annotate("<p>平方差与立方差。</p><p>再看方差。</p>", terms)
        self.assertTrue(output.startswith("<p>平方差与立方差。</p>"))
        self.assertEqual(output.count('class="term"'), 1)
        self.assertIn("（variance）", output)
        self.assertEqual([term["zh"] for term in used], ["方差"])

    def test_all_terms_are_annotated_across_inline_markup_and_paragraphs(self):
        output, used = build.annotate(
            "<p><strong>预训练</strong>之后是<em>监督微调</em>和强化学习。</p>"
            "<p>强化学习再单独讲。</p>", self.terms)
        paragraphs = output.split("</p>")
        self.assertEqual(paragraphs[0].count('class="term"'), 3)
        self.assertEqual(paragraphs[1].count('class="term"'), 1)
        self.assertEqual([term["zh"] for term in used], ["预训练", "监督微调", "强化学习"])

    def test_existing_english_is_not_duplicated(self):
        for label in ("强化学习（reinforcement learning, RL）", "强化学习 (RL)"):
            with self.subTest(label=label):
                source = f"<p>{label}。</p><p>强化学习。</p>"
                output, used = build.annotate(source, self.terms)
                self.assertTrue(output.startswith(f"<p>{label}。</p>"))
                self.assertEqual(output.count('class="term"'), 1)
                self.assertIn('<span class="term-en">（RL）</span>', output)
                self.assertEqual([term["zh"] for term in used], ["强化学习"])

    def test_protected_content_remains_unchanged(self):
        for tag in ("h2", "a", "code", "pre", "button", "label", "summary", "svg"):
            with self.subTest(tag=tag):
                protected = f"<{tag}>强化学习</{tag}>"
                output, used = build.annotate(protected + "<p>强化学习</p>", self.terms)
                self.assertTrue(output.startswith(protected))
                self.assertEqual(output.count('class="term"'), 1)
        for css_class in ("widget", "curriculum-card", "curriculum-hero", "mermaid", "lesson-recipe"):
            protected = f'<div class="{css_class}"><p>强化学习</p></div>'
            output, used = build.annotate(protected + "<p>强化学习</p>", self.terms)
            self.assertTrue(output.startswith(protected))
            self.assertEqual(output.count('class="term"'), 1)

    def test_english_first_explanation_is_not_annotated_inside_parentheses(self):
        terms = [("反事实", "counterfactual", "改变条件，检查结果")]
        for label in ("counterfactual tests（反事实测试）", "Counterfactual tests (反事实测试)"):
            with self.subTest(label=label):
                source = f"<p>这些是 {label}。</p><p>反事实测试。</p>"
                output, used = build.annotate(source, terms)
                self.assertTrue(output.startswith(f"<p>这些是 {label}。</p>"))
                self.assertEqual(output.count('class="term"'), 1)
                self.assertIn('<span class="term-en">（counterfactual）</span>', output)
                self.assertEqual([term["zh"] for term in used], ["反事实"])

    def test_chinese_phrase_extension_keeps_existing_english(self):
        terms = [("反事实", "counterfactual", "改变条件，检查结果")]
        source = "<p>反事实测试（counterfactual tests），然后是反事实分析。</p>"
        output, used = build.annotate(source, terms)
        self.assertTrue(output.startswith("<p>反事实测试（counterfactual tests），然后是"))
        self.assertEqual(output.count('class="term"'), 1)
        self.assertIn('<span class="term-en">（counterfactual）</span>', output)
        self.assertEqual(len(used), 1)

    def test_existing_english_across_inline_tags_is_preserved(self):
        for paragraph in (
            '<p><strong>强化学习</strong>（reinforcement learning, RL）。</p>',
            '<p>强化学习（<em>RL</em>）。</p>',
            '<p><em>RL</em>（<strong>强化学习</strong>）。</p>',
            '<p>强化学习（<code>RL</code>）。</p>',
        ):
            with self.subTest(paragraph=paragraph):
                output, used = build.annotate(paragraph + '<p>强化学习。</p>', self.terms)
                self.assertTrue(output.startswith(paragraph))
                self.assertEqual(output.count('class="term"'), 1)
                self.assertEqual([term["zh"] for term in used], ["强化学习"])

    def test_translation_detection_does_not_cross_blocks_or_line_breaks(self):
        for source in (
            '<p>强化学习</p><p>（RL）</p>',
            '<p>强化学习<br>（RL）</p>',
            '<p>强化学习<br/>（RL）</p>',
            '<table><tr><td>强化学习</td><td>（RL）</td></tr></table>',
        ):
            with self.subTest(source=source):
                output, used = build.annotate(source, self.terms)
                self.assertEqual(output.count('class="term"'), 1)
                self.assertEqual([term["zh"] for term in used], ["强化学习"])

    def test_later_parentheses_only_explain_the_adjacent_occurrence(self):
        output, used = build.annotate('<p>强化学习和强化学习（RL）。</p>', self.terms)
        self.assertEqual(output.count('class="term"'), 1)
        self.assertTrue(output.endswith('和强化学习（RL）。</p>'))
        self.assertEqual([term["zh"] for term in used], ["强化学习"])

    def test_english_first_abbreviation_is_already_a_translation(self):
        terms = [("强化学习", "reinforcement learning, RL", "根据回报学习")]
        paragraph = '<p><strong>RL</strong>（强化学习）。</p>'
        output, used = build.annotate(paragraph + '<p>强化学习。</p>', terms)
        self.assertTrue(output.startswith(paragraph))
        self.assertEqual(output.count('class="term"'), 1)
        self.assertEqual([term["zh"] for term in used], ["强化学习"])

    def test_repeated_terms_in_table_cells_are_annotated(self):
        source = '<table><tr><td>强化学习和强化学习。</td><td>强化学习。</td></tr></table>'
        output, used = build.annotate(source, self.terms)
        self.assertEqual(output.count('class="term"'), 3)
        self.assertEqual([term["zh"] for term in used], ["强化学习"])

    def test_unrelated_parentheses_do_not_hide_a_needed_annotation(self):
        terms = [("反事实", "counterfactual", "改变条件，检查结果")]
        for source in ("<p>用 API（可以先做反事实测试）。</p>",
                       "<p>反事实测试结果（sample A）。</p>"):
            with self.subTest(source=source):
                output, used = build.annotate(source, terms)
                self.assertEqual(output.count('class="term"'), 1)
                self.assertEqual(len(used), 1)

    def test_annotation_is_escaped_and_idempotent(self):
        output, used = build.annotate("<p>词语 &amp; 文本</p>", [("词语", "<term>", 'a "quote" & b')])
        self.assertIn("（&lt;term&gt;）", output)
        self.assertIn("&quot;quote&quot; &amp; b", output)
        self.assertEqual(build.annotate(output, [("词语", "<term>", "解释")])[0], output)

    def test_glossary_and_mobile_copy(self):
        terms = build.load_glossary()
        names = [term[0] for term in terms]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(all(value.strip() for value in term) for term in terms))
        self.assertIn("会话级评估", names)
        self.assertIn("多模态学习", names)
        css = (build.SITE / "static/style.css").read_text()
        self.assertNotRegex(css, r"(?m)^\s*\.term-en\s*\{[^}]*display\s*:\s*none")
        self.assertNotIn('data-english-terms', css)
        self.assertNotIn('.term-en::before', css)
        template = (build.SITE / "template.html").read_text()
        for label in ("skip_label", "menu_label", "theme_label", "toc_label"):
            self.assertIn("{{" + label + "}}", template)


if __name__ == "__main__":
    unittest.main()
