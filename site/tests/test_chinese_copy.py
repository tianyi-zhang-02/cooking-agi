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

    def test_first_mention_and_longest_term(self):
        output, used = build.annotate("<p>多头注意力。多头注意力。</p>", self.terms)
        self.assertEqual(output.count('class="term"'), 1)
        self.assertIn("（multi-head attention）", output)
        self.assertEqual([term["zh"] for term in used], ["多头注意力"])

    def test_compound_word_does_not_consume_variance_annotation(self):
        terms = [("方差", "variance", "偏离均值的平方的期望")]
        output, used = build.annotate("<p>平方差与立方差。</p><p>再看方差。</p>", terms)
        self.assertTrue(output.startswith("<p>平方差与立方差。</p>"))
        self.assertEqual(output.count('class="term"'), 1)
        self.assertIn("（variance）", output)
        self.assertEqual([term["zh"] for term in used], ["方差"])

    def test_paragraph_limit_survives_inline_markup(self):
        output, used = build.annotate(
            "<p><strong>预训练</strong>之后是<em>监督微调</em>和强化学习。</p>"
            "<p>强化学习再单独讲。</p>", self.terms)
        paragraphs = output.split("</p>")
        self.assertEqual(paragraphs[0].count('class="term"'), 2)
        self.assertEqual(paragraphs[1].count('class="term"'), 1)
        self.assertEqual([term["zh"] for term in used], ["预训练", "监督微调", "强化学习"])

    def test_existing_english_is_not_duplicated(self):
        for label in ("强化学习（reinforcement learning, RL）", "强化学习 (RL)"):
            with self.subTest(label=label):
                source = f"<p>{label}。</p><p>强化学习。</p>"
                output, used = build.annotate(source, self.terms)
                self.assertEqual(output, source)
                self.assertEqual([term["zh"] for term in used], ["强化学习"])

    def test_protected_content_does_not_consume_first_mention(self):
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
                self.assertEqual(output, source)
                self.assertEqual([term["zh"] for term in used], ["反事实"])

    def test_chinese_phrase_extension_keeps_existing_english(self):
        terms = [("反事实", "counterfactual", "改变条件，检查结果")]
        source = "<p>反事实测试（counterfactual tests），然后是反事实分析。</p>"
        output, used = build.annotate(source, terms)
        self.assertEqual(output, source)
        self.assertEqual(len(used), 1)

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
