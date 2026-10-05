from html.parser import HTMLParser
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build


class GlossaryMarkup(HTMLParser):
    def __init__(self, markup):
        super().__init__()
        self.blocks = []
        self.counts = []
        self.terms = []
        self.feed(markup)

    def handle_starttag(self, tag, attributes):
        if tag in build.GLOSS_BLOCKS:
            self.blocks.append(0)
        attributes = dict(attributes)
        if "term" in attributes.get("class", "").split():
            self.terms.append(attributes)
            if self.blocks:
                self.blocks[-1] += 1

    def handle_endtag(self, tag):
        if tag in build.GLOSS_BLOCKS and self.blocks:
            self.counts.append(self.blocks.pop())


class DeepRLGlossaryTests(unittest.TestCase):
    def setUp(self):
        self.page = SimpleNamespace(section={"group": "deep-rl"})
        self.terms = build.glossary_for_page(self.page, build.load_glossary())

    def test_scoped_terms_do_not_change_other_sections(self):
        general = build.load_glossary()
        for group in ("career", "reference", "foundations"):
            page = SimpleNamespace(section={"group": group})
            self.assertIs(build.glossary_for_page(page, general), general)
        names = [term[0] for term in self.terms]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("自举", names)
        self.assertNotIn("自举", [term[0] for term in general])
        self.assertTrue(all(all(value.strip() for value in term) for term in self.terms))

    def test_hidden_state_uses_environment_meaning_in_rl_only(self):
        scoped = next(term for term in self.terms if term[0] == "隐藏状态")
        general = next(term for term in build.load_glossary() if term[0] == "隐藏状态")
        self.assertIn("环境", scoped[2])
        self.assertIn("递归模型", general[2])
        self.assertNotEqual(scoped, general)

    def test_paragraph_budget_and_first_use(self):
        output, used = build.annotate(
            "<p>回报、<strong>自举</strong>和行为策略。</p>"
            "<p>行为策略与目标策略。再次看回报。</p>", self.terms)
        self.assertEqual(GlossaryMarkup(output).counts, [2, 2])
        self.assertEqual([term["zh"] for term in used], ["回报", "自举", "行为策略", "目标策略"])

    def test_existing_parentheses_and_compound_terms(self):
        source = "<p>广义优势估计（GAE）。</p><p>广义优势估计。</p>"
        output, used = build.annotate(source, self.terms)
        self.assertEqual(output, source)
        self.assertEqual([term["zh"] for term in used], ["广义优势估计"])
        output, used = build.annotate("<p>目标策略和策略评估。</p>", self.terms)
        self.assertEqual([term["zh"] for term in used], ["目标策略", "策略评估"])
        self.assertEqual(len(GlossaryMarkup(output).terms), 2)

    def test_diagrams_and_labs_keep_their_original_labels(self):
        for component in ("drl-flow", "drl-paths", "drl-lab"):
            with self.subTest(component=component):
                diagram = f'<div class="{component}"><p>自举与回报</p></div>'
                output, used = build.annotate(diagram + "<p>自举与回报</p>", self.terms)
                self.assertTrue(output.startswith(diagram))
                self.assertEqual([term["zh"] for term in used], ["自举", "回报"])

    def test_math_is_opaque_without_consuming_first_use(self):
        for formula in (r"$\text{回报}$", r"$$\text{回报}$$",
                        r"\(\text{回报}\)", "\\[\n\\text{回报}\n\\]"):
            with self.subTest(formula=formula):
                source = f"<p>{formula}</p><p>回报</p>"
                output, used = build.annotate(source, self.terms)
                self.assertIn(f"<p>{formula}</p>", output)
                self.assertEqual(len(GlossaryMarkup(output).terms), 1)
                self.assertEqual([term["zh"] for term in used], ["回报"])
                self.assertNotIn("MATHSTASH", output)

    def test_code_and_escaped_text_are_unchanged(self):
        source = r'<pre><code>price = "$回报$"; value &lt; 2 &amp; x &gt; 1</code></pre>'
        output, used = build.annotate(source, self.terms)
        self.assertEqual(output, source)
        self.assertEqual(used, [])

    def test_every_chapter_has_annotations_but_english_stays_clean(self):
        nav = build.load_nav()
        previous_sources = dict(build.BY_SRC)
        try:
            pages, _ = build.discover(nav)
            known = {page.url for page in pages}
            selected = [page for page in pages if page.section["group"] == "deep-rl"]
            self.assertEqual(len(selected), 34)
            for page in selected:
                with self.subTest(page=page.url):
                    build.build_page(page, build.load_glossary(), nav["site"]["repo"], known)
                    parsed = GlossaryMarkup(page.body)
                    if page.lang == "en":
                        self.assertFalse(parsed.terms)
                        self.assertFalse(page.glossary)
                        self.assertEqual(build.glossary_html(page), "")
                    else:
                        self.assertGreater(len(parsed.terms), 0)
                        self.assertTrue(all(count <= 2 for count in parsed.counts))
                        self.assertTrue(all(term["tabindex"] == "0" and term["data-tip"]
                                            for term in parsed.terms))
                        self.assertIn("本页术语", build.glossary_html(page))
                    self.assertNotIn("MATHSTASH", page.body)
        finally:
            build.BY_SRC.clear()
            build.BY_SRC.update(previous_sources)


if __name__ == "__main__":
    unittest.main()
