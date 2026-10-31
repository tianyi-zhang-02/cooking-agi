from html.parser import HTMLParser
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build


class AlgorithmTerms(HTMLParser):
    def __init__(self, markup):
        super().__init__()
        self.terms = []
        self.active_blocks = []
        self.block_counts = []
        self.feed(markup)

    def handle_starttag(self, tag, attributes):
        if tag in build.GLOSS_BLOCKS:
            self.active_blocks.append(0)
        attributes = dict(attributes)
        if "term" in attributes.get("class", "").split():
            self.terms.append(attributes)
            if self.active_blocks:
                self.active_blocks[-1] += 1

    def handle_endtag(self, tag):
        if tag in build.GLOSS_BLOCKS and self.active_blocks:
            self.block_counts.append(self.active_blocks.pop())


class AlgorithmGlossaryTests(unittest.TestCase):
    def setUp(self):
        self.general = build.load_glossary()
        self.terms = build.glossary_for_page(
            SimpleNamespace(section={"group": "algorithms"}), self.general)

    def test_terms_are_scoped_and_complete(self):
        for group in ("career", "reference", "foundations"):
            page = SimpleNamespace(section={"group": group})
            self.assertIs(build.glossary_for_page(page, self.general), self.general)
        scoped = build.load_glossary("glossary.algorithms.tsv")
        self.assertTrue(all(len(term) == 3 and all(term) for term in scoped))
        names = [term[0] for term in self.terms]
        self.assertEqual(len(names), len(set(names)))
        translations = {term[0]: term[1] for term in self.terms}
        self.assertEqual(translations["递归"], "recursion")
        self.assertEqual(translations["迭代"], "iteration")
        self.assertEqual(translations["回溯"], "backtracking")
        self.assertEqual(translations["递归终止条件"], "base case")
        self.assertIn("dynamic programming", translations["动态规划"])

    def test_every_occurrence_has_english(self):
        output, used = build.annotate(
            "<p>递归、回溯、动态规划。</p><p>动态规划也可以用迭代写。递归。</p>",
            self.terms)
        parsed = AlgorithmTerms(output)
        self.assertEqual(parsed.block_counts, [3, 3])
        self.assertEqual([term["zh"] for term in used], ["递归", "回溯", "动态规划", "迭代"])
        self.assertIn('<span class="term-en">（recursion）</span>', output)

    def test_longest_term_is_not_split_into_recursion(self):
        output, used = build.annotate(
            "<p>递归终止条件与调用栈。</p><p>递归栈和递归。</p>", self.terms)
        self.assertEqual([term["zh"] for term in used],
                         ["递归终止条件", "调用栈", "递归栈", "递归"])
        self.assertEqual(len(AlgorithmTerms(output).terms), 4)

    def test_existing_english_is_not_duplicated(self):
        source = "<p>递归（recursion）和记忆化搜索（memoization）。</p><p>再看递归。</p>"
        output, used = build.annotate(source, self.terms)
        self.assertTrue(output.startswith("<p>递归（recursion）和记忆化搜索（memoization）。</p>"))
        self.assertEqual(len(AlgorithmTerms(output).terms), 1)
        self.assertEqual([term["zh"] for term in used], ["递归", "记忆化搜索"])

    def test_headings_links_code_and_controls_remain_unchanged(self):
        source = ('<h2 id="4">递归与迭代</h2><a href="#4">递归</a>'
                  '<pre><code>递归 &lt; 回溯</code></pre>'
                  '<button>递归</button><svg><text>递归</text></svg>')
        output, used = build.annotate(source + "<p>递归</p>", self.terms)
        self.assertTrue(output.startswith(source))
        self.assertEqual([term["zh"] for term in used], ["递归"])

    def test_algorithm_pages_have_glosses_only_in_chinese(self):
        nav = build.load_nav()
        previous_sources = dict(build.BY_SRC)
        try:
            pages, _ = build.discover(nav)
            known = {page.url for page in pages}
            selected = [page for page in pages if page.section["group"] == "algorithms"]
            self.assertEqual(len(selected), 30)
            for page in selected:
                with self.subTest(page=page.url):
                    build.build_page(page, self.general, nav["site"]["repo"], known)
                    parsed = AlgorithmTerms(page.body)
                    if page.lang == "en":
                        self.assertFalse(parsed.terms)
                        self.assertFalse(page.glossary)
                    else:
                        self.assertGreater(len(parsed.terms), 0)
                        self.assertEqual(len(parsed.terms), page.body.count('class="term-en"'))
                        self.assertEqual(len(page.glossary), len({term["zh"] for term in page.glossary}))
                        self.assertTrue(all(term["tabindex"] == "0" and term["data-tip"]
                                            for term in parsed.terms))
                        if page.src.name in ("traversal.md", "backtracking-and-dp.md"):
                            self.assertIn('<span class="term-en">（recursion）</span>', page.body)
        finally:
            build.BY_SRC.clear()
            build.BY_SRC.update(previous_sources)


if __name__ == "__main__":
    unittest.main()
