from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build


class PageTitleTests(unittest.TestCase):
    def test_explicit_heading_id_is_not_title_text(self):
        self.assertEqual(build.split_heading_id('A title {#stable-id}'), ('A title', 'stable-id'))
        self.assertEqual(build.split_heading_id('中文标题 {#中文锚点}'), ('中文标题', '中文锚点'))

    def test_braces_in_title_remain_text(self):
        for title in ['The set {a, b}', 'A title {not-an-id}', 'A {#fragment} in the middle']:
            self.assertEqual(build.split_heading_id(title), (title, None))

    def test_read_title_omits_heading_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'note.md'
            path.write_text('# A useful title {#stable-id}\n\nBody')
            self.assertEqual(build.read_title(path), 'A useful title')

    def test_html_and_filename_fallbacks_remain_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'note.md'
            path.write_text('<h1>A &amp; B</h1>')
            self.assertEqual(build.read_title(path), 'A & B')
            path.write_text('No heading')
            self.assertEqual(build.read_title(path), 'note')

    def test_both_agent_pages_keep_anchor_without_showing_source_syntax(self):
        old_nav, old_sources = build.NAV.copy(), build.BY_SRC.copy()
        try:
            navigation = build.load_nav()
            pages, sections = build.discover(navigation)
            known = {page.url for page in pages}
            for language, anchor in [('zh', 'agents-frontier-api-serve'),
                                     ('en', 'agents-frontier-api-or-self-hosted')]:
                with self.subTest(language=language):
                    page = build.BY_SRC['10-agents/model-choice.md'][language]
                    self.assertNotIn('{#', page.title)
                    build.build_page(page, [], navigation['site']['repo'], known)
                    self.assertNotIn('{#', page.title)
                    self.assertIn(f'<span id="{anchor}"></span>', page.body)
                    output = build.assemble(page, sections, navigation, 'test',
                                            (build.SITE / 'template.html').read_text())
                    self.assertNotIn('{#' + anchor + '}', output)
        finally:
            build.NAV.clear()
            build.NAV.update(old_nav)
            build.BY_SRC.clear()
            build.BY_SRC.update(old_sources)


if __name__ == '__main__':
    unittest.main()
