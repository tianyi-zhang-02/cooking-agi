import copy
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build


class NoteReadingTests(unittest.TestCase):
    def page(self, language):
        suffix = '.en' if language == 'en' else ''
        section = {'zh': '核心机制', 'en': 'Core mechanisms', 'dir': '00-foundations/core', 'pages': []}
        page = SimpleNamespace(lang=language, src=build.ROOT / f'00-foundations/core/attention{suffix}.md',
                               section=section, url=f'00-foundations/core/attention{suffix}.html',
                               kind='article', title='Attention <&>', read_minutes=10, position=2,
                               section_count=3, reviewed='', rel=lambda url: '../../' + url,
                               previous=None, next=None)
        overview = copy.copy(page)
        overview.url = f'00-foundations/core/index{suffix}.html'
        overview.src = build.ROOT / f'00-foundations/core/README{suffix}.md'
        following = copy.copy(page)
        following.url = f'00-foundations/core/next{suffix}.html'
        page.next = following
        section['pages'] = [{language: overview}, {language: page}, {language: following}]
        return page

    def test_both_languages_link_back_to_their_own_chapter(self):
        for language in ('zh', 'en'):
            page = self.page(language)
            suffix = '.en' if language == 'en' else ''
            link = f'href="../../00-foundations/core/index{suffix}.html"'
            header = build.page_header_html(page)
            self.assertIn(link, header)
            self.assertIn(link, build.page_nav_html(page))
            self.assertIn('data-reading-note="00-foundations/core/attention.md"', header)
            self.assertIn('data-note-resume hidden', header)
            self.assertIn('data-side-save', header)
            self.assertIn('Attention &lt;&amp;&gt;', header)
            self.assertIn('本章 2 / 3 篇' if language == 'zh' else 'Note 2 of 3', header)

    def test_overviews_do_not_get_long_article_controls(self):
        page = self.page('zh')
        page.kind = 'index'
        self.assertNotIn('data-reading-note', build.page_header_html(page))

    def test_missing_overview_does_not_create_broken_link(self):
        page = self.page('zh')
        page.section['pages'] = []
        self.assertNotIn('index.html', build.page_header_html(page))
        self.assertNotIn('page-turn-context', build.page_nav_html(page))

    def test_template_loads_reading_script(self):
        self.assertIn('static/note-reading.js', (build.SITE / 'template.html').read_text())

    @unittest.skipUnless(shutil.which('node'), 'Node is required for reading behavior tests')
    def test_browser_behavior(self):
        for filename in ('note-reading-behavior.cjs', 'sidebar-library-behavior.cjs'):
            with self.subTest(filename=filename):
                result = subprocess.run([shutil.which('node'), str(Path(__file__).with_name(filename))],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
