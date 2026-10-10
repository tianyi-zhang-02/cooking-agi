from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build
import paritycheck


class Elements(HTMLParser):
    def __init__(self, markup):
        super().__init__()
        self.elements = []
        self.feed(markup)

    def handle_starttag(self, tag, attributes):
        self.elements.append((tag, dict(attributes)))


class LanguageEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_nav, cls.old_sources = build.NAV.copy(), build.BY_SRC.copy()
        cls.nav = build.load_nav()
        cls.pages, cls.sections = build.discover(cls.nav)
        cls.known = {page.url for page in cls.pages}
        cls.home = build.BY_SRC['README.md']

    @classmethod
    def tearDownClass(cls):
        build.NAV.clear()
        build.NAV.update(cls.old_nav)
        build.BY_SRC.clear()
        build.BY_SRC.update(cls.old_sources)

    def test_repository_default_and_home_are_english(self):
        self.assertEqual(self.home['en'].src, build.ROOT / 'README.md')
        self.assertEqual(self.home['en'].url, 'index.html')
        self.assertEqual(self.home['en'].title, 'AGI Study Notes')
        self.assertEqual(self.home['zh'].src, build.ROOT / 'README.zh.md')
        self.assertEqual(self.home['zh'].url, 'index.zh.html')
        self.assertEqual(self.home['zh'].title, 'AGI 学习笔记')
        self.assertTrue(all(page.kind == 'home' for page in self.home.values()))
        self.assertEqual(len([page for page in self.pages if page.kind == 'home']), 2)

    def test_plain_html_has_full_english_home_and_chinese_link(self):
        for language, target in [('en', 'index.zh.html'), ('zh', 'index.html')]:
            page = self.home[language]
            build.build_page(page, [], self.nav['site']['repo'], self.known)
            output = build.assemble(page, self.sections, [], self.nav, 'test',
                                    (build.SITE / 'template.html').read_text())
            elements = Elements(output).elements
            self.assertIn(('html', {'lang': 'en' if language == 'en' else 'zh-Hans',
                                     'class': 'lang-' + language}), elements)
            switches = [attrs for tag, attrs in elements if tag == 'a' and attrs.get('class') == 'lang-btn']
            self.assertEqual([attrs['href'] for attrs in switches], [target])
            self.assertIn('ML and LLM notes' if language == 'en' else '学 AI 时', output)
            self.assertIn('class="hero"', output)
            self.assertNotIn('{{', output)

    def test_home_navigation_stays_in_its_language(self):
        for language in ['en', 'zh']:
            page = build.BY_SRC['learn/README.md'][language]
            home = 'index.html' if language == 'en' else 'index.zh.html'
            self.assertIn(f'href="../{home}"', build.tabs_html(page))
            self.assertEqual(build.page_category(self.home[language]), 'home')

    def test_existing_articles_keep_their_urls(self):
        for language, suffix in [('zh', ''), ('en', '.en')]:
            self.assertEqual(build.BY_SRC['career/journey.md'][language].url, f'career/journey{suffix}.html')
            self.assertEqual(build.BY_SRC['learn/README.md'][language].url, f'learn/index{suffix}.html')

    def test_home_readmes_resolve_to_pages_not_github_sources(self):
        page = build.BY_SRC['learn/README.md']['en']
        for source, target in [('README.md', 'index.html'), ('README.en.md', 'index.html'),
                               ('README.zh.md', 'index.zh.html')]:
            markup = f'<a href="../{source}">Home</a>'
            self.assertEqual(build.rewrite_links(markup, page, self.nav['site']['repo'], self.known),
                             f'<a href="../{target}">Home</a>')

    def test_github_source_fallback_keeps_section_and_line_anchors(self):
        page = build.BY_SRC['learn/README.md']['en']
        for source, target in [
            ('../CONTRIBUTING.md#english', 'blob/main/CONTRIBUTING.md#english'),
            ('../site/build.py#L1', 'blob/main/site/build.py#L1'),
            ('../templates/#readme', 'tree/main/templates#readme'),
        ]:
            output = build.rewrite_links(f'<a href="{source}">Read</a>', page,
                                         self.nav['site']['repo'], self.known)
            self.assertEqual(output,
                             f'<a href="https://github.com/{self.nav["site"]["repo"]}/{target}">Read</a>')

    def test_old_english_home_is_an_explicit_english_redirect(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(build, 'OUT', Path(folder)):
            build.write_redirects({'redirects': {'index.en.html': 'index.html'}}, self.known)
            output = (Path(folder) / 'index.en.html').read_text()
            self.assertIn('url=index.html?lang=en', output)
            self.assertIn('location.hash', output)
            self.assertIn('href="index.html?lang=en"', output)

    def test_readme_parity_checks_actual_bilingual_content(self):
        english = (build.ROOT / 'README.md').read_text()
        chinese = (build.ROOT / 'README.zh.md').read_text()
        self.assertEqual(paritycheck.shape(english), paritycheck.shape(chinese))
        self.assertGreater(paritycheck.shape(chinese)['h2'], 3)
        self.assertIn('main README', (build.ROOT / 'README.en.md').read_text())

    def test_readmes_offer_the_same_routes(self):
        routes = {}
        for language, filename in [('en', 'README.md'), ('zh', 'README.zh.md')]:
            text = (build.ROOT / filename).read_text()
            route_section = re.split(r'(?m)^## ', text)[1]
            self.assertEqual(len(re.findall(r'(?m)^- \*\*\[', route_section)), 4)
            local = {target.split('#')[0] for target in re.findall(r'\]\(([^)]+)\)', text)
                     if not target.startswith(('https:', 'http:'))}
            for target in local:
                self.assertTrue((build.ROOT / target).is_file(), target)
            routes[language] = {target.replace('.en.md', '.md') for target in local
                                if target not in {'README.md', 'README.zh.md'}}
            self.assertTrue({'learn/README.md', 'interview/README.md',
                             'practice/README.md', 'career/README.md',
                             'learn/coverage.md'} <= routes[language])
        self.assertEqual(routes['en'], routes['zh'])

    def test_readme_site_links_resolve_to_each_language(self):
        for language in ['en', 'zh']:
            page = self.home[language]
            build.build_page(page, [], self.nav['site']['repo'], self.known)
            links = {attrs['href'] for tag, attrs in Elements(page.body).elements
                     if tag == 'a' and 'href' in attrs}
            suffix = '.en.html' if language == 'en' else '.html'
            for chapter in ['learn', 'interview', 'practice', 'career']:
                self.assertIn(f'{chapter}/index{suffix}', links)
            self.assertIn('learn/coverage' + suffix, links)
            self.assertFalse(any('quant/' in target or 'discussions/' in target
                                 for target in links))

    def test_coverage_summary_does_not_claim_global_completion(self):
        for filename, heading, status in [
            ('learn/coverage.md', '## 这轮先整理什么', '不代表全站已经核对完'),
            ('learn/coverage.en.md', '## What this update focuses on', 'not a completed site-wide audit'),
        ]:
            text = (build.ROOT / filename).read_text()
            self.assertIn(heading, text)
            self.assertIn(status, text)
            self.assertIn('../README.zh.md', text)
            self.assertIn('../README.md', text)
            self.assertIn('{#current-update}', text)

    def test_old_chinese_home_anchors_remain_valid_on_the_default_home(self):
        page = self.home['en']
        build.build_page(page, [], self.nav['site']['repo'], self.known)
        anchors = {attrs['id'] for tag, attrs in Elements(page.body).elements if 'id' in attrs}
        self.assertTrue({f'_{number}' for number in range(1, 6)} <= anchors)
        self.assertIn('start-here', anchors)

    def test_guide_is_hidden_without_javascript_and_is_not_a_modal(self):
        template = (build.SITE / 'template.html').read_text()
        guide = next(attrs for tag, attrs in Elements(template).elements if attrs.get('id') == 'language-guide')
        self.assertIn('hidden', guide)
        self.assertEqual(guide['aria-labelledby'], 'language-guide-title')
        self.assertNotIn('aria-modal', guide)
        self.assertIn('data-site-root="{{prefix}}"', template)
        self.assertIn('static/language.css', template)
        self.assertLess(template.index('static/language.js'), template.index('static/style.css'))

    @unittest.skipUnless(shutil.which('node'), 'Node is required for browser behavior tests')
    def test_language_preferences_and_guide_behavior(self):
        result = subprocess.run([shutil.which('node'), str(build.SITE / 'tests/language-behavior.cjs')],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
