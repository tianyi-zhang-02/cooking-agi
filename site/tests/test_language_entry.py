from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urljoin, urlsplit

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
        for language, target in [('en', 'index.zh.html'), ('zh', 'index.html?lang=en')]:
            page = self.home[language]
            build.build_page(page, [], self.nav['site']['repo'], self.known)
            output = build.assemble(page, self.sections, self.nav, 'test',
                                    (build.SITE / 'template.html').read_text())
            elements = Elements(output).elements
            self.assertIn(('html', {'lang': 'en' if language == 'en' else 'zh-Hans',
                                     'class': 'lang-' + language}), elements)
            switches = [attrs for tag, attrs in elements if tag == 'a' and attrs.get('class') == 'lang-btn']
            self.assertEqual([attrs['href'] for attrs in switches], [target])
            self.assertIn('ML and LLM notes' if language == 'en' else '这里整理了中英文的', output)
            self.assertIn('class="hero"', output)
            self.assertNotIn('{{', output)

    def test_home_navigation_stays_in_its_language(self):
        for language in ['en', 'zh']:
            page = build.BY_SRC['learn/README.md'][language]
            home = 'index.html?lang=en' if language == 'en' else 'index.zh.html'
            self.assertIn(f'href="../{home}"', build.tabs_html(page))
            self.assertEqual(build.page_category(self.home[language]), 'home')

    def test_home_hides_repository_only_links_but_keeps_them_in_readmes(self):
        for language in ['en', 'zh']:
            page = self.home[language]
            source = page.src.read_text()
            repository_block = re.search(
                r'<!-- repository-only -->(.*?)<!-- /repository-only -->',
                source, re.S).group(1)
            self.assertIn('badge.svg', repository_block)
            self.assertIn(f'.github/media/hero.{language}.svg', repository_block)
            self.assertIn('https://tianyi-zhang-02.github.io/cooking-agi/', repository_block)
            build.build_page(page, [], self.nav['site']['repo'], self.known)
            self.assertNotIn('repository-only', page.body)
            self.assertNotIn('badge.svg', page.body)
            self.assertNotIn('.github/media/', page.body)
            intro = page.body.split('<h2', 1)[0]
            self.assertNotIn('<a ', intro)
            self.assertIn('English or 中文' if language == 'en' else '中文或 English', intro)

    def test_home_routes_precede_release_details_and_fit_two_columns(self):
        for language in ['en', 'zh']:
            page = self.home[language]
            build.build_page(page, [], self.nav['site']['repo'], self.known)
            table = re.search(r'<table>.*?</table>', page.body, re.S).group(0)
            rows = re.findall(r'<tr>(.*?)</tr>', table, re.S)
            self.assertEqual(len(rows), 6)
            for row in rows:
                self.assertEqual(len(re.findall(r'<t[hd]>', row)), 2)
            self.assertLess(page.body.index('</table>'), page.body.index('v1.0'))
            disclosures = re.findall(r'<details[^>]*>(.*?)</details>', page.body, re.S)
            self.assertTrue(any('CODEOWNER' in content and 'AI' in content
                                for content in disclosures))

    def test_existing_articles_keep_their_urls(self):
        for language, suffix in [('zh', ''), ('en', '.en')]:
            self.assertEqual(build.BY_SRC['career/journey.md'][language].url, f'career/journey{suffix}.html')
            self.assertEqual(build.BY_SRC['learn/README.md'][language].url, f'learn/index{suffix}.html')

    def test_question_bank_links_to_foundations_in_its_language(self):
        for language, suffix, label in [('zh', '', '基础与原理'), ('en', '.en', 'Foundations')]:
            with self.subTest(language=language):
                page = build.BY_SRC['interview/questions.md'][language]
                build.build_page(page, [], self.nav['site']['repo'], self.known)
                self.assertIn(f'<a href="../learn/index{suffix}.html">{label}</a>', page.body)

    def test_home_readmes_resolve_to_pages_not_github_sources(self):
        page = build.BY_SRC['learn/README.md']['en']
        for source, target in [('README.md', 'index.html?lang=en'), ('README.en.md', 'index.html?lang=en'),
                               ('README.zh.md', 'index.zh.html')]:
            markup = f'<a href="../{source}">Home</a>'
            self.assertEqual(build.rewrite_links(markup, page, self.nav['site']['repo'], self.known),
                             f'<a href="../{target}">Home</a>')

    def test_english_home_links_are_explicit_but_canonical_is_clean(self):
        for source in ['README.md', 'README.en.md']:
            page = self.home['zh']
            markup = f'<a href="{source}#start-here">English</a>'
            self.assertEqual(build.rewrite_links(markup, page, self.nav['site']['repo'], self.known),
                             '<a href="index.html?lang=en#start-here">English</a>')
        for page in [self.home['en'], self.home['zh'], build.BY_SRC['learn/README.md']['en']]:
            build.build_page(page, [], self.nav['site']['repo'], self.known)
            output = build.assemble(page, self.sections, self.nav, 'test',
                                    (build.SITE / 'template.html').read_text())
            for tag, attrs in Elements(output).elements:
                href = attrs.get('href', '')
                if tag == 'link' and attrs.get('rel') == 'canonical':
                    self.assertNotIn('?', href)
                target = urlsplit(urljoin('https://example.org/' + page.url, href))
                if tag == 'a' and not href.startswith('#') and target.netloc == 'example.org' and target.path == '/index.html':
                    self.assertIn('lang=en', target.query, href)

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

    def test_redirect_to_a_home_section_keeps_language_and_drops_old_anchor(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(build, 'OUT', Path(folder)):
            build.write_redirects({'redirects': {
                'community/old.en.html': 'index.html#contributors',
                'community/old.html': 'index.zh.html#contributors',
            }}, self.known)
            english = (Path(folder) / 'community/old.en.html').read_text()
            chinese = (Path(folder) / 'community/old.html').read_text()
            self.assertIn('url=../index.html?lang=en#contributors', english)
            self.assertIn('url=../index.zh.html#contributors', chinese)
            for output in (english, chinese):
                self.assertNotIn('location.hash', output)
            with self.assertRaises(ValueError):
                build.write_redirects({'redirects': {'old.html': 'missing.html#contributors'}}, self.known)

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
            self.assertEqual(len(re.findall(r'(?m)^- \*\*\[', route_section)), 5)
            local = {target.split('#')[0] for target in re.findall(r'\]\(([^)]+)\)', text)
                     if not target.startswith(('https:', 'http:', '#'))}
            for target in local:
                self.assertTrue((build.ROOT / target).is_file(), target)
            routes[language] = {target.replace('.en.md', '.md') for target in local
                                if target not in {'README.md', 'README.zh.md'}}
            self.assertTrue({'learn/README.md', 'interview/README.md',
                             'practice/README.md', 'career/README.md',
                             'learn/coverage.md', 'learn/technology-timeline.md'} <= routes[language])
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

    def test_new_home_details_do_not_shift_legacy_chinese_anchors(self):
        page = self.home['zh']
        build.build_page(page, [], self.nav['site']['repo'], self.known)
        expected = [('_1', '从哪里开始'), ('_2', '内容一览'), ('roadmap', '路线图'), ('contributors-reviewers', '贡献者与审稿人（Contributors / Reviewers）'), ('_3', '加入我们！'),
                    ('_4', '本地阅读与检查'), ('_5', '为什么整理这些'), ('_6', '更新记录')]
        self.assertEqual([(heading['id'], heading['name']) for heading in page.toc], expected)
        self.assertIn('id="how-to-use-the-notes"', page.body)
        self.assertIn('id="whats-in-each-section"', page.body)

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
