import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build


class ReadingRoomTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(build.NAV), patch.dict(build.BY_SRC):
            cls.nav = build.load_nav()
            pages, sections = build.discover(cls.nav)
            cls.known = {page.url for page in pages}
            cls.home = {}
            for language in ('zh', 'en'):
                page = build.BY_SRC['README.md'][language]
                build.build_page(page, [], cls.nav['site']['repo'], cls.known)
                cls.home[language] = build.assemble(page, sections, cls.nav, 'test', (build.SITE / 'template.html').read_text())
        cls.styles = (build.SITE / 'static/reading-room.css').read_text()

    def test_cover_and_routes_are_localized(self):
        for language, output in self.home.items():
            with self.subTest(language=language):
                cover = build.home_data()['cover']
                self.assertIn(cover['eyebrow_' + language], output)
                for line in cover['title_' + language]:
                    self.assertIn(line, output)
                self.assertIn('class="side-primary"', output)
                self.assertIn('class="side-search" data-search-open', output)
                self.assertIn('static/reading-room.css', output)

    def test_old_home_anchors_and_complete_example_are_preserved(self):
        for output in self.home.values():
            for anchor in ('_1', '_2', 'start-here', 'first-calculation', 'whats-inside',
                           'how-the-notes-are-written', 'interview-coach', *build.HOME_LEGACY_IDS):
                self.assertIn(f'id="{anchor}"', output)
            for marker in ('data-home-example=', 'data-part="predict"', 'data-part="loss"',
                           'data-part="update"', 'data-rate="0.1"', 'data-rate="1"'):
                self.assertIn(marker, output)

    def test_home_has_no_forced_hidden_learning_steps(self):
        for output in self.home.values():
            route = re.search(r'<section class="home-block home-route".*?</section>', output, re.S).group()
            self.assertEqual(route.count('<li>'), 6)
            self.assertNotIn('hidden', route)
            self.assertNotIn('<details', route)
            self.assertLess(output.index('id="whats-inside"'), output.index('id="first-calculation"'))

    def test_layout_keeps_mobile_controls_and_text_preferences(self):
        self.assertIn('--site-header-height: 64px', self.styles)
        self.assertIn('body .topbar-brand .brand-full { position: static;', self.styles)
        self.assertIn('grid-template-columns: repeat(2, minmax(0, 1fr))', self.styles)
        self.assertIn('var(--text-scale, 1)', self.styles)
        self.assertIn('height: auto; min-height: var(--home-cover-height)', self.styles)
        self.assertNotIn('localStorage', self.styles)
        self.assertNotIn('text-overflow', self.styles)
        self.assertNotIn('line-clamp', self.styles)

    def test_sky_measures_its_displayed_area(self):
        script = (build.SITE / 'static/sky.js').read_text()
        self.assertIn('canvas.getBoundingClientRect()', script)
        self.assertIn('prefers-reduced-motion: reduce', script)
        self.assertIn('document.hidden', script)


if __name__ == '__main__':
    unittest.main()
