"""The reading layout: quieter structure on every page but the cover, with the site's own touches kept."""
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "site"))
import build

READING = (ROOT / "site/static/reading.css").read_text()
TEMPLATE = (ROOT / "site/template.html").read_text()


class ReadingLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(build.NAV), patch.dict(build.BY_SRC):
            cls.nav = build.load_nav()
            pages, cls.sections = build.discover(cls.nav)
            known = {page.url for page in pages}
            cls.output = {}
            for key, source in (("home", "README.md"), ("note", "00-foundations/core/multi-head-attention.md")):
                for language in ("zh", "en"):
                    page = build.BY_SRC[source][language]
                    build.build_page(page, [], cls.nav["site"]["repo"], known)
                    cls.output[key, language] = build.assemble(page, cls.sections, cls.nav, "test", TEMPLATE)

    def test_reading_pages_keep_the_sky_and_the_ember_touches(self):
        for language in ("zh", "en"):
            with self.subTest(language=language):
                for key in ("home", "note"):
                    self.assertIn('<canvas class="sky"', self.output[key, language])
                    self.assertIn("static/sky.js", self.output[key, language])
        rules = re.sub(r"/\*.*?\*/", "", READING, flags=re.S)
        for touch in ("--ember:", "reading-progress", "study-buddy", "sb-bubble", "widget-kicker", "canvas.sky"):
            with self.subTest(touch=touch):
                self.assertNotIn(touch, rules)

    def test_reading_pages_are_marked_and_the_cover_is_not(self):
        for language in ("zh", "en"):
            with self.subTest(language=language):
                self.assertRegex(self.output["note", language], r'<body class="page-article page-reading">')
                self.assertNotIn("page-reading", self.output["home", language])

    def test_the_directory_toggle_works_without_scripts(self):
        labels = {"zh": ("查看全部目录", "只看当前章节"), "en": ("Show all chapters", "Current chapter only")}
        for language, (more, less) in labels.items():
            with self.subTest(language=language):
                note = self.output["note", language]
                toggle = re.search(r'<label class="side-all"><input type="checkbox" data-side-all>(.*?)</label>', note)
                self.assertIsNotNone(toggle)
                self.assertIn(more, toggle[1])
                self.assertIn(less, toggle[1])
                self.assertNotIn("data-side-all", self.output["home", language])
        self.assertIn(".side-nav:has(.side-all input:not(:checked)) .nav details:not(:has(a.active)) { display: none; }", READING)
        self.assertNotRegex(re.sub(r"/\*.*?\*/", "", READING, flags=re.S), r":has\([^()]*:has\(")   # :has() cannot nest; browsers drop the rule

    def test_reading_styles_load_last_and_stay_scoped(self):
        self.assertLess(TEMPLATE.index("static/mobile-reading.css"), TEMPLATE.index("static/reading.css"))
        rules = re.sub(r"/\*.*?\*/", "", READING, flags=re.S)
        for selector in re.findall(r"(?m)^\s*([^@{}\n][^{}]*?)\s*\{", rules):
            for part in re.split(r",(?![^()]*\))", selector):   # commas inside :not(...) / :has(...) stay
                part = part.strip()
                with self.subTest(selector=part):
                    self.assertTrue(part.startswith((":root .page-reading", ":root[data-bar-hidden] .page-reading", ":root[data-text", ".side-all")), part)

    def test_top_bar_leaves_only_while_reading_onward(self):
        app = (ROOT / "site/static/app.js").read_text()
        self.assertIn('classList.contains("page-reading")', app)
        self.assertIn('y > lastScroll && y > 120 && !barRoot.classList.contains("drawer-is-open")', app)
        self.assertIn(":root[data-bar-hidden] .page-reading .topbar:not(:has(:focus-visible)) { transform: translateY(-100%); }", READING)
        self.assertIn("@media (prefers-reduced-motion: reduce)", READING)

    def test_text_size_scales_note_text_and_survives_reloads(self):
        hub = (ROOT / "site/static/reader-hub.js").read_text()
        app = (ROOT / "site/static/app.js").read_text()
        head = TEMPLATE[:TEMPLATE.index("</head>")]
        self.assertIn("const TEXT_SIZES = [['s',", hub)
        self.assertIn("scope.dispatchEvent(new CustomEvent('notes:text-size', {detail: value}))", hub)
        self.assertIn('window.addEventListener("notes:text-size"', app)
        self.assertIn('localStorage.setItem("text-size", value)', app)
        self.assertIn('localStorage.getItem("text-size")', head)
        for size, scale in (("s", ".94"), ("l", "1.12"), ("xl", "1.24")):
            self.assertIn(f':root[data-text="{size}"] {{ --text-scale: {scale}; }}', READING)
        # without a choice the note text keeps its default size
        self.assertIn("font-size: calc(1.0625rem * var(--text-scale, 1))", READING)

    def test_touch_screens_keep_the_gloss_underline(self):
        hover = READING.index("@media (hover: hover)")
        self.assertGreater(READING.index(":root .page-reading .term { border-bottom: 0; }"), hover)


if __name__ == "__main__":
    unittest.main()
