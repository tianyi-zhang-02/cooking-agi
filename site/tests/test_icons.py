import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build


class InterfaceIconTests(unittest.TestCase):
    """Interface icons are Lucide: one 24-unit grid, stroke width 2, and a licence notice."""

    def test_top_bar_and_search_icons_share_one_grid_and_stroke(self):
        template = (build.SITE / "template.html").read_text(encoding="utf-8")
        for svg in re.findall(r"<svg\b[^>]*>", template):
            if 'viewBox="0 0 28 28"' in svg or 'viewBox="0 0 16 16"' in svg:
                continue   # the site's own book mark and GitHub's mark keep their drawings
            with self.subTest(svg=svg[:60]):
                self.assertIn('viewBox="0 0 24 24"', svg)
                self.assertIn('stroke-width="2"', svg)
        self.assertIn('viewBox="0 0 24 24"', build.icon("chevron-right", 13, "chev"))
        self.assertIn('stroke-width="2"', build.icon("search", 20))

    def test_close_buttons_draw_the_icon_instead_of_a_letter(self):
        sources = {
            "template.html": (build.SITE / "template.html").read_text(encoding="utf-8"),
            "collaboration.py": (build.SITE / "collaboration.py").read_text(encoding="utf-8"),
            "reader-hub.js": (build.SITE / "static/reader-hub.js").read_text(encoding="utf-8"),
            "mobile-reading.js": (build.SITE / "static/mobile-reading.js").read_text(encoding="utf-8"),
        }
        for name, source in sources.items():
            with self.subTest(name=name):
                self.assertNotIn(">×</button>", source)
                self.assertNotIn("text: '×'", source)
                self.assertNotIn("textContent = '×'", source)

    def test_lucide_licence_ships_with_the_site(self):
        notice = (build.SITE / "static/LUCIDE-LICENSE.txt").read_text(encoding="utf-8")
        self.assertIn("ISC License", notice)
        self.assertIn("Lucide Contributors", notice)
        licence = (build.ROOT / "LICENSE.md").read_text(encoding="utf-8")
        self.assertEqual(licence.count("site/static/LUCIDE-LICENSE.txt"), 4)   # English and Chinese, text and link


if __name__ == "__main__":
    unittest.main()
