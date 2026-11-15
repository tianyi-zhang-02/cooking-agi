import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build

STATIC = build.SITE / "static"
css = lambda name: (STATIC / name).read_text(encoding="utf-8")


def rule(text, selector):
    """The last top-level rule for exactly this selector (the stylesheets layer older rules first;
    rules inside @media blocks are adjustments for one screen size and are left out)."""
    text = re.sub(r"@media[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", "", text)
    return re.findall(r"(?m)^\s*" + re.escape(selector) + r"\s*\{([^}]*)\}", text)[-1]


class InterfaceTests(unittest.TestCase):
    """One style for what surrounds the text, while the site's own touches stay."""

    def test_search_reads_as_a_field_with_its_shortcut(self):
        template = (build.SITE / "template.html").read_text(encoding="utf-8")
        button = re.search(r'<button type="button" class="icon-btn search-toggle".*?</button>', template, re.S)[0]
        self.assertIn('<span class="search-label">{{search_ph}}</span><kbd>⌘ K</kbd>', button)
        search = css("search.css")
        self.assertIn(".search-toggle :is(.search-label, kbd) { display: none; }", search)          # an icon on phones
        self.assertIn("@media (min-width: 1081px)", search)

    def test_top_bar_controls_share_one_ghost_style(self):
        style = css("style.css")
        lang = rule(style, ".topbar-right .lang-btn")
        self.assertIn("border-color: transparent", lang)
        self.assertIn("text-transform: none", lang)
        pill = rule(css("reader-hub.css"), ".repo-btn.repo-pill")
        self.assertIn("border: 0", pill)
        self.assertNotIn("repo-pill-divider", (STATIC / "reader-hub.js").read_text(encoding="utf-8"))
        self.assertIn(".lang-btn.language-cue { border-color: var(--ember); color: var(--ember); }", css("language.css"))   # the first-visit cue stays

    def test_labels_drop_tracked_capitals_and_the_chapter_numeral(self):
        style = css("style.css")
        kicker = rule(style, ".article-kicker span")
        self.assertNotIn("uppercase", kicker)
        self.assertNotIn("letter-spacing", kicker)
        self.assertNotIn("chapter-mark", style)
        self.assertNotIn("chapter-mark", (build.SITE / "build.py").read_text(encoding="utf-8"))

    def test_boxes_become_one_left_rule(self):
        line = "border-left: 2px solid color-mix(in srgb, var(--ember) 45%, var(--rule))"
        self.assertIn(line, rule(css("curriculum.css"), ".chapter-context"))
        self.assertIn(line, rule(css("style.css"), ".prose blockquote"))
        self.assertIn("background: none", rule(css("style.css"), ".article-body blockquote"))
        self.assertEqual(rule(css("style.css"), ".rail-inner").strip(), "position: relative; padding: 0;")
        route = rule(css("curriculum.css"), ".study-route a")
        self.assertIn("border-top: 1px solid var(--rule)", route)
        self.assertNotIn("background", route)

    def test_the_current_note_is_one_ember_line(self):
        active = rule(css("sidebar.css"), ".nav .chapter-pages > li > a.active")
        self.assertIn("box-shadow: inset 2px 0 var(--ember)", active)
        self.assertNotIn("background", active)

    def test_the_sites_own_touches_stay(self):
        template = (build.SITE / "template.html").read_text(encoding="utf-8")
        self.assertIn('<canvas class="sky"', template)
        self.assertIn("static/sky.js", template)
        self.assertIn('<div class="reading-progress"', template)
        reading = css("reading.css")
        self.assertIn("the ember", reading)                                   # the reading layout's own promise
        self.assertIn(".study-buddy {", css("study-buddy.css"))
        buddy = css("study-buddy.css")
        self.assertIn("width: min(280px, calc(100vw - 32px))", rule(buddy, ".sb-card"))
        self.assertIn("grid-template-columns: 44px minmax(0, 1fr);", buddy)       # on phones the column fits the smaller art
        self.assertIn(".sb-art { width: 44px; height: 44px; }", buddy)
        self.assertIn('.nav details:has(a.active) > summary .grp-name::after { content: " ·"; color: var(--ember); }', css("sidebar.css"))


if __name__ == "__main__":
    unittest.main()
