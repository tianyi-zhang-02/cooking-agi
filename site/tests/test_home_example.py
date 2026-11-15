import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build


class FirstCalculationTests(unittest.TestCase):
    """The home page's first calculation is the worked example of Starting out, step for step."""

    @classmethod
    def setUpClass(cls):
        cls.example = build.home_data()["example"]
        cls.numbers = build.example_numbers(cls.example)

    def test_the_numbers_are_the_lessons_numbers(self):
        n = self.numbers
        self.assertEqual(n["predictions"], [1, 2])
        self.assertEqual(n["errors"], [1, 4])
        self.assertEqual(n["loss"], 2.5)
        self.assertEqual(n["terms"], [-2, -8])
        self.assertEqual(n["gradient"], -5)
        self.assertEqual([(s["rate"], s["weight"], s["loss"]) for s in n["steps"]], [(0.1, 1.5, 0.625), (1, 6, 40)])
        for source in ("interview/speedrun-start.md", "interview/speedrun-start.en.md"):
            with self.subTest(source=source):
                lesson = (build.ROOT / source).read_text(encoding="utf-8")
                for text in ("`[1,2]`", "`[2,4]`", "`weight = 1`", "(1+4)/2 = 2.5", "−2", "−8", "−5",
                             "1 − 0.1 × (−5) = 1.5", "(0.25+1)/2 = 0.625", "`weight=6`", "40"):
                    self.assertIn(text, lesson)

    def test_the_lesson_anchors_exist_in_both_languages(self):
        for key in ("lesson", "update"):
            source, anchor = self.example[key].split("#")
            for path in (source, source.replace(".md", ".en.md")):
                with self.subTest(path=path, anchor=anchor):
                    self.assertRegex((build.ROOT / path).read_text(encoding="utf-8"), r"\{#" + anchor + r"\}")

    def test_without_javascript_the_whole_calculation_is_on_the_page(self):
        for language in ("en", "zh"):
            with self.subTest(language=language):
                markup = build.example_html(language)
                for part in ("predict", "loss", "update"):
                    self.assertIn(f'data-part="{part}"', markup)
                self.assertIn("(1 + 4) / 2 = <b>2.5</b>", markup)
                self.assertIn("(−2 − 8) / 2 = <b>−5</b>", markup)
                self.assertIn('data-rate="0.1"', markup)
                self.assertIn('data-rate="1"', markup)
                self.assertIn("= <b>1.5</b>", markup)
                self.assertIn("→ <b>40</b>", markup)
                self.assertIn('<div class="hx-plot" hidden></div>', markup)   # the curve is drawn by the script only
                self.assertIn(f'href="{self.example["lesson"]}"', markup)

    def test_on_a_phone_the_table_scrolls_in_its_own_box_never_the_page(self):
        styles = (build.SITE / "static/style.css").read_text(encoding="utf-8")
        box = re.search(r"\.prose \.hx-table \.table-wrap \{([^}]*)\}", styles)[1]
        self.assertIn("max-width: 100%", box)
        self.assertIn("overflow-x: auto", box)
        # headers wrap between words before anything scrolls; all four columns stay
        self.assertIn(".prose .hx-table .table-wrap th { color: var(--muted); white-space: normal; overflow-wrap: normal; word-break: normal; }", styles)
        self.assertIn(".prose .hx-table .table-wrap :is(th, td):first-child { min-width: 0; }", styles)
        self.assertEqual(build.example_html("en").count('<th scope="col"'), 4)

    def test_the_script_loads_only_where_the_example_is(self):
        template = (build.SITE / "template.html").read_text(encoding="utf-8")
        self.assertIn('<!-- if:data-home-example --><script defer src="{{prefix}}static/home-example.js', template)
        script = (build.SITE / "static/home-example.js").read_text(encoding="utf-8")
        self.assertNotIn("innerHTML", script)
        self.assertNotIn("setInterval", script)        # nothing moves on its own
        self.assertIn("aria-pressed", script)
        self.assertIn("aria-live", script)
        styles = (build.SITE / "static/style.css").read_text(encoding="utf-8")
        self.assertIn("@media (prefers-reduced-motion: reduce) { .hx-after, .hx-move, .hx-head, .hx-drop { animation: none; } }", styles)


class TypeScaleTests(unittest.TestCase):
    """The home page and the notes share one type scale, the GOV.UK points on a 5px grid."""

    def test_the_scale_has_the_govuk_points(self):
        styles = (build.SITE / "static/style.css").read_text(encoding="utf-8")
        for size, line in (("48", "3.125rem"), ("36", "2.5rem"), ("24", "1.875rem"), ("19", "1.5625rem"), ("16", "1.25rem")):
            with self.subTest(point=size):
                self.assertRegex(styles, rf"--fs-{size}: [\d.]+rem;\s+--lh-{size}: {re.escape(line)};")
        for value in re.findall(r"--lh-\d+: ([\d.]+)rem", styles):   # every line height lands on the 5px grid
            self.assertEqual(float(value) * 16 % 5, 0, value)

    def test_notes_and_home_draw_on_the_same_scale(self):
        reading = (build.SITE / "static/reading.css").read_text(encoding="utf-8")
        for token in ("--fs-48", "--fs-24", "--fs-19", "--fs-16"):
            self.assertIn(f"var({token})", reading)
        styles = (build.SITE / "static/style.css").read_text(encoding="utf-8")
        home = styles[styles.index("/* below the cover: a reading room"):styles.index("/* the finish: one radius")]
        self.assertIn("font-size: var(--fs-36)", home)
        self.assertNotRegex(home, r"font-size: 1\.\d+rem")   # section and card sizes come from the scale


if __name__ == "__main__":
    unittest.main()
