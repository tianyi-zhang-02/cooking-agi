from html.parser import HTMLParser
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build


class ReadingElements(HTMLParser):
    def __init__(self, markup):
        super().__init__()
        self.elements = []
        self.feed(markup)

    def handle_starttag(self, tag, attributes):
        self.elements.append((tag, dict(attributes)))


class ReadingModesTests(unittest.TestCase):
    def setUp(self):
        self.chinese = SimpleNamespace(lang="zh", url="learn/example.html", rel=lambda url: "../" + url)
        self.english = SimpleNamespace(lang="en", url="learn/example.en.html", rel=lambda url: "../" + url)
        self.chinese.sibling = self.english
        self.english.sibling = self.chinese

    def test_single_click_links_to_the_same_note_without_javascript(self):
        for page, target_language, label in ((self.chinese, "en", "EN"), (self.english, "zh-Hans", "中文")):
            with self.subTest(language=page.lang):
                markup = build.reading_controls_html(page)
                elements = ReadingElements(markup).elements
                self.assertEqual(len(elements), 1)
                tag, attributes = elements[0]
                self.assertEqual(tag, "a")
                self.assertEqual(attributes["href"], "../" + page.sibling.url)
                self.assertEqual(attributes["hreflang"], target_language)
                self.assertEqual(attributes["lang"], target_language)
                self.assertIn("Read in", attributes["aria-label"])
                self.assertTrue(markup.endswith(f">{label}</a>"))
                self.assertNotIn("<details", markup)
                self.assertNotIn("data-term-toggle", markup)

    def test_missing_counterpart_is_disabled_not_a_dead_link(self):
        for page in (self.chinese, self.english):
            page.sibling = None
            elements = ReadingElements(build.reading_controls_html(page)).elements
            self.assertEqual(len(elements), 1)
            tag, attributes = elements[0]
            self.assertEqual(tag, "span")
            self.assertEqual(attributes["aria-disabled"], "true")
            self.assertNotIn("href", attributes)

    def test_link_target_is_escaped(self):
        self.english.url = 'learn/example.en.html?view="reading"&source=note'
        markup = build.reading_controls_html(self.chinese)
        self.assertIn('&quot;reading&quot;&amp;source=note', markup)

    def test_english_annotations_no_longer_depend_on_saved_preferences(self):
        css = (build.SITE / "static/style.css").read_text()
        self.assertNotIn("data-english-terms", css)
        self.assertNotIn("reading-panel", css)
        self.assertIn(".lang-btn:focus-visible", css)
        template = (build.SITE / "template.html").read_text()
        self.assertIn("{{reading_controls}}", template)
        self.assertNotIn("static/reading.js", template)
        self.assertFalse((build.SITE / "static/reading.js").exists())
        output, used = build.annotate("<p>注意力</p>", [("注意力", "attention", "按相关程度组合信息")])
        self.assertIn('<span class="term-en">（attention）</span>', output)
        self.assertNotIn("hidden", output)
        self.assertEqual(len(used), 1)


if __name__ == "__main__":
    unittest.main()
