from html.parser import HTMLParser
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "site"))
import build


class StoryMarkup(HTMLParser):
    def __init__(self, body):
        super().__init__()
        self.headings = 0
        self.fold_depth = 0
        self.stories = []
        self.labs = []
        self.images = []
        self.feed(body)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "h2":
            self.headings += 1
        if tag == "details":
            self.fold_depth += 1
        if "data-bert-story" in attrs:
            self.stories.append((attrs, self.headings, self.fold_depth))
        if "data-encoder-lab" in attrs:
            self.labs.append((attrs, self.headings, self.fold_depth))
        if tag == "img":
            self.images.append(attrs)

    def handle_endtag(self, tag):
        if tag == "details":
            self.fold_depth -= 1


class BertStoryTests(unittest.TestCase):
    def test_state_and_task_contracts(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js unavailable")
        result = subprocess.run([node, str(ROOT / "site/tests/bert-story-behavior.cjs")],
                                capture_output=True, text=True, check=True)
        self.assertIn("72 valid states", result.stdout)

    def test_reading_order_and_static_fallback_in_both_languages(self):
        for suffix, language in [("", "zh"), (".en", "en")]:
            with self.subTest(language=language):
                source = (ROOT / f"00-foundations/core/bert{suffix}.md").read_text()
                body, _ = build.render_markdown(source)
                markup = StoryMarkup(body)
                self.assertEqual(len(markup.stories), 1)
                story, heading_count, fold_depth = markup.stories[0]
                self.assertEqual(story["data-lang"], language)
                self.assertEqual(story["id"], "bert-walkthrough")
                self.assertEqual((heading_count, fold_depth), (0, 0))
                self.assertEqual(len(markup.labs), 1)
                self.assertEqual(markup.labs[0][1:], (4, 1))
                self.assertEqual(len(markup.images), 1)
                self.assertIn("bert-flow", markup.images[0]["src"])
                self.assertTrue(markup.images[0]["alt"])
                self.assertEqual(markup.headings, 8)
                self.assertIn("0.804719", body)
                self.assertIn("cold", body)

    def test_assets_are_scoped_to_the_walkthrough(self):
        with patch.dict(build.NAV), patch.dict(build.BY_SRC):
            nav = build.load_nav()
            pages, sections = build.discover(nav)
            known = {page.url for page in pages}
            for source, expected in [("00-foundations/core/bert.md", True),
                                     ("00-foundations/core/vanilla-transformer.md", False),
                                     ("career/journey.md", False)]:
                for language in ["zh", "en"]:
                    with self.subTest(source=source, language=language):
                        page = build.BY_SRC[source][language]
                        build.build_page(page, [], nav["site"]["repo"], known)
                        output = build.assemble(page, sections, [], nav, "test",
                                                (ROOT / "site/template.html").read_text())
                        self.assertEqual("static/bert-story.js" in output, expected)
                        self.assertEqual("static/bert-story.css" in output, expected)


if __name__ == "__main__":
    unittest.main()
