import copy
import shutil
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import discussions


class DiscussionTests(unittest.TestCase):
    def setUp(self):
        self.config = discussions.load_config()

    def page(self, language="zh", slug="self-worth"):
        suffix = ".en.md" if language == "en" else ".md"
        return SimpleNamespace(lang=language, src=discussions.ROOT / "discussions" / (slug + suffix),
                               section={"dir": "discussions"}, rel=lambda path: "../" + path)

    def test_standalone_category_is_after_papers(self):
        nav = tomllib.loads((discussions.ROOT / "site/nav.toml").read_text())
        categories = [category["id"] for category in nav["category"]]
        self.assertEqual(categories.index("discussions"), categories.index("reference") + 1)
        group = next(group for group in nav["group"] if group["id"] == "discussions")
        self.assertEqual(group["category"], "discussions")

    def test_language_pair_shares_thread(self):
        chinese = discussions.topic_for(self.page(), self.config)
        english = discussions.topic_for(self.page("en"), self.config)
        self.assertEqual(chinese, english)
        for language in ("zh", "en"):
            output = discussions.comments(self.page(language), self.config)
            self.assertIn(discussions.thread_url(chinese, self.config), output)

    def test_each_topic_has_its_own_thread_and_matching_bilingual_titles(self):
        topics = self.config["topic"]
        self.assertTrue({"self-worth", "perspectives", "room-to-live", "earning-and-enough", "your-own-path"}.issubset({topic["slug"] for topic in topics}))
        self.assertEqual(len({topic["number"] for topic in topics}), len(topics))
        for topic in topics:
            for language in ("zh", "en"):
                page = self.page(language, topic["slug"])
                self.assertEqual(page.src.read_text().splitlines()[0], "# " + topic[f"title_{language}"])
                markup = discussions.comments(page, self.config)
                self.assertIn(discussions.thread_url(topic, self.config), markup)

    def test_classical_quote_has_a_primary_source_in_both_languages(self):
        for language in ("zh", "en"):
            source = self.page(language, "room-to-live").src.read_text()
            self.assertIn("> 持而盈之，不如其已。", source)
            self.assertIn("https://www.gutenberg.org/cache/epub/7337/pg7337.html", source)

    def test_comments_stay_off_other_pages_and_overview(self):
        self.assertEqual(discussions.comments(self.page(slug="README"), self.config), "")
        page = self.page()
        page.section = {"dir": "career"}
        self.assertIsNone(discussions.topic_for(page, self.config))

    def test_disabled_embed_is_honest_and_has_working_destination(self):
        self.config["embed_enabled"] = False
        markup = discussions.comments(self.page(), self.config)
        self.assertNotIn("data-discussion-widget", markup)
        self.assertNotIn("<script", markup)
        self.assertIn("目前在 GitHub 留言", markup)
        self.assertIn("https://github.com/tianyi-zhang-02/cooking-agi/discussions/46", markup)

    def test_optional_embed_is_on_demand_with_public_disclosure(self):
        self.config["embed_enabled"] = True
        for language in ("zh", "en"):
            markup = discussions.comments(self.page(language), self.config)
            self.assertIn('data-number="46"', markup)
            self.assertIn("data-load-comments hidden", markup)
            self.assertNotIn("<script", markup)
            self.assertIn("留言公开" if language == "zh" else "Comments are public", markup)

    def test_text_fields_are_escaped(self):
        self.config["topic"][0]["title_zh"] = '<img src=x onerror="bad()">'
        output = discussions.topic_cards(self.page(slug="README"), self.config)
        self.assertNotIn("<img", output)
        self.assertIn("&lt;img", output)

    def test_no_duplicate_or_unsafe_targets(self):
        for field, value in (("slug", "../private"), ("number", -1), ("number", True),
                             ("author", 'bad"'), ("date", "2026-22-99")):
            config = copy.deepcopy(self.config)
            config["topic"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                discussions.validate(config)
        self.config["topic"].append(copy.deepcopy(self.config["topic"][0]))
        with self.assertRaises(ValueError):
            discussions.validate(self.config)

    def test_cards_and_back_links_use_current_language(self):
        for language in ("zh", "en"):
            suffix = ".en" if language == "en" else ""
            self.assertIn(f'self-worth{suffix}.html', discussions.topic_cards(self.page(language), self.config))
            self.assertIn(f'href="index{suffix}.html"', discussions.header(self.page(language), self.config))

    @unittest.skipUnless(shutil.which("node"), "Node required")
    def test_comment_loader_behavior(self):
        result = subprocess.run([shutil.which("node"), str(Path(__file__).with_name("discussions-behavior.cjs"))],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
