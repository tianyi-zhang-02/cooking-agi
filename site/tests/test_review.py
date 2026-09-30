import copy
import html
import json
import re
import sys
import tomllib
import unittest
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import review


class Tags(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.tags = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.config = review.load_config()
        self.nav = tomllib.loads((review.ROOT / "site/nav.toml").read_text())
        self.pages = {}
        for deck in self.config["decks"]:
            for card in deck["cards"]:
                source = card["source"]
                base = source.removesuffix(".md").removesuffix("README")
                if base.endswith("/"):
                    base += "index"
                self.pages[source] = {locale: SimpleNamespace(
                    url=f'{base}{".en" if locale == "en" else ""}.html')
                    for locale in ("zh", "en")}

    def render(self, section="00-foundations", language="zh", config=None):
        page = SimpleNamespace(section={"dir": section}, lang=language,
                               rel=lambda path: "../" + path)
        return review.render(page, config or self.config, self.pages)

    def test_every_existing_chapter_is_covered(self):
        review.validate(self.config, self.nav)

    def test_missing_chapter_is_rejected(self):
        self.config["decks"].pop()
        with self.assertRaises(ValueError):
            review.validate(self.config, self.nav)

    def test_duplicate_card_and_section_are_rejected(self):
        broken = copy.deepcopy(self.config)
        broken["decks"][1]["cards"][0]["id"] = broken["decks"][0]["cards"][0]["id"]
        with self.assertRaises(ValueError):
            review.validate(broken, self.nav)
        self.config["decks"].append(self.config["decks"][0])
        with self.assertRaises(ValueError):
            review.validate(self.config, self.nav)

    def test_bilingual_fields_are_required(self):
        for field in ("question", "answer", "pitfall"):
            broken = copy.deepcopy(self.config)
            del broken["decks"][0]["cards"][0][field]["en"]
            with self.assertRaises(ValueError):
                review.validate(broken, self.nav)

    def test_sources_must_be_safe_and_paired(self):
        for source in ("../../README.md", "/tmp/example.md", "missing.md", "https://example.com/read.md"):
            broken = copy.deepcopy(self.config)
            broken["decks"][0]["cards"][0]["source"] = source
            with self.assertRaises(ValueError):
                review.validate(broken, self.nav)

    def test_answers_collapsed_without_javascript(self):
        tags = Tags(self.render()).tags
        answers = [attrs for tag, attrs in tags if tag == "details"]
        self.assertEqual(len(answers), len(self.config["decks"][0]["cards"]))
        self.assertTrue(all("open" not in attrs for attrs in answers))
        controls = [attrs for tag, attrs in tags if "data-review-controls" in attrs]
        self.assertTrue(all("hidden" in attrs for attrs in controls))
        cards = [attrs for tag, attrs in tags if "data-card-id" in attrs]
        self.assertTrue(all("hidden" not in attrs for attrs in cards))

    def test_all_chapters_render_in_both_languages(self):
        for deck in self.config["decks"]:
            identities = []
            for language in ("zh", "en"):
                tags = Tags(self.render(deck["section"], language)).tags
                card_tags = [attrs for tag, attrs in tags if "data-card-id" in attrs]
                identities.append([(attrs["data-card-id"], attrs["data-version"]) for attrs in card_tags])
                for tag, attrs in tags:
                    if "data-review-copy" in attrs:
                        self.assertEqual("hidden" in attrs, attrs["data-review-copy"] != language)
            self.assertEqual(identities[0], identities[1])

    def test_home_and_governance_have_no_decks(self):
        for section in (".", "community", "templates"):
            self.assertEqual(self.render(section), "")

    def test_content_is_escaped(self):
        card = self.config["decks"][0]["cards"][0]
        payload = '<script>alert("test")</script>'
        card["question"]["zh"] = payload
        output = self.render()
        self.assertIn(html.escape(payload), output)
        self.assertNotIn("<script>", output)

    def test_answer_edits_invalidate_marks(self):
        card = self.config["decks"][0]["cards"][0]
        original = review.fingerprint(card)
        self.assertEqual(original, review.fingerprint(json.loads(json.dumps(card))))
        card["answer"]["en"] += " Additional explanation."
        self.assertNotEqual(original, review.fingerprint(card))

    def test_new_note_examples_run_and_languages_match(self):
        sources = ["01-data-and-feedback/feedback-to-objectives", "02-memory/memory-lifecycle",
                   "07-evaluation/ablation-and-slices", "00-foundations/model-families/how-to-read"]
        for source in sources:
            code_by_language = []
            for suffix in (".md", ".en.md"):
                content = (review.ROOT / (source + suffix)).read_text()
                blocks = re.findall(r"```python\n(.*?)\n```", content, re.S)
                code_by_language.append(blocks)
                for block in blocks:
                    exec(compile(block, source, "exec"), {})
            self.assertEqual(code_by_language[0], code_by_language[1])

    def test_card_sources_are_discoverable(self):
        import build
        build.discover(self.nav)
        for deck in self.config["decks"]:
            for card in deck["cards"]:
                self.assertIn(card["source"], build.BY_SRC)
                self.assertIsNotNone(build.BY_SRC[card["source"]]["en"])

    def test_general_checklist_is_not_a_manifest(self):
        import build
        output, terms = build.annotate("<p>准备清单与实验清单</p>", build.load_glossary())
        self.assertIn("<p>准备清单与<span", output)
        self.assertEqual([term["zh"] for term in terms], ["实验清单"])


if __name__ == "__main__":
    unittest.main()
