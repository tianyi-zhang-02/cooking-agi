"""README cards and hero: standalone SVGs GitHub can show as images."""
from pathlib import Path
import re
import sys
import unittest
import urllib.error
from unittest import mock
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "site"))
import build
import readme_cards

CJK = re.compile(r"[　-〿一-鿿＀-￯]")


class ReadmeCardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nav = build.load_nav()
        build.discover(nav)
        cls.topics = readme_cards.topic_rows(build.NAV, build.NAV["_sections"])
        cls.milestones = readme_cards.load_milestones(ROOT / "site/milestones.toml")
        people = [{"name": "Ada", "login": None, "commits": 3, "kind": "human"},
                  {"name": "Codex", "login": None, "commits": None, "kind": "ai"}]
        collaboration = {"area": [{"zh": "评估", "en": "Evaluation", "reviewers": ["someone"]}]}
        outcomes = {"companies": {"example": "Example"}, "records": [{"company": "example", "consent": True},
                                                                      {"company": "example", "consent": False}]}
        cls.team = {"contributors": [{"name": "Ada", "login": None, "commits": 3, "avatar": None}],
                    "ai": ["Codex"], "reviewers": [{"login": "someone", "areas": [("评估", "Evaluation")],
                                                    "reviews": 0, "avatar": None}],
                    "next": readme_cards.next_stops(outcomes)}
        cls.signals = {"reviewed": 10, "figures": 20, "paper": 30}
        cls.cards = {}
        for lang in ("zh", "en"):
            for palette in readme_cards.PALETTES:
                cls.cards[("topics", lang, palette)] = readme_cards.topics_svg(cls.topics, lang, palette, "2026-10-17", cls.signals)
                cls.cards[("roadmap", lang, palette)] = readme_cards.roadmap_svg(cls.milestones, lang, palette, "2026-10-17")
                cls.cards[("people", lang, palette)] = readme_cards.people_svg(cls.team, lang, palette, "2026-10-17")

    def test_cards_are_valid_standalone_svg(self):
        for key, svg in self.cards.items():
            root = ET.fromstring(svg)
            self.assertTrue(root.tag.endswith("svg"), key)
            self.assertNotIn("http://", svg.replace("http://www.w3.org/2000/svg", ""), key)
            self.assertNotIn("https://", svg, key)

    def test_english_cards_have_no_chinese(self):
        for (name, lang, _), svg in self.cards.items():
            if lang == "en":
                self.assertIsNone(CJK.search(svg), name)

    def test_topics_cover_every_zone_and_mark_drafts(self):
        zones = [zone for zone in build.NAV["zone"]]
        self.assertEqual(len(self.topics["zones"]), len(zones))
        self.assertTrue(any(d for *_, groups in self.topics["zones"] for *_, d in groups))
        # the label wraps onto two lines of the stat column
        for part in ("notes re-read in", "both languages"):
            self.assertIn(part, self.cards[("topics", "en", "dark")])

    def test_next_stop_counts_only_consented_records(self):
        self.assertEqual([company["name"] for company in self.team["next"]], ["Example"])

    def test_roadmap_has_all_four_columns(self):
        for key in ("done", "now", "next", "open"):
            self.assertTrue(self.milestones.get(key), key)


class ContributorIdentityTests(unittest.TestCase):
    def test_one_entry_per_person_through_mailmap(self):
        # Michael Wang (Juggernaut0825) also committed as "Zijian Wang"; .mailmap makes it one person,
        # for authors and for Co-authored-by credits alike
        offline = mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline"))
        with offline:
            people = build.contributors("tianyi-zhang-02/cooking-agi")
        names = [p["name"] for p in people]
        self.assertNotIn("Zijian Wang", names)
        self.assertEqual(names.count("Michael Wang"), 1)
        self.assertEqual(next(p for p in people if p["name"] == "Michael Wang")["login"], "Juggernaut0825")


class ReadmeHeroTests(unittest.TestCase):
    def test_hero_is_self_contained_and_small(self):
        for lang in ("en", "zh"):
            path = ROOT / f".github/media/hero.{lang}.svg"
            svg = path.read_text(encoding="utf-8")
            ET.fromstring(svg)
            self.assertLess(path.stat().st_size, 600_000)
            self.assertNotRegex(svg, r'href="https?://')
            self.assertIn("prefers-reduced-motion", svg)
            self.assertIn(f".github/media/hero.{lang}.svg", (ROOT / ("README.md" if lang == "en" else "README.zh.md")).read_text(encoding="utf-8"))

    def test_tour_stills_exist_for_both_themes(self):
        for lang in ("en", "zh"):
            for name in ("figure", "paper", "frontier"):
                for theme in ("dark", "light"):
                    self.assertTrue((ROOT / f".github/media/tour/{lang}-{name}-{theme}.webp").is_file())


if __name__ == "__main__":
    unittest.main()
