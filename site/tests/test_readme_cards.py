"""README cards and hero: standalone SVGs GitHub can show as images."""
from pathlib import Path
import re
import sys
import unittest
from datetime import date
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
        people = [{"name": "Codex", "login": None, "commits": 1, "kind": "ai", "first": "2026-10-13", "last": "2026-10-13"},
                  {"name": "Ada Lovelace", "login": "ada", "commits": 3, "kind": "human", "first": "2026-08-01", "last": "2026-09-01"}]
        collaboration = {"area": [{"zh": "评估", "en": "Evaluation", "reviewers": ["someone"]}]}
        outcomes = {"companies": {"example": "Example"}, "records": [{"company": "example", "consent": True},
                                                                      {"company": "example", "consent": False}]}
        with mock.patch.object(readme_cards, "avatar_uri", return_value=None), \
                mock.patch.object(readme_cards, "review_counts", return_value={}):
            cls.team = readme_cards.people_data(people, collaboration, "example/notes", outcomes, today=date(2026, 10, 17))
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
        # the draft count and the amber-bar legend appear only while some notes are still drafts
        drafts = sum(d for *_, groups in self.topics["zones"] for *_, d in groups)
        for lang, label in (("en", "drafts in review"), ("zh", "篇草稿待审核")):
            self.assertEqual(label in self.cards[("topics", lang, "dark")], bool(drafts))
        # the label wraps onto two lines of the stat column; read together they give the whole label
        lines = re.findall(r">([^<]+)</text>", self.cards[("topics", "en", "dark")])
        self.assertIn("notes re-read in both languages", " ".join(lines))

    def test_crew_is_numbered_by_first_participation_and_shown_by_username(self):
        crew = self.team["crew"]
        self.assertEqual([(m["cg"], m["handle"]) for m in crew], [("CG 001", "@ada"), ("CG 002", "Codex")])
        self.assertEqual([m["this_week"] for m in crew], [False, True])  # week of 2026-10-12
        self.assertTrue(crew[1].get("mark"))  # AI collaborators carry their mark, not an avatar
        for lang in ("zh", "en"):
            card = self.cards[("people", lang, "dark")]
            self.assertIn("@ada", card)
            self.assertNotIn("Lovelace", card)  # GitHub usernames, not real names

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
