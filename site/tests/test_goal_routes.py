from html.parser import HTMLParser
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build
import paritycheck


class RouteLinks(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.links = []
        self.feed(text)

    def handle_starttag(self, tag, attributes):
        if tag == "a":
            self.links.append(dict(attributes))


class GoalRoutesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_nav = build.NAV.copy()
        cls.previous_sources = build.BY_SRC.copy()
        cls.nav = build.load_nav()
        cls.pages, cls.sections = build.discover(cls.nav)
        cls.catalog = build.BY_SRC.copy()

    @classmethod
    def tearDownClass(cls):
        build.NAV.clear()
        build.NAV.update(cls.previous_nav)
        build.BY_SRC.clear()
        build.BY_SRC.update(cls.previous_sources)

    def test_new_notes_have_matching_structure_and_stable_anchors(self):
        for source in ("learn/choose-your-path.md",):
            pair = self.catalog[source]
            chinese, english = [pair[lang].src.read_text() for lang in ("zh", "en")]
            self.assertEqual(paritycheck.shape(chinese), paritycheck.shape(english))
            self.assertEqual(re.findall(r"\{#([^}]+)\}", chinese), re.findall(r"\{#([^}]+)\}", english))

    def test_goal_cards_work_without_scripts(self):
        expected = ["getting-started", "interviews", "research", "projects", "topics"]
        for language in ("zh", "en"):
            source = self.catalog["learn/choose-your-path.md"][language].src.read_text()
            cards = re.search(r'<nav class="study-route goal-route".*?</nav>', source, re.S)[0]
            self.assertEqual([link["href"] for link in RouteLinks(cards).links], [f"#{anchor}" for anchor in expected])
            for anchor in expected:
                self.assertIn(f"{{#{anchor}}}", source)
            self.assertNotIn("onclick", cards)
            self.assertNotIn("<button", cards)

    def test_all_primary_areas_have_a_path_back_to_the_guide(self):
        for source in ("learn/README.md", "interview/README.md", "practice/README.md"):
            for language in ("zh", "en"):
                page = self.catalog[source][language]
                markup = build.sidebar_html(page, self.sections, self.nav["group"])
                routes = [link for link in RouteLinks(markup).links if link.get("class") == "side-route-link"]
                self.assertEqual(len(routes), 1)
                suffix = "" if language == "zh" else ".en"
                self.assertEqual(routes[0]["href"], page.rel(f"learn/choose-your-path{suffix}.html"))
                self.assertIn(f"choose-your-path{suffix}.md", page.src.read_text())

    def test_guide_does_not_link_to_itself_in_the_sidebar(self):
        for language in ("zh", "en"):
            page = self.catalog["learn/choose-your-path.md"][language]
            markup = build.sidebar_html(page, self.sections, self.nav["group"])
            self.assertNotIn('class="side-route-link"', markup)

    def test_all_guide_note_links_exist_and_stay_in_language(self):
        for language in ("zh", "en"):
            page = self.catalog["learn/choose-your-path.md"][language]
            source = page.src.read_text()
            links = re.findall(r"\]\(([^)]+\.md)(?:#[^)]*)?\)", source)
            self.assertGreater(len(links), 30)
            self.assertEqual((page.src.parent / links.pop(0)).resolve(), page.sibling.src.resolve())
            known_sources = {note.src.resolve() for note in self.pages if note.lang == language}
            for link in links:
                self.assertTrue((page.src.parent / link).resolve() in known_sources, f"{page.src.name}: {link}")

    def test_retired_job_search_route_keeps_its_old_anchor(self):
        # The career section was removed on 2026-11-05; old #applications links land on the interview route.
        for language in ("zh", "en"):
            source = self.catalog["learn/choose-your-path.md"][language].src.read_text()
            self.assertNotIn("../career/", source)
            self.assertNotIn("{#applications}", source)
            self.assertEqual(source.count('<span id="applications"></span>'), 1)
            self.assertLess(source.index('<span id="applications"></span>'), source.index("{#interviews}"))


if __name__ == "__main__":
    unittest.main()
