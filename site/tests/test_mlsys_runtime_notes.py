from html.parser import HTMLParser
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "site"))
import build
import paritycheck
import prep_progress


CHAPTERS = (
    "mlsys-llmsys-map",
    "execution-boundaries",
    "megatron-basics",
    "rl-systems",
    "async-rl",
)


def source(chapter, language="zh"):
    suffix = ".en" if language == "en" else ""
    return (ROOT / "06-systems" / f"{chapter}{suffix}.md").read_text()


class Markup(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids = set()
        self.annotations = []
        self.feed(text)

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if "id" in attributes:
            self.ids.add(attributes["id"])
        if "term" in attributes.get("class", "").split():
            self.annotations.append(attributes)


def diagram_edges(text):
    result = []
    for diagram in re.findall(r"```mermaid\n(.*?)```", text, re.S):
        graph = re.sub(r"\[[^\]]*\]", "", diagram)
        result.append(re.findall(r"\b(\w+)\s*-->\s*(\w+)\b", graph))
    return result


class MLSystemRuntimeNotesTests(unittest.TestCase):
    def test_bilingual_structure_math_and_explicit_anchors(self):
        for chapter in CHAPTERS:
            with self.subTest(chapter=chapter):
                chinese, english = source(chapter), source(chapter, "en")
                self.assertEqual(paritycheck.shape(chinese), paritycheck.shape(english))
                anchors = re.findall(r"\{#([^}]+)\}", chinese)
                self.assertEqual(anchors, re.findall(r"\{#([^}]+)\}", english))
                self.assertEqual(len(anchors), len(set(anchors)))
                self.assertGreater(len(anchors), 3)
                self.assertEqual(re.findall(r"\$\$(.*?)\$\$", chinese, re.S),
                                 re.findall(r"\$\$(.*?)\$\$", english, re.S))
                inline = r"(?<!\$)\$(?!\$)([^\n$]+)\$"
                self.assertEqual(re.findall(inline, chinese), re.findall(inline, english))
                english_body = re.sub(r"\[中文\]\([^)]*\)", "", english)
                self.assertNotRegex(english_body, r"[\u3400-\u9fff]")

    def test_one_navigation_location_per_language_in_foundations_first_order(self):
        with patch.dict(build.NAV), patch.dict(build.BY_SRC):
            pages, sections = build.discover(build.load_nav())
            section = next(item for item in sections if item["dir"] == "practice/llm-runtime")
            self.assertEqual(section["group"], "ml-systems")
            self.assertEqual(section["home"], "06-systems/mlsys-llmsys-map.md")
            self.assertEqual([pair["zh"].src.stem for pair in section["pages"]], list(CHAPTERS))
            catalog = prep_progress.catalog(build.BY_SRC)["notes"]
            for chapter in CHAPTERS:
                canonical = f"06-systems/{chapter}.md"
                entry = catalog[prep_progress.identifier(canonical)]
                for language, suffix in (("zh", ""), ("en", ".en")):
                    expected = ROOT / "06-systems" / f"{chapter}{suffix}.md"
                    matches = [page for page in pages if page.src == expected]
                    self.assertEqual(len(matches), 1)
                    self.assertEqual(matches[0].section["group"], "ml-systems")
                    self.assertEqual(entry["pages"][language]["url"], f"06-systems/{chapter}{suffix}.html")

    def test_local_links_keep_language_and_existing_fragments(self):
        for chapter in CHAPTERS:
            for language in ("zh", "en"):
                text = source(chapter, language)
                for label, href in re.findall(r"\[([^\]]*)\]\(([^)]+)\)", text):
                    parsed = urlsplit(href)
                    if parsed.scheme or parsed.netloc:
                        continue
                    with self.subTest(chapter=chapter, language=language, link=href):
                        target = (ROOT / "06-systems" / unquote(parsed.path)).resolve()
                        self.assertTrue(target.is_file())
                        if label not in ("中文", "English"):
                            self.assertEqual(target.name.endswith(".en.md"), language == "en")
                        if parsed.fragment:
                            raw = re.sub(r"^#\s+.+$", "", target.read_text(), count=1, flags=re.M)
                            rendered, _ = build.render_markdown(build.protect_mermaid(raw))
                            self.assertIn(unquote(parsed.fragment), Markup(rendered).ids)

    def test_reading_route_returns_to_cluster_section_after_megatron(self):
        for language, suffix in (("zh", ""), ("en", ".en")):
            route = source("mlsys-llmsys-map", language).split("{#learning-route}", 1)[1]
            links = re.findall(r"\]\(([^)]+)\)", route)
            self.assertEqual(links[:5], [
                f"execution-boundaries{suffix}.md#runtime",
                f"megatron-basics{suffix}.md",
                f"execution-boundaries{suffix}.md#scheduling",
                f"rl-systems{suffix}.md",
                f"async-rl{suffix}.md",
            ])

    def test_localized_diagrams_keep_edges_and_queue_boundary(self):
        for chapter in ("mlsys-llmsys-map", "rl-systems"):
            chinese, english = source(chapter), source(chapter, "en")
            self.assertEqual(diagram_edges(chinese), diagram_edges(english))
            for diagram in re.findall(r"```mermaid\n(.*?)```", chinese, re.S):
                self.assertRegex(diagram, r"[\u4e00-\u9fff]")
        edges = diagram_edges(source("rl-systems"))[0]
        self.assertIn(("R", "Q"), edges)
        self.assertIn(("Q", "L"), edges)
        self.assertNotIn(("R", "L"), edges)

    def test_glossary_stays_scoped_and_annotates_every_occurrence(self):
        general = build.load_glossary()
        page = SimpleNamespace(section={"group": "ml-systems"})
        terms = build.glossary_for_page(page, general)
        names = [term[0] for term in terms]
        self.assertEqual(len(names), len(set(names)))
        for name in ("背压", "策略滞后", "微批次", "优化器状态", "张量"):
            self.assertIn(name, names)
        for group in ("career", "systems", "infra", "inference"):
            self.assertIs(build.glossary_for_page(SimpleNamespace(section={"group": group}), general), general)
        rendered, used = build.annotate("<p>背压与策略滞后。</p><p>再次检查背压与策略滞后。</p>", terms)
        self.assertEqual(len(Markup(rendered).annotations), 4)
        self.assertEqual([term["zh"] for term in used], ["背压", "策略滞后"])

    def test_memory_example_matches_both_languages(self):
        for language in ("zh", "en"):
            text = source("megatron-basics", language)
            dimensions = {name: int(value) for name, value in re.findall(r"\b([BSHV])=(\d+)", text)}
            hidden_bytes = dimensions["B"] * dimensions["S"] * dimensions["H"] * 2
            logits_bytes = dimensions["B"] * dimensions["S"] * dimensions["V"] * 2
            self.assertEqual(hidden_bytes / 2**20, 16)
            self.assertEqual(logits_bytes / 2**20, 500)
            self.assertIn("16 MiB", text)
            self.assertIn("500 MiB", text)
            self.assertIn("1000 MiB", text)
            self.assertIn(r"$2\times2\times2=8$", text)
            self.assertIn(r"$2\times2\times4=16$", text)

    def test_pinned_paths_and_existing_sao_note_are_kept_distinct(self):
        for language, suffix in (("zh", ""), ("en", ".en")):
            text = source("async-rl", language)
            self.assertIn("d25f2929b2a1e962e9d366717fbf872ab46a4997/docs/en/algorithms/async.md", text)
            self.assertIn("dd2e251da8ea1b4de6e76394b543e89d1e8cbe9e/docs/guides/async-grpo.md", text)
            self.assertIn(f"../05-post-training/async-policy-learning{suffix}.md", text)
            self.assertIn("max_head_offpolicyness", text)
            self.assertIn("max_trajectory_age_steps", text)
            self.assertIn("staleness_threshold", text)


if __name__ == "__main__":
    unittest.main()
