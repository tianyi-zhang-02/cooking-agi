from html import escape
from html.parser import HTMLParser
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "site"))
import build
import paritycheck
import technology_chronicle

SPEC = importlib.util.spec_from_file_location(
    "rl_tradeoffs", ROOT / "05-post-training/deep-rl/code/rl_tradeoffs.py")
EXAMPLES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLES)
CHAPTERS = ("learn/technology-timeline", "learn/rl-timeline",
            "05-post-training/deep-rl/methods-map", "05-post-training/deep-rl/why-rl-is-hard")


class TreeMarkup(HTMLParser):
    def __init__(self, body):
        super().__init__()
        self.panels, self.links, self.ids = [], [], []
        self.elements = []
        self.headings = 0
        self.feed(body)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        self.elements.append((tag, attrs))
        if "data-tree-panel" in attrs:
            self.panels.append(attrs)
        if tag == "a":
            self.links.append(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "h2":
            self.headings += 1


class TechnologyTreeTests(unittest.TestCase):
    def test_bilingual_content_math_and_primary_sources(self):
        for chapter in CHAPTERS:
            with self.subTest(chapter=chapter):
                chinese, english = [(ROOT / f"{chapter}{suffix}.md").read_text()
                                    for suffix in ("", ".en")]
                self.assertEqual(paritycheck.shape(chinese), paritycheck.shape(english))
                self.assertEqual(re.findall(r"\{#([^}]+)\}", chinese), re.findall(r"\{#([^}]+)\}", english))
                self.assertEqual(re.findall(r"\$\$(.*?)\$\$", chinese, re.S),
                                 re.findall(r"\$\$(.*?)\$\$", english, re.S))
                self.assertEqual(set(re.findall(r"https://[^)\s\"]+", chinese)),
                                 set(re.findall(r"https://[^)\s\"]+", english)))
                for source in (chinese, english):
                    self.assertFalse(paritycheck.source_issues(source))
                    self.assertIn("2026-10", source)

    def test_chronicle_page_keeps_history_without_editorial_or_study_guides(self):
        data = json.loads(technology_chronicle.DATA.read_text())
        for suffix, language in (("", "zh"), (".en", "en")):
            source = (ROOT / f"learn/technology-timeline{suffix}.md").read_text()
            body, _ = build.render_markdown(source)
            body = technology_chronicle.expand(body, language)
            markup = TreeMarkup(body)
            self.assertEqual(len(markup.panels), 0)
            self.assertEqual(body.count('class="chronicle-work ')+body.count('class="chronicle-work"'), len(data["nodes"]))
            self.assertNotIn("\n## ", body)
            self.assertNotIn('markdown="1"', body)
            self.assertEqual(len(markup.ids), len(set(markup.ids)))
            for unwanted in ("world-system-example", "world-scope", "choose-your-path", "tree-inspector",
                             "哪些工作会放进这棵树", "What belongs on the tree", "继续读你感兴趣的 AI 分支"):
                self.assertNotIn(unwanted, body)
            for alias in ("overview", "language", "vision", "rl", "robotics", "systems"):
                self.assertIn(f"branch-{alias}", markup.ids)
            for node in data["nodes"]:
                self.assertIn(f'event-{node["id"]}', markup.ids)
                self.assertIn(escape(node["sources"][0][1], quote=True), body)

    def test_domain_routes_have_targets_in_both_languages(self):
        for suffix in ("", ".en"):
            history = (ROOT / f"learn/rl-timeline{suffix}.md").read_text()
            body, _ = build.render_markdown(history)
            ids = TreeMarkup(body).ids
            for anchor in ("dqn", "robot-control", "llm-post-training"):
                self.assertIn(anchor, ids)
            self.assertIn("https://arxiv.org/html/1801.01290", history)
            self.assertIn("https://arxiv.org/abs/2307.15818", history)

    def test_assets_are_scoped_and_new_pages_discoverable(self):
        with patch.dict(build.NAV), patch.dict(build.BY_SRC):
            nav = build.load_nav()
            pages, sections = build.discover(nav)
            known = {page.url for page in pages}
            for source in (*CHAPTERS, "00-foundations/core/bert"):
                for language in ("zh", "en"):
                    page = build.BY_SRC[f"{source}.md"][language]
                    self.assertEqual(sum(item.src == page.src for item in pages), 1)
                    build.build_page(page, [], nav["site"]["repo"], known)
                    output = build.assemble(page, sections, nav, "test",
                                            (ROOT / "site/template.html").read_text())
                    for retired in ("technology-tree.js", "technology-atlas.js", "world-exhibit.js", "world-grove.js"):
                        self.assertNotIn(f"static/{retired}", output)
                    self.assertEqual("static/chronicle.js" in output, source == CHAPTERS[0])
                    self.assertEqual("static/chronicle-artifacts.js" in output, source == CHAPTERS[0])
                    self.assertEqual("static/chronicle-designs.js" in output, source == CHAPTERS[0])
                    self.assertEqual("static/chronicle-starmap.js" in output, source == CHAPTERS[0])
                    self.assertEqual("static/chronicle.css" in output, source == CHAPTERS[0])
                    self.assertEqual("static/technology-tree.css" in output, source == CHAPTERS[1])
                    self.assertEqual("page-technology-tree" in output, source == CHAPTERS[0])
                    if source == CHAPTERS[0]:
                        self.assertNotIn("static/sky.js", output)
                        self.assertNotIn('href="rl-timeline.md', output)
                        self.assertIn(f'rl-timeline{".en" if language == "en" else ""}.html', output)
                        self.assertNotIn('.html#robot-control', output)

    def test_timeline_lists_every_milestone_once_and_folds_to_three(self):
        data = json.loads((ROOT / "site/static/technology-tree-data.json").read_text())
        for language in ("zh", "en"):
            output = technology_chronicle.render(data, language)
            self.assertEqual(output, technology_chronicle.render(data, language))
            markup = TreeMarkup(output)
            items = [link for link in markup.links if "ch-item" in link.get("class", "")]
            self.assertEqual(sorted(link["href"] for link in items), sorted(f'#event-{node["id"]}' for node in data["nodes"]))
            self.assertNotIn("<svg", output)  # the tree is drawn by chronicle.js as a background, not in the markup
            for era in data["eras"]:
                works = [name for branch in era["branches"] for name in branch["works"]]
                section = output[output.index(f'id="era-{era["id"]}"'):]
                section = section[:section.index("</section>")]
                self.assertEqual(section.count("data-key"), min(3, len(works)))
                self.assertEqual("data-more " in section, len(works) > 3)
    def test_tracks_are_declared_and_used(self):
        original = json.loads(technology_chronicle.DATA.read_text())
        for topic in original["topics"]:
            tracks = {track["id"] for track in topic.get("tracks", [])}
            for node in original["nodes"]:
                if node["topic"] == topic["id"]:
                    self.assertIn(node.get("track"), tracks)
        for failure in ("unknown", "duplicate", "untranslated"):
            data = json.loads(json.dumps(original))
            if failure == "unknown":
                data["nodes"][0]["track"] = "nowhere"
            elif failure == "duplicate":
                data["topics"][0]["tracks"].append(dict(data["topics"][0]["tracks"][0]))
            else:
                del data["topics"][0]["tracks"][0]["title"]["en"]
            with self.assertRaises(ValueError):
                technology_chronicle.validate(data)

    def test_atlas_data_has_bilingual_evidence_and_valid_routes(self):
        data = json.loads((ROOT / "site/static/technology-tree-data.json").read_text())
        self.assertEqual(data["version"], 3)
        technology_chronicle.validate(data)
        nodes = {node["id"]: node for node in data["nodes"]}
        self.assertEqual(len(nodes), len(data["nodes"]))
        for node in nodes.values():
            with self.subTest(node=node["id"]):
                self.assertRegex(node["id"], r"^[a-z0-9-]+$")
                self.assertIn(node["status"], ("historical", "adopted", "evolving"))
                self.assertIn(node["topic"], {topic["id"] for topic in data["topics"]})
                self.assertIn(node["kind"], ("research", "system", "product", "open"))
                self.assertLessEqual(node["year"], int(data["reviewed"][:4]))
                for field in ("title", "body", "limit"):
                    self.assertEqual(set(node[field]), {"zh", "en"})
                    self.assertTrue(node[field]["zh"].strip())
                    if field != "title":
                        self.assertRegex(node[field]["zh"], r"[\u4e00-\u9fff]")
                    self.assertNotRegex(node[field]["en"], r"[\u4e00-\u9fff]")
                self.assertTrue(node["sources"])
                for label, url in node["sources"]:
                    self.assertTrue(label.strip())
                    self.assertTrue(url.startswith("https://"))
        self.assertEqual(nodes["resnet"]["year"], 2015)
        self.assertEqual(nodes["gcn"]["year"], 2016)
        self.assertEqual(nodes["alphago"]["kind"], "system")
        self.assertEqual(nodes["chatgpt"]["kind"], "product")
        self.assertTrue({"gcn", "graphsage", "gat", "alphafold2"}.issubset(nodes))
        self.assertEqual(nodes["grpo"]["status"], "adopted")
        self.assertEqual(nodes["posttraining-open"]["status"], "evolving")
        adoption = " ".join(url for _, url in nodes["grpo"]["sources"])
        self.assertIn("huggingface.co/docs/trl", adoption)
        self.assertIn("verl.readthedocs.io", adoption)

    def test_world_tree_keeps_infrastructure_and_people_first_class(self):
        data = json.loads(technology_chronicle.DATA.read_text())
        self.assertEqual({topic["id"] for topic in data["topics"]}, {
            "energy", "chips", "memory", "networks", "software", "distributed",
            "algorithms", "data", "models", "applications", "ecosystem"})
        for language in ("zh", "en"):
            output = technology_chronicle.render(data, language)
            for topic in data["topics"]:
                self.assertEqual(output.count(f'id="field-{topic["id"]}"'), 1)
                self.assertNotIn(escape(topic["next"][language]), output)
                self.assertIn(f'data-field="{topic["id"]}" aria-pressed="false" aria-controls="chronicle-content" disabled', output)
            self.assertIn('id="event-software-linux-kernel-1991"', output)
            self.assertIn('class="chronicle-connections"', output)
            self.assertIn('class="ch-links"', output)
        self.assertEqual(data["eras"][0]["start"], 1882)

    def test_era_branches_explain_their_period_instead_of_repeating_topic_copy(self):
        data = json.loads(technology_chronicle.DATA.read_text())
        for language in ("zh", "en"):
            descriptions = {topic["description"][language] for topic in data["topics"]}
            summaries = [branch["summary"][language] for era in data["eras"] for branch in era["branches"]]
            self.assertEqual(len(summaries), len(set(summaries)))
            self.assertFalse(descriptions.intersection(summaries))

    def test_milestone_reader_layers_details_without_removing_sources(self):
        data = json.loads(technology_chronicle.DATA.read_text())
        for language in ("zh", "en"):
            output = technology_chronicle.render(data, language)
            markup = TreeMarkup(output)
            for class_name in ("chronicle-more", "chronicle-references"):
                disclosures = [attributes for tag, attributes in markup.elements
                               if tag == "details" and attributes.get("class") == class_name]
                self.assertEqual(len(disclosures), len(data["nodes"]))
                self.assertTrue(all("open" not in attributes for attributes in disclosures))
            self.assertEqual(output.count('class="chronicle-lead"'), len(data["nodes"]))
            self.assertNotIn("chronicle-work-meta", output)
            for node in data["nodes"]:
                for field in ("body", "limit"):
                    self.assertIn(escape(node[field][language]), output)
                for label, url in node["sources"]:
                    self.assertIn(escape(label), output)
                    self.assertIn(escape(url, quote=True), output)

    def test_optional_milestone_explanations_are_bilingual_escaped_and_server_rendered(self):
        data = json.loads(technology_chronicle.DATA.read_text())
        expanded = [node for node in data["nodes"] if "detail" in node]
        self.assertTrue(expanded)
        for language in ("zh", "en"):
            output = technology_chronicle.render(data, language)
            paragraphs = [paragraph for node in expanded for paragraph in node["detail"][language].split("\n\n")]
            self.assertEqual(output.count('class="chronicle-explanation"'), len(paragraphs))
            for paragraph in paragraphs:
                self.assertIn(f'<p class="chronicle-explanation">{escape(paragraph)}</p>', output)
        for detail in ({"zh": "只有中文"}, {"zh": "有内容", "en": ""}, "untranslated"):
            expanded[0]["detail"] = detail
            with self.assertRaisesRegex(ValueError, "detail translation"):
                technology_chronicle.validate(data)
        expanded[0]["detail"] = {"zh": "第一段\n\n第二段", "en": "One paragraph only"}
        with self.assertRaisesRegex(ValueError, "detail paragraphs differ"):
            technology_chronicle.validate(data)
        expanded[0]["detail"] = {"zh": "第一段\n\n第二段", "en": "First\n\nSecond"}
        self.assertIn('<p class="chronicle-explanation">第一段</p><p class="chronicle-explanation">第二段</p>',
                      technology_chronicle.render(data, "zh"))
        expanded[0]["detail"] = {"zh": '<script>alert("test")</script>', "en": "Safe explanation"}
        output = technology_chronicle.render(data, "zh")
        self.assertIn("&lt;script&gt;", output)
        self.assertNotIn('<script>alert(', output)
        for node in expanded:
            del node["detail"]
        self.assertNotIn('class="chronicle-explanation"', technology_chronicle.render(data, "en"))

    def test_world_tree_rejects_bad_taxonomy_and_unsupported_connections(self):
        original = json.loads(technology_chronicle.DATA.read_text())
        for failure in ("topic", "layer", "empty-topic", "relation", "evidence", "endpoint", "duplicate-topic", "root-field", "root-language"):
            with self.subTest(failure=failure):
                data = json.loads(json.dumps(original))
                rooted = next(topic for topic in data["topics"] if "root" in topic)
                if failure == "topic":
                    data["nodes"][0]["topic"] = "missing"
                elif failure == "layer":
                    data["topics"][0]["layer"] = "missing"
                elif failure == "empty-topic":
                    for node in data["nodes"]:
                        if node["topic"] == "energy":
                            node["topic"] = "chips"
                elif failure == "relation":
                    data["connections"][0]["type"] = "caused-everything"
                elif failure == "evidence":
                    data["connections"][0]["source"] = "https://example.com/unsupported"
                elif failure == "endpoint":
                    data["connections"][0]["to"] = "missing"
                elif failure == "root-field":
                    rooted["root"]["from"] = "alexnet"
                elif failure == "root-language":
                    del rooted["root"]["note"]["en"]
                else:
                    data["topics"].append(data["topics"][0])
                with self.assertRaises(ValueError):
                    technology_chronicle.validate(data)

    def test_internal_links_resolve_and_rendering_is_deterministic(self):
        data = json.loads(technology_chronicle.DATA.read_text())
        for language in ('zh', 'en'):
            output = technology_chronicle.render(data, language)
            self.assertEqual(output, technology_chronicle.render(data, language))
            markup = TreeMarkup(output)
            self.assertEqual(len(markup.ids), len(set(markup.ids)))
            for link in markup.links:
                if link.get('href', '').startswith('#'):
                    self.assertIn(link['href'][1:], markup.ids)
            self.assertEqual(output.count('class="ch-item"'), len(data['nodes']))
            self.assertIn('data-filter-empty hidden', output)
            self.assertNotIn('这条线还缺什么？' if language == 'zh' else 'What is still missing?', output)

    def test_sparse_domains_keep_honest_counts_and_native_reading_paths(self):
        data = json.loads(technology_chronicle.DATA.read_text())
        selected = {}
        for node in data["nodes"]:
            selected.setdefault(node["topic"], node)
        data["nodes"] = list(selected.values())
        names = {node["id"] for node in data["nodes"]}
        for era in data["eras"]:
            for branch in era["branches"]:
                branch["works"] = [name for name in branch["works"] if name in names]
            era["branches"] = [branch for branch in era["branches"] if branch["works"]]
            era["featured"] = [name for branch in era["branches"] for name in branch["works"]][:1]
        data["eras"] = [era for era in data["eras"] if era["branches"]]
        data["connections"] = []
        for topic in data["topics"]:
            topic.pop("root", None)
        for language in ("zh", "en"):
            output = technology_chronicle.render(data, language)
            markup = TreeMarkup(output)
            self.assertEqual(len(markup.ids), len(set(markup.ids)))
            self.assertEqual(len(re.findall(r'data-field="[a-z-]+"[^>]*>.*?<small>1</small></button>', output)), len(data["topics"]))
            self.assertEqual(output.count("data-work-title>"), len(names))
            self.assertEqual(output.count('class="ch-era-summary"'), len(data["eras"]))
            for tag, attributes in markup.elements:
                if tag == "details":
                    self.assertNotIn("open", attributes)
                if "data-field" in attributes:
                    self.assertIn("disabled", attributes)
            for name in names:
                self.assertIn(f"event-{name}", markup.ids)

    def test_period_bar_and_featured_works_are_data_backed(self):
        data = json.loads(technology_chronicle.DATA.read_text())
        for language in ("zh", "en"):
            output = technology_chronicle.render(data, language)
            markup = TreeMarkup(output)
            periods = [attrs["data-era"] for tag, attrs in markup.elements if tag == "section" and "data-era" in attrs]
            self.assertEqual(periods, [era["id"] for era in data["eras"]])
            # The featured milestones are the ones a period shows before it is opened.
            shown = {name for name in re.findall(r'data-key><a class="ch-item" href="#event-([a-z0-9-]+)"', output)}
            self.assertTrue({name for era in data["eras"] for name in era["featured"]}.issubset(shown))

    def test_chronicle_script_stays_local_and_parses(self):
        names = ("chronicle.js", "chronicle-designs.js", "chronicle-artifacts.js", "chronicle-starmap.js")
        for name in names:
            script = (ROOT / "site/static" / name).read_text()
            for forbidden in ("localStorage", "sessionStorage", "fetch(", "XMLHttpRequest", "innerHTML"):
                self.assertNotIn(forbidden, script, name)
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js unavailable")
        for name in names:
            subprocess.run([node, "--check", str(ROOT / "site/static" / name)], check=True)

    def test_star_map_and_its_switch_wait_for_javascript(self):
        data = json.loads((ROOT / "site/static/technology-tree-data.json").read_text(encoding="utf-8"))
        for language in ("zh", "en"):
            output = technology_chronicle.render(data, language)
            self.assertRegex(output, r'<div class="ch-views" data-views[^>]* hidden>')
            self.assertEqual(output.count('data-view="'), 2)
            self.assertRegex(output, r'<div class="ch-starmap" data-starmap hidden><canvas role="img" aria-label="[^"]+">')
            # the map is drawn from the library: every milestone and its links are in the page
            self.assertEqual(output.count('class="chronicle-work"'), len(data["nodes"]))

    def test_every_milestone_card_has_a_hidden_object_canvas(self):
        data = json.loads((ROOT / "site/static/technology-tree-data.json").read_text(encoding="utf-8"))
        for language in ("zh", "en"):
            output = technology_chronicle.render(data, language)
            canvases = re.findall(r'<canvas class="ch-art" data-art="([a-z0-9-]+)"[^>]* hidden></canvas>', output)
            self.assertEqual(sorted(canvases), sorted(node["id"] for node in data["nodes"]))

    def test_chronicle_language_and_history_navigation(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js unavailable")
        subprocess.run([node, "--test", str(ROOT / "site/tests/chronicle-behavior.test.cjs")], check=True, capture_output=True)

    def test_external_reading_routes_have_canonical_targets_and_scroll_offset(self):
        data = json.loads(technology_chronicle.DATA.read_text())
        for language in ("zh", "en"):
            markup = TreeMarkup(technology_chronicle.render(data, language))
            for identifier in ("field-energy", "event-alexnet", "event-vllm"):
                self.assertEqual(markup.ids.count(identifier), 1)
        stylesheet = (ROOT / "site/static/chronicle.css").read_text()
        self.assertRegex(stylesheet, r"\.chronicle-work \{[^}]*scroll-margin-top: 7rem;")

    def test_chronicle_is_rendered_without_javascript_and_has_stable_ids(self):
        data = json.loads(technology_chronicle.DATA.read_text())
        versions = []
        for language, suffix in (("zh", ""), ("en", ".en")):
            raw = (ROOT / f"learn/technology-timeline{suffix}.md").read_text()
            body, _ = build.render_markdown(raw)
            output = technology_chronicle.expand(body, language)
            markup = TreeMarkup(output)
            self.assertEqual(len(markup.ids), len(set(markup.ids)))
            self.assertEqual(output.count('id="tree-map"'), 1)
            self.assertNotIn(technology_chronicle.MARKER, output)
            self.assertIn('class="technology-tree tree-chronicle-ready"', output)
            self.assertEqual(output.count('class="ch-era"'), len(data["eras"]))
            self.assertIn('id="event-alphago"', output)
            self.assertIn('id="twig-branching-graphs"', output)
            self.assertNotIn('<details open', output)
            versions.append([name for name in markup.ids if name.startswith(("era-", "twig-", "event-", "field-"))])
        self.assertEqual(versions[0], versions[1])

    def test_chronicle_validates_dates_coverage_and_safe_sources(self):
        original = json.loads(technology_chronicle.DATA.read_text())
        for failure in ("duplicate", "unknown", "order", "date", "source", "language"):
            with self.subTest(failure=failure):
                data = json.loads(json.dumps(original))
                if failure == "duplicate":
                    data["nodes"].append(data["nodes"][0])
                elif failure == "unknown":
                    data["eras"][0]["branches"][0]["works"].append("missing")
                elif failure == "order":
                    data["eras"].reverse()
                elif failure == "date":
                    data["nodes"][0]["year"] = 2025
                elif failure == "source":
                    data["nodes"][0]["sources"][0][1] = "javascript:alert(1)"
                elif failure == "language":
                    del data["eras"][0]["title"]["en"]
                with self.assertRaises(ValueError):
                    technology_chronicle.validate(data)
        original["nodes"][0]["body"]["zh"] = '<img src=x onerror="alert(1)">'
        self.assertIn('&lt;img src=x', technology_chronicle.render(original, "zh"))
        self.assertNotIn('<img src=x', technology_chronicle.render(original, "zh"))

    def test_q_learning_terminal_bootstrap_and_validation(self):
        self.assertAlmostEqual(EXAMPLES.q_update(2, 1, [3, 4], .9, .5), 3.3)
        self.assertEqual(EXAMPLES.q_update(2, 1, [], .9, .5, True), 1.5)
        self.assertEqual(EXAMPLES.q_update(2, 1, [999], .9, .5, True), 1.5)
        self.assertEqual(EXAMPLES.q_update(2, 1, [999], 0, 1), 1)
        for next_values, discount, rate in (([], .9, .5), ([3], 1.1, .5),
                                           ([3], .9, 0), ([math.nan], .9, .5)):
            with self.assertRaises(ValueError):
                EXAMPLES.q_update(2, 1, next_values, discount, rate)

    def test_ppo_positive_negative_and_zero_advantages(self):
        self.assertAlmostEqual(EXAMPLES.clipped_surrogate(1.3, 2), 2.4)
        self.assertAlmostEqual(EXAMPLES.clipped_surrogate(.7, -2), -1.6)
        self.assertAlmostEqual(EXAMPLES.clipped_surrogate(1.3, -2), -2.6)
        self.assertAlmostEqual(EXAMPLES.clipped_surrogate(.7, 2), 1.4)
        self.assertEqual(EXAMPLES.clipped_surrogate(100, 0), 0)
        for ratio in (-1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                EXAMPLES.clipped_surrogate(ratio, 2)

    def test_dpo_margin_example_counterexample_and_stability(self):
        loss = EXAMPLES.dpo_loss(math.log(.6), math.log(.4), math.log(.5), math.log(.5))
        self.assertAlmostEqual(loss, -math.log(.6))
        self.assertAlmostEqual(EXAMPLES.dpo_loss(-1, -1, -1, -1), math.log(2))
        counterexample = EXAMPLES.dpo_loss(math.log(.15), math.log(.05), math.log(.2), math.log(.2))
        self.assertAlmostEqual(counterexample, math.log(4 / 3))
        self.assertLess(counterexample, loss)
        self.assertEqual(EXAMPLES.dpo_loss(-1000, 0, 0, 0), 1000)
        self.assertEqual(EXAMPLES.dpo_loss(0, -1000, 0, 0), 0)
        with self.assertRaises(ValueError):
            EXAMPLES.dpo_loss(0, 0, 0, 0, beta=0)

    def test_rollout_accounting_and_pipeline_bound(self):
        budget = EXAMPLES.rollout_budget(64, 8, 2048, 16)
        self.assertEqual(budget, {"generated": 1048576, "retained": 262144, "excluded": 786432})
        self.assertEqual(budget["retained"] / budget["generated"], .25)
        self.assertEqual(EXAMPLES.rollout_budget(0, 8, 2048, 0)["generated"], 0)
        for counts in ((2, 0, 8, 1), (2, 8, 8, 3), (True, 8, 8, 0), (2, 8, -1, 1)):
            with self.assertRaises(ValueError):
                EXAMPLES.rollout_budget(*counts)
        stages = (60, 20, 30, 10)
        self.assertEqual(sum(stages), 120)
        self.assertEqual(max(stages), 60)


if __name__ == "__main__":
    unittest.main()
