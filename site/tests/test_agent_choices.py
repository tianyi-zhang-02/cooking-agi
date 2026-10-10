from contextlib import redirect_stdout
from html.parser import HTMLParser
import io
import itertools
from pathlib import Path
import re
import sys
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "site"))
import build
import paritycheck


class Figures(HTMLParser):
    def __init__(self, body):
        super().__init__()
        self.depth = 0
        self.figures = {}
        self.feed(body)

    def handle_starttag(self, tag, attributes):
        if tag == "details":
            self.depth += 1
        if tag == "figure":
            attrs = dict(attributes)
            self.figures[attrs.get("id")] = (attrs.get("lang"), self.depth)

    def handle_endtag(self, tag):
        if tag == "details":
            self.depth -= 1


class AgentChoicesTests(unittest.TestCase):
    def source(self, chapter, suffix=""):
        return (ROOT / f"10-agents/{chapter}{suffix}.md").read_text()

    def example(self, chapter, suffix=""):
        code = re.search(r"```python\n(.*?)```", self.source(chapter, suffix), re.S).group(1)
        namespace = {}
        with redirect_stdout(io.StringIO()):
            exec(compile(code, chapter, "exec"), namespace)
        return code, namespace

    def test_bilingual_structure_and_calculations_match(self):
        for chapter in ("model-choice", "tools-and-skills"):
            self.assertEqual(paritycheck.shape(self.source(chapter)),
                             paritycheck.shape(self.source(chapter, ".en")))
            self.assertEqual(self.example(chapter)[0], self.example(chapter, ".en")[0])

    def test_loading_counts_include_catalog(self):
        _, values = self.example("tools-and-skills")
        self.assertEqual((values["on_demand"], values["all_at_once"]), (3480, 23280))
        self.assertEqual(values["context_budget"]([90] * 12, [], []), 1080)

    def test_loading_counts_reject_invalid_values(self):
        _, values = self.example("tools-and-skills")
        count = values["context_budget"]
        self.assertEqual(count([], [], []), 0)
        for invalid in (-1, 1.5, True, "90", float("nan")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                count([90], [invalid], [])

    def test_routing_costs_and_outcomes(self):
        _, values = self.example("model-choice")
        score = values["score_plan"]
        for choices, overhead, expected in (("AAAA", 0, (3, 4.0)), ("BBBB", 0, (3, 20.0)),
                                            ("AABA", 0.1, (4, 8.4)), ("ABAA", 0.1, (3, 8.4))):
            self.assertEqual(score(choices, overhead), expected)

    def test_cheapest_perfect_allocation_by_exhaustion(self):
        _, values = self.example("model-choice")
        score = values["score_plan"]
        perfect = [(score(choices, 0.1)[1], "".join(choices))
                   for choices in itertools.product("AB", repeat=4) if score(choices, 0.1)[0] == 4]
        self.assertEqual(min(perfect), (8.4, "AABA"))

    def test_routing_rejects_truncated_plans_and_invalid_overhead(self):
        _, values = self.example("model-choice")
        for choices in ("", "AAA", "AAAAA", "AACB"):
            with self.assertRaises(ValueError):
                values["score_plan"](choices)
        for overhead in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                values["score_plan"]("AAAA", overhead)

    def test_history_formula_matches_explicit_steps(self):
        for calls in range(1, 20):
            self.assertEqual(sum(1000 + 200 * step for step in range(calls)),
                             calls * 1000 + 200 * calls * (calls - 1) // 2)
        self.assertEqual(sum((1000, 1200, 1400, 1600)), 5200)
        self.assertAlmostEqual(0.95 ** 20, 0.3584859224085419)

    def test_cascade_includes_initial_call(self):
        for fraction in (0.0, 0.25, 0.5, 1.0):
            router = 0.1 + (1 - fraction) + fraction * 5
            cascade = 1 + 0.2 + fraction * 5
            self.assertAlmostEqual(cascade - router, 0.1 + fraction)

    def test_figures_are_localized_and_not_folded(self):
        for chapter, identifier in (("model-choice", "reasoning-decisions"),
                                    ("tools-and-skills", "tool-request-path")):
            for suffix, lang in (("", "zh-CN"), (".en", "en")):
                body, _ = build.render_markdown(self.source(chapter, suffix))
                self.assertEqual(Figures(body).figures[identifier], (lang, 0))
                figure = re.search(r'<figure\b.*?</figure>', body, re.S).group()
                if suffix:
                    self.assertNotRegex(figure, r"[\u4e00-\u9fff]")
                else:
                    self.assertRegex(figure, r"[\u4e00-\u9fff]")
                    self.assertNotIn("Before generation", figure)

    def test_skill_file_is_review_only(self):
        for suffix in ("", ".en"):
            example = re.search(r"```markdown\n(.*?)```", self.source("tools-and-skills", suffix), re.S).group(1)
            self.assertIn("name: review-note", example)
            self.assertIn("description:", example)
            self.assertIn("Do not edit the note or publish the draft.", example)
            self.assertNotIn("allowed-tools:", example)

    def test_navigation_and_reference_reading_status(self):
        navigation = tomllib.loads((ROOT / "site/nav.toml").read_text())
        agents = next(section for section in navigation["section"] if section["dir"] == "10-agents")
        self.assertLess(agents["order"].index("patterns.md"), agents["order"].index("tools-and-skills.md"))
        coverage = tomllib.loads((ROOT / "site/appendix-coverage.toml").read_text())
        topics = {topic["id"]: topic for chapter in coverage["chapter"] for topic in chapter["topics"]}
        self.assertEqual((topics["mcp-skills"]["state"], topics["mcp-skills"]["read"]), ("article", "body"))
        self.assertEqual((topics["thinking-mode-routing"]["state"], topics["thinking-mode-routing"]["read"]), ("article", "body"))

    def test_widget_discloses_assumptions(self):
        source = (ROOT / "site/static/tx-lab.js").read_text()
        section = source.split("var fig = TX.figure('tx-model-router'", 1)[1].split("/* --------------------------------------------- probability", 1)[0]
        self.assertIn("by construction", section)
        self.assertIn("overhead are omitted", section)
        self.assertNotIn("a wrong guess is never caught", section)
        self.assertNotIn("p.judge", section)

    def test_skill_activation_is_not_neural_activations(self):
        terms = build.load_glossary()
        for sentence in ("激活一个 Skill", "激活账户", "调用不会自动激活它"):
            rendered, _ = build.annotate(f"<p>{sentence}</p>", terms)
            self.assertNotIn("activations", rendered)
        rendered, used = build.annotate("<p>这一层的激活值</p>", terms)
        self.assertIn("（activations）", rendered)
        self.assertEqual([term["zh"] for term in used], ["激活值"])


if __name__ == "__main__":
    unittest.main()
