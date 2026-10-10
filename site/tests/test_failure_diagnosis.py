import ast
from html.parser import HTMLParser
from pathlib import Path
import re
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "practice/post-training/experiments-and-release"
sys.path.insert(0, str(ROOT / "site"))
sys.path.insert(0, str(ROOT / "practice/rag/code"))
import build
from evidence_pipeline import Chunk, pack_evidence


class VisibleFigure(HTMLParser):
    def __init__(self):
        super().__init__()
        self.details_depth = 0
        self.figure_depth = 0
        self.figures = []

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if tag == "details":
            self.details_depth += 1
        if tag == "figure":
            self.figure_depth += 1
            self.figures.append({
                "language": attributes.get("lang"),
                "hidden": self.details_depth > 0,
                "text": "",
                "steps": 0,
            })
        if tag == "li" and self.figure_depth:
            self.figures[-1]["steps"] += 1

    def handle_endtag(self, tag):
        if tag == "details":
            self.details_depth -= 1
        if tag == "figure":
            self.figure_depth -= 1

    def handle_data(self, data):
        if self.figure_depth:
            self.figures[-1]["text"] += data


class FailureDiagnosisTests(unittest.TestCase):
    def source(self, language=""):
        return Path(f"{CHAPTER}{language}.md").read_text()

    def snippets(self, language=""):
        return re.findall(r"```python\n(.*?)```", self.source(language), re.S)

    def passages(self):
        return [
            Chunk("weekday", "hours", 1, "North Hall closes at 20:00 on weekdays.",
                  frozenset({"public"})),
            Chunk("sunday", "hours", 1, "North Hall closes at 18:00 on Sundays.",
                  frozenset({"public"})),
        ]

    def test_bilingual_code_is_identical(self):
        self.assertEqual(self.snippets(), self.snippets(".en"))
        self.assertEqual(len(self.snippets()), 1)
        ast.parse(self.snippets()[0])

    def test_documented_example_runs_from_repository_root(self):
        result = subprocess.run([sys.executable, "-c", self.snippets()[0]], cwd=ROOT,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [
            "retrieved: ['weekday', 'sunday']",
            "provided: ['weekday']",
            "oracle: ['sunday']",
        ])

    def test_budget_discards_required_passage_without_mutating_retrieval(self):
        retrieved = self.passages()
        packed, used = pack_evidence(retrieved, 7)
        self.assertEqual([chunk.chunk_id for chunk in retrieved], ["weekday", "sunday"])
        self.assertEqual([chunk.chunk_id for chunk in packed], ["weekday"])
        self.assertEqual(used, 7)

    def test_oracle_comparison_preserves_budget(self):
        retrieved = self.passages()
        baseline, baseline_used = pack_evidence(retrieved, 7)
        oracle, oracle_used = pack_evidence(list(reversed(retrieved)), 7)
        self.assertEqual(baseline_used, oracle_used)
        self.assertNotIn(retrieved[1], baseline)
        self.assertIn(retrieved[1], oracle)

    def test_under_budget_and_sufficient_budget_cases(self):
        for budget, expected_count, expected_used in [(0, 0, 0), (6, 0, 0), (7, 1, 7),
                                                       (13, 1, 7), (14, 2, 14)]:
            with self.subTest(budget=budget):
                packed, used = pack_evidence(self.passages(), budget)
                self.assertEqual(len(packed), expected_count)
                self.assertEqual(used, expected_used)
                self.assertLessEqual(used, budget)

    def test_bad_budgets_are_not_silently_accepted(self):
        for budget in [-1, 7.5, True, "7"]:
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                pack_evidence(self.passages(), budget)

    def test_figure_is_visible_and_localized(self):
        for suffix, language in [("", "zh-CN"), (".en", "en")]:
            body, _ = build.render_markdown(self.source(suffix))
            parser = VisibleFigure()
            parser.feed(body)
            self.assertEqual(len(parser.figures), 1)
            figure = parser.figures[0]
            self.assertEqual(figure["language"], language)
            self.assertFalse(figure["hidden"])
            self.assertEqual(figure["steps"], 3)
            if suffix:
                self.assertNotRegex(figure["text"], r"[\u4e00-\u9fff]")
            else:
                self.assertRegex(figure["text"], r"[\u4e00-\u9fff]")
                self.assertNotIn("Retrieved", figure["text"])

    def test_new_sections_have_stable_shared_anchors(self):
        for suffix in ["", ".en"]:
            body, _ = build.render_markdown(self.source(suffix))
            for anchor in ["trace-a-failure", "controlled-replay", "inspect-training",
                           "loss-versus-behavior", "choose-a-fix", "regression-families"]:
                self.assertEqual(body.count(f'id="{anchor}"'), 1)

    def test_sources_and_follow_up_paths_exist_in_both_languages(self):
        for suffix in ["", ".en"]:
            source = self.source(suffix)
            for address in ["aclanthology.org/2020.acl-main.442/", "arxiv.org/abs/2106.09685",
                            "huggingface.co/docs/transformers/chat_templating",
                            "huggingface.co/docs/trl/v0.29.0/en/sft_trainer"]:
                self.assertIn(address, source)
            for name in ["training-plan", "data-pipeline", "data-and-objectives"]:
                self.assertIn(f"({name}{suffix}.md)", source)
            case_studies = (ROOT / f"07-evaluation/llm-as-a-judge/case-studies{suffix}.md").read_text()
            self.assertIn(f"experiments-and-release{suffix}.md#trace-a-failure", case_studies)

    def test_ambiguous_correlation_is_not_annotated_as_relevance(self):
        for sentence in ["样本之间的相关性", "分组相关性", "相关性不等于因果"]:
            with self.subTest(sentence=sentence):
                output, _ = build.annotate(f"<p>{sentence}</p>", build.load_glossary())
                self.assertNotIn("relevance", output)

    def test_unambiguous_terms_get_the_right_english_annotation(self):
        for chinese, english in [("组内相关性", "within-group correlation"),
                                 ("统计相关性", "correlation"), ("检索相关性", "relevance")]:
            with self.subTest(term=chinese):
                output, used = build.annotate(f"<p>{chinese}</p>", build.load_glossary())
                self.assertIn(f"（{english}）", output)
                self.assertEqual([term["zh"] for term in used], [chinese])


if __name__ == "__main__":
    unittest.main()
