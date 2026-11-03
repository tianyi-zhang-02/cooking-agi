from fractions import Fraction
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "site"))
import build
import paritycheck
import prep_progress
from test_mlsys_runtime_notes import Markup
from test_system_note_figures import Figures


CHAPTERS = (
    "06-systems/README",
    "06-systems/agent-observability",
    "06-systems/failure-and-recovery",
    "06-systems/human-in-the-loop",
    "06-systems/decision-support",
    "08-model-experience/README",
    "09-personal-agi/README",
)


def source(chapter, suffix=""):
    return (ROOT / f"{chapter}{suffix}.md").read_text()


def table_rows(chapter, anchor, suffix=""):
    section = source(chapter, suffix).split("{#" + anchor + "}", 1)[1]
    section = re.split(r"(?m)^#{2,4} ", section, maxsplit=1)[0]
    lines = [line for line in section.splitlines() if line.startswith("|")]
    return [[cell.strip() for cell in line.strip("|").split("|")] for line in lines[2:]]


class SystemExperienceNotesTests(unittest.TestCase):
    def test_bilingual_shape_formulas_and_language(self):
        for chapter in CHAPTERS:
            with self.subTest(chapter=chapter):
                chinese, english = source(chapter), source(chapter, ".en")
                self.assertEqual(paritycheck.shape(chinese), paritycheck.shape(english))
                self.assertEqual(re.findall(r"\$\$(.*?)\$\$", chinese, re.S),
                                 re.findall(r"\$\$(.*?)\$\$", english, re.S))
                self.assertNotRegex(re.sub(r"\[中文\]\([^)]*\)", "", english), r"[\u3400-\u9fff]")

    def test_local_links_and_fragments_in_both_languages(self):
        for chapter in CHAPTERS:
            for suffix in ("", ".en"):
                origin = ROOT / f"{chapter}{suffix}.md"
                for label, href in re.findall(r"\[([^\]]*)\]\(([^)]+)\)", origin.read_text()):
                    parsed = urlsplit(href)
                    if parsed.scheme or parsed.netloc:
                        continue
                    target = (origin.parent / unquote(parsed.path)).resolve() if parsed.path else origin
                    if target.is_dir():
                        target = target / "README.md"
                    with self.subTest(origin=origin, href=href):
                        self.assertTrue(target.is_file())
                        if target.suffix == ".md" and label not in ("中文", "English"):
                            self.assertEqual(target.name.endswith(".en.md"), suffix == ".en")
                        if parsed.fragment:
                            raw = re.sub(r"^#\s+.+$", "", target.read_text(), count=1, flags=re.M)
                            markup, _ = build.render_markdown(build.protect_mermaid(raw))
                            self.assertIn(unquote(parsed.fragment), Markup(markup).ids)

    def test_navigation_keeps_product_topics_together(self):
        with patch.dict(build.NAV), patch.dict(build.BY_SRC):
            pages, sections = build.discover(build.load_nav())
            systems = next(section for section in sections if section["dir"] == "06-systems")
            self.assertEqual(systems["group"], "systems")
            self.assertEqual([pair["zh"].src.stem for pair in systems["pages"]],
                             ["README", "agent-observability", "failure-and-recovery", "human-in-the-loop", "decision-support"])
            catalog = prep_progress.catalog(build.BY_SRC)["notes"]
            for chapter in CHAPTERS:
                note = catalog[prep_progress.identifier(chapter + ".md")]
                for language, suffix in (("zh", ""), ("en", ".en")):
                    path = chapter.removesuffix("README") + "index" if chapter.endswith("README") else chapter
                    self.assertEqual(note["pages"][language]["url"], path + suffix + ".html")
                    self.assertEqual(sum(page.src == ROOT / f"{chapter}{suffix}.md" for page in pages), 1)

    def test_task_success_and_error_budget(self):
        for suffix in ("", ".en"):
            rows = table_rows("06-systems/README", "task-success", suffix)
            counts = [int(row[1].replace(",", "")) for row in rows]
            rates = [int(row[2].strip("%")) for row in rows]
            self.assertEqual(counts, [9800, 9400, 9100])
            self.assertEqual([count / 100 for count in counts], rates)
            self.assertEqual(Fraction(10000 - counts[-1], 500), Fraction(18, 10))

    def test_trace_parallel_and_serial_time(self):
        for suffix in ("", ".en"):
            rows = table_rows("06-systems/agent-observability", "trace-timing", suffix)
            durations = [int(row[2]) for row in rows]
            intervals = [[int(value.replace(",", "").strip()) for value in row[1].split("→")] for row in rows]
            self.assertEqual([end - start for start, end in intervals], durations)
            entry, retrieval, order, generation = durations
            self.assertEqual(entry + max(retrieval, order) + generation, 1200)
            self.assertEqual(sum(durations), 1400)
            self.assertEqual(entry + max(50, order) + generation, 1200)
            self.assertEqual(entry + max(retrieval, order) + 350, 850)

    def test_review_capacity_and_sampling(self):
        self.assertEqual(600 * Fraction(8, 100), 48)
        self.assertEqual(2 * 60 / 2, 60)
        self.assertEqual(600 * Fraction(12, 100) - 60, 12)
        self.assertAlmostEqual(1 - 0.99 ** 100, 0.634, places=3)
        self.assertAlmostEqual(1 - 0.99 ** 300, 0.951, places=3)
        self.assertAlmostEqual(1 - 0.999 ** 300, 0.259, places=3)
        for suffix, required in (("", "每次独立抽取"), (".en", "independent draws")):
            text = source("06-systems/human-in-the-loop", suffix)
            self.assertIn(required, text)
            self.assertIn("12%", text)
            self.assertIn("{#review-capacity}", text)

    def test_retry_budget_and_localized_recovery_figure(self):
        for suffix, language in (("", "zh-CN"), (".en", "en")):
            rows = table_rows("06-systems/failure-and-recovery", "retry-budget", suffix)
            durations = [Fraction(row[1].removesuffix(" s")) for row in rows]
            self.assertEqual(sum(durations), Fraction(9, 2))
            self.assertEqual(5 - sum(durations), Fraction(1, 2))
            text = source("06-systems/failure-and-recovery", suffix)
            self.assertIn("3 × 3 × 3 = 27", text)
            figure = Figures(text).figures["recovery-flow"]
            self.assertEqual(figure["steps"], 4)
            self.assertEqual(figure["attributes"]["lang"], language)
            self.assertIn("worked-update--pairs", figure["attributes"]["class"].split())

    def test_experience_time_does_not_double_count_first_token(self):
        for suffix in ("", ".en"):
            self.assertIn('<div class="worked-table" markdown="1">', source("08-model-experience/README", suffix))
            rows = table_rows("08-model-experience/README", "time-to-finish", suffix)
            for column, expected in ((1, 44), (2, 20)):
                self.assertEqual(sum(Fraction(row[column].removesuffix(" s")) for row in rows[1:4]), expected)
                self.assertIn("= " + str(expected) + " s", rows[-1][column])

    def test_reliance_denominators_and_baselines(self):
        for suffix in ("", ".en"):
            rows = table_rows("06-systems/decision-support", "reliance-denominators", suffix)
            counts = [int(row[1]) for row in rows]
            correct = [int(row[2]) for row in rows]
            self.assertEqual(sum(counts), 100)
            self.assertEqual(Fraction(correct[2], counts[2]), Fraction(4, 5))
            self.assertEqual(Fraction(correct[3], counts[3]), Fraction(3, 4))
            self.assertEqual((counts[0] + counts[3], counts[0] + counts[2], sum(correct)), (60, 70, 79))

    def test_new_terms_repeat_without_double_annotation(self):
        terms = build.load_glossary()
        for chinese, english in (("关键路径", "critical path"), ("幂等性", "idempotency"), ("流式输出", "streaming")):
            with self.subTest(term=chinese):
                markup, _ = build.annotate(f"<p>{chinese}，{chinese}。</p>", terms)
                self.assertEqual(markup.count(f"（{english}）"), 2)
                explicit, _ = build.annotate(f"<p>{chinese}（{english}）</p>", terms)
                self.assertEqual(explicit.count(english), 1)


if __name__ == "__main__":
    unittest.main()
