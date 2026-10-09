import importlib.util
import re
import tomllib
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "quant/probability"
SPEC = importlib.util.spec_from_file_location(
    "probability_checks", DIRECTORY / "code/proof_checks.py"
)
CHECKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKS)
PARITY_SPEC = importlib.util.spec_from_file_location(
    "probability_parity", ROOT / "site/paritycheck.py"
)
PARITY = importlib.util.module_from_spec(PARITY_SPEC)
PARITY_SPEC.loader.exec_module(PARITY)


class ProbabilityProofTests(unittest.TestCase):
    def test_exact_examples(self):
        self.assertTrue(CHECKS.run_checks())

    def test_extra_coin_including_empty_history(self):
        for trials in range(9):
            self.assertEqual(CHECKS.extra_coin_win(trials)[0], Fraction(1, 2))
        self.assertEqual(CHECKS.extra_coin_win(0, Fraction(1, 3))[0], Fraction(1, 3))
        self.assertEqual(CHECKS.extra_coin_win(5, 0)[0], 0)
        self.assertEqual(CHECKS.extra_coin_win(5, 1)[0], 1)

    def test_pattern_complement_symmetry(self):
        for pattern in ("H", "HH", "HT", "HHT", "HTH", "HHHT", "HTHT"):
            complement = pattern.translate(str.maketrans("HT", "TH"))
            self.assertEqual(
                CHECKS.pattern_wait(pattern, Fraction(1, 3)),
                CHECKS.pattern_wait(complement, Fraction(2, 3)),
            )

    def test_linear_solver_pivot_and_rejection(self):
        self.assertEqual(CHECKS.solve_exact([[0, 2], [1, 1]], [4, 3]), [1, 2])
        for matrix, right_side in (([[1, 1], [2, 2]], [1, 2]), ([[1, 2]], [1])):
            with self.assertRaises(ValueError):
                CHECKS.solve_exact(matrix, right_side)

    def test_pattern_validation(self):
        for pattern in ("", "HX"):
            with self.assertRaises(ValueError):
                CHECKS.pattern_wait(pattern)
        for probability in (0, 1, -1):
            with self.assertRaises(ValueError):
                CHECKS.pattern_wait("H", probability)

    def test_meeting_boundaries(self):
        self.assertEqual(CHECKS.meeting_probability(60, 0, 0), 0)
        self.assertEqual(CHECKS.meeting_probability(60, 60, 60), 1)
        self.assertEqual(CHECKS.meeting_probability(60, 10, 20), CHECKS.meeting_probability(60, 20, 10))
        for inputs in ((0, 0, 0), (60, -1, 10), (60, 61, 10)):
            with self.assertRaises(ValueError):
                CHECKS.meeting_probability(*inputs)

    def test_fair_ruin_equations(self):
        for boundary in range(2, 12):
            for state in range(1, boundary):
                hit = Fraction(state, boundary)
                self.assertEqual(
                    hit, (Fraction(state - 1, boundary) + Fraction(state + 1, boundary)) / 2
                )
                duration = state * (boundary - state)
                self.assertEqual(
                    duration,
                    1 + Fraction(
                        (state - 1) * (boundary - state + 1)
                        + (state + 1) * (boundary - state - 1), 2
                    ),
                )

    def test_retained_sources_are_bilingual_but_not_published(self):
        nav = tomllib.loads((ROOT / "site/nav.toml").read_text())
        self.assertFalse(any(item["dir"].startswith("quant") for item in nav["section"]))
        filenames = sorted(path.name for path in DIRECTORY.glob("*.md") if not path.name.endswith(".en.md"))
        self.assertEqual(len(filenames), 16)
        for filename in filenames:
            chinese = DIRECTORY / filename
            english = chinese.with_name(chinese.stem + ".en.md")
            self.assertTrue(english.exists(), filename)
            self.assertEqual(PARITY.shape(chinese.read_text()), PARITY.shape(english.read_text()), filename)
            for page in (chinese, english):
                content = page.read_text()
                self.assertNotIn("HRT", content)
                self.assertNotIn("review-deck", content)
                self.assertEqual(content.count("<details"), content.count("</details>"), str(page))
                self.assertEqual(len(re.findall(r"(?m)^\$\$$", content)) % 2, 0, str(page))

    def test_relative_source_links_exist(self):
        for page in DIRECTORY.glob("*.md"):
            for target in re.findall(r"\]\(([^)]+)\)", page.read_text()):
                if "://" in target or target.startswith("#"):
                    continue
                target = target.split("#", 1)[0]
                self.assertTrue((page.parent / target).exists(), f"{page.name}: {target}")

    def test_chapter_formulas_match_across_languages(self):
        for chinese in DIRECTORY.glob("*.md"):
            if chinese.name.endswith(".en.md"):
                continue
            english = chinese.with_name(chinese.stem + ".en.md")
            if chinese.stem in {"README", "conditional", "distributions"}:
                continue
            chinese_math = re.findall(r"(?ms)^\$\$\n(.*?)\n\$\$", chinese.read_text())
            english_math = re.findall(r"(?ms)^\$\$\n(.*?)\n\$\$", english.read_text())
            chinese_math = [formula.replace("甲领先", "A leads") for formula in chinese_math]
            self.assertEqual(chinese_math, english_math, chinese.name)


if __name__ == "__main__":
    unittest.main()
