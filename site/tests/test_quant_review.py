import importlib.util
import math
import re
import statistics
import tomllib
import unittest
from fractions import Fraction
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUANT = ROOT / "quant"
SPEC = importlib.util.spec_from_file_location("quant_review_checks", QUANT / "code/review_checks.py")
CHECKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKS)
PARITY_SPEC = importlib.util.spec_from_file_location("quant_parity", ROOT / "site/paritycheck.py")
PARITY = importlib.util.module_from_spec(PARITY_SPEC)
PARITY_SPEC.loader.exec_module(PARITY)


class QuantReviewTests(unittest.TestCase):
    def test_examples(self):
        self.assertTrue(CHECKS.run_checks())

    def test_hypergeometric_moments_and_edges(self):
        for population in range(2, 11):
            for targets in range(population + 1):
                probability = Fraction(targets, population)
                for draws in range(population + 1):
                    mean, variance = CHECKS.distribution_moments(
                        CHECKS.hypergeometric(population, targets, draws)
                    )
                    self.assertEqual(mean, draws * probability)
                    self.assertEqual(
                        variance,
                        draws * probability * (1 - probability)
                        * Fraction(population - draws, population - 1),
                    )
        self.assertEqual(CHECKS.hypergeometric(1, 1, 1), {1: Fraction(1)})
        for arguments in ((0, 0, 0), (5, 6, 1), (5, 1, 6), (5, -1, 0)):
            with self.assertRaises(ValueError):
                CHECKS.hypergeometric(*arguments)

    def test_derangements_against_enumeration(self):
        for size in range(7):
            total = sum(
                all(position != value for position, value in enumerate(order))
                for order in permutations(range(size))
            )
            self.assertEqual(
                CHECKS.derangement_probability(size), Fraction(total, math.factorial(size))
            )

    def test_die_values_are_monotone_and_bounded(self):
        previous = Fraction()
        for rolls in range(1, 12):
            value = CHECKS.die_stopping_value(rolls)
            self.assertGreaterEqual(value, previous)
            self.assertLessEqual(value, 6)
            previous = value
        self.assertEqual(CHECKS.die_stopping_value(3, sides=1), 1)

    def test_replication_matches_both_states(self):
        for payoff_up, payoff_down in ((20, 0), (0, 20), (7, 7), (-3, 8)):
            price, delta, cash, probability = CHECKS.replication(
                100, Fraction(6, 5), Fraction(4, 5), Fraction(21, 20), payoff_up, payoff_down
            )
            self.assertEqual(delta * 120 + cash * Fraction(21, 20), payoff_up)
            self.assertEqual(delta * 80 + cash * Fraction(21, 20), payoff_down)
            self.assertEqual(
                price, (probability * payoff_up + (1 - probability) * payoff_down) / Fraction(21, 20)
            )
        with self.assertRaises(ValueError):
            CHECKS.replication(100, 1.2, 0.8, 1.3, 20, 0)

    def test_bisection_boundaries_and_failure_modes(self):
        self.assertEqual(CHECKS.bisect_root(lambda value: value, 0, 1), 0)
        self.assertEqual(CHECKS.bisect_root(lambda value: value - 1, 0, 1), 1)
        self.assertAlmostEqual(CHECKS.bisect_root(lambda value: -value, -1, 2), 0, places=9)
        with self.assertRaises(ValueError):
            CHECKS.bisect_root(lambda value: value * value + 1, -1, 1)
        with self.assertRaises(ValueError):
            CHECKS.bisect_root(lambda value: math.nan, -1, 1)
        with self.assertRaises(ValueError):
            CHECKS.bisect_root(lambda value: value, 0, 1, tolerance=0)
        with self.assertRaises(RuntimeError):
            CHECKS.bisect_root(lambda value: value * value - 2, 0, 2, max_iterations=1)

    def test_running_moments_with_large_offset(self):
        values = [10**10 + value for value in range(1, 21)]
        running = CHECKS.RunningMoments()
        with self.assertRaises(ValueError):
            running.sample_variance()
        for value in values:
            running.add(value)
        self.assertAlmostEqual(running.mean, statistics.mean(values))
        self.assertAlmostEqual(running.sample_variance(), statistics.variance(values))
        with self.assertRaises(ValueError):
            running.add(math.inf)

    def test_streaming_median_every_prefix(self):
        running = CHECKS.StreamingMedian()
        with self.assertRaises(ValueError):
            running.median()
        values = [8, -2, 8, 1, 0, 300, -20, 3]
        for length, value in enumerate(values, start=1):
            running.add(value)
            self.assertEqual(running.median(), statistics.median(values[:length]))
        with self.assertRaises(ValueError):
            running.add(math.nan)

    def test_exact_control_variate_variance(self):
        mean_square = Fraction(1, 3)
        variance_square = Fraction(1, 5) - mean_square**2
        variance_uniform = Fraction(1, 12)
        covariance = Fraction(1, 4) - Fraction(1, 3) * Fraction(1, 2)
        coefficient = covariance / variance_uniform
        result = variance_square - 2 * coefficient * covariance + coefficient**2 * variance_uniform
        self.assertEqual(result, Fraction(1, 180))
        self.assertEqual(result / variance_square, Fraction(1, 16))

    def test_retained_quant_sources_are_bilingual_and_unpublished(self):
        nav = tomllib.loads((ROOT / "site/nav.toml").read_text())
        sections = [section for section in nav["section"] if section.get("group") == "quant"]
        self.assertEqual(sections, [])
        sources = [page for page in QUANT.rglob('*.md') if not page.name.endswith('.en.md')]
        self.assertEqual(len(sources), 30)
        for chinese in sources:
            english = chinese.with_name(chinese.stem + '.en.md')
            self.assertTrue(english.exists(), str(english))
            self.assertEqual(PARITY.shape(chinese.read_text()), PARITY.shape(english.read_text()), str(chinese))

    def test_links_and_details(self):
        for page in QUANT.rglob("*.md"):
            content = page.read_text()
            self.assertEqual(content.count("<details"), content.count("</details>"), str(page))
            self.assertNotIn("review-deck", content)
            for target in re.findall(r"\]\(([^)]+)\)", content):
                if "://" in target or target.startswith("#"):
                    continue
                self.assertTrue((page.parent / target.split("#", 1)[0]).exists(), f"{page}: {target}")

    def test_new_display_formulas_match(self):
        for page in QUANT.rglob("*.md"):
            if page.name.endswith(".en.md"):
                continue
            if page.parent.name == "probability" and page.stem not in {
                "counting", "distribution-toolkit", "joint-and-order"
            }:
                continue
            english = page.with_name(page.stem + ".en.md")
            formulas = re.findall(r"(?ms)^\$\$\n(.*?)\n\$\$", page.read_text())
            translated = re.findall(r"(?ms)^\$\$\n(.*?)\n\$\$", english.read_text())
            self.assertEqual(formulas, translated, str(page))

    def test_coverage_does_not_claim_exercise_completeness(self):
        for name, marker in (("README.md", "逐项验收"), ("README.en.md", "item by item")):
            self.assertIn(marker, (QUANT / name).read_text())


if __name__ == "__main__":
    unittest.main()
