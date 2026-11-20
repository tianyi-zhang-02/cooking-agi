import importlib.util
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
NOTES = ROOT / "05-post-training/deep-rl"
SPEC = importlib.util.spec_from_file_location("offline_model_checks", NOTES / "code/offline_model_checks.py")
CHECKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKS)


class OfflineModelChecks(unittest.TestCase):
    def test_stepwise_penalty_and_discounted_total_agree(self):
        result = CHECKS.penalized_return([2, 4], [0.1, 0.4], 3, 0.9)
        self.assertEqual(result["adjusted_rewards"], [1.7, 2.8])
        self.assertAlmostEqual(result["raw_return"], 5.6)
        self.assertAlmostEqual(result["penalty"], 1.38)
        self.assertAlmostEqual(result["adjusted_return"], 4.22)
        self.assertAlmostEqual(result["raw_return"] - result["penalty"], result["adjusted_return"])
        self.assertAlmostEqual(CHECKS.penalized_return([2, 4], [0.1, 0.4], 3, 1)["adjusted_return"], 4.5)
        self.assertAlmostEqual(CHECKS.penalized_return([2, 4], [0.1, 0.4], 3, 0)["adjusted_return"], 1.7)
        self.assertAlmostEqual(CHECKS.penalized_return([2, 4], [0.1, 0.4], 0, 0.9)["adjusted_return"], 5.6)

    def test_halt_tail_and_collapsed_target_agree(self):
        result = CHECKS.halt_values(1, 2, 0.9)
        self.assertAlmostEqual(result["halt"], -20)
        self.assertAlmostEqual(result["entry_target"], -17)
        tail = sum(0.9 ** step * -2 for step in range(1000))
        self.assertAlmostEqual(tail, result["halt"])
        self.assertAlmostEqual(1 + 0.9 * tail, result["entry_target"])
        self.assertNotAlmostEqual(result["entry_target"], -2)
        self.assertAlmostEqual(CHECKS.halt_values(1, 2, 0.99)["halt"], -200)
        self.assertEqual(CHECKS.halt_values(1, 2, 0)["entry_target"], 1)

    def test_conservative_fit_reverses_ranking(self):
        first = CHECKS.conservative_fit([2, 5], [0.9, 0.1], [0.5, 0.5], [0.7, 0.3], 1)
        second = CHECKS.conservative_fit([2, 5], [0.9, 0.1], [0.5, 0.5], [0.7, 0.3], 2)
        for actual, expected in zip(first, [18 / 7, 11 / 3]):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(second, [22 / 7, 7 / 3]):
            self.assertAlmostEqual(actual, expected)
        self.assertLess(first[0], first[1])
        self.assertGreater(second[0], second[1])

    def test_conservative_fit_minimizes_original_objective(self):
        targets, data, conservative, regression = [2, 5], [.9, .1], [.5, .5], [.7, .3]
        strength = 2
        solution = CHECKS.conservative_fit(targets, data, conservative, regression, strength)

        def objective(values):
            return sum(strength * (model_mass - data_mass) * value
                       + 0.5 * weight * (value - target) ** 2
                       for target, data_mass, model_mass, weight, value
                       in zip(targets, data, conservative, regression, values))

        for index in range(len(solution)):
            above, below = solution.copy(), solution.copy()
            above[index] += 1e-5
            below[index] -= 1e-5
            self.assertAlmostEqual((objective(above) - objective(below)) / 2e-5, 0, places=8)
            above[index] += 0.1
            below[index] -= 0.1
            self.assertGreater(objective(above), objective(solution))
            self.assertGreater(objective(below), objective(solution))
        self.assertEqual(CHECKS.conservative_fit(targets, data, data, regression, 2), targets)
        self.assertEqual(CHECKS.conservative_fit(targets, data, conservative, regression, 0), targets)

    def test_reject_invalid_inputs(self):
        for uncertainty in ([-0.1, .4], [math.nan, .4], [.1]):
            with self.assertRaises(ValueError):
                CHECKS.penalized_return([2, 4], uncertainty, 3, .9)
        for discount in (-.1, 1.1, math.nan):
            with self.assertRaises(ValueError):
                CHECKS.penalized_return([2], [.1], 3, discount)
        for discount in (-.1, 1, math.nan):
            with self.assertRaises(ValueError):
                CHECKS.halt_values(1, 2, discount)
        for distribution in ([0, 1], [.5, .4], [-.1, 1.1], [math.nan, .5], [1]):
            with self.assertRaises(ValueError):
                CHECKS.conservative_fit([2, 5], [.9, .1], [.5, .5], distribution, 1)
        for strength in (-1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                CHECKS.conservative_fit([2, 5], [.9, .1], [.5, .5], [.7, .3], strength)

    def test_new_note_keeps_matching_explicit_anchors(self):
        for suffix in ("", ".en"):
            source = (NOTES / f"offline-models{suffix}.md").read_text()
            for anchor in ("data-flow", "mopo-penalty", "morel-halt", "combo-critic", "combo-example",
                           "comparison", "rollout-cutoff", "implementation-checks", "review"):
                self.assertIn("{#" + anchor + "}", source)
            self.assertIn("code/offline_model_checks.py", source)


if __name__ == "__main__":
    unittest.main()
