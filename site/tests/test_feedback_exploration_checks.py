import importlib.util
import itertools
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "feedback_exploration_checks",
    ROOT / "05-post-training/deep-rl/code/feedback_exploration_checks.py")
checks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checks)


class RewardSignalTests(unittest.TestCase):
    def test_masked_example_and_numeric_gradient(self):
        logits, labels = [math.log(3), math.log(3), math.nan], [1, 0, None]
        result = checks.masked_binary_loss(logits, labels)
        self.assertAlmostEqual(result["loss"], (-math.log(0.75) - math.log(0.25)) / 2)
        self.assertEqual(result["gradient"], [-0.125, 0.375, 0.0])
        for index in [0, 1]:
            lower, upper = logits[:], logits[:]
            lower[index] -= 1e-5
            upper[index] += 1e-5
            numeric = (checks.masked_binary_loss(upper, labels)["loss"]
                       - checks.masked_binary_loss(lower, labels)["loss"]) / 2e-5
            self.assertAlmostEqual(numeric, result["gradient"][index], places=8)

    def test_unknown_is_not_a_negative_label(self):
        known = checks.masked_binary_loss([0, 2], [1, 0])
        masked = checks.masked_binary_loss([0, 2], [1, None])
        self.assertNotEqual(known["loss"], masked["loss"])
        self.assertEqual(masked["gradient"][1], 0)
        self.assertEqual(masked["gradient"][0], -0.5)

    def test_stability_soft_labels_and_validation(self):
        result = checks.masked_binary_loss([-1000, 1000, 0], [1, 0, 0.25])
        self.assertTrue(math.isfinite(result["loss"]))
        self.assertAlmostEqual(result["gradient"][2], 0.25 / 3)
        for logits, labels in [([], []), ([0], []), ([0], [None]), ([math.nan], [0]),
                               ([0], [1.1]), ([0], [math.nan])]:
            with self.assertRaises(ValueError):
                checks.masked_binary_loss(logits, labels)

    def test_aggregation_can_reverse_rankings(self):
        first, second = checks.aggregate_scores([0.7, 0.99]), checks.aggregate_scores([0.82, 0.82])
        self.assertAlmostEqual(first["product"], 0.693)
        self.assertAlmostEqual(second["product"], 0.6724)
        self.assertGreater(first["product"], second["product"])
        self.assertLess(first["minimum"], second["minimum"])
        self.assertAlmostEqual(checks.aggregate_scores([0.9] * 4)["product"], 0.6561)
        for scores in [[], [math.nan], [-0.1], [1.1]]:
            with self.assertRaises(ValueError):
                checks.aggregate_scores(scores)

    def test_equal_budget_verifier_example(self):
        cheap = checks.selection_success(4, 0.25, 0.9, 0.1)
        expensive = checks.selection_success(2, 0.25, 0.9, 0.02)
        self.assertAlmostEqual(cheap, 0.5785)
        self.assertAlmostEqual(expensive, 0.415)
        self.assertAlmostEqual(checks.selection_success(4, 0.25), 1 - 0.75 ** 4)
        self.assertEqual(checks.selection_success(4, 0, 1, 0), 0)
        self.assertEqual(checks.selection_success(4, 1), 1)
        self.assertAlmostEqual(checks.selection_success(4, 0.25, 0, 0), 0.25)

    def test_selection_matches_enumeration(self):
        for correct_rate, recall, false_positive in [(0.25, 0.9, 0.1), (0.25, 0.9, 0.02),
                                                     (0.4, 1, 0), (1, 1, 0), (0.25, 0, 0)]:
            states = [(True, True, correct_rate * recall),
                      (True, False, correct_rate * (1 - recall)),
                      (False, True, (1 - correct_rate) * false_positive),
                      (False, False, (1 - correct_rate) * (1 - false_positive))]
            for samples in range(1, 5):
                expected = 0.0
                for sequence in itertools.product(states, repeat=samples):
                    chosen = next((entry for entry in sequence if entry[1]), sequence[0])
                    expected += chosen[0] * math.prod(entry[2] for entry in sequence)
                self.assertAlmostEqual(checks.selection_success(samples, correct_rate, recall,
                                                                false_positive), expected)

    def test_continuation_rates_depend_on_policy(self):
        self.assertEqual(checks.empirical_success_rate([1, 0, 0, 0]), 0.25)
        self.assertEqual(checks.empirical_success_rate([1, 1, 1, 0]), 0.75)
        for outcomes in [[], [0.5], [math.nan]]:
            with self.assertRaises(ValueError):
                checks.empirical_success_rate(outcomes)


class ExplorationTests(unittest.TestCase):
    def test_ucb_example(self):
        scores = checks.ucb_indices([8, 1], [10, 2])
        self.assertAlmostEqual(scores[0], 0.8 + math.sqrt(2 * math.log(12) / 10))
        self.assertAlmostEqual(scores[1], 0.5 + math.sqrt(math.log(12)))
        self.assertGreater(scores[1], scores[0])
        self.assertTrue(all(score > 1 for score in scores))
        for successes, counts in [([], []), ([0], [0]), ([2], [1]), ([1], [2.0]), ([1], [2, 3])]:
            with self.assertRaises(ValueError):
                checks.ucb_indices(successes, counts)

    def test_beta_update(self):
        self.assertEqual(checks.beta_posterior(8, 2), {"alpha": 9, "beta": 3, "mean": 0.75})
        self.assertEqual(checks.beta_posterior(1, 1), {"alpha": 2, "beta": 2, "mean": 0.5})
        self.assertAlmostEqual(checks.beta_posterior(2, 1)["mean"], 0.6)
        self.assertAlmostEqual(checks.beta_posterior(1, 2)["mean"], 0.4)
        for successes, failures, prior in [(-1, 0, (1, 1)), (1, 0, (0, 1)), (1, 0, (math.nan, 1))]:
            with self.assertRaises(ValueError):
                checks.beta_posterior(successes, failures, prior)

    def test_information_gain_and_noise(self):
        result = checks.probe_information(0.8)
        self.assertAlmostEqual(result["information_bits"], 0.2780719051126377)
        self.assertAlmostEqual(result["information_bits"], result["posterior_kl_bits"])
        self.assertEqual(checks.probe_information(0.5)["information_bits"], 0)
        self.assertEqual(checks.probe_information(1)["information_bits"], 1)
        self.assertAlmostEqual(checks.probe_information(0.2)["information_bits"], result["information_bits"])
        for value in [-0.1, 1.1, math.nan]:
            with self.assertRaises(ValueError):
                checks.probe_information(value)

    def test_pseudo_count_reconstruction(self):
        result = checks.pseudo_counts(0.2, 0.25)
        self.assertAlmostEqual(result["local"], 3)
        self.assertAlmostEqual(result["total"], 15)
        for total in [2, 15, 100]:
            for local in range(1, total):
                before, after = local / total, (local + 1) / (total + 1)
                recovered = checks.pseudo_counts(before, after)
                self.assertAlmostEqual(recovered["local"], local)
                self.assertAlmostEqual(recovered["total"], total)
        for before, after in [(0.25, 0.2), (0.2, 0.2), (0, 0.2), (0.2, 1), (math.nan, 0.2)]:
            with self.assertRaises(ValueError):
                checks.pseudo_counts(before, after)


if __name__ == "__main__":
    unittest.main()
