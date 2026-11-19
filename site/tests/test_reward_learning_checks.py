import importlib.util
import contextlib
import io
import itertools
import math
import re
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "reward_learning_checks", ROOT / "05-post-training/deep-rl/code/reward_learning_checks.py")
checks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checks)


class MaxentTests(unittest.TestCase):
    def test_worked_counts_and_gradient(self):
        theta = math.log(2)
        result = checks.maxent_statistics([0, 1, 2], [1, 2, 7], theta)
        for actual, expected in zip(result["probabilities"], [1 / 7, 2 / 7, 4 / 7]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(result["empirical_mean"], 1.6)
        self.assertAlmostEqual(result["model_mean"], 10 / 7)
        self.assertAlmostEqual(result["gradient"], 6 / 35)
        self.assertAlmostEqual(result["hessian"], -26 / 49)
        step = 1e-5
        lower = checks.maxent_statistics([0, 1, 2], [1, 2, 7], theta - step)
        upper = checks.maxent_statistics([0, 1, 2], [1, 2, 7], theta + step)
        self.assertAlmostEqual((upper["log_likelihood"] - lower["log_likelihood"]) / (2 * step),
                               result["gradient"], places=8)
        self.assertAlmostEqual((upper["gradient"] - lower["gradient"]) / (2 * step),
                               result["hessian"], places=8)
        after = checks.maxent_statistics([0, 1, 2], [1, 2, 7], theta + 0.5 * result["gradient"])
        self.assertGreater(after["log_likelihood"], result["log_likelihood"])

    def test_dynamic_program_matches_enumeration(self):
        for theta in [-20, 0, math.log(2), 20]:
            exact = checks.maxent_statistics([0, 1, 2], [1, 2, 7], theta)
            tree = checks.deterministic_tree(theta)
            self.assertAlmostEqual(tree["root_value"], exact["log_partition"])
            for actual, expected in zip(tree["leaf_probabilities"], exact["probabilities"]):
                self.assertAlmostEqual(actual, expected)

    def test_stochastic_transition_cannot_be_reweighted(self):
        result = checks.stochastic_choice()
        self.assertEqual(result["causal_safe_risky"], [0.5, 0.5])
        self.assertEqual(result["actual_lucky_given_risky"], 0.5)
        self.assertGreater(result["naive_safe_risky"][1], 0.6)
        self.assertAlmostEqual(result["naive_lucky_given_risky"], 1 / (1 + math.exp(-2)))

    def test_shift_invariance_and_validation(self):
        for offset in [-1000, 1000]:
            for actual, expected in zip(checks.softmax([offset, offset + 1, offset + 2]),
                                        checks.softmax([0, 1, 2])):
                self.assertAlmostEqual(actual, expected)
        for features, counts in [([], []), ([0, 1], [1]), ([0, 1], [-1, 2]), ([0], [0])]:
            with self.assertRaises(ValueError):
                checks.maxent_statistics(features, counts, 1)
        for bad in [math.nan, math.inf]:
            with self.assertRaises(ValueError):
                checks.maxent_statistics([0, 1], [1, 1], bad)


class GailTests(unittest.TestCase):
    def test_labels_rewards_and_js_identity(self):
        result = checks.gail_statistics([0.75, 0.25], [0.25, 0.75])
        self.assertEqual(result["expert_probability"], [0.75, 0.25])
        self.assertAlmostEqual(result["rewards"][0], math.log(4))
        self.assertAlmostEqual(result["rewards"][1], -math.log(0.75))
        self.assertAlmostEqual(result["value"], -math.log(4) + 2 * result["js_divergence"])
        self.assertGreater(0.25 * 0.75 * (result["rewards"][0] - result["rewards"][1]), 0)

    def test_matching_and_disjoint_supports(self):
        match = checks.gail_statistics([0.7, 0.3, 0], [0.7, 0.3, 0])
        self.assertEqual(match["expert_probability"], [0.5] * 3)
        self.assertEqual(match["js_divergence"], 0)
        self.assertTrue(all(math.isclose(reward, math.log(2)) for reward in match["rewards"]))
        disjoint = checks.gail_statistics([1, 0], [0, 1])
        self.assertEqual(disjoint["value"], 0)
        self.assertAlmostEqual(disjoint["js_divergence"], math.log(2))
        self.assertEqual(disjoint["rewards"], [math.inf, 0])

    def test_invalid_distributions(self):
        for expert, learner in [([1], [0.5, 0.5]), ([-0.5, 1.5], [0.5, 0.5]),
                                ([1, 1], [0.5, 0.5]), ([math.nan, 0], [0.5, 0.5])]:
            with self.assertRaises(ValueError):
                checks.gail_statistics(expert, learner)


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.scores = [math.log(4), math.log(2), 0]
        self.order = [0, 1, 2]

    def test_worked_ranking_and_pair_products(self):
        result = checks.ranking_statistics(self.scores, self.order)
        self.assertAlmostEqual(result["probability"], 8 / 21)
        pairs = checks.all_pair_losses(self.scores, self.order)
        self.assertAlmostEqual(math.exp(-sum(pairs)), 16 / 45)
        self.assertNotAlmostEqual(result["loss"], sum(pairs))
        for actual, expected in zip(result["gradient"], [-3 / 7, -1 / 21, 10 / 21]):
            self.assertAlmostEqual(actual, expected)

    def test_probabilities_over_all_orders(self):
        total = sum(checks.ranking_statistics(self.scores, order)["probability"]
                    for order in itertools.permutations(self.order))
        pair_product_total = sum(math.exp(-sum(checks.all_pair_losses(self.scores, order)))
                                 for order in itertools.permutations(self.order))
        self.assertAlmostEqual(total, 1)
        self.assertNotAlmostEqual(pair_product_total, 1)
        partial = sum(checks.ranking_statistics(self.scores, order)["probability"]
                      for order in [[0, 1, 2], [1, 0, 2]])
        self.assertAlmostEqual(partial, 64 / 105)

    def test_article_code_matches_companion(self):
        for suffix in ["", ".en"]:
            source = (ROOT / f"05-post-training/reward-ranking{suffix}.md").read_text()
            blocks = re.findall(r"```python\n(.*?)```", source, flags=re.DOTALL)
            self.assertEqual(len(blocks), 1)
            namespace = {}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(blocks[0], namespace)
            for scores in [self.scores, [1000, -1000], [0, 0, 0], [2, 1, 3]]:
                self.assertAlmostEqual(namespace["ranking_loss"](scores),
                                       checks.ranking_statistics(scores, list(range(len(scores))))["loss"])

    def test_gradient_by_finite_differences(self):
        result = checks.ranking_statistics(self.scores, self.order)
        step = 1e-5
        for index in self.order:
            lower, upper = self.scores.copy(), self.scores.copy()
            lower[index] -= step
            upper[index] += step
            gradient = (checks.ranking_statistics(upper, self.order)["loss"]
                        - checks.ranking_statistics(lower, self.order)["loss"]) / (2 * step)
            self.assertAlmostEqual(gradient, result["gradient"][index], places=8)
        self.assertAlmostEqual(sum(result["gradient"]), 0)

    def test_two_answers_reduce_to_bradley_terry(self):
        for scores in [[2, 1], [1, 2], [-1000, 1000], [1000, 1000]]:
            result = checks.ranking_statistics(scores, [0, 1])
            self.assertAlmostEqual(result["loss"], checks.pair_loss(*scores))
            self.assertTrue(math.isfinite(result["loss"]))

    def test_offset_not_scale_invariant(self):
        original = checks.ranking_statistics(self.scores, self.order)
        shifted = checks.ranking_statistics([score + 1000 for score in self.scores], self.order)
        scaled = checks.ranking_statistics([score * 2 for score in self.scores], self.order)
        self.assertAlmostEqual(original["loss"], shifted["loss"])
        self.assertNotAlmostEqual(original["loss"], scaled["loss"])

    def test_prompt_and_pair_weighting(self):
        result = checks.comparison_means([[0.2], [0.8] * 6])
        self.assertAlmostEqual(result["pair_mean"], 5 / 7)
        self.assertAlmostEqual(result["prompt_mean"], 0.5)

    def test_invalid_orders_and_groups(self):
        for order in [[0, 1], [0, 0, 2], [0, 1, 3]]:
            with self.assertRaises(ValueError):
                checks.ranking_statistics(self.scores, order)
        for groups in [[], [[]], [[math.inf]]]:
            with self.assertRaises(ValueError):
                checks.comparison_means(groups)


if __name__ == "__main__":
    unittest.main()
