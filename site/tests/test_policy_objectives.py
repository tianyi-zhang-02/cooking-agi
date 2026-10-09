import math
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
LESSONS = ("policy-ratios", "async-policy-learning", "dapo")


def snippets(lesson, suffix=".md"):
    text = (ROOT / "05-post-training" / (lesson + suffix)).read_text()
    return re.findall(r"```python\n(.*?)```", text, re.S)


class PolicyObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespaces = {}
        for lesson in LESSONS:
            namespace = {}
            for snippet in snippets(lesson):
                exec(compile(snippet, lesson, "exec"), namespace)
            cls.namespaces[lesson] = namespace

    def test_bilingual_code_is_identical(self):
        for lesson in LESSONS:
            self.assertEqual(snippets(lesson), snippets(lesson, ".en.md"))

    def test_geometric_ratio_ignores_non_actions(self):
        ratio = self.namespaces["policy-ratios"]["sequence_ratio"]
        self.assertAlmostEqual(ratio([math.log(4), math.log(0.25)], [True, True]), 1)
        self.assertAlmostEqual(ratio([math.log(4), float("nan")], [True, False]), 4)
        self.assertAlmostEqual(ratio([0.1] * 1000, [True] * 1000), math.exp(0.1))
        for values, mask in [([], []), ([0], [False]), ([0], []), ([0], [1]), ([float("nan")], [True])]:
            with self.assertRaises(ValueError):
                ratio(values, mask)

    def test_ppo_clipping_depends_on_advantage_sign(self):
        objective = self.namespaces["policy-ratios"]["clipped_surrogate"]
        for ratio, advantage, expected in [(1.3, 2, 2.4), (0.7, 2, 1.4), (1.3, -2, -2.6), (0.7, -2, -1.6)]:
            self.assertAlmostEqual(objective(ratio, advantage), expected)
        for ratio in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                objective(ratio, 1)

    def test_stop_gradient_finite_difference(self):
        namespace = self.namespaces["policy-ratios"]
        self.assertAlmostEqual(namespace["derivative"], 2, places=6)
        log_probability = namespace["current_logp"]
        step = namespace["step"]
        wrong_derivative = (0.2 / math.exp(log_probability + step) - 0.2 / math.exp(log_probability - step)) / (2 * step)
        self.assertAlmostEqual(wrong_derivative, -2, places=6)

    def test_dis_forward_mask_and_extremes(self):
        weight = self.namespaces["async-policy-learning"]["dis_weight"]
        self.assertAlmostEqual(weight(-2, -2), 1)
        self.assertEqual(weight(-10000, -2), 0)
        self.assertEqual(weight(-2, -10000), 0)
        for current, behavior in [(float("nan"), -2), (1, -2), (-2, float("-inf"))]:
            with self.assertRaises(ValueError):
                weight(current, behavior)

    def test_dynamic_group_probability(self):
        self.assertEqual(1 - 0.5 ** 4 - (1 - 0.5) ** 4, 0.875)
        self.assertAlmostEqual(1 - 0.99 ** 4 - (1 - 0.99) ** 4, 0.03940398)

    def test_loss_reduction_and_empty_masks(self):
        aggregate = self.namespaces["dapo"]["aggregate_tokens"]
        self.assertEqual(aggregate([[1, 1, 999], [3] * 6], [[True, True, False], [True] * 6]), (2, 2.5))
        for values, masks in [([], []), ([[1]], [[]]), ([[1]], [[False]]), ([[1]], [[1]]), ([[float("inf")]], [[True]])]:
            with self.assertRaises(ValueError):
                aggregate(values, masks)

    def test_global_token_normalization_matches_ddp_average(self):
        local_sums = [2, 18]
        local_counts = [2, 6]
        world_size = 2
        global_count = sum(local_counts)
        scaled = [value * world_size / global_count for value in local_sums]
        self.assertEqual(sum(scaled) / world_size, 2.5)
        self.assertEqual(sum(value / count for value, count in zip(local_sums, local_counts)) / world_size, 2)

    def test_length_penalty_boundaries(self):
        penalty = self.namespaces["dapo"]["length_penalty"]
        for length, expected in [(0, 0), (80, 0), (81, -0.05), (90, -0.5), (100, -1), (10000, -1)]:
            self.assertEqual(penalty(length, 100, 20), expected)
        for args in [(-1, 100, 20), (1, 10, 20), (1, 100, 0), (True, 100, 20), (1.5, 100, 20)]:
            with self.assertRaises(ValueError):
                penalty(*args)


if __name__ == "__main__":
    unittest.main()
