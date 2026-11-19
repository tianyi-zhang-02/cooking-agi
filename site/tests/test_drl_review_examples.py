import itertools
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
NOTES = ROOT / '05-post-training/deep-rl'


class ReviewExamples(unittest.TestCase):
    def test_discounted_occupancy_matches_return_and_flow(self):
        discount = 0.5
        occupancy = [(1 - discount) / (1 - discount ** 2),
                     discount * (1 - discount) / (1 - discount ** 2)]
        self.assertAlmostEqual(sum(occupancy), 1)
        self.assertAlmostEqual(occupancy[0], 1 - discount + discount * occupancy[1])
        self.assertAlmostEqual(occupancy[1], discount * occupancy[0])
        direct = sum(discount ** step * (1 if step % 2 == 0 else 3) for step in range(100))
        self.assertAlmostEqual(direct, 10 / 3)
        self.assertAlmostEqual(direct, (occupancy[0] + 3 * occupancy[1]) / (1 - discount))
        self.assertNotAlmostEqual(direct, (0.5 + 3 * 0.5) / (1 - discount))

    def test_variance_optimal_baseline(self):
        outcomes = [(0.2, -0.8, 1), (0.8, 0.2, 3)]
        baseline = (sum(probability * score ** 2 * reward for probability, score, reward in outcomes)
                    / sum(probability * score ** 2 for probability, score, reward in outcomes))
        self.assertAlmostEqual(baseline, 1.4)
        for candidate, variance in [(0, 0.3136), (2.6, 0.2304), (baseline, 0)]:
            mean = sum(probability * score * (reward - candidate)
                       for probability, score, reward in outcomes)
            actual_variance = sum(probability * (score * (reward - candidate) - mean) ** 2
                                  for probability, score, reward in outcomes)
            self.assertAlmostEqual(mean, 0.32)
            self.assertAlmostEqual(actual_variance, variance)

    def test_multistep_action_distribution(self):
        discount, collect_probability = 0.9, 0.25
        exact = 1 + discount * 6
        raw_mean = collect_probability * exact + (1 - collect_probability)
        corrected = 1 + discount * 6 / collect_probability
        self.assertAlmostEqual(exact, 6.4)
        self.assertAlmostEqual(raw_mean, 2.35)
        self.assertAlmostEqual(corrected, 22.6)
        self.assertAlmostEqual(collect_probability * corrected + (1 - collect_probability), exact)

    def test_model_identity_changes_multistep_distribution(self):
        persistent = [-2, 2]
        resampled = [first + second for first, second in itertools.product([-1, 1], repeat=2)]
        self.assertEqual(sum(value ** 2 for value in persistent) / len(persistent), 4)
        self.assertEqual(sum(value ** 2 for value in resampled) / len(resampled), 2)
        self.assertEqual(sum(abs(value) < 1 for value in persistent), 0)
        self.assertEqual(sum(abs(value) < 1 for value in resampled) / len(resampled), 0.5)

    def test_small_prediction_error_does_not_control_derivative(self):
        for error_bound in [0.1, 0.01, 0.001]:
            action = math.pi * error_bound / 2
            model = lambda value: value + error_bound * math.sin(2 * value / error_bound)
            step = error_bound * 1e-5
            self.assertAlmostEqual(model(action), action)
            self.assertAlmostEqual((model(action + step) - model(action - step)) / (2 * step), -1)
            self.assertLess(action - step, action)
            self.assertGreater(model(action - step), model(action))

    def test_pathwise_threshold_counterexample(self):
        objective = lambda parameter: parameter + 1 + math.erf(parameter / math.sqrt(2))
        step = 1e-5
        numeric = (objective(step) - objective(-step)) / (2 * step)
        self.assertAlmostEqual(numeric, 1 + 2 / math.sqrt(2 * math.pi))
        self.assertGreater(numeric, 1.79)
        for noise in [-2, -0.1, 0.1, 2]:
            sample_return = lambda parameter: parameter + noise + 2 * (parameter + noise > 0)
            sample_gradient = (sample_return(step) - sample_return(-step)) / (2 * step)
            self.assertAlmostEqual(sample_gradient, 1)

    def test_terminal_value_state_gradient(self):
        objective = lambda parameter: -0.9 * (2 + parameter) ** 2
        step = 1e-5
        self.assertAlmostEqual((objective(step) - objective(-step)) / (2 * step), -3.6)

    def test_stitching_and_finite_sample_coverage(self):
        logged = [('direct', 'quit'), ('detour', 'collect')]
        target = ('direct', 'collect')
        weights = [math.prod(2 if observed == desired else 0
                             for observed, desired in zip(episode, target)) for episode in logged]
        self.assertEqual(weights, [0, 0])
        self.assertEqual(sum(weights), 0)
        self.assertEqual(0.5 ** len(target), 0.25)
        terminal_values = {'collect': 5, 'quit': 0}
        stitched_values = {'direct': 0 + max(terminal_values.values()),
                           'detour': -2 + max(terminal_values.values())}
        self.assertEqual(stitched_values, {'direct': 5, 'detour': 3})
        self.assertAlmostEqual(math.log(1 / 0.1), 2.302585092994046)

    def test_model_penalty_needs_a_valid_bound(self):
        for prediction, uncertainty, actual in [(0.4, 0.05, 0.4), (0.8, 0.7, 0.1)]:
            self.assertLessEqual(abs(prediction - actual), uncertainty + 1e-12)
            self.assertLessEqual(10 * (prediction - uncertainty), 10 * actual + 1e-12)
        invalid_lower_bound = 10 * (0.8 - 0.05)
        self.assertEqual(invalid_lower_bound, 7.5)
        self.assertGreater(invalid_lower_bound, 10 * 0.1)

    def test_review_anchors_have_both_languages(self):
        anchors = {
            'mdp-bellman': ['discounted-occupancy'],
            'policy-gradients': ['variance-optimal-baseline'],
            'dqn': ['multistep-off-policy'],
            'model-based': ['particle-model-identity', 'model-jacobian-check'],
            'planning-and-control': ['pathwise-gradient-conditions', 'short-horizon-actor-gradient'],
            'offline-and-ope': ['stitching-and-trajectory-support', 'support-versus-policy-distance',
                                'offline-model-pessimism'],
        }
        for stem, identifiers in anchors.items():
            for suffix in ['', '.en']:
                source = (NOTES / f'{stem}{suffix}.md').read_text()
                for identifier in identifiers:
                    self.assertIn('{#' + identifier + '}', source)


if __name__ == '__main__':
    unittest.main()
