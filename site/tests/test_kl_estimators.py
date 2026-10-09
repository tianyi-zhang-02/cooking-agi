import math
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '05-post-training/rlhf/kl-estimators.md'


class KLEstimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = {}
        cls.code = re.findall(r'```python\n(.*?)```', CHAPTER.read_text(), re.S)
        for snippet in cls.code:
            exec(compile(snippet, str(CHAPTER), 'exec'), cls.namespace)
        cls.terms = staticmethod(cls.namespace['kl_sample_terms'])

    def test_bilingual_code_matches(self):
        english = CHAPTER.with_name('kl-estimators.en.md').read_text()
        self.assertEqual(self.code, re.findall(r'```python\n(.*?)```', english, re.S))

    def test_exact_expectations_for_positive_distributions(self):
        for current in ([.8, .2], [.01, .99], [.2, .3, .5]):
            reference = [1 / len(current)] * len(current)
            rows = [self.terms(math.log(probability), math.log(anchor))
                    for probability, anchor in zip(current, reference)]
            exact = sum(probability * math.log(probability / anchor)
                        for probability, anchor in zip(current, reference))
            for column in (0, 2):
                self.assertAlmostEqual(sum(probability * row[column]
                                           for probability, row in zip(current, rows)), exact)
            self.assertTrue(all(row[2] >= -1e-15 for row in rows))

    def test_table_and_biased_quadratic(self):
        rows = self.namespace['terms']
        self.assertEqual([[round(value, 4) for value in row] for row in rows],
                         [[.4700, .1105, .0950], [-.9163, .4198, .5837]])
        means = self.namespace['means']
        self.assertEqual([round(value, 4) for value in means], [.1927, .1723, .1927])
        self.assertNotAlmostEqual(means[0], means[1])

    def test_old_policy_needs_action_correction(self):
        rows = self.namespace['terms']
        current, old = [.8, .2], [.5, .5]
        uncorrected = sum(probability * row[2] for probability, row in zip(old, rows))
        corrected = sum(behavior * policy / behavior * row[2]
                        for behavior, policy, row in zip(old, current, rows))
        self.assertAlmostEqual(uncorrected, .5625 + .5 * math.log(.64))
        self.assertAlmostEqual(corrected, self.namespace['means'][0])

    def test_missing_support_breaks_control_variate(self):
        row = self.terms(0, math.log(.5))
        self.assertAlmostEqual(row[2] - row[0], -.5)

    def test_equal_and_nearby_distributions(self):
        self.assertEqual(self.terms(math.log(.5), math.log(.5)), (0, 0, 0))
        for delta in (-1e-7, 1e-7):
            row = self.terms(-1, -1 + delta)
            self.assertAlmostEqual(row[2] / row[1], 1, places=6)

    def test_gradient_requires_sampling_derivative(self):
        theta = math.log(4)
        step = 1e-5
        frozen = [.8, .2]

        def probability(parameter):
            positive = 1 / (1 + math.exp(-parameter))
            return [positive, 1 - positive]

        def objectives(parameter):
            current = probability(parameter)
            rows = [self.terms(math.log(value), math.log(.5)) for value in current]
            exact = sum(value * math.log(value / .5) for value in current)
            fixed = sum(weight * row[2] for weight, row in zip(frozen, rows))
            corrected = sum(behavior * value / behavior * row[2]
                            for behavior, value, row in zip(frozen, current, rows))
            return exact, fixed, corrected

        derivatives = [(upper - lower) / (2 * step)
                       for upper, lower in zip(objectives(theta + step), objectives(theta - step))]
        self.assertAlmostEqual(derivatives[0], .16 * math.log(4))
        self.assertAlmostEqual(derivatives[1], .3)
        self.assertAlmostEqual(derivatives[2], derivatives[0])

    def test_invalid_values_and_overflow_are_not_silently_clipped(self):
        for value in (math.nan, math.inf, -math.inf, .1):
            with self.assertRaises(ValueError):
                self.terms(value, -1)
            with self.assertRaises(ValueError):
                self.terms(-1, value)
        with self.assertRaises(OverflowError):
            self.terms(-1000, -1)


if __name__ == '__main__':
    unittest.main()
