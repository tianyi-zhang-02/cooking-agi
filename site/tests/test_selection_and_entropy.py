import itertools
import math
from pathlib import Path
import re
import runpy
import unittest


ROOT = Path(__file__).resolve().parents[2]
FUNCTIONS = runpy.run_path(str(ROOT / "05-post-training/code/selection_and_entropy.py"))


class SelectionAndEntropyTests(unittest.TestCase):
    def assertVectorAlmostEqual(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for observed, wanted in zip(actual, expected):
            self.assertAlmostEqual(observed, wanted)

    def test_filter_example(self):
        rate, retained = FUNCTIONS["filtered_distribution"]([0.5, 0.3, 0.2], [1, 0.5, 0])
        self.assertAlmostEqual(rate, 0.65)
        self.assertVectorAlmostEqual(retained, [10 / 13, 3 / 13, 0])

    def test_zero_acceptance_is_not_a_distribution(self):
        with self.assertRaises(ValueError):
            FUNCTIONS["filtered_distribution"]([0.5, 0.5], [0, 0])

    def test_invalid_acceptance(self):
        for acceptance in ([1], [-0.1, 1], [1.1, 0], [math.nan, 1]):
            with self.subTest(acceptance=acceptance), self.assertRaises(ValueError):
                FUNCTIONS["filtered_distribution"]([0.5, 0.5], acceptance)

    def test_invalid_probability_vectors(self):
        for values in ([], [0.4, 0.4], [-1, 2], [math.nan, 0], [math.inf, 0]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                FUNCTIONS["probability_vector"](values)

    def test_classical_envelope_recovers_target(self):
        target, proposal = [0.6, 0.3, 0.1], [1 / 3] * 3
        acceptance = FUNCTIONS["rejection_acceptance"](target, proposal, 1.8)
        self.assertVectorAlmostEqual(acceptance, [1, 0.5, 1 / 6])
        rate, retained = FUNCTIONS["filtered_distribution"](proposal, acceptance)
        self.assertAlmostEqual(rate, 1 / 1.8)
        self.assertVectorAlmostEqual(retained, target)

    def test_larger_envelope_lowers_acceptance_not_target(self):
        target, proposal = [0.6, 0.3, 0.1], [1 / 3] * 3
        for envelope in (2, 3, 10):
            rate, retained = FUNCTIONS["filtered_distribution"](proposal, FUNCTIONS["rejection_acceptance"](target, proposal, envelope))
            self.assertAlmostEqual(rate, 1 / envelope)
            self.assertVectorAlmostEqual(retained, target)

    def test_missing_support_and_small_envelope_rejected(self):
        for target, proposal, envelope in (([0.5, 0.5], [1, 0], 2), ([0.8, 0.2], [0.5, 0.5], 1)):
            with self.assertRaises(ValueError):
                FUNCTIONS["rejection_acceptance"](target, proposal, envelope)

    def test_zero_mass_category_is_safe(self):
        self.assertEqual(FUNCTIONS["rejection_acceptance"]([1, 0], [1, 0], 1), (1.0, 0.0))

    def test_best_of_two(self):
        result = FUNCTIONS["best_of_n_distribution"]([0.5, 0.3, 0.2], [0, 1, 2], 2)
        self.assertVectorAlmostEqual(result, [0.25, 0.39, 0.36])

    def test_best_of_one_is_proposal(self):
        self.assertVectorAlmostEqual(FUNCTIONS["best_of_n_distribution"]([0.5, 0.3, 0.2], [2, 0, 1], 1), [0.5, 0.3, 0.2])

    def test_equal_scores_keep_first_draw(self):
        self.assertVectorAlmostEqual(FUNCTIONS["best_of_n_distribution"]([0.5, 0.3, 0.2], [1, 1, 1], 3), [0.5, 0.3, 0.2])

    def test_bad_scorer_is_amplified(self):
        for count in range(1, 6):
            result = FUNCTIONS["best_of_n_distribution"]([0.5, 0.3, 0.2], [0, 1, 2], count)
            self.assertAlmostEqual(result[2], 1 - 0.8 ** count)
            self.assertAlmostEqual(sum(result), 1)

    def test_enumeration_rejects_invalid_counts_and_unbounded_work(self):
        for count in (0, -1, 1.5, True, 20):
            with self.assertRaises(ValueError):
                FUNCTIONS["best_of_n_distribution"]([0.5, 0.5], [0, 1], count)

    def test_entropy_values_and_zero_support(self):
        self.assertAlmostEqual(FUNCTIONS["entropy"]([0.5, 0.5]), math.log(2))
        self.assertAlmostEqual(FUNCTIONS["entropy"]([0.9, 0.1]), 0.3250829733914482)
        self.assertEqual(FUNCTIONS["entropy"]([1, 0]), 0)

    def test_cross_entropy_requires_matching_sampler_for_entropy(self):
        self.assertAlmostEqual(FUNCTIONS["cross_entropy"]([0.9, 0.1], [0.9, 0.1]), FUNCTIONS["entropy"]([0.9, 0.1]))
        self.assertAlmostEqual(FUNCTIONS["cross_entropy"]([1, 0], [0.9, 0.1]), -math.log(0.9))
        self.assertEqual(FUNCTIONS["cross_entropy"]([0.5, 0.5], [1, 0]), math.inf)

    def test_weighting_and_masked_nan(self):
        actual = FUNCTIONS["entropy_means"]([[1, 1, math.nan], [0.1] * 8], [[True, True, False], [True] * 8])
        self.assertVectorAlmostEqual(actual, [0.55, 0.28])

    def test_invalid_masks_and_empty_responses_rejected(self):
        cases = (([], []), ([[1]], [[False]]), ([[math.nan]], [[True]]), ([[1]], [[1]]), ([[1]], [[True, True]]))
        for rows, masks in cases:
            with self.assertRaises(ValueError):
                FUNCTIONS["entropy_means"](rows, masks)

    def test_group_probability_matches_enumeration(self):
        for success in (0, 0.1, 0.5, 0.9, 1):
            for count in range(1, 5):
                expected = sum(math.prod(success if outcome else 1 - success for outcome in outcomes)
                               for outcomes in itertools.product((False, True), repeat=count)
                               if len(set(outcomes)) == 1)
                self.assertAlmostEqual(FUNCTIONS["homogeneous_group_probability"](success, count), expected)

    def test_group_numbers_in_prose(self):
        self.assertAlmostEqual(FUNCTIONS["homogeneous_group_probability"](0.1, 4), 0.6562)
        self.assertAlmostEqual(FUNCTIONS["homogeneous_group_probability"](0.5, 4), 0.125)

    def test_binary_entropy_derivative_matches_finite_difference(self):
        step = 1e-6
        for probability in (0.2, 0.8):
            updated, derivative = FUNCTIONS["binary_reward_update"](probability, step)
            observed = (FUNCTIONS["entropy"]([updated, 1 - updated]) - FUNCTIONS["entropy"]([probability, 1 - probability])) / step
            self.assertAlmostEqual(observed, derivative, places=7)
            self.assertGreater(updated, probability)
            self.assertEqual(derivative > 0, probability < 0.5)

    def test_sort_check_rejects_missing_or_reordered_elements(self):
        for candidate in ([1, 3], [], [3, 1, 1], [1, 1, 1, 3]):
            self.assertFalse(FUNCTIONS["verifies_sort"]([3, 1, 1], candidate))

    def test_sort_check_accepts_edge_cases(self):
        for original in ([], [1], [3, 1, 1], [-4, 2, 0, -4]):
            self.assertTrue(FUNCTIONS["verifies_sort"](original, sorted(original)))

    def test_entropy_does_not_identify_quality(self):
        original = [3, 1, 1]
        outputs = ([1, 1, 3], [1, 3], [])
        quality = [FUNCTIONS["verifies_sort"](original, output) for output in outputs]
        proxy = [all(left <= right for left, right in zip(output, output[1:])) for output in outputs]
        correct_policy, loophole_policy = [1, 0, 0], [0, 0.5, 0.5]
        self.assertEqual(sum(mass * value for mass, value in zip(correct_policy, quality)), 1)
        self.assertEqual(sum(mass * value for mass, value in zip(loophole_policy, quality)), 0)
        self.assertEqual(FUNCTIONS["entropy"](correct_policy), 0)
        self.assertAlmostEqual(FUNCTIONS["entropy"](loophole_policy), math.log(2))
        for policy in (correct_policy, loophole_policy):
            self.assertEqual(sum(mass * reward for mass, reward in zip(policy, proxy)), 1)
        for probabilities in ([0.9, 0.1], [0.5, 0.5]):
            self.assertAlmostEqual(FUNCTIONS["entropy"](probabilities), FUNCTIONS["entropy"](list(reversed(probabilities))))


class SelectionDocumentationTests(unittest.TestCase):
    def test_bilingual_sections_and_code(self):
        sections = {
            'rejection-sampling': ('four-uses', 'filtered-sft', 'filter-distribution', 'classical-rejection', 'best-of-n', 'prompt-selection', 'data-records', 'worked-code'),
            'alignment-tax': ('entropy-measurement', 'entropy-versus-hacking', 'entropy-research', 'entropy-diagnostics'),
            'verifiable-rewards': ('reward-entropy-example',),
        }
        for chapter, anchors in sections.items():
            versions = [(ROOT / f'05-post-training/{chapter}{suffix}').read_text() for suffix in ('.md', '.en.md')]
            self.assertEqual(re.findall(r'```python\n(.*?)```', versions[0], re.S), re.findall(r'```python\n(.*?)```', versions[1], re.S))
            for source in versions:
                for anchor in anchors:
                    self.assertIn('{#' + anchor + '}', source)
                self.assertEqual(source.count('<details'), source.count('</details>'))
                prose = re.sub(r'(?ms)^```.*?^```\s*$', '', source)
                for span in re.finditer(r'(`+).*?\1', prose, re.S):
                    self.assertNotIn('$$', span.group())

    def test_sort_snippets_execute_and_match_teaching_function(self):
        for suffix in ('.md', '.en.md'):
            source = (ROOT / f'05-post-training/verifiable-rewards{suffix}').read_text()
            snippet, = re.findall(r'```python\n(.*?)```', source, re.S)
            namespace = {}
            exec(snippet, namespace)
            for candidate in ([1, 1, 3], [1, 3], []):
                self.assertEqual(namespace['verifies_sort']([3, 1, 1], candidate), FUNCTIONS['verifies_sort']([3, 1, 1], candidate))


if __name__ == '__main__':
    unittest.main()
