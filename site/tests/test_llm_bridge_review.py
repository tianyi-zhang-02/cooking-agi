import math
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
NOTES = ROOT / "05-post-training/deep-rl"


class LLMBridgeReviewTests(unittest.TestCase):
    def test_reward_gradient_matches_direct_finite_difference(self):
        probability = 0.3
        rewards = [2.0, -1.0]
        probabilities = [probability, 1 - probability]
        scores = [1 - probability, -probability]
        contributions = [reward * score for reward, score in zip(rewards, scores)]
        self.assertAlmostEqual(contributions[0], 1.4)
        self.assertAlmostEqual(contributions[1], 0.3)
        expected_gradient = sum(weight * contribution
                                for weight, contribution in zip(probabilities, contributions))
        self.assertAlmostEqual(expected_gradient, 0.63)

        def expected_reward(parameter):
            current_probability = 1 / (1 + math.exp(-parameter))
            return 2 * current_probability - (1 - current_probability)

        parameter = math.log(probability / (1 - probability))
        step = 1e-5
        finite_difference = (expected_reward(parameter + step) - expected_reward(parameter - step)) / (2 * step)
        self.assertAlmostEqual(expected_gradient, finite_difference)

    def test_unfiltered_self_imitation_has_zero_initial_expected_gradient(self):
        sampling_probability = 0.3

        def fixed_data_log_likelihood(parameter):
            current_probability = 1 / (1 + math.exp(-parameter))
            return (sampling_probability * math.log(current_probability)
                    + (1 - sampling_probability) * math.log1p(-current_probability))

        parameter = math.log(sampling_probability / (1 - sampling_probability))
        step = 1e-5
        finite_difference = (fixed_data_log_likelihood(parameter + step)
                             - fixed_data_log_likelihood(parameter - step)) / (2 * step)
        self.assertAlmostEqual(finite_difference, 0)
        self.assertAlmostEqual(sampling_probability * (1 - sampling_probability)
                               + (1 - sampling_probability) * (-sampling_probability), 0)
        self.assertAlmostEqual(1 - sampling_probability, 0.7)
        changed_parameter = math.log(0.4 / 0.6)
        changed_derivative = (fixed_data_log_likelihood(changed_parameter + step)
                              - fixed_data_log_likelihood(changed_parameter - step)) / (2 * step)
        self.assertAlmostEqual(changed_derivative, sampling_probability - 0.4)
        self.assertNotAlmostEqual(changed_derivative, 0)

    def test_zero_success_probability_and_one_sided_upper_bound(self):
        attempts = 256
        zero_success_probability = (1 - 0.005) ** attempts
        self.assertAlmostEqual(zero_success_probability, 0.27714603575484426)
        upper_bound = -math.expm1(math.log(0.05) / attempts)
        self.assertAlmostEqual(upper_bound, 0.011633876163163285)
        self.assertAlmostEqual((1 - upper_bound) ** attempts, 0.05)
        self.assertLess((1 - (upper_bound + 0.001)) ** attempts, 0.05)
        two_sided_upper_bound = -math.expm1(math.log(0.025) / attempts)
        self.assertLess(upper_bound, two_sided_upper_bound)

    def test_rare_success_discovery_budget_is_minimal(self):
        probability = 0.0001
        attempts = math.ceil(math.log(0.05) / math.log1p(-probability))
        self.assertEqual(attempts, 29956)
        discovery_probability = -math.expm1(attempts * math.log1p(-probability))
        previous_probability = -math.expm1((attempts - 1) * math.log1p(-probability))
        self.assertGreaterEqual(discovery_probability, 0.95)
        self.assertLess(previous_probability, 0.95)
        self.assertEqual(attempts * 4096, 122699776)

    def test_added_arithmetic_and_anchors_match_in_both_languages(self):
        cases = (
            ("llm-bridge", "reward-gradient-path", ["0.63", "0.3", "1.4"]),
            ("why-rl-is-hard", "finite-exploration-evidence",
             ["27.7", "1.16", "29{,}956", "122,699,776"]),
        )
        for stem, anchor, numbers in cases:
            sections = []
            for suffix in ("", ".en"):
                source = (NOTES / f"{stem}{suffix}.md").read_text()
                section = source.split("{#" + anchor + "}", 1)[1].split("\n##", 1)[0]
                sections.append(section)
                for number in numbers:
                    self.assertIn(number, section)
            chinese_math = re.findall(r"\$\$(.*?)\$\$", sections[0], re.S)
            english_math = re.findall(r"\$\$(.*?)\$\$", sections[1], re.S)
            translated_math = [expression.replace("零次成功", "zero successes")
                               for expression in chinese_math]
            self.assertEqual(translated_math, english_math)


if __name__ == "__main__":
    unittest.main()
