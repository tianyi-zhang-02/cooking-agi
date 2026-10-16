import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ReviewedClaimsTests(unittest.TestCase):
    def test_unsampled_probability_can_rise_after_policy_gradient(self):
        probabilities = [0.2, 0.7, 0.1]
        logits = [
            math.log(probability) + 0.1 * ((index == 0) - probability)
            for index, probability in enumerate(probabilities)
        ]
        normalizer = sum(math.exp(logit) for logit in logits)
        updated = [math.exp(logit) / normalizer for logit in logits]
        self.assertGreater(updated[0], probabilities[0])
        self.assertGreater(updated[2], probabilities[2])
        self.assertLess(sum(updated[1:]), sum(probabilities[1:]))
        self.assertAlmostEqual(updated[2], 0.1022421676)
        for suffix in (".md", ".en.md"):
            source = (ROOT / ("05-post-training/rlhf/value-and-advantage" + suffix)).read_text()
            self.assertIn("[0.22374, 0.67402, 0.10224]", source)
