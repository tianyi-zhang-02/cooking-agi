import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ReviewedClaimsTests(unittest.TestCase):
    def test_gqa_reduces_kv_projection_weights_not_just_cache(self):
        model_width = 4096
        head_width = 128
        query_heads = 32
        projection_parameters = lambda kv_heads: 2 * model_width * head_width * (query_heads + kv_heads)
        self.assertEqual(projection_parameters(32), 67108864)
        self.assertEqual(projection_parameters(8), 41943040)
        self.assertGreater(projection_parameters(8) / projection_parameters(32), 8 / 32)

    def test_attention_variance_requires_zero_mean(self):
        components = (0, 2)
        products = [query * key for query in components for key in components]
        mean = sum(products) / len(products)
        variance = sum((product - mean) ** 2 for product in products) / len(products)
        self.assertEqual(variance, 3)
        for suffix, phrase in ((".md", "均值为 0"), (".en.md", "zero-mean")):
            source = (ROOT / ("00-foundations/transformer" + suffix)).read_text()
            self.assertIn(phrase, source)

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
