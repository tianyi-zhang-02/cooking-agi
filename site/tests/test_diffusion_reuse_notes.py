import itertools
import math
from pathlib import Path
import re
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "11-frontier/diffusion-lm"
NOTES = ("mdlm-walkthrough", "context-reuse")


def reverse_distribution(clean_probabilities, current_time, next_time):
    reveal_probability = (current_time - next_time) / current_time
    return [1 - reveal_probability] + [
        reveal_probability * probability for probability in clean_probabilities
    ]


def scalar_attention(query, keys, values):
    weights = [math.exp(query * key) for key in keys]
    normalizer = sum(weights)
    output = sum(weight * value for weight, value in zip(weights, values)) / normalizer
    return output, normalizer


def merge_attention(outputs, log_normalizers):
    maximum = max(log_normalizers)
    weights = [math.exp(normalizer - maximum) for normalizer in log_normalizers]
    return sum(weight * output for weight, output in zip(weights, outputs)) / sum(weights)


class DiffusionReuseNotesTests(unittest.TestCase):
    def test_weighted_loss_and_sequence_average(self):
        loss = -sum(math.log(probability) for probability in (0.8, 0.7, 0.9)) / 0.5
        self.assertAlmostEqual(loss, 1.370358021821537)
        self.assertEqual(round(loss, 4), 1.3704)
        self.assertEqual(round(loss / 6, 4), 0.2284)
        self.assertNotEqual(loss / 6, loss / 3)

    def test_reverse_probability_table(self):
        actual = reverse_distribution([0.7, 0.2, 0.1], 0.5, 0.25)
        for observed, expected in zip(actual, [0.5, 0.35, 0.1, 0.05]):
            self.assertAlmostEqual(observed, expected)
        self.assertAlmostEqual(sum(actual), 1)
        self.assertEqual(reverse_distribution([0.7, 0.2, 0.1], 0.5, 0)[0], 0)
        self.assertEqual(reverse_distribution([0.7, 0.2, 0.1], 0.5, 0.5)[0], 1)

    def test_reverse_kl_reduces_to_weighted_cross_entropy(self):
        model = reverse_distribution([0.7, 0.2, 0.1], 0.5, 0.25)
        truth = [0.5, 0.5, 0, 0]
        divergence = sum(
            probability * math.log(probability / predicted)
            for probability, predicted in zip(truth, model)
            if probability
        )
        self.assertAlmostEqual(divergence, 0.5 * -math.log(0.7))

    def test_correct_marginals_do_not_preserve_joint_constraints(self):
        combinations = list(itertools.product(("cats", "dogs"), repeat=3))
        valid = [
            sequence for sequence in combinations
            if sequence[0] != sequence[1] and sequence[1] == sequence[2]
        ]
        self.assertEqual(len(valid) / len(combinations), 0.25)
        self.assertEqual(
            [sequence for sequence in valid if sequence[0] == "cats"],
            [("cats", "dogs", "dogs")],
        )

    def test_fixed_kv_does_not_fix_attention_output(self):
        old_output, old_normalizer = scalar_attention(0, [0, 1], [0, 10])
        new_output, new_normalizer = scalar_attention(math.log(3), [0, 1], [0, 10])
        self.assertAlmostEqual(old_output, 5)
        self.assertAlmostEqual(old_normalizer, 2)
        self.assertAlmostEqual(new_output, 7.5)
        self.assertAlmostEqual(new_normalizer, 4)

    def test_partition_merge_matches_full_attention(self):
        full, _ = scalar_attention(math.log(3), [0, 1, 0], [0, 10, 2])
        combined = merge_attention([7.5, 2], [math.log(4), math.log(1)])
        stale = merge_attention([5, 2], [math.log(2), math.log(1)])
        self.assertAlmostEqual(full, 6.4)
        self.assertAlmostEqual(combined, full)
        self.assertAlmostEqual(stale, 4)
        self.assertEqual((7.5 + 2) / 2, 4.75)
        self.assertNotAlmostEqual(stale, full)

    def test_log_merge_remains_stable_for_large_scores(self):
        shifted = merge_attention([7.5, 2], [1000 + math.log(4), 1000])
        self.assertAlmostEqual(shifted, 6.4)

    def test_end_to_end_cost_includes_unaccelerated_work(self):
        accelerated = 60 / 3 + 40
        with_overhead = accelerated + 8
        self.assertEqual(accelerated, 60)
        self.assertEqual(with_overhead, 68)
        self.assertEqual(round(100 / accelerated, 2), 1.67)
        self.assertEqual(round(100 / with_overhead, 2), 1.47)

    def test_bilingual_navigation_and_examples(self):
        navigation = tomllib.loads((ROOT / "site/nav.toml").read_text(encoding="utf-8"))
        section = next(
            section for section in navigation["section"]
            if section["dir"] == "11-frontier/diffusion-lm"
        )
        order = section["order"]
        self.assertEqual(order.index("mdlm-walkthrough.md"), order.index("masked-diffusion.md") + 1)
        self.assertEqual(order.index("context-reuse.md"), order.index("review.md") - 1)
        expected_fragments = {
            "mdlm-walkthrough": ("1.3704", "0.2284", "25\\%", "2406.07524v2", "ddpm_cache"),
            "context-reuse": ("6.4", "4.75", "1.47", "2602.05305v3", "2602.14209v2"),
        }
        for name in NOTES:
            for suffix in (".md", ".en.md"):
                with self.subTest(name=name, suffix=suffix):
                    source = (DIRECTORY / (name + suffix)).read_text(encoding="utf-8")
                    for fragment in expected_fragments[name]:
                        self.assertIn(fragment, source)
                    overview = (DIRECTORY / ("README" + suffix)).read_text(encoding="utf-8")
                    self.assertIn(name + suffix, overview)

    def test_english_and_equations_are_complete(self):
        for name in NOTES:
            with self.subTest(name=name):
                chinese = (DIRECTORY / (name + ".md")).read_text(encoding="utf-8")
                english = (DIRECTORY / (name + ".en.md")).read_text(encoding="utf-8")
                body = english.replace(f"[中文]({name}.md)", "")
                self.assertIsNone(re.search(r"[\u3400-\u9fff]", body))
                self.assertEqual(
                    re.findall(r"\$\$(.*?)\$\$", chinese, re.DOTALL),
                    re.findall(r"\$\$(.*?)\$\$", english, re.DOTALL),
                )


if __name__ == "__main__":
    unittest.main()
