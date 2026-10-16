from fractions import Fraction
from itertools import product
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


def note_text(chapter, suffix):
    return (ROOT / (chapter + suffix)).read_text(encoding="utf-8")


def adamw_timescale(batch, learning_rate, weight_decay, tokens):
    return batch / (learning_rate * weight_decay * tokens)


class ScalingDiffusionReviewTests(unittest.TestCase):
    def test_batch_decay_scaling_with_fixed_learning_rate(self):
        original = adamw_timescale(1e6, 3e-4, 0.1, 1e11)
        for multiplier in (0.5, 2, 4):
            adjusted = adamw_timescale(1e6 * multiplier, 3e-4, 0.1 * multiplier, 1e11)
            self.assertAlmostEqual(adjusted, original)

    def test_batch_and_learning_rate_changes_require_the_ratio_rule(self):
        original = adamw_timescale(1e6, 3e-4, 0.1, 1e11)
        for batch_multiplier, lr_multiplier in ((4, 2), (4, 4), (2, 0.5)):
            decay_multiplier = batch_multiplier / lr_multiplier
            adjusted = adamw_timescale(
                1e6 * batch_multiplier, 3e-4 * lr_multiplier, 0.1 * decay_multiplier, 1e11,
            )
            self.assertAlmostEqual(adjusted, original)
        incorrect = adamw_timescale(4e6, 6e-4, 0.4, 1e11)
        self.assertAlmostEqual(incorrect, original / 2)
        self.assertNotAlmostEqual(incorrect, original)

    def test_timescale_preservation_does_not_silently_change_the_token_budget(self):
        original = adamw_timescale(1e6, 3e-4, 0.1, 1e11)
        doubled_tokens = adamw_timescale(4e6, 6e-4, 0.2, 2e11)
        self.assertAlmostEqual(doubled_tokens, original / 2)

    def test_bilingual_decay_guidance_includes_conditions_and_general_ratio(self):
        for suffix in (".md", ".en.md"):
            source = note_text("00-foundations/scaling/hyperparameters", suffix)
            self.assertIn(r"\frac{\lambda_{\rm new}}{\lambda_{\rm old}}", source)
            self.assertIn(r"\frac{B_{\rm new}/B_{\rm old}}{\eta_{\rm new}/\eta_{\rm old}}", source)
            self.assertIn("2505.13738v2", source)
            if suffix == ".en.md":
                for qualifier in ("peak learning rate stays fixed", "schedule shape", "critical batch", "fixed $D$", "not ×4"):
                    self.assertIn(qualifier, source)
                self.assertIn("does not prove", source)
            else:
                for qualifier in ("峰值学习率不变", "schedule 形状", "临界 batch", "固定 $D$", "不是 ×4", "不保证"):
                    self.assertIn(qualifier, source)

    def test_lifetime_cost_example_requires_more_training_compute(self):
        throughput = 64 * 989.4e12 * 0.4
        original_compute = 1e22
        revised_compute = 6 * 4.6e9 * 450e9
        original_days = original_compute / throughput / 86400
        revised_days = revised_compute / throughput / 86400
        self.assertAlmostEqual(revised_compute / original_compute, 1.242)
        self.assertEqual(round(original_days, 1), 4.6)
        self.assertEqual(round(revised_days, 1), 5.7)
        self.assertGreater(revised_compute, throughput * 4.6 * 86400)
        for suffix in (".md", ".en.md"):
            source = note_text("00-foundations/scaling/README", suffix)
            self.assertIn("24%", source)
            self.assertIn("5.7", source)
            self.assertIn("target loss fixed" if suffix == ".en.md" else "目标 loss 不变", source)
            self.assertIn("training plus inference cost" if suffix == ".en.md" else "训练加推理的总成本", source)

    def test_parallel_sampling_penalty_depends_on_the_target_distribution(self):
        dependent = {("full", "house"): Fraction(1, 2), ("high", "card"): Fraction(1, 2)}
        independent = dict.fromkeys(product(("full", "high"), ("house", "card")), Fraction(1, 4))
        for target, expected_valid in ((dependent, Fraction(1, 2)), (independent, Fraction(1))):
            marginals = [{}, {}]
            for sequence, probability in target.items():
                for position, token in enumerate(sequence):
                    marginals[position][token] = marginals[position].get(token, 0) + probability
            valid_probability = sum(
                marginals[0][first] * marginals[1][second]
                for first, second in target
            )
            self.assertEqual(valid_probability, expected_valid)
        for length in (2, 8, 64):
            self.assertEqual(2 ** length * Fraction(1, 2) ** length, 1)

    def test_diffusion_theorem_scope_is_retained_in_both_languages(self):
        for suffix in (".md", ".en.md"):
            source = note_text("11-frontier/diffusion-lm/decoding-and-speed", suffix)
            self.assertIn("2502.09622v2#S4", source)
            self.assertIn("n-gram", source)
            self.assertIn("4.2", source)
            self.assertIn("4.4", source)
            self.assertIn(r"\epsilon_{\rm learning}<1/128", source)
            for chapter in ("README", "review", "decoding-and-speed"):
                text = note_text("11-frontier/diffusion-lm/" + chapter, suffix)
                self.assertIn("HMM", text)
                if suffix == ".en.md":
                    self.assertIn("worst-case", text)
                    self.assertIn("learning-error", text)
                else:
                    self.assertIn("最坏情况", text)
                    self.assertIn("学习误差", text)
            if suffix == ".en.md":
                for qualifier in ("16-token vocabulary", "factorized", "support", "does not rule out speedups"):
                    self.assertIn(qualifier, source)
                self.assertNotIn("the efficiency advantage over AR models is gone", source)
            else:
                for qualifier in ("词表大小为 16", "因子化", "支持集", "不是说所有推理任务"):
                    self.assertIn(qualifier, source)
                self.assertNotIn("相对 AR 的效率优势就没了", source)


if __name__ == "__main__":
    unittest.main()
