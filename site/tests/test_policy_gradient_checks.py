import math
from pathlib import Path
import re
import runpy
import unittest


ROOT = Path(__file__).resolve().parents[2]
CHAPTERS = (
    "05-post-training/after-ppo",
    "05-post-training/rlhf/ppo-clipping",
    "05-post-training/rlhf/on-off-policy",
)
try:
    import torch
except ImportError:
    torch = None


class PolicyGradientDocumentationTests(unittest.TestCase):
    def test_inline_code_does_not_swallow_display_math(self):
        for chapter in CHAPTERS:
            for suffix in (".md", ".en.md"):
                source = (ROOT / (chapter + suffix)).read_text()
                prose = re.sub(r"(?ms)^```.*?^```\s*$", "", source)
                for span in re.finditer(r"(`+).*?\1", prose, re.S):
                    self.assertNotIn("$$", span.group(), chapter + suffix)

    def test_bilingual_code_matches(self):
        for chapter in CHAPTERS:
            chinese = (ROOT / f"{chapter}.md").read_text()
            english = (ROOT / f"{chapter}.en.md").read_text()
            self.assertEqual(re.findall(r"```python\n(.*?)```", chinese, re.S),
                             re.findall(r"```python\n(.*?)```", english, re.S))

    def test_new_sections_are_linkable_in_both_languages(self):
        sections = {
            CHAPTERS[0]: ("zero-loss-gradient", "zero-loss-assumptions"),
            CHAPTERS[1]: ("clipped-token-gradients", "other-gradient-paths"),
            CHAPTERS[2]: ("grpo-data-lifecycle", "stale-rollout-checks"),
        }
        for chapter, anchors in sections.items():
            for suffix in (".md", ".en.md"):
                source = (ROOT / (chapter + suffix)).read_text()
                for anchor in anchors:
                    self.assertIn("{#" + anchor + "}", source)
                self.assertEqual(source.count("<details"), source.count("</details>"))
                self.assertNotRegex(source, r"<details(?:\s[^>]*)?(?<!\")>")


@unittest.skipUnless(torch is not None, "PyTorch is unavailable; tensor examples are not executed")
class PolicyGradientNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions = runpy.run_path(str(ROOT / "05-post-training/code/policy_gradient_checks.py"))

    def test_zero_loss_has_nonzero_gradient_and_improves_chosen_probability(self):
        loss, gradient, probability = self.functions["zero_loss_example"]()
        self.assertAlmostEqual(loss, 0)
        self.assertAlmostEqual(gradient, -0.5)
        self.assertAlmostEqual(probability, 1 / (1 + math.exp(-0.1)))

    def test_zero_loss_gradient_matches_finite_difference(self):
        step = 1e-6
        def value(theta):
            return 1 - 2 / (1 + math.exp(-theta))
        derivative = (value(step) - value(-step)) / (2 * step)
        self.assertAlmostEqual(derivative, self.functions["zero_loss_example"]()[1], places=8)

    def test_both_advantage_signs_and_both_outside_regions(self):
        losses, gradients = self.functions["clipping_example"]()
        torch.testing.assert_close(losses, torch.tensor([-1.2, -0.6, 0.8, 1.4], dtype=torch.float64))
        torch.testing.assert_close(gradients, torch.tensor([0.0, -0.6, 0.0, 1.4], dtype=torch.float64))

    def test_clipped_objective_passes_gradcheck_away_from_boundaries(self):
        current = torch.tensor([0.7, 0.3, 0.3, 0.7], dtype=torch.float64).log().requires_grad_()
        old = torch.full_like(current, math.log(0.5))
        advantage = torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(lambda value: self.functions["clipped_terms"](value, old, advantage), (current,)))

    def test_old_log_probs_and_advantages_are_fixed(self):
        current = torch.tensor([-0.8], dtype=torch.float64, requires_grad=True)
        old = torch.tensor([-0.9], dtype=torch.float64, requires_grad=True)
        advantages = torch.ones_like(old, requires_grad=True)
        self.functions["clipped_terms"](current, old, advantages).sum().backward()
        self.assertIsNone(old.grad)
        self.assertIsNone(advantages.grad)
        self.assertNotEqual(current.grad.item(), 0)

    def test_detaching_only_denominator_preserves_gradient(self):
        current = torch.tensor(-0.8, dtype=torch.float64, requires_grad=True)
        incorrect = (current - current).exp()
        correct = (current - current.detach()).exp()
        self.assertEqual(torch.autograd.grad(incorrect, current)[0].item(), 0)
        self.assertEqual(torch.autograd.grad(correct, current)[0].item(), 1)

    def test_complete_centered_group_and_constant_rewards(self):
        rewards = torch.tensor([[1., 0., 1., 0.], [3., 3., 3., 3.]], dtype=torch.float64, requires_grad=True)
        advantages = self.functions["centered_advantages"](rewards)
        self.assertFalse(advantages.requires_grad)
        torch.testing.assert_close(advantages.sum(dim=-1), torch.zeros(2, dtype=torch.float64))
        torch.testing.assert_close(advantages[0], torch.tensor([1., -1., 1., -1.], dtype=torch.float64))
        self.assertEqual(advantages[1].abs().sum().item(), 0)

    def test_group_split_does_not_preserve_zero_mean(self):
        advantages = self.functions["centered_advantages"](torch.tensor([[1., 0., 1., 0.]], dtype=torch.float64))
        self.assertAlmostEqual(-advantages.mean().item(), 0)
        self.assertAlmostEqual(-advantages[:, :1].mean().item(), -1, places=7)

    def test_length_reduction_changes_initial_scalar_and_gradients(self):
        current = torch.full((2, 3), math.log(.5), dtype=torch.float64, requires_grad=True)
        old = current.detach().clone()
        valid = torch.tensor([[True, False, False], [True, True, True]])
        advantages = torch.tensor([1., -1.], dtype=torch.float64)
        sequence = self.functions["masked_policy_loss"](current, old, advantages, valid, "sequence")
        token = self.functions["masked_policy_loss"](current, old, advantages, valid, "token")
        self.assertAlmostEqual(sequence.item(), 0)
        self.assertAlmostEqual(token.item(), .5)
        sequence_grad = torch.autograd.grad(sequence, current, retain_graph=True)[0]
        token_grad = torch.autograd.grad(token, current)[0]
        torch.testing.assert_close(sequence_grad, torch.tensor([[-.5, 0., 0.], [1/6, 1/6, 1/6]], dtype=torch.float64))
        torch.testing.assert_close(token_grad, torch.tensor([[-.25, 0., 0.], [.25, .25, .25]], dtype=torch.float64))

    def test_padding_nan_is_excluded_before_arithmetic(self):
        current = torch.tensor([[math.log(.5), float("nan")]], dtype=torch.float64, requires_grad=True)
        old = current.detach().clone()
        valid = torch.tensor([[True, False]])
        loss = self.functions["masked_policy_loss"](current, old, torch.ones(1, dtype=torch.float64), valid)
        loss.backward()
        self.assertAlmostEqual(loss.item(), -1)
        torch.testing.assert_close(current.grad, torch.tensor([[-1., 0.]], dtype=torch.float64))

    def test_regularizer_keeps_gradient_when_policy_branch_is_flat(self):
        policy, total = self.functions["regularizer_example"]()
        self.assertAlmostEqual(policy, 0)
        self.assertAlmostEqual(total, .11 * .7 * .3 * math.log(7/3))

    def test_other_samples_can_move_shared_parameters(self):
        first, batch, probability = self.functions["shared_parameter_example"]()
        self.assertAlmostEqual(first, 0)
        self.assertAlmostEqual(batch, .21)
        self.assertLess(probability, .7)

    def test_reference_equality_is_separate_from_old_equality(self):
        policy = torch.tensor([.7, .3], dtype=torch.float64)
        reference = torch.tensor([.5, .5], dtype=torch.float64)
        self.assertAlmostEqual((policy / policy).mean().item(), 1)
        divergence = (policy * (policy.log() - reference.log())).sum()
        self.assertGreater(divergence.item(), 0)
        sampled_k3 = reference / policy - (reference / policy).log() - 1
        self.assertAlmostEqual((policy * sampled_k3).sum().item(), divergence.item())

    def test_stale_denominator_is_not_refreshed_during_reuse(self):
        old = torch.tensor([.4, .6], dtype=torch.float64)
        current = torch.tensor([.6, .4], dtype=torch.float64)
        torch.testing.assert_close(current / old, torch.tensor([1.5, 2/3], dtype=torch.float64))
        self.assertFalse(torch.equal(current / old, current / current))

    def test_invalid_contracts_are_rejected(self):
        with self.assertRaises(ValueError):
            self.functions["centered_advantages"](torch.ones(1, 1))
        with self.assertRaises(ValueError):
            self.functions["clipped_terms"](torch.zeros(2), torch.zeros(2), torch.ones(2), epsilon=0)
        for valid, reduction in ((torch.zeros(1, 2, dtype=torch.bool), "sequence"),
                                 (torch.ones(1, 2, dtype=torch.bool), "unknown")):
            with self.assertRaises(ValueError):
                self.functions["masked_policy_loss"](torch.zeros(1, 2), torch.zeros(1, 2), torch.ones(1), valid, reduction)

    def test_article_snippets_run_against_teaching_functions(self):
        for chapter in CHAPTERS:
            namespace = dict(self.functions)
            source = (ROOT / (chapter + ".md")).read_text()
            for snippet in re.findall(r"```python\n(.*?)```", source, re.S):
                exec(compile(snippet, chapter, "exec"), namespace)


if __name__ == "__main__":
    unittest.main()
