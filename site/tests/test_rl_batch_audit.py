import copy
import importlib.util
import itertools
import math
from pathlib import Path
import re
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("rl_batch_checks", ROOT / "practice/post-training/code/rl_batch_checks.py")
checks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checks)


class RlooTests(unittest.TestCase):
    def test_worked_group_and_mean_relationship(self):
        rewards = [1, 0, 0.5]
        advantages = checks.rloo_advantages(rewards)
        self.assertEqual(advantages, [0.75, -0.75, 0])
        for reward, advantage in zip(rewards, advantages):
            self.assertAlmostEqual(reward - 0.5, (2 / 3) * advantage)
        self.assertEqual(checks.rloo_advantages([1, 1, 1]), [0, 0, 0])
        for invalid in ([], [1], [math.nan, 0], [0, math.inf]):
            with self.assertRaises(ValueError):
                checks.rloo_advantages(invalid)

    def test_exact_expectation_for_independent_bernoulli_samples(self):
        probability = 0.3
        exact_gradient = probability * (1 - probability)
        for group_size in (2, 3, 4):
            expected_leave_one_out = expected_group_mean = 0.0
            for actions in itertools.product((0, 1), repeat=group_size):
                group_probability = math.prod(probability if action else 1 - probability for action in actions)
                advantages = checks.rloo_advantages(actions)
                scores = [action - probability for action in actions]
                expected_leave_one_out += group_probability * sum(
                    advantage * score for advantage, score in zip(advantages, scores)) / group_size
                mean_reward = sum(actions) / group_size
                expected_group_mean += group_probability * sum(
                    (action - mean_reward) * score for action, score in zip(actions, scores)) / group_size
            self.assertAlmostEqual(expected_leave_one_out, exact_gradient)
            self.assertAlmostEqual(expected_group_mean, (group_size - 1) / group_size * exact_gradient)

    def test_correlated_copies_do_not_qualify(self):
        self.assertEqual(checks.rloo_advantages([0, 0]), [0, 0])
        self.assertEqual(checks.rloo_advantages([1, 1]), [0, 0])
        self.assertGreater(0.3 * 0.7, 0)

    def test_group_order_and_bad_records(self):
        records = checks.demo_records()
        shuffled = [records[index] for index in (0, 3, 1, 4, 2, 5)]
        self.assertEqual(checks.grouped_advantages(records, 3), checks.grouped_advantages(shuffled, 3))
        for broken in (records[:-1], records + [records[0]]):
            with self.assertRaises(ValueError):
                checks.grouped_advantages(broken, 3)
        for field, value in (("score_status", "failed"), ("policy_version", "other"),
                             ("generation_round", "other"), ("reward_version", "other"),
                             ("reward", math.nan)):
            broken = copy.deepcopy(records)
            broken[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                checks.grouped_advantages(broken, 3)


class BatchTests(unittest.TestCase):
    def test_alignment_eos_and_masked_nonfinite_padding(self):
        values, total = checks.sequence_log_prob(checks.demo_logits(), [0, 1, 2, 3, 3],
                                                [False, True, True, True, False])
        for actual, expected in zip(values, [math.log(0.5), math.log(0.25), math.log(0.8), 0]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(total, math.log(0.1))
        broken = checks.demo_logits()
        broken[1][2] = math.nan
        with self.assertRaises(ValueError):
            checks.sequence_log_prob(broken, [0, 1, 2, 3, 3], [False, True, True, True, False])

    def test_reductions_and_padding_invariance(self):
        expected = {"sequence_sum": 5, "sequence_mean": 2, "token_mean": 5 / 3, "fixed_length": 1.25}
        plain = checks.loss_reductions([[2, 4], [1, 1, 1, 1]], [[True] * 2, [True] * 4], 4)
        padded = checks.loss_reductions([[2, 4, math.nan, math.inf], [1, 1, 1, 1]],
                                       [[True, True, False, False], [True] * 4], 4)
        self.assertEqual(plain, expected)
        self.assertEqual(plain, padded)
        for losses, masks, length in (([[1]], [[False]], 4), ([[math.nan]], [[True]], 4),
                                     ([[1, 2]], [[True, True]], 1), ([[1]], [[True]], 0)):
            with self.assertRaises(ValueError):
                checks.loss_reductions(losses, masks, length)

    def test_coefficients_by_finite_difference(self):
        losses = [[2.0, 4.0], [1.0] * 4]
        masks = [[True] * 2, [True] * 4]
        expected = {"sequence_sum": (0.5, 0.5), "sequence_mean": (0.25, 0.125),
                    "token_mean": (1 / 6, 1 / 6), "fixed_length": (0.125, 0.125)}
        step = 1e-5
        for row_index in (0, 1):
            above, below = copy.deepcopy(losses), copy.deepcopy(losses)
            above[row_index][0] += step
            below[row_index][0] -= step
            upper = checks.loss_reductions(above, masks, 4)
            lower = checks.loss_reductions(below, masks, 4)
            for name, coefficients in expected.items():
                self.assertAlmostEqual((upper[name] - lower[name]) / (2 * step), coefficients[row_index])


class ReadingRouteTests(unittest.TestCase):
    def test_review_route_links_to_specific_worked_examples(self):
        for suffix in (".md", ".en.md"):
            page = (ROOT / f"05-post-training/deep-rl/README{suffix}").read_text()
            self.assertIn("{#reward-review}", page)
            for target, anchor in (("../rloo", "why-leave-one-out"),
                                   ("maxent-irl", "gradient"),
                                   ("gail", "js"),
                                   ("../reward-ranking", "pairs-versus-ranking"),
                                   ("../../practice/post-training/rl-batch-audit", "alignment"),
                                   ("../../practice/post-training/rl-batch-audit", "reduction")):
                self.assertIn(f"{target}{suffix}#{anchor}", page)
                source = ROOT / "05-post-training/deep-rl" / f"{target}{suffix}"
                self.assertIn(f"{{#{anchor}}}", source.read_text())

    def test_navigation_and_language_examples(self):
        navigation = tomllib.loads((ROOT / "site/nav.toml").read_text())
        objectives = next(section for section in navigation["section"]
                          if section["dir"] == "05-post-training/preference-objectives")
        self.assertIn("05-post-training/rloo.md", objectives["include"])
        practice = next(section for section in navigation["section"] if section["dir"] == "practice/post-training")
        self.assertIn("rl-batch-audit.md", practice["order"])
        snippets = []
        for suffix in (".md", ".en.md"):
            page = (ROOT / f"practice/post-training/rl-batch-audit{suffix}").read_text()
            snippets.append(re.findall(r"```python\n(.*?)```", page, re.S))
            self.assertIn(f"rloo{suffix}", page)
            overview = (ROOT / f"05-post-training/after-ppo{suffix}").read_text()
            self.assertIn(f"rloo{suffix}", overview)
        self.assertEqual(snippets[0], snippets[1])
        self.assertEqual(len(snippets[0]), 1)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch required")
class TorchAlignmentTests(unittest.TestCase):
    def test_documented_snippet_and_gradients(self):
        import torch
        for suffix in (".md", ".en.md"):
            page = (ROOT / f"practice/post-training/rl-batch-audit{suffix}").read_text()
            namespace = {}
            exec(compile(re.findall(r"```python\n(.*?)```", page, re.S)[0], str(suffix), "exec"), namespace)
            logits = torch.tensor([checks.demo_logits()], dtype=torch.float64, requires_grad=True)
            ids = torch.tensor([[0, 1, 2, 3, 3]])
            mask = torch.tensor([[False, True, True, True, False]])
            token_values, sequence = namespace["completion_log_probs"](logits, ids, mask)
            self.assertEqual(tuple(token_values.shape), (1, 4))
            self.assertAlmostEqual(sequence.item(), math.log(0.1), places=6)
            gradient = torch.autograd.grad(-sequence.sum(), logits)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertTrue(torch.equal(gradient[:, 3:], torch.zeros_like(gradient[:, 3:])))
            self.assertLess(gradient[0, 2, 3].item(), 0)
            clean = torch.nan_to_num(logits.detach(), nan=0).requires_grad_()
            _, clean_sequence = namespace["completion_log_probs"](clean, ids, mask)
            self.assertTrue(torch.equal(sequence, clean_sequence))
            for broken in (torch.zeros_like(mask), torch.ones_like(mask), mask.int()):
                with self.assertRaises(ValueError):
                    namespace["completion_log_probs"](clean, ids, broken)


if __name__ == "__main__":
    unittest.main()
