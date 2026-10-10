import importlib.util
import math
from pathlib import Path
import re
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None


ROOT = Path(__file__).resolve().parents[2]


@unittest.skipIf(torch is None, "PyTorch is unavailable; loss examples were not verified")
class LossContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("loss_contracts", ROOT / "00-foundations/code/loss_contracts.py")
        cls.losses = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.losses)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, threads)

    def test_hard_target_value_and_gradient(self):
        logits = torch.tensor([[2., 3., 5.]], dtype=torch.float64).log().requires_grad_()
        loss = self.losses.hard_cross_entropy(logits, torch.tensor([2]))
        self.assertAlmostEqual(loss.item(), math.log(2))
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.tensor([[.2, .3, -.5]], dtype=torch.float64))

    def test_shift_invariance_and_large_logits(self):
        logits = torch.tensor([[1000., 999., -1000.]], dtype=torch.float64)
        targets = torch.tensor([2])
        expected = torch.nn.functional.cross_entropy(logits, targets)
        torch.testing.assert_close(self.losses.hard_cross_entropy(logits, targets), expected)
        torch.testing.assert_close(self.losses.hard_cross_entropy(logits + 400, targets), expected)

    def test_soft_target_value_and_gradient(self):
        logits = torch.tensor([[2., 3., 5.]], dtype=torch.float64).log().requires_grad_()
        targets = torch.tensor([[.1, .2, .7]], dtype=torch.float64)
        loss = self.losses.soft_cross_entropy(logits, targets)
        expected = torch.nn.functional.cross_entropy(logits, targets)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.tensor([[.1, .1, -.2]], dtype=torch.float64))

    def test_gradcheck_hard_and_soft(self):
        logits = torch.tensor([[.3, -.4, 1.2], [-.2, .5, .8]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda scores: self.losses.hard_cross_entropy(scores, torch.tensor([0, 2])), (logits,)))
        soft = torch.tensor([[.1, .2, .7], [.5, .1, .4]], dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(lambda scores: self.losses.soft_cross_entropy(scores, soft), (logits,)))

    def test_ignored_nan_row_does_not_enter_reduction(self):
        logits = torch.tensor([[0., 0.], [float("nan"), float("nan")]], requires_grad=True)
        loss = self.losses.hard_cross_entropy(logits, torch.tensor([1, -100]))
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        loss.backward()
        torch.testing.assert_close(logits.grad[1], torch.zeros(2))

    def test_empty_targets_raise(self):
        for targets in (torch.tensor([-100, -100]), torch.empty(0, dtype=torch.long)):
            with self.subTest(targets=targets), self.assertRaisesRegex(ValueError, "no supervised"):
                self.losses.hard_cross_entropy(torch.zeros(targets.numel(), 3), targets)

    def test_bad_hard_targets_raise(self):
        for targets in (torch.tensor([3]), torch.tensor([-1]), torch.tensor([1.])):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                self.losses.hard_cross_entropy(torch.zeros(1, 3), targets)

    def test_bad_soft_targets_raise(self):
        for targets in ([[.1, .1, .1]], [[-.1, .5, .6]], [[float("nan"), .5, .5]]):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                self.losses.soft_cross_entropy(torch.zeros(1, 3), torch.tensor(targets))

    def test_binary_extremes_and_zero_gradient(self):
        logits = torch.tensor([-1000., -100., 0., 100., 1000.], dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([1., 1., 1., 0., 0.], dtype=torch.float64)
        actual = self.losses.binary_cross_entropy_with_logits(logits, targets)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        torch.testing.assert_close(logits.grad, (logits.detach().sigmoid() - targets) / 5)

    def test_probability_clamping_loses_gradient(self):
        logits = torch.tensor([-100.], dtype=torch.float64, requires_grad=True)
        clipped = logits.sigmoid().clamp(1e-7, 1 - 1e-7)
        (-clipped.log()).backward()
        self.assertEqual(logits.grad.item(), 0.)
        logits.grad = None
        self.losses.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)).backward()
        self.assertEqual(logits.grad.item(), -1.)

    def test_binary_contract_rejects_broadcast(self):
        with self.assertRaises(ValueError):
            self.losses.binary_cross_entropy_with_logits(torch.zeros(2, 1), torch.zeros(2))
        with self.assertRaises(ValueError):
            self.losses.binary_cross_entropy_with_logits(torch.zeros(1), torch.tensor([2.]))

    def test_sft_shift_uses_preceding_position(self):
        logits = torch.zeros(1, 6, 6, dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([[-100, -100, -100, 4, 5, -100]])
        visible = torch.tensor([[True, True, True, True, True, False]])
        loss = self.losses.causal_sft_loss(logits, labels, visible)
        self.assertAlmostEqual(loss.item(), math.log(6))
        loss.backward()
        active = (logits.grad.abs().sum(-1) > 0).nonzero().tolist()
        self.assertEqual(active, [[0, 2], [0, 3]])

    def test_eos_same_id_as_padding(self):
        logits = torch.zeros(1, 4, 3, requires_grad=True)
        labels = torch.tensor([[-100, 1, 2, 2]])
        visible = torch.tensor([[True, True, True, False]])
        self.losses.causal_sft_loss(logits, labels, visible).backward()
        self.assertNotEqual(logits.grad[0, 1, 2].item(), 0.)
        torch.testing.assert_close(logits.grad[0, 2], torch.zeros(3))

    def test_left_padding_does_not_predict_first_real_token(self):
        logits = torch.zeros(1, 4, 3, requires_grad=True)
        labels = torch.tensor([[2, 0, 1, 2]])
        visible = torch.tensor([[False, True, True, True]])
        self.losses.causal_sft_loss(logits, labels, visible).backward()
        torch.testing.assert_close(logits.grad[0, 0], torch.zeros(3))

    def test_masked_context_can_receive_gradient(self):
        embeddings = torch.tensor([[[.2, -.1], [.3, .4], [-.2, .5]]], requires_grad=True)
        prefix_states = embeddings.cumsum(dim=1)
        projection = torch.tensor([[1., -1., .5], [.5, 1., -1.]])
        logits = prefix_states @ projection
        labels = torch.tensor([[-100, -100, 2]])
        self.losses.causal_sft_loss(logits, labels, torch.ones(1, 3, dtype=torch.bool)).backward()
        self.assertGreater(embeddings.grad[0, 0].abs().sum().item(), 0.)
        torch.testing.assert_close(embeddings.grad[0, 2], torch.zeros(2))

    def test_token_mean_differs_from_mean_of_means(self):
        first, second = torch.tensor([2.]), torch.tensor([1., 1., 1.])
        self.assertEqual(torch.cat([first, second]).mean().item(), 1.25)
        self.assertEqual(((first.mean() + second.mean()) / 2).item(), 1.5)

    def test_weighted_hard_and_one_hot_denominators(self):
        logits = torch.zeros(2, 2, dtype=torch.float64)
        targets = torch.tensor([0, 1])
        weights = torch.tensor([1., 3.], dtype=torch.float64)
        hard = torch.nn.functional.cross_entropy(logits, targets, weight=weights)
        soft = torch.nn.functional.cross_entropy(logits, torch.nn.functional.one_hot(targets, 2).double(), weight=weights)
        self.assertAlmostEqual(hard.item(), math.log(2))
        self.assertAlmostEqual(soft.item(), 2 * math.log(2))

    def test_dft_detach_changes_the_gradient_direction(self):
        logits = torch.tensor([[math.log(.99), math.log(.01)]], dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([1])
        loss = self.losses.dft_token_loss(logits, targets)
        gradient = torch.autograd.grad(loss, logits)[0]
        self.assertAlmostEqual(gradient[0, 1].item(), -.0099)
        probability = logits.softmax(-1)[0, 1]
        wrong = -probability * probability.log()
        wrong_gradient = torch.autograd.grad(wrong, logits)[0]
        self.assertGreater(wrong_gradient[0, 1].item(), 0.)

    def test_sft_bilingual_examples_match_and_run(self):
        snippets = []
        for suffix in ("", ".en"):
            text = (ROOT / f"05-post-training/sft-and-its-ceiling{suffix}.md").read_text()
            snippets.append(re.findall(r"```python\n(.*?)```", text, re.DOTALL))
        self.assertEqual(snippets[0], snippets[1])
        self.assertTrue(snippets[0])
        namespace = {}
        for snippet in snippets[0]:
            exec(compile(snippet, "sft-example", "exec"), namespace)


if __name__ == "__main__":
    unittest.main()
