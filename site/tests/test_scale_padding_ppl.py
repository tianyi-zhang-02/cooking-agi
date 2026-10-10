import itertools
import math
from pathlib import Path
import re
import unittest

import torch
import torch.nn.functional as functional


ROOT = Path(__file__).resolve().parents[2]
DECODER = "00-foundations/core/decoder-only"
METRICS = "07-evaluation/text-metrics"


def example(chapter, function, language=""):
    source = (ROOT / f"{chapter}{language}.md").read_text()
    return next(block for block in re.findall(r"```python\n(.*?)```", source, re.S)
                if f"def {function}(" in block)


class ScalePaddingPerplexityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = {}
        for chapter, function in ((DECODER, "last_valid_logits"), (METRICS, "causal_nll_totals")):
            exec(compile(example(chapter, function), chapter, "exec"), cls.namespace)

    def test_examples_match_across_languages(self):
        for chapter, function in ((DECODER, "last_valid_logits"), (METRICS, "causal_nll_totals")):
            self.assertEqual(example(chapter, function), example(chapter, function, ".en"))

    def test_padding_figure_is_localized(self):
        for language in ("", ".en"):
            source = (ROOT / f"{DECODER}{language}.md").read_text()
            section = source.split("{#left-padding}", 1)[1].split("{#padding-controls}", 1)[0]
            figure = re.search(r"<figure.*?</figure>", section, re.S).group()
            self.assertEqual(figure.count("<li>"), 2)
            if language:
                self.assertNotRegex(figure, r"[\u4e00-\u9fff]")
            else:
                self.assertRegex(figure, r"[\u4e00-\u9fff]")
            self.assertIn("536ecc007387a50e77603bb5d92100e9b07514cc", source)

    def test_softmax_shift_and_scale(self):
        scores = torch.tensor([8., -8.], dtype=torch.float64)
        unscaled = scores.softmax(-1)
        torch.testing.assert_close(unscaled, (scores - scores.max()).softmax(-1))
        self.assertAlmostEqual(unscaled[0].item(), 0.9999998874648379)
        self.assertAlmostEqual((scores / 8).softmax(-1)[0].item(), 0.8807970779778823)
        torch.testing.assert_close(torch.tensor([1000., 1000.]).softmax(-1), torch.tensor([0.5, 0.5]))

    def test_variance_by_exact_enumeration(self):
        dimension = 3
        samples = torch.tensor(list(itertools.product((-1., 1.), repeat=2 * dimension)), dtype=torch.float64)
        products = (samples[:, :dimension] * samples[:, dimension:]).sum(-1)
        self.assertEqual(products.mean().item(), 0)
        self.assertEqual(products.var(unbiased=False).item(), dimension)
        self.assertAlmostEqual((products / math.sqrt(dimension)).var(unbiased=False).item(), 1)
        self.assertAlmostEqual((products / dimension).var(unbiased=False).item(), 1 / dimension)
        correlated = samples[:, :dimension].square().sum(-1)
        self.assertEqual(correlated.mean().item(), dimension)

    def test_attention_gradient_is_not_cross_entropy_gradient(self):
        scores = torch.tensor([8., -8.], dtype=torch.float64, requires_grad=True)
        weights = scores.softmax(-1)
        attention_gradient, = torch.autograd.grad(weights[0], scores, retain_graph=True)
        loss = functional.cross_entropy(scores.unsqueeze(0), torch.tensor([1]))
        classifier_gradient, = torch.autograd.grad(loss, scores)
        self.assertLess(attention_gradient.abs().max().item(), 2e-7)
        self.assertGreater(classifier_gradient.abs().max().item(), 0.99)

    def test_last_valid_readout_for_both_directions(self):
        select = self.namespace["last_valid_logits"]
        scores = torch.arange(30).reshape(2, 5, 3)
        for mask, indices in (([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], [2, 4]),
                              ([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]], [4, 4]),
                              ([[0, 1, 0, 1, 0], [0, 0, 0, 0, 1]], [3, 4])):
            torch.testing.assert_close(select(scores, torch.tensor(mask)), scores[torch.arange(2), indices])

    def test_last_valid_readout_noncontiguous(self):
        scores = torch.arange(60).reshape(2, 5, 6)[..., ::2]
        mask = torch.tensor([[False, False, True, True, True], [True] * 5])
        self.assertFalse(scores.is_contiguous())
        torch.testing.assert_close(self.namespace["last_valid_logits"](scores, mask), scores[:, -1])

    def test_invalid_readout_inputs(self):
        select = self.namespace["last_valid_logits"]
        for scores, mask in ((torch.zeros(1, 2, 3), torch.zeros(1, 2)),
                             (torch.zeros(1, 2, 3), torch.tensor([[1, 2]])),
                             (torch.zeros(1, 2, 3), torch.ones(1, 3)),
                             (torch.zeros(1, 0, 3), torch.zeros(1, 0)),
                             (torch.zeros(1, 2), torch.ones(1, 2))):
            with self.subTest(shape=scores.shape), self.assertRaises(ValueError):
                select(scores, mask)

    def test_logical_positions_and_length_shortcut(self):
        mask = torch.tensor([[0, 0, 1, 1, 1]])
        positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)
        self.assertEqual(positions[mask.bool()].tolist(), [0, 1, 2])
        self.assertEqual(int(mask.sum(-1) - 1), 2)
        self.assertEqual(int(torch.arange(5).masked_fill(~mask.bool(), -1).amax()), 4)

    def test_ppl_uniform_example_and_overlapping_windows(self):
        score = self.namespace["causal_nll_totals"]
        logits = torch.zeros(1, 4, 6)
        first_total, first_count = score(logits, torch.tensor([[-100, 1, 2, 3]]))
        second_total, second_count = score(logits, torch.tensor([[-100, -100, 4, 5]]))
        self.assertEqual((first_count, second_count), (3, 2))
        self.assertAlmostEqual(float(first_total + second_total), 5 * math.log(6))
        self.assertAlmostEqual(math.exp(float((first_total + second_total) / 5)), 6)
        self.assertEqual(len(set([1, 2, 3] + [4, 5])), first_count + second_count)

    def test_ppl_weighted_aggregation(self):
        total = 2 * math.log(2) + 8 * math.log(8)
        self.assertAlmostEqual(math.exp(total / 10), 6.06286626604159)
        self.assertNotAlmostEqual(math.exp(total / 10), (2 + 8) / 2)
        self.assertAlmostEqual(math.exp(-(math.log(0.5) + math.log(0.25)) / 2), math.sqrt(8))
        self.assertAlmostEqual(2 ** (-(math.log2(0.5) + math.log2(0.25)) / 2), math.sqrt(8))

    def test_ppl_masked_gradient_and_direct_calculation(self):
        generator = torch.Generator().manual_seed(614)
        logits = torch.randn(2, 5, 7, generator=generator, dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([[-100, 1, 2, -100, -100], [-100, -100, 3, 4, 5]])
        total, count = self.namespace["causal_nll_totals"](logits, labels)
        shifted = labels[:, 1:]
        valid = shifted != -100
        direct = -logits[:, :-1].log_softmax(-1)[valid].gather(1, shifted[valid].unsqueeze(1)).sum()
        torch.testing.assert_close(total, direct)
        self.assertEqual(count, 5)
        gradient, = torch.autograd.grad(total, logits)
        self.assertEqual(torch.count_nonzero(gradient[:, :-1][~valid]).item(), 0)
        self.assertEqual(torch.count_nonzero(gradient[:, -1]).item(), 0)

    def test_ppl_unscored_or_misaligned_inputs(self):
        score = self.namespace["causal_nll_totals"]
        for logits, labels in ((torch.zeros(1, 3, 4), torch.full((1, 3), -100)),
                               (torch.zeros(1, 1, 4), torch.tensor([[1]])),
                               (torch.zeros(1, 3, 4), torch.ones(1, 4, dtype=torch.long))):
            with self.subTest(shape=logits.shape), self.assertRaises(ValueError):
                score(logits, labels)


if __name__ == "__main__":
    unittest.main()
