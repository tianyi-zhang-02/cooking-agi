import ast
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
CHAPTER = "06-systems/tensor-parallel"


def blocks(language=""):
    source = (ROOT / f"{CHAPTER}{language}.md").read_text()
    return re.findall(r"```python\n(.*?)```", source, re.S)


class TensorParallelDocumentationTests(unittest.TestCase):
    def test_bilingual_code(self):
        self.assertEqual(blocks(), blocks(".en"))
        for block in blocks():
            ast.parse(block)

    def test_localized_figures_and_stable_section_ids(self):
        for language in ("", ".en"):
            source = (ROOT / f"{CHAPTER}{language}.md").read_text()
            figures = re.findall(r"<figure.*?</figure>", source, re.S)
            self.assertEqual(len(figures), 2)
            for figure in figures:
                if language:
                    self.assertNotRegex(figure, r"[\u4e00-\u9fff]")
                else:
                    self.assertRegex(figure, r"[\u4e00-\u9fff]")
            for anchor in ("two-ranks", "column-row", "collectives", "backward",
                           "decoder-count", "sequence-parallel", "before-scaling"):
                self.assertIn("{#" + anchor + "}", source)
            self.assertIn("4603a836261fb39fd0050342d58dc66aecdf6f74", source)
            self.assertNotIn("Megatron-LM/blob/main/", source)

    def test_eight_gpu_weight_budgets(self):
        for tensor_ranks, data_ranks, expected in ((8, 1, 2), (1, 8, 16), (4, 2, 4)):
            self.assertEqual(tensor_ranks * data_ranks, 8)
            self.assertEqual(16 / tensor_ranks, expected)
            self.assertEqual(8 * expected, 16 * data_ranks)

    def test_pipeline_boundary_bytes_and_time(self):
        decode_bytes = 8 * 3072 * 2
        prefill_bytes = 8 * 4096 * 3072 * 2
        self.assertEqual(decode_bytes, 48 * 2**10)
        self.assertEqual(prefill_bytes, 192 * 2**20)
        self.assertAlmostEqual(decode_bytes / (12 * 2**30) * 1e6, 3.814697265625)
        self.assertAlmostEqual(prefill_bytes / (12 * 2**30) * 1e3, 15.625)


@unittest.skipIf(torch is None, "PyTorch is unavailable; tensor-parallel examples were not verified")
class TensorParallelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values = {}
        for block in blocks():
            exec(compile(block, CHAPTER, "exec"), cls.values)

    def test_worked_forward_and_input_gradient(self):
        torch.testing.assert_close(self.values["full"], torch.tensor([[9., 6.]], dtype=torch.float64))
        torch.testing.assert_close(self.values["combined"], torch.tensor([[9., 3.]], dtype=torch.float64))
        expected = ([[3., 3.]], [[6., 0.]])
        for actual, value in zip(self.values["input_contributions"], expected):
            torch.testing.assert_close(actual, torch.tensor(value, dtype=torch.float64))

    def test_partitioned_forward_backward_and_update(self):
        generator = torch.Generator().manual_seed(731)
        for tokens, hidden, intermediate, output_width in ((1, 3, 12, 3), (5, 4, 8, 7), (2, 2, 6, 2)):
            for parts in (1, 2, intermediate):
                with self.subTest(shape=(tokens, hidden, intermediate, output_width), parts=parts):
                    parameters = tuple(torch.randn(shape, generator=generator, dtype=torch.float64,
                                                   requires_grad=True)
                                       for shape in ((tokens, hidden), (hidden, intermediate),
                                                     (intermediate, output_width)))
                    inputs, first_weight, second_weight = parameters
                    full = torch.relu(inputs @ first_weight) @ second_weight
                    split = self.values["split_ffn"](*parameters, parts=parts)
                    torch.testing.assert_close(split, full)
                    target = torch.randn(full.shape, generator=generator, dtype=torch.float64)
                    full_grads = torch.autograd.grad((full - target).square().mean(), parameters)
                    split_grads = torch.autograd.grad((split - target).square().mean(), parameters)
                    for parameter, full_grad, split_grad in zip(parameters, full_grads, split_grads):
                        torch.testing.assert_close(split_grad, full_grad)
                        torch.testing.assert_close(parameter - 0.01 * split_grad,
                                                   parameter - 0.01 * full_grad)

    def test_noncontiguous_inputs(self):
        inputs = torch.arange(12, dtype=torch.float64).reshape(2, 6)[:, ::2]
        first = torch.arange(12, dtype=torch.float64).reshape(4, 3).T
        second = torch.arange(8, dtype=torch.float64).reshape(2, 4).T
        self.assertFalse(inputs.is_contiguous())
        torch.testing.assert_close(self.values["split_ffn"](inputs, first, second),
                                   torch.relu(inputs @ first) @ second)

    def test_empty_token_batch(self):
        result = self.values["split_ffn"](torch.empty(0, 2), torch.ones(2, 4), torch.ones(4, 2))
        self.assertEqual(tuple(result.shape), (0, 2))

    def test_invalid_partitions_and_shapes(self):
        inputs, first, second = (self.values[name] for name in ("inputs", "first_weight", "second_weight"))
        for parts in (0, -1, True, 2.0, 3, 5):
            with self.subTest(parts=parts), self.assertRaises(ValueError):
                self.values["split_ffn"](inputs, first, second, parts)
        invalid = ((inputs[0], first, second), (inputs, first[:1], second),
                   (inputs, first, second[:3]), (inputs, first[:, :0], second[:0]))
        for arguments in invalid:
            with self.assertRaises(ValueError):
                self.values["split_ffn"](*arguments)

    def test_nonlinearity_requires_complete_partial_sum(self):
        first = torch.tensor(3.)
        second = torch.tensor(-4.)
        self.assertEqual(torch.relu(first + second).item(), 0)
        self.assertEqual((torch.relu(first) + torch.relu(second)).item(), 3)

    def test_bias_is_added_once(self):
        partials = torch.tensor([[1., 2.], [8., 4.]])
        bias = torch.tensor([0.5, -0.25])
        correct = partials.sum(0) + bias
        wrong = (partials + bias).sum(0)
        torch.testing.assert_close(wrong - correct, bias)

    def test_collective_results_are_different(self):
        contributions = torch.tensor([[2, 5], [7, 9]])
        gathered = torch.cat(tuple(contributions.unbind()))
        reduced = contributions.sum(0)
        self.assertEqual(gathered.tolist(), [2, 5, 7, 9])
        self.assertEqual(reduced.tolist(), [9, 14])
        scattered = reduced.chunk(2)
        self.assertEqual([part.tolist() for part in scattered], [[9], [14]])
        torch.testing.assert_close(torch.cat(scattered), reduced)

    def test_sequence_and_channel_partitions(self):
        inputs = torch.arange(12).reshape(4, 3)
        token_shards = inputs.chunk(2, 0)
        gathered = torch.cat(token_shards, dim=0)
        torch.testing.assert_close(gathered, inputs)
        partials = (inputs * 2, inputs * 3)
        result_shards = sum(partials).chunk(2, 0)
        self.assertEqual(tuple(result_shards[0].shape), (2, 3))
        torch.testing.assert_close(torch.cat(result_shards), inputs * 5)


if __name__ == "__main__":
    unittest.main()
