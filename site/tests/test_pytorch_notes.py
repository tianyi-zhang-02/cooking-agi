import ast
from pathlib import Path
import re
import unittest
import warnings

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None


ROOT = Path(__file__).resolve().parents[2]
CHAPTERS = ("README", "tensors-and-storage", "operations-and-shapes", "autograd", "training-loop")


def code_blocks(chapter, language=""):
    source = ROOT / "00-foundations" / "pytorch" / f"{chapter}{language}.md"
    return re.findall(r"```python\n(.*?)```", source.read_text(), re.DOTALL)


class PyTorchDocumentationTests(unittest.TestCase):
    def test_bilingual_code_matches(self):
        for chapter in CHAPTERS:
            with self.subTest(chapter=chapter):
                self.assertTrue(code_blocks(chapter))
                self.assertEqual(code_blocks(chapter), code_blocks(chapter, ".en"))

    def test_python_examples_parse(self):
        for chapter in CHAPTERS:
            for position, block in enumerate(code_blocks(chapter)):
                with self.subTest(chapter=chapter, block=position):
                    ast.parse(block)


@unittest.skipIf(torch is None, "PyTorch is not installed; numerical examples were not verified")
class PyTorchNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        thread_count = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, thread_count)
        cls.namespaces = {}
        with torch.random.fork_rng(devices=[]):
            for chapter in CHAPTERS:
                namespace = {}
                for position, block in enumerate(code_blocks(chapter)):
                    exec(compile(block, f"{chapter}:{position}", "exec"), namespace)
                cls.namespaces[chapter] = namespace

    def test_all_chapter_assertions_executed(self):
        self.assertEqual(set(self.namespaces), set(CHAPTERS))

    def test_overview_diagram_matches_the_update(self):
        namespace = self.namespaces["README"]
        torch.testing.assert_close(namespace["weights"].detach(),
                                   torch.tensor([[0.05, -0.1], [-0.05, 0.1]], dtype=torch.float64))
        self.assertAlmostEqual(namespace["loss_before"].item(), 0.6931471806)
        self.assertAlmostEqual(namespace["loss_after"].item(), 0.5787059562)
        for suffix in (".md", ".en.md"):
            text = (ROOT / "00-foundations/pytorch" / f"README{suffix}").read_text()
            self.assertIn("0.6931", text)
            self.assertIn("0.5787", text)
            self.assertEqual(text.count("<li><small>"), 6)

    def test_scalar_empty_and_item(self):
        self.assertEqual(torch.tensor(5).shape, ())
        self.assertEqual(torch.tensor(5).numel(), 1)
        self.assertEqual(torch.empty(0).numel(), 0)
        self.assertEqual(torch.tensor([[5]]).item(), 5)
        with self.assertRaises(RuntimeError):
            torch.tensor([1, 2]).item()

    def test_assignment_view_and_clone(self):
        base = torch.arange(6).reshape(2, 3)
        alias = base
        view = base[:, 1:]
        copied = base.clone()
        view[0, 0] = 50
        self.assertIs(alias, base)
        self.assertIsNot(view, base)
        self.assertEqual(base[0, 1].item(), 50)
        self.assertEqual(copied[0, 1].item(), 1)

    def test_noncontiguous_compatible_view(self):
        strided = torch.arange(30).reshape(5, 6)[:, ::2]
        flattened = strided.view(-1)
        self.assertFalse(strided.is_contiguous())
        self.assertEqual(flattened.stride(), (2,))
        self.assertEqual(flattened.tolist(), list(range(0, 30, 2)))

    def test_transpose_view_fails_reshape_copies(self):
        transposed = torch.arange(30).reshape(5, 6).transpose(0, 1)
        with self.assertRaises(RuntimeError):
            transposed.view(-1)
        reshaped = transposed.reshape(-1)
        self.assertNotEqual(transposed.data_ptr(), reshaped.data_ptr())
        self.assertEqual(reshaped[:5].tolist(), [0, 6, 12, 18, 24])

    def test_advanced_index_read_copy_write_base(self):
        base = torch.arange(6)
        selected = base[[1, 3]]
        selected[0] = 99
        self.assertEqual(base[1].item(), 1)
        base[[1, 3]] = -1
        self.assertEqual(base.tolist(), [0, -1, 2, -1, 4, 5])

    def test_chained_slices_are_not_two_axes(self):
        matrix = torch.arange(16).reshape(4, 4)
        self.assertEqual(matrix[::2, ::2].shape, (2, 2))
        self.assertEqual(matrix[::2][::2].shape, (1, 4))
        self.assertEqual(matrix[-1, -1].item(), 15)
        with self.assertRaises(ValueError):
            matrix[::-1]

    def test_squeeze_may_remove_batch(self):
        batch = torch.ones(1, 3, 1)
        self.assertEqual(batch.squeeze().shape, (3,))
        self.assertEqual(batch.squeeze(-1).shape, (1, 3))

    def test_conversion_can_be_noop(self):
        values = torch.ones(2, dtype=torch.float32)
        self.assertIs(values.float(), values)
        self.assertIs(values.to(dtype=torch.float32, device="cpu"), values)

    def test_numpy_storage_contracts(self):
        try:
            import numpy
        except ModuleNotFoundError:
            self.skipTest("NumPy is not installed")
        array = numpy.array([1.0, 2.0], dtype=numpy.float32)
        shared = torch.from_numpy(array)
        copied = torch.tensor(array)
        array[0] = 9
        self.assertEqual(shared[0].item(), 9)
        self.assertEqual(copied[0].item(), 1)
        tracked = torch.tensor([2.0], requires_grad=True)
        with self.assertRaises(RuntimeError):
            tracked.numpy()
        forced = tracked.numpy(force=True)
        self.assertTrue(numpy.shares_memory(forced, tracked.detach().numpy()))

    def test_chunk_count_and_empty_tensor_split(self):
        self.assertEqual(len(torch.arange(4).chunk(3)), 2)
        self.assertEqual(len(torch.arange(4).tensor_split(3)), 3)
        self.assertEqual([part.numel() for part in torch.arange(2).tensor_split(4)], [1, 1, 0, 0])

    def test_broadcasting_silent_cross_batch_loss(self):
        predictions = torch.tensor([[1.0], [3.0]])
        targets = torch.tensor([1.0, 3.0])
        self.assertEqual((predictions - targets).square().mean().item(), 2.0)
        self.assertEqual((predictions.squeeze(-1) - targets).square().mean().item(), 0.0)

    def test_zero_dimension_and_inplace_broadcast(self):
        self.assertEqual((torch.empty(0, 3) + torch.ones(1, 3)).shape, (0, 3))
        with self.assertRaises(RuntimeError):
            torch.ones(1, 3).add_(torch.ones(2, 3))

    def test_expand_gradient_and_overlapping_write(self):
        source = torch.tensor([2.0], requires_grad=True)
        expanded = source.expand(3)
        self.assertEqual(expanded.stride(), (0,))
        expanded.sum().backward()
        self.assertEqual(source.grad.item(), 3.0)
        with self.assertRaises(RuntimeError):
            expanded.detach().add_(1)

    def test_masked_reduction_denominators(self):
        token_losses = torch.tensor([[1.0, 3.0, 99.0], [8.0, 99.0, 99.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        token_mean = (token_losses * mask).sum() / mask.sum()
        sample_mean = ((token_losses * mask).sum(-1) / mask.sum(-1)).mean()
        self.assertEqual(token_mean.item(), 4.0)
        self.assertEqual(sample_mean.item(), 5.0)

    def test_bmm_requires_matching_batch(self):
        queries = torch.ones(2, 3, 4)
        keys = torch.ones(1, 4, 5)
        self.assertEqual((queries @ keys).shape, (2, 3, 5))
        with self.assertRaises(RuntimeError):
            torch.bmm(queries, keys)

    def test_addbmm_reduces_batch(self):
        queries = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        keys = torch.arange(40, dtype=torch.float32).reshape(2, 4, 5)
        products = torch.bmm(queries, keys)
        torch.testing.assert_close(torch.addbmm(torch.zeros(3, 5), queries, keys), products.sum(0))
        torch.testing.assert_close(torch.baddbmm(torch.zeros(2, 3, 5), queries, keys), products)

    def test_stable_log_softmax_gradient(self):
        logits = torch.tensor([[1000.0, 999.0]], requires_grad=True)
        labels = torch.tensor([1])
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertAlmostEqual(logits.grad.sum().item(), 0, places=6)

    def test_nonleaf_gradient_not_retained_by_default(self):
        parameter = torch.tensor(2.0, requires_grad=True)
        intermediate = parameter.square()
        (3 * intermediate).backward()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.assertIsNone(intermediate.grad)
        self.assertEqual(parameter.grad.item(), 12.0)

    def test_nonscalar_backward_needs_seed(self):
        parameter = torch.tensor([1.0, 2.0], requires_grad=True)
        with self.assertRaises(RuntimeError):
            parameter.square().backward()
        parameter.square().backward(torch.tensor([1.0, 3.0]))
        torch.testing.assert_close(parameter.grad, torch.tensor([2.0, 12.0]))

    def test_second_backward_requires_saved_values(self):
        parameter = torch.tensor(2.0, requires_grad=True)
        loss = parameter.square()
        loss.backward()
        with self.assertRaises(RuntimeError):
            loss.backward()
        parameter.square().backward()
        self.assertEqual(parameter.grad.item(), 8.0)

    def test_inplace_version_check(self):
        parameter = torch.tensor(2.0, requires_grad=True)
        loss = parameter.square()
        with torch.no_grad():
            parameter.add_(1)
        with self.assertRaises(RuntimeError):
            loss.backward()

    def test_no_grad_factory_exception(self):
        with torch.no_grad():
            parameter = torch.tensor(2.0, requires_grad=True)
            output = parameter.square()
        self.assertTrue(parameter.requires_grad)
        self.assertFalse(output.requires_grad)

    def test_none_and_zero_gradient_updates_differ(self):
        absent = torch.nn.Parameter(torch.tensor(2.0))
        zero = torch.nn.Parameter(torch.tensor(2.0))
        optimizer = torch.optim.SGD([absent, zero], lr=0.1, weight_decay=0.5)
        zero.grad = torch.zeros_like(zero)
        optimizer.step()
        self.assertEqual(absent.item(), 2.0)
        self.assertAlmostEqual(zero.item(), 1.9, places=6)

    def test_dataset_reproducibility_and_alignment(self):
        namespace = self.namespaces["training-loop"]
        first = namespace["make_dataset"](23, 5)
        repeated = namespace["make_dataset"](23, 5)
        for actual, expected in zip(first.tensors, repeated.tensors):
            torch.testing.assert_close(actual, expected)
        loader = torch.utils.data.DataLoader(first, batch_size=7, shuffle=True)
        for features, labels in loader:
            expected = (features[:, 0] + 0.5 * features[:, 1] > 0).long()
            torch.testing.assert_close(labels, expected)

    def test_validation_is_invariant_to_batch_size(self):
        namespace = self.namespaces["training-loop"]
        scores = []
        for size in (1, 16, 47):
            loader = torch.utils.data.DataLoader(namespace["validation_data"], batch_size=size)
            scores.append(namespace["evaluate"](namespace["model"], loader))
        for score in scores[1:]:
            self.assertAlmostEqual(score["loss"], scores[0]["loss"], places=6)
            self.assertEqual(score["accuracy"], scores[0]["accuracy"])

    def test_validation_restores_mode_and_rejects_empty(self):
        namespace = self.namespaces["training-loop"]
        model = namespace["model"]
        for previous_mode in (False, True):
            model.train(previous_mode)
            namespace["evaluate"](model, namespace["validation_loader"])
            self.assertEqual(model.training, previous_mode)
            with self.assertRaises(ValueError):
                namespace["evaluate"](model, [])
            self.assertEqual(model.training, previous_mode)
        with self.assertRaises(ValueError):
            namespace["train_epoch"](model, [], namespace["optimizer"])

    def test_training_reduces_toy_holdout_loss(self):
        namespace = self.namespaces["training-loop"]
        self.assertLess(namespace["history"][-1]["loss"], namespace["initial_metrics"]["loss"])
        self.assertEqual(len(namespace["history"]), 25)

    def test_module_registration_and_buffer(self):
        model = self.namespaces["training-loop"]["TinyClassifier"]()
        self.assertEqual(len(list(model.parameters())), 4)
        self.assertIn("input_scale", model.state_dict())
        self.assertNotIn("input_scale", dict(model.named_parameters()))

    def test_snapshot_is_independent(self):
        namespace = self.namespaces["training-loop"]
        snapshot_weight = namespace["snapshot"]["model"]["weight"]
        model_weight = namespace["model"].weight
        self.assertNotEqual(snapshot_weight.data_ptr(), model_weight.data_ptr())
        torch.testing.assert_close(snapshot_weight, model_weight)

    def test_checkpoint_optimizer_state_and_next_update(self):
        namespace = self.namespaces["training-loop"]
        model = torch.nn.Linear(2, 2)
        restored_model = torch.nn.Linear(2, 2)
        model.load_state_dict(namespace["snapshot"]["model"])
        restored_model.load_state_dict(namespace["loaded"]["model"])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.15, momentum=0.9)
        restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.15, momentum=0.9)
        optimizer.load_state_dict(namespace["snapshot"]["optimizer"])
        restored_optimizer.load_state_dict(namespace["loaded"]["optimizer"])
        features, labels = next(iter(namespace["validation_loader"]))
        for current_model, current_optimizer in ((model, optimizer), (restored_model, restored_optimizer)):
            current_optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(current_model(features), labels).backward()
            current_optimizer.step()
        for actual, expected in zip(model.parameters(), restored_model.parameters()):
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
