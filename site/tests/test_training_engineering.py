import ast
import copy
from pathlib import Path
import re
import unittest

try:
    import torch
    from torch import nn
    from torch.utils.checkpoint import checkpoint
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None


ROOT = Path(__file__).resolve().parents[2]
CHAPTERS = (
    "practice/post-training/checkpoint-and-resume",
    "00-foundations/pytorch/training-loop",
    "00-foundations/deep-dives/precision-and-memory",
)


def blocks(chapter, language=""):
    return re.findall(r"```python\n(.*?)```", (ROOT / f"{chapter}{language}.md").read_text(), re.S)


def selected_namespace(chapter, function_name=None):
    namespace = {}
    for block in blocks(chapter):
        if function_name and not re.search(rf"^def {re.escape(function_name)}\(", block, re.M):
            continue
        exec(compile(block, chapter, "exec"), namespace)
    return namespace


class TrainingEngineeringDocsTests(unittest.TestCase):
    def test_code_is_equivalent_and_parses(self):
        for chapter in CHAPTERS:
            with self.subTest(chapter=chapter):
                self.assertEqual(blocks(chapter), blocks(chapter, ".en"))
                for block in blocks(chapter):
                    ast.parse(block)

    def test_figures_follow_page_language(self):
        for chapter in CHAPTERS:
            chinese = (ROOT / f"{chapter}.md").read_text()
            english = (ROOT / f"{chapter}.en.md").read_text()
            chinese_figures = re.findall(r'<figure class="worked-update[^"]*">(.*?)</figure>', chinese, re.S)
            english_figures = re.findall(r'<figure class="worked-update[^"]*">(.*?)</figure>', english, re.S)
            self.assertEqual(len(chinese_figures), len(english_figures))
            self.assertTrue(chinese_figures)
            for figure in chinese_figures:
                self.assertRegex(figure, r"[\u4e00-\u9fff]")
                self.assertNotIn("HOST MEMORY", figure)
                self.assertNotIn("RESTORE BOTH", figure)
            for figure in english_figures:
                self.assertNotRegex(figure, r"[\u4e00-\u9fff]")

    def test_cpu_scope_is_visible(self):
        for language in ("", ".en"):
            text = (ROOT / f"{CHAPTERS[1]}{language}.md").read_text()
            self.assertIn("CUDA", text)
            self.assertIn("CPU", text)
            self.assertIn("checkpoint-and-resume", text)


class TrainingEngineeringArithmeticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.memory = selected_namespace(CHAPTERS[2])
        cls.resume = selected_namespace(CHAPTERS[0])
        cls.pipeline = selected_namespace(CHAPTERS[1], "ideal_pipeline_ms")

    def test_momentum_counterexample(self):
        self.assertAlmostEqual(self.resume["saved_weight"], 0.9)
        self.assertEqual(self.resume["saved_velocity"], 1)
        self.assertAlmostEqual(self.resume["resumed_weight"], 0.71)
        self.assertAlmostEqual(self.resume["warm_weight"], 0.8)

    def test_complete_momentum_resume_matches_continuation(self):
        step = self.resume["momentum_step"]
        gradients = [1, -2, 0.5, 3, -1, 0.2]
        uninterrupted = (1.0, 0.0)
        for gradient in gradients:
            uninterrupted = step(*uninterrupted, gradient)
        saved = (1.0, 0.0)
        for gradient in gradients[:3]:
            saved = step(*saved, gradient)
        restored = tuple(saved)
        for gradient in gradients[3:]:
            restored = step(*restored, gradient)
        self.assertEqual(uninterrupted, restored)

    def test_full_state_and_units(self):
        ledger = self.memory["state_bytes"](1_000_000_000, 1_000_000_000)
        self.assertEqual(list(ledger.values()), [2_000_000_000, 2_000_000_000, 4_000_000_000, 8_000_000_000])
        self.assertAlmostEqual(sum(ledger.values()) / 2**30, 14.901161193847656)

    def test_adapter_keeps_frozen_weights(self):
        ledger = self.memory["state_bytes"](1_010_000_000, 10_000_000)
        self.assertEqual(ledger["weights"], 2_020_000_000)
        self.assertEqual(sum(ledger.values()), 2_160_000_000)

    def test_fp32_without_master_and_inference_only(self):
        counter = self.memory["state_bytes"]
        self.assertEqual(sum(counter(100, 100, weight_bytes=4, gradient_bytes=4, master_bytes=0).values()), 1600)
        self.assertEqual(sum(counter(100, 0).values()), 200)
        self.assertEqual(sum(counter(0, 0).values()), 0)

    def test_invalid_memory_counts_and_widths(self):
        counter = self.memory["state_bytes"]
        for total, trainable in ((-1, 0), (10, -1), (10, 11), (1.5, 1), (True, 1), (1, False)):
            with self.subTest(total=total, trainable=trainable), self.assertRaises(ValueError):
                counter(total, trainable)
        for width in (-1, 1.5, True, float("nan")):
            with self.subTest(width=width), self.assertRaises(ValueError):
                counter(10, 10, weight_bytes=width)

    def test_ideal_sharded_state_table(self):
        ledger = self.memory["state_bytes"](1_000_000_000, 1_000_000_000)
        weights = ledger["weights"] / 10**9
        gradients = ledger["gradients"] / 10**9
        optimizer = (ledger["master"] + ledger["moments"]) / 10**9
        self.assertEqual(weights + gradients + optimizer, 16)
        self.assertEqual(weights + gradients + optimizer / 8, 5.5)
        self.assertEqual(weights + (gradients + optimizer) / 8, 3.75)
        self.assertEqual((weights + gradients + optimizer) / 8, 2)

    def test_pipeline_fill_and_bottlenecks(self):
        calculate = self.pipeline["ideal_pipeline_ms"]
        self.assertEqual(calculate(1, 4, 2, 8), (14, 14))
        self.assertEqual(calculate(3, 4, 2, 8), (42, 30))
        self.assertEqual(calculate(3, 12, 2, 8), (66, 46))
        self.assertEqual(calculate(3, 2, 12, 4), (54, 42))
        self.assertEqual(calculate(3, 2, 2, 2), (18, 10))

    def test_pipeline_rejects_invalid_times(self):
        calculate = self.pipeline["ideal_pipeline_ms"]
        for count in (0, -1, 1.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                calculate(count, 1, 1, 1)
        for duration in (0, -1, float("nan"), float("inf"), True, "1"):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                calculate(3, duration, 1, 1)


@unittest.skipIf(torch is None, "PyTorch unavailable; CPU tensor checks not executed")
class TrainingEngineeringTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous_threads)

    def test_adamw_lazy_state_bytes(self):
        weight = nn.Parameter(torch.ones(100))
        optimizer = torch.optim.AdamW([weight], foreach=False)

        def current_bytes():
            weights = weight.numel() * weight.element_size()
            gradients = 0 if weight.grad is None else weight.grad.numel() * weight.grad.element_size()
            state = sum(value.numel() * value.element_size()
                        for entry in optimizer.state.values() for value in entry.values()
                        if isinstance(value, torch.Tensor))
            return weights, gradients, state

        self.assertEqual(current_bytes(), (400, 0, 0))
        weight.square().mean().backward()
        self.assertEqual(current_bytes(), (400, 400, 0))
        optimizer.step()
        self.assertEqual(current_bytes(), (400, 400, 804))
        optimizer.zero_grad(set_to_none=True)
        self.assertEqual(current_bytes(), (400, 0, 804))

    def test_documented_device_step_cpu_path(self):
        namespace = {"nn": nn}
        for block in blocks(CHAPTERS[1]):
            if "def device_step(" in block:
                exec(block, namespace)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(24)
            model = nn.Linear(2, 2)
            reference = copy.deepcopy(model)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
            features = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
            labels = torch.tensor([1, 0])
            returned = namespace["device_step"](model, optimizer, features, labels)
            loss = nn.functional.cross_entropy(reference(features), labels)
            loss.backward()
            reference_optimizer.step()
            torch.testing.assert_close(returned, loss.detach())
            self.assertFalse(returned.requires_grad)
            self.assertEqual(returned.device.type, "cpu")
            for parameter, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(parameter, expected)

    def test_scalar_momentum_agrees_with_torch(self):
        weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        optimizer = torch.optim.SGD([weight], lr=0.1, momentum=0.9)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            weight.backward()
            optimizer.step()
        self.assertAlmostEqual(weight.item(), 0.71)

    def test_activation_checkpoint_matches_dropout_gradients(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(12)
            block = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(0.5), nn.Linear(8, 2))
            copied = copy.deepcopy(block)
            features = torch.randn(3, 4, requires_grad=True)
            copied_features = features.detach().clone().requires_grad_()
            before_forward = torch.get_rng_state()
            loss = block(features).square().sum()
            loss.backward()
            after_backward = torch.get_rng_state()
            torch.set_rng_state(before_forward)
            copied_loss = checkpoint(copied, copied_features, use_reentrant=False,
                                     preserve_rng_state=True).square().sum()
            copied_loss.backward()
            torch.testing.assert_close(loss, copied_loss)
            torch.testing.assert_close(features.grad, copied_features.grad)
            for parameter, expected in zip(copied.parameters(), block.parameters()):
                torch.testing.assert_close(parameter.grad, expected.grad)
            self.assertTrue(torch.equal(after_backward, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
