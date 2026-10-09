import math
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
CHAPTERS = ("generalization", "activation-and-initialization", "optimizers")


def code_blocks(chapter, language=""):
    source = ROOT / "00-foundations" / "deep-dives" / f"{chapter}{language}.md"
    return re.findall(r"```python\n(.*?)```", source.read_text(), re.DOTALL)


class TrainingFundamentalsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = {}
        for chapter in CHAPTERS:
            for block in code_blocks(chapter):
                exec(compile(block, chapter, "exec"), cls.namespace)

    def test_bilingual_examples_match(self):
        for chapter in CHAPTERS:
            with self.subTest(chapter=chapter):
                self.assertTrue(code_blocks(chapter))
                self.assertEqual(code_blocks(chapter), code_blocks(chapter, ".en"))

    def test_l1_threshold_and_sign(self):
        solve = self.namespace["regularized_scalar"]
        for target, expected in ((0.4, 0.0), (0.5, 0.0), (2.0, 1.5), (-2.0, -1.5)):
            self.assertAlmostEqual(solve(target, 0.5, "l1"), expected)

    def test_l2_and_unregularized(self):
        solve = self.namespace["regularized_scalar"]
        self.assertAlmostEqual(solve(0.4, 0.5, "squared_l2"), 0.4 / 1.5)
        for kind in ("l1", "squared_l2"):
            self.assertEqual(solve(-2.0, 0.0, kind), -2.0)

    def test_regularized_minimizers(self):
        solve = self.namespace["regularized_scalar"]
        for target in (-2.0, -0.4, 0.0, 0.4, 2.0):
            for kind in ("l1", "squared_l2"):
                optimum = solve(target, 0.5, kind)
                def objective(value):
                    penalty = abs(value) if kind == "l1" else 0.5 * value ** 2
                    return 0.5 * (value - target) ** 2 + 0.5 * penalty
                for offset in (-0.1, -0.001, 0.001, 0.1):
                    self.assertLessEqual(objective(optimum), objective(optimum + offset))

    def test_invalid_regularization(self):
        solve = self.namespace["regularized_scalar"]
        for arguments in ((1, -1, "l1"), (math.nan, 1, "l1"), (1, math.inf, "l1"), (1, 1, "l2")):
            with self.assertRaises(ValueError):
                solve(*arguments)

    def test_confidently_wrong_cross_entropy(self):
        loss, gradient = self.namespace["categorical_loss_gradient"]([1000.0, 10.0, 5.0], 1)
        self.assertEqual(loss, 990.0)
        self.assertEqual(gradient, [1.0, -1.0, 0.0])
        correct_loss, correct_gradient = self.namespace["categorical_loss_gradient"]([1000.0, 10.0, 5.0], 0)
        self.assertEqual(correct_loss, 0.0)
        self.assertEqual(correct_gradient, [0.0, 0.0, 0.0])

    def test_cross_entropy_shift_and_gradient(self):
        calculate = self.namespace["categorical_loss_gradient"]
        logits = [-1.0, 2.0, 0.3]
        loss, gradient = calculate(logits, 0)
        shifted_loss, shifted_gradient = calculate([value + 1000 for value in logits], 0)
        self.assertAlmostEqual(loss, shifted_loss)
        self.assertAlmostEqual(sum(gradient), 0)
        for actual, shifted in zip(gradient, shifted_gradient):
            self.assertAlmostEqual(actual, shifted)
        step = 1e-5
        for index, actual in enumerate(gradient):
            upper, lower = logits.copy(), logits.copy()
            upper[index] += step
            lower[index] -= step
            numerical = (calculate(upper, 0)[0] - calculate(lower, 0)[0]) / (2 * step)
            self.assertAlmostEqual(actual, numerical, places=8)

    def test_invalid_cross_entropy(self):
        calculate = self.namespace["categorical_loss_gradient"]
        for arguments in (([], 0), ([math.inf], 0), ([math.nan], 0), ([1.0], -1), ([1.0], 1), ([1.0], True)):
            with self.assertRaises(ValueError):
                calculate(*arguments)

    def test_initialization_scales(self):
        scale = self.namespace["initialization_std"]
        self.assertEqual(scale(128, 128, "he"), 0.125)
        self.assertAlmostEqual(scale(128, 128, "xavier"), math.sqrt(1 / 128))
        self.assertAlmostEqual(scale(128, 128, "he", 1.0), math.sqrt(1 / 128))
        for arguments in ((0, 1, "he"), (1, 0, "he"), (1.5, 1, "he"), (True, 1, "he"), (1, 1, "unknown"), (1, 1, "he", -1)):
            with self.assertRaises(ValueError):
                scale(*arguments)

    def test_relu_second_moment_not_variance(self):
        activations = [max(0, value) for value in (-1.0, 1.0)]
        average = sum(activations) / 2
        second_moment = sum(value ** 2 for value in activations) / 2
        variance = second_moment - average ** 2
        self.assertEqual(second_moment, 0.5)
        self.assertEqual(variance, 0.25)

    def test_dropout_expectation_not_nonlinear_equivalence(self):
        outputs = [0.0, 8.0]
        self.assertEqual(sum(outputs) / 2, 4.0)
        self.assertEqual(sum(value ** 2 for value in outputs) / 2, 32.0)
        self.assertNotEqual(32.0, 4.0 ** 2)

    def test_quadratic_learning_rate_boundary(self):
        self.assertLess(abs((1 - 0.5) ** 10), 1)
        self.assertLess(abs((1 - 1.5) ** 10), 1)
        self.assertGreater(abs((1 - 2.1) ** 10), 1)

    def test_adam_first_step_and_state(self):
        update = self.namespace["adamw_scalar"]
        parameter, state = update(2.0, 2.0, (0.0, 0.0, 0))
        self.assertAlmostEqual(parameter, 1.9)
        self.assertAlmostEqual(state[0], 0.2)
        self.assertAlmostEqual(state[1], 0.004)
        self.assertEqual(state[2], 1)

    def test_adamw_decoupled_decay(self):
        update = self.namespace["adamw_scalar"]
        decoupled, _ = update(2.0, 0.0, (0.0, 0.0, 0), decay=0.2)
        coupled, _ = update(2.0, 0.2 * 2.0, (0.0, 0.0, 0))
        self.assertAlmostEqual(decoupled, 1.96)
        self.assertAlmostEqual(coupled, 1.9)
        self.assertGreater(abs(decoupled - coupled), 0.05)

    def test_adam_restore_requires_moments(self):
        update = self.namespace["adamw_scalar"]
        parameter, state = update(2.0, 2.0, (0.0, 0.0, 0))
        uninterrupted = update(parameter, -0.5, state)
        restored = update(float(parameter), -0.5, tuple(state))
        reset = update(parameter, -0.5, (0.0, 0.0, 0))
        self.assertEqual(uninterrupted, restored)
        self.assertNotAlmostEqual(uninterrupted[0], reset[0])

    def test_invalid_adam_state_and_settings(self):
        update = self.namespace["adamw_scalar"]
        for state in ((0, -1, 0), (0, 0, -1), (0, 0, True)):
            with self.assertRaises(ValueError):
                update(1, 1, state)
        for setting in ({"beta1": 1}, {"beta2": -1}, {"epsilon": 0}, {"decay": -1}, {"learning_rate": -1}):
            with self.assertRaises(ValueError):
                update(1, 1, (0, 0, 0), **setting)

    def test_gru_update_convention(self):
        self.assertAlmostEqual(0.75 * 0.8 + (1 - 0.75) * -0.2, 0.55)
        hidden, reset = [1, 2], [1, 0]
        before_linear = [sum(value * gate for value, gate in zip(hidden, reset))] * 2
        after_linear = [sum(hidden) * gate for gate in reset]
        self.assertEqual(before_linear, [1, 1])
        self.assertEqual(after_linear, [3, 0])
        self.assertNotEqual(before_linear, after_linear)


if __name__ == "__main__":
    unittest.main()
