import contextlib
import importlib.util
import io
import math
from pathlib import Path
import re
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("mlm_contracts", ROOT / "00-foundations/code/mlm_contracts.py")
MLM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MLM)


class MlmContractsTests(unittest.TestCase):
    def test_action_boundaries(self):
        for draw, expected in [(0, "mask"), (0.7999, "mask"), (0.8, "random"),
                               (0.8999, "random"), (0.9, "keep"), (0.9999, "keep")]:
            self.assertEqual(MLM.corruption_action(draw), expected)

    def test_invalid_draws(self):
        for draw in (-0.1, 1, math.inf, math.nan, True):
            with self.assertRaises(ValueError):
                MLM.corruption_action(draw)

    def test_demo_positions_and_masks(self):
        inputs, labels, attention, _ = MLM.demo_data()
        self.assertEqual(inputs, [1, 4, 5, 6, 3, 2, 0])
        self.assertEqual(labels, [-100, -100, 5, -100, 7, -100, -100])
        self.assertEqual(attention, [1, 1, 1, 1, 1, 1, 0])
        self.assertEqual(sum(label != MLM.IGNORE for label in labels), 2)
        self.assertEqual(inputs.count(3), 1)

    def test_random_replacement_keeps_original_target(self):
        inputs, labels, _ = MLM.build_example([1, 4, 5, 2], {2: 8})
        self.assertEqual(inputs[2], 8)
        self.assertEqual(labels[2], 5)

    def test_inputs_are_not_mutated(self):
        original = [1, 4, 5, 2]
        replacements = {2: 3}
        MLM.build_example(original, replacements)
        self.assertEqual(original, [1, 4, 5, 2])
        self.assertEqual(replacements, {2: 3})

    def test_special_and_padding_positions_are_excluded(self):
        for position in (0, 3, 4):
            with self.assertRaises(ValueError):
                MLM.build_example([1, 4, 5, 2, 0], {position: 3})

    def test_nondefault_special_ids(self):
        inputs, labels, attention = MLM.build_example([7, 0, 9], {1: 8}, pad_id=9, special_ids=(7,))
        self.assertEqual(inputs, [7, 8, 9])
        self.assertEqual(labels, [-100, 0, -100])
        self.assertEqual(attention, [1, 1, 0])

    def test_invalid_token_ids(self):
        for tokens in ([], [False], [-1], [2.5]):
            with self.assertRaises(ValueError):
                MLM.build_example(tokens, {})

    def test_special_id_validation_precedes_set_deduplication(self):
        for settings in ({"pad_id": False}, {"special_ids": (1, True)},
                         {"special_ids": (2, 2.0)}, {"pad_id": -1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                MLM.build_example([1, 4, 2, 0], {}, **settings)

    def test_invalid_selections(self):
        for replacements in ({-1: 3}, {3: 3}, {True: 3}, {1: -1}, {1: False}):
            with self.assertRaises(ValueError):
                MLM.build_example([1, 4, 2], replacements)

    def test_selected_mean_matches_hand_calculation(self):
        _, labels, _, logits = MLM.demo_data()
        expected = (-math.log(0.8) - math.log(0.25)) / 2
        self.assertAlmostEqual(MLM.masked_cross_entropy(logits, labels), expected)
        self.assertAlmostEqual(expected, 0.8047189562170501)

    def test_extra_padding_does_not_dilute_loss(self):
        _, labels, _, logits = MLM.demo_data()
        expected = MLM.masked_cross_entropy(logits, labels)
        self.assertAlmostEqual(MLM.masked_cross_entropy(logits + [[9000.0] * 9] * 12,
                                                       labels + [MLM.IGNORE] * 12), expected)

    def test_ignored_logits_do_not_affect_loss(self):
        _, labels, _, logits = MLM.demo_data()
        expected = MLM.masked_cross_entropy(logits, labels)
        logits[0] = [-1000, 1000] + [0] * 7
        self.assertEqual(MLM.masked_cross_entropy(logits, labels), expected)

    def test_large_shared_shift_is_stable(self):
        _, labels, _, logits = MLM.demo_data()
        expected = MLM.masked_cross_entropy(logits, labels)
        shifted = [[value + 10000 for value in row] for row in logits]
        self.assertAlmostEqual(MLM.masked_cross_entropy(shifted, labels), expected)

    def test_equal_extreme_logits_keep_log_vocabulary_loss(self):
        for magnitude in (1e20, -1e20):
            for vocabulary in (2, 9):
                self.assertAlmostEqual(
                    MLM.masked_cross_entropy([[magnitude] * vocabulary], [0]),
                    math.log(vocabulary),
                )

    def test_raising_target_logit_reduces_loss(self):
        _, labels, _, logits = MLM.demo_data()
        before = MLM.masked_cross_entropy(logits, labels)
        logits[4][labels[4]] += 0.3
        self.assertLess(MLM.masked_cross_entropy(logits, labels), before)

    def test_logit_gradients_match_probabilities_at_all_positions(self):
        _, labels, _, logits = MLM.demo_data()
        epsilon = 1e-5
        selected_count = sum(target != MLM.IGNORE for target in labels)
        for position, row in enumerate(logits):
            weights = [math.exp(value - max(row)) for value in row]
            for column, weight in enumerate(weights):
                plus = [values.copy() for values in logits]
                minus = [values.copy() for values in logits]
                plus[position][column] += epsilon
                minus[position][column] -= epsilon
                derivative = (MLM.masked_cross_entropy(plus, labels)
                              - MLM.masked_cross_entropy(minus, labels)) / (2 * epsilon)
                expected = (weight / sum(weights) - int(column == labels[position])) / selected_count
                if labels[position] == MLM.IGNORE:
                    expected = 0
                self.assertAlmostEqual(derivative, expected, places=7)

    def test_no_selected_positions_is_explicit(self):
        with self.assertRaises(ValueError):
            MLM.masked_cross_entropy([[0, 0]], [MLM.IGNORE])

    def test_loss_validation(self):
        cases = [([], []), ([[]], [0]), ([[0, 1]], []), ([[0], [0, 1]], [0, 0]),
                 ([[math.inf]], [0]), ([[math.nan]], [0]), ([[True]], [0]),
                 ([[0]], [1]), ([[0]], [-1]), ([[0]], [False])]
        for logits, labels in cases:
            with self.assertRaises(ValueError):
                MLM.masked_cross_entropy(logits, labels)

    def test_loss_gradient_uses_supervised_position_count(self):
        _, labels, _, logits = MLM.demo_data()
        epsilon = 1e-5
        for position, expected in ((2, (0.8 - 1) / 2), (0, 0)):
            column = labels[position] if position else 0
            plus = [row.copy() for row in logits]
            minus = [row.copy() for row in logits]
            plus[position][column] += epsilon
            minus[position][column] -= epsilon
            derivative = (MLM.masked_cross_entropy(plus, labels)
                          - MLM.masked_cross_entropy(minus, labels)) / (2 * epsilon)
            self.assertAlmostEqual(derivative, expected, places=7)

    def test_demo_matches_bilingual_articles(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            MLM.main()
        for suffix in ("", ".en"):
            source = (ROOT / f"00-foundations/core/bert{suffix}.md").read_text()
            self.assertIn(output.getvalue().strip(), re.findall(r"```text\n(.*?)\n```", source, re.S))

    def test_bert_follows_encoder_decoder_introduction(self):
        navigation = tomllib.loads((ROOT / "site/nav.toml").read_text())
        section = next(section for section in navigation["section"] if section["dir"] == "00-foundations/core")
        self.assertEqual(section["order"][-3:], ["vanilla-transformer.md", "bert.md", "decoder-only.md"])


if __name__ == "__main__":
    unittest.main()
