import contextlib
import importlib.util
import io
import itertools
import math
from pathlib import Path
import re
import tomllib
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "00-foundations/code/order_without_positions.py"
SPEC = importlib.util.spec_from_file_location("order_without_positions", SOURCE)
ORDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORDER)


class OrderWithoutPositionsTests(unittest.TestCase):
    def test_recurrence_example(self):
        self.assertEqual(ORDER.recurrent_trace([1, 2]), [1.0, 2.5])
        self.assertEqual(ORDER.recurrent_trace([2, 1]), [2.0, 2.0])

    def test_expansion_including_initial_state(self):
        for decay in (0, 0.2, 0.5, 1):
            for inputs in itertools.product((-2, 0, 1), repeat=4):
                initial = 0.75
                expected = decay ** len(inputs) * initial + sum(
                    decay ** (len(inputs) - position - 1) * value
                    for position, value in enumerate(inputs))
                self.assertAlmostEqual(ORDER.recurrent_trace(inputs, decay, initial)[-1], expected)

    def test_decay_boundaries(self):
        self.assertEqual(ORDER.recurrent_trace([1, 2], decay=0)[-1], 2)
        self.assertEqual(ORDER.recurrent_trace([1, 2], decay=1)[-1], 3)
        self.assertEqual(ORDER.recurrent_trace([2, 1], decay=1)[-1], 3)

    def test_common_final_input_preserves_a_difference(self):
        self.assertEqual(ORDER.recurrent_trace([1, 2, 0])[-1], 1.25)
        self.assertEqual(ORDER.recurrent_trace([2, 1, 0])[-1], 1)

    def test_split_execution_matches_uninterrupted(self):
        inputs = [1, -3, 0, 5, 2]
        expected = ORDER.recurrent_trace(inputs, initial=0.75)
        for split in range(len(inputs) + 1):
            prefix = ORDER.recurrent_trace(inputs[:split], initial=0.75)
            state = prefix[-1] if prefix else 0.75
            suffix = ORDER.recurrent_trace(inputs[split:], initial=state)
            self.assertEqual(prefix + suffix, expected)

    def test_independent_sequences_reset_state(self):
        state = ORDER.recurrent_trace([1, 2])[-1]
        self.assertNotEqual(ORDER.recurrent_trace([3], initial=state), ORDER.recurrent_trace([3]))

    def test_recurrence_rejects_invalid_values(self):
        for decay in (-0.1, 1.1, math.nan, math.inf, True):
            with self.assertRaises(ValueError):
                ORDER.recurrent_trace([1], decay=decay)
        for inputs in ([math.nan], [math.inf], [True], [1e308, 1e308]):
            with self.assertRaises(ValueError):
                ORDER.recurrent_trace(inputs, decay=1)
        with self.assertRaises(ValueError):
            ORDER.recurrent_trace([1], initial=math.inf)

    def test_empty_and_unmodified_inputs(self):
        inputs = [1, 2]
        ORDER.recurrent_trace(inputs)
        ORDER.prefix_means(inputs)
        self.assertEqual(inputs, [1, 2])
        self.assertEqual(ORDER.recurrent_trace([]), [])
        self.assertEqual(ORDER.prefix_means([]), [])

    def test_causal_means_one_and_two_layers(self):
        first = ORDER.prefix_means([1, 2, 0])
        second = ORDER.prefix_means([2, 1, 0])
        self.assertEqual(first, [1, 1.5, 1])
        self.assertEqual(second, [2, 1.5, 1])
        self.assertAlmostEqual(ORDER.prefix_means(first)[-1], 7 / 6)
        self.assertEqual(ORDER.prefix_means(second)[-1], 1.5)

    def test_global_mean_cannot_recover_order(self):
        first = [sum([1, 2, 0]) / 3] * 3
        second = [sum([2, 1, 0]) / 3] * 3
        self.assertEqual(first, second)
        self.assertEqual(ORDER.prefix_means(first), ORDER.prefix_means(second))

    def test_content_attention_permutation_equivariance(self):
        rows = [[1, 0], [0, 2], [-1, 1]]
        expected = ORDER.content_attention(rows)
        for permutation in itertools.permutations(range(len(rows))):
            actual = ORDER.content_attention([rows[position] for position in permutation])
            for result, position in zip(actual, permutation):
                for value, target in zip(result, expected[position]):
                    self.assertAlmostEqual(value, target)

    def test_causal_mask_breaks_general_permutation_equivariance(self):
        rows = [[1, 0], [0, 2], [-1, 1]]
        first = ORDER.content_attention(rows, causal=True)
        permuted = ORDER.content_attention([rows[1], rows[0], rows[2]], causal=True)
        self.assertGreater(sum(abs(left - right) for left, right in zip(first[1], permuted[0])), 0.1)
        for left, right in zip(first[-1], permuted[-1]):
            self.assertAlmostEqual(left, right)

    def test_attention_validation(self):
        for rows in ([], [[]], [[1], [2, 3]], [[math.nan]], [[math.inf]]):
            with self.assertRaises(ValueError):
                ORDER.content_attention(rows)
        with self.assertRaises(ValueError):
            ORDER.content_attention([[1]], causal="yes")
        with self.assertRaises(ValueError):
            ORDER.prefix_means([math.inf])

    def test_rotation_changes_scores(self):
        rotated_score = math.cos(math.pi / 2)
        self.assertAlmostEqual(rotated_score, 0)
        probability = math.exp(1) / (math.exp(1) + math.exp(rotated_score))
        self.assertAlmostEqual(probability, 0.7310585786300049)
        self.assertEqual(math.exp(1) / (math.exp(1) + math.exp(1)), 0.5)

    def test_channel_decay_example(self):
        actual = ORDER.channel_delta_step([[2], [10]], [1, 0], [6], [0.5, 0.9], 0.25)
        self.assertEqual(actual, [[2.25], [9]])
        scalar = ORDER.channel_delta_step([[2], [10]], [1, 0], [6], [0.5, 0.5], 0.25)
        self.assertEqual(scalar, [[2.25], [5]])

    def test_channel_transition_matches_factored_equation(self):
        state = [[2, -1], [10, 3]]
        key = [0.6, 0.8]
        value = [6, 4]
        decay = [0.5, 0.9]
        write = 0.25
        transition = [[(float(row == column) - write * key[row] * key[column]) * decay[column]
                       for column in range(2)] for row in range(2)]
        expected = [[sum(transition[row][inner] * state[inner][column] for inner in range(2))
                     + write * key[row] * value[column] for column in range(2)] for row in range(2)]
        actual = ORDER.channel_delta_step(state, key, value, decay, write)
        for result_row, expected_row in zip(actual, expected):
            for result, expected_entry in zip(result_row, expected_row):
                self.assertAlmostEqual(result, expected_entry)
        self.assertEqual(state, [[2, -1], [10, 3]])

    def test_channel_update_rejects_invalid_shapes_and_gates(self):
        cases = [([], [1], [2], [0.5], 1), ([[1, 2]], [1], [2], [0.5], 1),
                 ([[1]], [2], [2], [0.5], 1), ([[1]], [1], [2], [], 1),
                 ([[1]], [1], [2], [1.1], 1), ([[1]], [1], [2], [0.5], -1),
                 ([[math.nan]], [1], [2], [0.5], 1)]
        for args in cases:
            with self.assertRaises(ValueError):
                ORDER.channel_delta_step(*args)

    def test_three_step_expansion_preserves_matrix_order(self):
        def multiply(left, right):
            return [[sum(left[row][inner] * right[inner][column]
                         for inner in range(len(right)))
                     for column in range(len(right[0]))] for row in range(len(left))]

        parameters = [([1, 0], [2, -1], [0.5, 0.9], 0.4),
                      ([0.6, 0.8], [3, 4], [0.7, 0.3], 0.8),
                      ([0, 1], [-2, 1], [0.2, 0.9], 0.5)]
        state = [[0, 0], [0, 0]]
        transitions = []
        writes = []
        for key, value, decay, write in parameters:
            state = ORDER.channel_delta_step(state, key, value, decay, write)
            transitions.append([[(float(row == column) - write * key[row] * key[column])
                                 * decay[column] for column in range(2)] for row in range(2)])
            writes.append([[write * key[row] * value[column] for column in range(2)]
                           for row in range(2)])
        first_term = multiply(multiply(transitions[2], transitions[1]), writes[0])
        second_term = multiply(transitions[2], writes[1])
        reversed_term = multiply(multiply(transitions[1], transitions[2]), writes[0])
        self.assertNotEqual(first_term, reversed_term)
        for row in range(2):
            for column in range(2):
                self.assertAlmostEqual(state[row][column], first_term[row][column]
                                       + second_term[row][column] + writes[2][row][column])

    def test_decay_inverse_range(self):
        bf16_maximum = (2 - 2 ** -7) * 2 ** 127
        self.assertLess(math.exp(80), bf16_maximum)
        self.assertGreater(math.exp(100), bf16_maximum)
        self.assertAlmostEqual(math.exp(80) / 1e34, 5.54062238439351)

    def test_demo_matches_both_articles(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            ORDER.main()
        for suffix in ("", ".en"):
            source = ROOT / f"00-foundations/deep-dives/nope-and-order{suffix}.md"
            blocks = re.findall(r"```text\n(.*?)\n```", source.read_text(), re.S)
            self.assertIn(output.getvalue().strip(), blocks)

    def test_diagram_is_accessible_and_self_contained(self):
        root = ET.parse(ROOT / "00-foundations/assets/order-state.svg").getroot()
        namespace = {"svg": "http://www.w3.org/2000/svg"}
        self.assertTrue(root.find("svg:title", namespace).text)
        self.assertTrue(root.find("svg:desc", namespace).text)
        self.assertEqual(root.attrib["aria-labelledby"], "title desc")
        self.assertFalse(root.findall(".//svg:script", namespace))
        self.assertFalse(root.findall(".//svg:image", namespace))

    def test_navigation_places_order_after_position_basics(self):
        navigation = tomllib.loads((ROOT / "site/nav.toml").read_text())
        section = next(section for section in navigation["section"]
                       if section["dir"] == "learn/attention-systems")
        self.assertEqual(section["order"][:2], ["position-and-context.md", "nope-and-order.md"])
        self.assertEqual(section["include"].count("00-foundations/deep-dives/nope-and-order.md"), 1)

    def test_both_languages_link_to_primary_context(self):
        for suffix in ("", ".en"):
            source = (ROOT / f"00-foundations/deep-dives/nope-and-order{suffix}.md").read_text()
            self.assertIn("{#paper-context}", source)
            self.assertIn("https://arxiv.org/html/2203.16634v2", source)
            self.assertIn("https://arxiv.org/html/2510.26692v1#S5.SS2", source)
            self.assertIn("https://github.com/adihaviv/NoPos", source)
            self.assertNotIn("https://arxiv.org/html/2510.26692v1#S3)", source)


if __name__ == "__main__":
    unittest.main()
