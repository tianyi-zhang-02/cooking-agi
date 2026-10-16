from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
CHAPTER = "12-infra/quantization/formats-and-error"


def note_text(chapter, suffix):
    return (ROOT / (chapter + suffix)).read_text(encoding="utf-8")


def example_namespace(suffix):
    namespace = {}
    for block in re.findall(r"```python\n(.*?)```", note_text(CHAPTER, suffix), re.S):
        exec(compile(block, CHAPTER + suffix, "exec"), namespace)
    return namespace


def e4m3_positive_grid():
    grid = []
    for code in range(127):
        exponent, mantissa = divmod(code, 8)
        value = mantissa * 2 ** -9 if exponent == 0 else (1 + mantissa / 8) * 2 ** (exponent - 7)
        grid.append((code, value))
    return grid


def round_e4m3(value):
    _, rounded = min(
        e4m3_positive_grid(),
        key=lambda item: (abs(item[1] - abs(value)), item[0] % 2),
    )
    return -rounded if value < 0 else rounded


class QuantizationReviewTests(unittest.TestCase):
    def test_bilingual_examples_are_identical_and_execute(self):
        blocks = [
            re.findall(r"```python\n(.*?)```", note_text(CHAPTER, suffix), re.S)
            for suffix in (".md", ".en.md")
        ]
        self.assertTrue(blocks[0])
        self.assertEqual(*blocks)
        for suffix in (".md", ".en.md"):
            with self.subTest(suffix=suffix):
                self.assertIn("quantize_asym", example_namespace(suffix))

    def test_zero_point_is_representable_for_one_sided_and_constant_ranges(self):
        cases = ([1.0, 2.0], [-2.0, -1.0], [-2.0, 3.0], [2.0, 2.0], [-2.0], [0.0, 0.0])
        for suffix in (".md", ".en.md"):
            quantize = example_namespace(suffix)["quantize_asym"]
            for values in cases:
                for bits in (2, 4, 8):
                    with self.subTest(suffix=suffix, values=values, bits=bits):
                        scale, zero_point, codes, restored = quantize(values, bits)
                        maximum_code = 2 ** bits - 1
                        self.assertGreater(scale, 0)
                        self.assertIsInstance(zero_point, int)
                        self.assertTrue(0 <= zero_point <= maximum_code)
                        zero_code = max(0, min(maximum_code, round(0 / scale) + zero_point))
                        self.assertEqual(scale * (zero_code - zero_point), 0)
                        self.assertTrue(all(0 <= code <= maximum_code for code in codes))
                        self.assertEqual(quantize(values + [0.0], bits)[:2], (scale, zero_point))
                        for value, estimate in zip(values, restored):
                            self.assertLessEqual(abs(value - estimate), scale / 2 + 1e-12)

    def test_unextended_positive_range_reproduces_the_reported_failure(self):
        scale = (2 - 1) / 15
        zero_point = round(-1 / scale)
        zero_code = max(0, min(15, round(0 / scale) + zero_point))
        self.assertEqual(zero_point, -15)
        self.assertEqual(scale * (zero_code - zero_point), 1)
        for suffix in (".md", ".en.md"):
            self.assertEqual(example_namespace(suffix)["quantize_sym"]([0.0, 0.0])[2], [0.0, 0.0])

    def test_outlier_mse_uses_matching_populations(self):
        weights = [0.12, -0.53, 0.88, -0.07, 0.31, 1.40, -0.95, 0.02]
        outlier_weights = weights[:5] + [14.0] + weights[6:]
        for suffix in (".md", ".en.md"):
            namespace = example_namespace(suffix)
            original = namespace["quantize_sym"](weights)[2]
            changed = namespace["quantize_sym"](outlier_weights)[2]
            errors_before = [(value - estimate) ** 2 for value, estimate in zip(weights, original)]
            errors_after = [(value - estimate) ** 2 for value, estimate in zip(outlier_weights, changed)]
            self.assertAlmostEqual(errors_before[5], 0)
            self.assertAlmostEqual(errors_after[5], 0)
            self.assertAlmostEqual(sum(errors_before) / 8, 0.0042)
            self.assertAlmostEqual(sum(errors_after) / 8, 0.2592)
            self.assertAlmostEqual(sum(errors_before) / 7, 0.0048)
            self.assertAlmostEqual(sum(errors_after) / 7, 0.2962285714285714)
            self.assertAlmostEqual(sum(errors_after) / sum(errors_before), 61.71428571428571)
            for chapter in (CHAPTER, "12-infra/quantization/review"):
                source = note_text(chapter, suffix)
                for number in ("0.0048", "0.29623", "0.2592", "61.7"):
                    self.assertIn(number, source)
                self.assertNotRegex(source, r"seventy|七十")

    def test_e4m3_relative_bound_in_normal_unclipped_intervals(self):
        grid = e4m3_positive_grid()
        self.assertEqual(grid[1][1], 2 ** -9)
        self.assertEqual(grid[8][1], 2 ** -6)
        self.assertEqual(grid[-1][1], 448)
        for (_, lower), (_, upper) in zip(grid[8:-1], grid[9:]):
            for fraction in (0.25, 0.5, 0.75):
                for sign in (-1, 1):
                    value = sign * (lower + fraction * (upper - lower))
                    relative_error = abs(round_e4m3(value) - value) / abs(value)
                    self.assertLessEqual(relative_error, 1 / 16)

    def test_subnormal_underflow_and_clipping_are_not_bounded_by_six_percent(self):
        self.assertEqual(round_e4m3(0.001), 2 ** -9)
        self.assertAlmostEqual(abs(round_e4m3(0.001) - 0.001) / 0.001, 0.953125)
        self.assertEqual(round_e4m3(0.0001), 0)
        self.assertEqual(abs(round_e4m3(0.0001) - 0.0001) / 0.0001, 1)
        self.assertGreater(abs(round_e4m3(1000) - 1000) / 1000, 1 / 16)
        for suffix in (".md", ".en.md"):
            for chapter in (CHAPTER, "12-infra/quantization/review"):
                source = note_text(chapter, suffix)
                for qualifier in ("normal range", "subnormal", "95.3125%", "100%"):
                    self.assertIn(qualifier, source.lower())

    def test_kivi_layout_is_attributed_and_quarot_is_a_counterexample(self):
        for suffix in (".md", ".en.md"):
            for chapter in ("12-infra/quantization/fp8-and-kv-cache", "12-infra/quantization/review"):
                source = note_text(chapter, suffix)
                self.assertIn("KIVI", source)
                self.assertIn("QuaRot", source)
                self.assertIn("kernel", source)
                self.assertNotIn("keys must be quantized per channel", source)
                if suffix == ".en.md":
                    self.assertRegex(source, r"within each token and head")
                else:
                    self.assertIn("每个 token、每个 head 内", source)


if __name__ == "__main__":
    unittest.main()
