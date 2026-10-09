import contextlib
import io
import math
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ROOT / "00-foundations/model-families"


def example_namespace(name, suffix=".md"):
    source = FAMILIES / (name + suffix)
    snippets = re.findall(r"```python\n(.*?)```", source.read_text(), re.S)
    namespace = {}
    for snippet in snippets:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(snippet, str(source), "exec"), namespace)
    return namespace


class ModelFamilyTests(unittest.TestCase):
    def test_bilingual_examples_match_and_run(self):
        for name in ("gpt", "llama"):
            with self.subTest(name=name):
                snippets = [
                    re.findall(r"```python\n(.*?)```", (FAMILIES / (name + suffix)).read_text(), re.S)
                    for suffix in (".md", ".en.md")
                ]
                self.assertTrue(snippets[0])
                self.assertEqual(snippets[0], snippets[1])
                example_namespace(name)
                example_namespace(name, ".en.md")

    def test_sink_mass_is_not_renormalized_away(self):
        calculate = example_namespace("gpt")["weights_with_sink"]
        weights, sink = calculate([math.log(3), 0], math.log(4))
        self.assertAlmostEqual(sum(weights) + sink, 1)
        self.assertAlmostEqual(sum(weights), 0.5)
        raw = sum(weight * value for weight, value in zip(weights, [10, 2]))
        self.assertAlmostEqual(raw, 4)
        self.assertAlmostEqual(raw / sum(weights), 8)

    def test_sink_softmax_is_stable_under_large_common_shifts(self):
        calculate = example_namespace("gpt")["weights_with_sink"]
        for shift in (-10000, 0, 10000):
            weights, sink = calculate([shift + math.log(3), shift], shift + math.log(4))
            self.assertAlmostEqual(weights[0], 0.375)
            self.assertAlmostEqual(weights[1], 0.125)
            self.assertAlmostEqual(sink, 0.5)

    def test_sink_can_absorb_mass_without_nan(self):
        calculate = example_namespace("gpt")["weights_with_sink"]
        weights, sink = calculate([-math.inf, -math.inf], 0)
        self.assertEqual(weights, [0, 0])
        self.assertEqual(sink, 1)
        weights, sink = calculate([0, 0], 1000)
        self.assertEqual(sum(weights), 0)
        self.assertEqual(sink, 1)

    def test_kv_cache_accounting(self):
        calculate = example_namespace("llama")["kv_bytes"]
        self.assertEqual(calculate(1, 32, 8192, 8, 128, 2), 2**30)
        self.assertEqual(calculate(1, 32, 8192, 32, 128, 2), 4 * 2**30)
        self.assertEqual(calculate(1, 32, 131072, 8, 128, 2), 16 * 2**30)
        self.assertEqual(calculate(4, 32, 8192, 8, 128, 2), 4 * 2**30)
        self.assertEqual(calculate(1, 32, 0, 8, 128, 2), 0)
        self.assertEqual(calculate(1, 32, 8192, 8, 128, 1), 2**29)

    def test_dpo_relative_improvement_can_lower_preferred_probability(self):
        reference_chosen, reference_rejected = 0.1, 0.1
        chosen, rejected = 0.08, 0.02
        margin = math.log(chosen / reference_chosen) - math.log(rejected / reference_rejected)
        loss = math.log1p(math.exp(-margin))
        self.assertLess(chosen, reference_chosen)
        self.assertAlmostEqual(margin, math.log(4))
        self.assertLess(loss, math.log(2))
        for suffix in (".md", ".en.md"):
            text = (FAMILIES / ("llama" + suffix)).read_text()
            self.assertIn(f"{loss:.4f}", text)
            self.assertIn(f"{math.log(2):.4f}", text)


if __name__ == "__main__":
    unittest.main()
