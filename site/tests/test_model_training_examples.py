import math
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ModelTrainingExamplesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blocks = {}
        namespace = {}
        for chapter in ("gpt", "deepseek"):
            for suffix in (".md", ".en.md"):
                text = (ROOT / "00-foundations/model-families" / (chapter + suffix)).read_text()
                blocks = re.findall(r"```python\n(.*?)```", text, re.S)
                cls.blocks[chapter, suffix] = blocks
                if suffix == ".md":
                    for block in blocks:
                        exec(compile(block, chapter, "exec"), namespace)
        cls.compare = staticmethod(namespace["compare_runs"])
        cls.route = staticmethod(namespace["route_with_bias"])

    def test_bilingual_examples(self):
        for chapter in ("gpt", "deepseek"):
            self.assertEqual(self.blocks[chapter, ".md"], self.blocks[chapter, ".en.md"])

    def test_gains_can_hide_regressions(self):
        baseline = {"one": (True, 1), "two": (False, 1)}
        candidate = {"one": (False, 2), "two": (True, 2)}
        self.assertEqual(self.compare(baseline, candidate), (1, 1, 2))

    def test_protocol_rejects_missing_or_invalid_outcomes(self):
        for baseline, candidate in [({}, {}), ({"a": (True, 1)}, {"b": (True, 1)}),
                                    ({"a": (True, 1)}, {"a": (True, math.nan)})]:
            with self.assertRaises(ValueError):
                self.compare(baseline, candidate)

    def test_bias_changes_selection_not_selected_affinities(self):
        result = self.route([0.6, 0.3, 0.1], [0, 0, 0.4], 2)
        self.assertEqual([index for index, _ in result], [0, 2])
        self.assertAlmostEqual(result[0][1], 6 / 7)
        self.assertAlmostEqual(result[1][1], 1 / 7)
        self.assertEqual(self.route([0.6, 0.3, 0.1], [0, 0, 0], 1), [(0, 1)])

    def test_invalid_router_inputs(self):
        for args in [([], [], 1), ([0.5], [0], True), ([0.5], [math.inf], 1),
                     ([0.5], [0], 2), ([0], [0], 1)]:
            with self.assertRaises(ValueError):
                self.route(*args)


if __name__ == "__main__":
    unittest.main()
