import contextlib
import io
import itertools
import math
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build

ROOT = Path(__file__).resolve().parents[2]


class LessonDepthTests(unittest.TestCase):
    def test_bilingual_examples_execute_and_match(self):
        lessons = (
            "01-data-and-feedback/feedback-to-objectives",
            "02-memory/memory-lifecycle",
            "04-search/dual-encoder",
            "04-search/hybrid-and-reranking",
            "07-evaluation/metric-robustness",
        )
        for lesson in lessons:
            versions = []
            for suffix in (".md", ".en.md"):
                source = (ROOT / (lesson + suffix)).read_text()
                blocks = re.findall(r"```python\n(.*?)\n```", source, re.S)
                self.assertTrue(blocks, lesson)
                versions.append(blocks)
                for block in blocks:
                    with contextlib.redirect_stdout(io.StringIO()):
                        exec(compile(block, lesson + suffix, "exec"), {})
            self.assertEqual(*versions, lesson)

    def test_reading_shortcuts_are_in_the_table_of_contents(self):
        cases = {
            "04-search/dual-encoder": {"training-step", "index-version"},
            "04-search/hybrid-and-reranking": {"rank-fusion"},
            "07-evaluation/metric-robustness": {"population-mix", "paired-comparison"},
            "07-evaluation/evaluation-stack": {"failure-location", "controlled-changes"},
        }
        for lesson, anchors in cases.items():
            for suffix in (".md", ".en.md"):
                markup, toc = build.render_markdown((ROOT / (lesson + suffix)).read_text())
                self.assertTrue(anchors <= {entry["id"] for entry in toc}, lesson)
                for anchor in anchors:
                    self.assertIn(f'id="{anchor}"', markup)

    def test_search_sequence_is_explicit(self):
        search = next(section for section in build.load_nav()["section"] if section["dir"] == "04-search")
        self.assertEqual(search["order"], ["README.md", "dual-encoder.md", "hybrid-and-reranking.md"])

    def test_bellman_cycle_and_residual_bound(self):
        value_first, value_second = 2.4, 3.2
        self.assertAlmostEqual(value_first, 1 + 0.5 * (0.5 * value_first + 0.5 * value_second))
        self.assertAlmostEqual(value_second, 2 + 0.5 * value_first)
        for discount, expected in ((0.9, 0.1), (0.99, 1)):
            self.assertAlmostEqual(0.01 / (1 - discount), expected)

    def test_td_and_critic_error_calculations(self):
        self.assertAlmostEqual(1 + 0.9 * 1.5, 2.35)
        self.assertAlmostEqual(1 + 0.9 ** 2, 1.81)
        self.assertAlmostEqual(1 + 0.9 ** 2 * 2, 2.62)
        self.assertAlmostEqual(0.9 * -3 - 2, -4.7)
        self.assertEqual(sum([1, 1, 1, 9]) / 4, 3)
        self.assertEqual((1 + 9) / 2, 5)

    def test_ppo_four_branches(self):
        for advantage, ratio, expected in ((2, 1.3, 2.4), (2, 0.7, 1.4), (-2, 0.7, -1.6), (-2, 1.3, -2.6)):
            clipped = max(0.8, min(1.2, ratio))
            self.assertAlmostEqual(min(ratio * advantage, clipped * advantage), expected)

    def test_max_and_min_bias_examples(self):
        errors = list(itertools.product((-1, 1), repeat=2))
        self.assertEqual(sum(max(pair) for pair in errors) / 4, 0.5)
        self.assertEqual(sum(min(pair) for pair in errors) / 4, -0.5)

    def test_sac_variational_identity(self):
        temperature = 0.5
        probabilities = [1 / (1 + math.exp(2)), math.exp(2) / (1 + math.exp(2))]
        entropy = -sum(probability * math.log(probability) for probability in probabilities)
        value = probabilities[1] + temperature * entropy
        self.assertAlmostEqual(value, temperature * math.log1p(math.exp(2)))
        self.assertAlmostEqual(value, 1.063464, places=5)

    def test_offline_examples(self):
        estimate = (90 * (0.5 / 0.9) + 10 * (0.5 / 0.1) * 3) / 100
        self.assertEqual(estimate, 2)
        self.assertAlmostEqual(math.log(math.exp(1) + math.exp(5)) - 1, 4.01815, places=4)
        self.assertAlmostEqual(math.log(math.exp(1) + 1) - 1, 0.313262, places=5)
        self.assertAlmostEqual(0.8 * (4 - 3.2), 0.2 * 3.2)

    def test_control_shaping_and_preference_calculations(self):
        self.assertAlmostEqual(4 + 1.44 + 0.64 + 0.16 + 0.16, 1.6 * 4)
        potentials = [-3, -2, -1, 0]
        shaped = [0.9 * following - current for current, following in zip(potentials, potentials[1:])]
        self.assertAlmostEqual(sum(0.9 ** step * reward for step, reward in enumerate(shaped)), 3)
        probability = 1 / (1 + math.exp(-1))
        self.assertAlmostEqual(-math.log(probability), 0.313262, places=5)
        self.assertAlmostEqual(1.2 * 0.8 * 1.2, 1.152)

    def test_motion_is_finite_and_optional(self):
        css = (ROOT / "site/static/deep-rl-lab.css").read_text()
        self.assertIn("animation: drl-read-step 1.1s ease-out 1", css)
        self.assertIn("@media (prefers-reduced-motion: reduce)", css)
        self.assertIn(".drl-sequence > span { animation: none; }", css)
        self.assertNotIn("drl-read-step infinite", css)


if __name__ == "__main__":
    unittest.main()
