from contextlib import redirect_stdout
import io
from itertools import product
import math
from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
NOTES = ROOT / "05-post-training/deep-rl"
sys.path.insert(0, str(ROOT / "site"))
sys.path.insert(0, str(NOTES / "code"))
import build
from rl_checks import gae


class LongHorizonExamplesTests(unittest.TestCase):
    def source(self, suffix=""):
        return (NOTES / f"llm-bridge{suffix}.md").read_text()

    def snippets(self, suffix=""):
        return re.findall(r"```python\n(.*?)```", self.source(suffix), re.S)

    def execute(self, index):
        namespace = {}
        with redirect_stdout(io.StringIO()) as captured:
            exec(compile(self.snippets()[index], "long-horizon-example", "exec"), namespace)
        return namespace, captured.getvalue()

    def test_bilingual_examples_are_identical_and_executable(self):
        self.assertEqual(self.snippets(), self.snippets(".en"))
        self.assertEqual(len(self.snippets()), 2)
        for index in range(2):
            self.execute(index)

    def test_group_probability_matches_exhaustive_bernoulli_outcomes(self):
        calculate = self.execute(0)[0]["mixed_group_probability"]
        for group_size in range(1, 9):
            for probability in [0, 0.02, 0.2, 0.5, 0.98, 1]:
                enumerated = math.fsum(
                    probability ** sum(outcomes)
                    * (1 - probability) ** (group_size - sum(outcomes))
                    for outcomes in product([0, 1], repeat=group_size)
                    if 0 < sum(outcomes) < group_size
                )
                self.assertAlmostEqual(calculate(probability, group_size), enumerated)

    def test_displayed_group_numbers_match_code(self):
        output = self.execute(0)[1]
        self.assertEqual(output.splitlines(), ["2%: 14.9%", "20%: 83.2%", "98%: 14.9%"])
        for suffix in ["", ".en"]:
            self.assertIn("83.2%", self.source(suffix))
            self.assertIn("14.9%", self.source(suffix))

    def test_probability_rejects_invalid_inputs(self):
        calculate = self.execute(0)[0]["mixed_group_probability"]
        for probability in [-0.1, 1.1, math.nan, math.inf]:
            with self.assertRaises(ValueError):
                calculate(probability, 8)
        for size in [0, -1, 1.5, True, math.nan]:
            with self.assertRaises(ValueError):
                calculate(0.2, size)

    def test_probability_symmetry_and_group_size_one(self):
        calculate = self.execute(0)[0]["mixed_group_probability"]
        for probability in [0, 0.02, 0.2, 0.5, 0.98, 1]:
            self.assertAlmostEqual(calculate(probability, 1), 0)
            self.assertAlmostEqual(calculate(probability, 8), calculate(1 - probability, 8))

    def test_gae_matches_backward_hand_calculation(self):
        namespace, output = self.execute(1)
        self.assertEqual(output.splitlines(), ["0.5 [-0.3, 0.6, 0.2]", "1.0 [0.1, 0.7, 0.2]"])
        actual = gae(namespace["rewards"], namespace["values"], namespace["next_values"],
                     namespace["terminated"], namespace["truncated"], 1, 0.5)
        for calculated, expected in zip(actual, [-0.3, 0.6, 0.2]):
            self.assertAlmostEqual(calculated, expected)

    def test_lambda_one_recovers_complete_return_minus_value(self):
        values = [0.9, 0.3, 0.8]
        actual = gae([0, 0, 1], values, [0.3, 0.8, 0],
                     [False, False, True], [False] * 3, 1, 1)
        for calculated, value in zip(actual, values):
            self.assertAlmostEqual(calculated, 1 - value)

    def test_truncated_fragment_preserves_value_without_crossing_reset(self):
        actual = gae([0, 999], [0.6, 0], [0.8, 0],
                     [False, True], [True, False], 0.9, 0.5)
        self.assertAlmostEqual(actual[0], 0.12)
        terminal = gae([0], [0.6], [0.8], [True], [False], 0.9, 0.5)
        self.assertAlmostEqual(terminal[0], -0.6)

    def test_reduction_changes_task_weights_but_summing_fragments_does_not(self):
        tasks = [[1] * 2, [3] * 6]
        task_mean = math.fsum(math.fsum(task) / len(task) for task in tasks) / len(tasks)
        token_mean = math.fsum(map(math.fsum, tasks)) / sum(map(len, tasks))
        self.assertEqual(task_mean, 2)
        self.assertEqual(token_mean, 2.5)
        fragments = [[1], [1], [3] * 2, [3] * 4]
        recombined = math.fsum(map(math.fsum, fragments)) / sum(map(len, fragments))
        self.assertEqual(recombined, token_mean)

    def test_both_figures_are_localized_and_visible_without_expanding(self):
        for suffix, language in [("", "zh-CN"), (".en", "en")]:
            body, _ = build.render_markdown(self.source(suffix))
            figures = re.findall(r'<figure\b.*?</figure>', body, re.S)
            self.assertEqual(len(figures), 2)
            for figure in figures:
                self.assertIn(f'lang="{language}"', figure)
                self.assertEqual(figure.count("<li>"), 3)
                preceding = body[:body.index(figure)]
                self.assertEqual(preceding.count("<details"), preceding.count("</details>"))
                if suffix:
                    self.assertNotRegex(figure, r"[\u4e00-\u9fff]")
                else:
                    self.assertRegex(figure, r"[\u4e00-\u9fff]")
            for anchor in ["sparse-groups", "credit-example", "segment-boundary", "compare-designs"]:
                self.assertIn(f'id="{anchor}"', body)

    def test_sources_and_scope_remain_explicit(self):
        for suffix in ["", ".en"]:
            source = self.source(suffix)
            for url in ["https://z.ai/blog/glm-5.2", "1506.02438v6#S3", "2402.03300v3#S4.SS1.SSS3",
                        "gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/"]:
                self.assertIn(url, source)
            self.assertIn("post-training-infrastructure", source)
        self.assertIn("因果贡献", self.source())
        self.assertIn("causal contribution", self.source(".en"))


if __name__ == "__main__":
    unittest.main()
