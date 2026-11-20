import itertools
import math
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


def margin_loss(score_difference, margin):
    shifted = margin - score_difference
    return max(shifted, 0.0) + math.log1p(math.exp(-abs(shifted)))


def section(source, anchor):
    match = re.search(
        rf"^## [^\n]*\{{#{re.escape(anchor)}\}}\n(.*?)(?=^## |\Z)",
        source,
        re.M | re.S,
    )
    if match is None:
        raise AssertionError(f"Missing section: {anchor}")
    return match.group(1)


class RewardGeneralizationReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.margin_sections = []
        cls.ranking_sections = []
        for suffix in (".md", ".en.md"):
            preference_source = ROOT / ("05-post-training/where-preferences-come-from" + suffix)
            reward_source = ROOT / ("05-post-training/reward-signals" + suffix)
            cls.margin_sections.append(section(
                preference_source.read_text(encoding="utf-8"), "preference-strength-margin"
            ))
            cls.ranking_sections.append(section(
                reward_source.read_text(encoding="utf-8"), "ranking-versus-selection"
            ))

    def test_bilingual_margin_table_matches_loss_and_gradient(self):
        tables = [re.findall(
            r"^\| \$(.*?)\$ \| ([\d.]+) \| (-[\d.]+) \|$", content, re.M
        ) for content in self.margin_sections]
        self.assertEqual(tables[0], tables[1])
        self.assertEqual([row[0] for row in tables[0]], ["0", r"\ln3", r"\ln9"])
        margins = [0.0, math.log(3), math.log(9)]
        score_difference = math.log(3)
        for margin, (_, displayed_loss, displayed_gradient) in zip(margins, tables[0]):
            self.assertAlmostEqual(
                margin_loss(score_difference, margin), float(displayed_loss), places=6
            )
            gradient = -1 / (1 + math.exp(score_difference - margin))
            self.assertAlmostEqual(gradient, float(displayed_gradient))

    def test_margin_gradient_matches_finite_differences(self):
        for score_difference, margin in itertools.product((-4.0, 0.0, math.log(3), 5.0),
                                                          (0.0, math.log(3), math.log(9))):
            step = 1e-5
            numeric = (margin_loss(score_difference + step, margin)
                       - margin_loss(score_difference - step, margin)) / (2 * step)
            expected = -1 / (1 + math.exp(score_difference - margin))
            self.assertAlmostEqual(numeric, expected, places=9)

    def test_margin_is_not_a_regression_target(self):
        for margin in (0.0, math.log(3), math.log(9)):
            self.assertAlmostEqual(margin_loss(margin, margin), math.log(2))
            self.assertLess(margin_loss(margin + 1, margin), margin_loss(margin, margin))
            self.assertEqual(margin_loss(margin + 1000, margin), 0.0)
            self.assertEqual(margin_loss(margin - 1000, margin), 1000.0)

    def test_weight_matches_only_one_gradient(self):
        margin = math.log(3)
        for score_difference, weighted_expected, margin_expected in (
            (math.log(3), -0.5, -0.5),
            (0.0, -1.0, -0.75),
        ):
            self.assertAlmostEqual(-2 / (1 + math.exp(score_difference)), weighted_expected)
            self.assertAlmostEqual(
                -1 / (1 + math.exp(score_difference - margin)), margin_expected
            )

    def test_bilingual_rankings_reverse_metric_winner(self):
        tables = [re.findall(
            r"^\| ([A-E]) \| (\d+) \| (\d+) \|$", content, re.M
        ) for content in self.ranking_sections]
        self.assertEqual(tables[0], tables[1])
        self.assertEqual([row[0] for row in tables[0]], list("ABCDE"))
        reference_order = list("ABCDE")
        pairs = list(itertools.combinations(reference_order, 2))
        self.assertEqual(len(pairs), 10)
        for column, correct_pairs, winner in ((1, 9, "B"), (2, 4, "A")):
            scores = {row[0]: int(row[column]) for row in tables[0]}
            self.assertEqual(sum(scores[better] > scores[worse] for better, worse in pairs),
                             correct_pairs)
            self.assertEqual(max(scores, key=scores.get), winner)
            self.assertEqual(winner == "A", column == 2)
        for content in self.ranking_sections:
            self.assertIn("90%", content)
            self.assertIn("40%", content)

    def test_new_sections_link_primary_sources(self):
        for content in self.margin_sections:
            self.assertIn("https://arxiv.org/html/2307.09288v2", content)
        for content in self.ranking_sections:
            self.assertIn("https://aclanthology.org/2025.findings-naacl.96/", content)
            self.assertIn("https://arxiv.org/html/2305.20050v1#S2.SS1", content)


if __name__ == "__main__":
    unittest.main()
