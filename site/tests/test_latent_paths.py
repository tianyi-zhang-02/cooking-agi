import math
from pathlib import Path
import re
import tomllib
import unittest

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[2]
LATENT = "00-foundations/moe/latent-moe"
ATTENTION = "00-foundations/deep-dives/latent-and-sparse-attention"


def blocks(chapter, suffix=".md"):
    return re.findall(r"```python\n(.*?)```", (ROOT / (chapter + suffix)).read_text(), re.S)


def function_from(chapter, name):
    namespace = {}
    snippet = next(block for block in blocks(chapter) if f"def {name}(" in block)
    exec(compile(snippet, chapter, "exec"), namespace)
    return namespace[name]


class LatentPathsTests(unittest.TestCase):
    def test_code_matches_and_runs_in_both_languages(self):
        for chapter in (LATENT, ATTENTION):
            self.assertEqual(blocks(chapter), blocks(chapter, ".en.md"))
            for suffix in (".md", ".en.md"):
                namespace = {}
                for snippet in blocks(chapter, suffix):
                    exec(compile(snippet, chapter + suffix, "exec"), namespace)

    def test_chapter_has_a_reading_order_and_bilingual_label(self):
        navigation = tomllib.loads((ROOT / "site/nav.toml").read_text())
        section = next(item for item in navigation["section"] if item["dir"] == "00-foundations/moe")
        position = section["order"].index("latent-moe.md")
        self.assertEqual(section["order"][position - 1:position + 2],
                         ["fine-grained-and-shared.md", "latent-moe.md", "systems.md"])
        self.assertEqual(len(navigation["label"][LATENT + ".md"]), 2)
        for suffix in (".md", ".en.md"):
            overview = (ROOT / ("00-foundations/moe/README" + suffix)).read_text()
            self.assertIn("latent-moe" + suffix, overview)
            article = (ROOT / (LATENT + suffix)).read_text()
            for anchor in ("paths", "budget", "kimi-k3", "quantile-balancing", "validation"):
                self.assertIn("{#" + anchor + "}", article)

    def test_figure_uses_existing_layout_and_localized_copy(self):
        for suffix in (".md", ".en.md"):
            article = (ROOT / (LATENT + suffix)).read_text()
            figure = re.search(r"<figure\b.*?</figure>", article, re.S).group(0)
            self.assertIn("worked-update--pairs", figure)
            self.assertIn("<ol>", figure)
            self.assertEqual(figure.count("<li>"), 2)
            self.assertEqual(bool(re.search(r"[\u4e00-\u9fff]", figure)), suffix == ".md")
            self.assertIn("128 → 64 → 128", figure)

    def test_narrow_width_savings_include_projection_cost(self):
        budget = function_from(LATENT, "latent_budget")
        base = budget(128, 128, 256, 8, 2, 32, 2)
        narrow = budget(128, 64, 256, 8, 2, 32, 2)
        more = budget(128, 64, 256, 16, 4, 32, 2)
        self.assertEqual(narrow["parameters"], base["parameters"] // 2 + 16384)
        self.assertEqual(narrow["mac_per_token"], base["mac_per_token"] // 2 + 16384)
        self.assertAlmostEqual(more["parameters"] / base["parameters"] - 1, 1 / 48)
        self.assertAlmostEqual(more["mac_per_token"] / base["mac_per_token"] - 1, 1 / 12)

    def test_slot_bytes_are_not_only_the_latent_width(self):
        budget = function_from(LATENT, "latent_budget")
        base = budget(128, 128, 256, 8, 2, 32, 2)
        more = budget(128, 64, 256, 16, 4, 32, 2)
        self.assertEqual(base["slot_bytes"], more["slot_bytes"])
        self.assertEqual(budget(128, 64, 256, 8, 2, 64, 2)["slot_bytes"], base["slot_bytes"])

    def test_budget_rejects_bad_counts(self):
        budget = function_from(LATENT, "latent_budget")
        arguments = [128, 64, 256, 8, 2, 32, 2]
        for index in range(len(arguments)):
            for invalid in (0, -1, 1.5, True):
                changed = arguments.copy()
                changed[index] = invalid
                with self.assertRaises(ValueError):
                    budget(*changed)
        for changed in ([128, 129, 256, 8, 2, 32, 2], [128, 64, 256, 8, 9, 32, 2]):
            with self.assertRaises(ValueError):
                budget(*changed)

    def test_fixed_cutoff_matches_target_without_ties(self):
        bias_for = function_from(LATENT, "fixed_cutoff_bias")
        margins = [.31, .16, .07, -.08, -.19, -.34]
        for target in range(1, len(margins)):
            bias = bias_for(margins, target)
            self.assertEqual(sum(value + bias > 0 for value in margins), target)

    def test_ties_and_changed_batch_break_exact_count(self):
        bias_for = function_from(LATENT, "fixed_cutoff_bias")
        tied = [1, 1, 1, 0]
        bias = bias_for(tied, 2)
        self.assertNotEqual(sum(value + bias > 0 for value in tied), 2)
        margins = [.31, .16, .07, -.08, -.19, -.34]
        bias = bias_for(margins, 2)
        self.assertNotEqual(sum(value + 1 + bias > 0 for value in margins), 2)

    def test_fixed_cutoff_validation(self):
        bias_for = function_from(LATENT, "fixed_cutoff_bias")
        for margins, target in (([], 1), ([1, math.nan], 1), ([1, math.inf], 1),
                                ([1, 2], 0), ([1, 2], 2), ([1, 2], True), ([1, 2], 1.5)):
            with self.assertRaises(ValueError):
                bias_for(margins, target)

    def test_bias_centering_preserves_ranking_not_mixture_weights(self):
        affinities = [.7, .6, .2]
        bias = [-.1, .2, .0]
        centered = [value - sum(bias) / len(bias) for value in bias]
        rank = lambda values: sorted(range(len(values)), key=values.__getitem__, reverse=True)
        selected = rank([score + shift for score, shift in zip(affinities, bias)])[:2]
        self.assertEqual(selected, rank([score + shift for score, shift in zip(affinities, centered)])[:2])
        raw_weight = affinities[selected[0]] / sum(affinities[index] for index in selected)
        biased_weight = (affinities[selected[0]] + bias[selected[0]]) / sum(
            affinities[index] + bias[index] for index in selected)
        self.assertAlmostEqual(raw_weight, 6 / 13)
        self.assertNotAlmostEqual(raw_weight, biased_weight)

    def test_rotation_does_not_commute_with_projection(self):
        project = function_from(ATTENTION, "project_two")
        rotate = function_from(ATTENTION, "quarter_turn")
        self.assertEqual(rotate(project([1, 1])), [-2, 1])
        self.assertEqual(project(rotate([1, 1])), [-1, 2])
        self.assertNotEqual(rotate(project([1, 1]))[0], project(rotate([1, 1]))[0])

    def test_content_and_position_share_one_softmax(self):
        scores = [(content + position) / math.sqrt(4)
                  for content, position in zip([1, 1], [0, 2])]
        weights = [math.exp(score) / sum(math.exp(value) for value in scores) for score in scores]
        self.assertAlmostEqual(weights[0], .2689414213699951)
        self.assertAlmostEqual(weights[1], .7310585786300049)

    @unittest.skipIf(torch is None, "PyTorch unavailable")
    def test_normalization_cannot_move_across_mixture(self):
        experts = torch.tensor([[2., 0.], [0., 1.]], dtype=torch.float64)
        rms = lambda value: value / value.square().mean(dim=-1, keepdim=True).sqrt()
        after = rms(experts.mean(dim=0))
        before = rms(experts).mean(dim=0)
        torch.testing.assert_close(after, torch.tensor([1.264911064, .632455532], dtype=torch.float64))
        torch.testing.assert_close(before, torch.tensor([math.sqrt(.5), math.sqrt(.5)], dtype=torch.float64))
        self.assertFalse(torch.allclose(after, before))

    @unittest.skipIf(torch is None, "PyTorch unavailable")
    def test_decoupled_mla_outputs_and_gradients_match_expansion(self):
        generator = torch.Generator().manual_seed(42)
        shapes = [(2, 3), (2, 2), (4, 2), (4, 2), (2, 3, 2), (2, 3, 2)]
        inputs = [torch.randn(shape, generator=generator, dtype=torch.float64, requires_grad=True)
                  for shape in shapes]
        query_content, query_position, key_position, latents, key_up, value_up = inputs
        keys = torch.einsum("hcl,jl->hjc", key_up, latents)
        values = torch.einsum("hvl,jl->hjv", value_up, latents)
        positional = query_position @ key_position.T
        expanded_scores = (torch.einsum("hc,hjc->hj", query_content, keys) + positional) / math.sqrt(5)
        absorbed_query = torch.einsum("hc,hcl->hl", query_content, key_up)
        latent_scores = (absorbed_query @ latents.T + positional) / math.sqrt(5)
        visible = torch.tensor([True, True, True, False])
        expanded_weights = torch.softmax(expanded_scores.masked_fill(~visible, -torch.inf), dim=-1)
        latent_weights = torch.softmax(latent_scores.masked_fill(~visible, -torch.inf), dim=-1)
        expanded_output = torch.einsum("hj,hjv->hv", expanded_weights, values)
        latent_output = torch.einsum("hvl,hl->hv", value_up, latent_weights @ latents)
        torch.testing.assert_close(expanded_output, latent_output)
        expanded_grads = torch.autograd.grad(expanded_output.square().sum(), inputs, retain_graph=True)
        latent_grads = torch.autograd.grad(latent_output.square().sum(), inputs)
        for first, second in zip(expanded_grads, latent_grads):
            torch.testing.assert_close(first, second)
        self.assertEqual(torch.count_nonzero(latent_grads[2][-1]).item(), 0)
        self.assertEqual(torch.count_nonzero(latent_grads[3][-1]).item(), 0)


if __name__ == "__main__":
    unittest.main()
