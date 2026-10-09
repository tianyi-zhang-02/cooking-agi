import contextlib
import io
import math
from pathlib import Path
import random
import re
import tomllib
import unittest
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[2]
CHAPTERS = (
    "00-foundations/deep-dives/pretraining-pipeline",
    "00-foundations/deep-dives/precision-and-memory",
    "00-foundations/deep-dives/attention-kernels",
    "00-foundations/deep-dives/kv-cache-and-inference",
    "00-foundations/deep-dives/position-and-context",
    "00-foundations/deep-dives/latent-and-sparse-attention",
    "00-foundations/deep-dives/multi-token-prediction",
    "00-foundations/core/ffn-and-gates",
    "00-foundations/moe/fine-grained-and-shared",
    "00-foundations/moe/load-balancing",
    "00-foundations/moe/router",
    "05-post-training/parameter-efficient-tuning",
    "06-systems/distributed-training",
)


def snippets(chapter, suffix=".md"):
    return re.findall(r"```python\n(.*?)```", (ROOT / (chapter + suffix)).read_text(), re.S)


def namespace_for(chapter):
    namespace = {}
    for snippet in snippets(chapter):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(snippet, chapter, "exec"), namespace)
    return namespace


class TrainingExpansionTests(unittest.TestCase):
    def test_scrollable_tables_contain_accessible_math_layer(self):
        stylesheet = (ROOT / "site/static/style.css").read_text()
        rules = re.findall(r"\.article-body \.table-wrap\s*\{([^}]+)\}", stylesheet)
        self.assertTrue(any(re.search(r"position:\s*relative\s*;", rule) for rule in rules))

    def test_bilingual_examples_match_and_execute(self):
        for chapter in CHAPTERS:
            with self.subTest(chapter=chapter):
                self.assertEqual(snippets(chapter), snippets(chapter, ".en.md"))
                namespace_for(chapter)

    def test_each_chapter_is_named_in_navigation(self):
        nav = tomllib.loads((ROOT / "site/nav.toml").read_text())
        for chapter in CHAPTERS:
            path = Path(chapter + ".md")
            explicit = [section for section in nav["section"] if str(path) in section.get("include", [])]
            section = explicit[0] if explicit else next(
                section for section in nav["section"] if section["dir"] == str(path.parent)
            )
            self.assertIn(path.name, section["order"])
            self.assertEqual(len(nav["label"][str(path)]), 2)

    def test_local_links_exist_and_preserve_language(self):
        for chapter in CHAPTERS:
            for suffix in (".md", ".en.md"):
                source = ROOT / (chapter + suffix)
                for target in re.findall(r"\]\(([^)]+)\)", source.read_text()):
                    parsed = urlsplit(target)
                    if parsed.scheme or not parsed.path:
                        continue
                    resolved = (source.parent / unquote(parsed.path)).resolve()
                    self.assertTrue(resolved.exists(), f"{source}: {target}")
                    if suffix == ".en.md" and resolved.suffix == ".md" and resolved != ROOT / (chapter + ".md"):
                        self.assertTrue(resolved.name.endswith(".en.md"), f"{source}: {target}")

    def test_streaming_attention_matches_dense_for_blocks_and_offsets(self):
        attention = namespace_for("00-foundations/deep-dives/attention-kernels")["streaming_attention"]
        generator = random.Random(17)
        scores = [generator.uniform(-12, 12) for _ in range(17)]
        values = [generator.uniform(-5, 5) for _ in range(17)]
        weights = [math.exp(score - max(scores)) for score in scores]
        expected = sum(weight * value for weight, value in zip(weights, values)) / sum(weights)
        for block_size in (1, 2, 5, 17, 30):
            for offset in (-1000, 0, 1000):
                actual = attention([score + offset for score in scores], values, block_size)
                self.assertAlmostEqual(actual, expected, places=10)

    def test_unscale_before_clipping_preserves_the_threshold(self):
        clip = namespace_for("00-foundations/deep-dives/precision-and-memory")["clip_scalar"]
        for gradient in (-2.0, -0.25, 0.0, 0.25, 2.0):
            for scale in (1, 8, 1024):
                self.assertEqual(clip(gradient * scale / scale, 0.5), clip(gradient, 0.5))
        self.assertEqual(clip(0.25 * 8, 0.5) / 8, 0.0625)
        self.assertNotEqual(clip(0.25 * 8, 0.5) / 8, clip(0.25, 0.5))

    def test_activation_tensor_sizes_and_length_scaling(self):
        tensor_mib = namespace_for("00-foundations/deep-dives/precision-and-memory")["tensor_mib"]
        self.assertEqual(tensor_mib((2, 2048, 1024)), 8)
        self.assertEqual(tensor_mib((2, 4096, 1024)), 16)
        self.assertEqual(tensor_mib((2, 16, 2048, 2048)), 256)
        self.assertEqual(tensor_mib((2, 16, 4096, 4096)), 1024)
        self.assertEqual(tensor_mib((2, 2048, 1024), 4), 16)

    def test_gqa_cache_fraction_and_batch_scaling(self):
        kv_bytes = namespace_for("00-foundations/deep-dives/kv-cache-and-inference")["kv_bytes"]
        mha = kv_bytes(1, 32, 8192, 32, 128)
        gqa = kv_bytes(1, 32, 8192, 8, 128)
        mqa = kv_bytes(1, 32, 8192, 1, 128)
        self.assertEqual(gqa, 2**30)
        self.assertEqual(gqa / mha, 1 / 4)
        self.assertEqual(mqa / mha, 1 / 32)
        self.assertEqual(kv_bytes(8, 32, 8192, 8, 128), 8 * gqa)

    def test_streaming_attention_masked_blocks_and_invalid_rows(self):
        attention = namespace_for("00-foundations/deep-dives/attention-kernels")["streaming_attention"]
        self.assertEqual(attention([-math.inf, -math.inf, 0], [999, 999, 7], 2), 7)
        for scores, values, block_size in (([], [], 1), ([-math.inf], [1], 1), ([0], [], 1), ([0], [1], 0)):
            with self.assertRaises(ValueError):
                attention(scores, values, block_size)

    def test_rope_preserves_norm_and_relative_dot(self):
        namespace = namespace_for("00-foundations/deep-dives/position-and-context")
        rotate, dot = namespace["rotate_pair"], namespace["dot"]
        query, key = [1.3, -0.8], [2.1, 0.4]
        for frequency in (1, 0.01, 0.0001):
            for first, second in ((0, 1), (8, 30), (100, 500)):
                rotated_query = rotate(query, first * frequency)
                rotated_key = rotate(key, second * frequency)
                self.assertAlmostEqual(dot(rotated_query, rotated_query), dot(query, query))
                self.assertAlmostEqual(dot(rotated_query, rotated_key),
                                       dot(query, rotate(key, (second - first) * frequency)))

    def test_mtp_never_crosses_boundary_or_tail(self):
        pairs = namespace_for("00-foundations/deep-dives/multi-token-prediction")["future_pairs"]
        self.assertEqual(pairs(list("ABCDE"), [0, 0, 1, 1, 1], 1),
                         [("A", "B"), ("C", "D"), ("D", "E")])
        self.assertEqual(pairs(list("ABCDE"), [0, 0, 1, 1, 1], 2), [("C", "E")])
        self.assertEqual(pairs(list("ABC"), [0, 0, 0], 10), [])
        with self.assertRaises(ValueError):
            pairs(list("AB"), [0, 0], 0)
        with self.assertRaises(ValueError):
            pairs(list("AB"), [0], 1)

    def test_swiglu_gate_gradient_and_matched_budget(self):
        silu = namespace_for("00-foundations/core/ffn-and-gates")["silu"]
        step = 1e-6
        numerical = (3 * silu(step) - 3 * silu(-step)) / (2 * step)
        self.assertAlmostEqual(numerical, 1.5, places=6)
        self.assertEqual(2 * 12 * 48, 3 * 12 * 32)
        self.assertEqual(2 * 12 * 48, 1152)

    def test_peft_parameter_accounting(self):
        layers, width, prompt_length, rank = 6, 512, 16, 8
        self.assertEqual(prompt_length * width, 8192)
        self.assertEqual(2 * layers * prompt_length * width, 98304)
        self.assertEqual(layers * 2 * width * rank, 49152)

    def test_global_token_gradient_not_average_of_local_means(self):
        rank_gradients = [[3.0, 5.0], [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]]
        world_size = len(rank_gradients)
        global_tokens = sum(map(len, rank_gradients))
        expected = sum(map(sum, rank_gradients)) / global_tokens
        local_scaled = [world_size * sum(gradients) / global_tokens for gradients in rank_gradients]
        self.assertEqual(sum(local_scaled) / world_size, expected)
        self.assertNotEqual(sum(sum(gradients) / len(gradients) for gradients in rank_gradients) / world_size,
                            expected)

    def test_state_sharding_tp_and_pipeline_arithmetic(self):
        self.assertEqual(2 + 2 + 12 / 4, 7)
        self.assertEqual(2 + 2 / 4 + 12 / 4, 5.5)
        self.assertEqual((2 + 2 + 12) / 4, 4)
        self.assertEqual([1 + 4, 3 + 8], [5, 11])
        self.assertEqual(4 / (4 + 2 - 1), 0.8)

    def test_toy_cache_and_sparse_counterexample(self):
        self.assertEqual(2 * 8 * 64, 1024)
        self.assertEqual(128 + 32, 160)
        values = [0, 8, 0, 0, 0, 0, 0, 0]
        self.assertEqual(sum(values) / len(values), 1)
        selected = [values[position] for position in (0, 4, 5, 6, 7)]
        self.assertEqual(sum(selected) / len(selected), 0)
        self.assertEqual(math.ceil(5 / 4) * 4 + math.ceil(3 / 4) * 4 - 8, 4)

    def test_fine_grained_experts_preserve_only_the_declared_budget(self):
        budget = namespace_for("00-foundations/moe/fine-grained-and-shared")["expert_budget"]
        baseline = budget(128, 256, 8, 2)
        self.assertEqual(baseline, budget(128, 64, 32, 8))
        self.assertEqual(baseline, budget(128, 64, 31, 7, shared=1))
        self.assertEqual(budget(128, 64, 32, 8, shared=1)[1] / baseline[1], 1.125)
        with self.assertRaises(ValueError):
            budget(128, 64, 4, 5)

    def test_selection_bias_does_not_become_gate_weight(self):
        route = namespace_for("00-foundations/moe/load-balancing")["route_with_bias"]
        self.assertEqual(route([0.8, 0.7, 0.2], [0, 0, 0], 2)[0], [0, 1])
        chosen, weights = route([0.8, 0.7, 0.2], [-0.2, 0, 0.5], 2)
        self.assertEqual(chosen, [1, 2])
        self.assertAlmostEqual(weights[0], 7 / 9)
        self.assertAlmostEqual(sum(weights), 1)
        self.assertNotEqual(weights, [0.5, 0.5])
        self.assertEqual(route([0.8, 0.7, 0.2], [0.8, 1, 1.5], 2)[0], chosen)

    def test_causal_mask_is_invariant_to_future_scores_and_values(self):
        attention = namespace_for("00-foundations/deep-dives/latent-and-sparse-attention")["masked_scalar_attention"]
        for future_score in (-1000, 0, 20, 1000):
            for future_value in (-1e9, 0, 1e9):
                result = attention([0, 0, future_score], [2, 4, future_value], [True, True, False])
                self.assertEqual(result, 3)
        self.assertEqual(attention([0, 0, 0], [2, 4, 999], [True, True, False], False), 2)
        self.assertLess(attention([0, 0, 20], [2, 4, 999], [True, True, False], False), 0.001)
        with self.assertRaises(ValueError):
            attention([0], [1], [False])

    def test_yarn_toy_ramp_boundaries_and_identity(self):
        multiplier = namespace_for("00-foundations/deep-dives/position-and-context")["yarn_frequency_multiplier"]
        self.assertEqual([multiplier(rotations) for rotations in (0, 1, 2.5, 4, 100)],
                         [0.25, 0.25, 0.625, 1, 1])
        for rotations in (0, 0.5, 2.5, 100):
            self.assertEqual(multiplier(rotations, scale=1), 1)
        with self.assertRaises(ValueError):
            multiplier(1, low=4, high=1)

    def test_speculative_residual_restores_target_including_zero_support(self):
        correct = namespace_for("00-foundations/deep-dives/multi-token-prediction")["speculative_one_step_distribution"]
        cases = [([0.6, 0.4], [0.8, 0.2]), ([0.6, 0.4], [0.6, 0.4]),
                 ([1, 0], [0, 1]), ([0.2, 0.3, 0.5], [0, 0.4, 0.6])]
        for target, draft in cases:
            corrected, rejection = correct(target, draft)
            for actual, expected in zip(corrected, target):
                self.assertAlmostEqual(actual, expected)
            self.assertAlmostEqual(rejection, 1 - sum(min(first, second) for first, second in zip(target, draft)))
        self.assertEqual(correct([1, 0], [0, 1]), ([1, 0], 1))
        self.assertEqual(correct([0.6, 0.4], [0.6, 0.4])[1], 0)
        with self.assertRaises(ValueError):
            correct([0.6, 0.6], [0.5, 0.5])

    def test_nonlinear_projection_cannot_move_outside_average(self):
        values = [2, -2]
        self.assertEqual(max(0, sum(values) / 2), 0)
        self.assertEqual(sum(max(0, value) for value in values) / 2, 1)


if __name__ == "__main__":
    unittest.main()
