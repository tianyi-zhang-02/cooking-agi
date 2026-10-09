import importlib.util
import contextlib
import io
from collections import Counter
import json
import math
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'site'))
import paritycheck


class ParityAcceptanceTests(unittest.TestCase):
    def test_single_line_and_multiline_display_equations(self):
        for source in ('$$a=b$$', '$$\na=b\n$$'):
            self.assertEqual(paritycheck.shape(source)['math'], 1)
        self.assertEqual(paritycheck.shape('$$a=b$$\n\n$$c=d$$')['math'], 2)

    def test_inline_and_escaped_delimiters_are_not_display_math(self):
        self.assertEqual(paritycheck.shape(r'$a=b$ and \$\$quoted\$\$')['math'], 0)

    def test_fenced_code_is_not_a_mathematical_expression(self):
        source = '```python\ntext = "$$not math$$"\n```\n\n$$a=b$$'
        self.assertEqual(paritycheck.shape(source)['math'], 1)
        self.assertEqual(paritycheck.shape(source)['code'], 1)

    def test_translated_concept_face_is_not_counted_twice(self):
        source = '$$a=b$$\n<div class="concept-face concept-en">$$a=b$$</div>'
        self.assertEqual(paritycheck.shape(source)['math'], 1)

    def test_bad_multiline_and_unclosed_math_is_reported(self):
        self.assertTrue(paritycheck.source_issues('$\na=b\n$'))
        self.assertTrue(paritycheck.source_issues('$$a=b'))
        self.assertFalse(paritycheck.source_issues('$$\na=b\n$$'))
        self.assertFalse(paritycheck.source_issues('```text\n$\nexample\n$\n```'))

    def test_attention_matrix_preserves_three_rows(self):
        import build
        for suffix in ('.md', '.en.md'):
            source = (ROOT / ('00-foundations/core/multi-head-attention' + suffix)).read_text()
            self.assertFalse(paritycheck.source_issues(source))
            unused_prose, stashed = build.protect_math(source)
            matrix = next(tex for token, tex in stashed if 'S+M=' in tex and r'\begin{bmatrix}' in tex)
            self.assertTrue(matrix.startswith('$$'))
            self.assertEqual(matrix.count('\\\\'), 2)

    def test_workflow_rejects_bilingual_structure_gaps(self):
        workflow = (ROOT / '.github/workflows/site.yml').read_text()
        self.assertIn('run: python site/paritycheck.py --strict', workflow)
        self.assertNotIn('continue-on-error: true', workflow)
        self.assertIn('pip install markdown pygments numpy', workflow)

    def test_published_moe_anchors_survive_rewording(self):
        for suffix, anchor in (('.md', '{#moe-token}'), ('.en.md', '{#moe-more-parameters-same-compute-per-token}')):
            source = (ROOT / ('00-foundations/transformer-lab' + suffix)).read_text()
            self.assertIn(anchor, source)

    def test_previous_section_links_still_have_targets(self):
        import build
        records = json.loads((ROOT / 'site/legacy-section-anchors.json').read_text())
        for record in records:
            with self.subTest(source=record['source']):
                raw = (ROOT / record['source']).read_text()
                raw = re.sub(r'^#\s+.+$', '', raw, count=1, flags=re.M)
                rendered, unused_toc = build.render_markdown(build.protect_mermaid(raw))
                identifiers = Counter(re.findall(r'\bid="([^"]+)"', rendered))
                for anchor in record['lost']:
                    self.assertEqual(identifiers[anchor], 1, anchor)

    def test_decoder_projection_formulas_use_row_vectors(self):
        for suffix in ('.md', '.en.md'):
            source = (ROOT / ('00-foundations/core/decoder-only' + suffix)).read_text()
            self.assertIn(r'\right]W_{\text{down}}', source)
            self.assertNotIn(r'W_{\text{down}}\left[', source)
            self.assertIn(r'z_t=h_tW_{\text{vocab}}', source)
            self.assertIn(r'\operatorname{ReLU}(xW_1)W_2', source)

    def test_backprop_formula_has_an_unambiguous_mean_denominator(self):
        formula = r'\frac{\sigma(z_2)-y}{N}'
        for suffix in ('.md', '.en.md'):
            source = (ROOT / ('00-foundations/hand-write-kit' + suffix)).read_text()
            self.assertIn(formula, source)
            self.assertIn('2(p-y)p(1-p)/N', source)

    @unittest.skipUnless(shutil.which('node'), 'Node is needed for diagram calculations')
    def test_rope_diagram_oscillates_at_fixed_content(self):
        script = (ROOT / 'site/static/tx-lab.js').read_text()
        theta = re.search(r'function theta\(i\) \{[^\n]+\}', script).group()
        score = re.search(r'function score\(d\) \{[^\n]+\}', script).group()
        program = 'var DIM=64,base=10000;' + theta + score
        program += 'process.stdout.write(JSON.stringify(Array.from({length:129},(_,position)=>score(position))));'
        actual = json.loads(subprocess.check_output([shutil.which('node'), '-e', program], text=True))
        expected = [sum(math.cos(distance * 10000 ** (-2 * pair / 64)) for pair in range(32)) / 32
                    for distance in range(129)]
        for observed, target in zip(actual, expected):
            self.assertAlmostEqual(observed, target, places=13)
        self.assertTrue(any(later > earlier for earlier, later in zip(actual, actual[1:])))

    def test_mla_comparison_is_a_fixed_latent_configuration(self):
        script = (ROOT / 'site/static/tx-lab.js').read_text()
        self.assertIn("label: 'MLA · 512 + 64'", script)
        self.assertNotIn('return 4.5 * c.d', script)
        self.assertIn('complementary sequence-wise auxiliary loss', script)
        for suffix in ('.md', '.en.md'):
            source = (ROOT / ('00-foundations/transformer-lab' + suffix)).read_text()
            self.assertIn('$d_c+d_r$', source)
            self.assertIn('$512+64=576$', source)


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'NumPy is needed for handwritten modules')
class HandwrittenModuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy
        cls.numpy = numpy
        spec = importlib.util.spec_from_file_location('interview_kit', ROOT / '00-foundations/code/interview_kit.py')
        cls.kit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.kit)

    def test_bce_gradient_avoids_overflow_on_extreme_logits(self):
        numpy = self.numpy
        logits = numpy.array([-10000., -1000., 0., 1000., 10000.])
        targets = numpy.array([1., 0., 1., 0., 1.])
        with numpy.errstate(over='raise', invalid='raise', divide='raise'):
            actual = self.kit.bce_with_logits_grad(logits, targets)
        numpy.testing.assert_allclose(actual, [-1, 0, -0.5, 1, 0])
        probability = 0.8
        logit = math.log(probability / (1 - probability))
        self.assertAlmostEqual(self.kit.bce_with_logits_grad(logit, 1) / 2, -0.1)

    def test_batch_mean_gradient_is_invariant_to_duplicating_examples(self):
        numpy = self.numpy
        random = numpy.random.default_rng(23)
        inputs = random.normal(size=(3, 2))
        weights = random.normal(size=(2, 4))
        bias = random.normal(size=4)
        output_weights = random.normal(size=(4, 1))
        targets = numpy.array([0, 1, 0])
        logits, cache = self.kit.mlp_forward(inputs, weights, bias, output_weights, 0.1)
        original = self.kit.mlp_backward(logits, targets, cache, output_weights)
        repeated_logits, repeated_cache = self.kit.mlp_forward(numpy.tile(inputs, (2, 1)), weights, bias, output_weights, 0.1)
        repeated = self.kit.mlp_backward(repeated_logits, numpy.tile(targets, 2), repeated_cache, output_weights)
        for first, second in zip(original, repeated):
            numpy.testing.assert_allclose(first, second, atol=1e-12)

    def test_relu_positive_masks_are_equivalent(self):
        numpy = self.numpy
        preactivation = numpy.array([-100., -1e-12, 0., 1e-12, 100.])
        activation = numpy.maximum(preactivation, 0)
        numpy.testing.assert_array_equal(preactivation > 0, activation > 0)

    def test_swiglu_row_and_column_conventions_agree(self):
        numpy = self.numpy
        random = numpy.random.default_rng(29)
        inputs = random.normal(size=(3, 5))
        gate_weights = random.normal(size=(5, 7))
        up_weights = random.normal(size=(5, 7))
        down_weights = random.normal(size=(7, 5))
        gate = inputs @ gate_weights
        row_output = (gate / (1 + numpy.exp(-gate)) * (inputs @ up_weights)) @ down_weights
        self.assertEqual(row_output.shape, inputs.shape)
        for index, token in enumerate(inputs):
            column_gate = gate_weights.T @ token
            column_output = down_weights.T @ (column_gate / (1 + numpy.exp(-column_gate)) * (up_weights.T @ token))
            numpy.testing.assert_allclose(row_output[index], column_output, atol=1e-12)

    def test_cached_single_query_matches_causal_prefill(self):
        numpy = self.numpy
        random = numpy.random.default_rng(42)
        queries, keys, values = [random.normal(size=(5, 4)) for unused in range(3)]
        reference, unused_weights = self.kit.attention(queries, keys, values, self.kit.causal_mask(5))
        for position in range(5):
            actual, unused_weights = self.kit.attention(queries[position:position + 1], keys[:position + 1], values[:position + 1])
            numpy.testing.assert_allclose(actual[0], reference[position], atol=1e-12)
        block_unmasked, unused_weights = self.kit.attention(queries[-2:], keys, values)
        self.assertFalse(numpy.allclose(block_unmasked[0], reference[-2]))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is needed for normalization checks')
class NormalizationModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def test_one_sample_is_not_one_value_per_channel(self):
        torch = self.torch
        module = torch.nn.BatchNorm1d(3)
        with self.assertRaises(ValueError):
            module(torch.ones(1, 3))
        sequence = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        self.assertEqual(module(sequence).shape, sequence.shape)
        module.eval()
        self.assertTrue(torch.isfinite(module(torch.ones(1, 3))).all())

    def test_temporal_batch_statistics_leak_future_positions(self):
        torch = self.torch
        original = torch.tensor([[[1., 2., 3.], [2., 4., 6.]]])
        changed = original.clone()
        changed[:, :, -1] += 100
        module = torch.nn.BatchNorm1d(2, affine=False)
        self.assertFalse(torch.allclose(module(original)[:, :, 0], module(changed)[:, :, 0]))
        module.eval()
        torch.testing.assert_close(module(original)[:, :, 0], module(changed)[:, :, 0])
        layer_norm = torch.nn.LayerNorm(2)
        torch.testing.assert_close(layer_norm(original.transpose(1, 2))[:, 0], layer_norm(changed.transpose(1, 2))[:, 0])

    def test_fixed_running_statistics_are_not_scale_invariant(self):
        torch = self.torch
        inputs = torch.tensor([[1., 2.], [3., 5.]])
        batch_norm = torch.nn.BatchNorm1d(2, eps=0, affine=False)
        torch.testing.assert_close(batch_norm(inputs), batch_norm(inputs * 4))
        batch_norm.eval()
        self.assertFalse(torch.allclose(batch_norm(inputs), batch_norm(inputs * 4)))
        layer_norm = torch.nn.LayerNorm(2, eps=0, elementwise_affine=False)
        torch.testing.assert_close(layer_norm(inputs), layer_norm(inputs * 4))

    def test_teaching_script_demonstrates_all_three_modes(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            runpy.run_path(str(ROOT / '00-foundations/code/norm_compare.py'), run_name='__main__')
        self.assertIn('BatchNorm -> ValueError:', output.getvalue())
        self.assertIn('BatchNorm(train) [1, C, L] -> (1, 8, 5)', output.getvalue())
        self.assertIn('BatchNorm(eval) [1, C] -> (1, 8)', output.getvalue())
        self.assertNotIn('exactly the autoregressive decoding case', output.getvalue())

    def test_residual_graph_uses_the_same_upstream_gradient(self):
        sys.path.insert(0, str(ROOT / '00-foundations/code'))
        try:
            namespace = runpy.run_path(str(ROOT / '00-foundations/code/make_norm_figures.py'))
        finally:
            sys.path.pop(0)
        plain = namespace['measure_gradients'](depth=8, width=12, residual=False)
        residual = namespace['measure_gradients'](depth=8, width=12, residual=True)
        expected = 1 / math.sqrt(64 * 12)
        self.assertAlmostEqual(plain[-1], expected, places=7)
        self.assertAlmostEqual(residual[-1], expected, places=7)
        self.assertTrue(all(math.isfinite(value) and value >= 0 for value in plain + residual))

    def test_residual_identity_term_does_not_bound_the_total_gradient(self):
        torch = self.torch
        inputs = torch.tensor([1., 2., 3.], requires_grad=True)
        cancelled = inputs - inputs
        gradient, = torch.autograd.grad(cancelled.sum(), inputs)
        torch.testing.assert_close(gradient, torch.zeros_like(inputs))
        amplified = inputs
        for depth in range(10):
            amplified = amplified + amplified
        gradient, = torch.autograd.grad(amplified.sum(), inputs)
        torch.testing.assert_close(gradient, torch.full_like(inputs, 2 ** 10))
        self.assertEqual(cancelled.var(correction=0).item(), 0)


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'NumPy is needed for attention checks')
class AttentionBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy
        cls.numpy = numpy
        cls.namespace = runpy.run_path(str(ROOT / '00-foundations/code/attention_numpy.py'))

    def test_masking_after_softmax_requires_renormalization(self):
        numpy = self.numpy
        softmax = self.namespace['softmax']
        scores = numpy.array([[0., 0.], [2., -1.]])
        allowed = numpy.array([[True, False], [True, True]])
        expected = softmax(numpy.where(allowed, scores, -numpy.inf))
        zeroed = softmax(scores) * allowed
        self.assertFalse(numpy.allclose(zeroed, expected))
        numpy.testing.assert_allclose(zeroed / zeroed.sum(-1, keepdims=True), expected)
        numpy.testing.assert_allclose(softmax([8., 8.]), [0.5, 0.5])
        self.assertGreater(softmax([8., -8.])[0], 0.999999)

    def test_numpy_attention_rejects_entirely_blocked_rows(self):
        numpy = self.numpy
        model = self.namespace['MultiHeadAttention'](4, 2)
        with self.assertRaisesRegex(ValueError, 'at least one allowed key'):
            model(numpy.ones((1, 3, 4)), numpy.ones((3, 3), dtype=bool))

    def test_symmetric_scores_need_not_produce_symmetric_weights(self):
        numpy = self.numpy
        inputs = numpy.array([[1., 0.], [1., 2.]])
        scores = inputs @ inputs.T / math.sqrt(2)
        weights = self.namespace['softmax'](scores)
        numpy.testing.assert_allclose(scores, scores.T)
        self.assertFalse(numpy.allclose(weights, weights.T))

    def test_unmasked_attention_is_equivariant_not_invariant(self):
        numpy = self.numpy
        generator = numpy.random.default_rng(23)
        model = self.namespace['MultiHeadAttention'](8, 2)
        inputs = generator.normal(size=(1, 4, 8))
        permutation = [2, 0, 3, 1]
        original, unused_weights = model(inputs)
        reordered, unused_weights = model(inputs[:, permutation])
        numpy.testing.assert_allclose(reordered, original[:, permutation], atol=1e-12)
        self.assertFalse(numpy.allclose(reordered, original))
        original_causal, unused_weights = model(inputs, self.namespace['causal_mask'](4))
        reordered_causal, unused_weights = model(inputs[:, permutation], self.namespace['causal_mask'](4))
        self.assertFalse(numpy.allclose(reordered_causal, original_causal[:, permutation]))

    def test_published_numpy_snippets_reject_empty_support(self):
        numpy = self.numpy
        for suffix in ('.md', '.en.md'):
            source = (ROOT / ('00-foundations/core/multi-head-attention' + suffix)).read_text()
            snippet = next(block for block in re.findall(r'```python\n(.*?)\n```', source, re.S)
                           if 'import numpy' in block)
            namespace = {}
            exec(snippet, namespace)
            with self.assertRaisesRegex(ValueError, 'at least one allowed key'):
                namespace['softmax'](numpy.full((2, 3), -numpy.inf))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is needed for SDPA checks')
class AttentionImplementationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.namespace = runpy.run_path(str(ROOT / '00-foundations/code/attention_torch.py'))

    def test_published_sdpa_snippets_keep_boolean_mask_semantics(self):
        torch = self.torch
        generator = torch.Generator().manual_seed(11)
        query = torch.randn(1, 2, 4, 3, generator=generator, dtype=torch.float64)
        key = torch.randn(1, 2, 4, 3, generator=generator, dtype=torch.float64)
        value = torch.randn(1, 2, 4, 3, generator=generator, dtype=torch.float64)
        blocked = self.namespace['causal_mask'](4)
        expected = ((query @ key.transpose(-1, -2)) / math.sqrt(3)).masked_fill(blocked, -torch.inf).softmax(-1) @ value
        for suffix in ('.md', '.en.md'):
            source = (ROOT / ('00-foundations/core/multi-head-attention' + suffix)).read_text()
            snippet = next(block for block in re.findall(r'```python\n(.*?)\n```', source, re.S)
                           if 'output = F.scaled_dot_product_attention(' in block)
            namespace = {'q': query, 'k': key, 'v': value, 'mask': blocked}
            exec(snippet, namespace)
            torch.testing.assert_close(namespace['output'], expected)

    def test_sdpa_and_manual_output_and_gradients_agree(self):
        torch = self.torch
        torch.manual_seed(17)
        model = self.namespace['MultiHeadAttention'](8, 2).double()
        blocked = self.namespace['causal_mask'](4)
        for mask in (None, blocked):
            inputs = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
            manual, unused_weights = model(inputs, mask)
            sdpa = self.namespace['sdpa_version'](model, inputs, mask)
            torch.testing.assert_close(sdpa, manual)
            manual_gradient, = torch.autograd.grad(manual.square().sum(), inputs, retain_graph=True)
            sdpa_gradient, = torch.autograd.grad(sdpa.square().sum(), inputs)
            torch.testing.assert_close(sdpa_gradient, manual_gradient)

    def test_teaching_implementations_reject_empty_support(self):
        torch = self.torch
        model = self.namespace['MultiHeadAttention'](8, 2)
        inputs = torch.ones(1, 4, 8)
        blocked = torch.ones(4, 4, dtype=torch.bool)
        for function in (lambda: model(inputs, blocked),
                         lambda: self.namespace['sdpa_version'](model, inputs, blocked)):
            with self.assertRaisesRegex(ValueError, 'at least one allowed key'):
                function()


class LoopedBudgetTests(unittest.TestCase):
    def test_published_stopping_example_matches_in_both_languages(self):
        examples = []
        for suffix in ('.md', '.en.md'):
            source = (ROOT / ('00-foundations/looped/adaptive-depth' + suffix)).read_text()
            example = re.findall(r'```python\n(.*?)\n```', source, re.S)[0]
            examples.append(example)
            namespace = {}
            exec(example, namespace)
            distribution = namespace['stopping_distribution']
            for conditional in ([0.2, 0.5, 1], [0, 0, 0], [1, 0, 0], [0.1]):
                probabilities = distribution(conditional)
                self.assertAlmostEqual(sum(probabilities), 1)
                self.assertTrue(all(0 <= mass <= 1 for mass in probabilities))
            self.assertEqual(distribution([0, 0, 0]), [0, 0, 1])
            for invalid in ([], [-0.1], [1.1], [float('nan')]):
                with self.assertRaises(ValueError):
                    distribution(invalid)
        self.assertEqual(*examples)

    def test_depth_indexed_kv_units(self):
        bytes_per_token = 2 * 24 * 4 * 2048 * 2
        self.assertEqual(bytes_per_token, 786432)
        self.assertEqual(bytes_per_token // 1024, 768)
        self.assertEqual(bytes_per_token * 4096 // 2 ** 30, 3)

    def test_uniform_prior_kl_equals_negative_entropy_plus_constant(self):
        probabilities = [0.2, 0.4, 0.4]
        entropy = -sum(mass * math.log(mass) for mass in probabilities)
        divergence = sum(mass * math.log(mass * len(probabilities)) for mass in probabilities)
        self.assertAlmostEqual(divergence, -entropy + math.log(len(probabilities)))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is needed for routing gradient checks')
class RoutingBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def test_top_one_renormalization_removes_the_task_gradient(self):
        torch = self.torch
        logits = torch.tensor([2., 0., -1.], dtype=torch.float64, requires_grad=True)
        probabilities = logits.softmax(-1)
        selected = probabilities[:1]
        renormalized_loss = (selected / selected.sum() * 3).sum()
        normalized_gradient, = torch.autograd.grad(renormalized_loss, logits, retain_graph=True)
        unnormalized_gradient, = torch.autograd.grad(selected.sum() * 3, logits)
        torch.testing.assert_close(normalized_gradient, torch.zeros_like(logits))
        self.assertTrue((unnormalized_gradient.abs() > 0).all())

    def test_top_two_renormalization_cancels_unselected_logits(self):
        torch = self.torch
        logits = torch.tensor([2., 0., -1.], dtype=torch.float64, requires_grad=True)
        selected = logits.softmax(-1)[:2]
        output = (selected / selected.sum() * torch.tensor([3., -2.])).sum()
        gradient, = torch.autograd.grad(output, logits)
        self.assertGreater(gradient[:2].abs().min().item(), 0)
        self.assertAlmostEqual(gradient[-1].item(), 0, places=12)

    def test_zero_z_loss_does_not_imply_balanced_routing(self):
        torch = self.torch
        logits = torch.tensor([0.9, 0.1], dtype=torch.float64).log()
        self.assertAlmostEqual(logits.logsumexp(-1).square().item(), 0, places=12)
        torch.testing.assert_close(logits.softmax(-1), torch.tensor([0.9, 0.1], dtype=torch.float64))
        self.assertNotEqual(logits.argmax().item(), 1)

    def test_shared_recurrent_weights_do_not_make_identical_jacobians(self):
        torch = self.torch
        derivatives = []
        for state in (0., 2.):
            previous = torch.tensor(state, dtype=torch.float64, requires_grad=True)
            current = (previous * 0.8 + 0.2).tanh()
            derivative, = torch.autograd.grad(current, previous)
            self.assertAlmostEqual(derivative.item(), 0.8 * (1 - current.item() ** 2))
            derivatives.append(derivative.item())
        self.assertNotAlmostEqual(*derivatives)

    def test_truncated_recursion_changes_the_gradient(self):
        torch = self.torch
        weight = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        full = weight * (weight * (weight * 1))
        full_gradient, = torch.autograd.grad(full, weight)
        truncated = weight * (weight * (weight * 1)).detach()
        truncated_gradient, = torch.autograd.grad(truncated, weight)
        self.assertAlmostEqual(full.item(), truncated.item())
        self.assertAlmostEqual(full_gradient.item(), 0.75)
        self.assertAlmostEqual(truncated_gradient.item(), 0.25)


if __name__ == '__main__':
    unittest.main()
