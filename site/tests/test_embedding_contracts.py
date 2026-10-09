import contextlib
import importlib.util
import io
import math
from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "04-search/code/embedding_contracts.py"
SPEC = importlib.util.spec_from_file_location("embedding_contracts", CODE)
EMBEDDING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EMBEDDING)

try:
    import torch
except ImportError:
    torch = None


def lesson_namespace(language=""):
    source = ROOT / f"04-search/dual-encoder{language}.md"
    namespace = {}
    for block in re.findall(r"```python\n(.*?)\n```", source.read_text(), re.S):
        exec(compile(block, str(source), "exec"), namespace)
    return namespace


class EmbeddingContractTests(unittest.TestCase):
    def test_last_valid_pool_supports_both_padding_sides(self):
        left = [[99, 99], [1, 0], [2, 0], [3, 4]]
        right = [[1, 0], [2, 0], [3, 4], [99, 99]]
        self.assertEqual(EMBEDDING.last_valid_pool(left, [0, 1, 1, 1]), [3, 4])
        self.assertEqual(EMBEDDING.last_valid_pool(right, [1, 1, 1, 0]), [3, 4])
        self.assertEqual(EMBEDDING.last_valid_pool([[7, 8]], [1]), [7, 8])

    def test_mask_errors_are_not_silently_pooled(self):
        for mask in ([0, 0], [1], [1, 2], [1, math.nan]):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                EMBEDDING.last_valid_pool([[1, 0], [3, 4]], mask)

    def test_invalid_states_are_rejected(self):
        for states in ([], [[]], [[1], [2, 3]], [[math.inf]], [[math.nan]]):
            with self.subTest(states=states), self.assertRaises(ValueError):
                EMBEDDING.last_valid_pool(states, [1] * len(states))

    def test_pool_is_a_copy_and_does_not_mutate_input(self):
        states = [[3, 4]]
        pooled = EMBEDDING.last_valid_pool(states, [1])
        pooled[0] = 0
        self.assertEqual(states, [[3, 4]])

    def test_prefix_is_renormalized(self):
        self.assertEqual(EMBEDDING.normalized_prefix([3, 4]), [0.6, 0.8])
        self.assertEqual(EMBEDDING.normalized_prefix([0.6, 0, 0.8], 2), [1, 0])
        self.assertEqual(EMBEDDING.normalized_prefix([-4, 0], 1), [-1])

    def test_normalization_handles_large_and_small_scales(self):
        for scale in (1e-300, 1e300):
            result = EMBEDDING.normalized_prefix([3 * scale, 4 * scale])
            self.assertAlmostEqual(result[0], 0.6)
            self.assertAlmostEqual(result[1], 0.8)
            self.assertAlmostEqual(math.hypot(*result), 1)

    def test_invalid_prefixes_and_zero_vectors_are_rejected(self):
        for dimension in (0, -1, 4, 1.5, True):
            with self.subTest(dimension=dimension), self.assertRaises(ValueError):
                EMBEDDING.normalized_prefix([1, 0, 0], dimension)
        for vector, dimension in (([0, 0], None), ([0, 0, 1], 2), ([], None), ([math.nan], None)):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                EMBEDDING.normalized_prefix(vector, dimension)

    def test_same_dimensions_do_not_imply_compatible_spaces(self):
        query_old, query_new = [1, 0], [0, 1]
        documents_old = [[1, 0], [0, 1]]
        documents_new = [[0, 1], [-1, 0]]
        original = [EMBEDDING.dot(query_old, document) for document in documents_old]
        mixed = [EMBEDDING.dot(query_new, document) for document in documents_old]
        rebuilt = [EMBEDDING.dot(query_new, document) for document in documents_new]
        self.assertEqual(original, [1, 0])
        self.assertEqual(mixed, [0, 1])
        self.assertEqual(rebuilt, original)

    def test_dot_rejects_incompatible_or_nonfinite_vectors(self):
        for first, second in (([1], [1, 2]), ([], []), ([math.nan], [1]), ([1e308], [1e308])):
            with self.subTest(first=first), self.assertRaises(ValueError):
                EMBEDDING.dot(first, second)

    def test_maxsim_matches_the_table(self):
        queries, documents = [[1, 0], [0, 1]], [[0.8, 0.6], [0, 1]]
        self.assertAlmostEqual(EMBEDDING.mean_maxsim(queries, documents), 0.9)
        self.assertAlmostEqual(EMBEDDING.mean_maxsim(queries * 2, documents), 0.9)
        self.assertAlmostEqual(EMBEDDING.mean_maxsim(queries, documents * 2), 0.9)

    def test_maxsim_is_directional_and_order_insensitive_after_encoding(self):
        queries, documents = [[1, 0]], [[0.8, 0.6], [0, 1]]
        self.assertAlmostEqual(EMBEDDING.mean_maxsim(queries, documents), 0.8)
        self.assertAlmostEqual(EMBEDDING.mean_maxsim(documents, queries), 0.4)
        self.assertAlmostEqual(EMBEDDING.mean_maxsim(queries, documents[::-1]), 0.8)

    def test_maxsim_requires_valid_nonzero_matching_dimensions(self):
        for queries, documents in (([], [[1, 0]]), ([[1, 0]], []), ([[0, 0]], [[1, 0]]),
                                   ([[1, 0]], [[1]]), ([[1], [1, 2]], [[1]])):
            with self.subTest(queries=queries), self.assertRaises(ValueError):
                EMBEDDING.mean_maxsim(queries, documents)

    def test_raw_vector_storage_matches_both_notes(self):
        for dimensions, width, expected in ((1024, 4, "3.815"), (1024, 2, "1.907"), (256, 2, "0.477")):
            amount = 1_000_000 * dimensions * width
            self.assertEqual(f"{amount / 2**30:.3f}", expected)
            for language in ("", ".en"):
                source = (ROOT / f"04-search/embedding-models{language}.md").read_text()
                self.assertIn(f"{amount:,}", source)
                self.assertIn(expected, source)

    def test_bilingual_python_examples_execute(self):
        previous = sys.modules.get("embedding_contracts")
        sys.modules["embedding_contracts"] = EMBEDDING
        try:
            for language in ("", ".en"):
                path = ROOT / f"04-search/embedding-models{language}.md"
                for block in re.findall(r"```python\n(.*?)\n```", path.read_text(), re.S):
                    exec(compile(block, str(path), "exec"), {})
        finally:
            if previous is None:
                del sys.modules["embedding_contracts"]
            else:
                sys.modules["embedding_contracts"] = previous

    def test_demo_matches_documented_output(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            EMBEDDING.main()
        expected = "Last valid: [3.0, 4.0]\nMixed spaces: [0.0, 1.0]; rebuilt: [1.0, 0.0]\nMean MaxSim: 0.9\n"
        self.assertEqual(output.getvalue(), expected)

    def test_adding_candidate_changes_softmax_not_independent_sigmoid(self):
        for language in ("", ".en"):
            calculate = lesson_namespace(language)["contrastive_step"]
            loss_three, probabilities_three, gradients = calculate([0, 0, 0])
            loss_four, probabilities_four, _ = calculate([0, 0, 0, 0])
            self.assertAlmostEqual(loss_three, math.log(3))
            self.assertAlmostEqual(loss_four, math.log(4))
            self.assertAlmostEqual(probabilities_three[0], 1 / 3)
            self.assertAlmostEqual(probabilities_four[0], 1 / 4)
            self.assertAlmostEqual(gradients[0], -2 / 3)
            for count in (3, 4):
                sigmoid = [1 / (1 + math.exp(-score)) for score in [0] * count]
                self.assertEqual(sigmoid, [0.5] * count)
                self.assertAlmostEqual(-sum(math.log(value) for value in sigmoid) / count, math.log(2))

    @unittest.skipIf(torch is None, "PyTorch not installed; analytic examples still run")
    def test_infonce_matches_pytorch_ce_value_and_gradient(self):
        calculate = lesson_namespace()["contrastive_step"]
        for temperature in (0.3, 1.0, 2.0):
            scores = [2.0, 1.0, 0.0]
            expected_loss, _, expected_gradients = calculate(scores, temperature)
            logits = torch.tensor([scores], dtype=torch.float64, requires_grad=True)
            loss = torch.nn.functional.cross_entropy(logits / temperature, torch.tensor([0]))
            loss.backward()
            self.assertAlmostEqual(loss.item(), expected_loss)
            for observed, expected in zip(logits.grad[0].tolist(), expected_gradients):
                self.assertAlmostEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
