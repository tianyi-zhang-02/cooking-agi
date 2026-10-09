import itertools
import math
from pathlib import Path
import re
import unittest

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[2]


def snippets(name, language=""):
    source = ROOT / "00-foundations/deep-dives" / f"{name}{language}.md"
    return re.findall(r"```python\n(.*?)```", source.read_text(), re.S)


def load_examples(name):
    namespace = {}
    for snippet in snippets(name):
        exec(compile(snippet, name, "exec"), namespace)
    return namespace


class BilingualExamplesTests(unittest.TestCase):
    def test_examples_match_in_both_languages(self):
        for name in ("muon", "dflash"):
            self.assertTrue(snippets(name))
            self.assertEqual(snippets(name), snippets(name, ".en"))


class DraftVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        namespace = load_examples("dflash")
        cls.verify = staticmethod(namespace["verify_greedy_block"])
        cls.speedup = staticmethod(namespace["cycle_speedup"])

    def test_first_mismatch_discards_entire_suffix(self):
        for position in range(4):
            draft = [10, 20, 30, 40]
            targets = draft + [50]
            targets[position] = 99
            self.assertEqual(self.verify(draft, targets), (draft[:position] + [99], position))

    def test_full_acceptance_and_empty_draft(self):
        self.assertEqual(self.verify([1, 2], [1, 2, 3]), ([1, 2, 3], 2))
        self.assertEqual(self.verify([], [7]), ([7], 0))

    def test_eos_stops_at_acceptance_or_correction(self):
        self.assertEqual(self.verify([1, 0, 3], [1, 0, 3, 4], 0), ([1, 0], 2))
        self.assertEqual(self.verify([1, 2, 3], [1, 0, 3, 4], 0), ([1, 0], 1))
        self.assertEqual(self.verify([1], [1, 0], 0), ([1, 0], 1))

    def test_invalid_target_alignment(self):
        for choices in ([], [1], [1, 2], [1, 2, 3, 4]):
            with self.assertRaises(ValueError):
                self.verify([1, 2], choices)

    def test_matches_autoregressive_prefix_on_all_binary_drafts(self):
        def next_token(prefix):
            return (sum(prefix) + len(prefix)) % 2

        for length in range(5):
            for draft in itertools.product(range(2), repeat=length):
                choices = [next_token(draft[:position]) for position in range(length + 1)]
                actual, _ = self.verify(draft, choices)
                expected = []
                for _ in actual:
                    expected.append(next_token(expected))
                self.assertEqual(actual, expected)

    def test_cycle_cost_examples_and_break_even(self):
        self.assertEqual(self.speedup(10, 3, 12, 1, 4), 2.5)
        self.assertEqual(self.speedup(10, 3, 12, 1, 1), 0.625)
        self.assertEqual(self.speedup(10, 3, 12, 1, 1.6), 1)

    def test_invalid_cost_inputs(self):
        for inputs in ((0, 1, 1, 1, 1), (1, -1, 1, 1, 1),
                       (1, 0, 0, 0, 1), (1, 1, 1, 1, 0),
                       (1, 1, math.nan, 1, 1), (math.inf, 1, 1, 1, 1)):
            with self.assertRaises(ValueError):
                self.speedup(*inputs)


@unittest.skipUnless(torch is not None, "PyTorch is required for SVD reference checks")
class MuonGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.polar = staticmethod(load_examples("muon")["polar_reference"])

    def test_diagonal_singular_values_and_sign(self):
        actual = self.polar([[6.0, 0.0], [0.0, -2.0]])
        expected = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float64))
        torch.testing.assert_close(actual, expected)

    def test_rectangular_semi_orthogonality_and_rms(self):
        matrix = torch.tensor([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]], dtype=torch.float64)
        for source in (matrix, matrix.T):
            actual = self.polar(source)
            gram = actual @ actual.T if source.shape[0] < source.shape[1] else actual.T @ actual
            torch.testing.assert_close(gram, torch.eye(2, dtype=torch.float64))
            self.assertAlmostEqual(actual.square().mean().sqrt().item(), 1 / math.sqrt(3))

    def test_rank_deficiency_and_zero(self):
        torch.testing.assert_close(self.polar([[2.0, 0.0], [0.0, 0.0]]),
                                   torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64)))
        self.assertEqual(torch.count_nonzero(self.polar(torch.zeros(2, 3))), 0)

    def test_scale_invariance_and_not_elementwise_sign(self):
        matrix = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
        expected = self.polar(matrix)
        for scale in (1e-200, 1e200):
            torch.testing.assert_close(self.polar(matrix * scale), expected)
        self.assertFalse(torch.allclose(expected, matrix.sign()))

    def test_invalid_matrices(self):
        for matrix in ([], [1, 2], [[], []], [[math.inf]], [[math.nan]]):
            with self.assertRaises(ValueError):
                self.polar(matrix)

    def test_common_quintic_does_not_fix_one(self):
        self.assertAlmostEqual(3.4445 - 4.775 + 2.0315, 0.701)


if __name__ == "__main__":
    unittest.main()
