import ast
import importlib.util
import math
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "interview/code/number_theory.py"
SPEC = importlib.util.spec_from_file_location("number_theory", SOURCE)
NUMBERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NUMBERS)


def naive_prime(number):
    return number >= 2 and all(number % divisor for divisor in range(2, number))


class PrimeSieveTests(unittest.TestCase):
    def test_trial_division_against_exhaustive_divisors(self):
        for number in range(-10, 700):
            self.assertEqual(NUMBERS.is_prime(number), naive_prime(number), number)

    def test_sieve_against_exhaustive_divisors(self):
        for limit in range(80):
            self.assertEqual(list(NUMBERS.prime_flags(limit)), [int(naive_prime(number)) for number in range(limit + 1)])

    def test_perfect_squares_and_neighbors(self):
        for prime in (2, 3, 5, 7, 11, 29, 97):
            square = prime * prime
            self.assertFalse(NUMBERS.is_prime(square))
            flags = NUMBERS.prime_flags(square)
            self.assertEqual(flags[square], 0)
            self.assertEqual(flags[prime], 1)

    def test_documented_examples_and_counts(self):
        flags = NUMBERS.prime_flags(30)
        self.assertEqual([number for number, flag in enumerate(flags) if flag], [2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
        self.assertEqual(sum(NUMBERS.prime_flags(100)), 25)
        self.assertEqual(sum(NUMBERS.prime_flags(1000)), 168)
        self.assertEqual(NUMBERS.prime_factors(84), [2, 2, 3, 7])

    def test_inclusive_endpoint_and_small_limits(self):
        self.assertEqual(list(NUMBERS.prime_flags(0)), [0])
        self.assertEqual(list(NUMBERS.prime_flags(1)), [0, 0])
        self.assertEqual(list(NUMBERS.prime_flags(2)), [0, 0, 1])
        self.assertEqual(sum(NUMBERS.prime_flags(6)), 3)
        self.assertEqual(sum(NUMBERS.prime_flags(7)), 4)
        self.assertIsInstance(NUMBERS.prime_flags(7), bytearray)

    def test_factorization_product_and_primality(self):
        for number in range(1, 700):
            factors = NUMBERS.prime_factors(number)
            self.assertEqual(math.prod(factors), number)
            self.assertEqual(factors, sorted(factors))
            self.assertTrue(all(naive_prime(factor) for factor in factors))
        self.assertEqual(NUMBERS.prime_factors(2 ** 20), [2] * 20)

    def test_input_contracts(self):
        for function in (NUMBERS.is_prime, NUMBERS.prime_flags, NUMBERS.prime_factors):
            for value in (True, 7.0, "7", None):
                with self.assertRaises(TypeError):
                    function(value)
        with self.assertRaises(ValueError):
            NUMBERS.prime_flags(-1)
        for value in (-1, 0):
            with self.assertRaises(ValueError):
                NUMBERS.prime_factors(value)

    def test_segment_start_formula(self):
        for left, right in ((0, 0), (0, 2), (3, 7), (20, 30), (49, 59), (95, 120)):
            flags = [1] * (right - left + 1)
            for prime, is_prime in enumerate(NUMBERS.prime_flags(math.isqrt(right))):
                if is_prime:
                    start = max(prime * prime, ((left + prime - 1) // prime) * prime)
                    for multiple in range(start, right + 1, prime):
                        flags[multiple - left] = 0
            for number in (0, 1):
                if left <= number <= right:
                    flags[number - left] = 0
            self.assertEqual(flags, [int(naive_prime(number)) for number in range(left, right + 1)])

    def test_bilingual_code_and_output_match_implementation(self):
        source = SOURCE.read_text()
        functions = {
            node.name: ast.get_source_segment(source, node)
            for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
        }
        for suffix in (".md", ".en.md"):
            note = (ROOT / "interview/algorithms" / ("primes-and-sieves" + suffix)).read_text()
            blocks = re.findall(r"```python\n(.*?)\n```", note, re.S)
            self.assertEqual(blocks, [functions["is_prime"], functions["prime_flags"]])
            self.assertIn("Primes through 30: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]", note)


if __name__ == "__main__":
    unittest.main()
