import ast
import importlib.util
import itertools
from pathlib import Path
import re
import subprocess
import sys
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("deeper_patterns", ROOT / "interview/code/deeper_patterns.py")
PATTERNS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PATTERNS
SPEC.loader.exec_module(PATTERNS)
PAGES = ("string-matching", "monotonic-queue", "binary-search-trees",
         "knapsack", "sequence-dp", "state-machine-dp")


def subsequences(sequence):
    return {tuple(value for index, value in enumerate(sequence) if mask >> index & 1)
            for mask in range(1 << len(sequence))}


def build_tree(values):
    root = None
    for value in values:
        if root is None:
            root = PATTERNS.TreeNode(value)
            continue
        current = root
        while True:
            side = "left" if value < current.value else "right"
            child = getattr(current, side)
            if child is None:
                setattr(current, side, PATTERNS.TreeNode(value))
                break
            current = child
    return root


def exhaustive_profit(prices):
    def explore(day, owns, blocked, cash):
        if day == len(prices):
            return cash if not owns else float("-inf")
        outcomes = [explore(day + 1, owns, False, cash)]
        if owns:
            outcomes.append(explore(day + 1, False, True, cash + prices[day]))
        elif not blocked:
            outcomes.append(explore(day + 1, True, False, cash - prices[day]))
        return max(outcomes)

    return explore(0, False, False, 0)


class DeeperAlgorithmTests(unittest.TestCase):
    def test_kmp_all_short_binary_strings_and_overlaps(self):
        strings = ["".join(chars) for size in range(6)
                   for chars in itertools.product("ab", repeat=size)]
        for text in strings:
            for pattern in strings:
                expected = [index for index in range(len(text) + 1)
                            if text[index:index + len(pattern)] == pattern]
                self.assertEqual(PATTERNS.kmp_positions(text, pattern), expected, (text, pattern))
        self.assertEqual(PATTERNS.kmp_positions("回头回头回", "回头回"), [0, 2])

    def test_prefix_table_against_definition(self):
        for size in range(8):
            for chars in itertools.product("ab", repeat=size):
                pattern = "".join(chars)
                expected = [max(length for length in range(end + 1)
                                if pattern[:length] == pattern[end + 1 - length:end + 1])
                            for end in range(size)]
                self.assertEqual(PATTERNS.prefix_lengths(pattern), expected)
        self.assertEqual(PATTERNS.prefix_lengths("ababac"), [0, 0, 1, 2, 3, 0])

    def test_window_maxima_against_slicing(self):
        for size in range(1, 8):
            for values in itertools.product((-2, 0, 2), repeat=size):
                for width in range(1, size + 1):
                    expected = [max(values[start:start + width]) for start in range(size - width + 1)]
                    self.assertEqual(PATTERNS.window_maxima(values, width), expected)
        self.assertEqual(PATTERNS.window_maxima([4, 1, 3, 5, 2, 5], 3), [4, 5, 5, 5])

    def test_invalid_window_contracts(self):
        for values, width in [([], 1), ([1], 0), ([1], 2), ([1], 1.0), ([1], True)]:
            with self.assertRaises(ValueError):
                PATTERNS.window_maxima(values, width)

    def test_bst_global_bounds_and_duplicates(self):
        invalid = PATTERNS.TreeNode(8, None, PATTERNS.TreeNode(10, PATTERNS.TreeNode(6)))
        self.assertFalse(PATTERNS.valid_bst(invalid))
        self.assertFalse(PATTERNS.valid_bst(PATTERNS.TreeNode(8, PATTERNS.TreeNode(8))))
        self.assertTrue(PATTERNS.valid_bst(None))
        self.assertTrue(PATTERNS.valid_bst(build_tree([-10 ** 50, 10 ** 50, 0])))

    def test_bst_all_insertion_orders(self):
        for values in itertools.permutations(range(5)):
            root = build_tree(values)
            self.assertTrue(PATTERNS.valid_bst(root))
            self.assertEqual([PATTERNS.kth_smallest(root, rank) for rank in range(1, 6)],
                             list(range(5)))

    def test_bst_deep_chain_and_invalid_ranks(self):
        root = PATTERNS.TreeNode(0)
        current = root
        for value in range(1, 3000):
            current.right = PATTERNS.TreeNode(value)
            current = current.right
        self.assertTrue(PATTERNS.valid_bst(root))
        self.assertEqual(PATTERNS.kth_smallest(root, 3000), 2999)
        for rank in (0, -1, 3001, 1.5, True):
            with self.assertRaises(ValueError):
                PATTERNS.kth_smallest(root, rank)
        with self.assertRaises(ValueError):
            PATTERNS.kth_smallest(None, 1)

    def test_knapsack_against_subsets(self):
        options = ((1, -1), (1, 2), (2, 5), (3, 7))
        for count in range(5):
            for items in itertools.product(options, repeat=count):
                for capacity in range(7):
                    expected = max(
                        sum(items[index][1] for index in chosen)
                        for size in range(count + 1)
                        for chosen in itertools.combinations(range(count), size)
                        if sum(items[index][0] for index in chosen) <= capacity
                    )
                    self.assertEqual(PATTERNS.knapsack_once(items, capacity), expected)
        self.assertEqual(PATTERNS.knapsack_once([(2, 5)], 4), 5)
        self.assertEqual(PATTERNS.knapsack_once([(2, 5), (3, 7), (4, 8)], 5), 12)

    def test_coin_change_against_breadth_first_search(self):
        for size in range(5):
            for coins in itertools.combinations(range(1, 5), size):
                for amount in range(13):
                    frontier, visited, distance = {0}, {0}, 0
                    expected = -1
                    while frontier:
                        if amount in frontier:
                            expected = distance
                            break
                        next_frontier = {subtotal + coin for subtotal in frontier for coin in coins
                                         if subtotal + coin <= amount and subtotal + coin not in visited}
                        visited.update(next_frontier)
                        frontier = next_frontier
                        distance += 1
                    self.assertEqual(PATTERNS.minimum_coins(coins, amount), expected)
        self.assertEqual(PATTERNS.minimum_coins([1, 3, 4], 6), 2)
        self.assertEqual(PATTERNS.minimum_coins([2, 2], 4), 2)

    def test_invalid_capacity_and_coin_contracts(self):
        for capacity in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                PATTERNS.knapsack_once([], capacity)
            with self.assertRaises(ValueError):
                PATTERNS.minimum_coins([], capacity)
        for weight in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                PATTERNS.knapsack_once([(weight, 2)], 4)
            with self.assertRaises(ValueError):
                PATTERNS.minimum_coins([weight], 4)

    def test_lcs_against_subsequence_intersection(self):
        strings = ["".join(chars) for size in range(5)
                   for chars in itertools.product("ab", repeat=size)]
        for first in strings:
            for second in strings:
                expected = max(map(len, subsequences(first) & subsequences(second)))
                self.assertEqual(PATTERNS.lcs_length(first, second), expected)
        self.assertEqual(PATTERNS.lcs_length("CAB", "ACB"), 2)

    def test_lis_against_subsequence_enumeration(self):
        for size in range(7):
            for values in itertools.product((0, 1, 2), repeat=size):
                expected = max(len(sequence) for sequence in subsequences(values)
                               if all(left < right for left, right in zip(sequence, sequence[1:])))
                self.assertEqual(PATTERNS.lis_length(values), expected)
        self.assertEqual(PATTERNS.lis_length([3, 5, 6, 2, 4]), 3)
        self.assertEqual(PATTERNS.lis_length([2, 2, 2]), 1)

    def test_cooldown_against_action_enumeration(self):
        for size in range(7):
            for prices in itertools.product((0, 1, 3), repeat=size):
                self.assertEqual(PATTERNS.cooldown_profit(prices), exhaustive_profit(prices))
        self.assertEqual(PATTERNS.cooldown_profit([1, 2, 3, 0, 2]), 3)

    def test_article_code_matches_runnable_functions_in_both_languages(self):
        expected = {node.name: ast.dump(node, include_attributes=False)
                    for node in ast.parse((ROOT / "interview/code/deeper_patterns.py").read_text()).body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name != "main"}
        for language in ("", ".en"):
            found = set()
            for stem in PAGES:
                page = ROOT / f"interview/algorithms/{stem}{language}.md"
                for snippet in re.findall(r"```python\n(.*?)\n```", page.read_text(), re.S):
                    for node in ast.parse(snippet).body:
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            self.assertIn(node.name, expected, str(page))
                            self.assertEqual(ast.dump(node, include_attributes=False), expected[node.name], str(page))
                            found.add(node.name)
            self.assertEqual(found, set(expected))

    def test_paired_notes_have_equal_examples_and_headings(self):
        for stem in PAGES:
            zh = (ROOT / f"interview/algorithms/{stem}.md").read_text()
            en = (ROOT / f"interview/algorithms/{stem}.en.md").read_text()
            self.assertEqual(re.findall(r"^#+ ", zh, re.M), re.findall(r"^#+ ", en, re.M))
            self.assertEqual(re.findall(r"```python\n(.*?)\n```", zh, re.S),
                             re.findall(r"```python\n(.*?)\n```", en, re.S))
            self.assertEqual(re.findall(r"\$\$([\s\S]*?)\$\$", zh),
                             re.findall(r"\$\$([\s\S]*?)\$\$", en))
            self.assertNotIn("__", re.sub(r"```.*?```", "", zh, flags=re.S))
            self.assertGreater(len(zh), 1800, stem)
            self.assertGreater(len(en.split()), 420, stem)

    def test_navigation_and_pattern_map_link_every_note(self):
        nav = tomllib.loads((ROOT / "site/nav.toml").read_text())
        order = next(section["order"] for section in nav["section"]
                     if section.get("dir") == "interview/algorithms")
        for stem in PAGES:
            self.assertEqual(order.count(stem + ".md"), 1)
            for language in ("", ".en"):
                guide = (ROOT / f"interview/leetcode{language}.md").read_text()
                self.assertIn(f"algorithms/{stem}{language}.md", guide)

    def test_cli_examples(self):
        result = subprocess.run([sys.executable, str(ROOT / "interview/code/deeper_patterns.py")],
                                check=True, capture_output=True, text=True)
        self.assertIn("passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
