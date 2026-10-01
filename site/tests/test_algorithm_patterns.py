import ast
import bisect
import importlib.util
import itertools
import json
from pathlib import Path
import random
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("patterns", ROOT / "interview/code/patterns.py")
PATTERNS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATTERNS)


class AlgorithmPatternTests(unittest.TestCase):
    def test_hash_map_pair_and_self_reuse(self):
        self.assertEqual(PATTERNS.two_sum([4, 1, 7], 8), [1, 2])
        self.assertEqual(PATTERNS.two_sum([3, 3], 6), [0, 1])
        self.assertIsNone(PATTERNS.two_sum([3], 6))
        self.assertIsNone(PATTERNS.two_sum([], 0))

    def test_prefix_count_against_enumeration(self):
        for size in range(6):
            for values in itertools.product((-1, 0, 1), repeat=size):
                for target in (-1, 0, 1):
                    expected = sum(sum(values[left:right]) == target
                                   for left in range(size) for right in range(left + 1, size + 1))
                    self.assertEqual(PATTERNS.count_subarrays(values, target), expected)

    def test_sorted_pair_and_lower_bound(self):
        for values in ([], [2], [1, 1, 2, 4], [-3, -1, 0, 5]):
            for target in range(-5, 10):
                self.assertEqual(PATTERNS.lower_bound(values, target), bisect.bisect_left(values, target))
                pair = PATTERNS.sorted_pair(values, target)
                possible = any(first + second == target for first, second in itertools.combinations(values, 2))
                self.assertEqual(pair is not None, possible)
                if pair:
                    self.assertLess(pair[0], pair[1])
                    self.assertEqual(sum(values[index] for index in pair), target)

    def test_window_against_enumeration(self):
        for size in range(7):
            for letters in itertools.product("abc", repeat=size):
                text = "".join(letters)
                expected = max((right - left for left in range(size + 1)
                                for right in range(left, size + 1)
                                if len(set(text[left:right])) == right - left), default=0)
                self.assertEqual(PATTERNS.longest_unique(text), expected)

    def test_traversal_cycles_and_order(self):
        graph = {"A": ["B", "C"], "B": ["C", "D"], "C": ["A"], "D": ["D"]}
        self.assertEqual(PATTERNS.dfs_recursive(graph, "A"), ["A", "B", "C", "D"])
        self.assertEqual(PATTERNS.dfs_iterative(graph, "A"), PATTERNS.dfs_recursive(graph, "A"))
        self.assertEqual(PATTERNS.bfs_distances(graph, "A"), {"A": 0, "B": 1, "C": 1, "D": 2})
        self.assertEqual(PATTERNS.dfs_iterative({}, "isolated"), ["isolated"])
        self.assertEqual(PATTERNS.bfs_distances({}, "isolated"), {"isolated": 0})

    def test_iterative_deep_chain(self):
        graph = {index: [index + 1] for index in range(3000)}
        self.assertEqual(PATTERNS.dfs_iterative(graph, 0), list(range(3001)))
        self.assertEqual(PATTERNS.bfs_distances(graph, 0)[3000], 3000)

    def test_linked_list_identity_and_reverse(self):
        self.assertIsNone(PATTERNS.reverse_list(None))
        tail = PATTERNS.ListNode(2)
        head = PATTERNS.ListNode(1, tail)
        self.assertFalse(PATTERNS.has_cycle(head))
        reversed_head = PATTERNS.reverse_list(head)
        self.assertIs(reversed_head, tail)
        self.assertIs(tail.next, head)
        self.assertIsNone(head.next)
        head.next = tail
        self.assertTrue(PATTERNS.has_cycle(tail))

    def test_stack_and_heap_against_brute_force(self):
        generator = random.Random(17)
        for size in range(30):
            values = [generator.randrange(10) for _ in range(size)]
            expected = [next((later - index for later in range(index + 1, size)
                              if values[later] > value), 0) for index, value in enumerate(values)]
            self.assertEqual(PATTERNS.next_warmer(values), expected)
            for count in (0, 1, 3, 40):
                self.assertEqual(PATTERNS.largest_values(values, count), sorted(values, reverse=True)[:count])

    def test_subsets_outputs_are_independent(self):
        self.assertEqual(PATTERNS.subsets([]), [[]])
        result = PATTERNS.subsets([1, 2, 3])
        self.assertEqual(len({tuple(subset) for subset in result}), 8)
        self.assertEqual(sum(len(subset) for subset in result), 12)
        result[0].append(99)
        self.assertNotIn(99, result[1])

    def test_dp_against_all_valid_choices(self):
        for size in range(6):
            for values in itertools.product((0, 2, 5), repeat=size):
                expected = max((sum(value for index, value in enumerate(values) if mask >> index & 1)
                                for mask in range(1 << size) if not mask & (mask << 1)), default=0)
                self.assertEqual(PATTERNS.rob_memo(values), expected)
                self.assertEqual(PATTERNS.rob_iterative(values), expected)

    def test_intervals_preserve_input(self):
        intervals = [[2, 4], [1, 3], [6, 6], [6, 8]]
        self.assertEqual(PATTERNS.merge_intervals(intervals), [[1, 4], [6, 8]])
        self.assertEqual(intervals, [[2, 4], [1, 3], [6, 6], [6, 8]])
        self.assertEqual(PATTERNS.merge_intervals([]), [])

    def test_greedy_against_reachability(self):
        self.assertFalse(PATTERNS.can_jump([]))
        for size in range(1, 7):
            for jumps in itertools.product((0, 1, 2), repeat=size):
                reachable = {0}
                for index in range(size):
                    if index in reachable:
                        reachable.update(range(index + 1, min(size, index + jumps[index] + 1)))
                self.assertEqual(PATTERNS.can_jump(jumps), size - 1 in reachable)

    def test_note_templates_match_tested_functions(self):
        import re
        source = ast.parse((ROOT / "interview/code/patterns.py").read_text())
        expected = {node.name: ast.dump(node, include_attributes=False) for node in source.body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        checked = set()
        for page in (ROOT / "interview/algorithms").glob("*.md"):
            for snippet in re.findall(r"```python\n(.*?)\n```", page.read_text(), re.S):
                for node in ast.parse(snippet).body:
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in expected:
                        self.assertEqual(ast.dump(node, include_attributes=False), expected[node.name], str(page))
                        checked.add(node.name)
        self.assertEqual(checked, set(expected))

    @unittest.skipUnless(shutil.which("node"), "Node needed for traversal lab parity")
    def test_traversal_lab_matches_python(self):
        script = """
const lab = require(process.argv[1]);
const results = Object.entries(lab.graphs).map(([name, graph]) => ({
  name, graph, dfs: lab.traceGraph(graph, 'dfs'), bfs: lab.traceGraph(graph, 'bfs')
}));
process.stdout.write(JSON.stringify(results));
"""
        results = json.loads(subprocess.check_output([
            shutil.which("node"), "-e", script, str(ROOT / "site/static/traversal-lab.js")], text=True))
        for result in results:
            self.assertEqual(result["dfs"][-1]["order"], PATTERNS.dfs_recursive(result["graph"], "A"))
            self.assertEqual(result["bfs"][-1]["distances"], PATTERNS.bfs_distances(result["graph"], "A"))
            for mode in ("dfs", "bfs"):
                self.assertEqual(result[mode][-1]["frontier"], [])
                for state in result[mode]:
                    self.assertEqual(len(state["order"]), len(set(state["order"])))
                    self.assertEqual(len(state["frontier"]), len(set(state["frontier"])))
                    self.assertTrue(set(state["frontier"]).issubset(state["seen"]))

    def test_bilingual_code_blocks_match(self):
        import re
        for chinese in (ROOT / "interview/algorithms").glob("*.md"):
            if chinese.name.endswith(".en.md"):
                continue
            english = chinese.with_suffix(".en.md")
            pattern = r"```python\n(.*?)\n```"
            self.assertEqual(re.findall(pattern, chinese.read_text(), re.S),
                             re.findall(pattern, english.read_text(), re.S), str(chinese))


if __name__ == "__main__":
    unittest.main()
