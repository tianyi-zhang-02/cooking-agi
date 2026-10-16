from contextlib import redirect_stdout
import importlib.util
from io import StringIO
from pathlib import Path
import random
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "site"))
import build
import paritycheck

SPEC = importlib.util.spec_from_file_location(
    "comprehension_lab", ROOT / "interview/code/comprehension_lab.py")
LAB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAB)
CHAPTERS = ("speedrun", "speedrun-start", "code-comprehension")


class ComprehensionLabTests(unittest.TestCase):
    def test_versions_are_distinct_and_repeated_calls_hit(self):
        calls = []

        def encode(version, text):
            calls.append((version, text))
            return [len(text), int(version[-1])]

        cache = LAB.EmbeddingCache(encode)
        self.assertEqual(cache.get("v1", "cat"), [3, 1])
        self.assertEqual(cache.get("v2", "cat"), [3, 2])
        self.assertEqual(cache.get("v1", "cat"), [3, 1])
        self.assertEqual(calls, [("v1", "cat"), ("v2", "cat")])

    def test_empty_values_are_cached(self):
        calls = []

        def encode(version, text):
            calls.append((version, text))
            return []

        cache = LAB.EmbeddingCache(encode)
        self.assertEqual(cache.get("v1", ""), [])
        self.assertEqual(cache.get("v1", ""), [])
        self.assertEqual(len(calls), 1)

    def test_copies_isolate_encoder_and_caller_mutations(self):
        encoded = [1.0, 2.0]
        cache = LAB.EmbeddingCache(lambda version, text: encoded)
        returned = cache.get("v1", "cat")
        encoded[0] = 999
        returned[1] = 999
        self.assertEqual(cache.get("v1", "cat"), [1.0, 2.0])
        self.assertIsNot(cache.get("v1", "cat"), cache.get("v1", "cat"))

    def test_encoder_failure_does_not_cache_a_result(self):
        attempts = []

        def encode(version, text):
            attempts.append(text)
            if len(attempts) == 1:
                raise RuntimeError("Temporary failure")
            return [1.0]

        cache = LAB.EmbeddingCache(encode)
        with self.assertRaises(RuntimeError):
            cache.get("v1", "cat")
        self.assertEqual(cache.get("v1", "cat"), [1.0])
        self.assertEqual(len(attempts), 2)

    def test_buggy_version_has_all_three_demonstrated_failures(self):
        cache = LAB.BuggyEmbeddingCache(
            lambda version, text: [len(text), int(version[-1])])
        first = cache.get("v1", "cat")
        self.assertEqual(cache.get("v2", "cat"), [3, 1])
        first[0] = 99
        self.assertEqual(cache.get("v1", "cat"), [99, 1])
        calls = []

        def encode(version, text):
            calls.append(text)
            return []

        empty_cache = LAB.BuggyEmbeddingCache(encode)
        empty_cache.get("v1", "cat")
        empty_cache.get("v1", "cat")
        self.assertEqual(len(calls), 2)

    def test_token_mean_uses_the_intended_denominator(self):
        batches = [[2.0, 2.0], [8.0]]
        self.assertEqual(LAB.buggy_token_mean(batches), 5.0)
        self.assertEqual(LAB.token_mean(batches), 4.0)
        self.assertEqual(LAB.token_mean([[], [0.0], []]), 0.0)
        for batches in ([], [[], []]):
            with self.subTest(batches=batches), self.assertRaises(ValueError):
                LAB.token_mean(batches)

    def test_token_mean_is_invariant_to_batch_partitioning(self):
        generator = random.Random(29)
        losses = [generator.random() * 10 for index in range(41)]
        expected = sum(losses) / len(losses)
        for width in range(1, len(losses) + 1):
            batches = [losses[start:start + width]
                       for start in range(0, len(losses), width)]
            self.assertAlmostEqual(LAB.token_mean(batches), expected)


class SpeedrunContentTests(unittest.TestCase):
    def test_bilingual_structure_anchors_and_code_match(self):
        for chapter in CHAPTERS:
            with self.subTest(chapter=chapter):
                chinese, english = [(ROOT / f"interview/{chapter}{suffix}.md").read_text()
                                    for suffix in ("", ".en")]
                self.assertEqual(paritycheck.shape(chinese), paritycheck.shape(english))
                self.assertEqual(re.findall(r"\{#([^}]+)\}", chinese),
                                 re.findall(r"\{#([^}]+)\}", english))
                self.assertEqual(re.findall(r"```python\n(.*?)```", chinese, re.S),
                                 re.findall(r"```python\n(.*?)```", english, re.S))
                self.assertEqual(set(re.findall(r"https://[^)\s\"]+", chinese)),
                                 set(re.findall(r"https://[^)\s\"]+", english)))
                for source in (chinese, english):
                    self.assertFalse(paritycheck.source_issues(source))
                    body, _ = build.render_markdown(source)
                    self.assertNotIn('markdown="1"', body)
                    self.assertNotIn("\n## ", body)
                    for anchor in re.findall(r"\{#([^}]+)\}", source):
                        self.assertIn(f'id="{anchor}"', body)

    def test_beginner_examples_execute_and_match_the_numbers(self):
        source = (ROOT / "interview/speedrun-start.md").read_text()
        blocks = re.findall(r"```python\n(.*?)```", source, re.S)
        self.assertEqual(len(blocks), 2)
        namespace = {}
        captured = StringIO()
        with redirect_stdout(captured):
            for block in blocks:
                exec(compile(block, "speedrun-start.md", "exec"), namespace)
        self.assertEqual(captured.getvalue().splitlines(), ["2.5", "1.5 0.625"])
        self.assertEqual(namespace["gradient"], -5.0)
        self.assertEqual(namespace["weight"], 1.5)
        nearby_loss = namespace["mean_squared_error"](
            namespace["predict"]([1.0, 2.0], 1.01), [2.0, 4.0])
        self.assertAlmostEqual(nearby_loss, 2.45025)
        offset = 1e-5
        upper_loss = namespace["mean_squared_error"](
            namespace["predict"]([1.0, 2.0], 1.0 + offset), [2.0, 4.0])
        lower_loss = namespace["mean_squared_error"](
            namespace["predict"]([1.0, 2.0], 1.0 - offset), [2.0, 4.0])
        self.assertAlmostEqual((upper_loss - lower_loss) / (2 * offset), namespace["gradient"])
        predictions = namespace["predict"]([1.0, 2.0], 6.0)
        self.assertEqual(namespace["mean_squared_error"](predictions, [2.0, 4.0]), 40.0)
        for predictions, targets in (([], []), ([1], [1, 2])):
            with self.assertRaises(ValueError):
                namespace["mean_squared_error"](predictions, targets)

    def test_small_example_table_keeps_semantics_in_both_languages(self):
        for suffix in ("", ".en"):
            source = (ROOT / f"interview/speedrun-start{suffix}.md").read_text()
            body, _ = build.render_markdown(source)
            compact = re.search(r'<div class="compact-table">(.*?)</table>', body, re.S)
            self.assertIsNotNone(compact)
            self.assertNotIn('markdown="1"', body)
            self.assertEqual(compact.group(1).count('<th>'), 4)
            self.assertEqual(re.findall(r'<td>(\d+)</td>', compact.group(1)),
                             ['1', '2', '1', '1', '2', '4', '2', '4'])

    def test_code_comprehension_examples_are_executable(self):
        source = (ROOT / "interview/code-comprehension.md").read_text()
        blocks = re.findall(r"```python\n(.*?)```", source, re.S)
        self.assertEqual(len(blocks), 4)
        namespace = {}
        with redirect_stdout(StringIO()):
            for block in blocks:
                exec(compile(block, "code-comprehension.md", "exec"), namespace)
        self.assertEqual(namespace["first"], [3, 1])
        self.assertEqual(namespace["second"], [3, 1])
        cache = namespace["EmbeddingCache"](namespace["encode"])
        self.assertEqual(cache.get("v2", "cat"), [3, 2])
        self.assertEqual(namespace["token_mean"]([[2, 2], [8]]), 4)

    def test_suggested_time_blocks_add_up_in_both_languages(self):
        for chapter in ("speedrun", "speedrun-start"):
            for suffix in ("", ".en"):
                source = (ROOT / f"interview/{chapter}{suffix}.md").read_text()
                totals = []
                for table in re.findall(r"(?:^\|[^\n]+\n)+", source, re.M):
                    minutes = re.findall(r"^\| (\d+) (?:分钟|minutes) \|", table, re.M)
                    if minutes:
                        totals.append(sum(map(int, minutes)))
                self.assertEqual(totals, [120, 240, 360], (chapter, suffix))

    def test_entry_points_and_unpressured_routes_remain_available(self):
        navigation = (ROOT / "site/nav.toml").read_text()
        for chapter in CHAPTERS:
            self.assertIn(f'"{chapter}.md"', navigation)
        for suffix in ("", ".en"):
            main = (ROOT / f"interview/speedrun{suffix}.md").read_text()
            self.assertIn(f'speedrun-start{suffix}.md', main)
            self.assertIn('href="#two-hours"', main)
            self.assertIn(f'code-comprehension{suffix}.md', main)
            self.assertIn("有效学习时间" if not suffix else "focused learning time", main)
            self.assertIn("LeetCode", main)
            landing = (ROOT / f"interview/README{suffix}.md").read_text()
            self.assertIn(f'speedrun{suffix}.md', landing)
            self.assertIn(f'code-comprehension{suffix}.md', landing)

    def test_interview_policy_claims_keep_evidence_and_scope(self):
        for suffix in ("", ".en"):
            source = (ROOT / f"interview/code-comprehension{suffix}.md").read_text()
            # Site rule (2026-10-15): no company-specific hiring-process content.
            self.assertNotIn("https://www.businessinsider.com/", source)
            self.assertNotIn("deepmind-media/", source)
            self.assertIn("https://google.github.io/eng-practices/", source)
            body, _ = build.render_markdown(source)
            self.assertEqual(body.count("<details>"), 2)


if __name__ == "__main__":
    unittest.main()
