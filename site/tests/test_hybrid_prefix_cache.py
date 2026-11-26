import importlib.util
import inspect
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "hybrid_prefix_cache", ROOT / "00-foundations/code/hybrid_prefix_cache.py")
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)
NOTES = [ROOT / "00-foundations/deep-dives" / name
         for name in ("hybrid-prefix-cache.md", "hybrid-prefix-cache.en.md")]


def blocks(note):
    return re.findall(r"```python\n(.*?)```", note.read_text(encoding="utf-8"), re.S)


class HybridPrefixCacheTests(unittest.TestCase):
    """The prefix-cache note shows the script's code, and its examples hold."""

    def test_the_hand_examples(self):
        self.assertEqual(cache.checks(), {
            "concurrent_8k": 8, "concurrent_4k": 4, "concurrent_8k_no_overlap": 4,
            "warm": 4, "hold": 2, "hold_cap_3": 0, "hold_cap_4": 2,
        })

    def test_a_cap_within_the_free_slots_never_misses(self):
        prompts = list("ABCDEFGHIJ")
        for free in range(1, 6):
            for cap in range(1, free + 1):
                with self.subTest(free=free, cap=cap):
                    self.assertEqual(cache.queued_misses(prompts, free, "hold_cap", cap=cap), 0)
        # Without a cap, a pool smaller than the number of prompts loses some first samples.
        self.assertGreater(cache.queued_misses(prompts, 3, "warm"), 0)

    def test_the_notes_show_the_script_and_its_checks(self):
        sources = [inspect.getsource(cache.concurrent_misses).strip(),
                   inspect.getsource(cache.queued_misses).strip()]
        for note in NOTES:
            code = blocks(note)
            with self.subTest(note=note.name):
                self.assertEqual(len(code), 4)
                self.assertEqual(code[0].strip(), sources[0])
                self.assertEqual(code[2].strip(), sources[1])
                namespace = vars(cache).copy()
                exec(code[1], namespace)      # the asserts printed in the note must pass
                exec(code[3], namespace)

    def test_both_notes_quote_the_same_measurements(self):
        for note in NOTES:
            text = note.read_text(encoding="utf-8")
            for value in ("212", "592", "118", "169.5", "130.7", "243.5", "−26.7%",
                          "121–123", "388–455", "373,823", "142,739", "4.7%", "4.8k–5.9k"):
                with self.subTest(note=note.name, value=value):
                    self.assertIn(value, text)


if __name__ == "__main__":
    unittest.main()
