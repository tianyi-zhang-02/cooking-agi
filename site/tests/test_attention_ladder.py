import importlib.util
import inspect
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("attention_ladder", ROOT / "00-foundations/code/attention_ladder.py")
ladder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ladder)

STEPS = ["mha", "causal_mha", "causal_gqa", "causal_mla", "causal_mla_absorbed"]
HELPERS = ["softmax", "split", "merge", "future"]


def without_comments(code):
    return [re.sub(r"\s+#.*$", "", line).rstrip() for line in code.strip().splitlines()]


class AttentionLadderTests(unittest.TestCase):
    """One attention five ways: every step agrees with the step before, and the notes show the script's code."""

    def test_every_step_agrees_with_the_one_before(self):
        results = ladder.checks()
        self.assertEqual(len(results), 6)
        for name, gap in results.items():
            with self.subTest(check=name):
                self.assertLess(gap, 1e-12)
        for seed in (1, 2, 3):                       # not one lucky draw
            self.assertLess(max(ladder.checks(seed=seed, T=7, D=48, h=6, g=3, dc=12).values()), 1e-12)

    def test_the_cache_numbers_in_the_notes(self):
        self.assertEqual(ladder.cache_per_token(), {"MHA": 64, "GQA": 32, "MLA": 16})
        for name in ("attention-ladder.md", "attention-ladder.en.md"):
            note = (ROOT / "learn/ml-exercises" / name).read_text(encoding="utf-8")
            for value in ("| 64 |", "| 32 |", "| 16 |"):
                self.assertIn(value, note)

    def test_the_notes_show_the_script_line_for_line(self):
        for name in ("attention-ladder.md", "attention-ladder.en.md"):
            note = (ROOT / "learn/ml-exercises" / name).read_text(encoding="utf-8")
            blocks = re.findall(r"```python\n(.*?)```", note, re.S)
            self.assertEqual(len(blocks), 6, name)
            helpers = "\n\n\n".join(inspect.getsource(getattr(ladder, f)).strip() for f in HELPERS)
            with self.subTest(note=name, part="helpers"):
                self.assertEqual(without_comments(blocks[0]), without_comments(helpers))
            for block, step in zip(blocks[1:], STEPS):
                with self.subTest(note=name, step=step):
                    self.assertEqual(without_comments(block), without_comments(inspect.getsource(getattr(ladder, step))))

    def test_the_notes_quote_the_checks_the_script_makes(self):
        results = list(ladder.checks().values())
        self.assertAlmostEqual(results[4], 6.7e-16, delta=0.5e-16)    # "about 7e-16"
        self.assertAlmostEqual(results[5], 5.8e-16, delta=0.5e-16)    # "about 6e-16"
        for name, words in (("attention-ladder.md", ("约 7e-16", "约 6e-16")), ("attention-ladder.en.md", ("about 7e-16", "about 6e-16"))):
            note = (ROOT / "learn/ml-exercises" / name).read_text(encoding="utf-8")
            for word in words:
                self.assertIn(word, note)


if __name__ == "__main__":
    unittest.main()
