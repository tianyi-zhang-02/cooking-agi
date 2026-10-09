import contextlib
import importlib.util
import io
import math
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "00-foundations/code/tokenizer_from_scratch.py"
SPEC = importlib.util.spec_from_file_location("tokenizer_from_scratch", SOURCE)
TOKENIZER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOKENIZER
SPEC.loader.exec_module(TOKENIZER)


class MechanismExampleTests(unittest.TestCase):
    def test_bpe_keeps_base_symbols_after_merging(self):
        tokenizer = TOKENIZER.TinyBPE()
        tokenizer.train(["aa aa"], num_merges=10)
        self.assertTrue({"▁", "a", "aa", "▁aa"}.issubset(tokenizer.token_to_id))
        self.assertNotIn(tokenizer.token_to_id["<unk>"], tokenizer.encode("a"))

    def test_bpe_handles_new_combinations_of_known_symbols(self):
        tokenizer = TOKENIZER.TinyBPE()
        tokenizer.train(["rain rain rain rail rail"], num_merges=4)
        self.assertNotIn(tokenizer.token_to_id["<unk>"], tokenizer.encode("rair"))
        self.assertEqual(tokenizer.decode_tokens(tokenizer.tokenize("rair")), "rair")

    def test_character_bpe_does_not_claim_universal_coverage(self):
        tokenizer = TOKENIZER.TinyBPE()
        tokenizer.train(["rain rain"], num_merges=4)
        self.assertIn(tokenizer.token_to_id["<unk>"], tokenizer.encode("雨"))

    def test_retraining_replaces_the_vocabulary(self):
        tokenizer = TOKENIZER.TinyBPE()
        tokenizer.train(["aa aa"], num_merges=4)
        tokenizer.train(["bb bb"], num_merges=4)
        self.assertNotIn("a", tokenizer.token_to_id)
        self.assertEqual(tokenizer.encode("bb"), [tokenizer.token_to_id["▁bb"]])

    def test_bilingual_python_examples_match_and_run(self):
        for chapter in (
            "00-foundations/deep-dives/tokenizer-algorithms",
            "00-foundations/deep-dives/kv-cache-and-inference",
            "05-post-training/distillation",
        ):
            with self.subTest(chapter=chapter):
                snippets = []
                for suffix in (".md", ".en.md"):
                    text = (ROOT / (chapter + suffix)).read_text()
                    snippets.append(re.findall(r"```python\n(.*?)```", text, re.S))
                self.assertEqual(snippets[0], snippets[1])
                for snippet in snippets[0]:
                    with contextlib.redirect_stdout(io.StringIO()):
                        exec(compile(snippet, chapter, "exec"), {})

    def test_distillation_values_and_documented_rounding(self):
        teacher = [0.60, 0.35, 0.05]
        student = [0.40, 0.40, 0.20]
        loss = -sum(prob * math.log(pred) for prob, pred in zip(teacher, student))
        divergence = sum(prob * math.log(prob / pred) for prob, pred in zip(teacher, student))
        for suffix in (".md", ".en.md"):
            text = (ROOT / ("05-post-training/distillation" + suffix)).read_text()
            self.assertIn(f"approx{loss:.4f}", text)
            self.assertIn(f"approx{divergence:.4f}", text)

    def test_top_k_conditional_counterexample(self):
        first = [0.60, 0.35, 0.05]
        second = [0.30, 0.175, 0.525]
        self.assertNotEqual(first[-1], second[-1])
        for index in range(2):
            self.assertAlmostEqual(first[index] / sum(first[:2]), second[index] / sum(second[:2]))

    def test_lora_and_cache_arithmetic(self):
        self.assertEqual(8 * 4096 + 4096 * 8, 65536)
        self.assertEqual(65536 / (4096 * 4096) * 100, 0.390625)
        self.assertEqual(2 * 32 * 8192 * 8 * 128 * 2, 2**30)
        projected = sum(weight * value for weight, value in zip([1, 0, -1], [3, 1, 1]))
        self.assertEqual([2 * projected, -projected], [4, -2])


if __name__ == "__main__":
    unittest.main()
