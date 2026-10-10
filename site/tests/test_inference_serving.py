import ast
import contextlib
import io
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
MEMORY = "00-foundations/deep-dives/kv-cache-and-inference"
SERVING = "06-systems/llm-serving"
KERNELS = "00-foundations/deep-dives/attention-kernels"


def python_blocks(chapter, language=""):
    return re.findall(r"```python\n(.*?)```", (ROOT / f"{chapter}{language}.md").read_text(), re.S)


def namespace(chapter):
    values = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for block in python_blocks(chapter):
            exec(compile(block, chapter, "exec"), values)
    return values


class InferenceServingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.memory = namespace(MEMORY)
        cls.serving = namespace(SERVING)

    def test_bilingual_code_is_identical_and_valid(self):
        for chapter in (MEMORY, SERVING, KERNELS):
            self.assertEqual(python_blocks(chapter), python_blocks(chapter, ".en"))
            for block in python_blocks(chapter):
                ast.parse(block)

    def test_bilingual_figures_do_not_reuse_english_labels(self):
        for chapter in (MEMORY, SERVING):
            for language in ("", ".en"):
                source = (ROOT / f"{chapter}{language}.md").read_text()
                figures = re.findall(r"<figure.*?</figure>", source, re.S)
                self.assertTrue(figures)
                for figure in figures:
                    if language:
                        self.assertNotRegex(figure, r"[\u4e00-\u9fff]")
                    else:
                        self.assertRegex(figure, r"[\u4e00-\u9fff]")

    def test_pinned_implementation_links(self):
        for chapter in (SERVING, KERNELS):
            for language in ("", ".en"):
                source = (ROOT / f"{chapter}{language}.md").read_text()
                self.assertIn("10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf", source)
                self.assertNotIn("vllm-project/vllm/blob/main/", source)
                self.assertNotIn("sgl-project/sglang/blob/main/", source)

    def test_short_source_examples(self):
        self.assertEqual(self.memory["one_sequence"], 2**30)
        self.assertEqual(self.memory["eight_sequences"], 8 * 2**30)
        self.assertEqual(self.serving["prefill_chunks"](10, 2, 6), [4, 4, 2])
        self.assertEqual(self.serving["prefill_chunks"](10, 2, 12), [10])

    def test_page_rounding_against_explicit_slot_counts(self):
        calculate = self.memory["paged_kv_bytes"]
        for block_size in (1, 2, 4, 16):
            for first_length in range(18):
                for second_length in range(10):
                    slots = sum(len(range(0, length, block_size)) * block_size
                                for length in (first_length, second_length))
                    actual = calculate([first_length, second_length], layers=1,
                                       kv_heads=1, head_dim=1, bytes_per_value=1,
                                       block_tokens=block_size)
                    self.assertEqual(actual, 2 * slots)

    def test_unequal_lengths_have_tail_waste(self):
        calculate = self.memory["paged_kv_bytes"]
        self.assertEqual(calculate([5, 3, 8], 1, 1, 1, 1, 4), 40)
        self.assertEqual(calculate([], 1, 1, 1, 1, 4), 0)
        self.assertEqual(calculate(iter([4, 4])), calculate([4, 4]))

    def test_budget_and_units(self):
        self.assertAlmostEqual(14_000_000_000 / 2**30, 13.0385160446167)
        self.assertAlmostEqual(self.memory["kv_budget"] / 2**30, 6.9614839553833)
        calculate = self.memory["paged_kv_bytes"]
        self.assertEqual(int(self.memory["kv_budget"] // calculate([8192])), 6)
        self.assertEqual(int(self.memory["kv_budget"] // calculate([16384])), 3)

    def test_rank_partition_examples(self):
        calculate = self.memory["paged_kv_bytes"]
        full = calculate([8192])
        self.assertEqual(calculate([8192], kv_heads=4), full // 2)
        self.assertEqual(calculate([8192], layers=16), full // 2)
        self.assertEqual(calculate([8192, 8192]), 2 * full)

    def test_replicated_mqa_is_not_head_sharding(self):
        per_rank = self.memory["paged_kv_bytes"]([8192], kv_heads=1)
        self.assertEqual(per_rank, 2**27)
        self.assertEqual(8 * per_rank, 2**30)

    def test_memory_invalid_inputs(self):
        calculate = self.memory["paged_kv_bytes"]
        for lengths in ([-1], [1.5], [True]):
            with self.assertRaises(ValueError):
                calculate(lengths)
        for argument in ("layers", "kv_heads", "head_dim", "bytes_per_value", "block_tokens"):
            for value in (0, -1, 1.5, True):
                with self.assertRaises(ValueError):
                    calculate([1], **{argument: value})

    def test_chunks_conserve_prompt_and_respect_budget(self):
        split = self.serving["prefill_chunks"]
        for prompt in range(25):
            for decode in range(5):
                for budget in range(decode + 1, decode + 8):
                    chunks = split(prompt, decode, budget)
                    self.assertEqual(sum(chunks), prompt)
                    self.assertTrue(all(0 < chunk <= budget - decode for chunk in chunks))
                    self.assertEqual(len(chunks), len(range(0, prompt, budget - decode)))

    def test_no_prefill_capacity_is_explicit(self):
        split = self.serving["prefill_chunks"]
        for arguments in ((3, 2, 2), (3, 3, 2), (-1, 0, 4), (1, -1, 4),
                          (1, 0, 0), (1.5, 0, 3), (1, True, 3), (0, 3, 2)):
            with self.assertRaises(ValueError):
                split(*arguments)
        self.assertEqual(split(0, 2, 2), [])
