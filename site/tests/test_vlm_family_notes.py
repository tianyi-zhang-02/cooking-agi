import math
from pathlib import Path
import re
import tomllib
import unittest
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "03-multimodal-learning"
CHAPTERS = ("vlm-designs", "qwen-vl", "omni-streaming", "ocr-compression",
            "image-generation", "vlm-finetuning")


def snippets(chapter, suffix=".md"):
    return re.findall(r"```python\n(.*?)```", (FOLDER / (chapter + suffix)).read_text(), re.S)


def example_namespace(chapter):
    namespace = {}
    for snippet in snippets(chapter):
        exec(compile(snippet, chapter, "exec"), namespace)
    return namespace


class VlmFamilyNotesTests(unittest.TestCase):
    def test_bilingual_examples_match_and_run(self):
        for chapter in CHAPTERS:
            with self.subTest(chapter=chapter):
                self.assertEqual(snippets(chapter), snippets(chapter, ".en.md"))
                example_namespace(chapter)

    def test_vl2_includes_structural_tokens(self):
        count = example_namespace("vlm-designs")["vl2_sequence_length"]
        self.assertEqual(count(2, 3), 1415)
        self.assertEqual(count(1, 1), 421)
        self.assertEqual(count(2, 3) - 7 * 196, 43)
        for rows, columns in ((0, 1), (1, -1), (1.5, 2), (True, 2)):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                count(rows, columns)

    def test_dynamic_grid_is_not_fixed_length(self):
        count = example_namespace("qwen-vl")["merged_image_tokens"]
        self.assertEqual(count(280, 280), 100)
        self.assertEqual(count(280, 560), 200)
        self.assertEqual(count(560, 560), 400)
        for height, width in ((281, 560), (0, 280), (-280, 280), (True, 280)):
            with self.subTest(height=height), self.assertRaises(ValueError):
                count(height, width)

    def test_audio_frames_have_explicit_boundary_convention(self):
        count = example_namespace("omni-streaming")["frame_count"]
        self.assertEqual(count(32000, 400, 160), 198)
        self.assertEqual(count(399, 400, 160), 0)
        self.assertEqual(count(400, 400, 160), 1)
        self.assertEqual(count(560, 400, 160), 2)
        for values in ((-1, 400, 160), (10, 0, 1), (10, 1, 0), (10.5, 1, 1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                count(*values)

    def test_flow_direction_loss_and_euler_step(self):
        calculate = example_namespace("image-generation")["linear_flow_example"]
        self.assertEqual(calculate(2, 10, 0.25, 6, 0.1), (4.0, 8, 4, 4.6))
        self.assertEqual(calculate(2, 10, 0, 8, 1), (2, 8, 0, 10))
        self.assertEqual(calculate(10, 2, 0, -8, 1), (10, -8, 0, 2))
        for time, step in ((-0.1, 0), (0.9, 0.2), (0.5, -0.1)):
            with self.subTest(time=time), self.assertRaises(ValueError):
                calculate(2, 10, time, 6, step)

    def test_masks_preserve_answer_and_eos_only(self):
        namespace = example_namespace("vlm-finetuning")
        self.assertEqual(namespace["labels"], [-100, -100, -100, -100, 40, 2, -100])
        self.assertEqual(sum(label != -100 for label in namespace["labels"][1:]), 2)
        calculate = namespace["answer_labels"]
        self.assertEqual(calculate([1, 99, 7], [False, True, True],
                                   [True, True, True], [False, True, False]), [-100, -100, 7])
        self.assertEqual(calculate([1, 7, 0], [False, True, True],
                                   [True, True, False], [False, False, False]), [-100, 7, -100])

    def test_masks_reject_empty_or_misaligned_targets(self):
        calculate = example_namespace("vlm-finetuning")["answer_labels"]
        examples = [([], [], [], []), ([1], [True], [True], [False]),
                    ([1, 2], [False], [True, True], [False, False]),
                    ([1, 2], [False, 1], [True, True], [False, False]),
                    ([1, 2], [False, True], [True, True], [False, True])]
        for values in examples:
            with self.subTest(values=values), self.assertRaises(ValueError):
                calculate(*values)

    def test_budget_examples(self):
        self.assertAlmostEqual((3000 / 696) ** 2, 18.58, delta=0.01)
        self.assertAlmostEqual(300 / 2100, 1 / 7)
        self.assertEqual(160 + 30 + 90 + 20 + 40, 340)
        self.assertEqual(1000 / 6.25, 160)

    def test_notes_are_navigable_and_local_links_resolve(self):
        nav = tomllib.loads((ROOT / "site/nav.toml").read_text())
        section = next(item for item in nav["section"] if item["dir"] == "03-multimodal-learning")
        for chapter in CHAPTERS:
            self.assertIn(chapter + ".md", section["order"])
            self.assertEqual(len(nav["label"][f"03-multimodal-learning/{chapter}.md"]), 2)
            for suffix in (".md", ".en.md"):
                source = FOLDER / (chapter + suffix)
                for link in re.findall(r"\]\(([^)]+)\)", source.read_text()):
                    parsed = urlsplit(link)
                    if parsed.scheme or not parsed.path:
                        continue
                    target = (source.parent / parsed.path).resolve()
                    self.assertTrue(target.exists(), f"{source}: {link}")
                    if suffix == ".en.md" and target.suffix == ".md" and target != FOLDER / (chapter + ".md"):
                        self.assertTrue(target.name.endswith(".en.md"), link)

    def test_2026_updates_link_primary_sources(self):
        sources = {"ocr-compression": "2601.20552", "omni-streaming": "2609.25611",
                   "image-generation": "2605.10730", "qwen-vl": "Qwen3.5-397B-A17B"}
        for chapter, identifier in sources.items():
            for suffix in (".md", ".en.md"):
                self.assertIn(identifier, (FOLDER / (chapter + suffix)).read_text())


if __name__ == "__main__":
    unittest.main()
