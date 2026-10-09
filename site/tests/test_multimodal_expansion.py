import math
from pathlib import Path
import re
import tomllib
import unittest
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[2]
CHAPTERS = ("vit", "blip-and-q-former")
FOLDER = ROOT / "03-multimodal-learning"


def snippets(chapter, suffix=".md"):
    return re.findall(r"```python\n(.*?)```", (FOLDER / (chapter + suffix)).read_text(), re.S)


def namespace_for(chapter):
    namespace = {}
    for snippet in snippets(chapter):
        exec(compile(snippet, chapter, "exec"), namespace)
    return namespace


class MultimodalExpansionTests(unittest.TestCase):
    def test_bilingual_examples_match_and_run(self):
        for chapter in CHAPTERS:
            self.assertEqual(snippets(chapter), snippets(chapter, ".en.md"))
            namespace_for(chapter)

    def test_patch_order_and_projection(self):
        namespace = namespace_for("vit")
        self.assertEqual(namespace["patches"], [[1, 2, 5, 6], [3, 4, 7, 8],
                                                [9, 10, 13, 14], [11, 12, 15, 16]])
        self.assertEqual(namespace["patchify_gray"]([[1, 2, 3, 4], [5, 6, 7, 8]], 2),
                         [[1, 2, 5, 6], [3, 4, 7, 8]])

    def test_projection_collision_is_real(self):
        weights = namespace_for("vit")["projection_rows"]
        project = lambda patch: [sum(value * weight for value, weight in zip(patch, row))
                                 for row in weights]
        self.assertEqual(project([1, 2, 5, 6]), project([2, 3, 4, 5]))

    def test_patch_validation(self):
        patchify = namespace_for("vit")["patchify_gray"]
        for image, size in [([], 2), ([[]], 1), ([[1]], 0), ([[1]], 1.5),
                            ([[1, 2], [3]], 1), ([[1, 2, 3]], 2)]:
            with self.subTest(image=image, size=size), self.assertRaises(ValueError):
                patchify(image, size)

    def test_itc_isolates_modalities(self):
        mask = namespace_for("blip-and-q-former")["visibility_mask"]("itc", 2, 3)
        self.assertEqual(mask[:2], [[True, True, False, False, False]] * 2)
        self.assertEqual(mask[2:], [[False, False, True, True, True]] * 3)

    def test_itm_allows_full_fusion(self):
        mask = namespace_for("blip-and-q-former")["visibility_mask"]("itm", 2, 3)
        self.assertEqual(mask, [[True] * 5 for row in range(5)])

    def test_itg_hides_future_text_and_all_text_from_queries(self):
        build_mask = namespace_for("blip-and-q-former")["visibility_mask"]
        for query_count in (1, 2, 4):
            for text_count in (1, 3, 5):
                mask = build_mask("itg", query_count, text_count)
                for row, entries in enumerate(mask):
                    self.assertTrue(all(entries[:query_count]))
                    for column in range(query_count, query_count + text_count):
                        self.assertEqual(entries[column], row >= query_count and column <= row)

    def test_mask_validation(self):
        build_mask = namespace_for("blip-and-q-former")["visibility_mask"]
        for objective, queries, texts in [("unknown", 2, 3), ("itc", 0, 3), ("itg", 2, 0)]:
            with self.subTest(objective=objective), self.assertRaises(ValueError):
                build_mask(objective, queries, texts)

    def test_fixed_downstream_gradient(self):
        loss = lambda parameter: (2 * parameter * 3 - 1) ** 2
        delta = 1e-6
        derivative = (loss(0.5 + delta) - loss(0.5 - delta)) / (2 * delta)
        self.assertAlmostEqual(derivative, 24, places=6)

    def test_candidate_softmax_changes_without_score_changes(self):
        before = math.exp(2) / (math.exp(2) + 1)
        after = math.exp(2) / (2 * math.exp(2) + 1)
        self.assertAlmostEqual(before, 0.881, delta=0.001)
        self.assertAlmostEqual(after, 0.468, delta=0.001)

    def test_local_links_and_language(self):
        for chapter in (*CHAPTERS, "clip", "README"):
            for suffix in (".md", ".en.md"):
                source = FOLDER / (chapter + suffix)
                for target in re.findall(r"\]\(([^)]+)\)", source.read_text()):
                    parsed = urlsplit(target)
                    if parsed.scheme or not parsed.path:
                        continue
                    resolved = (source.parent / parsed.path).resolve()
                    self.assertTrue(resolved.exists(), f"{source}: {target}")
                    if suffix == ".en.md" and resolved.suffix == ".md" and resolved != FOLDER / (chapter + ".md"):
                        self.assertTrue(resolved.name.endswith(".en.md"), f"{source}: {target}")

    def test_navigation_and_coverage(self):
        nav = tomllib.loads((ROOT / "site/nav.toml").read_text())
        section = next(item for item in nav["section"] if item["dir"] == "03-multimodal-learning")
        self.assertLess(section["order"].index("vit.md"), section["order"].index("clip.md"))
        for chapter in CHAPTERS:
            name = f"03-multimodal-learning/{chapter}.md"
            self.assertIn(chapter + ".md", section["order"])
            self.assertEqual(len(nav["label"][name]), 2)
        coverage = tomllib.loads((ROOT / "site/reference-coverage.toml").read_text())
        chapter = next(item for item in coverage["chapter"] if item["en"] == "9 · Multimodal models")
        self.assertEqual(len(chapter["topics"]), 16)
        fine_tuning = next(item for item in chapter["topics"] if item["en"].startswith("Multimodal fine-tuning"))
        self.assertEqual(fine_tuning["state"], "article")
        self.assertEqual(fine_tuning["notes"], ["03-multimodal-learning/vlm-finetuning.md"])
        for suffix in (".md", ".en.md"):
            content = (FOLDER / ("vlm-finetuning" + suffix)).read_text()
            self.assertGreaterEqual(len(re.findall(r"^## ", content, re.M)), 8)
            self.assertIn("COCO", content)
            self.assertIn("processor", content)
            self.assertIn("GPU", content)


if __name__ == "__main__":
    unittest.main()
