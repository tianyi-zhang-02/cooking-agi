import bisect
import importlib.util
from pathlib import Path
import re
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("prep_checks", ROOT / "interview/code/prep_checks.py")
PREP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREP)
sys.path.insert(0, str(ROOT / "site"))
import build
import paritycheck


class InterviewPreparationTests(unittest.TestCase):
    def test_knn_modes_and_blocks_agree(self):
        generator = np.random.default_rng(19)
        points = generator.integers(-3, 4, size=(23, 5))
        queries = generator.integers(-3, 4, size=(11, 5))
        for count in (1, 4, len(points)):
            expected = PREP.nearest_indices(points, queries, count, mode="loop")
            np.testing.assert_array_equal(PREP.nearest_indices(points, queries, count, mode="vectorized"), expected)
            for block_size in (1, 3, 40):
                np.testing.assert_array_equal(PREP.nearest_indices(points, queries, count, block_size=block_size), expected)

    def test_knn_ties_votes_and_empty_queries(self):
        points = [[0, 0], [2, 0], [0, 0]]
        queries = [[1, 0]]
        np.testing.assert_array_equal(PREP.nearest_indices(points, queries, 3), [[0, 1, 2]])
        np.testing.assert_array_equal(PREP.knn_predict(points, [2, 1, 2], queries, 2), [1])
        np.testing.assert_array_equal(PREP.knn_predict(points, [2, 1, 2], queries, 3), [2])
        self.assertEqual(PREP.nearest_indices(points, np.empty((0, 2)), 1).shape, (0, 1))
        self.assertEqual(PREP.knn_predict(points, [0, 1, 0], np.empty((0, 2)), 1).shape, (0,))

    def test_knn_invalid_contracts(self):
        for count in (0, 4, True, 1.5):
            with self.assertRaises(ValueError):
                PREP.nearest_indices([[0], [1], [2]], [[1]], count)
        for points, queries in (([], [[0]]), ([[0]], [[0, 1]]), ([[np.nan]], [[0]]), ([[]], [[]])):
            with self.assertRaises(ValueError):
                PREP.nearest_indices(points, queries, 1)
        with self.assertRaises(ValueError):
            PREP.nearest_indices([[0]], [[0]], 1, block_size=0)
        with self.assertRaises(ValueError):
            PREP.nearest_indices([[0]], [[0]], 1, mode="unknown")
        with self.assertRaises(ValueError):
            PREP.knn_predict([[0]], [1.2], [[0]], 1)

    def test_binary_boundaries_match_standard_library(self):
        generator = np.random.default_rng(21)
        for size in range(30):
            values = sorted(generator.integers(-4, 5, size=size).tolist())
            for target in range(-6, 7):
                self.assertEqual(PREP.search_bounds(values, target),
                                 (bisect.bisect_left(values, target), bisect.bisect_right(values, target) - 1))

    def test_dependencies_layers_and_deduplication(self):
        graph = {"read": [], "clean": ["read", "read"], "audit": ["read"],
                 "train": ["clean", "audit"], "report": []}
        layers = PREP.dependency_layers(graph)
        self.assertEqual(set(layers[0]), {"read", "report"})
        self.assertEqual(set(layers[1]), {"clean", "audit"})
        self.assertEqual(layers[2], ["train"])
        positions = {task: index for index, layer in enumerate(layers) for task in layer}
        self.assertEqual(set(positions), set(graph))
        for task, predecessors in graph.items():
            self.assertTrue(all(positions[predecessor] < positions[task] for predecessor in predecessors))
        self.assertEqual(PREP.dependency_layers({}), [])

    def test_dependencies_reject_cycles_and_undeclared_tasks(self):
        for graph in ({"read": ["read"]}, {"read": ["train"], "train": ["read"]},
                      {"read": [], "train": ["typo"]}):
            with self.assertRaises(ValueError):
                PREP.dependency_layers(graph)

    def test_markov_results_and_periodic_counterexample(self):
        after_two, stationary, hitting = PREP.markov_example()
        transition = np.array([[0.8, 0.2], [0.4, 0.6]])
        np.testing.assert_allclose(after_two, [0.72, 0.28])
        np.testing.assert_allclose(stationary, [2 / 3, 1 / 3])
        np.testing.assert_allclose(stationary @ transition, stationary)
        self.assertAlmostEqual(hitting, 5)
        self.assertAlmostEqual(hitting, 1 + 0.8 * hitting)
        alternating = np.array([[0, 1], [1, 0]])
        np.testing.assert_allclose(np.array([0.5, 0.5]) @ alternating, [0.5, 0.5])
        np.testing.assert_array_equal(np.array([1, 0]) @ np.linalg.matrix_power(alternating, 9), [0, 1])

    def test_bilingual_shapes_anchors_and_examples(self):
        for chapter in ("interview/ai-roles-prep", "interview/coding-checks", "06-systems/inference-frameworks"):
            chinese = (ROOT / f"{chapter}.md").read_text()
            english = (ROOT / f"{chapter}.en.md").read_text()
            self.assertEqual(paritycheck.shape(chinese), paritycheck.shape(english), chapter)
            self.assertEqual(re.findall(r"\{#([^}]+)\}", chinese), re.findall(r"\{#([^}]+)\}", english))
            examples = re.findall(r"```python\n(.*?)```", chinese, re.S)
            self.assertEqual(examples, re.findall(r"```python\n(.*?)```", english, re.S))
            for example in examples:
                exec(compile(example, chapter, "exec"), {})

    def test_new_notes_are_published_once_in_interview_group(self):
        previous_nav, previous_sources = build.NAV.copy(), build.BY_SRC.copy()
        try:
            pages, _ = build.discover(build.load_nav())
            for name in ("ai-roles-prep", "coding-checks"):
                for suffix in ("", ".en"):
                    matches = [page for page in pages if page.src == ROOT / f"interview/{name}{suffix}.md"]
                    self.assertEqual(len(matches), 1)
                    self.assertEqual(matches[0].section["group"], "coding")
        finally:
            build.NAV.clear()
            build.NAV.update(previous_nav)
            build.BY_SRC.clear()
            build.BY_SRC.update(previous_sources)

    def test_sources_and_scope_are_preserved(self):
        for suffix in ("", ".en"):
            text = (ROOT / f"interview/ai-roles-prep{suffix}.md").read_text()
            for url in ("https://reyzharkov.com/blog/posts/interviews-2025-ml-research-engineer-uk",
                        "https://alisawuffles.github.io/blog/job-search/"):
                self.assertIn(f"]({url})", text)
            # Site rule (2026-10-15): no company-specific hiring-process content.
            for url in ("1point3acres.com", "openai.com/interview-guide"):
                self.assertNotIn(url, text)
            self.assertNotIn("bytebytego", text.lower())
            self.assertIn("RMSNorm", text)
            self.assertIn("GPT-2", text)

    def test_engine_maps_are_complete_and_localized(self):
        for suffix, language in (("", "zh-CN"), (".en", "en")):
            text = (ROOT / f"06-systems/inference-frameworks{suffix}.md").read_text()
            diagram = re.search(r'<figure class="engine-map".*?</figure>', text, re.S).group()
            self.assertIn(f'lang="{language}"', diagram)
            self.assertEqual(diagram.count("<section "), 2)
            self.assertEqual(diagram.count("<li>"), 8)
            for token in ("Scheduler", "ModelRunner", "DetokenizerManager", "GPU", "KV"):
                self.assertIn(token, diagram)
            if suffix:
                self.assertNotRegex(diagram, r"[\u4e00-\u9fff]")


if __name__ == "__main__":
    unittest.main()
