import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "site/static/recsys-lab.js"
SPEC = importlib.util.spec_from_file_location(
    "acceptance_retrieval", ROOT / "practice/recommender-systems/code/two_tower_reference.py")
RETRIEVAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RETRIEVAL
SPEC.loader.exec_module(RETRIEVAL)


class RecommenderAcceptanceTests(unittest.TestCase):
    def run_js(self, code):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js is required to verify the browser experiment")
        result = subprocess.run(
            [node, "-e", "const lab = require(process.argv[1]);\n" + code, str(SCRIPT)],
            capture_output=True, text=True, check=True,
        )
        return json.loads(result.stdout)

    def test_cosine_is_stable_at_extreme_finite_scales(self):
        scores = self.run_js("""
const result = [];
for (const leftScale of [1, 1e200, 1e-200, Number.MAX_VALUE, Number.MIN_VALUE]) {
  for (const rightScale of [1, 1e200, 1e-200, Number.MAX_VALUE, Number.MIN_VALUE]) {
    result.push(lab.cosine([leftScale, leftScale], [rightScale, 0]));
    result.push(lab.cosine([-leftScale, -leftScale], [rightScale, 0]));
  }
}
console.log(JSON.stringify(result));
""")
        self.assertEqual(len(scores), 50)
        for index, score in enumerate(scores):
            self.assertIsNotNone(score)
            self.assertAlmostEqual(score, (-1 if index % 2 else 1) / math.sqrt(2))

    def test_reranking_matches_an_independent_oracle_for_all_ui_controls(self):
        payload = self.run_js("""
const original = JSON.stringify(lab.candidates);
const cases = [];
for (let step = 0; step <= 20; step += 1) {
  for (const uniqueAuthors of [false, true]) {
    const penalty = step / 20;
    const selected = lab.rerank(lab.candidates, penalty, uniqueAuthors);
    cases.push({penalty, uniqueAuthors, selected, stats: lab.summarize(selected)});
  }
}
console.log(JSON.stringify({candidates: lab.candidates, cases,
  unchanged: JSON.stringify(lab.candidates) === original}));
""")
        self.assertTrue(payload["unchanged"])

        def cosine(left, right):
            return sum(first * second for first, second in zip(left, right)) / (
                math.sqrt(sum(value ** 2 for value in left)) *
                math.sqrt(sum(value ** 2 for value in right)))

        for case in payload["cases"]:
            with self.subTest(penalty=case["penalty"], authors=case["uniqueAuthors"]):
                remaining = {item["id"]: item for item in payload["candidates"]}
                chosen = []
                for position in range(3):
                    gains = []
                    for item in remaining.values():
                        if case["uniqueAuthors"] and any(
                                prior["author"] == item["author"] for prior in chosen):
                            continue
                        similarity = max([0] + [cosine(item["vector"], prior["vector"])
                                                for prior in chosen])
                        gains.append((item["score"] - case["penalty"] * similarity, item["id"]))
                    gain, identifier = min(gains, key=lambda pair: (-pair[0], pair[1]))
                    chosen.append(remaining.pop(identifier))
                    self.assertEqual(case["selected"][position]["id"], identifier)
                    self.assertAlmostEqual(case["selected"][position]["gain"], gain)
                distances = [1 - cosine(left["vector"], right["vector"])
                             for index, left in enumerate(chosen) for right in chosen[index + 1:]]
                self.assertAlmostEqual(case["stats"]["ild"], sum(distances) / len(distances))
                self.assertAlmostEqual(case["stats"]["mean"], sum(item["score"] for item in chosen) / 3)
                self.assertEqual(case["stats"]["topics"], len({item["topic"] for item in chosen}))

    def test_reranking_short_list_and_singleton_metrics(self):
        result = self.run_js("""
const sameAuthor = lab.candidates.map(item => ({...item, author: 'one'}));
console.log(JSON.stringify({empty: lab.summarize(lab.rerank([], .5, true)),
  singleton: lab.summarize(lab.rerank(sameAuthor, .5, true)),
  ids: lab.rerank(sameAuthor, .5, true).map(item => item.id)}));
""")
        self.assertEqual(result["empty"], {"mean": None, "topics": 0, "ild": None})
        self.assertEqual(result["singleton"], {"mean": .95, "topics": 1, "ild": None})
        self.assertEqual(result["ids"], ["A"])

    def test_serving_weight_table_changes_order_without_changing_predictions(self):
        for suffix in (".md", ".en.md"):
            text = (ROOT / f"practice/recommender-systems/03-ranking-and-diversity{suffix}").read_text()
            rows = re.findall(r"^\| ([AB]) \| (0\.\d+) \| (0\.\d+) \| (0\.\d+) \| (0\.\d+) \|$",
                              text, re.M)
            self.assertEqual(len(rows), 2)
            scores = {}
            for identifier, like, rejection, mild, strong in rows:
                like, rejection, mild, strong = map(float, (like, rejection, mild, strong))
                self.assertAlmostEqual(like - 2 * rejection, mild)
                self.assertAlmostEqual(like - 5 * rejection, strong)
                scores[identifier] = (mild, strong)
            self.assertGreater(scores["A"][0], scores["B"][0])
            self.assertLess(scores["A"][1], scores["B"][1])

    def test_documented_task_means_and_global_denominators(self):
        for suffix in (".md", ".en.md"):
            text = (ROOT / f"practice/recommender-systems/03-ranking-and-diversity{suffix}").read_text()
            losses = [json.loads(value) for value in re.findall(r"`(\[0\.\d+, 0\.\d+, 0\.\d+\])`", text)]
            self.assertEqual(len(losses), 2)
            means = [sum(values) / len(values) for values in losses]
            self.assertAlmostEqual(means[0] + .5 * means[1], .55)
            self.assertIn("**0.55**", text)
            rank_means = [losses[0][0], sum(losses[0][1:]) / 2]
            self.assertNotAlmostEqual(sum(rank_means) / 2, means[0])
            self.assertAlmostEqual((rank_means[0] + 2 * rank_means[1]) / 3, means[0])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch verifies masked loss gradients")
    def test_unknown_labels_do_not_dilute_loss_or_contribute_gradients(self):
        import torch

        losses = torch.tensor([[.2, .1], [.6, .3], [.4, .5], [100, 100]],
                              dtype=torch.float64, requires_grad=True)
        masks = torch.tensor([[1, 1], [1, 1], [1, 1], [0, 0]], dtype=torch.float64)
        weights = torch.tensor([1, .5], dtype=torch.float64)
        counts = masks.sum(dim=0)
        objective = (((losses * masks).sum(dim=0) / counts) * weights).sum()
        self.assertAlmostEqual(objective.item(), .55)
        objective.backward()
        torch.testing.assert_close(losses.grad[-1], torch.zeros(2, dtype=torch.float64))
        torch.testing.assert_close(losses.grad[:3], (weights / 3).expand(3, 2))

    def test_generated_ids_distinguish_resolution_eligibility_and_uniqueness(self):
        for suffix in (".md", ".en.md"):
            text = (ROOT / f"practice/recommender-systems/04-modern-recsys{suffix}").read_text()
            emissions = re.search(r"`\[([A-Z?, ]+)\]`", text)[1].split(", ")
            resolved = [identifier for identifier in emissions if identifier in {"A", "B", "D"}]
            eligible = [identifier for identifier in resolved if identifier in {"A", "B"}]
            self.assertEqual(len(resolved) / len(emissions), 4 / 5)
            self.assertEqual(len(eligible) / len(emissions), 3 / 5)
            self.assertEqual(set(eligible), {"A", "B"})
            self.assertIn("4/5", text)
            self.assertIn("3/5", text)

    def test_query_scaling_preserves_order_but_not_absolute_thresholds(self):
        spec = RETRIEVAL.EmbeddingSpec("acceptance-v1", 2, "inner_product")
        items = {"A": (1, 0), "B": (1 / math.sqrt(2), 1 / math.sqrt(2))}
        baseline = RETRIEVAL.exact_top_k((1, 0), spec, items, spec, 2)
        scaled = RETRIEVAL.exact_top_k((2, 0), spec, items, spec, 2)
        self.assertEqual([item for item, score in baseline], [item for item, score in scaled])
        for original, changed in zip(baseline, scaled):
            self.assertAlmostEqual(changed[1], 2 * original[1])
        self.assertEqual([item for item, score in baseline if score >= .8], ["A"])
        self.assertEqual([item for item, score in scaled if score >= .8], ["A", "B"])


if __name__ == "__main__":
    unittest.main()
