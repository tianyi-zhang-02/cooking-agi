import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "site"))
import paritycheck


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rag = load_module("practice_rag", "practice/rag/code/evidence_pipeline.py")
experiment = load_module("practice_experiment", "practice/post-training/code/experiment_checks.py")


class EvidencePipelineTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            rag.Chunk("old", "policy", 1, "registration deadline Friday", frozenset({"members"})),
            rag.Chunk("new", "policy", 2, "registration Wednesday", frozenset({"members"})),
            rag.Chunk("private", "budget", 1, "registration deadline budget", frozenset({"organizers"})),
        ]
        self.versions = {"policy": 2, "budget": 1}

    def test_filters_before_topk_even_when_ineligible_scores_are_higher(self):
        result = rag.retrieve("registration deadline", self.chunks, {"members"}, self.versions, limit=1)
        self.assertEqual([chunk.chunk_id for chunk in result], ["new"])

    def test_absent_identity_or_current_revision_fails_closed(self):
        self.assertEqual(rag.retrieve("registration", self.chunks, set(), self.versions), [])
        self.assertEqual(rag.retrieve("registration", self.chunks, {"members"}, {"budget": 1}), [])

    def test_index_lag_does_not_fall_back_to_previous_revision(self):
        self.assertEqual(rag.retrieve("registration", self.chunks, {"members"}, {"policy": 3}), [])

    def test_authorized_but_irrelevant_is_not_evidence(self):
        self.assertEqual(rag.retrieve("unknown topic", self.chunks, {"members"}, self.versions), [])

    def test_duplicate_chunk_ids_and_bad_limits_rejected(self):
        with self.assertRaises(ValueError):
            rag.retrieve("registration", self.chunks + self.chunks[:1], {"members"}, self.versions)
        for limit in (0, -1, 1.5, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                rag.retrieve("registration", self.chunks, {"members"}, self.versions, limit)

    def test_rrf_matches_article_arithmetic(self):
        result = rag.reciprocal_rank_fusion([["A", "B"], ["C", "A"]])
        self.assertEqual([identifier for identifier, score in result], ["A", "C", "B"])
        self.assertAlmostEqual(result[0][1], 1 / 11 + 1 / 12)
        self.assertAlmostEqual(result[1][1], 1 / 11)

    def test_rrf_deduplicates_within_a_source_and_breaks_ties_stably(self):
        self.assertEqual(rag.reciprocal_rank_fusion([["A", "A", "B"]]),
                         rag.reciprocal_rank_fusion([["A", "B"]]))
        self.assertEqual([identifier for identifier, score in rag.reciprocal_rank_fusion([["B"], ["A"]])],
                         ["A", "B"])
        self.assertEqual(rag.reciprocal_rank_fusion([]), [])
        for constant in (0, -1, float("nan"), float("inf")):
            with self.subTest(constant=constant), self.assertRaises(ValueError):
                rag.reciprocal_rank_fusion([["A"]], constant)

    def test_context_packing_skips_oversized_passage_without_truncation(self):
        chunks = [rag.Chunk(str(index), "document", 1, "word " * length, frozenset({"members"}))
                  for index, length in enumerate((6, 6, 4))]
        selected, used = rag.pack_evidence(chunks, 10)
        self.assertEqual([chunk.chunk_id for chunk in selected], ["0", "2"])
        self.assertEqual(used, 10)
        self.assertEqual(rag.pack_evidence(chunks, 0), ([], 0))
        self.assertEqual(len(chunks[1].text.split()), 6)

    def test_packing_rejects_empty_evidence_and_invalid_budget(self):
        for budget in (-1, 1.2, True):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                rag.pack_evidence(self.chunks, budget)
        with self.assertRaises(ValueError):
            rag.pack_evidence([rag.Chunk("blank", "blank", 1, "", frozenset({"members"}))], 10)

    def test_citation_validation_checks_membership_not_entailment(self):
        provided = [self.chunks[1]]
        self.assertEqual(rag.unknown_citations(["new", "old", "private", "old"], provided),
                         ["old", "private"])
        self.assertEqual(rag.unknown_citations(["new"], provided), [])
        self.assertEqual(rag.unknown_citations([], provided), [])


class ExperimentContractTests(unittest.TestCase):
    def test_masked_prompt_example_is_point_six(self):
        result = experiment.masked_objective([[4, 3, 0.8, 0.4]], [[0, 0, 1, 1]])
        self.assertAlmostEqual(result["token_mean"], 0.6)
        self.assertEqual(result["valid_tokens"], 2)
        self.assertAlmostEqual(sum([4, 3, 0.8, 0.4]) / 4, 2.05)

    def test_token_and_example_means_match_article(self):
        result = experiment.masked_objective([[0.2], [1, 1, 1]], [[1], [1, 1, 1]])
        self.assertAlmostEqual(result["token_mean"], 0.8)
        self.assertAlmostEqual(result["sample_mean"], 0.6)
        self.assertEqual(result["valid_tokens"], 4)

    def test_masks_require_alignment_finite_losses_and_targets(self):
        for losses, masks in [
            ([], []), ([[1]], []), ([[1, 2]], [[1]]), ([[1]], [[0]]),
            ([[1]], [[0.5]]), ([[float("nan")]], [[1]]),
            ([[float("inf")]], [[0]]), ([[-1]], [[1]]),
        ]:
            with self.subTest(losses=losses, masks=masks), self.assertRaises(ValueError):
                experiment.masked_objective(losses, masks)

    def test_question_families_can_repeat_within_one_split(self):
        rows = [
            {"sample_id": "en", "group_id": "one", "split": "train"},
            {"sample_id": "zh", "group_id": "one", "split": "train"},
        ]
        self.assertEqual(experiment.validate_group_splits(rows), {"one": "train"})
        rows[1]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            experiment.validate_group_splits(rows)

    def test_group_checks_reject_duplicate_or_incomplete_identifiers(self):
        row = {"sample_id": "one", "group_id": "family", "split": "train"}
        for rows in ([row, row], [{}], [{**row, "split": "unknown"}], [{**row, "group_id": ""}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                experiment.validate_group_splits(rows)

    def test_paired_report_surfaces_regression_behind_aggregate_gain(self):
        identifiers = ["q1", "q2", "q3", "q4", "q5", "q6"]
        baseline = dict(zip(identifiers, [0, 0, 1, 1, 1, 1]))
        candidate = dict(zip(identifiers, [1, 1, 1, 1, 0, 1]))
        slices = {identifier: "evidence" if index < 4 else "missing"
                  for index, identifier in enumerate(identifiers)}
        result = experiment.paired_report(baseline, candidate, slices)
        self.assertEqual(result["overall"]["wins"], 2)
        self.assertEqual(result["overall"]["losses"], 1)
        self.assertEqual(result["overall"]["ties"], 3)
        self.assertAlmostEqual(result["overall"]["delta"], 1 / 6)
        self.assertEqual(result["slices"]["missing"]["delta"], -0.5)
        self.assertEqual(result["slices"]["evidence"]["delta"], 0.5)

    def test_incomplete_pairs_and_nonbinary_outcomes_are_rejected(self):
        for baseline, candidate, slices in [
            ({}, {}, {}), ({"a": 1}, {}, {"a": "slice"}),
            ({"a": 1}, {"b": 1}, {"a": "slice"}),
            ({"a": 1}, {"a": 1}, {}),
            ({"a": 1}, {"a": 0.8}, {"a": "slice"}),
            ({"a": 1}, {"a": 1}, {"a": ""}),
        ]:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                experiment.paired_report(baseline, candidate, slices)

    def test_recommender_budget_and_composition_examples(self):
        relevant = {"B", "E"}
        self.assertEqual(len(set(["A", "B", "C", "D"]) & relevant), 1)
        self.assertEqual(len(set(["A", "B", "C", "E"]) & relevant), 2)
        self.assertEqual(len(set(["A", "C", "E", "F"]) & relevant), 1)
        self.assertAlmostEqual((20 * 0.8 + 80 * 0.2) / 100, 0.32)
        self.assertAlmostEqual((80 * 0.8 + 20 * 0.2) / 100, 0.68)


class PracticeArticleArithmeticTests(unittest.TestCase):
    def test_answer_and_abstention_table_totals_and_denominators(self):
        for suffix in (".md", ".en.md"):
            text = (ROOT / f"practice/rag/evidence-and-evaluation{suffix}").read_text()
            rows = re.findall(r"^\| [^|]+ \| (\d+) \| (\d+) \| (\d+) \| (\d+) \|$", text, re.M)
            counts = [tuple(map(int, row)) for row in rows]
            with self.subTest(language=suffix):
                self.assertIn('<div class="worked-table" markdown="1">', text)
                self.assertEqual(counts, [(8, 2, 2, 12), (0, 0, 8, 8)])
                for correct, wrong, abstained, total in counts:
                    self.assertEqual(correct + wrong + abstained, total)
                answered = sum(correct + wrong for correct, wrong, abstained, total in counts)
                self.assertEqual(sum(row[3] for row in counts), 20)
                self.assertAlmostEqual(sum(row[0] for row in counts) / answered, 0.8)
                self.assertAlmostEqual(counts[0][0] / counts[0][3], 2 / 3)
                self.assertEqual(counts[1][2] / counts[1][3], 1)

    def test_contrastive_temperature_example(self):
        for temperature in (0.25, 1.0, 2.0):
            for gap, probability in ((0, 0.5), (temperature * math.log(3), 0.75)):
                score = 1 / (1 + math.exp(-gap / temperature))
                with self.subTest(temperature=temperature, gap=gap):
                    self.assertAlmostEqual(score, probability)
        for suffix in (".md", ".en.md"):
            text = (ROOT / f"practice/recommender-systems/06-why-two-towers{suffix}").read_text()
            self.assertIn(f"{(-math.log(0.5)):.3f}", text)
            self.assertIn(f"{(-math.log(0.75)):.3f}", text)

    def test_retry_budget_distinguishes_retries_from_total_attempts(self):
        def leaf_calls(layers, attempts):
            if layers == 0:
                return 1
            return sum(leaf_calls(layers - 1, attempts) for attempt in range(attempts))

        self.assertEqual(leaf_calls(3, 3), 27)
        self.assertEqual(leaf_calls(3, 1 + 3), 64)
        for suffix in (".md", ".en.md"):
            text = (ROOT / f"practice/recommender-systems/08-serving-lifecycle{suffix}").read_text()
            self.assertIn("3×3×3=27", text)
            self.assertIn("4×4×4=64", text)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required for the gradient example")
    def test_prompt_loss_mask_does_not_remove_context_gradient(self):
        import torch

        hidden = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
                              dtype=torch.float64, requires_grad=True)
        causal_weights = torch.tril(torch.ones(4, 4, dtype=torch.float64))
        causal_weights = causal_weights / causal_weights.sum(dim=1, keepdim=True)
        projection = torch.tensor([[1.0, -1.0], [0.0, 0.0]], dtype=torch.float64)
        logits = (causal_weights @ hidden) @ projection
        logits.retain_grad()
        targets = torch.tensor([-100, -100, 0, 0])
        loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-100)
        loss.backward()
        self.assertEqual(logits.grad[:2].abs().sum().item(), 0)
        self.assertTrue(torch.all(hidden.grad[:2, 0].abs() > 0).item())
        self.assertTrue(torch.isfinite(hidden.grad).all().item())


class PracticeContentTests(unittest.TestCase):
    def test_new_chapters_have_matching_bilingual_structure(self):
        for folder in ("rag", "post-training"):
            for chinese in (ROOT / "practice" / folder).glob("*.md"):
                if chinese.name.endswith(".en.md"):
                    continue
                english = chinese.with_name(chinese.stem + ".en.md")
                with self.subTest(path=chinese):
                    self.assertTrue(english.is_file())
                    self.assertEqual(paritycheck.shape(chinese.read_text()),
                                     paritycheck.shape(english.read_text()))
                    for path in (chinese, english):
                        text = path.read_text()
                        headings = re.findall(r"^## (.+)$", text, re.M)
                        self.assertEqual(len(headings), len(set(headings)))
                        self.assertGreaterEqual(len(headings), 4)
                        if chinese.name == "README.md":
                            self.assertIn('<figure class="worked-update">', text)
                            self.assertEqual(text.count("<li><small>"), 6)
                            self.assertNotIn("```mermaid", text)

    def test_new_markdown_links_resolve_to_source_files(self):
        for folder in ("rag", "post-training"):
            for path in (ROOT / "practice" / folder).glob("*.md"):
                for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
                    if "://" in target or target.startswith("#"):
                        continue
                    with self.subTest(path=path, target=target):
                        self.assertTrue((path.parent / target.split("#")[0]).resolve().is_file())

    def test_navigation_and_review_ownership_include_both_projects(self):
        nav = tomllib.loads((ROOT / "site/nav.toml").read_text())
        config = tomllib.loads((ROOT / "site/collaboration.toml").read_text())
        practice = next(area for area in config["area"] if area["id"] == "practice")
        for group in ("rag-practice", "posttraining-practice"):
            self.assertIn(group, practice["groups"])
            sections = [section for section in nav["section"] if section.get("group") == group]
            self.assertEqual(len(sections), 1)
            expected = (["README.md", "data-and-retrieval.md", "evidence-and-evaluation.md"]
                        if group == "rag-practice" else
                        ["README.md", "training-plan.md", "data-pipeline.md", "data-and-objectives.md",
                         "distributed-training.md", "checkpoint-and-resume.md", "experiments-and-release.md"])
            self.assertEqual(sections[0]["order"], expected)

    def test_manifest_examples_are_valid_and_consistent(self):
        for suffix in (".md", ".en.md"):
            text = (ROOT / f"practice/post-training/experiments-and-release{suffix}").read_text()
            manifest = json.loads(re.search(r"```json\n(.*?)\n```", text, re.S)[1])
            self.assertEqual(manifest["decode"]["max_new_tokens"], 128)
            self.assertEqual(manifest["objective"], "assistant-target-token-mean")

    def test_both_cli_demos_execute_without_external_dependencies(self):
        for relative in ("practice/rag/code/evidence_pipeline.py",
                         "practice/post-training/code/experiment_checks.py"):
            result = subprocess.run([sys.executable, str(ROOT / relative)],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertGreater(len(result.stdout), 100)


if __name__ == "__main__":
    unittest.main()
