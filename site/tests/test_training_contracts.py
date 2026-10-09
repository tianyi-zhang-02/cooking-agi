import copy
import importlib.util
import json
import math
from pathlib import Path
import re
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "training_contracts", ROOT / "practice/post-training/code/training_contracts.py"
)
contracts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contracts)


class RecordTests(unittest.TestCase):
    def test_demo_and_abstention(self):
        record = contracts.demo_record()
        self.assertEqual(contracts.validate_records([record]), 1)
        record.update(answerable=False, citations=[], evidence=[], answer="The supplied evidence does not say.")
        self.assertEqual(contracts.validate_records([record]), 1)

    def test_reject_unknown_citations_and_empty_targets(self):
        for changes in ({"citations": ["missing"]}, {"answer": " "},
                        {"answerable": True, "citations": []},
                        {"answerable": False}, {"review_status": "pending"}):
            with self.subTest(changes=changes):
                record = contracts.demo_record()
                record.update(changes)
                with self.assertRaises(ValueError):
                    contracts.validate_records([record])

    def test_duplicate_ids_and_cross_split_group(self):
        first = contracts.demo_record()
        with self.assertRaises(ValueError):
            contracts.validate_records([first, copy.deepcopy(first)])
        second = copy.deepcopy(first)
        second.update(sample_id="demo-002", split="test")
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            contracts.validate_records([first, second])

    def test_invalid_evidence(self):
        for evidence in ([{"id": "policy-7"}], [None], "text"):
            with self.subTest(evidence=evidence):
                record = contracts.demo_record()
                record["evidence"] = evidence
                with self.assertRaises(ValueError):
                    contracts.validate_records([record])

    def test_documented_json_matches_fixture(self):
        for suffix in (".md", ".en.md"):
            text = (ROOT / f"practice/post-training/data-pipeline{suffix}").read_text()
            record = json.loads(re.search(r"```json\n(.*?)\n```", text, re.S)[1])
            self.assertEqual(record, contracts.demo_record())
            self.assertEqual(contracts.validate_records([record]), 1)


class BatchTests(unittest.TestCase):
    def test_padding_does_not_mask_real_eos(self):
        batch = contracts.collate_targets([
            {"input_ids": [11, 21, 2], "target_mask": [0, 1, 1]},
            {"input_ids": [11, 12, 21, 22, 2], "target_mask": [0, 0, 1, 1, 1]},
        ], pad_id=2)
        self.assertEqual(batch["labels"][0], [-100, 21, 2, -100, -100])
        self.assertEqual(batch["attention_mask"][0], [1, 1, 1, 0, 0])
        self.assertEqual(sum(map(sum, batch["attention_mask"])), 8)
        self.assertEqual(sum(target != -100 for row in batch["labels"] for target in row[1:]), 5)

    def test_collator_rejects_invalid_targets(self):
        for mask in ([0, 0, 0], [1, 0, 0], [0, 1], [0, 2, 1], [False, True, True]):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                contracts.collate_targets([{"input_ids": [1, 2, 3], "target_mask": mask}], 0)

    def test_batch_uses_data_parallel_degree(self):
        self.assertEqual(contracts.effective_batch(2, 8, 4), 64)
        for invalid in (0, -1, 1.5, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                contracts.effective_batch(invalid, 8, 4)

    def test_global_target_normalization_across_ranks_and_microsteps(self):
        local_gradients = [[[2.0], [4.0, 6.0]], [[8.0, 10.0], [12.0]]]
        total_targets = sum(len(microstep) for rank in local_gradients for microstep in rank)
        expected = sum(sum(microstep) for rank in local_gradients for microstep in rank) / total_targets
        ranks = len(local_gradients)
        averaged = sum(
            sum(ranks * sum(microstep) / total_targets for microstep in rank)
            for rank in local_gradients
        ) / ranks
        self.assertAlmostEqual(averaged, expected)
        self.assertAlmostEqual((0.2 + 3.0) / 4, 0.8)
        self.assertAlmostEqual((0.2 + 3.0 / 3) / 2, 0.6)

    def test_documented_resource_examples(self):
        self.assertEqual(4096 * 4096, 16777216)
        self.assertEqual(16 * (4096 + 4096), 131072)
        self.assertAlmostEqual(131072 / 16777216 * 100, 0.78125)
        self.assertAlmostEqual(0.6e9 * 16 / 2**30, 8.9406967163)
        self.assertAlmostEqual(2800 / 1000 / 4, 0.7)
        self.assertAlmostEqual(30 / (600 + 30) * 100, 4.7619047619)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required for the CPU gradient check")
    def test_token_mean_gradient_matches_concatenated_batch(self):
        import torch

        local_features = [[1.0], [2.0, 3.0, 4.0]]
        total_targets = sum(map(len, local_features))
        weight = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
        features = torch.tensor([value for rank in local_features for value in rank], dtype=torch.float64)
        ((weight * features - 1).square().sum() / total_targets).backward()
        expected = weight.grad.item()
        correct, naive = [], []
        for values in local_features:
            local = torch.tensor(values, dtype=torch.float64)
            parameter = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
            (2 * (parameter * local - 1).square().sum() / total_targets).backward()
            correct.append(parameter.grad.item())
            parameter = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
            (parameter * local - 1).square().mean().backward()
            naive.append(parameter.grad.item())
        self.assertAlmostEqual(sum(correct) / 2, expected)
        self.assertNotAlmostEqual(sum(naive) / 2, expected)


class RecoveryTests(unittest.TestCase):
    def test_resume_matches_every_interruption_boundary(self):
        continuous = contracts.ToyRun()
        continuous.advance(12)
        for boundary in range(13):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as directory:
                interrupted = contracts.ToyRun()
                interrupted.advance(boundary)
                path = Path(directory) / "checkpoint.json"
                contracts.save_checkpoint(path, interrupted.state())
                resumed = contracts.ToyRun(seed=99)
                resumed.restore(json.loads(path.read_text()))
                resumed.advance(12 - boundary)
                self.assertEqual(resumed.state(), continuous.state())

    def test_missing_momentum_changes_trajectory(self):
        source = contracts.ToyRun()
        source.advance(5)
        state = source.state()
        state["velocity"] = 0.0
        resumed = contracts.ToyRun()
        resumed.restore(state)
        source.advance(7)
        resumed.advance(7)
        self.assertFalse(math.isclose(source.weight, resumed.weight, abs_tol=1e-12))

    def test_data_revision_mismatch_is_not_resume(self):
        source = contracts.ToyRun(data_revision="toy-v1")
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            contracts.ToyRun(data_revision="toy-v2").restore(source.state())

    def test_reject_incomplete_mismatched_or_corrupt_state(self):
        original = contracts.ToyRun().state()
        states = []
        missing = copy.deepcopy(original)
        del missing["rng"]
        states.append(missing)
        for field, value in (("manifest", {}), ("weight", float("nan")),
                             ("rng", []), ("order", [0, 0, 0, 0]), ("cursor", 1)):
            state = copy.deepcopy(original)
            state[field] = value
            states.append(state)
        for state in states:
            with self.subTest(state_keys=list(state)):
                run = contracts.ToyRun()
                with self.assertRaises(ValueError):
                    run.restore(state)
                self.assertEqual(run.state(), original)

    def test_orphan_temp_file_does_not_replace_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            source = contracts.ToyRun()
            source.advance(5)
            contracts.save_checkpoint(path, source.state())
            (Path(directory) / "checkpoint.json.orphan.tmp").write_text('{"format":')
            resumed = contracts.ToyRun()
            resumed.restore(json.loads(path.read_text()))
            self.assertEqual(resumed.state(), source.state())

    def test_failed_serialization_preserves_previous_save(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            source = contracts.ToyRun()
            contracts.save_checkpoint(path, source.state())
            before = path.read_bytes()
            invalid = source.state()
            invalid["weight"] = float("nan")
            with self.assertRaises(ValueError):
                contracts.save_checkpoint(path, invalid)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
