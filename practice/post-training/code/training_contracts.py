"""Small data, batching, and recovery checks; not an LLM trainer."""

import copy
import json
import math
import os
from pathlib import Path
import random
import tempfile


def validate_records(records):
    if not records:
        raise ValueError("records must not be empty")
    sample_ids, group_splits = set(), {}
    for record in records:
        for field in ("sample_id", "group_id", "question", "answer", "label_source"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"missing text field: {field}")
        if record.get("split") not in {"train", "dev", "test"}:
            raise ValueError("invalid split")
        if record.get("review_status") != "approved":
            raise ValueError("record has not been approved")
        if type(record.get("answerable")) is not bool:
            raise ValueError("answerable must be a boolean")
        sample_id, group_id = record["sample_id"], record["group_id"]
        if sample_id in sample_ids:
            raise ValueError("duplicate sample ID")
        sample_ids.add(sample_id)
        if group_splits.setdefault(group_id, record["split"]) != record["split"]:
            raise ValueError("group crosses splits")
        evidence, citations = record.get("evidence"), record.get("citations")
        if not isinstance(evidence, list) or not isinstance(citations, list):
            raise ValueError("evidence and citations must be lists")
        evidence_ids = set()
        for document in evidence:
            if not isinstance(document, dict) or any(
                not isinstance(document.get(field), str) or not document[field].strip()
                for field in ("id", "revision", "text")
            ):
                raise ValueError("invalid evidence record")
            if document["id"] in evidence_ids:
                raise ValueError("duplicate evidence ID")
            evidence_ids.add(document["id"])
        if any(not isinstance(citation, str) for citation in citations):
            raise ValueError("citation IDs must be strings")
        if len(set(citations)) != len(citations) or not set(citations) <= evidence_ids:
            raise ValueError("unknown or duplicate citation")
        if record["answerable"] != bool(citations):
            raise ValueError("answerable examples need citations; abstentions must not cite")
    return len(records)


def collate_targets(examples, pad_id):
    if not examples or type(pad_id) is not int or pad_id < 0:
        raise ValueError("need examples and a nonnegative integer pad ID")
    for example in examples:
        tokens, mask = example.get("input_ids"), example.get("target_mask")
        if not isinstance(tokens, list) or not isinstance(mask, list):
            raise ValueError("tokens and masks must be lists")
        if not tokens or len(tokens) != len(mask):
            raise ValueError("unaligned tokens and mask")
        if any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("token IDs must be nonnegative integers")
        if any(type(flag) is not int or flag not in (0, 1) for flag in mask):
            raise ValueError("target mask must contain integer zeros and ones")
        if mask[0] != 0 or not any(mask[1:]):
            raise ValueError("need predictable targets after an unscored first token")
    width = max(len(example["input_ids"]) for example in examples)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for example in examples:
        tokens, mask = example["input_ids"], example["target_mask"]
        padding = width - len(tokens)
        batch["input_ids"].append(tokens + [pad_id] * padding)
        batch["attention_mask"].append([1] * len(tokens) + [0] * padding)
        batch["labels"].append(
            [token if flag else -100 for token, flag in zip(tokens, mask)] + [-100] * padding
        )
    return batch


def effective_batch(microbatch, accumulation, data_parallel):
    if any(type(value) is not int or value < 1 for value in (microbatch, accumulation, data_parallel)):
        raise ValueError("batch dimensions must be positive integers")
    return microbatch * accumulation * data_parallel


def nested_tuples(value):
    return tuple(nested_tuples(item) for item in value) if isinstance(value, list) else value


class ToyRun:
    samples = ((1.0, 0.5), (2.0, 1.0), (3.0, 1.5), (4.0, 2.0))

    def __init__(self, seed=7, data_revision="toy-v1"):
        self.manifest = {"data": data_revision, "algorithm": "momentum-regression-v1"}
        self.weight = 0.0
        self.velocity = 0.0
        self.step = 0
        self.cursor = 0
        self.order = list(range(len(self.samples)))
        self.rng = random.Random(seed)
        self.rng.shuffle(self.order)

    def advance(self, steps):
        if type(steps) is not int or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        for _ in range(steps):
            if self.cursor == len(self.order):
                self.rng.shuffle(self.order)
                self.cursor = 0
            feature, target = self.samples[self.order[self.cursor]]
            feature += self.rng.uniform(-0.1, 0.1)
            gradient = (self.weight * feature - target) * feature
            self.velocity = 0.9 * self.velocity + gradient
            learning_rate = 0.05 / (1 + self.step / 10)
            self.weight -= learning_rate * self.velocity
            self.cursor += 1
            self.step += 1

    def state(self):
        return copy.deepcopy({
            "format": 1,
            "manifest": self.manifest,
            "weight": self.weight,
            "velocity": self.velocity,
            "step": self.step,
            "cursor": self.cursor,
            "order": self.order,
            "rng": self.rng.getstate(),
        })

    def restore(self, state):
        if set(state) != set(self.state()) or state["format"] != 1:
            raise ValueError("incomplete or unsupported checkpoint")
        if state["manifest"] != self.manifest:
            raise ValueError("experiment manifest mismatch")
        for field in ("weight", "velocity"):
            if type(state[field]) not in (float, int) or not math.isfinite(state[field]):
                raise ValueError("non-finite model or optimizer state")
        if type(state["step"]) is not int or state["step"] < 0:
            raise ValueError("invalid step")
        if type(state["cursor"]) is not int or not 0 <= state["cursor"] <= len(self.samples):
            raise ValueError("invalid data cursor")
        if (not isinstance(state["order"], list)
                or any(type(index) is not int for index in state["order"])
                or sorted(state["order"]) != list(range(len(self.samples)))):
            raise ValueError("invalid data permutation")
        expected_cursor = 0 if state["step"] == 0 else (state["step"] - 1) % len(self.samples) + 1
        if state["cursor"] != expected_cursor:
            raise ValueError("step and data cursor disagree")
        restored_rng = random.Random()
        try:
            restored_rng.setstate(nested_tuples(state["rng"]))
        except (TypeError, ValueError, IndexError) as error:
            raise ValueError("invalid RNG state") from error
        self.weight, self.velocity = state["weight"], state["velocity"]
        self.step, self.cursor = state["step"], state["cursor"]
        self.order = list(state["order"])
        self.rng = restored_rng


def save_checkpoint(path, state):
    path = Path(path)
    payload = json.dumps(state, allow_nan=False, sort_keys=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def demo_record():
    return {
        "sample_id": "demo-001", "group_id": "registration-deadline", "split": "train",
        "question": "When does registration close?",
        "evidence": [{"id": "policy-7", "revision": "v2", "text": "Registration closes on Wednesday."}],
        "answer": "Registration closes on Wednesday [policy-7].", "citations": ["policy-7"],
        "answerable": True, "label_source": "synthetic-demonstration", "review_status": "approved",
    }


def main():
    print("Valid records:", validate_records([demo_record()]))
    examples = [
        {"input_ids": [11, 21, 2], "target_mask": [0, 1, 1]},
        {"input_ids": [11, 12, 21, 22, 2], "target_mask": [0, 0, 1, 1, 1]},
    ]
    print("Padded batch:", json.dumps(collate_targets(examples, pad_id=2)))
    print("Examples per update:", effective_batch(2, 8, 4))
    uninterrupted, interrupted = ToyRun(), ToyRun()
    uninterrupted.advance(12)
    interrupted.advance(5)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.json"
        save_checkpoint(path, interrupted.state())
        resumed = ToyRun(seed=999)
        resumed.restore(json.loads(path.read_text()))
        resumed.advance(7)
    print("12 steps equals 5 + restore + 7:", uninterrupted.state() == resumed.state())


if __name__ == "__main__":
    main()
