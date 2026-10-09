import json
import math


def validate_group_splits(rows):
    samples = set()
    assignments = {}
    for row in rows:
        if not all(isinstance(row.get(field), str) and row[field]
                   for field in ("sample_id", "group_id", "split")):
            raise ValueError("sample_id, group_id and split must be nonempty strings")
        if row["split"] not in {"train", "dev", "test"}:
            raise ValueError("unknown split")
        if row["sample_id"] in samples:
            raise ValueError("duplicate sample ID")
        samples.add(row["sample_id"])
        group = row["group_id"]
        if group in assignments and assignments[group] != row["split"]:
            raise ValueError("question family crosses splits")
        assignments[group] = row["split"]
    return assignments


def masked_objective(losses, masks):
    if not losses or len(losses) != len(masks):
        raise ValueError("nonempty, aligned batches required")
    loss_sums = []
    counts = []
    for row_losses, row_mask in zip(losses, masks):
        if not row_losses or len(row_losses) != len(row_mask):
            raise ValueError("losses and masks must align per sample")
        if any(not math.isfinite(value) or value < 0 for value in row_losses):
            raise ValueError("losses must be finite and nonnegative")
        if any(value not in (0, 1) for value in row_mask):
            raise ValueError("masks must be binary")
        count = sum(row_mask)
        if not count:
            raise ValueError("every sample needs a valid target")
        loss_sums.append(sum(value * keep for value, keep in zip(row_losses, row_mask)))
        counts.append(count)
    return {
        "token_mean": sum(loss_sums) / sum(counts),
        "sample_mean": sum(total / count for total, count in zip(loss_sums, counts)) / len(counts),
        "valid_tokens": sum(counts),
    }


def paired_report(baseline, candidate, slices):
    if not baseline or baseline.keys() != candidate.keys() or baseline.keys() != slices.keys():
        raise ValueError("baseline, candidate and slices need identical nonempty case IDs")
    if any(value not in (0, 1) for value in [*baseline.values(), *candidate.values()]):
        raise ValueError("outcomes must be binary pass/fail")
    if any(not isinstance(label, str) or not label for label in slices.values()):
        raise ValueError("slice labels must be nonempty strings")

    def summarize(identifiers):
        count = len(identifiers)
        old_rate = sum(baseline[identifier] for identifier in identifiers) / count
        new_rate = sum(candidate[identifier] for identifier in identifiers) / count
        return {
            "count": count,
            "baseline": old_rate,
            "candidate": new_rate,
            "delta": new_rate - old_rate,
            "wins": sum(candidate[identifier] > baseline[identifier] for identifier in identifiers),
            "losses": sum(candidate[identifier] < baseline[identifier] for identifier in identifiers),
            "ties": sum(candidate[identifier] == baseline[identifier] for identifier in identifiers),
        }

    return {
        "overall": summarize(sorted(baseline)),
        "slices": {
            label: summarize(sorted(identifier for identifier in slices if slices[identifier] == label))
            for label in sorted(set(slices.values()))
        },
    }


def demo():
    print("objective:", masked_objective([[0.2], [1.0, 1.0, 1.0]], [[1], [1, 1, 1]]))
    validate_group_splits([
        {"sample_id": "a", "group_id": "deadline", "split": "train"},
        {"sample_id": "b", "group_id": "deadline", "split": "train"},
        {"sample_id": "c", "group_id": "location", "split": "test"},
    ])
    baseline = dict(zip(["q1", "q2", "q3", "q4", "q5", "q6"], [0, 0, 1, 1, 1, 1]))
    candidate = dict(zip(["q1", "q2", "q3", "q4", "q5", "q6"], [1, 1, 1, 1, 0, 1]))
    slices = {identifier: "evidence" if identifier in {"q1", "q2", "q3", "q4"} else "no-evidence"
              for identifier in baseline}
    print(json.dumps(paired_report(baseline, candidate, slices), indent=2))


if __name__ == "__main__":
    demo()
