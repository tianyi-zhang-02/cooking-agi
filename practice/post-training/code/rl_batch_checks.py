import json
import math


def rloo_advantages(rewards):
    if len(rewards) < 2 or any(not math.isfinite(reward) for reward in rewards):
        raise ValueError("Expected at least two finite rewards")
    total = math.fsum(rewards)
    return [reward - (total - reward) / (len(rewards) - 1) for reward in rewards]


def grouped_advantages(records, group_size):
    if type(group_size) is not int or group_size < 2 or not records:
        raise ValueError("Expected records and an integer group size of at least two")
    groups = {}
    identifiers = set()
    for record in records:
        sample_id = record["sample_id"]
        prompt_id = record["prompt_id"]
        if not isinstance(sample_id, str) or not sample_id or sample_id in identifiers:
            raise ValueError("Sample identifiers must be nonempty and unique")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError("Prompt identifiers must be nonempty")
        if record["score_status"] != "ok":
            raise ValueError("Missing or failed scores are not zero rewards")
        identifiers.add(sample_id)
        groups.setdefault(prompt_id, []).append(record)
    result = {}
    for siblings in groups.values():
        if len(siblings) != group_size:
            raise ValueError("Incomplete or oversized prompt group")
        for field in ("policy_version", "generation_round", "reward_version"):
            values = [record[field] for record in siblings]
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError("Group versions must be nonempty strings")
            if len(set(values)) != 1:
                raise ValueError("Group versions must match")
        advantages = rloo_advantages([record["reward"] for record in siblings])
        result.update((record["sample_id"], advantage)
                      for record, advantage in zip(siblings, advantages))
    return result


def sequence_log_prob(logits, input_ids, action_mask):
    if len(logits) != len(input_ids) or len(action_mask) != len(input_ids) or len(input_ids) < 2:
        raise ValueError("Expected aligned logits, IDs, and action mask")
    if any(type(active) is not bool for active in action_mask) or action_mask[0] or not any(action_mask):
        raise ValueError("Expected sampled actions after the first position")
    vocabulary = len(logits[0])
    if vocabulary == 0 or any(len(row) != vocabulary for row in logits):
        raise ValueError("Expected one fixed vocabulary size")
    if any(type(token) is not int or not 0 <= token < vocabulary for token in input_ids):
        raise ValueError("Token ID outside vocabulary")
    selected = []
    for position in range(1, len(input_ids)):
        if not action_mask[position]:
            selected.append(0.0)
            continue
        row = logits[position - 1]
        if any(not math.isfinite(value) for value in row):
            raise ValueError("Active logits must be finite")
        maximum = max(row)
        shifted = [value - maximum for value in row]
        selected.append(shifted[input_ids[position]] - math.log(math.fsum(math.exp(value) for value in shifted)))
    return selected, math.fsum(selected)


def loss_reductions(losses, masks, fixed_length):
    if not losses or len(losses) != len(masks) or type(fixed_length) is not int or fixed_length <= 0:
        raise ValueError("Expected nonempty batches and a positive integer fixed length")
    totals, lengths = [], []
    for row, mask in zip(losses, masks):
        if len(row) != len(mask) or any(type(active) is not bool for active in mask):
            raise ValueError("Expected aligned losses and boolean masks")
        active_values = [value for value, active in zip(row, mask) if active]
        if not active_values or len(active_values) > fixed_length:
            raise ValueError("Invalid action count")
        if any(not math.isfinite(value) for value in active_values):
            raise ValueError("Active losses must be finite")
        totals.append(math.fsum(active_values))
        lengths.append(len(active_values))
    batch_size = len(totals)
    total = math.fsum(totals)
    return {
        "sequence_sum": total / batch_size,
        "sequence_mean": math.fsum(total / length for total, length in zip(totals, lengths)) / batch_size,
        "token_mean": total / sum(lengths),
        "fixed_length": total / (batch_size * fixed_length),
    }


def demo_records():
    return [
        {"prompt_id": prompt, "sample_id": f"{prompt}-{index}", "reward": reward,
         "score_status": "ok", "policy_version": "policy-0", "generation_round": "round-0",
         "reward_version": "rubric-0"}
        for prompt, rewards in (("Q1", [1, 0, 0.5]), ("Q2", [0, 0, 0]))
        for index, reward in enumerate(rewards)
    ]


def demo_logits():
    probabilities = [[0.125, 0.5, 0.125, 0.125, 0.125],
                     [0.1875, 0.1875, 0.25, 0.1875, 0.1875],
                     [0.05, 0.05, 0.05, 0.8, 0.05]]
    return [[math.log(value) for value in row] for row in probabilities] + [[math.nan] * 5] * 2


def main():
    records = demo_records()
    expected = {"Q1-0": 0.75, "Q1-1": -0.75, "Q1-2": 0.0,
                "Q2-0": 0.0, "Q2-1": 0.0, "Q2-2": 0.0}
    advantages = grouped_advantages(records, 3)
    assert advantages == expected == grouped_advantages(list(reversed(records)), 3)
    _, log_probability = sequence_log_prob(demo_logits(), [0, 1, 2, 3, 3], [False, True, True, True, False])
    assert math.isclose(log_probability, math.log(0.1))
    reductions = loss_reductions([[2, 4], [1, 1, 1, 1]], [[True] * 2, [True] * 4], 4)
    for name, expected_value in {"sequence_sum": 5, "sequence_mean": 2,
                                 "token_mean": 5 / 3, "fixed_length": 1.25}.items():
        assert math.isclose(reductions[name], expected_value)
    print(json.dumps({"advantages": advantages, "sequence_log_probability": log_probability,
                      "reductions": reductions}, indent=2))


if __name__ == "__main__":
    main()
