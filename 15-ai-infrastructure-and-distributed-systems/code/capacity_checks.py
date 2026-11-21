import json
import math


def nonnegative(*values):
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Require finite nonnegative quantities")


def backlog_after(initial, arrival_rate, service_rate, seconds):
    nonnegative(initial, arrival_rate, service_rate, seconds)
    return max(0, initial + (arrival_rate - service_rate) * seconds)


def drain_seconds(backlog, arrival_rate, service_rate):
    nonnegative(backlog, arrival_rate, service_rate)
    if backlog == 0:
        return 0.0
    if service_rate <= arrival_rate:
        return math.inf
    return backlog / (service_rate - arrival_rate)


def replica_count(arrival_rate, per_replica_capacity, target_fraction=0.8):
    nonnegative(arrival_rate, per_replica_capacity)
    if per_replica_capacity == 0 or not 0 < target_fraction <= 1:
        raise ValueError("Require positive capacity and a target fraction in (0, 1]")
    return math.ceil(arrival_rate / (per_replica_capacity * target_fraction))


def kv_bytes(layers, kv_heads, head_dim, element_bytes, tokens, sequences=1):
    values = (layers, kv_heads, head_dim, element_bytes, tokens, sequences)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("KV dimensions, lengths, and counts must be positive integers")
    return 2 * layers * kv_heads * head_dim * element_bytes * tokens * sequences


def token_weighted_mean(means, counts):
    if len(means) != len(counts) or not counts:
        raise ValueError("Means and counts must have the same nonzero length")
    if any(type(count) is not int or count < 0 for count in counts) or sum(counts) == 0:
        raise ValueError("Counts must be nonnegative integers with a positive total")
    if any(not math.isfinite(mean) for mean in means):
        raise ValueError("Means must be finite; use zero for an empty local contribution")
    return sum(mean * count for mean, count in zip(means, counts)) / sum(counts)


def example_results():
    backlog = backlog_after(0, 120, 100, 10)
    replicas = replica_count(120, 50)
    return {
        "backlog": backlog,
        "drain_seconds": drain_seconds(backlog, 80, 100),
        "capacity_replicas": replicas,
        "replicas_with_one_failure_reserve": replicas + 1,
        "short_request_kv_mib": kv_bytes(32, 8, 128, 2, 2560) / 2**20,
        "long_request_kv_mib": kv_bytes(32, 8, 128, 2, 8704) / 2**20,
        "short_batch_kv_gib": kv_bytes(32, 8, 128, 2, 2560, 32) / 2**30,
        "long_batch_kv_gib": kv_bytes(32, 8, 128, 2, 8704, 32) / 2**30,
        "token_weighted_loss": token_weighted_mean([1, 3], [2, 6]),
    }


if __name__ == "__main__":
    print(json.dumps(example_results(), indent=2))
