import json
import math


QUERY = ((1.0, 0.0), (0.0, 1.0))
KEY = ((0.5, 1.0), (1.0, -0.5))
VALUE = ((1.0, 0.25), (-0.5, 1.0))
TOKENS = ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (-1.0, 1.0))


def project(vector, matrix):
    return tuple(sum(value * weight for value, weight in zip(vector, row)) for row in matrix)


def softmax(logits):
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def attend(query, keys, values):
    scores = [
        sum(left * right for left, right in zip(query, key)) / math.sqrt(len(query))
        for key in keys
    ]
    weights = softmax(scores)
    return tuple(
        sum(weight * value[column] for weight, value in zip(weights, values))
        for column in range(len(values[0]))
    )


def full_attention(tokens, causal=True):
    keys = [project(token, KEY) for token in tokens]
    values = [project(token, VALUE) for token in tokens]
    outputs = []
    for position, token in enumerate(tokens):
        end = position + 1 if causal else len(tokens)
        outputs.append(attend(project(token, QUERY), keys[:end], values[:end]))
    return outputs


def cached_attention(chunks):
    keys, values, outputs = [], [], []
    for chunk in chunks:
        start = len(keys)
        keys.extend(project(token, KEY) for token in chunk)
        values.extend(project(token, VALUE) for token in chunk)
        for offset, token in enumerate(chunk):
            end = start + offset + 1
            outputs.append(attend(project(token, QUERY), keys[:end], values[:end]))
    return outputs


def max_difference(first, second):
    return max(
        abs(left - right)
        for left_row, right_row in zip(first, second)
        for left, right in zip(left_row, right_row)
    )


def normalization_example():
    token_counts = (100, 10)
    mean_losses = (1.0, 3.0)
    total_tokens = sum(token_counts)
    token_mean = sum(count * loss for count, loss in zip(token_counts, mean_losses)) / total_tokens
    microbatch_mean = sum(mean_losses) / len(mean_losses)
    rank_gradient_sums = (100.0, 30.0)
    rank_count = len(rank_gradient_sums)
    local_gradients = [rank_count * value / total_tokens for value in rank_gradient_sums]
    averaged_gradient = sum(local_gradients) / rank_count
    direct_gradient = sum(rank_gradient_sums) / total_tokens
    return token_mean, microbatch_mean, averaged_gradient, direct_gradient


def response_alignment_example():
    prompt = ("A", "B", "C", "D")
    answer = ("E", "F", "G")
    tokens = prompt + answer
    return [
        (source, target, int(position + 1 >= len(prompt)))
        for position, (source, target) in enumerate(zip(tokens[:-1], tokens[1:]))
    ]


def generation_timing_example():
    queue_ms, prefill_ms = 40, 60
    decode_ms = (20, 20)
    arrivals = [queue_ms + prefill_ms]
    for duration in decode_ms:
        arrivals.append(arrivals[-1] + duration)
    return arrivals


def norm_clipping_example():
    gradient = (3.0, 4.0)
    norm = math.sqrt(sum(component**2 for component in gradient))
    limit = 1.0
    scale = min(1.0, limit / norm)
    return tuple(component * scale for component in gradient)


def grouped_rewards_example():
    groups = ((1.0, 0.0, 1.0, 0.0), (1.0, 1.0, 1.0, 1.0))
    return [
        [reward - sum(group) / len(group) for reward in group]
        for group in groups
    ]


def run_checks():
    full = full_attention(TOKENS)
    incremental = cached_attention([(token,) for token in TOKENS])
    chunked = cached_attention([TOKENS[:3], TOKENS[3:]])
    incremental_error = max_difference(full, incremental)
    chunked_error = max_difference(full, chunked)
    missing_mask_error = max_difference(full, full_attention(TOKENS, causal=False))
    assert incremental_error < 1e-12
    assert chunked_error < 1e-12
    assert missing_mask_error > 1e-3
    token_mean, microbatch_mean, averaged_gradient, direct_gradient = normalization_example()
    assert math.isclose(token_mean, 13 / 11)
    assert microbatch_mean == 2
    assert math.isclose(averaged_gradient, direct_gradient)
    stable_nll = math.log1p(math.exp(-1))
    assert math.isclose(stable_nll, 0.31326168751822286)
    kv_bytes_per_token = 2 * 32 * 8 * 128 * 2
    assert kv_bytes_per_token == 128 * 1024
    assert kv_bytes_per_token * 4096 * 16 == 8 * 1024**3
    ratio = 0.3 / 0.2
    clipped_ratio = min(max(ratio, 0.8), 1.2)
    assert math.isclose(min(ratio, clipped_ratio), 1.2)
    assert math.isclose(min(-ratio, -clipped_ratio), -1.5)
    alignment = response_alignment_example()
    assert [row[2] for row in alignment] == [0, 0, 0, 1, 1, 1]
    arrivals = generation_timing_example()
    assert arrivals == [100, 120, 140]
    clipped_gradient = norm_clipping_example()
    assert math.isclose(sum(component**2 for component in clipped_gradient), 1)
    grouped_advantages = grouped_rewards_example()
    assert grouped_advantages == [[0.5, -0.5, 0.5, -0.5], [0.0, 0.0, 0.0, 0.0]]
    return {
        "scope": "Toy attention and arithmetic, not model training or a performance benchmark",
        "incremental_error": incremental_error,
        "chunked_error": chunked_error,
        "missing_causal_mask_error": missing_mask_error,
        "token_mean_loss": token_mean,
        "unweighted_microbatch_mean_loss": microbatch_mean,
        "distributed_gradient": averaged_gradient,
        "direct_gradient": direct_gradient,
        "stable_cross_entropy": stable_nll,
        "kv_per_token_bytes": kv_bytes_per_token,
        "sixteen_requests_kv_gib": kv_bytes_per_token * 4096 * 16 / 1024**3,
        "response_alignment": alignment,
        "output_arrivals_ms": arrivals,
        "ideal_norm_clipped_gradient": clipped_gradient,
        "mean_centered_group_rewards": grouped_advantages,
    }


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
