import math


def finite_values(values):
    result = list(values)
    if any(isinstance(value, bool) or not math.isfinite(value) for value in result):
        raise ValueError("Expected finite numeric values")
    return result


def recurrent_trace(inputs, decay=0.5, initial=0.0):
    values = finite_values(inputs)
    finite_values([decay, initial])
    if not 0 <= decay <= 1:
        raise ValueError("Decay must be in [0, 1]")
    state = initial
    trace = []
    for value in values:
        state = decay * state + value
        finite_values([state])
        trace.append(state)
    return trace


def prefix_means(inputs):
    values = finite_values(inputs)
    total = 0.0
    result = []
    for count, value in enumerate(values, start=1):
        total += value
        finite_values([total])
        result.append(total / count)
    return result


def channel_delta_step(state, key, value, decay, write):
    rows = [finite_values(row) for row in state]
    key = finite_values(key)
    value = finite_values(value)
    decay = finite_values(decay)
    finite_values([write])
    if not key or not value or len(rows) != len(key) or len(decay) != len(key):
        raise ValueError("State and gates must match the key dimension")
    if any(len(row) != len(value) for row in rows):
        raise ValueError("State columns must match the value dimension")
    if not 0 <= write <= 1 or any(not 0 <= gate <= 1 for gate in decay):
        raise ValueError("This example uses gates in [0, 1]")
    if not math.isclose(math.fsum(entry * entry for entry in key), 1, abs_tol=1e-9):
        raise ValueError("Expected a unit key")
    decayed = [[gate * entry for entry in row] for gate, row in zip(decay, rows)]
    prediction = [math.fsum(key_entry * row[column] for key_entry, row in zip(key, decayed))
                  for column in range(len(value))]
    error = [target - predicted for target, predicted in zip(value, prediction)]
    result = [[entry + write * key_entry * correction for entry, correction in zip(row, error)]
              for row, key_entry in zip(decayed, key)]
    for row in result:
        finite_values(row)
    return result


def content_attention(inputs, causal=False):
    rows = [finite_values(row) for row in inputs]
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Expected a nonempty rectangular matrix")
    if not isinstance(causal, bool):
        raise ValueError("Causal must be boolean")
    output = []
    dimension = len(rows[0])
    for position, query in enumerate(rows):
        visible = rows[:position + 1] if causal else rows
        logits = [math.fsum(left * right for left, right in zip(query, key))
                  / math.sqrt(dimension) for key in visible]
        finite_values(logits)
        maximum = max(logits)
        weights = [math.exp(logit - maximum) for logit in logits]
        denominator = math.fsum(weights)
        probabilities = [weight / denominator for weight in weights]
        output.append([math.fsum(probability * value[column]
                                 for probability, value in zip(probabilities, visible))
                       for column in range(dimension)])
    return output


def main():
    first = recurrent_trace([1, 2])[-1]
    second = recurrent_trace([2, 1])[-1]
    print(f"Recurrent final states: {first}, {second}")
    forward_prefixes = prefix_means([1, 2, 0])
    reversed_prefixes = prefix_means([2, 1, 0])
    print(f"Causal means, layer 1: {forward_prefixes[-1]}, {reversed_prefixes[-1]}")
    print(f"Causal means, layer 2: {round(prefix_means(forward_prefixes)[-1], 6)}, "
          f"{prefix_means(reversed_prefixes)[-1]}")


if __name__ == "__main__":
    main()
