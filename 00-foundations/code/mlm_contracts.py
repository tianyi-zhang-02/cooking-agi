import math


IGNORE = -100


def corruption_action(draw):
    if isinstance(draw, bool) or not math.isfinite(draw) or not 0 <= draw < 1:
        raise ValueError("Expected a draw in [0, 1)")
    if draw < 0.8:
        return "mask"
    if draw < 0.9:
        return "random"
    return "keep"


def build_example(original, replacements, *, pad_id=0, special_ids=(1, 2)):
    tokens = list(original)
    if not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("Expected nonnegative integer token IDs")
    structural_ids = list(special_ids) + [pad_id]
    if any(type(token) is not int or token < 0 for token in structural_ids):
        raise ValueError("Special token IDs must be nonnegative integers")
    excluded = set(structural_ids)
    inputs = tokens.copy()
    labels = [IGNORE] * len(tokens)
    for position, replacement in replacements.items():
        if type(position) is not int or not 0 <= position < len(tokens):
            raise ValueError("Selected position is outside the sequence")
        if tokens[position] in excluded:
            raise ValueError("Do not select padding or structural special tokens")
        if type(replacement) is not int or replacement < 0:
            raise ValueError("Replacement must be a nonnegative token ID")
        inputs[position] = replacement
        labels[position] = tokens[position]
    attention = [int(token != pad_id) for token in tokens]
    return inputs, labels, attention


def masked_cross_entropy(logits, labels):
    rows = [list(row) for row in logits]
    targets = list(labels)
    if not rows or len(rows) != len(targets) or not rows[0]:
        raise ValueError("Expected matching nonempty logits and labels")
    vocabulary = len(rows[0])
    if any(len(row) != vocabulary for row in rows):
        raise ValueError("Logits must be rectangular")
    if any(isinstance(value, bool) or not math.isfinite(value) for row in rows for value in row):
        raise ValueError("Logits must be finite numbers")
    losses = []
    for row, target in zip(rows, targets):
        if type(target) is not int or (target != IGNORE and not 0 <= target < vocabulary):
            raise ValueError("Invalid target token ID")
        if target == IGNORE:
            continue
        maximum = max(row)
        shifted = [value - maximum for value in row]
        losses.append(math.log(math.fsum(math.exp(value) for value in shifted)) - shifted[target])
    if not losses:
        raise ValueError("No supervised positions; skip or resample this batch explicitly")
    return math.fsum(losses) / len(losses)


def demo_data():
    original = [1, 4, 5, 6, 7, 2, 0]
    inputs, labels, attention = build_example(original, {2: 5, 4: 3})
    logits = [[0.0] * 9 for _ in original]
    for position, probability in ((2, 0.8), (4, 0.25)):
        logits[position] = [math.log((1 - probability) / 8)] * 9
        logits[position][labels[position]] = math.log(probability)
    return inputs, labels, attention, logits


def main():
    inputs, labels, attention, logits = demo_data()
    print(f"Input IDs: {inputs}")
    print(f"Labels: {labels}")
    print(f"Attention: {attention}")
    print(f"Selected-token loss: {masked_cross_entropy(logits, labels):.6f}")


if __name__ == "__main__":
    main()
