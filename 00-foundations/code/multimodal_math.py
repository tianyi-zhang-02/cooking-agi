"""Small, synthetic calculations for the embedding, CLIP, and LM-training notes."""

import argparse
import math


SCORES = [[0.8, 0.2, 0.1], [0.1, 0.7, 0.3], [0.2, 0.1, 0.9]]


def dot(first, second):
    if not first or len(first) != len(second):
        raise ValueError("Use nonempty vectors of the same length")
    return sum(left * right for left, right in zip(first, second))


def cosine(first, second):
    denominator = math.sqrt(dot(first, first) * dot(second, second))
    if denominator == 0:
        raise ValueError("Cosine is undefined for a zero vector")
    return dot(first, second) / denominator


def log_softmax(values):
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Use finite, nonempty logits")
    maximum = max(values)
    shifted = [value - maximum for value in values]
    normalizer = math.log(sum(math.exp(value) for value in shifted))
    return [value - normalizer for value in shifted]


def contrastive_loss(scores, temperature=1.0):
    count = len(scores)
    if not count or any(len(row) != count for row in scores):
        raise ValueError("Use a nonempty square score matrix")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    rows = [log_softmax([value / temperature for value in row]) for row in scores]
    columns = [log_softmax([scores[row][column] / temperature for row in range(count)])
               for column in range(count)]
    row_loss = -sum(rows[index][index] for index in range(count)) / count
    column_loss = -sum(columns[index][index] for index in range(count)) / count
    return (row_loss + column_loss) / 2


def next_token_loss(logits, tokens, target_mask):
    if len(tokens) < 2 or len(logits) != len(tokens) or len(target_mask) != len(tokens):
        raise ValueError("Logits, tokens, and target-position mask must align")
    losses = []
    for target_position in range(1, len(tokens)):
        if not target_mask[target_position]:
            continue
        prediction = logits[target_position - 1]
        target = tokens[target_position]
        if not 0 <= target < len(prediction):
            raise ValueError("Target must be a valid vocabulary index")
        losses.append(-log_softmax(prediction)[target])
    if not losses:
        raise ValueError("No valid next-token targets")
    return sum(losses) / len(losses)


def torch_demo():
    import torch
    import torch.nn.functional as functional

    scores = torch.tensor(SCORES, dtype=torch.float64)
    targets = torch.arange(len(SCORES))
    reference = (functional.cross_entropy(scores, targets)
                 + functional.cross_entropy(scores.T, targets)) / 2
    assert math.isclose(reference.item(), contrastive_loss(SCORES), abs_tol=1e-12)
    image_vectors = torch.nn.Parameter(torch.tensor(
        [[1.0, 0.2, 0.1], [0.1, 1.0, 0.2], [0.2, 0.1, 1.0]], dtype=torch.float64))
    text_vectors = torch.nn.Parameter(torch.tensor(
        [[1.0, 0.1, 0.3], [0.3, 1.0, 0.1], [0.1, 0.2, 1.0]], dtype=torch.float64))
    optimizer = torch.optim.SGD([image_vectors, text_vectors], lr=0.1)

    def objective():
        images = functional.normalize(image_vectors, dim=-1)
        texts = functional.normalize(text_vectors, dim=-1)
        logits = images @ texts.T / 0.5
        return (functional.cross_entropy(logits, targets)
                + functional.cross_entropy(logits.T, targets)) / 2

    optimizer.zero_grad()
    before = objective()
    before.backward()
    assert torch.isfinite(image_vectors.grad).all()
    assert torch.isfinite(text_vectors.grad).all()
    optimizer.step()
    after = objective()
    print(f"Synthetic vector update: {before.item():.6f} -> {after.item():.6f}")
    return before.item(), after.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch", action="store_true", help="Also check PyTorch and update toy vectors")
    args = parser.parse_args()
    print("Synthetic teaching example; no pretrained model or dataset is used.")
    print(f"Dot: {dot([1, 0], [2, 2]):.3f}; cosine: {cosine([1, 0], [2, 2]):.3f}")
    print("Row 1 probabilities:", [round(math.exp(value), 6) for value in log_softmax(SCORES[0])])
    print(f"Symmetric contrastive loss: {contrastive_loss(SCORES):.6f}")
    if args.torch:
        torch_demo()
