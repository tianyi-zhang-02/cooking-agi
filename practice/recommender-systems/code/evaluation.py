from collections.abc import Sequence, Set


def recall_at_k(ranked: Sequence[str], positives: Set[str], cutoff: int) -> float | None:
    if not isinstance(cutoff, int) or isinstance(cutoff, bool) or cutoff < 1:
        raise ValueError("cutoff must be a positive integer")
    if len(ranked) != len(set(ranked)):
        raise ValueError("ranking must contain unique candidate IDs")
    if not positives:
        return None
    hits = set(ranked[:cutoff]) & set(positives)
    return len(hits) / len(positives)


def main() -> None:
    small_pool = ["P", "N1", "N2"]
    larger_pool = ["N3", "N4", "P", "N1", "N2"]
    print("Synthetic example: the scoring model is unchanged.")
    print("Small pool Recall@2:", recall_at_k(small_pool, {"P"}, 2))
    print("Larger pool Recall@2:", recall_at_k(larger_pool, {"P"}, 2))
    positives = {"P1", "P2", "P3", "P4"}
    ranked = ["P1", "P2", "N1"]
    hits = len(set(ranked[:2]) & positives)
    print("Standard Recall@2 (denominator = all positives):", recall_at_k(ranked, positives, 2))
    print("Alternative normalization (denominator = min(positives, K)):", hits / min(len(positives), 2))


if __name__ == "__main__":
    main()
