"""Exact checks for the probability notes; finite examples are not proofs."""

from fractions import Fraction
from itertools import combinations, permutations, product
from math import comb


def ace_conditionals():
    pairs = list(combinations(range(52), 2))
    containing_ace = [pair for pair in pairs if any(card < 4 for card in pair)]
    two_aces = [pair for pair in containing_ace if all(card < 4 for card in pair)]
    return Fraction(3, 51), Fraction(len(two_aces), len(containing_ace))


def extra_coin_win(trials, probability=Fraction(1, 2)):
    probability = Fraction(probability)
    if trials < 0 or not 0 <= probability <= 1:
        raise ValueError("Require nonnegative trials and probability in [0, 1].")
    counts = [
        Fraction(comb(trials, heads))
        * probability**heads
        * (1 - probability) ** (trials - heads)
        for heads in range(trials + 1)
    ]
    tie = sum(mass**2 for mass in counts)
    lead = sum(
        first_mass * second_mass
        for first, first_mass in enumerate(counts)
        for second, second_mass in enumerate(counts)
        if first > second
    )
    return lead + probability * tie, tie


def fixed_point_mean(size):
    if size < 1:
        raise ValueError("Require at least one element.")
    counts = [
        sum(index == value for index, value in enumerate(order))
        for order in permutations(range(size))
    ]
    return Fraction(sum(counts), len(counts))


def coin_mixture_moments(trials=10):
    if trials < 0:
        raise ValueError("Require nonnegative trials.")
    distribution = {
        heads: sum(
            Fraction(1, 2)
            * comb(trials, heads)
            * probability**heads
            * (1 - probability) ** (trials - heads)
            for probability in (Fraction(1, 5), Fraction(4, 5))
        )
        for heads in range(trials + 1)
    }
    mean = sum(heads * mass for heads, mass in distribution.items())
    variance = sum((heads - mean) ** 2 * mass for heads, mass in distribution.items())
    return mean, variance


def solve_exact(matrix, right_side):
    size = len(right_side)
    if not size or len(matrix) != size or any(len(row) != size for row in matrix):
        raise ValueError("Require a nonempty square system.")
    augmented = [
        [Fraction(value) for value in row] + [Fraction(value)]
        for row, value in zip(matrix, right_side)
    ]
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if augmented[row][column]),
            None,
        )
        if pivot is None:
            raise ValueError("Singular system.")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [row[-1] for row in augmented]


def pattern_wait(pattern, probability=Fraction(1, 2)):
    probability = Fraction(probability)
    if not pattern or set(pattern) - {"H", "T"}:
        raise ValueError("Use a nonempty H/T pattern.")
    if not 0 < probability < 1:
        raise ValueError("Require probability strictly between 0 and 1.")
    size = len(pattern)
    matrix = [
        [Fraction(int(row == column)) for column in range(size)]
        for row in range(size)
    ]
    for state in range(size):
        for symbol, mass in (("H", probability), ("T", 1 - probability)):
            history = pattern[:state] + symbol
            if history.endswith(pattern):
                continue
            next_state = max(
                length
                for length in range(size)
                if history.endswith(pattern[:length])
            )
            matrix[state][next_state] -= mass
    return solve_exact(matrix, [1] * size)[0]


def meeting_probability(interval, wait_first, wait_second):
    interval, wait_first, wait_second = map(
        Fraction, (interval, wait_first, wait_second)
    )
    if interval <= 0 or not 0 <= min(wait_first, wait_second):
        raise ValueError("Require positive interval and nonnegative waits.")
    if max(wait_first, wait_second) > interval:
        raise ValueError("This formula uses waits no longer than the interval.")
    return 1 - (
        (interval - wait_first) ** 2 + (interval - wait_second) ** 2
    ) / (2 * interval**2)


def run_checks():
    assert ace_conditionals() == (Fraction(1, 17), Fraction(1, 33))
    assert coin_mixture_moments() == (5, Fraction(53, 5))
    assert meeting_probability(60, 15, 15) == Fraction(7, 16)
    assert meeting_probability(60, 10, 20) == Fraction(31, 72)
    for size in range(1, 7):
        assert fixed_point_mean(size) == 1
    for trials, probability in product(
        range(6), (Fraction(1, 3), Fraction(1, 2), Fraction(2, 3))
    ):
        win, tie = extra_coin_win(trials, probability)
        assert win == Fraction(1, 2) + (probability - Fraction(1, 2)) * tie
    for probability in (Fraction(1, 3), Fraction(1, 2), Fraction(2, 3)):
        base = 1 / (probability**2 * (1 - probability))
        assert pattern_wait("HHT", probability) == base
        assert pattern_wait("HTH", probability) == base + 1 / probability
    assert [pattern_wait(pattern) for pattern in ("H", "HH", "HHH")] == [2, 6, 14]
    return {
        "aces: first ace / at least one ace": ace_conditionals(),
        "mixture: mean / variance": coin_mixture_moments(),
        "H / HH / HHT / HTH waiting times": [
            pattern_wait(pattern) for pattern in ("H", "HH", "HHT", "HTH")
        ],
        "meeting: equal / unequal waits": [
            meeting_probability(60, 15, 15),
            meeting_probability(60, 10, 20),
        ],
    }


if __name__ == "__main__":
    for label, result in run_checks().items():
        print(f"{label}: {result}")
    print("Exact finite checks passed. These checks do not replace the proofs.")
