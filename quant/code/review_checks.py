import heapq
import math
from fractions import Fraction


def hypergeometric(population, targets, draws):
    if not all(isinstance(value, int) for value in (population, targets, draws)):
        raise ValueError("Counts must be integers")
    if population < 1 or not 0 <= targets <= population or not 0 <= draws <= population:
        raise ValueError("Invalid population, targets, or draws")
    denominator = math.comb(population, draws)
    lower = max(0, draws - population + targets)
    upper = min(draws, targets)
    return {
        count: Fraction(
            math.comb(targets, count) * math.comb(population - targets, draws - count),
            denominator,
        )
        for count in range(lower, upper + 1)
    }


def distribution_moments(distribution):
    total = sum(distribution.values(), Fraction())
    if total != 1 or any(probability < 0 for probability in distribution.values()):
        raise ValueError("Probabilities must be nonnegative and sum to one")
    mean = sum((value * probability for value, probability in distribution.items()), Fraction())
    variance = sum(
        ((value - mean) ** 2 * probability for value, probability in distribution.items()),
        Fraction(),
    )
    return mean, variance


def coupon_mean(types):
    if not isinstance(types, int) or types < 1:
        raise ValueError("A positive integer number of types is required")
    return types * sum((Fraction(1, remaining) for remaining in range(1, types + 1)), Fraction())


def derangement_probability(size):
    if not isinstance(size, int) or size < 0:
        raise ValueError("Size must be a nonnegative integer")
    return sum(
        (Fraction((-1) ** index, math.factorial(index)) for index in range(size + 1)),
        Fraction(),
    )


def die_stopping_value(rolls, sides=6):
    if not all(isinstance(value, int) and value > 0 for value in (rolls, sides)):
        raise ValueError("Rolls and sides must be positive integers")
    value = Fraction(sides + 1, 2)
    for remaining in range(1, rolls):
        value = sum((max(Fraction(face), value) for face in range(1, sides + 1)), Fraction()) / sides
    return value


def replication(spot, up, down, gross_return, payoff_up, payoff_down):
    spot, up, down, gross_return, payoff_up, payoff_down = map(
        Fraction, (spot, up, down, gross_return, payoff_up, payoff_down)
    )
    if spot <= 0 or not 0 < down < gross_return < up:
        raise ValueError("Require positive spot and 0 < down < gross return < up")
    delta = (payoff_up - payoff_down) / (spot * (up - down))
    cash = (payoff_down - delta * spot * down) / gross_return
    risk_neutral = (gross_return - down) / (up - down)
    price = delta * spot + cash
    return price, delta, cash, risk_neutral


def bisect_root(function, lower, upper, tolerance=1e-10, max_iterations=1000):
    if not all(math.isfinite(value) for value in (lower, upper, tolerance)):
        raise ValueError("Bounds and tolerance must be finite")
    if lower >= upper or tolerance <= 0 or max_iterations < 1:
        raise ValueError("Require ordered bounds, positive tolerance and iteration limit")
    left_value, right_value = function(lower), function(upper)
    if not all(math.isfinite(value) for value in (left_value, right_value)):
        raise ValueError("Function values must be finite")
    if left_value == 0:
        return lower
    if right_value == 0:
        return upper
    if (left_value > 0) == (right_value > 0):
        raise ValueError("An opposite-sign bracket is required")
    for iteration in range(max_iterations):
        midpoint = lower / 2 + upper / 2
        middle_value = function(midpoint)
        if not math.isfinite(middle_value):
            raise ValueError("Function values must be finite")
        if middle_value == 0 or upper / 2 - lower / 2 <= tolerance:
            return midpoint
        if midpoint == lower or midpoint == upper:
            raise ArithmeticError("Requested tolerance is below floating-point resolution")
        if (middle_value > 0) == (left_value > 0):
            lower, left_value = midpoint, middle_value
        else:
            upper = midpoint
    raise RuntimeError("Bisection did not reach the requested tolerance")


class RunningMoments:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.squared_deviations = 0.0

    def add(self, value):
        if not math.isfinite(value):
            raise ValueError("Observations must be finite")
        self.count += 1
        difference = value - self.mean
        self.mean += difference / self.count
        self.squared_deviations += difference * (value - self.mean)

    def sample_variance(self):
        if self.count < 2:
            raise ValueError("At least two observations are required")
        return self.squared_deviations / (self.count - 1)


class StreamingMedian:
    def __init__(self):
        self.lower_heap = []
        self.upper_heap = []

    def add(self, value):
        if not math.isfinite(value):
            raise ValueError("Observations must be finite")
        heapq.heappush(self.lower_heap, -value)
        heapq.heappush(self.upper_heap, -heapq.heappop(self.lower_heap))
        if len(self.upper_heap) > len(self.lower_heap):
            heapq.heappush(self.lower_heap, -heapq.heappop(self.upper_heap))

    def median(self):
        if not self.lower_heap:
            raise ValueError("No observations")
        if len(self.lower_heap) == len(self.upper_heap):
            return -self.lower_heap[0] / 2 + self.upper_heap[0] / 2
        return -self.lower_heap[0]


def run_checks():
    assert distribution_moments(hypergeometric(20, 5, 4)) == (1, Fraction(12, 19))
    assert coupon_mean(3) == Fraction(11, 2)
    assert derangement_probability(4) == Fraction(3, 8)
    assert die_stopping_value(2) == Fraction(17, 4)
    assert die_stopping_value(3) == Fraction(14, 3)
    assert replication(100, Fraction(6, 5), Fraction(4, 5), Fraction(21, 20), 20, 0) == (
        Fraction(250, 21), Fraction(1, 2), Fraction(-800, 21), Fraction(5, 8)
    )
    assert abs(bisect_root(lambda value: value * value - 2, 0, 2) - math.sqrt(2)) < 1e-10
    moments, medians = RunningMoments(), StreamingMedian()
    for value in (1, 2, 3, 4, 5):
        moments.add(value)
        medians.add(value)
    assert moments.mean == 3
    assert moments.sample_variance() == 2.5
    assert medians.median() == 3
    return True


if __name__ == "__main__":
    run_checks()
    print("Exact examples and numerical checks passed.")
    print("Finite checks catch mistakes; they do not replace general proofs.")
