from collections import Counter
from itertools import product
import math


def probability_vector(values):
    probabilities = tuple(float(value) for value in values)
    if not probabilities or any(not math.isfinite(value) or value < 0 for value in probabilities):
        raise ValueError("Probabilities must be finite, nonnegative, and nonempty")
    if not math.isclose(math.fsum(probabilities), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Probabilities must sum to one")
    return probabilities


def filtered_distribution(proposal, acceptance):
    probabilities = probability_vector(proposal)
    acceptance = tuple(float(value) for value in acceptance)
    if len(acceptance) != len(probabilities):
        raise ValueError("Proposal and acceptance lengths differ")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in acceptance):
        raise ValueError("Acceptance probabilities must lie in [0, 1]")
    masses = tuple(probability * accepted for probability, accepted in zip(probabilities, acceptance))
    rate = math.fsum(masses)
    if rate == 0:
        raise ValueError("No accepted mass; conditional distribution is undefined")
    return rate, tuple(mass / rate for mass in masses)


def rejection_acceptance(target, proposal, envelope):
    target = probability_vector(target)
    proposal = probability_vector(proposal)
    if len(target) != len(proposal):
        raise ValueError("Target and proposal lengths differ")
    if not math.isfinite(envelope) or envelope <= 0:
        raise ValueError("Envelope must be positive and finite")
    acceptance = []
    for desired, proposed in zip(target, proposal):
        if desired > envelope * proposed:
            raise ValueError("Envelope does not dominate the target")
        acceptance.append(desired / (envelope * proposed) if proposed else 0.0)
    return tuple(acceptance)


def best_of_n_distribution(probabilities, scores, samples):
    probabilities = probability_vector(probabilities)
    scores = tuple(float(score) for score in scores)
    if len(scores) != len(probabilities) or any(not math.isfinite(score) for score in scores):
        raise ValueError("One finite score is required per category")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("Sample count must be a positive integer")
    if len(probabilities) ** samples > 100000:
        raise ValueError("Teaching enumeration is limited to 100000 outcomes")
    winners = [0.0] * len(probabilities)
    for draws in product(range(len(probabilities)), repeat=samples):
        probability = math.prod(probabilities[index] for index in draws)
        winner = max(draws, key=lambda index: scores[index])
        winners[winner] += probability
    return tuple(winners)


def entropy(probabilities):
    probabilities = probability_vector(probabilities)
    return -math.fsum(value * math.log(value) for value in probabilities if value)


def cross_entropy(sampling, scored):
    sampling = probability_vector(sampling)
    scored = probability_vector(scored)
    if len(sampling) != len(scored):
        raise ValueError("Distribution lengths differ")
    if any(weight > 0 and value == 0 for weight, value in zip(sampling, scored)):
        return math.inf
    return -math.fsum(weight * math.log(value) for weight, value in zip(sampling, scored) if weight)


def entropy_means(rows, masks):
    rows, masks = tuple(rows), tuple(masks)
    if not rows or len(rows) != len(masks):
        raise ValueError("Require matching nonempty rows and masks")
    selected_rows = []
    for values, mask in zip(rows, masks):
        values, mask = tuple(values), tuple(mask)
        if len(values) != len(mask) or any(type(keep) is not bool for keep in mask):
            raise ValueError("Each entropy value needs a Boolean mask")
        selected = tuple(value for value, keep in zip(values, mask) if keep)
        if not selected or any(not math.isfinite(value) or value < 0 for value in selected):
            raise ValueError("Each response needs finite nonnegative valid entropies")
        selected_rows.append(selected)
    response_mean = math.fsum(math.fsum(row) / len(row) for row in selected_rows) / len(selected_rows)
    token_mean = math.fsum(math.fsum(row) for row in selected_rows) / sum(map(len, selected_rows))
    return response_mean, token_mean


def homogeneous_group_probability(success, samples):
    if not math.isfinite(success) or not 0 <= success <= 1:
        raise ValueError("Success probability must lie in [0, 1]")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("Group size must be a positive integer")
    return success ** samples + (1 - success) ** samples


def binary_reward_update(probability, step):
    if not math.isfinite(probability) or not 0 < probability < 1:
        raise ValueError("Binary probability must lie strictly between zero and one")
    if not math.isfinite(step) or step < 0:
        raise ValueError("Step must be finite and nonnegative")
    logit = math.log(probability / (1 - probability))
    updated_logit = logit + step * probability * (1 - probability)
    if updated_logit >= 0:
        updated = 1 / (1 + math.exp(-updated_logit))
    else:
        exponent = math.exp(updated_logit)
        updated = exponent / (1 + exponent)
    entropy_derivative = (probability * (1 - probability)) ** 2 * math.log((1 - probability) / probability)
    return updated, entropy_derivative


def verifies_sort(original, candidate):
    ordered = all(
        left <= right
        for left, right in zip(candidate, candidate[1:])
    )
    return ordered and Counter(original) == Counter(candidate)


def main():
    rate, retained = filtered_distribution((0.5, 0.3, 0.2), (1.0, 0.5, 0.0))
    print("Filter:", round(rate, 6), [round(value, 6) for value in retained])
    acceptance = rejection_acceptance((0.6, 0.3, 0.1), (1 / 3,) * 3, 1.8)
    print("Classical accept-reject:", filtered_distribution((1 / 3,) * 3, acceptance))
    print("Best-of-2:", best_of_n_distribution((0.5, 0.3, 0.2), (0, 1, 2), 2))
    print("Entropy:", entropy((0.5, 0.5)), entropy((0.9, 0.1)))
    print("Response/token means:", entropy_means([[1] * 2, [0.1] * 8], [[True] * 2, [True] * 8]))
    print("Homogeneous groups:", [homogeneous_group_probability(success, 4) for success in (0.1, 0.5, 0.9)])
    for probability in (0.2, 0.8):
        print("Binary learning:", probability, binary_reward_update(probability, 0.01))
    print("Sorting:", [verifies_sort([3, 1, 1], candidate) for candidate in ([1, 1, 3], [1, 3], [])])


if __name__ == "__main__":
    main()
