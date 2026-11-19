import math


def finite_values(values):
    result = [float(value) for value in values]
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError("Expected a nonempty sequence of finite numbers")
    return result


def logsumexp(values):
    values = finite_values(values)
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def softmax(values):
    values = finite_values(values)
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    return [weight / sum(weights) for weight in weights]


def maxent_statistics(features, counts, theta):
    features = finite_values(features)
    counts = finite_values(counts)
    if len(features) != len(counts) or min(counts) < 0 or sum(counts) <= 0:
        raise ValueError("Each feature needs a nonnegative count, with positive total")
    if not math.isfinite(theta):
        raise ValueError("theta must be finite")
    scores = [theta * feature for feature in features]
    probabilities = softmax(scores)
    empirical_mean = sum(feature * count for feature, count in zip(features, counts)) / sum(counts)
    model_mean = sum(feature * probability for feature, probability in zip(features, probabilities))
    variance = sum(probability * (feature - model_mean) ** 2
                   for feature, probability in zip(features, probabilities))
    return {
        "probabilities": probabilities,
        "log_partition": logsumexp(scores),
        "log_likelihood": theta * empirical_mean - logsumexp(scores),
        "empirical_mean": empirical_mean,
        "model_mean": model_mean,
        "gradient": empirical_mean - model_mean,
        "hessian": -variance,
    }


def deterministic_tree(theta):
    continuation_scores = [theta, 2 * theta]
    root_scores = [0, logsumexp(continuation_scores)]
    root_policy = softmax(root_scores)
    continuation_policy = softmax(continuation_scores)
    return {
        "root_value": logsumexp(root_scores),
        "root_policy": root_policy,
        "leaf_probabilities": [root_policy[0]] + [root_policy[1] * probability
                                                  for probability in continuation_policy],
    }


def stochastic_choice():
    causal_policy = softmax([1, 0.5 * 0 + 0.5 * 2])
    naive_scores = [1, logsumexp([0, 2]) - math.log(2)]
    return {
        "causal_safe_risky": causal_policy,
        "naive_safe_risky": softmax(naive_scores),
        "actual_lucky_given_risky": 0.5,
        "naive_lucky_given_risky": softmax([0, 2])[1],
    }


def distribution(values):
    values = finite_values(values)
    if min(values) < 0 or not math.isclose(sum(values), 1, abs_tol=1e-12, rel_tol=1e-12):
        raise ValueError("Expected a normalized nonnegative distribution")
    return values


def gail_statistics(expert, learner):
    expert, learner = distribution(expert), distribution(learner)
    if len(expert) != len(learner):
        raise ValueError("Distributions must have equal lengths")
    discriminator = [expert_mass / (expert_mass + learner_mass)
                     if expert_mass + learner_mass else 0.5
                     for expert_mass, learner_mass in zip(expert, learner)]
    value = 0.0
    divergence = 0.0
    for expert_mass, learner_mass, probability in zip(expert, learner, discriminator):
        mixture = (expert_mass + learner_mass) / 2
        if expert_mass:
            value += expert_mass * math.log(probability)
            divergence += 0.5 * expert_mass * math.log(expert_mass / mixture)
        if learner_mass:
            value += learner_mass * math.log1p(-probability)
            divergence += 0.5 * learner_mass * math.log(learner_mass / mixture)
    rewards = [-math.log1p(-probability) if probability < 1 else math.inf
               for probability in discriminator]
    return {"expert_probability": discriminator, "rewards": rewards,
            "value": value, "js_divergence": divergence}


def pair_loss(preferred, rejected):
    preferred, rejected = finite_values([preferred, rejected])
    difference = rejected - preferred
    return max(0, difference) + math.log1p(math.exp(-abs(difference)))


def ranking_statistics(scores, order):
    scores = finite_values(scores)
    order = list(order)
    if len(order) != len(scores) or sorted(order) != list(range(len(scores))):
        raise ValueError("order must be a complete permutation without ties")
    gradient = [0.0] * len(scores)
    loss = 0.0
    for position, selected in enumerate(order[:-1]):
        remaining = order[position:]
        remaining_scores = [scores[index] for index in remaining]
        maximum = max(remaining_scores)
        shifted = [score - maximum for score in remaining_scores]
        loss += logsumexp(shifted) - (scores[selected] - maximum)
        for index, probability in zip(remaining, softmax(remaining_scores)):
            gradient[index] += probability
        gradient[selected] -= 1
    return {"loss": loss, "probability": math.exp(-loss), "gradient": gradient}


def all_pair_losses(scores, order):
    ranking_statistics(scores, order)
    return [pair_loss(scores[preferred], scores[rejected])
            for position, preferred in enumerate(order)
            for rejected in order[position + 1:]]


def comparison_means(groups):
    groups = [finite_values(group) for group in groups]
    if not groups:
        raise ValueError("At least one comparison group is required")
    return {
        "pair_mean": sum(sum(group) for group in groups) / sum(len(group) for group in groups),
        "prompt_mean": sum(sum(group) / len(group) for group in groups) / len(groups),
    }


def main():
    theta = math.log(2)
    before = maxent_statistics([0, 1, 2], [1, 2, 7], theta)
    after = maxent_statistics([0, 1, 2], [1, 2, 7], theta + 0.5 * before["gradient"])
    print("MaxEnt before:", before)
    print("MaxEnt after:", after)
    print("Tree:", deterministic_tree(theta))
    print("Stochastic choice:", stochastic_choice())
    print("GAIL:", gail_statistics([0.75, 0.25], [0.25, 0.75]))
    scores, order = [math.log(4), math.log(2), 0], [0, 1, 2]
    print("Ranking:", ranking_statistics(scores, order))
    print("Pair loss sum:", sum(all_pair_losses(scores, order)))
    print("Comparison weighting:", comparison_means([[0.2], [0.8] * 6]))


if __name__ == "__main__":
    main()
