import math


def bellman(gamma, steps):
    start = charged = 0.0
    for step in range(steps):
        start, charged = max(2.0, gamma * charged), 4.0
    return start, charged


def discounted_return(rewards, gamma):
    return sum(gamma ** step * reward for step, reward in enumerate(rewards))


def gae(rewards, values, next_values, terminated, truncated,
        gamma=0.99, trace_decay=0.95):
    lengths = {len(part) for part in
               (rewards, values, next_values, terminated, truncated)}
    if len(lengths) != 1:
        raise ValueError("trajectory arrays must have equal lengths")
    result = [0.0] * len(rewards)
    carry = 0.0
    for step in reversed(range(len(rewards))):
        bootstrap = 0.0 if terminated[step] else next_values[step]
        residual = rewards[step] + gamma * bootstrap - values[step]
        same_episode = not (terminated[step] or truncated[step])
        carry = residual + gamma * trace_decay * same_episode * carry
        result[step] = carry
    return result


def double_targets(online, target, reward=1.0, gamma=0.9):
    if not online or len(online) != len(target):
        raise ValueError("Q arrays must have equal nonzero lengths")
    action = max(range(len(online)), key=online.__getitem__)
    return reward + gamma * max(target), reward + gamma * target[action]


def effective_sample_size(weights):
    if not weights or any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("weights must be finite and nonnegative")
    total = sum(weights)
    if total == 0:
        raise ValueError("at least one weight must be positive")
    return total ** 2 / sum(weight ** 2 for weight in weights)


def projected_value_step(parameter, gamma=0.9):
    if not math.isfinite(parameter) or not 0 <= gamma < 1:
        raise ValueError("expected a finite parameter and discount in [0, 1)")
    features = (1.0, 2.0)
    target = gamma * features[1] * parameter
    return sum(feature * target for feature in features) / sum(
        feature ** 2 for feature in features)


def bernoulli_kl(old_probability, new_probability):
    if not 0 < old_probability < 1 or not 0 < new_probability < 1:
        raise ValueError("this toy check requires probabilities strictly between 0 and 1")
    return (old_probability * math.log(old_probability / new_probability)
            + (1 - old_probability) * math.log(
                (1 - old_probability) / (1 - new_probability)))


def run_checks():
    assert bellman(0.9, 1) == (2.0, 4.0)
    assert bellman(0.9, 2) == (3.6, 4.0)
    assert bellman(0.4, 2) == (2.0, 4.0)
    assert math.isclose(discounted_return([1, 0, 2], 0.9), 2.62)
    advantages = gae([1, 2, -1], [0, 0, 0], [0, 0, 0],
                     [False, False, True], [False] * 3, 0.9, 0.8)
    assert math.isclose(advantages[0], 1.9216)
    truncated = gae([1, 100], [0, 0], [5, 0],
                    [False, True], [True, False], 0.9, 1.0)
    assert truncated == [5.5, 100.0]
    terminated = gae([1], [0], [5], [True], [False], 0.9, 1.0)
    assert terminated == [1.0]
    assert all(math.isclose(actual, expected) for actual, expected in
               zip(double_targets([5, 4], [2, 6]), [6.4, 2.8]))
    assert math.isclose(effective_sample_size([1, 1, 8]), 100 / 66)
    state = 2.0
    action = -state / 2
    assert state ** 2 + action ** 2 + (state + action) ** 2 == 6.0
    assert math.isclose(projected_value_step(1), 1.08)
    assert math.isclose(projected_value_step(1, 0.5), 0.6)
    step = math.sqrt(2 * 0.01 / 0.25)
    new_probability = 1 / (1 + math.exp(-step))
    assert math.isclose(new_probability, 0.57024, abs_tol=1e-5)
    assert math.isclose(bernoulli_kl(0.5, new_probability), 0.009967, abs_tol=1e-6)
    print("PASS: Bellman, return, GAE boundaries, Double DQN, ESS, LQR, projection, KL")


if __name__ == "__main__":
    run_checks()
