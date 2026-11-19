import math


def masked_binary_loss(logits, labels):
    if len(logits) != len(labels) or not logits:
        raise ValueError("Logits and labels must have the same nonzero length")
    valid = [index for index, label in enumerate(labels) if label is not None]
    if not valid:
        raise ValueError("At least one label is required")
    gradients = [0.0] * len(logits)
    losses = []
    for index in valid:
        logit, label = logits[index], labels[index]
        if not math.isfinite(logit) or not math.isfinite(label) or not 0 <= label <= 1:
            raise ValueError("Labeled entries need finite logits and labels in [0, 1]")
        losses.append(max(logit, 0) - logit * label + math.log1p(math.exp(-abs(logit))))
        probability = (1 / (1 + math.exp(-logit)) if logit >= 0
                       else math.exp(logit) / (1 + math.exp(logit)))
        gradients[index] = (probability - label) / len(valid)
    return {"loss": math.fsum(losses) / len(valid), "gradient": gradients}


def aggregate_scores(scores):
    if not scores or any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("Use at least one finite score in [0, 1]")
    return {"product": math.prod(scores), "minimum": min(scores), "last": scores[-1]}


def empirical_success_rate(outcomes):
    if not outcomes or any(outcome not in (0, 1) for outcome in outcomes):
        raise ValueError("Use at least one binary outcome")
    return sum(outcomes) / len(outcomes)


def ucb_indices(successes, counts):
    if not counts or len(successes) != len(counts):
        raise ValueError("Successes and counts must have the same nonzero length")
    for success, count in zip(successes, counts):
        if (type(count) is not int or type(success) is not int
                or count <= 0 or not 0 <= success <= count):
            raise ValueError("Each arm needs positive integer trials and valid successes")
    total = sum(counts)
    return [success / count + math.sqrt(2 * math.log(total) / count)
            for success, count in zip(successes, counts)]


def beta_posterior(successes, failures, prior=(1.0, 1.0)):
    if (type(successes) is not int or type(failures) is not int
            or successes < 0 or failures < 0 or len(prior) != 2
            or any(not math.isfinite(value) or value <= 0 for value in prior)):
        raise ValueError("Counts must be nonnegative integers and prior shapes positive")
    alpha, beta = prior[0] + successes, prior[1] + failures
    return {"alpha": alpha, "beta": beta, "mean": alpha / (alpha + beta)}


def probe_information(accuracy):
    if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError("Accuracy must be in [0, 1]")
    posterior = [accuracy, 1 - accuracy]
    entropy = -math.fsum(probability * math.log2(probability)
                         for probability in posterior if probability > 0)
    divergence = math.fsum(probability * math.log2(probability / 0.5)
                           for probability in posterior if probability > 0)
    return {"information_bits": 1 - entropy, "posterior_kl_bits": divergence}


def pseudo_counts(before, after):
    if not (math.isfinite(before) and math.isfinite(after) and 0 < before < after < 1):
        raise ValueError("This example assumes 0 < before < after < 1")
    total = (1 - after) / (after - before)
    return {"local": before * total, "total": total}


def selection_success(samples, correct_rate, true_positive_rate=1.0, false_positive_rate=0.0):
    if type(samples) is not int or samples < 1:
        raise ValueError("Use a positive integer sample count")
    if any(not math.isfinite(rate) or not 0 <= rate <= 1
           for rate in [correct_rate, true_positive_rate, false_positive_rate]):
        raise ValueError("Rates must be in [0, 1]")
    positive_rate = correct_rate * true_positive_rate + (1 - correct_rate) * false_positive_rate
    if positive_rate == 0:
        return correct_rate
    return ((1 - (1 - positive_rate) ** samples)
            * correct_rate * true_positive_rate / positive_rate
            + (1 - positive_rate) ** (samples - 1) * correct_rate * (1 - true_positive_rate))


def main():
    print("Masked reward loss:", masked_binary_loss([math.log(3), math.log(3), math.nan], [1, 0, None]))
    print("Aggregation A:", aggregate_scores([0.7, 0.99]))
    print("Aggregation B:", aggregate_scores([0.82, 0.82]))
    print("Continuation rates:", empirical_success_rate([1, 0, 0, 0]),
          empirical_success_rate([1, 1, 1, 0]))
    print("UCB A, B:", ucb_indices([8, 1], [10, 2]))
    print("Posterior A:", beta_posterior(8, 2))
    print("Posterior B:", beta_posterior(1, 1))
    print("Probe:", probe_information(0.8))
    print("Pseudo-counts:", pseudo_counts(0.2, 0.25))
    print("Equal-budget selection:", selection_success(4, 0.25, 0.9, 0.1),
          selection_success(2, 0.25, 0.9, 0.02))


if __name__ == "__main__":
    main()
