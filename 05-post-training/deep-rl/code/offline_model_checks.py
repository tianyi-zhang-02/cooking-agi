import math


def penalized_return(rewards, uncertainties, penalty, discount):
    if not rewards or len(rewards) != len(uncertainties):
        raise ValueError("Rewards and uncertainties need equal nonzero lengths")
    if (not math.isfinite(penalty) or penalty < 0
            or not math.isfinite(discount) or not 0 <= discount <= 1):
        raise ValueError("Use a nonnegative finite penalty and a discount in [0, 1]")
    if (any(not math.isfinite(reward) for reward in rewards)
            or any(not math.isfinite(value) or value < 0 for value in uncertainties)):
        raise ValueError("Rewards must be finite and uncertainties nonnegative and finite")
    adjusted = [reward - penalty * uncertainty
                for reward, uncertainty in zip(rewards, uncertainties)]
    raw_return = math.fsum(discount ** step * reward for step, reward in enumerate(rewards))
    total_penalty = penalty * math.fsum(discount ** step * value
                                       for step, value in enumerate(uncertainties))
    return {
        "adjusted_rewards": adjusted,
        "raw_return": raw_return,
        "penalty": total_penalty,
        "adjusted_return": math.fsum(discount ** step * reward for step, reward in enumerate(adjusted)),
    }


def halt_values(entry_reward, cost, discount):
    if (not math.isfinite(entry_reward) or not math.isfinite(cost) or cost < 0
            or not math.isfinite(discount) or not 0 <= discount < 1):
        raise ValueError("Use finite rewards, a nonnegative cost, and a discount in [0, 1)")
    halt = -cost / (1 - discount)
    return {"halt": halt, "entry_target": entry_reward + discount * halt}


def check_distribution(probabilities, size):
    if (len(probabilities) != size
            or any(not math.isfinite(value) or value < 0 for value in probabilities)
            or not math.isclose(math.fsum(probabilities), 1, rel_tol=0, abs_tol=1e-10)):
        raise ValueError("Each distribution must match the target size and sum to one")


def conservative_fit(targets, data, conservative, regression, strength):
    if not targets or any(not math.isfinite(target) for target in targets):
        raise ValueError("Use at least one finite fixed target")
    for distribution in (data, conservative, regression):
        check_distribution(distribution, len(targets))
    if (any(weight <= 0 for weight in regression)
            or not math.isfinite(strength) or strength < 0):
        raise ValueError("This solver needs positive regression weights and nonnegative finite strength")
    return [target - strength * (model_mass - data_mass) / weight
            for target, data_mass, model_mass, weight in zip(targets, data, conservative, regression)]


def main():
    print("MOPO example:", penalized_return([2, 4], [0.1, 0.4], 3, 0.9))
    print("HALT and entry:", halt_values(1, 2, 0.9))
    for strength in (1, 2):
        print(f"COMBO fixed-target fit, beta={strength}:",
              conservative_fit([2, 5], [0.9, 0.1], [0.5, 0.5], [0.7, 0.3], strength))


if __name__ == "__main__":
    main()
