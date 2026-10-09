# GSPO and ASPO: how should one answer share its gradient?

[中文](policy-ratios.md) · **English**

> Last reviewed: 2026-10-08 · Prerequisites: [PPO clipping](rlhf/ppo-clipping.en.md) and [GRPO](rlhf/after-rlhf.en.md)

An answer is correct overall, yet one token has become less likely. Should training raise the whole answer together, or give that token extra attention? GSPO and ASPO take different approaches. Start with where their weights enter the update, rather than the acronyms.

## Separate 3 policies

| Role | What it does here | Why the distinction matters |
| --- | --- | --- |
| Current policy $\pi_\theta$ | Supplies the gradient | Changes with optimizer steps |
| Behavior / old policy $\pi_b$ | Generated the training response; supplies the denominator | Often a frozen old policy in synchronous training; record the actual sampling version under asynchrony |
| Reference $\pi_{\rm ref}$ | Anchors a KL or preference objective | Being frozen does not make it the behavior policy |

For an already sampled response $y$, the token probability ratio is:

$$
\rho_t=\exp\left(\log\pi_\theta(y_t\mid x,y_{<t})-\log\pi_b(y_t\mid x,y_{<t})\right).
$$

Both policies must see the same prefix. Dividing probabilities for different freely generated responses does not have this meaning. Prompt tokens, tool outputs, and padding are not automatically policy actions either.

## GSPO: aggregate the change over a response

[GSPO](https://arxiv.org/html/2507.18071v2) uses a length-normalized sequence ratio and sequence-level clipping. The response tokens share a group-relative advantage $A$:

$$
s=\exp\left(\frac{1}{T}\sum_{t=1}^{T}\log\rho_t\right),\qquad
J=\min\left(sA,\operatorname{clip}(s,1-\epsilon,1+\epsilon)A\right).
$$

This is the **geometric mean** of token ratios, not their arithmetic mean or the raw sequence ratio $\prod_t\rho_t$. Taking the $T$-th root reduces length sensitivity in the numerical scale but changes the weight. It is not an unbiased whole-trajectory importance-sampling identity.

### Two tokens moving in opposite directions

These are invented numbers, not a model measurement. Let $A=1$ and the token ratios be `[2, 0.5]`:

| Method / quantity | Value | Interpretation |
| --- | --- | --- |
| Raw sequence ratio | $2\times0.5=1$ | The probability of this complete response is unchanged |
| GSPO ratio | $\sqrt{2\times0.5}=1$ | Inside the illustrative `[0.8, 1.2]` interval; no clipping |
| Mean token-level PPO surrogate | $(1.2+0.5)/2=0.85$ | With positive advantage, the first token's gain is capped |

On the unclipped GSPO branch, with fixed advantage:

$$
\nabla J=\frac{sA}{T}\sum_t\nabla\log\pi_\theta(y_t\mid x,y_{<t}).
$$

Every token has the same external coefficient $sA/T$, **not the same parameter gradient**: their network Jacobians still differ. The example also reveals a tradeoff: sequence aggregation can conceal large local changes.

```python
import math

def clipped_surrogate(ratio, advantage, lower=0.8, upper=1.2):
    if not 0 < lower <= 1 <= upper or not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("invalid ratio or clipping interval")
    return min(ratio * advantage, min(upper, max(lower, ratio)) * advantage)

def sequence_ratio(log_ratios, action_mask):
    if len(log_ratios) != len(action_mask) or any(type(flag) is not bool for flag in action_mask):
        raise ValueError("one Boolean action mask per token is required")
    selected = [value for value, active in zip(log_ratios, action_mask) if active]
    if not selected or not all(math.isfinite(value) for value in selected):
        raise ValueError("at least one finite action log-ratio is required")
    return math.exp(sum(selected) / len(selected))

ratios = [2.0, 0.5]
assert math.isclose(sequence_ratio([math.log(value) for value in ratios], [True, True]), 1)
assert math.isclose(sum(clipped_surrogate(value, 1) for value in ratios) / 2, 0.85)
assert clipped_surrogate(0.5, -1) == -0.8
```

This checks objective values; it is not a trainer. GSPO thresholds need to be chosen for sequence ratios. The 0.2 margin is only convenient arithmetic, not a training recommendation. Identical group rewards, broken action masks, and misleading rewards do not disappear with GSPO.

## ASPO: reverse the positive-advantage weight

[ASPO](https://arxiv.org/html/2510.06062v1) changes token-level updates: retain an advantage-directed mask, use reciprocal weights for positive advantages and ordinary ratios for negative advantages, and bound extreme weights. The motivation is to treat tokens already moving strongly and those lagging behind differently.

The gradient is easy to get wrong. Write $\ell=\log p_\theta$, with behavior probability $p_b$. The positive branch uses:

$$
\widehat\rho=\frac{p_b p_\theta}{\operatorname{sg}(p_\theta^2)},
\qquad
\text{forward value}=\frac{p_b}{p_\theta},
\qquad
\frac{\partial\widehat\rho}{\partial\ell}=+\frac{p_b}{p_\theta}.
$$

$\operatorname{sg}$ means stop-gradient. A directly differentiable `1 / ratio` has a negative derivative, reversing the intended direction for a good token. **Equal forward values do not imply equal training behavior.**

### One local update

Suppose behavior probability is 0.2, current probability is 0.1, and $A=1$. The ordinary ratio is 0.5; ASPO's positive-branch weight is 2. Before masking or weight caps apply, their coefficients on $\nabla\log p_\theta$ are therefore 0.5 and 2.

This checks the local stop-gradient interpretation by holding the denominator fixed during differentiation, rather than recomputing it after each perturbation.

```python
import math

behavior_probability = 0.2
current_probability = 0.1
frozen_square = current_probability ** 2
current_logp = math.log(current_probability)
step = 1e-6

def detached_denominator_value(log_probability):
    return behavior_probability * math.exp(log_probability) / frozen_square

derivative = (
    detached_denominator_value(current_logp + step)
    - detached_denominator_value(current_logp - step)
) / (2 * step)
assert math.isclose(derivative, 2.0, rel_tol=1e-6)
```

Without masking, dual clipping, and loss reduction, these lines are not a complete ASPO implementation. Continue with the [authors' code](https://github.com/wizard-III/Archer2.0). The v1 paper's token-masking prose contains an upper-bound sign inconsistent with the standard PPO description; check the implementation before turning that sentence into a branch condition.

## Decide with diagnostics, not names

Record rewards, lengths, ratio distributions, and valid-token counts on the same rollouts before changing the loss. Keep data, generation budget, initialization, and evaluation matched where possible.

| Observation | Useful check | Insufficient evidence |
| --- | --- | --- |
| A few extreme token ratios | Does sequence aggregation stabilize updates or hide local outliers? | Mean ratio near 1 |
| Different positive / negative weight distributions | Split by advantage sign; inspect detach and clipping | Any entropy decrease labeled as collapse |
| Fewer active gradients | Report token and sequence mask rates separately | Directly comparing clip fractions at different granularities |
| Better scores and much longer answers | Match inference budgets; inspect correctness and failure types | Treating length as reasoning ability |

These methods change an update rule, not the validity of a reward. Continue with [DAPO's sampling and normalization](dapo.en.md), then [stale data in asynchronous training](async-policy-learning.en.md).
