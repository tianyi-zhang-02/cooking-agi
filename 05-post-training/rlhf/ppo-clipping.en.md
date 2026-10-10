# PPO clipping: why not keep increasing the probability?

[中文](ppo-clipping.md) · **English**

> Reading time: ~9 min · Level: core · Last reviewed: 2026-10-09

The model generates an answer, receives good feedback, and becomes more likely to produce that answer. So far, so reasonable.

But PPO trains on the same batch for several rounds. If one good result keeps driving large changes, those old samples can soon become poor guides to the current model. Clipping stops adding encouragement once a particular direction has already changed enough.

Start with what it does to one update.

## After a correct answer, how does the probability change? {#worked-update}

Reduce the task to one token: ask “What is 2 + 3?” and allow only “5” or “6.” A correct answer earns 1; an incorrect answer earns 0.

Suppose the rollout policy answers correctly 40% of the time, and the Critic estimates a return of 0.4. This sample answers “5” and earns 1, exceeding that expectation by $1-0.4=0.6$. That difference is the advantage in this example.

After some updates, the probability of “5” reaches 60%. Relative to the original 40%, the ratio is $0.6/0.4=1.5$. Multiplying ratio by advantage gives $1.5\times0.6=0.9$; making “5” even more likely would keep increasing this term.

PPO adds a clipped branch. With $\epsilon=0.2$ and positive advantage, that branch uses a ratio no greater than 1.2. The objective takes the smaller result:

$$
\min(1.5\times0.6,\;1.2\times0.6)=0.72.
$$

Increasing the probability further no longer increases this term. Yet **the actual probability can still be 60%**. Clipping does not force it back to 48%; it stops adding this particular incentive.

If the sample had answered “6,” it would fall below expectation, with advantage $0-0.4=-0.4$. Training would favor lowering its probability instead. The clipping direction reverses too, as we will see below.

## Back to language models

The example considered one token and omitted a KL reward term. For a longer answer, the calculation is still made at each generated position: state $s_t$ is the prompt plus prefix, and action $a_t$ is the sampled token.

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)}.
$$

$$
\begin{aligned}
c_t&=\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon),\\
\ell_t&=\min(\rho_t\hat A_t,c_t\hat A_t).
\end{aligned}
$$

Here $t$ is the generation position, not the training iteration. The denominator comes from **the policy that generated this batch** and stays fixed while the batch is reused. The Reference is a different training anchor; it does not supply this denominator.

PPO maximizes an objective built from these sampled terms; code usually minimizes its negative. The earlier 0.72 was one sample's contribution, not the whole batch's average score.

## Compare positive and negative advantages

The figure uses $A=1$ and $A=-1$ to show the two directions. Switch the sign and drag $\rho$; its vertical-axis values therefore differ from the earlier example with $A=0.6$.

<!-- widget:tx-ppo-clip -->

| Advantage | Probability change | Effect of clipping |
|---|---|---|
| $A_t>0$ | Increases | Stop adding incentive beyond ratio $1+\epsilon$ |
| $A_t>0$ | Decreases | Retain the gradient favoring an increase |
| $A_t<0$ | Decreases | Stop adding incentive below ratio $1-\epsilon$ |
| $A_t<0$ | Increases | Retain the gradient favoring a decrease |

This is not a flat cutoff at both ends. **A large move in the direction favored by the advantage stops earning more encouragement; a move against that direction still needs correction.**

<details markdown="1">
<summary>How does one min produce both clipping directions?</summary>

A negative $A_t$ reverses the ordering under multiplication. Written separately:

$$
\begin{aligned}
A_t>0:\quad\\
\ell_t=\min(\rho_t,1+\epsilon)A_t,\\[6pt]
A_t<0:\quad\\
\ell_t=\max(\rho_t,1-\epsilon)A_t.
\end{aligned}
$$

Positive advantage clips only the upper side; negative advantage clips only the lower side. With $A_t=0$, this sampled term is zero. Inside the plateau, its local derivative with respect to the ratio is zero; the boundary itself is a nondifferentiable point of the piecewise function.

</details>

## Does a clipped token still have a gradient? {#clipped-token-gradients}

Not every ratio outside the clipping interval loses its gradient. Computing `clamp` is only part of the objective: `min` must select that flat branch, and the advantage sign determines when it does.

Fix the old probability at 0.5 and $\epsilon=0.2$. These are 4 separate sampled positions. Use the loss that code minimizes, $L_t=-\ell_t$:

| Current probability | Ratio | Advantage | $L_t$ | Gradient with respect to current log-probability |
|---|---|---|---|---|
| 0.7 | 1.4 | +1 | −1.2 | **0**: enough movement in the favored direction |
| 0.3 | 0.6 | +1 | −0.6 | **−0.6**: a favored token became less likely; correct it |
| 0.3 | 0.6 | −1 | +0.8 | **0**: the unfavorable token is already much less likely |
| 0.7 | 1.4 | −1 | +1.4 | **+1.4**: an unfavorable token became more likely; correct it |

All 4 ratios lie outside the interval; only 2 terms have zero gradient. These derivatives are with respect to the **selected log-probability**, not directly to a logit or model parameter. Backpropagation through the rest of the model still follows. Old log-probabilities and advantages are fixed during this update.

<details markdown="1">
<summary>Check all 4 cases in PyTorch</summary>

This runs on CPU. Treating selected log-probabilities as leaf variables isolates the loss calculation; in a real model they come from a log-softmax over the vocabulary.

```python
import math
import torch

current_log_probs = torch.tensor(
    [0.7, 0.3, 0.3, 0.7], dtype=torch.float64
).log().requires_grad_()
old_log_probs = torch.full_like(current_log_probs, math.log(0.5))
advantages = torch.tensor([1., 1., -1., -1.], dtype=torch.float64)
ratios = (current_log_probs - old_log_probs).exp()
direct = ratios * advantages
clipped = ratios.clamp(0.8, 1.2) * advantages
losses = -torch.minimum(direct, clipped)
losses.sum().backward()
torch.testing.assert_close(
    current_log_probs.grad,
    torch.tensor([0., -0.6, 0., 1.4], dtype=torch.float64),
)
print(losses.detach().tolist(), current_log_probs.grad.tolist())
```

Replacing `sum()` with `mean()` divides the gradients by 4 without changing which are zero. The examples avoid the nondifferentiable boundaries at 0.8 and 1.2 so finite differences can check the derivatives.

</details>

### A flat term does not freeze the model {#other-gradient-paths}

Even when this token's policy-loss term is genuinely flat, its probability can still change.

| What else can update? | Small example |
|---|---|
| Other losses, such as KL or entropy | The current two-action distribution is 0.7/0.3 and the Reference is 0.5/0.5. The policy term can be flat while KL still favors moving toward the Reference |
| Other tokens and samples | Two different prompts share a parameter. The first sample contributes no gradient, but the second changes that parameter and therefore the first prompt's output |

Not every implementation uses a KL penalty or entropy bonus; check the actual configuration. Expand the two-action examples below if you want to follow the numbers.

<details markdown="1">
<summary>How large can the other gradients be when the policy term is flat?</summary>

The [CPU teaching script](../code/policy_gradient_checks.py) tests both mechanisms separately:

- **Other losses:** current probabilities are 0.7/0.3, both old and Reference probabilities are 0.5/0.5, and the first action has advantage +1. Add $0.1\,\mathrm{KL}(\pi\|\pi_{\rm ref})$ to the policy loss and subtract $0.01H(\pi)$. With respect to the first action's logit, the policy gradient is 0 but the total gradient is about 0.0196. This toy uses **exact KL summed over both actions**, not a sampled KL estimator.
- **Shared parameters:** two different prompts use the same scalar logit for their selected action. Both current probabilities are 0.7 and old probabilities are 0.5, with advantages +1 and −1. The first term is flat; the second is not. Their mean has gradient 0.21 with respect to the shared logit, so an update still lowers the first action's probability.

</details>

The precise claim is that **this term contributes no local gradient through this policy-loss branch once it is on the plateau**. It does not mean the token is frozen. If an implementation changes the objective or clipping rule, follow its computation graph rather than drawing conclusions from the presence of `clamp` alone.

## What clipping does not fix

It cannot tell whether the scoring standard is sensible. If a reward model favors verbose answers, the resulting advantages may encourage verbosity. The “good direction” is relative to the current estimate, not a guarantee of better outcomes for users.

It also does not impose a hard KL limit on the entire parameter update. Samples share parameters, and changes accumulate across rounds. [Spinning Up's PPO notes](https://spinningup.openai.com/en/latest/algorithms/ppo.html) therefore describe early stopping based on KL. PPO clipping is also different from gradient clipping, which acts directly on gradients.

GRPO often retains this kind of clipped objective while changing the advantage estimate. Continue with [GRPO, DPO, and RLVR](after-rlhf.en.md), or revisit [Reference and Critic](reference-and-critic.en.md) if the old policy and Reference still seem interchangeable.
