# How much should a policy change? From natural gradient to PPO

[中文](trust-region.md) · **English**

> Reading time: ~8 min · Last reviewed: 2026-10

A gradient gives a direction, not a safe distance. Adding 0.1 to one parameter may barely change action probabilities; adding it to another may change them substantially. **A small parameter step is not necessarily a small behavioral change.**

Read [policy gradients](policy-gradients.en.md) and [GAE](actor-critic-gae.en.md) first. This chapter concerns step size rather than another advantage derivation.

## Why old data is useful locally

Freeze states and actions collected by the old policy and define a surrogate objective:

$$
L(\theta)=\mathbb E_{s\sim d_{\rm old},\,a\sim\pi_{\rm old}}
\left[\frac{\pi_\theta(a\mid s)}{\pi_{\rm old}(a\mid s)}A^{\pi_{\rm old}}(s,a)\right].
$$

Here $d_{\rm old}$ is the normalized discounted state-visitation distribution; assume exact advantages for now. For infinite-horizon discounted return, $\nabla J=\nabla L/(1-\gamma)$ at the old parameters, so the directions agree. Farther away, the new policy visits different states, and an objective using fixed old states no longer exactly represents the new return.

The ratio corrects **action probabilities at a given state**. It does not transform old states into those visited by the new policy. Support is also necessary: dividing by a zero old probability cannot recover an action the behavior policy never sampled.

## Measure policy change rather than parameter distance

Consider average KL from the old policy to the new one:

$$
\bar D(\theta)=\mathbb E_{s\sim d_{\rm old}}
D_{\rm KL}\!\left(\pi_{\rm old}(\cdot\mid s)\,\|\,\pi_\theta(\cdot\mid s)\right).
$$

At the old parameters, both the KL and its first derivative are zero. Under smoothness and regular support conditions, a second-order expansion gives:

$$
\bar D(\theta_{\rm old}+\Delta)\approx\tfrac12\Delta^\top F\Delta,
\qquad F=\mathbb E_{s,a\sim d_{\rm old},\pi_{\rm old}}
\left[\nabla\log\pi_{\rm old}(a\mid s)\nabla\log\pi_{\rm old}(a\mid s)^\top\right].
$$

$F$ is the Fisher information matrix. Parameter directions that strongly change action probabilities incur a larger cost. The factor $1/2$ comes from Taylor expansion. Keep the KL direction and expectation distribution explicit rather than memorizing only the matrix notation.

## Solve for direction and scale together

Let $g=\nabla L(\theta_{\rm old})$ and solve the local approximation:

$$
\max_\Delta g^\top\Delta
\quad\text{s.t.}\quad \tfrac12\Delta^\top F\Delta\le\delta.
$$

With positive-definite $F$ and nonzero $g$, the Lagrange condition is $g-\eta F\Delta=0$, yielding direction $F^{-1}g$. Substituting into the constraint gives:

$$
\Delta^*=\sqrt{\frac{2\delta}{g^\top F^{-1}g}}F^{-1}g.
$$

This is the natural-gradient direction scaled by a KL budget. Implementations avoid building a huge inverse: Fisher-vector products and conjugate gradient solve the system, often with damping for conditioning. Damping and finite-sample estimates depart from the exact idealized problem above.

## Work through a coin policy

Action 1 earns 1 and action 0 earns 0, with $p=\pi(1)=\sigma(\theta)$. Start at $\theta=0$:

| Quantity | Value | Reason |
| --- | --- | --- |
| Current $p$ | 0.5 | $\sigma(0)=0.5$ |
| Return gradient $g$ | 0.25 | $J=p$ and $dp/d\theta=p(1-p)$ |
| Fisher $F$ | 0.25 | The Bernoulli score is $a-p$, with second moment $p(1-p)$ |
| KL budget $\delta$ | 0.01 | Chosen for this example |
| Proposed step $\Delta$ | 0.28284 | $\sqrt{2\delta/F}$ |
| Updated $p$ | 0.57024 | $\sigma(0.28284)$ |

The exact Bernoulli KL is approximately 0.009967. It is close to 0.01 here, but a local quadratic approximation does not guarantee every step satisfies the exact KL budget.

<div class="drl-lab" data-drl-lab="trust-step"><p>Change the logit step to compare probabilities, exact KL, and its quadratic approximation. At Δ=0.3, exact KL is about 0.01121, above δ=0.01. Changing the budget does not change the proposed step.</p></div>

<details markdown="1">
<summary>What if we simply add 2 to the logit?</summary>

The new probability is about 0.8808, apparently a larger improvement, but the exact KL is about 0.4338, far beyond the budget. This noiseless bandit has known rewards, so the large step happens to be harmless. In a complex environment, advantages are estimated and new states are uncertain. The example does not establish that larger steps are generally better.

</details>

## TRPO and PPO offer different controls

TRPO uses the local objective and KL constraint to propose an update, then backtracks to check sampled improvement and measured KL. The practical algorithm uses average KL, finite data, and approximate solves; its theoretical monotonic-improvement result is not a guarantee for every training update.

PPO-Clip instead uses a cheaper first-order objective. Write the probability ratio as $\rho_t(\theta)=\pi_\theta(a_t\mid s_t)/\pi_{\rm old}(a_t\mid s_t)$, avoiding $r_t$, which denotes immediate reward in these notes:

$$
L_{\rm clip}=\mathbb E\left[\min\left(\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right].
$$

It stops rewarding large ratio changes in certain directions. **It does not hard-constrain every probability ratio to the clipping interval**, nor impose a strict KL bound. Shared parameters, other samples, and other losses can still move ratios farther. Monitor KL, clip fraction, entropy, and actual task performance. See the [PPO clipping visualization](../rlhf/ppo-clipping.en.md).

## Calculate all four PPO cases

Take $\epsilon=0.2$. The table shows the per-sample objective to maximize; negate it for gradient descent.

| Advantage | Ratio $\rho$ | Unclipped term | Clipped term | Minimum |
| --- | --- | --- | --- | --- |
| +2 | 1.3 | 2.6 | 2.4 | 2.4: no further incentive to increase probability |
| +2 | 0.7 | 1.4 | 1.6 | 1.4: retains gradient in the worsening direction |
| −2 | 0.7 | −1.4 | −1.6 | −1.6: no further incentive to decrease probability |
| −2 | 1.3 | −2.6 | −2.4 | −2.6: retains gradient in the worsening direction |

A ratio outside the interval does not always give zero gradient. Clipping removes additional incentive for moving too far in the direction this sample considers helpful. Other samples, value loss, or entropy loss can still move shared parameters.

## Why track KL and data freshness?

Under old-policy sampling, the mean of $\log\pi_{\rm old}(a\mid s)-\log\pi_\theta(a\mid s)$ estimates forward KL. Its expectation is nonnegative, but a finite-sample estimate can be negative. Distinguish it from exact enumeration over actions.

Clip fraction measures ratios outside the interval, not samples with no gradient or policy distance. Read it alongside KL, entropy, and actual return.

More epochs reuse costly data but move the policy farther from its collection version. Measured-KL early stopping and fewer updates are practical controls, not strict guarantees. Asynchronous sampling also needs policy-version tracking: older-policy trajectories are not fresh data from the current rollout batch.

## Implementation checks before tuning

- Freeze old log-probabilities throughout updates on a collected batch; do not redefine them after each optimizer step.
- Detach advantages from Actor gradients; compute ratios by exponentiating log-probability differences.
- Match masks and reductions to valid samples; multiple epochs are not fresh interaction.
- State whether a reported KL is an average or another estimator. A small average does not bound every state.

References: [Original TRPO paper](https://proceedings.mlr.press/v37/schulman15.html) · [Original PPO paper](https://arxiv.org/abs/1707.06347). This chapter derives the local constrained step, not the complete improvement bound or conjugate-gradient implementation.
