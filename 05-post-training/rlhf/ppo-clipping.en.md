# PPO clipping: why not keep increasing the probability?

[中文](ppo-clipping.md) · **English**

> Reading time: ~4 min · Level: core · Last reviewed: 2026-10

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
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)},\qquad
\ell_t=\min\left(\rho_t\hat A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right).
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
\ell_t=
\begin{cases}
\min(\rho_t,1+\epsilon)A_t,& A_t>0,\\
\max(\rho_t,1-\epsilon)A_t,& A_t<0.
\end{cases}
$$

Positive advantage clips only the upper side; negative advantage clips only the lower side. With $A_t=0$, this sampled term is zero. Inside the plateau, its local derivative with respect to the ratio is zero; the boundary itself is a nondifferentiable point of the piecewise function.

</details>

## What clipping does not fix

It cannot tell whether the scoring standard is sensible. If a reward model favors verbose answers, the resulting advantages may encourage verbosity. The “good direction” is relative to the current estimate, not a guarantee of better outcomes for users.

It also does not impose a hard KL limit on the entire parameter update. Samples share parameters, and changes accumulate across rounds. [Spinning Up's PPO notes](https://spinningup.openai.com/en/latest/algorithms/ppo.html) therefore describe early stopping based on KL. PPO clipping is also different from gradient clipping, which acts directly on gradients.

GRPO often retains this kind of clipped objective while changing the advantage estimate. Continue with [GRPO, DPO, and RLVR](after-rlhf.en.md), or revisit [Reference and Critic](reference-and-critic.en.md) if the old policy and Reference still seem interchangeable.
