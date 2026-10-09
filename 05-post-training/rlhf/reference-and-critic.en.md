# Reference and Critic: measuring drift versus estimating return

[中文](reference-and-critic.md) · **English**

> Reading time: ~8–10 min · Last reviewed: 2026-10

The reward model scores an answer at 2, the Critic predicts 1, and the Reference produces a sequence of log-probabilities. All three return numbers, but they answer different questions. Separate the questions before working through the equations.

We use typical PPO-style RLHF as an example, not a mandatory template for every post-training method. For a numerical starting point, jump to the [two-answer example](#kl-example).

## 1. Five roles, not necessarily five model copies

| Role | Its question | Typical update behavior in PPO-style RLHF |
| --- | --- | --- |
| Actor $\pi_\theta$ | What should I generate now? | Updated by the policy loss |
| Rollout policy $\pi_{\rm old}$ | How was this batch generated? | Behavior probabilities stay fixed for the batch; refreshed next round |
| Reference $\pi_{\rm ref}$ | How far have we moved from the chosen anchor? | Usually frozen, often initialized from SFT |
| Reward Model $r_\phi$ | Does this complete answer match learned preferences? | Frozen during a typical PPO stage |
| Critic $V_\psi$ | What return should follow this prefix? | Fits returns under the current policy |

These are five **roles**, not five full weight copies that must coexist in GPU memory. Rollout-policy information can be stored as sampled-token log-probabilities; Actor and Critic may share parameters. Memory depends on implementation, not box counts. [InstructGPT](https://arxiv.org/abs/2203.02155) is a classic instance of the three-stage pipeline.

For example, ask, “Why does the model perform well on training data but poorly on test data?” Post-training may make the Actor more likely to give a clear, accurate explanation. The Reference retains its anchor parameters and assigns probabilities to **those same answers to the same question**, letting us measure how generation preferences changed. It may itself favor an inaccurate answer: **a reference model is not a reference answer or an oracle of correctness**.

<span id="why-the-reference-is-mandatory"></span>

## 2. Reference KL: charge for moving away from an anchor

For prompt $x$ and complete answer $y$, a common objective is:

$$
J(\theta)=\mathbb E_x\left[\mathbb E_{y\sim\pi_\theta(\cdot|x)}r_\phi(x,y)
-\beta D_{\rm KL}\big(\pi_\theta(\cdot|x)\|\pi_{\rm ref}(\cdot|x)\big)\right].
$$

KL compares **answer distributions**; it is not an intrinsic property of one answer. The policy pays for moving away from its reference while seeking reward. This does not define correctness or guarantee protection against reward hacking.

Expanding the definition expresses KL as an expectation:

$$
D_{\rm KL}(\pi_\theta\|\pi_{\rm ref})
=\mathbb E_{y\sim\pi_\theta}\left[\log\frac{\pi_\theta(y|x)}{\pi_{\rm ref}(y|x)}\right].
$$

This suggests a penalized reward for a sampled answer:

$$
\widetilde r(x,y)=r_\phi(x,y)-\beta\log\frac{\pi_\theta(y|x)}{\pi_{\rm ref}(y|x)}.
$$

**A sampled log-ratio can be negative; its KL expectation is nonnegative.** Under autoregressive generation, the sequence log-ratio is a sum over generated tokens, including the end token when it belongs to the sequence definition. Replacing that sum with a mean changes the objective.

This identity assumes sampling from $\pi_\theta$, reference support covering the policy, and a well-defined expectation. PPO reuses samples from $\pi_{\rm old}$, so distinguish rewards fixed at collection time from KL estimates recomputed during updates. A tensor named `kl` is not necessarily the current policy's full-distribution KL.

For the $k_1$, $k_2$, and $k_3$ expressions found in training code, read [three KL estimators](kl-estimators.en.md). A shared two-token example checks their values, sampling assumptions, and gradients separately.

## 3. Work through two answers {#kl-example}

Suppose A and B are the only complete answers. The Reference assigns each 0.5; the Actor assigns 0.8 and 0.2. To isolate KL, give both answers reward 1 and set $\beta=0.2$. These are teaching numbers, not experimental results.

| Answer | Actor probability | $\log(\pi_\theta/\pi_{\rm ref})$ | Penalized reward |
| --- | --- | --- | --- |
| A | 0.8 | 0.4700 | 0.9060 |
| B | 0.2 | −0.9163 | 1.1833 |

B receives a small bonus because its probability is lower than under the reference. After weighting by Actor probabilities, KL is approximately 0.1927 and the objective is 0.9615, still below the unpenalized value of 1.

```python
import math

policy = [0.8, 0.2]
reference = [0.5, 0.5]
beta = 0.2
log_ratios = [math.log(current / anchor)
              for current, anchor in zip(policy, reference)]
divergence = sum(probability * ratio
                 for probability, ratio in zip(policy, log_ratios))
shaped_rewards = [1 - beta * ratio for ratio in log_ratios]
objective = sum(probability * reward
                for probability, reward in zip(policy, shaped_rewards))
assert log_ratios[1] < 0 < divergence
assert math.isclose(divergence, 0.19274475702175753)
assert math.isclose(objective, 1 - beta * divergence)
print(round(divergence, 4), round(objective, 4))
```

Setting `beta` to zero removes this cost; it does not invalidate the mathematics. Outcomes need measurement. With reward scale held fixed, a larger coefficient usually favors reference behavior and a smaller one permits more change. **There is no theorem requiring a nonzero coefficient for every task.** Existing implementations allow disabling it; consult the version and options in the [TRL GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer). That is not a recommendation to disable it in your task.

<span id="why-a-critic-as-well"></span>

## 4. Critic: the same reward can mean different things

An easy task already averages 0.9; a difficult task averages 0.1. Receiving 1 gives residuals of 0.1 and 0.9. The question is how much better the outcome was than expected, not what to rename the score.

Let $G_t$ be return from position $t$. A Monte Carlo advantage estimate is $\hat A_t=G_t-V_\psi(s_t)$. Why subtract a baseline? For fixed state $s$ and a baseline $b(s)$ independent of the current action:

$$
\mathbb E_{a\sim\pi_\theta}[b(s)\nabla_\theta\log\pi_\theta(a|s)]
=b(s)\nabla_\theta\sum_a\pi_\theta(a|s)=0.
$$

Under conditions permitting interchange of differentiation and summation, this term does not change the expected gradient. Treat the baseline as a fixed target in the Actor loss rather than accidentally differentiating through advantage into the Critic. See the [Spinning Up baseline derivation](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#baselines-in-policy-gradients).

A suitable baseline can reduce variance; an arbitrary one need not help, and REINFORCE can train without a Critic. GAE with imperfect value bootstrapping also introduces bias–variance choices that the zero-expectation identity alone does not resolve. See the [GAE paper](https://arxiv.org/abs/1506.02438) and our [worked calculation](../deep-rl/actor-critic-gae.en.md).

## 5. PPO clipping addresses a different change

PPO divides the current policy probability by the **policy that collected this batch**, not by the Reference:

$$
\rho_t=\frac{\pi_\theta(a_t|s_t)}{\pi_{\rm old}(a_t|s_t)},\qquad
L_t=\min\left(\rho_t\hat A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right).
$$

For $\hat A_t=2$ and $\epsilon=0.2$, increasing $\rho_t$ from 1.2 to 1.5 leaves this term at 2.4. It removes further incentive in that direction; it **does not force the actual ratio back to 1.2**. Other samples share parameters, and successive updates interact.

The [official PPO teaching implementation](https://spinningup.openai.com/en/latest/algorithms/ppo.html) explicitly warns that clipping can still allow excessive movement, motivating KL monitoring and early stopping. PPO-Clip does not strictly enforce TRPO's KL constraint.

## 6. Checks worth doing before a larger run

| Check | Possible mistake | How to inspect it |
| --- | --- | --- |
| Ratio denominator | Recomputing old probabilities every update, or using Reference instead | Keep old log-probabilities fixed per rollout; identical policies should initially give ratios near 1 |
| Response mask | Including prompt or padding in the loss | Hand-check two responses with different lengths |
| KL units | Mixing token means with sequence sums | Record length, sequence sum, and token mean |
| Reward scale | Multiplying rewards by ten while treating the old $\beta$ as equivalent | Specify normalization before comparing coefficients |
| Critic target | Confusing returns with advantages, or bootstrapping through termination | Check value targets and advantages on a short trajectory |

Continue with [who generated the data](on-off-policy.en.md), then [GRPO, DPO, and RLVR](after-rlhf.en.md). These address three connected questions: the objective, its gradient estimate, and its data source.
