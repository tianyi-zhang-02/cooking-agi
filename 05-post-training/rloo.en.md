# RLOO: use the other answers as a baseline

[中文](rloo.md) · **English**

> Reviewed: 2026-11 · Prerequisites: [policy gradients](deep-rl/policy-gradients.en.md), [PPO, GRPO, and DPO](after-ppo.en.md)

Sample three answers to one question and assign rewards of 1, 0, and 0.5. Is the answer scoring 0.5 good? The score alone gives little context. If the other two answers also average 0.5, it did not outperform them.

**REINFORCE Leave-One-Out (RLOO) uses the other answers' average reward as a baseline**, without training a value model. We will calculate one group before examining why each answer excludes itself. All examples are teaching fixtures, not model experiments.

## Three answers, three baselines {#worked-group}

Suppose a task has two requirements, worth 0.5 points each. Independently sample three responses to the same input:

| Answer | Reward $R_i$ | Other answers' average $b_i$ | Advantage $A_i=R_i-b_i$ |
| --- | ---: | ---: | ---: |
| A: satisfies both | 1 | $(0+0.5)/2=0.25$ | 0.75 |
| B: satisfies neither | 0 | $(1+0.5)/2=0.75$ | −0.75 |
| C: satisfies one | 0.5 | $(1+0)/2=0.5$ | 0 |

A positive advantage contributes a gradient that favors the sampled answer; a negative one contributes the opposite. C has no direct reward-gradient contribution this time. Its probability can still change because the answers share parameters, and regularization may contribute too.

For $K\ge2$ responses to the same prompt $x$:

$$
b_i=\frac{1}{K-1}\sum_{j\ne i}R_j,
\qquad A_i=R_i-b_i.
$$

Each answer takes a turn as the sample being evaluated; the others supply its baseline. [Ahmadian et al., §2.2–2.3](https://arxiv.org/html/2402.14740v1#S2.SS3) apply this estimator to language-model preference training. The multi-sample leave-one-out estimator predates that paper.

## Why exclude the answer's own reward? {#why-leave-one-out}

Fix the prompt. Assume rewards do not depend on the parameters being updated, responses $y_i$ are i.i.d. samples from $\pi_\theta(\cdot\mid x)$, and differentiation can pass through the expectation. Write $g_i=\nabla_\theta\log\pi_\theta(y_i\mid x)$. Then

$$
\mathbb E[g_i\mid x]
=\sum_y\pi_\theta(y\mid x)\nabla_\theta\log\pi_\theta(y\mid x)
=\nabla_\theta\sum_y\pi_\theta(y\mid x)=0.
$$

The baseline uses only the other independent samples, so it is independent of $y_i$ conditional on $x$:

$$
\mathbb E[b_i g_i\mid x]
=\mathbb E[b_i\mid x]\mathbb E[g_i\mid x]=0.
$$

Subtracting it preserves the expected reward gradient. Samples, rewards, and baselines are held fixed during differentiation; gradients do not flow through the baseline. Variance reduction is the motivation, but unbiasedness alone does not guarantee lower variance in every task.

If we instead subtract the group mean $\bar R$, which includes the current answer:

$$
R_i-\bar R=\frac{K-1}{K}(R_i-b_i).
$$

Our example becomes `[0.5, −0.5, 0]`, exactly $2/3$ of the leave-one-out advantages. Under the sampling assumptions above, the naive group-mean estimator also scales the expected gradient by $(K-1)/K$. This is a common scale factor for fixed $K$, but not if different prompts use different group sizes. With clipping, an optimizer, gradient clipping, or additional losses, changing the learning rate alone need not reproduce the same updates.

Independent samples need not have distinct text: two independent draws can happen to match. Conversely, generating a large pool and retaining only its best answers changes the sampling distribution. The proof no longer applies unchanged.

## How does one reward update every token? {#sequence-gradient}

For one on-policy update without KL, minimize the sample loss

$$
\widehat L=-\frac1K\sum_i\operatorname{stopgrad}(A_i)
\sum_{t=1}^{T_i}\log\pi_\theta(y_{i,t}\mid x,y_{i,<t}).
$$

The inner operation is a **sum, not a mean over response length**. A sequence probability is a product of conditional probabilities, so its logarithm is their sum. Dividing by $T_i$ changes the relative weights of responses of different lengths. For masks, EOS, and denominators, see [auditing an RL training batch](../practice/post-training/rl-batch-audit.en.md).

The same $A_i$ weights every sampled action in a response. It does not identify token 7 as the source of an error or learn a separate value for each prefix. Avoiding Critic training also gives up that per-step estimate. Compare [Actor–Critic and GAE](deep-rl/actor-critic-gae.en.md) when stepwise credit assignment matters.

## Which differences from GRPO matter? {#comparison}

| Choice | REINFORCE with a leave-one-out baseline | Common group-relative training |
| --- | --- | --- |
| Baseline | Mean of the other responses | Group mean including the current response |
| Standard deviation | Not required by the basic estimator | Used within each group in original GRPO |
| Update | This note starts with one fresh on-policy gradient | Often uses a ratio and PPO-style clipping |
| Token aggregation | Sum sequence log-probabilities, then average samples | Inspect the implementation's length normalization |

Without standard-deviation normalization, the advantages have the fixed relationship derived above. Dividing by each group's standard deviation instead changes groups' relative weights. That standard deviation also depends on the current samples, so the leave-one-out unbiasedness proof does not establish unbiasedness here. [DeepSeekMath §4.1](https://arxiv.org/html/2402.03300v3#S4.SS1) defines the original GRPO objective; our [group-relative advantage section](after-ppo.en.md#grpos-group-relative-advantage) writes it out.

**RLOO does not mean that every implementation omits clipping.** The baseline and the update rule are separate choices. For example, [TRL v0.22.2's RLOO documentation](https://huggingface.co/docs/trl/v0.22.2/en/rloo_trainer#computing-the-loss) includes importance ratios and clipping when samples are reused. This pinned version illustrates one implementation, not a definition binding all frameworks.

## Adding KL requires another check {#kl}

One approach constructs a modified score at sampling time:

$$
\widetilde R_i=R_i-\beta\left[
\log\pi_{\mathrm{old}}(y_i\mid x)
-\log\pi_{\mathrm{ref}}(y_i\mid x)\right].
$$

Compute the leave-one-out advantages from $\widetilde R$. The old policy describes where the data came from; the reference policy supplies a regularization anchor. They have different jobs.

The bracketed quantity is a sampled log-ratio and may be negative. Its expectation under the appropriate sampling distribution yields the corresponding KL. Equal raw rewards need not remain equal after the KL adjustment.

The earlier proof assumed parameter-independent rewards. For the exact on-policy objective with $R-\beta\log(\pi_\theta/\pi_{\mathrm{ref}})$, direct differentiation contributes an additional $-\beta\mathbb E[g]$, which is zero. Holding that score fixed in the score-function estimator therefore still gives the objective's gradient. Old samples, repeated updates, clipping, or a different KL estimator require a new check rather than an automatic extension of this result. See [KL estimators](rlhf/kl-estimators.en.md).

## When is it worth trying? {#checks}

RLOO is a manageable starting point when same-prompt rewards vary, terminal rewards are useful, and Critic training is expensive. Sampling still costs time. Doubling $K$ may be less useful than covering more prompts.

| Observation | First check |
| --- | --- |
| All rewards in a group are 0, or all are 1 | Is feedback sparse, or the task too easy or hard? Without extra terms, equal rewards give zero leave-one-out advantages. |
| Less memory, but more wall-clock time | Measure generation, scoring, and synchronization for the extra responses. |
| Reordering a batch changes advantages | Check prompt membership; a reshape does not establish grouping. |
| One update works, repeated updates diverge | Check fixed old log-probabilities, sample age, and sampler/learner probability agreement. |
| Reward improves but answers get worse | Audit the reward and independent evaluation before replacing the algorithm. |

The [batch-check program](../practice/post-training/code/rl_batch_checks.py) verifies the arithmetic and a finite-sample example of the unbiasedness conditions. Run `python3 practice/post-training/code/rl_batch_checks.py` from the repository root; no model or GPU is needed. These checks validate mathematics and alignment, not a successful RL training run.
