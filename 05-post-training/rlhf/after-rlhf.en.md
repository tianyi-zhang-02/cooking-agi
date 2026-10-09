# GRPO, DPO, RLVR: which part does each change?

[中文](after-rlhf.md) · **English**

> Reading time: ~10 min · Level: core · Last reviewed: 2026-10

These names often appear together, as if they formed an upgrade path: PPO is expensive, so move to GRPO, then DPO, then RLVR. They do not. **Some change the update rule; others change the source of feedback.**

Start with a batch of prompts and a model that can answer them. Who decides whether an answer is good? What is the comparison point? How does that comparison become a gradient?

## Separate three decisions first

| Decision | Possible choices | Important distinction |
|---|---|---|
| Where feedback comes from | Human preferences, a Reward Model, program checks, environment outcomes | RLVR primarily addresses this axis |
| How to construct an update signal | Critic-based advantages, within-group rewards, preference pairs | GRPO and DPO construct different signals |
| How training obtains data | Fresh policy samples, reused rollouts, a fixed preference dataset | An algorithm name does not specify the entire data pipeline |

![Update methods and feedback sources are separate decisions, not fixed model counts](../assets/rlhf-model-count.svg)

For example, “GRPO with verifiable rewards” is a valid combination. RLVR can also use a Critic-based method; it does not inherently mean “only two models.” Weight sharing, caching, and offloading further change the actual memory footprint.

## GRPO: compare answers to the same question

Classic PPO typically trains a Critic to estimate expected return. GRPO instead samples several answers to the same prompt and constructs a relative signal from their rewards, without a separately trained Critic. The [DeepSeekMath paper](https://arxiv.org/abs/2402.03300) introduces this approach; it does not require rewards to come from a program.

### Four answers, one correct {#group-example}

For example, a model writes four implementations for the same coding problem, and only the second passes the current test suite. Assigning 1 for a pass and 0 otherwise gives `[0, 1, 0, 0]`. This is a teaching example, not an experimental result; passing those tests does not prove correctness on every input.

Using the population standard deviation within a group of size $G$, and temporarily ignoring the stabilizer $\epsilon$:

$$
\bar r = \frac{1}{G}\sum_i r_i,\qquad
s = \sqrt{\frac{1}{G}\sum_i(r_i-\bar r)^2},\qquad
\hat A_i = \frac{r_i-\bar r}{s+\epsilon}.
$$

The mean is $0.25$ and the standard deviation is about $0.433$. The implementation that passes receives an advantage of about $1.732$; each other answer receives $-0.577$. This does not identify useful lines of code. It says **which result was better than the group average under the current scoring rule**.

```python
import math

rewards = [0.0, 1.0, 0.0, 0.0]
mean_reward = sum(rewards) / len(rewards)
variance = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
deviation = math.sqrt(variance)
advantages = [(reward - mean_reward) / deviation for reward in rewards]
assert math.isclose(advantages[1], math.sqrt(3))
assert math.isclose(sum(advantages), 0, abs_tol=1e-12)
print([round(advantage, 3) for advantage in advantages])
```

If all four rewards are zero, or all four are one, centering produces zeros. That group provides no distinguishing policy-gradient signal. The total gradient need not vanish if other terms, such as KL, remain active.

### Removing the Critic does not make training free

| Choice | Benefit | Cost or check |
|---|---|---|
| Multiple answers per prompt | Relative signals without training a Critic | Generation, scoring, and long answers still cost compute |
| Subtract the group mean | Compare within a question rather than across difficulty levels | All-correct and all-wrong groups may provide no signal; small groups are noisy |
| Divide by group standard deviation | Rescale signals across groups | Also changes relative prompt weighting; it is not just numerical housekeeping |
| Share an outcome advantage across answer tokens | Terminal feedback can enter a token-level loss | Does not identify which reasoning steps caused success |

That last distinction matters. A correct final answer does not imply that every intermediate step was correct. Applying a sequence reward to its tokens is a training estimator, not stepwise causal attribution.

The [TRL GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer) exposes choices for reward scaling, KL, and loss aggregation. Record those settings when reading code: implementations called GRPO may differ in group standard deviation, length normalization, and KL handling. Evaluate the useful group signal alongside its sampling cost.

## DPO: put preference comparisons directly into the policy loss

Suppose each prompt already has a preferred answer $y_w$ and a less preferred answer $y_l$. Standard offline DPO trains on these pairs without a fresh rollout, separate RM training, or a Critic at every training step. **That does not mean the dataset never required generation, or that preference labels were free.**

Ask, for example, “Explain overfitting in one sentence.” One answer says, “The model learns incidental details in the training examples and performs worse on new data”; another says, “The model has not learned the training data yet.” An annotator prefers the first. DPO learns directly from that **relative preference**, without first asking a reward model to score each sentence separately. Learning this pair does not guarantee correct choices on every new question.

### Why does the Reference appear in the loss?

For a KL-regularized reward objective with $\beta>0$, optimizing over all policy distributions satisfying the support conditions, with a finite partition function $Z(x)$, gives:

$$
\pi^*(y\mid x)=\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)
\exp\!\left(\frac{r(x,y)}{\beta}\right).
$$

Rearrange this as $r(x,y)=\beta\log[\pi^*(y\mid x)/\pi_{\mathrm{ref}}(y\mid x)]+\beta\log Z(x)$. Under a Bradley–Terry preference model, comparing two answers to the same prompt cancels $\log Z(x)$, leading to:

$$
\mathcal L_{\mathrm{DPO}}=
-\mathbb E_{(x,y_w,y_l)}\log\sigma\!\left(
\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right]\right).
$$

This derivation comes from the [DPO paper](https://arxiv.org/abs/2305.18290). It depends on the reward objective, preference model, and support assumptions. It does not establish that all human preferences follow this model, or guarantee that a finite neural network trained on finite data reaches the global optimum.

### Calculate the loss for one pair {#dpo-example}

Let the policy's chosen / rejected sequence log-probabilities be $-2,-4$, the Reference's both $-3$, and $\beta=0.2$. The relative gap is $2$, the sigmoid input is $0.4$, and the loss is about $0.513$. A zero relative gap gives $\log 2\approx0.693$.

```python
import math

chosen_logp, rejected_logp = -2.0, -4.0
reference_chosen, reference_rejected = -3.0, -3.0
beta = 0.2
relative_gap = (chosen_logp - reference_chosen) - (rejected_logp - reference_rejected)
preference_logit = beta * relative_gap
loss = math.log1p(math.exp(-preference_logit))
assert math.isclose(preference_logit, 0.4)
assert math.isclose(loss, 0.5130152523999526)
assert loss < math.log(2)
print(round(loss, 4))
```

These small values are convenient for hand calculations; production implementations should use stable `logsigmoid` / `softplus` operations. Standard sequence log-probability sums answer-token log-probabilities, with correct prompt, padding, and EOS handling. Replacing the sum with a token mean changes the objective. The [TRL DPO documentation](https://huggingface.co/docs/trl/dpo_trainer) helps connect the equations to data formats and loss implementations.

DPO can improve the relative gap without necessarily increasing the chosen answer's absolute probability: lowering the rejected answer's probability can also reduce this loss. A fixed dataset also cannot reveal failures that never appear in it.

## RLVR: programs give feedback, but programs have blind spots

RLVR uses verifiable outcomes as rewards, such as answer matching, compilation results, or test passes. It specifies **where feedback comes from**, not a unique policy-update algorithm. [DeepSeek-R1](https://arxiv.org/abs/2501.12948) is one public example using rule-based rewards.

“Deterministic” does not mean “complete.” Suppose the task is sorting, but the model always returns `[1, 2, 3]`:

| Test input | Constant output | What the test establishes |
|---|---|---|
| `[3, 1, 2]` | `[1, 2, 3]` | This case passes; sorting ability is not established |
| `[8, 4]` | `[1, 2, 3]` | Exposes the hard-coded answer |
| `[2, 2, 1]` | `[1, 2, 3]` | Exposes incorrect handling of duplicates |

Other gaps can lie in answer extraction, empty test sets, exception handling, or timeouts. **An optimizer receives the score you implemented, not the requirements you forgot to implement.** A deterministic verifier can remove some subjective scoring errors without eliminating specification gaming. [Anthropic's controlled study](https://www.anthropic.com/research/reward-tampering) illustrates how gaps between reward and intent can be exploited; it does not establish that every training run will do so.

Separate training checks from independent evaluation: reserve unseen problems and boundary cases, test new input types, and add human audits where needed. Changing the question while retaining the same faulty checker is not enough independence.

<span id="what-happened-next"></span>

## Choose based on what you actually have

| Current situation | A reasonable starting point | What a comparison must include |
|---|---|---|
| Good static preference pairs | Standard DPO as a baseline | Label consistency, coverage, held-out preference accuracy, and task outcomes |
| Fresh generation and reliable scoring are available | PPO or GRPO | Equal generation budgets, reward quality, Critic cost or useful group signals |
| Reliable automatic task verification exists | Connect a verifier to an appropriate RL method | Coverage, shortcuts, independent tests, and inference cost |
| A multi-turn environment gives delayed feedback | Specify state, termination, and credit assignment before choosing updates | Do not treat a trajectory as unrelated responses |

Before asking which name is newer, sketch where data comes from, who scores it, how the loss is computed, and who independently checks the result. Those four questions tell you what an algorithm change actually changes.

Continue with [on-policy and off-policy learning](on-off-policy.en.md) to understand data reuse, then use the [evaluation checklist](evaluation-and-review.en.md) to test whether a higher score supports your claim.
