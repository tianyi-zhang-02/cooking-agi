# On-policy and off-policy: can we train on old answers?

[中文](on-off-policy.md) · **English**

> Reading time: ~12 min · Level: core · Last reviewed: 2026-10-09

Generating long answers is expensive. Throwing them away immediately feels wasteful, but reusing them indefinitely creates another problem: **the model that generated them is no longer the model being updated.**

This is not simply about how many days have passed. Ten seconds can be enough if the parameters or sampling rules changed.

<span id="on-policy-off-policy-and-offline-rl"></span>

## Three questions that often get mixed together

| Question | What it asks | Example |
|---|---|---|
| On-policy / off-policy | Does the policy being learned or evaluated match the data-generating policy? | Updating a new policy from old answers requires accounting for distribution differences |
| Online / offline | Can learning obtain new interaction data? | Offline RL on fixed logs cannot collect missing observations on demand |
| On / off distribution | Do test inputs differ from training inputs? | Fresh rollouts still need not cover a new application domain |

These are not synonyms. SAC interacts with an environment while using a replay buffer: it is **online and off-policy**. The [Spinning Up SAC documentation](https://spinningup.openai.com/en/latest/algorithms/sac.html) shows the full loop. Standard offline DPO uses fixed preference pairs, but that alone does not make it equivalent to offline RL with environment transitions and Bellman backups.

## Why is PPO called on-policy if it uses multiple epochs?

Follow one batch through its lifecycle:

```text
Policy v7 generates answers → save answers, rewards, and v7 log-probabilities
                                         ↓
                                update to v8, then v9
                                         ↓
                             finish this batch; sample again
```

The batch initially comes from the rollout policy $\pi_{\mathrm{old}}$. Once updates begin, $\pi_\theta$ changes, but the denominator still uses the probabilities that generated the batch:

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)}.
$$

PPO permits multiple minibatch updates on this batch, using its surrogate objective, clipping, and sometimes implementation-level KL early stopping to moderate change. It is conventionally classified as on-policy, and informally described as near-on-policy: this describes **frequent fresh rollouts and limited reuse**, not a guarantee that distributions remain identical after every step. The [PPO paper](https://arxiv.org/abs/1707.06347) and [Spinning Up implementation notes](https://spinningup.openai.com/en/latest/algorithms/ppo.html) connect the method to a concrete training loop.

Do not replace the denominator with the newest model's log-probabilities after each step. That would no longer measure change relative to the policy that generated this batch. This old policy is also not the Reference used as a KL anchor.

## Why call GRPO on-policy if an older model generated the answers? {#grpo-data-lifecycle}

After the first parameter update, even a freshly sampled batch comes from an “older model.” Forbidding all reuse of old parameters would rule out the second update on that very batch.

What matters is **how far the generating policy has drifted, how long the batch is reused, and when fresh answers replace it**. Original GRPO samples a group of answers and performs a bounded set of updates; it does not require a new rollout after every gradient step. [Algorithm 1 in DeepSeekMath](https://arxiv.org/html/2402.03300v3) separates generation from the inner $\mu$ update iterations.

| Step | What happens | What must not silently change |
|---|---|---|
| 1. Generate | Save policy v7; generate $G$ answers per prompt | Record the generating version, sampling settings, and corresponding log-probabilities |
| 2. Score | Compute rewards, group means, and advantages within each prompt's group | Do not combine unrelated prompts into a group or assume the old mean remains zero after filtering |
| 3. Update | Use these answers to update the model to v8 and v9 | The ratio denominator still corresponds to v7; keep the batch's advantages and training masks well defined |
| 4. Refresh | Finish these updates and generate a new batch with the updated policy | Refreshing the old policy does not automatically refresh the Reference |

If v7 assigned a sampled token probability 0.4 and the updated model assigns 0.6, the ratio is 1.5. Overwriting the stored probability with the current 0.6 would reset the ratio to 1. The code would run, but it would no longer measure change from the generating policy.

The Reference has a separate schedule. The original algorithm sets it at an outer iteration, then performs multiple rounds of sampling and updates inside; it does not follow the actor after every optimizer step. That algorithm also mentions a replay buffer, but the buffer is used to **update the reward model**. It is not evidence that the actor normally samples arbitrary historical answers.

Implementations offer further choices. The [TRL GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer) exposes `num_iterations` for updates per generation alongside different loss and rollout settings, checked on 2026-10-09. A configuration named `GRPO` therefore does not tell you the complete data lifecycle.

### Synchronous updates, batch reuse, or asynchronous rollouts? {#stale-rollout-checks}

A synchronous loop is the simplest starting point to debug: generate a batch, update, synchronize parameters, and repeat. Generation and training may spend time waiting for each other. More updates per batch spread the generation cost; an asynchronous pipeline overlaps the work but can deliver answers from policies several updates behind.

| Choice | What it aims to save | What to watch |
|---|---|---|
| Few updates, frequent fresh batches | Limit distribution drift and simplify debugging | Generation cost and GPU idle time |
| Multiple updates on one batch | Reduce rollout cost per update | Ratio distribution, fraction on the actual flat branch, KL, and held-out quality |
| Asynchronous generation and training | Overlap generation with updates | Policy-version lag, queue age, stale-sample handling, and sampler/trainer probability agreement |

Updates per second is not enough to compare these options. Under the same compute or time budget, also measure final quality, valid training tokens, and answers discarded for staleness. There is no universally best reuse count.

Importance weighting can correct specified distribution differences under assumptions. It does not change where the data came from or automatically make arbitrary old data on-policy. Nor do token-level PPO/GRPO ratios and clipping guarantee an exact correction of the full trajectory distribution. A small calculation makes the limits easier to see.

## A two-action example: why importance weighting matters {#reweighting}

Forget language models for a moment. Make one decision: action A pays 1, B pays 3. The behavior policy $\mu$ chooses A/B with probabilities $0.9/0.1$; the target policy $\pi$ chooses each with probability $0.5$.

| Action | Behavior $\mu$ | Target $\pi$ | Reward | Weight $\pi/\mu$ |
|---|---|---|---|---|
| A | 0.9 | 0.5 | 1 | $5/9$ |
| B | 0.1 | 0.5 | 3 | 5 |

Expected reward is $1.2$ under the behavior policy and $2$ under the target. Averaging old logs does not automatically estimate the new policy. If every action the target may select also has nonzero behavior probability—the support condition—we can change measure:

$$
\begin{aligned}
\mathbb E_{a\sim\pi}[r(a)]
&=\sum_a\mu(a)\frac{\pi(a)}{\mu(a)}r(a)\\
&=\mathbb E_{a\sim\mu}\left[\frac{\pi(a)}{\mu(a)}r(a)\right].
\end{aligned}
$$

We also assume the reward mechanism conditional on the action has not changed. This is an exact expectation identity for a single stochastic decision; finite-sample estimates still have error.

The following synthetic log deliberately contains 45 A actions and 5 B actions, matching the expected proportions for easy arithmetic. Real samples need not have those counts.

```python
import math

behavior = {"A": 0.9, "B": 0.1}
target = {"A": 0.5, "B": 0.5}
rewards = {"A": 1.0, "B": 3.0}
actions = ["A"] * 45 + ["B"] * 5
weights = [target[action] / behavior[action] for action in actions]
logged_mean = sum(rewards[action] for action in actions) / len(actions)
estimate = sum(weight * rewards[action] for weight, action in zip(weights, actions)) / len(actions)
effective_size = sum(weights) ** 2 / sum(weight ** 2 for weight in weights)
assert math.isclose(logged_mean, 1.2)
assert math.isclose(estimate, 2.0)
assert math.isclose(effective_size, 18.0)
print(round(logged_mean, 2), round(estimate, 2), round(effective_size, 2))
```

There are 50 records, but the weight-based effective sample size (ESS) is only 18: the few B observations carry a large share of the weight. This is a **weight-concentration diagnostic**, not a guarantee of equivalence to 18 independent observations or a confidence interval.

## Three problems the ratio cannot fix

### 1. Division cannot recover an action never observed

If the behavior policy never chooses B, we only observe A's reward of 1. B could pay 0 or 3. Both worlds agree with the logs, but imply target-policy values of $0.5$ and $2$ respectively.

Adding a tiny denominator prevents a numerical error without creating missing observations. We need exploration, additional assumptions, or more conservative decisions. See [offline RL and OPE](../deep-rl/offline-and-ope.en.md) and [Levine et al.'s offline RL tutorial](https://arxiv.org/abs/2005.01643) for a deeper treatment.

### 2. Small token-level differences can multiply

A full-trajectory importance weight typically contains a product of stepwise ratios. Even a ratio of $1.1$ per step becomes $1.1^{20}\approx6.73$ after twenty steps. Real ratios need not all move in the same direction; the point is that long sequences can amplify estimator variance.

Adding log-probabilities helps numerics, not statistical variance. Nor should PPO's token-level surrogate be described as an exact correction of the entire old trajectory distribution.

### 3. Clipping weights changes the estimate

If we cap weights at 2 in the example, the five B observations contribute less. The estimate becomes $(45\times5/9+5\times2\times3)/50=1.1$, rather than 2. The bias–variance tradeoff is concrete; “clipping makes it stable” is not a complete explanation.

This is ordinary importance-weight clipping, **not the complete PPO clipped objective**. PPO also takes a minimum depending on the advantage sign; see [PPO clipping](ppo-clipping.en.md).

## Back to LLMs: what should the logs retain?

| Information | Why it matters | Common mistake |
|---|---|---|
| Rollout policy version | Identifies the generating parameters | Recomputing with the current model and calling it old log-probability |
| Actual sampling rules and probabilities | Temperature and top-p change the behavior distribution | Treating raw model probabilities as sampler probabilities |
| Tokens, masks, termination reason | Separates prompt, padding, EOS, and length truncation | Treating truncation as natural termination, or training on padding |
| Reward / verifier version | A scoring change may change the objective | Mixing scores from different versions without recording it |
| Trajectory age and queue delay | Asynchronous generation can lag many updates behind training | Tracking seconds but not policy-update distance |

Truncated sampling such as top-p can set some actual token probabilities to zero. A claim of exact importance correction therefore needs the actual behavior probabilities, target distribution, and support conditions—not an assumed untruncated softmax. Approximations can be practical, but identify and test them explicitly.

<span id="a-reward-model-is-not-a-critic"></span>

## What do the Reward Model, Critic, and logs each tell us?

| Object | Question answered | What it cannot replace |
|---|---|---|
| Reward Model / verifier | How does this outcome score under the chosen standard? | Not the probability that generated the sample |
| Critic | What return is expected from this state under the evaluated policy? | Not an external oracle of correctness |
| Behavior logs | What did a particular policy choose, and what followed? | Unobserved counterfactuals cannot be read straight from them |

Old answers are not unusable. We need to know where they came from, what they cover, and which bias we accept. Reuse can reduce generation cost, but losing track of the estimator's assumptions can make even “did training help?” hard to answer.

Next, [Reference and Critic](reference-and-critic.en.md) separates reference KL, the old-policy ratio, and the baseline.
