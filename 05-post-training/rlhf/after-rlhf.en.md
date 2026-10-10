# GRPO, DPO, RLVR: which part does each change?

[中文](after-rlhf.md) · **English**

> Reading time: ~12 min · Level: core · Last reviewed: 2026-10

Suppose your model can already answer questions and you want to improve its answers. You might have a dataset of responses with labels saying which ones are better. Or you might generate new responses during training and score them as you go.

Standard offline DPO uses the first setup. PPO and GRPO are commonly used in the second, but turn scores into learning signals differently. RLVR answers a separate question: **can a program check the result and provide a reward?** It can be combined with PPO or GRPO; it is not the next algorithm in that sequence.

Start with the diagram to see which problem each method addresses. Then follow a group of answers from rewards to a training signal, and calculate DPO loss for a preference pair. Derivations and code are there to expand when you want them.

## Separate three decisions first

| Decision | Possible choices | Important distinction |
|---|---|---|
| Where feedback comes from | Human preferences, a Reward Model, program checks, environment outcomes | RLVR primarily addresses this axis |
| How to construct an update signal | Critic-based advantages, within-group rewards, preference pairs | GRPO and DPO construct different signals |
| How training obtains data | Fresh policy samples, reused rollouts, a fixed preference dataset | An algorithm name does not specify the entire data pipeline |

<section class="method-map" lang="en" aria-labelledby="method-map-title" id="method-comparison">
  <header class="method-map-header"><span>At a glance</span><strong id="method-map-title">The update rule and the reward source are separate choices</strong></header>
  <p class="method-map-label"><span>01</span> How does the model learn?</p>
  <div class="method-map-methods">
    <div class="method-map-card"><strong>PPO</strong><p>Generate → score → update</p><dl><dt>The comparison</dt><dd>Returns against value estimates.</dd><dt>Typical setup</dt><dd>A trained value model (Critic) helps estimate advantages.</dd></dl></div>
    <div class="method-map-card"><strong>GRPO</strong><p>Several answers to one prompt</p><dl><dt>The comparison</dt><dd>Each answer against the rest of its group.</dd><dt>What it removes</dt><dd>No separately trained Critic. Generation and scoring still cost compute.</dd></dl></div>
    <div class="method-map-card"><strong>Offline DPO</strong><p>Answer pairs with a preference</p><dl><dt>The comparison</dt><dd>Changes in the two answers’ probabilities relative to a reference model.</dd><dt>The data</dt><dd>A fixed preference dataset, not fresh rollouts at each update.</dd></dl></div>
  </div>
  <p class="method-map-label"><span>02</span> Who decides what is good?</p>
  <div class="method-map-feedback">
    <div><strong>PPO / GRPO</strong><p>A reward model can score answers, or a verifier can check results, such as passing tests. RLVR uses the latter; it does not require a particular update rule.</p></div>
    <div><strong>Offline DPO</strong><p>Preference labels say which answer is better. People or other judges can provide them; their quality still needs checking.</p></div>
  </div>
  <p class="method-map-note">Model counts alone do not determine memory use. Weight sharing, caching, and offloading matter too.</p>
</section>

For example, “GRPO with verifiable rewards” is a valid combination. RLVR can also use a Critic-based method; it does not inherently mean “only two models.” Weight sharing, caching, and offloading further change the actual memory footprint.

## GRPO: compare answers to the same question

Classic PPO typically trains a Critic to estimate expected return. GRPO instead samples several answers to the same prompt and constructs a relative signal from their rewards, without a separately trained Critic. The [DeepSeekMath paper](https://arxiv.org/abs/2402.03300) introduces this approach; it does not require rewards to come from a program.

### Four answers, one correct {#group-example}

For example, a model writes four implementations for the same coding problem, and only the second passes the current test suite. Assigning 1 for a pass and 0 otherwise gives `[0, 1, 0, 0]`. This is a teaching example, not an experimental result; passing those tests does not prove correctness on every input.

Using the population standard deviation within a group of size $G$, and temporarily ignoring the stabilizer $\epsilon$:

$$
\begin{aligned}
\bar r &= \frac{1}{G}\sum_i r_i,\\
s &= \sqrt{\frac{1}{G}\sum_i(r_i-\bar r)^2},\\
\hat A_i &= \frac{r_i-\bar r}{s+\epsilon}.
\end{aligned}
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

### Reward, advantage, and loss are different numbers {#reward-to-update}

The passing answer earned a reward of 1. Why did `1.732` then appear? The first number records a test result; the second compares that answer with its group. Updating the model also involves how its probability of producing the answer has changed.

<figure class="worked-update" lang="en" id="grpo-signal-path">
  <figcaption>The same four answers · Follow the one that passed the tests.</figcaption>
  <ol>
    <li><small>1 · Test result</small><strong>Reward = 1</strong><span>A pass earns one point. No comparison with other answers has happened yet.</span></li>
    <li><small>2 · Group comparison</small><strong>Advantage ≈ 1.732</strong><span>Above average in [0, 1, 0, 0]. A different group can give the same answer a different advantage.</span></li>
    <li><small>3 · Policy update</small><strong>Include the probability ratio</strong><span>Compare the current policy with the policy that sampled the answer, then apply clipping and the rest of the objective.</span></li>
  </ol>
</figure>

Take one token from that passing response. Given the same prompt and response prefix, suppose the old policy assigned probability `0.20` and the current policy assigns `0.26`. The ratio is `1.3`. Its denominator is neither the reward nor a probability from the Reference model.

With a clipping interval of `[0.8, 1.2]` and this positive advantage, the clipped objective takes the smaller of `1.3 × 1.732` and `1.2 × 1.732`, approximately **2.078**. That is a single-token policy objective to maximize; a loss to minimize uses its negative. It is not the whole batch loss and does not include KL or other terms. [DeepSeekMath §4.1.1–4.1.2](https://arxiv.org/html/2402.03300v3#S4.SS1)

<details markdown="1">
<summary>Check the calculation, then increase the probability ratio</summary>

```python
import math

def clipped_policy_term(ratio, advantage, epsilon=0.2):
    clipped_ratio = min(max(ratio, 1 - epsilon), 1 + epsilon)
    return min(ratio * advantage, clipped_ratio * advantage)

advantage = math.sqrt(3)
old_probability, current_probability = 0.20, 0.26
ratio = current_probability / old_probability
objective = clipped_policy_term(ratio, advantage)
loss_term = -objective
assert math.isclose(objective, 1.2 * math.sqrt(3))
assert math.isclose(clipped_policy_term(1.6, advantage), objective)
print(round(objective, 3), round(loss_term, 3))
```

Increasing the ratio from `1.3` to `1.6` does not increase this positive-advantage term. Clipping is not a hard limit on the whole model's change: other tokens, shared parameters, and other loss terms still affect the update. Negative advantages need their own sign analysis; see [PPO clipping](ppo-clipping.en.md).

</details>

Two roles are easy to confuse here. The **old policy** generated the batch, and its saved log-probabilities supply the ratio denominator. The **Reference** supplies a baseline for a KL constraint when one is enabled. Their weights may coincide at some point, but their jobs differ. With log-probabilities, compute the ratio as `exp(current_logp - old_logp)`, not by dividing them. If sampling settings change the generation distribution, check which distribution those probabilities describe.

Returning to that one-point reward: it records a pass on the current tests. The advantage gives a relative signal, and the probability ratio and clipping determine its contribution to the objective. None guarantees that the next answer will be correct. That still needs evaluation on new problems not used for training.

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

The Reference is a **frozen model, not a reference answer**. It assigns probabilities to both responses in the dataset, giving us a starting point for comparison. It does not need to generate another answer to imitate. The preference label—not the Reference—says which response is better.

For our overfitting example, score the same prompt and response with the current model and the Reference. This tells us whether that response has become more or less likely relative to the starting point. DPO compares that change for the preferred answer against the change for the rejected one. [Original paper §4](https://arxiv.org/html/2305.18290v3#S4)

<details markdown="1">
<summary>How does this comparison lead to the DPO loss?</summary>

For a KL-regularized reward objective with $\beta>0$, optimizing over all policy distributions satisfying the support conditions, with a finite partition function $Z(x)$, gives:

$$
\begin{gathered}
\pi^*(y\mid x)=\frac{\pi_{\mathrm{ref}}(y\mid x)}{Z(x)}\\
{}\times\exp\!\left(\frac{r(x,y)}{\beta}\right).
\end{gathered}
$$

Rearrange this as $r(x,y)=\beta\log[\pi^*(y\mid x)/\pi_{\mathrm{ref}}(y\mid x)]+\beta\log Z(x)$. Under a Bradley–Terry preference model, comparing two answers to the same prompt cancels $\log Z(x)$, leading to:

$$
\begin{aligned}
\delta_w&=\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)},\\
\delta_l&=\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}.
\end{aligned}
$$

The terms $\delta_w,\delta_l$ are the two responses' changes in log-probability relative to the Reference. Call the scaled gap $g$ and average over the preference dataset $\mathcal D$:

$$
\begin{aligned}
g&=\beta(\delta_w-\delta_l),\\
\mathcal L_{\mathrm{DPO}}&=-\mathbb E_{\mathcal D}\log\sigma(g).
\end{aligned}
$$

This derivation comes from the [DPO paper](https://arxiv.org/abs/2305.18290). It depends on the reward objective, preference model, and support assumptions. It does not establish that all human preferences follow this model, or guarantee that a finite neural network trained on finite data reaches the global optimum.

</details>

### Calculate the loss for one pair {#dpo-example}

Suppose the current model assigns sequence log-probabilities of $-2,-4$ to the preferred and rejected answers, while the Reference assigns $-3$ to both. These are illustrative numbers; a higher log-probability means the model assigns more probability to that response:

<figure class="worked-update worked-update--pairs" lang="en" id="dpo-score-changes">
  <figcaption>Each response scored by the Reference and the current model, in log-probability.</figcaption>
  <ol>
    <li><small>Preferred response</small><strong>−3 → −2</strong><span>Reference → current model<br>Change: +1</span></li>
    <li><small>Rejected response</small><strong>−3 → −4</strong><span>Reference → current model<br>Change: −1</span></li>
  </ol>
</figure>

Subtracting the two changes gives $1-(-1)=2$. With $\beta=0.2$, the sigmoid input is $0.4$ and loss is about $0.513$. If the current model equals the Reference, both changes are zero and loss is $\log 2\approx0.693$—even when the two responses originally had different probabilities.

<details markdown="1">
<summary>Check the calculation in Python</summary>

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

</details>

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
