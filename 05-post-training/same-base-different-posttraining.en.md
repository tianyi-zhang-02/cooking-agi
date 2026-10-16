# How far one base model can go

[中文](same-base-different-posttraining.md) · **English**

> Reading time: ~8 min · Type: case study · Last reviewed: 2026-10

## What to look for in this comparison {#what-this-comparison-can-actually-tell-us}

[The vendor describes GLM-5.3 and 5.2 as sharing a base](https://z.ai/blog/glm-5.3). If that holds, it removes one possible difference in starting conditions, but does not identify which post-training change contributed most.

Three questions help make sense of the results: **what changed, which budget produced the scores, and whether training and sampling agree on token probabilities**. Keeping these questions separate makes it easier to see which conclusions the evidence supports.

## Why this is a limited natural comparison {#why-this-is-a-natural-experiment}

Sharing a base does not mean that every post-training stage starts from the same checkpoint, or that training data, objectives, compute, sampling systems, and evaluation conditions are fixed. List what is reportedly unchanged, what changed, and what was not disclosed. Do not silently treat the last category as controlled.

For example, if both training tasks and rollout scheduling change, the final success rate cannot allocate credit between them. To estimate the scheduler's contribution, hold tasks and objectives fixed and compare under a specified training budget. If the factors interact, test their combinations.

⚠️ All numbers below are **vendor-reported and not reproduced here**. A same-base comparison narrows the attribution question; it does not automatically become a single-factor causal experiment.

## The numbers {#the-numbers}

Two public benchmarks and a separate budget comparison are retained from the [GLM-5.3 release](https://z.ai/blog/glm-5.3).

| Benchmark | 5.2 | 5.3 |
| --- | --- | --- |
| Terminal-Bench 3.0 | 4.6% | 28.3% |
| DeepSWE v1.1 | 46.2% | 66.9% |

Reported protocols: Terminal-Bench uses Claude Code 2.1.207 and avg@3; DeepSWE uses mini-swe-agent. **These rows do not share one scoring protocol.** For example, results `[0, 0, 1]` have mean $1/3$, whereas "at least one success" is $1$. Do not substitute the latter when reading avg@3. Reproduction also requires the release's time, context, sampling, and tool limits, plus confirmation that both versions were rerun under matching conditions.

The next table concerns **private Z.ai Code Bench**, not token usage on the two benchmarks above:

| Model | Effort | Reported score | Output tokens/task (approx.) |
| --- | --- | --- | --- |
| GLM-5.2 | Max | 23.4% | 96k |
| GLM-5.3 | Max | 34.5% | 75k |
| GLM-5.3 | High | 31.4% | 50k |

These percentages retain the release's wording rather than adding a more specific metric name: the extracted text does not identify their axis in the figure. The detailed scoring and aggregation rules still need to be checked against the figure and protocol.

- **Same effort label:** Max versus Max gives a token reduction of $(96-75)/96=21.875\%$. Sharing the name Max still does not imply equal inference compute.
- **Cross-effort:** Old Max versus new High gives $(96-50)/96\approx47.92\%$. This is a quality–output-length comparison across settings, not an experiment that changes only the model.
- **Do not overinterpret length:** Fewer output tokens do not imply proportional reductions in latency, price, or training cost. Average output length alone does not establish a better reasoning path; inspect per-task outcomes, failed trajectories, and actual resource use.

## But read that first row carefully {#but-read-that-first-row-carefully}

Separate relative increases from percentage-point changes, then examine difficulty and scoring. **A low baseline does not diagnose missing training coverage, and DeepSWE's mid-range 46.2% baseline is not, by itself, stronger evidence about which change caused the gain.** The [GLM-5.2 release](https://z.ai/blog/glm-5.2) already describes long-horizon coding training; a low score on a new benchmark cannot establish its absence.

A synthetic counterexample: suppose a task has 25 independent steps, each with the same success probability, and all must succeed. Raising per-step success from about 88.41% to 95.08% can raise overall success from 4.6% to 28.3%:

$$
0.046^{1/25}\approx0.884118,\qquad
0.283^{1/25}\approx0.950761.
$$

This is a teaching model with an independence assumption, **not a fit to GLM trajectories**. It shows that accumulated errors can produce a low total score without implying "essentially no training." Real diagnoses also need coverage checks, tool failures, budget truncation, and verifier rules.

## What they actually changed is the interesting part {#what-they-actually-changed-is-the-interesting-part}

The release distinguishes inherited slime / SAO with compaction from scaling environments, tasks, and training compute, alongside further algorithmic and system work. [Source](https://z.ai/blog/glm-5.3).

Analyze three questions rather than labeling every component "new in this version":

1. **Inherited mechanisms:** Did the earlier version already have them? Were their configurations unchanged?
2. **Increased investment:** Did task diversity, trajectory count, and compute change together? Is the comparison about efficiency at fixed compute or quality after greater investment?
3. **Added or modified mechanisms:** Is there an independent ablation? Does throughput improve quality at a fixed budget, or merely shorten the same training job?

Counting components does not allocate credit. Two system changes and one algorithm change cannot establish that systems caused most quality gains. Compaction can affect feasibility, available information, and learning trajectories together; it is not merely a switch that lets a task finish.

## Training-inference alignment: check ratios and error statistics {#training-inference-alignment-probably-the-important-one}

Let $q(a\mid s)$ be the actual rollout sampling probability and $p_\theta(a\mid s)$ the target probability evaluated on the training side. For actions with $q(a\mid s)>0$:

$$
r_\theta(a\mid s)
=\frac{p_\theta(a\mid s)}{q(a\mid s)}
=\exp\!\left(\log p_\theta(a\mid s)-\log q(a\mid s)\right).
$$

This illustrates conditional probabilities at one state, not a complete PPO implementation. With temperature, truncation, or masking, raw model logprobs need not represent the actual $q$. Importance weighting also needs the relevant support condition: samples cannot recover actions to which the target assigns positive probability but the sampler assigns none.

**A deliberately exaggerated example:** the weights have not changed, but two implementations produce:

$$
q(\cdot\mid s)=(0.5,0.5),\qquad
p_{\theta_0}(\cdot\mid s)=(0.6,0.4).
$$

The two token ratios are $(1.2,0.8)$, not one shared constant multiplier. In general, the discrepancy can depend on state and token. These numbers do not describe any real model's error; they show why identical weights need not imply identical probability distributions.

Such a ratio is not necessarily calculated incorrectly. If $q$ is genuinely the behavior distribution, it may express a real off-policy difference. Assessing the update also requires checking rollout weight versions, sampling probabilities, the objective, and clipping. A ratio different from 1 does not by itself prove a biased gradient.

[The release reports an **average logprob difference** around $10^{-7}$](https://z.ai/blog/glm-5.3), not a maximum-error guarantee over every state and token. Even for absolute error, one error of 1 among $10^7$ otherwise zero-error positions has mean $10^{-7}$. A signed mean can additionally hide cancellation.

First establish the mean's definition and measurement sample, then inspect quantiles, maxima, and slices by length, precision, and sampling configuration. A small mean is useful evidence, not proof that every trajectory is aligned or every update is correct.

## What this comparison can tell us {#what-this-experiment-shows}

**One: fixing the starting point narrows the question, not every variable.** It directs attention toward post-training and evaluation settings, but separating contributions still requires controlled comparisons.

**Two: eliciting existing capability is an explanation, not an established mechanism.** Improvement may involve new learning, more reliable existing behavior, or both. Shared initial weights neither distinguish these explanations nor measure an immutable capability ceiling.

**Three: discuss systems and algorithms under a common budget.** Report task quality, training resources, and inference resources together instead of declaring one category inherently more important.

## What it does not tell us {#what-it-does-not-show}

- **Transfer across tasks and bases.** One coding comparison does not replace evaluation of other capabilities or starting models.
- **Whether anything regressed.** Test unreported dimensions separately; silence is not evidence of no regressions.
- **Complete cost.** Environment count cannot be converted directly into training compute or a cost multiplier. Actual training time, rollout, verification, and serving resources are needed.
- **Which change contributed most.** Without controls and ablations, retain competing explanations rather than presenting engineering intuition as an experimental conclusion.

## Down to a checklist {#down-to-a-checklist}

1. Are task versions, aggregation, harnesses, retries, and budgets aligned?
2. Is this a comparison at the same effort label, at equal actual compute, or across different quality–cost settings?
3. Which components were inherited, which investments were scaled, and which changes received independent ablations?
4. Over which samples and statistic is logprob disagreement measured? Does the denominator describe actual sampling?
5. What else explains the low baseline? Were costs, failures, and unreported capabilities measured separately?

## Where to read next {#where-to-read-next}

- [PPO, GRPO, and DPO: turning feedback into an update](after-ppo.en.md): objectives and constraints
- [The three stages of RLHF](rlhf/three-stages.en.md): training roles and data sources
- [Dual-encoder retrieval](../04-search/dual-encoder.en.md): checking training and candidate-pool definitions

## Sources {#sources}

- [GLM-5.3 vendor release, 2026-08-14](https://z.ai/blog/glm-5.3): tables, budgets, and release scope.
- [GLM-5.2 vendor release, 2026-06-16](https://z.ai/blog/glm-5.2): earlier training and inherited mechanisms.

Checked 2026-10-10 using indexed text from official pages, not secondary reporting. No model, private evaluation, or training system was reproduced; these tables are not a complete evaluation protocol. The probability and arithmetic counterexamples are independently constructed, not vendor experiments.

## Quick learning: what can a same-base comparison prove? {#quick-learning-what-can-a-same-base-comparison-prove}

<details class="interview" markdown="1">
<summary>Natural experiments, confounders, and train–serve alignment</summary>

**Quick memory**: the same starting point is not the same process; the same effort name is not equal compute; mean error is not maximum error.

**Interview answer**

> Separate the vendor's observations from your explanation. Align tasks, scoring, and budgets before distinguishing inherited components, scale changes, and independent ablations. For probabilities, identify actual sampling and the training target; for errors, inspect means, tails, and measurement scope. Without a controlled experiment, do not allocate gains by the names of the changes.

<details markdown="1">
<summary><b>Deep dive</b>: how would you design a more informative comparison?</summary>

Choose a factor, such as a scheduling strategy. Fix initial weights, training data, objectives, and evaluation protocol. Specify whether the budget fixes training time or sample count, record changes in the other, repeat runs, and report uncertainty. If task distribution must also change, treat it as a second factor rather than hiding it under "everything else equal."

</details>
</details>
