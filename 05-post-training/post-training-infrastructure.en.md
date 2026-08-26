# Post-training infrastructure: sampling, numerics, context

[中文](post-training-infrastructure.md) · **English**

> Reading time: ~7 min · Type: chapter · Last reviewed: 2026-08

## In one sentence

Public discussion of post-training is almost entirely about algorithms — PPO or GRPO, Critic or no Critic. What actually stalls a post-training run is usually two other things: **how much experience you can collect per hour**, and **whether the training side and the inference side compute the same number**. Neither appears in any loss function.

## Why this layer gets ignored

It doesn't make papers. A new loss can be written as an equation, plotted as a curve, and given a name. Tripling rollout throughput cannot be written as an equation.

But on return per unit of effort the order is often reversed: **an algorithm generation buys tens of percent, while doubling sampling throughput means running the entire experiment twice in the same wall-clock time.**

## One: sampling throughput, RL's real cost center

Supervised learning's data already exists. **RL's data has to be generated on the spot**, by the current policy — that is what on-policy means.

So every training step is preceded by an inference run. The longer the task, the more lopsided this gets:

- one single-turn dialogue sample: a few hundred milliseconds
- one multi-step agent task: minutes — with tool calls, environment responses, and waiting in between

**In a synchronous implementation the training process idles for that entire window.** On long-horizon tasks, GPU utilization can get uncomfortable to look at.

Going asynchronous is the standard fix: a pool of actors continuously pushes experience into a queue and the trainer pulls from it. Throughput jumps.

**The cost isn't zero**, and we've met it before: **the experience in that queue came from an older policy.** You traded a bit of on-policy for throughput.

That's the real tradeoff:

| Asynchrony | Throughput | Freshness | Consequence |
| --- | --- | --- | --- |
| Fully synchronous | lowest | strictly on-policy | GPUs idle a lot |
| Lightly async | high | one or two steps behind | importance sampling can correct it |
| Heavily async | highest | many steps behind | ratio variance explodes, clipping fires constantly — you're discarding gradient |

**The test isn't "more async is better," it's "does the importance-ratio distribution still look sane."** Ratios piling up against the clip boundary mean the experience is too stale for the current policy — the throughput you gained didn't turn into learning.

## Two: numerical agreement, a silent crack

Rollout runs on an inference engine; training runs in a training framework. Kernels, precision, and fusion all differ, so **the same text through the same weights can yield different logprobs on each side**.

And the importance ratio takes one from each:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

The denominator is **reported by the inference engine**; the numerator is **computed by the training framework**. A systematic offset injects a factor unrelated to the policy changing.

The consequence: you believe you are on-policy while an unmodeled gap sits between the sampling policy and the optimized one. **Clipping still runs — it just clips a contaminated ratio.**

**And nothing errors.** Loss descends, metrics move, and every update carries a small systematic tilt.

The check is direct: take one batch of text, compute logprobs on both sides, look at the distribution of the difference.

```
diff = logp_train - logp_rollout
# what matters isn't the mean (it can be 0 with a very wide spread),
# it's the quantiles and the max. A long tail means some tokens'
# ratios are badly distorted.
```

One real system reported significant gains after driving this difference to the $10^{-7}$ range (see [how far one base model can go](same-base-different-posttraining.en.md)). The gain appears in no formula. It appears as **every previous update finally meaning what it claimed to mean.**

## Three: context management, so long tasks can finish

A multi-step agent task's context grows monotonically — every tool call and every environment response adds to it. Run long enough and it overflows.

The default behavior on overflow is truncation, and what gets truncated is usually **the earliest part** — the task goal and the initial constraints. The model then forgets what it was doing halfway through.

Compaction (summarizing history, folding redundant tool output) addresses **whether the task can finish at all**, not how well. That distinction matters: it isn't a quality trick, it's an admission requirement for long-horizon work.

An easily missed side effect: **compaction changes the state.** Same task, different compaction policy, different context — which makes it a different MDP. **The compaction policy has to be versioned alongside the model**, or you'll compare results from two different environments.

## What the three have in common

None of them changes the objective, and all of them change what you can do to it:

- sampling throughput decides **how many steps you can take in a given time**
- numerical agreement decides **whether each step points the right way**
- context management decides **whether the task can be walked to the end at all**

A very practical corollary: **when post-training isn't working, check these three before reaching for a different algorithm.** Inverting that order means using a more complex algorithm to fix a problem the algorithm didn't cause, probably failing, and then concluding — wrongly — that the algorithm doesn't work.

## Down to a checklist

1. What are my GPUs doing during sampling? Have I measured the train/sample time split?
2. What does the importance-ratio distribution look like? What fraction sits at the clip boundary?
3. How large is the logprob difference between training and inference? Did I check quantiles, or only the mean?
4. How is long-task context truncated or compacted? Is the task goal what's getting cut?
5. Is my compaction policy versioned — or did it quietly change while I compared two different environments?

## Where to read next

- [How far one base model can go](same-base-different-posttraining.en.md): these three in a real system
- [After PPO](after-ppo.en.md): the algorithm half
- [How big is your negative pool, really](../practice/recommender-systems/negative-pool-size.en.md): another "numerically right, semantically wrong" case
