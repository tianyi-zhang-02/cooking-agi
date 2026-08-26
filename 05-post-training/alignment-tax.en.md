# The alignment tax: what you lose by becoming agreeable

[中文](alignment-tax.md) · **English**

> Reading time: ~7 min · Type: chapter · Last reviewed: 2026-08

## In one sentence

Alignment isn't free. What you buy is a single output that better matches preferences; what you pay is **the width of the distribution** — how many different good answers the model can still give to the same question. And the bill is invisible on any average score, because **an average looks at one sample, and what you lost was the second sample being different.**

## What the tax looks like

**Output diversity drops.** Sample the same prompt repeatedly and an aligned model's responses resemble each other more. This is the most direct and most measurable one.

**Entropy collapse.** The policy distribution sharpens over training and low-probability tokens get harder to select. [The algorithms chapter](after-ppo.en.md) covered one mechanism: PPO's symmetric clipping permits only a minuscule absolute increase for a token whose current probability is tiny, so it has almost no path back. DAPO's Clip-Higher targets exactly this.

**Collapse to the safe answer.** If cautious responses are systematically preferred in the data, then the optimal policy is to **always give the cautious response**. The model learns a house style: tidy structure, full disclaimers, low information density.

**Over-refusal.** Harmlessness boundaries can't be drawn exactly, and refusing usually costs less than being wrong, so the policy shifts the boundary toward the safe side.

## Why this is close to inevitable

It isn't sloppy implementation. It falls out of the objective.

RL optimizes **expected return**. For a given prompt, the optimum is usually **a point, not a distribution** — if the reward model prefers style A over style B by even a little, then "always emit A" strictly beats "sometimes A, sometimes B."

$$\pi^* = \arg\max_\pi \mathbb{E}_{y \sim \pi}[r(x,y)]$$

Nothing in that expression rewards diversity. **Diversity wasn't sacrificed; it was never requested.**

SFT suffers less because cross-entropy fits a distribution — however many ways of writing appear in the data, the model learns that many. RL climbs a scalar.

This is also the KL penalty's second job: beyond confining the policy to [where the reward model is still valid](where-preferences-come-from.en.md), it slows the narrowing. **But it's a price dial, not a safety switch** — turn it up and you keep diversity while forgoing the alignment gain.

## But it's a trade, not a pure loss

The other half has to be said, or this reads as a case against alignment.

Research has found that compared to SFT, **RLHF generalizes better out of distribution**, especially when test and training distributions differ substantially. So the real shape of the deal is:

> **You trade width of the distribution for robustness outside it.**

Which explains why both camps are right: "RLHF made the model dumber" and "RLHF made it far more usable" can be simultaneously true — **they measure different things.** One measures width, the other measures single-shot hit rate.

The actual mistake isn't making the trade. It's **making it without ever pricing it.**

## How to measure it: no single score will show you

The key property: **diversity loss is only visible across multiple samples.**

**The pass@k versus pass@1 gap.** This is the hardest criterion available.

- pass@1 up and pass@k up → the model genuinely got stronger;
- **pass@1 up and pass@k down → you compressed the distribution into a point.**

In the second case the model merely concentrated correct answers that were previously spread across k samples into the first one. **Total capability did not increase; sampling efficiency did** — and in an evaluation that only reports pass@1, that looks like pure progress.

For anything that relies on multiple attempts — agent retries, best-of-n, test-time scaling — this is a real loss: **you cut the raw material best-of-n runs on.**

Other things worth measuring:

| What | How |
| --- | --- |
| Lexical diversity | distinct-n / n-gram entropy across samples of one prompt |
| Semantic diversity | pairwise embedding similarity among samples (the vector version of self-BLEU) |
| Distribution sharpness | mean per-token entropy at generation, logged per training step |
| Refusal rate | measured on a control set of clearly benign prompts, to catch over-refusal |
| Length drift | mean output length as a curve over training steps |

**Entropy and length must be plotted per step, not measured once at the end.** Collapse is gradual, and only the curve tells you when it started and whether it can still be stopped.

## One practical recommendation

Before training starts, measure all of the above on the base model and **save it as a baseline.**

Because the perverse thing about the alignment tax is that **you can only see what you paid by comparing against "before."** By the time outputs start feeling samey, that "before" is often gone — checkpoints cleaned up, or the eval recipe changed and comparison is no longer valid.

This is the same point as [freezing the eval recipe](../07-evaluation/): **you need the instrument before improvement means anything.**

## Down to a checklist

1. Am I reporting pass@1 or pass@k? Which way is the gap between them moving?
2. Do I have a diversity baseline from the base model, or only post-alignment numbers?
3. What does generation entropy look like per training step? Monotone decline to a plateau?
4. Is output length climbing monotonically through training? If so, more content or just denser hedging?
5. Do I have a clearly-benign control set specifically for over-refusal?
6. What set my KL coefficient — a measurement, or a number I copied?

## Where to read next

- [After PPO](after-ppo.en.md): the mechanism linking symmetric clipping to entropy collapse
- [Where preferences come from](where-preferences-come-from.en.md): which step the bias entered at
- [How far one base model can go](same-base-different-posttraining.en.md): a live case of "no regressions mentioned" not meaning "no regressions"
- [Evaluation](../07-evaluation/): why one aggregate score is never enough

## Starting papers

- [A General Language Assistant as a Laboratory for Alignment](https://arxiv.org/abs/2112.00861) — one origin of the term "alignment tax"
- [Understanding the Effects of RLHF on LLM Generalisation and Diversity](https://arxiv.org/abs/2310.06452) — the systematic generalization-up, diversity-down comparison
- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) — early evidence on KL penalties and over-optimization
