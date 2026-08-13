# What counts as a positive, and as a negative

[中文](positive-negative-design.md) · **English**

> Reading time: ~5 min · Type: practice note · Last reviewed: 2026-08

## In one sentence

"Didn't click" is not a label — it is **at least four situations** collapsed together: never saw it, saw it and wasn't interested, was interested but had no time, saw it but the surface made clicking awkward. Encode them as the same 0 and no amount of training sophistication does anything but learn the wrong target more reliably.

## Four classes with genuinely different semantics

| Class | Definition | What it implies | Use in training |
| --- | --- | --- | --- |
| Reliable positive | strong behavioural evidence, not one click | relevant | positive |
| Exposed negative | **was actually shown**, with clear negative behaviour | not relevant | true negative, highest weight |
| Hard negative | semantically close or high-scoring, but wrong for this context | "close but wrong" | teaches the boundary |
| Unobserved | never shown | **nothing at all** | must not be a negative |

Rows two and four are the whole point. An exposed negative carries real information: the user had the chance and did not react positively. An unobserved candidate carries zero. Merging them encodes "dislikes it" and "never saw it" as the same number.

## The positive threshold matters more than it looks

A single click is weak evidence — clickbait earns clicks, so do mis-taps. The usual fix is to require several signals: dwell past a threshold, a follow-up action, no negative action within some window.

There is a hidden trade-off: **the stricter the threshold, the more reliable the positives, and the more they skew toward heavy users.**

Light users generate less behaviour to begin with; raising the bar filters them out of the training set entirely. The model then gets steadily better for heavy users and worse for cold-start ones, while the aggregate metric rises — because heavy users contributed most of the volume.

So a positive definition has to be **validated per activity bucket**, never on the aggregate. Change the definition, rerun the buckets; the blast radius is larger than it looks.

## Label semantics and optimisation roles are different

Nearly every design doc of this kind says "unobserved candidates must not be treated as negatives," then says "we use in-batch negatives." This looks contradictory only because it mixes two levels:

- **Label semantics:** unobserved does not mean disliked and must not be stored as explicit negative feedback.
- **Optimisation role:** sampled softmax may still place an unobserved candidate in the denominator as a distractor for the current training question.

The second use does not claim that the user dislikes the candidate. It uses a computable sampled objective to approximate a much larger comparison. The risks are proposal bias, false negatives, and objective mismatch—not the absence of a binary negative label.

When the target is full-catalog softmax, a common correction subtracts the log proposal probability:

$$s'(u,c) = s(u,c) - \log Q(c)$$

where $Q(c)$ is the probability or expected count of sampling $c$. Its definition must match replacement, deduplication, and batch construction.

Skipping it does not automatically make the model wrong. It means the model optimises a proposal-weighted objective rather than one automatically equivalent to full-catalog softmax. That distinction belongs in the experiment definition.

Exposed negatives come from a different data-generating process and should not inherit the same $Q(c)$ blindly. They may enter a separate pointwise or pairwise loss, or a denominator with separate weighting; the intended objective determines the treatment.

## The hard-negative sampling trap

"Mine hard negatives from high-scoring but irrelevant items" sounds natural and is self-referential: **the current model picked them**, so you are teaching it to correct errors it already knows about, not the ones it doesn't.

Two mitigations:

- mine hard negatives with the **previous** checkpoint rather than the one being trained;
- mix in semantic nearest neighbours (pure embedding distance), which don't depend on the current model's judgment.

The hard-negative ratio is also a real hyperparameter, not a more-is-better knob. Too high and the model turns conservative — it learns "close but wrong" at the cost of suppressing genuinely relevant items that merely look unlike the query. That failure is invisible in aggregate recall; it shows up in tail coverage.

## The checklist

1. What fraction of my "negatives" are actually unobserved?
2. Do exposed negatives and unobserved candidates travel the **same code path**?
3. What is the proposal for in-batch distractors, and does my intended objective require a correction?
4. After changing the positive threshold, did I recheck per activity bucket?
5. Were hard negatives mined by the current checkpoint or the previous one?

## Where to read next

- [Offline went up, online didn't](offline-online-skew.en.md) — the same problem at the evaluation layer
- [From noisy feedback to a servable retrieval system](noise-to-signal-retrieval.en.md) — the whole chain
- [Post-training](../05-post-training/README.en.md)
- [How big is your negative pool, really](negative-pool-size.en.md): local/global pools, differentiable gather, and the limits of logQ
