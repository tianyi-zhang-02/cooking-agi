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

## The contradiction nobody admits

Nearly every design doc of this kind says "unobserved candidates must not be treated as negatives." Then the training section says "we use in-batch negatives."

**Those two statements contradict each other.** In-batch negatives take other rows' candidates as negatives for the current row — and those are precisely randomly-drawn unobserved items.

The contradiction doesn't need to be avoided; the scale in-batch negatives provide is real and contrastive training needs it. But it needs an **explicit correction**, because the sampling distribution isn't uniform: an item appears in a batch with probability proportional to its frequency, so **popular items get sampled as negatives far more often than tail items**, and the model systematically suppresses them.

The standard fix is a logQ correction — subtract the log sampling probability from the score:

$$s'(u,c) = s(u,c) - \log Q(c)$$

with $Q(c)$ the estimated probability of $c$ appearing in a batch, which a streaming frequency estimate covers.

Skip it and you still get a popularity-debiased scorer — but with a direction and magnitude you never chose, set instead by the frequency distribution of your data. **You end up with a prior you didn't design.**

Exposed negatives need no such correction: they weren't sampled, they were genuinely shown. That is why separating the two negative types isn't merely "more careful" — they require mathematically different treatment.

## The hard-negative sampling trap

"Mine hard negatives from high-scoring but irrelevant items" sounds natural and is self-referential: **the current model picked them**, so you are teaching it to correct errors it already knows about, not the ones it doesn't.

Two mitigations:

- mine hard negatives with the **previous** checkpoint rather than the one being trained;
- mix in semantic nearest neighbours (pure embedding distance), which don't depend on the current model's judgment.

The hard-negative ratio is also a real hyperparameter, not a more-is-better knob. Too high and the model turns conservative — it learns "close but wrong" at the cost of suppressing genuinely relevant items that merely look unlike the query. That failure is invisible in aggregate recall; it shows up in tail coverage.

## The checklist

1. What fraction of my "negatives" are actually unobserved?
2. Do exposed negatives and unobserved candidates travel the **same code path**?
3. Is there a sampling correction on in-batch negatives? If not, what am I actually optimising?
4. After changing the positive threshold, did I recheck per activity bucket?
5. Were hard negatives mined by the current checkpoint or the previous one?

## Where to read next

- [Offline went up, online didn't](offline-online-skew.en.md) — the same problem at the evaluation layer
- [From noisy feedback to a servable retrieval system](noise-to-signal-retrieval.en.md) — the whole chain
- [Post-training](../05-post-training/README.en.md)
