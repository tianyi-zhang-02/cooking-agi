# Multi-token prediction: what do the extra targets teach?

[中文](multi-token-prediction.md) · **English**

To compare MTP with DFlash, establish target alignment here, then read [DFlash drafting, verification, and costs](dflash.en.md). That chapter distinguishes the original method, DFlash 2, and recent suffix-refinement work without conflating training objectives and inference algorithms.

> Reviewed: 2026-10 · Prerequisite: [language-model objectives](language-model-objective.en.md)

Ordinary language modeling predicts the next token at each position. Multi-token prediction adds further-ahead targets during training, encouraging representations useful beyond the immediate next step.

It does not eliminate autoregression or guarantee a twofold generation speedup by adding a head. Separate the training objective from the inference algorithm.

## How do targets shift on one sequence?

Let A, B, C, and D denote four tokens:

| Current representation | Next-token target | One further target |
| --- | --- | --- |
| $h_A$ | B | C |
| $h_B$ | C | D |
| $h_C$ | D | None |
| $h_D$ | None | None |

The second task has fewer valid positions. Do not invent targets past document boundaries, padding, or sequence ends. An extra loss introduces extra shifting, masking, and normalization contracts.

This constructs labels for **independent future-prediction heads**, with document IDs describing contiguous segments:

```python
def future_pairs(tokens, document_ids, horizon):
    if horizon < 1 or len(tokens) != len(document_ids):
        raise ValueError("Invalid horizon or document IDs")
    return [(tokens[position], tokens[position + horizon])
            for position in range(len(tokens) - horizon)
            if len(set(document_ids[position:position + horizon + 1])) == 1]

assert future_pairs(list("ABCD"), [0, 0, 0, 0], 1) == [
    ("A", "B"), ("B", "C"), ("C", "D")]
assert future_pairs(list("ABCD"), [0, 0, 0, 0], 2) == [
    ("A", "C"), ("B", "D")]
assert future_pairs(list("ABCD"), [0, 0, 1, 1], 2) == []
```

There is no model forward pass here. The purpose is to verify supervision relationships before they become a silent training bug.

## Independent heads and sequential modules differ

[Gloeckle et al.](https://arxiv.org/abs/2404.19737) study multiple prediction heads over shared representations. [DeepSeek-V3](https://arxiv.org/abs/2412.19437) uses sequential auxiliary modules preserving the causal conditioning chain.

At one position:

```text
Independent heads: h_A ─→ B
                      └→ C

Sequential sketch: h_A ─→ B
                   h_A + embedding(B) ─→ auxiliary module ─→ C
```

Sequential training uses the true B to predict the later C. That is not observing C, but deployment must use accepted or drafted tokens rather than assume ground-truth B is available. Test the resulting distribution difference.

The auxiliary module may observe B; the **base next-token head** must not see B while predicting it. Parallel tensor execution still has to respect this causal dependency.

## Combining the objectives

A useful teaching formulation is

$$
\mathcal L=\mathcal L_{\mathrm{NTP}}
+\frac{\lambda}{K}\sum_{k=1}^{K}\mathcal L_{\mathrm{aux},k}.
$$

Each auxiliary loss uses its valid positions; independent heads and sequential modules differ in conditioning inputs. Here each task uses a valid-token mean. A paper or implementation may choose different denominators, which must be made explicit.

For NTP loss 2.0, auxiliary losses 2.4 and 2.8, and $\lambda=0.1$, total loss is $2.0+0.1(2.4+2.8)/2=2.26$. Comparing that total directly against NTP-only loss 2.0 does not establish a regression.

The shared trunk receives gradients from several tasks. They can improve representations or interfere. Denser supervision is not independent information or a quality guarantee. Tune $\lambda$ and evaluate the base head and downstream tasks, not only auxiliary losses.

## Training with MTP does not require using it at inference

One deployment drops auxiliary modules and generates from the base model token by token. Training may improve quality, but there is no automatic multi-token generation speedup.

Another uses auxiliary modules for speculative decoding: propose tokens, then verify them with the target model. Frequent rejection can make drafting and verification more expensive than ordinary decoding.

Preserving the target sampling distribution requires correct acceptance, rejection, and resampling—not just retaining tokens both models like. [Speculative Decoding](https://arxiv.org/abs/2211.17192) develops such verification. Greedy and stochastic decoding also require distinct handling.

## What does acceptance mean? A two-token example

At one position under a fixed prefix, let $p$ be the target distribution and $q$ the draft distribution. Accept a draft token $x$ with probability $\min(1,p(x)/q(x))$; on rejection, sample from normalized $[p-q]_+$. This is the [original algorithm's](https://proceedings.mlr.press/v202/leviathan23a.html) one-step rule, not agreement between two argmaxes.

Suppose the next token is either “yes” or “no”:

| Token | Target $p$ | Draft $q$ | Acceptance probability | Directly accepted mass |
| --- | --- | --- | --- | --- |
| yes | 0.6 | 0.8 | 0.6 / 0.8 = 0.75 | 0.8 × 0.75 = 0.6 |
| no | 0.4 | 0.2 | 1 | 0.2 |

Acceptance totals 0.8, leaving rejection mass 0.2. “No” is short by exactly 0.2, so the residual distribution always selects “no.” Final probabilities are 0.6 and $0.2+0.2=0.4$, matching the target.

Keeping only accepted samples and renormalizing would give 0.75 / 0.25 instead. Rejection handling is essential.

```python
import math

def speculative_one_step_distribution(target, draft):
    if not target or len(target) != len(draft):
        raise ValueError("Expected matching distributions")
    for distribution in (target, draft):
        if any(not math.isfinite(probability) or probability < 0 for probability in distribution):
            raise ValueError("Invalid probability")
        if not math.isclose(sum(distribution), 1.0, abs_tol=1e-12):
            raise ValueError("Probabilities must sum to one")
    accepted = [min(target_prob, draft_prob)
                for target_prob, draft_prob in zip(target, draft)]
    residual = [max(target_prob - draft_prob, 0.0)
                for target_prob, draft_prob in zip(target, draft)]
    rejection = sum(residual)
    corrected = [accepted_prob + remainder
                 for accepted_prob, remainder in zip(accepted, residual)]
    return corrected, rejection

corrected, rejection = speculative_one_step_distribution([0.6, 0.4], [0.8, 0.2])
assert all(math.isclose(actual, expected) for actual, expected in zip(corrected, [0.6, 0.4]))
assert math.isclose(rejection, 0.2)
```

This computes probability mass rather than generating samples. Residual mass sums to rejection probability, so multiplying the normalized residual by rejection probability recovers the residual. Identical distributions reject nothing; no division by zero is needed.

For a draft sequence, a causal target forward evaluates the conditional distributions, then acceptance proceeds left to right. After the first rejection, later drafts cannot simply be retained: their prefix is no longer valid. Cache state must also return to the valid prefix. Sequential MTP modules have drafting dependencies; they do not automatically produce every future token in parallel.

## Count milliseconds, not just model calls

Use **hypothetical timings**, not benchmark results: ordinary decoding takes 10 ms per token; drafting plus verification takes 18 ms per round and emits three final tokens on average. That is 6 ms per token, about 1.67× faster. If it emits only one, it costs 18 ms per token and is slower.

Count accepted drafts plus any token emitted on rejection or full acceptance. Include sampling, cache handling, and scheduling in timing. A small module forward and a longer target verification do not have equal costs, so counting both as one step does not establish a $K/2$ speedup.

## Establish the benefit you actually want

| Question | Control | Measure |
| --- | --- | --- |
| Better representations? | Same data, comparable training compute | Base-head validation loss and downstream tasks |
| Worth the extra training? | Include auxiliary and output-head computation | Quality versus training cost |
| Faster inference? | Same target model, output conditions, hardware, lengths | Acceptance rate, output tokens/s, latency |
| Correct implementation? | Fixed tiny sequences and document boundaries | Shifts, masks, leakage, denominators |

The useful lesson is how additional objectives shape representations without confusing training quality, inference speed, and extra compute.
