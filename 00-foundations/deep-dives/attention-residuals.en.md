# Attention Residuals: choosing information across depth

[中文](attention-residuals.md) · **English**

Token attention asks which tokens a position should read. Attention Residuals asks a different question: **at this depth, how should the same token combine information from earlier layers?** Keeping sequence position and network depth separate avoids confusing it with sparse token attention.

## Start with an ordinary residual connection

A residual layer has $h_{l+1}=h_l+f_l(h_l)$. Expanding the recurrence gives an embedding plus accumulated layer updates. This provides a direct path through depth, but each update enters that sum with a fixed coefficient of one.

Attention Residuals weights earlier depth representations. Let $v_i$ be an embedding or earlier layer output, omitting batch and token dimensions:

$$
s_{l,i}=w_l^\top\operatorname{RMSNorm}(v_i),\qquad
\alpha_{l,i}=\frac{\exp(s_{l,i})}{\sum_j\exp(s_{l,j})},\qquad
h_l=\sum_i\alpha_{l,i}v_i.
$$

$w_l$ is a learned layer-specific pseudo-query, not a fresh $W_Qh$ projection of the current token. Weights are still content-dependent because keys come from earlier representations of this sample and token. Keys are RMS-normalized; values retain their original representations.

```text
Same token: embedding ─┐
            early output ├─→ softmax across depth ─→ weighted sum ─→ current sublayer
            middle output┤
            latest output┘
```

## Combine two sources

Take scalar values 2 and 10 with scores 0 and $\log 3$. Softmax gives weights 1/4 and 3/4, producing 8. Ordinary addition produces 12; a simple average produces 6.

```python
import math

def depth_mix(scores, values):
    if not scores or len(scores) != len(values):
        raise ValueError("Expected matching nonempty scores and values")
    if any(not math.isfinite(value) for value in scores + values):
        raise ValueError("Expected finite inputs")
    shift = max(scores)
    weights = [math.exp(score - shift) for score in scores]
    total = sum(weights)
    return sum(weight * value for weight, value in zip(weights, values)) / total

assert math.isclose(depth_mix([0, math.log(3)], [2, 10]), 8)
assert depth_mix([0, 0], [2, 10]) == 6
assert math.isclose(depth_mix([1000, 1000 + math.log(3)], [2, 10]), 8)
```

This function checks mixing only, not RMSNorm or query learning. It also exposes an initialization difference: a zero pseudo-query gives a uniform average, not the original residual sum. Changing residual structure requires training adaptation; it is not a lossless patch for arbitrary checkpoints.

## Why introduce blocks?

Retaining and reading every preceding sublayer output adds depth-wise memory traffic. Block AttnRes groups consecutive sublayers: updates accumulate within a block, while attention operates across blocks. Sources include the embedding, completed block representations, and the current block's partial accumulation. Current updates do not disappear until the block finishes.

A block here groups **depth**, not consecutive text positions. Attention and MLP count as separate sublayers in the paper; align this convention before comparing depth numbers.

| Simplified cost for $L$ sublayers and $N$ blocks | Full AttnRes | Block AttnRes |
| --- | --- | --- |
| Depth states retained per token | $O(Ld)$ | $O(Nd)$ |
| Total mixing computation across layers | $O(L^2d)$ | $O(LNd)$ |
| Selection granularity | Individual sublayer outputs | Block outputs and current partial sum |

These costs describe depth mixing, not whole-model training memory or FLOPs. They do not directly shrink sequence KV caches or make token attention linear in sequence length.

## Why not average locally normalized outputs?

Processing sources in batches can reduce intermediate storage. But each batch has its own softmax denominator, so normalized batch outputs cannot simply be averaged.

Suppose the first batch contains value 0 and the second contains three values of 4, all with zero scores. Local outputs are 0 and 4; their average is 2. The correct global output is 3 because the second batch carries three times the probability mass.

Merging requires each batch's log-sum-exp, or equivalent maximum and exponential sum. This is the same normalization issue as [FlashAttention's online softmax](attention-kernels.en.md), now across depth sources.

## What should an evaluation check?

Start with numerical agreement between full and batched mixing, prefix invariance under future-token changes, and no missing or double-counted partial sums at block boundaries. Disable randomness before comparing; dropout differences are not implementation errors.

Match training tokens, data, and total budget when comparing ordinary residuals, Full AttnRes, and block sizes. Measure training throughput, peak memory, decode latency, and long-input tasks alongside loss. Finer depth selection may not justify extra memory traffic, and a good grouping on one device may not transfer to another.

Checked 2026-10-08 against [Attention Residuals](https://arxiv.org/abs/2603.15031) and the [official implementation](https://github.com/MoonshotAI/Attention-Residuals). The examples explain normalization and cost; they do not reproduce training gains.
