# MoE: load balancing

[中文](load-balancing.md) · **English**

> Reading time: ~7 min · Level: advanced · Last reviewed: 2026-10-09

## Why a few experts can keep getting busier {#left-alone-it-collapses}

An expert receiving more tokens early gets more training opportunities and may become even more likely to be selected. Less-used experts can fall further behind. This is a possible feedback loop, not inevitable collapse. An unselected expert's parameters usually receive no task gradient from that token; router gradients also depend on gate normalization and auxiliary objectives. Those are different claims.

Below is a toy simulation; switch between the three approaches and watch the load:

<!-- widget:tx-moe-balance -->

## Capacity factor: how many tokens an expert may take

In Switch top-1 routing, capacity factor sets the token-slot cap per expert, helping plan compute and buffers. It does not guarantee equal runtime on every device:

$$\text{capacity} = \frac{\text{tokens in the batch}}{N} \times \text{capacity factor}$$

Switch skips overflowing expert branches while tokens continue through the residual path. A small CF saves slots but may drop more branches; a large CF reduces overflow but can leave more empty slots. Choose it with routing and throughput measurements. Dropless implementations keep expert assignments but still need load and buffer management. The top-1 formula is not a universal capacity definition for top-k implementations.

## Auxiliary loss: put the imbalance into the loss

Switch Transformer's auxiliary loss:

$$\mathcal L_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i P_i$$

- $f_i$: the fraction of tokens in the batch whose argmax is expert $i$, a count with no gradient;
- $P_i$: the mean router probability the batch gives to expert $i$, which does have a gradient;
- $\alpha = 10^{-2}$.

Multiplied together, the gradient reaches the router through $P_i$, and how hard it pushes is set by the real load $f_i$. At perfect balance $f_i = P_i = 1/N$ and the loss equals $\alpha$; this is a uniform reference value, not a strict lower bound over all routing distributions. [GShard](https://arxiv.org/abs/2006.16668) similarly multiplies assignment fractions by mean gates: one factor of the otherwise nondifferentiable squared fraction is replaced with a differentiable approximation. It does not differentiate the discrete count. The first sparsely-gated MoE used two coefficient-of-variation losses, one on importance and one on load.

## Router z-loss: for numerics, not balance

ST-MoE adds router z-loss to pull each token's log-sum-exp toward zero:

$$L_z = \frac{1}{B}\sum_{i=1}^{B}\Big(\log \sum_{j=1}^{N} e^{x^{(i)}_j}\Big)^2$$

The paper uses coefficient $c_z=0.001$. Here B counts participating tokens, not sequences. The penalty constrains the log-normalizer, not every logit's absolute value: logits equal to the logarithms of `[0.9, 0.1]` have zero z-loss, yet routing can remain skewed. It therefore cannot replace load balancing. See [ST-MoE](https://arxiv.org/abs/2202.08906).

## No auxiliary loss: adjust a bias instead

The auxiliary loss's gradient acts on the same router scores the language-model loss is learning, and the two pull against each other. DeepSeek-V3 does it differently:

1. each expert has a bias $b_i$ that is added to its score **only when choosing the top k**;
2. gate weights use the original affinity scores after selection; the bias neither enters that formula directly nor updates by backpropagation, but changing the selected set can change outputs and task gradients;
3. after every step an overloaded expert's $b_i$ goes down by $\gamma$ and an idle one's goes up by $\gamma$, with $\gamma = 0.001$ for the first 14.3T tokens and 0 for the last 500B.

It is not entirely free of balance losses: DeepSeek-V3 keeps a very small sequence-wise balance loss ($\alpha = 0.0001$) to prevent extreme imbalance inside a single sequence.

Another route is Qwen3's global-batch balance loss: balance is computed over the global batch rather than each micro-batch, which lets experts specialise more locally.

## Why can a selection-only bias change the output?

Consider a toy top-2 router, omitting grouped routing and additional scaling:

| Expert | Original affinity | Balance bias | Selection score |
| --- | --- | --- | --- |
| 0 | 0.8 | -0.2 | 0.6 |
| 1 | 0.7 | 0 | 0.7 |
| 2 | 0.2 | 0.5 | 0.7 |

Without bias, experts 0 and 1 run; with bias, experts 1 and 2 run. Their weights still come from 0.7 and 0.2: approximately 0.778 and 0.222, not 0.5 each from the equal selection scores. Changing the experts can change the output. This is why [DeepSeek-V3](https://arxiv.org/abs/2412.19437) separates selection scores from gate weights.

```python
def route_with_bias(affinities, biases, top_k):
    if len(affinities) != len(biases) or not 1 <= top_k <= len(affinities):
        raise ValueError("Invalid routing shape or top_k")
    if any(score <= 0 for score in affinities):
        raise ValueError("This example expects positive affinities")
    chosen = sorted(range(len(affinities)),
                    key=lambda expert: (-(affinities[expert] + biases[expert]), expert))[:top_k]
    denominator = sum(affinities[expert] for expert in chosen)
    return chosen, [affinities[expert] / denominator for expert in chosen]

chosen, weights = route_with_bias([0.8, 0.7, 0.2], [-0.2, 0.0, 0.5], 2)
assert chosen == [1, 2]
assert abs(weights[0] - 7 / 9) < 1e-12
```

## Expert balance and device balance are different

Put four experts on two devices: experts 0–1 on A and 2–3 on B. Count assignments, temporarily ignoring differences in execution cost.

| Expert assignment counts | Device A / B | Observation |
| --- | --- | --- |
| 4, 0, 4, 0 | 4 / 4 | Devices balance, but half the experts get no examples |
| 4, 4, 0, 0 | 8 / 0 | Both expert and device load are skewed |
| 2, 2, 2, 2 | 4 / 4 | Both balance for this batch, not necessarily every step |

For capacity, eight tokens with top-2 and four experts create 16 assignments, averaging four per expert. If capacity is defined over assignments with factor 1.25, the cap is five. **Dropping one assignment does not delete a training example**: another selected expert and the residual path may remain. Overflow and rerouting policies depend on the implementation.

Monitor expert histograms, per-device compute and communication, overflow, and task loss together. A flat histogram does not establish balanced runtime or better learning.
