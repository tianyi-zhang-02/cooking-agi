# MoE: load balancing

[中文](load-balancing.md) · **English**

> Reading time: ~3 min · Level: advanced · Last reviewed: 2026-09

## Left alone, it collapses

The router only gets gradient from the experts it chose. An expert that happens to get a few more tokens gets trained a bit better, scores higher next time, and gets more tokens; the rest are chosen less and less and barely learn. You end up paying for $N$ experts and using a handful.

Below is a toy simulation; switch between the three approaches and watch the load:

<!-- widget:tx-moe-balance -->

## Capacity factor: how many tokens an expert may take

To keep the compute on every device fixed during training, Switch Transformer capped each expert:

$$\text{capacity} = \frac{\text{tokens in the batch}}{N} \times \text{capacity factor}$$

Tokens over the cap are **not deleted**: they skip this layer's expert and pass to the next layer through the residual connection. Switch tried 1.0, 1.25, and 2.0 and found 1.0 to 1.25 worked better; ST-MoE trains at 1.25 and evaluates at 2.0. DeepSeek-V3 drops no tokens at all.

## Auxiliary loss: put the imbalance into the loss

Switch Transformer's auxiliary loss:

$$\mathcal L_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i P_i$$

- $f_i$: the fraction of tokens in the batch whose argmax is expert $i$, a count with no gradient;
- $P_i$: the mean router probability the batch gives to expert $i$, which does have a gradient;
- $\alpha = 10^{-2}$.

Multiplied together, the gradient reaches the router through $P_i$, and how hard it pushes is set by the real load $f_i$. At perfect balance $f_i = P_i = 1/N$ and the loss reaches its minimum, $\alpha$. GShard uses the same idea (the mean gate value stands in for the count), and the first sparsely-gated MoE used two coefficient-of-variation losses, one on importance and one on load.

## Router z-loss: for numerics, not balance

ST-MoE adds a router z-loss that penalises large router logits:

$$L_z = \frac{1}{B}\sum_{i=1}^{B}\Big(\log \sum_{j=1}^{N} e^{x^{(i)}_j}\Big)^2$$

with coefficient $c_z = 0.001$. Its purpose is stable training and fewer round-off errors at low precision, not load balancing.

## No auxiliary loss: adjust a bias instead

The auxiliary loss's gradient acts on the same router scores the language-model loss is learning, and the two pull against each other. DeepSeek-V3 does it differently:

1. each expert has a bias $b_i$ that is added to its score **only when choosing the top k**;
2. the gate weight is still computed from the original score, so the bias changes neither how outputs are weighted nor any gradient;
3. after every step an overloaded expert's $b_i$ goes down by $\gamma$ and an idle one's goes up by $\gamma$, with $\gamma = 0.001$ for the first 14.3T tokens and 0 for the last 500B.

It is not entirely free of balance losses: DeepSeek-V3 keeps a very small sequence-wise balance loss ($\alpha = 0.0001$) to prevent extreme imbalance inside a single sequence.

Another route is Qwen3's global-batch balance loss: balance is computed over the global batch rather than each micro-batch, which lets experts specialise more locally.
