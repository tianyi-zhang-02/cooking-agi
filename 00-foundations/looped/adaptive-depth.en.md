# Looped Transformers: how many loops per token

[中文](adaptive-depth.md) · **English**

> Reading time: ~5 min · Level: advanced · Last reviewed: 2026-10-09

## Which positions benefit from more computation? {#a-fixed-number-of-loops-wastes-most-of-them}

Some positions benefit from further computation; others may already be adequately represented. Adaptive depth tries to learn that distinction. It cannot be decided from word classes alone: an article is not guaranteed to need one loop, nor a pronoun several. Context matters.

<!-- widget:tx-loop-exit -->

## ACT and PonderNet

- [ACT / Universal Transformer](https://arxiv.org/abs/1807.03819) accumulates halting values and stops updating a position at a threshold or cap. Its output also uses weighted states and a final remainder, rather than simply selecting the last state.
- [PonderNet](https://arxiv.org/abs/2107.05407) (2021) makes halting a probability distribution. The probability of stopping at step $n$ is

$$p_n = \lambda_n \prod_{j<n} (1 - \lambda_j)$$

$\lambda_n$ is the conditional probability of stopping after reaching step n; $p_n$ is the probability of stopping there from the start. The objective is $\sum_n p_n \mathcal L_n + \beta\, \mathrm{KL}(p\|p_G)$, with a geometric prior $p_G$ using a compatible truncation convention. Training takes an expectation over halting steps; inference can sample a stop decision at each step. The paper's unbiased-gradient claim concerns this probabilistic objective, not arbitrary thresholding or truncation implementations.

For a toy three-step cap, let the first two conditional probabilities be 0.2 and 0.5, and force a stop at step three.

| Stop at step | Probability | Calculation |
| --- | --- | --- |
| 1 | 0.2 | Stop immediately |
| 2 | 0.4 | Continue, then stop: 0.8 × 0.5 |
| 3 | 0.4 | Absorb all remaining probability at the cap |

The probabilities sum to 1 and the expected depth is 2.2. The values 0.2 and 0.5 are not independent fractions of all exits. Discarding the remaining 0.4 after step two changes the objective. This code illustrates residual mass at the cap, not a full PonderNet trainer.

```python
def stopping_distribution(conditional):
    if not conditional or any(not 0 <= value <= 1 for value in conditional):
        raise ValueError("Expected conditional probabilities in [0, 1]")
    remaining = 1.0
    probabilities = []
    for position, probability in enumerate(conditional):
        mass = remaining if position == len(conditional) - 1 else remaining * probability
        probabilities.append(mass)
        remaining -= mass
    return probabilities

probabilities = stopping_distribution([0.2, 0.5, 1.0])
assert abs(sum(probabilities) - 1) < 1e-12
expected_steps = sum(step * mass for step, mass in enumerate(probabilities, 1))
assert abs(expected_steps - 2.2) < 1e-12
```

## Ouro's exit gate

[Ouro](https://arxiv.org/abs/2510.25741) attaches an exit gate after every loop: $\lambda_t = \sigma(\mathrm{Linear}(h_t))$. Training has two stages:

1. **Stage one**: the loss is $\sum_t p(t \mid x)\, \mathcal L_t - \beta\, H(p)$. At a fixed depth cap, $\mathrm{KL}(p\|U)=-H(p)+\log T$, so the entropy term is a weighted uniform-prior KL up to a constant. Calling it a standard negative ELBO additionally requires matching likelihood and weighting conventions; arbitrary $\beta$ does not establish that equivalence;
2. **Stage two**: the gate is trained on its own, with "how much would one more loop lower the loss" as the label.

In [Ouro-1.4B snapshot `7ea635b`](https://huggingface.co/ByteDance/Ouro-1.4B/blob/7ea635ba1575ae9ab4ae1d83d83e16a6e47fe696/modeling_ouro.py), all configured loops run before output states are selected. With the default four loops, selecting the second output does not mean computing only two. This describes that snapshot, not every Ouro engine: real savings require skipping work and handling KV dependencies.

## Mixture-of-Recursions: a router assigns depth

[Mixture-of-Recursions](https://arxiv.org/abs/2507.10524) (2025) routes tokens to recursion depths. Its recursion-wise cache stores participating tokens at each depth; recursive sharing is a separate policy. Cross-token top-k in expert-choice can leak future information, motivating auxiliary predictors for inference. Token-choice avoids that cross-position selection but faces load imbalance. The reported maximum 2.06× throughput gain uses a particular maximum-batch setting, not arbitrary serving conditions.

Record average depth, accuracy, per-token latency, and peak KV together. A lower selected depth is not a real compute saving if the kernels still execute every loop.
