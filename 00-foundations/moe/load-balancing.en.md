# MoE: load balancing

[中文](load-balancing.md) · **English**

> Reading time: ~12 min · Level: advanced · Last reviewed: 2026-10-09

Top-k controls how many experts each token uses, not how many tokens each expert receives. Eight tokens can all choose the same two experts and leave the rest idle. Load balancing addresses that second problem.

Separate three questions: how much capacity to allocate, how training encourages balance, and where to aggregate the statistics. A histogram over the whole batch can hide a very different pattern within individual sequences.

## Why a few experts can keep getting busier {#left-alone-it-collapses}

An expert receiving more tokens early gets more training opportunities and may become even more likely to be selected. Less-used experts can fall further behind. This is a possible feedback loop, not inevitable collapse. An unselected expert's parameters usually receive no task gradient from that token; router gradients also depend on gate normalization and auxiliary objectives. Those are different claims.

Below is a toy simulation; switch between the three approaches and watch the load:

<!-- widget:tx-moe-balance -->

## Capacity factor: how many tokens an expert may take

In Switch top-1 routing, capacity factor sets the token-slot cap per expert, helping plan compute and buffers. It does not guarantee equal runtime on every device:

$$C = \frac{T}{N}\cdot\mathrm{CF}$$

$T$ counts tokens participating in routing, $N$ is the expert count, CF is the capacity factor, and $C$ is the slot budget per expert. Allocating integer slots also requires an implementation-specific rounding rule.

Switch skips overflowing expert branches while tokens continue through the residual path. A small CF saves slots but may drop more branches; a large CF reduces overflow but can leave more empty slots. Choose it with routing and throughput measurements. Dropless implementations keep expert assignments but still need load and buffer management. The top-1 formula is not a universal capacity definition for top-k implementations.

## Auxiliary loss: put the imbalance into the loss

Switch Transformer's auxiliary loss:

$$\mathcal L_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i P_i$$

- $f_i$: the fraction of tokens in the batch whose argmax is expert $i$, a count with no gradient;
- $P_i$: the mean router probability the batch gives to expert $i$, which does have a gradient;
- $\alpha = 10^{-2}$.

Multiplied together, the gradient reaches the router through $P_i$, and how hard it pushes is set by the real load $f_i$. At perfect balance $f_i = P_i = 1/N$ and the loss equals $\alpha$; this is a uniform reference value, not a strict lower bound over all routing distributions. [GShard](https://arxiv.org/abs/2006.16668) similarly multiplies assignment fractions by mean gates: one factor of the otherwise nondifferentiable squared fraction is replaced with a differentiable approximation. It does not differentiate the discrete count. The first sparsely-gated MoE used two coefficient-of-variation losses, one on importance and one on load.

## A balanced batch can contain unbalanced sequences {#sequence-balance}

Take two sequences of four valid tokens, two experts, and top-1 routing. Set the auxiliary coefficient $\alpha$ to 1 just to keep the arithmetic visible:

| Scope | Expert probabilities per token | Assignment counts | Mean probabilities |
| --- | --- | --- | --- |
| Sequence A | `[0.9, 0.1]` | `[4, 0]` | `[0.9, 0.1]` |
| Sequence B | `[0.1, 0.9]` | `[0, 4]` | `[0.1, 0.9]` |
| Combined batch | Four of each kind | `[4, 4]` | `[0.5, 0.5]` |

The batch is balanced, yet A only uses expert 0 and B only uses expert 1. The auxiliary loss above is `1.0` over the combined batch. Computing it separately for each sequence and then averaging gives `1.8`. These are different constraints, not two ways of calculating the same number.

Local skew is not automatically a defect. If the sequences come from different domains, it may reflect useful specialization. Forcing every short sequence toward uniform routing restricts that freedom; looking only at global statistics can miss extreme concentration within one sequence. Choose the scope against model quality and system load, rather than increasing the coefficient until every histogram looks flat.

### Why divide top-k counts by $KT$?

A sequence with $T$ valid tokens and $K$ distinct experts per token produces $KT$ assignments. Let $c_i$ count assignments to expert $i$. We use $q_i$ for its fraction, since papers use different scaling conventions for $f_i$:

$$\begin{aligned}
c_i &= \sum_{t=1}^{T}\mathbf{1}[i\in S_t],\\
q_i &= \frac{c_i}{KT},\qquad \sum_i q_i=1,\\
P_i &= \frac{1}{T}\sum_{t=1}^{T}p_{t,i},\\
L_{\mathrm{seq}} &= \alpha N\sum_{i=1}^{N}q_iP_i.
\end{aligned}$$

$S_t$ is the selected set. Here $p_{t,i}$ is normalized over **all $N$ routed experts**, not just the selected experts used to combine outputs. At the uniform reference $q_i=P_i=1/N$, the loss equals $\alpha$, not zero. This reference is not a universal strict minimum.

For four experts, four tokens, and top-2, suppose the selected sets are `{0,1}`, `{0,1}`, `{0,2}`, and `{1,3}`. Counts are `[3,3,1,1]`; dividing by eight gives `[0.375,0.375,0.125,0.125]`. Dividing by four instead makes the fractions sum to two and adds a factor of $K$ if the rest of the formula is unchanged.

[DeepSeek-V3 §2.1.2, equations 17–20](https://arxiv.org/html/2412.19437v1#S2.SS1.SSS2) absorbs $N$ into $f_i=Nq_i$ and forms $P_i$ from sigmoid affinities normalized over all routed experts. Its auxiliary statistic is written using top-k of the original affinities; bias-adjusted dispatch has a separate formula. Check which selection each implementation uses rather than assuming every `topk` is identical. Our teaching code has neither bias nor device-group restrictions.

### Why not minimize squared count errors directly? {#balance-gradient}

Squared deviations can describe how much more skewed `[4,0]` is than `[2,2]`. But ordinary hard top-k does not differentiate its indices or counts with respect to logits. Adding $\sum_i(c_i-KT/N)^2$ to the loss **does not create a router training gradient**. It can be a monitoring statistic; optimizing through it requires a differentiable approximation or another gradient estimator.

The product above instead treats the observed assignment fractions $q_i$ as fixed coefficients and differentiates $P_i$. With softmax probabilities and a fixed selected set, the single-sequence derivative is:

$$\begin{aligned}
\frac{\partial L_{\mathrm{seq}}}{\partial z_{t,j}}
&=\frac{\alpha N}{T}p_{t,j}\\
&\quad\cdot\left(q_j-\sum_i q_i p_{t,i}\right).
\end{aligned}$$

For sequence A, $q=[1,0]$, $p=[0.9,0.1]$, and $T=4$. One token's logit gradients are `[0.045,-0.045]`: gradient descent slightly lowers expert 0's score and raises expert 1's. Averaging over two sequence losses adds another factor of one half.

That does not guarantee balanced assignments on the next step. Top-k choices stay unchanged until a boundary is crossed, and the language-model loss is updating parameters too. Inference usually omits this training auxiliary term, but it uses parameters shaped by that term. Omitting the calculation at inference does not mean it had no effect on generation quality.

<details markdown="1">
<summary>Padding, variable lengths, and normalization in code</summary>

The [complete CPU script](../code/moe_routing.py) provides `balance_statistics` and `balance_loss`, taking `[batch, time, experts]` logits and a boolean `[batch, time]` mask named `valid`:

```python
logits = torch.tensor([[[0.9, 0.1]] * 4, [[0.1, 0.9]] * 4]).double().log()
valid = torch.ones(2, 4, dtype=torch.bool)
sequence_loss = balance_loss(logits, valid, top_k=1, scope="sequence")
batch_loss = balance_loss(logits, valid, top_k=1, scope="batch")
assert abs(sequence_loss.item() - 1.8) < 1e-7
assert abs(batch_loss.item() - 1.0) < 1e-7
```

Run this after the function definitions in the script. The default `alpha=1` is for checking arithmetic, **not a recommended training weight**. With `scoring="sigmoid"`, `softmax(logsigmoid(logits))` stably normalizes sigmoid affinities across experts; this is not ordinary `softmax(logits)`.

- **Exclude padding from counts, probability means, and denominators.** The code replaces invalid logits before normalization so NaNs in padding cannot contaminate the result; those positions get zero gradient. It rejects sequences with no valid tokens.
- **Averaging sequence losses differs from pooling tokens.** `sequence` weights each sequence equally in the final mean. `batch` pools valid tokens before forming the two statistics and their product. Unequal lengths change weighting; even with equal lengths, averaging products is not the product of averages.
- **Packing does not erase document boundaries.** Two documents sharing a tensor row need document IDs if the intended balancing unit is a document rather than the packed row.
- **Post-overflow counts do not measure original routing demand.** Counting only assignments that survive a capacity cap can hide overloaded experts. Track attempted assignments and executed assignments separately.
- **Our batch is an input tensor, not an automatically distributed global batch.** Multi-device code must define the reduction scope, total valid-token count, and loss scaling. Averaging per-device means unconditionally can change the objective.

</details>

## Router z-loss: for numerics, not balance

ST-MoE adds router z-loss to pull each token's log-sum-exp toward zero:

$$L_z = \frac{1}{B}\sum_{i=1}^{B}\Big(\log \sum_{j=1}^{N} e^{x^{(i)}_j}\Big)^2$$

The paper uses coefficient $c_z=0.001$. Here B counts participating tokens, not sequences. The penalty constrains the log-normalizer, not every logit's absolute value: logits equal to the logarithms of `[0.9, 0.1]` have zero z-loss, yet routing can remain skewed. It therefore cannot replace load balancing. See [ST-MoE](https://arxiv.org/abs/2202.08906).

## No auxiliary loss: adjust a bias instead

The auxiliary and language-model losses update the same router scores, but do not necessarily favor the same direction. DeepSeek-V3 takes another approach:

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

Kimi K3 estimates the next routing bias with quantiles instead of fixed increments. [The LatentMoE example](latent-moe.en.md#quantile-balancing) works through that threshold and explains why the next batch is not guaranteed to split evenly.
