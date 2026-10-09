# Can a model track order without positional encoding?

[中文](nope-and-order.md) · **English**

“Pay, then ship” and “ship, then pay” use nearly the same words but describe different agreements. A model needs order information, but attaching a position vector to every token is not the only way to provide it. **An ordered computation can leave a record of order in its outputs.**

That is the starting point for NoPE (No Position Encoding). The first two examples give the basic distinction. Continue to the [two-layer attention example](#causal-order) for the role of a causal mask, or the [Kimi K3 section](#kimi-k3) for a specific architecture.

Prerequisites: [how attention works](../core/multi-head-attention.en.md) and [relative positions with RoPE](position-and-context.en.md). This article concerns order in text sequences; it does not automatically describe positional encoding in a vision encoder.

## Swap the inputs, change the result {#order-example}

Forget neural networks for a moment. Keep one number. For each new input, halve the old number and add the input:

$$
h_t=0.5h_{t-1}+x_t,\qquad h_0=0.
$$

| Input order | First step | Second step | Final state |
| --- | --- | --- | --- |
| 1, then 2 | $0.5\times0+1=1$ | $0.5\times1+2=2.5$ | 2.5 |
| 2, then 1 | $0.5\times0+2=2$ | $0.5\times2+1=2$ | 2 |

![Reversing the same two inputs changes the decayed running state from 2.5 to 2; an ordinary sum is 3 in both orders.](../assets/order-state.svg)

We never supplied an extra “position 1” input, yet the result changed: **the earlier value experienced one more decay step.** This scalar example illustrates order, not the KDA algorithm. One number cannot faithfully store a whole passage.

Replace 0.5 with a constant $\alpha$ and expand:

$$
h_T=\sum_{j=1}^{T}\alpha^{T-j}x_j.
$$

Older inputs pass through more factors of $\alpha$. There is a cost: for $0<\alpha<1$, distant information weakens. At $\alpha=1$, the final state is a sum and ignores permutations. At $\alpha=0$, it keeps only the last input. **Sensitivity to order, long retention, and exact recall of individual inputs are different properties.**

## Why consider positions in ordinary attention? {#unordered-attention}

Consider attention without positional encoding, a causal mask, or distance bias. Every position uses shared projections, and computation depends only on token content. Permuting the inputs generally permutes the outputs in the same way; it does not reveal which token originally came first. This is **permutation equivariance**, not a claim that all output positions are identical.

For example, globally averaging `[1, 2, 0]` gives 1 at every position. Swap the first two inputs and the output is still `[1, 1, 1]`. This is a minimal equal-score attention example. Real scores differ, but the simultaneous input/output permutation property still holds under the stated assumptions.

<details>
<summary>The derivation: why does the output follow the permutation?</summary>

Let $\Pi$ permute rows, with $Q=XW_Q$, $K=XW_K$, and $V=XW_V$. Without position-dependent operations, the permuted projections are $\Pi Q,\Pi K,\Pi V$. Row-wise softmax obeys

$$
\operatorname{softmax}(\Pi A\Pi^\top)=\Pi\operatorname{softmax}(A)\Pi^\top.
$$

Consequently,

$$
\operatorname{Attn}(\Pi X)
=\operatorname{softmax}\!\left(\Pi QK^\top\Pi^\top/\sqrt d\right)\Pi V
=\Pi\operatorname{Attn}(X).
$$

A final mean over positions removes even this permutation: the pooled result is **permutation invariant**. The argument assumes deterministic computation, shared projections, and no other position-dependent operation. It does not directly apply to a decoder with a causal mask.

</details>

## A causal mask changes the situation {#causal-order}

A lower-triangular mask lets the first position see only itself and the second see the first two. **Different positions receive different prefixes**, so they are no longer interchangeable in the same way.

Keep only an average over the visible prefix—no projections, residuals, or normalization. Apply this operation twice:

| Input | All first-layer outputs | Final position after the second layer |
| --- | --- | --- |
| `[1, 2, 0]` | $[1,1.5,1]$ | $(1+1.5+1)/3=7/6$ |
| `[2, 1, 0]` | $[2,1.5,1]$ | $(2+1.5+1)/3=1.5$ |

The final position is 1 in both first-layer outputs. At the second layer, however, it reads different earlier states. We deliberately keep the same final input, 0, to separate a change in order from a change in the last token's content.

This establishes only that **causal computation can distinguish order without added positional encoding**. It does not prove that every NoPE model will learn positions well or extrapolate to arbitrary lengths. Haviv et al. also observed learned positional information in causal LMs without explicit positional encoding; their evidence applies to the models and training conditions studied, not all possible architectures. [Paper](https://arxiv.org/abs/2203.16634v2)

## What problem was each paper addressing? {#paper-context}

In 2022, NoPos challenged the necessity of common explicit position schemes in causal LMs. It compared learned, sinusoidal, and ALiBi encodings at 125M–1.3B parameters—not today's very-long-context models. Read the setup, position probes, and limitations together. [Paper §3–6, §9](https://arxiv.org/html/2203.16634v2) · [Authors' code and models](https://github.com/adihaviv/NoPos)

In 2025, Kimi Linear also targeted long-sequence computation and KV storage without sacrificing expressivity. Its baselines included full MLA and a Gated DeltaNet hybrid. Separate the reason for mixing layers from the choice to omit RoPE. [Paper §4–5](https://arxiv.org/html/2510.26692v1#S4) · [Official implementation](https://github.com/MoonshotAI/Kimi-Linear)

These are different research questions, not a complete account of industry practice in either year. That distinction also helps when reading K3: “no positional encoding” is not a universal upgrade recipe.

## Kimi K3: order information comes from elsewhere {#kimi-k3}

Kimi K3 v1 interleaves KDA and Gated MLA at 3:1, ending with an extra MLA layer. KDA supplies ordered state updates; MLA omits explicit Q/K positional encoding. Global access remains causal. [Report §2.1](https://arxiv.org/html/2607.24653v1#S2.SS1)

This is not deleting the rotation function from an existing RoPE model. Inspect the complete token-mixing path. KDA's matrix state is richer than our scalar example: start with error-based writes in [Gated DeltaNet](gated-deltanet.en.md), then channel-wise forgetting in [Kimi Linear](https://arxiv.org/abs/2510.26692).

The report still uses long-context data and progressive context training. **No RoPE retuning does not mean no long-context training.** We have not reproduced K3 training or its long-context scores. [Report §3.4](https://arxiv.org/html/2607.24653v1#S3.SS4)

<details>
<summary>One level deeper: how does a matrix state retain order?</summary>

KDA extends scalar forgetting to channel-wise gates and uses short convolutions in its Q/K/V paths. Here $S\in\mathbb R^{d_k\times d_v}$ is transposed relative to our Gated DeltaNet article; do not mix conventions. [Kimi Linear](https://arxiv.org/abs/2510.26692)

$$
D_t=\operatorname{Diag}(\alpha_t),\qquad
\bar S_t=D_tS_{t-1},
$$
$$
S_t=\bar S_t+\beta_tk_t(v_t^\top-k_t^\top\bar S_t),\qquad o_t=S_t^\top q_t.
$$

$\alpha_t$ has $d_k$ components; $k_t,q_t$ have dimension $d_k$, $v_t$ has dimension $d_v$, and $\beta_t$ controls write strength. The update still means “decay, then correct the read error,” not simply add the new value.

Take $S=[2,10]^\top$, $\alpha=[0.5,0.9]^\top$, $k=[1,0]^\top$, $v=6$, and $\beta=0.25$:

| Operation | Result |
| --- | --- |
| Channel-wise decay | $\bar S=[1,9]^\top$ |
| Read along the key | $k^\top\bar S=1$ |
| Error | $6-1=5$ |
| Write | $S=[1,9]^\top+0.25[1,0]^\top\times5=[2.25,9]^\top$ |

The second entry retains 9, rather than 5 from halving both channels. Channel-wise gates offer different retention scales; training still determines which information deserves them.

Write the same update as $S_t=A_tS_{t-1}+B_t$, where

$$
A_t=(I-\beta_tk_tk_t^\top)D_t,\qquad B_t=\beta_tk_tv_t^\top.
$$

Starting from zero, three steps give

$$
S_3=A_3A_2B_1+A_3B_2+B_3.
$$

The first write passes through $A_2$, then $A_3$. Matrices generally do not commute; even when some do, the placement of the writes still matters. This expansion makes multiplication order clearer than an unspecified $\prod A_t$.

Both recurrence and RoPE can introduce distance through successive transformations, but they are not the same algorithm: rotary transformations preserve norms; these states can decay, correct, and overwrite. [Kimi Linear's §5.2 ablation](https://arxiv.org/html/2510.26692v1#S5.SS2) supports NoPE in its hybrid design. The authors' short-range-bias explanation is not a theorem that RoPE always harms long-context performance.

</details>

<details>
<summary>An implementation hazard: inverse cumulative decay can become huge</summary>

K3 lower-bounds per-step log-decay to control numerical range in chunkwise computation. [Report §2.1.1](https://arxiv.org/html/2607.24653v1#S2.SS1.SSS1)

For example, with $g_r\in(-5,0)$ and $\alpha_r=\exp(g_r)$, a 16-step product is $\exp(\sum_r g_r)$. Its inverse is below $e^{80}\approx5.54\times10^{34}$. Without that bound, a cumulative log-decay of $-100$ gives an inverse near $2.69\times10^{43}$, outside BF16's finite range.

This explains a bound on one intermediate, not whole-kernel stability or measured throughput. Restricting decay also constrains available forgetting rates; hardware convenience and learning behavior need joint evaluation.

</details>

## Why not disable RoPE in an existing checkpoint? {#checkpoint-change}

Because that changes similarities the model learned to use. Take a two-dimensional query and key, both $(1,0)$. Their unrotated dot product is 1; with relative rotation $\pi/2$, it becomes 0. Suppose two candidate keys produce logits `[1, 0]` with positional rotation and `[1, 1]` without it. The first softmax probability changes from approximately 0.731 to 0.5.

This changes how each layer reads context, not just the cost of a rotation. Adaptation needs training and evaluation; one successful hybrid design does not validate a configuration-only change to another checkpoint. See [RoPE](position-and-context.en.md) for rotation formulas and pairing conventions.

## Trade-offs: where does order come from, and where is history stored? {#tradeoffs}

| Mechanism | What it supplies | Remaining cost or limitation |
| --- | --- | --- |
| Explicit positional encoding | Position or relative distance in the reading relationship | Checkpoint and context-extension conventions must match |
| Causal mask | Different visible prefixes at different positions | Does not guarantee learned distance or long-range reasoning; dense attention still costs compute |
| Recurrent state | Ordered updates with a fixed-shape history summary | Compression causes interference or forgetting; not an exact record of every token |
| Recurrence plus global attention | State updates alongside per-position access | Both recurrent state and KV cache need management; the latter still grows with length |

NoPE is not a cache format. Removing positional encoding does not remove every token's keys and values. A fixed state also does not imply constant training memory: activations, gradients, chunks, and recomputation still matter.

Similarly, MLA's compression projections remain useful without rotation. Changing latent dimensions, head shapes, and cache layout introduces additional variables. Separate those comparisons rather than assigning every difference to “removing positional encoding.”

## Run the three distinctions {#run-example}

The [teaching script](../code/order_without_positions.py) uses only Python's standard library:

```bash
python3 00-foundations/code/order_without_positions.py
```

```text
Recurrent final states: 2.5, 2.0
Causal means, layer 1: 1.0, 1.0
Causal means, layer 2: 1.166667, 1.5
```

First set decay to 1 and check whether the final state still distinguishes order. Then restore the decay and append 0 to both input sequences: does the difference survive? Finally replace causal prefix means with a global mean. Can a second layer recover the distinction that the first removed?

The script also supports splitting a sequence into two runs. Passing the first run's final state into the second should match an uninterrupted run. An independent new sample should instead reset state. This boundary is more useful to test than merely saying that recurrence “remembers things.”

The matrix example uses `channel_delta_step`: one state update, without trainable projections, ShortConv, or a GPU chunked kernel. `content_attention` checks how outputs follow input permutations without a mask. The first two loops take linear time; this naive attention takes quadratic time and is not a performance comparison.

## What should a real-model evaluation check? {#evaluation}

Small calculations explain mechanisms; model experiments establish usefulness. Start with focused comparisons rather than immediately testing the maximum context window:

| Question | Controlled change | Measure |
| --- | --- | --- |
| Does event order matter? | Keep events and query fixed; reverse event order | Whether answers change with order |
| Does it merely favor the latest sentence? | Keep a valid rule and add similar but obsolete rules afterward | Selection of valid evidence, not automatic trust in the last sentence |
| Can it retrieve distant evidence? | Move evidence and add irrelevant text | Accuracy by position and length |
| Can it combine information? | Require two separated pieces of evidence | Multi-step success, not only single-fact lookup |
| Is caching correct? | Same tokens, uninterrupted versus split-and-resumed execution | Logits / state agreement within specified tolerances |

Fix the checkpoint, tokenizer, prompt, and decoding settings. Report quality, time to first token, subsequent-token latency, and memory separately. Architecture comparisons also need training and compute budgets; differences between training recipes cannot all be credited to NoPE.

Checked 2026-10-09. The calculations and script illustrate mechanisms, not K3 performance. For more on compressed history, continue to [MLA and sparse attention](latent-and-sparse-attention.en.md).
