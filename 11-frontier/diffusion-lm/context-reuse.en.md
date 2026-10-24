# Caching in diffusion decoding: what can stay, what must change?

[中文](context-reuse.md) · **English**

> Reading time: about 10 minutes · Last reviewed: 2026-10

If a diffusion language model has already read a long document, why access it repeatedly while generating the next small block? [Block diffusion](ar-and-diffusion.en.md#block-cache) can preserve the completed prefix's KV. But avoiding the computation that creates those representations does not eliminate the cost of reading them again.

This note follows that data flow: first the conditions for exact reuse, then the different costs targeted by E2D2, FlashBlock, and MAGE. These are not three interchangeable switches, and their headline speedups do not establish a ranking.

## Does fixed KV mean a fixed attention output?

No. Tokens in the active block change, and their queries can change with them. A query matches against historical keys to determine attention weights, which are then used to sum the values. Even with identical historical keys and values, the output can differ:

$$
A(q)=\operatorname{softmax}\left(\frac{qK^\top}{\sqrt d}\right)V.
$$

Consider a one-dimensional example with two historical positions: $K=[0,1]$, $V=[0,10]$, and $d=1$. Let $Z$ be the softmax denominator, the sum of exponentiated scores.

| Current query | Exponentiated scores | $Z$ | Historical attention output |
| --- | --- | --- | --- |
| $q=0$ | $[1,1]$ | 2 | 5 |
| $q=\log3$ | $[1,3]$ | 4 | 7.5 |

Nothing in the history changed, yet the output moved from 5 to 7.5 because the query changed. Reusing the historical **KV** can be exact. Reusing the previous **attention output** requires additional conditions and is generally approximate.

```text
Completed prefix → cached K, V ──┐
                                ├→ attention → layer output
Active block → current query ────┘
                    ↑
              may change next step
```

Exact prefix KV reuse still assumes that block-causal attention isolates the prefix from updates in the active block. It does not follow for an ordinary fully bidirectional model; see the [prefix-cache example](ar-and-diffusion.en.md).

## Three methods, three different changes

| Method and version | Reused object | Remaining work |
| --- | --- | --- |
| [E2D2, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/172e31282df16359324717fcfbdbead3-Paper-Conference.pdf) | Encoder features or KV for clean context | A smaller decoder denoises repeatedly; the encoder processes the completed new block |
| [FlashBlock, 2026-07 v3](https://arxiv.org/html/2602.05305v3) | Block-external attention outputs and normalization statistics | Recompute within-block attention, merge the two contributions, and refresh caches according to policy |
| [MAGE, 2026-06 v2](https://arxiv.org/html/2602.14209v2) | Historical KV indices selected from the initial all-`[MASK]` block | Recompute attention over selected KV with each step's new queries |

The blockwise version of E2D2 separates expensive context encoding from repeated denoising. Its two connections use final hidden states or shared layerwise KV, not necessarily a single compressed history vector. It is a jointly trained architecture, not an arbitrary split of an existing checkpoint. [§3.2–3.3](https://proceedings.neurips.cc/paper_files/paper/2025/file/172e31282df16359324717fcfbdbead3-Paper-Conference.pdf)

FlashBlock relies on observed cross-step stability, not guaranteed invariance. Version 3 also offers optional LoRA distillation to accommodate reuse errors; not every reported result is training-free. [§4](https://arxiv.org/html/2602.05305v3#S4)

MAGE caches **where to look**, not the answer returned by attention. Indices are selected per layer and KV head and reused across that block's later steps, rather than one list shared by every layer. It reduces KV access, but retains the complete KV cache and does not thereby shrink its memory footprint. [§4 and §6](https://arxiv.org/html/2602.14209v2#S4)

[Block Transformer](ar-and-diffusion.en.md#hybrids) is another potential source of confusion: it separates global and local modeling, but local generation is still autoregressive (AR). Hierarchy and blocks alone do not make a model a diffusion model.

## Why averaging two attention outputs is wrong

Split the keys into history and the active block. Each partial output has already been normalized over its own keys, but those two sets need not receive equal total weight.

For one query, layer, and head, call the outputs $A_{\mathrm{hist}}$ and $A_{\mathrm{block}}$, with softmax denominators $Z_{\mathrm{hist}}$ and $Z_{\mathrm{block}}$. The correct merge is:

$$
A=\frac{Z_{\mathrm{hist}}A_{\mathrm{hist}}+Z_{\mathrm{block}}A_{\mathrm{block}}}
{Z_{\mathrm{hist}}+Z_{\mathrm{block}}}.
$$

Reuse the new query from the earlier example: historical attention returns 7.5 with denominator 4. Suppose the active block has one key with score 0 and value 2, giving output 2 and denominator 1. The full output is:

$$
A=\frac{4\times7.5+1\times2}{4+1}=6.4.
$$

A plain average gives $(7.5+2)/2=4.75$, which is wrong. Reusing both the old historical output and its denominator gives $(2\times5+1\times2)/3=4$, also different from the current full computation.

An output vector alone is therefore insufficient: merging requires its normalization statistic. Implementations commonly store $L=\log Z$ and subtract the largest log-normalizer for numerical stability:

$$
\begin{aligned}
m&=\max(L_{\mathrm{hist}},L_{\mathrm{block}}),\\
A&=\frac{e^{L_{\mathrm{hist}}-m}A_{\mathrm{hist}}+e^{L_{\mathrm{block}}-m}A_{\mathrm{block}}}
{e^{L_{\mathrm{hist}}-m}+e^{L_{\mathrm{block}}-m}}.
\end{aligned}
$$

The merge is exact when both sets of statistics come from the **current query**. If the historical part comes from an older query, an accurate merge does not remove the staleness error. That is what attention-output reuse needs to evaluate. [FlashBlock's decomposition](https://arxiv.org/html/2602.05305v3#S4.SS2)

## Faster history access is not the same as faster generation

Suppose generating a block takes 100 ms: 60 ms for reading history and computing attention, and 40 ms for projections, FFNs, sampling, and other operations. Accelerating the first part by three times gives:

$$
T_{\mathrm{new}}=60/3+40=60\text{ ms},\qquad
S=100/60\approx1.67.
$$

If selection, refresh, and merging add 8 ms, the total becomes 68 ms: about $1.47\times$ faster. This teaching example assumes no overlap between operations; it is not a paper's measurement.

Short contexts may offer little historical access to save. With only a few denoising steps, the initial selection cost is harder to amortize too. Measure complete block latency rather than just the accelerated kernel, and record peak memory separately: less access does not automatically mean less storage.

## Separate sources of error before testing a reuse scheme

Suppose a model extracts an amount from a long contract, then explains its penalty conditions. An average score can hide a specific failure: the amount is correct, but the penalty requires reading a different passage, and stale attention outputs or indices fail to follow the change.

The following comparisons can help diagnose this. They are an experiment design, not completed measurements or a claim that adding a cache correction network is a validated new method.

| Comparison | What it separates |
| --- | --- |
| Full attention on fixed input states at every step | Layerwise reference results, without diverging sampling trajectories |
| Exact completed-prefix KV caching only | Bugs in the basic cache implementation |
| Frozen historical outputs and normalizers, varying refresh intervals | Stale-query errors and refresh costs |
| Frozen historical indices, attention recomputed with new queries | Selecting the wrong history versus reusing an old output |
| Free generation with the same decoding budget | Whether local errors change later choices and task outcomes |

Layerwise output errors are useful, but also measure token distributions, answer quality, and latency. Identical random seeds do not guarantee identical generation trajectories across implementations; both fixed-state comparisons and free-generation evaluations are needed.

If the design also compresses history into fewer representations, check information loss separately. Can it still retrieve exact amounts, dates, and negations? Smaller representations, fewer reads, and less frequent refreshes are different choices, even when all three are described as context reuse.

Return to [fair AR–diffusion comparisons](ar-and-diffusion.en.md#fair-test), or try the [review questions](review.en.md).
