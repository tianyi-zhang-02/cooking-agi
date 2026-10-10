# LatentMoE: must experts be as wide as the backbone?

[中文](latent-moe.md) · **English**

> Reading time: ~14 min · Level: advanced · Last reviewed: 2026-10-10 · Prerequisites: [Routing](router.en.md), [fine-grained and shared experts](fine-grained-and-shared.en.md)

Sending a token to eight experts means moving its representation to those experts. Even small expert computations can spend substantial time reading weights and moving data. What if the experts could work with a narrower representation?

That is the idea behind LatentMoE. **The routed experts' input and output become narrower; the backbone does not.** Start with the paths and budget table for the main idea. The later sections cover stability and balancing.

## Which part becomes narrower? {#paths}

<figure class="worked-update worked-update--pairs" aria-label="Full-width and latent paths in LatentMoE">
  <figcaption><strong>One input, two paths</strong><span>Follow a toy 128 → 64 → 128 computation.</span></figcaption>
  <ol>
    <li><small>Shared path · every token</small><strong>128 → 128</strong><span>Shared experts read and return width-128 vectors. This computation has not been narrowed.</span></li>
    <li><small>Routed path · selected experts</small><strong>128 → 64 → 128</strong><span>Project down to 64 for dispatch, expert computation, and combination. Project back to 128, then add the shared output.</span></li>
  </ol>
</figure>

One important connection is easy to omit: **the router can select experts from the original width-128 input**, not the width-64 latent. The vector being dispatched and the input used to score experts are different objects. [NVIDIA's original explanation](https://research.nvidia.com/labs/nemotron/LatentMoE/) makes this distinction explicit.

Using column vectors, let $D$ project down, $U$ project up, and $\mathcal T(x)$ denote the chosen experts. A basic version is

$$
\begin{gathered}
z=Dx,\\
u=\sum_{i\in\mathcal T(x)}g_i(x)E_i(z),\\
y=S(x)+Uu.
\end{gathered}
$$

$S$ combines the shared experts; the outer residual is omitted. Backbone width $d$, latent width $\ell$, and expert intermediate width $f$ are different dimensions. An expert maps $\ell\rightarrow f\rightarrow\ell$. Not every dimension has been halved.

## Three different ways to make something smaller {#three-widths}

| Change | What shrinks? | Direct effect, holding other choices fixed |
| --- | --- | --- |
| Fine-grained experts | FFN intermediate width $f$ | Internal expert computation; input/output may remain width $d$ |
| LatentMoE | Routed interface width $\ell$ | Expert matrices and dispatch vectors; adds down/up projections |
| MLA | Attention's historical KV representation | What must be cached for each past position |

LatentMoE and MLA can coexist, but LatentMoE does not itself compress attention's KV cache. For why MLA separates its RoPE branch, see [matrix absorption and position](../deep-dives/latent-and-sparse-attention.en.md#rope-absorption).

## Does the saving pay for more experts? {#budget}

Work through the savings and added costs with small dimensions: $d=128,f=256$, eight SwiGLU experts, and two active experts per token. Ignoring bias, each expert has $3df$ weights. LatentMoE changes this to $3\ell f$ and adds $2d\ell$ projection weights.

| Toy design | Routed experts | $\ell$ | Expert + projection parameters | Main MACs per token |
| --- | --- | --- | --- | --- |
| Ordinary MoE | 2 of 8 | 128 | 786,432 | 196,608 |
| Narrow the interface | 2 of 8 | 64 | 409,600 | 114,688 |
| Add more experts | 4 of 16 | 64 | 802,816 | 212,992 |

The third row matches the first row's **expert-only** budget, but the projections still cost something. Parameters rise about 2.1%; main MACs rise about 8.3%. The change is not free. Routing, shared experts, normalization, activations, and attention are excluded. Counting multiply and add separately, one MAC is about two FLOPs.

<details markdown="1">
<summary>Recalculate parameters, MACs, and dispatch bytes in Python</summary>

```python
def latent_budget(width, latent, intermediate, experts, top_k, tokens, item_bytes):
    values = (width, latent, intermediate, experts, top_k, tokens, item_bytes)
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError("Expected positive integer dimensions and counts")
    if latent > width or top_k > experts:
        raise ValueError("Invalid latent width or expert selection")
    projections = 0 if latent == width else 2 * width * latent
    per_expert = 3 * latent * intermediate
    return {
        "parameters": experts * per_expert + projections,
        "mac_per_token": top_k * per_expert + projections,
        "slot_bytes": 2 * tokens * top_k * latent * item_bytes,
    }

base = latent_budget(128, 128, 256, 8, 2, 32, 2)
small = latent_budget(128, 64, 256, 8, 2, 32, 2)
more = latent_budget(128, 64, 256, 16, 4, 32, 2)
assert [row["parameters"] for row in (base, small, more)] == [786432, 409600, 802816]
assert [row["mac_per_token"] for row in (base, small, more)] == [196608, 114688, 212992]
assert [row["slot_bytes"] for row in (base, small, more)] == [32768, 16384, 32768]
```

`slot_bytes` counts a separately dispatched input and returned output for every token–expert slot. With 32 tokens and two bytes per element, the round-trip totals are 32, 16, and 32 KiB. These are not measured network bytes: local experts, destination coalescing, padding, metadata, and topology change that accounting. Here `latent == width` represents ordinary MoE without the two projections.

</details>

Halving the width but doubling top-k brings the slot bytes back to the baseline. The question becomes whether quality improves at that cost, not just how much a dimension shrank. The original [LatentMoE paper](https://arxiv.org/html/2601.18089v1) distinguishes compute from bandwidth and communication constraints. The numbers above are independently constructed examples, not its benchmarks.

## Kimi K3: what else needs attention? {#kimi-k3}

K3 selects 16 of 896 routed experts per token. Its [official release](https://www.kimi.com/news/kimi-k3-open-source) calls the design Stable LatentMoE. [Report §2.3](https://arxiv.org/html/2607.24653v1#S2.SS3) specifies a width-7168 backbone, width-3584 routed path, and two full-width shared experts; the router scores the full input. Dispatching a 3584-dimensional vector does not mean the router must read one.

It adds RMSNorm between the combined expert output and up-projection, uses SiTU-GLU to control internal activation range, and uses Quantile Balancing for load. These address output scale, internal products, and uneven dispatch, respectively.

### Why the normalization cannot simply move {#normalization}

Consider two expert outputs, `[2, 0]` and `[0, 1]`, weighted equally. Ignore epsilon and learned gains:

| Computation order | Output |
| --- | --- |
| Combine, then RMSNorm | About `[1.265, 0.632]` |
| RMSNorm each, then combine | About `[0.707, 0.707]` |

The first preserves the 2:1 direction; the second changes it to 1:1. **RMSNorm is not a linear projection that can move through a sum.** An implementation can have the right shapes and still compute a different model.

SiTU has a related intuition: multiplying two large coordinates makes outliers worse, so smoothly cap both branches before multiplying. This is neither a constant replacement for every activation nor a guarantee against all network overflow. Learned projections can still amplify outputs, and gradients can become small in saturated regions.

## Quantile Balancing: measure now, use next step {#quantile-balancing}

Six tokens, three experts, and top-1 produce six assignments, giving an average target of two per expert. Average load alone does not tell us how to adjust a bias. We also need to know how far each expert is from being selected.

K3 uses the current biased $(k+1)$-th score as a cutoff and expert-wise score margins to estimate the next bias. Mixture weights still use raw affinities. **The bias takes effect next step and is frozen for inference.** Global quantiles are approximated with histograms. [Report §2.3.3](https://arxiv.org/html/2607.24653v1#S2.SS3.SSS3)

There are two important limits. A quantile calculated against old cutoffs does not guarantee perfectly balanced top-k assignments after every bias changes; the next batch can differ too. And “aux-loss-free” does not mean ordinary task gradients update the bias. A separate state-update rule still exists.

<details markdown="1">
<summary>Find a threshold for one expert</summary>

Suppose its raw-affinity-minus-current-cutoff margins for six tokens are `[0.31, 0.16, 0.07, -0.08, -0.19, -0.34]`. To retain two with **these cutoffs held fixed**, use bias `-0.07`. Only the first two adjusted margins are strictly positive.

```python
def fixed_cutoff_bias(margins, target):
    import math

    if not margins or any(not math.isfinite(value) for value in margins):
        raise ValueError("Expected finite margins")
    if type(target) is not int or not 0 < target < len(margins):
        raise ValueError("Expected an interior integer target")
    ordered = sorted(margins, reverse=True)
    return -ordered[target]

margins = [0.31, 0.16, 0.07, -0.08, -0.19, -0.34]
bias = fixed_cutoff_bias(margins, 2)
assert sum(margin + bias > 0 for margin in margins) == 2
```

This is an order-statistic example with a strict threshold, not full QB. Ties can retain fewer assignments than requested; a noninteger $Tk/N$ cannot give every expert an identical integer load. An implementation must also specify quantile interpolation, histogram bins, cross-rank aggregation, and state restoration. Subtracting a common mean from biases preserves ranking; including biases in mixture weights changes the computation.

Timing matters: if all tokens in a batch set its bias before that same batch is routed, an earlier token can depend on later tokens. Record the bias actually used, update the next-step state after collecting statistics, and save that state in checkpoints.

</details>

## How would you decide whether to use it? {#validation}

First identify the bottleneck. Small-batch decode may mainly read weights. Larger distributed batches may wait for all-to-all or overloaded experts. Top-k and parameter count alone cannot distinguish these cases.

| Question | Comparison | What else to track |
| --- | --- | --- |
| Does compression lose useful information? | Sweep $\ell$ with fixed data and training budget | Domain and rare-case quality, not just mean loss |
| Do gains come from compression or extra experts? | Compare all three budget rows | Router and projection overhead |
| Do the stability changes help? | Change norm, activation, and balancing separately | Gradients, activation tails, drops, load, and quality |
| Is serving faster? | Match hardware, precision, batch, and length | Separate prefill/decode; report throughput and p95 latency |

Nemotron 3 Super also uses LatentMoE, with different dimensions: its report specifies a $4096\rightarrow1024$ routed path. [Nemotron 3 Super §2.1.1](https://arxiv.org/html/2604.12374v1#S2.SS1.SSS1) Compression ratio is a design choice, not part of the definition. Published gains belong to those experiments. This page checks arithmetic and mathematical relationships, not model training or multi-GPU performance.

If you can explain the unchanged backbone, the difference between router input and dispatch payload, and the cost of extra projections, you have the main idea. Continue with [system costs](systems.en.md) to turn these dimensions into storage and time on actual hardware.
