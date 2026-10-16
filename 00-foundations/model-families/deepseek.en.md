# DeepSeek deep dive: architecture, systems, and reasoning training together

[中文](deepseek.md) · **English**

> Reading time: ~13 min · Type: model family deep dive · Last reviewed: 2026-10

<div class="lesson-recipe advanced">
  <div><span>Core question</span><strong>How do you control attention, FFN, and reasoning cost in a high-capacity model?</strong></div>
  <div><span>Main components</span><strong>MLA · fine-grained MoE · shared expert · MTP</strong></div>
  <div><span>Training line</span><strong>V3-Base → R1 multi-stage training; small-model distillation is a separate branch</strong></div>
  <div><span>After reading</span><strong>do not collapse V3 architecture and R1 post-training into one story</strong></div>
</div>

## One-sentence position

The most useful lesson in DeepSeek is **co-design**. MLA compresses attention state, MoE expands FFN capacity, the training and communication system makes the sparse model practical, and R1 shifts post-training toward verifiable reasoning. Each layer manages a different cost; “it uses MoE” is not enough.

## Separate the two lines first

<figure class="worked-update" id="deepseek-architecture" lang="en">
  <figcaption>First, separate V3's architectural and training components</figcaption>
  <ol>
    <li><strong>MLA</strong><span>Compress the KV stored per token, rather than collapsing the entire history into one record.</span></li>
    <li><strong>DeepSeekMoE</strong><span>Use a subset of experts for each token's FFN computation while retaining greater total capacity.</span></li>
    <li><strong>MTP and training systems</strong><span>Add future-token targets; parallelism and communication determine how the computation runs.</span></li>
  </ol>
</figure>

<figure class="worked-update worked-update--pairs" id="deepseek-training-routes" lang="en">
  <figcaption>Then, distinguish two post-training routes from V3-Base</figcaption>
  <ol>
    <li><strong>V3 chat model</strong><span>Apply SFT and RL to V3-Base to improve response behavior.</span></li>
    <li><strong>R1 and smaller students</strong><span>Apply multi-stage post-training to V3-Base. Separately, generate and filter R1 text for SFT of smaller Qwen or Llama models.</span></li>
  </ol>
</figure>

V3 chat post-training and R1 follow different routes from V3-Base. R1 is not a module attached to the V3 chat model. This diagram covers V2 / V3 / R1; V4 introduces a different attention design.

The branches distinguish weight initialization, not complete data independence: V3 used an internal R1 model to help produce reasoning SFT data. [V3 §5.1](https://arxiv.org/html/2412.19437v2#S5.SS1)

Two updates sit between V3 and V4; this note does not cover them in depth. [V3.1](https://api-docs.deepseek.com/news/news250821), released on 2025-08-21, puts thinking and non-thinking modes into one model: in the API, `deepseek-reasoner` uses thinking mode and `deepseek-chat` uses non-thinking mode. The [V3.2 report](https://arxiv.org/abs/2512.02556) (December 2025) adds DeepSeek Sparse Attention (DSA) next to MLA: a cheap indexer picks the top-k positions, and core attention reads only those. For DSA's complexity and the cost of a wrong selection, see the DSA section of [MLA and sparse attention](../deep-dives/latent-and-sparse-attention.en.md).

**Version scope:** this note checks V2 report v5, V3 report v2, and the original R1 report v1, not every Lite, minor-version, or dated checkpoint. The official configurations below were read on 2026-10-10; their `main` URLs are mutable. The R1 report was later published in [Nature 645, 633–638](https://doi.org/10.1038/s41586-025-09422-z) (2025), and arXiv [v2](https://arxiv.org/abs/2501.12948v2) followed on 2026-01-04: R1-Zero is in §2, R1's two RL stages are in §3, and the data recipe moved to Appendix B.3. The four-stage order below is unchanged in v2. The statement that the third stage fine-tunes V3-Base again comes from v1 §2.3.3; v2's main text and Appendix B.3 no longer state that stage's starting checkpoint.

| Configuration | Attention | Routed / selected per token / shared experts in an MoE layer | Distinction to preserve |
| --- | --- | --- | --- |
| [DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/config.json) | MLA | 160 / 6 / 2 | Softmax affinity; the first FFN is dense |
| [Original DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json) | MLA | 256 / 8 / 1 | Sigmoid affinity; the first three FFNs are dense |
| [Original DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/config.json) | V3-style MLA | 256 / 8 / 1 | This is the large model, not an R1-Distill architecture |

## Three ideas worth keeping

1. **MLA stores less per token, not one vector for the entire history.** Each token adds a latent and a decoupled positional key. Going from 1,000 to 2,000 tokens still increases cache entries. Projection absorption can avoid expanding full K/V during inference. [V2 §2.1](https://arxiv.org/html/2405.04434v5#S2.SS1)
2. **MoE separates total capacity from per-token computation.** Routed experts provide capacity; each token uses a subset. Shared experts aim to capture reusable patterns, without guaranteeing neatly separated domains. Routing, communication, and storage remain costs.
3. **R1 investigates what post-training can change.** R1-Zero applies RL to a pretrained V3-Base, not random weights. It does not establish that pretraining is unimportant or that RL explains every capability beyond the reward.

“Fine-grained” means cutting wide experts into more narrow ones and selecting more of them per token: total expert parameters and the main multiply-adds per token stay the same, while the possible combinations and the routing and communication costs change. For the worked comparison of 8-choose-2 with 32-choose-8, and why adding a shared expert increases the budget, see [Fine-grained and shared experts](../moe/fine-grained-and-shared.en.md).

### MLA cache accounting: one latent, not two {#mla-cache-ledger}

Let $B$ be batch size, $T$ cached tokens, $L$ layers, $d_c$ the KV latent width, $d_R$ the decoupled RoPE key width, and $s$ bytes per element for both components. The logical payload of a compact MLA cache is:

$$
\text{bytes}=BTL(d_c+d_R)s.
$$

There is no extra factor of two: K and V share a latent, and the positional key is shared across query heads. For content query $q$, cached latent $c$, and key up-projection $U_K$, scores satisfy $q^T U_Kc=(U_K^Tq)^Tc$, allowing a transformed query to address the latent directly. RoPE uses a separate branch; position-dependent rotations cannot simply be absorbed too. [V2 §2.1](https://arxiv.org/html/2405.04434v5#S2.SS1)

```python
def mla_cache_bytes(batch, tokens, layers, latent_width, rope_width, bytes_per_element):
    dimensions = (batch, layers, latent_width, rope_width, bytes_per_element)
    if any(type(value) is not int or value <= 0 for value in dimensions):
        raise ValueError("Expected positive integer dimensions and element bytes")
    if type(tokens) is not int or tokens < 0:
        raise ValueError("Expected a nonnegative integer token count")
    return batch * tokens * layers * (latent_width + rope_width) * bytes_per_element

assert mla_cache_bytes(1, 8192, 61, 512, 64, 2) == 549 * 2**20
```

Using V3's configured `61` layers and `512 + 64` dimensions, with assumed BF16 cache storage (2 bytes/element), one 8,192-token request needs **549 MiB** (1 MiB = $2^{20}$ bytes). This is derived payload, not measured device memory. It excludes weights, MTP, paging slack, quantization metadata, workspaces, and parallel replicas; runtimes need not use the compact layout. MLA alone still attends over visible history: narrower cached vectors do not imply sparse token selection.

## V3 load balance: selection scores are not mixture weights

V3 selects routed experts using affinity plus a load bias, but normalizes the selected original affinities for output mixing. Overload decreases an expert's bias; underload increases it. A small sequence-wise balance loss remains, so “auxiliary-loss-free” does not mean every balancing auxiliary term is absent. [V3 §2.1.2](https://arxiv.org/html/2412.19437v2#S2.SS1.SSS2)

Take three invented affinities `[0.6, 0.3, 0.1]` and select two experts. Biases `[0, 0, 0.4]` give the third expert more opportunities:

| Step | Values | Result |
| --- | --- | --- |
| Selection scores | `[0.6, 0.3, 0.5]` | Select experts 1 and 3 |
| Mixture weights | Normalize the original `0.6` and `0.1` | `6/7` and `1/7` |
| Incorrect shortcut | Normalize the biased selection scores | `6/11` and `5/11` |

With selected outputs 2 and 10, the correct mixture is `22/7`; the shortcut produces `62/11`. Reusing the wrong scores changes the forward computation.

```python
import math

def route_with_bias(affinities, biases, selected_count):
    if len(affinities) != len(biases) or not affinities:
        raise ValueError("affinities and biases must have matching nonzero lengths")
    if type(selected_count) is not int or not 1 <= selected_count <= len(affinities):
        raise ValueError("invalid selected_count")
    if not all(math.isfinite(value) for value in [*affinities, *biases]):
        raise ValueError("scores must be finite")
    if any(value <= 0 for value in affinities):
        raise ValueError("this example expects positive affinities")
    selected = sorted(
        range(len(affinities)), key=lambda index: (-(affinities[index] + biases[index]), index)
    )[:selected_count]
    denominator = sum(affinities[index] for index in selected)
    return [(index, affinities[index] / denominator) for index in selected]

routed = route_with_bias([0.6, 0.3, 0.1], [0.0, 0.0, 0.4], 2)
assert [index for index, _ in routed] == [0, 2]
assert math.isclose(sum(weight * [2, 0, 10][index] for index, weight in routed), 22 / 7)
```

This illustrates only the routed branch, without shared experts, node constraints, full scaling, or communication. A bias updated from prior load statistics is not new evidence of relevance for the current token. Measure tokens received per device too: balanced mean expert loads do not guarantee balanced inter-node traffic.

For another set of numbers, where adding the bias changes which experts are selected, and why expert balance and device balance are separate tables, see [Load balancing](../moe/load-balancing.en.md#bias-example).

### MTP adds a target, not permission to skip verification {#mtp-causal-chain}

V3 uses MTP depth 1. The backbone state $h_i$ predicts $t_{i+1}$; an extra module combines $h_i$ with the ground-truth embedding of $t_{i+1}$ to predict $t_{i+2}$. Modules preserve causal order and share the backbone's embedding and output head, rather than making independent predictions from separate heads. [V3 §2.2 and §4.2](https://arxiv.org/html/2412.19437v2#S2.SS2)

For the invented sequence `A B C D`, the backbone at `A` predicts `B`, while MTP consumes `B` to predict `C`. Its input `B` is a condition, not its target; feeding `C` would leak the answer. This helper lists alignments for one unpadded sequence, not model computations or losses. Packed data additionally require masking targets that cross document boundaries:

```python
def mtp_training_rows(tokens):
    return [
        (tuple(tokens[:position + 1]), tokens[position + 1], tokens[position + 2])
        for position in range(len(tokens) - 2)
    ]

assert mtp_training_rows(["A", "B", "C", "D"]) == [
    (("A",), "B", "C"), (("A", "B"), "C", "D")
]
```

Training adds the auxiliary objective to next-token loss. At inference, MTP can be removed for ordinary autoregressive generation or used for speculative decoding. Speculation requires backbone verification, not unconditional acceptance of every draft. Training quality, draft acceptance, and end-to-end latency are distinct: predicting two positions does not imply a twofold speedup.

## R1 has four stages; distillation is separate

The [original R1 report, §2.3–2.4](https://arxiv.org/html/2501.12948v1#S2.SS3), distinguishes:

| Stage | Purpose |
| --- | --- |
| Cold-start SFT | Give V3-Base a small set of readable reasoning demonstrations |
| Reasoning-oriented RL | Train reasoning with verifiable task rewards |
| Rejection sampling + SFT | Filter RL-generated samples, add general-task data, then fine-tune V3-Base again |
| All-scenarios RL | Further balance reasoning, helpfulness, and safety |

At stage three, **the data-generating checkpoint and the SFT starting point play different roles**. A diagram must distinguish data transfer from weight updates.

Small-model distillation is another branch: use the curated data for Qwen / Llama SFT. The report's distilled models neither compress V3's MoE architecture nor each repeat R1's RL pipeline.

### Checking the outcome does not necessarily check the reasoning

R1-Zero uses rule-based accuracy and format rewards. Later R1 stages also involve language consistency and general preferences. Do not extrapolate one phase's rule-based reward to the entire training pipeline. [R1 §2.2–2.3](https://arxiv.org/html/2501.12948v1#S2.SS2) For why answers grow longer on their own when only the answer is rewarded, see the R1-Zero section of [How long chains of thought are trained](../../11-frontier/test-time-compute/long-chains.en.md).

For an original small example, suppose an addition problem has answer 12. A response ends with 12 but explains it as “5 + 8.” A final-number checker rewards it; a step check catches the inconsistency. Conversely, a brittle parser might reject a valid algorithm's answer expressed in different units. Test **verifier reliability, reward coverage, and whether training actually uses the intended signal** separately. For how a verifier's error rate caps the gains of best-of-N, see [Sampling, voting and verifiers](../../11-frontier/test-time-compute/parallel-and-verify.en.md).

Local fixtures should include a correct response, a correct response in a different format, a correct result with faulty reasoning, and a fluent but wrong response. The training checker should not be the only independent evaluation.

Rejection sampling creates another choice. Suppose four samples per question produce four correct responses for one question and only one for another. Keeping all five as independent SFT examples weights the first question four times as much; retaining one per question weights them equally. These are invented counts, not paper statistics. “Keep correct answers” still leaves sampling and weighting decisions: curation is part of the objective.

## Continue to V4: the object being compressed changes

The [V4 walkthrough](deepseek-v4.en.md) examines CSA / HCA hybrid attention, mHC, and post-training. Sequence compression is not MLA's channel compression; constraints on mHC residual mappings are not Muon's update orthogonalization; a fixed local window does not make the whole history cache constant-size.

Keep mechanisms separate from measurements. Compression is not a losslessness guarantee, orthogonalization does not guarantee convergence, and a FLOPs ratio at one context length is not a universal latency ratio. V4.1-Flash, released after V4, changes the architecture, and the old API names now route to the new weights; see the note at the top of the V4 walkthrough.

## Trade-offs

| Choice | What it buys | What it costs |
| --- | --- | --- |
| MLA | Smaller logical KV payload; compact implementations can reduce cache traffic | Implementation complexity; latency still depends on kernels and parallelism |
| Fine-grained MoE | Large total capacity with fewer active parameters | All-to-all communication, routing, and load balance |
| Multi-token prediction | Denser training signal and possible decoding gains | More complex objective and implementation |
| Reasoning RL | Reward-guided task optimization; longer reasoning need not be correct | Rollout cost, reward boundaries, and behaviour regressions |

## How I would use the family

DeepSeek is good practice for not mixing levels. For every result I would ask: is this a cache or compute systems gain, an expert-capacity architecture gain, a pretraining-data gain, or a reasoning post-training gain? If the report does not isolate them, the uncertainty should remain explicit.

## Self-check

To work through the components, continue with [MLA and sparse attention](../deep-dives/latent-and-sparse-attention.en.md), [MTP targets and causality](../deep-dives/multi-token-prediction.en.md), and [FFN and SwiGLU](../core/ffn-and-gates.en.md). These explain mechanisms; they do not replace checking a specific model version.

- Why does MLA primarily improve the KV cache rather than directly prove stronger reasoning?
- Which compute does MoE save, and where does the cost move?
- What does the difference between R1-Zero and R1 teach us?
- Why is a distilled dense model not simply “the R1 architecture, but smaller”?

## Primary sources

- [DeepSeek-V3 Technical Report, v2](https://arxiv.org/abs/2412.19437v2)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, v1](https://arxiv.org/abs/2501.12948v1)
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model, v5](https://arxiv.org/abs/2405.04434v5)
