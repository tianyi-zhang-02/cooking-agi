# DeepSeek-V4: What Remains After Compressing History?

[中文](deepseek-v4.md) · **English**

A longer context limit lets a model accept more input. The harder questions are what history to retain, which parts to read next, and whether the savings discard information the task needs.

Continue here after [V2 / V3 / R1](deepseek.en.md). Attention, residual paths, and training change different things. The 2026-10-10 audit covers report v1 and the public configurations and inference source for the V4-Pro / V4-Flash previews, not every subsequent release. It does not reproduce reported performance.

> **Later releases (as of October 2026):** On 2026-08-13 [V4-Pro reached general availability](https://api-docs.deepseek.com/news/news260813) under the same API names, and V4-Pro and V4-Flash gained three reasoning-effort levels: low, high and max. [V4.1-Flash](https://api-docs.deepseek.com/news/news260910), released on 2026-09-10, moves to a new Causal Encoder–Decoder architecture. According to its [model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), it is an MoE with 552B backbone parameters that activates 8B per token during prefill and 16B during decode, and it accepts images natively. Its attention is CSA2, which gives each layer one of three fixed modes (Full, Reindex or Reuse); the card does not mention HCA. The September announcement said V4-Flash would retire and its API name would temporarily route to V4.1-Flash, with `deepseek-v4-pro` following from September 14 until V4.1-Pro launched. However, when checked on 2026-10-21, the [official model table](https://api-docs.deepseek.com/quick_start/pricing/) still identified `deepseek-v4-pro` as V4-Pro-0813. Those sources disagree, so the announcement alone does not establish today's routing. The same table now recommends the name `deepseek-flash`, served by [V4.1-Flash-1017](https://api-docs.deepseek.com/news/news261017) from the 2026-10-17 update; the legacy name `deepseek-v4-flash` is still accepted and routes to the same version. The CSA / HCA analysis below describes the V4 weights; confirm the actual served model and revision before comparing API results.

## 1. What do the two attention types save?

The [V4 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) describes interleaved CSA and HCA layers: CSA compresses then selects sparse entries; HCA compresses more heavily and reads the compressed history. Both retain a local sliding window. These are different layer configurations, not two consecutive attention operations inside every layer.

Pro and Flash are not identical stacks. These configuration values were read on the audit date; the linked `main` files can change:

| Configuration field | [V4-Pro config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json) | [V4-Flash config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json) |
| --- | --- | --- |
| Backbone `num_hidden_layers` | 61 | 43 |
| Attention in the first two layers | HCA plus local window | Sliding-window attention (SWA) only |
| CSA `index_topk` | 1024 | 512 |
| Routed experts per MoE layer | 384 | 256 |

Both use CSA / HCA ratios 4 / 128 and a 128-position window, selecting 6 routed experts alongside 1 shared expert. Their first three MoE layers use token-ID hash routing, not content top-k; later affinity scoring uses `sqrtsoftplus`, not V3's sigmoid. Both configure one extra MTP layer: the final `0` in `compress_ratios` describes this SWA-only MTP layer, not another backbone layer. [Source interpreting these fields: Gate / Transformer](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/model.py)

This diagram omits projections, normalization, and position handling:

<figure class="worked-update worked-update--pairs" id="v4-history-paths" lang="en">
  <figcaption>Two layer configurations: the local window stays; distant-history access differs</figcaption>
  <ol>
    <li><strong>CSA layer</strong><span>History → lighter compression → select entries. Core attention reads those entries alongside recent original entries.</span></li>
    <li><strong>HCA layer</strong><span>History → heavier compression → read every compressed entry. Core attention combines them with the local window too.</span></li>
  </ol>
</figure>

The branches represent alternative layers, not a head running both. CSA uses overlapping projected blocks while emitting one entry per m positions, not one per 2m. See [report §2.3](https://arxiv.org/html/2606.19348v1#S2.SS3).

Core attention uses shared-KV MQA, so V3's MLA cache formula cannot simply be reused. Local and compressed entries share one normalization, whose denominator also includes a learned attention sink. Separately normalizing each branch and adding the results is not equivalent. Consult the source for positional handling and other implementation details.

## 2. Storage and reads are different budgets

Consider a reduced accounting example: 1,024 processed positions, compression ratios 4 and 128, 32 selected entries, and a 128-position local window. Here `history` includes the current query position: this is the ledger after processing that position, not the preceding prefix length.

| Quantity | Lighter compression + selection | Heavier compression + full read |
| --- | --- | --- |
| Completed compressed entries | 256 | 8 |
| Compressed entries read by a query | At most 32 | 8 |
| Main-attention entries including the window | At most 160 | At most 136 |
| Omitted work | Compressor, indexer, projections, temporary state | Compressor, projections, temporary state |

The ratios and window match V4, but top-k is reduced to 32; this is not an actual Pro / Flash configuration or a latency prediction. Compressed entries do not contain the same information as individual tokens. Reading 32 entries also does not mean storing only 32: a later query may need different ones. Local and compressed entries remain distinct even when they cover overlapping text. These counts neither deduplicate original tokens nor include the scalar sink.

```python
def attention_ledger(history, light_ratio, heavy_ratio, selected, window):
    values = (history, light_ratio, heavy_ratio, selected, window)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("Expected positive integer counts")
    if heavy_ratio <= light_ratio:
        raise ValueError("Heavy compression must use a larger ratio")
    light_entries = history // light_ratio
    heavy_entries = history // heavy_ratio
    local_entries = min(history, window)
    return {
        "light_stored": light_entries,
        "heavy_stored": heavy_entries,
        "light_main_reads": min(selected, light_entries) + local_entries,
        "heavy_main_reads": heavy_entries + local_entries,
        "light_tail": history % light_ratio,
        "heavy_tail": history % heavy_ratio,
    }

assert attention_ledger(1024, 4, 128, 32, 128)["light_main_reads"] == 160
assert attention_ledger(1024, 4, 128, 32, 128)["heavy_main_reads"] == 136
assert attention_ledger(9, 4, 128, 32, 128)["light_tail"] == 1
```

The ledger counts completed blocks. `light_tail` / `heavy_tail` count only the unfinished current block, not all compressor state; CSA also retains the previous block for overlap. Its indexer still scores visible compressed candidates: top-k bounds subsequent core-attention reads, not total work. Compressed storage grows with history, and HCA core-attention reads grow too.

Byte accounting must also distinguish the report from the demonstration code. Report §2.3.4 describes BF16 RoPE dimensions and FP8 remaining KV dimensions; the inspected `inference/model.py` stores demonstration caches in BF16 while simulating quantization. Entry counts alone do not establish actual memory use. [Report](https://arxiv.org/html/2606.19348v1#S2.SS3.SSS4) · [Inference source](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/model.py)

## 3. Two errors that are easy to miss

**Future leakage.** A zero-based query at 8 with block width 4 has processed 9 positions and has two completed entries; an entry containing the entire 8–11 block would leak future tokens. In CSA, the second entry combines 4–7 with the previous block 0–3, rather than representing 4–7 alone. Overlap does not change completion time. The inspected source counts completed entries with `(query_position + 1) // block_width`, so position 7 may already read the newly completed second entry. Test boundaries such as 3, 4, 7, and 8: changing future tokens must not affect the current output. The causal local window also retains current and nearby content.

**Irreversible information loss.** Suppose a simplified compressor stores only a mean. Both `[1, 9]` and `[5, 5]` become 5, so their maxima can no longer be distinguished. V4 is not mean pooling; this counterexample simply explains why compression must be evaluated on tasks rather than judged by its ratio.

Test single evidence, multiple evidence, exact numbers, similar distractors, and unanswerable cases. See [benchmarks and long-context tests](../../07-evaluation/benchmark-protocols.en.md).

## 4. mHC constrains residual mixing; Muon changes updates

mHC uses multiple residual streams:

$$
X_{\ell+1}=B_\ell X_\ell+C_\ell F_\ell(A_\ell X_\ell).
$$

Rows of $X\in\mathbb{R}^{n\times d}$ are the $n$ streams. $A\in\mathbb{R}^{1\times n}$ combines them, $F$ still acts on a $d$-dimensional vector, and $C\in\mathbb{R}^{n\times1}$ redistributes the result. Nonnegativity and row/column-sum constraints apply to $B\in\mathbb{R}^{n\times n}$, approximated with finite Sinkhorn iterations. [Report §2.2](https://arxiv.org/html/2606.19348v1#S2.SS2)

Take a **fixed** example:

$$
B=\begin{bmatrix}0.8&0.2\\0.2&0.8\end{bmatrix},
\quad X=\begin{bmatrix}2\\10\end{bmatrix},
\quad BX=\begin{bmatrix}3.6\\8.4\end{bmatrix}.
$$

The sum remains 12; the sum of squares falls from 104 to 83.52. More generally, convexity gives, for fixed nonnegative $B$ with unit row and column sums:

$$
\sum_i\left(\sum_j B_{ij}x_j\right)^2
\leq \sum_{i,j}B_{ij}x_j^2
=\sum_jx_j^2.
$$

This is **not a contraction proof for the entire model**. The transformed branch remains, and actual mappings depend on the input. Finite normalization iterations also have numerical error; the theoretical target is not an exact floating-point identity.

Muon acts on parameter-update matrices. It is an optimizer, not attention or the same constraint on residual mixing. Both can affect stability while operating at entirely different locations. V4 does not apply Muon to every parameter: embeddings, output heads, RMSNorm weights, and static mHC biases / gating factors still use AdamW. [Report §2.4](https://arxiv.org/html/2606.19348v1#S2.SS4)

## 5. Post-training: specialists, then consolidation

The [official V4 description](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) separates specialist SFT / GRPO from subsequent on-policy distillation. Here specialists are teacher models, not individual MoE FFN experts. OPD is not weight averaging.

The reported OPD aligns the student with task-appropriate teachers on student-generated trajectories using full-vocabulary reverse KL. This differs from the original R1-Distill models' SFT on curated text. [Report §5.1.2](https://arxiv.org/html/2606.19348v1#S5.SS1.SSS2)

For an invented two-token distribution, let the student assign `[0.8, 0.2]` and the teacher `[0.5, 0.5]`:

$$
D_{\mathrm{KL}}(p_s\Vert p_t)
=0.8\log(1.6)+0.2\log(0.4)\approx0.193.
$$

Natural logarithms give units of **nats**, and this is KL at **one prefix**. Reversing its direction yields about 0.223 nats, a different objective. Training must also specify prefix collection, teacher selection, masks, normalization, and updates. Student-generated and teacher-generated prefixes visit different states: changing their source changes the experiment. Continue with [knowledge distillation](../../05-post-training/distillation.en.md).

## 6. What should a reproduction check first?

| Check | Small test | Likely issue if it fails |
| --- | --- | --- |
| Causality | Change future tokens; compare earlier outputs | Compressed-block boundaries or masks |
| Prefill / decode equivalence | Compare whole-sequence and incremental execution | Tail caches, positions, or restored state |
| mHC constraints | Row/column sums, minimum entry, mixing error | Normalization or precision |
| Compression quality | Vary evidence type, position, and distractors at fixed length | Aggregate scores conceal specific losses |
| Serving cost | Matched requests and concurrency; TTFT, latency, memory | FLOPs do not capture total latency |
| Post-training | Fix tasks, teachers, and generation budgets | Model, data, and inference compute are confounded |

Only the accounting and mathematical examples are tested here, not the model. Before deployment, check the [official inference instructions](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/README.md), weight precision, message encoder, and runtime version rather than trusting an automatically generated one-line launch command.
