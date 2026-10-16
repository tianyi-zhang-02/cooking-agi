# Qwen deep dive: one family across scales and reasoning modes

[中文](qwen.md) · **English**

> Reading time: ~12 min · Type: model family deep dive · Last reviewed: 2026-10-10

<div class="lesson-recipe advanced">
  <div><span>Core question</span><strong>How can one family cover different budgets and workloads?</strong></div>
  <div><span>Main components</span><strong>Dense / MoE · GQA · QK-norm · multilingual</strong></div>
  <div><span>Training line</span><strong>four-stage large-model post-training; a separate strong-to-weak distillation route</strong></div>
  <div><span>After reading</span><strong>separate architectural sparsity, inference-time compute, and small-model distillation</strong></div>
</div>

What interests me in Qwen3 is this: **how does one family adapt to different tasks and budgets?** It has dense and MoE models, many sizes, and both thinking and non-thinking modes. Only by putting these versions side by side can we tell which differences come from model size and which from architecture or inference mode.

The original Qwen3 is the baseline here, compared with specific Qwen3-2507, Qwen3-Next, Qwen3.5, Qwen3.6, and Qwen3.8 models, checked as of 2026-10-10; Qwen 1–2.5 are covered in the [early-version walkthrough](qwen-early.en.md).

## One-sentence position

Qwen offers comparable versions along all three axes—scale, sparsity, and inference budget—so it is good practice for accounting for them separately.

Mode switching here refers primarily to the original Qwen3 hybrid checkpoints. [Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) is non-thinking-only, whereas [Thinking-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507) is thinking-only. These are different weights, not two request settings for one checkpoint.

### Original Qwen3: inspect the config, not just the size label {#original-qwen3-config}

The [2025 Qwen3 report, §2](https://arxiv.org/html/2505.09388v1#S2), gives these GQA head counts. They do not describe Next, 3.5, or 3.8:

| Original model | Q heads / KV heads | KV fraction of matched MHA |
| --- | --- | --- |
| Qwen3-0.6B / 1.7B | 16 / 8 | 1/2 |
| Qwen3-4B / 8B | 32 / 8 | 1/4 |
| Qwen3-14B | 40 / 8 | 1/5 |
| Qwen3-32B | 64 / 8 | 1/8 |
| Qwen3-30B-A3B | 32 / 4 | 1/8 |
| Qwen3-235B-A22B | 64 / 4 | 1/16 |

Each fraction changes only that model's KV-head count; it is not a cache comparison across models. Original Qwen3 removes QKV bias and adds QK-norm relative to Qwen2.5. Normalizing Q/K helps stabilize training; it neither replaces softmax nor reduces the head count. Both original MoEs select eight of 128 experts per token and have **no shared expert**.

The [Qwen3-30B-A3B config](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json) illustrates another trap: `hidden_size=2048` and 32 query heads, but `head_dim=128`, not $2048/32=64$. Query projection width can differ from residual-stream width, so `hidden_size / num_attention_heads` is not a universal way to recover head dimension.

The [Qwen3-8B config](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json) has `vocab_size=151936` for matrix dimensions, while the report gives a tokenizer vocabulary of 151,669. Its [tokenizer config](https://huggingface.co/Qwen/Qwen3-8B/blob/main/tokenizer_config.json) also defines `<think>` / `</think>` IDs and the template. Check all of these, not just the BPE algorithm. Context figures have different meanings too: the 8B card separates 32,768 native tokens from 131,072 with YaRN. The default `max_position_embeddings=40960` neither proves a training sequence length nor enables 128K extrapolation by itself.

### What does a thinking switch actually change? {#qwen-thinking-switches}

In the [original Qwen3-8B interface](https://huggingface.co/Qwen/Qwen3-8B#switching-between-thinking-and-non-thinking-mode), `enable_thinking=False` is a hard chat-template switch. The `/think` and `/no_think` instructions are soft controls when thinking is enabled; they cannot override the hard-off setting. With `add_generation_prompt=True` and thinking disabled, the current template puts an empty `<think>…</think>` segment in the **prompt**, not in newly generated reasoning. Before testing for tags, distinguish template input, generated continuation, and fields extracted by a serving parser.

This does not retrain the weights per request or change the number of routed experts. Post-training teaches the protocol; templates and sampling guide its use at inference. `max_new_tokens` limits the entire continuation, not a separate reasoning allowance, and reaching that limit does not guarantee that a final answer has been produced. For separate thinking budgets and length control, see [Long chains: budgets and length control](../../11-frontier/test-time-compute/long-chains.en.md).

Later models add two more controls (as of October 2026). The [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) tunes reasoning depth with `reasoning_effort`, which accepts `xhigh` (the default), `medium` or `low`; its example passes it directly as a request parameter. `preserve_thinking` decides whether a multi-turn conversation keeps the thinking blocks from earlier turns: Qwen3.8 keeps them by default, while [Qwen3.6](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) by default keeps only the latest turn's. The first changes how much reasoning this answer spends; the second changes how much old reasoning the next turn's input carries. Like `enable_thinking`, both change the request and template, not the weights.

## Separate three ways to spend less compute

<figure class="worked-update" id="qwen-compute-choices" lang="en">
  <figcaption>Three choices, not three sequential steps</figcaption>
  <ol>
    <li><strong>Model size</strong><span>Choose smaller or larger weights. Distillation can help a smaller model learn from a larger one.</span></li>
    <li><strong>Dense or MoE</strong><span>Change which parameters each token uses. The remaining weights still need storage.</span></li>
    <li><strong>Inference budget</strong><span>Choose thinking or non-thinking where supported, then set the generation budget.</span></li>
  </ol>
</figure>

- **Size** sets static capacity and the basic deployment threshold.
- **MoE** lets total parameters grow faster than active computation per token, at the cost of routing and communication.
- **Thinking budget** changes how much generation compute one request may spend. It is neither MoE nor a change in parameter count.

Mixing the three makes “larger,” “sparser,” and “thinking longer” all look like the same kind of scaling.

## Three ideas worth keeping

1. **The family covers workloads, not just parameter points.** Small dense models target cheap deployment, large MoEs add capacity, and thinking mode reserves extra compute for hard problems.
2. **Original hybrid checkpoints support mode switching.** This is a checkpoint-specific interface, not a permanent promise for every family member. Report quality, length, and latency separately by mode.
3. **Distillation connects the family internally.** Knowledge and reasoning traces from large models can train smaller ones, so sizes are no longer isolated training runs.

## Trade-offs

| Choice | What it buys | What it costs |
| --- | --- | --- |
| Dense and MoE product line | One ecosystem across cost bands | More complex training, serving, and evaluation matrix |
| Thinking mode | More test-time compute for hard problems | Different latency, token cost, and output stability |
| Multilingual expansion | Wider language coverage and transfer | Harder data balance and long-tail evaluation |
| In-family distillation | Small models inherit part of large-model capability | The ceiling and biases depend on the teacher and data |

## Not every size follows the same training route

The [Qwen3 report, §4](https://arxiv.org/html/2505.09388v1#S4), describes four large-model stages: long-CoT cold start, reasoning RL, thinking-mode fusion, and general RL. Smaller models use strong-to-weak distillation, not a universal fifth stage.

Suppose the student has generated an imperfect prefix. Imitating a prewritten teacher response differs from querying the teacher on **the student's own prefix**. These correspond to the off-policy and on-policy distinction here. What matters is who generated the training context, not whether the teacher is deployed as an online service.

### Qwen3-Next changes the architecture, not just the label {#qwen3-next-boundary}

[Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) uses a 3:1 Gated DeltaNet/Gated Attention layout. Its MoE selects ten of 512 routed experts and adds one shared expert, unlike the original 128/eight design without shared experts. This Instruct checkpoint is non-thinking-only. The card also describes MTP: predicting additional future positions during training is distinct from an inference engine using it for speculative decoding, and having the module is not a measured speedup.

## Qwen3.5: recurrent state versus full history

For [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B), the official layout repeats three Gated DeltaNet layers and one Gated Attention layer, with a vision encoder. Its MoE has 512 routed experts, activating ten plus one shared expert per token. These are model-specific numbers. The [config](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json) nests the language model under `text_config`; full-attention Q/KV heads and DeltaNet linear heads are separate fields, not one interchangeable top-level count.

This model supports disabling thinking through `enable_thinking`, but the original slash-command interface should not be carried over. Its self-hosted examples wrap the setting in `chat_template_kwargs`; the official hosted examples pass it directly. API compatibility does not establish identical nesting for extension fields.

A recurrent state compresses history into a fixed-shape matrix; full attention still accesses historical KV. Combining them does not make the entire cache constant-size or full-sequence computation linear. Single-step decode and whole-sequence prefill also need separate accounting.

Updating memory along a key direction is not interference-free record deletion. Similar keys can be affected together. The [Gated DeltaNet derivation and numerical example](../deep-dives/gated-deltanet.en.md) works through this limitation and separates model parameters from forward-pass state.

## 2026 update: Qwen3.8 is not one uniform architecture

Qwen3.6 sits between Qwen3.5 and 3.8. According to [Qwen's GitHub release log](https://github.com/QwenLM/Qwen3.8), the MoE Qwen3.6-35B-A3B was released on 2026-04-16 and the dense Qwen3.6-27B on 2026-04-22; both keep the 3:1 DeltaNet / attention layout and include a vision encoder.

This table covers official model cards actually read on 2026-10-10 (the Qwen3.6 row on 2026-10-16), not every 2026 release. These are mutable pages; pin the checkpoint revision, template, and inference-engine version when reproducing behavior.

| Public model | Token mixer / FFN | Input and deployment boundary |
| --- | --- | --- |
| [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | 3 Gated DeltaNet + 1 Gated Attention layout; MoE with 256 experts, 8 routed + 1 shared per token | Public vision-language model; thinking by default, can be disabled per request, no slash soft switches |
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Repeated 3 Gated DeltaNet + 1 Gated Attention layout; dense FFN | Public vision-language model with per-request thinking control, not renamed text-only Qwen3 |
| [Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) | Related hybrid attention with MoE; 2.4T total, 95B active | Public weights are text-only and thinking-only; hosted Max's vision, non-thinking, and built-in tools are not properties of these weights |
| [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | Gated DeltaNet + QSA, gated residuals, n-gram embeddings, and MoE | Public vision-language model with thinking control; QSA selects micro-blocks and the card calls it an architecture preview |

The 27B model still has full-attention layers, so hybrid recurrence does not imply constant total history storage. Active parameters in the 2.4T model describe a per-token computational subset, not permission to omit other weights from memory budgets. Flash-Next adds n-gram table storage and reads too.

### Why is 6B not Flash-Next's complete resource budget?

The card lists a 125B body, 51B n-gram embeddings, and 4B MTP under the language model: roughly 180B, with about 6B active in the body per token. That subtotal is not a complete deployment inventory including the vision encoder. These numbers answer different questions:

- 6B describes the principal active computation, not all resident weights or communication.
- Table lookups do not execute a 51B-parameter dense matmul per token, but still require storage, transfers, and caching.
- MTP training and inference use must be accounted for separately. Having the module does not guarantee faster decoding.

For that language-model subtotal alone, a rough BF16 raw-weight estimate is 180B × two bytes, about 360 GB (decimal). Quantization, offloading, sharing, and buffers change actual requirements; this is not a deployable GPU-memory plan.

The QSA card specifies 512 micro-blocks, or 2,048 tokens. **The 1/16 fraction applies to one query with 32,768 visible positions when counting 2,048 selected positions; it is not the fraction for an entire causal prefill.** In an idealized model where query $t$ reads exactly $\min(t,k)$ positions, total reads are $\sum_{t=1}^{S}\min(t,k)$, versus $S(S+1)/2$ for dense attention.

```python
def causal_read_pairs(tokens, budget=None):
    if type(tokens) is not int or tokens < 0:
        raise ValueError("tokens must be a non-negative integer")
    if budget is not None and (type(budget) is not int or budget <= 0):
        raise ValueError("budget must be a positive integer")
    capped = tokens if budget is None else min(tokens, budget)
    return capped * (capped + 1) // 2 + (tokens - capped) * capped


sequence_length, token_budget = 32768, 2048
decode_fraction = min(sequence_length, token_budget) / sequence_length
prefill_fraction = (
    causal_read_pairs(sequence_length, token_budget)
    / causal_read_pairs(sequence_length)
)
assert decode_fraction == 1 / 16
assert abs(prefill_fraction - 0.121092) < 1e-6
```

This token-budget example gives a prefill fraction of approximately 0.121092 (12.11%), not 6.25%. It excludes micro-block alignment, actual block selection, indexing, and kernel details, so it does not measure QSA's executed workload. Neither fraction is a whole-model speedup: MoE, recurrent updates, and other computations remain.

Likewise, its depth gating is not automatically [Block AttnRes](../deep-dives/attention-residuals.en.md), nor is every hashed n-gram scheme [Engram](../deep-dives/engram.en.md). Similar motivations do not establish identical implementations.

### How should we interpret model-card scores?

Align harness, tool permissions, context, output budget, sample count, and benchmark revision first. The 27B card notes corrected coding tasks and re-evaluated baselines; the 2.4T page also reports hosted Max results. Downloading weights and running an arbitrary script does not establish the same evaluation conditions.

For a useful local comparison, fix a task set and record success, failure categories, total generated tokens, tool calls, and latency. Architecture helps explain costs but cannot replace measurement. We have not rerun these model benchmarks, and model cards do not disclose every training datum or recipe.

## How I would use the family

Qwen is useful when the research question includes a **dynamic compute budget**. The same task can compare dense against MoE; a checkpoint that actually supports both modes also permits a fixed-weight thinking/non-thinking comparison. The discipline is not to report accuracy alone: also record generated tokens, latency, active parameters, and failure types.

## Self-check

- How is MoE sparsity fundamentally different from letting thinking mode run longer?
- Why must the evaluation protocol change when the same model switches modes?
- What does distillation give a model family, and which teacher biases can it propagate?
- If a small model approaches a large model on a benchmark, which cost and generalisation dimensions still matter?

## Primary sources

Reading scope: the Qwen3 report's architecture, pretraining, and post-training sections, plus the linked configs, tokenizer template, and relevant model-card sections. This is not a line-by-line inference-kernel audit, weight inspection, or training reproduction. The 2026 architecture/interface descriptions are attributed to official cards/configs; performance and undisclosed training details remain unverified.


- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Official Qwen3 code and models](https://github.com/QwenLM/Qwen3)
- [Gated Delta Networks](https://arxiv.org/abs/2412.06464)
