# Qwen 1–2.5: separating architecture changes from training changes

[中文](qwen-early.md) · **English**

Early Qwen releases provide a useful exercise: when a generation improves, separate tokenizer, architecture, data, and post-training rather than crediting one component. This page covers historical releases; Qwen3 (2025) and later models are in [Reading Qwen](qwen.en.md).

## Identify the exact version first

| Version | Main change examined here | What does not follow |
| --- | --- | --- |
| Qwen | Large vocabulary, decoder, RoPE/RMSNorm/SwiGLU, separate base/chat models | Every chat behavior comes from pretraining |
| Qwen1.5 | Multilingual behavior, alignment, standardized interfaces | Every size has identical internals |
| Qwen2 | GQA, long context, dense and MoE options | Active parameters equal deployment memory |
| Qwen2.5 | Similar core blocks, expanded data and post-training | Gains require a new attention formula |

Sources: [Qwen](https://arxiv.org/abs/2309.16609), [Qwen1.5's official introduction](https://qwenlm.github.io/blog/qwen1.5/), [Qwen2](https://arxiv.org/abs/2407.10671), and [Qwen2.5](https://arxiv.org/abs/2412.15115). Publication dates, later model additions, and API aliases are distinct. An updated blog's model list is not necessarily the launch lineup.

For orientation, [Qwen1.5](https://qwenlm.github.io/blog/qwen1.5/) launched on **February 4, 2024**, [Qwen2](https://qwenlm.github.io/blog/qwen2/) on **June 7, 2024**, and [Qwen2.5](https://qwenlm.github.io/blog/qwen2.5/) on **September 19, 2024**. Technical reports may follow a launch; additional sizes and specialist versions arrive later too. Before comparing configurations or results in an older account, check which release it describes.

## Qwen: follow the input through to the loss

Byte-level BPE produces IDs, embeddings enter a causal decoder, attention and FFNs transform representations, and an output projection predicts the next token. RoPE handles positions, RMSNorm representation scale, and SwiGLU the FFN activation. They are not interchangeable attention optimizations.

The original report uses untied input embeddings and output projections. With hypothetical vocabulary $V=150000$ and hidden size $d=4096$, one matrix has $Vd=614400000$ parameters. An additional BF16 copy costs roughly 1.14 GiB before gradients or optimizer state. This is a teaching budget, not a particular Qwen configuration.

**Tokenizer inventory and model matrix dimensions are different counts.** [Qwen2 §2.1](https://arxiv.org/html/2407.10671v1#S2.SS1) specifies 151,643 regular tokens plus three control tokens and explicitly notes larger embedding dimensions. [Qwen2.5 §2](https://arxiv.org/html/2412.15115v1#S2) expands the control-token set to 22. The [Qwen2-7B config](https://huggingface.co/Qwen/Qwen2-7B/blob/main/config.json) instead has `vocab_size=152064` for model tensor shapes—not 152,064 ordinary vocabulary entries. Check ID mappings, added tokens, templates, and matrix shapes separately; matching row counts do not establish interchangeable protocols.

A larger vocabulary may shorten tokenized text but increases embedding and output-classification costs. When comparing Chinese and English throughput, record characters, tokens, batch size, and generated length. More tokens per second need not mean more source text per second.

## Qwen1.5: interfaces are part of the experiment

This generation moved toward standard Transformers interfaces, using tokenizer chat templates and `generate` rather than assuming an earlier model-specific `chat` method. The useful lesson is not memorizing historical dependency versions: **the same messages can become different model inputs**.

Before upgrading, save old and new token IDs for identical messages. Inspect roles, turn boundaries, assistant prefixes, EOS, and padding. Do not manually wrap a prompt and then apply the template again. The [Transformers template guide](https://huggingface.co/docs/transformers/chat_templating) explains that `apply_chat_template(..., tokenize=True)` can return IDs directly; if you render text with `tokenize=False` and tokenize it separately, set `add_special_tokens=False` to avoid adding special tokens twice.

An implementation class is not a release label. Qwen1.5-7B already uses `Qwen2ForCausalLM` and `model_type="qwen2"`, but it is not Qwen2-7B: even their Q/KV head counts differ below.

| Symptom | First check |
| --- | --- |
| Model keeps extending the user's message | Missing assistant generation prefix |
| Generation continues into another turn | Mismatched stopping tokens or rules |
| Loss unexpectedly falls after an upgrade | Prompt/template tokens incorrectly included, or misaligned labels |

These are diagnostic possibilities, not reported model defects. A base checkpoint should not be evaluated as if it already learned the chat protocol.

## Qwen2: calculate GQA's savings separately

GQA lets several query heads share K/V heads. The report's 7B model has 28 query heads and four KV heads. Holding everything else fixed, raw KV storage is $4/28=1/7$ of a 28-KV-head variant. That cache ratio is not a sevenfold whole-model speedup: the 28 query paths and the FFN do no less work, for the same reason as in the [Llama GQA example](llama.en.md#gqa-cache).

Nor is the boundary “all Qwen1.5 models use MHA; GQA starts with Qwen2.” These are specific inspected configurations, not a launch lineup:

| Config | Q heads / KV heads | KV fraction of a matched MHA variant |
| --- | --- | --- |
| [Qwen1.5-7B](https://huggingface.co/Qwen/Qwen1.5-7B/blob/main/config.json) | 32 / 32 | 1 (MHA) |
| [Qwen1.5-32B](https://huggingface.co/Qwen/Qwen1.5-32B/blob/main/config.json) | 40 / 8 | 1/5 |
| [Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B/blob/main/config.json) | 28 / 4 | 1/7 |
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json) | 28 / 4 | 1/7 |

Each row holds layers, head dimension, sequence length, and dtype fixed; these are not cross-model memory ratios. Using the [cache formula from the Llama section](llama.en.md#gqa-cache), this example counts only Qwen2-7B's KV tensors for one request, 28 layers, 8,192 tokens, and BF16:

```python
def kv_cache_bytes(*, batch, layers, tokens, kv_heads, head_dim, bytes_per_value=2):
    dimensions = (batch, layers, kv_heads, head_dim, bytes_per_value)
    if any(type(value) is not int or value <= 0 for value in dimensions):
        raise ValueError("cache dimensions must be positive integers")
    if type(tokens) is not int or tokens < 0:
        raise ValueError("tokens must be a non-negative integer")
    return 2 * batch * layers * tokens * kv_heads * head_dim * bytes_per_value


qwen2_gqa = kv_cache_bytes(batch=1, layers=28, tokens=8192, kv_heads=4, head_dim=128)
qwen2_mha = kv_cache_bytes(batch=1, layers=28, tokens=8192, kv_heads=28, head_dim=128)
assert qwen2_gqa == 448 * 2**20
assert qwen2_mha == 7 * qwen2_gqa
```

The result is 448 MiB, versus 3,136 MiB for a hypothetical 28-KV-head variant. This counts only the KV tensors and assumes shared KV is not physically expanded into 28 copies; weights, activations and framework overhead are excluded.

Qwen2 includes a 57B-A14B MoE alongside dense models. Record total parameters, activated experts, and shared experts separately before accounting for communication and resident weights. Its long-context methods include DCA/YaRN; see [positions and context](../deep-dives/position-and-context.en.md). Extending context is not simply changing a maximum-length constant.

## Qwen2.5: similar architecture still leaves substantial training choices

The public dense models retain the GQA, SwiGLU, RoPE, and RMSNorm core. Pretraining data scale, filtering, and mixtures change; the report separately describes SFT, offline DPO, and online GRPO. “It uses RL” hides these distinctions.

| Stage | Signal | Why it is not interchangeable |
| --- | --- | --- |
| Pretraining | Next-token learning on mixed corpora | Acquires language and knowledge patterns |
| SFT | Tasks with demonstrated outputs | Teaches desired response behavior |
| Offline preference | Previously compared response pairs | Limited by existing response coverage |
| Online generation/reward | Current-policy samples and scores | Closer to current behavior, with rollout and reward-design costs |

This tracks learning signals, not an identical recipe for every size, API, and specialist model. Coder, Math, VL, and later 1M releases are not merely one checkpoint with different context settings.

## A small experiment that keeps factors separate

Suppose structured extraction improves after an upgrade. Fix 50–100 original or authorized examples spanning Chinese, English, mixed-language, short, and long inputs. Keep the output schema fixed; test field correctness and uncertainty handling, not just JSON parsing.

1. Use each model's correct template and record exact revision, tokenizer, quantization, sampling, and truncation.
2. Compare base/instruct separately so post-training changes are not misattributed to architecture.
3. Compare at a shared context length and output budget before a separate long-context experiment.
4. Report error categories and costs. If data, scale, and template all changed, the result supports a version-level difference—not a causal claim about one component.

Checked 2026-10-10: tokenizer, architecture, and training sections of the reports above, Qwen1.5's interface introduction, and the linked model-specific configs—not every historical checkpoint. Config links target mutable `main`; pin revisions for reproduction. Full release benchmarks have not been rerun. Budgets and diagnostic examples are teaching cases.
