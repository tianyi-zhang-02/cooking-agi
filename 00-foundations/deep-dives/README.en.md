# Advanced reading: what does each component change?

[中文](README.md) · **English**

Once the Transformer makes sense, new names can start to blur together: FlashAttention, MLA, MoE, and various gates all seem to promise “efficiency.” Place each method back in the computation before memorizing its name. They do not change the same thing.

If Q, K, V, and residuals are unfamiliar, revisit the [complete Transformer](../transformer.en.md). For a sequential guide, follow [pretraining](../../learn/pretraining/README.en.md) or [generation and inference](../../learn/inference/README.en.md). This page helps distinguish mechanisms.

## Separate sequence positions, layers, and parameters

A sentence contains many tokens; a model contains many layers. **Reading across a sequence, mixing representations across depth, and choosing parameters to execute are different operations.**

| Where the change happens | Read | A small example that makes the distinction |
| --- | --- | --- |
| Attention output | [Gated Attention](gated-attention.en.md) | Multiply two head outputs by different gates; passing less information after computing it is not computing less |
| Historical state | [Gated DeltaNet](gated-deltanet.en.md) | Track an existing association after writing a new key |
| Local-pattern lookup | [Engram](engram.en.md) | The same phrase retrieves the same entry, but context controls its use; this is not chat memory |
| Network depth | [Attention Residuals](attention-residuals.en.md) | Combine earlier layer outputs for one token, rather than attending to more tokens |
| FFN parameters | [MoE](../moe/README.en.md) | Choose experts for a token; check whether saved computation becomes communication |
| Repeated computation | [Looped Transformers](../looped/README.en.md) | Reuse parameters across iterations; fewer parameters need not mean lower latency |

These components can coexist. In a particular configuration, still check which layers use them, where they sit, and whether the combination was trained together.

Place newer work next to the operation it changes: [Muon](muon.en.md) follows optimizers, while [DFlash / DFlash 2](dflash.en.md) follows KV caching. Parameter updates and draft verification need different prerequisites, not one publication-date reading list.

## Learning: from inputs to an update

| What to understand first | Read | What to check next |
| --- | --- | --- |
| Splitting text | [BPE, WordPiece, and Unigram](tokenizer-algorithms.en.md) | Vocabulary, byte coverage, and compatibility with model weights |
| Gradients through a sequence | [BPTT and gates](recurrent-dynamics.en.md) | Why products can shrink and what the LSTM additive path changes |
| Good training performance, poor validation | [Generalization and diagnosis](generalization.en.md) | Overfitting, leakage, and distribution shift require different responses |
| Initialization and gradient scale | [Activations and initialization](activation-and-initialization.en.md) | Xavier / He assumptions and scale through depth |
| Turning gradients into updates | [SGD to AdamW](optimizers.en.md) | The separate roles of momentum, second moments, and weight decay |
| Which positions supply supervision | [Language-model objectives](language-model-objective.en.md) → [One training update](training-step.en.md) | Shifting, masks, valid-token counts, and accumulation |
| Feeding data into training | [Pretraining pipeline](pretraining-pipeline.en.md) | Deduplication, mixtures, packing, validation, and resuming |
| Predicting several positions | [Multi-token prediction](multi-token-prediction.en.md) | Align extra targets; separate training gains from generation speed |
| Training memory use | [Precision and memory](precision-and-memory.en.md) | Account for weights, gradients, optimizer states, and activations separately |

To implement these steps, use the [four PyTorch chapters](../pytorch/README.en.md). They start with storage and shapes and finish with a training loop, without requiring distributed-training experience.

## Generation: different ways of saving resources

Suppose context grows from 4K to 32K. Some methods store less per position, some read fewer positions, and others improve data movement. Those changes affect different costs.

| Mechanism | Main change | What it does not mean |
| --- | --- | --- |
| [KV cache, MQA / GQA](kv-cache-and-inference.en.md) | Reuse historical K/V; reduce K/V head count | Historical information no longer needs processing |
| [RoPE, interpolation, and YaRN](position-and-context.en.md) | How positions enter attention | Raising the context limit guarantees useful access to distant evidence |
| [NoPE and sequence order](nope-and-order.en.md) | Where order can come from without explicit positional encoding | Any checkpoint can simply disable RoPE |
| [FlashAttention](attention-kernels.en.md) | Tiling and memory traffic for exact attention | Automatically removing attention pairs |
| [PagedAttention](attention-kernels.en.md) | Block-based KV storage across requests | Changing the model's learned attention rule |
| [MLA and sparse attention](latent-and-sparse-attention.en.md) | Compress state per position, or choose which positions to read | Every method performs the same kind of compression |

Hold the task, context length, batch, and output length fixed before comparing memory, time to first token, and subsequent generation speed. “Faster” alone does not explain the benefit.

## Return to a model report

The [model family notes](../model-families/README.en.md) put these components back into specific versions. Keep a short record: **what changed → which bottleneck it addresses → the comparison → the added cost**.

A worked example explains how a component operates; a report's ablation tests whether it helped in that experiment. Both matter, but neither replaces the other.
