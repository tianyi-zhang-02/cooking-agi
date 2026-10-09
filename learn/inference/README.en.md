# Inference: from the next token to a complete request

[中文](README.md) · **English**

“The model is slow” is not specific enough. The input may be long, caches may fill memory, or requests may wait for one another. Locate the delay before choosing an optimization.

## Follow one request

```text
input text → token IDs → prefill → first token
                                      ↓
                   update KV cache ← decode → sample next token
                                      ↓
                              stop and release state
```

This simplified autoregressive path does not prescribe a scheduler. Prefill processes the existing context; decode uses cached state to continue generation. Sampling decides which token to select, while caching decides which state to retain and read. They solve different problems.

## Similar names, different interventions

| Method | Main intervention | What does not follow |
| --- | --- | --- |
| Temperature / top-p | Token selection from the output distribution | More randomness does not imply greater capability |
| KV cache | Reuse of prefix keys / values | Caching does not remove subsequent reads of history |
| MQA / GQA | Number and sharing of KV heads | Memory reduction is not end-to-end speedup |
| RoPE / YaRN | Position representation and long-context handling | Accepting a long input is not using its evidence well |
| [NoPE and sequence order](../../00-foundations/deep-dives/nope-and-order.en.md) | How causal attention and recurrent state carry order information | Removing RoPE from an existing checkpoint is a different experiment |
| FlashAttention | Attention-related memory traffic | Dense attention pairs do not disappear |
| PagedAttention | Physical allocation and sharing of KV state | It does not make attention sparse |
| MLA | Representation stored for each position | Compressing state differs from reading fewer positions |
| Sparse attention | Query–key pairs included in computation | Omitted pairs are not necessarily unimportant |

Start with generation and caching below, then move to context and efficient attention. If you already understand decoders, find the particular mechanism you need to distinguish.

## Chapter contents

Once decode and caching are clear, [DFlash and MTP](../../00-foundations/deep-dives/dflash.en.md) examines reducing target-model rounds: follow drafting and verification, then rejection, cache rollback, and costs. Beginners do not need every acceleration method first.

<!-- widget:study-atlas -->

## A small performance diagnosis

Suppose a request waits 800 ms for its first token, then receives a token every 20 ms. These intervals answer different questions. The first can include queuing, prefill, and the first generation step; the second is closer to ongoing decode. Reporting only average tokens/s hides the distinction.

If inputs grow longer, inspect prefill and cache reads. If concurrency delays the first token, inspect queuing, scheduling, and resource contention. This example illustrates measurement decomposition, not proof of a particular bottleneck.

| Hold fixed when comparing | Report alongside it |
| --- | --- |
| Model, weights, tokenizer, precision | Output quality and numerical tolerance |
| Input/output length distributions, concurrency | Time to first token, inter-token intervals, end-to-end latency |
| Hardware, batch, cache state | Useful throughput, peak memory, tail latency |

Training-side memory belongs in [pretraining and training mechanics](../pretraining/README.en.md), rather than an inference-cache checklist. Version-specific claims still need the model report and implementation.
