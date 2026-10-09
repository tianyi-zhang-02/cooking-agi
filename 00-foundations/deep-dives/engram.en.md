# Engram: looking up recurring local patterns

[中文](engram.md) · **English**

Recognizing a familiar phrase and interpreting it in context are related but different jobs. Engram adds a learned lookup module for local patterns, then uses context to decide how much of the retrieved information to use. It is not a personal chat-history store or another name for RAG.

## What happens during a lookup?

```text
Recent tokens ─→ normalized IDs ─→ multiple n-gram hashes ─→ embedding tables
                                                               ↓ concatenate
Current hidden state ─→ match retrieved key ─→ gate × value ─→ local conv ─→ residual
```

A token-ID normalization map lets some surface variants share lookup space. This branch does not mean replacing every token in the main model input. Suffix n-grams of different lengths are hashed into multiple tables, and their embeddings are concatenated into $e_t$. The table vectors are learned parameters, not manually written answers.

## Why consult a table when context is available?

It separates some of the work: recurrent local patterns use deterministic addressing, while context-sensitive interpretation and long-distance relations remain model computations. Retrieving the same local pattern does not require using it identically.

For example, `the bank` could occur in a river or finance context. The local lookup can be identical while the hidden states differ. A gate can use that context to modulate the same entry. This is an explanatory example, not a reported experiment.

A simplified single-branch form is:

$$
k_t=W_Ke_t,\quad v_t=W_Ve_t,\qquad
\alpha_t=\sigma\left(\frac{\operatorname{RMSNorm}(h_t)^\top
\operatorname{RMSNorm}(k_t)}{\sqrt d}\right),\quad u_t=\alpha_tv_t.
$$

A causal depthwise convolution adds local interaction while retaining the $u_t$ path:

$$
y_t=u_t+\operatorname{SiLU}\big(\operatorname{Conv}_{\rm causal}(\operatorname{RMSNorm}(u))_t\big).
$$

The result enters the residual stream. This notation omits the paper's multi-branch residual details and is not a complete checkpoint implementation. The convolution must be causal; otherwise training can leak future tokens.

## What does a hash collision look like?

Use the teaching hash $(3a+b)\bmod m$ for a bigram. This is not the authors' hashing scheme.

```python
def bigram_bucket(tokens, table_size):
    if len(tokens) != 2 or type(table_size) is not int or table_size <= 0:
        raise ValueError("Expected two IDs and a positive table size")
    if any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("Expected nonnegative integer token IDs")
    return (3 * tokens[0] + tokens[1]) % table_size

assert bigram_bucket([1, 2], 5) == bigram_bucket([2, 4], 5) == 0
assert bigram_bucket([1, 2], 7) == 5
assert bigram_bucket([2, 4], 7) == 3
```

Two n-grams collide in one table but separate in another. Concatenating several lookups reduces dependence on any single collision; it does not guarantee collision-free addressing. Updating a shared entry also affects other n-grams mapped to it.

## Reduced computation still uses resources

Consider one hypothetical table: one million entries, 64 dimensions, two bytes per BF16 value.

| Quantity | Calculation | Result |
| --- | --- | --- |
| Complete parameter table | $10^6\times64\times2$ | 128 MB, about 122 MiB |
| Two reads per token | $2\times64\times2$ | 256 bytes |
| Raw reads at 100K tokens/s | $10^5\times256$ | 25.6 MB/s |

This is not a deployment estimate. Multiple layers, n-gram lengths, hash heads, addressing, transfers, and training state all add costs. Random-access effective bandwidth also differs from sequential bandwidth.

Deterministic addresses allow precomputation and prefetching. The tradeoff is possible random-read or CPU/GPU transfer bottlenecks. Having many parameters but reading only a few per step does not make additional knowledge free.

## How would we establish that it helps?

First test prefix invariance: changing future tokens must not affect earlier lookups or outputs. Save normalization, hashing versions, vocabulary, and table weights with the checkpoint, not just the main model.

Compare matched training budgets, separating frequent phrases, rare combinations, contextual disambiguation, and long-distance tasks. Ablate the lookup branch and inspect gate saturation. Selected examples of remembered phrases are not enough.

| Misinterpretation | More accurate description |
| --- | --- |
| Automatically remembers a new user | The table contains learned parameters, not per-user writable records |
| Provides a source quotation | Embeddings are not traceable documents; retrieval and citations need extra machinery |
| Always suppresses incorrect entries | Gates can fail; ambiguity and collision tests remain necessary |
| Replaces attention | It adds local-pattern memory, not every long-distance relationship |

Checked 2026-10-08 against the [Engram paper](https://arxiv.org/abs/2601.07372) and [authors' repository](https://github.com/deepseek-ai/Engram). This page tests hashing and resource arithmetic, not full-model training.
