# MoE: fine-grained and shared experts

[中文](fine-grained-and-shared.md) · **English**

> Reading time: ~6 min · Last reviewed: 2026-10

## Cut the experts finer

[DeepSeekMoE (2024)](https://arxiv.org/abs/2401.06066) narrows each expert's intermediate width to $1/m$ of the original: $N$ experts become $mN$, and each token activates $mK$ instead of $K$. This preserves the main expert matrix-multiplication budget, not necessarily whole-model runtime.

In the paper's example, choosing 2 of 16 experts gives 120 combinations. With 64 narrower experts and 8 selected per token, the count becomes

$$\binom{64}{8} = 4{,}426{,}165{,}368$$

More combinations offer more ways for experts to specialize, but useful specialization must still be learned. This changes the expert configuration; it does not let us split an already-trained wide expert into narrower ones without affecting its behavior.

## Work through a matched budget

Take hidden size $d=128$ and a SwiGLU expert with intermediate width $f=256$. Ignoring biases, one expert has $3df=98,304$ weights and roughly as many main multiply-accumulates (MACs) per token. A MAC is not one FLOP: counting multiplication and addition separately gives roughly two FLOPs.

| Toy design | Expert intermediate width $f$ | Total experts | Active per token | Total expert parameters | Active parameters / main MACs |
| --- | --- | --- | --- | --- | --- |
| Wider experts | 256 | 8 | 2 | 786,432 | 196,608 |
| Four times as many | 64 | 32 | 8 | 786,432 | 196,608 |
| Make one shared | 64 | 31 routed + 1 shared | 7 routed + 1 shared | 786,432 | 196,608 |

```python
def expert_budget(width, intermediate, routed, top_k, shared=0):
    if min(width, intermediate, routed, top_k) < 1 or shared < 0 or top_k > routed:
        raise ValueError("Invalid expert configuration")
    per_expert = 3 * width * intermediate
    return per_expert * (routed + shared), per_expert * (top_k + shared)

coarse = expert_budget(128, 256, 8, 2)
fine = expert_budget(128, 64, 32, 8)
with_shared = expert_budget(128, 64, 31, 7, shared=1)
assert coarse == fine == with_shared == (786432, 196608)
```

The last row matters: **shared computation is not free**. Keeping eight routed experts active and adding one shared raises the active count to 221,184, an increase of 12.5%. Any improvement then mixes a design change with extra compute.

<!-- widget:step-expert-budget -->

This table excludes routing, activations, attention, and communication. Smaller matrix operations and more dispatches can run slower at equal MACs. Equal total parameters also do not imply equal per-device peak memory.

## Shared experts

Set aside $K_s$ shared experts that **every token passes through**, providing a common computation path intended to reduce redundancy among routed experts. That is a design goal, not a prescribed knowledge partition: grammar is not guaranteed to live in the shared expert, or mathematics in a particular routed expert.

The paper reports DeepSeekMoE 16B matching LLaMA2 7B with about 40% of the compute. DeepSeek-V3 keeps both changes: of its 61 layers the first 3 are dense, and each of the remaining 58 layers has 1 shared expert plus 256 routed experts, 8 chosen per token; each expert's intermediate width is only 2048 (the model's hidden size is 7168).

## Not everyone does this

| Model | Routed experts per layer | Experts per token | Shared expert |
| --- | --- | --- | --- |
| DeepSeek-V3 | 256 | 8 | 1 |
| Original Qwen3-235B-A22B / 30B-A3B | 128 | 8 | none |
| Llama 4 Maverick | 128 | 1 | 1 (MoE layers alternate with dense layers) |
| Llama 4 Scout | 16 | 1 | 1 |
| gpt-oss-120b / 20b | 128 / 32 | 4 | none |
| DeepSeek-V4-Pro (2026) | 384 | 6 | 1 |
| Qwen3.5-397B-A17B / Qwen3.8-2.4T-A95B (2026) | 512 | 10 | 1 |
| Gemma 4 26B A4B (2026) | 128 | 8 | 1 |
| Kimi K3 (2026) | 896 | 16 | 2 (full width) |

The [Qwen3 report](https://arxiv.org/abs/2505.09388) explicitly removes shared experts compared with its Qwen2.5-MoE baseline. Llama 4 Maverick uses one shared expert plus one of 128 routed experts per token. Starting with [Qwen3-Next-80B-A3B](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) in September 2025, Qwen's MoE models brought the shared expert back: it and the 2026 [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) and [Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) all use 512 routed experts plus 1 shared, and [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) uses 256 plus 1, so “Qwen3 has no shared expert” holds only for the original release. Record the checkpoint, not just the family name. Sources for the other 2026 rows are in the [MoE overview's parameter table](README.en.md) (as of October 2026).

## How to choose

For fine-grained experts, match total expert parameters and active compute before comparing validation loss, throughput, and load. For shared experts, state whether their budget replaces routed computation or adds to it.

A larger combination count is not enough. Common tokens might improve while rare domains regress, or communication might slow every step. Architecture makes alternatives available; experiments establish whether they are useful.

Another option narrows the vectors sent to experts, rather than only their internal FFN dimension. [The LatentMoE chapter](latent-moe.en.md) compares the costs with a small example, then explains why Kimi K3 also changes normalization and load balancing.
