# Quantization: what fewer bits buy, and what they cost

[中文](README.md) · **English**

> Reading time: ~7 min · Draft, pending review

A 70B-parameter dense model has 140 GB of BF16 weights, which does not fit on one 80 GB GPU; even when it fits, generating one token at a time at batch 1 means reading every weight from memory at every step. Quantization stores these numbers in fewer bits. You can compress the weights, the activations or the KV cache, and each buys something different. **Find the bottleneck first: if the model does not fit, or decode is limited by memory reads, compress the weights; if prefill and large batches are limited by compute, compress the activations as well, so the lower-precision Tensor Cores can be used; if the KV cache dominates at long context, compress the KV cache. Whichever you compress, compare the two models item by item on the same set of questions, and look specifically at the kinds of task that break first; only then do you know what you lost.**

> **Draft, pending review.** The site owner has not reviewed this note yet; facts, emphasis and wording may change. Check the primary sources linked in the text before relying on it.

## First, the ledger: a 70B model and one 80 GB GPU

These are **teaching assumptions**: a 70B dense model with 80 layers, 8 KV heads and head dimension 128, so each token's BF16 KV takes $2\times80\times8\times128\times2=327{,}680$ bytes (320 KiB); a GPU with the 80 GB of memory and 3,352 GB/s bandwidth of an H100 SXM5 ([H100 architecture whitepaper V1.04](https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf)); 4 GB reserved for the runtime; all weights stored in the same format, with no separate accounting for embeddings, the output layer or metadata beyond the scales.

| Weight format | Per parameter | Weights | Left for KV | Tokens of KV that fit (BF16 / FP8 KV) | Batch-1 time per token, at least |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 16 bit | 140 GB | does not fit | — | not possible on one GPU |
| FP8 | 8 bit | 70 GB | 6 GB | about 18,000 / 37,000 | 20.9 ms |
| INT4, one group per 128 weights | 4.125 bit | 36.1 GB | 39.9 GB | about 122,000 / 244,000 | 10.8 ms |

The 4.125 bits are a 4-bit integer plus one 16-bit scale shared by every 128 weights ($16/128=0.125$). The last column is weight bytes divided by bandwidth, assuming the bandwidth is used at 100%, which real kernels do not reach; with two GPUs, the BF16 floor is also 20.9 ms, see [NVIDIA generations](../nvidia-generations.en.md). This table only says whether the model fits and how fast it could possibly go. **It says nothing about quality**, which is the second half of this series.

## Three things can be compressed, and each buys something different

How to read the names: W4A16 means 4-bit weights and 16-bit activations; W8A8 means both at 8 bits; g128 means one scale per 128 values; KV8 or FP8 KV means the KV cache is stored in 8 bits.

| What you compress | Common names | What it buys | What it needs | Where it usually breaks first |
| --- | --- | --- | --- | --- |
| Weights only | W4A16, W8A16 (GPTQ, AWQ) | Weight memory; bytes read per decode step | A kernel that dequantizes as it reads; GPTQ and AWQ also need a small calibration set | The advantage disappears at large batch; error grows sharply at 3 bits and below |
| Weights and activations | W8A8 (INT8, FP8), W4A4 (NVFP4, MXFP4) | Both of the above, plus the compute of lower-precision Tensor Cores | Hardware support for the format; a way to set activation scales | Outliers in the activations; INT8 often needs preprocessing such as SmoothQuant |
| KV cache | FP8 KV, INT4 / INT2 KV | KV memory and bytes read per step at long context | Attention-kernel support; separate scales for K and V | Long-context tasks; fixed outlier channels in the keys |

Which one to pick depends on which side of the roofline you are on ([Performance](../performance.en.md#decode-batching)). Batch-1 decode does about 1 operation per byte read, far to the left of the H100 BF16 ridge point (about 295 FLOP/byte), so reading fewer bytes makes it faster directly. Once the batch is large enough to be compute-bound, W4A16 still computes on BF16 Tensor Cores and its advantage disappears, whereas FP8 in W8A8 raises the dense peak from 989.4 to 1,978.9 TFLOPS (same whitepaper).

## The teaching example used in this series

The later notes use one **teaching model**: an 8B dense model with 32 layers, 32 query heads, 8 KV heads and head dimension 128, the same assumed configuration as in the [KV cache](../../00-foundations/deep-dives/kv-cache-and-inference.en.md) and [Performance](../performance.en.md) notes; the GPU is still worked out by hand from the public H100 SXM5 specifications. What each note covers is in the reading order at the end.

## Timeline

| Date | Work | In one line |
| --- | --- | --- |
| 2017-12 | [Jacob et al.](https://arxiv.org/abs/1712.05877) (Google) | Scale and zero-point mapping for integer inference; the zero-point guarantees that the real value 0 is represented exactly |
| 2021-06 | [Nagel et al., quantization white paper](https://arxiv.org/abs/2106.08295) | Post-training quantization (PTQ) and quantization-aware training (QAT) pipelines; in most cases PTQ alone reaches 8 bits close to floating point |
| 2022-08 | [LLM.int8()](https://arxiv.org/abs/2208.07339) (NeurIPS 2022) | Systematic outlier features appear around 6.7B; about 0.1% of dimensions are multiplied in 16 bits instead |
| 2022-09 | [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) | Proposes the E4M3 and E5M2 FP8 encodings |
| 2022-10 | [GPTQ](https://arxiv.org/abs/2210.17323) (ICLR 2023) | Uses second-order information to push quantization error onto weights not yet quantized; a 175B model goes to 3–4 bits in about four GPU hours |
| 2022-11 | [SmoothQuant](https://arxiv.org/abs/2211.10438) (ICML 2023) | Moves activation outliers onto the weights with an equivalent transformation, enabling W8A8 |
| 2023-06 | [AWQ](https://arxiv.org/abs/2306.00978) (MLSys 2024 best paper) | Finds important channels by activation size, scales them up, then quantizes |
| 2023-09 | [OCP MX specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) | Authors from AMD, Arm, Intel, Meta, Microsoft, NVIDIA and Qualcomm define it: every 32 values share one power-of-two scale |
| 2024-02 | [KIVI](https://arxiv.org/abs/2402.02750) (ICML 2024) | Keys per channel, values per token; the KV cache goes down to 2 bits |
| 2024-03 | [QuaRot](https://arxiv.org/abs/2404.00456) | Rotate first, then quantize; weights, activations and KV cache all at 4 bits |
| 2024-07 | [Accuracy is Not All You Need](https://arxiv.org/abs/2407.09141) (NeurIPS 2024) | Accuracy barely changes, yet many individual answers flip between right and wrong |
| 2024-08 | [Marlin](https://arxiv.org/abs/2408.11743) | A kernel for 4-bit weights times FP16 activations; close to the ideal 4x for large matrices on an A10 up to batch 16–32 |
| 2024-11 | ["Give Me BF16 or Give Me Death"?](https://arxiv.org/abs/2411.02355) (ACL 2025) | More than 500,000 evaluations on Llama-3.1: FP8 is near lossless, INT8 loses 1–3%, W4A16 is comparable to 8 bits |
| 2024-12 | [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | A 671B-parameter MoE trained with fine-grained FP8 |
| 2025-05 | [MXFP8 training recipes](https://arxiv.org/abs/2506.08027) (NVIDIA) | Models up to 8B and 15T tokens, with MXFP8 compute simulated on Hopper; self-reported validation perplexity within 0.5% of BF16 |
| 2025-05 | [Quantization on long-context tasks](https://arxiv.org/abs/2505.20276) (EMNLP 2025) | 8 bits lose at most about 0.8 points on average; 4 bits lose up to 59 points on long-input tasks |
| 2025-06 | [NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) | One E4M3 scale per 16 values, multiplied by one tensor-level FP32 scale |
| 2025-09 | [NVFP4 pretraining](https://arxiv.org/abs/2509.25149) | A 12B model on 10T tokens; self-reported loss and downstream accuracy comparable to an FP8 baseline |

## What is fairly settled, and what is still moving

Each cell corresponds to a dated public record; the third column says what kind of result would change the assessment, not a prediction.

| Fairly settled | Still moving | Signal to watch next |
| --- | --- | --- |
| 8-bit weights are essentially lossless on large models: Kurtic et al. (2024-11) measured FP8 W8A8 as near lossless on Llama-3.1 8B to 405B; Dutta et al. measured GPTQ W8A16 on MMLU with only 0.26%–1.40% of answers flipped | How to go to 4 bits and below: besides GPTQ and AWQ, there are rotations (QuaRot) and hardware block formats such as NVFP4 and MXFP4 | 4-bit results compared item by item on long-context, multilingual and reasoning tasks |
| At small batch, W4A16 mainly buys decode speed, and the advantage disappears at large batch: Marlin (2024-08), deployment advice in Kurtic et al. | Inference quality of W4A4: NVIDIA self-reports that DeepSeek-R1-0528 converted from FP8 to NVFP4 loses at most 1 percentage point on the 7 listed evaluations ([NVFP4 blog post](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)) | Third-party W4A4 evaluations reported by slice |
| Average accuracy hides item-level changes (Dutta et al., 2024-07); long context, non-English and hard tasks such as math break first (Mekala et al. 2025-05, Marchisio et al. 2024-07) | Low-precision training: FP8 has been used at 671B (DeepSeek-V3), MXFP8 up to 8B / 15T (simulated), NVFP4 up to 12B / 10T | FP4 training at larger scale with a fully public recipe |
| Hardware decides whether you actually get a speedup: FP8 Tensor Cores need compute capability 8.9 or higher, and 9.0 (Hopper) has no INT4 Tensor Core input ([CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)) | Trade-offs in new hardware: on the [HGX page](https://www.nvidia.com/en-us/data-center/hgx/) as of October 2026 (8 GPUs, sparse figures, dense is half), HGX B300 has only 3 POPS of INT8 against 72 PFLOPS of FP8, while HGX B200 has 72 for both; AMD's [MI350 series](https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi350.html) adds MX formats | The INT8, FP8 and FP4 throughput in the next generation's datasheets |

## How to read this series

These help first: [Precision and memory](../../00-foundations/deep-dives/precision-and-memory.en.md) (the FP16 versus BF16 trade-off), [GPU architecture](../gpu-architecture.en.md) (the data types each Tensor Core generation supports, and a hand-worked NVFP4 example), [Performance](../performance.en.md) (the roofline and the batch ledger for decode), [KV cache](../../00-foundations/deep-dives/kv-cache-and-inference.en.md) (the KV memory budget). Neighboring notes: [NVIDIA generations](../nvidia-generations.en.md) covers how many GPUs of each generation a 70B model needs; [vLLM and SGLang](../../06-systems/inference-frameworks.en.md) covers which layer of a generation engine quantization lives in; [LoRA and QLoRA](../../05-post-training/lora-and-qlora.en.md) covers NF4 and training on a 4-bit base; [The score improved. What does that actually establish?](../../07-evaluation/metric-robustness.en.md) covers the statistics of paired comparisons.

1. [Formats and error](formats-and-error.en.md): scale and zero-point, per-tensor, per-channel and per-group, INT versus FP grids, block-scaled MXFP4 and NVFP4, why outliers do so much damage, and which hardware generation computes which format
2. [Weight-only quantization and post-training quantization](weight-only-and-ptq.en.md): why W4A16 speeds up decode and at what batch it stops helping, GPTQ's error compensation, AWQ's channel scaling, SmoothQuant's W8A8
3. [FP8 inference and KV-cache quantization](fp8-and-kv-cache.en.md): how FP8 scales are set, what FP8 / MXFP8 / NVFP4 training has to handle, what KV quantization saves at 128K context, and why KIVI chooses per-channel key quantization
4. [How to measure what was lost](checking-quality.en.md): perplexity, KL divergence, task accuracy and flipped answers, paired comparison, long-context and multilingual slices, what to record for every quantization, common failures
5. [Review](review.en.md): interview questions and a self-check
