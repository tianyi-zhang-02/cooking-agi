# Quantization: FP8 inference and KV-cache quantization

[中文](fp8-and-kv-cache.md) · **English**

> Reading time: ~11 min · Last reviewed: 2026-10

At 128K context, one sequence of this series' 8B teaching model has 16 GiB of BF16 KV, more than the BF16 weights themselves (16 GB, about 14.9 GiB); an 80 GB H100 holds only 3 such sequences. Store the KV in FP8 and it holds 6; with the same 3 sequences, the bytes read per step drop by 38%. **At long context the KV cache has more bytes than the weights, so compressing the KV decides more directly than compressing the weights how many requests you can serve at once and how many bytes each step reads. FP8 is today's most common 8-bit starting point, usable for weights, activations and KV, and whether it is stable depends mostly on how the scales are set. At 4 or 2 bits, KIVI uses per-channel keys and per-token values; that is one method's choice, not a requirement for all KV quantization. Test the chosen layout on the target model and long-context workload.**

The E4M3 and E5M2 grids and block scaling are in [Formats and error](formats-and-error.en.md); the basic KV memory formula is in [KV cache](../../00-foundations/deep-dives/kv-cache-and-inference.en.md).

## FP8 W8A8: weights and activations both in 8 bits

FP8 W8A8 buys two things at once: weight bytes are halved, so decode gets faster (2.54 against 4.77 ms at batch 1 in the previous note's table); and the activations are FP8 too, so matrix multiplications run on FP8 Tensor Cores, doubling the H100 SXM5 dense peak from 989.4 TFLOPS in BF16 to 1,978.9 TFLOPS ([whitepaper V1.04](https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf)), which helps prefill and large batches as well. The precondition is FP8 Tensor Cores, i.e. compute capability 8.9 or higher; the [vLLM v0.31.0 FP8 documentation](https://docs.vllm.ai/en/v0.31.0/features/quantization/llm_compressor/fp8/) states that on Turing and Ampere the same FP8 model can only run as W8A16 with Marlin kernels, which saves weight memory without raising compute.

The common way to set the scales is the `FP8_DYNAMIC` scheme that this documentation recommends: one static scale per output channel for the weights, and dynamic per-token scales for the activations, with no calibration data needed. Another, online mode, `fp8_per_tensor`, gives the weights one per-tensor scale and computes the activations' min and max on every forward pass; the documentation says latency improvements are limited in this mode. It self-reports that FP8 halves model memory and improves throughput by up to 1.6×, without stating the hardware or workload.

On quality, the FP8 W8A8 in [Kurtic et al. (2024-11, ACL 2025)](https://arxiv.org/abs/2411.02355) uses only RTN: symmetric per-output-channel weights and dynamic per-token activations, with no calibration data. On Llama-3.1 8B, 70B and 405B, the average recovery on Open LLM Leaderboard v2 (quantized score divided by BF16 score) is 101.2%, 100.0% and 99.9%, which the paper calls near lossless; in the same paper a well-tuned INT8 W8A8 loses 1–3%.

### Dynamic scales or static scales

The E4M3 maximum is 448. Two tokens' activations (teaching numbers): A is $[0.01,\ -0.8,\ 3.2]$ and B is $[2.0,\ -45,\ 180]$.

- **Dynamic, per token**: A's scale is $3.2/448\approx0.00714$ and B's is $180/448\approx0.402$. Each token's maximum lands exactly on 448, and all three values read back accurately; A's 0.01 reads back as 0.0098.
- **Static, per tensor**: suppose the largest value seen during calibration was 120, so $s=120/448\approx0.268$. A is fine: 0.01 reads back as 0.0105, a 4.6% error, since FP8's exponent bits keep relative precision for small values. B's 180 divided by $s$ is 672, above 448; in saturating mode it is clamped to 448 and reads back as 120, a third short.

The cost of dynamic scales is a maximum over each token on every forward pass; static scales cost nothing at run time but clip any input more extreme than the calibration set. What clipping does also depends on configuration: OCP FP8 conversion requires support for both a saturating and an overflow mode, and in overflow mode an out-of-range E4M3 value becomes NaN (Table 3 of the [OCP MX specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf), reproduced there). In [SmoothQuant](https://arxiv.org/abs/2211.10438)'s measurements, static per-tensor O3 was faster than dynamic O1, so they recommend the coarser granularity whenever accuracy allows.

## Training in FP8 or lower: the extra problems

Inference only runs the forward pass; training must also keep gradients and weight updates correct at low precision, which adds several problems. This section only lists each group's approach; the example of small BF16 updates being rounded away is in [Precision and memory](../../00-foundations/deep-dives/precision-and-memory.en.md#rounded-away-update), and the principle is the same.

| Approach | Source | How scales are set | What stays out of low precision | Self-reported result |
| --- | --- | --- | --- | --- |
| FP8 delayed scaling | [Transformer Engine 2.18 docs](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.18/user-guide/examples/fp8_primer.html) | the scale comes from the largest absolute values of previous iterations (amax history); the HYBRID format uses E4M3 forward, E5M2 backward | decided by the framework | — |
| Fine-grained FP8 | [DeepSeek-V3 (2024-12)](https://arxiv.org/abs/2412.19437) §3.3 | one scale per 1×128 activation tile and per 128×128 weight block, from the current maximum computed online; E4M3 everywhere | embedding, output head, MoE gating, normalization, attention; master weights and gradients in FP32, AdamW moments in BF16 | on validation models of about 16B and 230B, loss stays within 0.25% relative error of the BF16 baseline; then used for the 671B MoE |
| MXFP8 | [Mishra et al. (NVIDIA, 2025-05)](https://arxiv.org/abs/2506.08027) | one E8M0 scale per 32 values, exponent rounded up; E4M3 for weights, activations and activation gradients | attention matrix multiplications and softmax, embedding, output layer | up to 8B and 15T tokens (MXFP8 simulated on Hopper), validation perplexity within 0.5% of BF16 |
| NVFP4 | [NVIDIA (2025-09)](https://arxiv.org/abs/2509.25149), TE 2.18 | one E4M3 scale per 16 values plus a tensor-level FP32 scale; weights in 16×16 2D blocks; stochastic rounding for gradients; random Hadamard transforms to tame outliers, applied only in the weight-gradient computation | about 16% of linear layers (the first 2 and last 8 blocks) stay in BF16; TE recommends MXFP8 for the last few layers but does not apply it automatically | 12B (hybrid Mamba-Transformer) on 10T tokens, loss and downstream accuracy comparable to an FP8 baseline |

Two details deserve their own sentence. First, accumulation precision. DeepSeek-V3 found that FP8 GEMMs on the H800 keep only about 14 bits in accumulation, giving a maximum relative error of nearly 2% in a test with random matrices at $K=4096$; every 128 accumulated elements (equivalent to 4 WGMMA instructions), they move the partial sum to CUDA cores and accumulate it in FP32. Second, transposes. The TE documentation explains that Blackwell Tensor Cores need MXFP8 data contiguous along the reduction dimension, while the backward pass uses the transposed matrix, and transposing MXFP8 data amounts to requantizing it, so TE quantizes a regular and a transposed copy separately from the original high-precision tensor. NVFP4 weights use 16×16 2D blocks for a related reason: so a tensor and its transpose share the same scales.

## KV cache: the real bulk at long context

Teaching configuration: 32 layers, 8 KV heads, head dimension 128. Each token's K and V together hold $2\times32\times8\times128=65{,}536$ values:

| KV format | Average bits per value | Per token | One 128K-token sequence |
| --- | ---: | ---: | ---: |
| BF16 | 16 | 128 KiB | 16 GiB |
| FP8, per-tensor scale | 8 | 64 KiB | 8 GiB |
| INT4, groups of 32, FP16 scale and zero-point | 5 | 40 KiB | 5 GiB |
| INT2, same grouping | 3 | 24 KiB | 3 GiB |

The INT4 and INT2 rows are estimated with [KIVI](https://arxiv.org/abs/2402.02750)'s grouping (group length 32), each group storing one FP16 scale and one FP16 zero-point, which adds $32/32=1$ bit per value on average; KIVI also keeps the most recent 128 tokens in full precision; that part is only 16 MiB in BF16 and is not counted in the table.

Fit them into the 80 GB of an H100 SXM5, reserving 6 GB for the runtime (a teaching assumption), and count how many 128K sequences fit:

| Weights | BF16 KV | FP8 KV | INT4 KV |
| --- | ---: | ---: | ---: |
| BF16 (16 GB) | 3 sequences | 6 sequences | 10 sequences |
| FP8 (8.5 GB, output layer BF16) | 3 sequences | 7 sequences | 12 sequences |

At batch 3 with 128K each, every decode step reads the weights plus 3 copies of KV. The floor at 3,352 GB/s:

| Weights + KV | Bytes read per step | Per step, at least | KV share of bytes read |
| --- | ---: | ---: | ---: |
| BF16 + BF16 | 67.5 GB | 20.1 ms | 76% |
| FP8 + BF16 | 60.0 GB | 17.9 ms | 86% |
| BF16 + FP8 | 41.8 GB | 12.5 ms | 62% |
| FP8 + FP8 | 34.3 GB | 10.2 ms | 75% |
| BF16 + INT4 | 32.1 GB | 9.6 ms | 50% |

At this length, compressing only the weights goes from 20.1 ms to 17.9 ms, while compressing only the KV goes to 12.5 ms: **compress whatever holds the bulk of the bytes first.** During decode the attention part has an arithmetic intensity that does not depend on the batch, only a few operations per byte, see [Performance](../performance.en.md#decode-batching), so almost every byte saved turns into time saved. At short context the conclusion flips: at 2K tokens one sequence's BF16 KV is only 0.27 GB, and the weights are the bulk.

### Why keys and values are quantized differently

KIVI (2024-02, ICML 2024) measured the KV cache's distributions: the keys have a few fixed channels with very large magnitudes, the values have no clear outlier pattern, and the attention output is mainly a weighted sum of the values of a few important tokens, so quantizing values per token keeps each error within its own token. Their scheme is therefore **keys per channel, values per token**.

The axes are not universal. [QuaRot, §5](https://arxiv.org/html/2404.00456v2#S5) rotates the keys before 4-bit quantization, then groups channels within each token and head rather than sharing a per-channel scale across tokens. The right layout depends on the distribution after any transforms, and on the kernel that will consume it. The example below explains KIVI's choice for keys with channel outliers; it does not establish a rule for every model.

**Worked once.** A key with 4 tokens and 4 channels, channel 3 being the outlier channel (teaching numbers); the query is $q=[1,\ 1,\ 0.1,\ 1]$; asymmetric 2 bits, 4 grid points per group. This example uses a real-valued minimum offset: $s=(x_{\max}-x_{\min})/3$, $\hat{x}=x_{\min}+s\,\operatorname{clamp}(\operatorname{round}((x-x_{\min})/s),0,3)$. Unlike the integer zero-point quantizer in [Formats and error](formats-and-error.en.md), it does not require zero to be exactly representable. Keep this convention fixed when comparing the two grouping layouts.

$$
K=\begin{bmatrix}0.4&-0.2&9.0&0.1\\-0.3&0.5&11.0&-0.4\\0.2&-0.6&10.0&0.3\\0.5&0.1&12.0&-0.2\end{bmatrix},\qquad Kq=\begin{bmatrix}1.2\\0.9\\0.9\\1.6\end{bmatrix}.
$$

- **Per token**: row 1 has minimum −0.2 and maximum 9.0, a step of $9.2/3\approx3.07$ and grid points $-0.2,\ 2.87,\ 5.93,\ 9.0$, so all three ordinary channels become −0.2. The four scores become $[0.3,\ -0.1,\ -0.8,\ 0.6]$; tokens 2 and 3, which had equal scores, are pulled apart, and the largest error is 1.7.
- **Per channel**: column 1, $[0.4,-0.3,0.2,0.5]$, is its own group with step 0.267; the outlier channel is its own group too. The scores become $[1.23,\ 0.90,\ 0.93,\ 1.67]$, with a largest error of 0.067.

The relative error of $K$ itself (Frobenius norm) is 8.6% against 0.6%, neither of which looks large, yet per token the scores are almost all wrong: the outlier channel inflates the norm while the query mostly looks at the other channels. **To measure the error of KV quantization, look at the attention scores and outputs, not only at the relative error of K.**

KIVI's measurements (Llama-2-7B, Table 3): GSM8K is 13.50 at 16 bits; with simulated quantization of all tokens to 2 bits, it is 5.76 with keys per channel and values per token, and 0.83 with both per token; full KIVI-2 (group length 32, most recent 128 tokens in full precision) reaches 12.74. The LongBench average is 44.52 at 16 bits and 44.27 for KIVI-2 (Table 4). The paper self-reports (Llama-2-7B on one A100 80GB) 2.6× less peak memory including weights, batches up to 4× larger and 2.35–3.47× throughput. [KVQuant (2024-01, NeurIPS 2024)](https://arxiv.org/abs/2401.18079) also quantizes keys per channel, and quantizes keys before RoPE to reduce the positional embedding's effect on quantization, adding per-layer non-uniform grids and per-vector separation of outliers; it self-reports less than 0.1 perplexity loss on WikiText-2 and C4 at 3 bits, and LLaMA-7B running 1M tokens of context on one A100-80GB and 10M on 8 GPUs.

### Turning it on in an engine

As of October 2026, the [Quantized KV Cache documentation](https://docs.vllm.ai/en/v0.31.0/features/quantization/quantized_kvcache.html) for vLLM v0.31.0 lists `kv_cache_dtype` values `auto`, `fp8`, `fp8_e4m3` (CUDA 11.8+ and ROCm) and `fp8_e5m2` (CUDA 11.8+), and lists no INT4 or INT2 (the API reference shows an INT4 KV attention kernel module, but the feature documentation gives no usage). The main points:

- Without calibration, **all scales are set to 1.0**; the documentation recommends calibrating on data with llm-compressor.
- Scales can be one per tensor (one each for Q, K and V) or one per attention head; the latter is only available on the Flash Attention backend and requires calibration.
- With the Flash Attention 3 backend, attention itself runs in FP8, and the queries are quantized to FP8 too.
- The documentation warns that layer types such as sliding-window attention are more sensitive to KV quantization, and `--kv-cache-dtype-skip-layers` keeps those layers at the original precision.

What a scale of 1.0 means can be worked out from E4M3 (this is this note's reasoning, not a statement in the documentation): K and V values beyond 448 in magnitude are clipped; the smallest subnormal is $2^{-9}\approx0.00195$, so 0.001 rounds up to it, an error of almost 100%, and anything a little smaller becomes 0. So whether uncalibrated FP8 KV is safe depends on the range this model's K and V fall in; measure that before deciding.

## Compared with other ways of saving KV

| Approach | What it saves | Does the model change? | Cost |
| --- | --- | --- | --- |
| KV quantization | bits per value | no; it can be done after training | error at long context; needs kernel support |
| GQA / MQA | the number of KV heads | yes (trained from scratch or further trained) | the representation changes; heads cannot simply be deleted |
| MLA | compresses KV into a smaller latent vector | yes | more complex implementation, see [MLA and sparse attention](../../00-foundations/deep-dives/latent-and-sparse-attention.en.md) |
| Sliding window | keeps only the most recent positions | decided by the architecture | information outside the window is not visible |
| Paging, prefix sharing, offload | fragmentation and duplicate storage, or moving KV out of GPU memory | no | management and transfer overhead, see [vLLM and SGLang](../../06-systems/inference-frameworks.en.md) |

These multiply: going from 32 KV heads to 8 and storing in FP8 leaves $\tfrac{8}{32}\times\tfrac12=\tfrac18$ of the MHA BF16 KV. The GQA arithmetic is in [KV cache](../../00-foundations/deep-dives/kv-cache-and-inference.en.md).

## Where these numbers stop

- The engine documentation gives no accuracy numbers for FP8 KV; the KIVI and KVQuant results come from Llama-2-era models and are self-reported by the authors. A different model or context length needs a new measurement, on long-context tasks, see [How to measure what was lost](checking-quality.en.md).
- The times above are floors of bytes divided by peak bandwidth; they leave out attention compute, kernel efficiency and paging fragmentation.
- The training section only lists each group's published approach and self-reported results; the recipes have not been compared against each other on the same model.
