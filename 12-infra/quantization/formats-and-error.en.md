# Quantization: where the error comes from when numbers are squeezed onto a grid

[中文](formats-and-error.md) · **English**

> Reading time: ~11 min · Last reviewed: 2026-10

Take a group of 8 weights and compress them to INT4 with one shared scale: the mean squared error (MSE) is 0.0042. Replace the value 1.40 with 14.0 and the other 7 values all become 0. Why does changing one value affect all the others?

In this example, quantization error depends on the step between grid points, which is set by the largest value in the group sharing a scale. The format determines how the grid is laid out: evenly spaced for integers, proportional for floating point. Granularity determines which values share a scale. An outlier increases the step for its whole group, increasing the rounding error for the other values too.

The bit layouts and rounding of FP16 and BF16 are in [Precision and memory](../../00-foundations/deep-dives/precision-and-memory.en.md); the table of formats each Tensor Core generation supports and a small NVFP4 example are in [GPU architecture](../gpu-architecture.en.md). They are not repeated here. Except where a source is given, the numbers in this note are teaching examples you can work out by hand.

## How a real number is stored as an integer: scale and zero-point

The most common way to store a real number $x$ as a $b$-bit integer $q$ is an affine mapping:

$$
q=\operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{x}{s}\right)+z,\ q_{\min},\ q_{\max}\right),\qquad \hat x=s\,(q-z).
$$

$s$ is the scale, the step between two neighboring grid points; $z$ is the zero-point, the integer that corresponds to the real value 0; $\hat x$ is the approximation you read back. [Jacob et al. (Google, 2017-12)](https://arxiv.org/abs/1712.05877) define $z$ this way so that 0 is represented exactly, because many operators pad with zeros at the borders. Two common choices:

- **Symmetric**: $z=0$, INT4 uses $-7$ to $7$ (leaving out $-8$ to keep positive and negative symmetric), $s=\max|x|/7$ for a nonzero tensor.
- **Asymmetric**: use unsigned 4-bit codes from $0$ to $15$. First include zero in the calibration interval: $a=\min(0,\min x)$, $b=\max(0,\max x)$. Then $s=(b-a)/15$, $z=\operatorname{clamp}(\operatorname{round}(-a/s),0,15)$. The zero-point is now a valid code, so real zero is reconstructed exactly.

For an all-zero tensor, either scheme can use $s=1$, $z=0$, with all codes zero, avoiding division by zero. A constant nonzero tensor also works with the extended asymmetric interval. For example, calibration values $[1,2]$ need the interval $[0,2]$: $s=2/15$, $z=0$. Using $[1,2]$ directly would give $z=-15$; quantizing real zero would then clamp its code to 0 and reconstruct **1**, not 0.

As long as $x$ is in range, the rounding error is at most $s/2$; if the error is uniform on $[-s/2,s/2]$, the MSE is about $s^2/12$. **Halve the step and the squared error drops to a quarter.**

Consider six non-negative numbers, $[0,\ 0.2,\ 0.5,\ 0.9,\ 1.3,\ 2.0]$, such as values after an activation function. With symmetric quantization, $s=2.0/7\approx0.286$: the codes use only 0 to 7, leaving the negative half of the grid unused, and the MSE is about 0.00514. With asymmetric quantization, $s=2.0/15\approx0.133$ and $z=0$: the codes are $[0,2,4,7,10,15]$, and the MSE is about 0.00130. That is roughly a fourfold reduction, close to the squared ratio of steps $(0.286/0.133)^2\approx4.6$; with only 6 numbers, $s^2/12$ is only an approximation.

## Eight weights by hand: how much error does grouping save?

Teaching weights $w=[0.12,\ -0.53,\ 0.88,\ -0.07,\ 0.31,\ 1.40,\ -0.95,\ 0.02]$, symmetric INT4.

- **per-tensor**: all 8 share one scale, $s=1.40/7=0.2$.
- **per-group, 4 per group**: the largest magnitude among the first 4 is 0.88, so $s=0.88/7\approx0.1257$; the last 4 keep $s=0.2$.

| Original | per-tensor code | Read back | per-group code | Read back |
| ---: | ---: | ---: | ---: | ---: |
| 0.12 | 1 | 0.2 | 1 | 0.1257 |
| −0.53 | −3 | −0.6 | −4 | −0.5029 |
| 0.88 | 4 | 0.8 | 7 | 0.88 |
| −0.07 | 0 | 0 | −1 | −0.1257 |
| 0.31 | 2 | 0.4 | 2 | 0.4 |
| 1.40 | 7 | 1.4 | 7 | 1.4 |
| −0.95 | −5 | −1.0 | −5 | −1.0 |
| 0.02 | 0 | 0 | 0 | 0 |

The per-tensor MSE is 0.0042, the same order as $s^2/12=0.0033$; with groups it drops to 0.00186. The price is one more stored scale: in practice the group length is often 128, with one 16-bit scale per group, so each weight costs $16/128=0.125$ extra bits on average.

### Swap in one outlier

Replace 1.40 with 14.0. The per-tensor scale becomes $14/7=2$; the other 7 values have magnitudes below 0.5 after dividing by 2 and all round to 0. Compare the **same seven unchanged values** before and after: their MSE rises from 0.0048 to about 0.29623, roughly 61.7 times as much. Averaging over all eight values instead gives 0.0042 → 0.2592, the same ratio: the maximum is reconstructed exactly in both cases. With groups, the first group is unaffected, the other 3 values in the second group become 0, and the MSE over the seven unchanged values is about 0.143. **Grouping does not remove an outlier; it only confines the damage to its own group.**

[SmoothQuant (2022-11)](https://arxiv.org/abs/2211.10438) puts this into one number: with 8-bit per-tensor quantization, a channel whose maximum is $m_i$ gets only about $2^8\cdot m_i/m$ grid points, where $m$ is the maximum of the whole matrix. In the activations of OPT and similar models they saw outliers about 100 times larger than most values, leaving non-outlier channels with only two or three grid points. The outliers are not random either:

- [LLM.int8() (2022-08)](https://arxiv.org/abs/2208.07339) reports that around 6.7B, large-magnitude features appear in every layer and in about 75% of sequence positions, yet they are concentrated in only 6 feature dimensions of the whole model (about 0.1% of input features), with magnitudes up to 20 times those of other dimensions; setting these dimensions to zero worsens validation perplexity by 600%–1000%.
- SmoothQuant observed that outliers stay in a few fixed channels, large in the same channels for every token, while the weight distribution is much flatter and much easier to quantize.

So in the same model, weights often survive 4 bits while activations can already cause trouble at 8. SmoothQuant and AWQ in the next note both exploit the fact that outliers sit in a few fixed channels.

## Granularity: who shares a scale

Take a linear layer $Y=XW$, where $X$ is $T\times C_i$ ($T$ tokens, $C_i$ input channels) and $W$ is $C_i\times C_o$.

| Granularity | What shares one scale | Extra bits per value (16-bit scale) | Can it be applied after the integer GEMM? | Common use |
| --- | --- | --- | --- | --- |
| per-tensor | the whole matrix | ≈ 0 | yes | static FP8 activations |
| per-channel (weight output channel) | one column of $W$ | $16/C_i$, about 0.004 at $C_i=4096$ | yes | INT8 and FP8 weights |
| per-token (one row of activations) | all channels of one token | ≈ 0 | yes | dynamic activation scales |
| per-group / per-block (along $C_i$) | 16, 32 or 128 consecutive values | about 0.125 for g128 | no; dequantize inside the kernel first, or use an MMA that supports block scaling | INT4 weights, MX formats, NVFP4 |
| per-channel (activation input channel) | all tokens of one input channel | — | no | SmoothQuant sidesteps it with an equivalent transformation |

The key column is whether the scale can be applied afterward. SmoothQuant's derivation is that only scales along the two outer dimensions (the token dimension $T$ and the output channel $C_o$) can move past the matrix multiplication:

$$
Y=\operatorname{diag}(\Delta_X)\,\big(\bar X_{\mathrm{INT8}}\,\bar W_{\mathrm{INT8}}\big)\,\operatorname{diag}(\Delta_W).
$$

$\bar X$ and $\bar W$ are the integer codes, $\Delta_X$ has one scale per token, and $\Delta_W$ has one scale per output channel. If you group along the inner dimension $C_i$, the scale changes partway through the dot-product sum, and it cannot be inserted into the integer accumulation of a Tensor Core. Either the kernel dequantizes the weights to FP16 on the fly as it reads them, as the [GPTQ](https://arxiv.org/abs/2210.17323) kernel does, or the hardware multiplies block scales inside the MMA instruction. Blackwell has the latter: in [PTX ISA 9.4](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html), both `tcgen05.mma` on datacenter Blackwell (sm_100a, §9.7.18.10.7) and `mma.sync` on RTX Blackwell (sm_120a, §9.7.16.3) have block-scaled forms that compute $(A\cdot \text{scale}_A)(B\cdot\text{scale}_B)$ and add it to the accumulator, with UE8M0 scales for MX formats and UE4M3 scales for NVFP4.

## Floating-point formats: a proportional grid

An INT grid is evenly spaced; an FP grid divides each power-of-two interval evenly, so it is denser near 0. There are two 8-bit floating-point formats ([Micikevicius et al., 2022-09](https://arxiv.org/abs/2209.05433); values from Table 2 of the [OCP MX specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf), which reproduces the OCP FP8 specification):

| Format | Sign / exponent / mantissa bits | Exponent bias | Maximum | Smallest normal | Neighbor spacing (reference specified) |
| --- | --- | ---: | ---: | ---: | ---: |
| E4M3 | 1 / 4 / 3 | 7 | 448 | $2^{-6}$ | $2^{-3}$, i.e. 12.5%, relative to the normal interval's lower endpoint |
| E5M2 | 1 / 5 / 2 | 15 | 57,344 | $2^{-14}$ | $2^{-2}$, i.e. 25%, relative to the normal interval's lower endpoint |
| INT8 (symmetric, for comparison) | — | — | $127s$ | — | absolute spacing $s$ everywhere |

The floating-point percentages use the normal interval's lower endpoint $2^e$ as their denominator, not each value itself. With $m$ mantissa bits, the spacing within $[2^e,2^{e+1})$ is $\Delta=2^{e-m}$. Relative to a positive value $u$ in that interval, $\Delta/u\le2^{-m}$ and decreases as $u$ grows. INT8 has constant absolute spacing $s$, but its spacing relative to a nonzero dequantized value $\hat x$ is $s/|\hat x|$, which is not constant.

E4M3 has no infinities and uses only one mantissa pattern for NaN, and the encodings saved push its maximum to 448; E5M2 keeps the IEEE 754 special values. E4M3 has a gap of 1 between 8 and 16 and a gap of 4 between 32 and 64. **For scaled values in the normal range, without overflow or clipping**, round-to-nearest has a relative error bounded by about 6.25%; the absolute error grows with magnitude.

That bound does not cover subnormal values or underflow. At scale 1, E4M3 rounds $0.001$ to $2^{-9}=0.001953125$, a relative error of 95.3125%. Smaller nonzero values can round to zero, giving 100% relative error. The scale determines which values land in this region.

Compare by hand. Four numbers, $[0.02,\ -0.3,\ 1.7,\ 12.0]$, each with a per-tensor scale that maps the maximum to the top of the format: E4M3 uses $s=12/448\approx0.0268$, INT8 uses $s=12/127\approx0.0945$.

| Original | E4M3: $x/s$ → rounded → read back | Relative error | INT8: code → read back | Relative error |
| ---: | --- | ---: | --- | ---: |
| 0.02 | 0.747 → 0.75 → 0.0201 | 0.45% | 0 → 0 | 100% |
| −0.3 | −11.2 → −11 → −0.2946 | 1.8% | −3 → −0.2835 | 5.5% |
| 1.7 | 63.47 → 64 → 1.714 | 0.84% | 18 → 1.701 | 0.05% |
| 12.0 | 448 → 448 → 12.0 | 0 | 127 → 12.0 | 0 |

Near the maximum INT8 is more accurate; far below the maximum FP8 is. [Van Baalen et al. of Qualcomm AI Research (2023-03)](https://arxiv.org/abs/2303.17951) sum it up as "all about the outliers": when a distribution has significant outliers FP8 gives the smaller error, and when it is well behaved INT8 is more accurate. In their dedicated-hardware estimate, the roughly 50%–180% figure describes the FP8 multiply-accumulate unit's **additional gate-count cost** over INT8, not a throughput or efficiency loss exceeding 100%. Specifically, Figure 3 compares against INT8 with a 27-bit fixed-point accumulator: FP8 E4 with FP16 and FP32 accumulators needs 53% and 183% more gates, respectively. Gate count is a first-order proxy for area and power, not a measured GPU speed. The authors conclude that edge inference does not justify a dedicated FP8 implementation. On datacenter GPUs the choice also depends on what the Tensor Cores support; see the hardware section below.

Being called FP8 does not mean the bit patterns are the same. AMD's [HIP 7.16 documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/fp8_numbers.html) states that CDNA 3 (the MI300 series) supports the FNUZ variant in hardware and CDNA 4 (the MI350 series) supports the OCP variant. As defined in the [ONNX float8 notes](https://onnx.ai/onnx/technical/float8.html), E4M3FNUZ has exponent bias 8 (7 for OCP), no infinities and no negative zero, with NaN taking the negative-zero pattern, so the all-ones `S.1111.111` is an ordinary number too. That definition gives a maximum of $1.875\times2^{15-8}=240$ (the ONNX table gives only the bit pattern), against 448 for OCP E4M3. For finite nonzero OCP E4M3FN values, reading the same bits as FNUZ halves the value, so doubling the scale preserves the dequantized value. Special encodings require separate conversion: OCP negative zero becomes NaN in FNUZ and should map to positive zero, while OCP NaN patterns become ordinary $\pm240$ values and must map to the target NaN. Doubling the scale is therefore not a conversion rule for every bit pattern (these conclusions follow from the bit definitions).

## Why 4-bit formats need block scaling: MXFP4 and NVFP4

With only 4 bits per element, E2M1 can represent just 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives. One scale per tensor is not enough; small blocks need their own scales.

| Format | Element | Block size | Block scale | Average bits per value |
| --- | --- | ---: | --- | ---: |
| MXFP8 | E4M3 or E5M2 | 32 | E8M0 (powers of two only) | 8.25 |
| MXFP4 | E2M1 | 32 | E8M0 | 4.25 |
| NVFP4 | E2M1 | 16 | E4M3, times one tensor-level FP32 scale | 4.5 |
| INT4 g128 (for comparison) | INT4 | 128 | FP16 | 4.125 |

The MX rows come from Table 1 of the OCP MX specification v1.0 (which also lists MXFP6 and MXINT8); NVFP4 comes from [NVIDIA's NVFP4 blog post (2025-06-24)](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/). Section 6.3 of the MX specification gives a conversion that implementations should support at a minimum: the block scale $X$ is the largest power of two not exceeding the largest magnitude in the block, divided by the largest power of two the element format can represent (4 for E2M1); each element is $V_i/X$ rounded, with anything above the element maximum clamped to the maximum.

Work one block by hand, $[0.3,\ -1.2,\ 2.4,\ 5.0]$ (a teaching example with only 4 values, ignoring NVFP4's tensor-level scale):

- **MXFP4**: the largest power of two not exceeding 5.0 is 4, so $X=4/4=1$. The elements are the original values: 0.3 → 0.5, −1.2 → −1, 2.4 → 2, and 5.0, exactly halfway between 4 and 6, rounds to the even choice, 4. Read back: $[0.5,-1,2,4]$, MSE 0.31. The maximum loses 20%, and the top grid point 6 is never used.
- **NVFP4**: the scale is $5.0/6\approx0.833$, and the nearest E4M3 value is 0.8125. The elements $[0.369,-1.477,2.954,6.154]$ round to $[0.5,-1.5,3,6]$, which read back as $[0.406,-1.219,2.438,4.875]$, an MSE of about 0.0072, about 43 times lower than MXFP4, at the cost of 0.25 more bits per value.

A power-of-two scale can only jump between 0.5, 1 and 2. When the block maximum falls between two powers of two, either the top grid points are wasted or the maximum is clipped. Try a block maximum of 3.5: $X=2/4=0.5$, and $3.5/0.5=7$ exceeds 6, so it is clamped to 6 and reads back as 3.0. The specification's rule amounts to rounding the scale down, so the largest element can be clipped. NVIDIA's [MXFP8 training recipe (2025-05)](https://arxiv.org/abs/2506.08027) rounds the scale exponent up instead, precisely so that no scaled value exceeds the element maximum.

## Which hardware computes which format directly

A format that saves memory is not automatically fast: if the Tensor Cores do not support it, the data has to be dequantized to 16 bits before computing. As of October 2026, the Tensor Core input-type table in the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html) (page updated 2026-09-10) lists the following for 8 bits and below; the mapping from GPU to compute capability is from the [CUDA GPU list](https://developer.nvidia.com/cuda-gpus):

| Compute capability | Examples | Tensor Core inputs at 8 bits and below |
| --- | --- | --- |
| 7.5, 8.0, 8.6 | T4, A100, A10 | INT8, INT4 |
| 8.9 | L4, L40S, RTX 4090 | FP8, INT8, INT4 |
| 9.0 | H100, H200 | FP8, INT8 (no INT4) |
| 10.0, 10.3, 12.x | B200, B300, RTX PRO 6000 Blackwell, RTX 5090 | FP8, FP6, FP4, INT8 |

The 9.0 row agrees with the [H100 whitepaper V1.04](https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf), which lists FP8, FP16, BF16, TF32, FP64 and INT8 as the supported MMA data types, with no INT4.

Being in the table does not mean being fast. As of October 2026, NVIDIA's [HGX page](https://www.nvidia.com/en-us/data-center/hgx/) (8-GPU systems, sparse figures, dense is half) gives HGX B300 3 POPS of INT8 and 72 PFLOPS of FP8, and HGX B200 72 for both. On the same B300, INT8 is 1/24 of FP8, so on B300 INT8 W8A8 is no longer a route to speed, and 8-bit compute should go through FP8. Throughput across generations is compared in [NVIDIA generations](../nvidia-generations.en.md). On the AMD side, the [precision-support page](https://rocm.docs.amd.com/en/latest/reference/precision-support.html) of the ROCm 10.1 documentation, as of October 2026, lists INT8 and FP8 for the CDNA 3 (MI300 series) matrix cores, with FP8 only in the FNUZ variant, and INT8 plus OCP FP8, FP6 and FP4 for CDNA 4 (MI350 series). The [MI350 architecture page](https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi350.html) also lists MXFP8, MXFP6 and MXFP4, and its peak table gives INT8 the same rate as FP8. When the hardware lacks a format, engines often fall back to weight-only quantization; note 3 has a vLLM example.

## Compute it yourself

```python
def quantize_sym(values, bits=4):
    qmax = 2 ** (bits - 1) - 1
    maximum = max(abs(value) for value in values)
    scale = maximum / qmax if maximum else 1.0
    codes = [max(-qmax, min(qmax, round(value / scale))) for value in values]
    return scale, codes, [code * scale for code in codes]

def quantize_asym(values, bits=4):
    qmax = 2 ** bits - 1
    lower = min(0.0, min(values))
    upper = max(0.0, max(values))
    scale = (upper - lower) / qmax if upper > lower else 1.0
    zero_point = max(0, min(qmax, round(-lower / scale)))
    codes = [max(0, min(qmax, round(value / scale) + zero_point)) for value in values]
    return scale, zero_point, codes, [scale * (code - zero_point) for code in codes]

def mse(expected, restored):
    return sum((value - estimate) ** 2 for value, estimate in zip(expected, restored)) / len(expected)

weights = [0.12, -0.53, 0.88, -0.07, 0.31, 1.40, -0.95, 0.02]
scale, codes, restored = quantize_sym(weights)
assert codes == [1, -3, 4, 0, 2, 7, -5, 0]
assert abs(mse(weights, restored) - 0.0042) < 1e-9

grouped = quantize_sym(weights[:4])[2] + quantize_sym(weights[4:])[2]
assert abs(mse(weights, grouped) - 0.0018592) < 1e-6

outlier_weights = weights[:5] + [14.0] + weights[6:]
_, codes_out, restored_out = quantize_sym(outlier_weights)
assert codes_out == [0, 0, 0, 0, 0, 7, 0, 0]
assert abs(mse(outlier_weights, restored_out) - 0.2592) < 1e-9

unchanged_weights = weights[:5] + weights[6:]
before = mse(unchanged_weights, restored[:5] + restored[6:])
after = mse(unchanged_weights, restored_out[:5] + restored_out[6:])
assert abs(before - 0.0048) < 1e-9
assert abs(after - 0.2962285714) < 1e-9
assert abs(after / before - 61.7142857143) < 1e-9

scale, zero_point, codes, restored = quantize_asym([1.0, 2.0])
assert abs(scale - 2.0 / 15) < 1e-12
assert zero_point == 0
assert scale * (zero_point - zero_point) == 0.0
assert quantize_asym([0.0, 0.0]) == (1.0, 0, [0, 0], [0.0, 0.0])
```

Change `bits` to 8, or change the group length to 2, and watch the MSE. Python's `round` rounds half to even, matching the default rounding of most hardware.

## What these grids cannot tell you

| What to try | What you should see | What you cannot conclude |
| --- | --- | --- |
| Change the 8 weights from per-tensor to groups of 4 | MSE drops from 0.0042 to 0.00186 | That model accuracy improves in the same proportion |
| Quantize the same numbers with E4M3 and with INT8 | FP8 is more accurate for small values, INT8 for large ones | Which format is better for the whole model; that depends on each layer's distribution |
| Switch the same block from MXFP4 to NVFP4 | MSE about 43 times lower in this example | That end-to-end quality differs by 43 times; NVFP4 also spends 0.25 more bits per value |
| Give the group holding the outlier its own scale | Only that group is damaged | That outliers can simply be clipped: LLM.int8() shows that zeroing those dimensions badly worsens perplexity |

The MSE of one tensor is only a starting point. We also need to examine how error accumulates through dozens of layers and whether errors within one output cancel or add up. In the GPTQ example in the next note, two errors in the same direction double the output error. A single tensor's MSE also says nothing about accumulation precision or which layers are most sensitive; note 3 covers the accumulation problem DeepSeek-V3 encountered in FP8 GEMMs. Next: [Weight-only quantization and post-training quantization](weight-only-and-ptq.en.md).
