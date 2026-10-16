# Quantization: review questions

[中文](review.md) · **English**

> Reading time: ~3 min · Draft, pending review

This page condenses the series into interview questions and a self-check; every number and conclusion in the answers can be traced to the first four notes.

> **Draft, pending review.** The site owner has not reviewed this note yet; facts, emphasis and wording may change. Check the primary sources linked in the text before relying on it.

## Interview questions

<details class="interview" markdown="1">
<summary>What do W4A16, W8A8 and an FP8 KV cache each buy? When do you pick which?</summary>

W4A16 compresses only the weights: it saves memory and cuts the bytes read per decode step, so at small batch the speedup is almost proportional to the bytes saved, but after dequantizing it still computes on 16-bit Tensor Cores and gains nothing once compute-bound. W8A8 (INT8 or FP8) compresses the activations too: besides halving the bytes, it uses 8-bit Tensor Cores, doubling the H100 SXM5 dense peak from 989.4 to 1,978.9 TFLOPS, so prefill and large batches get faster. FP8 KV compresses the KV cache: at long context its bytes often exceed the weights; one 128K sequence of the teaching model is 16 GiB, and in FP8 the number of sequences that fit on an 80 GB GPU goes from 3 to 6. Work out first whether the bottleneck is weight bytes, compute or KV, then choose.

</details>

<details class="interview" markdown="1">
<summary>Why can one outlier ruin per-tensor INT4? How much do per-channel and per-group fix? Why can activations not have per-input-channel scales?</summary>

The step is set by the largest value among the numbers sharing a scale. In the 8-weight example, with a maximum of 1.40, $s=0.2$ and the MSE over all eight weights is 0.0042; with 14.0 instead, $s=2$, the other 7 values all round to 0 and the eight-weight MSE is 0.2592. Comparing only the same seven unchanged values gives 0.0048 → 0.29623; either consistent denominator gives about 61.7×. Grouping confines the damage to one group; it does not remove the outlier. The weights' output channels and the activations' tokens are the outer dimensions of the GEMM, so their scales can be applied after the integer matrix multiplication; the activations' input channels are the summed dimension, so the scale would change partway through the dot product, which cannot be inserted into the Tensor Core's integer accumulation. LLM activation outliers sit precisely in a few fixed input channels, which is why SmoothQuant uses an equivalent transformation.

</details>

<details class="interview" markdown="1">
<summary>How does the error of FP8 (E4M3) differ from INT8? And how do MXFP4 and NVFP4 differ?</summary>

INT8's grid is evenly spaced, so its absolute rounding-error bound is constant within the unclipped range. E4M3 has 8 grid points in each normal power-of-two interval: for scaled values in the normal range, without overflow or clipping, the round-to-nearest relative error is bounded by about 6.25%. Subnormal values are different: at scale 1, $0.001$ rounds to $2^{-9}$ with 95.3125% relative error, and underflow to zero gives 100%. Quantizing $[0.02,-0.3,1.7,12]$ per tensor, 0.02 becomes 0 in INT8 but has a 0.45% error in E4M3, while 1.7 is more accurate in INT8. Qualcomm's conclusion is that it comes down to outliers: FP8 is better when a distribution has significant outliers, INT8 when it is well behaved. At 4 bits both need block scaling: MXFP4 shares one power-of-two scale (E8M0) across 32 values, 4.25 bits on average; NVFP4 uses one E4M3 scale per 16 values times a tensor-level FP32, 4.5 bits on average. For the block $[0.3,-1.2,2.4,5.0]$, the MX scale can only be 1, 5.0 reads back as 4, and the MSE is 0.31; the NVFP4 scale is 0.8125 and the MSE is about 0.0072.

</details>

<details class="interview" markdown="1">
<summary>Why is W4A16 more than three times faster at batch 1, and why does the advantage disappear at large batch? How do you estimate the crossover?</summary>

Batch-1 decode does about 1 operation per byte read, so the time is roughly weight bytes divided by bandwidth. The teaching 8B model is 16 GB and 4.77 ms in BF16; W4A16 g128 (output layer kept in BF16) is 4.87 GB and 1.45 ms, about 3.29×, short of 4× because of the uncompressed output layer and the scales. W4A16 computes in BF16, so the compute time at batch $B$ is $B\times2N/989.4$ TFLOPS; setting it equal to 1.45 ms gives $B\approx90$, after which it is compute-bound like BF16, whereas the FP8 W8A8 roof is twice as high. Measurements fall off earlier: Marlin self-reports close to 4× up to batch 16–32; Kurtic et al. recommend W4A16 for synchronous low-latency serving and W8A8 for high-throughput continuous batching.

</details>

<details class="interview" markdown="1">
<summary>How does GPTQ compensate for quantization error? How does AWQ's idea differ?</summary>

GPTQ minimizes $\|WX-\hat WX\|^2$ layer by layer, with Hessian $H=2XX^\top$; after quantizing a weight, it updates the unquantized weights by $\delta_F=-\frac{w_q-\mathrm{quant}(w_q)}{[H_F^{-1}]_{qq}}(H_F^{-1})_{:,q}$. With two highly correlated inputs, $w=[0.62,0.37]$ and step 0.25: RTN gives $[0.5,0.25]$ with a sum of squared output errors of 0.353; GPTQ first quantizes $w_1$ to 0.5 and pushes its 0.12 error onto $w_2$ via $H_{12}/H_{22}\approx1.01$, giving $[0.5,0.5]$ and an error of 0.00176. In engineering terms, it scales to 175B by sharing one $H^{-1}$ through a fixed order, lazy updates in blocks of 128 columns, and a Cholesky decomposition. AWQ does not change rounding directions; it finds important channels by activation size, multiplies their weights by $s$ and divides their inputs by $s$, grid-searching $s=s_X^{\alpha}$; it needs less calibration data and is steadier across domains (perplexity up 0.5–0.6 on OPT-6.7B, against 2.3–4.9 for GPTQ).

</details>

<details class="interview" markdown="1">
<summary>How does SmoothQuant make W8A8 work? What does $\alpha$ do?</summary>

Activation outlier channels leave ordinary channels with only two or three grid points under per-tensor or per-token quantization, and per-input-channel activation scales cannot enter an INT8 GEMM. SmoothQuant applies the equivalent transformation $Y=(X\operatorname{diag}(s)^{-1})(\operatorname{diag}(s)W)$ with $s_j=\max|X_j|^{\alpha}/\max|W_j|^{1-\alpha}$, moving difficulty from activations to weights, and folds $\operatorname{diag}(s)^{-1}$ offline into the previous layer. At $\alpha=0.5$, every channel on both sides ends up with maximum $\sqrt{\max|X_j|\max|W_j|}$; in the teaching example channel 1 goes from about 3 usable grid points to about 40, and the relative output error drops from 1.95% to 0.15%. Too small an $\alpha$ leaves activations hard to quantize, too large makes weights hard; on OPT-175B, 0.4–0.6 works best.

</details>

<details class="interview" markdown="1">
<summary>Why does KIVI quantize keys per channel and values per token? Is that the only option?</summary>

KIVI observed that keys have a few fixed channels with very large magnitudes, values have no clear outlier pattern, and the attention output comes mainly from the values of a few important tokens. Quantizing keys per token lets the outlier channel stretch each row's step: in the 4×4 example at 2 bits, the ordinary channels all collapse to one value and the scores go from $[1.2,0.9,0.9,1.6]$ to $[0.3,-0.1,-0.8,0.6]$; per channel, the largest error is only 0.067. The relative error of $K$ itself, 8.6% against 0.6%, understates how badly the scores are damaged. In KIVI's Llama-2-7B results at 2 bits, GSM8K is only 0.83 with both per token, against 12.74 for KIVI-2 (13.50 at 16 bits). This does not make the grouping universal: QuaRot rotates keys, then quantizes channel groups within each token and head. Choose the layout for the resulting distribution and kernel. In deployment, watch the engine's defaults too: vLLM sets all KV scales to 1.0 without calibration.

</details>

<details class="interview" markdown="1">
<summary>How do you accept a quantized model? Why is accuracy alone not enough?</summary>

When accuracy barely changes, many individual answers may still trade places: Dutta et al. measured GPTQ W4A16 on Llama2-70B-chat losing only 0.45 points on MMLU while 4.26% of answers flipped. Do three things: mean KL and top-1 agreement on fixed text (once a near-tied position flips, the rest of the text diverges); paired comparison on the same questions, where 1,000 items, 56 flips and a net change of −0.4 give a 95% interval of about $[-1.87,+1.07]$ points; and slices defined in advance: long context (4-bit methods lose up to 59 points), non-English (Japanese, 1.7% on automatic metrics against 16.0% by human judgment), math reasoning. Also record whether calibration data overlaps the evaluation, the format and scales of each part, and the engine and kernels, since without hardware support it falls back to weight-only.

</details>

## Self-check

<div class="taste-check">
  <strong>If you really understand it, you should be able to explain:</strong>
  <ol>
    <li>Given a 70B model and one 80 GB GPU, how much do BF16, FP8 and INT4 g128 weights take, how much is left for KV, and what is the batch-1 floor per token?</li>
    <li>Why can a weight's output-channel scale be applied after the GEMM, while scales grouped along the input dimension need dequantization inside the kernel or Blackwell's block-scaled MMA?</li>
    <li>The MX specification sets the block scale to the power of two not exceeding the maximum; what happens to FP4 elements when the block maximum is 3.5? Why does the MXFP8 training recipe round up instead?</li>
    <li>At 128K context and batch 3, by how much do the bytes read per step fall when only the weights are compressed, and when only the KV is? Why does the conclusion flip at short context?</li>
    <li>After a quantization, which fields would you record and which metrics and slices would you report, so others can reproduce it and believe that "not much was lost"?</li>
  </ol>
</div>

## References

- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877): the affine mapping with scale and zero-point
- [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295): standard PTQ and QAT pipelines
- [LLM.int8()](https://arxiv.org/abs/2208.07339): systematic outlier features in large-model activations, mixed-precision decomposition
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433): E4M3 and E5M2
- [FP8 versus INT8 for efficient deep learning inference](https://arxiv.org/abs/2303.17951): error and hardware cost of the two formats
- [OCP Microscaling Formats (MX) v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) and [Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537): MXFP8, MXFP6, MXFP4, MXINT8
- [Introducing NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/): a 4-bit format with groups of 16 and an E4M3 scale
- [GPTQ](https://arxiv.org/abs/2210.17323): post-training quantization with second-order error compensation
- [AWQ](https://arxiv.org/abs/2306.00978): finding and scaling important channels by activation
- [SmoothQuant](https://arxiv.org/abs/2211.10438): moving quantization difficulty from activations to weights
- [QuaRot](https://arxiv.org/abs/2404.00456): 4-bit weights, activations and KV after rotation
- [MARLIN](https://arxiv.org/abs/2408.11743): a kernel for 4-bit weights times FP16 activations, and how it depends on batch
- [KIVI](https://arxiv.org/abs/2402.02750), [KVQuant](https://arxiv.org/abs/2401.18079): per-channel and per-token KV-cache quantization
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437): fine-grained FP8 training and accumulation precision
- [Recipes for Pre-training LLMs with MXFP8](https://arxiv.org/abs/2506.08027), [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149): training recipes for block-scaled formats
- [Accuracy is Not All You Need](https://arxiv.org/abs/2407.09141): flips and KL divergence
- ["Give Me BF16 or Give Me Death"?](https://arxiv.org/abs/2411.02355): large-scale quantization evaluation on Llama-3.1 and deployment advice
- [Does quantization affect models' performance on long-context tasks?](https://arxiv.org/abs/2505.20276), [How Does Quantization Affect Multilingual LLMs?](https://arxiv.org/abs/2407.03211), [Quantization Hurts Reasoning?](https://arxiv.org/abs/2504.04823): long-context, multilingual and reasoning slices
