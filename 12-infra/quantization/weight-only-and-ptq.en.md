# Quantization: weight-only compression and post-training quantization

[中文](weight-only-and-ptq.md) · **English**

> Reading time: ~11 min · Last reviewed: 2026-10

The roofline estimate below uses an 8B teaching model and the H100 SXM5's peak compute and memory bandwidth, initially excluding KV-cache traffic and other overheads. Under those assumptions, reading the BF16 weights at batch 1 has a theoretical lower bound of 4.77 ms per step. INT4 storage in groups of 128 (W4A16) reduces that bound to 1.45 ms. At batch 256, the same estimate gives per-step lower bounds of 4.77 and 4.14 ms. These are predictions from the model, not measured latencies.

Weight-only compression reduces the bytes read during decode. In this simplified model, when weight reads dominate, reducing those bytes lowers the time bound proportionally. Once both schemes are limited by the same compute ceiling, the model no longer predicts a speedup from weight-only compression. Actual performance also depends on dequantization overhead, kernel implementation and efficiency, and activation and KV-cache traffic; these ratios do not directly predict measured speedups.

Several post-training quantization (PTQ) methods use a small calibration set to limit the effect of error on each layer's output. GPTQ compensates for each column's error by adjusting columns not yet quantized, while AWQ scales up important channels based on activation magnitude. For INT8 activations, SmoothQuant offers one way to handle outliers: an equivalent transformation transfers the quantization difficulty to the weights. This is not a required step for every 8-bit scheme, and in particular should not be assumed necessary for FP8.

The previous note, [Formats and error](formats-and-error.en.md), covers grids, scales and groups. [Performance](../performance.en.md#decode-batching) explains the roofline and how batching affects decode.

## Count bytes first: why batch 1 gets more than three times faster

Teaching assumptions: 8B parameters, of which 7.5B sit in the quantized linear layers and 0.5B are the output layer (lm_head), kept in BF16; the GPU is worked out from the H100 SXM5's 3,352 GB/s memory bandwidth, 989.4 TFLOPS dense BF16 and 1,978.9 TFLOPS dense FP8 ([H100 whitepaper V1.04](https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf)); the KV cache is left out for now. At batch 1 the floor for each step is simply weight bytes divided by bandwidth:

| Scheme | Weight bytes | Batch-1 time per token, at least | Relative to BF16 |
| --- | ---: | ---: | ---: |
| BF16 | 16 GB | 4.77 ms | 1× |
| W8A8 FP8 (output layer BF16) | 8.5 GB | 2.54 ms | 1.88× |
| W4A16 g128 (output layer BF16) | 4.87 GB | 1.45 ms | 3.29× |

The W4A16 bytes are $7.5\times10^9\times4.125/8+0.5\times10^9\times2\approx4.87\times10^9$. It falls short of 4× because the unquantized output layer takes 1 GB and the scales add 0.125 bits: **the harder you compress, the larger the share of what you left uncompressed**, already 21% here.

A public measurement: [GPTQ (2022-10)](https://arxiv.org/abs/2210.17323) compressed OPT-175B to 3 bits and generated 128 tokens at batch 1. Each token took 71 ms on one A100 80GB, against 230 ms for FP16 on 5 GPUs, about 3.24×. Their kernel multiplies a quantized matrix by a full-precision vector, dequantizing the weights as it reads them and leaving the activations unquantized; a matrix-vector product is bandwidth-bound anyway, so the extra dequantization work stays hidden.

### When the batch grows

W4A16 still computes on BF16 Tensor Cores after dequantizing, so in this model its compute ceiling is the same as BF16. In W8A8, the layers using FP8 have roughly twice the peak compute, but the output layer remains BF16, giving a whole-model compute-throughput ceiling about 1.88× as high. Each step takes $\max(\text{bytes}/\text{bandwidth},\ \text{operations}/\text{peak})$, still without KV:

| Batch | BF16: per step · tokens/s | W4A16: per step · tokens/s | W8A8 FP8: per step · tokens/s |
| ---: | ---: | ---: | ---: |
| 1 | 4.77 ms · 210 | 1.45 ms · 689 | 2.54 ms · 394 |
| 16 | 4.77 ms · 3,352 | 1.45 ms · 11,019 | 2.54 ms · 6,310 |
| 64 | 4.77 ms · 13,408 | 1.45 ms · 44,076 | 2.54 ms · 25,239 |
| 128 | 4.77 ms · 26,816 | 2.07 ms · 61,838 | 2.54 ms · 50,477 |
| 256 | 4.77 ms · 53,632 | 4.14 ms · 61,838 | 2.54 ms · 100,954 |
| 512 | 8.28 ms · 61,838 | 8.28 ms · 61,838 | 4.40 ms · 116,405 |

The lower bound for reading the W4A16 weights is 1.45 ms; at batch $B$, the compute-time lower bound is $B\times1.6\times10^{10}/(989.4\times10^{12})$ seconds. They are equal at $B\approx90$. Beyond that point W4A16 becomes compute-bound in the model, while BF16 may still be bandwidth-bound: BF16 does not cross over until $B\approx295$. Only once both are compute-bound does the model predict the same time lower bound.

The W8A8 FP8 compute-time lower bound must account separately for the quantized layers and the BF16 output layer: $B\times[1.5\times10^{10}/(1978.9\times10^{12})+1.0\times10^9/(989.4\times10^{12})]$ seconds. Equating this to the time to read 8.5 GB of weights also gives $B\approx295$. Its throughput ceiling is the table's 116,405 tokens/s, about 1.88× BF16's 61,838 tokens/s rather than exactly twice as high. These are theoretical time lower bounds and throughput upper bounds under peak assumptions, not measurements.

Actual kernel speedups can start declining at smaller batches: [Marlin (2024-08)](https://arxiv.org/abs/2408.11743) self-reports, in single-layer tests on large matrices on an A10, close to the ideal 4× up to batch 16–32, shrinking gradually after that to about 1.5× at batch 128. [Kurtic et al. (2024-11, ACL 2025)](https://arxiv.org/abs/2411.02355) reach the same direction on vLLM: in synchronous, latency-sensitive deployments W4A16 is the most cost-efficient, in asynchronous continuous batching W8A8 gives the highest throughput, and the crossover depends on model, hardware and task. This table leaves out the KV cache. Once each request's own KV is added, the bytes read per step grow with the batch. Under the same roofline assumptions, decode becomes compute-bound later, but weights are only part of what is read, diluting the predicted speedup: with 4K tokens of BF16 KV per request at batch 64, the estimated W4A16 speedup is only about 1.28× (11.7 against 15.0 ms), see [Performance](../performance.en.md#decode-batching).

## The simplest approach: round to nearest (RTN)

RTN (round-to-nearest) is what the previous note did: round each weight independently to the nearest grid point, without looking at any data. At 4 bits with g128 it often holds up: in Table 4 of [AWQ](https://arxiv.org/abs/2306.00978), the Llama-2 WikiText-2 perplexities are 5.47 for FP16, 5.73 for RTN, 5.69 for GPTQ and 5.60 for AWQ at 7B, and 3.32, 3.46, 3.42 and 3.41 at 70B, so the methods differ by a few tenths at most.

With fewer bits and no groups, RTN's error rises sharply. GPTQ's Table 3 uses per-row scales: OPT-175B is 8.34 in FP16, 10.54 with 4-bit RTN (worse than the FP16 13B model) and 8.37 with GPTQ; at 3 bits RTN gives 7.3×10³ and GPTQ 8.68. The problem is that each weight is rounded independently: when input channels are correlated, their errors add up in the output. To know which directions matter, we need the inputs this layer actually receives, which is what calibration data provides.

## GPTQ: quantize a column, push its error onto the columns not yet quantized

GPTQ solves this problem one layer at a time:

$$
\hat W=\arg\min_{\hat W}\ \big\|WX-\hat WX\big\|_2^2,
$$

where $W$ is the layer's weight (one row per output) and $X$ is the layer's input on the calibration data, one column per token. For one row of weights $w$ this is a quadratic function with Hessian $H=2XX^\top$. GPTQ inherits the formula from OBQ (Optimal Brain Quantization): after quantizing the $q$-th weight, the remaining unquantized weights $F$ are updated as follows to minimize the output error:

$$
\delta_F=-\frac{w_q-\operatorname{quant}(w_q)}{[H_F^{-1}]_{qq}}\,\big(H_F^{-1}\big)_{:,q}.
$$

**Worked once.** One output, two input channels, 4 calibration tokens (teaching numbers): $x_1=[1,\ 2,\ -1,\ 0.5]$ and $x_2=[1,\ 1.8,\ -1.2,\ 0.6]$, two highly correlated channels; $w=[0.62,\ 0.37]$; grid step 0.25.

- $H_{12}=2\sum_t x_{1t}x_{2t}=2\times6.1=12.2$, $H_{22}=2\sum_t x_{2t}^2=2\times6.04=12.08$.
- **RTN**: $[0.5,\ 0.25]$. Both errors are +0.12 and multiply almost identical inputs, so the output errors over the 4 tokens are $[0.24,\ 0.456,\ -0.264,\ 0.132]$, with a sum of squares of 0.353.
- **GPTQ**: first quantize $w_1\to0.5$, with error $e_1=0.12$. With only one weight left, the formula reduces to $w_2\leftarrow w_2+e_1H_{12}/H_{22}=0.37+0.12\times1.0099\approx0.491$, which then quantizes to 0.5. The output errors are $[-0.01,\ 0.006,\ 0.036,\ -0.018]$, a sum of squares of 0.00176, about 1/200 of RTN.

The intuition: $x_2\approx x_1$, so the small extra in $w_2$ makes up for what $w_1$ lost. In this example the GPTQ result also happens to be the best point on the grid (two dimensions can be checked by brute force).

```python
X = [[1.0, 2.0, -1.0, 0.5],
     [1.0, 1.8, -1.2, 0.6]]
w = [0.62, 0.37]
step = 0.25

def quant(v):
    return step * round(v / step)

def output_error(w_hat):
    total = 0.0
    for t in range(4):
        exact = w[0] * X[0][t] + w[1] * X[1][t]
        approx = w_hat[0] * X[0][t] + w_hat[1] * X[1][t]
        total += (exact - approx) ** 2
    return total

h12 = 2 * sum(a * b for a, b in zip(X[0], X[1]))
h22 = 2 * sum(b * b for b in X[1])
rtn = [quant(w[0]), quant(w[1])]
q1 = quant(w[0])
gptq = [q1, quant(w[1] + (w[0] - q1) * h12 / h22)]
assert rtn == [0.5, 0.25] and gptq == [0.5, 0.5]
assert round(output_error(rtn), 4) == 0.3527
assert round(output_error(gptq), 5) == 0.00176
```

Change $x_2$ to numbers unrelated to $x_1$ and the compensation barely helps: $H_{12}$ is close to 0.

To get from OBQ to 175B, GPTQ changes three things: all rows are quantized in the same fixed order and share one $H^{-1}$ (on large layers this is barely worse than OBQ's greedy order); updates run in blocks of 128 columns, with later columns updated once each block is done, avoiding the memory-bandwidth bottleneck; and a Cholesky decomposition plus a damping of 1% of the average diagonal value keeps it numerically stable. The calibration data is 128 random segments of 2,048 tokens from C4, and only one Transformer block is in GPU memory at a time. Quantizing OPT-175B takes 4.2 hours on one A100 80GB (Table 2).

## AWQ: look at the activations first, scale up the channels that matter

[AWQ (2023-06, MLSys 2024 best paper)](https://arxiv.org/abs/2306.00978) starts from an experiment (Table 1, OPT-6.7B, INT3-g128, WikiText perplexity): FP16 is 10.86 and RTN is 23.54; keeping just 1% of channels in FP16, chosen by activation magnitude, brings perplexity back to 11.39; chosen by weight norm it is 22.37, and chosen at random 24.23. **Whether a weight matters depends on the activation it multiplies, not on how large the weight itself is.**

Mixed precision is unfriendly to hardware, so AWQ scales instead: multiply the important input channel's weights by $s>1$ and divide the corresponding input by $s$, which is mathematically equivalent. The quantizer is $Q(w)=\Delta\cdot\operatorname{Round}(w/\Delta)$, and after scaling that channel's error becomes $\Delta'\cdot\operatorname{RoundErr}(ws/\Delta')\cdot x/s$. As long as scaling one element does not change the group's maximum, $\Delta'\approx\Delta$ and that channel's error shrinks to about $1/s$. In the paper's Table 2 (OPT-6.7B), $s=2$ is best, at perplexity 11.92; at $s=4$, 21.2% of groups have a changed $\Delta$, larger by 21% on average, and perplexity rises back to 12.36.

**Worked once.** A group of 4 weights $w=[0.90,\ 0.20,\ -0.35,\ 0.50]$, with average activation magnitudes per input channel $s_X=[1,\ 8,\ 1,\ 1]$; channel 2 is the important one. Symmetric INT4 with $\Delta=\max|w|/7$. Teaching assumption: the channel inputs are independent with mean 0 and $\mathbb E[x_j^2]=s_{X,j}^2$, so the output's mean squared error is about $\sum_j(w_j-\hat w_j)^2s_{X,j}^2$.

- **No scaling**: $\Delta=0.9/7\approx0.1286$. 0.20 is quantized to $2\Delta\approx0.257$, an error of 0.057, which after multiplying by 8 and squaring gives 0.209, almost all of the total error of 0.2105.
- **Channel 2 times 2**: $0.4$ is still below 0.9, so $\Delta$ is unchanged; $0.4/\Delta\approx3.11$ rounds to 3 and reads back as 0.386, which divided by 2 is an effective weight of 0.193, an error of 0.007, and the total error drops to 0.0047.

AWQ does not pick $s$ by hand. It sets $s=s_X^{\alpha}$ and grid-searches 20 values of $\alpha$ in $[0,1]$ to minimize $\|Q(W\operatorname{diag}(s))(\operatorname{diag}(s)^{-1}X)-WX\|$, then adds weight clipping chosen by MSE. In this small example:

| $\alpha$ | $s$ for channel 2 | $\Delta$ | Weighted error |
| ---: | ---: | ---: | ---: |
| 0 | 1 | 0.1286 | 0.2105 |
| 0.25 | 1.68 | 0.1286 | 0.0566 |
| 0.5 | 2.83 | 0.1286 | 0.0226 |
| 0.75 | 4.76 | 0.1359 | 0.0079 |
| 1 | 8 | 0.2286 | 0.0135 |

At $\alpha=1$, $0.2\times8=1.6$ becomes the group maximum, $\Delta$ grows 1.78 times, and quantization error increases in the other three channels. This is why we need to search for a suitable $s$; changes in rounding also make the error fluctuate as $s$ changes.

AWQ defaults to g128 and targets INT4 and INT3; it needs no backpropagation and no reconstruction. The paper reports that it depends less on calibration data (OPT-6.7B, INT3-g128): with 16 sequences it beats GPTQ with 192, and when calibration and evaluation data come from different domains (PubMed and Enron swapped), its perplexity rises by only 0.5–0.6 against 2.3–4.9 for GPTQ.

## Activations too: SmoothQuant's W8A8

Weight-only compression does not help the compute-bound phases (prefill, large batches). To use INT8 Tensor Cores, the activations must be 8-bit too, and as the previous note showed, activation outliers sit in a few fixed input channels, while per-input-channel activation scales cannot be applied after the GEMM. [SmoothQuant (2022-11, ICML 2023)](https://arxiv.org/abs/2211.10438) first applies an equivalent transformation:

$$
Y=\big(X\operatorname{diag}(s)^{-1}\big)\big(\operatorname{diag}(s)W\big),\qquad s_j=\frac{\max|X_j|^{\alpha}}{\max|W_j|^{1-\alpha}}.
$$

$X_j$ is the activation of the $j$-th input channel (measured on calibration data) and $W_j$ is the $j$-th row of $W$. $\alpha$ is the "migration strength": OPT and BLOOM use 0.5, and GLM-130B, with more outliers, uses 0.75; on OPT-175B with LAMBADA, values between 0.4 and 0.6 work best. $\operatorname{diag}(s)^{-1}$ can be folded offline into the preceding LayerNorm or linear layer, so there is no extra cost at run time.

**Worked once.** Three tokens, three input channels, channel 2 being the outlier channel (teaching numbers):

$$
X=\begin{bmatrix}0.8&-40&0.3\\-1.2&48&0.5\\0.6&-36&-0.9\end{bmatrix},\qquad
W=\begin{bmatrix}0.8&-0.3\\0.2&0.1\\-0.5&0.6\end{bmatrix}.
$$

The largest activation magnitudes per input channel are $[1.2,\ 48,\ 0.9]$, and the largest magnitudes of the rows of $W$ are $[0.8,\ 0.2,\ 0.6]$. With $\alpha=0.5$, $s=[1.22,\ 15.5,\ 1.22]$. After the transformation, every channel of both the smoothed $\hat X=X\operatorname{diag}(s)^{-1}$ and $\hat W=\operatorname{diag}(s)W$ has maximum $\sqrt{\max|X_j|\cdot\max|W_j|}=[0.98,\ 3.10,\ 0.73]$. With per-tensor INT8 activations, the step falls from $48/127\approx0.378$ to $3.10/127\approx0.0244$, and channel 1's codes go from $[2,-3,2]$ to $[27,-40,20]$, from about 3 grid points to about 40. With per-tensor activations and per-channel weights, the relative error of $Y$ falls from 1.95% to 0.15%; with per-token activations, from 1.76% to 0.048%. The price is that the second row of the weights is scaled up 15.5 times and the weights become somewhat harder to quantize, which is why $\alpha$ cannot be too large.

The paper self-reports (average accuracy in Table 4): on OPT-175B, O1 through O3 all stay within 0.5 points of FP16; on BLOOM-176B, O1 and O2 match FP16 while the coarsest level, O3, loses 0.8 points; GLM-130B is harder to quantize, and only O1 matches, with O2 and O3 losing 1.0–1.3 points. MT-NLG 530B also runs in W8A8. The abstract gives a maximum speedup of 1.56× (prefill of OPT-13B and 30B, FasterTransformer implementation, one GPU) with memory roughly halved, and coarser granularity, from O1 (per-token dynamic) to O3 (per-tensor static), is faster. The alternative, [LLM.int8()](https://arxiv.org/abs/2208.07339), splits off the outlier dimensions above magnitude 6 and multiplies them in FP16; it self-reports no accuracy loss up to 175B with memory halved, but its own end-to-end generation is slightly slower than BF16 (BLOOM-176B on 8 A100 80GB GPUs, batch 1: 253 against 239 ms/token).

## The methods side by side

| Method | What it compresses | What data it needs | Main idea | Costs and risks |
| --- | --- | --- | --- | --- |
| RTN | weights (activations too) | none | round each value to nearest | collapses at 3 bits without groups |
| GPTQ | weights | a small calibration text set | use $H^{-1}$ to push error onto later columns | can fit the calibration distribution (AWQ's cross-domain experiment) |
| AWQ | weights | an even smaller calibration set | scale up important channels by activation size | only tunes scales and clipping, not the rounding direction of individual weights |
| SmoothQuant | weights and activations (W8A8) | activation maxima from calibration | move activation difficulty onto the weights | $\alpha$ needs tuning; weights get harder to quantize |
| LLM.int8() | W8A8, outlier dimensions in FP16 | none | mixed-precision decomposition | usually not faster than FP16 |
| Rotation (e.g. [QuaRot](https://arxiv.org/abs/2404.00456)) | weights, activations and KV all at 4 bits | depends on implementation | an orthogonal rotation spreads outliers out without changing the output | extra rotation compute; self-reports at most 0.47 more perplexity on Llama2-70B |
| Quantization-aware training (QAT) | anything | training data and compute | simulate quantization during training | requires training; [Nagel et al.](https://arxiv.org/abs/2106.08295) note that it supports lower bit widths |

## Where these results stop

- Most of the accuracy numbers quoted above are perplexities on WikiText-2 or C4, on older models such as OPT, BLOOM and Llama-2. GPTQ itself notes that its calibration data comes from the C4 training set, so its C4 perplexity is not fully zero-shot. Task accuracy, flipped answers and long context need separate measurement, see [How to measure what was lost](checking-quality.en.md).
- The speedups were measured with specific kernels, on specific GPUs, at batch 1; the advantage shrinks at large batch, and a new engine version needs a new measurement.
- The output layer and embeddings are often kept at high precision, and the KV cache was not compressed here; at long context the KV cache is the bulk, see the next note, [FP8 inference and KV-cache quantization](fp8-and-kv-cache.en.md). Training further on a 4-bit base (QLoRA, NF4) is in [LoRA and QLoRA](../../05-post-training/lora-and-qlora.en.md).
