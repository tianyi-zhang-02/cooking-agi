# Scaling: for a given compute budget, how big a model and how much data

[中文](compute-optimal.md) · **English**

> Reading time: ~12 min · Last reviewed: 2026-10

Suppose you can run 64 H100s for four or five days, which works out to roughly $10^{22}$ floating-point operations. That budget could train a 3B-parameter model on a lot of data, or a 30B-parameter model on very little. Which gives the lower loss?

The Chinchilla fit offers a starting point: as the training budget grows, scale parameters and data in roughly equal proportion, with about 20 tokens per parameter. At that ratio, $10^{22}$ FLOPs corresponds to about 9B parameters and 180B tokens. The ratio is fitted on small runs and then extrapolated; it holds only when the data and optimization recipe are similar and the data is seen about once. The further you extrapolate, the more you should treat the result as a range rather than a precise point.

## First turn the budget into FLOPs

For each training token, a dense Transformer does about $2N$ floating-point operations in the forward pass ($N$ is the parameter count; each weight takes part in one multiply-add), and the backward pass costs about twice the forward, so the total is about $6N$. Training on $D$ tokens costs

$$
C\approx6ND.
$$

The formula comes from [Kaplan et al. (2020-01) §2.1](https://arxiv.org/abs/2001.08361). Their $N$ excludes embedding parameters, and they drop the attention term that grows with context length (their models satisfy $d_{\rm model}\gg n_{\rm ctx}/12$, so it is small). [Chinchilla (Hoffmann et al., 2022-03)](https://arxiv.org/abs/2203.15556) counts embeddings in both parameters and FLOPs; Appendix F counts every term and finds a ratio to $6ND$ between 0.99 and 1.10. The two papers count $N$ differently, and we will see that this matters. $6ND$ also appears in [the pretraining pipeline](../deep-dives/pretraining-pipeline.en.md), which covers what it leaves out.

The budget itself comes from hardware:

$$
C=n_{\rm GPU}\times P_{\rm peak}\times\mathrm{MFU}\times t.
$$

Teaching assumption: 64 H100 SXM5 GPUs, a BF16 dense peak of 989.4 TFLOPS, and 40% MFU (how MFU is computed and why it is often only 30–40% is covered in [performance: roofline and MFU](../../12-infra/performance.en.md#mfu)). The effective rate is $64\times989.4\times10^{12}\times0.4\approx2.53\times10^{16}$ FLOP/s, so $10^{22}$ FLOPs takes $3.95\times10^{5}$ seconds, about 4.6 days; on 8 GPUs the same budget takes 36.6 days.

Once $C$ is fixed, $N\times D=C/6\approx1.67\times10^{21}$ is a hyperbola: double the parameters and you halve the tokens you can afford. The question becomes where on that curve the loss is lowest.

## Loss falls as a power law in scale

Kaplan et al. found on WebText2 that, as long as the other two factors are not the bottleneck, test loss is approximately a power law in parameters, data and compute separately. For example, with plenty of data and training close to convergence:

$$
L(N)=\left(\frac{N_c}{N}\right)^{\alpha_N},\qquad \alpha_N\approx0.076,\quad N_c\approx8.8\times10^{13}.
$$

On a log scale this is a straight line: $\log L=\alpha_N(\log N_c-\log N)$. Doubling the parameters multiplies the loss by $2^{-0.076}\approx0.949$, about a 5% drop; ten times the parameters multiplies it by 0.839. The paper also notes that constants such as $N_c$ depend on the vocabulary and tokenization and have no absolute meaning; the exponents are what can be compared across setups.

A pure power law keeps falling to zero, but text has uncertainty that cannot be removed. Chinchilla's third approach uses a form with a constant term:

$$
L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}.
$$

$E$ is the loss even an ideal model cannot remove, the second term comes from limited model capacity, and the third from limited data and optimization steps. The fitted values in the paper are $E=1.69$, $A=406.4$, $B=410.7$, $\alpha=0.34$, $\beta=0.28$ (Appendix D.2). Minimizing under the constraint $6ND=C$ has a closed form, $N_{\rm opt}=G\,(C/6)^{a}$ with $a=\beta/(\alpha+\beta)$ and $G=(\alpha A/\beta B)^{1/(\alpha+\beta)}$ (Eq. 4). So "how fast parameters should grow with compute" is the exponent $a$.

## Fit a line by hand from a few small runs

In practice, a common choice is Chinchilla's second approach, called IsoFLOP. Pick a few small budgets; Chinchilla used 9, from $6\times10^{18}$ to $3\times10^{21}$ FLOPs. Train 5 to 10 model sizes at each budget, setting the token count to $C/6N$ and matching the learning-rate decay to each run's length. Plot loss against $\log N$, fit a parabola, and use its minimum as $N_{\rm opt}$ for that budget. Finally, fit a straight line through those minima on log–log axes.

The four minima below are made-up teaching numbers, not measurements from any paper:

| Budget $C$ (FLOPs) | Minimum $N_{\rm opt}$ | $\log_{10}N_{\rm opt}$ | Implied $D=C/6N$ | $D/N$ |
| --- | --- | --- | --- | --- |
| $10^{18}$ | 90M | 7.954 | 1.85B | 20.6 |
| $10^{19}$ | 300M | 8.477 | 5.56B | 18.5 |
| $10^{20}$ | 900M | 8.954 | 18.5B | 20.6 |
| $10^{21}$ | 2.9B | 9.462 | 57.5B | 19.8 |

First the slope by hand, from the first and last points: $(9.462-7.954)/(21-18)\approx0.503$. Then least squares over all four points:

```python
import math

points = [(1e18, 9.0e7), (1e19, 3.0e8), (1e20, 9.0e8), (1e21, 2.9e9)]  # teaching numbers
xs = [math.log10(c) for c, _ in points]
ys = [math.log10(n) for _, n in points]
mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sum((x - mean_x) ** 2 for x in xs)
intercept = mean_y - slope * mean_x
n_opt = 10 ** (slope * 22 + intercept)          # extrapolate to C = 1e22
print(round(slope, 3), round(intercept, 3), f"{n_opt:.2e}", round(1e22 / (6 * n_opt) / n_opt, 1))
# 0.5 -1.041 9.17e+09 19.8
```

This gives $a\approx0.500$ and an intercept of $-1.041$, with all four residuals within 0.016 (the teaching numbers were built around $a=0.5$). Extrapolating to $C=10^{22}$: $\log_{10}N=0.500\times22-1.041=9.96$, so $N\approx9.2$B and $D=C/6N\approx182$B, about 20 tokens per parameter, matching the claim at the top.

### A small change in the exponent, a large change far out

A fitted exponent carries error. The table below keeps the $10^{21}$ point (2.9B parameters) fixed and changes only the exponent, extrapolating to 10 times and 1000 times the budget:

| Exponent $a$ | Source | $N$ at $10^{22}$ | $N$ at $10^{24}$ | $D/N$ at $10^{24}$ |
| --- | --- | --- | --- | --- |
| 0.46 | Chinchilla approach 3 | 8.4B | 70B | 34 |
| 0.50 | Chinchilla approach 1 | 9.2B | 92B | 20 |
| 0.54 | Hypothetical error | 10.1B | 121B | 11 |
| 0.73 | Kaplan et al. | 15.6B | 449B | 0.8 |

At 10 times the budget, 0.46 and 0.54 differ by only about ±10%; at 1000 times, the answer ranges from 70B to 121B and the tokens per parameter from 34 to 11. The Kaplan row is only illustrative: attaching its exponent to someone else's anchor point is not rigorous, but it shows what an exponent difference of 0.2 means.

## Why Kaplan and Chinchilla disagree

Kaplan et al. (2020) found $N_{\rm opt}\propto C^{0.73}$ (using compute adjusted to the critical batch size): ten times the compute should buy more than five times the parameters but only about twice the data. Chinchilla (2022) found 0.46 to 0.50 across three approaches. Under the first, the GPT-3 generation's 175B parameters on 300B tokens made sense; under the second, the same compute should go to a smaller model that sees more data. Using Gopher's budget ($5.76\times10^{23}$ FLOPs), Chinchilla trained a 70B model on 1.4T tokens that beat the 280B Gopher in their evaluations, with 67.5% on MMLU, more than 7 points above Gopher (self-reported).

Three analyses explain where the gap comes from:

| Cause | Who pointed it out | Mechanism |
| --- | --- | --- |
| The learning-rate schedule did not match the training length; the models were small | Chinchilla §2, Appendix B | All of Kaplan's runs used the same fixed-length cosine decayed to zero, so losses read off midway were too high and underestimated what training on less data achieves; Chinchilla found that a cosine cycle more than 25% longer than the actual run clearly hurts the loss. Most of Kaplan's runs were also much smaller, many under 100M parameters |
| Counting only non-embedding parameters, and fitting at small scale | [Pearce and Song (2024-06, TMLR)](https://arxiv.org/abs/2406.12907) | Embeddings are a large share of small models; simulating the Chinchilla study under Kaplan's conditions gives a biased exponent close to 0.73 |
| Last-layer compute, warmup length, optimizer settings tuned per scale | [Porian et al. (2024-06, NeurIPS 2024)](https://arxiv.org/abs/2406.19146) | On OpenWebText2 they first reproduce 0.864 under Kaplan's setup (Kaplan's fit against unadjusted compute was 0.88); counting last-layer FLOPs brings it to 0.699; shortening warmup with model size, 0.603; matching cosine to the run length, 0.574 (little change); finally tuning batch size, learning rate and AdamW's $\beta_2$ per model size, with a constant learning rate, gives 0.518 |

Porian et al. also found something that differs from Chinchilla's hypothesis: the final step used a constant learning rate, with no decay, and still recovered Chinchilla's exponent, so careful learning-rate decay is not required for the law to hold. They also found that at small batch sizes (128 sequences or fewer), $\beta_2=0.95$ works poorly, and results improved only after adding 0.99 and 0.999 to the sweep.

Chinchilla's Appendix D.4 ran a direct comparison at $10^{21}$ FLOPs: Kaplan's formula calls for 4.68B parameters and approach 1 for 2.86B; they trained 4.74B and 2.80B models, and the smaller one reached the lower final training loss.

## Where "20 tokens per parameter" comes from

Chinchilla's Table 3 lists compute-optimal training lengths from approach 1: 400M parameters with 8.0B tokens, 1B with 20.2B, 10B with 205.1B, 67B with 1.5T. Divide and each is close to 20; that is the origin of "20 tokens per parameter". Chinchilla itself, 70B parameters on 1.4T tokens, is exactly 20.

The number comes with conditions:

- **Data seen about once.** The data is MassiveText. Appendix C repeats the IsoFLOP analysis on C4 and GitHub code and still finds exponents close to 0.5; the authors' reading is that the result does not depend on the dataset as long as training stays within one epoch.
- **Units.** $N$ includes embeddings, and tokens are counted with their SentencePiece tokenizer. A different tokenizer gives a different token count for the same text, and the ratio moves with it.
- **Recipe.** A cosine decayed by 10x with the cycle matched to the run length, and a smoothed training loss.
- **Fitting method.** Approach 3 (the parametric fit) prefers smaller models: 40B rather than 67B at Gopher's budget. [Besiroglu et al. (2024-04)](https://arxiv.org/abs/2404.10102) reconstructed the data from the paper's figures and refit approach 3, getting $E=1.82$, $A=482$, $B=2085$, $\alpha=0.348$, $\beta=0.366$, which implies about 20 tokens per parameter, consistent with approaches 1 and 2. They also point out that the original confidence intervals are implausibly narrow: intervals that tight would need about 600,000 experiments, while the original probably ran fewer than 500.
- **Data quality.** [DeepSeek LLM (2024-01)](https://arxiv.org/abs/2401.02954) fitted three datasets and got exponents $a$ of 0.450 (early in-house data), 0.524 (current in-house data) and 0.578 (OpenWebText2). Their reading is that the higher the data quality, the more of any extra compute should go to the model.

### How flat it is near the optimum

Below, Besiroglu et al.'s refit $L=1.82+482/N^{0.348}+2085/D^{0.366}$ serves as a **teaching surface**: it was fitted to data reconstructed from the Chinchilla paper's figures and describes only that data and recipe; the numbers in the table are our own calculations on this surface, not anyone's measurements. Look at the cost of missing the optimal size at $C=10^{22}$:

| Model size / optimum | Parameters | $D/N$ | Loss above optimum | Compute needed to reach the optimal loss |
| --- | --- | --- | --- | --- |
| 0.25 | 2.3B | 326 | 0.040 | 4.5x |
| 0.5 | 4.5B | 82 | 0.010 | 1.26x |
| 1 | 9.0B | 20 | 0 | 1x |
| 2 | 18.1B | 5 | 0.010 | 1.15x |
| 4 | 36.2B | 1.3 | 0.041 | 1.59x |

Between half and twice the optimal size, the loss is only about 0.01 higher, and 15% to 26% more compute makes up the difference. Kaplan et al. (Figure 12) reached a similar conclusion: models between 0.6 and 2.2 times the optimal size need only about 20% more compute. [The Llama 3 report (2024-07)](https://arxiv.org/abs/2407.21783) also observed that IsoFLOP curves get flatter around the minimum as the budget grows, so the flagship's performance is not very sensitive to how compute is split between parameters and data. Read "20" as the center of a range.

## How far to trust a small-model fit

A fit is reliable for interpolating near the range it was fitted on; the further you extrapolate, the more the following matter:

- **The curve bends.** Chinchilla's Appendix E finds slight curvature in the compute–loss frontier: fits to the first, middle and last thirds of the points extrapolate differently, and the authors suggest the optimal model at large budgets may be even smaller than a straight line predicts.
- **The law eventually breaks.** Kaplan et al. §6.3 compute a contradiction point themselves: extrapolating $L(C_{\min})$, around $10^{4}$ PF-days and $10^{12}$ parameters, falls below the loss that the slow growth of data allows; they note the location is very sensitive to the exponents and may be off by an order of magnitude.
- **Noise propagates into the exponent.** Each minimum comes from a parabola, and each parabola from a few runs. Kaplan et al. estimate that changing the random seed moves the loss by about 0.02.
- **Loss is not capability.** Loss extrapolates smoothly. Downstream accuracy can appear to jump, and part of that "emergence" comes from metrics that are step functions ([Schaeffer et al., 2023-04](https://arxiv.org/abs/2304.15004)).

The public success stories are all self-reported by the labs. [The GPT-4 technical report (2023-03)](https://arxiv.org/abs/2303.08774) says that fitting $L(C)=aC^b+c$ on models trained with at most one ten-thousandth of GPT-4's compute predicted the final loss accurately, right after the run started; models with at most one thousandth of the compute predicted the pass rate on a subset of HumanEval; the same section admits that some capabilities remain hard to predict. DeepSeek LLM says small experiments accurately predicted the bits-per-byte of its 7B and 67B models at 1000 times the compute. Llama 3 used a two-step method (compute to the negative log-likelihood of the correct answer, then to accuracy) to extrapolate ARC Challenge across four orders of magnitude, and slightly underestimated the final 405B result.

| What you want to do | How | What to observe | What you cannot conclude |
| --- | --- | --- | --- |
| Choose $N$ and $D$ | IsoFLOP at 3–5 budgets with 5–10 sizes each, schedule matched to run length | Whether the minima are clear, whether log–log is straight, how large the residuals are | That the point 1000x out is the optimum |
| Estimate extrapolation error | Bootstrap the minima, or refit without the smallest or largest budget | Intervals on the exponent and on the prediction | That a narrow interval proves the law |
| Predict a large run's loss | Fit $L(C)=E+A\,C^{-\gamma}$ and compare with the first few checkpoints after launch | Whether the actual curve falls inside the interval | That matching loss means matching downstream scores |
| Change data or tokenizer | Redo at least one set of IsoFLOP runs on the new recipe | How much the exponent and constants moved | That the paper's 20 still applies |

Next: [why today's models train far longer than Chinchilla](beyond-chinchilla.en.md), which looks at where the optimum moves once inference cost and data limits enter the accounting.
