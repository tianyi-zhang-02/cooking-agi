# Quantization: how to measure what was lost

[中文](checking-quality.md) · **English**

> Reading time: ~9 min · Last reviewed: 2026-10

When [Dutta et al. (2024-07, NeurIPS 2024)](https://arxiv.org/abs/2407.09141) compressed Llama2-70B-chat to W4A16 with GPTQ, MMLU accuracy fell by only 0.45 points, yet 4.26% of the answers flipped between right and wrong. When [Marchisio et al. (2024-07)](https://arxiv.org/abs/2407.03211) compressed the 103B Command R+ to 4-bit weights with group scales, automatic evaluations in Japanese dropped by only 1.7% on average, while human evaluators on realistic prompts saw a 16.0% drop.

An average score alone misses the most common quality losses from quantization. Evaluating a quantized model requires three complementary checks: compare next-token distributions on fixed text using KL divergence and top-1 agreement; compare answers to the same questions item by item, counting flips and examining the interval around the net change; and check the slices most prone to degradation, including long context, non-English, math reasoning and long outputs.

The statistics of paired comparison (McNemar's test, the paired bootstrap) are worked out in [The score improved. What does that actually establish?](../../07-evaluation/metric-robustness.en.md#paired-binary); which fields of the evaluation setup to record is in [Before choosing a benchmark](../../07-evaluation/benchmark-protocols.en.md). This note covers only what to watch for when applying them to quantization.

## Several rulers, each measuring something different

| Metric | How it is computed | What it is good for | What it cannot see |
| --- | --- | --- | --- |
| Perplexity | the mean negative log-likelihood per token, exponentiated | cheap and sensitive to quantization | large errors at a few positions get averaged away; optimistic when calibration data overlaps the evaluation text |
| KL divergence, top-1 agreement | compare the two models' distributions position by position on the same text | measures directly how far you are from the original model; needs no labels | does not say which kind of task is hurt, or in which direction |
| Task accuracy | one score per model | close to real use | when the net change is small, it cannot see right and wrong answers trading places |
| Flipped answers (flips) | right-to-wrong plus wrong-to-right, divided by the number of items | shows directly how much behavior changed | needs both models' per-item results saved |
| Slice scores | long context, each language, math, long outputs reported separately | where things break first | wide intervals when a slice has few items |
| Pairwise comparison of free-form outputs | a judge model or humans compare two answers | closest to what users feel | expensive; judges have their own biases, see [LLM-as-a-Judge](../../07-evaluation/llm-as-a-judge/README.en.md) |

## Perplexity and KL: what the average hides

Perplexity is

$$
\mathrm{PPL}=\exp\Big(-\frac1N\sum_{t=1}^{N}\log p(x_t\mid x_{<t})\Big).
$$

[AWQ](https://arxiv.org/abs/2306.00978) reports Llama-2-7B going from 5.47 in FP16 to 5.60 with INT4-g128, which in mean negative log-likelihood per token is only $\ln(5.60/5.47)\approx0.0235$ nats more. That number cannot say at which positions the loss is concentrated.

KL divergence measures how far apart the two distributions are at the same position, with $p$ the original model and $q$ the quantized one:

$$
D_{\mathrm{KL}}(p\,\|\,q)=\sum_{v}p(v)\log\frac{p(v)}{q(v)}.
$$

**Work out two positions** (teaching numbers, a vocabulary of only 4 tokens):

- Position 1: $p=[0.60,\ 0.25,\ 0.10,\ 0.05]$, $q=[0.52,\ 0.30,\ 0.12,\ 0.06]$. KL is about 0.0129; the most likely token is unchanged.
- Position 2: $p=[0.40,\ 0.38,\ 0.12,\ 0.10]$, $q=[0.37,\ 0.41,\ 0.12,\ 0.10]$. KL is about 0.0023, less than a fifth of position 1's, yet the top two swap places.

Under greedy decoding, position 2 outputs a different token, every later step builds on a different prefix, and the text diverges from here. So report **mean KL** together with **top-1 agreement**, and look at the positions with the largest KL or with the top two close together. Dutta et al. report a Spearman correlation of 0.981 between KL and flip rate on MMLU; for Llama2-13B on WikiText-2, the KL is $7.9\times10^{-5}$ for GPTQ W8A16 and 0.02 for GPTQ W4A16.

[Kurtic et al. (2024-11)](https://arxiv.org/abs/2411.02355) compared answers before and after quantization with greedy decoding on Arena-Hard prompts (FP8, INT8 and INT4 schemes); for the 8B model, ROUGE-L is only 0.46 while BERTScore is 0.92. This shows that literal overlap and embedding similarity can give different assessments. [BERTScore](https://arxiv.org/abs/1904.09675) measures token similarity through contextual embeddings; a high score alone does not guarantee that key facts, negations or answer correctness are preserved. For open-ended answers that allow different wording, exact text agreement should not be the sole quality criterion: compare distributions and task scores as well. Exact matching remains useful when a task requires verbatim quotations, fixed fields or a specified output format; code tasks also need appropriate execution or structural checks.

One more easily missed bias: measure perplexity on text outside the calibration data. [GPTQ](https://arxiv.org/abs/2210.17323) itself notes that its calibration data comes from the C4 training set, so the C4 perplexity it reports is not fully zero-shot. Use held-out text from your own domain, with no overlap with the calibration set.

## Paired comparison: same questions, item by item

Teaching example: 1,000 multiple-choice questions, both the original and the quantized model run on all of them, compared item by item.

| | Quantized right | Quantized wrong |
| --- | ---: | ---: |
| Original right | 690 | 30 |
| Original wrong | 26 | 254 |

Accuracy goes from 72.0% to 71.6%, a net change of −0.4 points; the flip rate is $(30+26)/1000=5.6\%$. With $\ell=30$ items lost and $g=26$ gained, the paired difference and its standard error are:

$$
\hat d=\frac{g-\ell}{n},\qquad \widehat{\mathrm{SE}}(\hat d)=\sqrt{\frac{(g+\ell)/n-\hat d^{\,2}}{n}}.
$$

Plugging in, $\hat d=-0.004$, the standard error is about 0.0075, and the 95% interval is about $[-1.87,\ +1.07]$ points; the exact McNemar p-value is about 0.69. **1,000 questions cannot tell "unchanged" from "down 1.5 points"**, and the 56 flipped items are the ones to read one by one.

```python
from math import comb, sqrt

def paired_report(base_correct, quant_correct):
    pairs = list(zip(base_correct, quant_correct))
    n = len(pairs)
    lost = sum(b and not q for b, q in pairs)
    gained = sum(q and not b for b, q in pairs)
    d = (gained - lost) / n
    se = sqrt(((gained + lost) / n - d * d) / n)
    k = gained + lost
    tail = sum(comb(k, j) for j in range(min(gained, lost) + 1))
    p = min(1.0, 2 * tail / 2 ** k) if k else 1.0
    return {"flips": k / n, "diff": d, "ci95": (d - 1.96 * se, d + 1.96 * se), "p": p}

base = [True] * 720 + [False] * 280
quant = [True] * 690 + [False] * 30 + [True] * 26 + [False] * 254
r = paired_report(base, quant)
assert r["flips"] == 0.056 and abs(r["diff"] + 0.004) < 1e-12
assert abs(r["p"] - 0.6889) < 1e-4
```

Public measurements (Dutta et al., MMLU 5-shot, Table 5; flips count only right-to-wrong and wrong-to-right, not one wrong answer turning into another):

| Model | Quantization | Accuracy change (points) | Flip rate |
| --- | --- | ---: | ---: |
| Llama2-70B-chat | GPTQ W4A16 | −0.45 | 4.26% |
| Llama2-7B-chat | GPTQ W4A16 | −0.58 | 8.26% |
| Llama2-7B-chat | AWQ W4A16 | −0.62 | 7.50% |
| Llama2-7B-chat | SmoothQuant W8A8 | −2.36 | 13.62% |
| Llama2 and Yi, several sizes | GPTQ W8A16 | — | 0.26%–1.40% |

The last row is 8-bit weight-only quantization, with far fewer flips. For generation tasks with sampled decoding, use the same decoding parameters on both sides and run several seeds each; otherwise the randomness of sampling is counted as flips too.

## Slices: look first where things break first

Continuing with the 1,000 questions, suppose 100 of them are long-context items (teaching numbers):

| Slice | Items | Original | Quantized | Change |
| --- | ---: | ---: | ---: | ---: |
| Short input | 900 | 675 (75.0%) | 677 (75.2%) | +0.2 points |
| Long context | 100 | 45 (45%) | 39 (39%) | −6 points |
| Total | 1,000 | 720 (72.0%) | 716 (71.6%) | −0.4 points |

The total drops by only 0.4 points, while the long-context slice drops by 6. The interval is wide with only 100 items, but the result warrants adding questions and investigating further. Define slices in advance: picking the worst of dozens of slices after the fact overstates the problem, as explained in [Controlled comparisons and slices](../../07-evaluation/ablation-and-slices.en.md). In published studies, the following tasks show degradation first:

- **Long context.** [Mekala et al. (2025-05, EMNLP 2025)](https://arxiv.org/abs/2505.20276) used 9.7K test examples from 5 datasets (long inputs meaning over 64K tokens, plus long-output tasks) to test FP8, GPTQ-int8, AWQ-int4, GPTQ-int4 and BNB-nf4 on Llama-3.1 8B and 70B and Qwen-2.5 7B, 32B and 72B: 8 bits lose only 0.2 (FP8) and 0.8 (GPTQ-int8) points on average; 4 bits lose 1.8 (AWQ) to 6.9 (BNB-nf4) points on average; non-English inputs fare worse; and it depends heavily on the model: with the same BNB-nf4 on the same OneRuler benchmark, Llama-3.1 70B loses 59 points while Qwen-2.5 72B barely moves (+0.6). Kurtic et al.'s RULER results are milder: 81.1 for 8B W4A16 against 82.8 for BF16.
- **Multilingual.** Marchisio et al. (Findings of EMNLP 2024) found languages with non-Latin scripts hit hardest and hard tasks such as math degrading fastest; automatic metrics underestimate the degradation people perceive (the 1.7% against 16.0% above).
- **Reasoning models.** [Liu et al. (2025-04, COLM 2025)](https://arxiv.org/abs/2504.04823) tested DeepSeek-R1 distilled models, QwQ-32B and Qwen3-8B on AIME, MATH-500, GSM8K, GPQA and LiveCodeBench: W8A8 or W4A16 can be lossless (within 1%), lower bit widths carry clear risk, outputs do not get longer under mild settings but do get longer under aggressive ones such as W3G128 and W4A4KV4, and model size, model origin and task difficulty are the key factors.
- **Overall.** Kurtic et al. ran more than 500,000 evaluations on Llama-3.1: FP8 is near lossless, well-tuned INT8 loses 1–3%, and W4A16 is comparable to 8 bits. This is a conclusion on average; it does not replace measuring your own slices.

## What to record for every quantization

| What to record | Why |
| --- | --- |
| Versions or hashes of the original and quantized models | so you can reproduce and roll back |
| Quantization method, tool and version; bits, group length, symmetric or not; which layers were left uncompressed (output layer, embeddings, sliding-window layers) | two things both called "INT4" can be done very differently |
| Source, count and length of the calibration data, and whether it overlaps the evaluation text | GPTQ's C4 perplexity is one example |
| The format and scale method for weights, activations and KV separately (static, dynamic, per head) | uncalibrated vLLM KV scales are 1.0 |
| Engine version, attention backend, kernels, GPU model | without hardware support, it silently falls back to W8A16 |
| Evaluation protocol: template, decoding parameters, seeds, context length | must be identical before and after quantization |
| Per-item results for both models; mean KL and top-1 agreement on fixed text | needed to count flips and compute paired intervals |
| Slice scores, output-length distribution, format-error rate, refusal rate | where things break first is often not in the average |
| Latency and throughput at the target batch | quality and speed have to be decided together |

## Common failures: symptom and first check

| Symptom | Likely cause | Check first |
| --- | --- | --- |
| Perplexity barely moves, but users say answers got worse | the average hides a few positions; a short English test set is insensitive | flips; long-context and non-English slices; the positions with the largest KL |
| One domain gets clearly worse while others are fine | the calibration data's distribution differs from that domain | redo it with calibration data from the target domain; AWQ reported GPTQ perplexity rising by 2.3–4.9 across domains |
| Errors start in the second half of long conversations | KV quantization error builds up with length, or some layers (such as sliding window) are more sensitive | turn KV quantization off as a control; use the option to skip sensitive layers |
| Occasional NaNs or individual broken outputs | FP8 out of range (E4M3 becomes NaN in overflow mode); a static scale that is too small | measure the maximum of activations and KV; switch to saturating mode or dynamic scales |
| Memory went down but speed did not change | the hardware lacks the matching Tensor Core and falls back to dequantizing, or the batch is already compute-bound | the kernel actually used, in the engine log; the GPU's compute capability; the batch table in [note 2](weight-only-and-ptq.en.md) |
| Outputs before and after quantization differ word for word | a near-tied token flips and the rest of the text diverges | compare distributions and task scores for open-ended answers; retain exact or structural checks where the task requires precise outputs |

This table is a starting point for debugging, not a diagnosis; each row has to be confirmed with a controlled comparison.

## What this note cannot answer for you

- Acceptable quality loss depends on product requirements. Specify in advance which slices must retain their performance and where some decline is acceptable.
- Statistical power: with the 1,000 questions above, the 95% interval is about 3 points wide; to resolve smaller changes you need more items or a finer paired design.
- A conclusion on one model does not carry over to another: Mekala et al. found that with the same BNB-nf4 on the same OneRuler benchmark, Qwen-2.5 72B barely moved while Llama-3.1 70B lost 59 points.

Finally: [Review](review.en.md).
