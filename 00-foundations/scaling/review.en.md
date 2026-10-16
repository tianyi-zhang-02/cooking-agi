# Scaling: review questions

[中文](review.md) · **English**

> Reading time: ~5 min · Draft, pending review

This page condenses the series into interview questions and a self-check; every number in the answers can be traced to the first three notes, and the hand-worked parts use teaching numbers.

> **Draft, pending review.** The site owner has not reviewed this note yet; facts, emphasis and wording may change. Check the primary sources linked in the text before relying on it.

## Interview questions

<details class="interview" markdown="1">
<summary>Where does $C\approx6ND$ come from? With 64 H100s for 4.6 days, how big a model does Chinchilla suggest?</summary>

For each training token a dense Transformer does about $2N$ FLOPs in the forward pass (one multiply-add per weight) and about twice that in the backward pass, $6N$ in total, multiplied by the number of tokens $D$. It drops the attention term that grows with context; Kaplan's $N$ excludes embeddings, Chinchilla's includes them. The budget is GPUs × peak × MFU × time: $64\times989.4$ TFLOPS (BF16 dense) $\times0.4\times3.95\times10^{5}$ seconds $\approx10^{22}$ FLOPs. At 20 tokens per parameter, $N=\sqrt{C/120}\approx9.1\times10^{9}$ and $D\approx1.8\times10^{11}$. Using the sparse peak would overstate the budget by a factor of two.

</details>

<details class="interview" markdown="1">
<summary>Why do Kaplan and Chinchilla give different optimal allocations?</summary>

Kaplan (2020) found $N_{\rm opt}\propto C^{0.73}$; Chinchilla (2022) found 0.46–0.50. Chinchilla points out that all of Kaplan's runs used one fixed-length cosine, so when losses were read off midway the learning rate had not yet decayed, the losses were too high, and the effect of less data was underestimated; Kaplan's models were also small. Pearce and Song show that counting only non-embedding parameters and fitting at small scale pushes the exponent up. Porian et al. start from 0.864 on OpenWebText2 and, correcting last-layer compute, warmup, cosine decay and per-scale hyperparameters (including AdamW's $\beta_2$) in turn, get down to 0.518, and find that Chinchilla's result holds even without learning-rate decay.

</details>

<details class="interview" markdown="1">
<summary>Under what conditions does "20 tokens per parameter" hold, and how costly is it to miss?</summary>

It comes from Table 3 of Chinchilla's approach 1 (for example, 1B parameters with 20.2B tokens). Conditions: MassiveText, data seen about once; $N$ includes embeddings and tokens use their tokenizer; a cosine decayed 10x and matched to the run length. The original approach-3 fit favours smaller models, but Besiroglu et al.'s refit also returns to about 20; DeepSeek LLM found that data quality moves the exponent from 0.450 to 0.578. The optimum is flat: on the refit surface at $10^{22}$ FLOPs, a model half or twice the optimal size has a loss only about 0.01 higher, and 15%–26% more compute makes up the gap.

</details>

<details class="interview" markdown="1">
<summary>How do you fit a scaling law from a few small runs, and how far can you trust the extrapolation?</summary>

IsoFLOP: 3–5 small budgets, 5–10 sizes each, schedule matched to run length; fit a parabola to loss versus $\log N$, take its minimum, then fit a line through the minima on log–log axes. The teaching points from $(10^{18},9\times10^7)$ to $(10^{21},2.9\times10^9)$ give a slope of about 0.50 and extrapolate to about 9.2B parameters at $10^{22}$. An exponent error of ±0.04 changes a 10x extrapolation by only about 10%, but a 1000x extrapolation ranges from 70B to 121B. Risks: the frontier bends (Chinchilla Appendix E), the law must fail somewhere (Kaplan §6.3), seed noise is about 0.02, and smooth loss does not mean smooth downstream metrics. The successful predictions by GPT-4, DeepSeek LLM and Llama 3 are all self-reported.

</details>

<details class="interview" markdown="1">
<summary>Why train Llama 3 8B on 15T tokens? How does inference volume change the optimum?</summary>

Chinchilla minimizes training compute only. Sardana et al. fix a target loss and minimize $6ND_{\rm tr}+2ND_{\rm inf}$: the more inference, the smaller the model and the more data. On the refit surface, targeting the loss of the compute-optimal model at $10^{22}$ FLOPs, $D_{\rm inf}=10^{12}$ gives an optimum of about 4.6B parameters and 447B tokens: 24% more training compute, 23% less in total. Sardana et al. also found that fits on typical ratios overestimate the value of data at extreme ratios, so the savings at very large inference volumes are optimistic upper bounds. The Llama 3 report self-reports that the smaller models were trained far past compute-optimal and do better at the same inference budget, while 405B sits close to its own fitted optimum (402B, 16.55T).

</details>

<details class="interview" markdown="1">
<summary>When data runs short, what is repeated training worth?</summary>

Muennighoff et al. convert repetition into effective data: $D'=U_D+U_DR_D^*(1-e^{-R_D/R_D^*})$ with $R_D^*\approx15.4$. 100B unique tokens trained for 4 epochs is worth about 372.6B (93%); 16 epochs about 1.06T (66%); the ceiling is about 16.4 times. The paper concludes that at fixed compute, repeating about 4 times is nearly as good as new data (an 8.7B model at 4 epochs had only about 0.5% higher validation loss), after which returns fall fast. The decay constant for excess parameters, $R_N^*\approx5.3$, is smaller, so when data is limited more passes beat a bigger model.

</details>

<details class="interview" markdown="1">
<summary>What is the critical batch size? How should batch size and learning rate be converted together?</summary>

McCandlish et al.: $S=S_{\min}(1+B_{\rm crit}/B)$, $E=E_{\min}(1+B/B_{\rm crit})$, $B_{\rm crit}=E_{\min}/S_{\min}$, roughly the gradient noise scale. With the teaching value $B_{\rm crit}=2$M, going from 4M to 16M saves only 25% of the steps but costs 3 times the data. It grows as loss falls (Kaplan: it doubles for about every 13% drop in loss), and recent work finds it grows mainly with data size (Zhang et al.: with data fixed, growing only the model gives an exponent of just 0.087). Conversion: the linear rule for SGD; for Adam the square-root rule (learning rate times $\sqrt\kappa$, $1-\beta$ times $\kappa$), valid only in the noise-dominated small-batch regime. DeepSeek LLM's fit can serve as a starting point: about $5.5\times10^{-4}$ and 4.6M tokens at $10^{22}$ FLOPs.

</details>

<details class="interview" markdown="1">
<summary>What problem does µP solve? Why should the hidden-layer learning rate shrink with width? What does not transfer?</summary>

Under the standard parametrization (SP) the optimal learning rate moves with width (about an order of magnitude from width 256 to 8192 for TP5's MLP). Adam's first update is about $-\eta\,{\rm sign}(g)$ and is aligned with the input $x$, so $(\Delta Wx)_i\approx\eta\,n\,\mathbb E|x|$: with $x=\pm1$ and $\eta=0.01$, the change is 5.12 at width 512 and 40.96 at width 4096. µP makes the Adam learning rate of hidden and output layers $\propto1/\text{fan\_in}$, sets the output initialization variance to $1/\text{fan\_in}^2$, and uses $1/d$ attention, so the hidden learning rate at width 4096 is 0.00125. What does not transfer: regularization such as dropout and weight decay; depth, batch size and training length are only validated empirically, and multi-layer Transformer blocks face fundamental limits in depth (TP6). Everett et al. found that SP with the right per-layer learning rates also transfers.

</details>

## Self-check

<div class="taste-check">
  <strong>If you really understand it, you should be able to explain:</strong>
  <ol>
    <li>Given GPU count, peak, MFU and days, how do you get to Chinchilla's $N$ and $D$ in one pass? Which step most easily doubles the budget by mistake?</li>
    <li>Why does Kaplan's fixed cosine length make models trained on less data look worse and push the exponent up?</li>
    <li>If the fitted exponent is off by 0.04, how large is the effect at 10x and at 1000x extrapolation? Why does the flatness near the optimum make this less alarming?</li>
    <li>As inference tokens grow from 0 to $10^{13}$, how does the optimal model change? Why are the savings on the right of the table optimistic?</li>
    <li>When width grows 8 times under SP, by how much does Adam's first step change the activations? How does µP bring it back, and what can it not guarantee?</li>
  </ol>
</div>

## References

- [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162): gradient noise scale and critical batch size
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361): power laws, $C\approx6ND$, $N\propto C^{0.73}$
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556): Chinchilla's three approaches and about 20 tokens per parameter
- [Chinchilla Scaling: A replication attempt](https://arxiv.org/abs/2404.10102): refitting approach 3 and its confidence intervals
- [Reconciling Kaplan and Chinchilla Scaling Laws](https://arxiv.org/abs/2406.12907), [Resolving Discrepancies in Compute-Optimal Scaling](https://arxiv.org/abs/2406.19146): where the two laws diverge
- [DeepSeek LLM](https://arxiv.org/abs/2401.02954): compute fits for learning rate and batch size; data quality changes the allocation
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783): its own IsoFLOP fit, downstream prediction and training recipe
- [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774): predictable scaling (self-reported)
- [Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004): metric choice and "emergence"
- [LLaMA](https://arxiv.org/abs/2302.13971), [Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448): choosing the model by inference cost
- [Language models scale reliably with over-training and on downstream tasks](https://arxiv.org/abs/2403.08540): extrapolating over-training and downstream error
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264): effective data under repetition
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437), [OLMo 2](https://arxiv.org/abs/2501.00656), [SmolLM2](https://arxiv.org/abs/2502.02737), [Gemma 3](https://arxiv.org/abs/2503.19786), [Qwen3](https://arxiv.org/abs/2505.09388), [Kimi K2](https://arxiv.org/abs/2507.20534), [DeepSeek-V4 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) and [report](https://arxiv.org/abs/2606.19348), [Llama 3 model card](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md): training volumes of open models
- [Accurate, Large Minibatch SGD](https://arxiv.org/abs/1706.02677), [On the SDEs and Scaling Rules for Adaptive Gradient Algorithms](https://arxiv.org/abs/2205.10287): linear and square-root conversion rules
- [How Does Critical Batch Size Scale in Pre-training?](https://arxiv.org/abs/2410.21676), [Power Lines](https://arxiv.org/abs/2505.13738): how batch size scales with data
- [SGDR](https://arxiv.org/abs/1608.03983), [MiniCPM](https://arxiv.org/abs/2404.06395), [Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations](https://arxiv.org/abs/2405.18392), [Understanding Warmup-Stable-Decay Learning Rates](https://arxiv.org/abs/2410.05192): cosine and WSD
- [Tensor Programs V](https://arxiv.org/abs/2203.03466), [Tensor Programs VI](https://arxiv.org/abs/2310.02244), [Scaling Exponents Across Parameterizations and Optimizers](https://arxiv.org/abs/2407.05872): µP, depth transfer and other parametrizations
