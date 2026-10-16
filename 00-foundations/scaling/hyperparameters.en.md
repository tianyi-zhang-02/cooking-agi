# Scaling: how learning rate, batch size and schedule scale up

[中文](hyperparameters.md) · **English**

> Reading time: ~13 min · Draft, pending review

Sweep the best learning rate on a narrow small model, copy it unchanged to a model 8 times wider, and a common outcome is unstable training or a clearly suboptimal result. There is more than one reason: under the standard parametrization the optimal learning rate moves with width; the useful batch size limit grows as the loss falls and as the data grows; and a cosine schedule's shape is tied to the total training length. **Scaling up means handling three things separately. For batch size, look at the critical batch size first: past it, each doubling of the batch saves fewer steps and costs more data. For learning rate, either start from a fitted empirical formula for your scale, or use a parametrization such as µP that makes it transferable across width. For the schedule, either match it to the training length, or use warmup-stable-decay to postpone deciding how long to train until the end.**

> **Draft, pending review.** The site owner has not reviewed this note yet; facts, emphasis and wording may change. Check the primary sources linked in the text before relying on it.

It helps to read [one training step](../deep-dives/training-step.en.md) and [from SGD to AdamW](../deep-dives/optimizers.en.md) first: we use optimizer steps, gradient accumulation and Adam's update rule. The teaching budget is the $C=10^{22}$ FLOPs from [note 1](compute-optimal.en.md).

## Batch size: when bigger stops paying

How many tokens go into one update determines two things: how many steps you take (wall-clock time) and how much data you see in total (compute). [McCandlish et al. (2018-12)](https://arxiv.org/abs/1812.06162) give a simple model: to reach a fixed loss, the number of steps $S$ and the number of examples processed $E=BS$ satisfy

$$
\left(\frac{S}{S_{\min}}-1\right)\left(\frac{E}{E_{\min}}-1\right)=1,\qquad B_{\rm crit}=\frac{E_{\min}}{S_{\min}}.
$$

$S_{\min}$ is the minimum number of steps with an infinitely large batch, and $E_{\min}$ the minimum data with a very small batch. Rearranged: $S=S_{\min}(1+B_{\rm crit}/B)$ and $E=E_{\min}(1+B/B_{\rm crit})$. They argue that $B_{\rm crit}$ is roughly the gradient noise scale, whose simplified form is $\mathcal B_{\rm simple}={\rm tr}(\Sigma)/|G|^2$: the summed variance of per-example gradients divided by the squared length of the true gradient. The larger the noise relative to the signal, the more a big batch is worth for averaging it out.

Teaching numbers: $S_{\min}=10{,}000$ steps and $B_{\rm crit}=2\text{M}$ tokens, so $E_{\min}=20$B tokens.

| Batch (tokens) | Steps $S$ | Tokens processed $E$ | Relative to min steps | Relative to min data |
| --- | --- | --- | --- | --- |
| 0.5M | 50,000 | 25B | 5.0x | 1.25x |
| 1M | 30,000 | 30B | 3.0x | 1.5x |
| 2M | 20,000 | 40B | 2.0x | 2.0x |
| 4M | 15,000 | 60B | 1.5x | 3.0x |
| 16M | 11,250 | 180B | 1.1x | 9.0x |

Below $B_{\rm crit}$, doubling the batch nearly halves the steps with little extra data, so more GPUs buy time; above it, going from 4M to 16M saves only 25% of the steps but costs 3 times the data. Exactly at $B_{\rm crit}$, both steps and data are twice their minimums.

$B_{\rm crit}$ is not a constant. Kaplan et al. fit $B_{\rm crit}(L)\approx2\times10^{8}/L^{1/0.21}$ tokens ($L$ is the loss on WebText2): the critical batch doubles for roughly every 13% drop in loss, and as the loss falls from 3 to 2 it grows from about 1.1M to about 7.4M tokens. That is why many large runs start with a smaller batch and increase it: Llama 3 405B starts at 4M tokens, doubles to 8M after 252M tokens and to 16M after 2.87T tokens; DeepSeek-V3 raises its batch size from 3072 to 15360 over the first 469B tokens (the report gives no unit; with its 4K pretraining sequence length these are presumably sequences, about 12.6M to 62.9M tokens).

What does it mostly depend on? [Zhang et al. (2024-10, ICLR 2025)](https://arxiv.org/abs/2410.21676) trained 85M to 1.2B models on C4, defined the critical batch as the batch where steps exceed linear scaling by 20%, and then separated model size from data size: scaling both together at Chinchilla ratios, the critical batch grows roughly as $D^{0.47}$; fixing the data and growing only the model, the exponent drops to 0.087; fixing a 302M model and adding only data, the critical batch still grows. So the growth comes mainly from the amount of training data $D$, not from model size. [Bergsma et al. (2025-05, NeurIPS 2025)](https://arxiv.org/abs/2505.13738) reach a similar conclusion: the optimal batch scales roughly as $D^{0.4}$ and the critical batch roughly as $D^{0.5}$, independent of model size. How a batch is split across GPUs is covered in [multi-GPU training](../../06-systems/distributed-training.en.md).

## Learning rate: change it with batch size and scale

When the batch changes, the learning rate usually has to change too. Two common conversion rules:

| Rule | Source | When the batch is multiplied by $\kappa$ | Scope |
| --- | --- | --- | --- |
| Linear | [Goyal et al. (2017-06)](https://arxiv.org/abs/1706.02677) | Multiply the learning rate by $\kappa$, with warmup early on | SGD; they used batches up to 8,192 images on ImageNet |
| Square root | [Malladi et al. (2022-05)](https://arxiv.org/abs/2205.10287) | Multiply the learning rate by $\sqrt\kappa$; multiply $1-\beta_1$ and $1-\beta_2$ by $\kappa$; divide $\epsilon$ by $\sqrt\kappa$ | Adam / RMSprop; the derivation assumes gradient noise dominates the gradient, and the error bound loosens as $\kappa$ grows |

Example: AdamW at a batch of 1M tokens, learning rate $3\times10^{-4}$ and $\beta_2=0.95$, moving to a batch of 4M ($\kappa=4$). The square-root rule gives a learning rate of $6\times10^{-4}$ and $\beta_2=1-4\times0.05=0.80$; the linear rule gives $1.2\times10^{-3}$. A $\beta_2$ of 0.80 is rare in large-model pretraining, which shows that the rule is derived for the noise-dominated, small-batch regime and should not be applied blindly once the batch is near the critical value.

The other route is to fit directly. [DeepSeek LLM (2024-01)](https://arxiv.org/abs/2401.02954) ran grid searches from $10^{17}$ to $2\times10^{19}$ FLOPs, counted any configuration within 0.25% of the best loss as "near-optimal", and fitted

$$
\eta_{\rm opt}=0.3118\,C^{-0.1250},\qquad B_{\rm opt}=0.2920\,C^{0.3271}.
$$

Here $C=M\cdot D$, where $M$ is the non-embedding FLOPs per token (including the attention computation), and $B$ is in tokens. At $C=10^{22}$: $\eta_{\rm opt}=0.3118\times10^{-2.75}\approx5.5\times10^{-4}$ and $B_{\rm opt}=0.2920\times10^{7.196}\approx4.6\text{M}$ tokens. The paper itself stresses that the near-optimal region is wide, so this is only a starting point, and that at the same compute a different split between model and data moves the optimal region. Bergsma et al. argue that fitting batch size against $C$ is misleading because it mainly depends on $D$. They also use AdamW's timescale $\tau=B/(\eta\lambda D)$ ($\lambda$ is weight decay) to set weight decay: at fixed $N$ and $D$ the optimal $\lambda$ is proportional to $B$, so doubling the batch should double the weight decay; the optimal $\tau$ itself falls as a power law in tokens per parameter, roughly $(D/N)^{-0.5}$. The Qwen3 report (2025-05) also says it fitted scaling laws to predict the learning-rate schedule and batch size for each pretraining stage, without publishing the coefficients.

## Schedule: cosine needs the length up front, WSD can decide it last

Cosine is the most common shape: after warmup, the learning rate follows

$$
\eta(t)=\eta_{\min}+\tfrac12(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

from $\eta_{\max}$ down to $\eta_{\min}$, an annealing shape that comes from [SGDR](https://arxiv.org/abs/1608.03983). Chinchilla decays by 10x, so $\eta_{\min}=0.1\eta_{\max}$: at $t/T=0.5$ the learning rate is $0.55\eta_{\max}$, and at $t/T=0.8$ it is $0.19\eta_{\max}$. The trouble is that $T$ has to be fixed in advance. Plan for 100B tokens but stop at 50B, and the learning rate is still $0.55\eta_{\max}$ with no annealing, so the loss is higher than a run planned for 50B from the start. Chinchilla's Appendix B finds that a cycle more than 25% longer than the actual run clearly hurts, and that decaying by 10x beats decaying by only 5x. Part of the Kaplan bias discussed in note 1 comes from here.

Warmup-stable-decay (WSD) splits the schedule into three parts: warmup, a long stretch at $\eta_{\max}$, and a short, fast decay at the end. [MiniCPM (2024-04)](https://arxiv.org/abs/2404.06395) coined the name and found that a decay phase of 10% of the total tokens is enough while 2.5% is not; [Hägele et al. (2024-05, NeurIPS 2024)](https://arxiv.org/abs/2405.18392) found that a cooldown 20% long matches cosine, and that a $1-\sqrt{\cdot}$ shaped decay beats a linear one. [Wen et al. (2024-10)](https://arxiv.org/abs/2410.05192) assume the loss landscape looks like a river valley and use it to explain why: a large learning rate makes the parameters bounce between the valley walls while moving along the river at the bottom, so the loss seems to fall slowly; the fast decay damps the bouncing and reveals the distance already travelled along the river.

WSD's advantage is that the training length can be decided at the end. Teaching example: you want to compare three training lengths, 25B, 50B and 100B tokens.

| Approach | How to run it | Total training tokens |
| --- | --- | --- |
| Cosine | One run per length | 25 + 50 + 100 = 175B |
| WSD, 20% cooldown | One constant-learning-rate trunk to 80B, branching at 20B, 40B and 80B with decays of 5B, 10B and 20B | 80 + 5 + 10 + 20 = 115B |

That saves about a third, and the more lengths you compare, the more it saves. Hägele et al. estimate that reusing runs with cooldowns would have cut the compute of Chinchilla's sweep from about $5.59\times10^{23}$ to $2.36\times10^{23}$ FLOPs.

Recipes used for large models mostly sit between the two:

| Model | Learning-rate shape | Source |
| --- | --- | --- |
| DeepSeek LLM 7B / 67B | 2,000 warmup steps; drop to 31.6% of peak after 80% of tokens and to 10% after 90% | 2024-01 report §2.3 |
| Llama 3 405B | 8,000 warmup steps to $8\times10^{-5}$, cosine to $8\times10^{-7}$ over 1.2M steps; linear decay to 0 over the last 40M tokens | 2024-07 report §3.4 |
| DeepSeek-V3 | 2,000 warmup steps to $2.2\times10^{-4}$, held until 10T tokens, then cosine to $2.2\times10^{-5}$ over 4.3T tokens; in the last 500B tokens, held at that value for 333B, then dropped to $7.3\times10^{-6}$ for 167B | 2024-12 report §4.2 |
| Kimi K2 | Called WSD in the report: 500 warmup steps, constant $2\times10^{-4}$ for the first 10T tokens, then cosine to $2\times10^{-5}$ over 5.5T tokens | 2025-07 report |
| OLMo 2 | After warmup, cosine to 10% of peak (truncated at 4T tokens for 7B), then linear decay to 0 during mid-training | 2024-12 report |

## µP: making the learning rate transferable across width

Under the standard parametrization (SP: the framework's default initialization and one learning rate for all layers), the optimal learning rate moves with width. [Tensor Programs V (Yang et al., 2022-03; TP5 below)](https://arxiv.org/abs/2203.03466) trained MLPs on CIFAR-10 with SGD and saw the optimal learning rate shift by roughly an order of magnitude as width went from 256 to 8192; using the narrowest model's optimum on the widest model performed very badly or diverged. Transformers behave the same way.

Why does it move? Take one hidden layer $h=Wx$ with an $n\times n$ matrix $W$ and an input $x$ whose entries are $O(1)$. After bias correction, Adam's first update is approximately $\Delta W_{ij}=-\eta\,{\rm sign}(g_{ij})$. With batch size 1, $g_{ij}=\delta_ix_j$ ($\delta_i$ is the gradient of the loss with respect to $h_i$), so

$$
(\Delta W x)_i=-\eta\,{\rm sign}(\delta_i)\sum_{j=1}^{n}|x_j|\approx-\eta\,n\,\mathbb E|x|.
$$

The key is that the update is aligned with $x$: $n$ terms of the same sign add up, so the size grows linearly in $n$. If the update direction were unrelated to $x$, positive and negative terms would cancel and the size would grow only as $\sqrt n$. By hand with every $x_j=\pm1$: at $\eta=0.01$ and $n=512$, $|\Delta h_i|=5.12$; the same learning rate at $n=4096$ gives 40.96, so the activation changes 8 times more in one step. To keep it at 5.12, the hidden-layer learning rate must be multiplied by $512/4096$, giving $0.00125$. That is the intuition behind µP setting Adam's hidden-layer learning rate $\propto1/\text{fan\_in}$. Recomputing once with Gaussian inputs:

```python
import random

def one_step_change(width, lr, seed=0):
    rng = random.Random(seed)
    x = [rng.gauss(0, 1) for _ in range(width)]          # previous-layer activations, entries O(1)
    update = [-lr * (1 if xj > 0 else -1) for xj in x]   # Adam's first step: -lr * sign(delta * x_j), with delta > 0
    return abs(sum(u * xj for u, xj in zip(update, x)))  # |(dW x)_i|

for width in (512, 4096):
    print(width, round(one_step_change(width, 0.01), 2),            # SP: same learning rate at every width
          round(one_step_change(width, 0.01 * 512 / width), 2))     # muP: hidden learning rate times base/width
# 512 4.17 4.17
# 4096 32.78 4.1
```

With Gaussian inputs $\mathbb E|x|=\sqrt{2/\pi}\approx0.80$, so the predictions are $0.01\times512\times0.80\approx4.1$ and $0.01\times4096\times0.80\approx32.7$, matching the output; an unaligned random direction would give only $\eta\sqrt n$, that is 0.23 and 0.64.

Table 3 of TP5 gives the differences between µP and SP under Adam (only the scaling with width is shown; constants can be added as multipliers):

| Parameter | Initialization variance | Adam learning rate | Difference from SP |
| --- | --- | --- | --- |
| Input weights (embedding) and all biases | $1/\text{fan\_in}$ | Does not change with width | None |
| Hidden weights (attention projections, MLP) | $1/\text{fan\_in}$ | $\propto1/\text{fan\_in}$ | Learning rate shrinks with width |
| Output weights (logits) | $1/\text{fan\_in}^2$ | $\propto1/\text{fan\_in}$ | Smaller initialization, learning rate shrinks with width |
| Attention logit | Use $q^\top k/d$ | Not applicable | SP uses $q^\top k/\sqrt d$ |

### Work one transfer by hand

The proxy model has width 512, and a sweep on it finds a global Adam learning rate $\eta=0.01$ (plus output multipliers and so on). The target has width 4096, a width ratio $m=8$. Following the table, and writing the width factors relative to the proxy as $m=n/n_0$ the way the paper's Eq. 4 does: embeddings, biases and norm gains keep learning rate 0.01; hidden matrices use $0.01/8=0.00125$, with initialization standard deviation still $1/\sqrt{\text{fan\_in}}$; the output layer takes initialization variance $1/(\text{fan\_in}\cdot m)$, 8 times smaller than SP, and learning rate $0.00125$; the attention logit becomes $\sqrt{d_0}\,q^\top k/d$ ($d$ is the head dimension and $d_0$ the proxy's head dimension; if widening only adds heads and keeps the head dimension, this term does not change). At $m=1$ all these factors are 1, so the proxy is an ordinary SP model and trains as usual. The paper also gives two equivalent forms: Table 8 keeps the output initialization variance constant in width and multiplies the forward pass by $1/\text{fan\_in}$ instead, so all vector-like parameters share one rule, which is easier to implement and lets input and output embeddings be tied; Table 9 follows the notation of the earlier TP4 paper.

[MiniCPM](https://arxiv.org/abs/2404.06395) uses the same kind of scheme: it searches on a 0.009B model, scales the learning rate of 2-D weights by $1/(d_m/d_{\rm base})$ and the initialization standard deviation by $1/\sqrt{d_m/d_{\rm base}}$; as the model grew from 0.04B to 0.5B, the optimal base learning rate stayed around 0.01, and a check at 2.1B also gave 0.01 (self-reported). The TP5 authors report that hyperparameters tuned on a 40M-parameter proxy, used to train a 6.7B GPT-3, beat the published numbers for the original 6.7B model, at a tuning cost of about 7% of total pretraining compute (self-reported by the authors).

Keep the limits in mind:

- **What does not transfer.** TP5's Table 1 lists regularization such as dropout and weight decay as not transferable; transfer across batch size, training length, sequence length and depth is only validated empirically.
- **Depth.** [Tensor Programs VI (2023-10, TP6)](https://arxiv.org/abs/2310.02244) gives Depth-µP for residual blocks with a single layer, and points out fundamental limitations in the infinite-depth limit for multi-layer blocks like those in Transformers.
- **Not only µP.** [Everett et al. (2024-07, ICML 2024)](https://arxiv.org/abs/2407.05872) trained tens of thousands of models (up to 26.8B) and found that all four parametrizations they compared, not only µP, can achieve hyperparameter transfer; their per-layer learning-rate rule for SP even beat µP. Adam's $\epsilon$ must also be scaled correctly with width, or gradients underflow.
- **A new optimizer needs its own derivation.** Muon scales its update by matrix shape so that the per-element update RMS does not change with matrix size, which is another approach; see [Muon](../deep-dives/muon.en.md).

## A checklist before launch

| What you want to do | How | What to observe | What you cannot conclude |
| --- | --- | --- | --- |
| Choose the batch | At small scale, measure steps and tokens to reach the same loss for several batch sizes | Whether doubling the batch nearly halves the steps | That the critical batch stays fixed for the whole run |
| Choose the learning rate | Sweep 5–7 values at 2x spacing on the proxy and check the optimum is not at the edge | The location and width of the minimum in loss versus learning rate | That the SP optimum carries over to a model 8 times wider |
| Check a µP implementation | Short runs at 3 widths with the same learning rate, logging each layer's activation size | Whether each layer's activation RMS stays stable across widths | That transfer across width implies transfer across depth |
| Choose the schedule | If the length is not fixed, use WSD and branch off decays at several lengths | The drop in loss across the cooldown | That the loss during the constant phase reflects final quality |

Finally: [review questions](review.en.md).
