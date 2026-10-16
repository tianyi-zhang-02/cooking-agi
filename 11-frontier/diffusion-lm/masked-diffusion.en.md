# Diffusion language models: what masked diffusion trains

[中文](masked-diffusion.md) · **English**

> Reading time: ~13 min · Draft, pending review

BERT also hides a few words and guesses them. Yet if you take a trained BERT, start from an input that is all `[MASK]` and keep filling in blanks, you do not get a decent paragraph; LLaDA uses a Transformer without a causal mask and a cross-entropy loss too, and it can hold a conversation. The difference is not the network but two details: **each training example draws its own masking ratio $t$, anywhere from almost nothing masked to everything masked, and the loss on masked positions is multiplied by $1/t$. The resulting objective is exactly "the autoregressive (AR) loss averaged over all generation orders", and it is an upper bound on the negative log-likelihood (NLL). The price is that the model has to learn every order, including the hard ones.**

> **Draft, pending review.** The site owner has not reviewed this note yet; facts, emphasis and wording may change. Check the primary sources linked in the text before relying on it.

This note covers only the training objective. How the blanks are filled step by step, and when that beats AR decoding, is in [the next note](decoding-and-speed.en.md).

## The forward process: each token is masked with probability t

Write an original sequence of length $L$ as $x_0=(x_0^1,\dots,x_0^L)$. Masked diffusion (also called absorbing-state discrete diffusion: once a position becomes `[M]` it never changes back) has a forward process with a single rule: at time $t\in[0,1]$, each position is independently replaced by `[M]` with probability $t$ and kept with probability $1-t$, giving $x_t$. At $t=0$ you have the original text; at $t=1$ every position is blank. [D3PM](https://arxiv.org/abs/2107.03006) (Austin et al., 2021-07) lists this absorbing-state transition as one kind of discrete diffusion and points out its connection to mask-based generative models; [MDLM](https://arxiv.org/abs/2406.07524) (Sahoo et al., 2024-06, NeurIPS 2024) and [LLaDA](https://arxiv.org/abs/2502.09992) (Nie et al., 2025-02) both use this form.

Reuse the sentence from [the BERT note](../../00-foundations/core/bert.en.md), `the tea is cold` ($L=4$). The number of masked positions follows a binomial distribution:

| Positions masked | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| $t=0.25$ | 0.316 | 0.422 | 0.211 | 0.047 | 0.004 |
| $t=0.5$ | 0.0625 | 0.25 | 0.375 | 0.25 | 0.0625 |

On average $tL$ positions are masked: 1 at $t=0.25$, 2 at $t=0.5$. There is no continuous noise level here as in image diffusion; "how much noise" simply means "how many blanks". For how continuous diffusion moves step by step in a latent space, see [flow matching in the Qwen-Image note](../../03-multimodal-learning/image-generation.en.md).

## The objective: cross-entropy on masked positions only, times 1/t

LLaDA §2.1, Eq. (3), writes it as:

$$
\mathcal{L}(\theta)=\mathbb{E}_{t\sim U(0,1]}\;\mathbb{E}_{x_t\sim q(x_t\mid x_0)}\Big[\frac{1}{t}\sum_{i=1}^{L}\mathbf{1}\big[x_t^i=\texttt{[M]}\big]\,\big(-\log p_\theta(x_0^i\mid x_t)\big)\Big]
$$

$p_\theta(\cdot\mid x_t)$ is a Transformer without a causal mask: it reads the whole partially masked sequence and outputs a vocabulary distribution at every position; $\mathbf{1}[\cdot]$ means only masked positions are scored. LLaDA's Eq. (4) states that this quantity upper-bounds the negative log-likelihood $-\mathbb{E}\log p_\theta(x_0)$. The model does not need to know $t$: the number of blanks is already visible in the input. In MDLM's experiments, dropping time conditioning leaves perplexity about the same; Theorem 1 of [RADD](https://arxiv.org/abs/2406.03736) (Ou et al., 2024-06) explains why: the conditional distributions the model has to learn do not depend on $t$ in the first place.

Compute it once on two hand-made examples (the probabilities are teaching numbers, not model outputs):

- Draw $t=0.5$, masking `tea` and `cold`: the input is `the [M] is [M]`. Suppose the model gives $p(\texttt{tea})=0.4$ and $p(\texttt{cold})=0.5$. The cross-entropy is $0.916+0.693=1.609$; times $1/t=2$, this example's loss is 3.219.
- Draw $t=0.25$, masking only `tea`: the input is `the [M] is cold`. Suppose $p(\texttt{tea})=0.6$. The cross-entropy is 0.511; times 4, it gives 2.043.

The second example has a single blank, more context and a more confident model, yet it is multiplied by 4. The reason: at small $t$ only $tL$ positions contribute on average, and after multiplying by $1/t$ every $t$ carries an expected "supervision weight" of $tL/t=L$ tokens. Without the factor, lightly masked examples would carry little weight in the total loss, yet the last steps of generation are made in exactly that state, with one or two blanks left.

Other masking schedules give the same result. MDLM writes $\alpha_t$ for the fraction of original tokens kept at time $t$; in continuous time the weight is $-\alpha_t'/(1-\alpha_t)$, which is exactly $1/t$ for the linear schedule $\alpha_t=1-t$. MDLM §3.4 also notes that the continuous-time objective does not depend on the shape of $\alpha_t$. This follows from a change of variables: a different schedule only changes how often each masking ratio is drawn during training, so it affects the variance of the estimate, not the value of the objective.

## How it differs from BERT

| | BERT's masked language modeling (MLM) | Masked diffusion (MDLM, LLaDA) | Autoregressive |
| --- | --- | --- | --- |
| How much is masked | A fixed ~15%, with an 80/10/10 corruption of selected positions | Each example draws $t\sim U(0,1]$, from almost nothing to everything | Nothing; a causal mask hides the right side |
| How the loss is weighted | Average over selected positions | Times $1/t$, giving an upper bound on the negative log-likelihood | $-\log p$ at every position, which sums to the exact negative log-likelihood |
| Trained to start from all blanks? | No | Yes: $t=1$ is all blanks | Not needed: it writes one token at a time from the left |
| Positions supervised per example | About 15% | Half on average | All $L$ |
| Attention | Bidirectional | Bidirectional | Causal |

BERT's masking details are in [the three masks section of the BERT note](../../00-foundations/core/bert.en.md#three-masks). MDLM §3.5 writes the diffusion objective as a "weighted average" of MLM losses and says this connects generative diffusion models with encoder-only BERT models. LLaDA §2.1 also notes that MaskGIT's objective, used for images, drops the $1/t$ and so has no link to maximum likelihood. "A diffusion LM is a BERT that can generate" is therefore only half right: the network can stay the same, but the distribution of masking ratios and the weighting have to change.

## By hand: with two tokens, the objective is the average of two orders

Take the two-token sequence `hot tea`. The model is only ever asked four kinds of question (the probabilities are teaching numbers again):

| Question the model answers | Teaching probability | Cross-entropy (nats) |
| --- | --- | --- |
| Both blank; position 1 is `hot` | $p(\texttt{hot}\mid\texttt{[M] [M]})=0.2$ | $\ell_1=1.609$ |
| Both blank; position 2 is `tea` | $p(\texttt{tea}\mid\texttt{[M] [M]})=0.1$ | $\ell_2=2.303$ |
| `tea` known; position 1 is `hot` | $p(\texttt{hot}\mid\texttt{[M] tea})=0.5$ | $\ell_{1\mid2}=0.693$ |
| `hot` known; position 2 is `tea` | $p(\texttt{tea}\mid\texttt{hot [M]})=0.8$ | $\ell_{2\mid1}=0.223$ |

At a given $t$, both positions are masked with probability $t^2$, and only the first or only the second is masked with probability $t(1-t)$ each. Integrate the objective over $t$:

$$
\int_0^1\frac{1}{t}\Big[t^2(\ell_1+\ell_2)+t(1-t)\,\ell_{1\mid2}+t(1-t)\,\ell_{2\mid1}\Big]dt=\tfrac12\big(\ell_1+\ell_{2\mid1}\big)+\tfrac12\big(\ell_2+\ell_{1\mid2}\big)
$$

The $1/t$ cancels one factor of $t$, and the remaining $\int_0^1 t\,dt$ and $\int_0^1(1-t)\,dt$ are both $1/2$. The first bracket on the right is the AR loss written left to right: first `hot`, then `tea` after `hot`, $1.609+0.223=1.833$. The second is right to left: $2.303+0.693=2.996$. Their average, 2.414, is the diffusion objective on this sequence. **With the $1/t$, the objective is exactly the average loss over the two generation orders.** Drop the $1/t$ and the integral becomes $(\ell_1+\ell_2)/3+(\ell_{1\mid2}+\ell_{2\mid1})/6=1.457$, which is not an average over any orders and has no likelihood interpretation.

Why is it an upper bound? The continuous-time sampler fills each position at an independent, uniformly distributed moment, which is the same as generating in a random order. This model generates `hot tea` with probability $\tfrac12(0.2\times0.8)+\tfrac12(0.1\times0.5)=0.105$, a negative log-likelihood of 2.254, below 2.414. The 0.16-nat gap comes from the two orders disagreeing; if the four conditional probabilities came from one joint distribution, both orders would give the same number and the bound would be tight. An AR model trains only the left-to-right order, with loss 1.833.

```python
import math
l1, l2 = -math.log(0.2), -math.log(0.1)      # both blank: guess position 1, position 2
l1_2, l2_1 = -math.log(0.5), -math.log(0.8)  # the other position known
objective = 0.5 * (l1 + l2) + 0.5 * (l1_2 + l2_1)       # diffusion objective after integrating
left_to_right, right_to_left = l1 + l2_1, l2 + l1_2
generator_nll = -math.log(0.5 * 0.2 * 0.8 + 0.5 * 0.1 * 0.5)
print(round(objective, 3), round((left_to_right + right_to_left) / 2, 3), round(generator_nll, 3))
# 2.414 2.414 2.254
```

The same holds at any length. RADD's Theorem 2 proves that the absorbing-diffusion training objective is equivalent to the any-order autoregressive objective: for a uniformly random permutation $\pi$, predict the tokens one by one in the order $\pi$ and take the expected sum of negative log-likelihoods. Earlier, [ARDM](https://arxiv.org/abs/2110.02037) (Hoogeboom et al., 2021-10, ICLR 2022) had already placed any-order autoregressive models and absorbing diffusion in one model family.

## What "any order" means, and what it costs

The phrase has two meanings. In training, the objective averages over all orders; in the example above each order gets half the weight. At inference, blanks can be filled in any order: the most confident position first, using content already known on the right, or filling a gap in the middle.

Published results show both sides:

- **The hard orders must be learned too.** [Kim et al. (2025-02)](https://arxiv.org/abs/2502.06768) show theoretically and empirically that masked diffusion has to solve computationally intractable subproblems during training that AR models avoid; choosing the order adaptively by confidence at inference sidesteps them, raising a pretrained model's Sudoku accuracy from under 7% to about 90%, ahead of AR models with 7 times as many parameters trained with teacher forcing on the right order.
- **Context on the right helps.** LLaDA §3.3 tests 496 pairs of lines from Chinese poems. Writing the first line from the second: LLaDA 8B Instruct 45.6, GPT-4o 34.3. Writing the second from the first: LLaDA 51.8, GPT-4o 82.7. Better in reverse, much worse forward.
- **Training costs more compute.** In IsoFLOP experiments from $6\times10^{18}$ to $10^{20}$ FLOPs, [Nie et al. (2024-10)](https://arxiv.org/abs/2410.18514) found that masked diffusion needs about 16 times the compute of an AR model to reach the same validation loss.
- **Less supervision per example, higher variance.** Only half the positions carry a loss on average. In [Block Diffusion](https://arxiv.org/abs/2503.09573) (Arriola et al., 2025-03, ICLR 2025), the diffusion objective with block size 1 equals the AR objective in expectation, yet trained that way it reaches an LM1B perplexity of ≤25.56 against 22.88 for AR; switching to full masking brings it back to 22.88. The authors attribute the gap to the high variance of the training objective.
- **Real decoding is not arbitrary.** When LLaDA 8B Instruct switches to block-wise, left-to-right sampling, GSM8K rises from 69.4 to 78.6 and MATH from 31.9 to 42.2 (§3.2); the authors' explanation is that the SFT data is padded with many EOS tokens, so low-confidence remasking makes the model write the ending too early. The third-party [Asaria et al. (2026-06)](https://arxiv.org/abs/2606.14620) recorded which positions DiffusionGemma commits at each step and found a weak left-to-right tendency (clearer the coarser the analysis), with large batches of positions often committed at once.

When data is scarce, learning every order becomes an advantage. [Prabhudesai et al. (2025-07)](https://arxiv.org/abs/2507.15857), with 7M to 2.5B parameters and data repeated over many epochs, found that repeating the same data is about as good as fresh data for roughly 4 epochs for AR models but for about 100 epochs for masked diffusion. They attribute this to random masking acting as training over many orders, a form of data augmentation.

## LLaDA: scaling the objective to 8B

LLaDA (submitted 2025-02, updated to v3 in 2025-10) trained an 8B model from scratch: 2.3T tokens, 0.13M H800 GPU hours, sequence length 4096, with 1% of the data cut to a random length between 1 and 4096 (§2.2). Because it cannot use a KV cache (see the next note), the authors used plain multi-head attention instead of GQA and reduced the FFN dimension to keep the parameter count. SFT uses 4.5 million prompt-response pairs and masks only the response (Eq. 5); short examples are padded with EOS, which is trained as an ordinary token, so the model learns where to stop.

**Check what the scalability claim compares.** The authors trained AR baselines of similar size on the same data (the 8B point is compared with a 7B AR model), from $10^{20}$ to $10^{23}$ FLOPs, compared them on six downstream tasks, and concluded that the overall trend is competitive with AR models and stronger on MMLU and GSM8K (§3.1). This is the authors' own experiment and compares downstream scores. Nie et al.'s "about 16 times" compares validation loss, at a much smaller scale. The two results do not contradict each other, but neither alone answers "for the same money, which is better".

**The comparison with LLaMA3 8B is not a controlled experiment.** Base models: MMLU 65.9 versus 65.4, GSM8K 70.3 versus 48.7 (Table 1); the instruction models go the other way: MMLU 65.5 versus 68.4, GSM8K 69.4 versus 78.3 (Table 2). The two were trained on different data; LLaDA's instruction model had SFT only, while LLaMA3 8B Instruct had SFT and RL. Note the decoding setting as well: in Appendix B.6 the instruction model's number of sampling steps equals the answer length, 512 steps for 512 tokens on GSM8K, MATH and HumanEval, one token per step. These scores were measured without any parallel speedup.

As of October 2026, most later large models are converted from AR weights: [Dream 7B](https://arxiv.org/abs/2508.15487) (2025-08) is initialized from an AR model; [LLaDA2.0](https://arxiv.org/abs/2512.15745) (2025-12) converts Ling-mini-2.0 and Ling-flash-2.0 into 16B and 100B MoE models, growing the training block size step by step from 1 to 4096 (the whole sequence) and then shrinking it step by step to 32, and self-reports average scores of 64.34 versus 65.77 for the original model (16B) and 73.18 versus 72.15 (100B); the [DiffusionGemma model card](https://ai.google.dev/gemma/docs/diffusiongemma/model_card), released by Google on 2026-06-10, says it is based on the 26B A4B MoE Gemma 4 architecture and lists it below the same-size Gemma 4 on most benchmarks, for example AIME 2026 at 69.1% versus 88.3% and MMLU Pro at 77.6% versus 82.6%.

## Alternatives: where the noise goes

| Approach | Where the noise goes | What the model learns | Examples | Main cost |
| --- | --- | --- | --- | --- |
| Autoregressive | No noise; the right side is hidden | Distribution of the next token | GPT, LLaMA | Only one token per step |
| Continuous diffusion | Gaussian noise on word vectors | A denoising direction in continuous space | [Diffusion-LM](https://arxiv.org/abs/2205.14217) (2022-05) | Vectors must be rounded back to tokens at the end |
| Score entropy | Transitions between discrete tokens | Ratios of the distribution (concrete score) | [SEDD](https://arxiv.org/abs/2310.16834) (2023-10, ICML 2024) | Needs time conditioning |
| Masked (absorbing) | Tokens replaced by `[M]` | The original token at masked positions | MDLM, LLaDA, LLaDA2.0 | Committed tokens are not revised by default |
| Uniform-state | Tokens replaced by random words | The original token at every position | [Sahoo et al. (2026-02)](https://arxiv.org/abs/2602.15014); by Google's account, DiffusionGemma | Likelihood is usually worse |

Uniform-state diffusion has no `[M]` in its input: every position holds a real word, which the model can change. Sahoo et al. (2026-02) found at 1.7B that uniform-state diffusion has worse validation perplexity yet beats AR and masked diffusion models on GSM8K, so they argue perplexity is only a fair comparison within one family.

**Which kind is DiffusionGemma?** Google's [developer guide](https://developers.googleblog.com/diffusiongemma-the-developer-guide/) (2026-06-10) says it starts from a span of random placeholder tokens and uses Uniform State Diffusion, and that a token whose confidence drops can be renoised and replaced; the model card says only that tokens not selected are "fully" renoised. The third-party Asaria et al. (2026-06-12) call it masked discrete diffusion in their abstract. This site follows the official description and files it under uniform-state. Google has not published a technical report, and the exact noise distribution is not public.

## What to check when you try it

| Try first | What to observe | What you can NOT conclude |
| --- | --- | --- |
| Check an implementation on the $L=2$ example (the code above) | The objective equals the average of the two orders | The objective is the model's exact negative log-likelihood |
| Log the average loss per $t$ bucket during training | Loss grows with $t$; buckets near $t=0$ have the highest variance | $1/t$ is the only workable weight (schedules can change; they affect variance) |
| Compare perplexity with an AR model | Diffusion gives an upper bound (≤); AR gives the exact value | The difference is how much "worse" diffusion really is |
| Compare across diffusion families | Within one family, perplexity and downstream results broadly agree | Between masked and uniform-state, perplexity decides which is better |
| Evaluate generations with generative perplexity | Also report token diversity or entropy | Lower is better: [Zheng et al. (2024-09, ICLR 2025)](https://arxiv.org/abs/2409.02908) found that categorical sampling at 32-bit precision lowers the effective temperature, biasing this metric toward low-diversity samples |

Next: [decoding steps, caching and real speed](decoding-and-speed.en.md).
