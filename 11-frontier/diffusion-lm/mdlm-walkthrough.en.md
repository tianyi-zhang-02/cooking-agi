# MDLM: from a masked training example to a generated sentence

[中文](mdlm-walkthrough.md) · **English**

> Reading time: about 10 minutes · Last reviewed: 2026-10

During training, the model predicts missing words. At generation time, it starts with an entirely blank sequence. How does one lead to the other? A dataset with just two sentences is enough to work through training, sampling, and caching.

This note covers [MDLM](https://arxiv.org/abs/2406.07524), the Masked Diffusion Language Model introduced in 2024. It uses masked diffusion. The sentences and probabilities below are teaching examples, not a claim that all diffusion language models use the same sampler. For the general training objective, see [what masked diffusion trains](masked-diffusion.en.md).

## A tiny language with two sentences

Suppose the training data contains only these two sentences, each with probability one half. Treat each word and the comma as one token, ignoring real tokenizer behavior.

```text
Slot   1     2      3    4    5    6
A     cats  chase  dogs  ,   dogs  run
B     dogs  chase  cats  ,   cats  run
```

Positions 2, 4, and 6 are fixed. Positions 1 and 3 differ; positions 3 and 5 match. Learning the word frequencies is not enough: the model needs these relationships too.

Compare two ways to generate sentence A. Autoregressive (AR) generation appends choices to a prefix; this MDLM example progressively fills six fixed positions.

| Generation method | Known at this step | Prediction |
| --- | --- | --- |
| Ordinary AR generation | `cats chase` | The probability that the next token is `dogs` |
| MDLM denoising | `cats [M] [M] , [M] run` | Word distributions at each blank, using information on both sides |

This compares **generation**. AR training already has the complete sentence and can compute next-token losses in parallel with a causal mask; it need not run word by word. An MDLM training example also typically uses just one masked input. The multiple rounds occur during generation.

## One training update does not unroll the whole trajectory

Choose sentence A and sample a time $t$. Under a linear masking schedule, replace each token with `[M]` independently with probability $t$. At $t=0.5$, each position has a half chance of being masked; the sequence need not have exactly three masks.

Suppose this draw produces:

```text
Input   cats  [M]   [M]   ,   [M]   run
Target   —   chase  dogs  —   dogs   —
```

Run the network once, predict the missing original tokens, compute the loss, and backpropagate. The next training example can use another sentence, time, and mask pattern. There is no need to generate a complete response before obtaining a training signal.

| Masked position | Correct token | Assumed probability assigned to it |
| --- | --- | --- |
| 2 | `chase` | 0.8 |
| 3 | `dogs` | 0.7 |
| 5 | `dogs` | 0.9 |

Using the continuous-time objective from the previous note, this example contributes the weighted loss:

$$
\widehat{\mathcal L}=\frac{-\log 0.8-\log 0.7-\log 0.9}{0.5}
\approx 1.3704.
$$

Averaging over all six positions gives about $0.2284$. Do not silently divide by the number of masks instead: that count is random, and the change alters this estimator. Visible positions have no direct reconstruction loss, but they still supply context. Their associated parameters can receive gradients through predictions at masked positions.

## Where the 1/t weight comes from

Consider position 3. It is currently `[M]`, and its original token is `dogs`. Moving from time $t$ to a cleaner time $s<t$, the conditional probability of revealing it under the linear schedule is:

$$
r=P(z_s\ne [M]\mid z_t=[M])=\frac{t-s}{t}.
$$

Given the original sentence, the true reverse distribution has two outcomes: remain `[M]` with probability $1-r$, or restore `dogs` with probability $r$. The model does not know the original token, so it distributes that second mass according to its clean-token prediction $f_\theta$: `dogs` receives $r f_\theta(\text{dogs})$, and other words can receive mass too.

In the KL between these distributions, the term for remaining masked is zero. What remains is:

$$
D_{\mathrm{KL}}(q\|p_\theta)
=r\log\frac{r}{r f_\theta(\text{dogs})}
=\frac{t-s}{t}[-\log f_\theta(\text{dogs})].
$$

Summing small time steps and taking the continuous-time limit gives the $1/t$ weight. It follows from the reverse process rather than an arbitrary difficulty adjustment. The full objective is a variational upper bound on negative log-likelihood, not the exact NLL. [MDLM §3](https://arxiv.org/html/2406.07524v2#S3)

## Predicting a word does not mean revealing it yet

MDLM's SUBS, or substitution-based parameterization, imposes two constraints: the **clean-token prediction** assigns no mass to `[M]`, and revealed tokens are carried over unchanged. Keep these two distributions separate:

| Distribution | Question it answers | Can it return `[M]`? |
| --- | --- | --- |
| Clean-token prediction $f_\theta$ | What word belongs here in the completed text? | No |
| Next state $p_\theta(z_s\mid z_t)$ | What occupies this position after this step? | Yes; it may remain unrevealed |

For $t=0.5\to s=0.25$, $r=0.5$. Suppose the clean prediction at a masked position is `dogs: 0.7`, `cats: 0.2`, and all other words together have probability `0.1`:

| Next state | Calculation | Probability |
| --- | --- | --- |
| `[M]` | $1-r$ | 0.50 |
| `dogs` | $r\times0.7$ | 0.35 |
| `cats` | $r\times0.2$ | 0.10 |
| Other words | $r\times0.1$ | 0.05 |

The original MDLM ancestral sampler reveals positions randomly according to the schedule. It does not select a few highest-confidence positions at each step. Confidence ranking is a different decoding choice, discussed in [decoding and speed](decoding-and-speed.en.md). Compare [§4.1 of the paper](https://arxiv.org/html/2406.07524v2#S4.SS1) with the [official sampler](https://github.com/kuleshov-group/mdlm/blob/master/diffusion.py).

## Why not fill every position at once?

With an all-mask input, a perfect model's marginal distribution at positions 1, 3, and 5 assigns one half to `cats` and one half to `dogs`: considered separately, each position is equally likely to contain either word. Independent sampling at these three positions gives $2^3=8$ equally likely combinations, only two of which belong to our dataset. **Perfect per-position probabilities still give only $2/8=25\%$ valid sentences when sampled independently in one step.**

If position 1 is revealed as `cats` first, the next step has new information. We now need the conditional distribution given that choice: in this tiny dataset, positions 3 and 5 must both be `dogs`. A perfect model can fill both correctly.

```text
t=1.00  [M]   [M]   [M]   [M]  [M]   [M]
t=0.50  cats  [M]   [M]    ,   [M]   run
t=0.25  cats  chase [M]    ,   [M]   run
t=0.00  cats  chase dogs   ,   dogs  run
```

This is one possible random trajectory, not a prescribed order. The endpoint $s=0$ is idealized for the example; implementations typically avoid numerical endpoints and apply a final denoising step. The 25% result applies only to this dataset and the one-step independent sampler, not to an MDLM benchmark.

Bidirectional attention shares the current context across positions. Within one forward pass, however, no position knows what the others will **actually sample afterward**. Multiple rounds expose those realized choices to later predictions; they need not proceed left to right.

## If nothing is revealed, one network call may be unnecessary

Sometimes a sampling step reveals no new positions. The input is unchanged. If the network is also independent of time, its clean-token predictions are unchanged too. Reuse them and update only the reveal probability.

| Change | Can the complete prediction be reused unchanged? |
| --- | --- |
| Same tokens; time-independent network | Yes, with the rest of the inference state unchanged |
| One `[M]` becomes `cats` | No; the context for other positions has changed |
| Same tokens; different time supplied to the network | Token equality alone is insufficient |

This is different from AR's layerwise KV cache. The official `ddpm_cache` path clears the prediction cache when the input changes or time conditioning is enabled; see `_sample` and `_ddpm_caching_update` in `diffusion.py`. It does not establish that a visible token's deeper-layer KV stays fixed. [Official code](https://github.com/kuleshov-group/mdlm/blob/master/diffusion.py)

The basic sampler also cannot revise committed tokens. If it has already produced `cats chase cats`, extra rounds do not guarantee a repair. Visible training tokens come from the original text; visible generation tokens come from the model's choices. Incorrect context therefore deserves a separate evaluation.

Three useful implementation checks follow from this example: preserve the masking weight in the loss, check that transition probabilities sum to one, and reuse predictions only when the input and network conditions permit it. Continue with [steps and actual speed](decoding-and-speed.en.md), then [what context reuse saves](context-reuse.en.md).
