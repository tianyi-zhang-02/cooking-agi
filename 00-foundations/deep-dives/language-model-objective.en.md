# Language-model objectives, training, and generation

[中文](language-model-objective.md) · **English**

> Reading time: ~9 min · Level: advanced · Last reviewed: 2026-10

One answer has 20 valid tokens and another has 200. Averaging over all tokens gives the second answer ten times as many loss terms. Averaging within each answer first gives the answers equal weight. **Before changing any architecture, the reduction already changes whose examples dominate learning.** [The numbers are worked out below](#dissection-one-cross-entropy-does-not-decide-what-matters-for-you).

## Why do training and generation run at different rhythms? {#the-core-question-why-training-and-generation-run-at-different-rhythms}

Training and generation use the same parameterized distribution:

$$p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t})$$

The only difference is whether the future is known. In training the whole ground-truth sequence is already laid out on the table; in generation the next token does not exist yet, so the model can only write one, look at it, and then keep writing. The [roofline](../../12-infra/performance.en.md#roofline) explains the split in the table's “main bottleneck” row.

| | training / prefill | autoregressive decode |
| --- | --- | --- |
| Input | a whole span of known tokens | the one or few most recently generated tokens |
| attention | parallel under a causal mask | queries the historical KV cache |
| Main bottleneck | compute, activation memory | memory bandwidth, cache, number of serial steps |
| Sources of error | data and objective | additionally, sampling and error accumulation |

## Cross-entropy does not decide “what matters” for you {#dissection-one-cross-entropy-does-not-decide-what-matters-for-you}

The loss at a single position:

$$\ell_t=-\log p_\theta(y_t\mid x_{\le t})$$

Here $y_t=x_{t+1}$ is the true token after position $t$. The gradient with respect to the logits is $p-e_{y_t}$, where $e_{y_t}$ is the one-hot vector for $y_t$; the formula is clean. But once all positions are summed, frequent tokens and long samples naturally hold more “votes”. So data mixing, sample weighting, and loss masking are not leftovers in the corner of a training script — they are what defines whom the model should actually care about.

Return to the two answers from the opening. Suppose the short answer (20 tokens) has a mean per-token loss of 1.0 and the long answer (200 tokens) has 2.0:

| How you average | Calculation | Result |
| --- | --- | ---: |
| Over all tokens together | $(20\times1.0+200\times2.0)/220=420/220$ | about 1.91 |
| Within each answer, then over samples | $(1.0+2.0)/2$ | 1.5 |

Under the token mean, the long answer carries $200/220\approx91\%$ of the weight, so the result is almost its own 2.0; under the per-sample mean, each answer counts for half. Which one fits depends on what you want the model to care about; the formula does not choose for you. Gradient accumulation across microbatches raises the same question; see [one training update](training-step.en.md#token-mean).

Perplexity is the exponential of the average token negative log-likelihood:

$$\text{PPL}=\exp\!\left(\frac{1}{N}\sum_t \ell_t\right)$$

Plug in two numbers. With a mean NLL of 2 nats per token, PPL is $e^2\approx7.39$, roughly like choosing among 7.39 equally likely tokens at every step. If the model gives every one of the $V$ vocabulary tokens probability $1/V$, each step's NLL is $\ln V$ and PPL is exactly $V$.

Different tokenizers have different token units, so perplexity cannot be compared directly across them. Take a teaching example (not a measurement): the same 100-byte text, and two models that both assign it a total NLL of 100 nats, differing only in how their tokenizers split it:

| Tokenizer splits it into | NLL per token | PPL | Bits per byte (bpb) |
| --- | ---: | ---: | ---: |
| 25 tokens | 4 nats | $e^4\approx54.6$ | $100/(100\ln2)\approx1.44$ |
| 50 tokens | 2 nats | $e^2\approx7.39$ | $100/(100\ln2)\approx1.44$ |

Both models give the text the same probability, yet their PPL differs by more than 7 times: finer splits make each token easier to guess. Converting the total NLL to bits and dividing by the byte count fixes the unit to the same bytes; the [GPT-2 report §3.1](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) likewise reports results per fixed unit such as a character, byte, or word. Bpb has limits too: under [UTF-8's encoding rules](https://www.rfc-editor.org/rfc/rfc3629#section-3), an ASCII character takes 1 byte and most CJK characters take 3. The same content written in another language has a different byte count, so cross-language bpb comparisons still need care.

## Training uses the true prefix; generation depends on the model's own output {#dissection-two-training-uses-the-true-prefix-generation-depends-on-the-models-own-output}

In training the model always predicts on top of the true prefix (teacher forcing); in generation it must continue from its own output. This mismatch is commonly called exposure bias; see [Seq2Seq](../core/seq2seq.en.md) for a worked example. A low-probability error at one step can carry the subsequent context into a region that the training data rarely covers.

It is very easy here to say, by reflex, “then use RL”. Before that, check where the problem actually comes from:

1. the training data does not cover the target behavior;
2. the loss mask or the template is wrong;
3. the sampling strategy is unsuitable (to tell 1 from 3, score a reference answer: a low log-likelihood suggests training never covered it; a reasonable likelihood with bad samples points to temperature, top-p, and similar settings);
4. a long-horizon objective cannot be expressed by per-token likelihood.

The fourth category points most directly to a sequence-level preference or an RL objective; if the problem is exactly that the model's own outputs carry the context away from the training distribution, training on the model's own prefixes (on-policy data, RL, and similar) is also worth considering.

## A cache must match the full forward pass {#the-correctness-floor-a-cache-is-not-fine-as-long-as-it-runs-it-must-be-equivalent}

Incremental decoding must agree with a one-shot forward pass within numerical tolerance. A small error in positions, the mask, or K/V order can still produce fluent sentences while the logits have already changed, so compare logits step by step rather than judging whether the text looks right. The item-by-item checklist is in the [KV cache note](kv-cache-and-inference.en.md#cache-checks), and [`../code/test_model.py`](../code/test_model.py) has the matching equivalence test.

<span id="self-check"></span><span id="quick-learning-one-factorization-two-execution-paths"></span>

## What post-training actually changes {#what-post-training-actually-changes}

SFT, preference learning, and RL do not switch to a different kind of model. On the same autoregressive distribution they change: where the training samples come from, which tokens count toward the loss, how different outputs are compared, and how the return is assigned across a whole trajectory.

Continue with [Post-Training](../../05-post-training/README.en.md).
