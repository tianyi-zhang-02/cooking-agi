# Rejection sampling: generate more, then keep what?

[中文](rejection-sampling.md) · **English**

> Reading time: ~12 min · Type: chapter · Last reviewed: 2026-10

A model produces four solutions to a coding task. Two pass the tests, one passes some tests, and one does not run. You could train on both passing solutions, select the highest-scoring one, or return that solution to the user without training at all.

All of these may appear under the label rejection sampling. They are not the same algorithm. Two questions usually clear things up: **What determines which answers survive? Are those answers returned at inference time, or used to update the model?**

For the training workflow, read the first two sections and [the data records](#data-records). The three small calculations in between explain how selection changes a distribution. Basic probability and [SFT](sft-and-its-ceiling.en.md) are enough to follow them.

## One name, several procedures {#four-uses}

| Procedure | What survives | Parameter updates? |
| --- | --- | --- |
| Classical accept–reject sampling | Samples accepted with a probability chosen to recover a target distribution | Not inherently a training method |
| Verifier filtering | Answers that pass checks or a threshold; possibly several per prompt | Filtering alone does not update weights; SFT may follow |
| Best-of-N | The highest-scoring of N generated candidates | Can be inference-only |
| Rejection-sampling fine-tuning | Selected answers become demonstrations | SFT updates the parameters |

[Llama 2 (2023), §3.2.3](https://arxiv.org/html/2307.09288v2) used multiple candidates per prompt, selected the highest reward, and fine-tuned on those answers. The [initial DeepSeek-R1 report (2025), §2.3.3](https://arxiv.org/html/2501.12948v1#S2.SS3.SSS3) describes retaining correct responses, with additional readability filters. Keeping exactly one reward-model winner is therefore not the definition of every such pipeline. Neither procedure is automatically exact statistical accept–reject sampling.

## What does filtered SFT learn? {#filtered-sft}

The workflow has four parts: fix a generator and sampling configuration, produce candidates, score and filter them, then run SFT on the retained answers. A later iteration can generate a fresh dataset from the updated model.

For the retained dataset $D_{\rm keep}$, first sum the supervised token losses within each answer to get $\ell(x,y)$. Then divide the total across answers by M, the number of supervised tokens:

$$
\ell(x,y)
=-\sum_{t\in\mathcal T_y}\log\pi_\theta(y_t\mid x,y_{<t}).
$$

$$
\mathcal L_{\rm SFT}
=\frac{1}{M}\sum_{(x,y)\in D_{\rm keep}}\ell(x,y).
$$

Here x is the prompt, y the retained response, $\mathcal T_y$ the supervised response positions, and M their total count. Prompt and padding positions do not enter that denominator. Averaging each response first and then averaging responses would give long and short answers different relative weights.

The verifier decides **which demonstrations enter the dataset**. Training still means imitating the retained answers; it needs neither gradients through the verifier nor PPO's policy ratio. Discarded answers are simply absent from this SFT update. They do not directly teach the model what to avoid, though a separate preference objective could use them.

## Filtering changes the distribution {#filter-distribution}

Fix one prompt. Suppose the generator produces three answer categories with probabilities $q=(0.5,0.3,0.2)$. All answers in the first category pass, half in the second pass, and none in the third pass. Their acceptance probabilities are $a=(1,0.5,0)$.

| Category | Original probability q | Acceptance a | Retained mass q × a |
| --- | --- | --- | --- |
| A | 0.50 | 1.00 | 0.50 |
| B | 0.30 | 0.50 | 0.15 |
| C | 0.20 | 0.00 | 0.00 |

The total acceptance rate is 0.65. **Among accepted answers**, A accounts for $0.5/0.65=10/13$, and B for $3/13$. In general:

$$
q_{\rm keep}(y\mid x)
=\frac{q(y\mid x)a(x,y)}{Z(x)}.
$$

$$
Z(x)=\sum_y q(y\mid x)a(x,y).
$$

This requires $Z(x)>0$. It is a conditional distribution, not a guarantee that filtering recovers an ideal answer distribution. False rejections, false acceptances, and solutions the generator rarely produces all matter. Here a summarizes pass rates within categories; a deterministic verifier can instead assign each individual answer an acceptance value of 0 or 1.

## Classical rejection sampling needs an envelope {#classical-rejection}

Suppose the aim is specifically to sample from a known target p. Generate from an easier proposal q and accept y with probability $a(y)=p(y)/(M q(y))$. A finite M must satisfy $p(y)\le Mq(y)$ across the target support. Wherever p is positive, q must also be positive. See the classical derivation in [CMU's Monte Carlo lecture, slide 8](https://www.cs.cmu.edu/~epxing/Class/10708/lectures/lecture16-MC.pdf#page=8).

Take $p=(0.6,0.3,0.1)$ and $q=(1/3,1/3,1/3)$. The smallest valid M is 1.8:

| Category | Acceptance p / (M q) | Proposal mass × acceptance |
| --- | --- | --- |
| A | 1 | 1/3 |
| B | 1/2 | 1/6 |
| C | 1/6 | 1/18 |

The total acceptance rate is $5/9=1/M$. Normalizing the retained masses gives exactly $(0.6,0.3,0.1)$. A larger valid M rejects more proposals but leaves the target unchanged. An acceptance probability above 1 means the envelope condition failed; clipping it to 1 does not preserve the exact-sampling guarantee.

An unnormalized target density also works with a suitable envelope. Its acceptance rate is the target's normalizing constant divided by the envelope constant, not simply $1/M$. Typical LLM filtering pipelines neither specify that target density nor establish a global envelope. That distinction is why the shared name needs care.

## Best-of-N can amplify a scoring error {#best-of-n}

Return to the original probabilities $(0.5,0.3,0.2)$, but let a scorer rank C > B > A. Draw two independent candidates and keep the higher-scoring one. A wins only when both draws are A, with probability $0.5^2=0.25$. C wins whenever it appears, with probability $1-0.8^2=0.36$. The remaining 0.39 belongs to B.

If C is an incorrect answer that fools the scorer, more candidates make it easier to select that error. This is a constructed counterexample, not a model measurement. It shows why selection budgets and scorer quality need to be evaluated together.

[Reward Model Overoptimization (2022; ICML 2023)](https://proceedings.mlr.press/v202/gao23h.html) also studies RL and best-of-N using a synthetic setup with a larger “gold” reward model standing in for true preferences. It motivates checking whether proxy scores separate from independently measured quality, not claiming that every increase in N is harmful.

## The prompt mix changes too {#prompt-selection}

Suppose easy and hard prompts are equally common. Generate ten answers per prompt; each attempt passes with probability 0.8 for an easy prompt and 0.1 for a hard one. Keeping all passing answers gives expected contributions of eight and one. Equal prompt counts become a training-row ratio approaching 8:1 over many prompts.

Capping each prompt at one retained answer does not completely fix this. The probability of retaining anything is $1-(1-p)^{10}$: approximately 1 for the easy prompt, but 0.651 for the hard one. Prompts without a successful candidate still disappear.

| Adjustment | What it helps | Cost or limitation |
| --- | --- | --- |
| Per-prompt caps or prompt-balanced sampling | Stops easy prompts dominating through answer count | Prompts with no successful answer remain absent |
| More attempts on hard prompts | Improves the chance of finding a usable demonstration | More generation and verification cost; no help when success probability is zero |
| Deduplicate while retaining distinct valid methods | Reduces repeated templates | Identifying genuinely different solutions takes additional judgment |
| Mix in trusted human or teacher demonstrations | Adds behavior the current generator struggles to produce | Introduces another source's errors and style biases |

Compare methods under a fixed generation-token or time budget. Report retention, prompt coverage, and independent evaluation, rather than treating the pass rate after filtering as evidence of learning.

## Keep enough records to investigate failures {#data-records}

Training may only consume prompts and responses. Debugging needs more context:

| Field | Why keep it? |
| --- | --- |
| prompt_id, origin, split | Paraphrases and near-duplicates must not leak between training and evaluation |
| Generator version, sampling settings, seed | Identifies the generating iteration; a seed alone does not ensure reproducibility across hardware |
| Raw response, termination reason, length | Separates completion, truncation, and format failures |
| Verifier version and individual check results | A single total score hides which rule misjudged an answer |
| Retention reason and candidate-group identifier | Allows selection changes without regenerating every candidate |

Split prompts before generating candidates, and do not evaluate solely with the same checks used to construct training data. When the verifier changes, rescore stored candidates without silently overwriting their original records. Executing model-generated code requires isolation and resource limits. The example here only evaluates finite probability tables; it does not execute generated programs.

## Check the arithmetic {#worked-code}

The [standard-library teaching script](code/selection_and_entropy.py) checks filtering and best-of-N by deterministic enumeration. It does not call a model or reproduce a training run:

```bash
python3 05-post-training/code/selection_and_entropy.py
```

It prints an acceptance rate of 0.65, retained probabilities near `[0.769231, 0.230769, 0]`, and best-of-2 probabilities `[0.25, 0.39, 0.36]`. Try giving C the lowest score instead. Does extra sampling now help a correct answer, or amplify the scorer's mistake?

## Where to go next {#next}

- [SFT](sft-and-its-ceiling.en.md): turning selected demonstrations into a training objective.
- [Verifiable rewards](verifiable-rewards.en.md): passing a check without completing the task.
- [Alignment tax and entropy](alignment-tax.en.md): useful concentration versus lost solution coverage.
- [Post-training practice](../practice/post-training/README.en.md): frameworks, data, and checkpoints.
