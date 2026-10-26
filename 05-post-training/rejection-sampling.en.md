# Rejection sampling: generate more, then keep what?

[中文](rejection-sampling.md) · **English**

> Reading time: ~12 min · Type: chapter · Last reviewed: 2026-10

A model produces four solutions to a coding task. Two pass the tests, one passes only some tests, and one does not run. You could train on both passing solutions, train only on the highest-scoring solution, or return that solution to the user without training at all.

All of these procedures may be called rejection sampling, but they are not the same algorithm. To distinguish them, ask: **How are answers selected? Are they returned to the user, or used to update the model?**

For the training workflow, start with the first two sections and [the record-keeping section](#data-records). The four worked examples in between explain how selection changes a distribution. Basic probability and [supervised fine-tuning (SFT)](sft-and-its-ceiling.en.md) are enough to follow them.

## One name, several procedures {#four-uses}

| Procedure | What is retained? | Are parameters updated? |
| --- | --- | --- |
| Classical accept–reject sampling | Candidates accepted with a probability chosen to recover a target distribution | The sampling algorithm itself does not train a model |
| Verifier filtering | Answers that pass checks or a threshold; possibly several per prompt | Filtering alone does not update weights; SFT may follow |
| Best-of-N | The highest-scoring of N generated candidates | Can be used at inference time without training; see [sampling, voting, and verifiers](../11-frontier/test-time-compute/parallel-and-verify.en.md) for the effect of verifier errors |
| Rejection-sampling fine-tuning | Selected answers become demonstrations | SFT updates the parameters |

[Llama 2 (2023), §3.2.3](https://arxiv.org/html/2307.09288v2) describes generating multiple candidates per prompt, selecting the one with the highest reward score, and fine-tuning on those answers. The [initial DeepSeek-R1 report (2025), §2.3.3](https://arxiv.org/html/2501.12948v1#S2.SS3.SSS3) describes sampling multiple responses per prompt and retaining the correct ones, with additional readability filters. Keeping exactly one reward-model winner is therefore only one possible procedure. Neither pipeline, as described, is the statistical accept–reject algorithm that guarantees an exact target distribution.

## What does filtered SFT learn? {#filtered-sft}

The workflow has four steps: fix a generator and sampling configuration; generate candidates; score, filter, and deduplicate them; then run SFT on the retained answers. The updated model can generate a fresh dataset for the next iteration. ReST-EM organizes this loop using expectation–maximization. For that approach and the limits imposed by the verifier, see the frontier draft [Self-improvement: generate, verify, retrain](../11-frontier/self-improvement/generate-verify-train.en.md).

For the retained dataset $D_{\rm keep}$, first sum the negative log-probabilities at supervised positions within each answer to get $\ell(x,y)$. Sum these answer losses, then divide by $M$, the total number of supervised tokens. Tokens are the units produced by a tokenizer and need not be whole words or characters:

$$
\ell(x,y)
=-\sum_{t\in\mathcal T_y}\log\pi_\theta(y_t\mid x,y_{<t}).
$$

$$
\mathcal L_{\rm SFT}
=\frac{1}{M}\sum_{(x,y)\in D_{\rm keep}}\ell(x,y).
$$

Here $x$ is the prompt, $y$ the retained response, $\mathcal T_y$ the supervised response positions, and $M$ their total count. Prompt and padding positions contribute neither to the loss sum nor to its denominator. Averaging within each response before averaging across responses would change the relative weights of long and short answers.

The verifier decides **which demonstrations enter the dataset**. Training then imitates the retained answers. It requires neither gradients through the verifier nor the new-to-old policy probability ratio used by proximal policy optimization (PPO). Discarded answers are simply absent from this SFT update; they do not directly teach the model what to avoid. A separate preference objective could make use of them.

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

This requires $Z(x)>0$. It is a conditional distribution, not a guarantee that filtering produces an ideal answer distribution. False rejections, false acceptances, and solutions the generator rarely produces all affect it. Here $a$ summarizes pass rates within categories; a deterministic verifier assigns each individual answer an acceptance value of 0 or 1. The formula describes per-answer acceptance under a fixed rule. Groupwise selection, deduplication, or retention caps can change the distribution further.

## Classical rejection sampling needs an envelope {#classical-rejection}

Now suppose the aim is to sample from a specified target distribution $p$, rather than merely retain answers that pass a check. Generate from an easier proposal distribution $q$ and accept $y$ with probability $a(y)=p(y)/(M q(y))$. Here $M$ is the envelope constant, unrelated to the supervised-token count in the earlier SFT formula. A finite $M$ must satisfy $p(y)\le Mq(y)$ wherever the target has positive probability. In particular, $q(y)$ cannot be zero where $p(y)>0$. See the derivation in [CMU's Monte Carlo lecture, slide 8](https://www.cs.cmu.edu/~epxing/Class/10708/lectures/lecture16-MC.pdf#page=8).

Take $p=(0.6,0.3,0.1)$ and $q=(1/3,1/3,1/3)$. The smallest valid $M$ is 1.8:

| Category | Acceptance p / (M q) | Proposal mass × acceptance |
| --- | --- | --- |
| A | 1 | 1/3 |
| B | 1/2 | 1/6 |
| C | 1/6 | 1/18 |

The total acceptance rate is $5/9=1/M$. Normalizing the retained masses gives exactly $(0.6,0.3,0.1)$. A larger valid $M$ rejects more proposals but leaves the accepted distribution unchanged. An acceptance probability above 1 means the envelope condition has failed; clipping it to 1 does not preserve the exact-sampling guarantee.

The method also works when only an unnormalized target density is available, provided the envelope bounds that density. The acceptance rate is then the target's normalizing constant divided by the envelope constant, not simply $1/M$. Large language model filtering pipelines typically neither specify such a target density nor establish a global envelope. They should not be confused with this exact-sampling procedure.

## Best-of-N can amplify a scoring error {#best-of-n}

Return to the original probabilities $(0.5,0.3,0.2)$, but let a scorer rank C > B > A. Draw two independent candidates from this distribution and keep the higher-scoring one. A wins only when both draws are A, with probability $0.5^2=0.25$. C wins whenever it appears, with probability $1-0.8^2=0.36$. The remaining probability, 0.39, is the chance that B wins.

If C is an incorrect answer that fools the scorer, more candidates make it easier to select that error. This is a constructed counterexample, not a model measurement. It shows why selection budgets and scorer quality need to be evaluated together.

<!-- widget:step-best-of-n -->

[Reward Model Overoptimization (2022; ICML 2023, §2.1)](https://proceedings.mlr.press/v202/gao23h.html) studies reinforcement learning and best-of-N in a synthetic setting. A larger reward model serves as the “gold standard” in place of real human preferences. The study motivates checking whether rising proxy scores are accompanied by falling independent quality scores. It does not show that every increase in candidate count is harmful.

## The prompt mix changes too {#prompt-selection}

Suppose easy and hard prompts are equally common. Generate ten answers per prompt, with a per-attempt pass probability of 0.8 for easy prompts and 0.1 for hard ones. Keeping every passing answer yields an expected eight examples per easy prompt but only one per hard prompt. Across many prompts, equal counts of the two prompt types produce close to an 8:1 ratio of training examples.

Capping retention at one answer per prompt does not fully solve the problem. If attempts for a given prompt are independent and each has the same pass probability $p$, the probability that at least one passes is $1-(1-p)^{10}$. That is approximately 1 for an easy prompt but only 0.651 for a hard one. Prompts with no successful candidate still disappear from training. Correlated attempts or changes to the sampling policy require a different calculation.

| Adjustment | What it helps | Cost or limitation |
| --- | --- | --- |
| Per-prompt caps or prompt-balanced sampling | Stops easy prompts dominating through answer count | Prompts with no successful answer remain absent |
| More attempts on hard prompts | Improves the chance of finding a usable demonstration | More generation and verification cost; no help when success probability is zero |
| Deduplicate while retaining distinct valid methods | Reduces repeated templates | Identifying genuinely different solutions takes additional judgment |
| Mix in trusted human or teacher demonstrations | Adds behavior the current generator struggles to produce | Introduces another source's errors and style biases |

Compare methods under a fixed budget of generated tokens or elapsed time. Report retained example counts, prompt coverage, and independent evaluation results. A higher pass rate after filtering may reflect the selection rule rather than anything the model has learned.

## Keep enough records to investigate failures {#data-records}

Training may only consume prompts and responses. Debugging needs more context:

| Field | Why keep it? |
| --- | --- |
| Prompt identifier (`prompt_id`), source, and dataset split | Prevents paraphrases and near-duplicates from leaking between training and evaluation |
| Generator version, sampling settings, seed | Identifies the generating iteration; a seed alone does not ensure reproducibility across hardware |
| Raw response, termination reason, length | Separates completion, truncation, and format failures |
| Verifier version and individual check results | A single total score hides which rule misjudged an answer |
| Reason for keeping or discarding the answer, and its candidate group | Allows selection changes without regenerating every candidate |

Assign prompts to training and evaluation splits before generating candidates. Do not evaluate solely with the checks used to construct the training data. When the verifier changes, rescore stored candidates while preserving their original records. Executing model-generated code requires isolation and resource limits. The example here only works with finite probability tables; it does not execute generated programs.

## Check the arithmetic {#worked-code}

The [teaching script](code/selection_and_entropy.py) uses only the standard library and checks filtering and best-of-N by enumerating all possible outcomes. It does not call a model or reproduce a paper's training run:

```bash
python3 05-post-training/code/selection_and_entropy.py
```

The first three lines show an acceptance rate of 0.65 with retained probabilities near `[0.769231, 0.230769, 0]`, the classical accept–reject rate of about 0.5556 (5/9) with distribution `(0.6, 0.3, 0.1)`, and best-of-2 probabilities `(0.25, 0.39, 0.36)`. Later lines report entropy, sorting checks, and other results used in separate notes. Try assigning the incorrect category C the lowest score, then compare each category's selection probability. After correcting the ranking, does generating more candidates still amplify the same error?

## Where to go next {#next}

- [Supervised fine-tuning](sft-and-its-ceiling.en.md): turning selected demonstrations into a training objective.
- [Verifiable rewards](verifiable-rewards.en.md): passing a check without completing the task.
- [Alignment tax and entropy](alignment-tax.en.md): useful concentration versus lost solution coverage.
- [Post-training practice](../practice/post-training/README.en.md): frameworks, data, and checkpoints.
