# Distillation: what is the student learning from the teacher?

[中文](distillation.md) · **English**

> Last reviewed: 2026-10 · Prerequisites: [SFT](sft-and-its-ceiling.en.md), [Language-model objectives](../00-foundations/deep-dives/language-model-objective.en.md)

A large model answers a question, and you train a small model on that answer. That is distillation, but it is not the only form. The student can learn the teacher's probabilities over alternatives, or try first and ask the teacher to assess the states it actually visits.

These approaches need different interfaces, budgets, and data. Before choosing a KL objective, ask: **what can the teacher actually provide?**

For a first pass, read the supervision table and soft-target example. For implementation, continue to tokenizer alignment and on-policy gradients: these determine whether the code learns the objective you intended.

## One question, three forms of supervision

Consider a fictional support classifier with three labels: refund, shipping, and product inquiry. “My package has not arrived. Can I get a refund?” is somewhat ambiguous.

| Teacher provides | Student learns from | What is retained or lost? |
| --- | --- | --- |
| One answer: “refund” | Hard-label CE / SFT | Simple, but alternative probabilities are lost |
| `[0.60, 0.35, 0.05]` | A target distribution | Retains the possibility of “shipping” |
| A distribution at the student's current prefix | States the student visits | Closer to student behavior, but requires repeated teacher calls |

These are teaching probabilities, not claims of calibrated confidence. More detailed supervision is not necessarily more correct. A teacher can misunderstand a question or pass its own biases to the student.

## Calculate a soft-target loss

Let the teacher distribution be $p=[0.60,0.35,0.05]$ and the student $q=[0.40,0.40,0.20]$. Over the same label space, forward KL is:

$$
\begin{aligned}
D_{KL}(p\|q)&=\sum_c p_c\log\frac{p_c}{q_c}\\
&\approx0.1272.
\end{aligned}
$$

Alternatively, train with soft-target cross entropy:

$$
\begin{aligned}
H(p,q)&=-\sum_c p_c\log q_c\\
&\approx0.9509.
\end{aligned}
$$

They differ by the teacher's entropy $H(p)$. With a fixed teacher, that difference does not depend on student parameters, so their student gradients agree. **Different loss values need not mean different learning directions.** The classic [knowledge distillation paper](https://arxiv.org/abs/1503.02531) develops soft distributions as supervision.

```python
import math

teacher = [0.60, 0.35, 0.05]
student = [0.40, 0.40, 0.20]
cross_entropy = -sum(prob * math.log(pred) for prob, pred in zip(teacher, student))
teacher_entropy = -sum(prob * math.log(prob) for prob in teacher)
forward_kl = sum(prob * math.log(prob / pred) for prob, pred in zip(teacher, student))
assert abs(cross_entropy - teacher_entropy - forward_kl) < 1e-12
print(round(cross_entropy, 4), round(forward_kl, 4))
```

With a softmax output, soft CE has student-logit gradient $q-p=[-0.20,0.05,0.15]$. Using only the teacher's argmax as a one-hot label gives `[-0.60, 0.40, 0.20]`. The second pushes “shipping” down more strongly; the first retains the teacher's judgment that it remains plausible.

## How temperature changes the target {#temperature-changes-the-distribution-not-the-teachers-intelligence}

Divide logits by temperature $T$ before softmax:

$$
\begin{aligned}
p_T&=\operatorname{softmax}(z^{teacher}/T),\\
q_T&=\operatorname{softmax}(z^{student}/T),\\
\mathcal L_{KD}&=T^2D_{KL}(p_T\|q_T).
\end{aligned}
$$

Increasing positive temperature softens a fixed set of logits and gives non-maximal entries more weight. The classic $T^2$ factor compensates for temperature-related gradient scaling; it does not make training at different temperatures exactly equivalent or remove the need to check loss weights.

If the teacher misunderstood the task, increasing temperature does not turn its mistake into truth. It redistributes supervision rather than adding knowledge.

With ground-truth labels, the two forms of supervision can be combined:

$$
\begin{aligned}
\mathcal L&=(1-\lambda)\operatorname{CE}(y,q_1)\\
&\quad+\lambda T^2D_{KL}(p_T\|q_T),\\
&\qquad 0\leq\lambda\leq1.
\end{aligned}
$$

The label term usually uses temperature 1; the distillation term uses the same $T$ for both models. At $\lambda=0$, training uses labels only; at $\lambda=1$, it uses the teacher only. Mixing is not automatically better. Investigate disagreements before choosing weights, and check the implementation's weighting convention.

## What must align in a language model?

A classifier has three fixed labels; a language model predicts a token at every prefix. Before comparing distributions, check:

- **The same prefix:** are teacher and student conditioned on the same generated content? Do different chat templates change its meaning?
- **The same prediction position:** logits predict the next token; an extra or missing shift changes the target.
- **The same output space:** identical integer IDs can mean different text under different tokenizers. Different vocabularies do not support naive column-wise KL.
- **The same valid positions:** decide how padding, prompt, response, and truncated tails enter the loss.

If the teacher is a VLM and the student is text-only, ask what the student will observe at deployment. Labels that depend on an image detail the student never sees can teach correlations, not magically recover missing evidence. Providing verifiable extracted visual evidence is one option, but extraction errors and missing-input behavior need separate evaluation.

When an API provides text but not full logits, response distillation is a more direct starting point. Do not describe it as full-distribution KL distillation. Cross-tokenizer distillation is possible with an explicit method, such as alignment at the text-output level—not merely arrays of matching shape.

## Different tokenizers: matching text is not matching a whole distribution

Consider an invented segmentation: the teacher represents `bluebird` with one token; the student uses `blue` and `bird`. The teacher's log-probability is −1.2; the student's two values are −0.7 and −0.9, totaling −1.6. You can compare scores for this fixed text path. You cannot apply tokenwise KL between one teacher prediction and two student predictions.

| Approach | Requires | Missing information or additional assumptions |
| --- | --- | --- |
| Teacher text followed by student SFT | Usable answers and each model's tokenizer | No distribution over the teacher's other choices |
| Strictly aligned positions / shared output events | Matching prefix and candidate meaning | Report coverage and retained probability mass |
| Aligned text spans with aggregated scores | Byte or text boundaries, consistent role meaning, span log-probabilities | A span score is not inherently a tokenwise target |
| Explicit cross-vocabulary distribution conversion | A common event space and probability mapping | More implementation work than swapping KL arrays |

Splitting the teacher's −1.2 into −0.6 and −0.6 preserves the sum; it does not establish the teacher's conditional probabilities for two steps. Allocating it proportionally to old student log-probabilities is also an objective-design choice, not a unique consequence of the probability chain rule.

Check whitespace cleanup, Unicode, special tokens, and templates before accepting a text match. Render each model's valid template from structured messages and inspect response boundaries rather than copying another model's role markers. If unaligned positions are masked, report which examples lose supervision. More coverage need not mean more reliable targets.

As of 2026-10-09, [cross-family OPD](https://arxiv.org/abs/2606.09456) studies transferring signals across tokenizers. A [2026-10 follow-up analysis](https://arxiv.org/abs/2610.08448) finds that extending supervision into mismatched spans can reduce accuracy for its tested model combinations. This is a result under specific conditions, not a rejection of every span method. This page reproduces neither training study and does not present an experimental branch's configuration as a universal verl API. Check the implementation version, objective, and gradients first.

<details markdown="1">
<summary>Work it out: allocating one span score to two student steps</summary>

For `bluebird`, the teacher span score is −1.2 and the old student's sum is −1.6. Allocating in proportion to the old log-probabilities gives a scale of `−1.2 / −1.6 = 0.75`:

| Student token | Old log-prob | Allocated target | Target minus old value |
| --- | ---: | ---: | ---: |
| `blue` | −0.7 | −0.525 | +0.175 |
| `bird` | −0.9 | −0.675 | +0.225 |
| Total | −1.6 | −1.2 | +0.4 |

This preserves the total span score and gives a larger adjustment to the originally more surprising step. [Cross-family OPD, equations 8–9](https://arxiv.org/html/2606.09456v1#S4.SS2), uses this allocation. **These are assigned training targets, not two conditional distributions actually emitted by the teacher.** An old span score near zero also needs explicit numerical handling.

A further distinction: token-path probability need not equal text probability. Suppose a toy tokenizer can produce `ab` through two complete paths, `[ab]` and `[a, b]`, with probabilities 0.2 and 0.3 including termination. The text probability is 0.5. Scoring only the canonical encoding `[ab]` gives 0.2. Span alignment does not sum over all tokenization paths, so matching text alone does not establish an exact text-level KL.

</details>

<details markdown="1">
<summary>Code: find shared byte boundaries before assigning token targets</summary>

This teaching implementation operates on byte pieces, not a real tokenizer. Both sides must concatenate to the same nonempty bytes. It returns half-open student / teacher token ranges. Special tokens, decoder cleanup, and role templates require additional handling in a real system.

```python
def aligned_spans(student_pieces, teacher_pieces):
    for pieces in (student_pieces, teacher_pieces):
        if not pieces or any(not isinstance(piece, bytes) or not piece for piece in pieces):
            raise ValueError("expected nonempty byte pieces")
    if b"".join(student_pieces) != b"".join(teacher_pieces):
        raise ValueError("decoded bytes differ")

    teacher_ends = {}
    offset = 0
    for index, piece in enumerate(teacher_pieces, 1):
        offset += len(piece)
        teacher_ends[offset] = index

    spans = []
    offset = 0
    student_start = teacher_start = 0
    for student_end, piece in enumerate(student_pieces, 1):
        offset += len(piece)
        if offset in teacher_ends:
            teacher_end = teacher_ends[offset]
            spans.append((student_start, student_end, teacher_start, teacher_end))
            student_start, teacher_start = student_end, teacher_end
    return spans

assert aligned_spans([b"blue", b"bird", b"!"], [b"bluebird", b"!"]) == [
    (0, 2, 0, 1), (2, 3, 1, 2)
]
```

Why bytes? A token can contain only part of a Unicode character's UTF-8 encoding; decoding each token separately may introduce replacement characters. Here we verify the complete byte strings before comparing boundaries. Runtime is linear in total bytes plus token count. Alignment alone does not decide how to allocate supervision.

</details>

## What is lost when you keep only top-k?

Return to the three labels. The teacher's top two probabilities `[0.60,0.35]` retain mass 0.95. Renormalizing them gives `[0.6316,0.3684]`. If the student is also renormalized over these same entries, the comparison is between **conditional distributions**, not the originals.

| Student | Renormalized within teacher top-2 | Probability on the third entry |
| --- | --- | ---: |
| A: `[0.60,0.35,0.05]` | `[0.6316,0.3684]` | 0.05 |
| B: `[0.30,0.175,0.525]` | `[0.6316,0.3684]` | 0.525 |

This conditional-only loss considers A and B equally good, missing B's large probability outside the retained set. Options include retaining residual mass in an “other” bucket or defining a truncated objective without this conditional renormalization. Even an aggregate tail bucket cannot reveal how probability is distributed inside the tail.

The counterexample concerns **renormalizing both distributions after selecting top-k**. It is not a claim about every top-k distillation objective. Record teacher normalization, student normalization, tail handling, and token masks together.

## Where does R1 data distillation fit?

[DeepSeek-R1's public description](https://github.com/deepseek-ai/DeepSeek-R1) describes fine-tuning Qwen / Llama students with R1-curated data. This fits the “teacher provides answers” row: the student tokenizes text itself, without requiring aligned teacher logits.

Different vocabularies alone do not establish an entire training recipe. Nor does SFT imply updating every parameter: supervision and the choice of full-parameter versus LoRA updates are separate decisions. Check the specific model's training report.

## On-policy changes where prefixes come from

Consider “3 boxes, 4 pens per box: how many pens?” A demonstration says “3 × 4 = 12.” The student starts with “3 + 4 =”. Training only along the demonstration never asks the teacher what to do after that plus sign.

On-policy distillation lets the student generate first, then asks the teacher to score **the prefixes the student actually produced**. The teacher is not simply solving the problem independently and comparing final answers. It is assigning probabilities to continuations given what the student has already written.

<figure class="worked-update worked-update--pairs">
<figcaption>The teacher can stay the same while the training prefixes change.</figcaption>
<ol>
<li><small>Fixed demonstration</small><strong>3 × 4 = → next step</strong><span>Supervise the demonstration's prefixes. Data can be prepared ahead of time, but may miss the student's mistakes.</span></li>
<li><small>Student rollout</small><strong>3 + 4 = → next step</strong><span>Score the student's prefixes. They reflect its current behavior, at the cost of fresh generation and scoring.</span></li>
</ol>
</figure>

The example also exposes a limitation. Predicting `7` after the plus sign is locally sensible but still solves the wrong problem. Dense feedback is not guaranteed task correctness; answer checks, backtracking, or other training choices may still be needed. RL can also use process rewards rather than a single terminal score.

### Prefixes, supervision, and loss are separate choices {#opd-choices}

| Decision | Options | What it does not determine |
| --- | --- | --- |
| Where prefixes come from | Fixed data, student samples, a mixture | KL direction |
| What the teacher returns | Text, full distributions, top-k, sampled-token log-probs | These are not all full-distribution targets |
| How the student updates | Distribution fitting at fixed prefixes, policy gradients for sampled actions | Sampling and stop-gradient cannot be left unspecified |

[GKD (ICLR 2024)](https://arxiv.org/html/2306.13649v3#S3.SS1) separates the prefix source from the divergence and can mix fixed data with student generations. It does not differentiate through prefix sampling; it fits teacher distributions at those sampled prefixes. **On-policy therefore means neither “reverse KL” nor “replace GRPO's reward.”**

### What does reverse KL do to one update? {#opd-gradient}

Hold one prefix fixed and allow just two next actions, A and B. The teacher is $p=[0.8,0.2]$ and the student is $q=[0.5,0.5]$.

| Sampled action | Teacher log-prob minus student log-prob | Interpretation |
| --- | ---: | --- |
| A | $\log(0.8/0.5)\approx+0.4700$ | The student underweights A relative to the teacher |
| B | $\log(0.2/0.5)\approx-0.9163$ | The student overweights B relative to the teacher |

Used as a **detached** learning signal, a positive difference encourages the sampled action and a negative one discourages it. This is not an accuracy score. A mistaken teacher can still point the update in the wrong direction.

Write $q=q_\theta$ and abbreviate the log-ratio as $f(a)=\log[q(a)/p(a)]$. With a fixed teacher and positive probabilities:

$$
\begin{aligned}
J(\theta)&=\sum_a q(a)f(a),\\
\nabla J&=\mathbb E_{a\sim q}
\left[f(a)\nabla\log q(a)\right].
\end{aligned}
$$

The constant term omitted from the second line has expectation $\sum_a\nabla q(a)=0$. Sampling actions from the student requires this score-function gradient, not merely differentiating `logq - logp` after sampling. Here the correct student-logit gradient is approximately `[-0.3466, +0.3466]`. Holding sampled actions fixed and differentiating only their log-ratios instead gives an expected gradient of `[0, 0]`.

<details markdown="1">
<summary>Finite-difference check: a correct loss value can still have the wrong gradient</summary>

This enumerates both actions, with no sampling noise. `score_gradient` uses detached sampling weights and log-ratios; `naive_gradient` differentiates only each sampled log-prob. It checks one fixed prefix, not a complete trainer.

```python
import math

teacher = [0.8, 0.2]
student = [0.5, 0.5]
log_ratio = [math.log(pred / target) for pred, target in zip(student, teacher)]
score_gradient = [
    sum(student[action] * log_ratio[action] * ((action == index) - student[index])
        for action in range(2))
    for index in range(2)
]
naive_gradient = [
    sum(student[action] * ((action == index) - student[index]) for action in range(2))
    for index in range(2)
]

def reverse_kl(logits):
    weights = [math.exp(value - max(logits)) for value in logits]
    probs = [value / sum(weights) for value in weights]
    return sum(prob * math.log(prob / target) for prob, target in zip(probs, teacher))

epsilon = 1e-5
finite_difference = (reverse_kl([epsilon, 0]) - reverse_kl([-epsilon, 0])) / (2 * epsilon)
assert abs(finite_difference - score_gradient[0]) < 1e-9
assert naive_gradient == [0.0, 0.0]
assert abs(score_gradient[0] + math.log(2) / 2) < 1e-12
```

</details>

Forward KL is not blind to unwanted tokens. With teacher `[1, 0]` and student `[0.5, 0.5]`, the soft-CE logit gradient is `[-0.5, +0.5]`. The second logit is pushed down because softmax entries share a normalization.

Nor does reverse KL make selecting one mode automatically sufficient. For teacher `[0.5, 0.5]` and student `[1, 0]`, reverse KL is $\log2$, not zero. Without capacity or other restrictions, both KL directions are minimized at $q=p$. Mode preferences depend on the distributions the model can represent and on optimization.

### From one prefix to a whole response {#opd-sequence}

The derivation deliberately fixed the prefix. In a full generation, earlier actions also determine which later states are reached.

- **Full-vocabulary KL at fixed sampled prefixes:** fit distributions at visited positions without differentiating through prefix sampling. This is a stated training convention, not an exact full-trajectory gradient.
- **Full-sequence reverse KL:** on a common token and termination space, the sequence log-ratio sums per-step log-ratios. Its exact policy gradient also accounts for future effects of actions, expressible through return-to-go.
- **Immediate log-prob differences as advantages with PPO clipping:** this is a local surrogate. It may be useful, but resemblance to the one-prefix equation does not make its gradient identical to the full-sequence objective.

An implementation must also track behavior / current student versions and avoid treating prompt, tool-result, or padding tokens as sampled actions. Sampled-token log-probs and full-vocabulary logits are different information. Continue to [rollout records](post-training-infrastructure.en.md#rollout-record) for the data entering training.

### Is the extra teacher cost worthwhile? {#opd-budget}

Establish a fixed-teacher-data baseline before adding student rollouts. Fixed data is easy to cache; on-policy training keeps producing new prefixes, so measure cache reuse and teacher cost. Batched teacher forcing can score an entire response's prefixes together—it need not require a remote request for every generated token.

Report student training, generated tokens, teacher-scored tokens, total GPU-hours / wall time, and independently measured task quality. Equal optimizer steps do not mean equal budgets. If teacher errors, missing inputs, or tokenizer mismatches remain unresolved, more sampling can simply collect more unreliable supervision.

## Evaluating the distilled student {#did-the-student-improve-or-merely-resemble-the-teacher}

| Comparison | Alternative explanation to rule out |
| --- | --- |
| Original student vs budget-matched ordinary SFT vs distillation | Gains came only from more data or training |
| Human labels or task verifiers independent of the teacher | The student is reproducing the teacher's preferences and mistakes |
| Familiar, unseen, and insufficient-evidence tasks separately | Fixed-pattern imitation or fabrication |
| Quality, latency, memory, and teacher-data cost | A cheap student does not repay the cost of obtaining it |

Non-overlapping prompts alone are not enough: using the same teacher to label training data and judge the winner can preserve its systematic biases across both stages. An independent basis for validation is needed.

A student is not mathematically doomed to remain worse than its teacher; additional data, different objectives, or ensemble signals can change the outcome. But outperforming the teacher still needs independent evidence. Small training KL is not task success.

Continue to [LLM-as-a-Judge](../07-evaluation/llm-as-a-judge/README.en.md) to examine teacher scoring, or [LoRA / QLoRA](lora-and-qlora.en.md) to reduce trainable parameters.
