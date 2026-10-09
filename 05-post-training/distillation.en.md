# Distillation: what is the student learning from the teacher?

[中文](distillation.md) · **English**

> Last reviewed: 2026-10 · Prerequisites: [SFT](sft-and-its-ceiling.en.md), [Language-model objectives](../00-foundations/deep-dives/language-model-objective.en.md)

A large model answers a question, and you train a small model on that answer. That is distillation, but it is not the only form. The student can learn the teacher's probabilities over alternatives, or try first and ask the teacher to assess the states it actually visits.

These approaches need different interfaces, budgets, and data. Before choosing a KL objective, ask: **what can the teacher actually provide?**

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
D_{KL}(p\|q)=\sum_c p_c\log\frac{p_c}{q_c}\approx0.1272.
$$

Alternatively, train with soft-target cross entropy:

$$
H(p,q)=-\sum_c p_c\log q_c\approx0.9509.
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

## Temperature changes the distribution, not the teacher's intelligence

Divide logits by temperature $T$ before softmax:

$$
p_T=\operatorname{softmax}(z^{teacher}/T),\quad
q_T=\operatorname{softmax}(z^{student}/T),\quad
\mathcal L_{KD}=T^2D_{KL}(p_T\|q_T).
$$

Increasing positive temperature softens a fixed set of logits and gives non-maximal entries more weight. The classic $T^2$ factor compensates for temperature-related gradient scaling; it does not make training at different temperatures exactly equivalent or remove the need to check loss weights.

If the teacher misunderstood the task, increasing temperature does not turn its mistake into truth. It redistributes supervision rather than adding knowledge.

With ground-truth labels, the two forms of supervision can be combined:

$$
\mathcal L=(1-\lambda)\operatorname{CE}(y,q_1)
+\lambda T^2D_{KL}(p_T\|q_T),\qquad 0\leq\lambda\leq1.
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

```text
Fixed demonstrations: prompt → existing answer prefix → teacher scores → student update
On-policy: prompt → current student generates prefix → teacher scores → update → regenerate
```

At inference, a student's small initial error can lead to prefixes absent from demonstration data. Teacher guidance at those states can reduce dependence on pristine training paths. See [On-policy Distillation](https://arxiv.org/abs/2306.13649) for this motivation.

The costs are real: generation, teacher inference, and version management. Some erroneous prefixes are already beyond repair; a teacher continuation does not automatically correct earlier mistakes. Establish a fixed-teacher-data baseline before deciding whether extra calls are worthwhile.

On-policy does not specify a unique KL direction. $D_{KL}(p\|q)$ penalizes student undercoverage where the teacher assigns probability; $D_{KL}(q\|p)$ weights by student probability and emphasizes where the student puts mass. Support, temperature, student capacity, and optimization affect the result. “Mode-covering versus mode-seeking” alone cannot choose the better objective.

## Did the student improve, or merely resemble the teacher?

| Comparison | Alternative explanation to rule out |
| --- | --- |
| Original student vs budget-matched ordinary SFT vs distillation | Gains came only from more data or training |
| Human labels or task verifiers independent of the teacher | The student is reproducing the teacher's preferences and mistakes |
| Familiar, unseen, and insufficient-evidence tasks separately | Fixed-pattern imitation or fabrication |
| Quality, latency, memory, and teacher-data cost | A cheap student does not repay the cost of obtaining it |

Non-overlapping prompts alone are not enough: using the same teacher to label training data and judge the winner can preserve its systematic biases across both stages. An independent basis for validation is needed.

A student is not mathematically doomed to remain worse than its teacher; additional data, different objectives, or ensemble signals can change the outcome. But outperforming the teacher still needs independent evidence. Small training KL is not task success.

Continue to [LLM-as-a-Judge](../07-evaluation/llm-as-a-judge/README.en.md) to examine teacher scoring, or [LoRA / QLoRA](lora-and-qlora.en.md) to reduce trainable parameters.
