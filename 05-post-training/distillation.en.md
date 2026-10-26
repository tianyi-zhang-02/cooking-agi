# Distillation: what is the student learning from the teacher?

[中文](distillation.md) · **English**

> Last reviewed: 2026-10 · Prerequisites: [Supervised fine-tuning (SFT)](sft-and-its-ceiling.en.md), [Language-model objectives](../00-foundations/deep-dives/language-model-objective.en.md)

A large model answers a question correctly, and you train a smaller model on its answer. That is one form of distillation. You could also train the student to match the teacher's probabilities over possible answers, or let the student generate first and ask the teacher to evaluate continuations along that path.

Each approach requires different interfaces, data, and compute. Before choosing a Kullback–Leibler (KL) divergence to compare distributions, ask: **what can the teacher actually provide?**

Start with the supervision table and the soft-target calculation. If you are implementing training, continue to tokenizer alignment and the gradients for student-generated data. These details determine whether your code optimizes the intended objective.

## One question, three forms of supervision

Consider a fictional support classifier with three labels: refund, shipping, and product inquiry. “My package has not arrived. Can I get a refund?” is somewhat ambiguous.

| Teacher provides | Student learns from | What is retained or lost? |
| --- | --- | --- |
| One answer: “refund” | Hard-label cross-entropy for classification; an SFT demonstration for generation | Simple, but alternative probabilities are lost |
| `[0.60, 0.35, 0.05]` | A target distribution | Retains the possibility of “shipping” |
| A distribution conditioned on what the student has generated so far | Teacher feedback at states the student visits | Tracks student behavior, but requires fresh teacher scores as the student changes |

The probabilities are illustrative; we are not assuming the teacher's confidence is calibrated. A more detailed target can still be wrong. The teacher may misunderstand the question or pass its biases on to the student.

## Calculate a soft-target loss

Let the teacher distribution be $p=[0.60,0.35,0.05]$ and the student distribution be $q=[0.40,0.40,0.20]$. With both distributions defined over the same labels, forward KL is:

$$
\begin{aligned}
D_{KL}(p\|q)&=\sum_c p_c\log\frac{p_c}{q_c}\\
&\approx0.1272.
\end{aligned}
$$

Alternatively, use cross-entropy (CE) with the teacher's distribution as a soft target:

$$
\begin{aligned}
H(p,q)&=-\sum_c p_c\log q_c\\
&\approx0.9509.
\end{aligned}
$$

The two losses differ by the teacher's entropy $H(p)$. For a fixed teacher, this term does not depend on the student's parameters, so both losses give the same gradient with respect to those parameters. Their values differ, but their update directions agree. The classic [knowledge distillation paper (2015, §2)](https://arxiv.org/abs/1503.02531) explains how to use these soft targets for supervision.

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

When softmax converts the student's unnormalized scores, or logits, into probabilities, soft-target cross-entropy has gradient $q-p=[-0.20,0.05,0.15]$ with respect to those logits. Keeping only the teacher's most likely label gives the target `[1, 0, 0]` and gradient `[-0.60, 0.40, 0.20]`. That hard target pushes “shipping” down more strongly. The soft target preserves the teacher's judgment that shipping remains plausible.

### What do entropy and cross-entropy measure? {#entropy-and-surprise}

For a moment, consider just two labels with teacher probabilities `[0.99, 0.01]`. The first label is easy to predict; the second would be a surprise. **A rare outcome can be surprising even when the distribution as a whole is highly predictable.**

The surprise of one outcome is its self-information, or surprisal: $-\log p(x)$. For the second label, this is $-\log0.01\approx4.6052$. Entropy averages over all outcomes: $H(p)=-\sum_xp(x)\log p(x)\approx0.0560$, well below the $0.6931$ of two equally likely labels. We use natural logarithms, so the unit is the nat. See [Deep Learning §3.13](https://www.deeplearningbook.org/contents/prob.html) for the definitions.

Returning to the three-label example, we can separate the cross-entropy into two parts:

<figure class="worked-update" lang="en" id="entropy-loss-parts">
  <figcaption>For a fixed teacher, cross-entropy includes both teacher entropy and student mismatch.</figcaption>
  <ol>
    <li><small>Teacher entropy H(p)</small><strong>0.8237</strong><span>This remains even if the student matches the teacher exactly.</span></li>
    <li><small>Mismatch KL(p ∥ q)</small><strong>+ 0.1272</strong><span>The extra loss from using a different distribution.</span></li>
    <li><small>Cross-entropy H(p, q)</small><strong>= 0.9509</strong><span>With the teacher fixed, lowering CE lowers forward KL.</span></li>
  </ol>
</figure>

A lower CE after changing the teacher or temperature does not, by itself, mean a closer match. Two students that perfectly match `[0.99, 0.01]` and `[0.5, 0.5]` both have zero KL, yet their cross-entropies are 0.0560 and 0.6931. Keep the target distribution fixed when comparing. Low teacher entropy also means confidence, not necessarily correctness.

<details markdown="1">
<summary>Why does H(p, q) = H(p) + KL(p ∥ q)?</summary>

For a finite label space, assume $q(x)>0$ wherever $p(x)>0$. Subtract the two definitions:

$$
\begin{aligned}
H(p,q)-H(p)
&=\sum_{x:p(x)>0}p(x)\big[\log p(x)-\log q(x)\big]\\
&=D_{KL}(p\|q).
\end{aligned}
$$

Terms with $p(x)=0$ contribute zero by the limiting convention. If the teacher assigns positive probability to a label that the student assigns zero probability, both CE and KL are infinite. If an implementation still reports a finite loss, check whether clipping or another numerical adjustment has hidden that support mismatch.

Here the expectation is under the **teacher p**. The note on [KL estimators in reinforcement learning (RL)](rlhf/kl-estimators.en.md) samples from the **current policy q** and discusses $D_{KL}(q\|p)$. The same letters appear, but the direction is reversed.

</details>

## How temperature changes the target {#temperature-changes-the-distribution-not-the-teachers-intelligence}

Divide logits by temperature $T$ before softmax:

$$
\begin{aligned}
p_T&=\operatorname{softmax}(z^{teacher}/T),\\
q_T&=\operatorname{softmax}(z^{student}/T),\\
\mathcal L_{KD}&=T^2D_{KL}(p_T\|q_T).
\end{aligned}
$$

Increasing positive temperature brings probability ratios closer to one, softening the distribution for a fixed set of logits. It does not increase every non-maximal probability: for logits `[2.0, 1.9, 0.0]`, raising $T$ from 1 to 2 lowers the second probability from about 0.4435 to 0.4102 while increasing the distribution's entropy.

The $T^2$ factor compensates for the change in gradient scale. [Section 2.1 of the original paper](https://arxiv.org/abs/1503.02531) derives approximate $1/T^2$ scaling when the temperature is high relative to the logits and each model's logits are centered. Multiplying by $T^2$ does not make training at different temperatures exactly equivalent; the relative loss weights still need checking.

Increasing temperature cannot correct the teacher's misunderstanding of a task. It changes how supervision is distributed, not what the teacher knows.

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

Our classifier has three fixed labels. A language model instead predicts the next token given a prefix: the content already available to it. Tokens are the units produced by a tokenizer and need not be whole words or characters. Before comparing two distributions, check:

- **The same prefix:** are teacher and student conditioned on the same generated content? Do different chat templates change its meaning?
- **The same prediction position:** logits predict the next token; an extra or missing shift changes the target.
- **The same output space:** the same integer ID can represent different text in different tokenizers. Matching array columns is not enough to make KL meaningful across vocabularies.
- **The same valid positions:** decide how padding, prompt, response, and truncated tails enter the loss.

If the teacher is a vision-language model and the student is text-only, ask what the student can observe at deployment. A label may depend on an image detail the student never sees. In that case, the student may learn a correlation, but distillation cannot recover the missing evidence. One option is to extract verifiable visual evidence into text for the student. Evaluate extraction errors and behavior with missing inputs separately.

When an interface returns text but not full logits, response distillation is a straightforward starting point: train the student on the teacher's answers as demonstrations. That is not full-distribution KL distillation. Models with different tokenizers can still be distilled, but they need a defined method, such as learning from teacher-generated text. Arrays with matching shapes are not necessarily comparable.

## Different tokenizers: matching text is not matching a whole distribution

Suppose, in a toy example, that the teacher encodes `bluebird` as one token while the student uses `blue` and `bird`. The teacher assigns its token a log-probability of −1.2. The student's two log-probabilities are −0.7 and −0.9, totaling −1.6. You can compare the scores of these particular token paths through the same text. You cannot directly compute tokenwise KL between one teacher prediction and two student predictions.

| Approach | Requires | Missing information or additional assumptions |
| --- | --- | --- |
| Teacher text followed by student SFT | Usable answers and each model's tokenizer | No distribution over the teacher's other choices |
| Strictly aligned positions and shared output events | Matching prefix and candidate meaning | Report coverage and retained probability mass—the sum of probabilities on those events |
| Aligned text spans with aggregated scores | Byte or text boundaries, consistent role meaning, span log-probabilities | A span score is not inherently a tokenwise target |
| Explicit cross-vocabulary distribution conversion | A common event space and probability mapping | Requires more than substituting a different array in the same KL calculation |

Splitting the teacher's −1.2 into −0.6 and −0.6 preserves the sum. It does not establish that these are the teacher's conditional log-probabilities for two steps. Allocating the score in proportion to the student's old log-probabilities is also a training choice, not something uniquely determined by the probability chain rule.

Check whitespace cleanup, Unicode, special tokens, and templates before accepting a text match. Render each model's valid template from structured messages and inspect response boundaries rather than copying another model's role markers. If unaligned positions are masked, report which examples lose supervision. More coverage need not mean more reliable targets.

Cross-tokenizer studies use different supervision objectives. [A June 2026 cross-family study](https://arxiv.org/html/2606.09456v1#S4.SS2) uses on-policy distillation (OPD): teacher feedback on student-generated content. It allocates a span score to individual student steps; the example below works through that rule. [A separate October 2026 study (§3.2)](https://arxiv.org/html/2610.08448v1#S3.SS2) adds mean squared error on mismatched-span log-probabilities to KL at strictly aligned positions. Across its three teacher–student pairs and tested positive weights, this lowers average math and code accuracy. The objectives differ, so that result does not rule out all span-based methods.

Neither training study is reproduced here. Compare model pairs, objectives, and budgets, not just alignment coverage. Check the training framework's version too: a configuration supported by an experimental branch is not necessarily part of the standard verl interface.

<details markdown="1">
<summary>Work it out: allocating one span score to two student steps</summary>

For `bluebird`, the teacher span score is −1.2 and the old student's sum is −1.6. Allocating in proportion to the old log-probabilities gives a scale of `−1.2 / −1.6 = 0.75`:

| Student token | Old log-prob | Allocated target | Target minus old value |
| --- | ---: | ---: | ---: |
| `blue` | −0.7 | −0.525 | +0.175 |
| `bird` | −0.9 | −0.675 | +0.225 |
| Total | −1.6 | −1.2 | +0.4 |

This preserves the total span score and assigns a larger target difference to the less likely step. The actual parameter update still depends on the gradients and optimizer. [Cross-family OPD, equations 8–9](https://arxiv.org/html/2606.09456v1#S4.SS2), uses this allocation. **These are assigned training targets, not two conditional distributions produced by the teacher.** An old span score near zero also needs explicit numerical handling.

Token-path probability is not necessarily text probability. Suppose a toy model can generate `ab` through two complete token paths, `[ab]` and `[a, b]`, with probabilities 0.2 and 0.3 including termination. The text probability is 0.5. Scoring only the tokenizer's canonical encoding `[ab]` gives 0.2. Span alignment does not sum over every encoding path, so matching text alone does not establish an exact text-level KL.

</details>

<details markdown="1">
<summary>Code: find shared byte boundaries before assigning token targets</summary>

This teaching implementation works with byte pieces; it does not call a tokenizer. Both sides must concatenate to the same nonempty byte string. It returns matching student and teacher token ranges, with the start included and the end excluded. A real system also needs to handle special tokens, decoder cleanup, and role templates.

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

Return to the three labels and keep only the teacher's two highest probabilities, or top-2: `[0.60,0.35]`, with total mass 0.95. Renormalizing them gives `[0.6316,0.3684]`. If the student is also renormalized over those entries, you are comparing **distributions conditioned on those two choices**, not the original full distributions.

| Student | Renormalized within teacher top-2 | Probability on the third entry |
| --- | --- | ---: |
| A: `[0.60,0.35,0.05]` | `[0.6316,0.3684]` | 0.05 |
| B: `[0.30,0.175,0.525]` | `[0.6316,0.3684]` | 0.525 |

This conditional-only loss considers A and B equally good, missing B's large probability outside the retained set. Options include retaining residual mass in an “other” bucket or defining a truncated objective without this conditional renormalization. Even an aggregate tail bucket cannot reveal how probability is distributed inside the tail.

This counterexample applies when **both distributions are renormalized over the teacher's top-k choices**. It does not apply to every top-k distillation objective. Record how each distribution is normalized, how the tail is handled, and which tokens contribute to the loss.

## Where does R1 data distillation fit?

[DeepSeek-R1's public description](https://github.com/deepseek-ai/DeepSeek-R1) and [initial report, §2.4](https://arxiv.org/html/2501.12948v1#S2.SS4) describe fine-tuning Qwen and Llama students on R1-curated data. This fits the “teacher provides answers” row: the student uses its own tokenizer to read the text, with no need for tokenwise alignment of teacher logits. For the benefits and limits of distilling long reasoning traces into smaller models, see “Distillation: teaching long chains to small models” in the frontier draft on [how long chains of thought are trained](../11-frontier/test-time-compute/long-chains.en.md).

Different vocabularies alone do not tell you a project's training recipe. Nor does SFT imply updating every parameter: the supervision source is separate from the choice between full-parameter updates and low-rank adaptation (LoRA). Check the model's training report for what was actually done.

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

The example also shows a limitation. Predicting `7` after the plus sign is a sensible continuation, but it still gives the wrong answer to the question. Feedback at every step does not guarantee a correct final answer; verification, backtracking, or other training choices may still be needed. RL can also use process rewards, not just a single terminal score.

### Prefixes, supervision, and loss are separate choices {#opd-choices}

| Decision | Options | What it does not determine |
| --- | --- | --- |
| Where prefixes come from | Fixed data, student samples, a mixture | KL direction |
| What the teacher returns | Text, full distributions, top-k, sampled-token log-probabilities | Whether full-distribution supervision is available |
| How the student updates | Distribution fitting at fixed prefixes, policy gradients for sampled actions | Which distribution supplies samples and which quantities are held constant during differentiation |

[Generalized Knowledge Distillation, or GKD (ICLR 2024, §3.1)](https://arxiv.org/html/2306.13649v3#S3.SS1), treats the prefix source and divergence as separate choices. It can mix fixed data with student generations and fits teacher distributions at those prefixes without differentiating through the sampling process. On-policy distillation therefore need not use reverse KL, nor is it simply group relative policy optimization (GRPO) with a different reward.

### What does reverse KL do to one update? {#opd-gradient}

Hold one prefix fixed and allow just two next actions, A and B. The teacher is $p=[0.8,0.2]$ and the student is $q=[0.5,0.5]$.

| Sampled action | Teacher log-probability minus student log-probability | Interpretation |
| --- | ---: | --- |
| A | $\log(0.8/0.5)\approx+0.4700$ | The student underweights A relative to the teacher |
| B | $\log(0.2/0.5)\approx-0.9163$ | The student overweights B relative to the teacher |

When the difference is **detached**, meaning no gradient flows through the signal itself, a positive value encourages the sampled action and a negative one discourages it. This is not an accuracy score. A mistaken teacher can still push the update in the wrong direction.

A single difference is not the full negative KL either. For a fixed prefix and teacher, with positive teacher probabilities on the student's support, taking expectation under the student gives $\mathbb E_{a\sim q}[\log p(a)-\log q(a)]=-D_{KL}(q\|p)\leq0$. Here the mean is $0.5\times0.4700+0.5\times(-0.9163)\approx-0.2231$, even though the term for sampled action A is positive. Sampling from another behavior distribution without correction does not preserve this identity.

Write $q=q_\theta$ and abbreviate the log-ratio as $f(a)=\log[q(a)/p(a)]$. With a fixed teacher and positive probabilities:

$$
\begin{aligned}
J(\theta)&=\sum_a q(a)f(a),\\
\nabla J&=\mathbb E_{a\sim q}
\left[f(a)\nabla\log q(a)\right].
\end{aligned}
$$

The constant term omitted from the second line has expectation $\sum_a\nabla q(a)=0$. To estimate the gradient from student-sampled actions, use this score-function estimator: weight $\nabla\log q(a)$ by the log-ratio. Merely differentiating `logq - logp` after sampling misses the dependence of the sampling distribution on the parameters. Here the correct gradient with respect to the student logits is approximately `[-0.3466, +0.3466]`; differentiating only the sampled log-ratios gives an expected gradient of `[0, 0]`.

<details markdown="1">
<summary>Finite-difference check: a correct loss value can still have the wrong gradient</summary>

The code enumerates both actions, eliminating sampling noise. `score_gradient` treats the sampling weights and log-ratios as constants; `naive_gradient` differentiates only the sampled action's log-probability. This checks the gradient at one fixed prefix, not an entire training system.

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

Forward KL does not ignore tokens that the teacher assigns zero probability. With teacher `[1, 0]` and student `[0.5, 0.5]`, the soft-target cross-entropy gradient with respect to the logits is `[-0.5, +0.5]`. The second logit is pushed down because all softmax entries share a normalization denominator.

Nor is matching just one teacher mode necessarily enough for reverse KL. For teacher `[0.5, 0.5]` and student `[1, 0]`, reverse KL is $\log2$, not zero. Without capacity or other restrictions, both KL directions reach their minimum at $q=p$. Whether training favors particular modes depends on which distributions the model can represent and how it is optimized.

### From one prefix to a whole response {#opd-sequence}

The derivation deliberately fixed the prefix. In a full generation, earlier actions also determine which later states are reached.

- **Full-vocabulary KL at fixed sampled prefixes:** fit distributions at visited positions without differentiating through prefix sampling. This is a stated training convention, not an exact full-trajectory gradient.
- **Full-sequence reverse KL:** when both models use a common token space and termination event, the sequence log-ratio is the sum of the per-step log-ratios. The exact policy gradient also accounts for an action's later effects through return-to-go: the cumulative reward from that step onward.
- **Immediate log-probability differences as advantages, with proximal policy optimization (PPO) clipping:** this is a local surrogate objective. It may be useful, but resembling the single-prefix equation does not make its gradient identical to that of the full-sequence objective.

[Thinking Machines' on-policy distillation (Lu et al., 2025-10-27)](https://thinkingmachines.ai/blog/on-policy-distillation/) is an example of a local signal without clipping. Each sampled token's advantage is the teacher's log-probability minus the student's sampling-time log-probability, held constant during the update. This is one sampled contribution to negative reverse KL, not KL summed over the vocabulary; its expectation has the fixed-prefix interpretation above. A discount of 0 means the update considers only the immediate next token. The implementation uses an importance-sampling policy-gradient loss, not PPO clipping, and needs teacher log-probabilities only for sampled tokens. The authors chose zero discount for simplicity after finding no improvement from nonzero discounts in their experiments.

The post cites [the Qwen3 technical report (2025, §4.7, Table 21)](https://arxiv.org/html/2505.09388v1#S4.SS7). Both runs start from the same off-policy-distilled Qwen3-8B checkpoint and use only math and code queries for this training comparison. On-policy distillation scores 74.4 on AIME'24 versus 67.6 for RL. Their reported GPU-hour totals are 1,800 and 17,920, respectively—roughly a tenfold difference. These are the Qwen team's results for that setup, not a general speedup claim. The table does not fully specify the hardware and compute-accounting details needed to estimate costs for other workloads.

Track both the student version that generated the samples and the version being updated. Do not treat prompt, tool-result, or padding tokens as student-sampled actions. Sampled-token log-probabilities and full-vocabulary logits also support different training objectives. See [rollout records](post-training-infrastructure.en.md#rollout-record) for the data passed into training.

### Is the extra teacher cost worthwhile? {#opd-budget}

Start with a baseline trained on fixed teacher data, then compare training on student-generated sequences. Fixed data is easy to cache; on-policy training continually produces new prefixes, so measure cache reuse and teacher cost. Once the student has generated a response, teacher forcing can score its continuations at all positions in one pass. There is no need to make a remote request after every generated token.

Report the amount of student training, generated and teacher-scored token counts, total GPU-hours, elapsed time, and independently evaluated task quality. Equal numbers of optimizer steps do not imply equal budgets. Unresolved teacher errors, missing inputs, or tokenizer mismatches can make extra sampling little more than a way to collect more unreliable supervision.

## Evaluating the distilled student {#did-the-student-improve-or-merely-resemble-the-teacher}

| Comparison | Alternative explanation to rule out |
| --- | --- |
| Original student, budget-matched ordinary SFT, and distilled student | Gains came only from more data or training |
| Human labels or task verifiers independent of the teacher | The student is reproducing the teacher's preferences and mistakes |
| Familiar, unseen, and insufficient-evidence tasks separately | Fixed-pattern imitation or fabrication |
| Quality, latency, memory, and teacher-data cost | Deployment savings do not offset the cost of distillation |

Non-overlapping prompts alone are not enough: using the same teacher to label training data and judge the winner can preserve its systematic biases across both stages. An independent basis for validation is needed.

A student need not always perform worse than its teacher. Additional data, different objectives, or ensemble signals can change the outcome. Any claim that it outperforms the teacher still needs independent evidence; a small training KL does not establish task success.

For more on teacher scoring, see [using language models as judges](../07-evaluation/llm-as-a-judge/README.en.md). To reduce the number of trainable parameters, see [LoRA and its quantized variant, QLoRA](lora-and-qlora.en.md).
