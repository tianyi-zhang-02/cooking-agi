# LLM-as-a-Judge: how exactly should a model do the scoring?

[中文](llm-as-a-judge.md) · **English**

## Quick learning: a judge is a measurement system, not one prompt

<details class="interview" markdown="1">
<summary>Criteria, scoring, calibration, and validation</summary>

**Quick memory**: define an observable criterion first, then choose pairwise, ordinal, or binary scoring; finally calibrate the judge with expert labels, perturbation tests, and distribution slices.

**Interview answer**

> LLM-as-a-Judge requires explicit input evidence, a single criterion, behavioral anchors, an output schema, and an aggregation rule. Pairwise scoring is usually steadier than absolute scores but still suffers from position, verbosity, self-preference, and reference-leakage biases. Before deployment it must be cross-validated against expert audits, and you report disagreement and confidence, not only a mean.

<details markdown="1">
<summary><b>Deep dive</b>: why can a weighted mean hide failure?</summary>

The same mean can come from "everyone gives a 3" or from "half give a 1, half give a 5," and the two carry completely different decision risks. Preserve the score distribution, entropy or disagreement, criterion slices, and the judge version; high-stakes cases should also be escalated to a human instead of collapsing the uncertainty into one scalar.

</details>
</details>

## A judge is an evaluation system, not a prompt

LLM-as-a-Judge is not one fixed prompt. It is a set of design choices that combine freely: **whether to give demonstrations, whether to give a reference answer, how candidates are compared, which criterion is judged, and how the scores are finally aggregated.**

Many of these concepts look alike because a single prompt often uses several of the methods at once.

## Why few-shot and reference-based are easy to confuse

They are two orthogonal dimensions:

| Dimension | Few-shot | Reference-based |
| --- | --- | --- |
| What it asks | Should the model first see a few scoring demonstrations? | Does the current answer have a reference answer to compare against? |
| Main function | Teaches the task format, how to use the rubric, and the output style | Gives the judge a task-level anchor |
| Core content | Several input → score examples | The current sample's `Expected Answer` |
| Opposite | Zero-shot | Reference-free |

So all four combinations exist; a prompt can use either, both, or neither:

| | Reference-free | Reference-based |
| --- | --- | --- |
| **Zero-shot** | Give only the criterion and let the judge score on its own | Give the criterion and the current reference answer, with no demonstrations |
| **Few-shot** | Give a few scoring demonstrations, but neither the demonstrations nor the current sample has a reference answer | Both the demonstrations and the current sample carry reference answers |

One sentence to remember:

> **Few-shot controls whether there are demonstrations; reference-based controls whether the current answer has a gold answer to compare against.**

## First separate a few fields that are easy to mix up

```text
Task / Input       What is the original task?
Candidate          Which response or trajectory is being evaluated?
Criterion          Which single quality dimension is judged this time?
Rubric             What observable behavior does each score stand for?
Reference          An ideal answer, key facts, or a set of acceptable answers
Demonstrations     Scoring examples that teach the judge how to use the rubric
Evidence           Retrieved content, tool results, environment state, or other verification information
Verdict            A score, category, preference, or pass/fail
Rationale          The specific evidence that supports the verdict
```

The difference between a reference and a demonstration matters most: a reference is **the answer anchor for the current task**; a demonstration is **a teaching example of how to carry out the evaluation task**.

## How to write a criterion

A good criterion usually has five properties.

### 1. Judge one thing at a time

"Is this a high-quality response?" is too vague. Split it into correctness, faithfulness, relevance, completeness, and style.

If several dimensions must be merged into a total score, still judge them separately first, then state explicitly how they are combined.

### 2. Point to observable evidence

Don't just write "helpful." Say what helpful looks like in the current task, for example: does the response give actionable steps, does it cover the user's explicit constraints, does it avoid irrelevant content?

### 3. State the scope

Faithfulness can mean "relies only on the given context," or it can mean "consistent with real-world facts." The two tasks need completely different evidence.

### 4. Give boundaries and counterexamples

Say which cases must fail, which cases may still pass, and whether `unknown` is allowed when information is insufficient.

### 5. Give each score a behavioral anchor

`4 = good`, `3 = okay` does not help much. A better rubric is:

```text
5  Fully correct, covers every necessary constraint, no substantive omission
4  Core is correct, with only small omissions that do not affect use
3  Partly correct, but misses one important requirement or needs the user to supply a correction
2  Has major errors; only a small part of the content is usable
1  Wrong, irrelevant, or violates a key constraint
```

## Common criteria

Different tasks should choose different combinations, rather than running every metric by default.

### RAG / question answering

- **Answer correctness**: is the answer correct?
- **Faithfulness / groundedness**: can the conclusion be supported by the evidence provided?
- **Answer relevance**: does it directly answer the user's question?
- **Context relevance**: is the retrieved evidence related to the question?
- **Completeness**: does it cover the key points needed to complete the task?

### Agent

- **Task completion**: was the real task completed?
- **Tool correctness**: were the tool choice and the arguments correct?
- **State integrity**: does the external state after execution match expectations?
- **Policy compliance**: were permissions and inviolable constraints respected?
- **Efficiency**: are there loops that make no progress, repeated searches, or runaway cost?
- **Recovery quality**: after a tool failure, did it correctly explain, retry, or escalate to a human?

### Personalization / Model Experience

- **Intent fit**: does it fit the current intent, rather than only matching long-term preference?
- **Memory appropriateness**: is the memory it used relevant, still valid, and open to correction?
- **Breadth**: are the results overly repetitive, or do they trap the user in one narrow topic?
- **Control**: can the user understand and intervene in the key decisions?
- **Longitudinal improvement**: do corrections actually take effect in later sessions?

### Open-ended generation

- **Correctness**, **coherence**, **instruction following**, **tone**, **clarity**, **originality**.

The more subjective these dimensions are, the more they need an explicit rubric and human calibration.

## The main scoring modes

### 1. Binary / categorical

Output `pass / fail / unknown`, or one of a finite set of categories.

Good for well-defined conditions, such as "does it cite a fact that is not in the context?" It is usually easier to calibrate than a vague continuous score.

### 2. Anchored ordinal score

Output ordered levels such as 1–5, each with a clear behavioral anchor.

It fits cases that need to express severity, but don't assume the gap between 4 and 3 necessarily equals the gap between 3 and 2.

### 3. Pairwise comparison

Give two candidates, A and B, and let the judge choose `A / B / tie / both bad`.

Good for A/B comparison of models, prompts, or policies. It is usually more natural than absolute scoring but is prone to position bias, so swap the order and judge again.

### 4. Listwise ranking

Rank several candidates at once. It saves calls, but as the candidates multiply, context and order effects become more complex. It usually needs randomized order or grouped comparison.

### 5. QAG / decomposable ratio

First split a complex target into several answerable yes/no questions, then compute the score from the fraction that pass.

For completeness, for example, first extract five required points, then judge how many of them the response covers:

```text
completeness = required points covered / total required points
```

Where this score comes from is much clearer than directly asking the model to "rate completeness 0.73."

### 6. DAG / hierarchical gating

Check the inviolable conditions first, then evaluate open-ended quality. For example:

```text
Tool arguments wrong? → fail immediately
Fact not supported by the evidence? → score capped at 2
Otherwise → go on to judge helpfulness and quality of expression
```

The value of a DAG is that it expresses the evaluation logic; it does not automatically make LLM judgment deterministic.

## What a probability-weighted score actually is

Suppose the judge can only output 1–5. Taking one discrete score produces many ties, and you cannot see the model hesitating between 3 and 4.

If you can get the probabilities of the rating tokens, you can compute the expectation `E[s] = Σ s·p(s)`:

```text
score = 1·p(1) + 2·p(2) + 3·p(3) + 4·p(4) + 5·p(5)
```

For example:

```text
p(3)=0.10, p(4)=0.70, p(5)=0.20
score = 3×0.10 + 4×0.70 + 5×0.20 = 4.10
```

That is the core idea of probability-weighted scoring in G-Eval: you end up with a finer, continuous score rather than only integers.

## What "generate 20 scores" means

Two implementations need to be told apart here.

### Method A: read token probabilities directly

If the model API exposes log probabilities for output tokens, you get `p(1)...p(5)` directly and then compute the weighted expectation.

### Method B: approximate the distribution by repeated sampling

If the full probabilities are not available, have the judge sample independently several times. Say 20 runs give:

```text
score 3:  4 times
score 4: 12 times
score 5:  4 times
```

The empirical probabilities are `0.2 / 0.6 / 0.2`, and the final expected score is 4.0.

So "generate 20 scores and take a weighted sum" is not a new criterion; it is **one way to estimate the score distribution**.

## Storing only the weighted mean is not enough

Both of the distributions below have a mean of 3:

```text
A: 100% give a 3
B: 50% give a 1, 50% give a 5
```

A means the judge consistently finds the answer mediocre; B means the judge is extremely uncertain, or the rubric admits two conflicting readings.

So it is best to store all of these together:

- the weighted mean;
- the variance or standard deviation;
- the probability of each score;
- the agreement rate across repeated samples;
- the `unknown / abstain` rate.

If the distribution is clearly bimodal, inspect the criterion and the input evidence first, rather than letting the mean paper over the disagreement.

## How I would choose a scoring mode

| Scenario | Better starting point |
| --- | --- |
| There is an explicit rule or an executor | Deterministic checks; an LLM is not the first choice |
| Judging whether one semantic condition is violated | Binary + rationale + evidence |
| Comparing two models or prompts | Pairwise, swap the order, allow ties |
| Judging how severe a quality problem is | A 1–5 ordinal rubric with behavioral anchors |
| A complex target made of several sub-conditions | QAG or DAG decomposition |
| Need a finer ranking and the API has logprobs | Rating-token probability expectation |
| No logprobs, but stability needs estimating | Repeated sampling, reporting the distribution and the variance as well |
| A high-quality gold answer exists | Reference-based |
| There is no single correct answer | Reference-free + a clear rubric + human calibration |

## Common biases

- **Position bias**: in pairwise judging, favoring the answer that appears first or the one that appears last.
- **Verbosity bias**: mistaking longer and more detailed for better.
- **Self-enhancement bias**: the judge favors outputs that resemble its own model family.
- **Style bias**: being swayed by headings, Markdown, tone, and how confident the text sounds.
- **Reference anchoring**: when the reference answer is incomplete, the judge may wrongly penalize other valid solutions.
- **Scale bias**: persistently favoring a few scores, so the rating range is never actually used.
- **Evidence leakage**: prompt injection or self-evaluation inside the candidate influences the judge.

## A more robust minimal workflow

```text
1. Use deterministic rules first for every condition that can be verified exactly
2. Each judge judges only one criterion
3. Write a rubric with behavioral anchors
4. Provide a reference when there is one, but allow other equivalent answers
5. Calibrate the prompt and the thresholds on a small set of human-labeled samples
6. Pairwise: swap the order; pointwise: inspect the score distribution
7. Store the verdict, rationale, evidence, and evaluator version
8. Run slice analysis by task, user, and failure type
9. Periodically re-measure agreement between the judge and humans
```

## Recommended reading

- [G-Eval](https://arxiv.org/abs/2303.16634): CoT evaluation steps, form filling, and probability-weighted scoring.
- [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685): pointwise and pairwise evaluation, and common biases.
- [Prometheus](https://arxiv.org/abs/2310.08491): training a dedicated evaluator with fine-grained rubrics and reference answers.
- [Judging the Judges](https://arxiv.org/abs/2406.07791): a systematic analysis of pairwise position bias.

### Engineering practice

- [Confident AI Blog](https://www.confident-ai.com/blog): a fairly complete set of practical articles on LLM-as-a-Judge, G-Eval, Arena-style comparison, RAG evaluation, agent evaluation, and observability. A useful engineering companion for implementation ideas, but no substitute for the primary papers or for calibration experiments on your own task.

## How this connects to other chapters

- [Evaluation overview](README.en.md)
- [Agent Observability](../06-systems/agent-observability.en.md)
- [Human-in-the-Loop](../06-systems/human-in-the-loop.en.md)
- [Data and feedback](../01-data-and-feedback/README.en.md)
