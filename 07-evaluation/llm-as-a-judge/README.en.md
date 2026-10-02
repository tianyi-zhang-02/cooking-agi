# LLM-as-a-Judge: what makes an evaluation useful?

[中文](README.md) · **English**

You change a prompt. The answers read better, and the average score goes up. Are users actually getting better answers?

LLM-as-a-Judge means asking a model to evaluate another model's output against a set of criteria. It can help us read many answers. But **producing a score and knowing whether to trust it are different jobs**.

Start with one support answer, not a list of metrics. We'll follow it through evidence, criterion-level judgments, a decision, and a check on the judge. The example uses authored rules and judgments, not a live model.

## Walk through one answer: would you send it?

<div data-judge-lab="walkthrough"><p>A fictional return answer can be relevant and friendly while inventing the policy. Passing 2 out of 3 criteria doesn't make that error acceptable. The interaction lets you change the answer, hide the policy, and compare two decision rules.</p></div>

Try leaving “Friendly, wrong policy” selected and opening step 3. Why does it pass a majority rule? Turn on required checks: **the judgments haven't changed, but the decision has**. Then choose the complete answer and hide the policy. The missing evidence doesn't prove the answer is wrong; it prevents the judge from checking it.

There are three distinct jobs:

- **The model being evaluated** writes the answer.
- **The judge** applies a rubric and gives criterion-level judgments with evidence.
- **The decision rule** accepts, blocks, or requests review. It shouldn't be invented on the spot by the model.

The same judgment may serve different purposes. Comparing two prompts offline and automatically sending an answer to a user have different error costs. One passing answer doesn't establish that a system is ready to ship.

## Three parts, one learning path

<div class="judge-route" aria-label="A reading path in three parts">
<section><span class="jr-number">01 · Define</span><h3>What would a good answer do?</h3><p>Turn “looks good” into checkable criteria. Decide whether you need labels, scores, or comparisons.</p><ul><li><a href="criteria.en.md">Criteria: what are we checking?</a></li><li><a href="scoring.en.md">Scoring: pass/fail, ratings, or pairs</a></li></ul><small>Leave with a rubric and boundary examples.</small></section>
<section><span class="jr-number">02 · Validate</span><h3>Where does the judge get it wrong?</h3><p>Look beyond the mean. Swap order, remove evidence, and compare against human judgments.</p><ul><li><a href="probability-scores.en.md">What do 20 scores tell us?</a></li><li><a href="bias-and-workflow.en.md">Which changes should leave judgments alone?</a></li><li><a href="calibration.en.md">False approvals, rejections, and review</a></li></ul><small>Report errors, sample counts, and uncertainty.</small></section>
<section><span class="jr-number">03 · Use</span><h3>Bring it back to a real task</h3><p>Different tasks need different evidence. Connect validation, failure handling, and reporting.</p><ul><li><a href="case-studies.en.md">RAG, agents, and memory</a></li><li><a href="implementation.en.md">Run a small Python workflow</a></li></ul><small>Leave an auditable record, not just a total score.</small></section>
</div>

**New to judges?** Follow 01 → 02 → 03. **Already using one?** Start with [bias checks](bias-and-workflow.en.md) and [calibration](calibration.en.md). **Ready to build?** Bring one rubric to the [minimal implementation](implementation.en.md), then return to the concepts you need.

Each page tackles one question. There's no checklist to memorize before moving on, and no need to finish everything in one sitting.

## Why ask a model when a rule would do?

A parser is more useful than “the format looks correct” for checking JSON. To find out whether an agent saved a draft, inspect the final state.

But identifying a missing qualification in a summary or checking whether an answer addresses the user's real question isn't always expressible as a regular expression. That's where an LLM judge may help. **Use deterministic checks for directly testable properties, and semantic judgment for the gaps.**

| What are you checking? | Start with | Keep in mind |
| --- | --- | --- |
| Format, numbers, tests, final state | Parsers, tests, state assertions | Passing tests only covers what those tests inspect |
| Meaning, omissions, support, preferences | Explicit rubrics + LLM or human judgment | Fluent reasons don't establish correct judgments |
| Whether the judge is reliable | Independent human comparisons, boundary cases, perturbations | Agreement among models isn't automatically ground truth |

[MT-Bench / Chatbot Arena](https://arxiv.org/abs/2306.05685) documents biases including position and verbosity. [JudgeBench](https://arxiv.org/abs/2410.12784) asks whether judges can distinguish correct from incorrect answers on harder problems. Being good at answering doesn't automatically make a model a good evaluator for your task.

One more distinction: **observing a score and optimizing against it create different risks.** Once it becomes a reward, a system has an incentive to find ways to raise it. Independent assessment matters even more: don't use one teacher both to supply all labels and as the only evidence that its student improved. [Bias and workflow](bias-and-workflow.en.md) returns to this.

## What does each term do?

| Term | In the example |
| --- | --- |
| Criterion | Is the policy correct? Are the application steps included? |
| Rubric | What passes, fails, or remains unknown, including boundary cases |
| Evidence | The store's return policy |
| Reference | An acceptable answer to this question, not necessarily the only wording |
| Few-shot examples | Other scored cases demonstrating how to apply the rubric |
| Verdict | Pass, fail, or unknown for each criterion |
| Meta-evaluation | Checking the judge's errors against human judgments |

Few-shot asks whether to supply demonstrations. Reference-based asks whether to supply a reference for the current question. They can be used together or separately. [The next chapter develops this distinction](criteria.en.md).

## Further reading

- [G-Eval](https://arxiv.org/abs/2303.16634): rubrics and probability-weighted scoring.
- [Prometheus 2](https://arxiv.org/abs/2405.01535): training specialized evaluators; still calibrate on your task.
- [Judging the Judges](https://arxiv.org/abs/2406.07791): measuring position bias.
- [Replacing Judges with Juries](https://arxiv.org/abs/2404.18796): one approach to judge panels, not a correctness guarantee.
- [Confident AI Blog](https://www.confident-ai.com/blog): more practical examples. Choosing a framework doesn't define “good” for you.

Every interaction in this series uses fictional data, without uploads or model calls. The scores and judgments explain mechanisms; they are not measurements of a real model.

Continue: [Criteria](criteria.en.md) · [Evaluation overview](../README.en.md)
