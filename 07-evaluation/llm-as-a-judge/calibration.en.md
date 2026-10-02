# Calibrating a judge: inspect its mistakes, not its confidence

[中文](calibration.md) · **English**

A judge can sound expert and give detailed reasons while missing the failures that matter most. Calibration needs concrete decisions: **what to block, what to accept, and where uncertain cases go.**

## Prepare human comparisons before tuning prompts

Separate the uses of your data:

| Split | Purpose |
| --- | --- |
| Rubric development | Discover error types, revise definitions, write demonstrations |
| Calibration / validation | Compare judges, choose thresholds and review rules |
| Frozen holdout | Final assessment without tuning to the observed results |
| Fresh audit | New samples to check changing distributions and versions |

These are roles, not mandatory proportions. Cross-validation can help with limited data, but tuning and final reporting still need separation. Don't split one user's data, paraphrases of the same problem, or one agent trajectory across sets in ways that leak information.

People disagree too. Label independently before discussing boundaries, and preserve pre-adjudication disagreement. Unclear requirements call for a better rubric.

## Try it: does a higher threshold improve everything?

<div data-judge-lab="calibration"><p>This interaction needs JavaScript. Higher thresholds can reduce erroneous approvals while blocking good answers; wider review bands reduce automatic coverage.</p></div>

These 12 examples are **deliberately constructed teaching data**, split into RAG and agent cases. Their numeric ratings are fictional scores, not correctness probabilities. An acceptable aggregate can conceal a poor slice.

Follow the individual tiles instead of watching only the percentages:

1. Keep all cases, a 3.5 threshold, and review off. R4 and A4 have high scores but are false approvals. **A high score isn't evidence of correctness.**
2. Raise the threshold to 4.0. False approvals fall from 3 to 2, while false rejections rise from 2 to 3. Accuracy stays at 7/12. The total looks unchanged, but the mistakes differ.
3. Turn on human review. Six cases leave the automatic decision set, so only six remain in the table. You haven't eliminated six errors; you've handed six decisions to a person.

The 2×2 table crosses whether the human thinks a case should pass with whether the rule actually approves it. Understand those two axes before memorizing TP and FP.

Near-threshold cases can go to human review. That doesn't make them automatically correct; it defers a decision and costs attention and time. Exploring thresholds here is educational. A real holdout isn't a tuning set.

## What a useful report includes

Treat human pass as the positive class. Accepting a human fail is a **false positive**; rejecting a human pass is a **false negative**.

$$\text{coverage}=\frac{N_{\text{automatic pass/fail}}}{N_{\text{all attempts}}}.$$

$$\text{error among approvals}=\frac{N_{\text{human fail, judge pass}}}{N_{\text{judge pass}}}.$$

The second quantity asks how many approvals are actually wrong. It is not the false positive rate, whose denominator is all human-fail examples. Use clear names and denominators. A zero denominator means N/A, not 0%.

| Report item | Why it matters |
| --- | --- |
| Confusion matrix | Separate erroneous approvals from erroneous rejections |
| Coverage + unknown + errors | High conditional accuracy can coexist with frequent abstention |
| Slice counts and errors | Language, long context, or tool failures may disappear in the aggregate |
| Human disagreement | Some tasks have genuinely difficult boundaries |
| Item-level and system-level results | Ranking two systems correctly doesn't imply catching individual failures |

If 95 of 100 cases are passes, an always-pass judge achieves 95% accuracy while missing every failure. That number alone doesn't establish usefulness.

## Agreement, correlation, calibration, and stability

- **Agreement:** whether the judge and human give the same category for an item.
- **Correlation:** whether rankings agree; high correlation doesn't imply correct item-level decisions.
- **Calibration:** whether predictions interpreted as 0.8 pass probability actually pass about 80% of the time.
- **Reliability / stability:** repeatability across runs or order changes; consistently wrong is still consistent.

Cohen's kappa can supplement raw agreement but depends on class distribution and annotation assumptions. Weighted kappa may suit ordinal labels; Spearman or Kendall suit rankings. Don't choose whichever metric looks best and discard the rest.

Without additional calibration, 4/5, token probability, and self-reported confidence are not correctness probabilities. Agreement among models is not automatically independent evidence.

## Can we trust a small improvement?

Compare A/B on the same tasks, retain per-task differences, and estimate uncertainty with methods such as paired bootstrap over independent task units. Cluster multi-turn data by conversation. Twenty judgments of one question are not twenty independent questions.

With few independent tasks, interpret intervals cautiously. Report effect sizes, intervals, data sources, and effective sample counts. Required sample size depends partly on the smallest effect worth detecting; there is no universal “100 cases is enough.”

Repeatedly selecting models and thresholds against one validation set can adapt to it. Preserve a final holdout. A changed rubric needs a new version, not scores quietly combined with the old scale.

<details markdown="1">
<summary>If the judge disagrees with a human, is the judge necessarily wrong?</summary>

No. Inspect the evidence, instructions, and annotation quality, with independent review where necessary. But don't change human labels merely because the model gives a persuasive explanation. Record reasons for corrections instead of revising the answer key to fit the evaluator.

</details>

Continue: [Task-specific examples](case-studies.en.md) · [Minimal implementation](implementation.en.md)
