# LLM-as-a-Judge: biases and a robust workflow

[中文](bias-and-workflow.md) · **English**

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
