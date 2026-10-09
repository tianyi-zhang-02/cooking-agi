# Compare experiments and replace models safely

[中文](experiments-and-release.md) · **English**

> Original teaching example · Reviewed: 2026-10. Pass rates below are hand-built examples, not model results.

After training, the tempting question is “How much did it improve?” An aggregate move from 60% to 70% doesn't tell you which questions improved, which regressed, or whether evaluation added an extra hint.

## Save enough information to repeat the experiment

You don't need a large platform first. A manifest saved with the results prevents many avoidable mix-ups:

```json
{
  "run_id": "citation-sft-demo-01",
  "base_revision": "local-fixture-v1",
  "data_snapshot": "examples-v3",
  "split_rule": "question-family-v1",
  "template_revision": "chat-template-v2",
  "objective": "assistant-target-token-mean",
  "eval_snapshot": "fixed-evidence-v2",
  "decode": {"sampling": false, "max_new_tokens": 128}
}
```

These are illustrative fields, not downloadable checkpoint names. Real runs also need content hashes, dependency versions, seeds, optimizer settings, training steps, and valid-token counts. A run ID indexes artifacts; it cannot replace a data snapshot.

If A gets 128 output tokens and B gets 512, the comparison can still be useful—but it compares systems with different budgets, not training alone.

## Six questions reveal what an aggregate hides

`1` means passing a predefined check; `0` means failing. Put old and new outcomes for the same question on one row:

| Question | Slice | Old | New | Change |
| --- | --- | --- | --- | --- |
| q1 | Evidence available | 0 | 1 | Win |
| q2 | Evidence available | 0 | 1 | Win |
| q3 | Evidence available | 1 | 1 | Tie |
| q4 | Evidence available | 1 | 1 | Tie |
| q5 | Evidence missing | 1 | 0 | Loss |
| q6 | Evidence missing | 1 | 1 | Tie |

Overall performance rises from `4/6` to `5/6`. But q5, previously handled correctly without evidence, now fails. That slice drops from `2/2` to `1/2`. The new model might answer better—or simply be more willing to answer. Inspect q5's actual output, then check the quantity and quality of missing-evidence demonstrations.

A **paired comparison** retains the change on each question: two wins, one loss, three ties. Six cases illustrate regression detection, not a stable improvement estimate. Real comparisons need sample sizes and uncertainty; resampling should respect related questions from the same user or document.

## Record incomplete cases separately

If the old model completes 100 questions and the new one completes only 80, averaging each model's successful responses can introduce selection bias. Our small implementation rejects mismatched IDs so incomplete pairs require explicit handling.

A real evaluator can retain `ok / timeout / invalid_output` per case and report two views:

- **Did the user get a usable result?** Use all requests as the denominator; timeouts and malformed outputs affect task success.
- **Which model answered better when both completed validly?** Compare that subset, with counts and reasons for exclusions. This is conditional performance, not an estimate automatically valid for all requests.

Both views matter. A timeout isn't a factual error, but it cannot disappear from the report. If the task requires JSON, invalid formatting can itself count as task failure.

```bash
python3 practice/post-training/code/experiment_checks.py
```

The [code](code/experiment_checks.py) prints the six-case overall and slice results. Removing a new-model result causes a missing-pair error; assigning a question family to both training and testing causes a leakage error.

## Increase cost in stages

| Stage | First check | Only then move to |
| --- | --- | --- |
| One example / batch | Tokens, masks, loss, finite gradients | Tiny-dataset training |
| Tiny dataset | Learn a few clear examples without breaking unsupported-answer behavior | More data and longer runs |
| Save and restore | Output comparisons under fixed settings, optimizer/scheduler and required random state | Long runs and interruption recovery |
| Full offline evaluation | Fixed protocol, slice regressions, resource cost | Controlled release with explicit rollback |

If a few examples cannot be learned, inspect inputs, objectives, and implementation before adding GPUs. Memorizing those examples doesn't establish generalization either; it is a training-path check.

See [training choices](training-plan.en.md), [multi-GPU batches](distributed-training.en.md), and [recovery comparisons](checkpoint-and-resume.en.md) for implementation decisions. GPU count alone is not a resource report: include valid target tokens, end-to-end time, peak memory, and failed-work replay costs.

If only offline evaluation is available, keep the conclusion offline. User simulation can stress-test behavior, but it cannot establish real-user product benefit.

## Release a compatible bundle

Weights belong with the tokenizer, template, generation settings, preprocessing, and tool/output schemas. Rolling back weights while retaining a new template may not restore the old system.

First run a shadow version on matching inputs without changing user-visible outputs, checking format, latency, and failure types. Then decide whether controlled online validation is appropriate. Shadow traffic cannot measure changed user behavior because users never saw the new outputs. Define stop conditions in advance, such as key-slice regressions, excessive invalid outputs, or latency beyond the product budget. Thresholds follow risk, not this six-question table.

“Don't release this run” is a valid conclusion. Explaining why, and what to test next, is more useful than presenting every training run as a win.

[Project introduction](README.en.md) · [Compare the recommender release path](../recommender-systems/08-serving-lifecycle.en.md)
