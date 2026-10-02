# Minimal implementation: finish one evaluation end to end

[中文](implementation.md) · **English**

No platform or API key is needed yet. We'll process fictional judge outputs through parsing, validation, and aggregation. The goal is to prevent invalid outputs from becoming approvals and failed calls from disappearing from the denominator.

## Run it locally

From the repository root:

~~~bash
python3 07-evaluation/llm-as-a-judge/code/judge_eval.py
~~~

[Full Python implementation](code/judge_eval.py) · [Example JSONL](code/example-results.jsonl)

The standard-library-only program reads local files without networking. It reports totals, decided cases, abstentions, invalid formats, call errors, and a confusion matrix. The fixture intentionally includes a false approval, false rejection, unknown, malformed JSON, and a failed call.

Edit a judge_raw value and rerun it to see coverage and errors change. **There is no training, live judge, or measured model performance here.**

## Inputs and outputs

A record contains both the human label and raw judge output for post-hoc meta-evaluation. **Do not send the human label to the judge** in a real request.

~~~json
{
  "case_id": "toy-01",
  "group_id": "conversation-01",
  "slice": "rag",
  "human": "pass",
  "allowed_evidence_ids": ["policy-1"],
  "judge_raw": "{\"criterion_id\":\"policy-grounding-v1\",\"verdict\":\"pass\",\"evidence_ids\":[\"policy-1\"],\"reason\":\"Eligibility and deadline are supported.\"}"
}
~~~

Send task, candidate, criterion, and allowed evidence. Include a reference only when deliberately using reference-based evaluation. Keep human labels in a separate comparison table.

The program validates verdict values, criterion_id, field types, unknown evidence IDs, and duplicate case IDs. It cannot establish that citing policy-1 makes the reasoning correct; semantic review is still needed. Extra output fields are rejected to prevent consumers from relying on an undefined contract.

## Four states, not one score

| Status | Meaning | Treatment |
| --- | --- | --- |
| decided | Valid pass/fail | Include in conditional accuracy and confusion matrix |
| unknown | Valid output, insufficient evidence | Count abstention and route for review |
| invalid | Malformed JSON or schema | Evaluator error, not automatic candidate failure |
| error | The call did not return successfully | Infrastructure failure; retain the attempt |

Demo coverage uses **every input case** as its denominator, with failures shown separately. After adding an API, record retries individually too. Keeping only successful final attempts doesn't justify claiming 100% call success.

## Where a real model goes

~~~text
Frozen cases / evidence
→ deterministic checks
→ judge request without human labels
→ selected provider adapter
→ raw response + versions + cost + attempt
→ validation / unknown / error routing
→ join human labels by case_id
→ criterion metrics + slices + human audit
~~~

Keep the adapter at the call step rather than mixing SDK calls, rubrics, and aggregation into one prompt. Use bounded retries for timeouts or rate limits and retain attempt counts and final failures. Begin with a small redacted batch and a budget limit.

Join parallel results by case_id, not completion order. Cache keys need candidate, evidence, rubric, model, and sampling configuration, or a changed rubric can silently reuse old scores.

## A pre-release checklist

- Explicit rubric boundaries, hard gates, and unknown semantics.
- No case leakage across development, calibration, and holdout; no human-label leakage into requests.
- Order, length, reference conflict, injection, and missing-evidence tests.
- Human comparisons with false approvals, false rejections, coverage, and slice counts.
- Samples from pass, fail, unknown, and parser errors—not only attractive explanations.
- Parallel re-evaluation before a judge or prompt change, with traceable versions and rollback.
- A report distinguishing offline evidence from claims about online experience.

This teaching implementation is not a production framework. It lacks an API adapter, persistence, access controls, and formal uncertainty estimation. It provides a small, inspectable starting point.

<details markdown="1">
<summary>Exercise: add a nonexistent evidence ID to a sample response</summary>

The result should become invalid even if the raw verdict says pass. Then try verdict unknown: that is valid abstention, not a schema error. The report needs to distinguish those paths.

</details>

Return: [Series overview](README.en.md) · [Calibration](calibration.en.md)
