# Scoring modes: what decision will this result support?

[中文](scoring.md) · **English**

Checking a violation and choosing between two prompts need not use the same output. Don't turn every question into a numerical rating just because the API can return one.

## Common options, side by side

| Mode | Example | Useful for | Trade-off |
| --- | --- | --- | --- |
| Binary / categorical | pass, fail, unknown | Specific conditions | Define boundaries and count unknown separately |
| Ordinal | Anchored 1–5 | Severity of errors or omissions | 4→5 need not equal the improvement from 2→3 |
| Pairwise | A, B, tie, both_bad, unknown | Comparing candidates on the same task | Order effects, extra calls; better doesn't mean acceptable |
| Listwise | Rank a candidate set | Whole-set comparisons | Long context and ordering effects; fewer calls may not cost less |
| Checklist / QAG | Cover 3 of 5 requirements | Decomposable conditions | Bad decomposition yields a precise but useless ratio |
| Gated / DAG | Hard checks before quality | Non-compensable failures | Upstream mistakes propagate; LLM nodes remain uncertain |

Reference-based evaluation concerns the information supplied, not a competing output format. Probability weighting aggregates a score distribution; it isn't a new quality criterion.

## Swap A/B without losing answer identity

<div data-judge-lab="pairwise"><p>This interaction needs JavaScript. Always choosing the first slot changes answer identity after a swap. Map choices back to stable IDs before checking consistency.</p></div>

Three deliberately simple “judges” follow policy, always pick the first slot, or prefer the longer response. They are fixed rules, not models, designed to make failure modes visible.

Keep the task, rubric, and answers unchanged; swap only their display order. Choosing A first and then choosing “the first answer” again does **not** necessarily support A twice.

For example, record:

~~~json
{"case_id":"return-01","order":["answer-A","answer-B"],"winner":"answer-A"}
{"case_id":"return-01","order":["answer-B","answer-A"],"winner":"answer-A"}
~~~

Both results map to the same answer ID here. Define how ties, both_bad, and unknown count as well. A consistently chosen answer can still be wrong: consistency isn't correctness. Read the [position-bias study](https://arxiv.org/abs/2406.07791) alongside this demonstration.

## Decide the win-rate denominator in advance

For 20 independent tasks, suppose A wins 8, B wins 6, with 4 ties, 1 both_bad, and 1 unknown.

One explicit convention records unknown as abstention and both_bad as task failure, then computes a quality win rate on the remaining 18:

$$W_A=\frac{8+0.5\times4}{8+6+4}=\frac{10}{18}.$$

This is a convention, not the only valid formula. Report all 20 tasks, the abstention, and the both_bad case alongside 55.6%. A relative win is not permission to ship when both versions fail the minimum requirements.

Pairwise results can also cycle: A beats B, B beats C, C beats A. Elo or Bradley–Terry can fit overall rankings under assumptions; a few comparisons do not reveal a single true model ability.

## Decompose a checklist carefully

For “explain eligibility, deadline, and application steps,” check three observable requirements instead of asking for a completeness score of 0.83.

They may not be equally important. Adding “nice punctuation” should not compensate for false eligibility information. Specify whether a requirement is inapplicable or missing rather than silently changing the denominator.

QAG generally decomposes evaluation using questions and answers; a DAG represents dependent evaluation steps. Framework terminology and APIs vary. Neither name guarantees a good measurement.

## Keep hard gates separate from quality scores

~~~text
Unparseable judge JSON → evaluator error, not automatic candidate failure
Necessary evidence absent → unknown / human review
Confirmed permission violation or incorrect final state → criterion fail
Otherwise → separate grounding, completeness, and expression judgments
~~~

You can record every dimension while defining the release gate separately. An agent that placed an unauthorized order shouldn't pass because its wording was polite.

The JSON above is **the judge's grading output**. If the candidate's task requires valid JSON and the candidate returns malformed output, its format criterion can fail directly. Identify whose output broke before assigning the error to the evaluation or task layer.

Continue: [Probability scores](probability-scores.en.md) · [Task examples](case-studies.en.md)
