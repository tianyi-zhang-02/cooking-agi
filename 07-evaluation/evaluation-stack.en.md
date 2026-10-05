# An answer went wrong. Which layer should you inspect?

[中文](evaluation-stack.md) · **English**

A user asks which evening is free this week. The assistant answers “Wednesday” and includes a citation. The format is valid and the prose is natural, but the cited record describes last month.

Would a score of 7 tell you what to fix? Perhaps old memory never expired, retrieval missed the new instruction, or the model received both records but confused their dates. **Evaluation should narrow the failure location, not merely rank outputs.**

We will use a fictional assistant with retrieval and memory. Start with [locating failures](#failure-location), then [controlled changes](#controlled-changes). No prior statistics course is needed.

## 1. Trace the request backward {#failure-location}

```text
Task → Available data and memory → Candidates → Selected context → Answer → Actual outcome
                   ↑                  ↑                ↑              ↑            ↑
            Was the fact stored? Was it retrieved? Was it retained? Was it used? Was the task done?
```

| Observation | First inspection | Likely area |
| --- | --- | --- |
| Correct instruction never entered storage | Write records, versions, permissions | Data or memory ingestion; a prompt cannot supply a missing fact |
| Stored instruction is absent from candidates | Exact versus ANN search, filter logs | Representation, index, or filtering |
| Candidate contains it, final context does not | Truncation, deduplication, reranking, token budget | Context selection |
| Context contains it, answer uses the old date | Temporal interpretation, citation support, conflict handling | Generation or evidence interpretation |
| Answer is correct, calendar was not changed | Tool response and final calendar state | Execution; “Done” is not proof of completion |

One request can have multiple failures. Preserve multiple labels if useful; define an additional earliest-failure convention for funnel reporting rather than mixing the two counting rules.

## 2. Match the check to the question

These methods are not ranked from basic to advanced, nor do they have a fixed cost order. An executor can be expensive; a human check can be quick. Choose based on what you need to establish.

| Method | Can answer | Cannot establish by itself |
| --- | --- | --- |
| Deterministic rules | Valid JSON? Existing citation ID? Allowed date? | An existing citation supports the claim |
| Reference or executor | Correct mathematical result? Passing tests? Actual calendar change? | Correct behavior outside test coverage |
| LLM judge | Does evidence support the claim? Which requirements are missing? | The model’s judgment is ground truth |
| Human review | How should ambiguity be handled? What does the rubric omit? | A single reviewer is necessarily consistent or unbiased |
| Online and longitudinal observation | Did users finish tasks or repeatedly correct the system? | A short-term click establishes long-term value |

If a citation ID does not exist, code already identifies a failure. If it exists but fails to support the claim, semantic inspection is needed. Keeping these checks separate also reveals whether a judge missed an obvious error or encountered genuinely ambiguous evidence.

## 3. How do 100 questions turn 80% into 55%?

Assume nested success conditions: each stage counts only examples that passed the previous stage.

| Stage | Remaining examples | Pass rate conditional on prior stage |
| --- | --- | --- |
| Fixed test set | 100 | — |
| Necessary evidence reaches the candidate set | 80 | 80% |
| Necessary evidence survives context selection | 70 | 87.5% |
| Key claims are supported | 60 | About 85.7% |
| The question is also answered completely | 55 | About 91.7% |

End-to-end success is $55/100$, not the final row’s $55/60$. Here $0.8\times(70/80)\times(60/70)\times(55/60)=0.55$ because the counts are nested; no stage-independence assumption is needed. Multiplying averages measured on unrelated samples has no equivalent interpretation.

Evaluating generation only where retrieval found evidence helps diagnose generation, but report excluded examples too. A worse retriever might leave only easy questions and make the conditional generation score look better.

## 4. What should a judge receive and return?

Separate “Is this key claim supported?” from “Does the answer address every requirement?” An answer can be grounded but incomplete, or apparently complete with fabricated details.

Supply the question, evidence actually available to the model, candidate answer, rubric, and necessary time/version context. Keep the decision, relevant evidence, and uncertainty reason in the output rather than a lone number.

```json
{
  "example_id": "demo-017",
  "claim": "Wednesday evening is allowed this week",
  "support": "insufficient",
  "evidence_ids": ["memory-old"],
  "reason": "The cited record applies to last month, not this week",
  "needs_review": true
}
```

This illustrates a response schema, not proof of correctness. Code can verify that IDs and quotations exist; string presence alone cannot establish entailment. Human spot checks should still connect the claim, evidence, and dates.

Reference answers can specify necessary facts without enforcing one exact wording. Allow `insufficient`, `tie`, or both-bad outcomes instead of forcing a good winner from bad answers. Swap A/B order to test position sensitivity, and calibrate with human-reviewed examples before scaling. See the [Judge chapter](llm-as-a-judge/README.en.md) for details.

## 5. How do you attribute a component improvement? {#controlled-changes}

Freeze candidates to test reranking. Freeze context to test generation. Then run the complete system to examine interactions and actual costs. Component experiments locate effects; end-to-end experiments test the assembled design. You need both.

| Experiment | Change | Temporarily hold fixed |
| --- | --- | --- |
| Retrieval | Encoder or retrieval policy | Corpus snapshot, queries, labels, candidate budget |
| Reranking | Ranker or fusion rule | Input candidates and downstream context budget |
| Generation | Prompt or generator | Evidence, decoding convention, evaluation criteria |
| Complete system | Full design | Tasks, total allowed budget, success criteria |

Manually supplying correct evidence is a useful oracle diagnostic: failure even then means retrieval is not the only issue. It is not a deployable system or an overall performance number. Likewise, changing one component may require recalibrating others; fixed-component tests cannot replace final end-to-end evaluation.

## 6. What does a first evaluation round look like?

Build a small set covering ordinary queries, temporal conflicts, unanswerable questions, unauthorized content, truncated evidence, and tasks requiring actual execution. Separate rubric-development examples from final holdouts; do not claim that a small set represents every user.

Save inputs, system versions, intermediate candidates, context, outputs, tool results, and failure reasons. Run deterministic checks, then semantic judgments, then inspect disagreements and boundary cases. Fixed examples detect regressions; new independent examples reveal failures you had not anticipated.

Read the report with three questions: how many tasks succeeded, where did failures concentrate, and what would fixing them cost? Continue with [metric robustness](metric-robustness.en.md) before deciding how broad a claim the gain supports.
