# AI Agent Observability: What Did the Agent Actually Just Do?

[中文](agent-observability.md) · **English**

## Quick learning: what does agent observability observe?

<details class="interview" markdown="1">
<summary>Traces, state transitions, and reproducible failures</summary>

**Quick memory**: metrics tell you where something is abnormal, logs give local events, and traces reconstruct the causal chain across model, retrieval, tools, and state within one run.

**Interview answer**

> The unit of agent observability is not a single API call but a stateful trajectory. Every span should record versions, input evidence, decisions, tool arguments and results, state changes, latency, tokens or cost, and the final outcome, all tied together by a trace ID to support replay and root-cause analysis.

<details markdown="1">
<summary><b>Deep dive</b>: why is “storing every prompt” still not observability?</summary>

Raw text has no explicit causal structure and may leak sensitive data. An observable system needs structured spans, parent-child relationships, versions, and state diffs, with PII redacted. The goal is to be able to tell “retrieval was wrong, the model was wrong, the tool was wrong, or the evaluator was wrong,” not to produce longer logs.

</details>
</details>

## Observability has to explain the decision process

The goal of agent observability is not only to know whether a service threw an error, but to be able to reconstruct: **what the agent saw, which decisions it made, how state changed, and at which step the failure began.**

The final answer and an error code are not enough. Failure may originate in retrieval, stale memory, tool behavior, state transitions, loops, evaluation mismatch, or an incorrect model decision over correct evidence.

## What happened inside one failed run

The user asks the agent to book a restaurant for Friday, and it ends up booked for Saturday.

Looking only at the final answer, all we know is that the date is wrong. A full trace might tell us:

```text
Original user request: Friday
Memory read: last week's conversation mentioned a Saturday dinner
Model decision: prioritize the old memory
Tool call: date=Saturday
Tool result: success
Final answer: booking confirmed
```

The tool did not fail, and the model did not hallucinate an order that does not exist. The real problem is that an old memory wrongly overrode an explicit current instruction.

## How metrics, logs, and traces differ

| Form | The question it answers best |
| --- | --- |
| Metrics | Did the overall error rate, latency, or cost change today? |
| Logs | What event did a given component report at a given time? |
| Traces | In what order did one request pass through which decisions and dependencies? |
| Replay | With the same versions and evidence, can this failure be reproduced? |
| Evaluation | Did this complete trajectory satisfy the task requirements? |
| Slicing | Which users, tasks, tools, or environments concentrate failure? |

Agents especially need traces, because one result is usually produced jointly by many model, retrieval, and tool calls.

## What one run should record

### Versions

Which version of the model, prompt, retriever, tools, memory policy, evaluator, data, and configuration was in use.

Alongside the versions, record index and feature versions and the identifiers that tie the run together: session, user, task, trace, and experiment IDs.

### Input evidence

Which instructions, memory, retrieved results, and tool outputs the model saw; which content was truncated, filtered, or reordered.

Also record how the token, latency, and cost budget was allocated.

### Decision process

Which action was chosen at each step, why it continued or stopped, and whether there was a retry, fallback, timeout, or human takeover.

### State changes

What changed in the user state, the task state, and the external system state before and after execution.

### Final outcome

Whether the task was really completed, rather than only whether a plausible-looking piece of text was generated.

Keep the evaluation with it: deterministic invariants, reference checks, and LLM-judge results, plus user corrections, subsequent behavior, and long-term outcomes.

## A trace should not be just a longer log

A good trace is organized around the causal chain. Common spans can include:

- `model`: context, output, tokens, latency, and version;
- `retrieval`: query, candidates, scores, filtering, and the final evidence;
- `tool`: arguments, results, exceptions, retries, and side effects;
- `memory`: reads, writes, compression, forgetting, and confidence changes;
- `policy`: routing, stopping, fallback, and risk judgments;
- `evaluation`: criteria, evidence, verdict, and evaluator version;
- `human_review`: why it was escalated, how the human changed it, and the rationale for the change.

## Common agent failures

- **Loop:** the same tool is called repeatedly while state does not materially change.
- **Goal drift:** later steps have already departed from the user's original task.
- **Memory contamination:** wrong, stale, or other-context information enters long-term state.
- **Tool-semantics mismatch:** the API behavior the model assumes differs from the actual implementation.
- **Silent fallback:** after falling back, the system still answers in its normal tone, and the user does not know quality has dropped.
- **Evaluation blind spot:** only the final text is checked, not the process, the tools, or the side effects.
- **Cost runaway:** more tokens, searches, and retries bring no new progress.

## What to do after recording

The end point of observability is not a dashboard. It is the ability to produce three things:

1. a stable, reproducible failure case;
2. a complete trajectory that can be added to the offline eval set;
3. a clear direction for the change: should it be the prompt, search, the tools, memory, the policy, or the training data?

If a trace cannot help answer “what do we change next,” it is closer to an expensive log warehouse.

## Connections to other chapters

- [Evaluation](../07-evaluation/README.en.md) decides how to judge a trajectory as a success or a failure.
- [Human-in-the-Loop](human-in-the-loop.en.md) decides which trajectories need human intervention.
- [Representation and memory](../02-memory/README.en.md) explains how state gets written and contaminated.
- [Search](../04-search/README.en.md) explains how evidence enters the context.
