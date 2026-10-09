# AI Agent Observability: What Did the Agent Actually Just Do?

[中文](agent-observability.md) · **English**

## Observability has to explain the decision process {#observability-has-to-explain-the-decision-process}

The goal of agent observability is not only to know whether a service threw an error, but to be able to reconstruct: **what the agent saw, which decisions it made, how state changed, and at which step the failure began.**

The final answer and an error code are not enough. Failure may originate in retrieval, stale memory, tool behavior, state transitions, loops, evaluation mismatch, or an incorrect model decision over correct evidence.

## What happened inside one failed run {#what-happened-inside-one-failed-run}

The user asks the agent to book a restaurant for Friday, and it ends up booked for Saturday.

Looking only at the final answer, all we know is that the date is wrong. Suppose the records show:

```text
Original user request: Friday
Memory read: last week's conversation mentioned a Saturday dinner
Model-produced tool arguments: date=Saturday
Tool call: date=Saturday
Tool result: success
Final answer: booking confirmed
```

The tool succeeded, but it received the wrong date. Because the old memory matches that date, interference from stale memory is worth investigating. **The trace alone does not prove causation.** In a test environment that cannot place real orders, hold other inputs fixed and compare keeping, removing, and updating that memory. A model's retrospective explanation is not sufficient evidence of why it acted.

## How metrics, logs, and traces differ {#how-metrics-logs-and-traces-differ}

| Form | The question it answers best |
| --- | --- |
| Metrics | Did the overall error rate, latency, or cost change today? |
| Logs | What event did a given component report at a given time? |
| Traces | In what order did one request pass through which decisions and dependencies? |
| Replay | With the same versions and evidence, can this failure be reproduced? |
| Evaluation | Did this complete trajectory satisfy the task requirements? |
| Slicing | Which users, tasks, tools, or environments concentrate failure? |

Agents especially need traces, because one result is usually produced jointly by many model, retrieval, and tool calls.

## What one run should record {#what-one-run-should-record}

### Versions {#versions}

Record the model, prompt, retriever, index, tools, memory policy, evaluator, data, and configuration versions. Connect the run through task, trace, and experiment IDs. Include user identifiers only when necessary, rather than logging email addresses or similar fields by default.

### Input evidence {#input-evidence}

Which instructions, memory, retrieved results, and tool outputs the model saw; which content was truncated, filtered, or reordered.

Also record token, latency, and cost budgets, so “nothing found” can be distinguished from “stopped because the budget ran out.”

Recording evidence does not mean copying all raw content. Prefer versioned IDs and controlled references where these suffice. When raw content is necessary, restrict access, retention, and purpose, and exclude credentials and unrelated personal data. Hashing a predictable user ID is not necessarily anonymization; see [OpenTelemetry's sensitive-data guidance](https://opentelemetry.io/docs/security/handling-sensitive-data/).

### Decision process {#decision-process}

Record the action taken, the rule that triggered continuation or stopping, and any retry, fallback, timeout, or human takeover. Keep system rules, observable outputs, and retrospective hypotheses distinct. This does not require access to a model's private reasoning.

### State changes {#state-changes}

What changed in the user state, the task state, and the external system state before and after execution.

### Final outcome {#final-outcome}

Whether the task was really completed, rather than only whether a plausible-looking piece of text was generated.

Link deterministic checks, judge versions, user corrections, and subsequent outcomes to the run. Distinguish a tool reporting success from the task actually succeeding.

## A trace should not be just a longer log {#a-trace-should-not-be-just-a-longer-log}

A trace connects call dependencies and state changes. It supports hypotheses about a failure, but does not by itself prove which component caused it. Common spans can include:

- `model`: context, output, tokens, latency, and version;
- `retrieval`: query, candidates, scores, filtering, and the final evidence;
- `tool`: arguments, results, exceptions, retries, and side effects;
- `memory`: reads, writes, compression, forgetting, and confidence changes;
- `policy`: routing, stopping, fallback, and risk judgments;
- `evaluation`: criteria, evidence, verdict, and evaluator version;
- `human_review`: why it was escalated, how the human changed it, and the rationale for the change.

## Common agent failures {#common-agent-failures}

- **Loop:** the same tool is called repeatedly while state does not materially change.
- **Goal drift:** later steps have already departed from the user's original task.
- **Memory contamination:** wrong, stale, or other-context information enters long-term state.
- **Tool-semantics mismatch:** the API behavior the model assumes differs from the actual implementation.
- **Silent fallback:** after falling back, the system still answers in its normal tone, and the user does not know quality has dropped.
- **Evaluation blind spot:** only the final text is checked, not the process, the tools, or the side effects.
- **Cost runaway:** more tokens, searches, and retries bring no new progress.

## What to do after recording {#what-to-do-after-recording}

The end point of observability is not a dashboard. It is the ability to produce three things:

1. a failure case with explicit inputs, environment, and failure conditions;
2. a complete trajectory that can be added to the offline eval set;
3. a clear direction for the change: should it be the prompt, search, the tools, memory, the policy, or the training data?

Be specific about what “replay” means:

| Method | What it tells you | Boundary |
| --- | --- | --- |
| Inspect historical records | Which inputs, calls, and outcomes were recorded | Not re-execution; sampling or redaction may leave gaps |
| Freeze tool responses and rerun in a sandbox | Whether behavior changes when a component changes | The external environment is frozen; this is not an online result |
| Call services again in a test environment | Whether failure recurs with current dependencies | Versions, sampling, and external state may have changed |

Replaying a trace must not place another booking, payment, or email by default. Start with stubs or dry-run execution; actual writes need fresh authorization, idempotency checks, and a recovery plan. Matching versions and random seeds still does not guarantee identical outputs from external APIs or different hardware. Fully documenting one failure is a better starting point than adding many dashboards.

## Connections to other chapters {#connections-to-other-chapters}

- [Evaluation](../07-evaluation/README.en.md) decides how to judge a trajectory as a success or a failure.
- [Human-in-the-Loop](human-in-the-loop.en.md) decides which trajectories need human intervention.
- [Representation and memory](../02-memory/README.en.md) explains how state gets written and contaminated.
- [Search](../04-search/README.en.md) explains how evidence enters the context.

## Quick learning: what does agent observability observe? {#quick-learning-what-does-agent-observability-observe}

<details class="interview" markdown="1">
<summary>Traces, state transitions, and reproducible failures</summary>

**Quick memory**: metrics reveal aggregate changes, logs record events, and traces connect model calls, retrieval, tools, and state changes within a run.

**Interview answer**

> Investigate the whole task, not just one API call. Spans should connect versions, input evidence, actions, tool results, state changes, and cost through a trace ID. Sensitive content should not be logged wholesale. First reconstruct what happened, then use controlled tests to investigate why.

<details markdown="1">
<summary><b>Deep dive</b>: why is “storing every prompt” still not observability?</summary>

Raw text alone makes call ordering and dependencies hard to recover and may leak sensitive data. Structured spans, parent-child relationships, versions, and state diffs help locate failures in retrieval, the model, tools, or evaluation. Apply PII redaction and collection limits rather than treating longer logs as the goal.

</details>
</details>
