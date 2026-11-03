# AI Agent Observability: What Did the Agent Actually Just Do?

[中文](agent-observability.md) · **English**

> Last reviewed: 2026-10

> Timing and asynchronous-correlation additions checked: 2026-11-02

## Observability has to explain the decision process {#observability-has-to-explain-the-decision-process}

Agent observability helps reconstruct **what the agent saw, which decisions it made, how state changed, and at which step the failure began**. Knowing only whether a service reported an error is not enough.

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

Retained traces do not necessarily represent all traffic. [Selecting traces by error or latency](https://opentelemetry.io/docs/concepts/sampling/#tail-sampling) helps debugging but changes the sample. If 100 requests include 10 failures, keeping every failure and only 9 successes gives a recorded failure rate of `10 / 19`, not the actual `10 / 100`. Use reliable population counts or explain sampling corrections for overall metrics. Operations requiring a complete audit trail cannot rely solely on sampled traces.

## What one run should record {#what-one-run-should-record}

### Versions {#versions}

Record the model, prompt, retriever, index, tools, memory policy, evaluator, data, and configuration versions. Connect the run through task, trace, and experiment IDs. Include user identifiers only when necessary, rather than logging email addresses or similar fields by default. What these fields are called in OpenTelemetry's GenAI conventions, and which stability level those conventions are at as of October 2026, is covered in [Taking evaluation online](../07-evaluation/eval-observability.en.md).

### Input evidence {#input-evidence}

Record which instructions, memories, retrieved results, and tool outputs the model saw, and which content was truncated, filtered, or reordered.

Also record token, latency, and cost budgets, so “nothing found” can be distinguished from “stopped because the budget ran out.”

Recording evidence does not mean copying all raw content. Prefer versioned IDs and controlled references where these suffice. When raw content is necessary, restrict access, retention, and purpose, and exclude credentials and unrelated personal data. Hashing a predictable user ID is not anonymization, because anyone who can enumerate the IDs can recompute the hashes; see [OpenTelemetry's sensitive-data guidance](https://opentelemetry.io/docs/security/handling-sensitive-data/).

Avoid collecting sensitive content at the source, then check allowed fields and redact before export, rather than only hiding values in a dashboard. Inspect automatic instrumentation, exception text, URL query parameters, and tool results too. Filtering a few top-level fields does not make free text secret-free.

For example, this synthetic record keeps the correlation needed to investigate a retry without copying the refund request body or credentials:

```text
trace_id=demo-17  tool=refund  attempt=1
operation_ref=demo-refund-8  transport_status=timeout
business_outcome=unknown
```

A timeout does not mean the refund failed. Check the actual result through the controlled `operation_ref`; access to that reference must also be restricted. This is a minimal illustration, not a universally safe logging format.

### Decision process {#decision-process}

Record the action taken, the rule that triggered continuation or stopping, and any retry, fallback, timeout, or human takeover. Keep system rules, observable outputs, and retrospective hypotheses distinct. This does not require access to a model's private reasoning.

### State changes {#state-changes}

Record changes to user state, task state, and external system state across execution.

### Final outcome {#final-outcome}

Check whether the task was completed, not just whether the generated text looked plausible.

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

## Reading latency from a trace {#trace-timing}

A support agent retrieves the refund policy and queries the order before drafting a reply. This synthetic trace uses milliseconds from request arrival. The two lookups start together, and generation requires both results.

| Span | Start → end | Duration | What it waits for |
| --- | --- | ---: | --- |
| Request entry | 0 → 50 | 50 | Validation and dispatch |
| Policy retrieval | 50 → 250 | 200 | Retrieval results |
| Order lookup | 50 → 500 | 450 | The order service |
| Reply generation | 500 → 1,200 | 700 | Both results, then generation |

Completion takes `50 + max(200, 450) + 700 = 1,200 ms`, not the sum of all four rows, `1,400 ms`. Policy retrieval finishes first, but the request still needs the order. The dependency chain determining completion is the critical path. Adding the root span, which covers the whole request, would double-count time again.

Reducing policy retrieval from 200 ms to 50 ms leaves completion at 1,200 ms. Reducing generation alone from 700 ms to 350 ms gives 850 ms. These calculations hold everything else fixed. Real changes affect concurrency, queueing, and contention, so measure again rather than presenting this arithmetic as a performance guarantee.

Separate queueing from execution too. A long model span may mean waiting for admission rather than slow GPU computation; client retries and transport may appear elsewhere. Across machines, account for clock skew and use dependency relationships alongside locally measured durations.

## One task can span several traces {#task-and-attempt}

A user may close the page and return, or a reviewer may approve two hours later. The trace ID of one HTTP request is not enough to identify the continuing task. Keep these relationships distinct; the names below are illustrative, not a required OTel schema.

| Field | What it connects |
| --- | --- |
| `task_id` | The user's continuing task, including pauses and resumptions |
| `trace_id` / `span_id` | Calls and dependencies within an execution |
| `operation_id` | One external action, such as a refund; useful for lookup, but not itself an idempotency guarantee |
| `attempt` | An attempt at one tool operation; count writes and status queries separately while linking both to the original external action |
| `action_version` | The approved argument version, so resuming cannot silently execute a different action |

Use [OpenTelemetry span links](https://opentelemetry.io/docs/concepts/signals/traces/#span-links) to associate asynchronous work with an earlier call when appropriate. Two hours waiting for approval should not look like two hours of model execution. Correlation IDs are not credentials; accessing their records still requires authorization.

For example, refund submission times out on `attempt=1`, then a status query confirms that the refund completed. Two tool calls recovered one task; no second refund was issued. The query is not a second attempt to submit the refund. Safe retry behavior still depends on the external API; see [failure and recovery](failure-and-recovery.en.md#unknown-outcome).

## Common agent failures {#common-agent-failures}

- **Loop:** the same tool is called repeatedly while state does not materially change.
- **Goal drift:** later steps have already departed from the user's original task.
- **Memory contamination:** wrong, stale, or other-context information enters long-term state.
- **Tool-semantics mismatch:** the API behavior the model assumes differs from the actual implementation.
- **Silent fallback:** after falling back, the system still answers in its normal tone, and the user does not know quality has dropped.
- **Evaluation blind spot:** only the final text is checked, not the process, the tools, or the side effects.
- **Cost runaway:** more tokens, searches, and retries bring no new progress.

## What to do after recording {#what-to-do-after-recording}

Use the records to produce more than a dashboard:

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

**In brief**: metrics reveal aggregate changes, logs record events, and traces connect model calls, retrieval, tools, and state changes within a run.

**Interview answer**

> Investigate the whole task, not just one API call. Spans should connect versions, input evidence, actions, tool results, state changes, and cost through a trace ID. Sensitive content should not be logged wholesale. First reconstruct what happened, then use controlled tests to investigate why.

<details markdown="1">
<summary><b>Deep dive</b>: why is “storing every prompt” still not observability?</summary>

Raw text alone makes call ordering and dependencies hard to recover and may leak sensitive data. Structured spans, parent-child relationships, versions, and state diffs help locate failures in retrieval, the model, tools, or evaluation. Apply PII redaction and collection limits rather than treating longer logs as the goal.

</details>
</details>
