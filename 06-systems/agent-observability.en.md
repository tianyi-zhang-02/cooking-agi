# AI Agent Observability

[中文](agent-observability.md) · **English**

## Quick learning: what should agent observability observe?

<details class="interview" markdown="1">
<summary>Traces, state transitions, and reproducible failures</summary>

**Quick memory**: metrics show where behavior changed, logs capture local events, and traces reconstruct the causal chain across model, retrieval, tools, and state.

**Interview answer**

> The observability unit for an agent is a stateful trajectory rather than one API call. Every span should record versions, evidence, decisions, tool arguments and results, state changes, latency, tokens or cost, and final outcome, all connected by a trace ID for replay and root-cause analysis.

<details markdown="1">
<summary><b>Deep dive</b>: why is storing every prompt still not observability?</summary>

Raw text lacks explicit causal structure and can leak sensitive data. Observability needs structured spans, parent-child relationships, versions, and state diffs with PII redaction. The goal is to distinguish retrieval, model, tool, and evaluator failures—not to create longer logs.

</details>
</details>

## Core view

Traditional software observability asks whether a service is healthy. Agent observability must also ask:

> **What did the agent see, what did it believe, why did it choose this action, and where did failure begin to propagate?**

The final answer and an error code are not enough. Failure may originate in retrieval, stale memory, tool behavior, state transitions, loops, evaluation mismatch, or an incorrect model decision over correct evidence.

## What one run should capture

### Identity and version

- Model, prompt, tool, retriever, memory policy, and evaluator versions
- Data, index, feature, and configuration versions
- Session, user, task, trace, and experiment identifiers

### Context and evidence

- Instructions, memory, retrieved evidence, and tool results shown to the model
- Candidates filtered, truncated, or reranked
- Token, latency, and cost-budget allocation

### Decisions and state transitions

- The action selected at each step
- Inputs, outputs, exceptions, and retries
- State before and after execution
- Cancellation, timeout, fallback, and human takeover

### Outcome and evaluation

- Task completion rather than text generation alone
- Deterministic invariants, reference checks, and LLM-judge results
- User correction, subsequent behavior, and long-term outcomes

## Traces are not accumulated logs

Useful traces organize a causal chain. Common spans include `model`, `retrieval`, `tool`, `memory`, `policy`, `evaluation`, and `human_review`. Each span should preserve the relevant version, evidence, transition, latency, cost, and outcome.

## From monitoring to understanding

| Layer | Question answered |
| --- | --- |
| Metrics | What changed across the system? |
| Logs | What event did a component report? |
| Traces | Which decisions and dependencies shaped one request? |
| Replay | Can the failure be reproduced with the same evidence and versions? |
| Evaluation | Did the trajectory satisfy the task contract? |
| Slicing | Which users, tasks, tools, or environments concentrate failure? |

## Important failure modes

- **Loop:** repeated reasoning or tool calls without material state change.
- **Context drift:** later actions no longer serve the original goal or evidence.
- **Memory contamination:** incorrect, stale, or cross-user information enters persistent state.
- **Tool mismatch:** model assumptions differ from actual API semantics.
- **Silent fallback:** degraded execution still produces a plausible-looking result.
- **Evaluator blind spot:** evaluation inspects final text but ignores process and side effects.
- **Cost runaway:** additional tokens, calls, or retries produce no corresponding progress.

## My current view

Observability should produce three actionable artifacts: reproducible failures, behavioral trajectories for offline evaluation, and clear attribution that can change prompts, policies, tools, memory, or post-training. If a trace cannot help decide what to change next, it is mostly expensive log storage.
