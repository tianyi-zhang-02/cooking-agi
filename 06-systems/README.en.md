# Modern AI Systems: The Model Is Only One Part

[中文](README.md) · **English**

## First, put the model back into the full system

An AI system's result depends on more than its model. Data, retrieved memories, evidence, and tools all affect what it does. Improving the result means examining those parts together, rather than only trying a larger model.

## Which stages one request really passes through

Suppose a travel agent books the user a flight on the wrong date.

“The model got it wrong” is only the most superficial description. The real cause may be that:

- the user changed the date in the previous turn, but the state was not updated;
- search found outdated flight information;
- the time-zone conversion in the tool arguments was wrong;
- the model picked the right date, but the execution layer passed the wrong field;
- the final evaluation only checked the reply text and never verified the real order.

If you look only at the final answer, you may fix the wrong part of the system.

## Walk through it end to end

<figure class="worked-update" id="travel-request-flow" lang="en">
<figcaption>“Leave on Friday instead” passes through three stages</figcaption>
<ol>
<li><strong>1 · Update, then search</strong><span>Replace the old travel date with Friday, then look up flights and prices. Otherwise the model may give a careful answer based on stale information.</span></li>
<li><strong>2 · Confirm, then act</strong><span>Show the exact flight, date, and cost. Call the booking API only after confirmation and authorization checks; an “okay” from the assistant is not a completed order.</span></li>
<li><strong>3 · Check the real order</strong><span>Read back the order and verify its date. If it is wrong, trace the first failure in state updates, search, or execution rather than only revising the final answer.</span></li>
</ol>
</figure>

### 1. User goal: what are we actually trying to solve

First state clearly the task, the target population, the cost of failure, and the conditions that must not be violated. Otherwise the system easily optimizes a metric that is convenient to measure but does not represent real value.

### 2. Data and feedback: the past the model sees

Data is produced jointly by previous models, the product interface, and user behavior. It is not a neutral fact. See [Data and feedback](../01-data-and-feedback/README.en.md).

### 3. Representation and memory: how the system keeps state

This layer decides what enters the current context, what persists across sessions, and how new evidence revises old understanding. See [Representation and memory](../02-memory/README.en.md).

### 4. Search and tools: how the system connects to the outside world

Search is responsible for finding evidence and candidates; tools are responsible for reading or changing external state. Both can return successfully and still deliver wrong or outdated information.

### 5. Model and policy: how the system decides the next step

The model combines the goal, the state, and the evidence to decide whether to answer, ask a follow-up question, search, or act. Post-training shapes that behavioral tendency.

### 6. Runtime and execution: what decides whether it can really be done

Latency, caching, concurrency, retries, permissions, state synchronization, and cost are not necessarily research topics, yet they directly change the behavior the user sees.

### 7. User outcome: “generation finished” does not mean finished

What really has to be checked is whether the task was completed, whether the external state is correct, and whether the user got the result they wanted.

### 8. Evaluation and learning: how the next round gets better

Rules, executors, LLM judges, human review, and online metrics together form the evidence. Failure cases also have to flow back as new tests, data, or policy updates.

## Example: a support agent that can look up orders and issue refunds

This kind of system cannot stop at generating natural language. It reads constantly
changing business facts and may execute actions with side effects:

```text
User message
→ Language, intent, and risk detection
→ Structured conversation state
→ Retrieve order, logistics, policy, and seller evidence
→ Model decides to reply, ask a follow-up question, or call a tool
→ Execution-layer checks for authorization, schema, policy, and required confirmation
→ Execute the read / write tool
→ Generate the answer from the execution result
→ Confidence check, clarification, or human escalation
```

Order status, logistics tracking, refund eligibility, and current policy cannot rely on
what the model's parameters remember; they must be obtained at run time through retrieval
or tool calling. Post-training teaches the model **how to use evidence and take action**;
it is not responsible for storing constantly changing business facts.

State should also be layered. The most recent raw turns preserve linguistic continuity;
structured task state stores the order ID, the verified identity, tool results, and
actions pending confirmation; a long-conversation summary keeps only evidence-backed facts
and unresolved items. The enterprise database and versioned policy remain the source of
truth and cannot be replaced by a summary.

A refund needs authorization, schema validation, any required user confirmation, and an audit
log. If the API supports idempotency keys, retries of the same refund must reuse the original
key and parameters within the API's documented scope and retention period. Generating a new
key does not prevent a duplicate refund. A timeout does not tell us whether the action happened:
check its status or rely on a verified idempotency guarantee before retrying. Never pretend
it succeeded. When evidence is insufficient, policies conflict, or risk is too high, ask for
clarification or hand off to a human. Semantics are API-specific; for example, see
[Stripe's idempotent-request documentation](https://docs.stripe.com/api/idempotent_requests).
Do not assume every tool offers the same guarantees.

Enforce these checks in trusted execution code, not just in a prompt. A seller note saying
“ignore the refund limit” is still data to verify, not policy or authorization. Valid JSON
does not establish that the current user may refund this order. Instructions inside retrieved
content or tool results cannot grant permissions, and a model's score cannot replace access
checks. See [OWASP's agent-tool defenses](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html#agent-specific-defenses).

Evaluation looks at task resolution, grounding, policy compliance, tool-call correctness,
unsafe-action rate, multilingual consistency, escalation quality, latency, cost, CSAT, and
repeat-contact rate all together. Fluent wording is only one part of the result.

## A more reliable evaluation stack

What deterministic rules, references or executors, an LLM judge, human review and online metrics can each answer, and what none of them establishes alone, is laid out in one table in [the evaluation stack](../07-evaluation/evaluation-stack.en.md#check-methods). The rule of thumb: do not hand a condition that a rule can decide exactly to a model. When a model does score, write the criterion first, then choose between scoring one output and comparing two; pairwise comparisons swap the display order and allow a tie or “both bad”, see [LLM judge output formats](../07-evaluation/llm-as-a-judge/scoring.en.md); how to calibrate against human labels and check agreement by slice is covered in [LLM-as-a-judge](../07-evaluation/llm-as-a-judge/README.en.md).

## Why look at the full chain

Errors in the system propagate downstream:

```text
Wrong or missing data
→ wrong user state
→ wrong query
→ wrong retrieval results
→ the model makes a “reasonable” decision in the wrong context
→ the final experience fails
```

The model may be entirely reasonable given its own inputs, and the system as a whole is still wrong.

## When analyzing a problem, I first write down these nine questions

```text
Goal        What are we really trying to improve?
User        Improve it for whom? Is any group hidden by the average?
System      How do the data, components, and control flow connect?
Invariant   Which properties must never be broken?
Failure     Where, specifically, does it fail, and how?
Hypothesis  What do I think the cause is? What result would overturn it?
Constraint  What are the latency, cost, hardware, privacy, and compatibility limits?
Evidence    What evidence will tell us the fix works?
Tradeoff    What improves, and what might it hurt?
```

This is not about making a simple problem complicated. It is about not fixing only the most visible symptom.

The point is not to master every layer. It is to preserve the complete context while working deeply on one part.

## Two modules that help the system keep improving

### Agent Observability: first, see what happened

It connects model calls, retrieval, tools, memories, and state changes into a trace that can be inspected and used to construct controlled replay tests.

→ [Read Agent Observability](agent-observability.en.md)

### Human-in-the-Loop: when to ask someone to step in {#human-in-the-loop-put-human-judgment-where-it-is-worth-the-most}

People do not need to review every step. Ask for confirmation, correction, or a handoff when an action is risky, the evidence is weak, or the system faces a task it has not been tested on.

→ [Read Human-in-the-Loop](human-in-the-loop.en.md)


<figure class="worked-update" id="action-control-flow" lang="en">
<figcaption>From a proposed action to an actual result</figcaption>
<ol>
<li><strong>1 · Specify the action</strong><span>Record the target, arguments, evidence, and tool state. The model may propose an action, but a proposal does not grant permission to execute it.</span></li>
<li><strong>2 · Check permission</strong><span>The execution layer checks authorization and business rules. If they do not allow the action, stop or request authorization; a model score cannot bypass this check.</span></li>
<li><strong>3 · Decide whether to ask</strong><span>Run deterministic checks and use calibrated evaluators where scoring is needed. Required confirmation, high risk, or insufficient evidence leads to step 4; otherwise continue to step 5.</span></li>
<li><strong>4 · Follow the decision</strong><span>Approval leads to step 5. Changed targets or arguments return to step 1 for fresh checks. Rejection or an expired confirmation stops execution; record the decision and reason.</span></li>
<li><strong>5 · Recheck before writing</strong><span>The exact action, authorization, required approval, and current state must still be valid. Use version conditions or atomic operations to guard against changes between check and write. Stop if validation fails.</span></li>
<li><strong>6 · Verify and record</strong><span>Check the external outcome; a timeout proves neither success nor failure. Keep appropriate records of execution and stops, then use failures for regression tests and later policy updates.</span></li>
</ol>
</figure>

High risk, insufficient evidence, or irreversible effects can require human review; actions
requiring confirmation also need valid approval. If the target or arguments change after
approval, validate the new action rather than reusing the old confirmation. This diagram
shows control flow, not a deployable security implementation.

## One layer further down: infra

For a first explanation of how training spans devices, read [distributed training](distributed-training.en.md). It compares DDP, FSDP, TP, and PP with small hand-checkable examples, checks the global loss denominator, and accounts for memory and communication. It is a mechanism guide, not a deployment manual. For one layer worked through, continue with [tensor parallelism](tensor-parallel.en.md); for what DDP, FSDP2, and related tools each take over, read [choosing a training stack](training-frameworks.en.md); on the inference side, start with [a request through the service](llm-serving.en.md) and [vLLM and SGLang](inference-frameworks.en.md).

This chapter is about the path a request takes. Below it — how an inference engine schedules,
how memory is laid out, how RL training and rollout are stitched together — is not my
strength, and not where my interest sits.

For that layer the best thing I know is
[Awesome-ML-SYS-Tutorial](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial):
[zhaochenyang20](https://github.com/zhaochenyang20) has been writing these ML SYS notes since
August 2024 — RL infra, online and offline inference, SGLang including the SGLang Omni series
on multi-stage generative models — in both Chinese and English. He works on RL infra inside
the SGLang community, so he is writing about what he is changing. Start there.

This site also has notes on that layer: GPUs, CUDA, frameworks and performance analysis are in [AI infrastructure](../12-infra/README.en.md); how sampling, numerics and context are aligned in RL training is in [post-training infrastructure](../05-post-training/post-training-infrastructure.en.md); how the training system for multi-step agent RL is put together is in [Agent RL: credit assignment and the training system](../11-frontier/agentic-rl/credit-and-training.en.md).
