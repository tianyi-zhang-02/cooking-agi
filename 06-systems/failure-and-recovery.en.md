# Failure and recovery: did the action happen before the timeout?

[中文](failure-and-recovery.md) · **English**

> Last reviewed: 2026-11-02 · The examples illustrate the design, not a production-ready implementation

A user asks a support assistant for a refund. The tool call waits, then times out. It is tempting to treat this as a failure and retry. But the refund may already have succeeded, with only the response lost. Another attempt can turn an uncertain outcome into a duplicate action.

Following the [systems overview](README.en.md), this note covers control logic outside the model: unknown outcomes, bounded retries, stopping, and resuming a task when the user returns. No particular agent framework is required.

## Tasks, requests, and external actions have different lifetimes {#three-lifetimes}

| Object | Refund example | Lifetime |
| --- | --- | --- |
| User task | Refund order R17 | May span page visits until completed or explicitly stopped |
| Request | The browser waiting for a server response | May end on disconnect, timeout, or page closure |
| External action | The payment service issuing the refund | Does not necessarily stop when the browser disconnects |

Ending an HTTP connection does not end the business task; stopping generation does not reverse a refund. A durable task record should identify the original action, its known outcome, and what to do next. A chat message saying “processing” is not enough. See [task and attempt correlation](agent-observability.en.md#task-and-attempt) for linking this record to traces.

## Store unknown as an outcome of its own {#unknown-outcome}

This simplified flow places authorization, action-version checks, and business rules in trusted execution code. The model proposes a request.

<figure class="worked-update worked-update--pairs" id="recovery-flow" lang="en">
<figcaption>A refund timeout does not trigger a fresh refund</figcaption>
<ol>
<li><strong>1 · Validate before submitting</strong><span>Check the target, amount, permissions, and valid approval. Save the action identifier and argument version.</span></li>
<li><strong>2 · Submit and wait</strong><span>Submit to the external service and verify success before marking the action complete. If the service confirms it rejected the action without executing it, use the reported reason to decide what to do next.</span></li>
<li><strong>3 · Lost response: outcome unknown</strong><span>Keep the original action rather than resubmitting with a new identifier. Tell the user that its status is being checked, not that the refund failed.</span></li>
<li><strong>4 · Reconcile before continuing</strong><span>Look up the original action, or retry with the original key and identical arguments when the API explicitly supports it. Unresolved outcomes stay pending, with an alert or handoff to an authorized person.</span></li>
</ol>
</figure>

Idempotency means repeated requests do not repeat the same effect within a defined scope, not that a request is transmitted only once. A unique `operation_id` provides this only if the server actually uses it for deduplication. [AWS's idempotent-API article](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/) examines duplicate requests and changes of intent under one identifier.

Check scope, retention, argument matching, and failure-response semantics. For example, [Stripe API v1](https://docs.stripe.com/api/idempotent_requests) reuses saved responses, including some 500 responses; reusing a key after pruning can create a new request. Other APIs may differ. Repeating a key indefinitely does not guarantee successful recovery.

## Different failures need different responses {#failure-policy}

| Observed condition | Possible response | Do not assume |
| --- | --- | --- |
| Temporary read-service failure | Retry within total time and attempt limits | Unlimited retries are harmless |
| Provider throttling | Back off according to the API and control concurrency | More parallel requests will help |
| Missing arguments or insufficient permission | Fix input or obtain authorization | Identical requests will eventually work |
| Write timeout with an unknown outcome | Query the original action or use verified idempotency semantics | A fresh key makes resubmission safe |
| Changed amount or recipient | Create a new action version and validate and confirm again | Old approval applies |
| Generation interrupted after the order completed | Preserve the order result; redisplay or regenerate its explanation | Rewriting the reply requires repeating the order |

There is an execution boundary between knowing that 20 units should be refunded and actually refunding them. Valid structured output, high model confidence, and an earlier button click cannot individually prove correct completion.

## Retries share one budget {#retry-budget}

Suppose the application, tool wrapper, and HTTP client each allow 3 attempts, including the initial call. If the bottom layer keeps failing and every layer exhausts its allowance, one user action can cause `3 × 3 × 3 = 27` downstream calls, not 3. This is an assumed upper bound, not an SDK default.

Check whether the SDK already retries before choosing the responsible layer. Propagate an overall deadline and limit total attempts, tokens, and cost. An unused attempt does not mean there is enough time left to make it.

For a read-only lookup with a 5-second budget:

| Stage | Duration |
| --- | ---: |
| Entry and setup | 0.2 s |
| First call times out | 1.5 s |
| Backoff on this attempt | 0.3 s |
| Second call | 1.5 s |
| Prepare and return the result | 1.0 s |

That totals 4.5 seconds, leaving 0.5 seconds: not enough for another full attempt with the same settings. This is budget arithmetic; real networking and queues consume time too. Partial results, a smaller task, or ending the wait require explicit product behavior, not a substitute answer presented as completion.

[AWS's timeout and retry guidance](https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/) explains how backoff and jitter reduce synchronized retries. They spread requests out but do not create capacity; sustained overload also needs admission limits and rejected work.

## What does the stop button stop? {#cancellation}

Do not label every stage “cancelled.” The interface can be simple without hiding the backend state.

| Stage when stop is requested | What the user can be told | Remaining backend work |
| --- | --- | --- |
| Draft generation only; no write submitted | Generation stopped; the draft can be kept | Propagate cancellation and release resources where possible |
| External action submitted; no result received | Further steps stopped; the submitted action is being checked | Reconcile the result; do not promise nothing happened |
| Send or refund verified as complete | Completed; a separate reversal may be available | Run a separately authorized compensation flow with its own conditions |

Cancellation can race with submission. The server must atomically decide whether an action may still be submitted; stopping the browser animation does not resolve that race. For an already submitted action, even an external cancellation API may be best-effort, requiring a final status check.

Compensation does not turn back time. An email may have been read, cancellation may incur a fee, and another operation may already depend on the result. [Azure's compensating-transaction pattern](https://learn.microsoft.com/en-us/azure/architecture/patterns/compensating-transaction) discusses these constraints: compensation is business-specific and can itself fail, so preserve progress and support manual intervention.

## Can the fallback still do the promised job? {#fallback}

If generation is unavailable, returning retrieved source links may help, but say that no summary was produced. A stale order cache may show a dated record, not current refund eligibility. A fallback model needs checks for tool support, output formats, data destinations, and evaluation results; a similar API does not establish equivalence.

Some capabilities can be reduced; some constraints cannot. Shorten an explanation, but do not bypass authorization. Pause writes, but do not fill in an invented outcome. Record the fallback path and measure its success and cost separately.

## Inject failures, not just successful responses {#failure-tests}

In a sandbox, fix tool responses and place failures before and after the action:

| Injected condition | Required check |
| --- | --- |
| Failure before submission | No external write; task can restart under its policy |
| Service executes, but the response is lost | Reconciliation recognizes completion without repeating the action |
| Double approval or two workers resuming together | At most one valid execution path; the UI receives the same business result |
| Arguments change after approval | Old approval cannot execute the changed action |
| Page refresh while waiting | Reload task status instead of resubmitting to restore the screen |
| Cancellation arrives after submission | No further actions start; reconcile the submitted part |

These are test requirements, not guarantees established by a flowchart. Also test crashes after the external write succeeds but before local state is saved. Without external idempotency or a way to verify the result, an end-to-end exactly-once claim is difficult to justify.

Continue with [human handoff](human-in-the-loop.en.md#handoff-packet) for unresolved tasks, [model experience](../08-model-experience/README.en.md#interaction-recovery) for progress and correction, and the [serving and feedback walkthrough](../practice/ml-system/serving-and-feedback.en.md#release) for releases and online feedback.
