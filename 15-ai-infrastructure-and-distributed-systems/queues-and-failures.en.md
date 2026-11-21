# Queues, capacity, and failures: when requests pile up

[中文](queues-and-failures.md) · **English**

“Add replicas” sometimes fixes a slow service and sometimes arrives too late. First locate the wait: how much deadline remains, when new instances become ready, and whether a failure turns one request into many more attempts.

The small models below are deliberately simplified. They explain the reasoning but do not directly size a real GPU cluster with variable request lengths and batching.

## Define a successful request first {#slo}

Latency among successful requests hides rejections and timeouts. Suppose our service level indicator (SLI) is the fraction of **valid, in-scope requests that receive a complete, valid response within the agreed deadline** over a measurement window. Requests rejected because of service overload count as failures; malformed inputs are tracked separately.

Set the service level objective (SLO) to 99.9%. With 2,000,000 eligible requests in the window, the allowed failure count is:

\[
B=2{,}000{,}000\times(1-0.999)=2{,}000.
\]

This is the example's request-based error budget. If 3,000 requests fail, success is 99.85%, exceeding the budget by 1,000 failures. It does not directly translate into minutes of downtime because traffic varies over time. [Google SRE's SLO guidance](https://sre.google/workbook/implementing-slos/) explains defining indicators and objectives around user-visible outcomes.

A valid response is not necessarily a correct answer. Availability, timely completion, and model quality can be tracked separately. An empty answer should not masquerade as success, while factual correctness on a complex task may need a separate evaluation; HTTP 200 does not settle it.

## Why does a short burst leave a long queue? {#queue}

Start with a fluid approximation: every request requires equal work, total service capacity is fixed at 100 requests per second, and the initial queue is empty. Arrivals run at 120 requests per second for ten seconds.

Twenty extra requests arrive each second, leaving:

\[
Q=(120-100)\times10=200.
\]

Arrivals then fall to 80 per second. Only 20 requests per second of capacity remain to drain the backlog, so clearing it takes:

\[
t_{\rm drain}=\frac{200}{100-80}=10\ \mathrm{s}.
\]

Under the constant-rate FIFO approximation, a request behind those 200 waits about \(200/100=2\) seconds. With only one second left on its deadline, queueing it may merely postpone failure. The model excludes batching, length differences, and randomness and cannot predict p99. It does explain why lower arrival rates do not immediately eliminate waiting.

[Little's law](https://www.mit.edu/course/other/urban_or_book/www/book/chapter4/4.4.html) gives another useful relationship. For a stable system, average in-flight requests \(L\), effective arrival rate \(\lambda\), and average time inside the system \(W\) satisfy:

\[
L=\lambda W.
\]

At 80 requests per second and 0.25 seconds average residence time, average in-flight count is 20. This includes requests waiting and being processed; **it is not GPU batch size**. All three quantities need matching system boundaries and accounting. Do not apply steady-state averages to a queue growing indefinitely under overload.

## How much capacity should scaling leave unused? {#autoscale}

Suppose one replica meets its latency target at 50 requests per second on a fixed representative workload. Expected demand is 120 per second, and we plan to use only 80% of that capacity:

\[
n=\left\lceil\frac{120}{0.8\times50}\right\rceil=3.
\]

The 80% target is an illustrative choice, not a universal optimum. Three replicas provide 150 requests per second in total, but losing one leaves 100—below demand. To preserve the same headroom after losing one replica, provision four equivalent replicas, placed so the assumed failure really is isolated.

This excludes startup delay. If pulling weights and warming up takes two minutes, a ten-second burst ends before the extra capacity is ready. Combine forecasting, reserve capacity, bounded queues, and admission control rather than treating autoscaling as a switch that solves overload.

| Signal | What it helps reveal | What it can miss |
| --- | --- | --- |
| GPU utilization | Whether computation is busy | High utilization says little about acceptable waiting; low utilization may mean communication stalls |
| Queue length and oldest-request age | Backlog and endangered deadlines | Different lengths require different amounts of work |
| KV occupancy and preemption | Memory limiting admission or uninterrupted execution | Architecture, precision, and prefix reuse change costs |
| Completion rate, TTFT, and ITL | Delivered work and user experience | Completed-only statistics exclude timeouts and rejections |

[Kubernetes HPA](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/) adjusts replicas using resource or custom metrics in a periodic control process; it does not create ready capacity instantly. The scaling unit must match deployment: a model-parallel worker group needs joint readiness, not one process mistaken for a complete replica.

## How can retries amplify a small failure? {#retry}

Suppose the client, gateway, and downstream call each allow at most three attempts, including the initial attempt. One user operation can cause up to:

\[
3\times3\times3=27
\]

attempts at the lowest layer. This is a worst-case bound when every upper-layer attempt exhausts the lower-layer budget, not an average retry rate. Retries do not create capacity; when overload causes failures, additional work can delay recovery.

Allocate a total retry budget along the call chain rather than letting every layer retry independently. Backoff and jitter spread attempts out but still need limits on count and elapsed time. Stop explicitly when insufficient deadline remains. [Google SRE's overload chapter](https://sre.google/sre-book/handling-overload/) discusses overload handling and degradation in more detail.

Streaming adds another complication. Once the user has seen part of an answer, starting generation again can produce a different continuation. Do not silently splice them. Use an explicit recovery protocol or report an interrupted request. Tool writes and other side effects also need appropriate idempotency handling; a repeated request ID alone does not guarantee downstream deduplication.

## What should survive a failure? {#failure}

| Failure or pressure | Possible response | Condition that must remain true |
| --- | --- | --- |
| A request can no longer meet its deadline | Reject or cancel early; release downstream resources | Count the failure; distinguish retryable and permanent errors |
| Feature-service timeout | Evaluated missing-value behavior or a fallback model | Do not bypass authorization, eligibility, or safety rules |
| Parallel worker failure | Stop the affected group, rebuild, or redirect new requests | Do not promise seamless continuation of existing KV without a recovery protocol |
| New release not warmed up | Keep the previous version serving | Process liveness is not serving readiness |
| A hot prefix overloads one replica | Reduce affinity and send some traffic elsewhere | Compare lower cache hits with reduced waiting, not one metric alone |
| Slow checkpoint writes | Investigate storage, frequency, and asynchronous saving | Async writes must represent a consistent snapshot; incomplete files are not recovery points |

Graceful degradation needs an explicit quality expectation. What is lost by using a smaller model, fewer candidates, or older but still permitted data? After recovery, how do you know users no longer receive fallback output? Track fallback rate and quality separately rather than hiding them inside overall success.

## Check the design with a small program {#exercise}

Run from the repository root; the site test requires Python 3.11 or newer:

```bash
python3 15-ai-infrastructure-and-distributed-systems/code/capacity_checks.py
python3 -m unittest discover -s site/tests -p 'test_public_learning_routes.py'
```

It reproduces the 200-request backlog, ten-second drain, three capacity replicas plus one failure reserve, and the previous chapter's KV and token-weighted loss examples. It does not contact external services or incur cloud charges.

Change three conditions yourself:

1. Keep recovery-period arrivals at 100 per second. The queue neither grows nor drains; the calculation must not invent a finite recovery time.
2. Increase input length from 2,048 to 8,192 tokens. Recompute KV, then remeasure per-replica capacity rather than preserving the old 50-requests-per-second assumption.
3. Cancel a queued request. A real implementation should reduce waiting work and subsequent computation; this fluid model has no cancellation events, so test that separately with request IDs in a discrete-event model.

Finally, annotate the design: where are the queues, who checks remaining time, who releases state, and who can scale capacity? The normal path is only the beginning. Failure handling makes the system description complete.
