# From request to GPU: taking a model service apart

[中文](serving-system.md) · **English**

Suppose a generative assistant already has weights and can answer one request. The next questions are operational: who accepts a request, decides how long it waits, allocates its cache, stops wasted work after a failure, and safely replaces the model?

This is a teaching system, not a company's internal architecture. Every calculation states its assumptions; none is a measured result for a particular GPU or framework.

## Which layers handle a request? {#stack}

Start with one request, then separate the decisions happening alongside it. These are responsibilities, not a requirement to deploy every step separately.

<div class="system-flow" role="group" aria-label="Request and deployment paths through a model service">
<section class="system-flow-lane">
<h3>Request path</h3>
<ol><li>The client supplies input, a deadline, and cancellation</li><li>The entry checks authentication, limits, and admission</li><li>The router chooses a ready replica</li><li>The engine queues work, allocates KV, and schedules computation</li><li>Workers execute the model and communication</li><li>Tokens stream back to the client</li></ol>
<p>Deadlines and cancellation must reach the engine, not stop at the entry.</p>
</section>
<section class="system-flow-lane">
<h3>Deployment and capacity</h3>
<ol><li>Obtain validated weights and configuration</li><li>Load, warm up, and check readiness</li><li>Register ready replicas with routing</li><li>Use queue and cache measurements to decide scaling</li><li>Stop new assignments to old replicas, then drain and retire them</li></ol>
<p>The engine supplies measurements; control adjusts capacity. New resources are not instantly ready.</p>
</section>
</div>

| Layer | Decision | Information to pass onward | Measurements |
| --- | --- | --- | --- |
| Entry | Is the request valid, and can it still be admitted? | Request ID, deadline, authorization scope, input limits | Rejections, input lengths, entry overhead |
| Replica routing | Which ready replica accepts it? | Model version, cache-related information, remaining time | Replica load, routing delay, version distribution |
| Engine scheduling | Which tokens run this step, and where does KV live? | Computed tokens, sampling state, cancellation | Queue time, cache occupancy, preemption, batch composition |
| Workers | Which operators and collectives execute? | Tensors, execution plans, parallel groups | Kernel time, bandwidth, communication waits, failures |
| Deployment control | Load, warm up, shift traffic, drain, roll back | Ready versions, health, resource requests | Time to readiness, failed releases, capacity shortfalls |

For example, the [vLLM V1 architecture](https://docs.vllm.ai/en/stable/design/arch_overview/) separates HTTP/input handling, engine-core scheduling and KV management, and GPU-worker model execution. These responsibilities provide useful source-reading entry points; process layouts and options should match the deployed version.

Choosing vLLM versus SGLang is not simply “throughput versus caching.” Check models, hardware, parallelism, quantization, and integrations, then compare on the same workload. The [framework chapter](../06-systems/inference-frameworks.en.md) covers mechanisms and ecosystems. Replacing the engine does not remove admission control, recovery, or end-to-end evaluation.

## A slow response may not be slow model computation {#timeline}

Prefill processes the input; decode continues generation. Users also wait for preparation, queues, and transport. Consider this simplified request:

| Stage | Assumed duration | What the user is waiting for |
| --- | --- | --- |
| Input validation and preparation | 6 ms | Entering the service |
| Queueing | 80 ms | Available resources |
| Prefill through first-token readiness | 60 ms | Input computation and initial sampling |
| Delivery of the first token | 10 ms | Serialization and transport |
| Each subsequent token arrival interval | 20 ms | Combined computation, scheduling, and delivery |

The 60 ms includes initial sampling, so time to first token (TTFT) is:

\[
T_{\rm first}=6+80+60+10=156\ \mathrm{ms}.
\]

For exactly 100 output tokens with a fixed subsequent arrival interval of 20 ms, completion time is:

\[
T_{\rm complete}=156+(100-1)\times20=2136\ \mathrm{ms}.
\]

Reducing computation from 60 to 40 ms brings TTFT from 156 to 136 ms; queueing still dominates much of the wait. Conversely, batching that increases throughput by waiting longer may feel slower to the user.

This is a single-request decomposition. **Adding per-layer p99s does not give end-to-end p99.** Trace complete requests and compare by input length, output length, cache behavior, and traffic period. Specify whether inter-token latency (ITL) is measured at the engine or client: buffering and transport affect the two differently.

## How does KV cache limit admission? {#memory}

Assume an ordinary GQA Transformer with 32 layers, 8 KV heads, head dimension 128, and 2 bytes per K or V element. Initially ignore page waste, shared prefixes, sliding windows, and other cache compression. Logical KV per cached token per sequence is:

\[
b_{\rm token}=2\times32\times8\times128\times2
=131072\ \mathrm{bytes}=128\ \mathrm{KiB}.
\]

The first 2 accounts for K and V; the last is bytes per element. This is **logical KV across one complete model**, not necessarily per-device allocation under arbitrary parallelism. Sharding and replication change placement; architectures such as MLA need a different formula.

| Request-length budget | Logical KV per sequence | Total for 32 sequences |
| --- | --- | --- |
| 2,048 input + 512 maximum output = 2,560 tokens | 320 MiB | 10 GiB |
| 8,192 input + 512 maximum output = 8,704 tokens | 1,088 MiB | 34 GiB |

The same “32 concurrent requests” differ by 24 GiB just by changing input length. Input plus maximum output is a conservative budget; an engine may allocate incrementally rather than reserve all output space immediately. It still needs a policy for cache exhaustion if all admitted requests approach their limits.

Subtract weights, peak activations, runtime allocations, and headroom before defining the KV budget. Even 10 GiB remaining does not guarantee the first row fits: allocation granularity and temporary storage are excluded. **A request-count limit is not a token-budget limit.**

Prefix caching reuses computation for genuinely identical, compatible prefixes, not merely similar text. Hit rates depend on routing, eviction, and traffic. Evaluate both cold caches and representative reuse instead of reporting only an ideal warm-cache workload.

## Replication, model parallelism, and disaggregation solve different problems {#placement}

| Choice | Main purpose | Added cost or constraint |
| --- | --- | --- |
| Independent replicas | More requests than one replica can handle | Repeated weights, fragmented caches, load balancing |
| Tensor parallelism (TP) | Split model computation and storage | Per-layer communication; interconnect sensitivity; workers execute together |
| Pipeline parallelism (PP) | Place different model layers separately | Transfers between stages, pipeline bubbles, scheduling complexity |
| Prefill/decode disaggregation | Reduce interference and provision phases separately | KV transfer, capacity balance, routing, and recovery complexity |

A TP group is generally one logical replica, not several complete replicas free to accept unrelated work independently. Losing a worker may stop or require rebuilding the group. Routing to another replica does not automatically move its KV state there.

Disaggregation needs a transfer budget too. [SGLang's documentation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation) explains how prefill can interfere with decode. For a framework-independent example, transferring 1 GiB of KV at an effective 8 GiB/s takes 125 ms for the payload alone, before queueing and protocol overhead.

If it removes only 40 ms of interference and all 125 ms sits on the critical path, single-request latency does not improve. Whether overlapping transfer, greater capacity, or better tail latency compensates needs a workload-specific measurement. A more elaborate architecture is not automatically a better one.

Cache-affinity routing has a similar tradeoff. Sending a prefix back to its previous replica can save computation while creating a queue at a hot replica. Compare **computation saved against waiting added**, not hit rate alone.

## How does training connect to this service? {#training}

Model updates should not sit on the request's critical path. A minimal offline update flow is:

<div class="system-flow" role="group" aria-label="From recoverable training to a serving release">
<section class="system-flow-lane">
<h3>Produce a recoverable training result</h3>
<ol><li>Fix the data snapshot and transform versions</li><li>Sample, shard, and construct batches</li><li>Run forward, backward, gradient synchronization, and updates</li><li>Save recoverable training state</li></ol>
<p>Data position and optimizer progress belong to recovery state too.</p>
</section>
<section class="system-flow-lane">
<h3>Validate before serving</h3>
<ol><li>Run task evaluations and regression checks</li><li>Prepare serving weights, tokenizer, and configuration</li><li>Load, warm up, and validate on small traffic</li><li>Expand traffic after checks pass</li></ol>
<p>Model updates enter the deployment path above without blocking active requests.</p>
</section>
</div>

DDP typically replicates the model and synchronizes gradients. FSDP2 shards training state, gathers parameter groups for computation, and uses the corresponding gradient communication. Consider memory and communication together, not forward speed alone. The [PyTorch FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) explains gathering, computation, and prefetching; see [training frameworks](../06-systems/training-frameworks.en.md) for the wider choice.

A subtler failure is a distributed program that runs successfully but changes the objective. Suppose two ranks have 2 and 6 valid tokens, with local mean losses of 1 and 3. If the objective averages all valid tokens:

\[
\bar L=\frac{2\times1+6\times3}{2+6}=2.5,
\qquad \frac{1+3}{2}=2\ne\bar L.
\]

Averaging local means equally overweights the shorter contribution. This is objective-value arithmetic. Implementing backward also requires checking gradient reduction, gradient accumulation, and global valid-token counts; changing the displayed loss to 2.5 is not a fix by itself.

Training recovery needs more than weights. Optimizer state, learning-rate progress, randomness, the data cursor, and sharding configuration determine what happens next. Serving may not need those states. The artifacts can share weights while having different purposes and validation rules; see [checkpointing and resume](../practice/post-training/checkpoint-and-resume.en.md).

## Walk the complete path before release {#checks}

| Check | Concrete action | Expected behavior |
| --- | --- | --- |
| Compatible versions | Check weights, tokenizer, template, sampling configuration, and cache isolation together | New and old releases do not reuse incompatible state |
| Readiness | Send test traffic before loading and warmup finish | Unready instances do not accept normal traffic |
| Cancellation | Disconnect a client midway | Cancellation reaches the queue and engine; associated resources are eventually freed |
| Draining | Stop new assignments to an old replica; wait for active requests | No silent interruption of streams; bounded handling of drain timeouts |
| Recovery | Inject a worker failure or dependency timeout | Retries have budgets, failures have an owner, queues remain bounded |

Build a workload with short, medium, and long inputs and varied output limits before comparing engine speed. Peak throughput on fixed short prompts does not describe mixed-length user experience. Keep rejected and timed-out requests visible in the measurements.

[Queueing, capacity, and failures](queues-and-failures.en.md) turns these boundaries into small calculations and runnable models. They are estimation and debugging exercises, not production deployment configurations.
