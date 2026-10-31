# What happens between an LLM request and the answer on screen?

[中文](llm-serving.md) · **English**

> Last reviewed: 2026-10 · Prerequisite: [KV cache](../00-foundations/deep-dives/kv-cache-and-inference.en.md)

A model answers quickly on its own, then feels slow behind a website. The forward pass may not be the problem. Requests can wait in a queue, CPU input processing can fall behind, and a newly arrived document can occupy the next batch.

Follow one request through the service: who prepares it, who decides when it runs, and who sends the result to the screen. A short token trace establishes the mechanics; scheduling and queueing examples then show why higher throughput need not mean an earlier answer or smoother streaming. We focus on ordinary autoregressive generation. Speculative decoding, hybrid-state models, and cross-machine caches add other paths.

## Start with who does what {#request-map}

<figure class="worked-update">
<ol>
<li><small>01 · API server</small><strong>Receive and prepare</strong><span>Validate the request, apply the chat template, and prepare text, images, or other inputs.</span></li>
<li><small>02 · Engine / scheduler</small><strong>Choose this iteration's work</strong><span>Manage queues, check cache and token budgets, and assign positions to compute.</span></li>
<li><small>03 · Worker → output</small><strong>Compute and return output</strong><span>Run the model and sample tokens; turn those tokens into text, handling stops and cleanup.</span></li>
</ol>
<figcaption>Unfinished requests return to scheduling. Requests make progress together; the service need not finish one answer before starting another.</figcaption>
</figure>

vLLM V1 separates API handling, engine-core scheduling and KV management, and GPU execution. These are responsibilities, not a claim that every deployment runs exactly three processes. Compare the [official architecture overview](https://docs.vllm.ai/en/latest/design/arch_overview/) with the [V1 process architecture in v0.31.0](https://docs.vllm.ai/en/v0.31.0/design/arch_overview/#v1-process-architecture). The versioned documentation helps identify responsibilities; it does not change the source revision pinned below.

## Trace a short request {#request-trace}

Suppose the prepared prompt is `[A, B, C]` and the generated answer is `[D, E]`. Letters stand in for token IDs. Keep track of two different events: processing a position and selecting the next token. Selecting a token does not mean it has already been fed back through the model.

| Stage | What happens | What you cannot conclude yet |
| --- | --- | --- |
| Receive | Validate parameters and length; the chat template adds roles and separators | Visible text length equals model token count |
| Register and queue | Store request ID and sampling settings; wait for resources | Accepting HTTP means GPU work has started |
| Match and schedule | Find reusable prefix state; assign KV slots and a compute budget | A cache hit eliminates queueing or all computation |
| Prefill | Process A, B, C; use C's output to sample D | D already has cached K/V |
| Next decode iteration | Feed D through the model, creating its K/V, then sample E | The answer was retrieved from a response cache |
| Output and finish | Update text, check stopping conditions, clean up request state | Each network message contains exactly one token |

The first generated token is sampled too. Prefill, decode, and sampling are not three stages that each run once. Whenever the model selects another token, it applies the relevant logit-processing and sampling rules. An intermediate prefill chunk usually cannot produce an answer yet; the request must reach a generation boundary.

Output processing is more than converting each token ID into one character. A character or stop string can span tokens, requiring buffering. “A token was computed” and “a text chunk is ready to send” are different events. Follow [`AsyncLLM.generate`](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/engine/async_llm.py) and the [output processor](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/engine/output_processor.py) at the pinned version to connect the two.

## A long prompt arrives while others are streaming {#token-budget}

Now put that request into a shared service. Two requests are decoding, each needing one input position this iteration. A new 10-token prompt arrives. To isolate chunking, allow six input positions per iteration, schedule decode first, and give the remaining budget to prefill.

| Iteration | Existing requests | New prompt processed | Prompt still waiting |
| --- | ---: | ---: | ---: |
| 1 | 2 tokens | 4 tokens | 6 |
| 2 | 2 tokens | 4 tokens | 2 |
| 3 | 2 tokens | 2 tokens | 0 |

The new request can produce its first token after completing prefill in iteration 3. Existing requests receive compute each iteration. With a budget of 12, the prompt fits in one iteration—but that iteration may take longer, pausing the streams already in progress. The tradeoff is between starting the new answer sooner and keeping existing answers moving steadily.

This is **budget arithmetic, not a latency prediction**. One prefill position and one decode position need not cost the same. KV capacity, batch shapes, priorities, and kernels also matter. Six scheduled positions do not imply a fixed number of milliseconds. Why a larger decode batch moves from bandwidth-bound toward compute-bound is covered in [Performance: decode batching](../12-infra/performance.en.md#decode-batching).

<!-- widget:step-chunked-prefill -->

<details markdown="1">
<summary>Check the chunk sizes with a small function</summary>

```python
def prefill_chunks(prompt_tokens, decode_tokens, token_budget):
    if any(type(value) is not int for value in
           (prompt_tokens, decode_tokens, token_budget)):
        raise ValueError("Token counts must be integers")
    if prompt_tokens < 0 or not 0 <= decode_tokens <= token_budget or token_budget < 1:
        raise ValueError("Invalid token budget")
    remaining_budget = token_budget - decode_tokens
    if prompt_tokens and remaining_budget <= 0:
        raise ValueError("No capacity left for prefill in this toy schedule")
    chunks = []
    while prompt_tokens:
        scheduled = min(prompt_tokens, remaining_budget)
        chunks.append(scheduled)
        prompt_tokens -= scheduled
    return chunks

assert prefill_chunks(10, 2, 6) == [4, 4, 2]
assert prefill_chunks(10, 2, 12) == [10]
```

This checks a budget with fixed decode demand. It does not simulate vLLM, request completion, preemption, or recomputation after preemption.

</details>

The [vLLM tuning guide](https://docs.vllm.ai/en/latest/configuration/optimization/) discusses the tradeoff between time to first token (TTFT) and inter-token latency (ITL). The [V1 scheduler at our pinned revision](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/core/sched/scheduler.py) does not maintain two entirely separate prefill and decode phases: it tracks computed positions and assigns additional work. Our table explains the budget without pretending to reproduce that implementation.

## Cancellation has to reach the engine {#request-lifecycle}

Scheduling also needs to account for requests that have left. A user asks for up to 50 tokens but closes the page after five. If the service computes the rest anyway, it consumes GPU time and cache capacity for output nobody will receive.

Both normal completion and cancellation need cleanup: stop accepting further output for the request, notify the engine, stop scheduling new work, and release cache references. Already-issued GPU work may not be interruptible at an arbitrary instruction, so cancellation needs a defined completion boundary. At the pinned version, [`generate` handles cancellation](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/engine/async_llm.py#L763) by aborting the request; the [engine iteration](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/engine/core.py#L647) also processes aborts received during execution.

Releasing a reference does not return memory to the operating system. Another request may still use the prefix, or an unused cached block may remain in a preallocated pool until reuse or eviction. Continue with the [cache lifecycle](../00-foundations/deep-dives/attention-kernels.en.md#cache-lifecycle) for that distinction.

## Measure the delay before tuning parameters {#latency-diagnosis}

Use the request path to choose measurement points: input processing, queueing, the model computation needed for the first token, and output handling / transport. These are useful measurement boundaries, not necessarily disjoint intervals. Adding separate mean durations does not give end-to-end P95 latency.

| Observation | First checks | An easy wrong turn |
| --- | --- | --- |
| TTFT rises under concurrency | Arrival rate, queues, KV capacity, preemption | Replacing the tokenizer without inspecting queueing |
| The answer starts promptly but streams unevenly | ITL distribution, long prefills, synchronization, network buffering | Reporting only aggregate tokens/s |
| Similar prompts remain slow | Actual token prefixes, adapters, replicas, reused positions | Treating text similarity as a cache hit |
| Low GPU activity, busy CPUs | Input processing, scheduling, serialization, multimodal preparation | Adding GPUs without examining CPU queues |
| Capacity stays occupied after cancellation | Request termination and cache references | Closing only the frontend connection |

Load generation matters too. In a fixed-concurrency test, a completed request is replaced by another; a slower service also receives replacements more slowly. With a fixed arrival rate, queues can keep growing. Both tests are useful, but their tail latencies are not directly interchangeable. Record lengths, arrival pattern, cold or warm caches, hardware, and versions. Report failures and timeouts separately.

### A small example: who benefits from waiting to batch? {#batching-wait}

Consider a **simplified queueing example, not a hardware measurement**. Four requests reach one idle executor at times 0, 1, 2, and 3. Assume processing one request takes 4 time units, while processing all four as a batch takes 6. Units are arbitrary, other overhead is ignored, and batches cannot overlap. To isolate queueing, treat each request as one indivisible computation rather than a stream of tokens.

| Scheduling choice | Execution intervals | Completion times for the four requests | First request's arrival-to-completion latency |
| --- | --- | --- | --- |
| Serve one at a time in arrival order, starting whenever idle | 0–4, 4–8, 8–12, 12–16 | 4, 8, 12, 16 | 4 |
| Wait for all four, then process them together | 3–9 | 9, 9, 9, 9 | 9, including 3 waiting |

Measured from time 0 until all requests finish, the completion rates are `4 / 16 = 0.25` and `4 / 9 ≈ 0.44` requests per time unit. Batching finishes the workload sooner but makes the first request finish later. These are **finite-window completion rates, not steady-state throughput or TTFT**. They do not justify always waiting for a full batch. Continuous batching can change membership at iteration boundaries rather than waiting for every answer in a batch to finish; the course section below explains that distinction.

## Use the flow to choose a deployment {#serving-choices}

Fitting model weights on one GPU does not establish that the GPU can serve your target concurrency. A [complete memory budget](../00-foundations/deep-dives/kv-cache-and-inference.en.md#inference-budget) includes weights, KV, and runtime headroom. Identify the resource you are short of before scaling. [vLLM and SGLang: several ways to use several GPUs](inference-frameworks.en.md#deployment-map) compares the benefits and costs of independent replicas, TP, PP, and prefill/decode separation, with an eight-GPU weight-storage example. For scheduling and scaling replicas across a cluster, see [Kubernetes and GPU clusters](../12-infra/kubernetes.en.md).

This is not a framework ranking. Start with a simple working deployment, measure quality, cost, and latency on representative traffic, and then decide whether another system layer is worth it. [Parallelism mechanics](distributed-training.en.md) explains partitioning; inference capacity and service behavior still need their own validation.

## Where to enter the source code {#source-route}

Start with two focused questions for [Stanford CS336 Lecture 10](https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py):

- `throughput_and_latency()`: batch size versus latency under a bandwidth model with ideal overlap and no overhead—not an end-to-end latency prediction.
- `continuous_batching()`: admitting requests at iteration boundaries. Compare it with our budget table, without treating the lecture sketch as a version-specific scheduler.

Then move into the implementation. This note checks vLLM `10cc2f6`; source links use the full commit so later changes to main do not invalidate the explanation. Start with the [completion API handler](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/entrypoints/openai/completion/serving.py), then follow `AsyncLLM.generate` → `EngineCore.step` → `Scheduler.schedule` → worker execution / sampling → output processor. Chat and speculative decoding introduce branches, so not every request follows the same call stack.

Async scheduling is on by default for compatible configurations at this commit (`AsyncScheduler` subclasses `Scheduler`). The CPU prepares the next step while the GPU runs the current one, so the two steps can overlap in a trace. Some pooling models, executors, or speculative-decoding configurations disable it. Check those conditions as well as the default setting.

We run the budget example and documentation checks, not a complete serving deployment of this version. No framework speedup is claimed. Reproducing performance also requires fixed models, hardware, engine settings, and request replay.
