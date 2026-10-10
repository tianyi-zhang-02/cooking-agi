# What happens between an LLM request and the answer on screen?

[中文](llm-serving.md) · **English**

> Last reviewed: 2026-10 · Prerequisite: [KV cache](../00-foundations/deep-dives/kv-cache-and-inference.en.md)

A model answers quickly on its own, then feels slow behind a website. The forward pass may not be the problem. Requests can wait in a queue, CPU input processing can fall behind, and a newly arrived document can occupy the next batch.

Follow one request through the service, then use a small scheduling example to see why higher throughput and smoother streaming are not always the same goal. We focus on ordinary autoregressive generation. Speculative decoding, hybrid-state models, and cross-machine caches add other paths.

## Start with who does what {#request-map}

<figure class="worked-update">
<ol>
<li><small>01 · API server</small><strong>Receive and prepare</strong><span>Validate the request, apply the chat template, and prepare text, images, or other inputs.</span></li>
<li><small>02 · Engine / scheduler</small><strong>Choose this iteration's work</strong><span>Manage queues, check cache and token budgets, and assign positions to compute.</span></li>
<li><small>03 · Worker → output</small><strong>Compute and return output</strong><span>Run the model and sample tokens; turn those tokens into text, handling stops and cleanup.</span></li>
</ol>
<figcaption>Unfinished requests return to scheduling. Requests make progress together; the service need not finish one answer before starting another.</figcaption>
</figure>

vLLM V1 separates API handling, engine-core scheduling and KV management, and GPU execution. These are responsibilities, not a claim that every deployment runs exactly three processes. [Official architecture overview](https://docs.vllm.ai/en/latest/design/arch_overview/)

## Trace a short request {#request-trace}

Suppose the prepared prompt is `[A, B, C]` and the generated answer is `[D, E]`. Letters stand in for token IDs.

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

Two requests are decoding, each needing one input position this iteration. A new 10-token prompt arrives. To isolate chunking, allow six input positions per iteration, schedule decode first, and give the remaining budget to prefill.

| Iteration | Existing requests | New prompt processed | Prompt still waiting |
| --- | ---: | ---: | ---: |
| 1 | 2 tokens | 4 tokens | 6 |
| 2 | 2 tokens | 4 tokens | 2 |
| 3 | 2 tokens | 2 tokens | 0 |

The new request can produce its first token after completing prefill in iteration 3. Existing requests receive compute each iteration. With a budget of 12, the prompt fits in one iteration—but that iteration may take longer, pausing the streams already in progress.

This is **budget arithmetic, not a latency prediction**. One prefill position and one decode position need not cost the same. KV capacity, batch shapes, priorities, and kernels also matter. Six scheduled positions do not imply a fixed number of milliseconds.

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

The [vLLM tuning guide](https://docs.vllm.ai/en/latest/configuration/optimization/) discusses the TTFT / ITL tradeoff. The current [V1 scheduler](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/core/sched/scheduler.py) does not maintain two entirely separate prefill and decode phases: it tracks computed positions and assigns additional work. Our table explains the budget without pretending to reproduce that implementation.

## Cancellation has to reach the engine {#request-lifecycle}

A user asks for up to 50 tokens but closes the page after five. If the service computes the rest anyway, it consumes GPU time and cache capacity for output nobody will receive.

Both normal completion and cancellation need cleanup: stop accepting further output for the request, notify the engine, stop scheduling new work, and release cache references. Already-issued GPU work may not be interruptible at an arbitrary instruction, so cancellation needs a defined completion boundary. At the pinned version, [`generate` handles cancellation](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/engine/async_llm.py#L763) by aborting the request; the [engine iteration](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/engine/core.py#L647) also processes aborts received during execution.

Releasing a reference does not return memory to the operating system. Another request may still use the prefix, or an unused cached block may remain in a preallocated pool until reuse or eviction. Continue with the [cache lifecycle](../00-foundations/deep-dives/attention-kernels.en.md#cache-lifecycle) for that distinction.

## Measure the delay before tuning parameters {#latency-diagnosis}

For the first token, instrument input processing, queueing, the model computation needed for first output, and output handling / transport. These are useful measurement boundaries, not necessarily disjoint intervals. Adding separate mean durations does not give end-to-end P95 latency.

| Observation | First checks | An easy wrong turn |
| --- | --- | --- |
| TTFT rises under concurrency | Arrival rate, queues, KV capacity, preemption | Replacing the tokenizer without inspecting queueing |
| The answer starts promptly but streams unevenly | ITL distribution, long prefills, synchronization, network buffering | Reporting only aggregate tokens/s |
| Similar prompts remain slow | Actual token prefixes, adapters, replicas, reused positions | Treating text similarity as a cache hit |
| Low GPU activity, busy CPUs | Input processing, scheduling, serialization, multimodal preparation | Adding GPUs without examining CPU queues |
| Capacity stays occupied after cancellation | Request termination and cache references | Closing only the frontend connection |

Load generation matters too. In a fixed-concurrency test, a completed request is replaced by another; a slower service also receives replacements more slowly. With a fixed arrival rate, queues can keep growing. Both tests are useful, but their tail latencies are not directly interchangeable. Record lengths, arrival pattern, cold or warm caches, hardware, and versions. Report failures and timeouts separately.

## Use the flow to choose a deployment {#serving-choices}

Fitting model weights on one GPU does not establish that the GPU can serve your target concurrency. A [complete memory budget](../00-foundations/deep-dives/kv-cache-and-inference.en.md#inference-budget) includes weights, KV, and runtime headroom. Identify the resource you are short of before scaling.

| Choice | What it may help | What it costs |
| --- | --- | --- |
| Independent replicas | Divide request load and increase total serving capacity | Duplicate weights; routing changes may lose warm-cache locality |
| Tensor parallelism | Distribute weights and some state within layers | Repeated communication; sensitivity to interconnect |
| Pipeline parallelism | Distribute model layers across devices | Activation transfers, bubbles, and scheduling complexity |
| Separate prefill and decode | Choose hardware and scheduling for each phase | KV transfers, coordinated queues, and failure handling |

This is not a framework ranking. Start with a simple working deployment, measure quality, cost, and latency on representative traffic, and then decide whether another system layer is worth it. [Parallelism mechanics](distributed-training.en.md) explains partitioning; inference capacity and service behavior still need their own validation.

## Where to enter the source code {#source-route}

This note checks vLLM `10cc2f6`; source links use the full commit rather than moving main. Start with the [completion API handler](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/entrypoints/openai/completion/serving.py), then follow `AsyncLLM.generate` → `EngineCore.step` → `Scheduler.schedule` → worker execution / sampling → output processor. Chat, asynchronous scheduling, and speculative decoding introduce branches; not every request follows one identical call stack.

We run the budget example and documentation checks, not a complete serving deployment of this version. No framework speedup is claimed. Reproducing performance also requires fixed models, hardware, engine settings, and request replay.
