# vLLM and SGLang: what a generation engine actually does

[中文](inference-frameworks.md) · **English**

Calling `generate()` is enough to try a model locally. Running a service introduces a different problem: one user sends a long document, another asks a short question, and a third disconnects halfway through an answer. Which requests should run together? Which should wait? Who gets the memory a finished request leaves behind?

That is much of the work vLLM and SGLang do. The model predicts tokens; the engine organizes many requests around the available GPUs. We will follow one request, then look at caching, scheduling, and multi-GPU deployment. For the generation basics, start with [prefill, decode, and KV caching](../00-foundations/deep-dives/kv-cache-and-inference.en.md). The surrounding service is covered in [a request through the service](llm-serving.en.md).

<details markdown="1">
<summary>Versions and source references</summary>

This chapter primarily uses vLLM v0.31.0 (2026-10-05) and SGLang v0.5.21 (2026-10-02). Documentation was checked on 2026-10-12; source links point to main-branch commits from 2026-10-10. On 2026-10-21 we also checked the release notes for [vLLM v0.32.0](https://github.com/vllm-project/vllm/releases/tag/v0.32.0) and [SGLang v0.5.22](https://github.com/sgl-project/sglang/releases/tag/v0.5.22). Among the defaults discussed here, the ROCm model runner changed; the source notes below explain that change.

Official references for ecosystem and integration choices were checked on **2026-10-31**. Those sections describe available integration paths. The implementation walkthrough still uses the versions stated alongside it; its defaults should not be assumed to apply to every newer release.

The numerical examples are calculations, not GPU benchmarks of either engine.

</details>

## Choosing an engine also means choosing an ecosystem {#ecosystem}

At the feature-list level, vLLM and SGLang have a lot in common: prefix caching, continuous batching, structured output, and multi-GPU serving. “One manages memory; the other is for conversations” is no longer a useful distinction. Implementation differences still matter, but a better starting question is: **which fits our model, hardware, and existing system?**

An ecosystem is more than a list of projects or a star count. It includes maintained model implementations, features that work together, deployment integrations, and reproducible bug reports and fixes. A few official entry points show how each project extends beyond its engine:

- **vLLM:** [Production Stack](https://docs.vllm.ai/en/latest/deployment/integrations/production-stack/) provides Kubernetes deployment, routing, monitoring configuration, and LMCache integration. It is a reference stack built around the engine, not functionality that appears automatically when a vLLM process starts.
- **SGLang:** [Model Gateway](https://docs.sglang.io/docs/advanced_features/sgl_model_gateway) routes requests across workers and provides health checks, rate limits, and routing policies. The [RL integration guide](https://docs.sglang.io/docs/advanced_features/sglang_for_rl) covers memory release and restoration, weight updates, and pausing generation. The gateway and engine instances are separate components and can be deployed separately.
- **These are overlapping ecosystems, not isolated alternatives:** llm-d documents integrations with both [vLLM](https://docs.vllm.ai/en/latest/deployment/integrations/llm-d/) and [SGLang](https://docs.sglang.io/docs/advanced_features/llm-d). [verl](https://verl.readthedocs.io/en/latest/start/install.html) also supports both rollout engines. Sharing a framework does not imply identical feature coverage, dependencies, or configuration across versions.

Both projects cover deployment and RL integration. These examples are starting points, not a maturity ranking. For your system, check the following:

| Goal | What to check | Why it affects the choice |
| --- | --- | --- |
| Serve a model | Pin the model revision, GPU SKU, and quantization format; check required combinations such as multimodal input, LoRA, and tool calling | Supporting a model does not establish that every combination has been tested |
| Join an existing service | Gateway, deployment configuration, metrics, scaling, cancellation, and failure recovery | Returning an answer is only part of deploying a service |
| Generate RL samples | Trainer and engine versions, weight synchronization, returned tokens and log probabilities, pause/resume behavior | Fast generation alone neither makes the full training loop fast nor guarantees correct attribution to policy versions |
| Maintain the system | Tests for the target model, relevant issues, release notes, and the team's familiarity with the execution path | One benchmark cannot reveal upgrade regressions, debugging effort, or rollback cost |

Look up the particular model rather than counting entries in a support list. For example, [vLLM's model support policy](https://docs.vllm.ai/en/latest/models/supported_models/) distinguishes levels of testing; [SGLang's model directory](https://docs.sglang.io/docs/supported-models) also needs to be followed through to the relevant model and feature documentation. A support claim is not an acceptance test for your configuration.

**Consider a team already serving a document assistant with vLLM on Kubernetes.** It has monitoring, alerts, rollback procedures, and load tests. To evaluate SGLang, it could start with shadow traffic and compare tail latency, failures, and cost. It would also need to check tool-call response formats, cancellation, and monitoring integration. A better single-GPU score alone does not justify the migration. The same reasoning applies to an established SGLang deployment: ecosystem reputation is not a reason to rebuild a working service.

**For RL rollouts, the integration questions change.** If both engines support the model, check the path through the training framework. Can updated weights reach the engine correctly? Can tokens and log probabilities return to the trainer? Is old KV invalidated when weights change? How long do sleep, wake-up, and recovery take? These steps can matter more to training efficiency than generation throughput alone. The rest of this chapter examines engine internals; [RL systems](rl-systems.en.md#engines) follows the full training loop.

Start by establishing that the required model and feature combination works, then check integration with the existing system, and finally compare performance on representative traffic. Ecosystem fit affects integration and maintenance cost; schedulers, kernels, and cache implementations can directly affect runtime cost. Account for both.

## Application, engine, and kernels {#layers}

<figure class="worked-update" lang="en">
<ol>
<li><small>01 / Application</small><strong>Prepare what the model will read</strong><span>Check permissions, assemble retrieved context, tools, and prompts, and set a deadline. Filter out unauthorized material before giving it to the model.</span></li>
<li><small>02 / Generation engine</small><strong>Choose which requests run together</strong><span>Maintain queues and batches, allocate KV space, dispatch workers, sample tokens, and return results. This is where vLLM and SGLang fit.</span></li>
<li><small>03 / Runtime and kernels</small><strong>Execute the work on GPUs</strong><span>Run the model’s matrix multiplications, attention kernels, and cross-device communication. This layer determines how a batch is computed.</span></li>
</ol>
<figcaption>These layers divide the work, not necessarily the processes. Permissions, request scheduling, and kernel performance need separate attention: a faster kernel will not fix an access-control mistake or a long queue.</figcaption>
</figure>

[vLLM V1](https://docs.vllm.ai/en/v0.31.0/design/arch_overview/) divides the engine's work further. API servers handle inputs and outputs, engine cores manage scheduling and caches, and GPU workers execute the model. Data-parallel deployments also use a DP coordinator. The deployment determines how many processes participate.

SGLang does similar work with a different organization and different component names. In the [Engine source linked here](https://github.com/sgl-project/sglang/blob/436d73ddda02d17d6d56e770a5362ebd5dd95ac0/python/sglang/srt/entrypoints/engine.py#L238), a request roughly follows this path:

```text
request → TokenizerManager → Scheduler → model execution
            ↑                                ↓
      response text ← DetokenizerManager ← output token IDs
```

TokenizerManager prepares inputs for execution and receives request results. Scheduler selects the next batch; after model execution, DetokenizerManager converts output tokens into text. This view leaves out multi-GPU topology and cross-instance routing, which can involve more processes.

So when the GPU is idle but answers are slow, look at where requests are waiting. Model computation may be fine while inputs are still being prepared, or results are taking too long to reach the client.

## Trace both implementations side by side {#engine-map}

For ordinary autoregressive text generation, both engines prepare inputs, select a batch, execute the model, and return results. The arrows below label what passes between components—mostly request information and scheduling results, not a copy of the entire model or KV cache at every step. Multimodal inputs and speculative decoding add other paths.

Although the diagram reads from top to bottom, CPU and GPU work can overlap: the CPU prepares the next batch while the GPU computes the current one. Compatible configurations enable this by default, through vLLM's async scheduling or SGLang's overlap scheduler.

<figure class="engine-map" lang="en">
<div class="engine-map__lanes">
<section aria-label="vLLM request path"><h3>vLLM V1</h3><ol>
<li><strong>API server / AsyncLLM</strong><span>Validate and prepare input; track requests and streaming output</span><small>↓ token IDs, sampling settings, request ID</small></li>
<li><strong>EngineCore → Scheduler</strong><span>Select queued requests, set this iteration's token counts, allocate KV blocks</span><small>↓ SchedulerOutput: request updates, token counts, block IDs</small></li>
<li><strong>Executor → GPU workers</strong><span>Dispatch scheduling results to workers that hold model shards</span><small>↓ SchedulerOutput → the worker's model runner</small></li>
<li><strong>Model runner → kernels</strong><span>Prepare input tensors and attention metadata, then run the model, access KV, and sample</span><small>↑ tokens / execution results return to scheduling and output</small></li>
</ol></section>
<section aria-label="SGLang request path"><h3>SGLang SRT</h3><ol>
<li><strong>HTTP → TokenizerManager</strong><span>Input processing, request state, returned results</span><small>↓ tokenized request</small></li>
<li><strong>Scheduler + cache / pools</strong><span>Select requests, match cached prefixes, assign cache locations</span><small>↓ ScheduleBatch; converted to ForwardBatch inside the worker</small></li>
<li><strong>Model worker → ModelRunner</strong><span>Prepare model inputs for the selected batch and run the forward pass</span><small>↓ input tensors and attention metadata</small></li>
<li><strong>Model → attention backend</strong><span>Run kernels, update KV, send output through detokenization</span><small>↑ DetokenizerManager → TokenizerManager</small></li>
</ol></section>
</div>
<figcaption>Unfinished requests continue into the next iteration. Boxes represent components, not necessarily separate processes; SGLang's ModelRunner is part of the scheduler's model-execution path.</figcaption>
</figure>

All these components contribute to the same request. Suppose request R has six prompt tokens, with reusable KV for the first four. For this example only, let a block hold two tokens: that gives us two complete cached blocks. Two tokens is not either engine's default block size.

| Step | What needs to happen |
| --- | --- |
| Find a cached prefix | Find the two blocks for the first four tokens and verify that they can be reused |
| Schedule this iteration | Allocate writable space for the two remaining tokens; wait if this iteration is already full |
| Process the rest of the prompt | Write KV for those two tokens and sample new token X from the final position's logits |
| Continue generation | Feed X into the next iteration to compute its KV and predict the following token |

Notice that **X's KV does not exist yet when X is sampled.** Sampling selects a token ID. The next forward pass feeds X into the model, writes its KV, and predicts the following token. A prefix hit does not supply a finished answer either: even a fully cached prompt may require recomputing its last position to obtain next-token logits, depending on the implementation.

Distinguish two kinds of traffic:

- **Control messages:** request metadata, cancellation, execution plans, completion notices. API-to-engine IPC does not imply sending weights with each message.
- **Tensor transfers:** TP collectives, PP activations, and moving KV between devices or instances. These have concrete bandwidth, synchronization, and topology costs.

<details markdown="1">
<summary>Source walkthrough: Scheduler to ModelRunner</summary>

For vLLM process boundaries, see the [architecture documentation](https://docs.vllm.ai/en/v0.31.0/design/arch_overview/), then follow [EngineCore](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/engine/core.py). It passes `SchedulerOutput` to the executor; the worker's model runner combines it with local request state to prepare tokens, positions, and attention metadata. The scheduler does not assemble all model-input tensors beforehand.

The pinned commit prefers [Model Runner V2](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/worker/gpu/model_runner.py) for compatible configurations. Unsupported features, missing Triton, some ROCm model defaults, or explicit settings can select the older [GPUModelRunner](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/worker/gpu_model_runner.py) instead. The later [v0.32.0](https://github.com/vllm-project/vllm/releases/tag/v0.32.0) makes V2 the default on ROCm too and drops the per-model exceptions that kept the older runner.

The SGLang path is also checked against a fixed commit: [Scheduler.run_batch](https://github.com/sgl-project/sglang/blob/436d73ddda02d17d6d56e770a5362ebd5dd95ac0/python/sglang/srt/managers/scheduler.py#L4405), the [worker's batch conversion and forward call](https://github.com/sgl-project/sglang/blob/436d73ddda02d17d6d56e770a5362ebd5dd95ac0/python/sglang/srt/managers/tp_worker.py#L598), and [ModelRunner](https://github.com/sgl-project/sglang/blob/436d73ddda02d17d6d56e770a5362ebd5dd95ac0/python/sglang/srt/model_executor/model_runner.py). Follow the example request through these entry points: where do its two remaining tokens become tensors, and where are their cache locations supplied?

</details>

## From ModelRunner to the GPU {#runtime-stack}

ModelRunner still has work to do before calling the model. It assembles the changing batch into tensors: which positions contain new tokens, where their KV belongs, and which attention implementation will process them. Model computation and device execution follow from those choices.

| Layer | Responsibility | A useful follow-up |
| --- | --- | --- |
| Model runner | Prepare tokens, positions, cache indices, and model calls | How does a changing batch become executable tensors? Why do shapes affect execution paths? |
| Model layers and attention backend | QKV projections, attention, MLPs, KV reads/writes | How do prefill and decode shapes differ? How does GQA affect the cache? |
| Compilation, kernels, CUDA Graphs | Generate or select kernels; reduce launch and scheduling overhead | Capture constraints on shapes and buffers; fallback paths when they do not fit |
| GPU memory and communication | HBM holds weights and KV; registers / shared memory reuse local data; devices exchange tensors | Compute, HBM bandwidth, or interconnect bound? Can transfers overlap useful computation? |

PyTorch, Triton, CUDA, and vendor libraries appear in different backends; they are not a fixed series of services that every request traverses. CUDA Graphs primarily reduce CPU overhead for launching GPU work (why that overhead stands out at small batch sizes is covered in [the framework layer](../12-infra/frameworks.en.md)). They do not make KV storage free or eliminate TP communication. FlashAttention's IO optimization and paged KV management also address different layers: how attention is computed versus how cached state is allocated and addressed. See [attention kernels and cache layouts](../00-foundations/deep-dives/attention-kernels.en.md).

The [SGLang ModelRunner](https://github.com/sgl-project/sglang/blob/436d73ddda02d17d6d56e770a5362ebd5dd95ac0/python/sglang/srt/model_executor/model_runner.py) is a useful place to read about these execution choices. Source explains why a path is taken; measurements show how it performs on a particular GPU. In an interview, distinguish implementations you have read from optimizations you have tested.

## Multi-GPU deployment: replicas, model shards, or separate stages {#deployment-map}

With more GPUs, you can run more model copies and divide the requests between them. Or several GPUs can cooperate on one model. A third option assigns prompt processing and output generation to different devices. These arrangements solve different problems.

<figure class="worked-update" lang="en">
<ol>
<li><small>Split requests / Replicas</small><strong>Request A → replica 1<br>Request B → replica 2</strong><span>Complete replicas share the incoming load. Each stores weights and its own cache; routing to another replica may lose a warm prefix.</span></li>
<li><small>Split the model / TP</small><strong>One request → GPU 0 + GPU 1</strong><span>Devices compute different parts of the same layer, then exchange results. Less work per device comes with more communication, so the interconnect matters.</span></li>
<li><small>Split stages / P–D</small><strong>Prefill worker → KV → decode worker</strong><span>Allocate resources separately for prompt processing and token generation. Handoffs require KV transfers, routing, and two queues, so first check that separation addresses the bottleneck.</span></li>
</ol>
<figcaption>These arrangements can be combined. Here, replicas are complete model copies; the MoE example below covers experts shared across replicas.</figcaption>
</figure>

For a teaching configuration with eight GPUs, TP=2 and four complete replicas means a request uses one two-GPU group, with four groups scheduling requests independently. It does not mean every request automatically uses all eight GPUs. Whether this beats TP=8 depends on model fit, KV headroom, single-request latency, arrivals, and interconnects. Start with the [memory budget](../00-foundations/deep-dives/kv-cache-and-inference.en.md#inference-budget), then measure representative traffic.

### Eight GPUs: TP8 or DP8? {#eight-gpus}

Start with a dense model and no shared experts. **TP8 uses eight GPUs to run one model cooperatively; DP8 / TP1 runs eight replicas handling separate requests.** A TP8 group can batch many requests—it is not limited to one user. The difference is who cooperates on each computation.

Suppose weights occupy exactly 16 GiB, each device has 24 GiB, and all weights can be partitioned evenly. Count only weights for now, excluding KV, workspace, and small replicated parameters:

| Configuration | Weights per GPU | Division of work |
| --- | --- | --- |
| TP8 · one cooperating group | Ideally 2 GiB | Eight ranks cooperate on the same batch. Weight storage per GPU falls, but layer computations need frequent communication |
| DP8 / TP1 · eight replicas | 16 GiB | Replicas share the incoming load without synchronizing dense forward computation. Each has its own cache and queue |
| DP2 / TP4 · two cooperating groups | Ideally 4 GiB | Each four-GPU group cooperates internally. Two groups share the load; this is not eight model replicas |

The table tells us weight storage, not how many requests can run or how many tokens they produce per second. DP8 leaves 8 GiB per GPU for KV and the runtime together, rather than 8 GiB dedicated to KV.

If weights fit and aggregate throughput matters, replicas are a useful baseline. If capacity or per-request computation is the problem, compare TP / PP. Smaller TP matrix multiplications do not guarantee a proportional latency reduction: communication, kernel efficiency, and scheduling can absorb the gains. [vLLM's scaling guide](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/) provides starting configurations; choose using your actual request distribution.

Hold GPU count, arrival load, input/output lengths, and cache warmup fixed. Alongside offline tokens/s, measure time to first token (TTFT), output intervals, tail latency, rejected requests, and peak memory. Higher throughput with longer queues may be a poor trade for an interactive product.

### Serving DP does not always mean independent replicas {#moe-deployment}

Training DDP synchronizes gradients; serving data parallelism typically distributes requests. With MoE, check whether experts are shared across those replicas. In [vLLM's EP deployment documentation](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/), checked on 2026-10-09, enabling EP forms an expert group of size TP × DP. Attention replication/sharding and expert placement are different decisions. This is a framework convention, not a universal definition.

Take a **hypothetical weight budget**: 6 GiB of shardable non-expert weights and 42 GiB of routed experts on four GPUs. Ignore replicated small tensors, quantization metadata, KV, and workspace:

| Configuration | Non-expert weights per GPU | Routed experts per GPU | Total per GPU |
| --- | --- | --- | --- |
| TP1 / DP4 / EP4 | 6 GiB | 10.5 GiB | 16.5 GiB |
| TP4 / DP1 / EP4 | 1.5 GiB | 10.5 GiB | 12 GiB |

The second saves 4.5 GiB per GPU but requires TP cooperation for attention. The first keeps attention local to each rank while still exchanging activations for remote experts. **It is not four completely independent services.** Consider replica caches and queues together with shared-expert scheduling.

Whether those 4.5 GiB are worth saving depends on performance too. Hold arrivals, input/output lengths, and cache warmth fixed, then compare first-token latency, output intervals, tail latency, and expert load. Very small matrix shards can save memory while spending more time waiting for communication.

### Where does PP keep KV, and what crosses a boundary? {#pp-kv}

Assume an ordinary decoder partitioned by layer, without cache offload or prefill/decode disaggregation. Each GPU retains KV for its own layers. Adjacent stages exchange boundary activations rather than repeatedly moving the entire historical KV cache.

The activation volume depends on how many tokens this iteration processes. With hidden size 3072 and two bytes per BF16 element:

| One boundary transfer | Element count | Raw bytes |
| --- | --- | --- |
| Eight requests decoding one token each | 8 × 3072 | 48 KiB |
| Eight requests prefilling 4096 tokens each | 8 × 4096 × 3072 | 192 MiB |

For the same eight requests, decode sends 48 KiB while full prefill sends 192 MiB. Neither figure includes extra residual tensors, metadata, protocol costs, or synchronization. Chunking a long prefill reduces each message's size but also changes how the pipeline must be scheduled.

[vLLM's scaling guide](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/) recommends pipeline parallelism instead of tensor parallelism for higher throughput when GPUs lack NVLink (for example, L40S). That is not proof of optimality on every PCIe machine. A single request still traverses stages sequentially; unequal stage work creates idle time. If the model fits on one GPU, compare independent replicas, quantized single-GPU serving, TP, and PP under the same workload. Disaggregated serving that transfers KV introduces a different communication path.

For a transfer lower bound, assume effective boundary bandwidth of 12 GiB/s and ignore startup latency. The 48 KiB message takes about 3.8 microseconds; the 192 MiB message takes about 15.6 milliseconds. Startup and synchronization can dominate the first, while bytes already matter in the second. This is assumed bandwidth, not a measured PCIe link or GPU specification.

PP can reduce the frequency of cross-device cooperation, but preserves sequential dependencies: with one request and one next token, later stages still wait for earlier results. More microbatches let different requests occupy different stages, provided there is enough traffic; queues and cache usage can also grow. Lack of NVLink is one input to the decision, alongside topology, prefill length, concurrency, and the latency budget.

## Refill a batch as requests finish {#batching}

Continuous batching reduces idle slots without adding GPUs. Consider a simplified decode schedule: A and B have completed prefill and need two and five more tokens. C is ready and needs three. Each round accommodates two requests, producing one token each; set other costs aside for now.

| Rounds | Wait for the entire batch | Refill between rounds |
| --- | --- | --- |
| 1–2 | A, B | A, B |
| 3–5 | B, with one empty slot | B, C |
| 6–8 | C | Finished |

After A finishes in round 2, C takes its slot and runs alongside B. The example therefore finishes in five rounds instead of eight. The real speedup depends on prefill time, KV capacity, and the cost of reading long histories; 8/5 is not a benchmark result. A replacement request still needs memory and room in the iteration's token budget. When those run out, requests wait, are rate-limited, or are rejected.

<!-- widget:step-continuous-batching -->

## Two ways to organize prefix reuse {#prefix}

Suppose two users ask about the same document. Each prompt contains the same 2,000-token document followed by a different 100-token question. If the shared prefix's KV is still available, the second request can skip that prefill computation. Its answer still attends to those cached keys and values; reuse avoids recomputation without removing the document from attention.

In the versions used here, both engines enable prefix reuse by default: `--enable-prefix-caching` in vLLM and radix cache in SGLang. A better comparison than “memory management versus caching” is how they find a prefix, how much of it they can reuse, and what they evict when space runs out.

| Question | vLLM V1 | SGLang radix tree (default `UnifiedRadixCache`) |
| --- | --- | --- |
| How is reusable state organized? | Prefix-dependent block hashes and a block pool, looked up by hash | A tree of shared token paths; v0.5.21 prefers a Rust tree core, with Python fallbacks for incompatible platforms or features |
| What is the reuse boundary? | Base default: 16-token blocks; the model, attention backend, and configuration can change the size. Only full blocks are reused | Page size 1 by default, so a match can stop at any token; some backends or models raise the page to 64, 128, and so on, and matches then align down to whole pages |
| How is state managed? | Reference counts, an LRU free queue for eviction, reallocation | Locking, references, node splitting; LRU by default, with LFU, SLRU, priority, and T-LRU as options |
| How should performance be compared? | Measure computation saved on real requests; block organization alone does not guarantee faster serving | Test the same requests; a tree is not faster for every workload |

Returning to the document example, visually identical text is not enough for reuse. Weights, adapters, token IDs, positions, and multimodal inputs must also be compatible; otherwise, the same token can produce different KV.

With two replicas, A and B, there is another question: **which replica has the cache?** Assume there is no cross-instance KV transfer. If A holds the document's KV but the next request goes to B, B still has to recompute it even though it supports prefix reuse. Sending the request to A may save computation, but a long queue at A could make recomputation on B faster. That is why routing outside the engine matters. [Production Stack's KV-aware routing](https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/kv-cache-aware-routing.html) and [SGLang Model Gateway](https://docs.sglang.io/docs/advanced_features/sgl_model_gateway) both provide relevant mechanisms; how to balance cache reuse against load still needs testing with your traffic.

Multi-tenant reuse also needs isolation. Cache-dependent timing can reveal whether content was present; caching is not authorization. [vLLM cache salting](https://docs.vllm.ai/en/v0.31.0/design/prefix_caching/) provides one reuse-isolation mechanism, and SGLang's OpenAI-compatible API also accepts `cache_salt`. The service must still assign trusted groups rather than letting clients claim another tenant's group.

<details markdown="1">
<summary>Cache configuration, page alignment, and implementation references</summary>

The model, backend, and configuration can change the defaults in the table. Check vLLM's [runner and async-scheduling conditions](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/config/vllm.py#L709), its [cache configuration](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/config/cache.py#L72), and SGLang's [tree-core fallback conditions](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/mem_cache/unified_cache/tree_core_registry.py#L44). Confirm the resolved configuration in startup logs before benchmarking.

See the pinned implementations and page-alignment example in [cache lifecycle](../00-foundations/deep-dives/attention-kernels.en.md#cache-lifecycle). The [original SGLang paper](https://arxiv.org/abs/2312.07104) explains RadixAttention's context, not the complete feature boundary of today's products.

</details>

## HiCache moves beyond GPU memory {#hicache}

[SGLang HiCache](https://docs.sglang.io/docs/advanced_features/hicache_design) extends KV storage across GPU memory, host memory, and storage backends. GPU and host tiers belong to inference instances; cross-instance reuse depends on the storage backend and namespace configuration. Two machines' RAM is not automatically one shared cache. vLLM now offers the same kind of tiering: the [OffloadingConnector](https://docs.vllm.ai/en/v0.31.0/features/kv_offloading_usage/) copies finished full blocks into CPU memory and can add secondary tiers such as files or object storage, or you can plug in LMCache or Mooncake Store instead. Neither engine turns this on by default.

Keeping more cache does not always make serving faster. Suppose recomputing a prefix takes 80 ms, while restoring its KV takes 30 ms plus 5 ms for lookup and scheduling. Reuse could save about 45 ms. But if transfer takes 100 ms, recomputing may be cheaper. We add the times serially here to make the tradeoff clear; actual execution can overlap, so check a performance trace.

This is why a high hit rate may not make the service faster. Hits might cover only short prefixes, or restoring their KV might take too long. Record reused tokens, transferred bytes, restore time, and the lookup cost of a miss to see whether the extra cache tier is worthwhile.

## Diagnose before adding another optimization {#bottlenecks}

| Symptom | Inspect first | Options to compare | Possible cost |
| --- | --- | --- | --- |
| Slow first token | Queueing, tokenization, prefill, cache | Chunking, prefix reuse, replicas | TTFT / ITL tradeoff; more memory |
| Uneven streaming | Decode batches, KV reads, contention | Scheduling, quantization, kernels, prefill/decode separation | Quality, transfers, operational complexity |
| Model does not fit | Weights, KV, workspace separately | Quantization, TP / PP | Communication and hardware constraints |
| Invalid JSON | Whether output constraints are enabled | Constrained decoding and validation | Valid syntax still permits false content |
| Speculative decoding is slower | Draft cost, acceptance, verification batch | Change or disable the draft method | Extra state and verification work |

Following the request path also shows how much the engines have in common. In the versions used here, both support:

- **Scheduling and caching:** paged KV, prefix reuse on by default, continuous batching, chunked prefill, and tiered KV storage.
- **Speeding up generation:** CPU scheduling overlapped with GPU execution, CUDA Graphs, and EAGLE / MTP / n-gram speculative decoding.
- **Output control and multi-GPU deployment:** structured output through xgrammar and others, TP / PP / DP + EP, and prefill/decode disaggregation.

The same feature names can still hide differences in scheduling, cache structures, and compatibility. In the source versions pinned here, the prefix index is a block-hash table in vLLM and a radix tree in SGLang, with base defaults of 16 and 1 token respectively; actual granularity depends on the backend and model. vLLM schedules decode tokens first and fills the rest of the same batch with prefill chunks. SGLang defaults to running a prefill batch whenever new prefill work is ready and mixes the two only with `--enable-mixed-chunk`. These are concrete scheduling examples, not permanent distinctions between all future releases.

Execution also differs: in those versions, vLLM uses torch.compile by default and SGLang does not. For disaggregated serving, vLLM connects transfer backends through KV connectors; its [v0.31.0 documentation](https://docs.vllm.ai/en/v0.31.0/features/disagg_prefill/) labels the feature experimental. [SGLang's deployment guide](https://docs.sglang.io/docs/advanced_features/pd_disaggregation) configures separate prefill and decode workers, with a router distributing requests. That router is not the scheduler inside each engine instance. These boundaries matter for deployment and debugging, but “built-in” and “experimental” are not enough to rank stability in a particular environment.

For engine selection, **pin the software versions and compare using the same model, hardware, and required features**. Labels such as “for agents” or “for concurrency” are not enough to choose between them. FSDP handles [training state](training-frameworks.en.md), a different job from the generation engines compared here.

## Design a meaningful comparison {#benchmark}

For a document assistant, test long documents and follow-up questions. If traffic often arrives in bursts, test that too. A single “hello” confirms that the service responds, but says little about which engine fits the workload. Include short questions, long documents, shared-prefix conversations, bursts, and cancellations, recording input and actual output lengths. Use the same weights, tokenizer, chat template, sampling settings, and hardware; report cold starts separately from warmed runs.

| Report | Reason |
| --- | --- |
| TTFT, ITL, end-to-end p50 / p95 / p99 | Throughput hides waiting; short outputs still contribute TTFT |
| Input and output token throughput separately | They represent different computation |
| Requests/s meeting quality and latency requirements | Timeouts do not have the same value as useful completions |
| Timeouts, cancellations, rejections, OOMs, errors | Successful requests alone create survivor bias |
| GPU / CPU memory, network traffic, total cost | The bottleneck may move away from the GPU |

How the load generator sends requests matters too. An open-loop test keeps sending on schedule, exposing queue buildup. In a typical closed-loop test, each client waits for its previous request to finish before sending another, so offered load falls when the service slows. Both are useful, but report requests sent as well as requests completed. A shared seed does not guarantee identical generated text across engines, batches, or kernels; check output quality and format separately.

Keep the software versions, configuration, inputs, results, and failures so you can rerun the comparison after changing models or upgrading an engine. [Serving and feedback](../practice/ml-system/serving-and-feedback.en.md) continues with integrating the engine into a product.

## Explain the path before listing features {#interview-walkthrough}

If an interviewer asks how vLLM works, following a request is often clearer than starting with a feature list. For example:

> I would trace a request first. The application handles access, prompts, and incoming traffic. The engine owns queues, compute budgets, and KV state, and schedules each iteration. Workers execute the model and kernels, then return results for streaming. Input/output lengths, arrival rate, and latency targets tell me which bottleneck to investigate before changing batches, replicas, or model/stage partitioning.

If the next question is “why vLLM rather than SGLang?”, explain the shared capabilities before describing your model, hardware, deployment or training framework, and the combinations you have validated. Then discuss measurements and maintenance costs. Knowing the prefix-index implementation is useful, but it is not a complete selection argument. For anything you have not measured, explain how you would test it.

That explanation naturally leads to questions about caching, communication, and scheduling. Use these follow-ups to check your understanding rather than memorize a script:

| Follow-up | Explain | Evidence to bring |
| --- | --- | --- |
| Why is a service slow despite a high prefix hit rate? | How many tokens hit, and on which replica? Queues, decode, and transfers remain after prefill reuse | Phase traces, reused tokens, cold/warm comparisons |
| What happens when KV is full? | Wait, reject, evict reclaimable blocks, preempt, or recompute; distinguish reclaimable cache from state still used by active requests | Capacity, references, preemption/recomputation, tail latency |
| Is TP=4 always faster than TP=2? | Less work per device can come with more communication or a larger communication share; small batches may underutilize hardware | Compute/collective timings, interconnect, batch and model structure |
| A client disconnected but memory did not drop—is that a leak? | Check termination and references; retaining a memory pool does not mean continuing to serve the request | Request lifecycle, active blocks versus idle pool, not process memory alone |
| How would you demonstrate an improvement? | Match versions and traffic; compare completions meeting quality and latency targets, cost, and errors | Open-arrival testing, failures, reproducible settings—not just tokens/s |

Read source by following a request object into a batch, cache indices, and execution. You do not need to read the repository in file order. For the surrounding system, continue with [CPU, GPU, and cluster scheduling](execution-boundaries.en.md) or [connecting generation to RL training](rl-systems.en.md#engines). For the broader preparation route, see [RE / MLE and inference interviews](../interview/ai-roles-prep.en.md#inference).

## Courses and further calculations {#further-reading}

- **Why do prefill and decode hit different bottlenecks?** [Stanford CS336 Lecture 10: Inference](https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py) works through arithmetic intensity by counting computation and data movement separately. Its throughput and latency section also explains why a larger batch need not make an individual request faster.
- **Want to calculate memory, bandwidth, and multi-device costs yourself?** Read the linear-layer, attention, and distributed-inference sections of [How To Scale Your Model: Transformer Inference](https://jax-ml.github.io/scaling-book/inference/), then try the worked problems. The configurations are examples, not optimal settings for every model and accelerator.

Use these resources for the calculations. For defaults, backend compatibility, and call paths, return to the versioned documentation and code linked in this chapter. An idealized cost model helps decide what to measure; it does not replace measurements with real queues, cache states, and communication costs.
