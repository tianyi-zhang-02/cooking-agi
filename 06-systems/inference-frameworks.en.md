# vLLM and SGLang: what a generation engine actually does

[中文](inference-frameworks.md) · **English**

> Checked 2026-10-12 against vLLM v0.31.0 (2026-10-05) and SGLang v0.5.21 (2026-10-02) documentation; source links are pinned to main-branch commits from 2026-10-10. On 2026-10-21 we also read the release notes for [vLLM v0.32.0](https://github.com/vllm-project/vllm/releases/tag/v0.32.0) and [SGLang v0.5.22](https://github.com/sgl-project/sglang/releases/tag/v0.5.22); among the defaults this page mentions, they show one change (the ROCm model runner), noted in the text. Numbers below are teaching examples, not framework benchmark results.

If `generate()` can answer a question, why add an engine? A service has many users: short and long requests arrive together, clients disconnect, and several questions may share the same document. An engine organizes that work rather than giving each request exclusive use of the GPU from scratch.

Start with [prefill, decode, and KV caching](../00-foundations/deep-dives/kv-cache-and-inference.en.md). [A request through the service](llm-serving.en.md) follows execution more closely; this page focuses on architecture and selection.

## Application, engine, and kernels {#layers}

<figure class="worked-update" lang="en">
<ol>
<li><small>01 / Application</small><strong>Decide what to do</strong><span>Authorization, retrieval, tools, prompts, and deadlines. Do not pass unauthorized evidence to a model and hope it keeps it secret.</span></li>
<li><small>02 / Generation engine</small><strong>Schedule work and reuse state</strong><span>Queues, batches, KV management, workers, sampling, and output: the role of vLLM or SGLang.</span></li>
<li><small>03 / Runtime and kernels</small><strong>Execute the current work</strong><span>Model execution, attention, matrix multiplication, and communication. A faster kernel may leave queueing untouched.</span></li>
</ol>
<figcaption>These are responsibilities, not a required process layout. The engine is not the entire production security boundary.</figcaption>
</figure>

[vLLM V1 architecture](https://docs.vllm.ai/en/v0.31.0/design/arch_overview/) separates input/output handling in API servers, scheduling and cache management in engine cores, and model execution in GPU workers; data-parallel deployments add a DP coordinator. Process counts depend on deployment. SGLang has its own request, scheduling, and execution paths; vLLM class names are not universal terminology.

More concretely, the [pinned SGLang Engine](https://github.com/sgl-project/sglang/blob/436d73ddda02d17d6d56e770a5362ebd5dd95ac0/python/sglang/srt/entrypoints/engine.py#L238) separates tokenization, scheduling, and detokenization:

```text
request → TokenizerManager → Scheduler → model execution
            ↑                                ↓
      response text ← DetokenizerManager ← output token IDs
```

TokenizerManager handles inputs and request results; Scheduler selects batches; the execution path runs forward passes; DetokenizerManager converts tokens to text. This omits multi-GPU topology and cross-instance routing, rather than promising exactly three processes in every deployment. It also helps diagnosis: low GPU load with slow requests may call for inspecting input preparation and output handling, not only attention kernels.

## Trace both implementations side by side {#engine-map}

Follow vLLM down the left column, then SGLang down the right. Arrows label what crosses an interface; they do not mean copying the whole model or KV cache at every step. This diagram covers ordinary autoregressive text generation; multimodal input and speculation add branches. It is drawn one step at a time, but with compatible configurations both engines default to letting the CPU schedule the next step while the GPU runs the current one (vLLM calls this async scheduling, SGLang the overlap scheduler).

<figure class="engine-map" lang="en">
<div class="engine-map__lanes">
<section aria-label="vLLM request path"><h3>vLLM V1</h3><ol>
<li><strong>API server / AsyncLLM</strong><span>Validate and prepare input; track requests and streaming output</span><small>↓ token IDs, sampling settings, request ID</small></li>
<li><strong>EngineCore → Scheduler</strong><span>Queues, compute budgets, KV block allocation</span><small>↓ SchedulerOutput: request updates, token counts, block IDs</small></li>
<li><strong>Executor → GPU workers</strong><span>Dispatch scheduling results to workers that hold model shards</span><small>↓ SchedulerOutput → the worker's model runner</small></li>
<li><strong>Model runner → kernels</strong><span>Prepare input tensors and attention metadata, then run the model, access KV, and sample</span><small>↑ tokens / execution results return to scheduling and output</small></li>
</ol></section>
<section aria-label="SGLang request path"><h3>SGLang SRT</h3><ol>
<li><strong>HTTP → TokenizerManager</strong><span>Input processing, request state, returned results</span><small>↓ tokenized request</small></li>
<li><strong>Scheduler + cache / pools</strong><span>Waiting and running requests, prefix matching, cache locations</span><small>↓ ScheduleBatch; converted to ForwardBatch inside the worker</small></li>
<li><strong>Model worker → ModelRunner</strong><span>Execute the scheduled batch; not a separate serving service</span><small>↓ input tensors and attention metadata</small></li>
<li><strong>Model → attention backend</strong><span>Run kernels, update KV, send output through detokenization</span><small>↑ DetokenizerManager → TokenizerManager</small></li>
</ol></section>
</div>
<figcaption>Unfinished requests continue into the next iteration. These are responsibility maps, not matching process diagrams. SGLang's ModelRunner belongs to the scheduler's execution path; a box does not imply a separate process.</figcaption>
</figure>

For vLLM process boundaries, see the [architecture documentation](https://docs.vllm.ai/en/v0.31.0/design/arch_overview/), then follow [EngineCore](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/engine/core.py). It passes `SchedulerOutput` to the executor; the worker's model runner combines it with local request state to prepare tokens, positions, and attention metadata. The scheduler does not assemble all model-input tensors beforehand. The pinned commit prefers [Model Runner V2](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/worker/gpu/model_runner.py) for compatible configurations; unsupported features, missing Triton, some ROCm model defaults, or explicit settings can instead select the older [GPUModelRunner](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/worker/gpu_model_runner.py); the later [v0.32.0](https://github.com/vllm-project/vllm/releases/tag/v0.32.0) makes V2 the default on ROCm too and drops the per-model exceptions that kept the older runner. The SGLang path is checked against a fixed commit: [Scheduler.run_batch](https://github.com/sgl-project/sglang/blob/436d73ddda02d17d6d56e770a5362ebd5dd95ac0/python/sglang/srt/managers/scheduler.py#L4405), the [worker's batch conversion and forward call](https://github.com/sgl-project/sglang/blob/436d73ddda02d17d6d56e770a5362ebd5dd95ac0/python/sglang/srt/managers/tp_worker.py#L598), and [ModelRunner](https://github.com/sgl-project/sglang/blob/436d73ddda02d17d6d56e770a5362ebd5dd95ac0/python/sglang/srt/model_executor/model_runner.py). These are reading coordinates, not names to memorize.

Suppose request R has six prompt tokens and the first four belong to reusable complete cache blocks. A hit does not finish the request: the scheduler must validate reuse, allocate writable space for two remaining tokens, and find iteration budget. After computation, sampling produces the first new token X. **X's KV does not exist merely because X was sampled.** Feeding X into the next iteration creates its KV and predicts the next token. For a fully cached prompt, obtaining next-token logits still depends on the implementation, such as recomputing the last position. A cache is not a stored answer.

Distinguish two kinds of traffic:

- **Control messages:** request metadata, cancellation, execution plans, completion notices. API-to-engine IPC does not imply sending weights with each message.
- **Tensor transfers:** TP collectives, PP activations, and moving KV between devices or instances. These have concrete bandwidth, synchronization, and topology costs.

## What sits beneath model execution? {#runtime-stack}

| Layer | Responsibility | A useful follow-up |
| --- | --- | --- |
| Model runner | Prepare tokens, positions, cache indices, and model calls | How does a changing batch become executable tensors? Why do shapes affect execution paths? |
| Model layers and attention backend | QKV projections, attention, MLPs, KV reads/writes | How do prefill and decode shapes differ? How does GQA affect the cache? |
| Compilation, kernels, CUDA Graphs | Generate or select kernels; reduce launch and scheduling overhead | Capture constraints on shapes and buffers; fallback paths when they do not fit |
| GPU memory and communication | HBM holds weights and KV; registers / shared memory reuse local data; devices exchange tensors | Compute, HBM bandwidth, or interconnect bound? Can transfers overlap useful computation? |

PyTorch, Triton, CUDA, and vendor libraries appear in different backends; they are not a fixed series of services that every request traverses. CUDA Graphs primarily reduce CPU overhead for launching GPU work (why that overhead stands out at small batch sizes is covered in [the framework layer](../12-infra/frameworks.en.md)). They do not make KV storage free or eliminate TP communication. FlashAttention's IO optimization and paged KV management also address different layers: how attention is computed versus how cached state is allocated and addressed. See [attention kernels and cache layouts](../00-foundations/deep-dives/attention-kernels.en.md).

Continue through the pinned [SGLang ModelRunner](https://github.com/sgl-project/sglang/blob/436d73ddda02d17d6d56e770a5362ebd5dd95ac0/python/sglang/srt/model_executor/model_runner.py) for execution choices. If you have only read a backend, say so; source reading is not measured optimization experience.

## Several GPUs can mean different things {#deployment-map}

<figure class="worked-update" lang="en">
<ol>
<li><small>Split requests / Replicas</small><strong>Request A → replica 1<br>Request B → replica 2</strong><span>Complete replicas handle different requests. Capacity may grow, but weights are duplicated and a selected replica may not hold the warm prefix.</span></li>
<li><small>Split the model / TP</small><strong>One request → GPU 0 + GPU 1</strong><span>Devices compute different parts of the same layer. These are not independent requests; layer-level communication can be expensive on a weak interconnect.</span></li>
<li><small>Split stages / P–D</small><strong>Prefill worker → KV → decode worker</strong><span>Schedule different compute shapes separately, at the cost of KV transfers, routing, and two queues. Slow decode alone is not a reason to split.</span></li>
</ol>
<figcaption>These choices can combine. “Replica” here means a complete inference replica, not every MoE DP / EP topology.</figcaption>
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

This is weight accounting, not measured throughput or supported concurrency. DP8 leaves 8 GiB per GPU for both caches and runtime—not 8 GiB of guaranteed KV capacity.

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

This estimates capacity, not throughput. Hold arrivals, input/output lengths, and cache warmth fixed; measure first-token latency, output intervals, tail latency, and expert load. Matrices sharded too finely may save memory while increasing waiting.

### Where does PP keep KV, and what crosses a boundary? {#pp-kv}

Assume an ordinary decoder partitioned by layer, without cache offload or prefill/decode disaggregation. Each GPU retains KV for its own layers. Adjacent stages exchange boundary activations rather than repeatedly moving the entire historical KV cache.

“Only activations” does not necessarily mean a small transfer. With hidden size 3072 and two bytes per BF16 element:

| One boundary transfer | Element count | Raw bytes |
| --- | --- | --- |
| Eight requests decoding one token each | 8 × 3072 | 48 KiB |
| Eight requests prefilling 4096 tokens each | 8 × 4096 × 3072 | 192 MiB |

Extra residual tensors, metadata, protocol costs, and synchronization are excluded. Chunking long prefill changes scheduling and pipeline behavior; a small decode message cannot describe every workload.

[vLLM's scaling guide](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/) recommends pipeline parallelism instead of tensor parallelism for higher throughput when GPUs lack NVLink (for example, L40S). That is not proof of optimality on every PCIe machine. A single request still traverses stages sequentially; unequal stage work creates idle time. If the model fits on one GPU, compare independent replicas, quantized single-GPU serving, TP, and PP under the same workload. Disaggregated serving that transfers KV introduces a different communication path.

For a transfer lower bound, assume effective boundary bandwidth of 12 GiB/s and ignore startup latency. The 48 KiB message takes about 3.8 microseconds; the 192 MiB message takes about 15.6 milliseconds. Startup and synchronization can dominate the first, while bytes already matter in the second. This is assumed bandwidth, not a measured PCIe link or GPU specification.

PP can reduce the frequency of cross-device cooperation, but preserves sequential dependencies: with one request and one next token, later stages still wait for earlier results. More microbatches let different requests occupy different stages, provided there is enough traffic; queues and cache usage can also grow. Lack of NVLink is one input to the decision, alongside topology, prefill length, concurrency, and the latency budget.

## Refill a batch as requests finish {#batching}

Take an idealized decode schedule. A and B have completed prefill and need two and five more tokens. C is ready and needs three. Each round accommodates two requests, producing one token each; ignore every other cost.

| Rounds | Wait for the entire batch | Refill between rounds |
| --- | --- | --- |
| 1–2 | A, B | A, B |
| 3–5 | B, with one empty slot | B, C |
| 6–8 | C | Finished |

Eight rounds versus five is a scheduling result under these assumptions, not a measured speedup. Real prefill, KV capacity, and history reads matter. Continuous batching does not admit unlimited work: requests still need queues, limits, or rejection when budgets are exhausted.

<!-- widget:step-continuous-batching -->

## Two ways to organize prefix reuse {#prefix}

Both engines support prefix reuse, and both turn it on by default (vLLM's `--enable-prefix-caching`, SGLang's radix cache). “vLLM manages memory; SGLang does caching” is not a useful distinction. What still differs is the index structure, the default granularity, and the eviction policy.

| Question | vLLM V1 | SGLang radix tree (default `UnifiedRadixCache`) |
| --- | --- | --- |
| How is reusable state organized? | Prefix-dependent block hashes and a block pool, looked up by hash | A tree of shared token paths; v0.5.21 prefers a Rust tree core, with Python fallbacks for incompatible platforms or features |
| What is the reuse boundary? | Base default: 16-token blocks; the model, attention backend, and configuration can change the size. Only full blocks are reused | Page size 1 by default, so a match can stop at any token; some backends or models raise the page to 64, 128, and so on, and matches then align down to whole pages |
| How is state managed? | Reference counts, an LRU free queue for eviction, reallocation | Locking, references, node splitting; LRU by default, with LFU, SLRU, priority, and T-LRU as options |
| What cannot be inferred? | Blocks do not prove better service performance | Trees do not prove better performance for every workload |

Defaults are not universal guarantees. Check vLLM's [runner and async-scheduling conditions](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/config/vllm.py#L709), its [cache configuration](https://github.com/vllm-project/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/config/cache.py#L72), and SGLang's [tree-core fallback conditions](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/mem_cache/unified_cache/tree_core_registry.py#L44). Confirm the resolved configuration in startup logs before benchmarking.

See the pinned implementations and page-alignment example in [cache lifecycle](../00-foundations/deep-dives/attention-kernels.en.md#cache-lifecycle). The [original SGLang paper](https://arxiv.org/abs/2312.07104) explains RadixAttention's context, not the complete feature boundary of today's products.

Suppose two requests share a 2,000-token document and each appends a 100-token question. The second may avoid recomputing shared prefill. Its new answer still attends to prefix KV. Reuse saves computation; it does not remove history. Weights, adapters, token IDs, positions, and multimodal inputs must be compatible too. Visually identical text is not sufficient.

Multi-tenant reuse also needs isolation. Cache-dependent timing can reveal whether content was present; caching is not authorization. [vLLM cache salting](https://docs.vllm.ai/en/v0.31.0/design/prefix_caching/) provides one reuse-isolation mechanism, and SGLang's OpenAI-compatible API also accepts `cache_salt`. The service must still assign trusted groups rather than letting clients claim another tenant's group.

## HiCache moves beyond GPU memory {#hicache}

[SGLang HiCache](https://docs.sglang.io/docs/advanced_features/hicache_design) extends KV storage across GPU memory, host memory, and storage backends. GPU and host tiers belong to inference instances; cross-instance reuse depends on the storage backend and namespace configuration. Two machines' RAM is not automatically one shared cache. vLLM now offers the same kind of tiering: the [OffloadingConnector](https://docs.vllm.ai/en/v0.31.0/features/kv_offloading_usage/) copies finished full blocks into CPU memory and can add secondary tiers such as files or object storage, or you can plug in LMCache or Mooncake Store instead. Neither engine turns this on by default.

Consider hypothetical serial costs: recomputing a prefix takes 80 ms; restoring its KV takes 30 ms plus 5 ms for lookup and scheduling. Reuse could save about 45 ms. If transfer takes 100 ms instead, this reuse can lose. Real overlap needs a trace; this arithmetic only explains the decision.

Measure reused tokens, transferred bytes, restore time, and miss overhead alongside hit rate. Many short-prefix hits may save less work than a few long-prefix hits.

## Diagnose before adding another optimization {#bottlenecks}

| Symptom | Inspect first | Options to compare | Possible cost |
| --- | --- | --- | --- |
| Slow first token | Queueing, tokenization, prefill, cache | Chunking, prefix reuse, replicas | TTFT / ITL tradeoff; more memory |
| Uneven streaming | Decode batches, KV reads, contention | Scheduling, quantization, kernels, prefill/decode separation | Quality, transfers, operational complexity |
| Model does not fit | Weights, KV, workspace separately | Quantization, TP / PP | Communication and hardware constraints |
| Invalid JSON | Whether output constraints are enabled | Constrained decoding and validation | Valid syntax still permits false content |
| Speculative decoding is slower | Draft cost, acceptance, verification batch | Change or disable the draft method | Extra state and verification work |

As of 2026-10 (vLLM v0.31.0, SGLang v0.5.21), most mechanisms exist in both engines: paged KV management, prefix reuse on by default, continuous batching and chunked prefill, CPU scheduling overlapped with GPU execution, CUDA graphs, EAGLE / MTP / n-gram speculative decoding, structured output through xgrammar and others, TP / PP / DP + EP, prefill/decode disaggregation, and tiered KV storage. Shared features do not make the engines interchangeable: scheduling paths, cache structures, and feature compatibility still matter. The prefix index is a block-hash table in vLLM and a radix tree in SGLang, with base defaults of 16 versus 1 token; actual granularity depends on the backend and model. vLLM schedules decode tokens first and fills the rest of the same batch with prefill chunks, while SGLang by default runs a prefill batch whenever new prefill work is ready and mixes the two only with `--enable-mixed-chunk`. vLLM uses torch.compile by default and SGLang does not. SGLang builds prefill/decode roles and a router into the engine, while vLLM disaggregates through KV connectors that its docs still mark experimental.

Compare a **versioned model/hardware/feature combination**, not slogans about which engine is for agents or concurrency. FSDP handles [training state](training-frameworks.en.md); it is not another generation engine in this comparison.

## Design a meaningful comparison {#benchmark}

A single “hello” request is insufficient. Include short questions, long documents, shared-prefix conversations, bursts, and cancellations. Record input and actual output lengths. Fix weights, tokenizer, chat template, sampling, and hardware; separate cold starts from warmed runs.

| Report | Reason |
| --- | --- |
| TTFT, ITL, end-to-end p50 / p95 / p99 | Throughput hides waiting; short outputs still contribute TTFT |
| Input and output token throughput separately | They represent different computation |
| Requests/s meeting quality and latency requirements | Timeouts do not have the same value as useful completions |
| Timeouts, cancellations, rejections, OOMs, errors | Successful requests alone create survivor bias |
| GPU / CPU memory, network traffic, total cost | The bottleneck may move away from the GPU |

An open-arrival test keeps sending on schedule and exposes queue buildup. A closed-loop test waits for prior completions, so offered load can fall when the system slows. Both are useful if labeled, with offered and completed work reported. A shared seed does not guarantee identical generated text across engines, batches, or kernels; run quality and format regressions separately.

The outcome should be a reproducible record: versions, configuration, inputs, results, and failures. Continue with [serving and feedback](../practice/ml-system/serving-and-feedback.en.md) to place the engine inside a product.

## Explain the path before listing features {#interview-walkthrough}

One short opening can establish the scope:

> I would trace a request first. The application handles access, prompts, and incoming traffic. The engine owns queues, compute budgets, and KV state, and schedules each iteration. Workers execute the model and kernels, then return results for streaming. Input/output lengths, arrival rate, and latency targets tell me which bottleneck to investigate before changing batches, replicas, or model/stage partitioning.

That is an answer structure, not a script. Continue into a part you know well:

| Follow-up | Explain | Evidence to bring |
| --- | --- | --- |
| Why is a service slow despite a high prefix hit rate? | How many tokens hit, and on which replica? Queues, decode, and transfers remain after prefill reuse | Phase traces, reused tokens, cold/warm comparisons |
| What happens when KV is full? | Wait, reject, evict reclaimable blocks, preempt, or recompute; distinguish reclaimable cache from state still used by active requests | Capacity, references, preemption/recomputation, tail latency |
| Is TP=4 always faster than TP=2? | Less work per device can come with more communication or a larger communication share; small batches may underutilize hardware | Compute/collective timings, interconnect, batch and model structure |
| A client disconnected but memory did not drop—is that a leak? | Check termination and references; retaining a memory pool does not mean continuing to serve the request | Request lifecycle, active blocks versus idle pool, not process memory alone |
| How would you demonstrate an improvement? | Match versions and traffic; compare completions meeting quality and latency targets, cost, and errors | Open-arrival testing, failures, reproducible settings—not just tokens/s |

Read source by following a request object into a batch, cache indices, and execution. You do not need to read the repository in file order. For the broader preparation route, see [RE / MLE and inference interviews](../interview/ai-roles-prep.en.md#inference). This chapter checks source and teaching examples; it does not run GPU benchmarks for either framework.
