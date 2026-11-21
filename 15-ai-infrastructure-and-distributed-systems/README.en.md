# AI Infra: running training jobs and services reliably

[中文](README.md) · **English**

Running a model once is different from serving requests all day. Longer inputs, a traffic burst, or a failed worker can expose limits that a successful demo never encountered. This route connects resources, scheduling, distributed execution, and recovery through those changes.

It is not a second vLLM or FSDP manual. [AI infrastructure](../12-infra/README.en.md) covers hardware and runtime foundations; [AI systems](../06-systems/README.en.md) covers mechanisms and frameworks. Here we place them inside **complete services and training jobs**, then examine capacity, interfaces, and failure behavior.

## Choose a starting point {#route}

| What you want to understand | Suggested order | Something concrete to explain |
| --- | --- | --- |
| Which part of a service does what? | [Request to GPU](serving-system.en.md#stack) → [first-token and completion latency](serving-system.en.md#timeline) | Distinguish the gateway, replica routing, engine scheduling, and workers |
| Capacity and failures beyond calling a framework | [KV budget](serving-system.en.md#memory) → [queues](queues-and-failures.en.md#queue) → [failures](queues-and-failures.en.md#failure) | Change input length or lose a replica and explain the new bottleneck |
| Training is the main interest | [Training to release](serving-system.en.md#training) → [training frameworks](../06-systems/training-frameworks.en.md) → [recovery](../practice/post-training/checkpoint-and-resume.en.md) | List the state required to resume, not just the weights |
| A design discussion or a hands-on check | [Capacity exercise](queues-and-failures.en.md#exercise) → [runtime source reading](../practice/runtime-reading/README.en.md) | Trace a slow request to a queue, cache, or communication step rather than “a slow GPU” |

If generation is unfamiliar, start with [KV cache](../00-foundations/deep-dives/kv-cache-and-inference.en.md) and [prefill / decode](../practice/runtime-reading/prefill-decode.en.md). You do not need every CUDA optimization—or a rented machine—to work through these calculations.

## Three decisions that are easy to confuse {#boundaries}

| Decision | Typical responsibility | Not the same as |
| --- | --- | --- |
| Which replica receives this request? | Gateway and routing | Which tokens the engine computes in this step |
| Which requests enter the next computation? | Engine scheduler | How much capacity the cluster allocates to the service |
| When can a new replica accept traffic? | Deployment and resource control | A process starting successfully before weights load and warmup finishes |

There are several kinds of scheduling. Naming the scheduled object and the time scale matters more than memorizing a component name. Different teams may own these layers, so their interfaces need to carry deadlines, cancellation, versions, and failure reasons.

## Go deeper in the existing chapters {#deeper}

| Component | Continue with | Question to answer |
| --- | --- | --- |
| Generation engines | [vLLM / SGLang](../06-systems/inference-frameworks.en.md) | When features overlap, how do ecosystem, model support, and deployment constraints affect the choice? |
| Parallel execution | [Tensor parallelism](../06-systems/tensor-parallel.en.md) · [distributed training](../06-systems/distributed-training.en.md) | Are we splitting parameters, computation, or requests? What communication is added? |
| Training stability | [Stability investigation](../practice/runtime-reading/training-stability.en.md) | Does an abnormal loss come from the objective, precision, normalization, or data version? |
| Clusters and performance | [Kubernetes](../12-infra/kubernetes.en.md) · [performance analysis](../12-infra/performance.en.md) | Where is the wait, and would scaling remove it? |
| Product feedback | [Production ML](../14-production-ml-and-recs/README.en.md) | How do latency and fallback affect impressions, labels, and user outcomes? |

These are connected concerns. A training change may improve answer quality while doubling output length, requiring a new serving-capacity estimate. Conversely, shortening outputs for speed requires checking whether the task is still completed.

The [systems learning routes](../practice/learning-routes.en.md) show how all five areas connect. This route first establishes service and training boundaries. Optimization and control, power and cooling, and product collaboration will be expanded later. Start with whichever route addresses your current question.
