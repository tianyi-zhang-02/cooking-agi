# From workers to GPUs: execution and scheduling

[中文](execution-boundaries.md) · **English**

“Why is work waiting while the GPU is idle?” The job may lack an assigned device, the model may await data, requests may lack KV-cache space, or a training process may be waiting for its communication peers. GPU utilization alone cannot distinguish these cases. Follow one computation to the device, then trace requests and jobs.

On a first read, start with the runtime section; return to cluster scheduling after learning distributed-training basics. The example topologies, jobs and traffic illustrate how the layers interact, rather than report a cluster deployment or GPU benchmark. Sources were checked on 2026-10-30.

## Runtime: how Python becomes GPU work {#runtime}

### Separate the CPU, GPU and device memory

A training program starts as a process on the CPU. It reads configuration, prepares data and sends suitable computation to the GPU. The GPU has execution units and device memory holding parameters, activations and other tensors. For the discrete GPUs considered here, treat VRAM as its working space.

A tensor's `device` tells you where it resides. Installing a GPU does not automatically turn a CPU token array into usable GPU input. The program and framework must manage placement, transfers and dependencies.

### Follow one operator downward

Suppose Python requests a matrix multiplication. The framework chooses an implementation for the input shapes and dtypes, then submits work to the device. A kernel is a program that performs computation on the device:

```text
CPU process prepares tensors and dependencies
  → PyTorch / compiler / operator library selects or generates an implementation
  → CUDA submits kernels, copies and other work
  → GPU executes, reading and writing device memory
  → dependencies or synchronization establish completion when results are needed
```

The arrows describe execution responsibilities. One process can handle several steps, and operators may take different library paths.

### Submission, execution and waiting are different moments

When a CPU call returns, GPU work may be running or may not have started. The program can prepare other data meanwhile. Dependencies or synchronization establish completion when a later operation needs the result.

Think of a CUDA stream as an ordered queue of submitted work. Operations within it obey the relevant ordering; different streams need coordination where data dependencies require it. Streaming multiprocessors (SMs) are compute units within the GPU. A stream does not permanently own a group of SMs. See the [CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html) and [asynchronous execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html).

Suppose `inputs=[[1,2],[3,4]]` and `weights=[[1],[2]]` are already tensors on the same GPU. Submit multiplication and a subsequent sum to one stream; the multiplication has shapes `[2,2] @ [2,1] → [2,1]`:

| Moment | What the CPU does | GPU results and dependencies |
| --- | --- | --- |
| Submit multiplication | Call `outputs = inputs @ weights`, then continue | Produces `[[5],[11]]` when complete; the call may return earlier |
| Submit sum | Call `total = outputs.sum()` without first waiting for multiplication to finish | The sum follows the multiplication and produces 16 |
| Use the result | Read 16 on the CPU | Wait for the relevant computation and transfer back before using the value |

A CPU timer around an async call may mostly measure submission. Use CUDA events or suitable synchronization for device timing, according to the experiment. Whole-service timing also includes queueing, inputs and outputs. State whether warmup, compilation and graph capture count. [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) reduce repeated submission overhead without removing memory, communication or arithmetic costs.

Continue with [CUDA: threads, memory and kernels](../12-infra/cuda.en.md) and [frameworks](../12-infra/frameworks.en.md).

## Scheduling has at least four scales {#scheduling}

### Begin with the unit each layer schedules

The cluster assigns devices to workloads, the distributed runtime places worker processes, the inference engine chooses this iteration's requests, and the device executes submitted computation. Here, “distributed runtime” concerns processes and tasks; the previous section followed the execution of an individual operator. Each layer has different information and makes different decisions.

| Scale | Unit scheduled | Resources and constraints | Example wait |
| --- | --- | --- | --- |
| Cluster workload | Pods, Jobs, distributed jobs | Allocatable CPU/GPU, memory, quotas, nodes and topology | Eight eligible GPUs requested, only four available |
| Distributed runtime | Ray tasks / actors and worker work | Logical resources, placement groups, process lifecycle | Resource bundles cannot be placed together |
| Inference engine | Requests, sequences, tokens for this iteration | Token budget, KV capacity, prefill/decode state | New requests queue while existing requests decode |
| Device execution | Thread blocks and other submitted kernel work | SMs, registers, shared memory, dependencies and capacity | Dependencies or resources prevent execution/overlap |

### Cluster: assign devices to workloads

In Kubernetes, a Pod is a basic unit for running containers. A [Job](https://kubernetes.io/docs/concepts/workloads/controllers/job/) commonly runs work that finishes; a [Deployment](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/) maintains service replicas. Slurm organizes computation through jobs and resource allocations. First identify required devices, processes and completion conditions, then choose the deployment tool.

### Runtime and engine: use resources for the current work

A Ray task is a remote function call. An [actor](https://docs.ray.io/en/latest/ray-core/actors.html) is a worker that retains state: for example, a process that keeps a model loaded while handling successive requests. Placement groups express placement requirements for sets of resource bundles. These concern where workers run.

A vLLM/SGLang scheduler also needs to know how far each request has progressed: prefill processes the input prompt, while decode generates subsequent tokens step by step. The engine uses this progress, KV-cache capacity and the iteration's token budget to form executable batches.

For example, a worker with a GPU and loaded model can still have request R waiting for KV space. Allocation and startup are complete; R's computation is submitted only after the engine selects it for a batch. Inspect the engine queue and KV usage to explain this wait.

This is why adding a service replica differs from increasing an engine's batch budget. One adds serving resources; the other changes how existing resources are used, affecting queueing, memory and per-request latency.

### Device: execute submitted work

Hardware and driver mechanisms implement device scheduling; CUDA programs express work and dependencies. This is separate from Kubernetes placing a Pod. Ray resource declarations are also not complete VRAM enforcement or physical isolation; see its [resource documentation](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html).

Kubernetes, Slurm and Ray are not three mandatory serial layers. Slurm can launch training directly; Ray can run on Kubernetes, with KubeRay managing Ray cluster workloads. See [Ray on Kubernetes](https://docs.ray.io/en/latest/cluster/kubernetes/index.html) and the existing [Kubernetes note](../12-infra/kubernetes.en.md).

## Eight GPUs: how are they connected? {#topology}

### Map process groups to physical machines

Suppose two machines each have four GPUs. Compare placing a tensor-parallel group (TP=4) within a fast-connected node versus across nodes, while checking actual NVLink/PCIe topology rather than assuming locality from node names. Network links, network interface cards (NICs), NUMA memory layout and storage also affect communication and input supply.

A two-GPU TP request differs from two TP=1 replicas handling separate requests. In training, TP, pipeline-parallel (PP) and data-parallel (DP) groups determine communication frequency and tensor types. Draw the [eight-rank groups](megatron-basics.en.md#eight-gpus), then map them to nodes. Kubernetes does not infer the model's optimal parallel layout.

### Enough GPUs on paper can still leave a job waiting

Suppose an eight-GPU training job needs every process to participate, with each process identified by a rank within its communication group. With only four devices available, starting half the ranks does not establish that training can progress. Group resource requests and admission mechanisms address such requirements, with behavior depending on the runtime and scheduling configuration.

Also check device types, node constraints, quotas and topology. Eight idle GPUs do not necessarily satisfy one job's requirements.

A device plugin makes GPUs requestable. Assignment does not establish that a model fits, NCCL connects or throughput meets requirements. See [Kubernetes GPU scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/). MIG, time-slicing and MPS have different sharing/isolation semantics; they do not simply divide arbitrary portions of 80 GB VRAM.

## Which waits does NCCL address? {#communication}

Even after acquiring devices, processes can wait for data from their peers. NCCL is a widely used GPU communication library. The collective operations below describe how participants exchange or combine tensors.

| Primitive | Meaning | Example use |
| --- | --- | --- |
| All-reduce | Reduce participant arrays and return results to all | Synchronize corresponding gradients |
| All-gather | Collect partitions into a result available to participants | Obtain a full parameter group when needed |
| Reduce-scatter | Reduce, then distribute partitions of the result | Retain owned gradient shards |
| All-to-all | Each participant sends different portions to others | Some MoE dispatch / combine paths |

### Inspect the results on two small arrays

Let rank 0 hold `[1,2]` and rank 1 hold `[3,4]`. Each row starts again from these original inputs:

| Operation | Calculation | Final data on rank 0 | Final data on rank 1 |
| --- | --- | --- | --- |
| Sum all-reduce | Add corresponding elements: `[1+3,2+4]=[4,6]` | `[4,6]` | `[4,6]` |
| All-gather | Concatenate in rank order without adding | `[1,2,3,4]` | `[1,2,3,4]` |
| Sum reduce-scatter | Form `[4,6]`, then partition it equally between ranks | `[4]` | `[6]` |

For the mean of two equally weighted gradients, divide the sum `[4,6]` by 2 to obtain `[2,3]`. The sum specified here does not perform that normalization automatically.

These primitives specify who receives what. The training framework decides when to call them, which parameters or gradients they carry, and whether communication can overlap computation.

### Participants need compatible contracts

The mapping depends on the training strategy. NCCL participants must agree on collective ordering, element counts, dtypes and other requirements. For example, rank 1 might exit after an OOM during backward, while rank 0 subsequently waits for a gradient all-reduce. The visible symptom may be a communication timeout on rank 0. When investigating a hang, find the earliest error across all ranks, not just the last device to time out. The [official collective documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) defines these semantics.

## Put failures back in their layer {#debugging}

| Symptom | First evidence to inspect | Does not establish |
| --- | --- | --- |
| Pod Pending | Scheduling events, quotas, requests, node conditions | A CUDA kernel bug |
| Pod Running but model unready | Worker startup, loading, warmup, health | Representative requests can already be served |
| Idle GPU with queued requests | CPU tokenization, data/environment time, queues, engine budgets | Insufficient GPU compute |
| CUDA OOM | Simultaneous parameters, activations, KV, buffers, allocator state | More DP ranks alone will solve it |
| NCCL timeout | First error per rank, collective order, topology | Necessarily a network fault |
| Completed generation but client timeout | Deadlines, cancellation, streaming, proxy logs | Model computation did not finish |

Pod recreation, Ray actor restart and training recovery are separate. Restarted processes may need model/optimizer state, random-number-generator state and data position. Reinforcement learning (RL) may also require unconsumed sampled trajectories (rollouts), policy versions and acknowledgement boundaries: which results have been confirmed as consumed. A platform can restart a process without reconstructing state the application never saved.

## Decide whether an optimization is worthwhile {#measurement}

Trace CPU, GPU, communication and queues to find the critical path, then change one variable. For inference batch budgets, hold model, input/output length distribution, precision, arrival rate and GPU topology constant. Report throughput, p50/p95/p99 latency, failures and cancellations, not just average tokens/s.

For training, align effective batch, loss normalization, update counts and target quality. A single-device tensor comparison does not establish multi-node fault tolerance; a tiny model's memory savings do not replace throughput measurement of the full training setup. Begin with local shape and event-order checks, then budget explicit real-device experiments. Locate the wait—resource allocation, request batching, device execution or communication—before choosing the next log.

Continue: [Megatron and distributed basics](megatron-basics.en.md), [inference engines](inference-frameworks.en.md), [RL systems](rl-systems.en.md).
