# Module 06 · GPU Clusters and Platform Engineering

[中文](06-gpu-platforms.md) · **English** · [Back to AI Infra](../README.en.md)

> Reading time: ~5 minutes · Level: Intermediate · Freshness: Evolving · Last reviewed: 2026-08

## What this module solves

One script running successfully is only the beginning. Platform engineering makes workloads from different users, models, and priorities reproducible, observable, and recoverable on a shared GPU cluster while controlling waste and limiting failures.

## Learning goals

- Understand the node software stack from drivers and containers to training or serving processes.
- Explain Kubernetes device plugins or Slurm GPU scheduling.
- Understand queues, quotas, priorities, preemption, and gang scheduling.
- Design a job lifecycle, checkpoint policy, and retry policy.
- Define metrics, SLOs, and alerts for a GPU platform.

## Core notes

### The stack on a GPU node

```text
GPU firmware / hardware
→ host driver
→ CUDA runtime and libraries
→ container runtime and device exposure
→ framework
→ training or serving process
```

Compatibility can fail between any adjacent layers. Containers package user-space dependencies but do not replace the need for a compatible host driver.

### Resource discovery and scheduling

A scheduler needs to know available GPUs, memory capacity, health, topology, and a workload's device count. Multi-GPU jobs often require gang scheduling: all workers receive resources together so that partial allocations do not hold GPUs while waiting for the rest.

Topology-aware scheduling places frequently communicating GPUs on faster links and considers relationships among CPU affinity, NUMA domains, NICs, and GPUs.

### Sharing and isolation

Options include exclusive GPUs, MIG, time slicing, and application-level multi-tenancy. Sharing can improve utilization but adds performance interference, memory-isolation, failure-domain, and security complexity.

The platform must define which resources can be oversubscribed, what can preempt what, how interactive services are isolated from batch training, and what is protected during overload.

### Job lifecycle

```text
submit → validate → queue → schedule → initialize
→ run → checkpoint → finish / retry / fail → retain artifacts
```

Reliable retry distinguishes transient infrastructure failures, deterministic code errors, data errors, and OOM. Unconditional retry may only repeat wasted GPU time.

### Observability

At least three signal types are needed:

- **Metrics:** GPU utilization, memory, power, temperature, networking, queue time, and failure rates.
- **Logs:** scheduler, node, container, framework, and user-process logs.
- **Traces/timelines:** computation, communication, I/O, and waiting across workers.

Observability should let an operator drill from “the job is slow” to a particular node, rank, kernel, collective, or storage request.

## Quantities to calculate

Cluster utilization should distinguish:

```text
allocation utilization = allocated GPU time / available GPU time
active utilization = useful busy time / allocated GPU time
```

Allocation utilization alone counts jobs that reserve GPUs without performing useful work.

Capacity planning should also measure:

- queue-time distributions;
- demand and fragmentation by GPU type;
- GPU-hours consumed by failures and retries;
- checkpoint and dataset bandwidth;
- service peaks and safety margins.

## Hands-on work

1. Draw a state machine for a training job from submission to completion.
2. Design resource requests for one-GPU, eight-GPU, and multi-node jobs.
3. Define different retry policies for OOM, node loss, network timeout, and code failure.
4. Write SLOs and dashboard metrics for training and online inference separately.
5. Plan GPU-node maintenance: draining, checkpointing, migration, and recovery.

## Common misconceptions

- An allocated GPU is not necessarily doing useful work.
- A runnable container does not prove full driver, CUDA, and framework compatibility.
- Automatic retry is not a substitute for fault-tolerance design.
- Optimizing mean utilization may harm P99 latency or high-priority workloads.
- Without unified job identity and version metadata, abundant logs remain difficult to correlate.

## Mastery check

- Why do distributed jobs need gang scheduling?
- What does a device plugin solve, and what does it not solve?
- When is GPU sharing appropriate?
- How do allocated, busy, and useful GPU time differ?
- How can a platform make a training run exactly reproducible?

Next: [Module 07 · Data, Evaluation, and Learning Loops](07-data-eval-learning-loop.en.md)
