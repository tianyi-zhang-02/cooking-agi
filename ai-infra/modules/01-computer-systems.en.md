# Module 01 · Computer Systems Foundations

[中文](01-computer-systems.md) · **English** · [Back to AI Infra](../README.en.md)

## What this module solves

Before optimizing GPUs, training, or inference, it helps to understand how a program is executed by the CPU, operating system, and memory system. This module establishes the first performance question: **is time spent computing, waiting for data, or waiting for other work?**

## Learning goals

- Explain processes, threads, and a context switch.
- Describe the basic path from instruction fetch to retirement.
- Explain caches, cache lines, TLBs, virtual memory, and page faults.
- Distinguish latency, throughput, concurrency, and parallelism.
- Diagnose whether a simple program is compute-bound or memory-bound.

## Core notes

### From a program to a running process

Source code is compiled or interpreted into executable instructions. The operating system creates a process, gives it a virtual address space, and maps its code, heap, stacks, and libraries onto physical memory.

A thread is a schedulable execution stream. Threads in one process share an address space but have their own register state and stacks. Switching threads requires saving and restoring execution state and may disrupt cache locality.

### How a CPU executes instructions

A simplified out-of-order CPU pipeline is:

```text
Fetch → Decode → Rename → Dispatch → Execute → Retire
```

- The branch predictor guesses where execution continues.
- Register renaming removes false register dependencies.
- The scheduler issues instructions whose inputs are ready.
- A reorder buffer retires results in program order.
- Work produced along a wrong speculative path is discarded.

A modern CPU improves performance by exploiting instruction-level, data-level, and thread-level parallelism at the same time.

### The memory hierarchy

```text
Registers → L1 → L2 → L3 → DRAM → SSD / remote storage
```

Storage closer to a core is smaller and more expensive but has lower latency. Sequential access benefits from spatial locality; repeatedly using recent data benefits from temporal locality. Random accesses across cache lines are usually much more expensive than sequential scans.

Virtual addresses are translated through page tables. A TLB caches recent translations. TLB misses, page faults, and remote NUMA accesses add different forms of delay.

### Concurrency, parallelism, and throughput

- **Concurrency:** multiple tasks make progress over overlapping periods.
- **Parallelism:** multiple tasks execute at the same instant.
- **Latency:** time required to finish one task.
- **Throughput:** tasks completed per unit time.

More threads may improve throughput without reducing request latency. Too many threads can instead add scheduling, synchronization, and cache contention.

## Quantities to calculate

A basic CPU-time approximation is:

```text
CPU time ≈ instruction count × cycles per instruction / clock frequency
```

A basic memory-throughput lower bound is:

```text
time ≥ bytes moved / effective bandwidth
```

These are not complete performance models, but they reveal useful physical limits.

## Hands-on work

1. Compare sequential and random access over the same large array.
2. Change the working-set size and observe transitions across cache levels.
3. Profile operator time and thread activity in a PyTorch CPU workload.
4. Run the same work with one thread and multiple threads; record latency and throughput.
5. Write a one-page diagnosis: is the bottleneck instructions, memory, synchronization, or I/O?

## Common misconceptions

- 100% CPU utilization does not mean every execution unit is useful.
- More threads do not guarantee more speed.
- Algorithms with the same Big-O complexity can behave very differently because of locality.
- Sufficient memory capacity does not imply sufficient bandwidth or low latency.
- A benchmark that includes initialization, disk reads, or warm-up can support the wrong conclusion.

## Mastery check

- Why is a sequential scan usually faster than random access?
- Why does a branch misprediction waste cycles?
- What do processes and threads share or not share?
- How do a cache miss, TLB miss, and page fault differ?
- Why can latency improve while throughput becomes worse?

Next: [Module 02 · GPU Programming](02-gpu-programming.en.md)
