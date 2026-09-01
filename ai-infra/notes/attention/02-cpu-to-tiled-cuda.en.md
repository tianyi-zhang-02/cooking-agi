# CPU Reference to Tiled CUDA Attention

[中文](02-cpu-to-tiled-cuda.md) · **English** · [Back to project](../../projects/01-attention-from-scratch.en.md)

> Reading time: ~5 minutes · Level: Intermediate · Freshness: Evolving · Last reviewed: 2026-08

## Core problem

How can attention move onto the GPU incrementally while every failure remains localizable?

## CPU baseline

The CPU reference first materializes:

```text
scores[Nq,Nk] → probabilities[Nq,Nk] → output[Nq,Dv]
```

Optimize CPU execution one change at a time: loop order, transpose copies, tiling, and parallelism. Profile whether time belongs to QKᵀ, softmax, PV, or layout conversion. Preserve the unoptimized reference permanently.

## Staged naive CUDA

Begin with four kernel groups:

```text
QKᵀ
→ scale + mask
→ row softmax
→ PV
```

Every stage can be copied to the host and compared with the corresponding CPU intermediate. This adds launches and HBM traffic but dramatically reduces debugging difficulty.

Observe:

- how threads and blocks map onto outputs;
- whether neighboring threads read neighboring addresses;
- whether reductions synchronize correctly;
- whether small shapes are dominated by launch overhead;
- the cost of writing score and probability matrices to HBM;
- whether registers or shared memory limit resident warps.

## Tiling

One QKᵀ output tile reuses sub-blocks of Q and K. Moving tiles into shared memory can reduce HBM reads:

```text
global Q/K tile
→ shared memory
→ repeated multiply-accumulate
→ output tile
```

PV can follow a similar structure. Tile size trades reuse against shared memory, registers, occupancy, and boundary handling. No one size fits every shape.

## Use the roofline model

```text
arithmetic intensity = FLOPs / bytes moved
time ≥ max(FLOPs/compute throughput, bytes/bandwidth)
```

Tiling aims to increase reuse and arithmetic intensity. Verify bytes, bandwidth, and time with profiles; using shared memory alone does not prove success.

## Hands-on check

1. Compare CPU and GPU intermediates on tiny shapes.
2. Measure kernel time and HBM traffic for all four stages.
3. Add boundary tests for sequence and head dimensions that are not tile multiples.
4. Compare two tile sizes and record occupancy and bandwidth.
5. Explain where the bottleneck moved after optimization.

## Key conclusions

Make errors localizable before introducing fusion. A naive multi-kernel version is not a failure; it is the essential test base for high-performance versions.

Next: [Online Softmax and IO-Aware Attention](03-online-softmax.en.md)
