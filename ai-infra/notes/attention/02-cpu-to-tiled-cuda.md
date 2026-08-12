# 从 CPU Reference 到 Tiled CUDA Attention

**中文** · [English](02-cpu-to-tiled-cuda.en.md) · [返回项目](../../projects/01-attention-from-scratch.md)

> 阅读时间：约 5 分钟 · 难度：Intermediate · 时效性：Evolving · 最近审阅：2026-08

## 只解决一个问题

怎样逐步把 attention 搬到 GPU，同时让每次错误仍然能定位？

## CPU baseline

CPU reference 先 materialize：

```text
scores[Nq,Nk] → probabilities[Nq,Nk] → output[Nq,Dv]
```

优化 CPU 时依次改变：loop order、transpose copy、tiling、parallelism。用 profiler 判断时间在 QKᵀ、softmax、PV 还是 layout conversion。始终保留未优化 reference。

## Naive CUDA 分阶段

先写四组 kernels：

```text
QKᵀ
→ scale + mask
→ row softmax
→ PV
```

每个 stage 都能 copy 回 host，与 CPU 中间值对比。这增加了 launch 和 HBM traffic，却极大降低 debug 难度。

需要观察：

- thread/block 如何映射 output；
- 相邻 thread 是否读取相邻地址；
- reduction 是否正确同步；
- 小 shape 是否主要被 launch overhead 控制；
- score/probability matrix 写回 HBM 的代价；
- register 和 shared memory 是否限制 resident warps。

## Tiling

QKᵀ 的一个 output tile 会重复使用 Q 与 K 的子块。把 tile 搬到 shared memory 可以减少 HBM 读取：

```text
global Q/K tile
→ shared memory
→ repeated multiply-accumulate
→ output tile
```

PV 也可以用类似方式处理。Tile size 影响复用、shared memory、register、occupancy 和边界处理，没有一个尺寸适合所有 shape。

## 用 roofline 判断

```text
arithmetic intensity = FLOPs / bytes moved
time ≥ max(FLOPs/compute throughput, bytes/bandwidth)
```

Tiling 的目标是增加复用和 arithmetic intensity。必须用 profile 验证 bytes、bandwidth 和 time，而不是因为使用 shared memory 就宣称成功。

## 动手检查

1. 对很小 shape 逐阶段比较 CPU/GPU 中间值。
2. 测量四阶段的 kernel time 和 HBM traffic。
3. 加入 sequence/head dimension 非 tile 整数倍的边界测试。
4. 比较两个 tile size，记录 occupancy 与 bandwidth。
5. 解释优化后瓶颈转移到了哪里。

## 记住

先让错误可以局部化，再做融合。Naive multi-kernel 版本不是失败，而是高性能版本不可缺少的测试基座。

下一篇：[Online softmax 与 IO-aware attention](03-online-softmax.md)
