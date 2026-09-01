# Project 00 · 用 C/C++ 手写一个小型 Tensor Core

**中文** · [English](00-c-cpp-tensor-core.en.md) · [项目路线](README.md)

> 阅读时间：约 5 分钟 · 难度：Foundation → Intermediate · 时效性：Stable · 最近审阅：2026-08

## 项目目标

实现一个只在 CPU 上运行的小型 tensor library。第一版不使用 BLAS、Eigen 或深度学习框架，让 shape、stride、ownership、loop order 和数值稳定性全部显式可见。

这里的 “Tensor Core” 指一个个人学习用的 tensor 核心库，不是 NVIDIA Tensor Core 硬件。

## 最终能力

```text
Tensor storage and views
+ elementwise operators
+ reduction
+ matrix multiplication
+ stable softmax
+ tests and benchmarks
```

完成后，它会成为 [从零实现 Attention](01-attention-from-scratch.md) 的底层组件。

## 建议目录

```text
cpp-tensor-core/
├── CMakeLists.txt
├── include/tiny_tensor/
│   ├── tensor.h
│   └── ops.h
├── src/
│   ├── tensor.cpp
│   └── ops.cpp
├── tests/
├── bench/
├── results/
└── README.md
```

## Milestone 0 · C 语言 primitives

先用 C 写：

- vector add；
- scalar multiply；
- dot product；
- sum 和 max reduction；
- stable softmax；
- naive matrix multiplication。

接口显式传递 pointer、length、shape 和 stride。所有分配和释放都必须有测试，并在 AddressSanitizer 下运行。

成功标准：随机小输入与 Python/NumPy 或 PyTorch reference 在容差内一致，没有 sanitizer error。

## Milestone 1 · C++ Tensor ownership

实现一个 owning `Tensor`：

- 连续 row-major storage；
- shape 与 stride；
- `numel()`、`data()` 和带边界检查的 indexing；
- `const` 与 non-`const` access；
- 由 `std::vector<float>` 管理内存；
- 使用 Rule of Zero。

明确区分：

- owning tensor；
- borrowed view；
- mutable view；
- `const` view。

第一版只支持 `float`，正确后再 template 化。

## Milestone 2 · Shape、stride 与 view

加入：

- reshape：只在元素数一致时允许；
- transpose view：改变 stride，不复制数据；
- contiguous copy：把非连续 view materialize；
- slice view：保存 offset、shape 和 stride。

重点测试 view 的 lifetime：view 不能在 owner 销毁后继续访问。第一版可以通过 API 约束或共享 owner 表达安全边界，并在 README 中明确选择。

## Milestone 3 · 算子

实现：

- elementwise add/multiply；
- row-wise sum/max；
- stable row-wise softmax；
- `matmul(A, B)`；
- transpose-aware matmul；
- causal mask helper。

Softmax 必须使用：

```text
softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
```

直接计算 `exp(x_i)` 的版本只作为失败对照。

## Milestone 4 · Correctness tests

至少覆盖：

- empty dimension 和 zero-sized tensor 的既定行为；
- 不兼容 shape；
- 非连续 view；
- 极大与极小 softmax 输入；
- 非方阵 matmul；
- aliasing 输入；
- copy、move 和 object destruction；
- 随机 property tests。

比较浮点结果时使用绝对与相对容差，不要直接 `==`。

## Milestone 5 · 性能实验

比较三种 matmul：

1. `i-j-k` naive loops；
2. 改变 loop order 以改善连续访问；
3. cache-blocked / tiled matmul。

记录 shape、编译 flags、median latency、GFLOP/s 和 profile。解释为什么 loop order 会影响 cache locality。

可选扩展：

- OpenMP parallelism；
- SIMD/vectorization report；
- aligned allocation；
- BLAS 作为性能上界对照；
- `float` / `double` template；
- pybind11 binding。

## Definition of done

- Debug、sanitizer 和 release build 都可以运行；
- 所有 ownership 和 view lifetime 有明确规则；
- 算子对 reference 通过测试；
- benchmark 不包含初始化和 warm-up；
- 至少一次优化由 profiler 证据支持；
- README 解释 correctness、复杂度、layout 与瓶颈；
- library 可以直接支持 Project 01 的 CPU attention。

下一步：[Project 01 · 从零实现 Attention](01-attention-from-scratch.md)
