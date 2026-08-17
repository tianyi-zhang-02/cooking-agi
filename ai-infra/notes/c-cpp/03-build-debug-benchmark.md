# 编译、Sanitizer 与 Benchmark

**中文** · [English](03-build-debug-benchmark.en.md) · [返回 C/C++ 模块](../../modules/00-c-cpp-foundations.md)

> 阅读时间：约 5 分钟 · 难度：Foundation · 时效性：Stable · 最近审阅：2026-08

## 核心问题

代码编译并运行了，为什么仍然不能证明它正确或快速？

```text
source → preprocess → compile → object file → link → executable
```

每个 `.c/.cpp` 是 translation unit。Header 提供声明和 template 定义，linker 解析跨文件符号与 library。Compile error、link error、runtime error 和 performance regression 是不同类别。

## 三种 build

### Debug

```text
-O0 -g -Wall -Wextra -Wpedantic
```

适合断点、变量和容易阅读的 stack。不代表 release 行为或性能。

### Checked

```text
-O1 -g -fsanitize=address,undefined
```

AddressSanitizer 检查许多越界、use-after-free 等内存错误；UndefinedBehaviorSanitizer 检查部分 UB。多线程 data race 需要 ThreadSanitizer 的独立构建。

### Release

```text
-O3 -DNDEBUG -march=native
```

用于性能测试。优化会内联、重排、vectorize，甚至删除没有可观察结果的计算。Release 中的错误可能与 debug 完全不同。

## Benchmark 最小纪律

1. 先用 reference 和 tests 证明正确。
2. 把初始化、I/O 和 warm-up 与 steady state 分开。
3. 使用计算结果，防止 dead-code elimination。
4. 固定输入分布、随机种子、线程和 affinity。
5. 报告 median 和分布，不只记录最好一次。
6. 保存 CPU、compiler、flags、library 和 commit。
7. 用 profiler 解释变化。

一个 benchmark 数字只有在 measurement contract 清楚时才有意义。

## Debugger 与 profiler

- `gdb/lldb`：断点、call stack、memory 和变量；
- sanitizer：发现特定 correctness violation；
- `perf` / Instruments：CPU sample、cache 与热点；
- compiler report：vectorization 和 inlining；
- Nsight：CUDA timeline 与 kernel metrics。

Profiler 不直接告诉你修复方案。它提供“时间和资源在哪里”的证据，再由系统模型解释原因。

## 动手验证

1. 建立 CMake 的 debug、checked 和 release targets。
2. 故意加入越界和 signed overflow，观察不同工具报告。
3. benchmark 三种 matmul loop order。
4. 检查编译器是否 vectorize inner loop。
5. 保存一张实验表，解释最快版本为什么更快。

## 记住

Test 证明已覆盖输入下的行为，sanitizer 暴露特定错误，profiler 描述资源使用，benchmark 比较受控 workload。四者互相补充，不能替代。

下一步：[Project 00 · C/C++ Tensor Core](../../projects/00-c-cpp-tensor-core.md)
