# Module 00 · 面向 AI Infra 的 C/C++ 基础

**中文** · [English](00-c-cpp-foundations.en.md) · [返回索引](../README.md)

> 阅读时间：约 5 分钟 · 难度：Foundation · 时效性：Stable · 最近审阅：2026-08

## 为什么需要它

CUDA、PyTorch runtime、NCCL、推理引擎和高性能算子大量使用 C/C++。目标不是背完整语言，而是能回答四个问题：

```text
数据放在哪里？
谁拥有它？
它能活多久？
编译器最终执行了什么？
```

这些问题不清楚时，常见结果是越界、use-after-free、错误 stride、data race，或者一个看似漂亮但完全没有性能的 abstraction。

## 最小知识地图

### C：直接面对内存

需要掌握：

- 基本类型、数组和 `struct`；
- pointer、address、dereference 和 pointer arithmetic；
- `const`、长度、shape 与 stride；
- `malloc/calloc/realloc/free`；
- header、source file、translation unit 和 linking；
- undefined behavior 与 sanitizer。

C 指针不携带长度和 ownership。一个 `float*` 只表示某个地址可能有 `float`，无法说明有多少元素、能否修改或由谁释放。

→ [五分钟：Pointer、数组与连续内存](../notes/c-cpp/01-pointers-and-memory.md)

### C++：显式表达 ownership

需要掌握：

- reference 与 `const`；
- `std::vector`、`std::array` 和 `std::span`；
- class、constructor 和 invariant；
- RAII、`unique_ptr` 与 Rule of Zero；
- copy、move、borrow 和 object lifetime；
- function template 与 lambda。

现代 C++ 的价值不是“面向对象”，而是用可优化的 abstraction 表达资源生命周期。默认优先标准容器和 RAII，只有在 ownership 完全清楚时才管理 raw resource。

→ [五分钟：RAII、copy、move 与 view](../notes/c-cpp/02-ownership-and-raii.md)

### Build、debug 与性能证据

需要熟悉：

```text
source → preprocess → compile → object → link → executable
```

学习项目至少保留三种配置：

```text
debug:   -O0 -g -Wall -Wextra -Wpedantic
checked: -O1 -g -fsanitize=address,undefined
release: -O3 -DNDEBUG -march=native
```

Warning、sanitizer、test、debugger 和 profiler 解决不同问题。Release benchmark 不能替代 checked build；代码运行一次也不能证明没有 undefined behavior。

→ [五分钟：编译、Sanitizer 与 Benchmark](../notes/c-cpp/03-build-debug-benchmark.md)

## 最小实战路径

1. 用 C 写 vector add、dot、stable softmax 和 naive matmul。
2. 为 pointer API 显式传递 length、shape 和 stride。
3. 用 AddressSanitizer 找出故意加入的越界和 use-after-free。
4. 用 `std::vector` 和 RAII 重写一个 C++ `Tensor`。
5. 比较不同 loop order 的 matmul，并用 profiler 解释差异。
6. 把实现拆成 library、tests 和 benchmark targets。

完整规格：[Project 00 · C/C++ Tensor Core](../projects/00-c-cpp-tensor-core.md)

## 掌握检查

- 数组退化为指针后丢失了什么？
- raw pointer 是否表示 ownership？
- stack、heap 和 object lifetime 有什么关系？
- 为什么 RAII 能覆盖异常和提前返回？
- debug、sanitizer 和 release build 分别证明什么？
- 为什么编译通过不代表程序有定义良好的行为？

下一篇：[Module 01 · 计算机系统](01-computer-systems.md)
