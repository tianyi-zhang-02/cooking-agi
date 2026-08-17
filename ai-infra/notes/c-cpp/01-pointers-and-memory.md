# Pointer、数组与连续内存

**中文** · [English](01-pointers-and-memory.en.md) · [返回 C/C++ 模块](../../modules/00-c-cpp-foundations.md)

> 阅读时间：约 5 分钟 · 难度：Foundation · 时效性：Stable · 最近审阅：2026-08

## 核心问题

一个 pointer 到底告诉了程序什么，又没有告诉什么？

```c
float x = 3.0f;
float *p = &x;
*p = 4.0f;
```

`x` 是对象，`&x` 是地址，`p` 保存地址，`*p` 访问该地址的对象。Pointer 本身通常不包含长度、shape、容量、ownership 或 lifetime。

## 数组与 pointer API

数组传给函数时经常退化为 pointer，因此长度必须显式传递：

```c
void saxpy(float *y, const float *x, float a, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}
```

`const float *x` 表示不能通过 `x` 修改输入。它不表示其他代码不能修改同一内存。

二维 row-major 矩阵常保存为一段连续内存：

```text
offset(row, col) = row × columns + col
```

更一般的 tensor 需要 shape、stride 和 offset。Transpose view 可以只交换 shape/stride，而不复制数据；这也意味着“逻辑连续”和“物理连续”不是一回事。

## 手动分配

- `malloc`：分配未初始化 raw memory；
- `calloc`：分配并清零；
- `realloc`：可能移动分配，旧 pointer 可能失效；
- `free`：结束分配的 lifetime。

每次分配都需要唯一、明确的释放责任。常见错误：leak、double free、use-after-free、out-of-bounds 和未初始化读取。

## Aliasing

两个 pointer 可能指向重叠区域。编译器必须保守处理可能 alias 的访问。C 的 `restrict` 可以承诺特定 pointer 在作用域内不重叠，从而帮助优化；承诺错误会产生 undefined behavior。

## 动手验证

1. 写一个 `Matrix {float* data; rows; cols; stride;}`。
2. 实现 row-major indexing、transpose view 和 contiguous copy。
3. 故意写一次越界访问，用 AddressSanitizer 捕获。
4. 让两个 pointer alias，观察 in-place operator 是否仍正确。

## 记住

Pointer 是地址，不是 tensor。安全、高性能的 tensor API 还必须表达长度、layout、mutability、ownership 和 lifetime。

下一篇：[RAII、copy、move 与 view](02-ownership-and-raii.md)
