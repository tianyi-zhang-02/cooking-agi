# Module 00 · 面向 AI Infra 的 C/C++ 基础

**中文** · [English](00-c-cpp-foundations.en.md) · [返回 AI Infra](../README.md)

## 这一模块解决什么

CUDA、PyTorch runtime、通信库和推理引擎的大量底层代码都建立在 C/C++ 上。本模块不追求背完语言，而是掌握实现 tensor、算子和系统组件真正需要的部分：**内存在哪里、谁拥有它、它能活多久，以及编译器最终看到了什么。**

## 学习目标

- 能读写基本的 C 和现代 C++；
- 理解数组、指针、引用、`const` 和 pointer arithmetic；
- 理解 stack、heap、object lifetime、ownership 与 RAII；
- 会使用 `struct`、`class`、constructor、destructor、copy 和 move；
- 会使用 `std::vector`、`std::array`、`std::span`、template 和 lambda；
- 能建立多文件项目，并使用 warning、sanitizer、debugger 和 profiler。

## 先理解编译过程

```text
source files
→ preprocessing
→ compilation
→ assembly
→ object files
→ linking
→ executable or shared library
```

Header 通常保存声明和 template 定义，source file 保存普通函数定义。每个 `.c` 或 `.cpp` 是一个 translation unit。Linker 负责把跨文件符号和库连接起来；声明不一致、重复定义和缺失符号往往在这个阶段暴露。

编译优化等级会改变性能和可调试性。学习阶段建议同时保留两种构建：

```text
debug:   -O0 -g -Wall -Wextra -Wpedantic
checked: -O1 -g -fsanitize=address,undefined
release: -O3 -DNDEBUG -march=native
```

## C：最重要的语法和心智模型

### 值、地址和指针

```c
float x = 3.0f;
float *p = &x;
*p = 4.0f;
```

- `x` 是一个值；
- `&x` 是它的地址；
- `p` 保存这个地址；
- `*p` 访问地址指向的对象。

指针本身不携带数组长度。API 必须通过额外参数或结构体保存 shape 和容量。

```c
void saxpy(float *y, const float *x, float a, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}
```

`const float *x` 表示函数不通过这个指针修改输入。`restrict` 可以在满足无别名约束时帮助 C 编译器优化，但违反约束会造成 undefined behavior。

### 数组和连续内存

二维矩阵常以一维连续数组保存：

```text
row-major offset = row × number_of_columns + column
```

这种 layout 使 shape、stride 和访问顺序显式化，也是理解 tensor layout 的基础。

### `struct` 与函数指针

C 使用 `struct` 把数据组织起来：

```c
typedef struct {
    float *data;
    size_t rows;
    size_t cols;
} Matrix;
```

函数指针常用于 callback、kernel dispatch 或策略表。需要先掌握普通函数和指针，再学习复杂声明。

### 手动内存管理

`malloc` 返回未初始化的 raw memory，`calloc` 会清零，`realloc` 可能移动分配，`free` 结束 ownership。每个成功分配都必须有唯一、清晰的释放责任。

常见错误包括 leak、double free、use-after-free、out-of-bounds 和未初始化读取。它们可能暂时“正常运行”，因此必须使用 sanitizer 检查。

## C++：在零成本抽象中表达 ownership

### 引用与 `const`

```cpp
void normalize(std::span<float> values);
float dot(std::span<const float> a, std::span<const float> b);
```

引用和 `span` 可以表达“借用但不拥有”。`const` 应尽量描述不会发生的修改，让接口更容易推理。

### RAII

RAII 把资源生命周期绑定到对象生命周期。`std::vector<float>` 会在对象销毁时自动释放内存；`std::unique_ptr` 表达唯一 ownership；`std::shared_ptr` 只应在确实需要共享生命周期时使用。

```cpp
class Matrix {
public:
    Matrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), data_(rows * cols) {}

    float& operator()(std::size_t r, std::size_t c) {
        return data_.at(r * cols_ + c);
    }

    const float& operator()(std::size_t r, std::size_t c) const {
        return data_.at(r * cols_ + c);
    }

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<float> data_;
};
```

这里 `Matrix` 拥有连续数据，constructor 建立 invariant，destructor 由成员自动完成。学习阶段优先使用标准容器，等 ownership 完全清楚后再写 raw allocator。

### Copy、move 与 Rule of Zero

Copy 产生独立对象，move 转移资源。由 `vector`、`string`、`unique_ptr` 等 RAII 类型组成的 class 通常不需要手写 destructor、copy 或 move，这叫 Rule of Zero。

只有管理 raw resource 时才需要认真实现 Rule of Five。错误的 shallow copy 很容易造成 double free。

### Template 和泛型代码

Template 让一个算法支持不同 dtype 或 layout：

```cpp
template <typename T>
T dot(std::span<const T> a, std::span<const T> b);
```

AI kernel 代码大量使用 template 在编译期选择 dtype、tile size、layout 和硬件路径。先理解简单 function template，再进入 metaprogramming。

### Lambda

Lambda 是可以捕获上下文的匿名函数，常用于算法、线程任务和 callback。必须注意按引用捕获的对象不能比 lambda 更早销毁。

## Undefined behavior 与安全边界

C/C++ 允许接近硬件，也把正确性责任交给程序员。常见 undefined behavior：

- 越界读写；
- 访问已经销毁的对象；
- signed integer overflow；
- 错误的类型转换或 alignment；
- 多线程 data race；
- 违反 `restrict` 或 lifetime 规则。

Undefined behavior 不是“返回随机值”，而是编译器不再需要维护任何语义保证。Debug、sanitizer、assertion 和测试应该成为日常工作流。

## 构建与调试工具

- `clang` / `gcc`：C 编译器；
- `clang++` / `g++`：C++ 编译器；
- CMake：描述 targets、依赖和 build configuration；
- AddressSanitizer：越界、use-after-free 等内存错误；
- UndefinedBehaviorSanitizer：部分 UB；
- ThreadSanitizer：data race；
- `gdb` / `lldb`：断点、栈和变量；
- `perf`、Instruments 或 profiler：性能证据。

## 动手练习

1. 用 C 实现 vector add、dot product、softmax 和 matrix multiply。
2. 用 `malloc/free` 写一个 `Matrix`，并用 sanitizer 找出故意加入的 leak 和越界。
3. 用 C++ `std::vector` 和 RAII 重写 `Matrix`。
4. 为 `float` 和 `double` 写 templated vector operations。
5. 用 debug 与 release 配置比较 correctness 和性能。
6. 把代码拆为 library、tests 和 benchmark 三个 targets。

详细项目：[Project 00 · C/C++ Tensor Core](../projects/00-c-cpp-tensor-core.md)

## 常见误区

- 会写 Python extension 不等于理解对象 ownership；
- raw pointer 不自动代表 ownership；
- `new/delete` 不是现代 C++ 的默认选择；
- 编译通过不代表没有 undefined behavior；
- microbenchmark 如果被编译器完全优化掉，会得到虚假的极快结果；
- C++ abstraction 不一定慢，关键是它是否能被内联和优化。

## 掌握检查

- 数组退化为指针后丢失了什么信息？
- stack 与 heap 的生命周期怎样不同？
- `const T*`、`T* const` 和 `const T* const` 有什么区别？
- RAII 怎样防止异常路径上的资源泄漏？
- copy、move 和 borrow 分别表达什么？
- 为什么 release build 中的 UB 可能比 debug build 更难预测？

下一步：[Module 01 · 计算机系统](01-computer-systems.md)
