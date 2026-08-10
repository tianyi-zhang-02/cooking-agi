# Module 00 · C/C++ Foundations for AI Infrastructure

[中文](00-c-cpp-foundations.md) · **English** · [Back to AI Infra](../README.en.md)

## What this module solves

Much of CUDA, the PyTorch runtime, communication libraries, and inference engines is built on C and C++. This module does not attempt to memorize the whole languages. It focuses on what is required to implement tensors, operators, and systems components: **where memory lives, who owns it, how long it remains valid, and what the compiler ultimately sees.**

## Learning goals

- Read and write foundational C and modern C++.
- Understand arrays, pointers, references, `const`, and pointer arithmetic.
- Understand stack, heap, object lifetime, ownership, and RAII.
- Use structs, classes, constructors, destructors, copy, and move.
- Use `std::vector`, `std::array`, `std::span`, templates, and lambdas.
- Build multi-file projects and use warnings, sanitizers, debuggers, and profilers.

## First understand compilation

```text
source files
→ preprocessing
→ compilation
→ assembly
→ object files
→ linking
→ executable or shared library
```

Headers normally contain declarations and template definitions, while source files contain ordinary function definitions. Each `.c` or `.cpp` file is a translation unit. The linker resolves symbols and libraries across files; inconsistent declarations, duplicate definitions, and missing symbols often surface here.

Optimization levels change both performance and debuggability. Keep several build configurations while learning:

```text
debug:   -O0 -g -Wall -Wextra -Wpedantic
checked: -O1 -g -fsanitize=address,undefined
release: -O3 -DNDEBUG -march=native
```

## C: essential syntax and mental models

### Values, addresses, and pointers

```c
float x = 3.0f;
float *p = &x;
*p = 4.0f;
```

- `x` is a value.
- `&x` is its address.
- `p` stores that address.
- `*p` accesses the object at the address.

A pointer does not carry an array length. An API must preserve shape and capacity separately or inside a struct.

```c
void saxpy(float *y, const float *x, float a, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}
```

`const float *x` says the function does not modify input through this pointer. `restrict` can enable optimization when no-alias requirements are satisfied in C, but violating the promise creates undefined behavior.

### Arrays and contiguous memory

A two-dimensional matrix is commonly stored in one contiguous allocation:

```text
row-major offset = row × number_of_columns + column
```

This makes shape, stride, and access order explicit and is the foundation of tensor layouts.

### Structs and function pointers

C organizes related data with a struct:

```c
typedef struct {
    float *data;
    size_t rows;
    size_t cols;
} Matrix;
```

Function pointers appear in callbacks, kernel dispatch, and strategy tables. Master ordinary functions and pointers before complicated declarations.

### Manual memory management

`malloc` returns uninitialized raw memory, `calloc` zeroes it, `realloc` may move an allocation, and `free` ends ownership. Every successful allocation needs one clear release responsibility.

Common errors include leaks, double free, use-after-free, out-of-bounds access, and uninitialized reads. They may appear to work temporarily, so sanitizer checks are essential.

## C++: expressing ownership with zero-cost abstractions

### References and `const`

```cpp
void normalize(std::span<float> values);
float dot(std::span<const float> a, std::span<const float> b);
```

References and spans can express borrowing without ownership. Use `const` to describe modifications that cannot occur and make interfaces easier to reason about.

### RAII

RAII binds a resource's lifetime to an object's lifetime. `std::vector<float>` releases memory when destroyed; `std::unique_ptr` expresses unique ownership; `std::shared_ptr` should be reserved for genuinely shared lifetime.

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

The matrix owns contiguous storage, its constructor establishes invariants, and members clean themselves up. Prefer standard containers while learning; write raw allocators only after ownership is completely clear.

### Copy, move, and the Rule of Zero

Copy creates an independent object; move transfers resources. A class composed of RAII types such as `vector`, `string`, and `unique_ptr` normally does not need a handwritten destructor, copy, or move operation. This is the Rule of Zero.

Only classes managing raw resources should carefully implement the Rule of Five. Incorrect shallow copies commonly cause double free.

### Templates and generic code

Templates let one algorithm support several dtypes or layouts:

```cpp
template <typename T>
T dot(std::span<const T> a, std::span<const T> b);
```

AI kernels use templates heavily to select dtype, tile size, layout, and hardware path at compile time. Learn simple function templates before metaprogramming.

### Lambdas

A lambda is an anonymous callable that may capture surrounding state. Lambdas appear in algorithms, thread tasks, and callbacks. An object captured by reference must outlive the lambda's use.

## Undefined behavior and safety boundaries

C and C++ expose hardware closely and give programmers responsibility for correctness. Common undefined behavior includes:

- out-of-bounds reads or writes;
- accessing destroyed objects;
- signed integer overflow;
- invalid casts or alignment;
- multithreaded data races;
- violating `restrict` or lifetime rules.

Undefined behavior does not mean “return a random value.” It removes the compiler's obligation to preserve meaningful semantics. Debug builds, sanitizers, assertions, and tests belong in the normal workflow.

## Build and debugging tools

- `clang` / `gcc`: C compilers;
- `clang++` / `g++`: C++ compilers;
- CMake: targets, dependencies, and build configurations;
- AddressSanitizer: out-of-bounds and use-after-free errors;
- UndefinedBehaviorSanitizer: selected forms of UB;
- ThreadSanitizer: data races;
- `gdb` / `lldb`: breakpoints, stacks, and variables;
- `perf`, Instruments, or a profiler: performance evidence.

## Hands-on work

1. Implement vector add, dot product, softmax, and matrix multiplication in C.
2. Write a `Matrix` with `malloc/free` and use sanitizers to find intentionally added leaks and bounds errors.
3. Rewrite the matrix using C++ `std::vector` and RAII.
4. Write templated vector operations for `float` and `double`.
5. Compare correctness and performance in debug and release builds.
6. Split the code into library, test, and benchmark targets.

Detailed project: [Project 00 · C/C++ Tensor Core](../projects/00-c-cpp-tensor-core.en.md)

## Common misconceptions

- Writing a Python extension does not imply correct ownership reasoning.
- A raw pointer does not automatically mean ownership.
- `new/delete` are not the default choice in modern C++.
- Successful compilation does not rule out undefined behavior.
- A microbenchmark optimized away by the compiler reports meaningless speed.
- C++ abstractions are not automatically slow; whether they inline and optimize is what matters.

## Mastery check

- What information is lost when an array decays to a pointer?
- How do stack and heap lifetimes differ?
- How do `const T*`, `T* const`, and `const T* const` differ?
- How does RAII prevent resource leaks on exceptional paths?
- What do copy, move, and borrow each express?
- Why can undefined behavior be less predictable in release builds?

Next: [Module 01 · Computer Systems](01-computer-systems.en.md)
