# RAII, Copy, Move, and Views

[中文](02-ownership-and-raii.md) · **English** · [Back to C/C++ module](../../modules/00-c-cpp-foundations.en.md)

> Reading time: ~5 minutes · Level: Foundation · Freshness: Stable · Last reviewed: 2026-08

## The central question

How does C++ express resource ownership without sacrificing performance?

RAII binds resource lifetime to object lifetime: a constructor establishes valid state and a destructor releases resources automatically. Early returns and exceptions still destroy local objects.

## An owning object

```cpp
class Matrix {
public:
    Matrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), data_(rows * cols) {}

    float& operator()(std::size_t r, std::size_t c) {
        return data_.at(r * cols_ + c);
    }

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<float> data_;
};
```

`Matrix` owns its vector and needs no handwritten destructor. This is the Rule of Zero: when members already manage resources correctly, compiler-generated copy, move, and destruction are usually safer.

## Copy, move, and borrow

- **Copy:** create an independent object and resource.
- **Move:** transfer a resource; the old object remains valid but its content is generally unspecified.
- **Borrow/view:** access a resource temporarily without owning or extending its lifetime.

`std::unique_ptr` expresses unique ownership. `std::shared_ptr` expresses shared lifetime but adds a control block, atomic updates, and a more complicated ownership graph. Use it only when shared lifetime is real.

`std::span<T>` is a non-owning view over contiguous data. It stores a pointer and length but cannot ensure that the owner still exists.

```cpp
float dot(std::span<const float> a,
          std::span<const float> b);
```

A `const` view provides read-only access. It is not a thread-safety guarantee and does not extend lifetime.

## The danger of views

```text
owner destroyed → view still exists → dangling access
```

Tensor transpose, slice, and reshape commonly create views. An API must choose among:

- requiring callers to keep the owner alive;
- letting the view retain shared ownership;
- restricting borrowing to a limited scope;
- materializing an independent contiguous copy when needed.

No choice is always best, but the lifetime contract must be explicit.

## Hands-on check

1. Implement an owning `Tensor` and non-owning `TensorView`.
2. Verify that modifying a copied tensor does not change the original.
3. Verify that a moved resource has one owner.
4. Construct a dangling view, then remove it through sanitizer evidence or API redesign.
5. Compare memory and access costs of a transpose view and contiguous transpose.

## Key conclusions

A raw pointer describes location, an RAII type describes ownership, and a view describes borrowing. Confusing them is a major source of correctness bugs in C++ tensor code.

Next: [Compilation, Sanitizers, and Benchmarks](03-build-debug-benchmark.en.md)
