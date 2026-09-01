# Pointers, Arrays, and Contiguous Memory

[中文](01-pointers-and-memory.md) · **English** · [Back to C/C++ module](../../modules/00-c-cpp-foundations.en.md)

> Reading time: ~5 minutes · Level: Foundation · Freshness: Stable · Last reviewed: 2026-08

## The central question

What does a pointer tell a program, and what does it not tell it?

```c
float x = 3.0f;
float *p = &x;
*p = 4.0f;
```

`x` is an object, `&x` is its address, `p` stores that address, and `*p` accesses the object. A pointer normally carries no length, shape, capacity, ownership, or lifetime.

## Arrays and pointer APIs

An array commonly decays to a pointer when passed to a function, so its length must be explicit:

```c
void saxpy(float *y, const float *x, float a, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}
```

`const float *x` prevents modification through `x`; it does not prevent other code from modifying the same memory.

A row-major matrix is commonly stored in one contiguous allocation:

```text
offset(row, col) = row × columns + col
```

A general tensor also needs shape, stride, and offset. A transpose view may swap shape and stride without copying data, which means logical and physical contiguity differ.

## Manual allocation

- `malloc` allocates uninitialized raw memory.
- `calloc` allocates and zeroes it.
- `realloc` may move storage and invalidate the old pointer.
- `free` ends the allocation's lifetime.

Every allocation needs one clear release responsibility. Common failures are leaks, double free, use-after-free, out-of-bounds access, and uninitialized reads.

## Aliasing

Two pointers may overlap. A compiler must handle potential aliases conservatively. C's `restrict` can promise that selected pointers do not overlap within a scope and enable optimization; a false promise creates undefined behavior.

## Hands-on check

1. Define `Matrix {float* data; rows; cols; stride;}`.
2. Implement row-major indexing, a transpose view, and a contiguous copy.
3. Add one deliberate bounds error and catch it with AddressSanitizer.
4. Alias two pointers and test whether an in-place operator remains correct.

## Key conclusions

A pointer is an address, not a tensor. A safe, high-performance tensor API also expresses length, layout, mutability, ownership, and lifetime.

Next: [RAII, Copy, Move, and Views](02-ownership-and-raii.en.md)
