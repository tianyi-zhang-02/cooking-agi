# RAII、Copy、Move 与 View

**中文** · [English](02-ownership-and-raii.en.md) · [返回 C/C++ 模块](../../modules/00-c-cpp-foundations.md)

> 阅读时间：约 5 分钟 · 难度：Foundation · 时效性：Stable · 最近审阅：2026-08

## 核心问题

C++ 怎样在不牺牲性能的前提下表达“谁拥有资源”？

RAII 把资源 lifetime 绑定到对象 lifetime：constructor 建立有效状态，destructor 自动释放资源。提前返回或异常不会跳过局部对象的 destructor。

## Owning object

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

`Matrix` 拥有 `vector`，因此不需要手写 destructor。这是 Rule of Zero：如果成员已经正确管理资源，class 让编译器生成 copy、move 和 destructor 通常更安全。

## Copy、move 与 borrow

- **Copy**：产生独立对象和独立资源；
- **Move**：把资源转移给新对象，旧对象仍有效但内容通常不再指定；
- **Borrow/view**：临时访问资源，不拥有、不延长 lifetime。

`std::unique_ptr` 表达唯一 ownership；`std::shared_ptr` 表达共享 lifetime，但增加控制块、atomic 更新和更复杂的 ownership graph。只在确实需要共享 lifetime 时使用。

`std::span<T>` 是连续数据的非 owning view。它保存 pointer 和 length，但不能保证 owner 仍存在。

```cpp
float dot(std::span<const float> a,
          std::span<const float> b);
```

`const` view 表达只读访问。它不是线程安全保证，也不延长底层对象 lifetime。

## View 的危险

```text
owner destroyed → view still exists → dangling access
```

Tensor transpose、slice 和 reshape 经常创建 view。API 必须选择：

- 由调用者保证 owner 活得足够久；
- view 持有共享 owner；
- 只允许受限作用域中的借用；
- 必要时 materialize 独立 contiguous copy。

没有一种方案永远最好，但 lifetime contract 必须明确。

## 动手验证

1. 实现 owning `Tensor` 和 non-owning `TensorView`。
2. 测试 copy 后修改一个 tensor 不影响另一个。
3. 测试 move 后资源只由新对象拥有。
4. 构造 dangling view，并用 sanitizer 或 API redesign 消除它。
5. 比较 view transpose 与 contiguous transpose 的内存和访问成本。

## 记住

Raw pointer 描述位置；RAII type 描述 ownership；view 描述借用。把三者混在一起，是 C++ tensor code 最常见的正确性来源之一。

下一篇：[编译、Sanitizer 与 Benchmark](03-build-debug-benchmark.md)
