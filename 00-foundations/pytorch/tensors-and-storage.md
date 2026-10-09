# 张量与存储：改的是哪一份数据？

**中文** · [English](tensors-and-storage.en.md)

> 阅读时间：约 15 分钟 · 前置知识：Python 列表与索引 · 最近审阅：2026-10

你从一批数据里取出两行，把其中一个值改了，回头发现原数据也变了。这里通常没有神秘的 bug：两个 tensor 正在看同一块内存。

理解 tensor 时，先分清三件事：数值放在哪里，按什么形状读取，以及操作是否需要被自动求导记录。前两件在本章讲，第三件留给[自动求导](autograd.md)。

## 先把每个维度叫对

假设一批有 2 条序列，每条 3 个 token，每个 token 用 4 个数表示。它的 shape 是 `[2,3,4]`，我们约定为 `[batch, time, feature]`。这些名字是我们赋予的，PyTorch 不会自动知道哪个轴是时间。

| 观察方式 | 这个例子的结果 | 回答什么 |
| --- | --- | --- |
| `values.shape` / `values.size()` | `(2,3,4)` | 每个轴多长 |
| `values.ndim` | 3 | 有几个轴 |
| `values.numel()` | 24 | 一共有多少个数 |
| `values.dtype` | 例如 `float32` | 每个数用什么格式 |
| `values.device` | 例如 `cpu` | 在哪里存储、计算 |
| `type(values)` | `torch.Tensor` | Python 对象是什么类型 |

`[4]` 只有一个轴，既不是“1 行 4 列”，也不是“4 行 1 列”。后两种分别是 `[1,4]` 和 `[4,1]`。标量的 shape 是 `[]`，有 1 个元素；`[0]` 则是空向量，有 0 个元素。

```python
import torch

values = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
assert values.shape == (2, 3, 4)
assert values.ndim == 3 and values.numel() == 24
assert torch.tensor(5.0).shape == ()
assert torch.empty(0).numel() == 0
assert values[0].shape == (3, 4)
assert values[0:1].shape == (1, 3, 4)
```

整数索引去掉一个轴，切片保留这个轴。最后两行数值对应同一条序列，但 shape 不同；下游要不要保留 batch 维度，应当明确选择。

## 创建张量时，不要把默认值当约定

`torch.tensor([1,2])` 通常推断为 `int64`；浮点列表使用当前默认浮点类型。训练数据最好显式写 dtype。类别 ID 通常用 `long`，连续特征通常用浮点数，mask 用 `bool`；不能因为“模型训练用浮点”，就把类别索引也全部转成浮点。

| 想创建什么 | 常用入口 | 容易混淆的地方 |
| --- | --- | --- |
| 固定数值 | `tensor`、`zeros`、`ones`、`full` | `full` 的填充值也影响类型推断 |
| 等间隔序列 | `arange`、`linspace` | 前者通常不含终点；后者按指定点数包含两端 |
| 随机值 | `rand`、`randn`、`randint` | 分别是 [0,1)、标准正态、左闭右开的整数范围 |
| 指定正态分布 | `normal` | 传的是标准差，不是方差 |
| 与现有张量匹配 | `zeros_like`、`randn_like` | 默认继承 dtype / device；整数张量不能直接做默认 `randn_like` |
| 只分配空间 | `empty` | 内容未初始化，不能当作“小数值”或零 |

`empty` 适合马上覆盖全部内容的缓冲区。启用特定确定性设置时，框架可能填入特殊值，但这仍不意味着可以依赖它作为初始化。[官方说明](https://docs.pytorch.org/docs/2.8/generated/torch.empty.html)列出了这个例外。

`.to(dtype=..., device=...)` 返回转换结果；类型和设备已经匹配时可能直接返回原对象。`.float()` 也不保证复制。类型提升按算子规则发生，不是“PyTorch 每次遍历所有元素，要求类型完全相同”：加法可以提升类型，常见矩阵乘法则要求匹配。遇到报错，检查对应算子。

## 一张图看懂 view

下面只画存储关系，不画梯度：

```text
storage:       [0, 1, 2, 3, 4, 5]
                └───────┬──────┘
base [2,3]:    [[0,1,2], [3,4,5]]   stride=(3,1)
transpose:     [[0,3], [1,4], [2,5]] stride=(1,3)

clone:         [0, 1, 2, 3, 4, 5]   independent storage
```

对普通 strided tensor，某个位置对应的存储偏移是 `storage_offset + Σ index[axis] × stride[axis]`。转置改变读取方式，不必搬动数值。两个对象因此可以 shape 不同，却共享 storage。

```python
base = torch.arange(6).reshape(2, 3)
window = base[:, 1:]
snapshot = window.clone()
window[0, 0] = 50
assert base[0, 1].item() == 50
assert snapshot[0, 0].item() == 1
assert window is not base
```

`alias = base` 是同一个 Python 对象。`window` 是不同对象，但共享存储。`clone` 分配独立存储。别把“新对象”和“新数据”混为一谈。[Tensor Views](https://docs.pytorch.org/docs/2.14/tensor_view.html)说明了这一区别。

## reshape 不是 transpose 的替代品

`reshape` 按当前逻辑顺序重新分组；`transpose` / `permute` 改的是轴的对应关系。`[B,T,D] → [B,D,T]` 要交换 time 和 feature，应该用 `transpose(1,2)`，而不是只把 shape 写成想要的样子。

`view` 要求目标形状与当前 size / stride 兼容。连续张量比较容易满足，但**非连续不等于一定失败**。`reshape` 能共享时共享，不行时复制；正确性不能依赖它一定选择哪一种。

```python
grid = torch.arange(30).reshape(5, 6)
spaced = grid[:, ::2]
flat = spaced.view(-1)
assert not spaced.is_contiguous()
assert flat.stride() == (2,)
assert flat.tolist() == list(range(0, 30, 2))

swapped = grid.transpose(0, 1)
repacked = swapped.reshape(-1)
assert repacked[:5].tolist() == [0, 6, 12, 18, 24]
assert repacked.data_ptr() != swapped.data_ptr()
```

第一段每隔两个存储位置取一个数，仍能按同一规律展平；第二段展平转置结果需要重新排数据。在这个具体例子里 `reshape` 复制了。`contiguous()` 也不是“总复制”：原来已经满足要求时可以原样返回。细则见 [view 的 stride 条件](https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.view.html)。

`flatten(start_dim=1)` 常用于保留 batch、合并后面各轴。`squeeze(dim)` 只移除指定的长度为 1 的轴；裸 `squeeze()` 可能把 batch size = 1 的 batch 轴也删掉。`unsqueeze(dim)` 则插入一个长度为 1 的轴。

## 取样、筛选和拼 batch

基本切片通常返回 view；整数数组 / 列表、布尔 mask 等 advanced indexing 通常返回拷贝。不过 `values[mask] = ...` 这样的赋值会修改原 tensor，不能据此推断“高级索引永不修改源数据”。

`values[-1]` 可以取末项；Python 列表式的 `values[::-1]` 不被支持，反转用 `flip`。`matrix[::2,::2]` 是两个轴各隔一个取；`matrix[::2][::2]` 则对第一个轴取了两次。

| 需要的操作 | 选择 | 结果的轴 |
| --- | --- | --- |
| 给全部行选择相同几列 | `index_select(dim, indices)` | 索引是一维；可以重复 |
| 每行选择不同列 | `gather(dim, indices)` | 结果形状由 index 决定；例子见下一章 |
| 只取满足条件的元素 | 布尔索引 / `masked_select` | 全形状 mask 取值通常展平成一维 |
| 条件分支选值 | `where(mask, left, right)` | 按广播规则逐元素选 |
| 沿已有轴合并 | `cat` | 其他轴必须匹配 |
| 增加一个轴 | `stack` | 输入 shape 必须一致 |
| 按每块大小拆开 | `split` | 整数参数是块大小，不是块数 |
| 请求若干块 | `chunk` | 可能返回更少的块；`tensor_split` 按指定块数拆 |

```python
first = torch.tensor([2, 4, 6])
second = torch.tensor([8, 10, 12])
assert torch.cat([first, second]).shape == (6,)
assert torch.stack([first, second]).shape == (2, 3)
assert [part.numel() for part in torch.arange(7).split(3)] == [3, 3, 1]
assert len(torch.arange(4).chunk(3)) == 2
assert len(torch.tensor_split(torch.arange(4), 3)) == 3
```

不要把 `stack` 类比成 `[1,2].append([3,4])`：后者是形状不齐的 Python 列表。拆分产生的 tuple 不能替换其中的项，但里面的 tensor 仍可能修改原存储。`chunk` 的返回数量以[文档](https://docs.pytorch.org/docs/2.8/generated/torch.chunk.html)为准，不要不检查就解包成固定数量。

## 和 NumPy、Python 交接时

`torch.tensor(array)` 会复制输入数据；`from_numpy` 与兼容的 `as_tensor` 可以共享存储。共享更省复制，却意味着另一边的修改也可能影响训练输入。

`.numpy()` 默认要求 CPU、无需梯度且 dtype / layout 等受支持，返回值通常共享存储。`.numpy(force=True)` 会处理 detach、CPU 和共轭等转换，但不能理解为“无条件独立复制”。真要独立快照，明确复制。见 [numpy 接口条件](https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.numpy.html)。

`.tolist()` 生成 Python 嵌套数值，`list(tensor)` 则沿第一个轴拿到子 tensor。`.item()` 要求恰好一个元素，不要求一定是零维；它得到的 Python 数不再带梯度关系。CUDA 上频繁 `item()` 还可能引入 CPU 等待，所以通常用于日志，而不是拼 loss。

下次遇到“改了 A，B 也变了”，先查是否共享存储；遇到 shape 正确但含义不对，先查是否错误地用 reshape 交换了轴。下一章看另一类更隐蔽的问题：[代码能跑，但算错了对象](operations-and-shapes.md)。
