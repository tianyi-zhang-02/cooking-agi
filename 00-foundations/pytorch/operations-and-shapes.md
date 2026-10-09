# 运算与维度：代码能跑，不代表算对了

**中文** · [English](operations-and-shapes.en.md)

> 阅读时间：约 15 分钟 · 前置知识：[张量与存储](tensors-and-storage.md) · 最近审阅：2026-10

两条样本，预测完全正确，均方误差却不是零。先别换 optimizer，看一眼 shape：预测可能是 `[2,1]`，标签却是 `[2]`。广播让这段代码顺利运行，也悄悄换了你要计算的问题。

这章不按函数名背 API，而是逐个回答：谁和谁算、沿哪个轴合并、结果还剩哪些轴。

## 广播先对齐末尾，不理解你的 batch

Broadcasting 从最右边对齐各轴，对应长度相等或其中一个为 1 才能匹配；缺少的前导轴按长度 1 处理。

```text
prediction    [B, 1]
target           [B]    → treated as [1, B]
result        [B, B]

Each prediction is compared with every target,
not just the target from the same sample.
```

```python
import torch

prediction = torch.tensor([[1.0], [3.0]])
target = torch.tensor([1.0, 3.0])
pairwise_error = (prediction - target).square()
assert pairwise_error.tolist() == [[0.0, 4.0], [4.0, 0.0]]
assert pairwise_error.mean().item() == 2.0
matched_error = (prediction.squeeze(-1) - target).square()
assert matched_error.mean().item() == 0.0
```

有时候 `[B,B]` 正是我们想要的，例如对比学习里所有 query 与 candidate 的相似度；回归里却通常不是。广播只检查尺寸，不检查样本含义。

长度为 0 的轴要特别看：`[0]` 与 `[1]` 可以得到 `[0]`，不是机械地取最大长度。原地操作也不能把左侧存储扩成更大的形状。规则见 [Broadcasting semantics](https://docs.pytorch.org/docs/2.14/notes/broadcasting.html)。

## 省了输入复制，不一定省结果内存

`expand` 可以用零 stride 表示重复读取，不真的复制这些输入值；`repeat` 会复制。但两个小输入广播成很大的结果，结果本身仍可能占据大量内存。

比如 `[B,1,D] - [1,N,D]` 会产生 `[B,N,D]`。算两两距离时，如果只要平方距离，可以比较 `||q||² + ||k||² - 2qkᵀ` 的矩阵实现，避免显式保存三维差值；还要留意消减误差，不能把它当作所有 dtype 下都更准确。

多个逻辑元素可能指向 `expand` 的同一存储位置，因此别直接往扩展视图里写值；需要可修改副本时先 `clone`。反向传播时，广播出去的多条路径会累加回原来的元素，见下一章的 bias 例子。

## reduction：分母是什么，剩下哪些轴

对 `[B,T,D]`，`mean(dim=-1)` 得到每个 token 的特征均值，shape 为 `[B,T]`；加上 `keepdim=True` 得到 `[B,T,1]`，方便沿特征轴减回去。

| 操作 | 返回内容 | 需要想清楚 |
| --- | --- | --- |
| `sum` / `mean` / `prod` | 归约后的值 | 不写 dim 时通常归约全部元素 |
| `max(dim=...)` / `min(dim=...)` | 值与索引 | 只要值可用 `amax` / `amin` |
| `argmax` / `argmin` | 索引 | 不是对应值，也不是可微的选择规则 |
| `topk` | values 与 indices | 相同分数的索引顺序不保证稳定 |
| `var` / `std` | 方差 / 标准差 | `correction` 决定分母，默认是 1 |
| `median` / `quantile` | 中位数 / 分位数 | 偶数个元素时，`median` 返回较小的中间值 |

归一化中常用总体方差 `correction=0`。只有一个元素却使用 `correction=1`，就没有足够自由度；空 tensor 的均值也没有定义。不能看到 NaN 就先加一个 epsilon，而不问统计量本来该是什么。

```python
samples = torch.tensor([2.0, 6.0])
assert samples.var(correction=0).item() == 4.0
assert samples.var(correction=1).item() == 8.0
assert samples.median().item() == 2.0
assert samples.quantile(0.5).item() == 4.0

token_loss = torch.tensor([[1.0, 3.0, 99.0], [2.0, 99.0, 99.0]])
valid = torch.tensor([[True, True, False], [True, False, False]])
token_average = token_loss[valid].mean()
sequence_average = torch.stack([row[mask].mean() for row, mask in zip(token_loss, valid)]).mean()
assert token_average.item() == 2.0
assert sequence_average.item() == 2.0
```

上例两个均值恰好相同，不说明它们等价。把第二条唯一的有效 loss 从 2 改为 8，token 平均变成 4，逐序列平均变成 5：前者按 token 加权，后者每条序列等权。训练目标选择不同，梯度也会不同。空有效集合应明确跳过或报错，不能默默制造 NaN。

## 矩阵乘法：最后两个轴负责什么

`*` 是逐元素乘法，可以广播；`@` / `matmul` 才是矩阵乘积。对于矩阵，`[M,K] @ [K,N] → [M,N]`，中间的 K 被求和消掉。

| 入口 | 典型输入 | 区别 |
| --- | --- | --- |
| `dot` / `vdot` | 两个一维向量 | 复数时 `vdot` 对第一个输入取共轭 |
| `mv` | 矩阵与向量 | `[M,K] × [K] → [M]` |
| `mm` | 两个二维矩阵 | 不广播 batch 轴 |
| `bmm` | 两个三维张量 | batch 数相同，不做 batch 广播 |
| `matmul` / `@` | 向量、矩阵、批量矩阵 | 最后两个轴相乘，前导 batch 轴可广播 |
| `addmm` | 偏置与两个矩阵 | `beta * input + alpha * (left @ right)` |
| `baddbmm` / `addbmm` | 偏置与批量矩阵 | 前者保留 batch；后者把 batch 乘积求和 |

```python
queries = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
keys = torch.arange(40, dtype=torch.float32).reshape(2, 5, 4)
scores = queries @ keys.transpose(-2, -1)
assert scores.shape == (2, 3, 5)
torch.testing.assert_close(scores, torch.bmm(queries, keys.transpose(1, 2)))
torch.testing.assert_close(scores, torch.einsum("bqd,bkd->bqk", queries, keys))

offset = torch.zeros(3, 5)
summed = torch.addbmm(offset, queries, keys.transpose(1, 2))
torch.testing.assert_close(summed, scores.sum(dim=0))
```

这里每个 batch 有 3 个 query、5 个 key、4 个特征；输出是每个 query 对每个 key 的分数。`einsum` 把这件事写成带字母的轴约定，便于核对，不保证比 `matmul` 更快。[addbmm](https://docs.pytorch.org/docs/2.8/generated/torch.addbmm.html)会消掉 batch 轴，不能把它当作带 bias 的普通 `bmm`。

常用的矩阵构造也有实际用途：`eye` 是单位矩阵，`diag` 可从向量构造对角矩阵或从二维矩阵取对角线；批量取对角可看 `diagonal`。`triu` / `tril` 只保留上 / 下三角，`diagonal=1` 的上三角常用于标记“未来位置”。`t()` 面向不超过二维的输入，批量转置用 `transpose(-2,-1)`，别拿整个高维 `.T` 当成只交换最后两个轴。

## 每行取一个目标值：gather

三分类任务中，每条样本的正确类别不同。`index_select` 会给所有行取同一组列；我们需要的是每行各取一个。

```python
logits = torch.tensor([[3.0, 1.0, -2.0], [0.0, 2.0, 4.0]])
labels = torch.tensor([0, 2], dtype=torch.long)
log_probs = logits.log_softmax(dim=-1)
selected = log_probs.gather(1, labels[:, None]).squeeze(1)
loss = -selected.mean()
torch.testing.assert_close(loss, torch.nn.functional.cross_entropy(logits, labels))
```

`gather` 的 index 与输入维数一致，这里是 `[B,1]`；它不替你广播 index。我们只是把标签加一个轴，选完再去掉。类别数来自任务定义，不要每个 batch 都用 `labels.unique()` 的数量重新决定输出层大小。

## 数值函数：选对写法比多留小数重要

`exp`、`log`、`sqrt` 等是逐元素运算；`logsumexp` 则计算“指数和的对数”，不是“小数求和”。大 logits 先 `exp` 可能溢出，先 softmax 再 log 也可能碰到零概率。

```python
large = torch.tensor([1000.0, 999.0])
assert torch.isinf(large.exp()).all()
stable = torch.logsumexp(large, dim=0)
torch.testing.assert_close(stable, torch.tensor(1000.3132617))
assert torch.isfinite(large.log_softmax(dim=0)).all()
assert torch.exp(torch.tensor([1, 2])).is_floating_point()
assert torch.round(torch.tensor([0.5, 1.5, 2.5])).tolist() == [0.0, 2.0, 2.0]
```

`round` 遇到正中间时取偶数；`ceil` 向正无穷取整，`floor` 向负无穷取整。`log1p` 在输入接近零时，比先算 `1 + value` 再取 log 更稳。`log` 的实数输入应为正，`sqrt` 的实数输入应非负，异常值要追到来源。

距离也要写清定义：`linalg.vector_norm(vector, ord=2)` 是欧氏长度，`ord=1` 是绝对值和；`dist(left,right,p=2)` 是两个张量之差的欧氏距离。它们不等于归一化后的 cosine similarity。

浮点数的结合顺序会影响结果。`equal` 不是近似比较，也不替你检查 dtype；`allclose` 使用绝对与相对容差，shape 还可能广播。测试模型时更适合明确检查 shape，再用 `torch.testing.assert_close` 检查值和类型。不要把“多打印几位”和“提高计算精度”混为一谈。

上述基础规则对照 [PyTorch 数值精度说明](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)、[归约与数学 API](https://docs.pytorch.org/docs/stable/torch.html)及 [CrossEntropyLoss](https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html)。本章只算小 tensor，没有声称复现大模型训练或性能。
