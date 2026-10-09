# 自动求导：梯度算出来了，为什么没存下来？

**中文** · [English](autograd.en.md)

> 阅读时间：约 16 分钟 · 前置：链式法则、张量运算 · 最近审阅：2026-10

写完 `loss.backward()`，打印某个中间结果的 `.grad`，却看到 `None`。先别急着认定计算图断了：**算过这个梯度**，和**把它存在这个张量的 `.grad` 里**，是两回事。

这一章用几个很小的计算把它们分开。代码只需要 CPU；数值检查环境为 PyTorch 2.8.0。这里讨论常见的实数自动求导，不涉及复数梯度约定。

## 1. 先跟着一条链算

设参数 $w=2$，中间值 $h=w^2$，损失 $L=3h$。手算得到：

$$
\begin{aligned}
\frac{\partial L}{\partial h}&=3,\\
\frac{\partial h}{\partial w}&=2w=4,\\
\frac{\partial L}{\partial w}&=3\times4=12.
\end{aligned}
$$

前向走过的是 `2 → 4 → 12`；反向传的是导数，不是把前向的数值倒着抄回来。

```text
前向：  w = 2  ── 平方 ──>  h = 4  ── 乘 3 ──>  L = 12
反向： dL/dw = 12 <── ×4 ── dL/dh = 3 <── ×3 ── dL/dL = 1
```

```python
import torch

weight = torch.tensor(2.0, requires_grad=True)
hidden = weight.square()
hidden.retain_grad()
loss = 3 * hidden
loss.backward()

assert weight.is_leaf and weight.grad_fn is None
assert not hidden.is_leaf and hidden.grad_fn is not None
assert weight.grad.item() == 12.0
assert hidden.grad.item() == 3.0
assert weight.item() == 2.0
```

`weight` 是我们创建的叶节点（leaf tensor）；`hidden` 是运算产生的非叶节点。默认情况下，反向会给需要梯度的叶节点累积 `.grad`；中间节点通常不保留这份结果。这里调用 `retain_grad()`，才方便事后查看 `hidden.grad`。如果删掉它，梯度仍然会经过 `hidden` 传到 `weight`，但读取 `hidden.grad` 通常会得到 `None` 和一条提醒。

最后一个断言也很重要：**反向传播没有更新参数**。它只是算出梯度；`optimizer.step()` 才负责按优化规则修改参数。`grad_fn` 不是“有没有梯度”的开关：叶节点的 `grad_fn` 是 `None`，照样可以接收梯度。

## 2. 输出不止一个数，要先说清怎样合起来

`backward()` 不带参数时，通常针对只有一个元素的实数输出，隐含地从导数 1 开始。如果输出是一个向量，PyTorch 不知道你想优化它的和、平均值，还是其中一个分量。

比如 $y=(w^2,3w)$。两个分量对 $w$ 的导数分别是 $2w$ 和 3。我们指定权重 $v=(1,2)$，求的是向量—雅可比积（vector–Jacobian product, VJP）：

$$
\begin{aligned}
v^\top J&=\frac{\partial (y_1+2y_2)}{\partial w}\\
&=2w+6.
\end{aligned}
$$

在 $w=2$ 时，结果是 10，不是 7，也不是整个 Jacobian。

```python
parameter = torch.tensor(2.0, requires_grad=True)
outputs = torch.stack((parameter.square(), 3 * parameter))
seed = torch.tensor([1.0, 2.0])
derivative, = torch.autograd.grad(outputs, parameter, grad_outputs=seed)

assert derivative.item() == 10.0
assert parameter.grad is None
```

`autograd.grad` 把导数作为返回值给你，通常不写入输入的 `.grad`。`outputs.backward(seed)` 则会沿图把结果累积到叶节点。两种接口算的是相关的导数，但用途不同。

日常训练更常见的做法，是先用 `mean()` 或 `sum()` 得到标量 loss。选哪一个会影响梯度尺度；上一章里“按样本平均”和“按有效 token 平均”的区别，在这里会直接改变参数更新。

## 3. 为什么第二次梯度变大了？

每次 `backward` 默认都是**累加**，不是覆盖。这个设计允许我们把多个小 batch 的梯度合起来，但也很容易因为忘记清零而训练出错。

```python
parameter = torch.tensor(2.0, requires_grad=True)
parameter.square().backward()
first_gradient = parameter.grad.item()
parameter.square().backward()
accumulated_gradient = parameter.grad.item()
parameter.grad = None
parameter.square().backward()

assert first_gradient == 4.0
assert accumulated_gradient == 8.0
assert parameter.grad.item() == 4.0
```

这段每次都重新计算 `parameter.square()`，所以是在两张新图上求导，再加到同一份 `.grad`。它不是在同一张已经释放中间结果的图上反复反向。

普通训练一般在每次更新前调用 `optimizer.zero_grad(set_to_none=True)`。若要做梯度累积，就按整个有效 batch 的目标来缩放 loss。小 batch 大小不等时，不能随便把每个 batch 的均值再等权平均。

`None` 和零也不同。`None` 可以表示本次没有梯度写到这个参数；零则是一份数值恰好为零的梯度。优化器可能跳过 `grad is None` 的参数，而对零梯度仍处理动量或 weight decay。排错时别把两种情况混在一起。

## 4. detach、no_grad 和 eval 各管什么

| 操作 | 改变什么 | 没有保证什么 |
| --- | --- | --- |
| `clone()` | 复制数据存储 | 不会自动切断梯度 |
| `detach()` | 得到不连接原计算图的张量 | 不会复制存储；改数据仍可能互相影响 |
| `with torch.no_grad():` | 块内通常不记录反向图 | 不切换 Dropout / BatchNorm 的训练行为 |
| `model.eval()` | 调整受训练模式影响的模块 | 不关闭自动求导 |
| `torch.inference_mode()` | 进一步减少推理时的 autograd 开销 | 比 `no_grad` 更严格；产生的张量不适合随意带回需记录梯度的计算 |

```python
source = torch.tensor([2.0, 3.0], requires_grad=True)
copied = source.clone()
detached = source.detach()
independent = source.detach().clone()
assert copied.requires_grad
assert not detached.requires_grad
assert detached.data_ptr() == source.data_ptr()
assert independent.data_ptr() != source.data_ptr()

copied.sum().backward()
assert torch.equal(source.grad, torch.ones(2))

model = torch.nn.Linear(2, 1)
model.eval()
assert model(torch.ones(1, 2)).requires_grad
with torch.no_grad():
    prediction = model(torch.ones(1, 2))
assert not prediction.requires_grad
```

要保存一份不再参与训练、也不会跟着原数据变化的结果，可以用 `detach().clone()`。只做日志时常用 `loss.item()`，但它返回 Python 数值，不能再用它反向；若 loss 在 GPU 上，还可能引入同步。

`no_grad` 有一个容易忽略的例外：某些创建新张量的工厂函数显式接受 `requires_grad`，不受这个上下文按同样方式限制。不要把它理解成“块里不可能出现任何需要梯度的张量”。

## 5. 图的寿命，和原地修改的风险

一次普通 eager 前向会按实际执行的运算建立图；下一次前向可以走不同分支。反向通常会释放不再需要的已保存中间结果。此时直接对原来的 loss 再做一次反向，可能因为这些结果已释放而报错。并非所有简单图都一定报错，所以不要靠“第二次居然能跑”推断图被完整保留。

这三个名字相近，作用却不同：

| 参数 / 方法 | 什么时候用 |
| --- | --- |
| `retain_grad()` | 想查看非叶节点的 `.grad` |
| `retain_graph=True` | 确实需要重复使用同一次前向的图 |
| `create_graph=True` | 想继续对导数求导，例如二阶导数 |

默认做法是下一步重新前向，而不是给所有 `backward` 都加 `retain_graph=True`。无必要地保留图会增加内存占用。

再看原地修改（in-place）：求 $w^2$ 的导数需要知道前向时的 $w$。如果你在前向和反向之间把它改了，反向到底应该用哪一版？PyTorch 会通过版本检查拦住许多这类错误。不要用 `.data` 绕过检查。自定义更新应在反向完成后、`no_grad` 下进行；通常直接交给优化器。

```python
parameter = torch.tensor(2.0, requires_grad=True)
objective = parameter.square()
objective.backward()
with torch.no_grad():
    parameter.add_(parameter.grad, alpha=-0.1)
parameter.grad = None
assert torch.isclose(parameter, torch.tensor(1.6))

coordinate = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
first, = torch.autograd.grad(coordinate.pow(3), coordinate, create_graph=True)
second, = torch.autograd.grad(first, coordinate)
assert first.item() == 12.0
assert second.item() == 12.0
```

`torch.compile` 可能捕获和编译部分执行过程，但这不意味着要把 eager autograd 的规则换成“整个程序永远只有一张静态图”。先把这里的梯度与状态弄对，再讨论编译边界。

## 6. 怎么知道导数不是“看起来对”？

对光滑的小函数，可以把自动求导和中心差分比较：

$$
f'(x)\approx\frac{f(x+\varepsilon)-f(x-\varepsilon)}{2\varepsilon}.
$$

$\varepsilon$ 太大会有截断误差，太小又受浮点舍入影响。下面用 double precision 检查 $f(x)=x^3+2x$，再让 `gradcheck` 检查同一个函数。不要拿不可导的拐点，例如 ReLU 的 0 点，直接套这个测试。

```python
def smooth_objective(value):
    return value.pow(3) + 2 * value

coordinate = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
analytic, = torch.autograd.grad(smooth_objective(coordinate), coordinate)
step = 1e-6
center = coordinate.detach()
numerical = (smooth_objective(center + step) - smooth_objective(center - step)) / (2 * step)
assert torch.allclose(analytic, numerical, atol=1e-7, rtol=1e-7)
assert torch.autograd.gradcheck(smooth_objective, (coordinate,))
```

遇到训练问题，可以按这个顺序查：loss 是否仍是连接计算图的张量；参数是否在图里；`.grad` 是没保留、没产生，还是数值为零；是否清过梯度；最后才看学习率和优化器。一次正常的反向，只能证明代码按这张图求了导，不能证明标签、mask 或训练目标选得对。

## 资料与下一步

上面的代码是教学实验，不是完整训练器。接着看[训练循环](training-loop.md)，把数据、参数、验证和日志接起来。

- [Autograd mechanics](https://docs.pytorch.org/docs/2.8/notes/autograd.html)：叶节点、梯度模式、计算图与原地修改。
- [Tensor.backward](https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.backward.html) 与 [autograd.grad](https://docs.pytorch.org/docs/2.8/generated/torch.autograd.grad.html)：梯度种子、累积和返回值。
- [retain_grad](https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.retain_grad.html) 与 [detach](https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.detach.html)：保留中间梯度不等于复制或保留整张图。
- [Gradcheck mechanics](https://docs.pytorch.org/docs/2.8/notes/gradcheck.html)：数值梯度检查的原理和限制。
