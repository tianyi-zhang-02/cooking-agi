# 从一个矩阵乘法，看懂梯度、显存与多卡

**中文** · [English](matrix-gradients-to-systems.en.md)

> 最近审阅：2026-11 · 前置：乘法、求和和导数的直觉

为什么只写了一行 `Z = X @ W`，训练时却要存激活、算反向，还要在多张卡之间通信？先拿一个能手算的线性层走完一次更新，再回头看这些系统开销从哪里来。

## 一次前向：把输入特征组合成输出 {#forward}

约定每一行是一条输入，暂时省略偏置：

$$
X\in\mathbb{R}^{B\times I},\quad W\in\mathbb{R}^{I\times O},\quad Z=XW\in\mathbb{R}^{B\times O}.
$$

`B` 是输入行数，`I` 是输入特征数，`O` 是输出特征数。矩阵乘法中的一项是

$$Z_{bo}=\sum_{i=1}^{I}X_{bi}W_{io}.$$

固定一行输入、固定一个输出，就把输入的每项乘上对应权重再相加。例如：

```text
X = [[1, 2]]        W = [[1, 0],       Z = [[1, 2]]
                         [0, 1]]
```

这里 W 是单位矩阵，输出与输入相同。真实模型会用学到的 W 重组特征。PyTorch `nn.Linear` 的参数实际存成 `[O,I]`，计算为 `X @ weight.T + bias`；它与这里的 W 只是转置约定不同。[Linear 的 shape 说明](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html)

## 有了 loss，才能说清要改变什么 {#loss}

让这条输入的目标为 `T=[[0,1]]`。为方便手算，选教学损失

$$L=\frac12\sum_{b,o}(Z_{bo}-T_{bo})^2.$$

这是平方误差的**和乘以一半**，不是默认按元素取平均的 MSE。现在误差为 `[1,1]`，所以 loss 为 1。

先求输出对 loss 的影响：

$$G=\frac{\partial L}{\partial Z}=Z-T=[[1,1]].$$

G 与 Z 的 shape 相同。G 的一个分量表示：在当前点附近，轻微改变这个输出，loss 会以多大的速率改变。它不是“应该把输出直接设成哪个值”。换成别的损失，G 的计算会改变，但下面这个线性层的反向规则仍然适用。

## 为什么权重梯度是 X 的转置乘 G？ {#backward}

单独看 `W[i,o]`。它影响每条输入的第 o 个输出；对第 b 条输入而言，影响的倍率是 `X[b,i]`。将各条输入的贡献加起来：

$$
\frac{\partial L}{\partial W_{io}}
=\sum_b\frac{\partial L}{\partial Z_{bo}}\frac{\partial Z_{bo}}{\partial W_{io}}
=\sum_bG_{bo}X_{bi}.
$$

把这些逐元素的结果排回矩阵，就是

$$\boxed{\frac{\partial L}{\partial W}=X^\top G}.$$

检查 shape：`[I,B] @ [B,O] → [I,O]`，与 W 相同。对上面的数值：

```text
dW = [[1],  @ [[1, 1]] = [[1, 1],
      [2]]                 [2, 2]]
```

第二个输入特征是 2，所以相同上游梯度在这一行产生两倍的权重梯度。这里转置的作用是把“沿 batch 汇总每个参数的贡献”写成矩阵乘法，**不是求 W 的逆矩阵**。

如果前面还有一层，还需要把梯度传回输入。一个输入特征会影响这一行的多个输出，所以这次要沿输出维求和：

$$
\frac{\partial L}{\partial X_{bi}}
=\sum_o\frac{\partial L}{\partial Z_{bo}}\frac{\partial Z_{bo}}{\partial X_{bi}}
=\sum_oG_{bo}W_{io}.
$$

排回矩阵后得到：

$$\boxed{\frac{\partial L}{\partial X}=GW^\top}.$$

shape 为 `[B,O] @ [O,I] → [B,I]`。有偏置 `Z=XW+b` 时，偏置梯度则把 G 沿输入行求和。若 X 是不需要求导的原始数据，这一处可能无需计算 dX；若 X 是前一层的输出，dX 就会继续参与前一层的链式法则。

本例中 W 是单位矩阵，所以 dX 也等于 `[[1,1]]`。它表示损失对输入的敏感程度，不是要求训练器修改数据。计算 dX 要用对应前向的旧权重，不能先更新 W，再把新权重代回这次反向。

## 算出梯度，还没有更新参数 {#update}

选最简单的 SGD，学习率为 0.1：

$$W_{\mathrm{new}}=W-0.1\frac{\partial L}{\partial W}.$$

得到：

```text
W_new = [[ 0.9, -0.1],      X @ W_new = [[0.5, 1.5]]
         [-0.2,  0.8]]      新 loss = 0.25
```

这次 loss 从 1 降到 0.25。我们验证的是这组数据、这个点、这个学习率的一步更新；不能据此保证任何学习率都会下降，更不能用它证明 LLM 的训练效果。

用 PyTorch 可以把手算与自动求导对照：

```python
import torch

inputs = torch.tensor([[1.0, 2.0]], dtype=torch.float64, requires_grad=True)
weights = torch.eye(2, dtype=torch.float64, requires_grad=True)
target = torch.tensor([[0.0, 1.0]], dtype=torch.float64)

loss = 0.5 * ((inputs @ weights - target) ** 2).sum()
loss.backward()
torch.testing.assert_close(weights.grad, torch.tensor(
    [[1.0, 1.0], [2.0, 2.0]], dtype=torch.float64))
torch.testing.assert_close(inputs.grad, torch.ones_like(inputs))
torch.testing.assert_close(weights, torch.eye(2, dtype=torch.float64))

with torch.no_grad():
    weights -= 0.1 * weights.grad
new_loss = 0.5 * ((inputs @ weights - target) ** 2).sum()
torch.testing.assert_close(new_loss, torch.tensor(0.25, dtype=torch.float64))
```

`backward()` 求出梯度；SGD 更新在这里由 `no_grad` 块内那行完成。实际代码通常交给 `optimizer.step()`。重复训练时还要管理梯度清零或累积，详见[自动求导](../pytorch/autograd.md)与[训练循环](../pytorch/training-loop.md)。这个示例已在 CPU、PyTorch 2.8.0 上检查通过。

## 公式里为什么藏着显存开销？ {#memory}

在中间层，X 通常是前面网络产生的激活。`dW=XᵀG` 要求反向时能拿到前向输入 X，因此系统需要安排这些值的生命周期。常见选择有：

- 保存到反向需要时，消耗内存容量；
- 重新计算部分前向结果，增加计算；
- 搬到其他存储位置，再在需要时取回，增加传输与管理开销。

具体该保留哪些 tensor，由算子、所需梯度和执行方式决定，并不意味着每个中间结果都必须一直原样留在 GPU。PyTorch 的自动求导会按算子需要保存相应数据。[Saved tensors](https://docs.pytorch.org/docs/2.14/notes/autograd.html#saved-tensors)

这只是训练显存的一部分。此外还有参数、梯度、优化器状态、临时缓冲等。推理通常没有这次训练的反向图与优化器状态，但自回归生成会有 KV cache 和引擎自己的缓冲。两种场景需要分别算账，接着看[精度与显存](precision-and-memory.md)。

## 多卡为什么要汇总梯度？ {#distributed}

把输入行分到两张卡上，暂时假定两张卡使用相同 W：

$$
\frac{\partial L_{\mathrm{sum}}}{\partial W}
=X_1^\top G_1+X_2^\top G_2.
$$

这就是数据并行汇总梯度的代数来源。每张卡知道局部数据的贡献，要得到整个目标的梯度，必须按照目标和归约约定合并。

给前面的例子再加一个样本，两张卡各处理一条，权重都从单位矩阵开始：

| 进程 | 输入 X | 目标 T | 输出梯度 G | 本地权重梯度 |
| --- | --- | --- | --- | --- |
| Rank 0 | [1, 2] | [0, 1] | [1, 1] | [[1, 1], [2, 2]] |
| Rank 1 | [0, 1] | [1, 0] | [−1, 1] | [[0, 0], [−1, 1]] |

相加得到 dW = `[[1,1],[1,3]]`。第一列的贡献部分抵消，第二列则同向增加。继续使用求和损失与 0.1 的学习率，更新后总 loss 从 2 变为 1.14；配套脚本的第二个例子复算了这一步。

如果改为按两个样本取平均，梯度就要除以 2。PyTorch DDP 通常平均各进程的梯度：这里每个进程处理一个样本，平均结果对应全局样本平均，却不等于前面使用的全局求和目标。见 [DDP 对梯度平均的说明](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)。再核对 MSE 的定义：本文的损失除以 B 是按样本平均；默认 MSE 则把平方误差除以 B×O，不带 1/2。它们只是在这个 O=2 的例子中碰巧相同。

如果目标是全局平均，而两张卡的有效 token 数不同，“各自求均值后再平均”通常不等于全局 token 平均。后端的归约可能采用求和或平均，loss 的分母必须与它配合，不能照搬一个缩放因子。[训练稳定性](../../practice/runtime-reading/training-stability.md#normalization)用 100 个与 10 个有效 token 的例子把这个差别算清。

张量并行则可能切分矩阵乘法的维度，改变部分结果何时通信；FSDP 主要解决训练状态的分片和按需获取。读这些机制前，先写清每张卡上的 X、W、G 的 shape 与归属，往往比先记缩写更有帮助，接着看[张量并行的一层计算](../../06-systems/tensor-parallel.md)。

## 从这里继续拆 LLM {#next}

线性层让我们看见了前向、梯度、参数更新、存储和通信之间的联系。接下来可以用同样办法拆 attention：先算 Q、K、V，再跟踪 `QKᵀ`、softmax 和加权求和。注意力权重依赖输入，softmax 又引入非线性，不能把整层当作一个固定 W。自回归 decoder 的自注意力还会用因果 mask；不是所有 attention 都禁止看后续位置。参见[原论文 §3.2](https://arxiv.org/html/1706.03762v7#S3.SS2)与[完整 Transformer](../transformer.md)。

每次只换一个条件：把 batch 加一行，改 loss 的分母，或者让 W 不再是方阵，重新核对结果。配套的[纯 Python 实验](../code/matrix_gradient_checks.py)对每个 X/W 元素做有限差分检查，覆盖三个小例子，包括非方阵、非单位权重；它检验导数，不测 GPU 性能：

```bash
python3 00-foundations/code/matrix_gradient_checks.py
```
