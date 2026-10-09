# PyTorch：把公式写成能训练的代码

**中文** · [English](README.en.md)

> 阅读时间：约 8 分钟 · 运行环境：CPU · 最近审阅：2026-10

知道公式和写对训练代码，中间还差几步：维度有没有对齐，改动会不会影响另一份数据，梯度到底流向哪里，日志里的 loss 又在平均什么。这几章就补这些地方。

不用先背完所有 API。跟着一个小 batch 从数据走到参数更新，遇到新函数再查它负责哪一步。

每章代码按出现顺序在同一个 Python 会话中运行；下一章重新开始即可。可选的 TensorBoard 接入另作说明，不影响前面的实验。

## 按这个顺序读

| 章节 | 先回答一个问题 | 动手检查 |
| --- | --- | --- |
| [1 · 张量与存储](tensors-and-storage.md) | 为什么我只改了切片，原数据也变了？ | shape、dtype、索引、view 与 copy |
| [2 · 运算与维度](operations-and-shapes.md) | 代码没报错，为什么算的不是我要的 loss？ | broadcasting、归约、矩阵乘法、数值稳定性 |
| [3 · 自动求导](autograd.md) | 有计算图，为什么 `.grad` 还是 None？ | 梯度累积、叶节点、detach、VJP |
| [4 · 从 batch 到训练循环](training-loop.md) | 怎样确认参数真的学到了，而不只是前向能跑？ | Module、DataLoader、验证、日志与恢复 |

## 先看一次完整的小更新

先不训练大模型。只拿两个二维点：`[1, 0]` 属于类别 0，`[0, 2]` 属于类别 1。一个没有 bias 的线性分类器给每个点输出两个 logits。我们故意把权重设为 0，方便手算；这不是深层网络的初始化建议。

<figure class="worked-update">
<ol>
<li><small>01 / 输入</small><strong>2 个样本 × 2 个特征</strong><span>数据 [[1, 0], [0, 2]]<br>标签 [0, 1]</span></li>
<li><small>02 / 前向</small><strong>每个类别都是 0 分</strong><span>W = 0 → logits = 0<br>两类概率各为 0.5</span></li>
<li><small>03 / 损失</small><strong>平均 CE ≈ 0.6931</strong><span>每个目标类别的概率都是 0.5，取 −log 后再平均。</span></li>
<li><small>04 / 反向</small><strong>算出梯度，还没更新</strong><span>第一类权重梯度 [−0.25, 0.5]<br>第二类 [0.25, −0.5]</span></li>
<li><small>05 / 更新</small><strong>沿梯度反方向走</strong><span>SGD 学习率 0.2<br>W ← W − 0.2 × grad</span></li>
<li><small>06 / 再算一次</small><strong>同一批数据的 CE ≈ 0.5787</strong><span>目标类别的分数变高了。这里只检查这一步，不代表泛化效果。</span></li>
</ol>
<figcaption>独立构造的手算例子。前向算预测，反向算梯度，optimizer 才改参数；三件事不要混在一起。</figcaption>
</figure>

为什么第一类权重的梯度是 `[−0.25, 0.5]`？Softmax + CE 对单个样本 logits 的梯度是“预测概率减 one-hot 标签”。第一类在两个样本上的误差分别是 −0.5 和 0.5，乘回各自输入，得到 `[−0.5, 0]` 和 `[0, 1]`。再对两个样本取平均：

$$
\begin{aligned}
\nabla W_{0,:}
&=\tfrac12\big([-0.5,0]+[0,1]\big)\\
&=[-0.25,0.5].
\end{aligned}
$$

所以更新后的两行权重分别是 `[0.05, −0.1]` 和 `[−0.05, 0.1]`。对第一个点，两类分数是 0.05 和 −0.05；对第二个点，是 −0.2 和 0.2。**标签没有传给模型前向，但它通过 loss 决定了更新方向。**

下面直接核对图里的数字，复制这一块就能运行：

```python
import math
import torch

features = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
labels = torch.tensor([0, 1])
weights = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.float64))
optimizer = torch.optim.SGD([weights], lr=0.2)
optimizer.zero_grad(set_to_none=True)
logits = features @ weights.T
loss_before = torch.nn.functional.cross_entropy(logits, labels)
loss_before.backward()
expected_gradient = torch.tensor([[-0.25, 0.5], [0.25, -0.5]], dtype=torch.float64)
torch.testing.assert_close(weights.grad, expected_gradient)
torch.testing.assert_close(weights.detach(), torch.zeros_like(weights))
assert math.isclose(loss_before.item(), math.log(2))
optimizer.step()
with torch.no_grad():
    torch.testing.assert_close(weights, -0.2 * expected_gradient)
    loss_after = torch.nn.functional.cross_entropy(features @ weights.T, labels)
assert math.isclose(loss_after.item(), 0.5787059562, abs_tol=1e-9)
```

第 1 章解释输入如何存，第 2 章解释矩阵乘法和平均，第 3 章追踪梯度，第 4 章再加上数据加载、验证和恢复。你不必现在就懂每个函数；先知道每一步交出了什么，再到对应章节拆开看。

这里使用无类别权重的平均 CE 和无 momentum、无 weight decay 的 SGD。换了这些条件，上面的更新数值也会变。接口约定见 [PyTorch 2.8 CE](https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html) 和 [SGD](https://docs.pytorch.org/docs/2.8/generated/torch.optim.SGD.html)。

## 已经会写训练代码了？

可以直接看几个反例：非连续张量不一定不能 `view`；`[B,1]` 减 `[B]` 会得到 `[B,B]`；`eval()` 不会关闭梯度；最后一个 batch 的 loss 不代表整轮平均值。能解释原因，再去写更大的模型会踏实一些。

代码是本站独立编写的小实验，使用 PyTorch 2.x 的常用接口。数值验证环境单独记录为 **PyTorch 2.8.0 / CPU**，不是“最新版”的代称，也不是 GPU 性能测试。可从仓库根目录运行：

```bash
python -m unittest discover -s site/tests -p test_pytorch_notes.py -v
```

没有安装 PyTorch 时，数值测试会明确跳过；文档结构检查仍会运行。跳过不算数值验证通过。

## 接着去哪里

想看模型实现，去[从零实现实验](../code/README.md)。想把这里的训练循环换成语言模型，去[一次训练更新](../deep-dives/training-step.md)：那里继续讲 token 标签、loss mask 和梯度累积。遇到训练不收敛，再回[泛化与训练诊断](../deep-dives/generalization.md)，不要先把问题归给模型不够大。
