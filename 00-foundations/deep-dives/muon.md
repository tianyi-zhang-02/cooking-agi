# Muon：怎样更新一张矩阵？

**中文** · [English](muon.en.md)

> 核对：2026-10-10 · 前置：[动量与 AdamW](optimizers.md) · 想先了解思路，读前 2 节和最后的比较表即可。

一个线性层的权重是矩阵，梯度也是。AdamW 按每个坐标的历史调整步长；Muon 则保留矩阵结构，调整整张更新矩阵的方向和尺度。**它改的是参数怎么更新，不是 loss，也不是 attention 的计算。**

## 先看一个 2 × 2 的例子

假设带动量的更新为

$$M=\begin{bmatrix}6&0\\0&2\end{bmatrix}.$$

两个方向都在更新，但第一个方向的幅度是第二个的 3 倍。把整张矩阵除以它的范数，只会一起缩小，比例仍然是 3:1。

Muon 的理想化操作不同。把 $M$ 做奇异值分解（SVD），写成 $U\Sigma V^\top$，再把非零奇异值的大小变成 1。只取这些方向对应的奇异向量 $U_r,V_r$，结果是 $U_rV_r^\top$。对这个满秩例子，它就是单位矩阵，两个方向变成 1:1。保留的是左右奇异向量，不是把每个元素都改成 1。[原作者的说明](https://kellerjordan.github.io/posts/muon/)用这个角度解释正交化（orthogonalization）。

| 对更新做什么 | 本例结果 | 两个方向的比例 |
| --- | --- | --- |
| 不处理 | $\operatorname{diag}(6,2)$ | 3:1 |
| 除以 Frobenius 范数 | $\operatorname{diag}(6,2)/\sqrt{40}$ | 3:1 |
| 理想化正交化 | $\operatorname{diag}(1,1)$ | 1:1 |

这能解释操作，却不能证明训练一定更好。小奇异值可能对应有用的方向，也可能主要是噪声；把幅度拉近不是免费得到更准确的梯度。

如果改成 $\operatorname{diag}(2,0)$，本文的参照结果是 $\operatorname{diag}(1,0)$，不是单位阵：零方向仍然为零。不能直接把数值库返回的全部 $U,V$ 相乘，否则可能把零奇异值对应的补全方向也加进来。后面的代码会用阈值决定哪些奇异值保留。

## 一次更新经过什么

```text
loss → 梯度矩阵 → 加入动量 → 近似正交化 → 尺度调整 → 更新权重
                                                           ↑
                                                   decoupled weight decay
```

用不带 Nesterov 的简化记法：

$$M_t=\mu M_{t-1}+G_t,$$
$$O_t\approx\operatorname{Polar}(M_t),\qquad
W_t=(1-\eta_t\lambda)W_{t-1}-\eta_t s(m,n)O_t.$$

这里 $s(m,n)$ 只依赖矩阵的行数与列数。$\mu$ 是动量，$\lambda$ 是 weight decay，$s$ 是实现选用的尺度系数。实际代码还可能用 Nesterov 或 EMA 形式的动量；不能从另一份代码抄一半公式，就假定两者逐步等价。

[PyTorch 的 Muon 文档](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html)把这些选项分开列出，包括不同的 learning-rate adjustment。使用时记录 PyTorch 版本和配置，不把网页上的默认值当成所有实现的定义。

## Newton–Schulz 在近似什么？

每一步都做 SVD 很贵。Muon 用矩阵乘法组成的 Newton–Schulz 迭代处理归一化后的更新。设 $X_0=M/(\lVert M\rVert_F+\epsilon)$，一类迭代写成：

$$A_k=X_kX_k^\top,\qquad
X_{k+1}=aX_k+(bA_k+cA_k^2)X_k.$$

这不是把权重矩阵 $W$ 变成正交矩阵；被处理的是**更新矩阵**。矩形矩阵也不能两边都满足单位阵：满列秩时可以有 $O^\top O=I$，满行秩时可以有 $OO^\top=I$。

还有一个容易讲错的细节：[固定版本的原实现](https://github.com/KellerJordan/Muon/blob/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd/muon.py)使用五次多项式系数 `(3.4445, -4.775, 2.0315)`，优先考虑少量迭代下的实用效果，**不保证反复迭代就精确得到理想的 polar factor**。代入奇异值 1 就能看见：

$$3.4445-4.775+2.0315=0.701.$$

连 1 都不是这个多项式的不动点，怎么能直接说“多跑几次必然精确收敛到 1”呢？“近似正交化”不等于任意迭代次数下的严格正交化。

<details markdown="1">
<summary>前面的 6 和 2，做五步后变成多少？</summary>

对角矩阵可以逐项检查这个多项式。下面用 Python 浮点数复算，只看变换，不模拟生产实现的 BF16 舍入：

```python
import math

diagonal = [6.0, 2.0]
scale = math.sqrt(sum(value * value for value in diagonal)) + 1e-7
transformed = [value / scale for value in diagonal]
for step in range(5):
    transformed = [3.4445 * value - 4.775 * value**3 + 2.0315 * value**5
                   for value in transformed]
assert math.isclose(transformed[0], 0.7530, abs_tol=1e-4)
assert math.isclose(transformed[1], 1.1337, abs_tol=1e-4)
```

比例不再是 3:1，但也没变成精确的 1:1。SVD 参照帮助理解几何方向；有限步迭代是另一个具体算法，不能拿前者的所有性质直接保证后者。

</details>

### 用 SVD 检查几何含义

下面是教学用的精确参照，不是高性能 Muon 实现，也没有动量和 weight decay。需要安装 PyTorch；只支持实数矩阵。代码对秩亏矩阵保留零空间，不给零奇异值凭空补方向。

```python
import torch


def polar_reference(matrix):
    try:
        inferred = torch.as_tensor(matrix)
    except RuntimeError:
        inferred = torch.as_tensor(matrix, dtype=torch.float64)
    if torch.is_complex(inferred):
        raise ValueError("expected a real matrix, not complex values")
    matrix = torch.as_tensor(matrix, dtype=torch.float64)
    if matrix.ndim != 2 or matrix.numel() == 0:
        raise ValueError("expected a nonempty real matrix")
    if not torch.isfinite(matrix).all():
        raise ValueError("matrix must be finite")
    scale = matrix.abs().max()
    if scale == 0:
        return torch.zeros_like(matrix)
    left, singular, right = torch.linalg.svd(matrix / scale, full_matrices=False)
    threshold = max(matrix.shape) * torch.finfo(matrix.dtype).eps * singular.max()
    retained = (singular > threshold).to(matrix.dtype)
    return (left * retained) @ right


direction = polar_reference([[6.0, 0.0], [0.0, 2.0]])
assert torch.allclose(direction, torch.eye(2, dtype=torch.float64))
assert torch.count_nonzero(polar_reference(torch.zeros(2, 3))) == 0
```

可以再试 `[[1, 1], [0, 1]]`：结果并不是对每个非零元素取正负号。近零奇异值的处理与精度、阈值有关；这段代码的阈值约定也不是某个生产优化器的兼容性保证。

## 为什么不能只换一个优化器名字？

**首先是尺度。** 对形状为 $m\times n$ 的满秩矩阵，理想化 $O$ 有 $\min(m,n)$ 个奇异值等于 1，因此

$$\operatorname{RMS}(O)=\sqrt{\frac{\min(m,n)}{mn}}
=\frac{1}{\sqrt{\max(m,n)}}.$$

同样的 learning rate，矩阵越大，逐元素 RMS 反而越小。[Muon 的扩展训练论文](https://arxiv.org/abs/2502.16982)采用 $0.2\sqrt{\max(m,n)}$ 的更新缩放来匹配 AdamW 的经验尺度。这个常数不是数学定理；原始形状缩放和 RMS 匹配也不是同一套配置。

例如按 [PyTorch v2.9.0 的尺度选项](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/optim/_muon.py#L69)，`original` 使用 $\sqrt{\max(1,m/n)}$。下面用理想满秩参照算 RMS，暂不乘学习率，也不把有限步 NS 当作精确正交化：

| 矩阵形状 | 不加尺度系数 | `original` 后 | RMS 匹配后 |
| --- | ---: | ---: | ---: |
| $8\times2$ | $1/\sqrt8\approx0.3536$ | 0.7071 | 0.2 |
| $2\times8$ | $1/\sqrt8\approx0.3536$ | 0.3536 | 0.2 |

截至 2026-10，稳定版 [PyTorch 2.14 文档](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html)多了第三个选项 `spectral_unclamped`：系数是 $\sqrt{m/n}$，不再把它的下限夹到 1。上表 $8\times2$ 仍是 0.7071，$2\times8$ 则变成 $0.3536\times0.5\approx0.1768$。不传参数时仍按 `original` 处理。

v2.9.0 实现里，weight decay 使用基础学习率，而不是形状调整后的学习率。若 $\eta=0.1,\lambda=0.2$，衰减项是 $0.98W$；更新方向乘 2，不会把衰减也改成 $0.96W$。

**其次是参数分组。** 经典混合用法让隐藏层矩阵用 Muon，embedding、输出层、bias 和 norm 参数继续用 AdamW。Embedding 明明也是二维矩阵，所以不能只用 `ndim == 2` 判断归属。共享 embedding / output 权重要按对象去重，不能被两个优化器更新两遍。[原仓库的用法说明](https://github.com/KellerJordan/Muon)明确区分了这些参数。

**最后是多卡。** 对矩阵的不同切片分别正交化，通常不等于对完整矩阵正交化再切分。若框架把参数展平或分片，先确认逻辑矩阵怎样还原、需要哪些通信；不能把 AdamW 的逐元素分片逻辑原封不动搬过来。

<details id="distributed-step" markdown="1">
<summary>两张卡各算一行，为什么会不一样？</summary>

用一个自拟的更新矩阵来看：

$$M=\begin{bmatrix}1&1\\0&1\end{bmatrix}.$$

假设两张卡各持有一行。如果分别做理想化正交化，每行都只会除以自己的长度；把结果拼起来，两行并不正交。整张矩阵一起算，结果则不同：

| 算法 | 拼好的更新矩阵 | 两行的内积 |
| --- | --- | --- |
| 每行分别处理 | $\begin{bmatrix}1/\sqrt2&1/\sqrt2\\0&1\end{bmatrix}$ | $1/\sqrt2$ |
| 整张矩阵一起处理 | $\frac1{\sqrt5}\begin{bmatrix}2&1\\-1&2\end{bmatrix}$ | $0$ |

这里没有网络传输，也没有低精度舍入；即使用精确 SVD，两种算法也已经不同。用前面的 `polar_reference` 可以直接检查：

```python
import math

momentum = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
whole_update = polar_reference(momentum)
row_updates = torch.cat([polar_reference(row) for row in momentum.split(1, dim=0)])
expected_whole = torch.tensor([[2.0, 1.0], [-1.0, 2.0]], dtype=torch.float64) / math.sqrt(5)
torch.testing.assert_close(whole_update, expected_whole)
assert math.isclose(torch.dot(row_updates[0], row_updates[1]).item(), 1 / math.sqrt(2))
assert not torch.allclose(row_updates, whole_update)
```

[Moonlight 报告 v1 §2.3、算法 1](https://arxiv.org/html/2502.16982v1#S2.SS3)给出一种保留完整矩阵计算的实现顺序：

1. 对梯度做 DP reduce-scatter，再用本地动量更新本地分片。
2. Gather 所需的动量更新分片，恢复完整矩阵，做 Newton–Schulz。
3. 只取本卡负责的更新切片，更新本地参数，再 all-gather 更新后的参数。

**正交化本身是计算；额外的通信来自第 2 步把动量分片 gather 成完整矩阵。** TP 启用后还要处理 TP 切分。报告里的通信量比例和“没有明显延迟”来自其配置与集群，不能直接预测你的训练速度；矩阵还原、临时缓冲和通信重叠都要实际测。这里没有运行多卡 benchmark。

</details>

## 跟 AdamW 怎么比才有意义？

| 问题 | 该看什么 | 常见误判 |
| --- | --- | --- |
| 学得更快了吗 | 同样数据、token 预算下的验证 loss；两边都给合理调参预算 | Muon 调过学习率，AdamW 直接用默认值 |
| 实际省时间了吗 | 达到同一质量所需 wall-clock time | 只看 step 数，漏掉矩阵运算与通信 |
| 省显存了吗 | 优化器状态、主权重、梯度、激活和临时缓冲分别统计 | 少一份二阶矩就说总显存减半 |
| 微调也适合吗 | 固定基座，比较保持能力与目标任务，不只看训练 loss | 预训练结果直接外推到小数据 SFT / RL |
| 能恢复训练吗 | 动量、参数分组、scheduler、步数和 RNG 状态 | 只加载模型权重，宣称无缝续训 |

一个可执行的起点：固定一个小模型和数据顺序，给 AdamW 与 Muon 各自相同数量的 learning-rate / weight-decay 试验；报告多次运行的验证曲线，再加每步耗时、总耗时与峰值显存。这里提供的是实验设计，**本站没有据此跑出“Muon 优于 AdamW”的训练结果**。

## 2026 年再读，哪些说法要留余地？

- “Embedding 必须永远用 AdamW”太绝对。2026 年 10 月的 [AF-Muon](https://arxiv.org/abs/2610.01395)已经研究 tied embedding 的专门更新；它是新的方案，不代表可以把经典 Muon 无修改地套上去。本页仅核对其摘要与问题设定，尚未复现实验。
- “换上 Muon 就能直接放大规模”太乐观。[Kimi K2 报告 §2.1](https://arxiv.org/abs/2507.20534)（2025-07）在一个激活 9B、总 53B 参数的实验里看到，普通 Muon 让最大 attention logit 很快超过 1000。他们的 MuonClip 在每次更新后按 head 检查最大 logit，超过阈值 τ（K2 用 100）就缩小这个 head 的 query / key 投影权重。K2 用它预训练 15.5T tokens，报告全程没有 loss spike。到 2026 年，Muon 已经用在这类生产级训练中：[DeepSeek-V4 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)也写明用 Muon（embedding 等参数仍用 AdamW，见 [V4 笔记](../model-families/deepseek-v4.md)）。规模上去以后，要额外监控 logit 这类新问题。
- “Muon 保证收敛”也太绝对。2026 年 8 月的[收敛性分析](https://arxiv.org/abs/2608.04607)讨论了特定随机优化问题中的反例与误差条件；不能由此推出它在所有 LLM 训练中无效，也不能忽略假设宣称普遍有效。

先理解更新的几何含义，再读新变体究竟改了哪一步，会比记“某优化器已经过时”有用。接着可以看[训练显存](precision-and-memory.md)和[多卡训练](../../06-systems/distributed-training.md)，把算法代价放回完整训练过程。
