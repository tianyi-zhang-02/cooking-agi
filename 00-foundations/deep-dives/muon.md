# Muon：怎样更新一张矩阵？

**中文** · [English](muon.en.md)

> 核对：2026-10-09 · 前置：[动量与 AdamW](optimizers.md) · 想先了解思路，读前 2 节和最后的比较表即可。

一个线性层的权重是矩阵，梯度也是。AdamW 按每个坐标的历史调整步长；Muon 则保留矩阵结构，调整整张更新矩阵的方向和尺度。**它改的是参数怎么更新，不是 loss，也不是 attention 的计算。**

## 先看一个 2 × 2 的例子

假设带动量的更新为

$$M=\begin{bmatrix}6&0\\0&2\end{bmatrix}.$$

两个方向都在更新，但第一个方向的幅度是第二个的 3 倍。把整张矩阵除以它的范数，只会一起缩小，比例仍然是 3:1。

Muon 的理想化操作不同。把 $M$ 做奇异值分解（SVD），写成 $U\Sigma V^\top$，再把非零奇异值的大小去掉，留下 $UV^\top$。对这个满秩例子，结果就是单位矩阵，两个方向变成 1:1。它保留左右奇异向量，而不是把每个元素都改成 1。[原作者的说明](https://kellerjordan.github.io/posts/muon/)用这个角度解释正交化（orthogonalization）。

| 对更新做什么 | 本例结果 | 两个方向的比例 |
| --- | --- | --- |
| 不处理 | $\operatorname{diag}(6,2)$ | 3:1 |
| 除以 Frobenius 范数 | $\operatorname{diag}(6,2)/\sqrt{40}$ | 3:1 |
| 理想化正交化 | $\operatorname{diag}(1,1)$ | 1:1 |

这能解释操作，却不能证明训练一定更好。小奇异值可能对应有用的方向，也可能主要是噪声；把幅度拉近不是免费得到更准确的梯度。

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

还有一个容易讲错的细节：[原实现](https://github.com/KellerJordan/Muon/blob/master/muon.py)常见的五次多项式系数是 `(3.4445, -4.775, 2.0315)`，它优先考虑少量迭代下的实用效果，**不保证反复迭代就精确得到 $UV^\top$**。代入奇异值 1 就能看见：

$$3.4445-4.775+2.0315=0.701.$$

连 1 都不是这个多项式的不动点，怎么能直接说“多跑几次必然精确收敛到 1”呢？“近似正交化”不等于任意迭代次数下的严格正交化。

### 用 SVD 检查几何含义

下面是教学用的精确参照，不是高性能 Muon 实现，也没有动量和 weight decay。需要安装 PyTorch；只支持实数矩阵。代码对秩亏矩阵保留零空间，不给零奇异值凭空补方向。

```python
import torch


def polar_reference(matrix):
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

**其次是参数分组。** 经典混合用法让隐藏层矩阵用 Muon，embedding、输出层、bias 和 norm 参数继续用 AdamW。Embedding 明明也是二维矩阵，所以不能只用 `ndim == 2` 判断归属。共享 embedding / output 权重要按对象去重，不能被两个优化器更新两遍。[原仓库的用法说明](https://github.com/KellerJordan/Muon)明确区分了这些参数。

**最后是多卡。** 对矩阵的不同切片分别正交化，通常不等于对完整矩阵正交化再切分。若框架把参数展平或分片，先确认逻辑矩阵怎样还原、需要哪些通信；不能把 AdamW 的逐元素分片逻辑原封不动搬过来。

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
- “Muon 保证收敛”也太绝对。2026 年 8 月的[收敛性分析](https://arxiv.org/abs/2608.04607)讨论了特定随机优化问题中的反例与误差条件；不能由此推出它在所有 LLM 训练中无效，也不能忽略假设宣称普遍有效。

先理解更新的几何含义，再读新变体究竟改了哪一步，会比记“某优化器已经过时”有用。接着可以看[训练显存](precision-and-memory.md)和[多卡训练](../../06-systems/distributed-training.md)，把算法代价放回完整训练过程。
