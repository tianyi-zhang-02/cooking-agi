# 从 SGD 到 AdamW：一次更新到底改了什么？

**中文** · [English](optimizers.en.md)

本文先讲逐坐标更新。想知道矩阵优化器改了什么，可接着读 [Muon：梯度是一张矩阵](muon.md)，不必为了入门先读完它。

> 阅读时间：约 15 分钟 · 难度：基础到进阶 · 最近审阅：2026-10

梯度告诉你往哪里走，optimizer 决定怎么走。它可能保留上几步的方向，也可能按每个坐标过去的变化调整步长。把这两件事分开，很多名字就不难记了。

以下公式讨论常见的实数、稠密参数、最小化设置。$t$ 表示一次 optimizer update，不是一个 epoch；$g_t=\nabla L_t(\theta_{t-1})$。所有平方、除法和平方根默认按坐标进行。

## SGD：先学会看学习率

最简单的更新是 $\theta_t=\theta_{t-1}-\eta g_t$。对 $L(w)=\frac12w^2$，就变成

$$w_t=(1-\eta)w_{t-1}.$$

从 $w_0=1$ 出发：$\eta=0.5$ 得到 `1 → 0.5 → 0.25`；$\eta=1.5$ 得到 `1 → -0.5 → 0.25`，虽然来回摆，仍收敛；$\eta=2.1$ 得到 `1 → -1.1 → 1.21`，越走越远。这个一维凸例子的稳定区间是 $0<\eta<2$，不应原样套给神经网络。

随机 / mini-batch 梯度还带有采样噪声。SGD 不只是“会卡在局部最优”，Momentum 也不是“保证跳出局部最优”的按钮。

## Momentum 与 Nesterov：方向要不要有记忆？

一种常见的 Momentum 约定是

$$v_t=\mu v_{t-1}+g_t,\qquad \theta_t=\theta_{t-1}-\eta v_t.$$

方向一致的梯度会积累，来回反向的部分会抵消一些。有的推导在 $g_t$ 前乘 $1-\mu$，那是另一种缩放约定；学习率不能不加说明地照搬。框架还可能对首步动量做特殊处理。

Nesterov 的直观写法是先在预测位置 $\theta_{t-1}-\eta\mu v_{t-1}$ 求梯度，再形成更新。实现可以通过重参数化避免真的临时移动权重。它是在特定假设下改善优化过程，不是提前知道下一步真实梯度。[PyTorch SGD](https://docs.pytorch.org/docs/main/generated/torch.optim.SGD.html)说明了实现与部分论文公式的差别。

## 自适应步长：为什么要记住平方梯度

设两个坐标的梯度分别是 1 和 100。用相同学习率会给后者大 100 倍的更新，但这不一定是合理的尺度。自适应方法利用历史平方梯度调整每个坐标的步长。

| 方法 | 保存什么 | 主要取舍 |
| --- | --- | --- |
| AdaGrad | $s_t=s_{t-1}+g_t^2$ | 累积量只增不减；长期更新的坐标步长可能越来越小 |
| RMSProp | $s_t=\rho s_{t-1}+(1-\rho)g_t^2$ | 更关注近期尺度；不是完整 Hessian |
| AdaDelta | 梯度平方与更新量平方的滑动平均 | 用更新尺度 / 梯度尺度形成比值，不只是把 AdaGrad 换成 EMA |
| Adam | 一阶矩 $m_t$ 与二阶矩 $v_t$ | 同时平滑方向、调整尺度；还有偏差修正与参数开销 |

AdaGrad 常见分母是 $\sqrt{s_t}+\epsilon$。平方累积较少的坐标可能得到更大有效步长，但“出现次数少”不完全等于“平方和小”。[AdaGrad 原论文](https://www.jmlr.org/papers/v12/duchi11a.html)讨论的是利用数据几何结构自适应调整。

AdaDelta 还维护 $u_t=\rho u_{t-1}+(1-\rho)\Delta_t^2$，用

$$\Delta_t=-\frac{\sqrt{u_{t-1}+\epsilon}}{\sqrt{s_t+\epsilon}}g_t.$$

它与只保存梯度平方 EMA 的基本 RMSProp 不相同。实际库可能再提供 learning-rate 系数，不能把“原始推导减少手动学习率需求”理解成所有实现都没有学习率。见 [AdaDelta](https://arxiv.org/abs/1212.5701)。

## Adam：为什么第一步要做偏差修正

从 $m_0=v_0=0$ 开始：

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.$$

零初始化会使早期 EMA 偏小，因此使用

$$\hat m_t=\frac{m_t}{1-\beta_1^t},\quad
\hat v_t=\frac{v_t}{1-\beta_2^t},\quad
\theta_t=\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.$$

若首步 $g_1=2,\beta_1=0.9,\beta_2=0.999$，原始状态为 $m_1=0.2,v_1=0.004$，修正后为 2 和 4。忽略 epsilon，更新幅度约为 $\eta$。这不是以后每步都固定长度：梯度方向变化、历史状态和 epsilon 都会影响结果。

Adam 来自 **adaptive moment estimation**，不是 AdaDelta 与 Momentum 名字拼接；二阶矩也不是二阶导数，Adam 不是 Newton 法。[Adam 原论文](https://arxiv.org/abs/1412.6980)给出算法与修正。NAdam 再把 Nesterov 思路结合进来，但不存在对所有任务都最好的“终极形态”。

## AdamW：weight decay 为什么要拿出来单独做

对没有 Momentum 的普通 SGD，把平方 L2 的梯度 $\lambda\theta$ 加入 $g$，有

$$\theta'=(1-\eta\lambda)\theta-\eta g.$$

这与直接衰减权重等价。但如果先把 $g+\lambda\theta$ 送进 Adam 的两个状态，惩罚也会改变历史矩和按坐标缩放，通常不再等价。

AdamW 把衰减与自适应梯度更新分开：

$$\theta_t=(1-\eta\lambda)\theta_{t-1}
-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.$$

用全新状态、$\theta=2$、数据梯度为 0、$\eta=0.1,\lambda=0.2$ 举例：AdamW 只衰减到 1.96。把 L2 梯度 0.4 送进 Adam，则首步约到 1.9。不是同一个更新。[Decoupled Weight Decay](https://arxiv.org/abs/1711.05101)专门区分了这两件事。

```python
import math

def adamw_scalar(parameter, gradient, state, learning_rate=0.1,
                 decay=0.0, beta1=0.9, beta2=0.999, epsilon=1e-8):
    first, second, steps = state
    values = (parameter, gradient, first, second, learning_rate, decay, beta1, beta2, epsilon)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("finite values required")
    if type(steps) is not int or steps < 0 or second < 0:
        raise ValueError("invalid optimizer state")
    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
        raise ValueError("betas must be in [0, 1)")
    if learning_rate < 0 or decay < 0 or epsilon <= 0:
        raise ValueError("invalid optimizer settings")
    steps += 1
    first = beta1 * first + (1 - beta1) * gradient
    second = beta2 * second + (1 - beta2) * gradient ** 2
    first_corrected = first / (1 - beta1 ** steps)
    second_corrected = second / (1 - beta2 ** steps)
    updated = parameter * (1 - learning_rate * decay)
    updated -= learning_rate * first_corrected / (math.sqrt(second_corrected) + epsilon)
    return updated, (first, second, steps)
```

这个标量实现只展示更新顺序，不支持稀疏梯度、AMSGrad、混合精度或分布式。`gradient=0` 也不等于框架中的 `grad=None`：后者可能让该参数整步跳过。真实行为以 [AdamW 实现](https://docs.pytorch.org/docs/main/generated/torch.optim.AdamW.html)为准。

## 接到真实训练循环里

| 决定 | 要检查的内容 |
| --- | --- |
| 学习率与 scheduler | warmup / decay 按成功的 optimizer update 还是 microbatch 计数？ |
| 梯度累积 | 多个 microbatch 的样本数 / 有效 token 数不同，分母是否仍正确？ |
| 梯度裁剪 | AMP 下是否先 unscale，再裁剪？裁剪的是哪组参数？ |
| 恢复训练 | 除权重外，是否恢复 moments、step、scheduler 与随机状态？ |
| 模块冻结 | optimizer 是否包含正确参数，旧状态是否还占用内存？ |

AdamW 是值得比较的基线，不是无需调参的结论。比较 optimizer 时，让各自有合理的学习率搜索预算，记录验证表现、达到目标所需时间和状态内存。只拿一个学习率跑所有算法，往往是在比较谁更适合这个学习率。

最后把这一篇接回[一次训练](training-step.md)：`backward()` 计算并累积梯度，`step()` 才根据这些状态更新权重。两者不是同一件事。
