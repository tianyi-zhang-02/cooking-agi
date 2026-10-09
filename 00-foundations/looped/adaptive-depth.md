# Looped Transformer：每个 token 该转几圈

**中文** · [English](adaptive-depth.en.md)

> 阅读时间：约 5 分钟 · 难度：进阶 · 最近审阅：2026-10-09

## 哪些位置值得多算几圈 {#_1}

有些位置继续计算还有收益，有些位置可能已经够用了。自适应深度想学会这个区别，把预算花在更有用的地方。但不能按词性预先断定“的只需一圈、代词必须多圈”：同一个词换了上下文，计算需求也可能不同。

<!-- widget:tx-loop-exit -->

## ACT 和 PonderNet

- [ACT / Universal Transformer](https://arxiv.org/abs/1807.03819)：累计 halting 值，达到阈值或上限后停止更新该位置；输出还涉及各步状态的加权组合和最后的 remainder，不只是挑最后一个状态。
- [PonderNet](https://arxiv.org/abs/2107.05407)（2021）把停止写成一个概率分布。第 $n$ 步停下的概率是

$$p_n = \lambda_n \prod_{j<n} (1 - \lambda_j)$$

这里 $\lambda_n$ 是“已经来到第 n 步时停下”的条件概率，$p_n$ 才是从开始看、恰在这一步停止的概率。损失为 $\sum_n p_n \mathcal L_n + \beta\, \mathrm{KL}(p\|p_G)$，其中 $p_G$ 是采用相同截断约定的几何先验。训练对停止步数求期望，推理可以逐步采样是否停止。论文的无偏梯度结论对应这个概率目标，不能直接推广到任意阈值或截断实现。

看一组自拟数值：最多 3 步，前两步的条件停止概率为 0.2、0.5，第三步强制停止。

| 停在第几步 | 停在这里的概率 | 为什么 |
| --- | --- | --- |
| 1 | 0.2 | 直接停下 |
| 2 | 0.4 | 先继续，再停：0.8 × 0.5 |
| 3 | 0.4 | 前两步都没停，剩余概率放在上限 |

总概率是 1，平均步数是 2.2，而不是把 0.2、0.5 当作两个独立的退出比例。若前两步后直接结束却丢掉剩余的 0.4，算出来的目标就变了。下面只演示这个“最后一步吸收剩余概率”的约定，不是完整 PonderNet trainer。

```python
def stopping_distribution(conditional):
    if not conditional or any(not 0 <= value <= 1 for value in conditional):
        raise ValueError("Expected conditional probabilities in [0, 1]")
    remaining = 1.0
    probabilities = []
    for position, probability in enumerate(conditional):
        mass = remaining if position == len(conditional) - 1 else remaining * probability
        probabilities.append(mass)
        remaining -= mass
    return probabilities

probabilities = stopping_distribution([0.2, 0.5, 1.0])
assert abs(sum(probabilities) - 1) < 1e-12
expected_steps = sum(step * mass for step, mass in enumerate(probabilities, 1))
assert abs(expected_steps - 2.2) < 1e-12
```

## Ouro 的退出 gate

[Ouro](https://arxiv.org/abs/2510.25741) 在每一圈后面接一个退出 gate：$\lambda_t = \sigma(\mathrm{Linear}(h_t))$。训练分两个阶段：

1. **第一阶段**：损失是 $\sum_t p(t \mid x)\, \mathcal L_t - \beta\, H(p)$。对固定最大步数，$\mathrm{KL}(p\|U)=-H(p)+\log T$，所以熵项等价于加权的均匀先验 KL，差一个常数；标准负 ELBO 还要匹配似然定义和权重，不能把任意 $\beta$ 都直接叫标准 ELBO；
2. **第二阶段**：拿「多转一圈损失能降多少」当标签，单独训练这个 gate。

核对 [Ouro-1.4B 实现 `7ea635b`](https://huggingface.co/ByteDance/Ouro-1.4B/blob/7ea635ba1575ae9ab4ae1d83d83e16a6e47fe696/modeling_ouro.py) 可见，它先执行配置中的全部循环，再选择输出状态。默认 4 圈时，“选第 2 圈输出”不等于只计算 2 圈。这里说的是该代码快照，不代表所有 Ouro 推理引擎；要省算力，还需调度器真正跳过后续计算并处理 KV 依赖。

## Mixture-of-Recursions：让 router 分配深度

[Mixture-of-Recursions](https://arxiv.org/abs/2507.10524)（2025）用 router 分配递归深度。其 recursion-wise cache 只为该圈参与的 token 存 KV；recursive sharing 是另一种方案，不要混成同一个缓存规则。Expert-choice 跨 token 选 top-k 有未来信息泄漏风险，论文用辅助预测来处理推理时的选择；token-choice 避开这类跨位置选择，却要面对负载不均。论文报告特定最大 batch 设置下最高 2.06× 吞吐，不能当作所有服务的预期加速。

实际比较时，把平均圈数、准确率、每 token 延迟和峰值 KV 一起记下来。平均圈数降了但算子仍照跑，账面省下的计算就还没变成真实收益。
