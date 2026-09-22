# Looped Transformer：每个 token 该转几圈

**中文** · [English](adaptive-depth.en.md)

> 阅读时间：约 2 分钟 · 难度：进阶 · 最近审阅：2026-09

## 固定圈数，很多是浪费

一句话里，「的」「在」这种词一圈就够；需要回指、需要多跳的词才值得多转几圈。每个 token 都转满，计算就花在了不需要的地方。自适应深度要解决的，就是让每个 token 自己决定转几圈。

<!-- widget:tx-loop-exit -->

## ACT 和 PonderNet

- **ACT**：Universal Transformer 给每个位置一个停止概率，累计超过阈值就停；停下的位置直接把状态往后拷贝，不再更新。
- **PonderNet**（2021）把停止写成一个概率分布。第 $n$ 步停下的概率是

$$p_n = \lambda_n \prod_{j<n} (1 - \lambda_j)$$

损失是各步损失按 $p_n$ 加权，再加一项把 $p$ 拉向几何分布先验的 KL：$\sum_n p_n \mathcal L_n + \beta\, \mathrm{KL}\big(p \,\|\, \mathrm{Geometric}(\lambda_p)\big)$。推理时按概率采样停在哪一步；和 ACT 不同，它的梯度是无偏的。

## Ouro 的退出 gate

Ouro 在每一圈后面接一个退出 gate：$\lambda_t = \sigma(\mathrm{Linear}(h_t))$。训练分两个阶段：

1. **第一阶段**：损失是 $\sum_t p(t \mid x)\, \mathcal L_t - \beta\, H(p)$。熵正则项等价于一个均匀先验下的 ELBO，作用是不让 gate 一开始就偏向早停；
2. **第二阶段**：用「多转一圈损失能降多少」作为标签，单独训练 gate。

一个容易误会的地方：在 Hugging Face 公开的 Ouro 代码里，4 圈**总是全部跑完**，early exit 只决定读哪一圈的隐藏状态作为输出，并不省计算。

## Mixture-of-Recursions：让 router 分配深度

Mixture-of-Recursions（2025）借用了 MoE 的思路：一个 router 给每个 token 决定递归深度（可以是 expert-choice，也可以是 token-choice）。某一圈没被选中的 token 直接跳过，这一圈也不为它存 KV。论文在 135M 到 1.7B 的模型上报告，吞吐最高提升到 2.06 倍。
