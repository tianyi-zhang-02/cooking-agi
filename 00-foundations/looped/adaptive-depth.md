# Looped Transformer：每个 token 该转几圈

**中文** · [English](adaptive-depth.en.md)

> 阅读时间：约 2 分钟 · 难度：进阶 · 最近审阅：2026-09

## 固定圈数，很多是浪费

一句话里，「的」「在」这种词转一圈就够了；要回指、要多跳的词才值得多转几圈。每个 token 都转满，计算就浪费在了不需要的地方。自适应深度做的事，就是让每个 token 自己决定转几圈。

<!-- widget:tx-loop-exit -->

## ACT 和 PonderNet

- **ACT**：Universal Transformer 给每个位置一个停止概率，累计过了阈值就停下；停下的位置把状态直接往后拷，不再更新。
- **PonderNet**（2021）把停止写成一个概率分布。第 $n$ 步停下的概率是

$$p_n = \lambda_n \prod_{j<n} (1 - \lambda_j)$$

损失是各步损失按 $p_n$ 加权，再加一项把 $p$ 拉向几何分布先验的 KL：$\sum_n p_n \mathcal L_n + \beta\, \mathrm{KL}\big(p \,\|\, \mathrm{Geometric}(\lambda_p)\big)$。推理时按这个概率采样停在第几步；和 ACT 不一样的是，它的梯度是无偏的。

## Ouro 的退出 gate

Ouro 在每一圈后面接一个退出 gate：$\lambda_t = \sigma(\mathrm{Linear}(h_t))$。训练分两个阶段：

1. **第一阶段**：损失是 $\sum_t p(t \mid x)\, \mathcal L_t - \beta\, H(p)$。那个熵正则项等价于均匀先验下的 ELBO，作用是别让 gate 一上来就偏向早停；
2. **第二阶段**：拿「多转一圈损失能降多少」当标签，单独训练这个 gate。

有个地方容易误会：Hugging Face 上公开的 Ouro 代码里，4 圈**每次都跑满**，early exit 只决定读哪一圈的隐藏状态当输出，并不省计算。

## Mixture-of-Recursions：让 router 分配深度

Mixture-of-Recursions（2025）把 MoE 的思路搬了过来：用一个 router 给每个 token 定递归深度（expert-choice 或 token-choice 都行）。某一圈没被选中的 token 直接跳过，这一圈也不给它存 KV。论文在 135M 到 1.7B 的模型上，吞吐最高提到 2.06 倍。
