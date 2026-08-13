# 序列梯度、BPTT 与门控

**中文** · [English](recurrent-dynamics.en.md)

> 阅读时间：约 10 分钟 · 难度：进阶 · 最近审阅：2026-08

## 核心问题

普通 RNN 的状态更新是 $h_t=f(a_t)$，其中 $a_t=W_hh_{t-1}+W_xx_t+b$。一个早期状态怎样影响很晚的 loss，取决于 Jacobian 连乘：

$$\frac{\partial h_T}{\partial h_t}=\prod_{k=t+1}^{T}\frac{\partial h_k}{\partial h_{k-1}}
=\prod_{k=t+1}^{T}\text{diag}\!\big(f'(a_k)\big)W_h$$

如果这些矩阵的典型奇异值小于 1，梯度随距离指数衰减；大于 1 则指数爆炸。长依赖首先是一个**优化路径长度**问题。

## BPTT 是什么

Backpropagation Through Time 只是把共享参数的 recurrent cell 展开，再按普通反向传播累计每个时间步对同一参数的梯度：

$$\frac{\partial \mathcal L}{\partial W_h}=\sum_t \frac{\partial \mathcal L}{\partial a_t}\frac{\partial a_t}{\partial W_h}$$

Truncated BPTT 每隔固定步数切断计算图，降低显存和延迟，但模型无法通过梯度直接归因到切断点以前。

## LSTM 为什么更稳

cell state 的核心更新：

$$c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$$

沿直接通路有

$$\frac{\partial c_t}{\partial c_{t-1}}=f_t$$

模型可以把 $f_t$ 学到接近 1，使梯度不必每步穿过一个饱和的 $\tanh(W_hh)$。门不是神秘记忆模块，而是**可学习的梯度与信息流控制器**。

## 工程上怎样处理

- gradient clipping 处理爆炸，不解决消失；
- orthogonal initialization 让 recurrent Jacobian 初始更接近保范数；
- forget-gate bias 设为正值，鼓励训练初期先保留记忆；
- packing / masking 避免 padding 更新隐藏状态；
- 明确 state 是跨 chunk 延续还是每个 sample 重置。

## 可证伪检查

1. 把依赖距离从 8 增加到 64，准确率如何变化？
2. 记录每个时间步 hidden-state gradient norm，是否随距离指数下降？
3. 打乱早期关键 token，输出是否真正变化？
4. LSTM 的 forget gate 是否一直饱和为 1，导致模型只会复制状态？

对应实验在 [`../code/sequence_torch.py`](../code/sequence_torch.py)。
