# 序列梯度、BPTT 与门控

**中文** · [English](recurrent-dynamics.en.md)

> 阅读时间：约 10 分钟 · 难度：进阶 · 最近审阅：2026-08

<div class="lesson-recipe advanced">
  <div><span>这次要拆什么</span><strong>远距离梯度为什么消失或爆炸</strong></div>
  <div><span>需要先会</span><strong>链式法则 · 矩阵乘 · RNN / LSTM 前向</strong></div>
  <div><span>真正的主角</span><strong>Jacobian 连乘 · BPTT · cell-state 加法通路</strong></div>
  <div><span>最后要能证明</span><strong>模型是真的记住，而不是碰巧猜中</strong></div>
</div>

## 快速学习：长距离梯度问题的最小解释

<details class="interview" markdown="1">
<summary>BPTT、Jacobian 连乘与 LSTM 加法通路</summary>

**快速记忆**：RNN 的远距离依赖经过 Jacobian 连乘；clipping 只能截住爆炸，不能恢复已经消失的梯度；LSTM 用接近恒等的 cell-state path 缩短有效优化路径。

**面试回答**

> BPTT 把时间递推展开成深网络，共享参数的梯度是所有时间步贡献之和。早期状态到晚期 loss 的梯度包含许多 Jacobian 的乘积，其奇异值决定指数衰减或增长。LSTM 用加法更新和 forget gate 给梯度提供更直接的路径。

<details markdown="1">
<summary><b>深挖</b>：怎样证明模型真的记住了，而不是利用 shortcut？</summary>

除看平均 accuracy，还要随依赖距离画性能与 gradient norm，干预早期关键 token，打乱无关局部线索，并检查 gates 是否长期饱和。只有预测随因果记忆干预而变化，才能把“会做题”与“真的保存远端信息”分开。

</details>
</details>

## 核心问题：为什么远距离依赖难以学习

普通 RNN 的状态更新是 $h_t=f(a_t)$，其中 $a_t=W_hh_{t-1}+W_xx_t+b$。一个早期状态怎样影响很晚的 loss，取决于 Jacobian 连乘：

$$\frac{\partial h_T}{\partial h_t}=\prod_{k=t+1}^{T}\frac{\partial h_k}{\partial h_{k-1}}
=\prod_{k=t+1}^{T}\text{diag}\!\big(f'(a_k)\big)W_h$$

如果这些矩阵的典型奇异值小于 1，梯度就会随着距离指数衰减；大于 1，则一路爆炸。所以“模型记不住很久以前的东西”不只是一个表示能力故事，它首先是一个**优化路径太长**的故事。

## 拆解一：BPTT 是时间展开后的链式法则

Backpropagation Through Time 只是把共享参数的 recurrent cell 展开，再按普通反向传播累计每个时间步对同一参数的梯度：

$$\frac{\partial \mathcal L}{\partial W_h}=\sum_t \frac{\partial \mathcal L}{\partial a_t}\frac{\partial a_t}{\partial W_h}$$

Truncated BPTT 每隔固定步数切断计算图，降低显存和延迟，但模型无法通过梯度直接归因到切断点以前。

## 拆解二：LSTM 真正聪明的是那条加法通路

cell state 的核心更新：

$$c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$$

沿直接通路有

$$\frac{\partial c_t}{\partial c_{t-1}}=f_t$$

模型可以把 $f_t$ 学到接近 1，使梯度不必每步穿过一个饱和的 $\tanh(W_hh)$。门不是神秘记忆模块，而是**可学习的梯度与信息流控制器**。

## 训练排查：优先检查这些问题

- gradient clipping 处理爆炸，不解决消失；
- orthogonal initialization 让 recurrent Jacobian 初始更接近保范数；
- forget-gate bias 设为正值，鼓励训练初期先保留记忆；
- packing / masking 避免 padding 更新隐藏状态；
- 明确 state 是跨 chunk 延续还是每个 sample 重置。

## 证据：怎么证明它真的在记，而不是碰巧猜对

1. 把依赖距离从 8 增加到 64，准确率如何变化？
2. 记录每个时间步 hidden-state gradient norm，是否随距离指数下降？
3. 打乱早期关键 token，输出是否真正变化？
4. LSTM 的 forget gate 是否一直饱和为 1，导致模型只会复制状态？

对应实验在 [`../code/sequence_torch.py`](../code/sequence_torch.py)。

## 自检

<div class="taste-check advanced">
  <strong>不看公式，能否说清：</strong>
  <ol>
    <li>为什么梯度问题来自一串 Jacobian，而不是某一个时间步？</li>
    <li>gradient clipping 解决爆炸后，为什么没有同时解决消失？</li>
    <li>哪个 intervention 能区分“模型真的利用了早期 token”和“数据里恰好有捷径”？</li>
  </ol>
</div>
