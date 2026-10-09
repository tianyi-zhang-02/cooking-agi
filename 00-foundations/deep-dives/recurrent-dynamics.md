# 序列梯度、BPTT 与门控

**中文** · [English](recurrent-dynamics.en.md)

> 阅读时间：约 10 分钟 · 难度：进阶 · 最近审阅：2026-08

先把矩阵缩成一个数：如果梯度每往前传一步都乘 0.9，走过 50 步只剩原来的约 0.005；每步乘 1.1，则变成约 117 倍。真实 RNN 用的是矩阵乘积，但这个小计算已经说明，为什么每一步看似温和的变化，跨很长序列后会让训练困难。

<span id="bptt"></span>

## 核心问题：为什么远距离依赖难以学习 {#_2}

普通 RNN 的状态更新是 $h_t=f(a_t)$，其中 $a_t=W_hh_{t-1}+W_xx_t+b$。一个早期状态怎样影响很晚的 loss，取决于 Jacobian 连乘：

$$\frac{\partial h_T}{\partial h_t}=\prod_{k=t+1}^{T}\frac{\partial h_k}{\partial h_{k-1}}
=\prod_{k=t+1}^{T}\text{diag}\!\big(f'(a_k)\big)W_h$$

若每一步 Jacobian 的算子范数都有统一上界 $c<1$，乘积范数至多按 $c^{T-t}$ 衰减。某一步最大奇异值大于 1 却不保证爆炸，还要看梯度方向和后续矩阵。因此远距离依赖不只涉及表示容量，也涉及很长的优化路径。权重会共享，Jacobian 则随着输入和状态变化，不能当成同一个固定矩阵。

## 拆解一：BPTT 是时间展开后的链式法则 {#bptt_1}

Backpropagation Through Time 只是把共享参数的 recurrent cell 展开，再按普通反向传播累计每个时间步对同一参数的梯度：

$$\frac{\partial \mathcal L}{\partial W_h}=\sum_t \frac{\partial \mathcal L}{\partial a_t}\frac{\partial a_t}{\partial W_h}$$

Truncated BPTT 每隔固定步数切断计算图，降低显存和延迟，但模型无法通过梯度直接归因到切断点以前。

## 拆解二：LSTM 真正聪明的是那条加法通路 {#lstm}

cell state 的核心更新：

$$c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$$

只看 cell 的直接加法通路，并固定门值时，有下面的逐元素导数。总导数还包含经隐藏状态影响门值的其他路径：

$$\frac{\partial c_t}{\partial c_{t-1}}=f_t$$

模型可以把 $f_t$ 学到接近 1，使梯度不必每步穿过一个饱和的 $\tanh(W_hh)$。门不是神秘记忆模块，而是**可学习的梯度与信息流控制器**。

## 训练排查：优先检查这些问题 {#_3}

- gradient clipping 处理爆炸，不解决消失；
- orthogonal initialization 可以让 recurrent 权重矩阵保范数，但乘上激活导数后，完整 Jacobian 不一定保范数；
- forget-gate bias 设为正值，鼓励训练初期先保留记忆；
- packing / masking 避免 padding 更新隐藏状态；
- 明确 state 是跨 chunk 延续还是每个 sample 重置。

## 证据：怎么证明它真的在记，而不是碰巧猜对 {#_4}

1. 把依赖距离从 8 增加到 64，准确率如何变化？
2. 记录每个时间步 hidden-state gradient norm，是否随距离指数下降？
3. 打乱早期关键 token，输出是否真正变化？
4. LSTM 的 forget gate 是否长期接近 1？再一起看 input gate 与 candidate，不能只凭 forget gate 判断模型只会复制状态。

## GRU：少一个状态，不是直接删掉 LSTM 的一扇门 {#gru}

GRU 只有隐藏状态 $h_t$，通常用 reset gate $r_t$ 影响候选状态，用 update gate $z_t$ 混合旧状态和新候选。按 PyTorch 约定：

$$n_t=\tanh(W_{in}x_t+b_{in}+r_t\odot(W_{hn}h_{t-1}+b_{hn})),\qquad
h_t=(1-z_t)\odot n_t+z_t\odot h_{t-1}.$$

这里 $z_t$ 越大，保留越多旧状态；有些教材反过来定义 $z_t$，先看公式再比较。固定门值时，$h_{t-1}=0.8,n_t=-0.2,z_t=0.75$ 得到 $h_t=0.55$。总导数还经过 gate 和 candidate，不能只写成 $z_t$。

实现上的一个坑：$W(r\odot h)$ 与 $r\odot(Wh)$ 通常不相等。令 $h=[1,2],r=[1,0],W=[[1,1],[1,1]]$，前者为 `[1,1]`，后者为 `[3,0]`。原始 GRU 与 PyTorch 的 reset 位置有这个区别，移植 checkpoint 时尤其要核对。[官方 GRU 文档](https://docs.pytorch.org/docs/main/generated/torch.nn.GRU.html)列出了两种公式。

同宽度下，GRU 通常比标准 LSTM 少一组门投影，但效果和速度仍受任务、序列长度和 kernel 影响。两者都沿时间递推；LSTM 有更直接的 cell-state 路径，也没有彻底解决所有长程学习问题。

对应实验在 [`../code/sequence_torch.py`](../code/sequence_torch.py)。

## 自检 {#_5}

<div class="taste-check advanced">
  <strong>不看公式，能否说清：</strong>
  <ol>
    <li>为什么梯度问题来自一串 Jacobian，而不是某一个时间步？</li>
    <li>gradient clipping 解决爆炸后，为什么没有同时解决消失？</li>
    <li>哪个 intervention 能区分“模型真的利用了早期 token”和“数据里恰好有捷径”？</li>
  </ol>
</div>

## 快速学习：长距离梯度问题的最小解释 {#_1}

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
