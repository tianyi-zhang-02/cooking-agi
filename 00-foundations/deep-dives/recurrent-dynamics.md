# 序列梯度、BPTT 与门控

**中文** · [English](recurrent-dynamics.en.md)

> 阅读时间：约 12 分钟 · 难度：进阶 · 最近审阅：2026-10

先把矩阵缩成一个数：如果梯度每往前传一步都乘 0.9，走过 50 步只剩原来的约 0.005；每步乘 1.1，则变成约 117 倍。真实 RNN 用的是矩阵乘积，但这个小计算已经说明，为什么每一步看似温和的变化，跨很长序列后会让训练困难。

<span id="bptt"></span>

## 为什么远距离依赖难学？ {#_2}

普通 RNN 的状态更新是 $h_t=f(a_t)$，其中 $a_t=W_hh_{t-1}+W_xx_t+b$。一个早期状态怎样影响很晚的 loss，取决于 Jacobian 连乘：

$$J_k=\frac{\partial h_k}{\partial h_{k-1}}=\operatorname{diag}\!\big(f'(a_k)\big)W_h,
\qquad \frac{\partial h_T}{\partial h_t}=J_TJ_{T-1}\cdots J_{t+1}.$$

这里用列向量：最早的变化先乘右边的 $J_{t+1}$，再往左传。矩阵一般不能交换，不能随意倒过来；标量例子看不出这个差别。

若每一步 Jacobian 的算子范数都有统一上界 $c<1$，乘积范数不超过 $c^{T-t}$，也就是至少按这个速度衰减。某一步最大奇异值大于 1 却不保证爆炸，还要看梯度方向和后续矩阵。因此远距离依赖不只涉及表示容量，也涉及很长的优化路径。权重会共享，Jacobian 则随着输入和状态变化，不能当成同一个固定矩阵。

## BPTT 是时间展开后的链式法则 {#bptt_1}

Backpropagation Through Time 只是把共享参数的 recurrent cell 展开，再按普通反向传播累计每个时间步对同一参数的梯度：

$$\nabla_{W_h}\mathcal L=\sum_t \delta_t h_{t-1}^{\mathsf T},\qquad
\delta_t=\nabla_{a_t}\mathcal L.$$

$\delta_t$ 已经包含从后续时间步回传的贡献；外积是当前 cell 对共享权重的那一次使用。不要在这一项里再把 $h_{t-1}$ 对权重的依赖展开一遍，否则会重复计数。[Pascanu 等（2013），§1.1–2.1](https://proceedings.mlr.press/v28/pascanu13.pdf)把这类梯度拆成沿时间传播的贡献。

Truncated BPTT 每隔固定步数切断计算图，降低显存和延迟，但模型无法通过梯度直接归因到切断点以前。

### 截断后，记忆值还在，梯度路径断了 {#truncated-example}

取一个最小递推：$h_0=1$，每步 $h_t=wh_{t-1}$，$w=0.5$，只在第 3 步计算 $\mathcal L=h_3^2/2$。

<figure class="worked-update" id="truncated-trace" lang="zh-CN">
<figcaption>相同的前向结果，不同的梯度</figcaption>
<ol>
<li><b>正常向前算</b><span>1 → 0.5 → 0.25 → 0.125；loss = 0.0078125。</span></li>
<li><b>完整反传</b><span>三步都用到了 w。梯度为 3w⁵ = 0.09375。</span></li>
<li><b>在 h₂ 后 detach</b><span>0.25 仍传入下一步，但被当作常数。梯度只剩 h₃ × h₂ = 0.03125。</span></li>
</ol>
</figure>

所以 detach 不是清空记忆。模型仍能使用前一个 chunk 的状态值，只是本次 loss 不再沿那条路径训练更早的计算。

<details markdown="1">
<summary>用 autograd 核对这两个梯度</summary>

```python
import torch

def recurrence_gradient(truncate):
    weight = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    hidden = torch.tensor(1.0, dtype=torch.float64)
    for step in range(3):
        if truncate and step == 2:
            hidden = hidden.detach()
        hidden = weight * hidden
    loss = hidden.square() / 2
    loss.backward()
    return hidden.item(), loss.item(), weight.grad.item()

assert recurrence_gradient(False) == (0.125, 0.0078125, 0.09375)
assert recurrence_gradient(True) == (0.125, 0.0078125, 0.03125)
```

这里只隔离 detach 的效果：两段之间没有 optimizer step。实际 truncated BPTT 若按 chunk 更新参数，还要区分“状态由哪个版本的参数产生”和“当前用哪个版本继续计算”。

</details>

## LSTM 的关键是那条加法通路 {#lstm}

cell state 的核心更新：

$$c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$$

只看 cell 的直接加法通路，并固定门值时，逐元素导数如下（这不是总导数：$c_{t-1}$ 还会经 $h_{t-1}$ 影响门值和候选值 $\tilde c_t$）：

$$\left.\frac{\partial c_t}{\partial c_{t-1}}\right|_{\text{fixed gates}}
=\operatorname{diag}(f_t).$$

模型可以把 $f_t$ 学到接近 1，这条通路就接近恒等，梯度不必每步穿过一个饱和的 $\tanh(W_hh)$，每步衰减更少。但要穿过的时间步数并没有变短。门不是神秘记忆模块，而是**可学习的梯度与信息流控制器**。

## GRU：少一个状态，不是直接删掉 LSTM 的一扇门 {#gru}

GRU 只有隐藏状态 $h_t$，通常用 reset gate $r_t$ 影响候选状态，用 update gate $z_t$ 混合旧状态和新候选。按 PyTorch 约定：

$$n_t=\tanh(W_{in}x_t+b_{in}+r_t\odot(W_{hn}h_{t-1}+b_{hn})),\qquad
h_t=(1-z_t)\odot n_t+z_t\odot h_{t-1}.$$

这里 $z_t$ 越大，保留越多旧状态；有些教材反过来定义 $z_t$，先看公式再比较。固定门值时，$h_{t-1}=0.8,n_t=-0.2,z_t=0.75$ 得到 $h_t=0.55$。只看 $z_t\odot h_{t-1}$ 这条直接通路，对 $h_{t-1}$ 的导数是 $\operatorname{diag}(z_t)$；总导数还经过 gate 和 candidate，不能只写成 $z_t$。

实现上的一个坑：$W(r\odot h)$ 与 $r\odot(Wh)$ 通常不相等。令 $h=[1,2],r=[1,0],W=[[1,1],[1,1]]$，前者为 `[1,1]`，后者为 `[3,0]`。原始 GRU 与 PyTorch 的 reset 位置有这个区别，移植 checkpoint 时尤其要核对。[PyTorch 2.8 GRU 文档](https://docs.pytorch.org/docs/2.8/generated/torch.nn.GRU.html)列出了两种公式；2026-10 的 [2.14 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html)写法相同。

同宽度下，GRU 通常比标准 LSTM 少一组门投影，但效果和速度仍受任务、序列长度和 kernel 影响。两者都沿时间递推，也都有门控的加法通路（LSTM 的 $f_t\odot c_{t-1}$，GRU 的 $z_t\odot h_{t-1}$），但都没有彻底解决所有长程学习问题。

## 训练排查：优先检查这些问题 {#_3}

- gradient clipping 处理爆炸，不解决消失：它只把过大的范数缩小，不能把已经接近 0 的梯度放大回来；
- orthogonal initialization 可以让 recurrent 权重矩阵保范数，但乘上激活导数后，完整 Jacobian 不一定保范数；
- forget-gate bias 设为正值，鼓励训练初期先保留记忆；
- packing / masking 避免 padding 更新隐藏状态；
- 明确 state 是跨 chunk 延续还是每个 sample 重置。

<span id="_5"></span><span id="_1"></span>

<span id="self-check"></span>

## 证据：怎么证明它真的在记，而不是碰巧猜对 {#_4}

1. 把依赖距离从 8 增加到 64，准确率如何变化？
2. 记录每个时间步 hidden-state gradient norm，是否随距离指数下降？
3. 打乱早期关键 token，输出是否真正变化？
4. LSTM 的 forget gate 是否长期接近 1？再一起看 input gate 与 candidate，不能只凭 forget gate 判断模型只会复制状态。
5. 打乱无关的局部线索，准确率是否基本不变？如果下降很多，模型可能在用捷径。

只有改动早期关键信息后预测跟着变化，才能把“会做题”和“真的记住了远处的信息”分开。

对应实验在 [`../code/sequence_torch.py`](../code/sequence_torch.py)。
