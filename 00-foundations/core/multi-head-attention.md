# 多头注意力：从公式到实现

**中文** · [English](multi-head-attention.en.md)

> 阅读时间：约 15 分钟 · 难度：必修 · 最近审阅：2026-10-10

先只看一次加权求和：两个位置提供的 value 是 `[2, 0]` 和 `[0, 4]`，注意力权重是 `0.75` 和 `0.25`，读出的结果就是 `[1.5, 1]`。Attention 的计算从这里展开：权重怎么由 Q、K 得到，哪些位置要被遮住，多组这样的计算又怎样拼起来。

<span id="_1"></span>

## 核心计算：匹配并聚合信息 {#_2}

每个位置拿着自己的**问题**（query）去问所有位置的**索引**（key）。对得越上，就从那个位置的**内容**（value）里取越多。取回来的是一个加权平均。

多头会使用不同的投影，各算一份加权结果，再拼起来。有些头可能学到位置邻近或指代关系，但这些分工不是预先指定的，也不保证每个头都对应一个人能命名的功能。

## Attention Matrix 表示什么 {#attention-matrix}

这里先讲 self-attention：$Q$、$K$、$V$ 来自同一个输入 $X$ 的三组可学习线性投影。Cross-attention 则可以从另一条序列取 K/V，后面再展开。

$$
\begin{gathered}
Q=XW_Q\\
K=XW_K\\
V=XW_V
\end{gathered}
$$

它们不是三个单独的维度，也没有人为规定好的语义。本例令 $d_v=d_k$：序列有 $T$ 个 token 时，$Q,K,V$ 都是 $(T,d_k)$；一般情况下 V 的宽度可以不同。公式赋予了它们不同的**计算角色**：Q/K 用来算权重，V 是之后被加权汇总的向量。

完整计算只有下面五步。

### 1. 每两个 token 算一个标量分数 {#1-token}

$$
S=\frac{QK^\top}{\sqrt{d_k}},\qquad
S_{ij}=\frac{\mathbf q_i^\top\mathbf k_j}{\sqrt{d_k}}
$$

$S$ 的形状是 $(T,T)$，这就是 attention score matrix。第 $i$ 行表示“当前位置 $i$ 想从哪里取信息”，第 $j$ 列表示“候选来源位置 $j$”。其中每个格子 $S_{ij}$ 只是一个标量；**这里还没有使用 $V$**。

### 2. Decoder-only 模型先把未来位置遮住 {#2-decoder-only}

GPT 在位置 $i$ 预测下一个 token 时，只允许使用位置 $i$ 及其左边的信息。因果 mask 写成：

$$
M_{ij}=
\begin{cases}
0, & j\le i\\
-\infty, & j>i
\end{cases}
$$

三个 token 的分数矩阵会变成：

$$
S+M=
\begin{bmatrix}
s_{11} & -\infty & -\infty\\
s_{21} & s_{22} & -\infty\\
s_{31} & s_{32} & s_{33}
\end{bmatrix}
$$

mask 不是让模型只看前一个 token，而是让它看到**自己和之前的所有 token**，同时看不到未来。训练时所有位置会并行计算；如果不遮住右上角，前面的位置就能直接读取后面的正确答案，训练目标会发生信息泄漏。双向 encoder 通常没有 causal mask，但仍可能使用 padding mask。

### 3. 对每一行做 softmax {#3-softmax}

$$
\begin{gathered}
A=\operatorname{softmax}_{j}(S+M)
\end{gathered}
$$

Softmax 沿列索引 $j$ 进行，因此每一行满足：

$$
\sum_j A_{ij}=1
$$

由于 $e^{-\infty}=0$，被 mask 的位置权重为 0。这里要求每行至少有一个允许的位置，且允许位置的分数有限；整行都屏蔽时，普通 softmax 没有有效的归一化分母，不能指望它自动返回全零。$A$ 才是 attention weight matrix：第 $i$ 行给出了当前位置从允许位置各取多少信息。

### 4. 用这一行权重汇总所有 value {#4-value}

$$
O=AV,\qquad
\mathbf o_i=\sum_j A_{ij}\mathbf v_j
$$

所以 $QK^\top$ 只负责决定“权重是多少”，真正被取出并混合的是 $V$。attention matrix 的一个格子是标量，而输出 $\mathbf o_i$ 是一个 $d_k$ 维向量。

### 5. 多个头各算一套，再合并 {#5}

每个头都有自己的投影、score matrix 和 attention weights，因此可以学到不同的匹配方式。所有头的输出先拼接，再经过 $W_O$ 投影回模型维度。

整条数据流可以压缩成一句：

$$
\begin{gathered}
S=QK^\top/\sqrt{d_k},\\
A=\operatorname{softmax}_{\rm row}(S+M),\\
O=AV.
\end{gathered}
$$

也就是：**三个投影产生 Q/K/V；Q 和 K 生成 token 两两之间的权重；mask 删除不允许的信息路径；每行 softmax 归一化；最后用这些权重对 V 求加权和。**

> MHA 里的 softmax 确实提供了非线性，但它主要在 token 维度上决定“和谁通信”。FFN 的激活函数则在每个 token 的特征维度上做非线性变换；两者作用不同。

## 单个头在算什么 {#_3}

$$
\begin{gathered}
\text{Attention}(Q,K,V)\\
= \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\end{gathered}
$$

拆到单个 query $\mathbf{q}_i$ 看：

$$
\begin{gathered}
s_{ij}=\mathbf q_i^\top\mathbf k_j/\sqrt{d_k},\\
\alpha_{ij}=\frac{\exp(s_{ij})}{\sum_{j\prime}\exp(s_{ij\prime})},\\
\mathbf o_i=\sum_j\alpha_{ij}\mathbf v_j.
\end{gathered}
$$

在没有 attention dropout 时，每行权重非负且和为 1，所以**单个头投影前的输出**是 value 的凸组合。开头的 `[1.5, 1]` 就是一个例子。别把这个结论套到整个 Transformer block：dropout、输出投影和残差连接都会改变它；权重本身也随输入变化，并不是固定的平均。

## 为什么除以 $\sqrt{d_k}$ {#sqrtd_k}

先看它改变了什么。假设一个头的维度为 64，某个 query 对两个 key 的点积分数是 `[8, −8]`：

| 怎么处理分数 | 送进 softmax 的值 | 两个位置分到的权重 |
| --- | --- | --- |
| 不缩放 | `[8, −8]` | 约 `[0.9999999, 0.0000001]` |
| 除以 $\sqrt{64}=8$ | `[1, −1]` | 约 `[0.881, 0.119]` |
| 只减去最大值 | `[0, −16]` | 与第一行相同 |

不缩放时，几乎全部权重都落在第一个位置上。缩放以后，第二个位置还能参与输出。**这不是说权重越平均越好，而是不希望仅仅因为维度变大，softmax 就过早变得很尖。**

这里要分清两件事：除以一个正数会改变分数差，也会改变权重；减去同一个数不改变分数差，只是让指数运算更稳。`[1000, 1000]` 仍然应该得到 `[0.5, 0.5]`，不能笼统地说“输入大就会饱和”。

<details markdown="1">
<summary>为什么恰好是平方根？把方差算一遍</summary>

用一个简化假设：$q_1,\ldots,q_{d_k},k_1,\ldots,k_{d_k}$ **全部相互独立**，每个分量均值为 0、方差为 1。先看一项：

$$
\begin{gathered}
\mathbb E[q_i k_i]=0,\\
\mathbb E[q_i^2]\mathbb E[k_i^2]=1,\\
\operatorname{Var}(q_i k_i)=1.
\end{gathered}
$$

不同项之间的协方差为 0，所以相加后：

$$
\begin{aligned}
\operatorname{Var}(q^\top k)&=d_k,\\
\operatorname{Var}\!\left(\frac{q^\top k}{\sqrt{d_k}}\right)&=1.
\end{aligned}
$$

维度为 64 时，未缩放点积的标准差是 8。除以 8，标准差回到 1；如果除以 64，方差反而变成 $1/64$。我们是在调整随机波动的尺度，不是在对 64 项求平均。

这也是 [Transformer 原论文 §3.2.1](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS1) 中缩放的出发点。但真实模型的 Q/K 是学出来的，不保证一直满足这些假设。比如令 $q=k$，即使各分量均值为 0、方差为 1，$q^\top k$ 的期望也会变成 $d_k$，不再是 0。只说“每个分量方差为 1”还不够。

</details>

<details markdown="1">
<summary>权重太尖，为什么会影响梯度？</summary>

记 softmax 输出为 $\alpha$，对输入分数 $z$ 的导数是：

$$
\frac{\partial\alpha_i}{\partial z_j}=\alpha_i(\delta_{ij}-\alpha_j).
$$

只有两个位置时，$\partial\alpha_1/\partial z_1=\alpha_1(1-\alpha_1)$。权重为 0.5 时，这一项是 0.25；接近 1 时，它就接近 0。若 value 和后续传来的梯度有界，经由 attention 权重传回分数的梯度会变小。

这里说的是 **attention 这条梯度路径**，不是“整个网络没有梯度了”。它也不能直接套到分类头的 softmax + cross-entropy：后者对 logits 的梯度是“预测概率减标签”，自信地预测错了，梯度未必小。

</details>

手写时用 **head dimension $d_k$**，不是整个模型的 $d_\text{model}$。如果模型还有 QK normalization 或可学习的温度，要以它的实际公式为准；缩放能缓解一个问题，不是训练稳定性的保证。

## 为什么使用多头：目的不是增加维度 {#_4}

$$
\begin{gathered}
\text{head}_i=\\
\operatorname{Attention}(XW_i^Q,XW_i^K,XW_i^V),\\
\operatorname{MultiHead}=\\
\operatorname{Concat}(\text{head}_1,\ldots,\text{head}_h)W^O.
\end{gathered}
$$

<div class="bilingual-note bilingual-intro">
  <span>逐概念双语 · CONCEPT-BY-CONCEPT</span>
  <p>下面三张卡默认中文；点 <strong>English ↻</strong> 可在原位置查看完整英文。</p>
</div>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 1. 多头的核心：多套注意力关系 {#1}

假设 $d_{\text{model}}=512$。一个完整维度的单头会计算

$$
\begin{gathered}
A=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{512}}\right)\\
O=AV.
\end{gathered}
$$

关键限制不是“512 维不够”，而是所有 value 通道共享同一套注意力矩阵 $A$。处理
“小明把书送给小红，因为她很喜欢阅读”中的“她”时，模型可能同时需要追踪指代、
语法依赖、语义角色和局部邻近；单头必须把这些关系压进一套分布。

多头让第 $i$ 个头学习自己的投影和权重：

$$
\begin{gathered}
Q_i=XW_i^Q\\
K_i=XW_i^K\\
V_i=XW_i^V,
\end{gathered}
$$

$$
A_i=\operatorname{softmax}\!\left(\frac{Q_iK_i^\top}{\sqrt{d_k}}\right).
$$

于是模型得到 $A_1,\ldots,A_h$ 多套读取方式。某些头可能偏向指代，另一些偏向
邻近或语法，但这些职责不是人工指定的，也可能彼此重叠。更准确的结论是：
**不同特征可以使用不同的注意力权重，不必全部共享一套分布。**

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">1. The core purpose: multiple attention relations</div>

Suppose $d_{\text{model}}=512$. One full-width attention head computes

$$
\begin{gathered}
A=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{512}}\right)\\
O=AV.
\end{gathered}
$$

The main limitation is not that 512 dimensions are insufficient. It is that every
value channel shares the same attention matrix $A$. Resolving a pronoun may require
coreference, syntactic dependency, semantic role, and local-neighborhood signals at
the same time; one head must compress all of them into one distribution.

Head $i$ instead learns its own projections and weights:

$$
\begin{gathered}
Q_i=XW_i^Q\\
K_i=XW_i^K\\
V_i=XW_i^V,
\end{gathered}
$$

$$
A_i=\operatorname{softmax}\!\left(\frac{Q_iK_i^\top}{\sqrt{d_k}}\right).
$$

The model therefore obtains $A_1,\ldots,A_h$: several ways to read the sequence.
Some heads may emphasize coreference, locality, or syntax, but those jobs are not
assigned by hand and can overlap. The precise advantage is that **different feature
groups can use different attention weights instead of sharing one distribution.**

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 2. 拆分维度是在控制预算 {#2}

原版使用 $d_{\text{model}}=512,h=8$，通常令

$$
d_k=d_v=\frac{512}{8}=64,
$$

所以 $8\times64=512$。如果 8 个头都保留完整 512 维，参数和计算会大幅增长；
把总宽度拆开，才能在接近单头的预算下得到 8 套关系。

单头完整投影有

$$
W_Q,W_K,W_V\in\mathbb{R}^{512\times512}.
$$

多头每组投影是 $512\times64$，8 组合计仍为

$$
8\times(512\times64)=512\times512.
$$

因此标准 MHA 的 Q/K/V 和输出投影总参数量约为 $4d_{\text{model}}^2$，与头数本身
无关。代码也通常只做一次大投影，再 reshape 成 `(B, H, T, d_head)`；不是顺序执行
8 次小模型。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">2. Splitting dimensions controls the budget</div>

The original model uses $d_{\text{model}}=512$ and $h=8$, usually with

$$
d_k=d_v=\frac{512}{8}=64,
$$

so $8\times64=512$. Giving all eight heads the full 512 dimensions would multiply
parameters and compute. Splitting a fixed total width yields eight attention
relations at roughly the budget of one full-width head.

A full-width projection has

$$
W_Q,W_K,W_V\in\mathbb{R}^{512\times512}.
$$

Eight $512\times64$ head projections contain the same total number of elements:

$$
8\times(512\times64)=512\times512.
$$

Standard MHA therefore has about $4d_{\text{model}}^2$ parameters across Q, K, V,
and the output projection, independent of head count. Implementations perform one
large projection and reshape to `(B, H, T, d_head)` rather than running eight small
models sequentially.

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 3. 表达能力不等于泛化保证 {#3}

多头首先增加的是表达能力：它允许多种 token 关系、匹配函数和上下文摘要并存。
更好的表示有时会改善未见数据上的表现，但“用了多头”并不自动推出 generalization
更好。

头数过多时可能出现每头维度太小、多个头功能重复、参数利用率低，甚至过拟合。
实践中经常可以剪掉部分头而几乎不损失性能。所以：

**多头不是为了把维度做大，而是在相近成本下获得多套注意力关系。**

“不同表示子空间”也不要过度解释。每个头确实有独立参数，因此可以学习不同匹配
函数；但“某个头一定负责公司语义、另一个一定负责水果语义”并不是预先设计或必然
可解释的事实。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">3. Expressivity is not a generalization guarantee</div>

Multi-head attention primarily increases expressivity: several token relations,
matching functions, and contextual summaries can coexist. Better representations may
improve performance on unseen data, but using multiple heads does not guarantee
better generalization.

Too many heads can make each head too narrow, create redundant attention patterns,
waste capacity, or contribute to overfitting. In practice, some heads can often be
pruned with little quality loss. Therefore:

**Multi-head attention obtains multiple relations at similar cost; it does not enlarge width for its own sake.**

“Different representation subspaces” should not be over-interpreted either. Separate
parameters let heads learn different matching functions, but no head is guaranteed
to have one clean, human-assigned semantic job.

</div>
</section>

## 从零实现时最容易错的六处 {#_5}

实现时先检查这六处，比盯着最终 loss 猜原因更直接：

| # | 坑 | 正确做法 |
| --- | --- | --- |
| 1 | reshape 顺序 | `view(B,T,H,dh).transpose(1,2)`，**不是** `view(B,H,T,dh)` |
| 2 | mask 时机 | 通常在 softmax 前屏蔽；之后只置零、不重新归一化，不等价 |
| 3 | mask 的值 | 将禁止位置设为 $-\infty$；把 logit 设为 0 仍会分到权重，给它加 0 则什么也没改变 |
| 4 | softmax 数值稳定 | 先减去每行最大值 |
| 5 | 除的维度 | $\sqrt{d_\text{head}}$，不是 $\sqrt{d_\text{model}}$ |
| 6 | 合头 | `transpose(1,2).contiguous().view(...)`；也可用 `reshape`，它必要时会复制。不能假设转置后的 strides 仍适合 `view` |

**第 1 条为什么必须这样。** 投影输出是 `(B, T, d_model)`，其中 `d_model` 这一维是 $h$ 个头**首尾相接**排列的。所以要先把最后一维拆成 `(H, dh)`，再把 `H` 挪到前面。直接 `view(B,H,T,dh)` 会横跨时间维乱切，得到的每个「头」是一堆不相干位置的碎片——形状对，数值全错，而且不报错。

**用两个分数检查 mask。** `[0, 0]` 的 softmax 是 `[0.5, 0.5]`。只允许第一个位置时，先屏蔽得到 `[1, 0]`；softmax 后只置零得到 `[0.5, 0]`，输出就缩小了。如果再除以剩余权重之和，数学上也能得到 `[1, 0]`，但有限精度下可能遇到剩余权重下溢，所以通常在 softmax 前屏蔽。下面的教学代码会拒绝整行被屏蔽的输入，而不是悄悄产生 NaN。

## 实验：验证三种实现等价 {#_6}

<details class="code-drop" markdown="1">
<summary><b>从零实现</b> · 纯 NumPy，不依赖任何框架</summary>

白板上要能默出来的就是这一版。

```python
import numpy as np

def softmax(x, axis=-1):
    if np.any(np.all(np.isneginf(x), axis=axis)):
        raise ValueError("Each query must have at least one allowed key")
    x = x - np.max(x, axis=axis, keepdims=True)   # 数值稳定：先减最大值
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def attention(q, k, v, mask=None):
    """q,k,v: (B, H, T, d_head)   mask: True = 屏蔽"""
    d = q.shape[-1]
    scores = q @ k.swapaxes(-2, -1) / np.sqrt(d)      # (B, H, Tq, Tk)
    if mask is not None:
        scores = np.where(mask, -np.inf, scores)      # softmax 之前
    w = softmax(scores, axis=-1)
    return w @ v, w

class MultiHeadAttention:
    def __init__(self, d_model, n_head, seed=0):
        assert d_model % n_head == 0
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(d_model)
        self.n_head, self.d_head = n_head, d_model // n_head
        self.Wq, self.Wk, self.Wv, self.Wo = (
            rng.uniform(-s, s, (d_model, d_model)) for _ in range(4))

    def split(self, x):                               # (B,T,C) -> (B,H,T,dh)
        b, t, _ = x.shape
        return x.reshape(b, t, self.n_head, self.d_head).transpose(0, 2, 1, 3)

    def __call__(self, x, mask=None):
        q, k, v = self.split(x @ self.Wq), self.split(x @ self.Wk), self.split(x @ self.Wv)
        out, w = attention(q, k, v, mask)
        b, h, t, dh = out.shape
        out = out.transpose(0, 2, 1, 3).reshape(b, t, h * dh)   # 合头
        return out @ self.Wo, w

def causal_mask(t):
    return np.triu(np.ones((t, t), dtype=bool), k=1)  # 严格上三角 = 未来
```

完整可运行版本（含形状打印和自检）：[`../code/attention_numpy.py`](../code/attention_numpy.py)

</details>

<details class="code-drop" markdown="1">
<summary><b>调包</b> · PyTorch，工程里实际会写的样子</summary>

```python
import math, torch, torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, bias=False):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head, self.d_head = n_head, d_model // n_head
        self.wq = nn.Linear(d_model, d_model, bias=bias)
        self.wk = nn.Linear(d_model, d_model, bias=bias)
        self.wv = nn.Linear(d_model, d_model, bias=bias)
        self.wo = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        split = lambda p: p(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        q, k, v = split(self.wq), split(self.wk), split(self.wv)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            att = att.masked_fill(mask, float("-inf"))
        if torch.isneginf(att).all(dim=-1).any():
            raise ValueError("Each query must have at least one allowed key")
        att = att.softmax(dim=-1)

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)   # 合头
        return self.wo(y), att

def causal_mask(t, device=None):
    return torch.triu(torch.ones(t, t, dtype=torch.bool, device=device), diagonal=1)
```

改用 SDPA 时，**mask 也要一起迁移**。这里的布尔 mask 用 `True` 表示屏蔽，而 PyTorch 2.8 SDPA 用 `True` 表示允许，方向正好相反。已有合法 mask、且只需要输出时，核心调用是：

```python
import torch.nn.functional as F

output = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None if mask is None else ~mask,
    dropout_p=0.0,
)
```

这行返回每个头的输出，不返回 attention weights；后面仍要合头和做输出投影。是否采用融合 kernel 取决于设备、dtype 和输入条件，不是调用这个 API 就一定运行 FlashAttention。教学实现里的逐行检查便于定位错误，不是性能优化。接口细节见 [PyTorch 2.8 SDPA](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html)。

</details>

[`../code/attention_torch.py`](../code/attention_torch.py) 把**四种实现**跑在同一组权重上并互相对答案：

```
  from-scratch vs F.scaled_dot_product_attention : 1.19e-07
  from-scratch vs nn.MultiheadAttention          : 1.19e-07
  from-scratch torch vs pure NumPy               : 1.19e-07
  every attention row sums to 1                  : 1.19e-07
  weight on any future position                  : 0.00e+00
```

能让从零实现的版本和 `nn.MultiheadAttention` 对上，比仅仅「能跑」更重要。对不上时定位差异，本身就是有效的调试练习。`nn.MultiheadAttention` 把 $W_Q, W_K, W_V$ 存成一个拼接的 `in_proj_weight`，而 `nn.Linear` 存的是 $(out, in)$，因此搬到 NumPy 时需要转置。

## 面试常见问题 {#_7}

<details class="interview" markdown="1">
<summary>多头注意力有多少参数？</summary>

四个 $d_\text{model} \times d_\text{model}$ 矩阵，$4d^2$（不含 bias）。和**头数无关**——切头只是把同样的参数重新分组。

</details>

<details class="interview" markdown="1">
<summary>时间和显存复杂度是多少？</summary>

设 batch 大小为 B、模型宽度为 d。包含投影的时间复杂度是 $O(BTd^2+BT^2d)$；显式保存各头注意力矩阵需要 $O(BhT^2)$ 空间。FlashAttention 通过分块避免把完整注意力矩阵写入显存，但 dense attention 的二次计算量并没有因此消失，其他激活、KV 和参数也仍占空间。

</details>

<details class="interview" markdown="1">
<summary>Q 和 K 为什么用两个不同的矩阵，不能共享？</summary>

可以共享，但会限制匹配函数。在本节没有额外位置变换的 self-attention 中，若 Q=K，原始分数 $QQ^\top$ 对称；逐行 softmax 的分母不同，最终权重却不一定对称。分开的投影允许两个方向连原始分数都不同；不要把“分数对称”说成“注意力权重对称”。

</details>

<details class="interview" markdown="1">
<summary>V 为什么不参与打分？</summary>

分开投影让匹配特征与传递的内容可以不同。例如匹配时更看重实体类型，读出时还要保留具体属性。Q/K/V 都来自输入，所以内容本来就会影响分数；不是让 V 参与打分就一定退化，而是标准结构选择了分开这两种计算角色。

</details>

<details class="interview" markdown="1">
<summary>多头会不会退化成一个头？</summary>

部分头可能冗余。[Head pruning 的研究](https://arxiv.org/abs/1905.10650)在一些模型和任务上删除了部分头，而性能变化不大。但不能推成任何模型都能直接删掉大部分头：剪哪些、是否微调、在哪些数据上验证，都影响结果。

</details>

<details class="interview" markdown="1">
<summary>mask 为什么要在 softmax 之前加？填 0 行不行？</summary>

给禁止位置的 logit 加 0 不会屏蔽它；把 logit 设成 0 也仍有 $e^0=1$ 的未归一化权重。softmax 前设成 $-\infty$ 才把它从分母里去掉。softmax 后置零再重新归一化，在非空有效支持集上数学等价，但仅置零不等价，数值稳定性也更难处理。

</details>

## 自检 {#_8}

<div class="taste-check">
  <strong>如果真的理解了，你应该能解释：</strong>
  <ol>
    <li>不看代码，写出缩放点积注意力，并用两个分数说明“先 mask”和“softmax 后只置零”的区别。</li>
    <li>为什么除的是 $\sqrt{d_\text{head}}$ 而不是 $\sqrt{d_\text{model}}$？不除会怎样？</li>
    <li><code>view(B,T,H,dh).transpose(1,2)</code> 和 <code>view(B,H,T,dh)</code> 的结果差在哪？为什么后者不报错却全错？</li>
    <li>多头把参数量变成几倍？</li>
  </ol>
</div>

## 继续阅读 {#_9}

没有位置编码、也没有依赖顺序的 mask 时，self-attention 对 token 置换是**等变**的：输入换序，输出跟着换序，并不是输出完全不变。Causal mask 本身已经区分了前后；位置编码还能提供更直接的位置信号。接着看[残差连接](residual-connections.md)和[归一化](normalization.md)如何帮助训练，再回到[原版 Transformer](vanilla-transformer.md) 把整块拼起来。

本章公式的起点是 [Attention Is All You Need §3.2](https://arxiv.org/abs/1706.03762)；实现中的分块显存优化见 [FlashAttention](https://arxiv.org/abs/2205.14135)。

## 快速学习：Multi-Head Attention 的一句话与边界条件 {#multi-head-attention}

<details class="interview" markdown="1">
<summary>匹配、归一化、聚合，以及为什么要多头</summary>

**快速记忆**：Q/K 决定“从哪些位置读”，V 提供“读到的内容”。每个 head 用自己的投影来学习不同关系；多头不是简单地把总维度变大。

**面试回答**

> 每个 query 与所有 keys 做缩放点积，mask 后逐行 softmax 得到权重，再加权汇总 values。Q/K 的最后一维必须相同才能点积，V 的维度只决定输出宽度。多个 heads 在固定计算预算下并行学习不同关系，拼接后由 $W_O$ 混合。

<details markdown="1">
<summary><b>深挖</b>：如果 Q=K，会发生什么？</summary>

Softmax 之前的 Gram matrix $QQ^\top$ 是对称且半正定的，但逐行 softmax 的分母不同，所以最终 attention matrix 一般不对称。若进一步强制 $W_Q=W_K$，模型失去一部分有方向的匹配自由度；分开的投影允许“谁查询谁”与反方向拥有不同分数。

</details>
</details>
