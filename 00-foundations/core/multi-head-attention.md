# 多头注意力：手搓版

**中文** · [English](multi-head-attention.en.md)

> 阅读时间：约 9 分钟 · 难度：必修 · 最近审阅：2026-08

<div class="lesson-recipe">
  <div class="recipe-flip" data-concept-card>
    <div class="recipe-face" data-concept-zh><span>解决什么问题 · PROBLEM</span><strong>让每个位置去序列里取它需要的信息</strong></div>
    <div class="recipe-face" data-concept-en><span>Problem · 问题</span><strong>Let each position retrieve the information it needs from the sequence</strong></div>
  </div>
  <div class="recipe-flip" data-concept-card>
    <div class="recipe-face" data-concept-zh><span>前置知识 · PREREQUISITES</span><strong>三个投影 W_Q, W_K, W_V · 一个输出投影 W_O</strong></div>
    <div class="recipe-face" data-concept-en><span>Prerequisites · 前置知识</span><strong>Three projections W_Q, W_K, W_V · one output projection W_O</strong></div>
  </div>
  <div class="recipe-flip" data-concept-card>
    <div class="recipe-face" data-concept-zh><span>核心机制 · CORE MECHANISM</span><strong>缩放点积 · 切头 · mask 在 softmax 之前</strong></div>
    <div class="recipe-face" data-concept-en><span>Core mechanism · 核心机制</span><strong>Scaled dot product · split heads · mask before softmax</strong></div>
  </div>
  <div class="recipe-flip" data-concept-card>
    <div class="recipe-face" data-concept-zh><span>常见错误 · COMMON MISTAKES</span><strong>reshape 的顺序、mask 的时机、除错了维度</strong></div>
    <div class="recipe-face" data-concept-en><span>Common mistakes · 常见错误</span><strong>Reshape order, mask timing, and scaling by the wrong dimension</strong></div>
  </div>
</div>

## 一句话：查字典，然后加权平均

每个位置拿着自己的**问题**（query）去问所有位置的**索引**（key）。对得越上，就从那个位置的**内容**（value）里取越多。取回来的是一个加权平均。

多头的意思是：同一句话同时问好几个不同的问题——一个头盯语法搭配，一个头盯指代，一个头盯位置邻近。问完各自取一份，再拼起来。

## 从输入到输出：attention matrix 到底装了什么

先把容易误解的比喻放下。$Q$、$K$、$V$ 本质上只是同一个输入 $X$ 经过三组不同的可学习线性投影：

$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$

它们不是三个单独的维度，也没有人为规定好的语义。假设序列有 $T$ 个 token，每个头的维度是 $d_k$，那么 $Q,K,V$ 的形状都是 $(T,d_k)$。公式赋予了它们不同的**计算角色**：$Q$ 和 $K$ 用来算权重，$V$ 是之后真正被加权汇总的向量。

完整计算只有下面五步。

### 1. 每两个 token 算一个标量分数

$
S=\frac{QK^\top}{\sqrt{d_k}},\qquad
S_{ij}=\frac{\mathbf q_i^\top\mathbf k_j}{\sqrt{d_k}}
$

$S$ 的形状是 $(T,T)$，这就是 attention score matrix。第 $i$ 行表示“当前位置 $i$ 想从哪里取信息”，第 $j$ 列表示“候选来源位置 $j$”。其中每个格子 $S_{ij}$ 只是一个标量；**这里还没有使用 $V$**。

### 2. Decoder-only 模型先把未来位置遮住

GPT 在位置 $i$ 预测下一个 token 时，只允许使用位置 $i$ 及其左边的信息。因果 mask 写成：

$
M_{ij}=
\begin{cases}
0, & j\le i\\
-\infty, & j>i
\end{cases}
$

三个 token 的分数矩阵会变成：

$
S+M=
\begin{bmatrix}
s_{11} & -\infty & -\infty\\
s_{21} & s_{22} & -\infty\\
s_{31} & s_{32} & s_{33}
\end{bmatrix}
$

mask 不是让模型只看前一个 token，而是让它看到**自己和之前的所有 token**，同时看不到未来。训练时所有位置会并行计算；如果不遮住右上角，前面的位置就能直接读取后面的正确答案，训练目标会发生信息泄漏。双向 encoder 通常没有 causal mask，但仍可能使用 padding mask。

### 3. 对每一行做 softmax

$
A=\operatorname{softmax}_{j}(S+M)
$

Softmax 沿列索引 $j$ 进行，因此每一行满足：

$
\sum_j A_{ij}=1
$

由于 $e^{-\infty}=0$，被 mask 的位置权重严格为 0。$A$ 才是通常所说的 attention weight matrix：第 $i$ 行给出了当前位置从所有允许位置各取多少信息。

### 4. 用这一行权重汇总所有 value

$
O=AV,\qquad
\mathbf o_i=\sum_j A_{ij}\mathbf v_j
$

所以 $QK^\top$ 只负责决定“权重是多少”，真正被取出并混合的是 $V$。attention matrix 的一个格子是标量，而输出 $\mathbf o_i$ 是一个 $d_k$ 维向量。

### 5. 多个头各算一套，再合并

每个头都有自己的投影、score matrix 和 attention weights，因此可以学到不同的匹配方式。所有头的输出先拼接，再经过 $W_O$ 投影回模型维度。

整条数据流可以压缩成一句：

$
X\xrightarrow{W_Q,W_K,W_V}(Q,K,V)
\xrightarrow{QK^\top/\sqrt{d_k}}S
\xrightarrow{+M,\,\text{row-softmax}}A
\xrightarrow{AV}O
$

也就是：**三个投影产生 Q/K/V；Q 和 K 生成 token 两两之间的权重；mask 删除不允许的信息路径；每行 softmax 归一化；最后用这些权重对 V 求加权和。**

> MHA 里的 softmax 确实提供了非线性，但它主要在 token 维度上决定“和谁通信”。FFN 的激活函数则在每个 token 的特征维度上做非线性变换；两者作用不同。

## 单个头在算什么

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

拆到单个 query $\mathbf{q}_i$ 看：

$$\alpha_{ij} = \frac{\exp(\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d_k})}{\sum_{j'}\exp(\mathbf{q}_i^\top \mathbf{k}_{j'}/\sqrt{d_k})}, \qquad \mathbf{o}_i = \sum_j \alpha_{ij}\mathbf{v}_j$$

每一行 $\alpha_{i\cdot}$ 加起来是 1。所以输出永远是 value 的凸组合——**注意力不创造新信息，它只决定从哪里搬**。

## 为什么非要除以 $\sqrt{d_k}$

面试最爱问这个，答案不是「经验值」。

设 $q, k$ 各分量独立、均值 0、方差 1，那么

$$\text{Var}(\mathbf{q}^\top\mathbf{k}) = \sum_{i=1}^{d_k}\text{Var}(q_i k_i) = d_k$$

标准差是 $\sqrt{d_k}$。$d_k = 64$ 时点积典型量级已经到 $\pm 8$，softmax 在这个尺度上接近 one-hot。而 softmax 的雅可比是

$$\frac{\partial\,\text{softmax}(z)_i}{\partial z_j} = \alpha_i(\delta_{ij}-\alpha_j)$$

当某个 $\alpha_i \to 1$、其余 $\to 0$ 时，整个雅可比趋近零矩阵，**梯度消失**。除以 $\sqrt{d_k}$ 把方差拉回 1。

⚠️ 除的是 $\sqrt{d_k} = \sqrt{d_\text{head}}$，**不是** $\sqrt{d_\text{model}}$。手写时很容易顺手写成后者。

## 为什么要多头：不是为了把维度做大

<div class="bilingual-note bilingual-intro">
  <span>逐概念双语 · CONCEPT-BY-CONCEPT</span>
  <p>下面三张卡默认中文；点 <strong>English ↻</strong> 可在原位置查看完整英文。</p>
</div>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 1. 多头的核心：多套注意力关系

假设 $d_{\text{model}}=512$。一个完整维度的单头会计算

$$A=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{512}}\right),\qquad O=AV.$$

关键限制不是“512 维不够”，而是所有 value 通道共享同一套注意力矩阵 $A$。处理
“小明把书送给小红，因为她很喜欢阅读”中的“她”时，模型可能同时需要追踪指代、
语法依赖、语义角色和局部邻近；单头必须把这些关系压进一套分布。

多头让第 $i$ 个头学习自己的投影和权重：

$$Q_i=XW_i^Q,\qquad K_i=XW_i^K,\qquad V_i=XW_i^V,$$

$$A_i=\operatorname{softmax}\!\left(\frac{Q_iK_i^\top}{\sqrt{d_k}}\right).$$

于是模型得到 $A_1,\ldots,A_h$ 多套读取方式。某些头可能偏向指代，另一些偏向
邻近或语法，但这些职责不是人工指定的，也可能彼此重叠。更准确的结论是：
**不同特征可以使用不同的注意力权重，不必全部共享一套分布。**

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">1. The core purpose: multiple attention relations</div>

Suppose $d_{\text{model}}=512$. One full-width attention head computes

$$A=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{512}}\right),\qquad O=AV.$$

The main limitation is not that 512 dimensions are insufficient. It is that every
value channel shares the same attention matrix $A$. Resolving a pronoun may require
coreference, syntactic dependency, semantic role, and local-neighborhood signals at
the same time; one head must compress all of them into one distribution.

Head $i$ instead learns its own projections and weights:

$$Q_i=XW_i^Q,\qquad K_i=XW_i^K,\qquad V_i=XW_i^V,$$

$$A_i=\operatorname{softmax}\!\left(\frac{Q_iK_i^\top}{\sqrt{d_k}}\right).$$

The model therefore obtains $A_1,\ldots,A_h$: several ways to read the sequence.
Some heads may emphasize coreference, locality, or syntax, but those jobs are not
assigned by hand and can overlap. The precise advantage is that **different feature
groups can use different attention weights instead of sharing one distribution.**

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 2. 拆分维度是在控制预算

原版使用 $d_{\text{model}}=512,h=8$，通常令

$$d_k=d_v=\frac{512}{8}=64,$$

所以 $8\times64=512$。如果 8 个头都保留完整 512 维，参数和计算会大幅增长；
把总宽度拆开，才能在接近单头的预算下得到 8 套关系。

单头完整投影有

$$W_Q,W_K,W_V\in\mathbb{R}^{512\times512}.$$

多头每组投影是 $512\times64$，8 组合计仍为

$$8\times(512\times64)=512\times512.$$

因此标准 MHA 的 Q/K/V 和输出投影总参数量约为 $4d_{\text{model}}^2$，与头数本身
无关。代码也通常只做一次大投影，再 reshape 成 `(B, H, T, d_head)`；不是顺序执行
8 次小模型。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">2. Splitting dimensions controls the budget</div>

The original model uses $d_{\text{model}}=512$ and $h=8$, usually with

$$d_k=d_v=\frac{512}{8}=64,$$

so $8\times64=512$. Giving all eight heads the full 512 dimensions would multiply
parameters and compute. Splitting a fixed total width yields eight attention
relations at roughly the budget of one full-width head.

A full-width projection has

$$W_Q,W_K,W_V\in\mathbb{R}^{512\times512}.$$

Eight $512\times64$ head projections contain the same total number of elements:

$$8\times(512\times64)=512\times512.$$

Standard MHA therefore has about $4d_{\text{model}}^2$ parameters across Q, K, V,
and the output projection, independent of head count. Implementations perform one
large projection and reshape to `(B, H, T, d_head)` rather than running eight small
models sequentially.

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 3. 表达能力不等于泛化保证

多头首先增加的是表达能力：它允许多种 token 关系、匹配函数和上下文摘要并存。
更好的表示有时会改善未见数据上的表现，但“用了多头”并不自动推出 generalization
更好。

头数过多时可能出现每头维度太小、多个头功能重复、参数利用率低，甚至过拟合。
实践中经常可以剪掉部分头而几乎不损失性能。所以：

$$\boxed{\text{多头不是为了把维度做大，而是在相近成本下获得多套注意力关系。}}$$

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

$$\boxed{\text{Multi-head attention obtains multiple relations at similar cost; it does not enlarge width for its own sake.}}$$

“Different representation subspaces” should not be over-interpreted either. Separate
parameters let heads learn different matching functions, but no head is guaranteed
to have one clean, human-assigned semantic job.

</div>
</section>

## 手搓时最容易错的六处

这些是真正会被看出来的地方：

| # | 坑 | 正确做法 |
| --- | --- | --- |
| 1 | reshape 顺序 | `view(B,T,H,dh).transpose(1,2)`，**不是** `view(B,H,T,dh)` |
| 2 | mask 时机 | 在 softmax **之前**加，不是之后置零 |
| 3 | mask 的值 | 填 $-\infty$，填 0 等于「概率相等」 |
| 4 | softmax 数值稳定 | 先减去每行最大值 |
| 5 | 除的维度 | $\sqrt{d_\text{head}}$，不是 $\sqrt{d_\text{model}}$ |
| 6 | 合头 | `transpose(1,2).contiguous().view(...)`，少了 `contiguous()` 会报错 |

**第 1 条为什么必须这样。** 投影输出是 `(B, T, d_model)`，其中 `d_model` 这一维是 $h$ 个头**首尾相接**排列的。所以要先把最后一维拆成 `(H, dh)`，再把 `H` 挪到前面。直接 `view(B,H,T,dh)` 会横跨时间维乱切，得到的每个「头」是一堆不相干位置的碎片——形状对，数值全错，而且不报错。

**第 3 条的直觉。** softmax 之后再把某些位置置零，剩下的权重加起来就不是 1 了；而且被屏蔽的位置在 softmax 时已经分走了概率质量。填 $-\infty$ 才是「这条路根本不存在」。

## 动手：三种写法，互相对答案

<details class="code-drop" markdown="1">
<summary><b>手搓</b> · 纯 NumPy，不依赖任何框架</summary>

白板上要能默出来的就是这一版。

```python
import numpy as np

def softmax(x, axis=-1):
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
        att = att.softmax(dim=-1)

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)   # 合头
        return self.wo(y), att

def causal_mask(t, device=None):
    return torch.triu(torch.ones(t, t, dtype=torch.bool, device=device), diagonal=1)
```

上线时把中间三行换成一句 `F.scaled_dot_product_attention(q, k, v)` 就行——它会自动选 FlashAttention 之类的融合 kernel，省掉 $T \times T$ 那个中间矩阵的显存。

</details>

[`../code/attention_torch.py`](../code/attention_torch.py) 把**四种实现**跑在同一组权重上并互相对答案：

```
  from-scratch vs F.scaled_dot_product_attention : 1.19e-07
  from-scratch vs nn.MultiheadAttention          : 1.19e-07
  from-scratch torch vs pure NumPy               : 1.19e-07
  every attention row sums to 1                  : 1.19e-07
  weight on any future position                  : 0.00e+00
```

能让手搓版和 `nn.MultiheadAttention` 对上，比「能跑」强得多。对不上时差异出在哪，本身就是最好的调试练习——顺带说，`nn.MultiheadAttention` 把 $W_Q, W_K, W_V$ 存成一个拼起来的 `in_proj_weight`，而 `nn.Linear` 存的是 $(out, in)$ 所以搬去 NumPy 要转置。

## 面试可能会问

<details class="interview" markdown="1">
<summary>多头注意力有多少参数？</summary>

四个 $d_\text{model} \times d_\text{model}$ 矩阵，$4d^2$（不含 bias）。和**头数无关**——切头只是把同样的参数重新分组。

</details>

<details class="interview" markdown="1">
<summary>时间和显存复杂度是多少？</summary>

时间 $O(T^2 d)$；注意力矩阵显存 $O(hT^2)$。长上下文的瓶颈是后者，这正是 FlashAttention 要消掉的东西——它分块计算，从不把 $T\times T$ 矩阵写进显存。

</details>

<details class="interview" markdown="1">
<summary>Q 和 K 为什么用两个不同的矩阵，不能共享？</summary>

共享的话 $\mathbf{q}_i^\top\mathbf{k}_j$ 就对称了，「A 该关注 B」会被迫等于「B 该关注 A」。而语言里的关系大多不对称——形容词修饰名词，反过来不成立。

</details>

<details class="interview" markdown="1">
<summary>V 为什么不参与打分？</summary>

打分决定「取多少」，V 决定「取什么」。混在一起会让内容影响自己的被取概率，容易退化。

</details>

<details class="interview" markdown="1">
<summary>多头会不会退化成一个头？</summary>

会，而且经常部分退化。多个头收敛到相近的注意力分布是已知现象，推理时剪掉大部分头影响也不大。所以「头多」本身不是优点，头之间是否**互补**才是。

</details>

<details class="interview" markdown="1">
<summary>mask 为什么要在 softmax 之前加？填 0 行不行？</summary>

不行。softmax 之后再置零，剩下的权重加起来就不是 1；而且被屏蔽的位置在 softmax 时已经分走了概率质量。填 $-\infty$ 才等价于「这条路不存在」——$e^{-\infty}=0$，归一化时它压根不进分母。

</details>

## 自检

<div class="taste-check">
  <strong>如果真的理解了，你应该能解释：</strong>
  <ol>
    <li>不看代码，写出缩放点积注意力，并说清 mask 为什么必须在 softmax 之前？</li>
    <li>为什么除的是 $\sqrt{d_\text{head}}$ 而不是 $\sqrt{d_\text{model}}$？不除会怎样？</li>
    <li><code>view(B,T,H,dh).transpose(1,2)</code> 和 <code>view(B,H,T,dh)</code> 的结果差在哪？为什么后者不报错却全错？</li>
    <li>多头把参数量变成几倍？</li>
  </ol>
</div>

## 继续读

注意力有了，但它对顺序完全不敏感，而且堆深了就训不动。先看[残差连接](residual-connections.md)和[归一化](normalization.md)怎么让深度变得可行，再回到[原版 Transformer](vanilla-transformer.md) 把整块拼起来。
