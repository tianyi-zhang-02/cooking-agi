# 多头注意力：手搓版

**中文** · [English](multi-head-attention.en.md)

> 阅读时间：约 9 分钟 · 难度：必修 · 最近审阅：2026-08

<div class="lesson-recipe">
  <div><span>这节要做什么</span><strong>让每个位置去序列里取它需要的信息</strong></div>
  <div><span>手里的食材</span><strong>三个投影 $W_Q, W_K, W_V$ · 一个输出投影 $W_O$</strong></div>
  <div><span>核心火候</span><strong>缩放点积 · 切头 · mask 在 softmax 之前</strong></div>
  <div><span>最容易翻车</span><strong>reshape 的顺序、mask 的时机、除错了维度</strong></div>
</div>

## 先尝一口：查字典，然后加权平均

每个位置拿着自己的**问题**（query）去问所有位置的**索引**（key）。对得越上，就从那个位置的**内容**（value）里取越多。取回来的是一个加权平均。

多头的意思是：同一句话同时问好几个不同的问题——一个头盯语法搭配，一个头盯指代，一个头盯位置邻近。问完各自取一份，再拼起来。

## 第一勺：单个头在算什么

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

## 为什么要多头

单个注意力只能算一种相似度、产出一个加权平均。切成 $h$ 份之后：

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V), \quad \text{MultiHead} = \text{Concat}(\text{head}_1..\text{head}_h)W^O$$

每份 $d_k = d_\text{model}/h$ 维，**总计算量不变**——切头是把同样的预算分给几个不同的问题，不是加预算。

实现上不需要 $h$ 组小矩阵：用一个 $(d_\text{model}, d_\text{model})$ 的投影再 reshape 成 $h$ 个头，数学上等价，但只有一次 GEMM。

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

## 出锅检查

<div class="taste-check">
  <strong>如果真的理解了，你应该能解释：</strong>
  <ol>
    <li>不看代码，写出缩放点积注意力，并说清 mask 为什么必须在 softmax 之前？</li>
    <li>为什么除的是 $\sqrt{d_\text{head}}$ 而不是 $\sqrt{d_\text{model}}$？不除会怎样？</li>
    <li><code>view(B,T,H,dh).transpose(1,2)</code> 和 <code>view(B,H,T,dh)</code> 的结果差在哪？为什么后者不报错却全错？</li>
    <li>多头把参数量变成几倍？</li>
  </ol>
</div>

## 下一道菜

注意力有了，但它对顺序完全不敏感，而且堆深了就训不动。先看[残差连接](residual-connections.md)和[归一化](normalization.md)怎么让深度变得可行，再回到[原版 Transformer](vanilla-transformer.md) 把整块拼起来。
