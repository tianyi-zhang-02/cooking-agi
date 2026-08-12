# Transformer 架构

**中文** · [English](transformer.en.md)

## 先用一句话讲清楚

Transformer 就是[上一页](from-linear-to-neural.md)那个「学出来的坐标变换 $\phi$」的一种具体做法：**注意力负责跨位置搬运信息，FFN 负责在单个位置上加工**，两者交替堆叠，最后仍然是一个线性分类器读出答案。

## 用做菜来理解

- **注意力**：每道工序前，先环顾整个案板，决定这一步该从哪几样食材取味。
- **FFN**：拿到取来的味道之后，在自己这一格里加工。
- **残差连接**：每一步都保留原样的一份，改动是叠加上去的，不是推倒重来。
- **堆 N 层**：反复「环顾—加工」，直到答案浮出来。

## 核心：缩放点积注意力

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

拆开看单个查询 $\mathbf{q}_i$：

$$\alpha_{ij} = \frac{\exp\!\big(\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d_k}\big)}{\sum_{j'} \exp\!\big(\mathbf{q}_i^\top \mathbf{k}_{j'} / \sqrt{d_k}\big)}, \qquad \mathbf{o}_i = \sum_j \alpha_{ij}\, \mathbf{v}_j$$

也就是：**用相似度当权重，对 value 做加权平均**。$\alpha_{ij}$ 每一行加起来是 1。

### 为什么要除以 $\sqrt{d_k}$

假设 $q$ 和 $k$ 的每个分量独立、均值 0、方差 1，那么

$$\mathbb{E}[\mathbf{q}^\top\mathbf{k}] = 0, \qquad \text{Var}(\mathbf{q}^\top\mathbf{k}) = \sum_{i=1}^{d_k}\text{Var}(q_i k_i) = d_k$$

即标准差是 $\sqrt{d_k}$。$d_k = 64$ 时点积的典型量级就有 $\pm 8$，$d_k = 128$ 时到 $\pm 11$。softmax 在这种量级上已经接近 one-hot，而它的雅可比是

$$\frac{\partial\, \text{softmax}(z)_i}{\partial z_j} = \alpha_i(\delta_{ij} - \alpha_j)$$

当某个 $\alpha_i \to 1$、其余 $\to 0$ 时，整个雅可比趋近于零矩阵——**梯度消失**。除以 $\sqrt{d_k}$ 把方差拉回 1，softmax 待在有梯度的区域。

### Python：注意力本体

```python
import torch, torch.nn.functional as F

def attention(q, k, v, mask=None):
    """q: (B, H, Tq, d)   k, v: (B, H, Tk, d)   mask: True = 屏蔽"""
    scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5   # (B, H, Tq, Tk)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn = scores.softmax(dim=-1)                          # 每行和为 1
    return attn @ v, attn
```

`-inf` 而不是 0：屏蔽要发生在 softmax **之前**，否则被屏蔽的位置仍会分到概率质量。

## 同一个模块，三种用法

这是原论文（2017）的 encoder-decoder 结构里最该盯住的地方。`self_attn(x, x, x)` 和 `cross_attn(x, memory, memory)` 是同一个类，只是喂进去的三个张量不同：

| 用法 | Q 来自 | K, V 来自 | mask | 注意力形状 |
| --- | --- | --- | --- | --- |
| encoder 自注意力 | src | src | 只挡 padding，**双向** | $(B,h,S,S)$ |
| decoder 自注意力 | tgt | tgt | padding **∨** 因果 | $(B,h,T,T)$ |
| **交叉注意力** | **tgt** | **memory** | 挡 src 的 padding | $(B,h,T,S)$ ← 非方阵 |

交叉注意力是两座塔唯一接触的地方：decoder 每生成一步，就拿当前状态当查询去 encoder 的输出里查一次。

[`code/vanilla_demo.py`](code/vanilla_demo.py) 训练它做「把序列反过来」，然后把交叉注意力矩阵打出来——正确的对齐是已知的（反对角线），所以能直接检查它学没学对：

```
        1  2  3  4  5  6  7  8   <- source (encoder)
  BOS                        @
    1                     @
    2                  @
    3               @
    4            #  :
    5         @
    6      @
    7   @
  ^ decoder step
```

64/64 序列完全正确。

## 多头：为什么不是一个大注意力

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O, \quad \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

单个注意力只能算一种相似度、输出一个加权平均。切成 $h$ 份、每份 $d_k = d_{\text{model}}/h$ 维之后，不同的头可以并行关注不同的关系（语法依赖、共指、位置邻近）。总计算量不变。

实现上不需要 $h$ 组小矩阵：用一个 $(d_{\text{model}}, d_{\text{model}})$ 的投影再 **reshape** 成 $h$ 个头，数学上等价，但只有一次 GEMM。

```python
q = self.w_q(x).view(B, T, h, d_k).transpose(1, 2)   # (B, T, C) -> (B, h, T, d_k)
# ... attention ...
y = out.transpose(1, 2).reshape(B, T, h * d_k)       # 拼回去
```

## post-norm 与 warmup 的绑定关系

论文写的是

$$\mathbf{x} \leftarrow \text{LayerNorm}\big(\mathbf{x} + \text{Sublayer}(\mathbf{x})\big)$$

**层归一化在残差加法外面**（post-norm）。今天所有实现都改成了

$$\mathbf{x} \leftarrow \mathbf{x} + \text{Sublayer}\big(\text{Norm}(\mathbf{x})\big)$$

差别不是风格。post-norm 把 norm 压在残差高速路上，堆 6 层之后早期梯度会炸，所以原论文的 Noam 调度

$$\text{lr}(t) = d_{\text{model}}^{-0.5} \cdot \min\big(t^{-0.5},\; t \cdot t_{\text{warmup}}^{-1.5}\big)$$

不是调参技巧，而是**训练能不能启动的前提**。pre-norm 之后梯度有一条完全不经过 norm 的通路，大家才敢用简单的常数 lr + 短 warmup。

> 实测提醒：`LambdaLR` 是拿 **base_lr 乘** lambda 的。把 Adam 的 `lr` 设成 0 再挂 Noam 调度，学习率会永远是 0，而 loss 因为 dropout 噪声看起来还在动。这个坑很常见。

## 位置编码：从正弦到 RoPE

注意力本身对顺序**完全不敏感**——打乱输入的顺序，输出只是跟着打乱。位置信息必须显式注入。

### 原版：固定正弦

$$PE_{(pos,\, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \qquad PE_{(pos,\, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

**加**到 embedding 上（不是拼接）。波长从 $2\pi$ 到 $10000\cdot 2\pi$ 排成等比数列，相当于一个多尺度的时钟。选它的理由：$PE_{pos+k}$ 是 $PE_{pos}$ 的线性函数，模型可以学相对偏移。

### 现在：RoPE

不加到输入上，而是在**每一层的 $q$ 和 $k$ 上做旋转**。把 $\mathbf{q}$ 的通道两两配对成复数，位置 $m$ 处旋转角 $m\theta_i$：

$$\tilde{\mathbf{q}}_m = R_m \mathbf{q}, \qquad R_m = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \ \ (\text{每个通道对})$$

关键性质：旋转矩阵满足 $R_m^\top R_n = R_{n-m}$，于是

$$\langle R_m\mathbf{q},\; R_n\mathbf{k}\rangle = \mathbf{q}^\top R_m^\top R_n \mathbf{k} = \mathbf{q}^\top R_{n-m}\mathbf{k}$$

注意力分数**只依赖相对距离 $n-m$**，绝对位置自动消掉。$\mathbf{v}$ 不旋转——它不该带位置信息。

```python
def apply_rope(x, cos, sin):
    """x: (B, H, T, d)   cos/sin: (T, d/2)"""
    x1, x2 = x.chunk(2, dim=-1)                     # split-half（Llama 约定）
    return torch.cat([x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos], dim=-1)
```

⚠️ 配对方式有两种：split-half（通道 $i$ 配 $i + d/2$，GPT-NeoX/Llama）和交错（原 RoPE 论文）。两者差一个通道置换，**权重不能互换**——转模型时这是经典踩坑点。

## 现代 decoder-only 还改了什么

| | vanilla (2017) | 现在 |
| --- | --- | --- |
| 结构 | encoder + decoder | 只有 decoder |
| 归一化 | post-norm LayerNorm | pre-norm RMSNorm |
| 位置 | 正弦，加在输入上 | RoPE，每层在 q/k 上旋转 |
| 注意力 | MHA（$h$ 个头各有 K/V） | GQA（K/V 头更少） |
| FFN | ReLU，$4d$ | SwiGLU，$\tfrac{8}{3}d$ |
| 推理 | 逐步重算 | KV cache |

**RMSNorm** 去掉了减均值那一步，也没有偏置：

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot \boldsymbol{\gamma}$$

省一次归约，效果不掉。平方和必须用 fp32 累加，否则 bf16 训练时方差会烂掉。

**SwiGLU** 用门控换掉 ReLU：

$$\text{SwiGLU}(\mathbf{x}) = \big(\text{SiLU}(W_g\mathbf{x}) \odot W_u\mathbf{x}\big)W_d, \qquad \text{SiLU}(z) = z\cdot\sigma(z)$$

它有**三个**矩阵而不是两个，所以隐藏维取 $\frac{8}{3}d$ 而非 $4d$，参数量才对得上。

**GQA** 让每组 $n_{\text{rep}}$ 个查询头共享一组 K/V 头。KV cache 的大小是

$$2 \cdot n_{\text{layer}} \cdot n_{\text{kv}} \cdot d_{\text{head}} \cdot T \cdot \text{sizeof(dtype)}$$

把 $n_{\text{kv}}$ 从 32 降到 8 就直接省下 4 倍显存——长上下文推理时这是主要瓶颈。

## 手搓一遍，并验证它是对的

[`code/`](code/) 里两版实现都不调 `nn.MultiheadAttention` 和 `F.scaled_dot_product_attention`，只用 `nn.Linear` 和裸张量运算：

- [`vanilla.py`](code/vanilla.py) —— 2017 原版 encoder-decoder，含 Noam 调度和标签平滑
- [`model.py`](code/model.py) —— 现代 decoder-only（RMSNorm + RoPE + GQA + SwiGLU + KV cache）

从零实现最容易错的四个地方，[`test_model.py`](code/test_model.py) 逐个验证：

```
  因果性：      改第 9 个 token，前 8 个位置 logits 差 0.0（严格 0）
  KV cache：    增量解码 vs 一次性前向，max 误差 3.6e-07
  RoPE 相对性： score(5,2) = score(20,17) = +5.6092，score(20,10) = +0.4579
  初始 loss：   4.19  vs  ln(V) = 4.16
```

带 cache 时最容易翻车的是 mask：query 的绝对位置是 `cache.pos + i`，key 从 0 数到 `cache.pos + T - 1`，所以 mask 是**非方阵**的 $(T, S)$；RoPE 的 cos/sin 也得从 `cache.pos` 切片。而且 `cache.pos` 每层前向只推进**一次**（在层循环之后），写在 `update()` 里会翻 $n_{\text{layer}}$ 倍。

## 从哪里继续读

- [从线性模型到神经网络](from-linear-to-neural.md) —— 为什么最后一层永远是线性分类器
- [Post-Training](../05-post-training/) —— 这些参数后来怎么被继续改
- [系统总览](../06-systems/) —— 它在整套系统里的位置

## 起始论文

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原版
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — pre-norm 为什么能去掉 warmup
- [RoFormer](https://arxiv.org/abs/2104.09864) — RoPE
- [GQA](https://arxiv.org/abs/2305.13245) — 分组查询注意力
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU
