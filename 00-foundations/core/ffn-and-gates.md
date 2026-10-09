# FFN 与 SwiGLU：Attention 之后还在算什么？

**中文** · [English](ffn-and-gates.en.md)

> 最近审阅：2026-10 · 前置：[多头注意力](multi-head-attention.md)、[残差连接](residual-connections.md)

画 Transformer 时，attention 往往占了最显眼的位置，FFN 却只是旁边一个小方框。但这个方框通常有不少参数和计算。它不是“再看一遍上下文”，而是对每个位置已经拿到的表示做非线性变换。

下面只看一个 token，省略 batch、bias 和 norm，使用列向量约定。

## 为什么先升维，再降回来？

普通两层 FFN 可以写作

$$
z=W_{\mathrm{up}}x,\qquad
y=W_{\mathrm{down}}\phi(z),
$$

其中 $x\in\mathbb R^d$，$W_{\mathrm{up}}\in\mathbb R^{h\times d}$，$W_{\mathrm{down}}\in\mathbb R^{d\times h}$。第一步构造更多中间特征，非线性改变特征组合，最后映射回 residual stream 的维度。

如果去掉 $\phi$，两个矩阵可以合成一个矩阵，整体还是线性变换。增大中间宽度并不会凭空产生非线性能力。

不同 token 使用同一组 FFN 权重，但在普通 dense FFN 内彼此不直接交换信息。上下文已经通过之前的 attention 进入当前向量。“逐位置计算”不等于“没有上下文”。

## SwiGLU 多了一条会随输入变化的通路

一种无 bias 的写法是：

$$
g=W_gx,\quad u=W_ux,\quad
h=\operatorname{SiLU}(g)\odot u,\quad
y=W_dh,
$$

$$
\operatorname{SiLU}(z)=z\sigma(z).
$$

两条投影分别产生门控分支和内容分支，再逐元素相乘。这里的 gate 不是概率，也不一定在 0 到 1 之间：SiLU 可能为负，也可能大于 1。不要把它理解成“选中某个 expert”的离散路由。

[GLU Variants](https://arxiv.org/abs/2002.05202)比较了这些门控 FFN 形式。SwiGLU 是其中一种设计；效果还依赖预算和训练，不能只把激活函数名字换掉就套用论文结论。

## 跟着两个中间特征算一次

为了单独看门控，假设投影后 $g=[0,1]$，$u=[3,2]$：

| 中间维度 | gate 分支 | SiLU | 内容分支 | 相乘结果 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 3 | 0 |
| 2 | 1 | 0.7311 | 2 | 1.4621 |

若 $W_d=[1,-1]$，则标量输出约为 -1.4621。这个小例子把输出维度缩成 1，真实 block 会回到 $d$。

```python
import math

def silu(value):
    return value / (1 + math.exp(-value))

gate = [0.0, 1.0]
content = [3.0, 2.0]
hidden = [silu(gate_value) * content_value
          for gate_value, content_value in zip(gate, content)]
output = hidden[0] - hidden[1]
assert math.isclose(output, -1.4621171572600098)
```

第一维输出为 0，不代表这一维永远收不到训练信号。设上游梯度为 $\delta$，则

$$
\frac{\partial\mathcal L}{\partial u}
=\delta\odot\operatorname{SiLU}(g),\qquad
\frac{\partial\mathcal L}{\partial g}
=\delta\odot u\odot\operatorname{SiLU}'(g).
$$

因为 $\operatorname{SiLU}'(0)=1/2$，当 $u=3,\delta=1$ 时，gate 梯度为 1.5。内容分支这一刻的梯度为 0，gate 仍能学习。这比把门想象成永久开关更准确。

## 三个矩阵，参数预算怎么比？

忽略 bias，普通 FFN 有 $2dh$ 个参数，SwiGLU 有 $3dh_g$。如果普通 FFN 的宽度是 $4d$，相同矩阵参数预算下：

$$
2d(4d)=3dh_g\quad\Rightarrow\quad h_g=\frac83d.
$$

例如 $d=12$：普通宽度 48，门控宽度 32，二者都是 1152 个参数。真实实现还会为硬件对齐调整宽度。

因此“换成三个矩阵”不必然增加总参数；反过来，若保持中间宽度相同，它确实更贵。比较方法要交代宽度、参数量和计算预算，否则可能把更多算力误当成结构优势。

## 从 dense FFN 到 MoE，哪件事又变了？

SwiGLU 的门在特征维度上做连续调制。MoE 的 router 则为 token 选择哪些 expert FFN 执行；expert 内部也可以使用 SwiGLU。这是两层不同的选择。

| 机制 | 处理什么 | 主要限制 |
| --- | --- | --- |
| Attention | 位置之间的信息组合 | 长度、缓存与读写成本 |
| Dense FFN / SwiGLU | 当前 token 的通道变换 | 中间激活、矩阵计算 |
| MoE routing | 分配 token 到部分 expert | 负载、通信与路由稳定性 |

想继续读，去 [MoE 专题](../moe/README.md)。写一个 FFN 的检查则很直接：核对 shape、确认非线性位置、对小输入做数值梯度检查，再比较等预算下的任务质量和耗时。
