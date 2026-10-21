# Gated Attention：算完注意力，再决定用多少

**中文** · [English](gated-attention.en.md)

普通 attention 把读到的信息交给后面的层。Gated Attention 多问了一句：这次读到的东西，需要原样传下去吗？**它给 attention 的输出加一个可学习的门。**这里讨论的是 2025 年论文中的 attention-output gating（[Qwen3-Next 和 Qwen3.5](../model-families/qwen.md#qwen3-next-boundary) 每 3 层 Gated DeltaNet 配 1 层这种带门的 attention），不是 MoE 选专家，也不是 Gated DeltaNet 更新循环状态。

## 门放在哪里？

先回忆一个 head 的计算。输入是 $X$，投影后得到 $Q_h,K_h,V_h$：

$$
O_h=\operatorname{softmax}\left(\frac{Q_hK_h^\top}{\sqrt{d_h}}+M\right)V_h.
$$

$M$ 是 causal / padding mask。本文关注的门放在 $O_h$ 后、输出投影 $W_O$ 前：

$$
G_h=\sigma(XW_{g,h}),\qquad
Y=\operatorname{Concat}_h(G_h\odot O_h)W_O.
$$

headwise gate 给每个 token、每个 head 一个数；elementwise gate 则给 head 内每个维度一个数。不要把 headwise 理解成“整个训练过程每个 head 只有一个固定开关”：门仍取决于当前 token 的表示。

```text
输入 ─→ Q/K/V ─→ SDPA ─→ × gate ─→ 拼接各个 head ─→ 输出投影
  └────────────→ sigmoid ──┘
```

各个门使用 sigmoid，互相独立，不需要加起来等于 1。与 softmax 竞争不同，所有 head 都可以少传一点，也可以同时多传。

## 两个 head，手算一次

假设两个 head 已经读出了 `[4, -2]` 和 `[1, 3]`，当前 token 对应的门是 0.1 和 0.5。

| 阶段 | Head 1 | Head 2 |
| --- | --- | --- |
| SDPA 输出 | `[4, -2]` | `[1, 3]` |
| 乘门以后 | `[0.4, -0.2]` | `[0.5, 1.5]` |

两个门加起来是 0.6，不是 1：每个 head 的门单独决定，一路调小，不会让另一路自动变大。第一路也没有消失，只是影响变小。接下来的 $W_O$ 仍会混合各个 head，所以不能把某个 head 的门值直接解释成最终答案的置信度。

```python
import math

def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)

def gated_heads(head_outputs, gate_logits):
    if not head_outputs or len(head_outputs) != len(gate_logits):
        raise ValueError("One logit per head is required")
    if any(not head or any(not math.isfinite(value) for value in head) for head in head_outputs):
        raise ValueError("Expected nonempty finite head outputs")
    if any(not math.isfinite(value) for value in gate_logits):
        raise ValueError("Expected finite logits")
    return [[sigmoid(logit) * value for value in head]
            for head, logit in zip(head_outputs, gate_logits)]

result = gated_heads([[4, -2], [1, 3]], [math.log(1 / 9), 0])
assert math.isclose(result[0][0], 0.4)
assert math.isclose(result[1][1], 1.5)
assert gated_heads([[4, -2]], [0]) == [[2.0, -1.0]]
```

最后一行很重要：零初始化 gate 的 logits，得到的是 **0.5 倍输出，不是原样通过**。不能在现成 checkpoint 上随手加个零初始化门，就认为保持了原模型行为。

## 为什么可能有用，又不保证什么？

从函数看，$G(X)\odot O(X)$ 增加了输入相关的乘性非线性。模型可以保留“读到了什么”，同时学习“要让它影响后续计算多少”。但“门很小”不是“模型没有计算这一路”：SDPA 已经执行了，因此这本身不是稀疏计算加速。

从优化看，门引入了新参数与新梯度。sigmoid 的导数是 $g(1-g)$，在 0.5 附近最大；接近 0 或 1 时变小。把门的 logits 初始化得很大、让门值接近 1 来近似 identity，也可能让门难学。

再看 attention sink。softmax 的权重必须加起来等于 1，所以一个 head 就算这次没什么可读，也得把这 1 份注意力分出去。论文引言引用的前人解释是：模型会把多出来的权重堆到某个 token 上，常见的是第一个 token，它的 value 接近 0。下面是**讲机制用的教学算例，不是论文的数据**：3 个 key 的 value 依次是 0（sink）、4、2。

| 做法 | 分数 | 注意力权重 | SDPA 读出 | 门 | 传下去 |
| --- | --- | --- | --- | --- | --- |
| 无门，靠 sink 少传 | `[3, 0, 0]` | $[0.909, 0.045, 0.045]$ | 0.27 | 无 | 0.27 |
| 有门，注意力照常分散 | `[0, 0, 0]` | $[1/3, 1/3, 1/3]$ | 2 | 0.1 | 0.2 |

两种做法传下去的输出都较小。在这个例子里，没有门的 head 通过提高 sink 的分数，让它获得 $e^3/(e^3+2)\approx0.909$ 的权重，从而减小输出；有门时，可以保持均匀的注意力权重，再由门把读出缩小到十分之一。

[原论文](https://arxiv.org/abs/2505.06708) 在 15B 总参数（2.54B 激活）的 MoE 和 1.7B dense 模型上比较了 30 种门的变体，训练量从 400B 到 3.5T token 不等；效果最好的是放在 SDPA 输出之后、各 head 独立的 elementwise sigmoid 乘性门（[§3.2](https://arxiv.org/html/2505.06708v1#S3.SS2)）。它报告基线各层平均有 46.7% 的注意力落在第一个 token，加这种门后降到 4.8%；作者把原因归于门带来的、随 query 变化的稀疏性（[§4.3](https://arxiv.org/html/2505.06708v1#S4.SS3)）。应把这些看成其训练设置下的证据，而不是“给任意模型加门都能消除 attention sink”。sink 也不能只靠看一张注意力热图判断。

门也有成本：每个 head 多一组投影参数和一次乘法；论文的 dense 实验为此缩窄了 FFN，让总参数量不变（[§3.2.2](https://arxiv.org/html/2505.06708v1#S3.SS2.SSS2)）。换一种架构或规模，门是否还有用，要单独做消融。

## 怎样做一个有意义的对照？

固定数据、训练 token 数、优化器和参数预算，比较无门、headwise 与 elementwise 三组。至少同时记录：

- held-out loss 与下游任务，而不只看 gate 分布好不好看。
- 按层、head、token 类型统计门值，检查是否长期饱和。
- 梯度范数、loss spike、短输入与长输入上的变化。
- prefill / decode 延迟与显存。多一个 gate 不能自动算作加速。

调试时可把门临时固定为 1，确认数值路径与原 attention 对齐；再固定为 0，确认只有该 attention 分支被关掉，residual 仍存在。这是实现检查，不是模型效果实验。

## 和其他“门”分开记

| 机制 | 选择或控制什么 |
| --- | --- |
| 本文 Gated Attention | 已算出的 attention 输出传多少 |
| MoE router | token 交给哪些专家计算 |
| Gated DeltaNet | 循环记忆保留多少、怎样用新信息纠正 |
| Engram gate | 查表得到的局部模式是否适合当前上下文 |

核对日期：2026-10-08。本文算例是局部前向检查，没有复现论文的完整训练。下一篇可以读 [Engram](engram.md)，比较“读历史 token”和“查一个学到的模式表”。
