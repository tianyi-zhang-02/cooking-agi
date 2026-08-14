# Vanilla Transformer

**中文** · [English](vanilla-transformer.en.md)

> 阅读时间：约 8 分钟 · 难度：必修 · 最近审阅：2026-08

<div class="lesson-recipe">
  <div><span>这节要做什么</span><strong>让所有位置并行交换信息，不再排队递归</strong></div>
  <div><span>手里的食材</span><strong>token matrix · position · attention mask</strong></div>
  <div><span>核心火候</span><strong>self-attention · cross-attention · FFN · residual</strong></div>
  <div><span>最容易翻车</span><strong>三处 attention 混淆，以及 causal mask 遮错方向</strong></div>
</div>

## 先尝一口：出去找信息，回来加工信息

先忘掉那张塞满箭头的大框图。Transformer 一层其实只反复做两件事：**attention 去别的位置拿信息，FFN 留在当前位置加工信息。** 原版仍然是 encoder–decoder，但 recurrence 被彻底拿掉了。

## 厨房分工：一层只有两种“工种”

1. **Attention mixing**：不同 token 之间交换信息。
2. **Channel mixing / FFN**：每个 token 独立变换自己的通道。

外面再套 residual connection 与 normalization。堆很多层，本质上还是反复“出去找信息 → 回来加工信息”。这样记，比背整张结构图轻松很多。

## 三口锅：三处 attention 在问不同的问题

| 位置 | Query | Key / Value | mask | 作用 |
| --- | --- | --- | --- | --- |
| encoder self-attention | source | source | padding | 双向理解输入 |
| decoder self-attention | target prefix | target prefix | causal + padding | 只能看已经生成的 token |
| decoder cross-attention | decoder state | encoder states | source padding | 从输入中取当前所需证据 |

scaled dot-product attention：

$$\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V$$

$M$ 是 mask：允许的位置加 0，禁止的位置加 $-\infty$。

## 为什么这套做法突然可以做大

- **训练并行**：所有位置的 $Q/K/V$ 可以一次算出；
- **路径更短**：任意两个 token 一层 attention 就能直接交互；
- **结构统一**：self-attention 与 cross-attention 只是张量来源不同。

代价是 self-attention 的分数矩阵大小为 $T\times T$，标准实现的时间和显存随序列长度近似二次增长。

## 缺的一味：Attention 根本不知道顺序

Attention 本身不知道顺序。原版把固定 sinusoidal position encoding 加到 token embedding：

$$z_t = E[x_t] + PE_t$$

没有位置编码时，模型只能看到一袋 token；调换顺序只会让输出跟着调换。

## 别串菜：2017 原版不是今天的 Decoder-only

| | 2017 vanilla | 现代 decoder-only |
| --- | --- | --- |
| 主体 | encoder + decoder | decoder stack only |
| norm | post-norm LayerNorm | 多为 pre-norm RMSNorm |
| position | sinusoidal | 多为 RoPE |
| attention | MHA | 常见 GQA / MQA |
| FFN | ReLU | 常见 SwiGLU |
| 目标 | 条件序列生成 | next-token prediction |

原版最适合学习三处 attention 的职责；现代版本最适合理解今天语言模型的训练与推理。

<details markdown="1">
<summary><b>进阶</b>：为什么训练能并行，生成仍不能并行</summary>

训练时真实目标序列已知，可以把它整体右移后一次喂给 decoder，causal mask 保证位置 $t$ 看不到未来。生成时第 $t+1$ 个输入正是第 $t$ 个预测，尚未产生，因此只能逐步解码。

</details>

## 动手：别只看图，跑一次

- 快速跑通：[`../code/vanilla_demo.py`](../code/vanilla_demo.py)
- 完整数学与现代组件：[Transformer 架构深拆](../transformer.md)
- 无 PyTorch attention 前向：[`../code/sequence_numpy.py`](../code/sequence_numpy.py)

## 出锅检查

<div class="taste-check">
  <strong>画完结构图后，再问自己：</strong>
  <ol>
    <li>encoder self-attention、decoder self-attention 和 cross-attention 的 Q/K/V 分别来自哪里？</li>
    <li>为什么 Transformer 训练可以并行，生成却仍然逐 token？</li>
    <li>如果拿掉位置编码，模型具体失去了什么信息？</li>
  </ol>
</div>

## 下一道菜

进入 [Decoder-only](decoder-only.md)，看怎样把条件生成、对话、代码与很多推理任务统一为一条 token stream 上的自回归预测。
