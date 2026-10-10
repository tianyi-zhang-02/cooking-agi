# Decoder-only：自回归生成

**中文** · [English](decoder-only.en.md)

> 阅读时间：约 32 分钟 · 难度：必修 · 最近审阅：2026-10-10

模型生成“今天下雨”时，要先有“今天”，才能根据这个前缀决定接下来写什么。训练时完整句子已经给定，所以多个位置的预测可以一起算；生成时后面的字还不存在，只能逐步继续。Decoder-only 的结构、causal mask 和 KV cache，都可以沿着这个区别来理解。

## 所有东西都排进同一条序列 {#_1}

把 instruction、context 和 answer 全都排进同一条 token 序列，用 causal mask 挡住未来，然后每个位置只做一件事：猜下一个 token。这就是 decoder-only 最迷人的地方——结构反而比 encoder–decoder 更统一。

## 一条序列自己就能当训练数据 {#_2}

给定 token 序列 $x_1,\ldots,x_T$：

$$
\mathcal{L}_{\text{LM}}=-\sum_{t=1}^{T-1}\log p_\theta(x_{t+1}\mid x_{\le t})
$$

输入与标签只是错开一位：

```text
input:   [BOS, 今, 天, 天, 气]
target:  [今,  天, 天, 气, 好]
```

每个位置都提供一次监督，因此大规模无标注文本天然能构造训练样本。

## 为什么可以去掉 Encoder {#encoder}

把“输入”和“输出”串在同一条序列里即可：

```text
[system] ... [user] 问题 [assistant] 回答
```

回答 token 能通过 self-attention 看见左侧 prompt；prompt token 不需要看见未来回答。原来 encoder–decoder 的条件关系，被 causal sequence 本身表达了。

这不代表 encoder 没价值。双向表征、分类和部分检索任务仍常使用 encoder；decoder-only 的优势是**一个目标统一预训练、条件生成与对话**。

## 从 Messages 到一轮或多轮生成 {#messages}

<div class="bilingual-note bilingual-intro">
  <span>逐概念双语 · CONCEPT-BY-CONCEPT</span>
  <p>下面两张卡默认中文；点 <strong>English ↻</strong> 可在当前位置查看等价英文。</p>
</div>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 1. 推理：最后一个 Assistant 标记是生成起点 {#1-assistant}

用户提交结构化 messages 后，应用先用 [chat template 与 tokenizer](tokenization.md)
生成带角色边界的 token IDs。典型 prompt 结束在 assistant 起始标记：

```text
<system> You are helpful <end>
<user> 你好吗？ <end>
<assistant>
```

然后模型重复同一循环：预测下一个 token，把它追加回上下文，再预测下一个；直到产生
end-of-message / EOS、命中其他 stop condition，或达到长度上限。

```text
<assistant> → 我 → 很好 → 。 → <end>
```

角色结构没有改变 decoder 的公式。它只是让“现在该由谁继续说”也成为 token context
的一部分；模型仍然执行 causal next-token prediction。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">1. Inference: the final assistant marker is the generation boundary</div>

After the user submits structured messages, the application uses a
[chat template and tokenizer](tokenization.md) to create token IDs with role boundaries.
A typical prompt ends at the assistant-start marker:

```text
<system> You are helpful <end>
<user> How are you? <end>
<assistant>
```

The model then repeats one loop: predict the next token, append it to the context, and
predict again. Generation stops on an end-of-message or EOS token, another configured
stop condition, or a maximum-length limit.

```text
<assistant> → I → am fine → . → <end>
```

Role structure does not change the decoder equation. It makes “whose turn is next” part
of the token context while the model continues ordinary causal next-token prediction.

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 2. 多轮对话只是更长的左侧上下文 {#2}

第二轮生成时，序列通常包含 system、第一轮 user、第一轮 assistant、第二轮 user，
最后再加新的 assistant 起始标记。由于 causal attention 可以读取左侧所有未被截断的
token，新回答能利用此前对话保持连贯。

这不等于模型拥有脱离输入的永久记忆。若历史消息没有重新放进 prompt，模型在当前
forward pass 里就看不到它；如果总长度超过 context window，应用还必须截断、总结，
或通过 retrieval / external memory 选回重要信息。

KV cache 只缓存**本次推理序列**里已计算的 K/V，减少重复计算；它不会自动把一次会话
变成跨会话知识库，也不会替你决定哪些历史值得长期保留。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">2. Multi-turn dialogue is a longer left context</div>

For the second response, the sequence usually contains the system message, first user
turn, first assistant turn, second user turn, and a new assistant-start marker. Causal
attention can read every untruncated token to the left, so the new response can remain
consistent with earlier dialogue.

This is not permanent memory independent of the input. If history is not placed back in
the prompt, the current forward pass cannot see it. When the sequence exceeds the
context window, the application must truncate, summarize, or recover important facts
through retrieval or external memory.

KV cache stores already-computed K/V for the **current inference sequence** to avoid
recomputation. It does not turn one session into a cross-session knowledge base or decide
which history deserves long-term retention.

</div>
</section>

## 现代 Decoder Block：哪些东西真的变了 {#decoder-block}

<div class="bilingual-note bilingual-intro">
  <span>逐概念双语 · CONCEPT-BY-CONCEPT</span>
  <p>下面十张卡默认中文；点 <strong>English ↻</strong> 可在当前位置查看等价英文。</p>
</div>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 1. 现代 Block 的整体结构 {#1-block}

典型的 pre-norm decoder block 可以写成：

$$
\begin{gathered}
U=\operatorname{RMSNorm}(X),\\
H=X+\operatorname{Attention}(U).
\end{gathered}
$$

$$
\begin{gathered}
G=\operatorname{RMSNorm}(H),\\
Y=H+\operatorname{SwiGLU}(G).
\end{gathered}
$$

实际路径是：RMSNorm → Q/K/V projection → 对 Q/K 应用 RoPE → causal
attention → output projection → residual addition → RMSNorm → SwiGLU → 第二次
residual addition。堆完所有 block 后通常还有一次 final norm，再投影到词表 logits。

这不是所有模型的硬性标准，而是一组常见设计。与 2017 原版相比，主体从
encoder–decoder 变为 decoder-only，post-norm 常被 pre-norm 替代，LayerNorm 常被
RMSNorm 替代，正弦位置编码常被 RoPE 替代，FFN 常使用 SwiGLU，注意力头也可能从
MHA 变为 GQA 或 MQA。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">1. The modern block at a glance</div>

A typical pre-norm decoder block can be written as

$$
\begin{gathered}
U=\operatorname{RMSNorm}(X),\\
H=X+\operatorname{Attention}(U).
\end{gathered}
$$

$$
\begin{gathered}
G=\operatorname{RMSNorm}(H),\\
Y=H+\operatorname{SwiGLU}(G).
\end{gathered}
$$

The full path is RMSNorm → Q/K/V projections → RoPE on Q and K → causal attention
→ output projection → residual addition → RMSNorm → SwiGLU → a second residual
addition. A final norm usually follows the entire stack before the vocabulary logits.

This is a common design, not a universal law. Relative to the 2017 encoder–decoder,
modern LLMs are often decoder-only, use pre-norm instead of post-norm, RMSNorm instead
of LayerNorm, RoPE instead of additive sinusoidal positions, SwiGLU instead of a ReLU
FFN, and sometimes GQA or MQA instead of standard MHA.

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 2. RoPE：把位置放进 Q/K 的相对相位 {#2-rope-qk}

原版把 position encoding 加到输入 embedding。RoPE 则先产生 Q/K，再按 token
位置旋转它们。下面 $Q_m,K_n$ 表示取出单个 head 后、转成列向量的位置表示：

$$
\begin{gathered}
Q=XW_Q\\
K=XW_K\\
Q'_m=R_mQ_m\\
K'_n=R_nK_n.
\end{gathered}
$$

二维旋转矩阵是

$$
R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}.
$$

真实向量会把通道两两成对，并让不同通道对使用不同频率。注意力点积满足

$$
\begin{gathered}
(R_mq_m)^\top(R_nk_n)=\\
q_m^\top R_{n-m}k_n,
\end{gathered}
$$

所以分数自然依赖相对距离 $n-m$。通常不旋转 V，因为位置主要影响“从哪里读”，
而不是被读取的内容。RoPE 也不意味着无限长度外推；远超训练长度后仍可能分布失配，
因此长上下文模型会使用频率调整或 RoPE scaling。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">2. RoPE: relative phase in Q and K</div>

The original Transformer adds positional encoding to input embeddings. RoPE first
forms Q and K, then rotates them according to token position. Here $Q_m,K_n$ denote
single-head position vectors written as columns:

$$
\begin{gathered}
Q=XW_Q\\
K=XW_K\\
Q'_m=R_mQ_m\\
K'_n=R_nK_n.
\end{gathered}
$$

The two-dimensional rotation matrix is

$$
R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}.
$$

Real implementations pair channels and use different frequencies across channel
pairs. Their dot product obeys

$$
\begin{gathered}
(R_mq_m)^\top(R_nk_n)=\\
q_m^\top R_{n-m}k_n,
\end{gathered}
$$

so attention scores naturally depend on relative distance $n-m$. V is normally not
rotated because position controls where to read rather than the content being read.
RoPE does not provide unlimited extrapolation: positions far beyond the training
length may still require frequency adjustment or RoPE scaling.

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 3. Pre-Norm 与 RMSNorm：保护残差主干 {#3-pre-norm-rmsnorm}

原版 post-norm 是

$$
Y=\operatorname{LayerNorm}(X+F(X)).
$$

现代 pre-norm 常写成

$$
Y=X+F(\operatorname{Norm}(X)).
$$

后者给残差状态和梯度保留了一条更直接的 identity path，通常更适合训练深层网络；
所有 block 结束后一般再做 final norm。

LayerNorm 会减均值并除以标准差：

$$
\operatorname{LN}(x)=\gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta.
$$

RMSNorm 只控制均方根尺度：

$$
\begin{gathered}
\operatorname{RMSNorm}(x)=\\
\gamma\frac{x}{\sqrt{\frac1d\sum_i x_i^2+\epsilon}}.
\end{gathered}
$$

所以 LayerNorm 调整中心与大小；RMSNorm 通常不减均值、没有 bias，只调整大小。
它计算更简单，但“用了 RMSNorm”本身不代表模型必然更好。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">3. Pre-norm and RMSNorm: protecting the residual stream</div>

The original post-norm form is

$$
Y=\operatorname{LayerNorm}(X+F(X)).
$$

Modern pre-norm blocks commonly use

$$
Y=X+F(\operatorname{Norm}(X)).
$$

Pre-norm preserves a more direct identity path for residual states and gradients,
which usually makes very deep networks easier to train. A final norm is typically
applied after the full block stack.

LayerNorm centers and scales:

$$
\operatorname{LN}(x)=\gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta.
$$

RMSNorm controls only root-mean-square magnitude:

$$
\begin{gathered}
\operatorname{RMSNorm}(x)=\\
\gamma\frac{x}{\sqrt{\frac1d\sum_i x_i^2+\epsilon}}.
\end{gathered}
$$

Thus LayerNorm adjusts center and scale; RMSNorm usually has no mean subtraction or
bias and adjusts only scale. It is simpler to compute, but choosing RMSNorm does not
by itself guarantee a better model.

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 4. SwiGLU：让 FFN 同时生成内容和门 {#4-swiglu-ffn}

本章把一个 token 写成行向量。原版 FFN 是升维 → ReLU → 降维：

$$
\operatorname{FFN}(x)=\operatorname{ReLU}(xW_1)W_2.
$$

现代模型常使用 SwiGLU：

$$
\begin{gathered}
g=\operatorname{SiLU}(xW_{\text{gate}}),\\
u=xW_{\text{up}},\\
\operatorname{SwiGLU}(x)=(g\odot u)W_{\text{down}}.
\end{gathered}
$$

$$
\operatorname{SiLU}(z)=z\sigma(z).
$$

$xW_{\text{up}}$ 产生候选内容，$\operatorname{SiLU}(xW_{\text{gate}})$ 决定每个
特征怎样调制，二者逐元素相乘后再降维。SiLU 不是概率，可以为负，也可以大于 1。
$W_{\text{gate}},W_{\text{up}}\in\mathbb R^{d\times h}$，$W_{\text{down}}\in\mathbb R^{h\times d}$，所以输出仍是 $d$ 维。它比两矩阵 ReLU FFN 多一个投影，因此为了
保持参数预算，隐藏维通常不会继续照搬 $4d$。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">4. SwiGLU: content and a learned gate</div>

This chapter represents each token as a row vector. The original FFN expands, applies ReLU, and projects back down:

$$
\operatorname{FFN}(x)=\operatorname{ReLU}(xW_1)W_2.
$$

Modern models often use SwiGLU:

$$
\begin{gathered}
g=\operatorname{SiLU}(xW_{\text{gate}}),\\
u=xW_{\text{up}},\\
\operatorname{SwiGLU}(x)=(g\odot u)W_{\text{down}}.
\end{gathered}
$$

$$
\operatorname{SiLU}(z)=z\sigma(z).
$$

$xW_{\text{up}}$ produces candidate content, while
$\operatorname{SiLU}(xW_{\text{gate}})$ controls how much of each feature passes.
SiLU is not a probability: it can be negative or exceed 1. With
$W_{\text{gate}},W_{\text{up}}\in\mathbb R^{d\times h}$ and $W_{\text{down}}\in\mathbb R^{h\times d}$,
the elementwise product is projected back to $d$ dimensions. Because SwiGLU uses three
projections rather than two, its hidden width is usually adjusted to keep a similar
parameter budget instead of blindly retaining $4d$.

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 5. MHA、MQA 与 GQA：省的是 KV 宽度 {#5-mhamqa-gqa-kv}

传统 MHA 为每个 query head 保留自己的 K/V head，例如

```text
Q heads: 32   K heads: 32   V heads: 32
```

MQA 让所有 query heads 共享一组 K/V；GQA 则让一组 query heads 共享一组 K/V：

```text
MQA: Q=32, KV=1
GQA: Q=32, KV=8   # 每 4 个 Q heads 共享一组 KV
```

不同 query heads 即使共享 K/V，仍能因 Q 不同而形成不同的 attention distributions。
MQA 最省内存带宽，但可能损失表示能力；GQA 在质量和推理效率之间折中。主要动机是
缩小推理时需要读取的 K/V 状态，而不是单纯减少总参数。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">5. MHA, MQA, and GQA: reducing KV width</div>

Standard MHA gives every query head its own K and V head:

```text
Q heads: 32   K heads: 32   V heads: 32
```

MQA shares one K/V head across all query heads. GQA shares one K/V head within each
group of query heads:

```text
MQA: Q=32, KV=1
GQA: Q=32, KV=8   # four Q heads share each KV head
```

Query heads can still produce different attention distributions because their Q
vectors differ even when K and V are shared. MQA saves the most memory bandwidth but
may lose representational capacity; GQA trades between quality and inference
efficiency. The primary motivation is narrower K/V state during inference, not merely
fewer total parameters.

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 6. KV Cache：用显存换掉历史重复计算 {#6-kv-cache}

自回归生成到位置 $t$ 时，过去 token 的 K/V 已经算过，而且模型参数没有变化。
因此每层保存

$$
\begin{gathered}
K_{\text{cache}}=[K_{\text{past}};k_t]\\
V_{\text{cache}}=[V_{\text{past}};v_t],
\end{gathered}
$$

新一步只计算 $q_t,k_t,v_t$，再让 query 读取整个缓存：

$$
\begin{gathered}
a_t=\operatorname{softmax}\!\left(\frac{q_tK_{\text{cache}}^\top}{\sqrt{d_k}}\right),\\
o_t=a_tV_{\text{cache}}.
\end{gathered}
$$

在权重、前缀、位置编码和可见范围相同、关闭 dropout 的条件下，未量化 / 未淘汰的 KV cache
与重算前缀在数学上等价；浮点实现仍可能有小误差。代价是 cache 随
层数、上下文长度和 KV heads 线性增长，长上下文时可能成为显存容量与读取带宽瓶颈。
GQA/MQA 正是通过减少 KV heads 来缩小这块状态。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">6. KV cache: trading memory for historical recomputation</div>

At generation step $t$, K and V for earlier tokens have already been computed and
model parameters have not changed. Each layer therefore stores

$$
\begin{gathered}
K_{\text{cache}}=[K_{\text{past}};k_t]\\
V_{\text{cache}}=[V_{\text{past}};v_t],
\end{gathered}
$$

computes only $q_t,k_t,v_t$ for the new token, and lets the query read the full cache:

$$
\begin{gathered}
a_t=\operatorname{softmax}\!\left(\frac{q_tK_{\text{cache}}^\top}{\sqrt{d_k}}\right),\\
o_t=a_tV_{\text{cache}}.
\end{gathered}
$$

With identical weights, prefix, positions, and visibility, and dropout disabled,
unquantized, unevicted KV caching is mathematically equivalent to recomputing the prefix;
floating-point implementations may differ slightly. The cache grows linearly with layer count, context length, and KV-head
count, so it can become a capacity and bandwidth bottleneck. GQA and MQA shrink this
state by reducing the number of KV heads.

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 7. FlashAttention：同一公式，更少显存读写 {#7-flashattention}

朴素实现会显式生成并写回

$$
\begin{gathered}
S=QK^\top\\
A=\operatorname{softmax}(S)\\
S,A\in\mathbb{R}^{L\times L}.
\end{gathered}
$$

FlashAttention 把 Q/K/V 分块，在 GPU 片上高速存储中逐块计算，并用 online softmax
维护正确的归一化结果，从而避免把完整 $L\times L$ 中间矩阵写回显存。

它不是稀疏注意力，也不是通过删 token pair 做近似；数学上计算的是同一 attention
（浮点重排会有微小数值差）。FLOP 复杂度仍约为 $O(L^2)$，主要收益来自 IO-aware
tiling、更少的高带宽显存访问和更小的中间状态。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">7. FlashAttention: the same formula with less memory traffic</div>

A naive implementation materializes and writes

$$
\begin{gathered}
S=QK^\top\\
A=\operatorname{softmax}(S)\\
S,A\in\mathbb{R}^{L\times L}.
\end{gathered}
$$

FlashAttention tiles Q, K, and V, computes blocks in fast on-chip memory, and uses an
online softmax to maintain the exact normalization without writing the full
$L\times L$ intermediates to high-bandwidth memory.

It is not sparse attention and does not approximate the result by dropping token
pairs. It evaluates the same mathematical attention, up to small floating-point
reordering differences. FLOP complexity remains approximately $O(L^2)$; the main
gain comes from IO-aware tiling, less memory traffic, and smaller intermediate state.

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 8. 长上下文：位置外推与平方成本是两件事 {#8}

RoPE scaling 调整旋转频率，解决“位置编码怎样延伸到更长序列”；它并不会消除
full attention 的 $O(L^2)$ 计算。

Sliding-window attention 让每个 token 只看固定邻域，成本接近 $O(Lw)$，但可能丢失
远距离信息。Sparse attention 只保留部分连接；一些架构混合局部层、全局层或特殊
global tokens 来恢复长距离通路。

所以要先问瓶颈是哪一种：

- 训练长度外的位置分布失配 → RoPE scaling / frequency adjustment；
- 完整 attention 的平方计算与中间状态 → windowed / sparse connectivity 或高效 kernel；
- 长距离证据无法到达 → 设计全局通路、检索或层间混合。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">8. Long context: position extrapolation and quadratic cost are different problems</div>

RoPE scaling changes rotation frequencies to extend positional behavior to longer
sequences. It does not remove the $O(L^2)$ cost of full attention.

Sliding-window attention limits each token to a neighborhood, approaching $O(Lw)$
cost but potentially losing distant information. Sparse attention keeps only selected
connections; some architectures mix local and global layers or special global tokens
to restore long-range paths.

Diagnose the bottleneck first:

- distribution shift beyond trained positions → RoPE scaling or frequency adjustment;
- quadratic full-attention compute and intermediates → windowed/sparse connectivity or
  a more efficient kernel;
- distant evidence cannot propagate → explicit global paths, retrieval, or layer mixing.

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 9. MoE：扩大 FFN 容量，而不是注意力头 {#9-moe-ffn}

Mixture-of-Experts 通常替换 FFN。Router 为每个 token 选择少数专家，例如 top-2：

$$
y=p_1E_1(x)+p_2E_2(x).
$$

模型可以拥有很多 expert parameters，但单个 token 只激活少量专家，因此总参数容量
可以远大于每 token 计算量。代价是 router quality、load balancing、expert capacity、
跨设备 all-to-all communication 和训练稳定性都更复杂。

专家可能形成一定功能分工，但和注意力头一样，不应假设每个 expert 都有清晰、固定、
可人工命名的语义。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">9. MoE: scaling FFN capacity, not attention heads</div>

Mixture-of-Experts usually replaces the FFN. A router selects a small number of
experts for each token, for example top-2 routing:

$$
y=p_1E_1(x)+p_2E_2(x).
$$

The model may contain many expert parameters while activating only a few per token,
so total parameter capacity can grow much faster than per-token compute. The cost is
greater complexity in routing quality, load balancing, expert capacity, cross-device
all-to-all communication, and training stability.

Experts may develop some functional specialization, but—as with attention heads—one
should not assume every expert has a clean, fixed, human-nameable semantic role.

</div>
</section>

<section class="concept-card" data-concept-card markdown="1">
<div class="concept-face concept-zh" data-concept-zh markdown="1">

### 10. 把一个现代 Block 从头走一遍 {#10-block}

对于第 $l$ 层输入 $X_l$，先归一化并产生 Q、K、V：

$$
U=\operatorname{RMSNorm}(X_l),
$$

$$
\begin{gathered}
Q=UW_Q\\
K=UW_K\\
V=UW_V.
\end{gathered}
$$

K/V 可能使用 GQA。下面按一个 query head 及其对应的 KV head 写公式；多头结果拼接后再做输出投影。接着只旋转 Q/K，并做 causal attention：

$$
\begin{gathered}
Q'=\operatorname{RoPE}(Q)\\
K'=\operatorname{RoPE}(K),
\end{gathered}
$$

$$
\begin{gathered}
S=Q'K'^\top/\sqrt{d_k},\\
A=\operatorname{softmax}(S+M_{\text{causal}}),\\
O=AV.
\end{gathered}
$$

这一步底层可以由 FlashAttention 高效实现，但公式不变。完成输出投影和第一次残差：

$$
H=X_l+OW_O.
$$

然后走第二条 pre-norm 分支：

$$
G=\operatorname{RMSNorm}(H),
$$

$$
\begin{gathered}
U_g=\operatorname{SiLU}(GW_{\text{gate}}),\\
U_u=GW_{\text{up}},\\
F=(U_g\odot U_u)W_{\text{down}}.
\end{gathered}
$$

$$
X_{l+1}=H+F.
$$

有些模型会把这里的 SwiGLU FFN 换成 MoE。重复 $N$ 层后：

$$
\begin{gathered}
H_{\text{final}}=\operatorname{RMSNorm}(X_N)\\
\text{logits}=H_{\text{final}}W_{\text{vocab}}.
\end{gathered}
$$

因此这些名词并不在同一层面：RoPE 改 Q/K 的位置关系；GQA 改 KV heads；KV cache
保存历史状态；FlashAttention 优化 attention kernel；MoE 则替换 FFN。

</div>
<div class="concept-face concept-en" data-concept-en markdown="1">

<div class="concept-title-en" role="heading" aria-level="3">10. Walking through one modern block end to end</div>

For layer-$l$ input $X_l$, first normalize and form Q, K, and V:

$$
U=\operatorname{RMSNorm}(X_l),
$$

$$
\begin{gathered}
Q=UW_Q\\
K=UW_K\\
V=UW_V.
\end{gathered}
$$

K and V may use GQA. The following attention equation is for one query head and its
associated KV head; concatenate head outputs before the output projection.
Next rotate only Q and K, then apply causal attention:

$$
\begin{gathered}
Q'=\operatorname{RoPE}(Q)\\
K'=\operatorname{RoPE}(K),
\end{gathered}
$$

$$
\begin{gathered}
S=Q'K'^\top/\sqrt{d_k},\\
A=\operatorname{softmax}(S+M_{\text{causal}}),\\
O=AV.
\end{gathered}
$$

FlashAttention can implement this step efficiently without changing the formula.
Apply the output projection and first residual connection:

$$
H=X_l+OW_O.
$$

Then follow the second pre-norm branch:

$$
G=\operatorname{RMSNorm}(H),
$$

$$
\begin{gathered}
U_g=\operatorname{SiLU}(GW_{\text{gate}}),\\
U_u=GW_{\text{up}},\\
F=(U_g\odot U_u)W_{\text{down}}.
\end{gathered}
$$

$$
X_{l+1}=H+F.
$$

Some models replace this SwiGLU FFN with MoE. After repeating $N$ layers:

$$
\begin{gathered}
H_{\text{final}}=\operatorname{RMSNorm}(X_N)\\
\text{logits}=H_{\text{final}}W_{\text{vocab}}.
\end{gathered}
$$

These terms therefore operate at different levels: RoPE changes positional relations
in Q/K; GQA changes KV heads; KV cache stores historical state; FlashAttention
optimizes the attention kernel; and MoE replaces the FFN.

</div>
</section>

## Prefill 与 Decode：并行处理前缀，再逐步生成 {#prefill-decode}

### Prefill {#prefill}

整段 prompt 已知，可以并行计算所有位置，并把每层的 K/V 保存起来。第一枚生成 token，就是从这一步最后有效位置的 logits 中选出的。

### Decode {#decode}

每次只输入新 token，查询历史 KV cache，再产生下一个 token。计算量少，但必须串行，常受内存带宽和 cache 大小限制。

```mermaid
flowchart LR
    F["预填充<br/>Prefill"] -->|末位 logits| S["选择下一 token"]
    S -->|新 token| D["解码一步<br/>Decode"]
    D -->|新 logits| S
    F -->|保存 K/V| K[("KV cache")]
    K -->|读取历史| D
    D -->|追加 K/V| K
```

## 批量生成时，为什么常用左填充？ {#left-padding}

两条 prompt 长度不同，要放进同一个矩形 tensor，就得把短的补齐。假设它们分别有 3 个和 5 个 token，`·` 表示 padding，字母只是 token 的占位符：

<figure class="worked-update worked-update--pairs" aria-label="左填充与右填充的最后位置对照">
<figcaption><strong>下一步读哪一格的 logits？</strong> 常见生成循环统一取每一行的最后一格。</figcaption>
<ol>
<li><strong>右填充（right padding）</strong><br><code>A B C · ·</code><br><code>D E F G H</code><br>短句最后一格是 padding；长句最后一格是 H。</li>
<li><strong>左填充（left padding）</strong><br><code>· · A B C</code><br><code>D E F G H</code><br>最后一格分别是 C 和 H，都是实际前缀的末尾。</li>
</ol>
</figure>

在常见的 decoder-only 生成循环里，下一 token 的分数取自 `logits[:, -1, :]`。左填充让不同长度的 prompt 都能使用这个索引。右填充时，短句会取到 padding 位置的输出，而不是 C 后面该接什么的分数。

**传了 attention mask，也不等于取对了 logits。** Mask 控制哪些位置能被 attention 读取，不负责把 `-1` 改成最后一个有效 token 的下标。把解码结果中的 padding 隐藏掉，也修不好已经取错位置的预测。

### Mask、位置编号和输出位置，各管一件事 {#padding-controls}

| 设置 | 在短句 `· · A B C` 上 | 解决什么问题 |
| --- | --- | --- |
| Attention mask | `[0, 0, 1, 1, 1]` | 有效 token 不读取 padding 的 K/V；仍需 causal mask |
| Position IDs | padding 忽略，A/B/C 编为 0/1/2 | 让 token 使用模型预期的位置编号 |
| Logits 索引 | 最后一格，即 C | 从实际前缀末尾预测下一 token |
| Loss mask | 只统计目标位置 | 训练或评估时，不把 padding 算进 loss |

一种常见的位置编号方式是 `attention_mask.cumsum(-1) - 1`，再给 padding 位置填一个合法占位值。它让有效 token 从 0 开始编号，**但不要把这行公式当作所有模型的接口约定**：多模态位置、已缓存前缀和专用推理引擎可能有各自的处理方式。

<details markdown="1">
<summary>用几行代码，检查“最后一格”和“最后一个有效 token”的区别</summary>

下面只演示一次完整 prefill 后的索引，假设 logits 的前两维和输入 mask 对齐。不下载模型，也不比较真实生成质量；每个位置的 3 个数只是便于核对的假分数。

```python
import torch


def last_valid_logits(logits, attention_mask):
    if logits.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("Expected [batch, length, vocab] and [batch, length]")
    if logits.shape[:2] != attention_mask.shape or logits.shape[1] == 0:
        raise ValueError("Logits and mask must align with a nonempty sequence")
    if logits.device != attention_mask.device:
        raise ValueError("Logits and mask must be on the same device")
    if not torch.all((attention_mask == 0) | (attention_mask == 1)):
        raise ValueError("Attention mask must contain only zero and one")
    valid = attention_mask.bool()
    if not valid.any(dim=-1).all():
        raise ValueError("Every row needs a valid token")
    positions = torch.arange(logits.shape[1], device=logits.device)
    last_positions = positions.expand_as(valid).masked_fill(~valid, -1).amax(-1)
    rows = torch.arange(logits.shape[0], device=logits.device)
    return logits[rows, last_positions]


scores = torch.arange(30).reshape(2, 5, 3)
right_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
left_mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
assert last_valid_logits(scores, right_mask).tolist() == [[6, 7, 8], [27, 28, 29]]
assert torch.equal(last_valid_logits(scores, left_mask), scores[:, -1, :])
```

注意，`mask.sum(-1) - 1` 只适合有效 token 从第 0 格开始、连续排放的情况。左填充短句有 3 个有效 token，但最后一个在下标 4，不在下标 2。这里直接找最后一个非零位置，不依赖填充方向。

这个 helper 只修正一次读出。要让右填充下的完整生成也正确，还要一起处理后续 token 的位置、mask 扩展和 KV cache；已经只返回最后一步 logits 的接口，也不能套用这个函数。

</details>

所以不是“decoder-only 只能左填充”，也不是“padding 一出现，语义就被打断”。**左填充是在常见批量生成接口下，方便又不容易取错位置的选择。** 训练通常能用右填充，甚至通过 packing 去掉 padding；此时关键是 causal / sample boundary 与 loss mask 是否正确。若用左填充训练，还要检查 shift 后的首个有效标签是否意外由 padding 位置预测。

没有独立 pad token 时，有些模型允许用 EOS 充当 padding；这时尤其要显式传入 mask，因为仅看 token ID 无法分清填充值和真实 EOS。默认停止条件通常看新生成的结束 token，不是看到 prompt 左侧已有 EOS 就停；自定义停止逻辑仍需另查。

可对照 [Transformers 生成指南](https://huggingface.co/docs/transformers/llm_tutorial#padding-side)。本次核对的[固定实现 `536ecc0`](https://github.com/huggingface/transformers/blob/536ecc007387a50e77603bb5d92100e9b07514cc/src/transformers/generation/utils.py#L3226)在普通生成路径取最后一格，并按 mask 准备位置编号。专用 serving 引擎可以用变长序列和自己的索引，不必照搬这一布局。

## 模型给分数，Sampling 决定怎么选 {#sampling}

最后一层 hidden state 经过线性层得到词表上每个 token 的 logits：

$$
\begin{gathered}
z_t=h_tW_{\text{vocab}}\\
p_t=\text{softmax}(z_t / \tau)
\end{gathered}
$$

- temperature $\tau$ 调整分布尖锐程度；
- top-$k$ 只保留概率最高的 $k$ 个候选；
- top-$p$ 保留累计概率达到阈值的最小候选集合；
- greedy 每步取最大值，不等于全序列概率最大。

Sampling 决定怎样从模型给出的概率分布中选择 token，但不会改变模型本身的 logits。temperature 较高不代表模型突然获得了更强的创造力，只是低概率 token 更容易被选中。

## Post-Training 如何作用于同一架构 {#post-training}

| 阶段 | 数据告诉模型什么 | 常见目标 |
| --- | --- | --- |
| pre-training | 语言、知识与模式 | next-token cross-entropy |
| SFT | 什么输入应该对应什么回答 | 对目标回答 token 做 cross-entropy |
| preference learning | 两个回答哪个更好 | pairwise / policy objective |
| RL | 行为怎样产生更高回报 | trajectory-level objective |

这些阶段通常不改变 decoder-only 的主体结构，改变的是数据分布、loss 和哪些位置直接计入 loss。Prompt 不计分，也仍可能通过后续回答的 attention 路径接收梯度。

<details markdown="1">
<summary><b>进阶</b>：为什么 SFT 常把 prompt token mask 掉</summary>

训练样本包含 prompt 和 response，但目标通常是学习“在给定 prompt 下怎样回答”，而不是重新学习复述用户输入。因此 loss mask 常只保留 assistant response。若多轮对话里所有 assistant turns 都训练，需要精确处理角色模板和边界 token。

</details>

## 实验：验证训练与生成路径 {#_3}

[`../code/model.py`](../code/model.py) 是手写的现代 decoder-only；[`../code/test_model.py`](../code/test_model.py) 验证 causal mask、RoPE、GQA 和 KV cache；[`../code/train.py`](../code/train.py) 让它学习一个需要跨位置复制的任务。

## 自检 {#_4}

<div class="taste-check">
  <strong>这一课真正要带走的是：</strong>
  <ol>
    <li>为什么输入和标签只需要错开一个 token？</li>
    <li>Prefill 与 decode 使用同一模型，性能特征为什么完全不同？</li>
    <li>temperature、top-k 和 top-p 改的是模型，还是读取模型分布的方法？</li>
  </ol>
</div>

## 继续阅读 {#_5}

继续读 [语言模型目标与生成](../deep-dives/language-model-objective.md)，再接到 [Post-Training](../../05-post-training/)。

## 快速学习：现代 LLM 的完整生成路径 {#llm}

<details class="interview" markdown="1">
<summary>从 messages 到 logits，再到 KV-cached decode</summary>

**快速记忆**：Chat Template 把不同角色的消息按格式拼成一条序列，模型根据前文预测下一个 token。推理时，prefill 先并行处理 prompt，随后逐个 token 生成；历史 K/V 存进 cache，留给后续步骤复用。

**面试回答**

> Decoder-only 模型把 system、user、assistant 与工具消息序列化到同一上下文中，用 causal self-attention 保证每个位置只看左侧。训练对所有位置并行做 next-token prediction；推理先 prefill prompt，再逐 token decode，缓存每层历史 K/V 避免重复投影。

<details markdown="1">
<summary><b>深挖</b>：KV Cache 缓存什么，为什么不缓存 Q？</summary>

历史 token 的 K/V 会被未来每个 query 重复读取，因此缓存后只需为新 token 计算一次。Query 只用于当前 token 发起读取，下一步会产生新的 query，没有跨步复用价值。KV cache 省的是重复计算，不应该改变 attention 的结果：逐步生成得到的结果，要和完整序列的 causal forward 一致。

</details>
</details>
