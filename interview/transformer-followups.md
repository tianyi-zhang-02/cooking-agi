# Transformer 进阶速查：那些一定会被追问的

**中文** · [English](transformer-followups.en.md)

> 阅读时间：约 8 分钟 · 最近审阅：2026-09

> **先读这个**：这里只整理公开的基础知识，不是任何一家公司的面试题，也不涉及任何公司的面试流程。模型的具体配置变得很快，数字以[交互图解](../00-foundations/transformer-lab.md)文末的论文和公开 config 为准。

问到 Transformer，很少有人停在“attention 公式是什么”。真正拉开差距的是后面的追问：KV cache、MQA / GQA / MLA、RoPE、MoE、FlashAttention。

这一页是**考前过一遍**用的：一问一答，答案压到能在白板前两句话讲完。想真的学透——看推导、动手拖图——去知识板块那边的 [Transformer 交互图解](../00-foundations/transformer-lab.md) 和 [Transformer 架构](../00-foundations/transformer.md)；每一节末尾都给了对应的图。

## KV cache

<details class="qa" markdown="1">
<summary>为什么解码需要 KV cache？不用会怎样？</summary>

生成是一个 token 一个 token 来的。causal mask 下，过去 token 的 K 和 V 永远不会变，所以没必要每一步重算。不用 cache，第 $t$ 步要重算 $t$ 个 K、$t$ 个 V 和约 $t^2/2$ 个分数；用了 cache，只算一对新的 K/V 和一行分数。

</details>

<details class="qa" markdown="1">
<summary>有了 cache，单步复杂度是多少？瓶颈变成了什么？</summary>

单步 attention 从 $O(t^2)$ 降到 $O(t)$。代价是显存随 token 数线性增长，而且每一步都要把整个 cache 读一遍——解码从 compute-bound 变成 **memory-bound**。后面所有“压 cache”的工作（GQA、MLA、sliding window）都是冲着这一点来的。

</details>

<details class="qa" markdown="1">
<summary>cache 的大小怎么算？</summary>

$2 \times n_{\text{layers}} \times n_{\text{kv heads}} \times d_{\text{head}} \times \text{tokens} \times \text{bytes}$。注意里面没有 query head 的数量。Llama-2-7B 形状（32 层、32 个 KV head、$d_{\text{head}}=128$、fp16）是每个 token 0.5 MB，32K 上下文就是 16 GB。

</details>

<details class="qa" markdown="1">
<summary>怎样证明带 cache 的实现是对的？</summary>

cache 路径逐位置输出的 logits，必须和不带 cache 的 full forward **数值等价**。最容易错的是 mask 和位置：query 的绝对位置是 `cache.pos + i`，mask 是非方阵的 $(T, S)$，RoPE 的 cos/sin 也要从 `cache.pos` 开始切。

</details>

看图：[KV cache：解码每一步到底算了什么](../00-foundations/transformer-lab.md#tx-kv-cache)

## MHA → GQA → MQA → MLA

<details class="qa" markdown="1">
<summary>MQA、GQA 各自改了什么？为什么只动 K/V 不动 Q？</summary>

cache 的大小只和 KV head 数有关，和 query head 数无关，所以省显存只需要在 K/V 一侧动手。**MQA**：所有 query head 共享一个 K/V head，cache 最小，质量有损失。**GQA**：一组 query head 共享一个 K/V head，是两者的折中；Llama 3 8B 是 32 个 query head 配 8 个 KV head。

</details>

<details class="qa" markdown="1">
<summary>MLA 缓存的是什么？为什么它的 cache 和 head 数无关？</summary>

MLA 不缓存每个 head 的 K 和 V，而是每个 token 缓存一个低维 latent 向量，用的时候再通过上投影展开成每个 head 自己的 K 和 V。DeepSeek 的设置是 latent $4\,d_{\text{head}}$ 加一个 $d_{\text{head}}/2$ 的 RoPE key，共 $4.5\,d_{\text{head}}$——相当于 GQA 只用 2.25 个 KV head，但每个 head 仍然有自己的 K/V。

</details>

<details class="qa" markdown="1">
<summary>MLA 里的 RoPE 为什么要单独拿出来？</summary>

RoPE 是和位置相关的旋转，没法被吸收进一个与位置无关的低秩投影里。所以 MLA 把位置信息放在一个单独的、不压缩的小 key（64 维）上，其余内容走 latent。

</details>

看图：[MHA → GQA → MQA → MLA 与显存计算器](../00-foundations/transformer-lab.md#tx-kv-heads)

## Sliding window 与 attention sink

<details class="qa" markdown="1">
<summary>Sliding window 每层只看 w 个 token，远处的信息怎么传过来？</summary>

靠层数叠加。每多一层，可达范围就再多一个窗口，$L$ 层之后大约是 $L\,(w-1)+1$ 个 token。Mistral 7B 是 $w=4096$、32 层，理论可达约 131K。

</details>

<details class="qa" markdown="1">
<summary>它省了什么？</summary>

计算从 $O(n^2)$ 变成 $O(n\,w)$，每层的 KV cache 封顶在 $w$。Gemma 3 用 5 层 local 配 1 层 global，只有 global 层需要完整长度的 cache。

</details>

<details class="qa" markdown="1">
<summary>Attention sink 是什么？为什么把最开头的 token 逐出 cache 会出事？</summary>

softmax 的权重必须加起来等于 1，当一个 head“没什么可看”时，模型学会了把多余的权重丢在最开头的几个 token 上。把它们逐出 cache，这些权重无处可去，分布就乱了。StreamingLLM 的做法是永久保留开头几个 token；gpt-oss 则在 softmax 分母里加了一个可学习的 bias，起同样的作用。

</details>

看图：[谁能看到谁](../00-foundations/transformer-lab.md#tx-windows)

## RoPE

<details class="qa" markdown="1">
<summary>RoPE 怎么表示位置？为什么说它是“相对”的？</summary>

把 $q$ 和 $k$ 的每一对维度旋转一个与位置成正比的角度，每一对的转速不同。两个旋转后向量的点积只取决于角度之差，所以分数只依赖相对距离 $m-n$。

</details>

<details class="qa" markdown="1">
<summary>为什么必须同时旋转 Q 和 K？只转一个行不行？</summary>

不行。相对性来自两个旋转相互抵消：$\langle R_m q, R_n k\rangle = \langle q, R_{n-m} k\rangle$。只旋转一边，点积里剩下的就是绝对位置。

</details>

<details class="qa" markdown="1">
<summary>长上下文为什么要调大 RoPE 的 base θ？</summary>

base 越大，各对维度转得越慢，分数随距离衰减得也越慢，更远的位置仍然可区分。Llama 3 用 $\theta = 500\text{k}$，Gemma 3 在 global 层用 $\theta = 1\text{M}$。

</details>

看图：[RoPE：位置就是旋转](../00-foundations/transformer-lab.md#tx-rope)

## MoE

<details class="qa" markdown="1">
<summary>MoE 的总参数量和激活参数量分别随什么增长？</summary>

把 FFN 换成 $N$ 个 expert，router 只把每个 token 送到得分最高的 $k$ 个。总参数随 $N$ 增长，单个 token 的计算量只随 $k$ 增长。Mixtral 是 8 选 2（约 47B 总参数、约 13B 激活）；DeepSeek-V3 是 256 个 routed expert 选 8 个，再加 1 个所有 token 都经过的 shared expert（671B 总参数、37B 激活）。

</details>

<details class="qa" markdown="1">
<summary>MoE 训练最典型的问题是什么？</summary>

负载不均：router 容易把 token 都塞给少数几个 expert，其余 expert 学不到东西。常见做法是加辅助的 load-balancing loss；DeepSeek-V3 改用一个 bias 项来调节，不引入辅助 loss。

</details>

看图：[MoE：按真实 expert 数量画的路由](../00-foundations/transformer-lab.md#tx-moe)

## FlashAttention

<details class="qa" markdown="1">
<summary>FlashAttention 是近似算法吗？它到底省了什么？</summary>

不是近似，结果是**精确的**。它省的不是算术，而是访存：标准实现要把完整的 $n\times n$ 分数矩阵写进较慢的 HBM 再读回来；FlashAttention 把 Q、K、V 切块放进片上 SRAM，一块一块地算，只写回 $n\times d$ 的输出，额外显存对 $n$ 线性。

</details>

<details class="qa" markdown="1">
<summary>分块之后 softmax 怎么算？整行不是要一起归一化吗？</summary>

用 online softmax：每一行维护一个 running max 和一个归一化因子，每来一个新 block 就更新这两个量并修正已经累积的输出。数学上和一次性算整行完全等价。

</details>

看图：[FlashAttention：attention 矩阵放在哪里](../00-foundations/transformer-lab.md#tx-flash)

## 结构与模型家族

<details class="qa" markdown="1">
<summary>从原版 Transformer 到 Llama 3，block 里改了哪几处？</summary>

四处：norm 从 post-LN 的 LayerNorm 变成 pre-norm 的 RMSNorm（深层训练更稳）；位置编码从加在 embedding 上的绝对位置变成 RoPE；FFN 的激活从 ReLU / GELU 变成 SwiGLU；attention 从 MHA 变成 GQA。另外去掉了 encoder 和 cross-attention，也不再用 bias。

</details>

<details class="qa" markdown="1">
<summary>Pre-norm 和 post-norm 差在哪？</summary>

post-norm 把 norm 放在残差相加之后，梯度每过一层都要穿过一次 norm，深层网络依赖 warmup 才能训稳；pre-norm 把 norm 放进残差分支里，主干上留下一条不被缩放的恒等通路。

</details>

看图：[结构图：在模型家族之间变形](../00-foundations/transformer-lab.md#tx-arch)

## 继续

- [考前速查：按知识板块过一遍考题](questions.md)
- [Transformer 交互图解](../00-foundations/transformer-lab.md) —— 同样的内容，学习版
