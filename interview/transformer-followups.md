# Transformer 进阶速查：常见的追问

**中文** · [English](transformer-followups.en.md)

> 阅读时间：约 8 分钟 · 最近审阅：2026-10

> **先读这个**：这里只整理公开的基础知识，不是任何一家公司的面试题，也不涉及任何公司的面试流程。模型的具体配置变得很快，数字以[交互图解](../00-foundations/transformer-lab.md)文末的论文和公开 config 为准。

复习 Transformer 时，除了 attention 公式，也可以继续问：解码为什么要用 KV cache？MQA / GQA / MLA 怎样改变缓存？RoPE、MoE 和 FlashAttention 又分别解决什么问题？

这一页适合**考前复习**：用一问一答回想关键机制，练习怎样在白板前解释。这里的简答不能代替完整推导；想进一步学习，可以看 [Transformer 交互图解](../00-foundations/transformer-lab.md) 和 [Transformer 架构](../00-foundations/transformer.md)，边读推导边操作图中的参数。每一节末尾都给了对应的图。

## KV cache

<details class="qa" markdown="1">
<summary>为什么解码需要 KV cache？不用会怎样？</summary>

生成是一个 token 一个 token 来的。causal decoder 里的旧位置看不到后来追加的 token，因此可以复用旧位置的 K 和 V，不用每一步重算。这里比较的是同一次推理：权重、位置编码和注意力配置不变，dropout 等随机性已经关闭。

不用 cache，第 $t$ 步要重算 $t$ 个 K、$t$ 个 V 和约 $t^2/2$ 个分数；用了 cache，只算一对新的 K/V 和一行分数。

</details>

<details class="qa" markdown="1">
<summary>有了 cache，单步复杂度是多少？瓶颈变成了什么？</summary>

单步 attention 从 $O(t^2)$ 降到 $O(t)$。代价是显存随 token 数线性增长，而且每一步都要把整个 cache 读一遍。小 batch 解码通常受读取权重和 cache 的带宽限制（**memory-bound**）；batch 变大后为什么会转向算力受限，见 [decode 的 roofline 算例](../12-infra/performance.md#decode-batching)。后面讨论的 GQA、MLA 和 sliding window 都会减少 cache 的存储与读取开销。

</details>

<details class="qa" markdown="1">
<summary>cache 的大小怎么算？</summary>

$2 \times n_{\text{layers}} \times n_{\text{kv heads}} \times d_{\text{head}} \times \text{tokens} \times \text{bytes}$。注意里面没有 query head 的数量。Llama-2-7B 形状（32 层、32 个 KV head、$d_{\text{head}}=128$、fp16）是每个 token 0.5 MiB（$2\times32\times32\times128\times2$ 字节），32K（32,768）个 token 就是 16 GiB。

</details>

<details class="qa" markdown="1">
<summary>怎样证明带 cache 的实现是对的？</summary>

固定输入、权重和推理配置，关闭 dropout，再对比 cache 路径逐位置输出的 logits 与不带 cache 的 full forward。两者应在所用精度的容差内一致，不要求不同 kernel 的结果逐位相同。最容易错的是 mask 和位置：query 的绝对位置是 `cache.pos + i`，mask 是非方阵的 $(T, S)$，RoPE 的 cos/sin 也要从 `cache.pos` 开始切。

</details>

看图：[KV cache：解码每一步到底算了什么](../00-foundations/transformer-lab.md#tx-kv-cache)

## MHA → GQA → MQA → MLA

<details class="qa" markdown="1">
<summary>MQA、GQA 各自改了什么？为什么只动 K/V 不动 Q？</summary>

其他维度不变时，cache 的大小随 KV head 数变化，不直接取决于 query head 数，所以可以只改 K/V 来省缓存。**MQA**：所有 query head 共享一个 K/V head，在这几种 KV head 共享方式中缓存最小，可能牺牲部分质量；[原论文](https://arxiv.org/abs/1911.02150)报告的是其任务上的小幅下降，不是所有模型必然下降。**GQA**：一组 query head 共享一个 K/V head，是两者的折中；Llama 3 8B 是 32 个 query head 配 8 个 KV head。

</details>

<details class="qa" markdown="1">
<summary>MLA 缓存的是什么？为什么它的 cache 和 head 数无关？</summary>

MLA 不缓存每个 head 的 K 和 V，而是每个 token 缓存一个低维 latent 向量。结构上，这个向量可以通过上投影得到各 head 的 K/V；推理时则可把内容侧的上投影吸收到 query 和输出投影中，不必真的展开全部历史 K/V。[DeepSeek-V2](https://arxiv.org/abs/2405.04434) 的设置是 latent $4\,d_{\text{head}}$ 加一个 $d_{\text{head}}/2$ 的 RoPE key，共 $4.5\,d_{\text{head}}$——相当于 GQA 只用 2.25 个 KV head，但每个 head 仍然有自己的 K/V 投影。

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

softmax 的权重必须加起来等于 1。某些 head 会把较多权重分给开头几个 token，即使这些 token 的内容与当前预测关系不大；这就是 attention sink 的一种表现。移除这些 token 会改变注意力分配，可能明显损伤流式生成。StreamingLLM 因此保留开头几个 token。[gpt-oss 的实现](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py)则为每个 head 增加一个可学习的 sink logit，参与 softmax，但不对应实际的 value。分母多出的是这个 logit 的指数项，不是直接加一个 bias。

</details>

看图：[谁能看到谁](../00-foundations/transformer-lab.md#tx-windows)

## RoPE

<details class="qa" markdown="1">
<summary>RoPE 怎么表示位置？为什么说它是“相对”的？</summary>

把 $q$ 和 $k$ 的每一对维度旋转一个与位置成正比的角度，每一对的转速不同。固定原来的 $q$、$k$ 后，旋转引入的位置依赖只与角度之差有关，因此能用相对距离 $m-n$ 表示。分数仍然依赖 $q$、$k$ 的内容，不是只看两个 token 隔了多远。

</details>

<details class="qa" markdown="1">
<summary>为什么必须同时旋转 Q 和 K？只转一个行不行？</summary>

不行。相对性来自两个旋转相互抵消：$\langle R_m q, R_n k\rangle = \langle q, R_{n-m} k\rangle$。只旋转一边，点积里剩下的就是绝对位置。

</details>

<details class="qa" markdown="1">
<summary>长上下文为什么要调大 RoPE 的 base θ？</summary>

base 越大，除频率固定为 1 的第一对外，其余维度都转得更慢，低频维度在更长距离内不会转完一圈，远处位置更不容易混淆。只调 base 不保证长上下文好用，还要配合训练和评估。Llama 3 用 $\theta = 500\text{k}$，Gemma 3 在 global 层用 $\theta = 1\text{M}$。

</details>

看图：[RoPE：位置就是旋转](../00-foundations/transformer-lab.md#tx-rope)

## MoE

<details class="qa" markdown="1">
<summary>MoE 的总参数量和激活参数量分别随什么增长？</summary>

把 FFN 换成 $N$ 个 expert，router 只把每个 token 送到得分最高的 $k$ 个。expert 大小不变时，总 expert 参数随 $N$ 增长，每个 token 的 routed-expert 计算随 $k$ 增长；总成本还包括 router、共享模块和通信，不能只数 $k$。Mixtral 是 8 选 2（约 47B 总参数、约 13B 激活）；DeepSeek-V3 是 256 个 routed expert 选 8 个，再加 1 个所有 token 都经过的 shared expert（671B 总参数、37B 激活）。

</details>

<details class="qa" markdown="1">
<summary>MoE 训练最典型的问题是什么？</summary>

负载不均：router 容易把 token 都塞给少数几个 expert，其余 expert 学不到东西。常见做法是加辅助的 load-balancing loss；DeepSeek-V3 主要改用一个 bias 项来调节，只保留一个权重很小的序列级辅助 loss。

</details>

看图：[MoE：按真实 expert 数量画的路由](../00-foundations/transformer-lab.md#tx-moe)

## FlashAttention

<details class="qa" markdown="1">
<summary>FlashAttention 是近似算法吗？它到底省了什么？</summary>

标准的 dense FlashAttention 不近似 attention 公式；这里的“精确”是数学等价，不保证不同浮点实现逐位相同。它主要省的是访存：标准实现要把完整的 $n\times n$ 分数矩阵写进较慢的 HBM 再读回来；FlashAttention 把 Q、K、V 切块放进片上 SRAM，一块一块地算。它保存 $n\times d$ 的输出及反向所需的归一化统计，不保存完整分数矩阵，额外显存对 $n$ 线性。[原论文](https://arxiv.org/abs/2205.14135)还通过反向重计算，用额外算术换取更少访存。HBM 和片上 SRAM 的容量、带宽各差多少，见 [GPU 的内存层级](../12-infra/gpu-architecture.md)。

</details>

<details class="qa" markdown="1">
<summary>分块之后 softmax 怎么算？整行不是要一起归一化吗？</summary>

用在线 softmax（online softmax）：每一行维护当前最大值（running max）和一个归一化因子。每处理一个新块，就更新这两个量，并相应调整已经累积的输出。数学上和一次性算整行完全等价。

</details>

看图：[FlashAttention：attention 矩阵放在哪里](../00-foundations/transformer-lab.md#tx-flash)

## 结构与模型家族

<details class="qa" markdown="1">
<summary>从原版 Transformer 到 Llama 3，block 里改了哪几处？</summary>

四处：norm 从 post-LN 的 LayerNorm 变成 pre-norm 的 RMSNorm；位置编码从加在 embedding 上的绝对位置变成 RoPE；FFN 的激活从原版的 ReLU 变成 SwiGLU（GELU 出现在其他后续模型中）；attention 从 MHA 变成 GQA。另外去掉了 encoder 和 cross-attention，也不再用 bias。

</details>

<details class="qa" markdown="1">
<summary>Pre-norm 和 post-norm 差在哪？</summary>

post-norm 把 norm 放在残差相加之后，梯度每过一层都要穿过一次 norm；pre-norm 把 norm 放进残差分支里，主干上留下一条不被缩放的恒等通路。[Xiong 等](https://arxiv.org/abs/2002.04745)分析了原版 post-LN 在初始化时靠近输出层的较大梯度，并说明 warmup 为什么有帮助。这不等于所有 post-norm 都必须用 warmup，或所有 pre-norm 都可以不用；初始化、深度和优化设置也要一起看。

</details>

看图：[结构图：在模型家族之间变形](../00-foundations/transformer-lab.md#tx-arch)

## 继续

- [考前速查：按知识板块过一遍考题](questions.md)
- [Transformer 交互图解](../00-foundations/transformer-lab.md) —— 同样的内容，学习版
