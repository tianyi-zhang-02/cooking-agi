# Transformer 交互图解：结构、优化与模型家族

**中文** · [English](transformer-lab.en.md)

> 阅读时间：约 15 分钟，动手玩会更久 · 难度：入门到进阶 · 最近审阅：2026-10-09
>
> 这一页是 [Transformer 架构](transformer.md) 的动手版：公式和推导在那边，这里只做一件事——让每个机制都能拖、能点、能看到数字怎么变。图里的数字都在浏览器里实时计算；模型家族的具体配置变化很快，以文末的论文和公开 config 为准。

<div class="lesson-recipe advanced">
  <div><span>这次要拆什么</span><strong>一张会变形的结构图，和七个改变了它的优化</strong></div>
  <div><span>需要先会</span><strong>attention 公式 · residual · causal LM</strong></div>
  <div><span>真正的主角</span><strong>KV cache · GQA / MLA · sliding window · RoPE · MoE · FlashAttention</strong></div>
  <div><span>最后要能证明</span><strong>每个优化省的是显存、访存还是计算，代价又是什么</strong></div>
</div>

图例贯穿全页：**蓝色**是 query 和 attention 相关的东西，**橙色**是 key、FFN 和 expert，**绿色**是 value 和 embedding。

## 结构图：一个 block，很多家族

先看最常见的自回归 Transformer：把 token 变成向量，经过多层 block，预测下一个 token。这里的“重复”指结构相似，不是每层共用同一组参数。标准 block 里，attention 在 token **之间**传递信息，FFN 对每个 token **单独**做变换，两条分支都有残差连接。混合架构也可能把部分 attention 换成循环状态更新，不能把这张图当成所有 LLM 的统一结构。

先比较几处最容易看见的差别：norm 放在哪、位置怎么编码、K/V head 怎么共享、FFN 是 dense 还是 MoE。选一个家族，看哪里动了；右侧带圆点的行，就是相对上一个选择发生变化的地方。数据、训练目标和推理配置不在这张结构图里，却同样影响效果。

<!-- widget:tx-arch -->

最值得对比的三步：`Transformer ’17 → GPT-2 / 3`（去掉 encoder 和 cross-attention，norm 移进残差分支）、`GPT-2 / 3 → Llama 3`（LayerNorm → RMSNorm，绝对位置 → RoPE，GELU → SwiGLU，MHA → GQA）、`Llama 3 → DeepSeek-V3`（GQA → MLA，dense FFN → MoE）。

## Self-attention：一步一步算

每个 token 产生三个向量：**query**（我在找什么）、**key**（我能提供什么）和 **value**（我要传递什么）。分数是点积 $q\cdot k$，除以 $\sqrt{d_k}$，加上 mask 保证看不到未来，再过 softmax；输出是 value 的加权平均，也就是 $\operatorname{softmax}(QK^\top/\sqrt{d_k})\,V$。

<!-- widget:tx-attention -->

这是一个玩具例子：6 个 token、$d_k=4$，权重是手工设定的（不是训练出来的），方便逐步核对。把鼠标放到某一行，可以跟着这个 token 走完整个计算；关掉 causal mask，就能看到双向 attention，但这还不是完整的 BERT：训练目标和其他层也不同。为什么要除以 $\sqrt{d_k}$、三个投影矩阵各自的含义，见 [Transformer 架构](transformer.md)。

## 为什么解码离不开 KV cache

生成是一个 token 一个 token 来的。如果每次都完整重算长度为 $t$ 的前缀，单层每个 head 要重算 $t$ 个 key、$t$ 个 value，以及 $t(t+1)/2$ 个有效 causal 分数。模型权重、前缀和位置配置不变，且关闭 dropout 时，过去的 K/V 可以复用；每步只需算新 token 的 K/V 和一行 attention 分数。修改前缀、权重或位置处理后，旧 cache 不一定还能用。

<!-- widget:tx-kv-cache -->

切到“没有 cache”再看一遍，注意两条累计柱的差距。在层数和维度固定时，attention 的单步工作量从 $O(t^2)$ 降到 $O(t)$，代价是 cache 随长度增长。小 batch 解码常受读取权重和 KV 的带宽限制；大 batch、不同 kernel 或硬件下，瓶颈可能不同。下一节只比较 KV 存储，不据此保证吞吐量。

## 压缩 cache：MHA → GQA → MQA → MLA

每层每个 token 的 cache 大小是

$$
2 \times n_{\text{kv heads}} \times d_{\text{head}}
$$

个数值，假设 K/V 维度相同、每层配置一致。再乘层数、token 数和每个数的字节数，才是存储字节数。固定 KV head 数和维度后，query head 数不会额外进入这个公式：

- **MQA**（2019）：所有 query head 共享一个 K/V head。相同维度下，它在这类 head 共享方案中缓存最小；质量是否下降要看训练和任务。
- **GQA**（2023）：一组 query head 共享一个 K/V head，在存储和表达能力之间折中。下表中的 Llama 3、Mistral、Qwen3 和 Gemma 3 都采用了它。
- **MLA**（DeepSeek-V2，2024）：缓存压缩后的 KV latent，再单独存 RoPE key。每层每 token 是 $d_c+d_r$ 个数值，不是通用的 $4.5\,d_{\text{head}}$。DeepSeek-V3 的配置是 $512+64=576$；矩阵吸收（weight absorption）还能避免在解码时显式还原全部 K/V，细节见 [MLA 推导](deep-dives/latent-and-sparse-attention.md)。

<!-- widget:tx-kv-heads -->

下半部分是控制变量计算器，不是实际部署配置：固定层数、长度与存储精度，再替换 attention 方案。MLA 行固定使用 512 维 latent + 64 维 RoPE key。它不包含权重、临时激活、KV 分页碎片或跨卡副本，也不表示每个模型原生支持这四种方案。

## 不是每个 token 都要看到所有 token

完整的 causal attention 是 $O(n^2)$。**Sliding-window attention** 让每个 token 只看最近的 $w$ 个 token：计算量变成 $O(n\,w)$，cache 也封顶在 $w$。信息仍然能传得更远，因为每多一层，可达范围就再多一个窗口——$L$ 层之后大约是 $L\,(w-1)+1$ 个 token。

<!-- widget:tx-windows -->

拖动“堆叠层数”看浅色区域怎么扩张。这里的范围是信息可能经过的路径，不保证远处内容一定被记住。Gemma 3 采用 5 层 local 配 1 层 global；gpt-oss 交替使用窗口层与全局层。StreamingLLM 则保留开头的 sink token 和最近窗口，减少直接丢掉开头 KV 带来的失稳；已经移出窗口的细节并不会因此完整保留。

## RoPE：把位置变成旋转

RoPE 不往 embedding 上加位置向量，而是旋转 $q$ 和 $k$ 的成对维度，每对使用不同频率。**固定内容向量**后，位置对点积的影响取决于相对距离 $m-n$；真实 attention 分数仍然依赖内容，不是只看距离。

<!-- widget:tx-rope -->

点播放：同时移动两个位置，分数不变。再拖动距离 $\Delta$，会看到曲线起伏，**不是逐点单调下降**。单个旋转平面就能说明：把两个内容向量都设为 $(1,0)$，点积随相位差是 $\cos\Delta$；从 $\pi$ 到 $2\pi$，它反而从 -1 升到 1。调大 base 会减慢部分维度的旋转，但仅改这个数不保证长上下文好用，还要结合训练和评估。推导见 [Transformer 架构 · RoPE](transformer.md)。

## MoE：参数更多，不必每次都算完 {#moe-token}

MoE 把一个 FFN 换成多个 expert，再由 router 为每个 token 选其中 $k$ 个。固定每个 expert 的大小和 $k$ 时，增加 expert 总数不增加被选中 FFN 的算术量。但 router、通信和负载不均仍有成本；选 2 个同样大的 expert，也不能说和原来 1 个 dense FFN 一样便宜。

<!-- widget:tx-moe -->

格子按真实数量画（8、128 或 256 个 expert），所以你看到的稀疏度就是真实的稀疏度。DeepSeek-V3 多了一个所有 token 都会经过的 shared expert。图里的 router 是随机的，只为演示负载分布；真实的 router 是一个学出来的线性层，负载均衡要靠辅助 loss 或 bias 项来维持。完整的一组笔记见 [MoE 系列](moe/)。

## FlashAttention：数学不变，少搬显存

朴素 attention 会把完整的 $n\times n$ 分数和概率矩阵写到 HBM，来回读写可能很贵。FlashAttention 用分块和 online softmax 累积输出，不需要把这两张完整矩阵存到 HBM。它不靠稀疏化来近似 attention，但浮点计算顺序不同，结果不保证逐位相同。省下多少时间仍取决于形状、硬件和 kernel，不能一概说 attention 总是访存瓶颈。

<!-- widget:tx-flash -->

切到“标准 attention”，看 S 和 P 两个 $n\times n$ 矩阵怎样被完整写进 HBM；再切回来，注意右上角整块被 causal mask 直接跳过的 block。

## 模型家族速览

下面保留的是这些**具体历史版本**，不是各家最新型号的排行榜。数字来自对应报告和 config；后续版本见[模型家族精读](model-families/README.md)。

| 家族 | 整体结构 | Norm | 位置编码 | Attention | FFN | 上下文 | 规模 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Transformer（2017，base） | Encoder–decoder，6 + 6 层 | Post-LN · LayerNorm | 正弦，加在 embedding 上 | MHA · 8 head，另有 cross-attention | Dense · ReLU · $d_{ff}$ 2,048 | 句子对（机器翻译） | 65M · $d_{model}$ 512 |
| GPT-2 / GPT-3（2019–20） | Decoder-only | Pre-LN · LayerNorm，末尾再加一层 | 可学习的绝对位置 | MHA · 25 head（GPT-2 XL）· 96 head（GPT-3） | Dense · GELU · 4× | 1,024 · 2,048 | 1.5B：48 层 · 175B：96 层 |
| Llama 3（2024，8B） | Decoder-only | Pre-norm · RMSNorm | RoPE · $\theta$ = 500,000 | GQA · 32 query / 8 KV head | Dense · SwiGLU · $d_{ff}$ 14,336 | 8K → 128K（Llama 3.1） | 8B：32 层，$d$ 4,096 |
| Mistral 7B（2023） | Decoder-only | Pre-norm · RMSNorm | RoPE | GQA · 32 / 8 · sliding window 4,096 | Dense · SwiGLU · $d_{ff}$ 14,336 | 8K，每层窗口 4,096 | 7B：32 层，$d$ 4,096 |
| Mixtral 8x7B（2023） | Decoder-only | Pre-norm · RMSNorm | RoPE | GQA · 32 / 8 · 完整 32K attention | MoE · 8 个 SwiGLU expert，选 2 个 | 32K | 总参数约 47B · 激活约 13B |
| Gemma 3（2025，27B） | Decoder-only | Pre-norm + post-norm · RMSNorm | RoPE · local 层 10k，global 层 1M | GQA · 32 / 16 · QK-norm · 5 local（窗口 1,024）: 1 global | Dense · GeGLU | 128K | 27B：62 层，$d$ 5,376 |
| Qwen3（2025，235B-A22B） | Decoder-only | Pre-norm · RMSNorm | RoPE · 长上下文用 YaRN | GQA · 64 / 4 · QK-norm | MoE · 128 个 expert，选 8 个，无 shared expert | 原生 32K · YaRN 到 131K | 总参数 235B · 激活 22B · 94 层 |
| DeepSeek-V3（2024） | Decoder-only | Pre-norm · RMSNorm | RoPE 作用在单独的 64 维 key 上 | MLA · 128 head · 缓存 latent 512 + 64 | MoE · 1 shared + 256 routed，选 8 个 | 128K | 总参数 671B · 激活 37B · 61 层 |
| gpt-oss-120b（2025） | Decoder-only | Pre-norm · RMSNorm | RoPE · YaRN 到 131K | GQA · 64 / 8 · 隔层 128 窗口 · learned sinks | MoE · 128 个 expert，选 4 个 | 131K | 总参数 117B · 激活 5.1B · 36 层 |

## 自检

<div class="taste-check advanced">
  <strong>完成这一页后，至少要能解释六个问题：</strong>
  <ol>
    <li>从原版 Transformer 到 Llama 3，block 里改了哪四处？各自解决什么问题？</li>
    <li>KV cache 在什么条件下可以复用？为什么减少计算后，仍可能卡在内存带宽？</li>
    <li>GQA、MQA、MLA 各自缓存的是什么？MLA 的 cache 为什么和 head 数无关？</li>
    <li>Sliding window 每层只看 $w$ 个 token，信息是怎么传到更远的地方的？</li>
    <li>FlashAttention 为什么是精确的？它省掉的是哪一类开销？</li>
    <li>MoE 的总参数量和激活参数量，分别随什么增长？</li>
  </ol>
</div>

## 继续阅读

- [Transformer 架构](transformer.md) —— 这一页所有机制的公式、推导和从零实现
- [Multi-head attention](core/multi-head-attention.md) 与 [Decoder-only](core/decoder-only.md) —— 必修章节里的逐步拆解

## 参考论文

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原版结构
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) · [RMSNorm](https://arxiv.org/abs/1910.07467) · [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — MQA · [GQA](https://arxiv.org/abs/2305.13245) · [DeepSeek-V2](https://arxiv.org/abs/2405.04434) — MLA
- [RoFormer](https://arxiv.org/abs/2104.09864) — RoPE · [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)
- [FlashAttention](https://arxiv.org/abs/2205.14135) · [FlashAttention-2](https://arxiv.org/abs/2307.08691) · [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
- [Sparsely-Gated Mixture-of-Experts](https://arxiv.org/abs/1701.06538) · [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- 模型报告：[GPT-3](https://arxiv.org/abs/2005.14165) · [Llama 3](https://arxiv.org/abs/2407.21783) · [Mistral 7B](https://arxiv.org/abs/2310.06825) · [Gemma 3](https://arxiv.org/abs/2503.19786) · [Qwen3](https://arxiv.org/abs/2505.09388) · [DeepSeek-V3](https://arxiv.org/abs/2412.19437) · [gpt-oss](https://arxiv.org/abs/2508.10925)
