# Transformer 交互图解：结构、优化与模型家族

**中文** · [English](transformer-lab.en.md)

> 阅读时间：约 12 分钟，动手玩会更久 · 难度：入门到进阶 · 最近审阅：2026-09
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

所有现代 LLM 跑的都是同一个循环：把 token 变成向量，经过 $N$ 个相同的 block，预测下一个 token。每个 block 有两个子层：attention 在 token **之间**搬运信息，FFN 对每个 token **单独**做变换，外面都包着残差连接。

家族之间真正的差别只是一张很短的清单：norm 放在哪、位置怎么编码、K/V head 怎么共享、FFN 是 dense 还是 MoE。选一个家族，看哪里动了；右侧带圆点的行，就是相对上一个选择发生变化的地方。

<!-- widget:tx-arch -->

最值得对比的三步：`Transformer ’17 → GPT-2 / 3`（去掉 encoder 和 cross-attention，norm 移进残差分支）、`GPT-2 / 3 → Llama 3`（LayerNorm → RMSNorm，绝对位置 → RoPE，GELU → SwiGLU，MHA → GQA）、`Llama 3 → DeepSeek-V3`（GQA → MLA，dense FFN → MoE）。

## Self-attention：一步一步算

每个 token 产生三个向量：**query**（我在找什么）、**key**（我能提供什么）和 **value**（我要传递什么）。分数是点积 $q\cdot k$，除以 $\sqrt{d_k}$，加上 mask 保证看不到未来，再过 softmax；输出是 value 的加权平均，也就是 $\operatorname{softmax}(QK^\top/\sqrt{d_k})\,V$。

<!-- widget:tx-attention -->

这是一个玩具例子：6 个 token、$d_k=4$，权重是手工设定的（不是训练出来的），目的是让每一步的数字都能心算验证。把鼠标放到某一行，可以跟着这个 token 走完整个计算；关掉 causal mask，就得到 BERT 式的 encoder。为什么要除以 $\sqrt{d_k}$、三个投影矩阵各自的含义，见 [Transformer 架构](transformer.md)。

## 为什么解码离不开 KV cache

生成是一个 token 一个 token 来的。没有 cache 的话，每一步都要对整个前缀重算一遍 attention：第 $t$ 步要重算 $t$ 个 key、$t$ 个 value 和约 $t^2/2$ 个分数。但在 causal mask 下，过去的 key 和 value 永远不会变，所以把它们存起来：第 $t$ 步只算一对新的 K/V 和一行分数。

<!-- widget:tx-kv-cache -->

切到“没有 cache”再看一遍，注意下面两条累计柱的差距。代价是显存随 token 数线性增长——单步计算从 $O(t^2)$ 降到 $O(t)$，解码却因此变成了访存受限（memory-bound）：每一步都要把整个 cache 读一遍。下一节解决的就是这个问题。

## 压缩 cache：MHA → GQA → MQA → MLA

每层每个 token 的 cache 大小是

$$
2 \times n_{\text{kv heads}} \times d_{\text{head}},
$$

再乘上层数、token 数和每个数的字节数。query head 的数量不影响 cache，所以可以只在 K/V 这一侧动手：

- **MQA**（2019）：保留所有 query head，只用一个共享的 K/V head。cache 最小，质量有损失。
- **GQA**（2023）：一组 query head 共享一个 K/V head，是 MHA 和 MQA 之间的折中。Llama 3、Mistral、Qwen、Gemma 都在用。
- **MLA**（DeepSeek-V2，2024）：每个 token 只缓存一个很小的 latent 向量，用的时候再展开成每个 head 自己的 K 和 V。cache 大小是 $4.5\,d_{\text{head}}$，和 head 数无关，相当于 GQA 只用 2.25 个 KV head。

<!-- widget:tx-kv-heads -->

下半部分是计算器：选一个模型形状，拖动上下文长度和并发数，看四种方案各要多少显存。把形状切到 DeepSeek-V3（128 个 head），再对比 MHA 和 MLA 两行。

## 不是每个 token 都要看到所有 token

完整的 causal attention 是 $O(n^2)$。**Sliding-window attention** 让每个 token 只看最近的 $w$ 个 token：计算量变成 $O(n\,w)$，cache 也封顶在 $w$。信息仍然能传得更远，因为每多一层，可达范围就再多一个窗口——$L$ 层之后大约是 $L\,(w-1)+1$ 个 token。

<!-- widget:tx-windows -->

拖动“堆叠层数”看浅色区域怎么扩张。另外三种模式对应真实模型：Gemma 3 用 5 层 local 配 1 层 global，只有 global 层需要完整长度的 cache；gpt-oss 让窗口层和全局层交替；StreamingLLM 永久保留最开头几个 sink token，因为 softmax 的权重总得放在某处，模型学会了把多余的权重丢在那里。

## RoPE：把位置变成旋转

RoPE 不往 embedding 上加位置向量，而是把 $q$ 和 $k$ 里每一对维度旋转一个与位置成正比的角度，每一对的转速各不相同。两个旋转后向量的点积只取决于角度之**差**，所以分数只依赖相对距离 $m-n$。

<!-- widget:tx-rope -->

点播放：两个向量一起转，分数纹丝不动——这就是“相对位置”的含义。再拖动距离 $\Delta$，或者直接在右侧曲线上滑动，看分数随距离衰减。base $\theta$ 越大转得越慢，曲线衰减得也越慢，这就是 Llama 3（$\theta=500\text{k}$）和 Gemma 3（global 层 $\theta=1\text{M}$）撑起长上下文的办法。推导见 [Transformer 架构 · RoPE](transformer.md)。

## MoE：参数更多，单个 token 的计算量不变

在 dense block 里，FFN 大约占三分之二的参数。MoE 层把它换成 $N$ 个 expert FFN，再加一个 router，只把每个 token 送到得分最高的 $k$ 个 expert：总参数量随 $N$ 增长，计算量只随 $k$ 增长。

<!-- widget:tx-moe -->

格子按真实数量画（8、128 或 256 个 expert），所以你看到的稀疏度就是真实的稀疏度。DeepSeek-V3 多了一个所有 token 都会经过的 shared expert。图里的 router 是随机的，只为演示负载分布；真实的 router 是一个学出来的线性层，负载均衡要靠辅助 loss 或 bias 项来维持。完整的一组笔记见 [MoE 系列](moe/)。

## FlashAttention：数学不变，少搬显存

在 GPU 上，attention 的瓶颈是访存而不是算术：标准实现会把完整的 $n\times n$ 分数矩阵写进较慢的 HBM，再读回来。FlashAttention 把 Q、K、V 切成能放进片上 SRAM 的小块，一块一块地算，每行只维护一个 running max 和归一化因子（online softmax），最后只写回 $n\times d$ 的输出。结果是**精确的**，不是近似，额外显存对 $n$ 线性。

<!-- widget:tx-flash -->

切到“标准 attention”，看 S 和 P 两个 $n\times n$ 矩阵怎样被完整写进 HBM；再切回来，注意右上角整块被 causal mask 直接跳过的 block。

## 模型家族速览

同一张选择清单，并排对比。数字来自各自的论文、技术报告和公开 config。

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
    <li>KV cache 把单步计算从 $O(t^2)$ 降到 $O(t)$，为什么解码反而变成了 memory-bound？</li>
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
