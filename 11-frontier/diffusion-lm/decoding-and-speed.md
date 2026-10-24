# 扩散语言模型（dLLM）：解码步数、缓存与真实速度

**中文** · [English](decoding-and-speed.en.md)

> 阅读时间：约 11 分钟 · 最近审阅：2026-10

LLaDA 8B Instruct 在 GSM8K 上的分数，是每次前向只落下 1 个 token 测出来的：答案长 512，就走 512 步（[LLaDA 附录 B.6](https://arxiv.org/abs/2502.09992)）。第三方在生成长度 128、块大小 32 的设置下，把同一个模型改成每步固定落 4 个 token，GSM8K 从 73.6 掉到 61.7（[Kim 等，2025-10](https://arxiv.org/abs/2510.04146)）。

估算扩散模型的速度，需要看两个数：在质量可接受时，平均每步能确定几个 token（记作 $\tau$），以及每一步要重算多少个位置（记作 $G$）。下文的简化 roofline 模型预测：低并发、decode 受显存带宽限制且算力尚有余量时，速度最多提高约 $\tau$ 倍；并发高到算力用满后，每个 token 的计算量约为 AR 的 $G/\tau$ 倍，预测的优势会消失，甚至反转。这些比例依赖模型的假设；实际交叉点还要计入 attention、KV 读写及实现开销后测量。

本篇接着[掩码扩散的训练目标](masked-diffusion.md)，算账用 [roofline 一篇](../../12-infra/performance.md#decode-batching)的记号和 H100 数字。

## 一步在做什么：预测所有空位，只留下有把握的

这一节用 LLaDA 的置信度选择举例；[MDLM 的基础采样器](mdlm-walkthrough.md)按日程随机揭开位置，两者不要混成同一个算法。

LLaDA §2.4 的采样从全空的回答开始，生成长度和步数都是超参数。从时刻 $t$ 走到 $s$ 的一步：把提示词和部分被遮的回答一起送进模型，一次前向预测所有空位，再把其中约 $s/t$ 的预测重新遮住。默认的“低置信度重遮”（low-confidence remasking）留下最有把握的预测，遮回其余的；EOS 之后的内容最后丢掉。

拿 `the tea is cold` 走一遍：生成长度 4，分 2 步，每步留 2 个。置信度是教学数字。

| 步 | 输入 | 每个空位最可能的词和置信度 | 留下 | 输出 |
| --- | --- | --- | --- | --- |
| 1 | `[M] [M] [M] [M]` | the 0.9，tea 0.3，is 0.6，cold 0.4 | the、is | `the [M] is [M]` |
| 2 | `the [M] is [M]` | tea 0.7，cold 0.8 | 全部 | `the tea is cold` |

第 1 步 `tea` 只有 0.3，被遮回去；第 2 步有了 `the` 和 `is` 作上下文，升到 0.7。这里的“重遮”只针对本步的新预测：已经留下的 `the` 和 `is` 以后不会再改。写成伪代码：

```python
def sample(model, prompt, gen_len, steps):
    x = prompt + [MASK] * gen_len
    for k in range(steps):
        probs = model(x)                          # 一次完整的双向前向
        masked = [i for i in range(len(x)) if x[i] == MASK]
        n_keep = len(masked) // (steps - k)       # 剩下的空位平摊到剩下的步数
        conf = {i: probs[i].max() for i in masked}
        for i in sorted(masked, key=conf.get, reverse=True)[:n_keep]:
            x[i] = probs[i].argmax()              # 落下，之后不再改
    return x
```

**能改已经确定的 token 吗？** 上面的基础采样器不能，但这不是所有掩码扩散方法的限制。[Mask-Predict](https://arxiv.org/abs/1904.09324)（Ghazvininejad 等，2019-04，EMNLP 2019）在机器翻译里会反复遮住所有位置中最没把握的词，重新预测。[ReMDM](https://arxiv.org/abs/2503.00307)（Wang 等，2025-03，NeurIPS 2025）给预训练好的掩码扩散模型加上重遮已解码 token 的采样器，作者报告增加步数后，质量更接近自回归。[LLaDA2.1](https://arxiv.org/abs/2602.08676)（2026-02）在“把空位填成词”之外加了“把词改成另一个词”的编辑。

DiffusionGemma 的[模型卡](https://ai.google.dev/gemma/docs/diffusiongemma/model_card)写，每步没被选中的 token 会被完全重新加噪；[开发者指南](https://developers.googleblog.com/diffusiongemma-the-developer-guide/)也说，置信度下降的 token 可以重新加噪、换掉。能不能修改已生成内容，要看具体采样器，而不只是模型是否叫扩散模型。

## 一步落几个：并行解码为什么会写出 high house

一步里同时确定的几个 token，分别从各位置的边缘分布采样，还不知道其他位置最终选了什么。[Fast-dLLM](https://arxiv.org/abs/2505.22618)（Wu 等，2025-05）用扑克牌型举例。设答案一半是 `full house`、一半是 `high card`（教学分布）。第 1 位的边缘分布是 full 0.5、high 0.5，第 2 位是 house 0.5、card 0.5。

一步独立采样两个位置，拼出合法牌型的概率只有 $0.5\times0.5+0.5\times0.5=0.5$，另一半是 `full card` 或 `high house`。若分两步，先选一个，再按条件分布选另一个，理想模型的合法率就回到 1。[训练目标一篇](masked-diffusion.md)里的 `hot tea` 也类似：一步独立采样两个位置，生成它的概率是 $0.2\times0.1=0.02$；按例子里的两种顺序等概率逐词生成，则是 0.105。

Fast-dLLM 的办法是按置信度决定落几个：置信度超过阈值（默认 0.9）的位置都落下，一个都没有就只落最高的那个。它的定理 1 说：若 $n$ 个位置的边缘置信度都高于 $1-\epsilon$，且 $(n+1)\epsilon\le1$，那么各自取最大值，和对联合分布取最大值，选出的是同一个序列。3 个位置都在 0.95 以上时，$(3+1)\times0.05=0.2\le1$，可以一起落；扑克的例子里 $\epsilon=0.5$、$n=2$，$3\times0.5=1.5>1$，没有保证，而且阈值规则也只会先落一个。于是 $\tau$ 会随内容变化：容易确定的片段可以一步落很多，难的地方一次一个；DiffusionGemma 的模型卡也写，简单的提示和代码这类结构化任务需要的去噪步数更少。

固定 $\tau$ 的代价有第三方测量。Kim 等（2025-10）用 LLaDA-8B-Instruct，生成长度 128、块大小 32，并开着 Fast-dLLM 的近似缓存：

| 每步固定落下 | 1 | 2 | 4 | 8 | 32 |
| --- | --- | --- | --- | --- | --- |
| 步数 | 128 | 64 | 32 | 16 | 4 |
| GSM8K（4-shot） | 73.6 | 70.6 | 61.7 | 29.7 | 6.1 |
| HumanEval（pass@1） | 35.4 | 32.3 | 17.7 | 9.1 | 3.7 |

73.6 是这套设置下每步落 1 个的基线，和 LLaDA 自报的 69.4（生成长度 512、不分块）不是同一套设置，两者不能直接比；表里能比的是同一行从左到右的下降。

按置信度自适应要好得多，但数字来自方法作者自己：Fast-dLLM 报告 LLaDA 在 GSM8K（5-shot，生成长度 256，A100 80GB）上从 79.3 到 78.5，吞吐从 6.7 升到 54.4 tokens/s，即 8.1 倍，这是缓存和并行解码合起来的效果。Google 的模型卡自报 DiffusionGemma 每次前向生成 15–20 个 token，最多 48 个去噪步。

[Feng 等（2025-02）的定理 4.2–4.4](https://arxiv.org/html/2502.09622v2#S4)讨论的范围更具体：目标分布和误差指标不同，所需步数会怎样变。对于 **n-gram 语言**，定理 4.2 在固定 n-gram 阶数和精度、满足学习误差与长度条件时，给出了不随序列长度增长的步数界。定理 4.4 则构造了一个**词表大小为 16 的隐马尔可夫模型（HMM）作为最坏情况**：对论文研究的逐位置因子化掩码反向过程，在假设 4.1、逐 token 反向分布的 KL 误差 $\epsilon_{\rm learning}<1/128$ 下，要把序列错误率降下来，至少需要随长度线性增长的步数。这里的“序列错误”指生成了 HMM 支持集之外的序列，不是对任意推理任务测出的错误率。

它提醒我们：困惑度不错，不代表多个位置一起选就可靠。**但这不是说所有推理任务、所有采样器都不可能加速。** 真正选模型时，仍要在相同任务准确率下比较延迟，不能只凭一个最坏情况定理，也不能只凭 tokens/s。

## KV cache：双向注意力为什么缓存不了

AR 里，位置 $i$ 的 K、V 只依赖 $x_{\le i}$，这些 token 不会再变，缓存是精确的（见 [KV cache 篇](../../00-foundations/deep-dives/kv-cache-and-inference.md)和[缓存必须等价](../../00-foundations/deep-dives/language-model-objective.md#cache)）。双向注意力下，从第 2 层起，每个位置的隐藏状态都读了所有位置，包括每步都在变的空位，连提示词的 K、V 也跟着变。LLaDA §2.2 直接写明它和 KV cache 不兼容，每一步都要把提示词和整段回答重算一遍。

有两种补法：

- **近似缓存。** Fast-dLLM 观察到同一块内相邻两步的 KV 非常相似，于是在一个块内复用前缀（PrefixCache）甚至后缀（DualCache）的 KV，每写完一个块再整体刷新。它是近似，作者报告准确率保持在基线 1–2 分之内。
- **改结构：分块扩散。** Block Diffusion（BD3-LM）让块与块之间从左到右，块内做扩散；前面的块不看后面的块，所以已完成块的 KV 能精确缓存。块大小为 1 就退回 AR。OpenWebText 上，块越小困惑度越接近 AR：AR 17.54，块大小 4、8、16 分别是 ≤20.73、≤21.68、≤22.27，整段的 MDLM ≤22.98。LLaDA2.0 评测时块大小是 32；DiffusionGemma 每次去噪一块 256 个 token，块写完后用一次因果注意力的前向把它写进 KV cache（[开发者指南](https://developers.googleblog.com/diffusiongemma-the-developer-guide/)）。SGLang 在 2025-12-19 的[博客](https://lmsys.org/blog/2025-12-19-diffusion-llm)里说，它把分块扩散接在已有的 chunked prefill 路径上。

分块还顺带提高质量：Kim 等测到，不分块时 GSM8K 63.2，块大小 32 时 73.6，16 时 75.7（生成长度 128）。代价是块间又变回串行，块越小越像 AR。

## 手算一次账（教学模型）：同一张卡上各花多少时间

**这一节是教学模型，不是实测。** 它只用 [roofline 一篇](../../12-infra/performance.md#decode-batching)的三个数：7B 模型的 BF16 权重约 $1.4\times10^{10}$ 字节，H100 SXM5 带宽 3.352 TB/s，BF16 dense 峰值 989.4 TFLOPS。读一遍权重的时间下限约为 $\frac{1.4\times10^{10}\ \text{bytes}}{3.352\times10^{12}\ \text{bytes/s}}\approx4.1766\times10^{-3}\ \text{s}\approx4.18\ \text{ms}$；每处理一个位置的计算约 $2N=1.4\times10^{10}$ FLOPs，按峰值是 0.01415 ms。忽略 attention 的计算、KV 的读取和各种开销，一次前向处理 $n$ 个位置的时间下限是

$$
T_{\text{step}}(n)=\max\big(4.18\ \text{ms},\;n\times0.01415\ \text{ms}\big)
$$

上式展示舍入值，下面的表格和代码用未舍入常数计算，交叉点约为 295.167 个位置。低于交叉点，读权重说了算，多算几个位置几乎不花时间；高于交叉点，计算说了算。先看单条请求、提示词 512、生成 256 的教学估算：

| 方案 | 每步处理的位置 | 每步下限 | 步数 | 总时间 | 相对 AR |
| --- | --- | --- | --- | --- | --- |
| AR + KV cache | 1（另有一次 512 的 prefill，7.24 ms） | 4.18 ms | 256 | 1,076 ms | 1× |
| 整段扩散，无缓存，每步落 1 个 | 768 | 10.87 ms | 256 | 2,782 ms | 0.39× |
| 同上，每步落 4 个 | 768 | 10.87 ms | 64 | 696 ms | 1.55× |
| 分块 $G=32$ + 前缀缓存，每步落 1 个 | 32 | 4.18 ms | 256 | 1,076 ms | 1.0× |
| 同上，每步落 4 个 | 32 | 4.18 ms | 64 | 275 ms | 3.9× |

在这个估算中，无缓存时每步都要重算 768 个位置，越过了 295，一步的时间下限就是 AR 的 2.6 倍；每步只落 1 个 token 时，总时间下限也是 AR 的 2.6 倍。分块加缓存后，每步只算 32 个位置，仍由权重读取决定时间下限，所以每步落 4 个，估算的速度就接近 AR 的 4 倍。若每完成一块再做一次写缓存的前向（DiffusionGemma 就是这样做的），8 块多出 $8\times4.18=33$ ms，总时间下限为 308 ms，估算加速约 3.5 倍。

再看多条请求并发（仍是教学模型，只算 decode 阶段，分块扩散 $G=32$、$\tau=4$）：

| 并发请求数 | AR 每步位置 | AR 吞吐上限 | 扩散每步位置 | 扩散吞吐上限 |
| --- | --- | --- | --- | --- |
| 1 | 1 | 239 tok/s | 32 | 958 tok/s |
| 8 | 8 | 1,915 tok/s | 256 | 7,662 tok/s |
| 64 | 64 | 15,323 tok/s | 2,048 | 8,834 tok/s |
| 295 | 295 | 70,631 tok/s | 9,440 | 8,834 tok/s |

按未舍入常数，扩散的算力交叉点在约 $295.167/32\approx9.224$ 条并发。9 条时只有 $9\times32=288$ 个位置，仍略受带宽限制；10 条起才封顶在约 8,834 tok/s。表中 AR 在 295 条并发时的 70,631 tok/s 也略低于计算上限，296 条起达到约 70,671 tok/s。按未舍入的平台值计算，扩散恰好是 AR 的 1/8：每落下 1 个 token，要算 $G/\tau=8$ 个位置。

AR 追平扩散的连续交点约为 36.90 条并发，因此整数并发从 37 起，AR 更快。两张表都能用下面几行复算：

```python
W = 1.4e10 / 3.352e12 * 1e3      # 读一遍权重：4.18 ms
C = 1.4e10 / 989.4e12 * 1e3      # 每个位置的计算：0.01415 ms
step = lambda n: max(W, n * C)   # 一次前向处理 n 个位置的下限
prefill = step(512)              # 7.24 ms
print(round(prefill + 256 * step(1)), round(256 * step(768)), round(64 * step(768)))  # 1076 2782 696
print(round(prefill + 64 * step(32)), round(prefill + 72 * step(32)))     # 275 308（后者含 8 次写缓存）
for b in (1, 8, 64, 295):
    print(b, round(b / step(b) * 1e3), round(4 * b / step(32 * b) * 1e3))  # 并发 b 时 AR、扩散的 tok/s
```

两种情形合起来：

$$
\frac{T_{\mathrm{AR}}}{T_{\mathrm{diff}}}\Big|_{\text{low batch}}\approx\tau,\qquad
\frac{\Theta_{\mathrm{diff}}}{\Theta_{\mathrm{AR}}}\Big|_{\text{compute-bound}}\approx\frac{\tau}{G}
$$

$\Theta$ 是吞吐。这个账忽略了 attention 和 KV。roofline 一篇的第二张表把 KV 算进来后，上下文 4096、batch 64 的 AR 上限只剩 4,440 tok/s，所以真实的交叉点要另外量。定性地看，扩散读 KV 的开销同样按 $\tau$ 摊薄（每步读一次、落下 $\tau$ 个），但 attention 的计算也要多付 $G/\tau$ 倍。

**实测方向和这个账一致。** Kim 等（A100 80GB，提示词和生成都是 128，块大小 32）测到 LLaMA-3-8B-Instruct 的吞吐能随 batch 增长到 16，LLaDA 在 8 左右就不涨了；提示词 2k、batch 1、不分块时，即使生成很短，LLaDA 的延迟也更高。Fast-dLLM 附录（A100，prefill 256，batch 1–32）发现小 batch 时带前缀缓存的 LLaDA 吞吐不低于 LLaMA，batch 变大后跟不上，因为 LLaMA 从带宽受限走向算力受限。Google 对 DiffusionGemma 的[发布文章](https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/)（2026-06-10）也写：优势在单卡、中低 batch 时最明显，高 QPS 的云端服务里收益递减，还可能抬高服务成本；Apple Silicon 这类统一内存设备未必有同样的加速。

## 厂商数字怎么读

下面都是公司或方法作者自报，条件各不相同，不能互相比较：

| 日期 | 系统 | 自报的数字 | 没说清的 |
| --- | --- | --- | --- |
| 2025-05-20 | [Gemini Diffusion](https://deepmind.google/models/gemini-diffusion/) | 各评测平均采样速度 1479 tokens/s，不计 0.84 s 的固定开销 | 模型大小、硬件、batch |
| 2025-06 | [Mercury Coder](https://arxiv.org/abs/2506.17298) Mini / Small | H100 上 1109 / 737 tokens/s，报告说来自 Artificial Analysis 对 Inception API 的独立测评 | 模型大小、batch；推理引擎只说是自研 |
| 2025-12 | [LLaDA2.0](https://arxiv.org/abs/2512.15745)-flash-CAP | 535 TPS，两个 AR 基线 256 和 237 TPS，“最多 2.1 倍” | 硬件和 batch；扩散跑 dInfer、基线跑 SGLang |
| 2026-02-24 | [Mercury 2](https://www.inceptionlabs.ai/blog/introducing-mercury-2) | Blackwell GPU 上 1009 tokens/s，“快 5 倍以上” | 基线、负载、谁测的 |
| 2026-06-10 | DiffusionGemma 26B A4B | [发布文章](https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/)：最多快 4 倍，单张 H100 上 1000+ tokens/s；[模型卡](https://ai.google.dev/gemma/docs/diffusiongemma/model_card)：低 batch、H100、FP8 时 1100+ | 4 倍的基线和 batch |
| 2026-09-08 | [Mercury 2.5](https://www.inceptionlabs.ai/blog/introducing-mercury-2-5) | 1107 tokens/s，“广泛可得的 NVIDIA GPU” | GPU 型号、基准分数、参数量 |

几个具体的坑。Mercury 报告里，对照模型的速度是 Artificial Analysis 在各家云服务商上测得的中位数，硬件、引擎和模型大小都不同；负载是约 1000 token 输入、1000 token 输出的编程提示。LLaDA2.0 论文没说 256 和 237 分别对应 Ling-flash-2.0 和 Qwen3-30B-A3B 中的哪一个，而且两边引擎不同；SGLang 的博客把两边都放在 SGLang 上（H20、TP8、小 batch），同一个 CAP 模型是 500 TPS 对 258 和 237，约 1.9 倍。LLaDA 作者自己在附录 B.7 报告，A100、batch 1、输出 256 时，基座模型在 GSM8K 和 MATH 上达到相当准确率时，吞吐是带 KV cache 的 LLaMA3 8B 的 1.5 倍和 1.8 倍。质量也要一起看：Gemini Diffusion 自己的表里，GPQA Diamond 40.4% 对 Gemini 2.0 Flash-Lite 的 56.5%，HumanEval 89.6% 对 90.2%。

读到“N 倍快”时先问四件事：多少并发；基线是不是同一张卡、同一个引擎；这个速度下质量是多少、用的什么阈值；时间里算没算 prefill、每块的提交前向和固定开销。

## 和 DFlash 的关系：扩散只管写草稿

[DFlash](../../00-foundations/deep-dives/dflash.md)（2026-02，ICML 2026）走的是另一条路：用一个小的分块扩散模型，读目标模型的特征，一次前向写出一块草稿；最终由 AR 的目标模型验证，只留下连续接受的前缀。目标模型的输出分布不变，质量风险就不在扩散一侧。作者自报在多种模型和任务上无损加速 6 倍以上，比 EAGLE-3 的加速高最多 2.5 倍。它的账和上面同构：草稿和验证都要算一整块，接受得越多越划算。

| 路线 | 谁决定最终 token | 质量风险 | 速度从哪里来 | 高并发时 |
| --- | --- | --- | --- | --- |
| 原生扩散 LM（LLaDA、DiffusionGemma） | 扩散模型自己 | 并行落下的 token 不一致；同族对比质量更低 | 每步落 $\tau$ 个 | 每个 token 多付约 $G/\tau$ 倍计算 |
| 扩散草稿 + AR 验证（DFlash） | AR 目标模型 | 不改变目标模型的分布 | 每轮接受多个草稿 token | 草稿和验证都占算力，收益同样缩小 |
| 纯自回归 | AR 模型 | 无额外风险 | 无并行 | 吞吐随 batch 涨到算力饱和 |

## 动手时怎么检查

| 先试什么 | 应该观察什么 | 不能据此推出什么 |
| --- | --- | --- |
| 固定一个质量目标，扫阈值或步数，画“质量–平均 $\tau$”曲线 | $\tau$ 增大到哪里质量开始掉 | 某一个点上的速度代表这个模型 |
| 同一张卡、同一个引擎，并发从 1 扫到 64 | AR 和扩散吞吐曲线的交点 | batch 1 的加速代表整台服务器的收益 |
| 打开近似缓存前后，逐题对比准确率 | 差距是否在 1–2 分以内 | KV 相似度接近 1，就说明结果等价 |
| 计时包括 prefill、每块的提交前向和固定开销 | 端到端延迟和首 token 时间 | 只看“采样速度”就能估出用户等多久 |
| 记录每一步落下几个 token 的分布 | 难题和简单片段的 $\tau$ 差多少 | 平均 $\tau$ 能代表每一道题 |

下一篇：[自回归与扩散：怎么比较，又怎么结合](ar-and-diffusion.md)，用一个分块例子看缓存，再看混合生成和近期进展。
