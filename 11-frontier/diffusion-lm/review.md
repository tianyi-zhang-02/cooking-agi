# 扩散语言模型（dLLM）：复习题

**中文** · [English](review.en.md)

> 阅读时间：约 6 分钟 · 最近审阅：2026-10

这一页整理本系列的面试题和自检题；答案里的数字和结论都能在前面的笔记中找到出处。

## 面试常见问题

<details class="interview" markdown="1">
<summary>掩码扩散的前向过程和训练目标是什么？$1/t$ 起什么作用？</summary>

前向过程：给定 $t\in[0,1]$，每个位置独立地以概率 $t$ 换成 `[M]`，变成 `[M]` 以后不再变回（吸收态）。训练目标：抽 $t\sim U(0,1]$ 和对应的 $x_t$，只在被遮的位置算交叉熵，再乘 $1/t$（LLaDA 式 3）。$t$ 小时平均只有 $tL$ 个位置贡献损失，乘 $1/t$ 后每个 $t$ 的期望监督权重都是 $L$，但实际损失仍取决于预测难度。完整的期望目标是负对数似然（NLL）的上界，不是每个随机遮盖样本都单独给出一个上界。

对同一理想去噪条件分布，用覆盖相同端点的平滑单调日程重参数化，并使用对应权重 $-\alpha_t'/(1-\alpha_t)$，连续时间目标不变；训练采样的方差和实际优化仍可能不同。独立、与内容无关的遮盖下，给定可见内容后的理想词分布不依赖 $t$，所以预测网络可以不接收时间。采样器仍需要时间来决定揭开概率，不能把这两层混在一起。见 [MDLM 算例](mdlm-walkthrough.md)。

</details>

<details class="interview" markdown="1">
<summary>它和 BERT 的掩码语言模型（MLM）有什么区别？为什么 BERT 不能直接拿来生成？</summary>

网络都可以是双向 Transformer，区别在训练问题和生成过程。BERT 通常选约 15% 的位置作为预测目标，其中只有一部分替换成 `[MASK]`；它的原始目标没有同时指定“从全空开始逐步生成”的过程。掩码扩散抽取不同遮盖程度，再按 $1/t$ 加权，MDLM 将这个目标写成 MLM 损失的加权平均，并推导出对应反向生成过程的似然下界。

不宜简单记成“BERT 没有似然解释”：MLM 可以用于[伪似然评分](https://arxiv.org/abs/1910.14659)，但这不等于直接得到序列的归一化联合概率。类似地，删掉 $1/t$ 会改变这里的生成目标，不能沿用同一个上界推导；并不意味着任何未加权的填空模型都不可能定义生成过程。见[训练目标里的两词推导](masked-diffusion.md)和 [MDLM 的训练、采样对照](mdlm-walkthrough.md)。

</details>

<details class="interview" markdown="1">
<summary>“任意顺序”是什么意思？用两个 token 的例子说明目标等于什么。</summary>

训练时目标对所有生成顺序取平均；推理时可以按不同顺序填空。两个 token 都被遮住的概率是 $t^2$，只遮一个的两种情况各有概率 $t(1-t)$。乘 $1/t$ 后积分，两项权重都是 $1/2$，目标等于“从左往右”和“从右往左”两种自回归损失的平均。

代入前文的教学数字，目标是 $(1.833+2.996)/2=2.414$；随机选择这两种顺序时，模型生成该序列的负对数似然是 $-\ln 0.105=2.254$。前者是上界；两种顺序给出的概率不一致时，上界就不紧。

一般长度下，RADD 定理 2 证明它等价于任意顺序自回归的目标。难的顺序也要学，但不能由此推出固定的计算倍率：Nie 等（2024-10）报告的约 16 倍对应其评分与训练设置，[后续比较](ar-and-diffusion.md)需要区分评分、架构与数据条件。

</details>

<details class="interview" markdown="1">
<summary>一步落多个 token 为什么会出错？怎样决定一步落几个？</summary>

一步里的几个 token 各自从边缘分布选出，彼此没看到。答案一半 `full house`、一半 `high card` 时，一步落两个，合法率只有 0.5。Fast-dLLM 按置信度决定：超过阈值（默认 0.9）的都落，没有就落最高的一个；它的定理给出条件：$n$ 个位置的置信度都高于 $1-\epsilon$ 且 $(n+1)\epsilon\le1$ 时，并行取最大与联合取最大一致。固定步数的代价有第三方测量：LLaDA-8B-Instruct 在生成长度 128、块大小 32 的设置下每步固定 4 个时，GSM8K 从 73.6 降到 61.7（73.6 是同一设置下每步 1 个的基线）。Feng 等的定理 4.4 对词表大小为 16 的最坏情况 HMM 给出线性步数下界，前提是论文中的因子化反向过程和小学习误差假设。它把生成 HMM 支持集之外的序列计为错误，不是所有推理任务的加速上限。实际仍要比较相同任务准确率下的延迟。

</details>

<details class="interview" markdown="1">
<summary>为什么扩散 LM 不能直接用 KV cache？有哪两种补法，各有什么代价？</summary>

自回归里位置 $i$ 的 K、V 只依赖不会再变的 $x_{\le i}$。双向注意力则不同：前一层读过的空位可能在下一轮变成词，所以从第 2 层起，连提示词的 K、V 也可能变化。原始 LLaDA 因此每轮重新计算整段，不能直接套用 AR 的精确缓存规则。

一种办法是近似缓存，例如 Fast-dLLM：利用相邻步骤 KV 较相似的现象，在块内复用并定期刷新，再检查质量损失。另一种是分块扩散：块间从左到右，块内去噪，用块因果注意力保证已完成前缀不再受当前块影响。此时前缀 KV 可以精确复用，但块完成后通常还要根据最终 token 计算可缓存的干净表示。

小块更接近 AR，也减少了一次并行生成的位置数。Block Diffusion 论文在其训练设置下报告了小块更低的困惑度上界，但不能保证任意模型只要缩小块就更好。具体缓存过程见[分块例子](ar-and-diffusion.md#block-cache)，attention 输出能否复用则是[另一个问题](context-reuse.md)。

</details>

<details class="interview" markdown="1">
<summary>并行解码为什么不会自动更快？用 roofline 估一次低并发和高并发。</summary>

用[解码与速度](decoding-and-speed.md)里的教学模型：只算线性层，采用 7B、BF16、H100 SXM5 的峰值规格。一次前向的耗时下限是 $\max(4.18\,\text{ms},\ n\times0.01415\,\text{ms})$；约 295 个位置以下，主要受读取权重限制。

单条请求、提示词 512、生成 256：AR 约 1,076 ms；整段扩散不缓存，每步算 768 个位置，每步落 1 个要 2,782 ms，比 AR 还慢；分块 $G=32$ 加缓存、每步落 4 个约 275 ms，快 3.9 倍；若每完成一块再做一次提交前向，8 块多出约 33 ms，总计约 308 ms，快约 3.5 倍。

并发上去后，扩散每落 1 个 token 要算 $G/\tau=8$ 个位置。这个简化模型给出的吞吐上限是 8,834 tok/s，AR 是 70,671 tok/s，约 37 条并发时 AR 追平。所以低并发加速约为 $\tau$，算力饱和时吞吐比约为 $\tau/G$。这些是估算，不是实测；第三方测量和 Google 对 DiffusionGemma 的说明也提示了低并发延迟与高并发吞吐之间的差别。

</details>

<details class="interview" markdown="1">
<summary>读到“某扩散模型 1000+ tokens/s、比自回归快 N 倍”，你会问什么？</summary>

四件事：多少并发；基线是不是同一张卡、同一个引擎；这个速度下质量是多少、用的什么阈值；计时里算没算 prefill、每块的提交前向和固定开销。例子：Gemini Diffusion 的 1479 tokens/s 不计 0.84 s 开销；Mercury 报告的对照速度是 Artificial Analysis 在各家云服务商上测得的中位数；LLaDA2.0 论文的扩散模型和基线用的是不同引擎（dInfer 对 SGLang），SGLang 博客把两边放在同一引擎上时，倍数从约 2.1 降到约 1.9；Mercury 2.5 没写 GPU 型号。质量也要同时看：DiffusionGemma 模型卡上多数基准低于同规格 Gemma 4。截至 2026 年 10 月，这些数字都来自厂商或方法作者。

</details>

<details class="interview" markdown="1">
<summary>DFlash 和原生扩散 LM 有什么区别？</summary>

DFlash 用分块扩散写草稿：小模型读取目标模型的特征，一次前向预测一块，再由 AR 目标模型验证，保留连续通过的前缀。草稿不直接决定最终输出。[DFlash 原文](https://arxiv.org/html/2602.06036v1#S4)

这里“保持目标分布”有条件：验证和后续采样必须采用正确的精确算法。贪心解码要保留目标模型的选择；随机采样则需要相应的接受、拒绝与补采样规则，不能只按置信度留下看起来不错的词。满足这些条件时，草稿预测差主要影响接受长度和速度，而不是改变目标分布。[投机解码的原始推导](https://proceedings.mlr.press/v202/leviathan23a.html)

原生扩散 LM 自己决定最终 token，因此必须评估并行采样的一致性及实际任务质量，不能借用另一个目标模型的质量保证。两类方法都要计算草稿或去噪、验证或提交的成本。一次通过的 token 越多，通常越容易摊薄成本；高并发时还要看额外计算与调度开销，不能直接沿用单请求的加速倍数。

</details>

## 自检

<div class="taste-check">
  <strong>可以用下面的问题检查理解：</strong>
  <ol>
    <li>把 $1/t$ 去掉，两个 token 的目标变成什么？为什么不能直接沿用原来的似然上界推导？</li>
    <li>LLaDA 在 GSM8K 上的分数是每步落几个 token 测的？这对“又好又快”的说法意味着什么？</li>
    <li>扑克的例子里，为什么一步落两个只有一半是合法牌型？Fast-dLLM 的阈值规则会怎么处理它？</li>
    <li>整段扩散不缓存时，为什么每步只落 1 个 token 反而比 AR 慢？换成分块加缓存后，账哪里变了？</li>
    <li>在算力饱和的区间，扩散和 AR 的吞吐比为什么约等于 $\tau/G$？给出一组 $\tau$ 和 $G$，算一次交叉点。</li>
  </ol>
</div>

## 参考论文

- [Mask-Predict](https://arxiv.org/abs/1904.09324)：翻译里的并行预测和迭代重遮
- [D3PM](https://arxiv.org/abs/2107.03006)：离散扩散与吸收态
- [Autoregressive Diffusion Models](https://arxiv.org/abs/2110.02037)：任意顺序自回归和吸收态扩散的统一
- [Diffusion-LM](https://arxiv.org/abs/2205.14217)：在词向量上做连续扩散
- [SEDD](https://arxiv.org/abs/2310.16834)：score entropy 离散扩散
- [MDLM](https://arxiv.org/abs/2406.07524)：简化的目标，MLM 损失的加权平均
- [RADD](https://arxiv.org/abs/2406.03736)：去掉时间条件；与任意顺序自回归等价的定理
- [Masked Diffusion Models are Secretly Time-Agnostic Masked Models](https://arxiv.org/abs/2409.02908)：类别采样的精度问题和 generative perplexity
- [Scaling up Masked Diffusion Models on Text](https://arxiv.org/abs/2410.18514)：scaling law 和约 16 倍的计算差距
- [Train for the Worst, Plan for the Best](https://arxiv.org/abs/2502.06768)：难的子问题与自适应解码顺序
- [Theoretical Benefit and Limitation of Diffusion Language Model](https://arxiv.org/abs/2502.09622)：步数与序列正确率
- [LLaDA](https://arxiv.org/abs/2502.09992)：8B 从零训练、采样和自报的扩展性
- [ReMDM](https://arxiv.org/abs/2503.00307)：重遮已解码 token 的采样器
- [Block Diffusion](https://arxiv.org/abs/2503.09573)：分块扩散与精确 KV cache
- [d1](https://arxiv.org/abs/2504.12216)：掩码扩散 LM 上的 RL（diffu-GRPO）
- [Fast-dLLM](https://arxiv.org/abs/2505.22618)：近似 KV cache 和按置信度并行解码
- [Mercury](https://arxiv.org/abs/2506.17298)：Inception 的技术报告（速度数字自报）
- [Diffusion Beats Autoregressive in Data-Constrained Settings](https://arxiv.org/abs/2507.15857)：数据重复多轮时的比较
- [Seed Diffusion](https://arxiv.org/abs/2508.02193)、[Dream 7B](https://arxiv.org/abs/2508.15487)：2025 年的大规模扩散 LM
- [Beyond Next-Token Prediction](https://arxiv.org/abs/2510.04146)：LLaDA 对 LLaMA-3 的第三方性能刻画
- [LLaDA2.0](https://arxiv.org/abs/2512.15745)、[LLaDA2.1](https://arxiv.org/abs/2602.08676)：从自回归转换的 100B MoE 与 token 编辑
- [DFlash](https://arxiv.org/abs/2602.06036)：分块扩散草稿模型加 AR 验证
- [Scaling Beyond Masked Diffusion Language Models](https://arxiv.org/abs/2602.15014)：均匀态扩散的 scaling law
- [Neither Parallel Nor Sequential](https://arxiv.org/abs/2606.14620)：DiffusionGemma 的落 token 顺序
- [Gemini Diffusion 发布](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-diffusion/)与[模型页](https://deepmind.google/models/gemini-diffusion/)：实验性演示与自报速度
- [DiffusionGemma 发布](https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/)、[模型卡](https://ai.google.dev/gemma/docs/diffusiongemma/model_card)、[开发者指南](https://developers.googleblog.com/diffusiongemma-the-developer-guide/)：结构、采样器、自报速度与质量
- [Mercury 2](https://www.inceptionlabs.ai/blog/introducing-mercury-2)、[Mercury 2.5](https://www.inceptionlabs.ai/blog/introducing-mercury-2-5)：Inception 2026 年的自报数字
- [SGLang 的 dLLM 支持](https://lmsys.org/blog/2025-12-19-diffusion-llm)与[文档](https://lmsysorg.mintlify.app/docs/supported-models/diffusion_language_models)：分块扩散接在 chunked prefill 上；FDFO 调度
