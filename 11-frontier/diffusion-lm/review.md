# 扩散语言模型（dLLM）：复习题

**中文** · [English](review.en.md)

> 阅读时间：约 6 分钟 · 最近审阅：2026-10

这一页把本系列的要点压成面试题和自检题；答案里的数字和结论都能在前两篇找到出处。

## 面试常见问题

<details class="interview" markdown="1">
<summary>掩码扩散的前向过程和训练目标是什么？$1/t$ 起什么作用？</summary>

前向过程：给定 $t\in[0,1]$，每个位置独立地以概率 $t$ 换成 `[M]`，变成 `[M]` 以后不再变回（吸收态）。训练目标：抽 $t\sim U(0,1]$ 和对应的 $x_t$，只在被遮的位置算交叉熵，再乘 $1/t$（LLaDA 式 3）。$t$ 小时平均只有 $tL$ 个位置贡献损失，乘 $1/t$ 后每个 $t$ 的期望份量都是 $L$ 个 token，整个目标成为负对数似然（NLL）的上界。换遮盖调度时，连续时间权重是 $-\alpha_t'/(1-\alpha_t)$，目标本身不变，变的是梯度方差。模型不需要知道 $t$，空位数已经写在输入里。

</details>

<details class="interview" markdown="1">
<summary>它和 BERT 的掩码语言模型（MLM）有什么区别？为什么 BERT 不能直接拿来生成？</summary>

网络可以一样，都是双向 Transformer；区别在遮盖比例的分布和加权。BERT 固定遮约 15%，选中位置取平均，没学过从全空开始，也没有似然解释。掩码扩散每条样本抽一个 $t$，从几乎不遮到全遮都练过，乘 $1/t$ 后与最大似然挂钩；MDLM 把它写成 MLM 损失的加权平均。LLaDA 也指出 MaskGIT 省掉 $1/t$，就失去了和最大似然的联系。

</details>

<details class="interview" markdown="1">
<summary>“任意顺序”是什么意思？用两个 token 的例子说明目标等于什么。</summary>

训练时目标对所有生成顺序取平均；推理时可以按任何顺序填空。两个 token：两个都遮的概率 $t^2$，只遮一个各 $t(1-t)$，乘 $1/t$ 后积分，两项权重都是 $1/2$，目标等于“从左往右”和“从右往左”两种自回归（AR）损失的平均。教学数字下是 $(1.833+2.996)/2=2.414$；模型实际生成这条序列的负对数似然是 $-\ln 0.105=2.254$，所以是上界，两种顺序不一致时就不紧。一般长度下，RADD 定理 2 证明它等价于任意顺序自回归的目标。代价是难的顺序也要学：同样验证损失约需 AR 16 倍计算（Nie 等，2024-10）。

</details>

<details class="interview" markdown="1">
<summary>一步落多个 token 为什么会出错？怎样决定一步落几个？</summary>

一步里的几个 token 各自从边缘分布选出，彼此没看到。答案一半 `full house`、一半 `high card` 时，一步落两个，合法率只有 0.5。Fast-dLLM 按置信度决定：超过阈值（默认 0.9）的都落，没有就落最高的一个；它的定理给出条件：$n$ 个位置的置信度都高于 $1-\epsilon$ 且 $(n+1)\epsilon\le1$ 时，并行取最大与联合取最大一致。固定步数的代价有第三方测量：LLaDA-8B-Instruct 在生成长度 128、块大小 32 的设置下每步固定 4 个时，GSM8K 从 73.6 降到 61.7（73.6 是同一设置下每步 1 个的基线）。Feng 等的定理 4.4 对词表大小为 16 的最坏情况 HMM 给出线性步数下界，前提是论文中的因子化反向过程和小学习误差假设。它把生成 HMM 支持集之外的序列计为错误，不是所有推理任务的加速上限。实际仍要比较相同任务准确率下的延迟。

</details>

<details class="interview" markdown="1">
<summary>为什么扩散 LM 不能直接用 KV cache？有哪两种补法，各有什么代价？</summary>

自回归里位置 $i$ 的 K、V 只依赖不会再变的 $x_{\le i}$；双向注意力下，从第 2 层起每个位置都读了每步都在变的空位，连提示词的 K、V 也会变，LLaDA 每步都要整段重算。补法一是近似缓存（Fast-dLLM）：相邻步的 KV 很相似，块内复用、块后刷新，要另外验证准确率。补法二是分块扩散：块间从左到右、块内扩散，已完成块的 KV 精确可缓存，块越小越接近自回归，困惑度更好，但并行度也更低。

</details>

<details class="interview" markdown="1">
<summary>并行解码为什么不会自动更快？用 roofline 估一次低并发和高并发。</summary>

用第二篇的教学模型（只算线性层，数字取自 roofline 一篇：7B、BF16、H100 SXM5 峰值），一次前向的下限是 $\max(4.18\,\text{ms},\ n\times0.01415\,\text{ms})$，约 295 个位置以下是读权重说了算。单条请求、提示词 512、生成 256：AR 约 1,076 ms；整段扩散不缓存，每步算 768 个位置，每步落 1 个要 2,782 ms，比 AR 还慢；分块 $G=32$ 加缓存、每步落 4 个约 275 ms，快 3.9 倍。并发上去后，扩散每落 1 个 token 要算 $G/\tau=8$ 个位置，封顶 8,834 tok/s，AR 封顶 70,671 tok/s，约 37 条并发时 AR 追平。所以低并发加速约为 $\tau$，算力饱和时吞吐比约为 $\tau/G$。第三方测量和 Google 对 DiffusionGemma 的说明都指向同一方向。

</details>

<details class="interview" markdown="1">
<summary>读到“某扩散模型 1000+ tokens/s、比自回归快 N 倍”，你会问什么？</summary>

四件事：多少并发；基线是不是同一张卡、同一个引擎；这个速度下质量是多少、用的什么阈值；计时里算没算 prefill、每块的提交前向和固定开销。例子：Gemini Diffusion 的 1479 tokens/s 不计 0.84 s 开销；Mercury 报告的对照速度是 Artificial Analysis 在各家云服务商上测得的中位数；LLaDA2.0 论文的扩散模型和基线用的是不同引擎（dInfer 对 SGLang），SGLang 博客把两边放在同一引擎上时，倍数从约 2.1 降到约 1.9；Mercury 2.5 没写 GPU 型号。质量也要同时看：DiffusionGemma 模型卡上多数基准低于同规格 Gemma 4。截至 2026 年 10 月，这些数字都来自厂商或方法作者。

</details>

<details class="interview" markdown="1">
<summary>DFlash 和原生扩散 LM 有什么区别？</summary>

DFlash 只用分块扩散写草稿：小模型读目标模型的特征，一次前向写一块，最后由自回归的目标模型验证，只保留连续接受的前缀，所以输出分布等于目标模型的分布，质量风险不在扩散一侧。原生扩散 LM 自己决定最终 token，要承担并行落 token 不一致和同族质量更低的风险。两者的速度账同构：都是一次前向算一整块，接受（或落下）得越多越划算，高并发时收益都会缩小。

</details>

## 自检

<div class="taste-check">
  <strong>如果真的理解了，你应该能解释：</strong>
  <ol>
    <li>把 $1/t$ 去掉，两个 token 的目标变成什么？为什么它不再有似然解释？</li>
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
