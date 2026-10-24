# 扩散语言模型（dLLM）：不从左到右，整块去噪

**中文** · [English](README.en.md)

> 阅读时间：约 7 分钟 · 最近审阅：2026-10

自回归通常根据已经写出的前缀，接着预测下一个 token。扩散语言模型换了一种生成方式：从带噪的状态出发，逐步恢复文本。

本系列主要讲掩码扩散——从一些空位开始，每轮预测缺失的词，再按采样规则确定其中一部分。

两者都可以用 Transformer，区别不只是“同一轮出几个词”，还包括训练时预测什么、生成时能看见什么，以及哪些计算可以留到下一轮复用。先看同一句话的生成过程，再往下拆这些区别。

扩散语言模型包括 Google 2025-05 展示的 Gemini Diffusion、Inception 的 Mercury 系列，以及 2026-06 开放权重的 DiffusionGemma。它们使用的噪声和解码方法并不完全相同，有些细节尚未公开。这个系列主要从公开的掩码扩散方法讲起。

掩码扩散的训练目标可以写成负对数似然（NLL）的上界；LLaDA 作者也报告，从零训练的 8B 模型在不少基准上与同量级 AR 模型相当。但并行预测并不等于一定更快：训练要覆盖不同生成顺序，解码要反复计算整块内容，同时确定太多 token 还可能产生不一致。已有厂商的同族对比也报告了质量差距。后面会分别看这些取舍，以及为什么速度优势往往出现在低并发、显存带宽受限的场景。

## 两种写法，同一句话

```text
自回归：每步落 1 个，只看左边
the → the tea → the tea is → the tea is cold

掩码扩散：每步预测所有空位，按采样规则揭开一部分
[M] [M] [M] [M] → the [M] is [M] → the tea is cold
```

| | 自回归 | 掩码扩散（整段） | 分块扩散（block diffusion） |
| --- | --- | --- | --- |
| 每步做什么 | 用左边预测下一个 | 双向读整段，预测所有空位 | 块与块之间从左到右，块内像整段扩散 |
| 每步确定几个 token | 常规解码为 1 | 由步数或自适应采样规则决定 | 在当前块内确定一个或多个 |
| KV cache | 已处理前缀可精确复用 | 上下文变化时，不能直接复用整段旧 KV | 已完成的块可精确复用；当前块另算 |
| 输出长度 | 写到 EOS 为止 | 事先给定 | 按块往后延 |
| 例子 | GPT、LLaMA | LLaDA 8B | BD3-LM、LLaDA2.0、DiffusionGemma |

后面的推导先以掩码扩散为例：把 token 换成 `[M]` 就是在加噪。它只是离散扩散的一种。DiffusionGemma 也按块生成，但 Google 介绍的噪声是随机替换 token，即均匀态（uniform-state）扩散，不能直接套用掩码扩散的所有公式。两者的区别见[噪声加在哪里](masked-diffusion.md)。

图像里常见的连续扩散通常从高斯噪声出发，见 [Qwen-Image 的 flow matching](../../03-multimodal-learning/image-generation.md)。文本也有在词向量上做连续扩散的工作；这里主要讨论的 LLaDA、Dream、LLaDA2.0 和 DiffusionGemma 则在离散 token 上建模。

还要区分“并行预测”和“确定输出”：模型可以同时预测很多位置，但采样器未必全都留下。即使一次确定了多个 token，也不等于计算更少，因为这一轮仍可能重新处理整块。

## 时间线

| 日期 | 事件 | 一句话 |
| --- | --- | --- |
| 2019-04 | [Mask-Predict](https://arxiv.org/abs/1904.09324) | 翻译里先并行预测全部目标词，再反复遮住最没把握的重填（EMNLP 2019） |
| 2021-07 | [D3PM](https://arxiv.org/abs/2107.03006) | 离散扩散框架，包括吸收态：变成 `[M]` 就不回来 |
| 2021-10 | [ARDM](https://arxiv.org/abs/2110.02037) | 任意顺序自回归和吸收态扩散是同一类模型的特例（ICLR 2022） |
| 2023-10 | [SEDD](https://arxiv.org/abs/2310.16834) | score entropy；困惑度比之前的扩散 LM 低 25–75%，胜过 GPT-2（ICML 2024） |
| 2024-06 | [MDLM](https://arxiv.org/abs/2406.07524)、[RADD](https://arxiv.org/abs/2406.03736) | 目标化简成掩码语言模型（MLM）损失的加权平均；模型不需要知道时间 |
| 2024-10 | [SMDM 的 scaling law](https://arxiv.org/abs/2410.18514) | 原实验中，同验证损失约需 AR 16 倍计算；不是所有扩散模型的固定倍率 |
| 2025-02 | [LLaDA](https://arxiv.org/abs/2502.09992) | 8B 从零训练 2.3T token；作者报告与同数据 AR 基线的扩展趋势相当 |
| 2025-03 | [Block Diffusion](https://arxiv.org/abs/2503.09573) | 块间自回归、块内扩散，已完成的块能精确缓存（ICLR 2025） |
| 2025-05-20 | [Gemini Diffusion](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-diffusion/) | 实验性演示，需排队申请；模型页自报平均 1479 tokens/s，不计固定开销 |
| 2025-05 | [Fast-dLLM](https://arxiv.org/abs/2505.22618) | 近似 KV cache 加按置信度并行落 token；自报 LLaDA 吞吐最多提高 27.6 倍（生成长度 1024） |
| 2025-06 | [Mercury 技术报告](https://arxiv.org/abs/2506.17298) | Mercury Coder Mini / Small 在 H100 上 1109 / 737 tokens/s，报告说数字来自 Artificial Analysis 的独立测评 |
| 2025-09 | [Fast-dLLM v2](https://arxiv.org/abs/2509.26328) | AR 权重转换为分块扩散，结合块级和子块缓存；[比较条件](ar-and-diffusion.md) |
| 2025-10；2026-03 修订 | [Planned Diffusion](https://arxiv.org/abs/2510.18087v2) | 同一模型用 AR 生成计划，再用扩散展开内容；计划开销也计入延迟 |
| 2025-10 | [性能刻画](https://arxiv.org/abs/2510.04146) | 第三方测 LLaDA 对 LLaMA-3：batch 一大，AR 吞吐更高 |
| 2025-12 | [LLaDA2.0](https://arxiv.org/abs/2512.15745) | 从 AR 模型转换的 16B 和 100B MoE；SGLang 2025-12-19 起支持 |
| 2026-02 | [LLaDA2.1](https://arxiv.org/abs/2602.08676)；[Mercury 2](https://www.inceptionlabs.ai/blog/introducing-mercury-2)（02-24） | LLaDA2.1 能改已落下的 token；Mercury 2 自报 Blackwell 上 1009 tokens/s |
| 2026-06-10 | [DiffusionGemma](https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/) | 开放权重 26B A4B MoE；自报最多快 4 倍，质量低于同规格 Gemma 4 |
| 2026-09-08 | [Mercury 2.5](https://www.inceptionlabs.ai/blog/introducing-mercury-2-5) | 自报 1107 tokens/s，没写 GPU 型号 |
| 2026-10-13 | [重新检查 16× 训练差距](https://arxiv.org/abs/2610.17590v1) | 预印本：评分方法和分块架构都会影响倍率，不能直接外推到前沿规模 |

## 哪些比较确定，哪些还在变

每格都对应一条带日期的公开记录。“接下来看什么信号”写的是什么样的结果会改变判断，不是预测它会发生。

| 比较确定 | 还在变 | 接下来看什么信号 |
| --- | --- | --- |
| 掩码扩散的目标是负对数似然的上界，并等于任意顺序自回归的损失：RADD（2024-06）、ARDM（2021-10） | 掩码和均匀态哪个更好：Sahoo 等（2026-02）在 1.7B 上发现均匀态困惑度更差、GSM8K 却更好；Google 的开发者指南说 DiffusionGemma 用均匀态，第三方的 Asaria 等（2026-06）则称它为掩码扩散 | 更大规模、同数据的跨家族对照；DiffusionGemma 的技术报告 |
| 同样计算下验证损失落后于 AR：SMDM 约 16 倍（$6\times10^{18}$–$10^{20}$ FLOPs） | 下游任务上的扩展：LLaDA 自报到 $10^{23}$ FLOPs 与同数据 AR 相当；数据重复多轮时扩散反超（Prabhudesai 等，2025-07） | 10B 以上、同数据同计算、公开 checkpoint 的对照 |
| 固定并行落 token 可能损失质量：Kim 等（2025-10）在生成长度 128、块大小 32 的设置下，每步 4 个时 GSM8K 73.6 → 61.7。另外，Feng 等对最坏情况 HMM 证明了线性步数下界，但有反向过程和学习误差假设，不能直接套到所有推理任务上（[适用范围](decoding-and-speed.md)） | 自适应落 token 和可修改：Fast-dLLM 的置信度阈值、LLaDA2.1 的编辑、DiffusionGemma 自报每次前向 15–20 个 token | 第三方测的“同质量下平均每步落几个” |
| 双向上下文在变化时，旧 KV 通常也会变；分块注意力隔离了已完成前缀，可以精确复用：[缓存例子](ar-and-diffusion.md) | 块多大合适，当前块近似缓存的误差有多大 | 主流引擎里的稳定接口和公开基准：SGLang（2025-12 起）、vLLM（DiffusionGemma 指南给了 `vllm serve` 命令） |
| 低并发下，扩散可以利用 AR decode 未充分使用的算力；Kim 等测到并发升高后优势缩小 | 不是所有实现都如此；Fast-dLLM v2 自报高 batch 仍有优势，[条件见这里](ar-and-diffusion.md) | 第三方在同硬件、同引擎、多个并发档位、同等质量下的对照 |
| 厂商自己的同族对比里质量更低：DiffusionGemma 模型卡上 AIME 2026 69.1% 对同规格 Gemma 4 的 88.3%；Gemini Diffusion 的 GPQA Diamond 40.4% 对 2.0 Flash-Lite 的 56.5% | 从 AR 权重转换能否补上差距：LLaDA2.0-flash 自报平均 73.18，原模型 Ling-flash-2.0 是 72.15 | 转换后的模型在推理和长上下文任务上与原模型的差距 |

“比较确定”的意思是：有一手结果，在原论文的设置里成立；不代表换任务、换规模也成立。

训练成本一行保留的是 SMDM 的历史结果。2026-10 的新预印本重新检查了评分方法与架构，见[AR 对比与近期进展](ar-and-diffusion.md)；目前还不宜把其中任何一个倍率当成普遍结论。

## 接下来看什么

下面是开放问题，以及什么样的结果会改变现在的判断；不是预测。

- **同条件的速度对照。** 截至 2026 年 10 月，公开的加速数字多是 batch 1 或厂商自报。要看到优势在哪个并发区间成立，需要第三方在同一硬件、同一引擎、同等质量下扫一遍并发。
- **前沿规模的质量差距。** DiffusionGemma 与 Gemma 4 的差距能否靠转换训练和 RL 缩小：[d1](https://arxiv.org/abs/2504.12216)（2025-04）给掩码扩散设计了 diffu-GRPO，LLaDA2.1（2026-02）加入了大规模 RL。重点看推理和长上下文。
- **服务栈。** SGLang 的 dLLM 调度默认用 FDFO：一个请求的当前块一解完就离开批次，不用等同批里慢的请求（[SGLang 文档](https://lmsysorg.mintlify.app/docs/supported-models/diffusion_language_models)，截至 2026-10）。它和 vLLM 对 DiffusionGemma 的支持在高并发下的吞吐和尾延迟，还没有公开实测。
- **数据受限时的优势。** Prabhudesai 等（2025-07）发现数据重复多轮时扩散更划算；高质量数据成为瓶颈时，这一点能否在更大的模型上成立。
- **扩散当草稿模型。** DFlash 一类“扩散写草稿、AR 验证”的做法不改变输出分布，它会不会比原生扩散 LM 更早成为部署的主流。

## 这一组怎么读

先读这些会更顺：[BERT](../../00-foundations/core/bert.md)（遮盖与双向 encoder）、[语言模型目标](../../00-foundations/deep-dives/language-model-objective.md)（自回归的似然和训练、生成的错位）、[解码策略](../../00-foundations/core/decoding.md)、[KV cache](../../00-foundations/deep-dives/kv-cache-and-inference.md)和 [roofline 与 decode 的 batch](../../12-infra/performance.md#decode-batching)。相邻的几篇：[DFlash](../../00-foundations/deep-dives/dflash.md)把分块扩散用作 AR 模型的草稿模型；[MTP](../../00-foundations/deep-dives/multi-token-prediction.md)讲一次预测多个 token 的训练目标；[测试时计算](../test-time-compute/README.md)讲多花推理计算换质量，ReMDM 一类“步数越多越好”的采样器也属于这个问题。

1. [掩码扩散在训练什么](masked-diffusion.md)：前向遮盖、$1/t$ 加权，和 BERT 的 MLM 差在哪里；用两个 token 手算“目标等于所有顺序的平均”；LLaDA 的规模和扩展说法该怎么读
2. [MDLM：从训练到生成](mdlm-walkthrough.md)：用两句话手算训练损失和揭开概率；为什么预测正确的边缘分布，仍可能写出相互矛盾的词
3. [解码步数、缓存与真实速度](decoding-and-speed.md)：逐步填空和重遮，一步落几个，KV cache 为什么失效、分块扩散怎么补；用教学模型在同一张 H100 上手算低并发和高并发的账；厂商数字和第三方测量；和 DFlash 的关系
4. [自回归与扩散：怎么比较，又怎么结合](ar-and-diffusion.md)：条件对照、分块缓存的小例子、AR 规划与扩散生成，以及近期论文的结果和限制
5. [缓存与上下文复用](context-reuse.md)：固定 KV 不代表固定 attention 输出；用小算例比较 E2D2、FlashBlock、MAGE 省掉的计算与引入的误差
6. [复习题](review.md)：面试题和自检
