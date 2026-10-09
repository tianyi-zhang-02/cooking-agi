# 内容覆盖：有哪些，还缺什么？

**中文** · [English](coverage.en.md)

这份清单用来防止两件事：整理目录时把旧内容落下，以及只加了一个术语就以为已经讲完。它不是完成率排行榜。

**截至 2026-10-09，还不能说内容已经全部补齐。** 主线清单有 82 项，附录已辨认的入口从截图中的 36 个扩到 67 个，两份清单有重叠，不能相加当作篇数。有的内容已经讲过，只是入口不好找；有的只出现了名字；也有确实没写的。下面把这几种情况分开。

想直接学知识，可以回到[学习路线](README.md)。这一页是查漏清单，不是要求你按顺序学完的题库。

本轮先整理、提交已有内容，后续继续补充。下面的“部分讲解”和未完成核对会继续保留；通过构建、链接和算例检查，不等于每篇都已完成技术审稿。

## 怎样读这张表

- **有正文**：站内有专门的文章或实质讲解，能点进去读；不代表与参考资料的所有细节完全等价。
- **部分讲解**：已有相关内容，但独立推导、例子或完整范围仍需补充。
- **待补**：主题保留在清单里，但不造一个空页面当作已经完成。

参考范围来自[大模型学习地图](https://tcn6r3nlptym.feishu.cn/wiki/N2Yjwoez0iUA0AkGQBRcSps1nPf)。主线 9 章的 71 个页面已通过浏览器读完正文，包括多模态全部 16 页；第 7 章的 7.1–7.7 在同一页，按 1 篇计算。附录 3 的 7 篇 ML 基础与 20 篇 PyTorch 正文也已读到末尾。这里统计的是正文阅读，不是全部代码、图片和外链的验证。

基础附录对应新增 [泛化与训练诊断](../00-foundations/deep-dives/generalization.md)、[激活与初始化](../00-foundations/deep-dives/activation-and-initialization.md)、[优化器](../00-foundations/deep-dives/optimizers.md)。用小例子区分训练失败与泛化差、ReLU 的二阶矩与方差、Adam 的偏差修正与 AdamW 的解耦衰减；原有 RNN / GRU、归一化和基础题的相关解释也同步修正。例子为本站原创，不复制参考代码。

新增 [PyTorch 实践路线](../00-foundations/pytorch/README.md)，按张量存储、运算维度、自动求导和训练循环分成 4 章，中英文都有完整代码。重点不是罗列 API，而是解释那些“不报错却算错了”的情况：跨 batch 广播、错误的 loss 分母、没保留的中间梯度，以及只有模型权重却没有完整训练状态的恢复。数值测试使用 PyTorch 2.8.0 / CPU，软件版本与教学边界单独注明。

多模态新增双语 [LLaVA / DeepSeek-VL](../03-multimodal-learning/vlm-designs.md)、[Qwen-VL](../03-multimodal-learning/qwen-vl.md)、[Omni](../03-multimodal-learning/omni-streaming.md)、[OCR 压缩](../03-multimodal-learning/ocr-compression.md)、[图像生成](../03-multimodal-learning/image-generation.md)与[微调检查](../03-multimodal-learning/vlm-finetuning.md)。保留之前的 [ViT](../03-multimodal-learning/vit.md)、[CLIP](../03-multimodal-learning/clip.md)和 [BLIP / Q-Former](../03-multimodal-learning/blip-and-q-former.md)。2026 年的 OCR 2、Qwen3.5 / 3.8-Omni 与 Qwen-Image 后续版本单独核对，不把旧报告当最新版本。算例不是模型训练实测，多模态 GPU 微调尚未执行。

评估新增 [BLEU、ROUGE 与编辑距离](../07-evaluation/text-metrics.md)，包括短句算分、动态规划代码、PPL 比较条件与反例；[Benchmark 与长上下文测试](../07-evaluation/benchmark-protocols.md)补充固定协议、证据位置、无答案控制、配对比较与污染检查。代码只生成测试夹具，没有虚构模型评测结果。

后训练新增 [DAPO](../05-post-training/dapo.md)、[GSPO / ASPO](../05-post-training/policy-ratios.md)与 [SAO / 异步训练](../05-post-training/async-policy-learning.md)。补充采样成本、loss reduction、stop-gradient 与轨迹过期的算例；修正了旧文中“clipping 硬限制概率”“Critic 只用来做 baseline”等不准确表述。小例子有测试，但并未复现这些论文的完整训练结果。

上一批新增的双语 [RAG：证据与评估](../04-search/rag-evidence.md)、[向量索引：IVF 与 PQ](../04-search/vector-indexes.md)，以及扩充的 [Agent 执行过程](../10-agents/patterns.md) 保留。新增 [Prompt](../10-agents/prompting.md) 的任务约定与对照实验设计，以及 [Deep Research](../10-agents/deep-research.md) 的证据队列、引用检查与停止逻辑。可运行的本地例子不代表已完成模型对比实验。

此前对 [DeepSeek](../00-foundations/model-families/deepseek.md)、[Qwen](../00-foundations/model-families/qwen.md)和[多卡训练](../06-systems/distributed-training.md)的补充保留；这批继续加入路由权重、奖励边界、Ring 通信、权重版本和恢复的算例。BLIP 的筛选审计与 InstructBLIP 消融方案也已补充，均不冒充完整训练实验。

此前新增的 [GPT](../00-foundations/model-families/gpt.md) 与扩充的 [Llama](../00-foundations/model-families/llama.md) 保留。新增 [Qwen1–2.5](../00-foundations/model-families/qwen-early.md) 与 [Gated DeltaNet 推导](../00-foundations/deep-dives/gated-deltanet.md)，Qwen 主文更新到 3.8 的公开型号，区分文本权重、视觉模型与托管服务。[DeepSeek-V4](../00-foundations/model-families/deepseek-v4.md) 已补混合注意力、因果边界、mHC 与后训练算例；gpt-oss 补充公开训练范围、effort / 工具对照及配对成本记账，不填补未披露的训练配方。

DCA 的 chunk 边界位置、DSA 的 indexer 训练与错选例子已补入长上下文和稀疏注意力篇；新增 [Gated Attention](../00-foundations/deep-dives/gated-attention.md)、[Engram](../00-foundations/deep-dives/engram.md)与 [Block AttnRes](../00-foundations/deep-dives/attention-residuals.md) 双语讲解。局部算例覆盖 gate 初始化、hash 冲突、深度 softmax 与资源成本，未声称复现完整训练。之前的 MoE、MLA、NSA、YaRN 和推测解码算例保留。

参考代码没有执行，图片内公式、嵌入 PDF 和外链视频也未全部核查。附录 2 的算法分类页已读完，字符串匹配、单调队列、BST、背包与序列 / 状态机 DP 已补 6 对笔记及局部测试。附录 1 已读 8 篇独立问答，包括 KV cache 估算与 K3 / NoPE；其余不能算完成。讲解、例子与代码自己写，技术结论回到原论文或官方实现核对。全站时效性审查和逐篇技术审稿还没完成，不把构建成功当成内容全部完成。

<span id="appendix-questions"></span>

## 附录问题：逐条查漏 {#appendix-question-map}

重新打开参考目录后，又辨认出 **31 个入口**，目前共登记 **67 个**，包括算法和 PyTorch 两个附录入口。这不是整个参考库的总篇数，也不代表 67 篇都读过。正文阅读与本站覆盖分开记，图片里的公式和外链也不自动算已核验。

这轮最值得补的，不只是新模型名字：

| 核对后发现什么 | 具体例子 | 接下来怎么处理 |
| --- | --- | --- |
| 有完整讲解，不必另开一篇 | Online softmax、手写 MHA、RoPE、BF16 / FP16、Norm 统计轴、熵与 CE / KL | 把入口连好，继续核对参考的附加问题 |
| 有内容，但还不够直接 | 序列级 MoE 均衡、GRPO 初始零 loss、被 clip 的 token 梯度、拒绝采样、熵坍塌 | 补缺少的推导、小例子或诊断步骤，不重复定义 |
| 原先只提名字，本轮补成专题 | [BM25 对比 TF-IDF](../04-search/tfidf-and-bm25.md)、[质数与筛法](../interview/algorithms/primes-and-sieves.md) | 算分、证明和边界有了，继续核对参考正文是否还有遗漏 |
| 已确认具体型号并补入专题 | [Kimi K3 / NoPE](../00-foundations/deep-dives/nope-and-order.md) | 从换序算例到矩阵递推；区分不用 RoPE 与不需要长上下文训练 |

例如，已有的蒸馏篇确实算过同一组分布的熵、交叉熵和 KL，这项不该误报成缺失；BM25 原先则只出现在混合检索里，本轮才补上专门算分的文章。这才是逐项检查要区分的事情。

2026-10-09 补入 [Muon](../00-foundations/deep-dives/muon.md)、[DFlash / MTP 对照](../00-foundations/deep-dives/dflash.md)、BM25 / TF-IDF、质数筛法、[Qwen3 Embedding 与 BGE-M3](../04-search/embedding-models.md)、[InfoNCE 与 CE](../04-search/dual-encoder.md#infonce-and-ce)，以及 NoPE 与顺序信息。

当前 **24 项有专门讲解，43 项仍需补充或对照**。清单里暂时没有完全缺少专题的项目，不代表已经验收完成。参考附录 1 的正文累计读完 8 篇，本次新增 K3；其余条目按实际阅读情况保留状态。教学算例不算模型实测，已经写过的文章也仍要核对来源、细节和呈现。

### 这些问题放在哪里，读者才找得到？

不按参考问答的顺序堆新页面，也不把所有内容塞进“进阶”。保留本站现有的 4 类目录，按知识依赖安排：

| 想怎么读 | 从哪里开始 | 哪些可以之后再看 |
| --- | --- | --- |
| 先入门，弄懂模型怎么学 | Attention、Norm、交叉熵和一次参数更新 | Muon、分布式通信、推测解码 |
| 只关心后训练 | SFT → PPO / GRPO → 采样、KL、loss 排错 | 各代模型和 serving 引擎源码 |
| 只关心检索或部署 | 检索从双塔、混合检索走；部署从生成、KV cache 走 | 另一条路线不必全部读完 |
| 想自己实现 | 手写模块 → shape / 梯度检查 → 完整小实验 | 新算法名不如先把一个模块写对 |

下面仍按 4 个主题组查漏；这是维护用的分类，不是新的难度等级。“需补充或对照”可能表示本站缺了细节，也可能表示参考正文还没读完，具体原因会写在每项下面。

<!-- widget:appendix-coverage -->

## 逐项对照

<!-- widget:reference-coverage -->

## 本站额外保留的内容

参考资料不是本站内容的上限。RNN / BPTT、完整 Deep RL 路线、Looped Transformer、评估校准、工具与记忆、系统设计、公开推荐系统拆解、Python 与算法练习都会保留。

这次调整只改归类与阅读入口，不搬文章的公开地址，也不把旧例子、代码和图解缩成摘要。此前明确下架的 Quant 和“聊聊”不在这轮重新上线。

## 接下来怎样核对

按章节继续在浏览器里读，分别记录正文、图片公式和代码读到了哪里，再对照本站的定义、推导、例子与实现。发现缺口就补进已有章节，确实是独立问题再新开一篇；不会为了对齐别人目录而复制一套重复文章。

如果你发现漏项，可以在 issue 里贴主题、公开来源和你希望讲清楚的问题。维护记录见仓库中的 `CONTENT_COVERAGE.md`；主线清单来自 `site/reference-coverage.toml`，截图与浏览器目录的附录对照来自 `site/appendix-coverage.toml`。
