# 内容覆盖：有哪些，还缺什么？

**中文** · [English](coverage.en.md)

这页记录已经写到哪里、还有哪些地方没核对完。整理目录时靠它防止漏项；读笔记时不必照着它逐项打卡。

**截至 2026-10-16，参考清单登记的主题都有阅读入口，但还不能说全部审完。** 主线 82 项中，81 项已有正文，1 项是面试复习导读；附录有 65 个专题和算法、PyTorch 两个汇总入口。有些条目指向同一篇笔记，所以这些数字不能相加当作文章总数。参考清单之外新加的板块和它们的审阅状态，见[这轮先整理什么](#current-update)。

附录 1 登记的参考正文已经读完，逐项记录见下方自动汇总。接下来要检查的是本站讲得是否准确、例子是否清楚，而不是继续凑阅读数量。读完正文，也不等于原图、外链和参考实现都验证过了。

想直接学知识，可以回到[学习路线](README.md)。这一页是查漏清单，不是要求你按顺序学完的题库。

已发布的内容会继续修订，新补充的内容先在本地检查。下面会保留尚未讲完或核对完的部分；构建、链接和算例通过测试，也不等于每篇文章都已审过。

## 这轮先整理什么 {#current-update}

2026-10-15 加了第四部分「前沿观察」、两个新板块和几篇单独的笔记。标“待审核”的是草稿：开头有提示框，侧栏标「（待审核）」，站主读过后才去掉。引用草稿里的事实前，请回到文中链接的一手资料核对。

| 内容 | 篇数 | 审阅状态 |
| --- | ---: | --- |
| [前沿观察](../11-frontier/README.md) · [世界模型](../11-frontier/world-models/README.md) | 7 | 已审阅 |
| 前沿观察 · [递归自我改进](../11-frontier/self-improvement/README.md) | 5 | 已审阅 |
| 前沿观察 · [测试时计算](../11-frontier/test-time-compute/README.md) | 5 | 待审核草稿 |
| 前沿观察 · [用 RL 训练 agent](../11-frontier/agentic-rl/README.md) | 4 | 待审核草稿 |
| 前沿观察 · [机制可解释性](../11-frontier/interpretability/README.md) | 5 | 待审核草稿 |
| [AI 基础设施](../12-infra/README.md)：晶圆与封装、CPU、GPU、NVIDIA 各代、CUDA、性能、框架、Kubernetes | 10 | 待审核草稿 |
| [感知、机器人与自动驾驶](../13-embodied/README.md)：感知与几何、自动驾驶、机器人学习 | 14 | 待审核草稿 |
| 强化学习：[学什么样的模型](../05-post-training/deep-rl/model-types.md)、[规划与学习](../05-post-training/deep-rl/planning-and-learning.md)、[离线评估的估计量](../05-post-training/deep-rl/ope-estimators.md) | 3 | 已审阅 |
| 其他章节：[一个还是多个 agent](../10-agents/single-vs-multi-agent.md)、[LLM 辅助决策](../06-systems/decision-support.md)、[评估接到线上](../07-evaluation/eval-observability.md) | 3 | 待审核草稿 |

篇数按中文笔记计（含各组导读），每篇另有英文版。这些内容不在参考清单里，下面的对照表也不统计它们。这张表只记审阅状态，不代表全站已经核对完；每批改动的检查范围和结果记在仓库的 `CONTENT_RELEASE_CHECKLIST.md`。

上一轮（2026-10-10）整理了中英文入口（首次访问默认英文，顶部一键切换同一篇）、仓库 [README](../README.md) 与[中文版](../README.zh.md)，以及 BERT、RL / post-training、Python 和检索的例子与图解。

**还要做什么：**继续逐章检查本站的解释、例子、图和双语表达；对实际用来支撑结论的图片公式和外链，补上核验记录。没有采用的参考材料会注明范围，不把“看过”写成“验证过”。技术结论以原论文或官方实现为准；工程项目里的教学算例不等于真实 GPU 训练结果。

## 怎样读这张表

例如 [LatentMoE](../00-foundations/moe/latent-moe.md) 已有宽度、路由和成本的讲解，也有能自己算的例子；[MLA / RoPE](../00-foundations/deep-dives/latent-and-sparse-attention.md#rope-absorption) 用旋转顺序说明计算为什么不能随意重排。两篇都可以标为“有正文”，但这不表示我们复现了模型训练，或验证了论文里的所有实验。

- **有正文**：站内有专门的文章或实质讲解，能点进去读；不代表与参考资料的所有细节完全等价。
- **汇总入口**：一个入口串起多篇文章，例如算法和 PyTorch 路线；不是尚未填上的空章。
- **待补**：主题保留在清单里，但不造一个空页面当作已经完成。

参考范围来自[大模型学习地图](https://tcn6r3nlptym.feishu.cn/wiki/N2Yjwoez0iUA0AkGQBRcSps1nPf)。主线 9 章的 71 个页面已通过浏览器读完正文，包括多模态全部 16 页；第 7 章的 7.1–7.7 在同一页，按 1 篇计算。附录 3 的 7 篇 ML 基础与 20 篇 PyTorch 正文也已读到末尾。这里统计的是正文阅读，不是全部代码、图片和外链的验证。

基础附录对应新增 [泛化与训练诊断](../00-foundations/deep-dives/generalization.md)、[激活与初始化](../00-foundations/deep-dives/activation-and-initialization.md)、[优化器](../00-foundations/deep-dives/optimizers.md)。用小例子区分训练失败与泛化差、ReLU 的二阶矩与方差、Adam 的偏差修正与 AdamW 的解耦衰减；原有 RNN / GRU、归一化和基础题的相关解释也同步修正。例子为本站原创，不复制参考代码。

新增 [PyTorch 实践路线](../00-foundations/pytorch/README.md)，按张量存储、运算维度、交叉熵与 loss、自动求导和训练循环分成 5 章，中英文都有完整代码。重点不是罗列 API，而是解释那些“不报错却算错了”的情况：跨 batch 广播、错误的 loss 分母、没保留的中间梯度，以及只有模型权重却没有完整训练状态的恢复。数值测试使用 PyTorch 2.8.0 / CPU，软件版本与教学边界单独注明。

多模态新增双语 [LLaVA / DeepSeek-VL](../03-multimodal-learning/vlm-designs.md)、[Qwen-VL](../03-multimodal-learning/qwen-vl.md)、[Omni](../03-multimodal-learning/omni-streaming.md)、[OCR 压缩](../03-multimodal-learning/ocr-compression.md)、[图像生成](../03-multimodal-learning/image-generation.md)与[微调检查](../03-multimodal-learning/vlm-finetuning.md)。保留之前的 [ViT](../03-multimodal-learning/vit.md)、[CLIP](../03-multimodal-learning/clip.md)和 [BLIP / Q-Former](../03-multimodal-learning/blip-and-q-former.md)。2026 年的 OCR 2、Qwen3.5 / 3.8-Omni 与 Qwen-Image 后续版本单独核对，不把旧报告当最新版本。算例不是模型训练实测，多模态 GPU 微调尚未执行。

评估新增 [BLEU、ROUGE 与编辑距离](../07-evaluation/text-metrics.md)，包括短句算分、动态规划代码、PPL 比较条件与反例；[Benchmark 与长上下文测试](../07-evaluation/benchmark-protocols.md)补充固定协议、证据位置、无答案控制、配对比较与污染检查。代码只生成测试夹具，没有虚构模型评测结果。

后训练新增 [DAPO](../05-post-training/dapo.md)、[GSPO / ASPO](../05-post-training/policy-ratios.md)与 [SAO / 异步训练](../05-post-training/async-policy-learning.md)。补充采样成本、loss reduction、stop-gradient 与轨迹过期的算例；修正了旧文中“clipping 硬限制概率”“Critic 只用来做 baseline”等不准确表述。小例子有测试，但并未复现这些论文的完整训练结果。

上一批新增的双语 [RAG：证据与评估](../04-search/rag-evidence.md)、[向量索引：IVF 与 PQ](../04-search/vector-indexes.md)，以及扩充的 [Agent 执行过程](../10-agents/patterns.md) 保留。新增 [Prompt](../10-agents/prompting.md) 的任务约定与对照实验设计，以及 [Deep Research](../10-agents/deep-research.md) 的证据队列、引用检查与停止逻辑。可运行的本地例子不代表已完成模型对比实验。

新增[工具、MCP 与 Skills](../10-agents/tools-and-skills.md)和重写的[模型选择与路由](../10-agents/model-choice.md)，用笔记检查和 4 条假设请求解释调用、权限、预算与取舍；没有真实模型或服务性能测量。

此前对 [DeepSeek](../00-foundations/model-families/deepseek.md)、[Qwen](../00-foundations/model-families/qwen.md)和[多卡训练](../06-systems/distributed-training.md)的补充保留；这批继续加入路由权重、奖励边界、Ring 通信、权重版本和恢复的算例。BLIP 的筛选审计与 InstructBLIP 消融方案也已补充，均不冒充完整训练实验。

此前新增的 [GPT](../00-foundations/model-families/gpt.md) 与扩充的 [Llama](../00-foundations/model-families/llama.md) 保留。新增 [Qwen1–2.5](../00-foundations/model-families/qwen-early.md) 与 [Gated DeltaNet 推导](../00-foundations/deep-dives/gated-deltanet.md)，Qwen 主文更新到 3.8 的公开型号，区分文本权重、视觉模型与托管服务。[DeepSeek-V4](../00-foundations/model-families/deepseek-v4.md) 已补混合注意力、因果边界、mHC 与后训练算例；gpt-oss 补充公开训练范围、effort / 工具对照及配对成本记账，不填补未披露的训练配方。

DCA 的 chunk 边界位置、DSA 的 indexer 训练与错选例子已补入长上下文和稀疏注意力篇；新增 [Gated Attention](../00-foundations/deep-dives/gated-attention.md)、[Engram](../00-foundations/deep-dives/engram.md)与 [Block AttnRes](../00-foundations/deep-dives/attention-residuals.md) 双语讲解。局部算例覆盖 gate 初始化、hash 冲突、深度 softmax 与资源成本，未声称复现完整训练。之前的 MoE、MLA、NSA、YaRN 和推测解码算例保留。

参考代码没有执行，图片内公式、嵌入 PDF 和外链视频也未全部核查。附录 2 的算法分类页已读完，字符串匹配、单调队列、BST、背包与序列 / 状态机 DP 已补 6 对笔记及局部测试。讲解、例子与代码自己写，技术结论回到原论文或官方实现核对。全站时效性审查和逐篇技术审稿还没完成，不把构建成功当成内容全部完成。

推理部分补充了[请求完整流程](../06-systems/llm-serving.md)、[完整显存预算](../00-foundations/deep-dives/kv-cache-and-inference.md#inference-budget)和[缓存生命周期](../00-foundations/deep-dives/attention-kernels.md#cache-lifecycle)。新读完参考的请求流程与推理显存两篇，公开实现固定到具体版本。算例可本地核对，但不把容量估算写成 GPU 性能实测。

<span id="appendix-questions"></span>

## 附录问题：逐条查漏 {#appendix-question-map}

目前登记 **67 个入口**：65 个专题，以及算法和 PyTorch 两个汇总入口。它们不是 67 篇独立文章，也不是整个参考库的总篇数。下方分别记录本站讲解和参考阅读状态，图片公式与外链不会自动算作已核验。

这轮最值得补的，不只是新模型名字：

| 核对后发现什么 | 具体例子 | 接下来怎么处理 |
| --- | --- | --- |
| 有完整讲解，不必另开一篇 | Online softmax、手写 MHA、RoPE、BF16 / FP16、Norm 统计轴、熵与 CE / KL | 连到相关小节，核对附加问题，不重复建章 |
| 原来的讲解不够直接 | 拒绝采样、熵坍塌 | 已补推导、小例子或诊断步骤，继续检查是否好理解 |
| 原先只提名字，本轮补成专题 | [BM25 对比 TF-IDF](../04-search/tfidf-and-bm25.md)、[质数与筛法](../interview/algorithms/primes-and-sieves.md) | 两篇参考正文均已读；本站另写算分、证明与边界例子 |
| 已确认具体型号并补入专题 | [Kimi K3 / NoPE](../00-foundations/deep-dives/nope-and-order.md) | 从换序算例到矩阵递推；区分不用 RoPE 与不需要长上下文训练 |

例如，已有的蒸馏篇确实算过同一组分布的熵、交叉熵和 KL，这项不该误报成缺失；BM25 原先则只出现在混合检索里，本轮才补上专门算分的文章。这才是逐项检查要区分的事情。

2026-10-09 补入 [Muon](../00-foundations/deep-dives/muon.md)、[DFlash / MTP 对照](../00-foundations/deep-dives/dflash.md)、BM25 / TF-IDF、质数筛法、[Qwen3 Embedding 与 BGE-M3](../04-search/embedding-models.md)、[InfoNCE 与 CE](../04-search/dual-encoder.md#infonce-and-ce)，以及 NoPE 与顺序信息。

另补 [BERT 从输入到微调](../00-foundations/core/bert.md)：通过一条短序列分清输入遮盖、attention mask 和 loss mask，再接 MLM / NSP 与任务输出层。

当前 **65 项有专门讲解，2 项为汇总入口**。清单里没有完全缺少相关专题的项目，不代表已经验收完成。最近补充了 [Muon 的分片反例](../00-foundations/deep-dives/muon.md)、[DFlash 的特征与 shape](../00-foundations/deep-dives/dflash.md#feature-shapes)和 [clipping 变体](../05-post-training/clipping-variants.md)。之前的重算、训练显存、ZeRO 与检索专题仍保留；[Python：变量、拷贝与函数调用](../interview/python-objects.md)仍在代码复习路线里。不会因为读完参考，就采用其中过时或过于绝对的结论。

工程部分新增[沿一次请求排查错误](../practice/post-training/experiments-and-release.md#trace-a-failure)，不把所有错误都归为“需要微调”。此前的[长任务中怎样选 GRPO / PPO](../05-post-training/deep-rl/llm-bridge.md#long-horizon-choice)、[学生采样与蒸馏](../05-post-training/distillation.md#opd-choices)和[rollout 到训练数据](../05-post-training/post-training-infrastructure.md#rollout-record)也已扩充。先看小例子，再按需要展开梯度和代码；CPU 检查不代表完整训练或 GPU 性能复现。

此前的[两张卡算同一层](../06-systems/tensor-parallel.md)与[同样 8 张卡的部署选择](../06-systems/distributed-training.md#eight-gpus)也保留了，从输入梯度、通信到资源开销都可以继续追下去。

训练工程补了 [checkpoint 与恢复](../practice/post-training/checkpoint-and-resume.md)、[CPU 到 GPU 的数据路径](../00-foundations/pytorch/training-loop.md)、[训练显存记账](../00-foundations/deep-dives/precision-and-memory.md)。先用动量、3 批数据和 10 亿参数的小例子讲清机制，再接代码与工程限制。上述参考正文和 ZeRO stages 均已读，数据路径另补了 kernel、SM 与存储的区别。本站的[重算例子](../00-foundations/deep-dives/precision-and-memory.md#recompute-example)和[通信账](../06-systems/distributed-training.md#zero-communication)可直接接着看。CPU 结果不代表 GPU 显存或通信性能。

后训练补了三个容易在训练日志里遇到的问题：[loss 为 0 但梯度不为 0](../05-post-training/after-ppo.md#zero-loss-gradient)、[ratio 越界不一定没有梯度](../05-post-training/rlhf/ppo-clipping.md#clipped-token-gradients)，以及[一组回答何时生成、复用和刷新](../05-post-training/rlhf/on-off-policy.md#grpo-data-lifecycle)。小算例和 CPU 代码对应，不把日志数值、梯度和真实训练效果混在一起。

本站的 [MoE 路由](../00-foundations/moe/router.md#dispatch-example)现在有完整的分发、计算、加权回填和梯度检查；[负载均衡](../00-foundations/moe/load-balancing.md#sequence-balance)用两条序列说明为什么全 batch 均衡不等于逐序列均衡。离散计数不能直接反传、归一化分母和 padding 等细节都单独核对。教学算例不算模型实测；参考中不准确的说法不会因为“已读”就直接采用。

### 这些问题放在哪里，读者才找得到？

不按参考问答的顺序堆新页面，也不把所有内容塞进“进阶”。保留本站现有的目录结构，按知识依赖安排：

| 想怎么读 | 从哪里开始 | 哪些可以之后再看 |
| --- | --- | --- |
| 先入门，弄懂模型怎么学 | Attention、Norm、交叉熵和一次参数更新 | Muon、分布式通信、推测解码 |
| 只关心后训练 | SFT → PPO / GRPO → 采样、KL、loss 排错 | 各代模型和 serving 引擎源码 |
| 只关心检索或部署 | 检索从双塔、混合检索走；部署从生成、KV cache 走 | 另一条路线不必全部读完 |
| 想自己实现 | 手写模块 → shape / 梯度检查 → 完整小实验 | 新算法名不如先把一个模块写对 |

下面按 4 个主题组查漏，不是新的难度等级。每项会分别说明本站讲了什么、参考读到哪里，以及还没验证什么。算法和 PyTorch 是汇总入口，不计入专题正文的阅读数量。

<!-- widget:appendix-coverage -->

## 逐项对照

<!-- widget:reference-coverage -->

## 本站额外保留的内容

参考资料不是本站内容的上限。RNN / BPTT、完整 Deep RL 路线、Looped Transformer、评估校准、工具与记忆、系统设计、公开推荐系统拆解、Python 与算法练习都会保留。

这次调整只改归类与阅读入口，不搬文章的公开地址，也不把旧例子、代码和图解缩成摘要。此前明确下架的 Quant 和“聊聊”不在这轮重新上线。

## 接下来怎样核对

按章节继续在浏览器里读，分别记录正文、图片公式和代码读到了哪里，再对照本站的定义、推导、例子与实现。发现缺口就补进已有章节，确实是独立问题再新开一篇；不会为了对齐别人目录而复制一套重复文章。

如果你发现漏项，可以在 issue 里贴主题、公开来源和你希望讲清楚的问题。维护记录见仓库中的 `CONTENT_COVERAGE.md`；主线清单来自 `site/reference-coverage.toml`，截图与浏览器目录的附录对照来自 `site/appendix-coverage.toml`。
