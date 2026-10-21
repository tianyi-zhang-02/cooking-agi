# 内容覆盖：有哪些，还缺什么？

**中文** · [English](coverage.en.md)

这页记录已经写到哪里、还有哪些地方没核对完。整理目录时靠它防止漏项；读笔记时不必照着它逐项打卡。

**截至 2026-10-16，参考清单登记的主题都有阅读入口，但还不能说全部审完。** 主线 82 项中，81 项已有正文，1 项是面试复习导读；附录有 65 个专题和算法、PyTorch 两个汇总入口。有些条目指向同一篇笔记，所以这些数字不能相加当作文章总数。参考清单之外新加的板块和它们的审阅状态，见[这轮先整理什么](#current-update)。

附录 1 登记的参考正文已经读完，逐项记录见下方自动汇总。接下来要检查的是本站讲得是否准确、例子是否清楚，而不是继续凑阅读数量。读完正文，也不等于原图、外链和参考实现都验证过了。

想直接学知识，可以回到[学习路线](README.md)。

已发布的内容会继续修订，新补充的内容先在本地检查。下面会保留尚未讲完或核对完的部分；构建、链接和算例通过测试，也不等于每篇文章都已审过。

## 这轮先整理什么 {#current-update}

2026-10-15 加了第四部分“前沿观察”、两个新板块和几篇单独的笔记。标“待审核”的是草稿：开头有提示框，侧栏标「（待审核）」，站主读过后才去掉。引用草稿里的事实前，请回到文中链接的一手资料核对。

| 内容 | 篇数 | 审阅状态 |
| --- | ---: | --- |
| [前沿观察](../11-frontier/README.md) · [世界模型](../11-frontier/world-models/README.md) | 7 | 已审阅 |
| 前沿观察 · [递归自我改进（RSI）](../11-frontier/self-improvement/README.md) | 5 | 已审阅 |
| 前沿观察 · [测试时计算（test-time compute）](../11-frontier/test-time-compute/README.md) | 5 | 待审核草稿 |
| 前沿观察 · [用 RL 训练 agent](../11-frontier/agentic-rl/README.md) | 4 | 待审核草稿 |
| 前沿观察 · [机制可解释性](../11-frontier/interpretability/README.md) | 5 | 待审核草稿 |
| 前沿观察 · [扩散语言模型（dLLM）](../11-frontier/diffusion-lm/README.md) | 4 | 待审核草稿（2026-10-16） |
| 大模型基础 · [Scaling law 与超参数](../00-foundations/scaling/README.md) | 5 | 待审核草稿（2026-10-16） |
| AI 基础设施 · [量化](../12-infra/quantization/README.md) | 6 | 待审核草稿（2026-10-16） |
| [AI 基础设施（AI infra）](../12-infra/README.md)：晶圆与封装、CPU、GPU、NVIDIA 各代、CUDA、性能、框架、Kubernetes | 10 | 待审核草稿 |
| [感知、机器人与自动驾驶](../13-embodied/README.md)：感知与几何、自动驾驶、机器人学习 | 14 | 待审核草稿 |
| 强化学习：[学什么样的模型](../05-post-training/deep-rl/model-types.md)、[规划与学习](../05-post-training/deep-rl/planning-and-learning.md)、[离线策略评估（OPE）的估计量](../05-post-training/deep-rl/ope-estimators.md) | 3 | 已审阅 |
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

当前 **65 项有专门讲解，2 项为汇总入口**。清单里没有完全缺少相关专题的项目，不代表已经验收完成；各项具体补了什么，见[此前补充了什么](#earlier-updates)。不会因为读完参考，就采用其中过时或过于绝对的结论。

### 这些问题放在哪里，读者才找得到？

不按参考问答的顺序堆新页面，也不把所有内容塞进“进阶”。保留本站现有的目录结构，按知识依赖安排：

| 想怎么读 | 从哪里开始 | 哪些可以之后再看 |
| --- | --- | --- |
| 先入门，弄懂模型怎么学 | Attention、Norm、交叉熵和一次参数更新 | Muon、分布式通信、推测解码（speculative decoding） |
| 只关心后训练 | SFT → PPO / GRPO → 采样、KL、loss 排错 | 各代模型和 serving 引擎源码 |
| 只关心检索或部署 | 检索从双塔、混合检索走；部署从生成、KV cache 走 | 另一条路线不必全部读完 |
| 想自己实现 | 手写模块 → shape / 梯度检查 → 完整小实验 | 新算法名不如先把一个模块写对 |

下面按 4 个主题组查漏，不是新的难度等级。每项会分别说明本站讲了什么、参考读到哪里，以及还没验证什么。算法和 PyTorch 是汇总入口，不计入专题正文的阅读数量。

<!-- widget:appendix-coverage -->

## 逐项对照

<!-- widget:reference-coverage -->

## 此前补充了什么 {#earlier-updates}

下面按主题汇总 2026-10-14 之前记在这页的补充，详细讲解都在各篇正文里。小算例和 CPU 检查只核对机制，不是模型训练或 GPU 性能实测；读完参考，也不代表采用其中每个结论。

- **训练基础**：[泛化与训练诊断](../00-foundations/deep-dives/generalization.md)、[激活与初始化](../00-foundations/deep-dives/activation-and-initialization.md)、[优化器](../00-foundations/deep-dives/optimizers.md)、[Muon 与分片反例](../00-foundations/deep-dives/muon.md)。例子为本站原创，不复制参考代码；RNN / GRU、归一化和基础题的相关解释同步修正。
- **PyTorch**：[实践路线](../00-foundations/pytorch/README.md) 5 章，讲跨 batch 广播、错误的 loss 分母、没保留的中间梯度、只恢复权重这类“不报错却算错”的情况。数值测试用 PyTorch 2.8.0 / CPU。
- **多模态**：[LLaVA / DeepSeek-VL](../03-multimodal-learning/vlm-designs.md)、[Qwen-VL](../03-multimodal-learning/qwen-vl.md)、[Omni](../03-multimodal-learning/omni-streaming.md)、[OCR 压缩](../03-multimodal-learning/ocr-compression.md)、[图像生成](../03-multimodal-learning/image-generation.md)、[微调检查](../03-multimodal-learning/vlm-finetuning.md)，以及 BLIP 的筛选审计和 InstructBLIP 消融方案。2026 年的 OCR 2、Qwen3.5 / 3.8-Omni 与 Qwen-Image 后续版本单独核对；多模态 GPU 微调尚未执行。
- **评估**：[BLEU、ROUGE 与编辑距离](../07-evaluation/text-metrics.md)、[Benchmark 与长上下文测试](../07-evaluation/benchmark-protocols.md)。代码只生成测试夹具，没有虚构模型评测结果。
- **后训练**：[DAPO](../05-post-training/dapo.md)、[GSPO / ASPO](../05-post-training/policy-ratios.md)、[SAO / 异步训练](../05-post-training/async-policy-learning.md)、[clipping 变体](../05-post-training/clipping-variants.md)，以及 [loss 为 0 但梯度不为 0](../05-post-training/after-ppo.md#zero-loss-gradient)、[ratio 越界不一定没有梯度](../05-post-training/rlhf/ppo-clipping.md#clipped-token-gradients)、[一组回答何时生成、复用和刷新](../05-post-training/rlhf/on-off-policy.md#grpo-data-lifecycle)。修正了“clipping 硬限制概率”“Critic 只用来做 baseline”等旧说法；没有复现论文的完整训练结果。
- **检索**：[RAG：证据与评估](../04-search/rag-evidence.md)、[向量索引：IVF 与 PQ](../04-search/vector-indexes.md)、[BM25 对比 TF-IDF](../04-search/tfidf-and-bm25.md)（2026-10-09）、[Qwen3 Embedding 与 BGE-M3](../04-search/embedding-models.md)（2026-10-09）、[InfoNCE 与 CE](../04-search/dual-encoder.md#infonce-and-ce)（2026-10-09）。
- **Agents**：[Agent 执行过程](../10-agents/patterns.md)、[Prompt](../10-agents/prompting.md)、[Deep Research](../10-agents/deep-research.md)、[工具、MCP 与 Skills](../10-agents/tools-and-skills.md)、[模型选择与路由](../10-agents/model-choice.md)。本地例子不代表做过模型对比或服务性能测量。
- **模型家族**：[GPT](../00-foundations/model-families/gpt.md)（含 gpt-oss 的公开训练范围与成本记账）、[Llama](../00-foundations/model-families/llama.md)、[Qwen1–2.5](../00-foundations/model-families/qwen-early.md)、[Qwen](../00-foundations/model-families/qwen.md)（更新到 3.8 的公开型号）、[DeepSeek](../00-foundations/model-families/deepseek.md)、[DeepSeek-V4](../00-foundations/model-families/deepseek-v4.md)。不填补未披露的训练配方。
- **架构组件与长上下文**：[BERT 从输入到微调](../00-foundations/core/bert.md)、[Gated DeltaNet 推导](../00-foundations/deep-dives/gated-deltanet.md)、[Gated Attention](../00-foundations/deep-dives/gated-attention.md)、[Engram](../00-foundations/deep-dives/engram.md)、[Block AttnRes](../00-foundations/deep-dives/attention-residuals.md)、[Kimi K3 / NoPE](../00-foundations/deep-dives/nope-and-order.md)（2026-10-09）；DCA、DSA 的例子补进长上下文和稀疏注意力篇。没有声称复现完整训练。
- **MoE**：[路由](../00-foundations/moe/router.md#dispatch-example)的分发、加权回填和梯度检查；[负载均衡](../00-foundations/moe/load-balancing.md#sequence-balance)说明全 batch 均衡不等于逐序列均衡。
- **训练工程与多卡**：[多卡训练](../06-systems/distributed-training.md)、[两张卡算同一层](../06-systems/tensor-parallel.md)、[同样 8 张卡的部署选择](../06-systems/distributed-training.md#eight-gpus)、[通信账](../06-systems/distributed-training.md#zero-communication)、[训练显存记账](../00-foundations/deep-dives/precision-and-memory.md)与[重算例子](../00-foundations/deep-dives/precision-and-memory.md#recompute-example)、[CPU 到 GPU 的数据路径](../00-foundations/pytorch/training-loop.md)、[checkpoint 与恢复](../practice/post-training/checkpoint-and-resume.md)。对应的参考正文和 ZeRO stages 均已读；CPU 结果不代表 GPU 显存或通信性能。
- **推理**：[请求完整流程](../06-systems/llm-serving.md)、[完整显存预算](../00-foundations/deep-dives/kv-cache-and-inference.md#inference-budget)、[缓存生命周期](../00-foundations/deep-dives/attention-kernels.md#cache-lifecycle)、[DFlash / MTP 对照](../00-foundations/deep-dives/dflash.md)与[特征和 shape](../00-foundations/deep-dives/dflash.md#feature-shapes)（2026-10-09）。参考里的请求流程与推理显存两篇已读；公开实现固定到具体版本，容量估算不是 GPU 实测。
- **工程实践**：[沿一次请求排查错误](../practice/post-training/experiments-and-release.md#trace-a-failure)、[长任务中怎样选 GRPO / PPO](../05-post-training/deep-rl/llm-bridge.md#long-horizon-choice)、[学生采样与蒸馏](../05-post-training/distillation.md#opd-choices)、[rollout 到训练数据](../05-post-training/post-training-infrastructure.md#rollout-record)。
- **算法与 Python**：附录 2 的算法分类页已读完，据此补了字符串匹配、单调队列、BST、背包和序列 / 状态机 DP 共 6 对笔记及局部测试，[质数与筛法](../interview/algorithms/primes-and-sieves.md)（2026-10-09），以及代码复习路线里的 [Python：变量、拷贝与函数调用](../interview/python-objects.md)。

参考代码没有执行，图片内公式、嵌入 PDF 和外链视频也未全部核查。全站时效性审查和逐篇技术审稿还没完成，不把构建成功当成内容全部完成。

## 本站额外保留的内容

参考资料不是本站内容的上限。RNN / BPTT、完整 Deep RL 路线、Looped Transformer、评估校准、工具与记忆、系统设计、公开推荐系统拆解、Python 与算法练习都会保留。

这次调整只改归类与阅读入口，不搬文章的公开地址，也不把旧例子、代码和图解缩成摘要。此前明确下架的 Quant 和“聊聊”不在这轮重新上线。

## 接下来怎样核对

按章节继续在浏览器里读，分别记录正文、图片公式和代码读到了哪里，再对照本站的定义、推导、例子与实现。发现缺口就补进已有章节，确实是独立问题再新开一篇；不会为了对齐别人目录而复制一套重复文章。

如果你发现漏项，可以在 issue 里贴主题、公开来源和你希望讲清楚的问题。维护记录见仓库中的 `CONTENT_COVERAGE.md`；主线清单来自 `site/reference-coverage.toml`，截图与浏览器目录的附录对照来自 `site/appendix-coverage.toml`。
