# 面试核心章节：先读哪些，练到哪里

**中文** · [English](core-chapters.en.md)

准备面试时，不一定有时间把站内文章都过一遍。这页先挑出 ML / LLM 方向值得复习的几篇，再按模型训练、推理系统和应用方向往下读。需要补细节时，直接回到对应文章看推导、例子和代码。

先确认下一轮面试的形式。如果只是算法轮，直接去 [Python 与 LeetCode](leetcode.md)；需要安排时间，配合[快速复习](speedrun.md)。这里主要解决“该读哪篇，以及读完能做什么”。

## 先把这六件事串起来 {#core}

这不是所有岗位的共同题库。下面偏向做模型的机器学习工程师（MLE）和研究工程师（RE）：从一次参数更新，走到 Transformer、生成和评估。熟悉的部分先试试右栏；说不清或写不出来，再打开正文。

| 要弄清什么 | 先读这一篇 | 读完试着做什么 |
| --- | --- | --- |
| 矩阵怎样得到梯度，再更新参数 | [矩阵梯度到训练系统](../00-foundations/deep-dives/matrix-gradients-to-systems.md) | 手算一个线性层的输出、dW、dX 和一步 SGD；解释反向为什么需要前向输入 |
| 一次训练到底经过哪些步骤 | [PyTorch 训练循环](../00-foundations/pytorch/training-loop.md) | 写出清零、前向、loss、反向、更新、验证的顺序；检查梯度累积和 loss 分母 |
| Token 之间怎样交换信息 | [多头注意力](../00-foundations/core/multi-head-attention.md) | 标出 Q/K/V 与 attention 分数的形状；说明缩放、causal mask、softmax 各做什么 |
| 模型怎样一个 token 一个 token 地生成 | [一次生成：prefill、decode 与缓存](../practice/runtime-reading/prefill-decode.md) | 用四个输入 token、三个输出 token 画出计算过程；分清采样时间与 KV 写入时间 |
| 训练变好，为什么新数据上反而变差 | [泛化与训练诊断](../00-foundations/deep-dives/generalization.md) | 区分过拟合、数据泄漏和分布变化，为每种情况选一个检查办法 |
| 怎样判断一个改动真的有用 | [评估基础](../07-evaluation/README.md) | 为自己的项目说明数据切分、baseline、指标、重要失败样本和比较条件 |

如果梯度这一行还吃力，先从[线性模型到神经网络](../00-foundations/from-linear-to-neural.md)补起，再回矩阵算例。想把这些概念接到源码，可以沿[数学 → 模型 → 系统 → NeMo RL](../practice/runtime-reading/README.md#small-steps)分段读，不必一口气把框架读完。

### 怎么知道自己不只是看懂了？ {#practice}

挑一个概念，做下面三件事就能发现不少缺口：

- **换一个小条件。** 两个样本的有效 token 数不同，原来的 loss 平均方式还对吗？沿[训练稳定性](../practice/runtime-reading/training-stability.md)复算，不只记“分母很重要”。
- **写一个能发现错误的测试。** 写 causal attention 时改掉未来 token，检查更早位置的输出。实现练习在 [ML Coding](../learn/ml-exercises/README.md)；先跑通小函数，再加 padding 和 cache。
- **用面试语言讲一遍。** 比如英文可以从 “The weight gradient sums contributions from every sample that uses the same weights.” 开始，再写出形状和求和轴。不是背这句话，而是确认切到英文以后仍能说明白。

读到这里，还不需要知道每个模型家族的配置，也不需要记住框架里的全部类名。接下来选与你的工作方向有关的一组。

## 模型训练、后训练与研究 {#training}

| 顺序 | 选读章节 | 可以继续追问什么 |
| --- | --- | --- |
| 先对齐数据与目标 | [数据与训练目标](../practice/post-training/data-and-objectives.md) | 什么位置参与 loss？样本来自哪里？训练、验证与线上请求有哪些差别？ |
| 再比较方法 | [SFT](../05-post-training/sft-and-its-ceiling.md) → [RL 与偏好方法对照](../05-post-training/deep-rl/methods-map.md) | 为什么选 SFT、DPO、PPO 或 GRPO？监督、采样成本和更新方式分别有什么不同？ |
| 需要多卡训练时 | [训练框架](../06-systems/training-frameworks.md) → [一次更新与多卡训练](../06-systems/megatron-basics.md) | 状态放在哪里？哪些数据需要通信？省显存以后多出了什么成本？ |
| 做实验与排错 | [训练稳定性](../practice/runtime-reading/training-stability.md) → [实验与发布](../practice/post-training/experiments-and-release.md) | loss 或 reward 变好了，哪些能力可能没变，甚至退步？怎样安排对照？ |

**可以练一次这样的讨论：**拿自己的任务，比较“不微调”和“SFT”两个方案。写清数据、评估集、成本和上线条件，再说明什么结果会让你放弃微调。涉及 RL 的岗位，再加 rollout、奖励、策略更新和旧样本的问题，接着读 [NeMo RL 源码拆解](../practice/runtime-reading/nemo-rl.md)。

准备研究科学家（Research Scientist，RS）岗位，还要回到自己的方向：选一个具体假设，解释证据、失败结果和下一个实验。这里的 LLM 阅读顺序不能替代计算机视觉（CV）、语音或机器人领域本身的知识。

## 推理、Serving 与 ML Systems {#systems}

这一组从前面的生成例子继续，不从框架名称开始背。

| 顺序 | 选读章节 | 可以继续追问什么 |
| --- | --- | --- |
| 先算资源 | [KV cache 与推理成本](../00-foundations/deep-dives/kv-cache-and-inference.md) | 权重、KV、临时张量各占多少？batch 和上下文长度改变后，哪项先增长？ |
| 再看请求怎么被安排 | [LLM serving](../06-systems/llm-serving.md) | 长输入与正在 decode 的请求怎样共存？吞吐、首 token 延迟和尾延迟怎样取舍？ |
| 选框架与接生态 | [vLLM / SGLang](../06-systems/inference-frameworks.md#ecosystem) | 模型、硬件、后端、观测与部署组件是否适配？哪些差别要在自己的负载上测？ |
| 用测量找瓶颈 | [性能与 profiling](../12-infra/performance.md#profiling) | 时间花在排队、计算、内存访问还是通信？对照实验该固定哪些条件？ |

**可以练一次这样的设计：**给一个文档问答服务加上长输入，用户反馈首字越来越慢。先画请求链路，再说明要补哪些测量；比较调整调度、增加副本或拆分 prefill / decode，而不是直接说“换引擎”。

目标是训练系统时，把重点转到梯度归约、训练状态和恢复；目标是 kernel 时，还要继续 GPU 内存层次与算子实现。相同的 systems title，不代表要准备同一套细节。

## 应用向 MLE：检索、推荐与完整系统 {#applications}

不必把所有题目都讲成 LLM 服务。先选与你项目或岗位最接近的一条：

| 方向 | 阅读顺序 | 练习重点 |
| --- | --- | --- |
| 检索 / RAG | [双塔检索](../04-search/dual-encoder.md) → [混合检索与重排](../04-search/hybrid-and-reranking.md) → [带引用的问答设计](../learn/system-design/rag.md) | 分开检查召回、排序与答案；找一个能定位错误发生在哪一段的例子 |
| 推荐 / 排序 | [推荐系统拆解](../practice/recommender-systems/README.md) → [Feed 设计题](../learn/system-design/feed.md) | 标签、曝光与采样、特征时效性、离线与线上指标，怎样影响模型选择？ |
| 端到端 ML 系统 | [完整 ML 系统](../practice/ml-system/README.md) → [上线与反馈](../practice/ml-system/serving-and-feedback.md) | 从数据到预测再到反馈，说明接口、版本、失败处理，以及哪些模块由其他团队负责 |

多模态岗位可以接[视觉特征怎样进入 LLM](../03-multimodal-learning/vision-to-language.md)和[VLM 微调与评估](../03-multimodal-learning/vlm-finetuning.md)。先把输入、融合位置、训练目标和评估讲通，再深入具体模型。

## 技术阅读之外，留出两件事 {#coding-and-projects}

**算法与读代码要单独练。** [LeetCode 方法地图](leetcode.md)按解题方法组织；[代码理解](code-comprehension.md)练已有实现的追踪与排错。能解释 attention，不代表就能处理算法边界；会写算法，也不代表能发现一个训练循环的错误。

**用自己的项目把知识串起来。** 按[项目准备](../career/prepare.md)选一段真实经历，讲清问题、个人贡献、替代方案、证据和限制。比如被问“为什么用两阶段检索”，既能解释原理，也能说明当时的延迟预算、数据和测量结果。没有测过的地方直接说明，不补一个听起来合理的数字。

这些章节是整理知识的入口，不是对某家公司面试范围的预测。想之后继续读，可以收藏具体文章；它们仍使用原有链接和「我的阅读」，这份索引不会另建一套进度。
