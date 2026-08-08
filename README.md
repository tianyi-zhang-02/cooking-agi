# 学习如何炒一盘 AGI

**中文** · [English](README.en.md)

这里记录我对 **Personal AGI、Search 与 Model Experience** 的持续理解。

这不是一份 AGI 菜谱，而是尝试把模型、数据、记忆、搜索、反馈与评估放到同一张操作台上：理解每种“食材”解决什么问题，它们怎样相互影响，以及系统最后为什么会呈现出某种行为。

它不是论文收藏夹，也不是按领域堆叠的知识库。我更想回答一个贯穿始终的问题：一个模型怎样在长期交互中理解一个人，找到此刻真正有用的信息，并从不完整的反馈中持续改善自己的行为？

## 我目前理解的主线

Personal AGI 不是一个拥有无限上下文的更大模型，而是一个能够闭环学习的完整系统。它需要维护可修正的用户与世界状态，在需要时主动搜索外部证据，通过模型完成推理和行动，再利用评估与真实交互更新自身。

```mermaid
flowchart LR
    A[真实目标与约束] --> B[交互与观察]
    B --> C[用户与世界状态]
    C --> D[记忆、搜索与工具]
    D --> E[模型推理与行动]
    E --> F[Model Experience]
    F --> G[评估与真实反馈]
    G --> H[数据与训练目标]
    H --> C
```

这条链路里，三个兴趣并不是三个分离的方向：

- **Personal AGI 是目标**：模型能否长期理解、适应并帮助一个具体的人。
- **Search 是与世界连接的接口**：决定模型能看到什么证据、候选和行动空间。
- **Model Experience 是最终可观察结果**：用户感受到的不只是一次回答，而是相关性、控制感、广度、可信度和长期改善。
- **Post-training 是行为更新机制**：把示范、偏好和交互反馈转化为模型行为。
- **Multimodal learning 是证据层**：文字、图像、视频、行为与社会上下文共同描述意图和内容价值。
- **Evaluation 是闭环的测量系统**：判断系统是否真的变好，以及改善是否只是指标幻觉。

## 怎样理解一个现代 AI 系统

一个 AI 产品的能力从来不只来自模型权重。更完整的表达是：

> **系统行为 = 数据 × 表征与记忆 × 搜索与工具 × 模型策略 × 运行时 × 评估闭环**

任何一层失真，都会传播到最终体验：

| 层次 | 核心问题 | 常见失真 |
| --- | --- | --- |
| 目标与约束 | 真正要改善什么？对谁、在什么场景下？ | 用容易测量的 proxy 替代真实目标 |
| 数据与反馈 | 观察来自哪里？谁没有被观察到？ | 稀疏、延迟、策略偏置、缺少纵向历史 |
| 表征与记忆 | 什么信息被保留、压缩或遗忘？ | 多重意图被平均成一个点，旧状态无法修正 |
| 搜索与上下文 | 模型能够看到哪些证据与候选？ | 召回空间过窄，相关但重复，证据排序错误 |
| 模型与策略 | 模型如何推理、选择和行动？ | 训练目标与真实使用场景不一致 |
| 运行时与工具 | 正确能力能否稳定、及时地执行？ | 超时、工具失败、状态不同步、成本失控 |
| 评估 | 什么证据足以支持“系统更好”？ | 聚合分数掩盖子群体、长尾和行为退化 |
| 产品反馈环 | 上线后的行为如何进入下一轮学习？ | 把曝光后的点击误当成自然偏好 |

因此，研究一个局部模块时，我希望总能追问它的上游数据、下游消费者、必须保持的 invariant、失败传播路径，以及最终由什么证据验证。

更完整的系统框架见 [`systems/`](systems/)。

## 我怎样看待 LLM Evaluation

LLM-as-a-Judge 很有价值，但它更像一个**可扩展的语义传感器**，而不是真值本身。可靠评估通常需要多层证据：

1. **确定性检查**：格式、schema、工具调用、状态转移与结构 invariant。
2. **参考答案或可执行验证**：代码测试、数学答案、检索证据、任务完成状态。
3. **单样本与 pairwise judge**：用于开放式质量、相关性、帮助性和偏好比较。
4. **人工审计与校准**：检查 rubric、边界案例，以及 judge 的位置偏差、自我偏好和风格偏好。
5. **在线与长期行为**：判断离线提升是否真的改善用户体验，而不是优化一个代理指标。

把复杂 rubric 分解成原子判断、交换 pairwise 顺序、加入参考与 few-shot calibration，通常比直接要求模型输出一个模糊的 1–10 分更可靠。真正重要的不是“用了哪个 judge”，而是测量是否可解释、可复现，并能发现自己何时失效。

## 推荐的起点

这些不是完整书单，而是帮助建立这张概念地图的论文。

| 主线 | 论文 | 我为什么把它放在这里 |
| --- | --- | --- |
| 持续型 Agent | [Generative Agents](https://arxiv.org/abs/2304.03442) | 把记忆、反思与规划连接成长期行为。 |
| 记忆系统 | [MemGPT](https://arxiv.org/abs/2310.08560) | 把上下文管理理解为系统问题，而不是假设 prompt 无限。 |
| Dense Search | [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) | 双编码器检索的清晰起点。 |
| Late Interaction | [ColBERT](https://arxiv.org/abs/2004.12832) | 不把所有匹配信号过早压缩进单一向量。 |
| Retrieval + Generation | [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) | 把检索视为生成过程可更新的外部证据。 |
| 推理与行动 | [ReAct](https://arxiv.org/abs/2210.03629) | 让模型在解决任务时主动获取信息。 |
| 上下文体验 | [Lost in the Middle](https://arxiv.org/abs/2307.03172) | 能访问上下文不等于能有效使用上下文。 |
| 人类反馈 | [InstructGPT](https://arxiv.org/abs/2203.02155) | 理解 SFT、reward modeling 与 RLHF 的经典链路。 |
| 偏好优化 | [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) | 将偏好学习转化为直接的策略优化目标。 |
| 多模态表征 | [CLIP](https://arxiv.org/abs/2103.00020) | 大规模语言监督视觉表征的基础工作。 |
| 多模态交互 | [Flamingo](https://arxiv.org/abs/2204.14198) | 研究图文交错上下文中的少样本学习。 |

## 仓库结构

- [`systems/`](systems/) — 如何把数据、模型、运行时、评估与产品闭环串起来
- [`systems/agent-observability.md`](systems/agent-observability.md) — 从 trace、状态和结果理解 Agent 为什么失败
- [`systems/human-in-the-loop.md`](systems/human-in-the-loop.md) — 用风险与不确定性决定何时引入人工判断
- [`personal-agi/`](personal-agi/) — 用户状态、长期记忆、适应与 Agent
- [`search/`](search/) — 检索、排序、探索与 retrieval-augmented reasoning
- [`model-experience/`](model-experience/) — 行为评估、交互质量与可控性
- [`post-training/`](post-training/) — SFT、偏好学习、RL、训练目标与数据质量
- [`multimodal-learning/`](multimodal-learning/) — 多模态表征与内容理解
- [`papers/`](papers/) — 单篇论文笔记
- [`templates/paper-note.md`](templates/paper-note.md) — 中文论文笔记模板
- [`templates/paper-note.en.md`](templates/paper-note.en.md) — English paper-note template

## 我的笔记方式

我尽量不按章节复述论文。每篇笔记至少回答：

- 它真正解决了什么问题？
- 方法成立依赖哪些假设？
- 核心机制或 invariant 是什么？
- 什么证据支持 claim，什么结果会推翻它？
- 它在完整系统的什么位置？
- 它怎样改变我现在的研究地图？

这是一个持续变化的工作笔记。我的理解会随着阅读、复现和实践不断修正。
