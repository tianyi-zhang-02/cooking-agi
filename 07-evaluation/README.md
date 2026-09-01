# Evaluation：我们凭什么说系统变好了？

**中文** · [English](README.en.md)

## 评估先定义“什么算好”

Evaluation 不是给模型生成一个分数，而是收集足够证据，判断一个具体改动是否真的改善了目标行为，同时没有在别的地方造成更严重的问题。

## 一个平均分为什么不够

一个旅行 Agent 给出了语言流畅、行程漂亮的回答，但订错了日期。

如果只用“表达是否清晰”的 LLM Judge，它可能拿到高分；如果检查工具调用和最终订单，它就是明确失败。

这说明不同问题需要不同证据：

- 格式是否正确，可以用规则；
- 日期是否一致，可以用确定性检查；
- 酒店是否真的可订，可以调用工具验证；
- 行程是否符合用户偏好，可能需要语义判断；
- 用户长期是否满意，需要看后续行为。

没有一个 evaluator 能独自回答所有问题。

## 先写清“好”是什么意思

开始评估之前，先定义 task contract：

```text
Goal        用户真正想完成什么？
Success     什么结果算完成？
Invariant   哪些条件绝对不能破坏？
Tradeoff    速度、成本、质量和风险怎样平衡？
Slice       哪些用户、任务和环境必须分别观察？
```

如果这些问题没有答案，最后的分数再精确也没有意义。

## 一套从便宜到昂贵的评估栈

### 1. 确定性检查

适合 schema、格式、状态变化、工具参数、权限和业务规则。能写成代码判断的，不要先交给概率模型。

### 2. Reference 或 Executor

代码可以跑测试，数学题可以核对答案，检索结果可以检查证据，Agent 可以验证任务是否完成。这类信号通常比语言评分更接近真值。

### 3. LLM-as-a-Judge

适合相关性、帮助性、连贯性、风格和开放式质量。它能扩展到大量样本，但会受到 prompt、模型版本、位置和表达风格影响。

### 4. 人工审查

适合定义 rubric、检查边界案例、发现自动 evaluator 没想到的新失败。人工也会不一致，所以需要校准和记录理由。

### 5. 在线与长期结果

最终还要看真实任务完成、用户纠正、重复使用和长期行为。离线指标改善不一定会转化为更好的体验。

## LLM Judge 应该怎样用

可以把 LLM Judge 想成一个**语义传感器**，而不是裁判席上的最终真理。

更稳健的做法包括：

- 一次只判断一个清楚的问题；
- 能提供 reference 时尽量提供；
- 比较两个方案时交换 A/B 顺序；
- 允许 `tie` 和 `both bad`，不要强迫选赢家；
- 用人工样本检查 judge 在哪些 slice 上失效；
- 保存 prompt、模型版本、temperature 和输入证据；
- 不要求 judge 编造看似精确但没有依据的 1–10 分。

把 rubric 写成 DAG 可以帮助拆分判断，但 DAG 本身不会自动让指标正确。最重要的仍然是每个节点是否对应真实、可验证的标准。

Few-shot、reference-based、pairwise 和概率加权经常被混在一起讲，但它们其实是不同维度。更完整的拆解见：

→ [LLM-as-a-Judge：criterion、reference 与 scoring 到底怎样组合？](llm-as-a-judge.md)

## 为什么一定要看 slice

一个平均 Recall、平均胜率或平均 judge score，可能同时包含：

- 主流用户大幅提升；
- 新用户明显退化；
- 简单任务提升；
- 长任务和多轮任务失败；
- relevance 上升，但 diversity 下降。

平均值告诉我们总体发生了什么，slice 才告诉我们**对谁、在什么条件下发生**。

## 评估也需要被评估

一个 evaluator 上线后也会漂移。模型变了、任务变了、用户变了，旧 rubric 可能不再覆盖真实失败。

所以需要持续问：

- evaluator 与人工判断的一致性有没有变化？
- 它是否偏爱更长、更自信或特定风格的回答？
- 它能不能发现新 failure mode？
- 指标提升是否和真实结果相关？

## 它和其他知识点怎样连接

- [Agent Observability](../06-systems/agent-observability.md) 提供过程证据，而不只看最终回答。
- [Human-in-the-Loop](../06-systems/human-in-the-loop.md) 负责高风险判断和 evaluator 校准。
- [数据与反馈](../01-data-and-feedback/) 决定 eval set 是否代表真实任务。
- [Model Experience](../08-model-experience/) 把离线指标连接到长期感受。

## 工程实践阅读

- [Confident AI Blog](https://www.confident-ai.com/blog)：持续更新 LLM evaluation、Agent evaluation、observability、Human-in-the-Loop、RAG metrics 与实验工作流。适合了解工程实现和常见设计模式；具体方法仍需要结合原始论文与自己的数据验证。
