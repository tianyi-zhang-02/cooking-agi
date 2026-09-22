# Evaluation：从便宜到昂贵的评估栈

**中文** · [English](evaluation-stack.en.md)

没有一个 evaluator 能独自回答所有问题。按成本从低到高，可以把评估手段排成五层。

## 1. 确定性检查

适合 schema、格式、状态变化、工具参数、权限和业务规则。能写成代码判断的，不要先交给概率模型。

## 2. Reference 或 Executor

代码可以跑测试，数学题可以核对答案，检索结果可以检查证据，Agent 可以验证任务是否完成。这类信号通常比语言评分更接近真值。

## 3. LLM-as-a-Judge

适合相关性、帮助性、连贯性、风格和开放式质量。它能扩展到大量样本，但会受到 prompt、模型版本、位置和表达风格影响。

## 4. 人工审查

适合定义 rubric、检查边界案例、发现自动 evaluator 没想到的新失败。人工也会不一致，所以需要校准和记录理由。

## 5. 在线与长期结果

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

→ [LLM-as-a-Judge：criterion、reference 与 scoring 到底怎样组合？](llm-as-a-judge/)
