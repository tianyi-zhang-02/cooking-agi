# LLM-as-a-Judge：打分方式怎么选

**中文** · [English](scoring.en.md)

同一个 criterion，可以用不同的方式打分。

## 1. Binary / categorical

输出 `pass / fail / unknown`，或者有限类别。

适合明确条件，例如“是否引用了 context 中不存在的事实”。它通常比模糊的连续分数更容易校准。

## 2. Anchored ordinal score

输出 1–5 等有序等级，每一级都有清楚的行为锚点。

它适合需要表达严重程度的场景，但不要假设 4 和 3 的差距一定等于 3 和 2。

## 3. Pairwise comparison

给出 A、B 两个候选，让 judge 选择 `A / B / tie / both bad`。

适合模型、prompt 或策略的 A/B 比较。它通常比绝对打分自然，但容易受位置偏差影响，所以应交换顺序重新判断。

## 4. Listwise ranking

一次排列多个候选。它节省调用，但候选变多以后上下文和顺序效应更复杂。通常需要随机化顺序或分组比较。

## 5. QAG / 可分解比例

先把复杂目标拆成多个可回答的 yes/no 问题，再根据通过比例计算分数。

例如 completeness 可以先抽取五个必要要点，再判断回答覆盖了几个：

```text
completeness = 覆盖的必要要点数 / 必要要点总数
```

这种分数的来源比直接要求模型“给完整性打 0.73 分”更清楚。

## 6. DAG / 分层 gating

先检查不可违反的条件，再评估开放式质量。例如：

```text
工具参数错误？ → 直接 fail
事实不受证据支持？ → 分数上限为 2
否则 → 再判断帮助性与表达质量
```

DAG 的价值是表达评估逻辑，不是自动让 LLM 判断变成确定性。

## 我会怎样选择 scoring

| 场景 | 更合适的起点 |
| --- | --- |
| 有明确规则或 executor | 确定性检查，不优先用 LLM |
| 判断是否违反一个语义条件 | Binary + rationale + evidence |
| 比较两个模型或 prompt | Pairwise，交换顺序，允许 tie |
| 判断质量严重程度 | 有行为锚点的 1–5 ordinal rubric |
| 复杂目标由多个子条件组成 | QAG 或 DAG 分解 |
| 需要更细排序且 API 有 logprobs | Rating-token probability expectation |
| 无 logprobs 但需要估计稳定性 | 重复采样，同时报告分布与方差 |
| 有高质量标准答案 | Reference-based |
| 没有唯一答案 | Reference-free + 清晰 rubric + 人工校准 |
