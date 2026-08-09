# LLM-as-a-Judge：到底应该怎样让模型打分？

**中文** · [English](llm-as-a-judge.en.md)

## 先用一句话讲清楚

LLM-as-a-Judge 不是一种固定 prompt，而是一组可以自由组合的设计选择：**给不给示范、给不给参考答案、怎样比较候选、判断什么 criterion，以及最后怎样聚合分数。**

很多概念看起来相似，是因为一段 prompt 往往同时使用了好几种方法。

## Few-shot 和 Reference-based 为什么容易混淆

它们是两个正交维度：

| 维度 | Few-shot | Reference-based |
| --- | --- | --- |
| 它在问什么 | 要不要先给模型看几个评分示范？ | 当前答案有没有一个参考答案可以对照？ |
| 主要作用 | 教任务格式、rubric 用法和输出风格 | 给 judge 一个任务级锚点 |
| 核心内容 | 多个输入 → 评分示例 | 当前样本的 `Expected Answer` |
| 对立面 | Zero-shot | Reference-free |

因此四种组合都存在：

| | Reference-free | Reference-based |
| --- | --- | --- |
| **Zero-shot** | 只给 criterion，让 judge 独立评分 | 给 criterion 和当前参考答案，不给示范 |
| **Few-shot** | 给几个评分示范，但示范和当前样本都没有参考答案 | 示范与当前样本都带参考答案 |

一句话记忆：

> **Few-shot 管“有没有示范”，reference-based 管“当前答案有没有标准答案作对照”。**

## 先把几个容易混在一起的字段分开

```text
Task / Input       原始任务是什么？
Candidate          正在被评估的回答或轨迹是什么？
Criterion          这一次只想判断哪个质量维度？
Rubric             不同分数分别代表什么可观察行为？
Reference          一个理想答案、关键事实或允许的答案集合
Demonstrations     教 judge 怎样使用 rubric 的评分示例
Evidence           检索内容、工具结果、环境状态或其他验证信息
Verdict            分数、类别、偏好或通过/失败
Rationale          支持 verdict 的具体证据
```

Reference 和 demonstration 的区别尤其重要：reference 是**当前任务的答案锚点**，demonstration 是**怎样执行评估任务的教学样本**。

## Criterion 应该怎样写

一个好 criterion 通常有五个特点。

### 1. 一次只判断一件事

“这是不是一个高质量回答”太模糊。可以拆成 correctness、faithfulness、relevance、completeness 和 style。

如果必须把多个维度合成总分，也应该先分别判断，再明确怎样组合。

### 2. 能指向可观察证据

不要只写“helpful”。应说明 helpful 在当前任务中表现为什么，例如：是否给出可执行步骤、是否覆盖用户明确约束、是否避免无关内容。

### 3. 写清作用范围

Faithfulness 可以指“只依据给定 context”，也可以指“与现实事实一致”。这两个任务需要的证据完全不同。

### 4. 给出边界和反例

说明什么情况一定失败、什么情况仍然可以通过，以及信息不足时是否允许 `unknown`。

### 5. 让不同分数有行为锚点

`4 = good`、`3 = okay` 没有太大帮助。更好的 rubric 是：

```text
5  完全正确，覆盖所有必要约束，没有实质遗漏
4  核心正确，只有不影响使用的小遗漏
3  部分正确，但遗漏一个重要要求或需要用户补充修正
2  存在主要错误，只有少量内容可用
1  错误、无关，或违反关键约束
```

## 常见 criterion

不同任务应该选择不同组合，而不是默认把所有指标都跑一遍。

### RAG / 问答

- **Answer correctness**：答案是否正确？
- **Faithfulness / groundedness**：结论是否能由提供的证据支持？
- **Answer relevance**：是否直接回答用户问题？
- **Context relevance**：检索证据是否与问题有关？
- **Completeness**：是否覆盖完成任务需要的关键点？

### Agent

- **Task completion**：真实任务是否完成？
- **Tool correctness**：工具选择与参数是否正确？
- **State integrity**：执行后的外部状态是否符合预期？
- **Policy compliance**：是否遵守权限和不可违反的约束？
- **Efficiency**：是否存在无进展的循环、重复搜索或成本失控？
- **Recovery quality**：工具失败后是否正确解释、重试或升级给人？

### Personalization / Model Experience

- **Intent fit**：是否符合当前意图，而不是只匹配长期偏好？
- **Memory appropriateness**：使用的记忆是否相关、仍然有效且允许被纠正？
- **Breadth**：结果是否过度重复或把用户困在一个狭窄主题？
- **Control**：用户能否理解并干预关键决定？
- **Longitudinal improvement**：纠正是否在后续 session 中真正生效？

### 开放式生成

- **Correctness**、**coherence**、**instruction following**、**tone**、**clarity**、**originality**。

这些维度越主观，越需要明确 rubric 和人工校准。

## 几种主要 scoring 方式

### 1. Binary / categorical

输出 `pass / fail / unknown`，或者有限类别。

适合明确条件，例如“是否引用了 context 中不存在的事实”。它通常比模糊的连续分数更容易校准。

### 2. Anchored ordinal score

输出 1–5 等有序等级，每一级都有清楚的行为锚点。

它适合需要表达严重程度的场景，但不要假设 4 和 3 的差距一定等于 3 和 2。

### 3. Pairwise comparison

给出 A、B 两个候选，让 judge 选择 `A / B / tie / both bad`。

适合模型、prompt 或策略的 A/B 比较。它通常比绝对打分自然，但容易受位置偏差影响，所以应交换顺序重新判断。

### 4. Listwise ranking

一次排列多个候选。它节省调用，但候选变多以后上下文和顺序效应更复杂。通常需要随机化顺序或分组比较。

### 5. QAG / 可分解比例

先把复杂目标拆成多个可回答的 yes/no 问题，再根据通过比例计算分数。

例如 completeness 可以先抽取五个必要要点，再判断回答覆盖了几个：

```text
completeness = 覆盖的必要要点数 / 必要要点总数
```

这种分数的来源比直接要求模型“给完整性打 0.73 分”更清楚。

### 6. DAG / 分层 gating

先检查不可违反的条件，再评估开放式质量。例如：

```text
工具参数错误？ → 直接 fail
事实不受证据支持？ → 分数上限为 2
否则 → 再判断帮助性与表达质量
```

DAG 的价值是表达评估逻辑，不是自动让 LLM 判断变成确定性。

## 概率加权分数到底是什么

假设 judge 只能输出 1–5。直接取一个离散分数会产生很多 ties，也看不到模型在 3 和 4 之间的犹豫。

如果能够得到评分 token 的概率，可以计算期望：

```text
score = 1·p(1) + 2·p(2) + 3·p(3) + 4·p(4) + 5·p(5)
```

例如：

```text
p(3)=0.10, p(4)=0.70, p(5)=0.20
score = 3×0.10 + 4×0.70 + 5×0.20 = 4.10
```

这就是 G-Eval 中 probability-weighted scoring 的核心思路：最终得到更细的连续分数，而不是只有整数。

## “生成 20 个 score”是什么

这里要区分两种实现。

### 方法 A：直接读取 token probabilities

如果模型 API 提供输出 token 的 log probabilities，可以直接得到 `p(1)...p(5)`，再计算加权期望。

### 方法 B：重复采样近似分布

如果拿不到完整概率，可以让 judge 独立采样多次。例如 20 次结果为：

```text
3 分：4 次
4 分：12 次
5 分：4 次
```

经验概率为 `0.2 / 0.6 / 0.2`，最终期望分数为 4.0。

所以“生成 20 个分数再 weighted sum”不是新的 criterion，而是**估计评分分布的一种方法**。

## 只保存 weighted mean 还不够

下面两组分布的平均值都是 3：

```text
A: 100% 都给 3 分
B: 50% 给 1 分，50% 给 5 分
```

A 表示 judge 稳定认为答案一般；B 表示 judge 极度不确定，或 rubric 存在两种冲突解释。

因此最好同时保存：

- 加权平均；
- 方差或标准差；
- 各分数概率；
- 重复采样的一致率；
- `unknown / abstain` 比例。

如果分布明显双峰，应该先检查 criterion 和输入证据，而不是用平均值掩盖分歧。

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

## 常见偏差

- **Position bias**：pairwise 中偏爱先出现或后出现的答案。
- **Verbosity bias**：把更长、更详细误判为更好。
- **Self-enhancement bias**：judge 偏爱与自己模型家族相似的输出。
- **Style bias**：被标题、Markdown、语气和自信程度影响。
- **Reference anchoring**：参考答案不完整时，judge 可能错误惩罚其他有效解法。
- **Scale bias**：长期偏爱某几个分数，导致评分范围实际没有被使用。
- **Evidence leakage**：candidate 中的提示注入或自我评价影响 judge。

## 一个更稳健的最小流程

```text
1. 先用确定性规则处理能明确验证的条件
2. 每个 judge 只判断一个 criterion
3. 写出有行为锚点的 rubric
4. 有 reference 就提供，但允许其他等价答案
5. 用少量人工样本校准 prompt 和阈值
6. Pairwise 交换顺序；pointwise 检查分数分布
7. 保存 verdict、rationale、evidence 和 evaluator version
8. 按任务、用户和 failure type 做 slice analysis
9. 定期重新测 judge 与人的一致性
```

## 推荐阅读

- [G-Eval](https://arxiv.org/abs/2303.16634)：CoT evaluation steps、form filling 与 probability-weighted scoring。
- [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)：pointwise、pairwise 评估与常见偏差。
- [Prometheus](https://arxiv.org/abs/2310.08491)：使用细粒度 rubric 与 reference answer 训练专门 evaluator。
- [Judging the Judges](https://arxiv.org/abs/2406.07791)：系统分析 pairwise position bias。

## 和其他章节的连接

- [Evaluation 总览](README.md)
- [Agent Observability](../systems/agent-observability.md)
- [Human-in-the-Loop](../systems/human-in-the-loop.md)
- [数据与反馈](../data-and-feedback/)
