# LLM-as-a-Judge：到底应该怎样让模型打分？

**中文** · [English](README.en.md)

## 快速学习：Judge 不是一个 prompt，而是一套 measurement system

<details class="interview" markdown="1">
<summary>Criterion、scoring、calibration 与 validation</summary>

**快速记忆**：先定义可观察 criterion，再选 pairwise/ordinal/binary scoring；最后用专家标签、扰动测试与分布切片校准 judge。

**面试回答**

> LLM-as-a-Judge 需要明确输入证据、单一 criterion、行为锚点、输出 schema 和 aggregation。Pairwise 通常比绝对分更稳定，但仍有 position、verbosity、self-preference 和 reference leakage。上线前必须与专家审计交叉验证，并报告 disagreement 与置信度而不只给均值。

<details markdown="1">
<summary><b>深挖</b>：为什么 weighted mean 会隐藏失败？</summary>

相同均值可能来自“大家都给 3 分”或“半数 1 分、半数 5 分”，两者决策风险完全不同。应保留 score distribution、entropy/disagreement、criterion slices 与 judge version；对 high-stakes case 还要升级给人，而不是把不确定性压成一个标量。

</details>
</details>

## Judge 是评估系统，不是一个 Prompt

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

## 这一组怎么读

- [Criterion 怎么写](criteria.md)：好 criterion 的五个特点，以及不同任务常用的 criterion。
- [打分方式怎么选](scoring.md)：binary、ordinal、pairwise、listwise、QAG、DAG 各适合什么，以及我会怎样选。
- [概率加权分数](probability-scores.md)：G-Eval 式的期望分数、重复采样，以及为什么只存均值不够。
- [偏差与稳健流程](bias-and-workflow.md)：judge 常见的偏差，和一个更稳健的最小流程。

## 推荐阅读

- [G-Eval](https://arxiv.org/abs/2303.16634)：CoT evaluation steps、form filling 与 probability-weighted scoring。
- [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)：pointwise、pairwise 评估与常见偏差。
- [Prometheus](https://arxiv.org/abs/2310.08491)：使用细粒度 rubric 与 reference answer 训练专门 evaluator。
- [Judging the Judges](https://arxiv.org/abs/2406.07791)：系统分析 pairwise position bias。

### 工程实践

- [Confident AI Blog](https://www.confident-ai.com/blog)：有较完整的 LLM-as-a-Judge、G-Eval、Arena-style comparison、RAG evaluation、Agent evaluation 与 observability 实践文章。适合补充实现思路，但不替代原始论文和针对自己任务的校准实验。

## 和其他章节的连接

- [Evaluation 总览](../)
- [Agent Observability](../../06-systems/agent-observability.md)
- [Human-in-the-Loop](../../06-systems/human-in-the-loop.md)
- [数据与反馈](../../01-data-and-feedback/)
