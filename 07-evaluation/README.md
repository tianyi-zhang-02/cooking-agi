# Evaluation：我们凭什么说系统变好了？

**中文** · [English](README.en.md)

## 评估先定义“什么算好”

评估不是算出一个分数就结束了。我们想知道的是：这个改动有没有解决原来的问题，又有没有带来新的问题？只看一个总分，往往回答不了这些问题。

## 一个平均分为什么不够

比如让一个旅行 Agent 帮忙安排行程。它的回答看起来很周到，却把酒店日期订错了。

只给文字表达打分，它可能表现不错；但一查最终订单，就会发现任务没有完成。

这说明不同问题需要不同证据：

- 格式是否正确，可以用规则；
- 日期是否一致，可以用确定性检查；
- 酒店是否真的可订，可以调用工具验证；
- 行程是否符合用户偏好，可能需要语义判断；
- 用户长期是否满意，需要看后续行为。

所以通常要把几种评估方式结合起来，不能只靠一个 evaluator。

## 先写清“好”是什么意思

开始评估前，先约定这项任务怎么算完成（task contract）：

```text
Goal        用户真正想完成什么？
Success     什么结果算完成？
Invariant   哪些条件绝对不能破坏？
Tradeoff    速度、成本、质量和风险怎样平衡？
Slice       哪些用户、任务和环境必须分别观察？
```

如果这些问题没有答案，最后的分数再精确也没有意义。

## 这一块怎么读

- [评估栈](evaluation-stack.md)：从确定性检查到在线结果，每一层适合回答什么问题。
- [指标靠得住吗](metric-robustness.md)：平均值掩盖了什么，evaluator 自己又会怎样漂移。
- [LLM-as-a-Judge](llm-as-a-judge/)：让模型打分时，criterion、打分方式和偏差分别怎么处理。

## 它和其他知识点怎样连接

- [Agent Observability](../06-systems/agent-observability.md) 记录中间步骤，方便查出问题发生在哪里。
- [Human-in-the-Loop](../06-systems/human-in-the-loop.md) 介绍哪些判断需要人来确认，以及怎么用人工评审检查 evaluator 的评分。
- [数据与反馈](../01-data-and-feedback/) 决定 eval set 是否代表真实任务。
- [Model Experience](../08-model-experience/) 检查离线指标能否反映用户长期使用的体验。

## 工程实践阅读

- [Confident AI Blog](https://www.confident-ai.com/blog)：持续更新 LLM evaluation、Agent evaluation、observability、Human-in-the-Loop、RAG metrics 与实验工作流。适合了解工程实现和常见设计模式；具体方法仍需要结合原始论文与自己的数据验证。
