# Evaluation：指标本身靠得住吗？

**中文** · [English](metric-robustness.en.md)

分数上去了，不代表系统真的变好了。最容易骗人的有两处：平均值，和 evaluator 本身。

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
