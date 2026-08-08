# Model Experience：用户真正感受到的模型是什么？

**中文** · [English](README.en.md)

## 先用一句话讲清楚

Model Experience 不是某一次回答写得多漂亮，而是用户在连续使用中感受到：模型是否理解上下文、是否值得信任、是否给人控制感，以及是否真的越来越有帮助。

## 一个最简单的例子

两个助手都能正确回答“帮我规划一周健身”。

助手 A 每次都重新问你的目标、器械和伤病情况。

助手 B 记得你的膝盖曾经受伤，但在使用这条信息前会确认它是否仍然有效；当计划不合适时，它能吸收纠正，而不是下次继续犯同样的错。

单次 benchmark 上，两者可能得分接近。长期体验却完全不同。

## 用户体验到的几个维度

### Useful：有没有真正完成任务

不只是语言通顺，而是结果能否执行、能否减少用户工作。

### Relevant：是否理解当前意图

同一个用户在不同时间可能有完全不同的目标。长期偏好不能覆盖当前任务。

### Broad：有没有把用户困在一个狭窄区域

高度个性化可能提高短期 relevance，却减少新内容和新选择。

### Controllable：用户能不能纠正和干预

用户应该能查看、修改记忆，确认高风险行动，并知道系统为什么这样做。

### Calibrated：模型是否知道自己不知道

证据不足时，追问或承认不确定通常比自信生成更好。

### Consistent：跨 session 是否保持连贯

模型不应该今天学会一件事，明天完全忘记；也不应该把一次偶然行为永久化。

## 为什么单轮评估不够

很多重要失败只会在交互轨迹里出现：

- 错误记忆不断被重复使用；
- 用户已经纠正，系统却没有更新；
- 每轮回答都不错，但整体目标逐渐偏移；
- 个性化越来越窄；
- 工具失败后，系统假装任务完成。

所以需要 session-level 和 longitudinal evaluation，而不只是单个 prompt—response pair。

## 离线指标怎样连接体验

Recall、NDCG、judge score、task success 都有用，但每个指标只看见一部分。

更完整的评估可以组合：

- 任务完成率和可执行验证；
- relevance、coverage 与 diversity；
- 用户纠正次数和重复解释成本；
- 记忆更新与撤回是否成功；
- 工具和行动的实际结果；
- 多次 session 后的满意度和依赖程度。

## 它和其他知识点怎样连接

- [Personal AGI](../personal-agi/) 定义长期关系的目标。
- [表征与记忆](../memory/) 决定系统怎样保持连续性。
- [Search](../search/) 决定模型能提供怎样的信息空间。
- [Evaluation](../evaluation/) 把主观体验拆成可以验证的证据。
- [Human-in-the-Loop](../systems/human-in-the-loop.md) 提供控制、升级和纠正机制。

## 起始论文

- [Lost in the Middle](https://arxiv.org/abs/2307.03172)
- [InstructGPT](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
