# 五分钟笔记与时效性规则

**中文** · [English](EDITORIAL.en.md) · [返回索引](README.md)

> 阅读时间：约 5 分钟 · 类型：编辑规范 · 最近审阅：2026-08

## 一篇只回答一个问题

标题应该能变成一个明确问题，例如“BF16 为什么比 FP16 更适合很多训练任务？”而不是“低精度大全”。如果一个页面同时解释硬件、算法、API、benchmark 和职业路线，就应该拆分。

建议正文目标：

- 英文约 500–900 words；
- 中文约 1,000–2,200 字符；
- 一个主要心智模型；
- 最多一到两个必要公式或图；
- 一个可以执行的练习或判断题。

代码、表格和引用不需要为了满足字数被机械删除，但它们必须服务于同一个问题。

## 固定结构

```text
Title
Reading time · Level · Freshness · Last reviewed
Why this matters
Core mental model
What to calculate or observe
Hands-on check
Common failure
Next note
```

Project page可以稍长，但 overview 仍控制在五分钟左右。具体 milestone 拆成独立子笔记。

## 分类原则

主索引按工作问题分类，而不是按论文年代或厂商分类：

- Foundations；
- Training systems；
- Inference systems；
- Platforms and reliability；
- Data, evaluation, and learning loops；
- Hands-on projects。

一篇笔记只放在一个主要分类中，其他关系用 “Related notes” 表达，避免同一链接到处重复。

## 时效性标签

| 标签 | 含义 | 复核频率 |
| --- | --- | --- |
| Stable foundation | 体系结构、内存、数值等长期概念 | 每年 |
| Evolving | 常用设计仍稳定，但工具和实现持续变化 | 每 6 个月 |
| Fast-moving | API、硬件格式、serving 和分布式实现 | 每季度 |
| Version-specific | 依赖某个明确版本或硬件 | 发布前重新验证 |

每篇都写 `Last reviewed: YYYY-MM`。这表示内容在该时间被重新检查，不表示引用资料的发布日期。

## 什么时候需要更新

- 官方 API、默认值或支持矩阵变化；
- 新硬件改变主流 dtype、通信或执行路径；
- 主流 serving/training 方案发生结构变化；
- 原有 benchmark 不再代表真实 workload；
- 链接失效或来源被弃用；
- 实战结果与正文结论冲突。

更新时优先修改心智模型和决策规则，不要只堆版本发布记录。

## 什么不进入主路线

- 与当前 LLM systems 没有直接关系的传统模型目录；
- 只因为历史上重要、却不能解释当前设计的长篇回顾；
- 没有测量条件的性能数字；
- 未注明版本的 vendor feature；
- 只有框架调用、没有解释数据移动和瓶颈的教程；
- 把即将过时的 API 当作永久概念。

历史内容只有在解释当前系统时保留，例如 softmax 数值稳定性、SIMD/SIMT 差异或 attention 的 IO 问题。

## 来源规则

技术事实优先使用：

1. 官方硬件和软件文档；
2. 原始论文；
3. 可复现的代码与 benchmark；
4. 高质量工程解释作为补充。

涉及最新 API、硬件支持或默认行为时必须重新检查官方资料。推断和实测结论要明确区分。

## 完成检查

- 能否用一句话说出这篇解决的问题？
- 是否能在约五分钟读完？
- 是否包含读者可以验证的量或练习？
- 是否删除了与当前决策无关的历史内容？
- 快速变化的事实是否有审阅日期和可靠来源？
- 下一篇链接是否自然，而不是强迫线性阅读？

满足这些条件后，笔记才进入主索引。
