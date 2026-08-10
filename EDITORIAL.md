# Modern-first 写作与时效性规范

**中文** · [English](EDITORIAL.en.md) · [返回首页](README.md)

> 阅读时间：约 5 分钟 · 最近审阅：2026-08

## 核心原则

仓库从当前 AI system 的真实问题开始，而不是从算法历史开始：

```text
用户目标
→ 数据、memory、search 与 tools
→ LLM / multimodal policy
→ training、serving 与 runtime
→ evaluation、feedback 与持续改进
```

一项旧工作是否进入主路线，不取决于年份，而取决于它能否解释今天仍然存在的机制、约束或失败。

## 主路线包含什么

- 当前 LLM 和 multimodal model 的行为与系统机制；
- Agent memory、search、retrieval、tools 和 state；
- SFT、preference learning、RL、distillation 和 model updating；
- evaluation、LLM judge、human feedback 和 online outcomes；
- training infra、inference、GPU、data 和 observability；
- personal AI、长期交互、control、privacy 和 model experience；
- 能够验证上述概念的现代实战项目。

## 什么不单独展开

- 传统 classifier、CNN、RNN、SVM 等模型的目录式介绍；
- 只按年代排列、不能解释当前决策的论文史；
- 已经过时的 framework API 教程；
- 没有版本、硬件和 workload 的性能数字；
- 只有术语列表、没有系统关系和验证方法的内容；
- 为了“完整”而加入、却不会改变理解或实践的背景知识。

例外：如果历史机制仍然活跃，就应该解释。例如 stable softmax、embedding retrieval、SIMD/SIMT、attention 的二次方中间状态和 exposure bias 都直接影响现代系统。

## 五分钟默认值

每篇默认：

- 只回答一个主要问题；
- 约五分钟阅读；
- 一个核心例子或心智模型；
- 一个可验证的量、实验或判断；
- 明确的下一篇或相关模块。

长主题用 index + short notes 组织。Project overview 保持短，具体 milestone 拆到子笔记或代码目录。

## 时效性

| 内容 | 复核节奏 |
| --- | --- |
| 数学、内存、统计和系统基础 | 每年 |
| Search、evaluation、post-training 方法 | 每 6 个月 |
| API、serving engine、distributed stack、低精度和硬件支持 | 每季度 |
| 明确版本或产品默认行为 | 发布前重新验证 |

快速变化的页面应包含 `Last reviewed: YYYY-MM`。无法确认的当前事实不应以永久结论写入。

## 来源顺序

1. 官方文档和规范；
2. 原始论文；
3. 可复现代码、实验和 benchmark；
4. 高质量二手解释。

事实、实测结果和推断要明确区分。Benchmark 必须包含模型、数据、硬件、版本、配置和 workload。

## Modern 不等于追新

Modern-first 不是追逐每个新名词。真正的判断标准是：

- 是否已经改变主流系统设计；
- 是否有可靠证据；
- 是否解决真实瓶颈；
- 是否能与现有模块建立清楚关系；
- 六个月后是否仍可能帮助做决策。

如果答案只是“最近很热”，它应该先进入实验记录，而不是主学习路线。

AI Infra 使用更具体的 [五分钟笔记与时效性规则](ai-infra/EDITORIAL.md)。
