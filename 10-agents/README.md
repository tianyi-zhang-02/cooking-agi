# Agents：先说清楚 agent 是什么

**中文** · [English](README.en.md)

> 阅读时间：约 2 分钟 · 难度：进阶 · 最近审阅：2026-09

<div class="lesson-recipe">
  <div><span>解决什么问题</span><strong>什么时候该让模型自己决定下一步，什么时候不该</strong></div>
  <div><span>前置知识</span><strong>LLM 调用 · 工具调用 · 评估</strong></div>
  <div><span>核心机制</span><strong>模型 + 工具 + 循环 + 停止条件</strong></div>
  <div><span>常见错误</span><strong>什么都做成 agent；没有停止条件，也没有验证信号</strong></div>
</div>

## 一个 agent 由四样东西组成

- **模型**：看到当前的情况，决定下一步做什么；
- **工具**：真正对外界动手的接口，比如搜索、读写文件、调 API、跑代码；
- **循环**：工具的结果回到模型，模型再决定下一步；
- **停止条件**：什么时候算做完了，或者预算用完了必须停。

再加上把中间状态记下来的地方（上下文、记忆、外部存储），就是大多数 agent 的全部。

## Workflow 和 agent 的区别

Anthropic 在 [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) 里分得很清楚：**workflow** 是用事先写好的代码路径把模型和工具串起来；**agent** 是让模型自己决定流程和用哪个工具。下面同一个任务，两种做法各走一遍：

<!-- widget:tx-agent-loop -->

## 先问：真的需要 agent 吗

agent 用更高的延迟和成本，换对开放式任务的适应能力。同一篇文章的建议是先找最简单的方案，只有在确实需要时才加复杂度。几个判断问题：

1. **步骤能不能事先列出来？** 能列出来，就写成 workflow。
2. **有没有验证信号？** 测试能不能跑、答案能不能对、订单能不能查。没有验证信号的 agent，很难知道它什么时候做对了。
3. **做错一步代价多大？** 发邮件、付钱、删数据这种不可逆的动作，要么不交给 agent，要么必须有人确认。

## ReAct：想一步，做一步

ReAct（Yao 等，2022）把推理和行动交替起来：模型先写下这一步的想法，再调用一个工具，看到结果后再想下一步。今天大多数 agent 的循环，都是这个模式的变体。

## 这一组怎么读

1. [几种常见结构](patterns.md)：从单次调用到 orchestrator-workers，各适合什么
2. [不同场景怎么用](scenarios.md)：写代码、搜索研究、客服、数据分析、操作电脑、个人助理
3. [用 frontier API 还是自己 serve](model-choice.md)：怎么选模型，以及一个可以自己填数字的成本模型
4. [复习题](review.md)：面试题和自检

相关：[Agent Observability](../06-systems/agent-observability.md) 讲怎样看清一次运行，[Human-in-the-Loop](../06-systems/human-in-the-loop.md) 讲什么时候让人介入。
