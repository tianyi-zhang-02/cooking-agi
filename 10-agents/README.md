# Agents：先说清楚 agent 是什么

**中文** · [English](README.en.md)

> 阅读时间：约 5 分钟 · 难度：进阶 · 最近审阅：2026-09

<div class="lesson-recipe">
  <div><span>解决什么问题</span><strong>什么时候该让模型自己决定下一步，什么时候不该</strong></div>
  <div><span>前置知识</span><strong>LLM 调用 · 工具调用 · 评估</strong></div>
  <div><span>核心机制</span><strong>模型 + 工具 + 循环 + 停止条件</strong></div>
  <div><span>常见错误</span><strong>什么都做成 agent；没有停止条件，也没有验证信号</strong></div>
</div>

## 先说一句：这个词一直在变

我第一次听到「agent」想到的是贪吃蛇和吃豆人 :) 在很长一段时间里，agent 就是游戏里那个会动的东西：
它看一眼环境，挑一个动作，环境给它一点回报，如此循环。强化学习整个领域都建立在这个循环上，
今天说的 LLM agent 也还是这个循环——只不过「怎么挑动作」从写死的规则、学出来的策略，换成了问一个模型。

<!-- widget:tx-agent-maze -->

强化学习这边有几件事挺有意思，顺手说说：

- **它算的是以后，不是眼前。** 价值函数 $V(s)$ 是「站在这一格，往后还能拿多少分」，
  折扣 $\gamma$ 决定你有多在乎以后。$\gamma$ 小就短视，$\gamma$ 大就愿意绕远路。上面那张价值图就是它。
- **探索和利用要权衡。** 一直走已知最好的路，就发现不了更好的路；总去尝试新的，又拿不到分。
- **你给什么分，它就优化什么。** 这是最好玩也最吓人的一点。OpenAI 那个划船游戏的例子很出名
  （[Faulty reward functions in the wild](https://openai.com/index/faulty-reward-functions/)，2016）：
  奖励是沿途的分数道具，结果 agent 发现原地转圈刷道具比跑完比赛得分更高，于是它一直转圈，
  一路起火、撞船、逆行，得分却比正常跑完还高。今天 RLHF 里的 reward hacking，和它是同一类问题。
- **LLM agent 也会这样。** 如果把「测试通过」当成奖励，它可能会去改测试，而不是修代码。

所以看到「agent」这个词，最好先问清对方指的是哪一种。这个领域变化很快，定义也一直有争议：
有人认为只有能自己决定流程的系统才算 agent，也有人把一次带工具的调用也叫 agent。
下面讲的是现在最常见的用法。

## 一个 agent 由四样东西组成

- **模型**：看一眼现在的情况，决定下一步做什么；
- **工具**：真正对外面动手的接口：搜索、读写文件、调 API、跑代码；
- **循环**：工具的结果回到模型，模型再决定下一步；
- **停止条件**：什么时候算做完，或者预算花光了必须收手。

再加上一个存中间状态的地方（上下文、记忆、外部存储），大多数 agent 也就这些东西。

## Workflow 和 agent 的区别

Anthropic 在 [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) 里把这条线划得很清楚：**workflow** 是用事先写好的代码路径把模型和工具串起来；**agent** 是让模型自己决定流程、自己挑工具。同一个任务，下面两种做法各走一遍：

<!-- widget:tx-agent-loop -->

## 先问：真的需要 agent 吗

用 agent，是拿更高的延迟和成本，换来处理开放式任务的能力。那篇文章建议先用最简单的方案，确实需要时再增加复杂度。可以先问自己三个问题：

1. **步骤能不能事先列出来？** 能列出来，就写成 workflow。
2. **有没有验证信号？** 比如测试能不能跑通、答案能不能核对、订单状态能不能查到。没有验证信号，就很难知道 agent 到底做对了没有。
3. **做错一步代价多大？** 发邮件、付钱、删数据这种不可逆的动作，要么不交给 agent，要么必须有人确认。

## ReAct：想一步，做一步

ReAct（Yao 等，2022）让推理和行动交替进行：模型先写下这一步在想什么，再调一个工具，看到结果接着想下一步。今天大多数 agent 的循环，都是它的变体。

## 这一组怎么读

1. [几种常见结构](patterns.md)：从单次调用到 orchestrator-workers，各适合什么
2. [不同场景怎么用](scenarios.md)：写代码、搜索研究、客服、数据分析、操作电脑、个人助理
3. [用 frontier API 还是自己 serve](model-choice.md)：怎么选模型，以及一个可以自己填数字的成本模型
4. [复习题](review.md)：面试题和自检

相关：[Agent Observability](../06-systems/agent-observability.md) 讲怎样看清一次运行，[Human-in-the-Loop](../06-systems/human-in-the-loop.md) 讲什么时候让人介入。
