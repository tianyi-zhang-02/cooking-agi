# 学习笔记：今天想学哪一块？

**中文** · [English](README.en.md)

不用先给自己定一个岗位方向。想补基础、写点代码，或试着设计一个系统，都可以从下面找一块开始。

<details markdown="1">
<summary>想切换语言，或对照英文术语？</summary>

点右上角的 **EN / 中文**，就能直接切到同一篇笔记的另一种语言。中文版默认带英文术语对照，比如「注意力（attention）」；每个术语首次出现时补充，不反复堆括注。模型名、公式和代码保持原样。

带有 **English ↻** 按钮的概念卡还能就地切换，不用离开这一段。遇到不熟的词，可以点一下带下划线的术语，或展开文末的「本页术语」看解释。

</details>

## 01 · 基础与模型

先弄明白每个模块在算什么，再看不同模型为什么会选不一样的做法。

- [数学与量化复习](../quant/README.md)：从概率证明出发，接到计数、线代、统计、随机过程、数值与金融数学；覆盖表标明推导深度，复习时能直接找到缺的那一块。
- [大模型学习路线](../00-foundations/study-guide.md)：从向量、Token 到 Transformer，不熟的地方随时回去补。
- [模型家族](../00-foundations/model-families/README.md)：带着同一组问题读 Llama、Qwen、DeepSeek、Gemma。
- [交互图解](../00-foundations/transformer-lab.md)：拖动滑块、改一个参数，看看计算过程和结果怎样变化。

## 02 · 训练与应用

看懂模型里的计算以后，还会碰到几个问题：用什么数据训练？怎么判断它学得好不好？怎样把它用到实际任务里？

| 想弄明白什么 | 从这里开始 |
| --- | --- |
| 从奖励学会一连串决策 | [强化学习 · Deep RL](../05-post-training/deep-rl/README.md)：基础推导 → 核心算法 → 数据与实验 |
| 反馈怎样变成训练目标 | [数据与反馈](../01-data-and-feedback/README.md) → [Post-training](../05-post-training/README.md) |
| 分数涨了，究竟哪里变好了 | [评估](../07-evaluation/README.md) → [LLM-as-a-Judge](../07-evaluation/llm-as-a-judge/README.md) |
| 模型怎么用图片、记忆和外部信息 | [多模态](../03-multimodal-learning/README.md) · [记忆](../02-memory/README.md) · [检索](../04-search/README.md) |
| 一次 AI 请求怎样完成 | [Agents](../10-agents/README.md) · [系统与可观测性](../06-systems/README.md) |

## 03 · 代码与题解

这里可以直接动手练，不是只有准备面试时才用得上。

- [Python](../interview/python.md)：容器、常用函数、语法和容易踩的坑。
- [LeetCode 方法地图](../interview/leetcode.md)：以 Easy / Medium 为主，练能迁移的方法，不拼难题数量。
- [ML 问答与手写](ml-exercises/README.md)：解释原理、算一个小例子，再把它写出来。

每次挑一个小问题：先自己做，再看答案，最后换个条件试一次。能说清为什么这样写，比记住一段代码更有用。

## 04 · 系统设计

[设计题与推演](system-design/README.md)里有几道自拟练习：推荐信息流、带引用的知识库问答、允许更正和删除的长期记忆。每题先交代需求和限制，再比较方案，最后想一想哪里可能出错。架构图是讨论的起点，不是标准答案。

## 想从一个问题一路学下去？

同一套知识可以从不同问题进入。下面不是必修顺序，也不是岗位路线；哪件事让你好奇，就从那儿读。

| 现在的疑问 | 可以连着读什么 |
| --- | --- |
| 日志这么多，为什么模型还是学不好？ | [反馈与目标](../01-data-and-feedback/feedback-to-objectives.md) → [双塔如何学习](../04-search/dual-encoder.md) → [分数能说明什么](../07-evaluation/metric-robustness.md) |
| 做一个能找到资料、记得住更正的助手 | [混合检索与重排](../04-search/hybrid-and-reranking.md) → [记忆的更新与遗忘](../02-memory/memory-lifecycle.md) → [怎样定位评估失败](../07-evaluation/evaluation-stack.md) |
| 想明白模型为什么这样计算 | [注意力](../00-foundations/core/multi-head-attention.md) → [从零实现](../00-foundations/hand-write-kit.md) → [模型家族精读](../00-foundations/model-families/README.md) |
| 想从算法走到实验 | [Deep RL](../05-post-training/deep-rl/README.md) → [实验设置](../05-post-training/deep-rl/experiments.md) → [对照实验与切片](../07-evaluation/ablation-and-slices.md) |

正文中的公式和代码不必第一次就全部读完。先跟着例子理解问题，再回来看推导；已经熟悉概念的话，可以直接去实现和失败案例。这里不按“初级／高级”给读者分组，也不要求把所有章节顺着刷完。

## 想看别的内容？

- **学习笔记**：概念、代码、题解和设计练习。
- **[工程实践](../practice/README.md)**：读公开项目，看看这些方法在代码里怎么实现。可以从 [Twitter / X 推荐系统系列](../practice/recommender-systems/README.md)开始。
- **[求职](../career/README.md)**：找工作的经历、心态、准备节奏和选择。
- **[论文](../papers/README.md)**：回到原始研究，看看作者到底证明了什么。

顶部切换大板块，左侧目录跟着显示对应章节。“我的阅读”里能收藏笔记，也能找回最近读过的内容。想找一个具体概念，可以用顶部的搜索。
