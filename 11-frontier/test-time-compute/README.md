# 推理模型与测试时计算：多想一会儿，什么时候有用

**中文** · [English](README.en.md)

> 阅读时间：约 7 分钟 · 最近审阅：2026-10

同一个模型、同一道竞赛题，只采样一次，和采样 64 次再投票、或者让它先想几万个 token，准确率可能差十几个点，花的钱也能差几十倍。2024 年以后的“推理模型”把这件事做成了训练目标和产品开关。**多花推理计算要起作用，需要两个条件：模型有相当的概率碰到正确答案，并且有办法把对的那份挑出来，或者让模型在一条链里自己改对。缺一个，多想只是多花钱。**

## 测试时计算指什么

权重训练完就不再改，回答一个问题时多花的计算都算测试时计算（test-time compute，也叫 inference-time compute）。生成一个 token 大约要 $2N$ 次浮点运算（$N$ 是参数量，不计 attention），所以多花的计算基本就是多生成的 token；[Snell 等（2024-08）](https://arxiv.org/abs/2408.03314)就按 $2N$ 乘以推理 token 数来记账。花法有四种，实际系统常常组合使用：

| 花法 | 多花在哪里 | 需要什么 | 例子 | 典型失效 |
| --- | --- | --- | --- | --- |
| 多采样再挑 | 同一个问题独立生成 N 份 | 答案能判等，或者有验证器 | self-consistency 投票、best-of-N | 正确答案不是众数；验证器放过错解 |
| 想得更长 | 一条链里检查、回溯、换方法 | 训练出会用长链的模型 | o1、DeepSeek-R1 | 简单题想太多，难题来回换思路 |
| 在中间步骤上搜索 | 按步打分，只保留有希望的前缀 | 过程奖励模型或价值估计 | beam search、Tree of Thoughts | 搜索把打分器的漏洞找出来 |
| 生成后检查、修改 | 让程序或模型检查，再改 | 检查器，或会自我检查的模型 | 单元测试、Lean、证明验证器 | 检查的规格和真正想问的问题不一致 |

<!-- paper-figure:snell-search-methods -->

## 两条轴：并行和串行

并行是同时生成 N 份互不依赖的回答，再用投票或验证器挑一份。只要服务端放得下，延迟几乎不随 N 增加，费用随 N 线性增加。串行是在一条链里多想，或者看着上一版改下一版：后面的步骤能用上前面的发现，代价是延迟跟着长度涨。

Snell 等在 PaLM 2-S* 和 MATH 数据集上发现，各种方法的效果“critically varies depending on the difficulty of the prompt”（[摘要](https://arxiv.org/abs/2408.03314)），即很大程度上取决于题目难度，两条轴适合的题也不一样：简单题从串行修改里得到的收益更多，难题上最好在两边之间找一个比例。今天的系统多半两条都用。DeepSeekMath-V2（2025-11）每题先生成 64 份证明、每份验证 64 次，再挑高分的修改，最多迭代 16 轮；OpenAI 的 GPT-5 system card（2025-08）说 ChatGPT 里的 gpt-5-thinking-pro 用的是“并行测试时计算”设置；Google DeepMind 说在 IMO 2025 拿到金牌分数的 Gemini Deep Think 会同时探索并组合多个候选解。后两者没有公开具体做法。

<!-- paper-figure:snell-parallel-sequential -->

本系列用同一个教学例子：**一道答案可以核对的数学题，模型单次答对的概率是 $p$。** 第 1 篇看采样多少次、怎么挑；第 2 篇看一条链怎么被训练得更长，长度又怎么控制；第 3 篇看发布里的数字该怎么比较。

## 时间线

| 日期 | 事件 | 机构 | 一句话 |
| --- | --- | --- | --- |
| 2021-10 | [Training Verifiers](https://arxiv.org/abs/2110.14168) | OpenAI | GSM8K；生成很多份，用训练出的验证器挑一份 |
| 2022-03 | [Self-consistency](https://arxiv.org/abs/2203.11171) | Google | 采样多条推理路径，对最终答案投票 |
| 2023-05 | [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) | OpenAI | 过程奖励模型挑答案胜过结果奖励模型和投票；公开 PRM800K |
| 2024-07 | [Large Language Monkeys](https://arxiv.org/abs/2407.21787) | Stanford 等 | 覆盖率随采样数近似对数线性增长 |
| 2024-08 | [Compute-optimal 测试时扩展](https://arxiv.org/abs/2408.03314) | UC Berkeley／Google DeepMind | 按难度分配测试时计算；它和放大参数不能 1:1 互换 |
| 2024-09-12 | [o1-preview](https://openai.com/index/learning-to-reason-with-llms/) | OpenAI | 用 RL 训练长链；自报性能随训练算力和思考时间平滑上升 |
| 2024-12-20 | [o3 在 ARC-AGI-1 上](https://arcprize.org/blog/oai-o3-pub-breakthrough) | ARC Prize／OpenAI | 6 个样本 75.7%；1024 个样本（约 172 倍计算）87.5% |
| 2025-01-22 | [DeepSeek-R1](https://arxiv.org/abs/2501.12948)（[*Nature* 版](https://doi.org/10.1038/s41586-025-09422-z) 2025-09-17） | DeepSeek | 公开权重和报告：只用规则奖励的 RL 让回答变长、准确率上升 |
| 2025-02-24 | [Claude 3.7 Sonnet 扩展思考](https://www.anthropic.com/news/visible-extended-thinking) | Anthropic | 同一个模型可快答也可长想，调用方用 token 预算控制 |
| 2025-04-16 | [o3 与 o4-mini](https://openai.com/index/introducing-o3-and-o4-mini/) | OpenAI | 推理过程中自己决定何时调用工具 |
| 2025-07-21 | [Gemini Deep Think 的 IMO 结果](https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/) | Google DeepMind | 35/42，由 IMO 协调员按学生标准评分 |
| 2025-08 | [GPT-5 system card](https://arxiv.org/abs/2601.03267) | OpenAI | 实时路由在快模型和思考模型之间选；pro 设置用并行测试时计算 |
| 2025-11-27 | [DeepSeekMath-V2](https://arxiv.org/abs/2511.22570) | DeepSeek | 训练证明验证器，用它当奖励，推理时扩大验证计算 |
| 2026-02-05 | [Claude Opus 4.6](https://www.anthropic.com/news/claude-opus-4-6) | Anthropic | adaptive thinking：模型自己决定想不想、想多久，调用方只调 effort |
| 2026-02-23 | [停报 SWE-bench Verified](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) | OpenAI | 测试有缺陷，前沿模型见过题目和答案 |
| 2026-03-25 | [ARC-AGI-3](https://arcprize.org/blog/arc-agi-3-launch) | ARC Prize | 交互式环境；发布时人类 100%，前沿 AI 0.51% |
| 2026-04-24 | [DeepSeek V4 预览](https://api-docs.deepseek.com/news/news260424/) | DeepSeek | 同一模型有思考和不思考两种模式；2026-08-13 加入 low／high／max 三档 |
| 2026 年 9 月 | [Navier–Stokes 相关证明](https://openai.com/index/navier-stokes-solution/) | OpenAI | 自报约 1 万个并发 agent 跑约 88 小时，附 Lean 形式化；结论范围有争议（[Scientific American，2026-09-21](https://www.scientificamerican.com/article/did-openai-solve-the-wrong-navier-stokes-problem/)） |

## 哪些已经比较确定，哪些还在变

每格都对应一条带日期的公开记录；“接下来看什么”写的是什么样的结果会改变判断，不是预测它会发生。

| 比较确定 | 还在变 | 接下来看什么信号 |
| --- | --- | --- |
| 多采样能提高覆盖率，但没有自动验证器时挑不出来：Brown 等（2024-07）在 SWE-bench Lite 上采 1 次 15.9%、采 250 次 56%（测试能直接判对错）；投票和奖励模型在几百个样本后就不再涨 | 更好的挑法：DeepConf（2025-08）用模型自身置信度过滤推理轨迹，自报在 AIME 2025 上比全量并行少生成最多 84.7% 的 token；DeepSeekMath-V2（2025-11）专门训练证明验证器 | 在没有标准答案的任务上，经独立评审，挑选准确率接近覆盖率 |
| 只用可验证奖励的 RL 能训练出长链，长度随训练增长：DeepSeek-R1-Zero（2025-01 报告，2025-09 *Nature* 版） | RL 是扩展了能力边界，还是只让已有解法更容易采到：Yue 等（2025-04，NeurIPS 2025）与 ProRL（2025-05）、Wen 等（2025-06）结论相反；Wu 等 2026-10-14 更新的逐题分析仍是收缩多于扩展 | 跨多个模型族、检查推理过程、大 $k$ 下 pass@k 也稳定高于基座的结果 |
| 测试时计算的收益依赖题目难度，不能和预训练 1:1 互换：Snell 等（2024-08） | 由谁决定想多久：Anthropic 2026-02-05 起让模型自己决定；截至 2026 年 10 月，OpenAI、Anthropic、Google 的 API 都以离散的 effort 档位为主要开关；Fan 等（2026-08）发现多道题共享预算时，模型不会按难度和分值分配 | 共享预算的评测里，模型按难度分配计算的成绩 |
| 写出来的推理不一定是模型真正依赖的计算：Turpin 等（2023-05）；Chen 等（2025-05）测到 Claude 3.7 Sonnet 和 DeepSeek R1 用了提示后说出来的比例平均只有 25% 和 39% | 思维链监控还能用多久：Baker 等（2025-03）发现把监控器放进奖励后，优化强了，agent 学会在思维链里隐藏意图；多家机构 2025-07 联名呼吁保持可监控性 | 实验室在 system card 里公开可监控性的测量，并说明是否对思维链施加训练压力 |
| 老基准被污染或做满：MathArena（2025-05）发现多数模型在 AIME 2024 上比按 2025 年题推算的水平高 10–20 分；OpenAI 2026-02-23 停报 SWE-bench Verified | 新尺子：ARC-AGI-3（2026-03-25）发布时前沿 AI 0.51%；MathArena 用新出的竞赛题实时评测 | 按等计算量、带置信区间报告的新基准结果 |

## 这一组怎么读

先读这些会更顺：[解码策略](../../00-foundations/core/decoding.md)（采样参数）、[拒绝采样与 best-of-N](../../05-post-training/rejection-sampling.md)、[可验证奖励](../../05-post-training/verifiable-rewards.md)、[GRPO 的组相对信号](../../05-post-training/rlhf/after-rlhf.md#group-example)和 [DAPO](../../05-post-training/dapo.md)。相邻的几篇：[生成、验证、再训练](../self-improvement/generate-verify-train.md)讲同一个验证器上限在训练回路里的样子；[Agent 选模型](../../10-agents/model-choice.md#thinking-control)讲“谁决定要不要思考”；[Looped Transformer](../../00-foundations/looped/README.md)讲不写出 token、在隐藏状态里多算几圈；judge 的可靠性见 [LLM-as-a-Judge](../../07-evaluation/llm-as-a-judge/README.md)。

1. [多采样、投票与验证器](parallel-and-verify.md)：投票为什么只在正确答案是众数时有用，验证器的误判率怎样定下上限，过程奖励模型和搜索，以及 Snell 等的 compute-optimal 到底说了什么
2. [长思维链是怎样训练出来的](long-chains.md)：R1-Zero 做了什么、长度为什么涨，蒸馏，想多久怎么控制，想太多与想太少，思维链是否忠实
3. [怎么比较、怎么读数字](measuring.md)：等计算量比较，pass@1、pass@k 与 maj@k，“RL 只是锐化”之争，污染与饱和
4. [复习题](review.md)：面试题和自检
