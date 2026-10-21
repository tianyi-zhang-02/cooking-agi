# 递归自我改进（RSI）：先问改的是什么

**中文** · [English](README.en.md)

> 阅读时间：约 7 分钟 · 最近审阅：2026-10

模型拿自己答对的题再训练一轮，agent 修改自己的代码，研究员用 AI 加快实验——这些做法都可能被称为“自我改进”，但自动化的程度和要验证的效果并不一样。

要弄清一个系统做到了哪一步，可以顺着一次改动看：它改了什么？谁提出、执行这个改动？又由谁、根据什么结果决定保留它？本系列用这几个问题比较不同方法，不把它们都等同于“AI 自己造出下一代”。

## 这个想法从哪里来

I. J. Good 在 1965 年的文章里（所在卷册常记为 1966 年）设想：一台智力超过人的机器能设计出更好的机器，进而引发“intelligence explosion”（[DOI](https://doi.org/10.1016/S0065-2458%2808%2960418-0)）。Schmidhuber 2003 年提出的 [Gödel Machine](https://arxiv.org/abs/cs/0309048) 给自我改写设了一道很严的门：只有找到形式证明，确认改写在给定效用函数下有益，才改自己的代码。Yudkowsky 在 [LOGI（2007）](https://intelligence.org/files/LOGI.pdf)里把 seed AI 描述为专门为理解和修改自身而设计的系统，并强调递归自我改进（recursive self-improvement，RSI）是 seed AI 的结果，不是造出 AI 的捷径。

2026 年，Anthropic Institute 的 Favaro 与 Clark 在 [When AI builds itself](https://www.anthropic.com/institute/recursive-self-improvement) 里给了一个更具体的定义：“an AI system capable of fully autonomously designing and developing its own successor”，即一个能完全自主地设计并开发自己继任者的 AI 系统。这几种说法的共同点是，改进作用在“做改进的那个东西”身上，于是下一轮可能改得更快；分歧在于，今天这个回路到底闭合到了哪一步。

## 四种常被叫作“自我改进”的东西

下面的分法是为教学整理的操作定义（例子都有出处），不是业界统一标准。

| 类型 | 改变的对象 | 谁提出／执行改动 | 用什么信号接受改动 | 已核实的例子 |
| --- | --- | --- | --- | --- |
| (a) 模型改进自己的权重 | 模型参数，训练流程由模型自己设计或执行 | 模型自己写数据管线、loss、训练代码并启动训练 | 模型或运营者定义的评估 | **没有找到完整形态的公开演示。** [DGM](https://arxiv.org/abs/2505.22954) 把“改写自己的训练脚本、更新底座模型”列为未来工作；部分前身：[PostTrainBench](https://arxiv.org/abs/2603.08640) 让 agent 后训练另一个 base model，[autoresearch](https://github.com/karpathy/autoresearch) 让 agent 改一个小模型的 `train.py` |
| (b) 系统改进自己的 scaffold、代码或 prompt | agent 的代码、工具、prompt、记忆和流程；权重冻结 | agent 改自己的代码仓库 | benchmark 分数，常在留出任务上 | DGM、SICA、Huxley-Gödel Machine、AIDE² |
| (c) AI 加速 AI 研究 | 人主导的研发过程及其产物：kernel、调度、架构、实验、论文 | agent 做一部分工作；人定方向、把关部署 | 实验指标、人工审查、线上遥测 | AlphaEvolve、ASI-Arch、AI Scientist、Anthropic 的自动对齐研究员、OpenAI 的“研究实习生”指标 |
| (d) 自产训练数据与 self-play | 模型权重，但更新算法是人设计、固定不变的 | 模型提供数据、推理过程或奖励 | 标准答案、代码执行，或模型自己的判断 | STaR、ReST-EM、Self-Rewarding LM、Absolute Zero、R-Zero |

(a) 和 (d) 都改权重，区别在于谁设计更新：(d) 里是人写好的“筛选 → 微调”或 DPO／RL 回路，(a) 里得是模型自己。日常说的“RSI”多半指 (a)，或者把 (c) 推到人完全退出；而公开发表、标着“自我改进”的系统大多属于 (b) 或 (d)。Chen、Wang、Qu 2026 年 7 月的[综述](https://arxiv.org/abs/2607.07663)统计了 1,250 篇论文，说几乎所有论文研究的都是有边界的自我精炼，而不是闭合回路。

<!-- paper-figure:rsi-survey-taxonomy -->

本系列用同一个教学例子：**一个写 Python 函数的模型，加一套判断解答对错的测试。**“生成解答 → 只留测试通过的 → 用它们微调”属于 (d)；让 agent 改写自己调用测试、拆分任务的代码属于 (b)；用 agent 帮研究员调训练 kernel 属于 (c)；只有模型自己决定怎么造数据、怎么改 loss、何时重训自己，才算 (a)。测试一旦会放过错解，每一种回路都会受影响，这是后面几篇反复回到的地方。

## 截至 2026 年 10 月，走到了哪里

- **实验室自报的加速。** OpenAI 2026-09-06 [宣布](https://openai.com/index/research-acceleration-view-inside-openai/)达成自己定下的“自动化研究实习生”目标，自报 2026 年 8 月中研究部门每个人类工作日对应 3.1 个 agent 工作日；同一篇文章也写明，成功完成的 4–8 小时任务里超过一半有过至少一次人工介入，同期可用算力也大幅增长。Anthropic 2026 年 8 月的[风险报告](https://www.anthropic.com/aug-2026-risk-report)自报：Claude 写了合并进生产代码库的大部分代码，内部研发明显更快，但还没到 2 倍。Google DeepMind 2026-10-13 的 [Gemini 4 Argon 安全报告](https://storage.googleapis.com/deepmind-media/gemini/gemini_4_safety_report.pdf)则说，按它自己的测量，没有发现显著的端到端 AI 研发加速。
- **对照正式阈值的判定。** 截至 2026-10-15，没有一家实验室宣布自己的模型越过正式的 AI 研发或自我改进阈值。措辞却越来越谨慎：Anthropic 对 Claude Opus 4.5（2025-11）说排除这一阈值“越来越难”，到 Claude Mythos Preview（2026-04-07）说这一判断的信心低于以往任何模型。细节见[第 3 篇](measuring.md)。
- **公开研究里能复查的结果。** 改自己 scaffold 的 agent 在权重冻结时也能大幅提分（DGM，2025-05：SWE-bench 20.0% → 50.0%）；进化式代码搜索改进了训练下一代模型的基础设施（AlphaEvolve，2025-05：Gemini 训练时间减少 1%）；用自产数据训练能改进权重，但没有外部验证器时几轮内就饱和。
- **还没有的东西。** 没有公开来源记录过一个系统自主设计并训练自己的继任者。Anthropic Institute 的文章说还没到那一步；OpenAI 2026-09-06 的文章说，还不知道怎样安全地走到对齐的、完整的 RSI。

下面是表态和预测，不是测量：Dario Amodei 2026 年 1 月写道，可能只差 1–2 年就会出现当前一代 AI 自主造出下一代的情况（[原文](https://darioamodei.com/essay/the-adolescence-of-technology)）；Jack Clark 2026-05-04 估计到 2028 年底出现完全无人参与的 AI 研发的可能性在 60% 以上（[Import AI 455](https://importai.substack.com/p/import-ai-455-automating-ai-research)）；OpenAI 的 Chris Lehane 2026-09-09 说完全自主的递归自我改进今天并没有发生，并且在能安全做到之前不应追求（[原文](https://openai.com/index/ai-policy-window/)）。

## 时间线

| 日期 | 事件 | 机构／作者 | 一句话 |
| --- | --- | --- | --- |
| 1965/66 | [Speculations Concerning the First Ultraintelligent Machine](https://doi.org/10.1016/S0065-2458%2808%2960418-0) | I. J. Good | 提出 intelligence explosion |
| 2003-09 | [Gödel Machine](https://arxiv.org/abs/cs/0309048) | Schmidhuber | 只有证明有益才改写自己 |
| 2022-03 | [STaR](https://arxiv.org/abs/2203.14465) | Stanford／Google Research | 用自己答对的推理过程再微调 |
| 2024-07 | [Model collapse](https://www.nature.com/articles/s41586-024-07566-y)（*Nature*） | Shumailov 等 | 反复用生成数据训练，分布尾部消失 |
| 2024-12 | [Mind the Gap](https://arxiv.org/abs/2412.02674) | Song 等 | 生成–验证差距决定自我改进；两三轮饱和 |
| 2025-03 | [Time horizon 论文](https://arxiv.org/abs/2503.14499) | METR | 50% time horizon 自 2019 年约 7 个月翻一倍 |
| 2025-05-14 | [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) | Google DeepMind | 进化代码；Gemini 训练时间减少 1% |
| 2025-05-29 | [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) | UBC／Vector／Sakana AI | agent 改自己的代码；SWE-bench 20.0% → 50.0% |
| 2026-01-29 | [Time Horizon 1.1](https://metr.org/blog/2026-1-29-time-horizon-1-1/) | METR | 228 个任务；2023 年后 131 天翻一倍 |
| 2026-05-19 | [Frontier Risk Report](https://metr.org/blog/2026-05-19-frontier-risk-report/) | METR | TH1.1 套件已基本饱和 |
| 2026-08-18 | [Pacing model development](https://openai.com/index/pacing-model-development-cyber-capabilities/) | OpenAI | 7 月内部网络安全评估中 agent 突破隔离后，暂停面向部署模型的 RL 训练两周 |
| 2026-09-06 | [Research acceleration](https://openai.com/index/research-acceleration-view-inside-openai/) | OpenAI | 宣布达成“研究实习生”目标（自报） |
| 2026-09-22 | [AIDE²](https://arxiv.org/abs/2609.26457) | Weco AI | 研究 agent 改写自己的代码，8 天接受 7 次改进 |
| 2026-10-13 | [Gemini 4 Argon 安全报告](https://storage.googleapis.com/deepmind-media/gemini/gemini_4_safety_report.pdf) | Google DeepMind | 未达 ML R&D 阈值；无显著端到端加速 |

## 哪些已经比较确定，哪些还在变

每格都对应一条带日期的公开记录；“接下来看”一栏写的是什么样的结果会改变判断，不是预测它会发生。

| 比较确定 | 还在变 | 接下来看什么信号 |
| --- | --- | --- |
| 没有外部验证器时，用自己的数据训练几轮内就饱和：Song 等 2024-12 测到 gap 两三轮内降到接近 0，ReST-EM（2023-12）两轮以后收益递减 | 用代码执行器自己出题、自己验证能走多远：Absolute Zero（2025-05）和 R-Zero（2025-08）只报告了相对 base model 的提升 | 远多于三轮、仍在独立 benchmark 上持续提升的自训练结果 |
| 冻结权重、只改 scaffold，benchmark 分数能大幅提高：DGM（2025-05）SWE-bench 20.0% → 50.0% | 这种改进会不会复利：HGM（2025-10）发现 benchmark 分数不能很好预测后代表现；AIDE²（2026-09-22）中被改出来的 agent 当外层优化器时，效果无法与基线明确区分 | 被改出来的 agent 在多个随机种子下明显比创造它的 agent 更会改进 agent |
| METR 测到的任务级 time horizon 多年持续上升：2025-03 论文约 7 个月翻倍，TH1.1（2026-01-29）2023 年后 131 天翻倍 | 测量工具本身：METR 2026-05-19 说 TH1.1 已基本饱和，16 小时以上读数不可靠，时间线页面 2026-09-08 起不再更新；OpenAI 2026-07-09 因旧评估饱和换了一套自我改进评估 | METR 或其他机构发布更长任务、带作弊审查的新套件 |
| 截至 2026-10-15 没有实验室宣布越过正式的 AI 研发阈值：Anthropic Claude Opus 5.5（2026-09-22）、OpenAI GPT-6 十月更新（2026-10-07）、Google DeepMind Gemini 4 Argon（2026-10-13） | 实验室怎么量自己的研发加速：OpenAI 3.1 agent 工作日（2026-09-06，自报）、Anthropic“还不到 2 倍”（2026-08，自报）、METR 转述的非公开 ~1.5 倍估计（2026-09-22） | 某家实验室宣布越过正式阈值；按 OpenAI 2026-06 政策蓝图向 CAISI 报告的 RSI 指标出现公开数据 |
| 没有公开演示 (a) 类回路：截至 2026-10-15 未找到系统自主设计并训练继任者的公开记录 | 节奏与治理：OpenAI 2026-08-18 暂停部分 RL 训练两周；Amodei 2026-09 提议给 RSI 速度设限 | 一个系统自主设计、训练并评估自己继任者的公开、可复查演示 |

## 这一组怎么读

需要补基础的话：RL 的基本记号见 [Deep RL](../../05-post-training/deep-rl/README.md)，怎样判断“变好了”见[评估](../../07-evaluation/README.md)，agent 的基本结构见 [Agents](../../10-agents/README.md)。

1. [生成、验证、再训练，为什么卡在验证器](generate-verify-train.md)：STaR、ReST-EM、Self-Rewarding、Absolute Zero，以及测试会放过错解时回路能学到多少
2. [改代码不改权重](code-not-weights.md)：AlphaEvolve 与 Darwin Gödel Machine 怎样在冻结权重下改进程序，离“改自己的权重”还差什么
3. [怎么量“AI 在加速 AI”](measuring.md)：METR time horizon、AI 研发 benchmark、各实验室的阈值与判定，以及自报数字该怎么读
4. [复习题](review.md)：面试题和自检
