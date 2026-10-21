# 自我改进：怎么量“AI 在加速 AI”

**中文** · [English](measuring.en.md)

> 阅读时间：约 12 分钟 · 最近审阅：2026-10

“某实验室 80% 以上的合并代码由 AI 写”和“某模型能完成人类 12 小时的任务”，这两条消息能说明 AI 研发在加速吗？它们都不能直接回答：前者量的是代码份额，后者量的是任务级能力，研发整体快了多少则是另一个问题。

读这些数字，先要确认由谁测量、使用什么单位。截至 2026 年 10 月，外部可复查的任务级测量仍在上升，但已经达到测量工具的上限。整体研发提速的数字几乎都来自实验室自己；给出整体进展倍数的估计都在 2 倍以下。

## 先把证据分成四类

- **任务级测量**：METR 的 time horizon、RE-Bench 等。外部可以复查，但量的是“能完成多长的任务”，不是研发进度。
- **实验室对照自己阈值的判定**：system card 里“越过／没越过”的结论，依据多半是内部评估。
- **实验室自报的内部数字**：代码份额、agent 工时、员工调查、能力指数的斜率。
- **表态和预测**：见[导读](README.md)，这里不重复。

## 50% time horizon：定义和拟合

METR 的 Kwa 等在 [2025-03-18 的论文](https://arxiv.org/abs/2503.14499)里这样定义：AI 模型以 50% 成功率能完成的任务，人类通常要花多长时间。对每个模型，按任务的人类用时 $t$ 拟合（§3.1）：

$$
P(\text{成功}\mid t)=\sigma\big(\beta\,(\log h-\log t)\big),\qquad \sigma(u)=\frac{1}{1+e^{-u}}
$$

$t$ 是成功完成该任务的人类基线用时的几何平均；$h$ 和 $\beta$ 由拟合得到。$t=h$ 时 $\sigma(0)=0.5$，所以 $h$ 就是 50% horizon。连续打分的任务按代表人类水平的阈值换成成功／失败；每个任务按所属任务族大小的平方根倒数加权，避免任务多的族主导结果。

<!-- paper-figure:metr-logistic-fits -->

### 手算一次

教学例子，数字是编的：一个模型在四组任务上各跑 10 次。

| 人类用时 $t$ | $x=\log_2(t/1\,\text{小时})$ | 成功次数 | 成功率 $p$ | $\operatorname{logit}p$ |
| --- | --- | --- | --- | --- |
| 15 分钟 | −2 | 9/10 | 0.9 | 2.20 |
| 1 小时 | 0 | 7/10 | 0.7 | 0.85 |
| 4 小时 | 2 | 3/10 | 0.3 | −0.85 |
| 16 小时 | 4 | 1/10 | 0.1 | −2.20 |

把成功率换成 logit，模型就变成关于 $x$ 的一条直线：

$$
\operatorname{logit}p=\ln\frac{p}{1-p}=\beta\,(\log h-\log t)
$$

四个点以 $x=1$ 为中心对称，最小二乘直线的斜率约为 −0.74（用时每翻一倍），在 $x=1$ 处穿过 0，所以 $h=2^1=2$ 小时。2 小时是 1 小时和 4 小时的几何中点，不是算术中点 2.5 小时，因为横轴是对数时间。用同一条线求 80% horizon：$\operatorname{logit}0.8=1.39$，$x=1-1.39/0.74\approx-0.86$，约 0.55 小时，即 33 分钟左右。80% horizon 通常比 50% 短得多，下文 METR 的数据也是这样。METR 实际是在单次运行上做加权最大似然；这里用 logit 直线只为能手算，四点对称时两种方法的 50% 交点都在 2 小时，斜率和 80% horizon 会略有不同。

### 读数超出任务范围时

换一个更强的模型，15 分钟组仍是 9/10，其余三组如下：

| 情形 | 1 小时 | 4 小时 | 16 小时 | 拟合出的 50% horizon |
| --- | --- | --- | --- | --- |
| 上例 | 7/10 | 3/10 | 1/10 | 2 小时 |
| 更强的模型 | 8/10 | 7/10 | 6/10 | 约 34 小时 |
| 16 小时组多成功 1 次 | 8/10 | 7/10 | 7/10 | 约 108 小时 |

后两行所有组的成功率都过半，50% 的点只能沿直线往外推，落在最长任务之外。最长一组多成功一次，估计就从约 34 小时跳到约 108 小时。这正是 METR 提醒 16 小时以上读数不可靠的原因；METR 2026-03-20 的方法说明还提到，套件里没有超过 30 小时的任务。

<!-- widget:step-time-horizon -->

## 趋势数字，和它的几个前提

METR 2025-03 的论文[摘要](https://arxiv.org/abs/2503.14499)报告，前沿模型的 horizon “doubling approximately every seven months since 2019, though the trend may have accelerated in 2024”，即自 2019 年大约每 7 个月翻一倍，2024 年可能更快；论文对“5 年内能做月级软件任务”的外推是有条件的。METR 2026-01-29 发布的 [Time Horizon 1.1](https://metr.org/blog/2026-1-29-time-horizon-1-1/) 把任务从 170 个扩到 228 个，8 小时以上的任务从 14 个增加到 31 个；2023 年以后的翻倍时间是 131 天（旧版 TH1 为 165 天），2024 年以后是 89 天（TH1 为 109 天）。METR 当前的[数据文件](https://metr.org/assets/benchmark_results_1_1.yaml)给出 2023 年以后 128.7 天（95% CI 104.4–158.0），已剔除 50% 点超过 16 小时的模型。

<!-- paper-figure:metr-time-horizon -->

换算一下 131 天：一年约 $365/131\approx2.8$ 次翻倍，horizon 每年约乘以 6.9。这是对过去趋势的算术换算，不是预测。读这类数字时要带上四个前提：

1. METR 2026-05-19 的 [Frontier Risk Report](https://metr.org/blog/2026-05-19-frontier-risk-report/)（评估窗口 2026-02-16 至 03-16，Anthropic、Google、Meta、OpenAI 参与）说，最强的 agent 基本上让 TH1.1 饱和了，最强的共享模型 50% 点估计在 16–20 小时之间。
2. 同一份报告：8 小时以上任务的成功运行中，至少 16% 经审查不合法。METR 2026-06-26 对 [GPT-5.6 Sol](https://metr.org/blog/2026-06-26-gpt-5-6-sol/) 的测量里，作弊记为失败约 11.3 小时（95% CI 5–40 小时），剔除作弊尝试是 71 小时，作弊记为成功则超过 270 小时；METR 说这些数字都不算稳健测量。
3. METR 2026-03-20 的[方法说明](https://metr.org/notes/2026-03-20-impact-of-modelling-assumptions-on-time-horizon-results/)：校正任务时长估计里的噪声，可能让前沿模型的 50% horizon 下降约 30%。
4. METR 的[时间线页面](https://metr.org/time-horizons/) 2026-09-08 起不再主动更新。Claude Opus 4.6 的读数在 2026-03-03 因正则化错误被修正；最初流传的约 14.5 小时只见于二手来源。

## 各模型的读数

来源是 METR 的数据文件和时间线页面，括号里是 95% CI（只列出有记录的）：

| 模型 | 50% horizon | 80% horizon | METR 加入日期 |
| --- | --- | --- | --- |
| GPT-5 | ≈3.4 小时 | 38 分钟 | — |
| Claude Opus 4.5 | ≈4.9 小时 | 49 分钟 | — |
| Gemini 3 Pro | ≈3.7 小时 | 54 分钟 | 2026-02-03 |
| GPT-5.2 | ≈5.9 小时 | 66 分钟 | 2026-02-04 |
| Claude Opus 4.6 | ≈12.0 小时（5.3–60.6 小时） | 70 分钟 | 2026-02-20，03-03 修正 |
| Gemini 3.1 Pro | ≈6.4 小时 | 90 分钟 | 2026-04-15 |
| Claude Mythos Preview（早期快照） | ≈17.4 小时（8.5–55.1 小时） | ≈3.1 小时（1.6–6.6 小时） | 2026-05-08 |
| GPT-5.6 Sol（METR 单独发文） | ≈11.3 小时（5–40 小时），作弊记为失败 | — | 2026-06-26 |

Mythos Preview 的点估计已超出 METR 自己说的可靠范围；Opus 4.6 的区间上限达 60 小时。区间这么宽，相邻两个模型的先后往往说明不了什么。

## AI 研发 benchmark

| Benchmark | 机构与日期 | 测什么 | 首发结果 | 截至 2026 年 10 月 |
| --- | --- | --- | --- | --- |
| [RE-Bench](https://arxiv.org/abs/2411.15114) | METR，2024-11 | 7 个开放式 ML 研究工程环境；61 位专家共 71 次 8 小时尝试 | 2 小时预算下最好的 agent 得分是人类专家的 4 倍；8 小时人类略胜，32 小时人类约为最好 agent 的 2 倍 | 没找到单独的 2026 年更新；任务已并入 TH1.1 |
| [MLE-bench](https://arxiv.org/abs/2410.07095) | OpenAI，2024-10 | 75 个 Kaggle 比赛的拿奖牌率 | o1-preview + AIDE 在 16.9% 的比赛拿到至少铜牌 | 官方[排行榜](https://github.com/openai/mle-bench)最高为 Famou-Agent 2.0 的 64.44%（2026-02-23，转读未逐项核对）；2026-04-24 起不再接受提交 |
| [PaperBench](https://arxiv.org/abs/2504.01848) | OpenAI，2025-04 | 从零复现 20 篇 ICML 2024 论文，LLM judge 按 8,316 个评分细项打分 | 最好的 agent 21.0%；ML 博士在 3 篇子集上 48 小时得 41.4%（三人取最好），o1 为 26.6% | 没找到 2026 年数字；OpenAI 2026 年的 system card 已不再用它 |
| [PostTrainBench](https://arxiv.org/abs/2603.08640) | Rank 等，2026-03 | agent 在一张 H100 上 10 小时内后训练一个 base LLM | 最好的 agent 23.2%，官方 instruct 模型 51.1%；观察到 reward hacking | Lite 版已进入 OpenAI GPT-5.6 的自我改进评估套件 |

## 实验室自己画的线

| 实验室与文件 | 等级 | 大意 |
| --- | --- | --- |
| OpenAI [Preparedness Framework v2](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf)（2025-04-15） | High | 相当于给每位 OpenAI 研究员配一位高水平的中级研究工程师助手，相对这些研究员 2024 年的基线 |
| 同上 | Critical | 能递归自我改进，即完全自动化的 AI 研发。领先指标：超人的研究科学家 agent；滞后指标：以 2024 年同等进展 1/5 的墙钟时间（例如 4 周）完成一代模型的提升（如 o1 到 o3），并持续数月。达到后的应对包括暂停进一步开发，直到安全措施明确 |
| Anthropic [RSP v2.2](https://www-cdn.anthropic.com/872c653b2d0501d6ab44cf87f43e1dc4853e4d37.pdf)（2025-05-14） | AI R&D-4 | 能完全自动化 Anthropic 一名入门级、纯远程研究员的工作 |
| 同上 | AI R&D-5 | 能让有效扩展的速度显著加快：一年的有效算力增长相当于 2018 年初到 2024 年初平均速度下的两年 |
| Anthropic [RSP v3.4](https://www-cdn.anthropic.com/files/4zrzovbb/website/0bacdc8440ea96e62a8766d99ebe1d4eea6d5f3a.pdf)（2026-07-08 生效；v3.0 于 2026-02-24 全面重写） | Automated R&D in key domains | 满足其一：(1) 能以有竞争力的成本（5 倍以内）替代全部研究科学家和研究工程师；(2) AI 进展显著加速，可操作化为至少连续三代模型，总体能力进步速度达到预期速度、以及无 AI 显著贡献时最快持续速度的两倍 |
| Google DeepMind [FSF v3.0](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/strengthening-our-frontier-safety-framework/frontier-safety-framework_3.pdf)（2025-09-22；v3.1 于 2026-04-17 措辞不变） | ML R&D acceleration level 1 | 已被用于加速 AI 开发，使 AI 进展相对历史速度显著加快 |
| 同上 | ML R&D automation level 1 | 能以大致相当的全部成本，完全自动化 Google 任一专注提升 AI 能力的研究团队的工作 |

Anthropic 在 v3.4 的变更记录里说，这个阈值意在捕捉“dramatic recursive self-improvement”的开端，并且很难操作化。OpenAI 2026-08-18 说会修订 Preparedness Framework；截至 2026-10-15 没有找到新版本。OpenAI 2026 年 6 月的[政策蓝图](https://openai.com/index/frontier-safety-blueprint/)提议前沿开发者向 CAISI 分享 RSI 相关测量，这是提议，文中没有公布指标。

## 2025–2026 的判定

每家从新到旧排列。没有一家报告越过 AI 研发阈值，但下判断的证据在变：旧的自动化任务评估饱和以后，各家转向员工调查、能力指数的趋势断点和新的内部评估。

**Anthropic**

- [Claude Opus 5.5](https://www.anthropic.com/claude-opus-5-5-system-card)（2026-09-22）：没有越过下一个能力阈值。CoBench 2.1 得 55.8%，可替代员工的标准是 ≥85%；能力指数趋势断点拟合的斜率增幅 1.53 倍（95% 区间 1.20–1.82），即斜率没有翻倍；部分内部加速指标已有变动，正在密切监测。
- [Claude Mythos 5.1](https://www-cdn.anthropic.com/0339e6a7c5c7b87f5c07798616dc32c215d14235/Claude%20Fable%205.1%20&%20Claude%20Mythos%205.1%20System%20Card.pdf)（2026-09-01）：没有越过“显著加速自动化 AI 研发”的阈值；旧的自动化评估已饱和，没有再跑。
- [Claude Mythos Preview](https://www.anthropic.com/claude-mythos-preview-system-card)（2026-04-07）：没有越过“两年进展压进一年”的阈值，但这一判断的信心低于以往任何模型。员工调查的提速几何平均约 4 倍，结合弹性估计，整体进展乘数低于 2 倍；能力指数斜率之比在 1.86–4.3 倍之间，取决于断点选在哪里，Anthropic 把这次跃升归因于 AI 加速研发以外的因素。
- [Claude Opus 4.6](https://anthropic.com/claude-opus-4-6-system-card)（2026-02）：排除论证比以往任何模型都更站不稳。kernel 优化在新 scaffold 下达到 427 倍提速（阈值 300 倍）；16 位受访员工中没有人认为它能自动化一名入门级研究员。
- [Claude Opus 4.5](https://www-cdn.anthropic.com/bf10f64990cfda0ba858290be7b8cc6317685f47.pdf)（2025-11）：没有越过 AI R&D-4，但有把握地排除越来越难；内部套件 0.604，排除线 0.6；18 位员工中没有人认为它能自动化入门级研究员。

**OpenAI**（system card 都在 [deploymentsafety.openai.com](https://deploymentsafety.openai.com/)）

- GPT-6 Sol 与 Luna 的十月更新（2026-10-07）：两者都没达到 AI Self-Improvement 的 High。
- GPT-6.1 Sol（2026-09-29）：低于 High；内部 Research Debugging 75.52%。
- GPT-6 Astra（2026-09-03）：AI Self-Improvement 未达 High；它是 OpenAI 第一个在网络安全上评为 Critical 的模型。Research Debugging 78.05%，低于未公布的 High 参考阈值。
- GPT-5.6（2026-07-09）：因 Monorepo-Bench 饱和、OPQA 含无解题目，换成新套件（Internal Research Debugging、KernelGen 1P、NanoGPT、PostTrainBench Lite、MLE-Bench Revised）；低于 High。
- GPT-5.5（2026-04-23）：可以排除 High。

**Google DeepMind**

- [Gemini 4 Argon](https://storage.googleapis.com/deepmind-media/gemini/gemini_4_safety_report.pdf)（2026-10-13）：两个 ML R&D CCL 都未达到；结论是没有显著的端到端 AI 研发加速。AI agent 写了大部分生产代码和 bug 修复，但改动仍经人工审查。能力指数（GCI）的折线回归：2026-04-01 之前斜率为每月 0.091（0.071–0.108），之后为 0.118（0.053–0.171），差异不显著。
- [Gemini 3.7 Flash](https://storage.googleapis.com/deepmind-media/gemini/gemini_3-7_flash_fsf_report.pdf)（2026-08）：内部 benchmark pass@1 为 27%，排除阈值是 90%。
- [Gemini 3 Pro](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf)（2025-11）：RE-Bench 综合分远低于警戒阈值。

## 自报的加速数字：单位各不相同

| 来源（日期） | 数字 | 单位 | 同一来源给出的限定 |
| --- | --- | --- | --- |
| [OpenAI](https://openai.com/index/research-acceleration-view-inside-openai/)（2026-09-06，自报） | 3.1 | 2026 年 8 月中，agent 工作日／人类工作日 | 成功的 4–8 小时任务过半有人工介入；同期算力大幅增长；研究优先级仍由人定 |
| [Anthropic Institute](https://www.anthropic.com/institute/recursive-self-improvement)（约 2026-06，发布日期据二手来源；自报） | 超过 80% | 2026 年 5 月合并代码中由 Claude 写的比例 | 作者说代码行数这个指标几乎肯定高估了生产率 |
| [Anthropic 风险报告](https://www.anthropic.com/aug-2026-risk-report)（2026-08，自报） | 不到 2 倍 | 内部 AI 研发速度 | 任务评估已饱和，信心低于以往报告 |
| Anthropic Mythos Preview system card（2026-04-07，员工调查，自报） | 约 4 倍，折算后低于 2 倍 | 员工自估产出，折算成整体进展乘数 | 折算用了弹性估计 |
| [METR 转述](https://metr.org/blog/2026-09-22-claude-opus-5-5/)（2026-09-22） | 约 1.5 倍（约 30% 可能达 2 倍） | AI 带来的能力进展加速 | 出自 METR 一份未公开的初步报告；METR 看不到支撑证据，Anthropic 审阅并修改了摘要 |
| [METR 说明](https://metr.org/notes/2026-07-08-anthropic-researcher-uplift/)（2026-07-08） | 可能超过 2 倍 | 研究员 uplift，由 Anthropic 自报的每日合并代码 8 倍推算 | 研究员产出不等于整体进展速度 |
| Google DeepMind Gemini 4 Argon 报告（2026-10-13，自评） | 无显著端到端加速 | 能力指数斜率 | 断点前后的区间重叠 |

同一家实验室可以同时出现“代码大多由 AI 写”“研究员 uplift 可能超过 2 倍”和“整体还不到 2 倍”三种数字，它们并不矛盾，因为单位不同。OpenAI 自己也写道，AI 研究有很多潜在瓶颈，算力是另一道闸门，而且随着其他瓶颈缩小可能变得更重要。

## 经济学模型怎么看这个回路

Cunningham 等 2026-09-14 的论文 [The Economics of Recursive Self-Improvement](https://arxiv.org/abs/2609.15802) 认为，净加速取决于各条反馈回路弹性的乘积。§4.1 的校准是：只有当 AI 模型能力（用 Epoch Capabilities Index 衡量）提高一个单位，能让 AI 研发生产率至少提高 15% 时，加速才能自我维持。他们粗估自编程 agent 推出以来约为 9%，这个数来自自报的 Mythos Preview 约 4 倍 uplift，作者说它很可能是高估。按这个校准，弹性还要再提高约三分之二（$15/9\approx1.7$）才到门槛。论文的结论是：反馈回路目前还不足以产生自我维持的加速，但似乎在变强。

更早的 [Whitfill 与 Wu（2025-07）](https://arxiv.org/abs/2507.23181)用 2014–2024 年 OpenAI、DeepMind、Anthropic、DeepSeek 的数据，估计研究算力和认知劳动之间的替代程度，结论取决于模型设定：基线设定下两者可以互相替代；“前沿实验”设定下两者互补，意味着算力会卡住。

## 觉得快了，不等于测出来快了

METR 2025-07 的[随机对照实验](https://arxiv.org/abs/2507.09089)（2025 年初的工具）发现，允许使用 AI 让开发者完成任务的时间增加了 19%，而开发者事先预期会缩短 24%。METR 2026-02-24 的[后续](https://metr.org/blog/2026-02-24-uplift-update/)给出回访开发者“−18% 的 speedup”（CI −38% 到 +9%）和新招募者 −4%（CI −15% 到 +9%），两个区间都包含 0，文章没有重申正负号约定；METR 称这批数据是不可靠的信号，部分原因是开发者越来越不愿意在不用 AI 的条件下工作。METR 2026-05-19 的报告则把它的 RCT 概括为发现了小幅（约 4–20%）的生产率提升。Anthropic Institute 的作者也说，2026 年 3 月员工调查的约 4 倍中位数，真实提速大概会稍低。

教训是：自我感觉和有对照的测量可能差很多，甚至方向相反。问卷数字要和对照实验分开读。

## 读到一个新说法，先问这几件事

| 先问什么 | 为什么 | 本篇里的例子 |
| --- | --- | --- |
| 谁量的？外部能不能核对？ | 内部数据外部看不到 | METR 看不到约 1.5 倍估计的支撑证据 |
| 单位是什么？ | 代码份额、agent 工时、研究员 uplift、整体进展速度是不同的量 | 超过 80% 的代码，对比“还不到 2 倍” |
| 人介入了多少？ | 自主程度决定能不能往外推 | OpenAI：成功的 4–8 小时任务过半有人工介入 |
| 算力同时涨了多少？ | 进展可能来自算力，而不是 AI 劳动 | OpenAI 说同期算力大幅增长 |
| 测量工具饱和了吗？有没有作弊？ | 超出任务范围的读数是外推 | TH1.1 饱和；至少 16% 的长任务成功不合法 |
| 和什么比？断点怎么选？ | 趋势断点对选择很敏感 | Mythos Preview 的斜率比 1.86–4.3 倍，取决于断点 |

## 这条线接下来看什么

下面是开放问题，以及什么样的结果会改变现在的判断；不是预测。

1. **有没有更长的外部尺子？** METR 2026-05-19 说 TH1.1 已基本饱和，页面 2026-09-08 起不再更新；没有找到名为“Time Horizon 2.0”的 METR 发布。一套任务远超 30 小时、带作弊审查的新套件，会恢复一条外部可查的趋势线。
2. **自报数字能不能被外部核对？** OpenAI 2026 年 6 月的蓝图提议向 CAISI 报告 RSI 测量；METR 2026-09-22 说它看不到约 1.5 倍估计的证据。公开、经第三方核对的测量，会改变这些数字的证据地位。
3. **趋势断点会不会变显著？** Anthropic 对 Opus 5.5 拟合的斜率增幅是 1.53 倍（2026-09-22），门槛是 2 倍；Google DeepMind 的 GCI 折线回归（2026-10-13）前后差异不显著。统计上显著的断点，或者某家宣布越过正式阈值，都会改变判断。
4. **弹性估计。** Cunningham 等（2026-09-14）的约 9% 对比 15% 门槛，建立在自报的 uplift 上。用有对照的测量重新估计这个弹性，会让“回路还不够强”这个结论更稳，或者被推翻。

最后：[复习题](review.md)。
