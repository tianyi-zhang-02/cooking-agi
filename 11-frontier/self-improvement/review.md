# 递归自我改进（RSI）：复习题

**中文** · [English](review.en.md)

> 阅读时间：约 4 分钟 · 最近审阅：2026-10

## 面试常见问题

<details class="interview" markdown="1">
<summary>“递归自我改进”在不同场合指什么？怎么快速分辨？</summary>

先问三件事：改的是什么对象，谁提出并执行改动，用什么信号接受改动。按这三个问题可以分出四类：(a) 模型自己设计并执行训练、改进自己的权重；(b) 系统改自己的 scaffold、代码或 prompt，权重冻结；(c) AI 加速人主导的研发；(d) 用自产数据或 self-play 训练，但更新规则是人写的。(a) 和 (d) 都改权重，区别在谁设计更新。公开的“自我改进”系统大多是 (b) 或 (d)；截至 2026-10-15，没有找到 (a) 的完整公开演示。

</details>

<details class="interview" markdown="1">
<summary>STaR 这类“筛选再微调”和 RL 是什么关系？</summary>

STaR 把它看成对 0/1 奖励目标做 policy gradient 的近似：答错的样本梯度为零，所以“只留答对的再做 SFT”就近似一次策略梯度更新，筛选规则就是奖励。ReST-EM 把它写成 EM：M 步是在上一轮模型的样本上做按奖励加权的最大似然。和 PPO 这类在线 RL 不同，采样策略在这一步固定，数据收集和优化是分开的。推论是：验证器定义了被优化的东西，测试写成什么样，模型就往什么方向学。

</details>

<details class="interview" markdown="1">
<summary>为什么用自己的数据训练，几轮就饱和？</summary>

Song 等定义了 generation–verification gap：按验证器重新加权后真实得分能涨多少。若更新能把重新加权的分布学到 $\varepsilon$ 以内，一轮的提升大约就是 gap 加减 $\varepsilon$。他们测到迭代时 gap 两三轮内降到接近 0，小模型在多数验证方式下 gap 不为正，大 $k$ 的 pass@k 还会下降。ReST-EM 两轮以后收益递减，Self-Rewarding 只跑了 3 轮、作者预计会饱和。这些结果限制的是相应验证信号和更新方式下的持续收益，不能推广成“没有外部新信息就完全无法学习”。

</details>

<details class="interview" markdown="1">
<summary>模型真实正确率 40%，测试放过 10% 的错解。按测试过滤再微调，真实正确率最多到多少？</summary>

这是教学假设。通过测试的占 $0.40+0.60\times0.10=0.46$，其中正确的是 $0.40/0.46\approx87\%$。假设微调能完美模仿这批解，真实正确率到 87%，测试却显示 100%。再转一轮，测试对“正确”和“错但能过”一视同仁，gap 为 0，回路停在 87%。错解放过率是 30% 时上限是 69%，降到 2% 时上限升到约 97%。要提高上限，得加强验证器，比如加隐藏测试和边界用例。

</details>

<details class="interview" markdown="1">
<summary>Absolute Zero 的 learnability reward 为什么对“总能解出”和“从不能解出”的题都给 0？</summary>

这两类题在该组内二值奖励比较下都没有成功与失败的对比信号，不代表其他训练目标无法利用它们。出题者的奖励是：平均成功率 $\bar r$ 为 0 时给 0，否则给 $1-\bar r$。$\bar r=0.5$ 得 0.5，$\bar r=1$ 得 0，$\bar r=0.25$ 得 0.75，所以奖励最高的是偶尔能解出的题；做 $G$ 次尝试时最多是 $1-1/G$。“Zero”指不用人工整理的题和答案，预训练模型和 Python 执行器仍然必需。

</details>

<details class="interview" markdown="1">
<summary>AlphaEvolve 和 Darwin Gödel Machine 算“改进自己”吗？和“模型改自己的权重”差在哪？</summary>

两者改的都是代码，LLM 权重不动。AlphaEvolve 按人写的评估函数进化程序；它找到的 kernel tiling 让 Gemini 训练时间减少 1%，其中包括训练 AlphaEvolve 背后的模型，作者也说这条反馈周期以月计。DGM 让 coding agent 改自己的代码仓库，用 archive 保留所有还能工作的 agent，80 轮后 SWE-bench 从 20.0% 到 50.0%，但作者写明受底座 FM 能力限制，把“改写训练脚本更新 FM”留作未来工作。要成为 (a) 类，改的对象得是权重，更新由模型设计，接受信号还得是被改系统碰不到的。

</details>

<details class="interview" markdown="1">
<summary>METR 的 50% time horizon 怎么算？为什么超过 16 小时的读数不可靠？</summary>

对每个模型，把成功率对任务的人类用时（取对数）做 logistic 拟合，成功率等于 50% 处的人类用时就是 horizon。若所有任务组的成功率都过半，50% 的点只能沿曲线外推到最长任务之外，一次成功的差别就能让估计变几倍；METR 的套件里没有超过 30 小时的任务。METR 2026-05-19 还说 TH1.1 已基本饱和，8 小时以上任务至少 16% 的成功不合法；校正任务时长的噪声可能让 horizon 再降约 30%。

</details>

<details class="interview" markdown="1">
<summary>明天读到一个新的“AI 在加速 AI”的说法，你会怎么评估？</summary>

先归类：它是任务级测量、实验室对照阈值的判定、实验室自报数字，还是表态预测。再问五件事：谁量的、外部能否核对；单位是代码份额、agent 工时、研究员 uplift，还是整体进展速度；人介入了多少；同期算力涨了多少；测量工具是否饱和、有没有作弊审查，以及趋势断点选在哪里。最后和现有锚点对照：截至 2026 年 10 月，凡是给出整体进展倍数的估计都在 2 倍以下（Anthropic 自报、METR 转述约 1.5 倍），Google DeepMind 2026-10-13 没有测到显著的端到端加速，没有实验室宣布越过正式阈值。还要记得 METR 的对照实验里，开发者自我感觉和测量结果方向都可能相反。

</details>

## 自检

<div class="taste-check">
  <strong>可以用下面的问题检查理解：</strong>
  <ol>
    <li>给你一个“自我改进”系统，怎样说出它改的对象、谁提出改动、用什么信号接受，并判断它属于 (a)–(d) 哪一类？</li>
    <li>为什么 STaR 的筛选规则相当于 0/1 奖励？把“答对”换成“通过测试”后，回路实际在优化什么？</li>
    <li>错解放过率从 10% 降到 2%，过滤再微调的上限怎么变？为什么理想情况下第二轮不再涨？</li>
    <li>DGM 的父代选择里，得分和子代数各起什么作用？HGM 为什么认为只看分数不够？</li>
    <li>当模型在所有任务组上的成功率都过半，50% time horizon 是怎么得到的？这个数可信到什么程度？</li>
  </ol>
</div>

## 参考论文

- [Speculations Concerning the First Ultraintelligent Machine](https://doi.org/10.1016/S0065-2458%2808%2960418-0)：I. J. Good，intelligence explosion 的出处
- [Gödel Machines](https://arxiv.org/abs/cs/0309048)：以形式证明为门槛的自我改写
- [Levels of Organization in General Intelligence](https://intelligence.org/files/LOGI.pdf)：seed AI 的定义
- [When AI builds itself](https://www.anthropic.com/institute/recursive-self-improvement)：Anthropic Institute 2026 年的定义与自报数字
- [Recursive Self-Improvement in AI（综述）](https://arxiv.org/abs/2607.07663)：1,250 篇论文与 verification hierarchy
- [STaR](https://arxiv.org/abs/2203.14465)：用自己答对的推理过程微调
- [ReST-EM](https://arxiv.org/abs/2312.06585)：把自训练写成 EM
- [Expert Iteration](https://arxiv.org/abs/1705.08439)：搜索当专家、网络当学徒
- [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020)：模型给自己打分再做 DPO
- [Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798)：没有外部反馈的自我纠错
- [Absolute Zero](https://arxiv.org/abs/2505.03335)：自己出题、执行器验证、learnability reward
- [R-Zero](https://arxiv.org/abs/2508.05004)：Challenger 与 Solver 共同演化
- [Mind the Gap](https://arxiv.org/abs/2412.02674)：generation–verification gap 与两三轮饱和
- [AI models collapse when trained on recursively generated data](https://www.nature.com/articles/s41586-024-07566-y)：model collapse
- [AlphaEvolve 白皮书](https://arxiv.org/abs/2506.13131)与[一周年文章](https://deepmind.google/blog/alphaevolve-impact/)：进化代码与其限制
- [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954)：archive、父代选择、objective hacking
- [SICA](https://arxiv.org/abs/2504.15228)：最好的 agent 改自己的代码
- [Huxley-Gödel Machine](https://arxiv.org/abs/2510.21614)：metaproductivity–performance mismatch
- [AIDE²](https://arxiv.org/abs/2609.26457)：研究 agent 改写自己的代码与 ignition test
- [PostTrainBench](https://arxiv.org/abs/2603.08640)：agent 后训练 base LLM
- [Measuring AI Ability to Complete Long Software Tasks](https://arxiv.org/abs/2503.14499)、[Time Horizon 1.1](https://metr.org/blog/2026-1-29-time-horizon-1-1/)、[Frontier Risk Report](https://metr.org/blog/2026-05-19-frontier-risk-report/)：METR 的 time horizon 及其局限
- [RE-Bench](https://arxiv.org/abs/2411.15114)、[MLE-bench](https://arxiv.org/abs/2410.07095)、[PaperBench](https://arxiv.org/abs/2504.01848)：AI 研发 benchmark
- [OpenAI Preparedness Framework v2](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf)、[Anthropic RSP v3.4](https://www-cdn.anthropic.com/files/4zrzovbb/website/0bacdc8440ea96e62a8766d99ebe1d4eea6d5f3a.pdf)、[Google DeepMind FSF v3.0](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/strengthening-our-frontier-safety-framework/frontier-safety-framework_3.pdf)：各实验室的阈值
- [Research acceleration: The view inside OpenAI](https://openai.com/index/research-acceleration-view-inside-openai/)、[Anthropic 2026 年 8 月风险报告](https://www.anthropic.com/aug-2026-risk-report)、[Gemini 4 Argon 安全报告](https://storage.googleapis.com/deepmind-media/gemini/gemini_4_safety_report.pdf)：三家对自身研发加速的报告
- [The Economics of Recursive Self-Improvement](https://arxiv.org/abs/2609.15802)：反馈回路弹性的校准
- [METR 开发者随机对照实验](https://arxiv.org/abs/2507.09089)：感觉的提速与测量的提速
