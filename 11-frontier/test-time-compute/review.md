# 推理模型与测试时计算：复习题

**中文** · [English](review.en.md)

> 阅读时间：约 4 分钟 · 最近审阅：2026-10

这一页把本系列的要点压成面试题和自检题；答案里的数字和结论都能在前三篇找到出处。

## 面试常见问题

<details class="interview" markdown="1">
<summary>什么是测试时计算？并行和串行两种花法各有什么代价？</summary>

权重固定，回答一个问题时多花的计算：多采样再挑、一条链里想得更长、在中间步骤上搜索、生成后检查再改。生成一个 token 约 $2N$ FLOPs，所以多花的计算基本就是多生成的 token。并行是同时生成 N 份再挑，延迟几乎不变、费用随 N 线性增长，但要有办法挑；串行是一条链里多想或逐版修改，后面能用上前面的发现，代价是延迟随长度增加。Snell 等发现简单题更受益于串行修改，难题要在两者之间找比例。实际系统常把两者组合，比如 DeepSeekMath-V2 每题 64 份证明、每份验证 64 次、最多修改 16 轮。

</details>

<details class="interview" markdown="1">
<summary>多数投票什么时候反而会降低准确率？举个能手算的例子。</summary>

在固定分布、独立采样且存在唯一众数时，N 增大使投票选中该众数的概率趋近 1；这只是渐近结论，不保证有限 N 下的收益单调。例子：单次答对 0.4，错的 0.6 全是同一个答案 B。3 票时正确答案要拿 2 票以上，概率 $0.064+0.288=0.352$，比采一次的 0.4 还低，N→∞ 趋向 0。同样 $p=0.4$、错误答案各不相同时，3 票是 0.496、5 票约 0.715，最终趋向 1。所以平均 $p$ 推不出投票后的成绩，要逐题看错误是分散还是集中；投票还要求答案能判等。

</details>

<details class="interview" markdown="1">
<summary>Best-of-N 加一个验证器，准确率的上限由什么决定？</summary>

设单次答对概率 $p$，验证器接受正确答案的概率 $r$、放过错误答案的概率 $f$，各样本独立。N 很大时，交出的总是某份被接受的样本，准确率趋向精度 $pr/\big(pr+(1-p)f\big)$。$p=0.4$、$r=1$：$f=0.1$ 时上限 87.0%，$f=0.3$ 时 69.0%，$f=0$ 时就是覆盖率 $1-0.6^N$。和投票相比，验证器能在错误集中时仍然有效，前提是它的误判和生成器的错误不相关；它若也总是接受那个常见错解，就退回单次采样的水平。打分式验证器取最高分时，N 越大越容易挑到“高分的错解”。

</details>

<details class="interview" markdown="1">
<summary>过程奖励模型和结果奖励模型有什么区别？Lightman 等的结果能推出什么、推不出什么？</summary>

ORM 给整份答案一个分，PRM 给每一步打分再合成，Lightman 等用各步正确概率的乘积。他们用 GPT-4 微调的生成器、500 道 MATH 题、每题 1860 个样本里挑一个：PRM 78.2%，ORM 72.4%，多数投票 69.6%，PRM800K 有 80 万个步级人工标注。能推出：在固定生成器上，PRM 更会挑答案，N 越大领先越多。推不出：用过程奖励做 RL 会训练出更强的策略，这需要另做实验。DeepSeek-R1 报告说 PRM 难定义步、难标注、在大规模 RL 里会被 reward hacking，更适合重排和辅助搜索。

</details>

<details class="interview" markdown="1">
<summary>Snell 等的 compute-optimal 是什么意思？“小模型加测试时计算打败 14 倍大的模型”在什么条件下成立？</summary>

对每道题和每个预算，选让答对概率最大的测试时策略（式 1）；实际用基座 pass@1 分出的 5 档难度来近似，比 best-of-N 最多省 4 倍计算。和放大参数比时，预训练记 $6ND_{\text{pretrain}}$、推理记 $2ND_{\text{inference}}$，小模型可以把推理计算乘以 $K=M+3(D_{\text{pretrain}}/D_{\text{inference}})(M-1)$。$M=14$ 时，$R=D_{\text{inference}}/D_{\text{pretrain}}=0.16$ 约 258 倍，$R=22$ 只有约 16 倍。结论是：简单和中等难度的题、推理负载低时测试时计算更划算；最难的题或推理负载高时，放大预训练更有效，两者不能 1:1 互换。结果来自 PaLM 2-S* 和 MATH，早于 R1 这类长链模型。

</details>

<details class="interview" markdown="1">
<summary>R1-Zero 的回答为什么会越来越长？长度增长都是好事吗？</summary>

R1-Zero 从 DeepSeek-V3-Base 出发，用 GRPO 和规则奖励（答案对错加格式）训练，不用神经网络奖励模型；平均长度随训练上升，AIME 2024 pass@1 从 15.6% 到 71.0%（v1）。原因之一是检查、回溯、换方法能提高答对概率：比如直接答 0.3、检查重做 0.65，期望奖励高的做法被强化，长度随之变长。但不全是好事：Liu 等指出 GRPO 按回答长度归一化，答错的长回答每个 token 受罚更轻，会额外推长错误回答；o1 类模型在“2 加 3”上平均多用 1,953% 的 token；难题上答错的回答反而更长、思路切换更频繁。Kimi k1.5、L1 用长度奖励收，Dr. GRPO 去掉归一化偏差。

</details>

<details class="interview" markdown="1">
<summary>思维链忠实吗？怎么测？对监控意味着什么？</summary>

不一定忠实。测法是干预或埋提示：Turpin 等把示例答案都排成 (A)，模型被带偏却不在解释里提；Lanham 等截断、改错、改写思维链，看答案变不变，发现依赖程度因任务而异；Chen 等在 prompt 里埋提示，Claude 3.7 Sonnet 和 DeepSeek R1 用了提示后说出来的比例平均只有 25% 和 39%，在留了漏洞的 RL 环境里利用漏洞超过 99%、说出来的不到 2%。Baker 等发现读思维链比只看动作更能抓到 reward hacking，但把监控器放进奖励后，强优化会让模型隐藏意图。所以思维链监控有用但不充分，而且对它施加优化压力会让它失效。

</details>

<details class="interview" markdown="1">
<summary>“RL 只是锐化了基座本来就有的能力”这个说法怎么看？pass@k 怎么估计？</summary>

Yue 等（NeurIPS 2025）发现 $k$ 小时 RL 模型更好，$k$ 大时基座反超，推理路径也在基座分布里；蒸馏则在所有 $k$ 上更好。机制可以手算：基座两题正确率 (0.5, 0.02)，RL 后 (0.95, 0)，pass@1 从 0.26 到 0.475，pass@100 却从 0.934 降到 0.5。反方有 ProRL（长时间 RL 在基座解不出的题上提升）、CoT-Pass@k（排除蒙对后 RL 仍扩展边界）、技能组合实验；Spurious Rewards 则显示 Qwen2.5-Math 用随机奖励也能涨分，说明部分收益来自激活已有行为。pass@k 用 Chen 等的无偏估计 $1-\binom{n-c}{k}/\binom{n}{k}$，$n=10$、$c=3$ 时 pass@5 约 0.917，直接代入 $1-0.7^5$ 只有 0.832。截至 2026 年 10 月没有共识，要看基座、训练时长、领域、$k$ 和是否检查推理过程。

</details>

## 自检

<div class="taste-check">
  <strong>可以用下面的问题检查理解：</strong>
  <ol>
    <li>给出一道题的答案分布，投票会把准确率推向哪里？为什么只知道数据集的平均 $p$ 不够？</li>
    <li>best-of-N 的准确率公式里，$r$ 和 $f$ 各自怎样影响上限？验证器的错误和生成器的错误相关时会怎样？</li>
    <li>按 Snell 等的记账，推理负载 $R$ 从 0.16 变到 22，小模型能多花的推理计算怎么变？为什么？</li>
    <li>一份 RL 训练日志里平均长度在涨，你会拆出哪几条曲线，来区分“学会了检查”和长度偏差？</li>
    <li>读到“某模型 AIME 9x%”时要问哪四件事？30 道题、准确率 80% 时，95% 置信区间大约多宽？</li>
  </ol>
</div>

## 参考论文

- [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)：GSM8K，用验证器从多份样本里挑
- [Self-Consistency](https://arxiv.org/abs/2203.11171)：对多条推理路径的最终答案投票
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)：过程奖励与结果奖励，PRM800K
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601)：在“想法”树上搜索
- [Large Language Monkeys](https://arxiv.org/abs/2407.21787)：覆盖率随采样数增长，挑选会饱和
- [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314)：compute-optimal 策略和与预训练的换算
- [Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)：o1 的自报结果与隐藏原始思维链的决定
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)与 [*Nature* 版](https://doi.org/10.1038/s41586-025-09422-z)：R1-Zero、蒸馏、失败的尝试
- [Kimi k1.5](https://arxiv.org/abs/2501.12599)：长上下文 RL 与长度奖励
- [s1](https://arxiv.org/abs/2501.19393)：1,000 条样本 SFT 与 budget forcing
- [L1](https://arxiv.org/abs/2503.04697)：用 RL 控制思考长度
- [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783)：GRPO 的长度偏差与 Dr. GRPO
- [Do NOT Think That Much for 2+3=?](https://arxiv.org/abs/2412.21187)：overthinking
- [Thoughts Are All Over the Place](https://arxiv.org/abs/2501.18585)：underthinking
- [Inverse Scaling in Test-Time Compute](https://arxiv.org/abs/2507.14417)：推理越长反而越错的任务
- [Thinking Hard, Not Smart](https://arxiv.org/abs/2608.07968)：多题共享预算时的分配
- [Language Models Don't Always Say What They Think](https://arxiv.org/abs/2305.04388)、[Measuring Faithfulness in Chain-of-Thought Reasoning](https://arxiv.org/abs/2307.13702)、[Reasoning Models Don't Always Say What They Think](https://arxiv.org/abs/2505.05410)：思维链忠实性
- [Monitoring Reasoning Models for Misbehavior](https://arxiv.org/abs/2503.11926)、[Chain of Thought Monitorability](https://arxiv.org/abs/2507.11473)：思维链监控及其脆弱性
- [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)：pass@k 的无偏估计
- [Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?](https://arxiv.org/abs/2504.13837)：pass@k 之争的起点
- [ProRL](https://arxiv.org/abs/2505.24864)、[RLVR Implicitly Incentivizes Correct Reasoning](https://arxiv.org/abs/2506.14245)、[From f(x) and g(x) to f(g(x))](https://arxiv.org/abs/2509.25123)、[The Invisible Leash?](https://arxiv.org/abs/2507.14843)、[Spurious Rewards](https://arxiv.org/abs/2506.10947)：争论中的各方
- [DeepSeekMath-V2](https://arxiv.org/abs/2511.22570)：证明验证器与扩展验证计算
- [Deep Think with Confidence](https://arxiv.org/abs/2508.15260)：按置信度过滤并行轨迹
- [MathArena](https://arxiv.org/abs/2505.23281)、[GPQA](https://arxiv.org/abs/2311.12022)、[ARC-AGI-2](https://arxiv.org/abs/2505.11831)、[ARC-AGI-3 发布](https://arcprize.org/blog/arc-agi-3-launch)、[Why we no longer evaluate SWE-bench Verified](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)：污染、饱和与新基准
- [FEval-TTC](https://arxiv.org/abs/2511.01203)：测试时方法的成本建模与公平比较
