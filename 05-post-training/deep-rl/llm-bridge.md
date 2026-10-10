# 接到 LLM：哪些 RL 直觉能搬过去？

**中文** · [English](llm-bridge.en.md)

> 阅读时间：约 14 分钟 · 最近审阅：2026-10

让模型答一道题，和让它反复改代码、跑测试，训练起来有什么不同？两者都可以用 RL，但“一次动作”“任务结束”和“做得更好”未必是同一回事。

这篇先从 3 个 token 的更新讲起，再看任务变长以后，组内比较、Critic 和上下文压缩各会遇到什么问题。只想先弄懂取舍，可以看[组内奖励的小例子](#sparse-groups)和[长任务怎么选](#long-horizon-choice)；想算一遍，再看 [3 步 GAE](#credit-example)。

## 先写一张对应表

| RL 概念 | 放到文本生成里 | 容易混淆的地方 |
| --- | --- | --- |
| 状态 / 观察（state / observation） | 提示文字与当前上下文 | 文本不一定包含完整环境状态 |
| 动作（action） | 一个 token，或一次工具调用 | 两种粒度的时长和成本不同 |
| 策略（policy） | 自回归模型 | 要记录实际采样分布的 log-prob |
| 奖励（reward） | 偏好、测试、任务结果 | 能优化的分数未必代表用户目标 |
| 一轮任务（episode） | 一次回答或多轮任务 | EOS、超时、工具失败不是同一种结束 |

单段生成中，拼接 token 的转移近乎确定，但采样与奖励不一定确定。工具 agent 的下一观察还来自外部世界，可能延迟、有权限限制或发生失败。

## PPO 与 GRPO 各估计了什么

PPO 用采样策略收集的轨迹估计 advantage，再通过概率比值和 clipping，抑制某些方向上过大的更新收益；它不把比值硬锁在区间内。参考模型 KL 惩罚与 PPO clipping 也不同：前者惩罚偏离固定基准的行为，后者调整这批数据上的局部优化目标。

GRPO 常用同一 prompt 下多个回答的相对奖励构造组内 advantage，减少对独立 value Critic 的依赖。若组内奖励全相同，标准化后的相对信号可能为零；分组与归一化还会影响 prompt 的相对权重。它不自动解决稀疏奖励、错误 verifier 或跨任务公平比较。

如果 advantage 和概率比还不熟悉，可以先看 [PPO clipping](../rlhf/ppo-clipping.md)。想比较 PPO、GRPO、DPO 用什么数据，再看[方法对照](../rlhf/after-rlhf.md)。

## 用三个 token 看清训练到底在改什么

假设 prompt 后生成三个 token，最后一个是 EOS，完整回答得到奖励 1。生成过程可以静态地拆成：

<div class="drl-flow drl-sequence" aria-label="文本生成与奖励归因">
<span>Prompt<small>不是本轮动作</small></span><b>→</b><span>Token 1、2<small>每步保存 log-prob</small></span><b>→</b><span>EOS<small>完整答案得到评分</small></span><b>→</b><span>更新<small>奖励归因到有效动作</small></span>
</div>

假设三个已采样 token 的条件概率，旧策略是 $[0.5,0.25,0.5]$，新策略是 $[0.6,0.2,0.6]$。逐 token 比值为 $[1.2,0.8,1.2]$，整段概率比却是它们的乘积 1.152。Token-level PPO 不能把这两者混用；长序列乘积的尺度和方差都不同。

如果只在最后给一个分数，前面的动作也会得到由它构造的学习信号；这不是声称每一步推理都被独立验证。过程奖励可以增加中间反馈，但它自己也可能有偏差，不能因为分数更密就叫它更准确。

## GRPO 的组内信号，什么时候会消失？

取同一 prompt 的四个回答，奖励 $[0,0,1,1]$。用总体标准差约定，均值 0.5、标准差 0.5，标准化优势为 $[-1,-1,1,1]$。高于本组平均的回答被鼓励，低于平均的被压低。

若四个回答都是 0，减均值后全部为 0；分母加 $\epsilon$ 可以避免除零，却不能制造任务奖励梯度。四个都是 1 也一样。这时 KL 等其他项仍可能产生梯度，不能说整个训练完全不更新。不同实现的方差约定、是否除标准差、按 token 还是按序列平均，也会改变更新。

### 一组采 8 次，能得到多少有用差异？ {#sparse-groups}

先做一个很小的假设：对同一道题，每次独立尝试的成功概率都是 $p$，成功得 1，失败得 0。一组采 $G$ 次，只有“至少一次成功，也至少一次失败”时，才有非零的组内奖励差异。

$$
\begin{gathered}
P_{\mathrm{mix}}=1-p^G-(1-p)^G.\\
G=8,\quad p=0.02\quad\Rightarrow\quad 14.9\%.
\end{gathered}
$$

这里 $P_{\mathrm{mix}}$ 表示一组里同时有成功和失败的概率。怎么来的？从全部情况里，减去“全成功”和“全失败”就行。下面固定采 8 次，只改变这道题的成功率。

<figure class="worked-update" lang="zh-CN" id="group-signal-comparison">
<figcaption>同样采 8 次，难度不同，能比较的机会也不同。数值来自独立采样的二元奖励假设。</figcaption>
<ol>
<li><small>成功率 2%</small><strong>14.9% 的组有差异</strong><span>多数时候 8 次都失败。继续采样有成本，却不一定得到新的相对信号。</span></li>
<li><small>成功率 20%</small><strong>83.2% 的组有差异</strong><span>更容易同时看到成功和失败，组内比较才有东西可比。</span></li>
<li><small>成功率 98%</small><strong>14.9% 的组有差异</strong><span>这次是多数时候全成功。信号少，不一定意味着模型不会做。</span></li>
</ol>
</figure>

**这不是 GRPO 的效果预测。** 真实任务的成功率不同，采样也可能相关，奖励不一定只有 0 和 1。这里算的是出现奖励差异的概率，不是梯度质量，更不是训练后的准确率。它提醒我们：先看全零组、全一组各占多少，再决定增加组大小、调整题目难度，还是增加可靠的中间反馈。

<details markdown="1">
<summary>用 Python 复算；不需要模型或 GPU</summary>

```python
def mixed_group_probability(success_probability, group_size):
    if not 0 <= success_probability <= 1:
        raise ValueError("success_probability must be in [0, 1]")
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
        raise ValueError("group_size must be a positive integer")
    return 1 - success_probability ** group_size - (1 - success_probability) ** group_size

for success_probability in [0.02, 0.2, 0.98]:
    probability = mixed_group_probability(success_probability, 8)
    print(f"{success_probability:.0%}: {probability:.1%}")
```

</details>

| 选择 | 节省或获得什么 | 付出什么 |
| --- | --- | --- |
| 学 Value Critic | 跨状态估计回报，支持时序归因 | 多一条估值训练路径，可能有误差 |
| 组内相对奖励 | 不必单独训练同样的 Value Critic | 同题多次生成，组内无差异时信号弱 |
| 程序验证奖励 | 对可执行约束提供直接反馈 | 只覆盖验证器能检查的部分 |

这些不是同一个维度的互斥选项。可验证奖励可以用于不同策略优化算法，GRPO 也不能修复一个把错误答案判对的 verifier。

## Old policy、reference model、reward model 别混

在最简单的同步 PPO 中，Old policy 就是采这批动作时的策略，保存的 log-prob 在本批更新期间不变。异步训练就不能直接这样假定：真正产生动作的行为策略可能更旧，采样温度和推理后端也可能让实际概率与训练端不同。需要分别记录行为概率和更新所用的旧策略基准，例子见[训练数据从哪个策略来](../post-training-infrastructure.md)。

Reference model 是另一个角色：它通常较稳定，用来计算 KL 等约束，不是“标准答案”，也不必等于 Old policy。Reward model 或 verifier 则负责评分。模型都叫 model，不代表这些工作可以混着算。

保存 prompt / response 边界、EOS、padding、采样温度和策略版本，才能知道哪些位置是动作、概率比对应谁。对工具 agent，工具返回的文字是观察，不是模型采样的 token；不能因为拼进同一序列，就把它也当 Actor 的行为来训练。

这里的算例解释共同机制，不声称所有 GRPO / PPO 实现采用完全相同的 loss。读具体实现时，可接着比较 [DeepSeekMath 的 GRPO 设计](https://arxiv.org/abs/2402.03300) 与站内 [RLHF 的模型分工](../rlhf/three-stages.md)。

## 长任务：memory 改变的是什么

先区分两个容易放在一起的词。**长上下文（long context）**问的是一次能读多少；**长任务（long horizon）**问的是要连续做多少次决策，以及多久才能知道结果。读完一份很长的报告就回答，可以很长但不需要多轮交互；每次只看几条日志、反复排错几十轮，则可能上下文不大，但决策链很长。若把每个 token 都算一步，两者又会相关，所以比较实验时要写清“步”指 token、工具调用，还是交互轮次。

历史越积越多，迟早要选择保留什么。例如，摘要只写“测试失败”，却漏掉“已经试过方案 A，错误出在输入边界”，下一轮就可能重复走老路。这里缺的不是更多奖励，而是决策需要的信息。记忆可以帮忙，但不能保证摘要已经包含完整环境状态，也不会自动满足 Markov 假设。

同样是停下来，任务失败、服务超时、主动取消和预算耗尽也不同。先约定这些情况怎样记录、是否继续、怎么评分，再去增加 rollout 并发，不然跑得更快也可能只是在更快地收错数据。

## 长轨迹一定要从 GRPO 换成 PPO 吗

<span id="long-horizon-choice"></span>

不一定。两次尝试都在修同一个 bug，一次 4 步修好，一次 10 步仍失败，完整结果仍可以比较。**GRPO 不要求第 5 步对齐，也不要求两条轨迹一样长。** 麻烦在于：一次完整尝试已经很贵，再对同一任务采一组，是否划算？只看最后的成败，能否提供足够的学习信号？

[GLM-5.2 的官方说明](https://z.ai/blog/glm-5.2)提供了一个具体选择：该团队在长任务训练中采用 Critic-based PPO，以处理压缩后数量、长度不同的训练子轨迹，并使用 token-level loss。这是 2026-06-16 公布的实现选择，不是“业界证明 GRPO 已经过时”，也不是 PPO 一定更好的对照实验。

还要区分完整轨迹和训练片段。一次尝试拆成 3 段、另一次拆成 6 段，不代表把这 9 段当成同一 prompt 下的 9 次独立作答就合理；它们起点、历史和后续回报都不同。但如果完整任务身份与最终奖励还在，也不能因此断言组内方法在数学上失效。

### 最后成功了，前面每一步都该鼓励吗？ {#credit-example}

假设一个修复任务经过 3 次决策，最后测试通过，奖励依次为 $[0,0,1]$。只使用最终结果的 GRPO，会给这条轨迹中的动作同一个组相对 advantage。这不等于各 token 梯度相同，只是它们共用一个“这次尝试比本组平均好多少”的信号。

Critic 想多估计一件事：**已经走到这里，接下来大约能拿多少回报？** 以下估值是手造的，不来自实际模型。为便于手算，取 $\gamma=1$、$\lambda=0.5$，结束后的价值为 0。

<figure class="worked-update" lang="zh-CN" id="credit-assignment-example">
<figcaption>先算第 3 步，再把结果带回第 2、1 步。同一条成功轨迹里，局部估计也可以不同。</figcaption>
<ol>
<li><small>第 1 步 · 先试一处修改</small><strong>δ₁ = 0 + 0.3 − 0.9 = −0.6</strong><span>Â₁ = −0.6 + 0.5 × 0.6 = −0.3。下一状态的估值比原先低。</span></li>
<li><small>第 2 步 · 根据测试重新定位</small><strong>δ₂ = 0 + 0.8 − 0.3 = 0.5</strong><span>Â₂ = 0.5 + 0.5 × 0.2 = 0.6。后面一步的信号也传了回来。</span></li>
<li><small>第 3 步 · 改对并通过测试</small><strong>δ₃ = 1 + 0 − 0.8 = 0.2</strong><span>Â₃ = 0.2。任务确实结束，不再接未来价值。</span></li>
</ol>
</figure>

别急着把负数理解为“第一步确实有害”。也可能是最初 0.9 的估值太乐观。Critic 给的是统计估计，不是每步的因果贡献证明。把同一例子的 $\lambda$ 改成 1，完整回报减去各步估值，advantage 就变成 $[0.1,0.7,0.2]$，第一步又是正的。估计方法本身也会改变信号。

<details markdown="1">
<summary>公式与代码：为什么要从后往前算？</summary>

令 $d_t$ 表示任务在这一步真正结束，$V_t$ 是当前状态的价值估计。先用 $B_t$ 表示要接上的未来价值；若任务终止，它就是 0。再算 TD residual：

$$
\begin{gathered}
B_t=(1-d_t)V_{t+1},\\
\delta_t=r_t+\gamma B_t-V_t.
\end{gathered}
$$

连续轨迹内的 GAE 递推是：

$$
\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}.
$$

后一条从片段末尾向前算；真正终止、重置或不可接续的片段边界会截断递推。较小的 $\lambda$ 更依赖局部估值；完整终止轨迹上，$\lambda=1$ 会得到折扣实际回报减去 baseline。更多推导见 [GAE 原论文 §3](https://arxiv.org/html/1506.02438v6#S3)和[站内实现](actor-critic-gae.md)。

下面复用 [rl_checks.py](code/rl_checks.py) 中的 `gae`。在仓库的 `05-post-training/deep-rl/code/` 目录运行；两种语言的代码一致。

```python
from rl_checks import gae

rewards = [0, 0, 1]
values = [0.9, 0.3, 0.8]
next_values = [0.3, 0.8, 0]
terminated = [False, False, True]
truncated = [False, False, False]

for trace_decay in [0.5, 1.0]:
    advantages = gae(rewards, values, next_values, terminated, truncated,
                     gamma=1.0, trace_decay=trace_decay)
    print(trace_decay, [round(advantage, 3) for advantage in advantages])
```

这里每一步代表一次决策，实际语言模型通常还要展开到 token，处理 action mask、KL 和 loss reduction；这段代码不是完整 trainer。

</details>

以上比较针对只用最终奖励的设计。[DeepSeekMath §4.1.3](https://arxiv.org/html/2402.03300v3#S4.SS1.SSS3)也讨论了过程奖励版 GRPO，所以不能说“没有 Critic 就永远只能看到最终成败”。中间反馈是否可靠，仍要单独检验。

### 压缩了上下文，不等于任务结束 {#segment-boundary}

假设片段最后一步奖励为 0，当前估值 0.6，接续状态估值 0.8，取 $\gamma=0.9$。如果任务还会继续，一步目标是 $0+0.9\times0.8=0.72$，TD residual 为 0.12；若误标成终止，目标变成 0，residual 变成 −0.6。同一条数据，仅仅改错一个边界标志，就从鼓励变成了压低。

因此要保存实际用来继续任务的上下文、环境状态和下一状态估值；不能拿重置后新任务的状态来续接。摘要只是 observation 的改变，不自动意味着终止。是否能跨片段继续 GAE，还取决于片段是否真的连续、数据如何对齐；不是把文件拼在一起就行。终止与外部截断的区别见 [Gymnasium 的说明](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)。

切片还会改变权重。假设两个任务分别贡献 2 个和 6 个有效动作位置，每个位置的 loss 分别为 1 和 3。先对任务取均值再平均，结果是 2；对全部 8 个位置取均值，结果是 2.5。后者给长任务更多权重，不是某种自动消除长度偏差的修复。无论选哪种，都应在切片、padding 和多卡聚合后保留原定的分母。

### 怎么比较这两种设计，才不会比错？ {#compare-designs}

| 现象 | 能考虑的设计 | 新的风险 |
| --- | --- | --- |
| 同题能收集多份完整结果 | 组内 outcome advantage | 奖励全同、rollout 很贵、归因粗 |
| 轨迹很长，需要中间估值 | Critic + TD / GAE | 估值偏差、额外训练与显存 |
| 训练片段结束，任务没结束 | 用下一状态价值续接，或等完整回报再处理 | 不能把片段边界误标为失败终态 |
| 历史被压缩 | 保存压缩后的 observation 与策略版本 | 关键信息可能被摘要丢掉 |

我会先固定任务集、起始模型、工具权限、验证器和测试时的采样预算，再比较两种训练方法。不要只固定 update 次数：GRPO 每题采 8 次、另一种方法采 1 次，生成成本已经不同；增加 Critic 又会增加训练成本。至少同时报告完整 rollout 数、生成 token、工具耗时和训练资源，而不是只报 loss。

接着看两类证据。训练侧，组内奖励是否经常全同、Critic 对未参与拟合的完整回报预测得怎样、切片后权重是否变化？任务侧，同等测试预算下，成功率是否提高，是否更容易从工具失败中恢复，是否只会在熟悉的仓库或模板上做题？题目和 verifier 有问题时，换估计器救不了它们。

最后单独检查作弊：代码任务不能因为 agent 改了测试、读了受保护答案，就给它成功奖励。GLM-5.2 的公开说明也把反作弊列为长任务训练的一部分。本站例子只验证计算，没有训练真实模型，也不声称哪一种方案必胜。

## 多个 agent：环境不再固定

其他 agent 也在学习时，你的状态转移与奖励分布会随它们改变。Centralized training / decentralized execution 可以在训练时给 Critic 更多联合信息，执行时各 Actor 只用允许的局部信息。

这只是多智能体入口，不是完整 MARL 课程。通信、博弈目标、非平稳性和协作归因需要另开篇章。参考：[MADDPG](https://arxiv.org/abs/1706.02275)。

## Reward 上涨后，还缺什么证据

检查独立任务成功率、长度与成本、未见场景、工具失败恢复，以及 reward hacking。训练时用同一个 judge 提供奖励，评估又只看它的分数，不能视作独立证据。

接下来可以读[评估栈](../../07-evaluation/evaluation-stack.md)，把任务结果、失败情况和成本分开记录；需要模型评分时，再看 [LLM-as-a-Judge](../../07-evaluation/llm-as-a-judge/README.md)。Reward 涨了值得高兴，但还得看看模型到底把什么做得更好了。
