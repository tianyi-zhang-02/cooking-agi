# 技术编年史：RL 与后训练（post-training）

**中文** · [English](rl-timeline.en.md)

[← 返回技术树](technology-timeline.md#branch-rl)

> 阅读时间：约 9–12 分钟 · 最近审阅：2026-10

为什么已经研究几十年的 RL，今天还会让人觉得难训？沿着它解决过的问题看一遍，就不难理解：我们从估计几个动作的价值，走到了让大模型生成很长的轨迹；学习目标、反馈来源和系统成本都在变。

这一页先整理 **RL 到 LLM 后训练（post-training）**的一条技术线，不冒充完整 AI 史。年份按原始工作首次发表或公开的时间标注；后续修订不改写首发年份。每个节点回答三件事：当时遇到什么问题、方法改了哪里、今天还留下什么取舍。详细公式放在相应章节，不要求先把整条时间线背下来。

<nav class="timeline-jumps" aria-label="按阶段浏览技术编年史">
<a href="#learning">1989–2015 · 学会决策</a>
<a href="#feedback">2017–2024 · 学习反馈</a>
<a href="#scale">2025–2026 · 成本与稳定性</a>
</nav>

## 先学会估值，再学会直接改策略 {#learning}

<div class="tech-timeline" markdown="1">

### 1989 / 1992 · Q-learning：不用先知道环境怎么转移 {#q-learning}

如果转移规律已知，可以用动态规划计算价值；只拿得到实际经历时，就需要从一次次转移中更新估值。Q-learning 把当前奖励和下一状态的最大估值接在一起。1989 是 Watkins 提出该方法的时间，1992 是 Watkins 与 Dayan 的收敛论文，不是同一件事。

表格情形的条件不能原样搬到大网络上。今天看到 bootstrap、off-policy 和估值发散，仍然是在面对这条路线的难点。[原论文](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf) · [算一次 Q 更新](../05-post-training/deep-rl/methods-map.md#q-update)

### 1992 · REINFORCE：直接提高高回报动作的概率 {#reinforce}

另一条路是不先选出最大的 Q，而是直接调整随机策略。Williams 的 REINFORCE 工作给出了沿期望奖励梯度调整参数的方法。它不是后来语言模型训练才出现的想法。

可以直接改概率，不代表一次采样就能看准方向。理解 baseline 和方差，才能理解后来的 Actor–Critic。[原论文](https://gwern.net/doc/reinforcement-learning/model-free/1992-williams.pdf) · [策略梯度](../05-post-training/deep-rl/policy-gradients.md)

### 2013 · Deep Q-learning：输入可以是像素，不再是一张小表 {#dqn}

Atari 的画面没法逐个状态填表。早期 DQN 用神经网络估计动作价值，并用经验回放抽取过去的转移，缓解相邻样本相关、数据分布变化的问题。这里选取的是 2013 年报告，不把后续版本的所有设计都归到这一年。

更强的表示能力扩大了能做的任务，也让估值和训练稳定性更复杂。今天读 DQN，值得看 replay、target 和数据分布，不只是记一个游戏成绩。[原论文 §2、§4](https://arxiv.org/html/1312.5602) · [DQN 与 Double DQN](../05-post-training/deep-rl/dqn.md)

### 2015 · TRPO：一步改太多，旧数据的判断就不可靠了 {#trpo}

用旧策略的数据判断新策略，前提之一是两者不能离得太远。TRPO 从策略改进分析出发，构造带信赖域约束的实用算法；实用版包含近似，并不等于无条件的单调改进保证。

这条线后来接到 PPO。今天的 KL、概率比和更新幅度讨论，仍和这个问题有关。[原论文 §3–5](https://arxiv.org/html/1502.05477) · [Natural Gradient 与 TRPO](../05-post-training/deep-rl/trust-region.md)

</div>

## 不只有算法，还要弄清谁来判断好坏 {#feedback}

<div class="tech-timeline" markdown="1">

### 2017 · 人类偏好：奖励也可以从比较中学 {#human-preferences}

有些行为很难写成奖励函数，但人能判断两个尝试哪个更好。Christiano 等人的工作用轨迹片段比较学习 reward，再用 RL 优化策略；实验包括 Atari 和模拟机器人，而不是聊天模型。

它说明“人类反馈 + RL”早于今天的 LLM。学到的评分器仍只是近似，模型可能利用它的遗漏。[原论文 §2.2](https://arxiv.org/html/1706.03741) · [模仿与奖励](../05-post-training/deep-rl/imitation-and-rewards.md)

### 2017 · PPO：让策略更新更容易实现 {#ppo}

PPO 研究便于一阶优化的策略更新目标，clipped surrogate 是其中广泛使用的一种。它简化了部分更新机制，不负责解决所有奖励设计和采样问题。

PPO 比 TRPO 更方便用，但这不意味着裁剪会硬性限制实际 KL。今天仍需要检查 advantage 和更新幅度。[原论文 §3](https://arxiv.org/pdf/1707.06347) · [PPO clipping](../05-post-training/rlhf/ppo-clipping.md)

### 2022 · InstructGPT：把反馈流程接到指令模型 {#instructgpt}

语言模型会续写，不代表按用户需求回答。InstructGPT 将示范微调、偏好奖励模型和 PPO 接成一套指令训练流程。它沿用了更早的反馈学习工作，不是 RLHF 的起点。

算法之外，谁来标注、题目来自哪里、如何判断可靠，都直接影响模型行为。[原论文 §3](https://arxiv.org/html/2203.02155v1) · [三个阶段各做什么](../05-post-training/rlhf/three-stages.md)

### 2023 · DPO：有偏好对，能否少搭一套在线训练循环？ {#dpo}

DPO 把特定 KL 正则化奖励目标与偏好模型联系起来，得到直接优化策略的成对 loss。普通离线设置不需要在训练循环里持续 rollout。

这是另一种偏好学习路线，不是“更高版本的 PPO”；省掉的环节不等于消除了数据偏差。[原论文 §3–4](https://arxiv.org/html/2305.18290v3) · [DPO 算例](../05-post-training/deep-rl/methods-map.md#dpo-pair)

### 2024 · GRPO：用一组回答代替单独训练的 value baseline {#grpo}

DeepSeekMath 用同一问题下多次回答的奖励估计组内相对好坏，减少对独立 value model 的需要。GRPO 早于 DeepSeek-R1。

省下 Critic 的一部分开销，代价是组采样和信号质量。全对、全错以及相关性很高的回答，仍要分别处理。[原论文 §4.1](https://arxiv.org/html/2402.03300v3) · [组内信号](../05-post-training/deep-rl/llm-bridge.md#sparse-groups)

</div>

## 规模上来以后，成本和稳定性更难分开 {#scale}

<div class="tech-timeline" markdown="1">

### 2025 · DeepSeek-R1：把可验证反馈用于推理训练 {#r1}

R1-Zero 展示了在其基座与任务设置下、不先做 SFT 的 RL 训练；完整 R1 则还包含冷启动数据和多个训练阶段。两者不能混成一句“只靠 RL”。首发是 2025 年，这里链接的是后续 v2 报告。

可验证的数学和代码反馈很有用，但不能直接推出开放任务都能照搬。比较结果还要固定推理时的采样和长度预算。[报告 §2–3](https://arxiv.org/html/2501.12948v2) · [LLM 与 Agent 中的 RL](../05-post-training/deep-rl/llm-bridge.md) · [长思维链（long CoT）怎样训练出来](../11-frontier/test-time-compute/long-chains.md)

### 2025 · DAPO、AReaL、GSPO：分别改采样、流水线和更新 {#engineering}

这些不是一个算法的三次改名：[DAPO](https://arxiv.org/html/2503.14476v2)处理采样和优化细节；[AReaL](https://arxiv.org/html/2505.24298v2)研究异步生成与训练；[GSPO](https://arxiv.org/html/2507.18071v1)调整序列级策略更新。年份相同，不等于在解决同一个问题。

比较时先问：省了生成、计算还是等待？改变了哪个训练目标或数据分布？各篇的收益有各自的实验条件。[成本与稳定性拆解](../05-post-training/deep-rl/why-rl-is-hard.md)

### 2026 · 当前核对点：别把工程限制当作已经解决 {#current-checkpoint}

这是本站截至 2026-10-11 的复核节点，不是一篇叫“2026 RL”的新算法。按 [verl 的当前文档](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html)，训练与推理之间的分布差异仍要单独处理；按 [AReaL 的异步说明](https://areal-ai.io/docs/en/algorithms/async.html)，策略陈旧度仍要加以控制。

这提醒我们：有了可用框架，不等于每个任务都有可靠配方。后续更新会注明新工作实际改了什么，不只给旧方法换一个新年份。[实验怎么排查](../05-post-training/deep-rl/why-rl-is-hard.md#diagnosis)

</div>

## 换个环境，RL 的难点就变了 {#domains}

游戏、机器人和语言模型都可以写成“观察 → 动作 → 反馈”，但这不代表训练配方能直接搬过去。

| 场景 | 一步动作 | 反馈与采样 | 特别需要小心 |
| --- | --- | --- | --- |
| Atari 等游戏 | 选择按键 | 模拟器给分，可以反复开局 | 一局的分数怎样分给前面的动作；是否只记住训练环境 |
| 机器人控制 | 关节力矩、速度或动作序列 | 仿真或真机，可能要人工复位 | 安全、采样成本，以及仿真与现实的差别 |
| LLM 后训练 | token，或更高层的工具调用 | 通常生成完再打分，工具也可能提供中间反馈 | 长轨迹、奖励漏洞、生成成本，以及离线评估与实际使用的差别 |

### 机器人：数据重用很有价值，失败也真的有成本 {#robot-control}

想象训练机械臂把杯子放进托盘。失败不只是得到一个低分：杯子要重新放好，设备需要检查，下一次尝试才能开始。这解释了为什么机器人 RL 很关心 **sample efficiency**，而不仅是每秒能跑多少次梯度更新。

2018 年的 [SAC](https://arxiv.org/html/1801.01290) 将 off-policy 数据重用与最大熵目标结合，面向连续控制。旧轨迹能继续参与训练，但不代表任何旧数据都同样可靠。状态覆盖、价值估计误差和环境变化仍然重要。模拟器可以降低采样成本，不能自动消除 sim-to-real 的差距。

到了 2023 年，[RT-2](https://arxiv.org/abs/2307.15818) 展示了另一条连接：把视觉语言知识和机器人示范接到动作预测上。它是 [VLA](../13-embodied/robot-learning/vla.md) 路线的例子，不能因为输出了动作就算作 RL。实际系统可以结合示范学习、规划、RL 与安全控制；每一部分处理的问题不同。

继续读：[连续控制](../05-post-training/deep-rl/continuous-control.md) → [SAC 的目标](../05-post-training/deep-rl/soft-actor-critic.md) → [规划与控制](../05-post-training/deep-rl/planning-and-control.md)。

### 语言模型：先有一个会生成的起点，再决定怎样反馈 {#llm-post-training}

假设模型已经能写排序函数。现在想让它更可靠：可以给正确代码做 SFT，可以收集两个回答的偏好对做 DPO，也可以让当前模型生成代码、跑测试，再做在线 RL。任务相似，**数据来自哪里、何时生成、如何更新**却不同。

这条线连接了 2017 年的[人类偏好学习](#human-preferences)、2022 年的 [InstructGPT](#instructgpt)、2023 年的 [DPO](#dpo)，以及后来面向推理的可验证反馈。DPO 是相关的偏好优化分支，不是“RL 必须经过的下一代”。同样，可验证奖励也不等于只能用 GRPO。

LLM 的动作可以是离散 token，成本却一点都不“小”：同一问题采样八条长回答，每条都需要自回归生成；最后才得到一个分数时，如何把信号分给前面的选择仍然困难。代码测试能检查一部分行为，但通过有限测试不等于对所有输入都正确。工具调用还会引入执行成本与安全边界，多步任务怎样训练见[用 RL 训练 agent](../11-frontier/agentic-rl/README.md)。

继续读：[Q-learning、PPO 与 DPO](../05-post-training/deep-rl/methods-map.md) → [LLM / Agent 的 RL](../05-post-training/deep-rl/llm-bridge.md) → [rollout 成本与稳定性](../05-post-training/deep-rl/why-rl-is-hard.md)。

## 年份看完，再按问题回到正文 {#read-next}

| 你真正想问的 | 继续读 |
| --- | --- |
| Q-learning 和 DPO 为什么不该直接比“谁更先进”？ | [方法地图](../05-post-training/deep-rl/methods-map.md) |
| 为什么生成很多，模型却学得不多？ | [训练信号与 rollout 成本](../05-post-training/deep-rl/why-rl-is-hard.md) |
| 这里没有展开的连续控制、离线 RL 呢？ | [SAC](../05-post-training/deep-rl/soft-actor-critic.md) · [Offline RL](../05-post-training/deep-rl/offline-and-ope.md) · [离线策略评估（OPE）估计量](../05-post-training/deep-rl/ope-estimators.md) |
| 算法背后的数据和评估怎么设计？ | [数据作为监督](../01-data-and-feedback/data-as-supervision.md) · [评估栈](../07-evaluation/evaluation-stack.md) |

这条时间线只选了能帮助理解当前后训练设计的节点；动态规划、控制、探索、模型式 RL（[学什么样的模型](../05-post-training/deep-rl/model-types.md)、[规划与学习](../05-post-training/deep-rl/planning-and-learning.md)）、多智能体等都有更长的历史。一个方向没放进来，不代表它不重要。读新方法时，最值得追问的仍是：它解决了哪一个旧问题，代价又转移到了哪里？
