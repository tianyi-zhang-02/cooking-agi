# 深度强化学习 {#deep-rl}

**中文** · [English](README.en.md)

> 阅读时间：约 8–10 分钟 · 最近审阅：2026-11

眼前有 2 分可以拿，多等一步可能拿到 4 分。该怎么选？强化学习（reinforcement learning，RL）要考虑眼前的奖励，也要考虑一次选择对后续的影响。这组笔记从小环境开始，逐步讲到价值估计、策略更新，以及怎样把它们写成训练循环。

<nav class="study-route reading-entry" aria-label="强化学习阅读入口">
<a href="mdp-bellman.md"><strong>第一次学 RL</strong><span>从状态、动作和奖励开始，手算一次更新。</span></a>
<a href="#reward-review"><strong>复习与面试</strong><span>按具体问题，回看计算、推导和实现。</span></a>
<a href="experiments.md"><strong>训练出了问题</strong><span>检查 target、mask 和梯度，再看曲线。</span></a>
</nav>

每篇先用例子把问题讲清楚，再看公式和代码。第一次读可以跟着例子走，复习时再用表格对照算法、检查实现。算例都是自拟的小环境，不依赖课程作业，也不转载课件。

算法名字容易混在一起，先看[Q-learning、PPO 与 DPO 的区别](methods-map.md)。想知道为什么现在仍难训，再看[信号、成本与稳定性](why-rl-is-hard.md)；想按来路梳理，可以打开[技术树里的 RL 编年史](../../learn/rl-timeline.md)。

## 按主题找章节 {#_1}

| 路线 | 读什么 | 读完应该能做什么 |
| --- | --- | --- |
| 先把问题写对 | [MDP 与 Bellman](mdp-bellman.md) → [MC、TD 与终止条件](returns-and-td.md) | 从一段轨迹算出 return 和训练 target |
| 直接改策略 | [Policy Gradient](policy-gradients.md) → [Actor–Critic 与 GAE](actor-critic-gae.md) → [Natural Gradient 与 TRPO](trust-region.md) | 解释梯度方向、baseline、两个 mask，以及步长为什么需要约束 |
| 通过价值选动作 | [Fitted Q 与收敛边界](fitted-q.md) → [DQN](dqn.md) → [DDPG / TD3](continuous-control.md) → [SAC](soft-actor-critic.md) | 分清回归 loss 与策略表现，画出各个网络的梯度路径 |
| 用模型想几步 | [Model-based RL](model-based.md) → [学什么样的模型](model-types.md) → [规划与 LQR](planning-and-control.md) → [规划与学习怎样配合](planning-and-learning.md) | 分清学习动力学、规划动作和训练策略，说出模型该预测什么、搜索结果怎样变成训练标签 |
| 数据与奖励从哪里来 | [Offline RL 与 OPE](offline-and-ope.md) → [逐步权重、双重稳健与 LSTD](ope-estimators.md) · [探索与层级](exploration-and-hierarchy.md) · [模仿与奖励](imitation-and-rewards.md) | 说出各自增加的假设，而不只是记名字 |
| 用模型扩充离线训练数据 | [Offline RL 与 OPE](offline-and-ope.md) → [离线模型式 RL](offline-models.md) | 手算奖励惩罚、HALT 与保守 Critic，分清预测轨迹和真实证据 |
| 下一次该试什么 | [探索与层级](exploration-and-hierarchy.md) → [探索与不确定性](exploration-uncertainty.md) | 手算 UCB、后验和信息增益，区分新颖、随机与值得探索 |
| 从示范里学习 | [模仿与奖励](imitation-and-rewards.md) → [MaxEnt IRL](maxent-irl.md) → [GAIL](gail.md) · [奖励排序训练](../reward-ranking.md) | 拟合特征期望、计算判别器奖励，区分成对偏好与完整排序 |
| 真正跑一个实验 | [实验与排错](experiments.md) → [接到 LLM](llm-bridge.md) | 检查实现，再判断 reward 是否代表任务进步 |

这些章节不只介绍算法名字。我们会一起算 Bellman 残差界、TD target、梯度方向、PPO 裁剪和控制轨迹，再看 CQL、IQL 与 FQE 如何处理离线数据。模仿学习这一组继续用算例讲奖励拟合与分布匹配；层级和多智能体目前只作引入，还不是完整的理论综述。

接到语言模型训练时，可以先读 [RLOO 的留一基线](../rloo.md)，再动手[核对一批回答怎样变成 loss](../../practice/post-training/rl-batch-audit.md)。前一篇补基线的证明和手算，后一篇检查分组、token 移位、EOS 与损失分母；不需要先搭多卡训练。

奖励本身也值得单独拆开：[奖励模型在预测什么](../reward-signals.md)用一道小算术题区分答案正确、步骤正确和后续成功概率，再算标签 mask 与分数聚合。如果正在复习推理时搜索，可以接着算[同一预算怎样分给生成与验证](../../11-frontier/test-time-compute/parallel-and-verify.md#budget-example)。

## 复习：六个计算与实现问题 {#reward-review}

已经学过策略梯度，想准备技术讨论或排查训练代码，可以从这张表开始。每行都连到具体的推导或算例，不必为找一个公式重读整个章节。若 baseline、return 和 advantage 还分不清，先读上面的基础路线。

| 问题 | 回答时要说清什么 | 推导与算例 |
| --- | --- | --- |
| 同题奖励是 1、0、0.5，留一优势是多少？为什么要排除自己？ | 优势是 0.75、−0.75、0。无偏性用到给定题目后的独立采样，以及 score function 的期望为 0；不是“减任何均值都一样” | [RLOO 的计算与证明](../rloo.md#why-leave-one-out) |
| 专家走过的路线奖励越高，IRL 就学得越好吗？ | 还要与其他可行路线比较，不能丢掉配分函数。三路线例子中，梯度是 $1.6-10/7$；它匹配特征期望，不要求拟合每条路线的频率 | [MaxEnt IRL 的梯度](maxent-irl.md#gradient) |
| GAIL 的判别准确率到了 50%，就模仿成功了吗？ | 可能是分布接近，也可能是判别器没学会。还要看独立任务表现；理想匹配时本文约定的每步奖励为 $\log2$，不是 0 | [判别器与分布匹配](gail.md#js) |
| 三个回答的完整排序，能直接当成三个独立偏好对吗？ | 可以选择 pairwise loss，但它不等于完整排序的 PL loss。还要说明按题还是按 pair 平均，避免候选多的题无意中占更大权重 | [排序与偏好对的差别](../reward-ranking.md#pairs-versus-ranking) |
| EOS 和 PAD 共用 ID，能把这个 ID 都 mask 掉吗？ | 不能。实际生成的 EOS 是动作，补齐长度的 PAD 不是；用真实生成边界构造 mask，并把 logits 与下一个 token 对齐 | [移位与 EOS 检查](../../practice/post-training/rl-batch-audit.md#alignment) |
| 两条回答的逐 token 损失为 $[2,4]$ 和 $[1,1,1,1]$，为什么三种 mean 会不同？ | 序列求和后平均为 5，逐回答平均为 2，有效 token 平均为 $5/3$。分母改变了样本权重；拆 microbatch 后也应保留同一目标 | [损失归约的手算表](../../practice/post-training/rl-batch-audit.md#reduction) |

算出数字后，可以再换一个条件：同题采样不独立、候选数不同、轨迹长度不固定，或者奖励服务超时。原来的计算还成立吗？如果不成立，要改的是哪一步？

## 用反例检查理解 {#assumption-review}

下面这些问题不需要记新的算法名字。先猜答案，再用小环境或数字检查自己漏了哪个条件。

| 问题 | 用哪个例子检查 |
| --- | --- |
| 策略不随时间变，访问到的状态分布也不变吗？ | [L、R 交替的链](mdp-bellman.md#discounted-occupancy)：平稳分布是各半，折扣访问分布却是 $2/3$、$1/3$ |
| baseline 等于平均回报，梯度方差就最小吗？ | [两个动作的 bandit](policy-gradients.md#variance-optimal-baseline)：平均回报 2.6，本例的最小方差基线却是 1.4 |
| Q-learning 能用旧数据，多步回报也可以直接用吗？ | [两步 replay](dqn.md#multistep-off-policy)：目标价值 6.4，未校正的两步 target 均值只有 2.35 |
| 模型预测很准，沿它的梯度优化就可靠吗？ | [预测相同、导数相反](model-based.md#model-jacobian-check)；再看[零方差但有偏的梯度](planning-and-control.md#pathwise-gradient-conditions) |
| 日志里最好的一整段只有 3 分，离线策略最多也只能拿 3 分吗？ | [拼接两段兼容的经验](offline-and-ope.md#stitching-and-trajectory-support)：在给定的确定性 MDP 中能拿 5 分，但轨迹 IS 可能恰好没有一条非零权重 |
| 保守价值学习，会把每个 Q 值都压低吗？ | [COMBO 的两项回归](offline-models.md#combo-example)：A 从 2 升到约 2.5714，B 从 5 降到约 3.6667；方向取决于采样分布 |
| 很难预测的结果，就值得多探索吗？ | [探测器与随机硬币](exploration-uncertainty.md#information-gain)：前者提供约 0.2781 bit，后者对隐藏状态的信息为 0 |
| 一个前缀后来更容易答对，说明前缀本身变正确了吗？ | [换续写策略](../reward-signals.md#prefix-value)：前缀没变，样本成功率却从 1/4 变成 3/4 |
| 奖励模型排对了更多偏好对，最后选出的回答一定更好吗？ | [五份回答的反例](../reward-signals.md#ranking-versus-selection)：90% 的排序准确率仍可能把唯一合格答案排到第二 |
| 固定一道题采了 256 次都失败，就能说模型不会吗？ | [零次成功与发现预算](why-rl-is-hard.md#finite-exploration-evidence)：真实成功率 0.5% 时，零次成功仍有约 27.7% 的概率 |

每个例子都保留了适用条件。复习时不只报出反例的数字，也说说换成随机环境、不同策略或更少数据以后，原来的计算还能不能用。

## 例子、推导和代码怎么配合 {#_2}

| 今天想读到哪一步 | 可以怎么读 | 应该留下什么 |
| --- | --- | --- |
| 先弄懂直觉 | 顺着开头的任务、图和手算表走，暂时跳过推导 | 能说出算法为什么需要这个部件 |
| 想把原理弄扎实 | 回到公式，逐个确认条件期望、符号和假设 | 能算一个例子，也能指出保证何时不成立 |
| 想动手实现 | 看数据来源、target、梯度路径、mask 和更新时机 | 用小环境查错，而不是只盯训练曲线 |

关键算例都在正文里，不用操作交互图也能读完。想试不同参数时，再用图里的控件。流程图的短暂动画只是提示顺序，停下后内容仍然完整；开启系统的“减少动态效果”也不影响阅读。

## 看一个算法，先找这三步 {#3}

<div class="drl-flow drl-sequence" aria-label="强化学习训练循环">
<span>采样<br><small>谁产生轨迹？</small></span><b>→</b><span>估计<br><small>用什么 target？</small></span><b>→</b><span>更新<br><small>梯度流向谁？</small></span>
</div>

| 方法 | 主要学什么 | 动作怎么来 | 容易忽略的代价 |
| --- | --- | --- | --- |
| REINFORCE | policy | 从策略分布采样 | 回报方差大，旧数据不能直接混用 |
| DQN | Q-value | 离散动作中取最大值，训练时加探索 | bootstrap、函数逼近和 off-policy 叠加不稳定 |
| 状态价值 Actor–Critic | policy + V | 从策略分布采样 | Critic 的误差会影响 Actor |
| TD3 / SAC | policy + 两个 Q | 连续动作 policy | 重用数据省交互，不代表省计算 |
| Model-based | dynamics，可能再学 policy / value | 规划或 policy | 模型预测准，不等于规划出的行为可靠 |

On-policy / off-policy 比较的是**收集数据的行为策略，与正在评价或改进的目标策略是否一致**；online / offline 说的是**训练中能不能继续和环境交互**。所以，边交互边用旧数据训练，可以同时是 online 和 off-policy。

## 先补哪几块基础 {#_3}

能读条件期望、链式法则和梯度就可以开始。需要补基础时，可以先看[ML 数学](../../00-foundations/ml-math-interview.md)里的概率与梯度；PyTorch 部分需要知道自动求导和张量形状。

统一记号：$s_t$ 是状态，$a_t$ 是动作，$r_t$ 是执行该动作后收到的奖励，$\gamma$ 是折扣。有限时域写 $t=0,\ldots,T-1$，终止后的 value 为 0。概率策略用 $\pi_\theta$，value 参数用 $\phi$。不要把 reward、return 和 value 当成同一个数。

后面会反复遇到这些词，先用一句话认清它们：

| 术语 | 在这里是什么意思 |
| --- | --- |
| 策略（policy） | 给定状态，决定选哪个动作或怎样分配动作概率 |
| 奖励（reward） | 执行一步后收到的反馈，不是整段表现 |
| 回报（return） | 从某一步起，把后续奖励按约定累加起来 |
| 价值（value） | 在某个策略下，对未来回报的期望 |
| 训练目标值（target） | 这轮拿来拟合的数；可能只是估计，并非真值 |
| 自举（bootstrapping） | 用已有估值，补上还没观察到的未来 |
| 轨迹（trajectory / rollout） | 与环境交互得到的一串状态、动作和奖励；rollout 也可能只是一段 |

## 练习：每次改一个条件 {#_4}

读完算例，试着只改一件事：把终止换成超时、把 $\lambda$ 从 0 调到 1、让两个 Q-network 的排序不同。网页里的小实验就是为这个准备的。它们演示计算机制，不模拟一场真实训练，也不展示虚构的性能提升。

想检查公式有没有落实到代码，运行[配套小程序](code/rl_checks.py)；想看 stop-gradient 是否正确，再运行 [PyTorch 检查](code/torch_updates.py)。两者都不需要 GPU。

## 可以动手试的图解 {#_5}

| 想弄明白 | 去哪一篇 | 自己改什么 |
| --- | --- | --- |
| 奖励怎么一步步传回起点 | [MDP 与 Bellman](mdp-bellman.md) | 调折扣，逐轮更新 |
| 未来哪几步影响这次 advantage | [Actor–Critic 与 GAE](actor-critic-gae.md) | 调 λ，展开每步贡献表 |
| 回归变好为何还会发散 | [Fitted Q](fitted-q.md) | 逐轮拟合，对照 value 曲线与 MSE |
| 一次策略更新到底改了多少 | [Natural Gradient 与 TRPO](trust-region.md) | 调 logit 步长和 KL 预算 |
| 模型预测错了，反馈有什么用 | [Model-based RL](model-based.md) | 调模型误差和规划长度，对照两条真实轨迹 |

调完参数，先猜一下结果为什么变了，再展开计算表核对。曲线往哪边走不用背，要看懂的是这个改变怎样一步步传下去。

## 公开参考 {#_6}

推导与术语可对照 [Spinning Up：RL 基础](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)；各章另列原始论文。PPO 与 RLHF 已有独立系列，读完这里再接过去，不必在两个地方重复背定义。想看这些方法用到多步 agent 和世界模型上，接着读前沿里的[用强化学习训练 agent](../../11-frontier/agentic-rl/README.md)和[世界模型](../../11-frontier/world-models/README.md)。
