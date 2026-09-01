# RLHF 的三个阶段，和后来发生了什么

**中文** · [English](rlhf-pipeline.en.md)

> 阅读时间：约 14 分钟 · 难度：必修 · 最近审阅：2026-08

<div class="lesson-recipe">
  <div><span>解决什么问题</span><strong>把「人更喜欢哪个回答」变成可优化的目标</strong></div>
  <div><span>前置知识</span><strong>SFT · 偏好数据 · next-token prediction</strong></div>
  <div><span>核心机制</span><strong>学一个奖励模型，再用 RL 去优化它</strong></div>
  <div><span>常见错误</span><strong>以为四个模型都在训练；以为 KL 项是可选的</strong></div>
</div>

## 为什么需要这么绕

想让模型「回答得更好」，最直接的想法是写个损失函数。但「更好」写不出来——它没有闭式表达，也没有标准答案可以对照。

人却能**比较**：给两个回答，说哪个更好。RLHF 的全部思路就是把这件事变成可优化的目标：

```text
人能比较 → 学一个模型来预测「人会更喜欢哪个」→ 拿这个模型当奖励函数做 RL
```

代价是引入了一层间接：你优化的不再是「人的偏好」，而是「一个模型对人偏好的拟合」。后面所有的麻烦都从这里来。

## 先把普通 RL 映射到语言模型

监督学习通常直接给出目标答案；强化学习（Reinforcement Learning, **RL**）只评价一段行为的结果。模型知道这次得了多少分，却未必知道是哪一步造成的，也没有一个逐步的标准答案可以照抄。

语言模型生成一句话，正好可以写成一段序列决策：

| RL 概念 | 在语言模型里是什么 |
| --- | --- |
| **Agent**（智能体） | 正在训练的语言模型 |
| **State**（状态）$s_t$ | prompt 加上已经生成的前缀 $(x, y_{<t})$ |
| **Action**（动作）$a_t$ | 下一枚 token $y_t$ |
| **Policy**（策略）$\pi_\theta(a_t\mid s_t)$ | 模型 softmax 给出的 next-token 分布 |
| **Trajectory**（轨迹）$\tau$ | 从回答开始到结束的整段 token 序列 |
| **Reward**（奖励）$r_t$ | Reward Model、verifier 或真实环境给出的标量反馈 |

这里的 Actor 不是包在语言模型外面的另一个决策器：**语言模型本身就是 policy**。在状态 $s_t=(x,y_{<t})$ 下从词表里采样下一枚 token，所用的概率就是

$$a_t=y_t\sim\pi_\theta(\cdot\mid x,y_{<t}).$$

token 被接到前缀后，便形成下一个状态。单纯文本生成时，这个状态转移几乎就是确定性的字符串拼接；到了工具调用或交互式 agent，环境还会返回搜索结果、执行结果或新的 observation。

### Reward、Return 和奖励归因

**Reward** $r_t$ 是某一步拿到的即时反馈；**Return** $G_t$ 是从这一步开始的累计未来奖励：

$$G_t=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots.$$

在经典偏好 RLHF 中，主要奖励经常等完整回答结束才由 Reward Model 给出。因此前面每一枚 token 都要共同为最后的分数负责：低分究竟是开头方向错了，还是中途出现事实错误？这就是 **credit assignment**（奖励归因）问题。实现里也常把逐 token 的 KL 惩罚当作较密集的 shaping reward，但它不等于人类偏好本身。

### Value、Q 和 Advantage

Critic 要学的不是「这个完整回答好不好」，而是当前前缀往后生成，预计还能拿到多少累计回报：

$$V^\pi(s)=\mathbb E_\pi[G_t\mid s_t=s].$$

如果还指定当前先选动作 $a$，则是 action value：

$$Q^\pi(s,a)=\mathbb E_\pi[G_t\mid s_t=s,a_t=a].$$

二者的差是 **Advantage**（优势）：

$$A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).$$

它问的不是「这次总分高不高」，而是「这个动作相对当前状态下的正常预期，好了多少」。同样拿到 $0.6$ 的回报：如果 Critic 原本预测 $0.8$，它低于预期；如果原本只预测 $0.2$，它就明显高于预期。减去这个 baseline 不会改变期望中的策略梯度，却能显著降低方差。

策略梯度的核心因此可以写成：

$$\nabla_\theta J(\theta)\approx\mathbb E\left[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\hat A_t\right].$$

$\hat A_t>0$ 时，提高这次采样动作的概率；$\hat A_t<0$ 时，降低它。这个公式只直接更新实际采样到的 token，并通过共享参数影响其他状态下的分布。Critic 则用 return 或 bootstrapped target 回归 $V_\phi(s_t)$；它的主要作用是**降低估计方差**，不是替 Actor 决定下一个 token。

### Reward Model 不是 Critic

这两个模型都输出标量，所以很容易混：

| | Reward Model | Critic / Value Model |
| --- | --- | --- |
| 输入 | prompt + 完整回答 | 当前 prompt + 生成前缀 |
| 输出 | 学到的偏好代理分数 | 从当前状态出发的预计 return |
| 回答的问题 | 「这个完成的回答看起来有多好？」 | 「从这里按当前策略继续，预计能拿多少分？」 |
| PPO 阶段 | 通常冻结 | 跟随当前 Actor 训练 |

Reward Model 给出的不是「真实人类满意度」，而是从有限偏好数据学到的 **proxy reward**（代理奖励）。它会判断错、偏爱表面风格，也可能被策略钻空子。Critic 学的则是当前 policy 下的条件期望；Actor 一变，它要估计的目标也会跟着变。

从完整回答粒度看，这套训练有一点像 contextual bandit：给一个 prompt，生成一个回答，最后拿一个整体分数。但 token 生成内部仍然是序列决策，状态会随前缀不断变化。两种说法只是抽象粒度不同。

## 三个阶段

**第一阶段 SFT。** 拿人写的示范数据微调预训练模型，得到一个至少会按指令格式回答的起点。它决定了后面 RL 的初始策略，起点太差 RL 也救不回来。

**第二阶段 训练奖励模型。** 收集**成对**的排序数据（同一个提示的两个回答，标注哪个更好），训练一个模型 $r_\phi$ 给回答打分。损失是 Bradley–Terry：

$$\mathcal{L}(\phi) = -\mathbb{E}_{(x, y_w, y_l)}\Big[\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\Big]$$

注意它只学**相对**关系。$r_\phi$ 的零点不可辨识：给所有分数加同一个常数，损失不会改变。因此原始分数不能直接解释成「人类满意度」，跨 prompt 或跨数据分布使用前也要检查校准；真正受训练目标直接约束的是同一 prompt 下 chosen 与 rejected 的分差。

**第三阶段 用 RL 优化。** 这一阶段有四个模型同时在场，而它们的角色完全不同。

## 第三阶段的四个模型，谁在训谁被冻住

这是最容易糊的地方：

| 模型 | 从哪来 | 训不训 | 干什么 |
| --- | --- | --- | --- |
| **Actor**（策略） | SFT 模型的副本 | **训练** | 真正要优化的那个，最后交付的就是它 |
| **Critic**（价值） | 常从奖励模型初始化 | **训练** | 估计状态价值 $V_t$，用来降低梯度方差 |
| **Reward**（奖励） | 第二阶段的产物 | 冻结 | 给完整回答打分 |
| **Reference**（参考） | SFT 模型的副本 | 冻结 | 提供 KL 基准，拴住 Actor 别跑偏 |

**只有前两个在更新参数。** Reward 和 Reference 全程只做前向。

Actor 和 Reference 一开始是同一份权重的两个副本——Actor 会被训练而慢慢偏离，Reference 停在原地当尺子。

这是四种**概念角色**，不保证工程上一定对应四个完全独立、始终常驻显存的大模型。Critic 可以是共享 backbone 上的 value head，冻结模型也可以分片或 offload；角色之间的训练关系不变。

## 为什么必须有 Reference

因为奖励模型是可以被钻空子的。它只是一个在有限偏好数据上拟合出来的函数，在训练分布之外的行为完全没有约束。策略如果放开了优化，会找到那些**奖励模型给高分、但人根本不认可**的回答——这就是 reward hacking。

所以真正被优化的目标里带一个 KL 惩罚：

$$r_{\text{total}}(x, y) = r_\phi(x, y) - \beta\,\mathrm{KL}\big(\pi_\theta(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\big)$$

意思是：你可以往奖励高的方向走，但不许离出发点太远。$\beta$ 就是这根绳子的松紧。

$\beta$ 太大，模型学不动，输出和 SFT 没区别；$\beta$ 太小，跑几百步之后开始输出人类看不懂但奖励模型很喜欢的东西。**这不是可选的正则项，是这套方法能不能用的前提。**

## 为什么还要一个 Critic

策略梯度需要知道「这个动作比平均水平好多少」，也就是优势 $A_t$。只用回报 $R_t$ 的话方差极大，训练会抖得没法用。Critic 学一个基线 $V_t$，用

$$A_t = R_t - V_t$$

来降方差。PPO 进一步用 GAE 做多步平滑，并把更新裁剪在信任域内：

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(\rho_t A_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)A_t\big)\Big], \qquad \rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

裁剪是为了防止一步走太远——策略一旦跑出旧策略的支撑集，重要性比 $\rho_t$ 就会爆炸。

这里有两把不同的尺子：**PPO clipping** 比较当前 Actor 与采样数据时的旧 Actor，限制一次优化更新；**Reference KL** 比较 Actor 与冻结的 SFT Reference，限制整个训练过程的累计漂移。前者不能替代后者。

## 后来发生了什么

四个模型、两个在训练，其中 Critic 还是个全尺寸网络。这个成本是后续所有工作要削的东西。

![各方法需要同时在显存里的模型数](assets/rlhf-model-count.svg)

**GRPO：把 Critic 去掉。** 对同一个提示采样一组回答，用组内的相对好坏当基线：

$$\hat A_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

既然只需要一个基线来降方差，那组内均值就够了，不必专门训一个网络去估它。省掉一个在训的全尺寸模型。

**DPO：连 RL 循环一起去掉。** 关键推导是：带 KL 约束的奖励最大化问题有闭式最优解，反解之后奖励可以用策略本身表示，于是偏好损失可以**直接对策略求**：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

不用训奖励模型，不用采样，不用 Critic。代价是它只能用**离线**的偏好对——策略在训练中改变了，数据却还是旧策略产生的，这个分布错配是 DPO 的主要局限。

**RLVR：奖励换成一个程序。** 数学题可以对答案，代码可以跑测试。这类任务的奖励不需要学，写一个检查器就行。学出来的奖励模型消失了，reward hacking 的空间也随之大幅收窄——**能被钻空子的是被拟合出来的奖励，不是被验证出来的**。

## 面试常见问题

<details class="interview" markdown="1">
<summary>RLHF 第三阶段有几个模型？分别训不训？</summary>

四个：Actor、Critic、Reward、Reference。

**只有 Actor 和 Critic 在更新参数。** Reward 是第二阶段训好后冻结的，Reference 是 SFT 模型的冻结副本。Actor 和 Reference 初始权重相同，训练中 Actor 逐渐偏离，Reference 停在原地作为 KL 的参照。

</details>

<details class="interview" markdown="1">
<summary>为什么需要 Reference model 和 KL 惩罚？去掉会怎样？</summary>

奖励模型只是在有限偏好数据上拟合的函数，训练分布之外没有任何约束。放开优化，策略会找到奖励模型给高分但人不认可的输出，也就是 reward hacking。典型表现是输出变长、堆砌讨好性措辞，或者干脆退化成某种重复模式。

KL 惩罚把策略拴在 SFT 附近。$\beta$ 太大学不动，太小会跑飞——它是这套方法的核心超参，不是可选正则。

</details>

<details class="interview" markdown="1">
<summary>Critic 是干什么的？GRPO 怎么把它省掉的？</summary>

Critic 估计状态价值 $V_t$，用来算优势 $A_t = R_t - V_t$。它的作用是**降方差**：直接用回报做策略梯度，方差大到训不动。

GRPO 的观察是：既然只是要一个基线，那对同一个提示采样一组回答、用组内均值和标准差归一化就够了，不必训练一个全尺寸网络来估。省掉的是一个正在训练的模型，显存和计算都是实打实的。

</details>

<details class="interview" markdown="1">
<summary>DPO 和 PPO 的本质区别是什么？</summary>

DPO 用了一个推导：带 KL 约束的奖励最大化有闭式最优解，把奖励反解成策略的表达式之后，偏好损失可以直接对策略求梯度。于是奖励模型和 RL 循环都不需要了。

代价是 DPO 用的是**离线**偏好对。训练中策略在变，数据却是旧策略采的，存在分布错配；PPO 每一轮都用当前策略重新采样，是 on-policy 的。所以 DPO 便宜很多，但在需要探索的任务上通常追不上。

</details>

<details class="interview" markdown="1">
<summary>奖励模型的分数可以直接比较大小吗？</summary>

Bradley–Terry 损失直接约束的是同一 prompt 下 chosen 与 rejected 的**分差**。给所有分数加一个常数，损失完全不变，所以零点没有可识别语义；一个 $2.4$ 不能直接读成「满意度 2.4」。

固定 Reward Model 的原始输出当然可以拿来计算，但跨 prompt、领域或模型版本比较时，必须先确认校准与尺度是否稳定。训练实现也可能做 reward whitening 或 normalization；那是优化选择，不是 Bradley–Terry 必然要求的规则。

</details>

<details class="interview" markdown="1">
<summary>为什么 RLVR 里 reward hacking 变少了？</summary>

因为奖励不再是被拟合出来的。数学题对答案、代码跑测试，检查器是写死的程序，没有「训练分布之外」这回事，也就没有可以钻的空子。

但它换来了另一个限制：只适用于结果可自动验证的任务。写作、对话这类没有检查器的场景，还是得回到学出来的奖励模型。而且验证器本身也可能被绕过——比如代码只过测试用例但逻辑是错的。

</details>

## 自检

<div class="taste-check">
  <strong>如果真的理解了，你应该能解释：</strong>
  <ol>
    <li>第三阶段四个模型分别从哪来，哪两个在更新参数？</li>
    <li>KL 惩罚去掉会发生什么？$\beta$ 调大调小分别是什么症状？</li>
    <li>Critic 的作用是降方差还是提高准确率？GRPO 用什么替代了它？</li>
    <li>为什么 DPO 不需要奖励模型？它因此付出了什么代价？</li>
  </ol>
</div>

## 继续阅读

- [Post-Training 总览](README.md)：SFT、偏好学习、RL 各自适合什么问题
- [数据与反馈](../01-data-and-feedback/)：偏好标签本身的质量问题
- [Evaluation](../07-evaluation/)：怎么判断对齐之后真的变好了

## 中文导读

- [大模型中的强化学习](https://zhuanlan.zhihu.com/p/693582342) — 知乎 @大家好我是爱因，专栏《机器学习小王子》。
  本章只讲 RLHF 这条主干，刻意没有铺开算法谱系。那一半在这篇里：从 MDP 的要素、
  贝尔曼方程、MC/TD/GAE 的偏差-方差权衡讲起，一路到 PPO 的四模型协同、DPO 与
  IPO/KTO，再到 GRPO、DAPO、Dr. GRPO、RLOO、REINFORCE++ 各自想解决什么问题。
  想看全景，从它开始。

## 参考论文

- [InstructGPT](https://arxiv.org/abs/2203.02155) — 三阶段流程的出处
- [PPO](https://arxiv.org/abs/1707.06347) — 裁剪目标与信任域
- [DPO](https://arxiv.org/abs/2305.18290) — 去掉奖励模型和 RL 循环
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — GRPO
- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) — KL 惩罚与 reward hacking 的早期实证
