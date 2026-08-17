# RLHF 的三个阶段，和后来发生了什么

**中文** · [English](rlhf-pipeline.en.md)

> 阅读时间：约 9 分钟 · 难度：必修 · 最近审阅：2026-08

<div class="lesson-recipe">
  <div><span>解决什么问题</span><strong>把「人更喜欢哪个回答」变成可优化的目标</strong></div>
  <div><span>前置知识</span><strong>SFT · 偏好数据 · 策略梯度</strong></div>
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

## 三个阶段

**第一阶段 SFT。** 拿人写的示范数据微调预训练模型，得到一个至少会按指令格式回答的起点。它决定了后面 RL 的初始策略，起点太差 RL 也救不回来。

**第二阶段 训练奖励模型。** 收集**成对**的排序数据（同一个提示的两个回答，标注哪个更好），训练一个模型 $r_\phi$ 给回答打分。损失是 Bradley–Terry：

$$\mathcal{L}(\phi) = -\mathbb{E}_{(x, y_w, y_l)}\Big[\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\Big]$$

注意它只学**相对**关系。$r_\phi$ 的绝对数值没有意义，加个常数不改变任何东西——这也是为什么奖励模型的分数不能跨批次直接比较。

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

## 面试可能会问

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

同一批内可以比，跨批次不行。Bradley–Terry 损失只约束**差值**，给所有分数加一个常数损失完全不变，所以绝对数值没有意义。

实践中通常会对奖励做批内归一化，否则 KL 系数 $\beta$ 的效果会随批次漂移。

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

## 继续读

- [Post-Training 总览](README.md)：SFT、偏好学习、RL 各自适合什么问题
- [数据与反馈](../01-data-and-feedback/)：偏好标签本身的质量问题
- [Evaluation](../07-evaluation/)：怎么判断对齐之后真的变好了

## 起始论文

- [InstructGPT](https://arxiv.org/abs/2203.02155) — 三阶段流程的出处
- [PPO](https://arxiv.org/abs/1707.06347) — 裁剪目标与信任域
- [DPO](https://arxiv.org/abs/2305.18290) — 去掉奖励模型和 RL 循环
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — GRPO
- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) — KL 惩罚与 reward hacking 的早期实证
