# Reference 与 Critic：一个管偏离，一个估回报

**中文** · [English](reference-and-critic.en.md)

> 阅读时间：约 8–10 分钟 · 最近审阅：2026-10

同一个回答，奖励模型打了 2 分，Critic 估了 1 分，Reference 又给出一串 log-probability。它们都输出数字，但回答的是三个不同的问题。先把问题分开，公式就没那么绕了。

这一篇用典型的 PPO-style RLHF 作例子，不把它当成所有后训练方法必须照搬的模板。想先算一遍，可以直接看[两种回答的小例子](#kl-example)。

## 1. 先分清五个角色，不要急着数模型

| 角色 | 它回答什么 | 典型 PPO-style RLHF 中怎样更新 |
| --- | --- | --- |
| Actor $\pi_\theta$ | 现在该生成什么？ | 用策略损失更新 |
| Rollout policy $\pi_{\rm old}$ | 这批回答当时是怎么采出来的？ | 一批数据的行为概率固定；下一轮换新版本 |
| Reference $\pi_{\rm ref}$ | 相对选定的参照，行为偏了多少？ | 通常冻结，常从 SFT 模型初始化 |
| Reward Model $r_\phi$ | 这个完整回答符合学到的偏好吗？ | 在典型 PPO 阶段冻结 |
| Critic $V_\psi$ | 从这个前缀继续，预期还会拿多少回报？ | 拟合当前策略的回报 |

这是五种**角色**，不等于显存里必须同时放五份完整权重。旧策略可以只留下已采样 token 的 log-probability；Actor 和 Critic 也可能共享部分参数。具体资源要看实现，不要靠数方框估显存。三阶段流程的经典实例见 [InstructGPT](https://arxiv.org/abs/2203.02155)。

比如，你问模型“为什么训练集表现很好，测试集却很差？”后训练可能让 Actor 更容易给出清楚、准确的解释。Reference 仍保留参照时的参数，给**同一个问题下的这些回答**计算概率，让我们知道生成倾向改了多少。它也可能偏爱一个不准确的回答，所以 **Reference model 不是标准答案（reference answer），也不是负责判对错的老师**。

<span id="reference"></span>

## 2. Reference KL：给高奖励加一个“偏离代价”

固定 prompt $x$，让 $y$ 表示完整回答。一个常见目标是：

$$
J(\theta)=\mathbb E_x\left[\mathbb E_{y\sim\pi_\theta(\cdot|x)}r_\phi(x,y)
-\beta D_{\rm KL}\big(\pi_\theta(\cdot|x)\|\pi_{\rm ref}(\cdot|x)\big)\right].
$$

这里 KL 是**两个回答分布**的差异，不是某一条回答自己的属性。它让模型在追求奖励时，也为偏离参照付出代价。它没有告诉模型什么一定正确，更不能保证奖励模型不会被钻空子。

展开定义后，KL 可以写成采样期望：

$$
D_{\rm KL}(\pi_\theta\|\pi_{\rm ref})
=\mathbb E_{y\sim\pi_\theta}\left[\log\frac{\pi_\theta(y|x)}{\pi_{\rm ref}(y|x)}\right].
$$

所以可以给一次采样构造惩罚后的回报：

$$
\widetilde r(x,y)=r_\phi(x,y)-\beta\log\frac{\pi_\theta(y|x)}{\pi_{\rm ref}(y|x)}.
$$

**单条 log-ratio 可以为负，KL 的期望才非负。** 如果回答按自回归方式生成，序列 log-ratio 等于各个生成 token 的 log-ratio 之和，包括所用序列定义里的结束符；不能悄悄改成均值，再声称目标完全一样。

这个期望关系假设样本来自 $\pi_\theta$、参照覆盖它的支持集，且期望有定义。PPO 实际会复用 $\pi_{\rm old}$ 的样本，因此还要区分“采样时固定的回报”和“更新时重新估计的 KL”；代码里出现一个叫 `kl` 的张量，不代表它必然就是当前策略完整分布的 KL。

想看训练代码里常见的 $k_1$、$k_2$、$k_3$ 怎么选，读 [KL 的三种估计](kl-estimators.md)：同一组两 token 算例，分别核对数值、采样和梯度。

## 3. 两种回答，算一次就能看出差别 {#kl-example}

假设只有 A、B 两个完整回答。Reference 各给 0.5，Actor 给 0.8、0.2。为隔离 KL 的作用，假设两条的奖励都为 1，$\beta=0.2$。这些全是教学数字。

| 回答 | Actor 概率 | $\log(\pi_\theta/\pi_{\rm ref})$ | 惩罚后回报 |
| --- | --- | --- | --- |
| A | 0.8 | 0.4700 | 0.9060 |
| B | 0.2 | −0.9163 | 1.1833 |

B 的单条回报反而被加了一点，因为 Actor 给它的概率比参照更低。按 Actor 概率加权后，KL 约为 0.1927，总目标约为 0.9615，仍低于未惩罚的 1。

```python
import math

policy = [0.8, 0.2]
reference = [0.5, 0.5]
beta = 0.2
log_ratios = [math.log(current / anchor)
              for current, anchor in zip(policy, reference)]
divergence = sum(probability * ratio
                 for probability, ratio in zip(policy, log_ratios))
shaped_rewards = [1 - beta * ratio for ratio in log_ratios]
objective = sum(probability * reward
                for probability, reward in zip(policy, shaped_rewards))
assert log_ratios[1] < 0 < divergence
assert math.isclose(divergence, 0.19274475702175753)
assert math.isclose(objective, 1 - beta * divergence)
print(round(divergence, 4), round(objective, 4))
```

把 `beta` 改为 0，只是去掉这一项代价，并不会让数学表达式失效。实际效果要测：相对同一奖励尺度，更大的系数通常更偏向保留参照行为；更小的系数通常允许更大改变。**没有“所有任务都必须非零”的定理。** 已有实现允许关闭该项，版本与选项可查 [TRL GRPO 文档](https://huggingface.co/docs/trl/grpo_trainer)；这不代表你的任务也应该关闭。

<span id="critic"></span>

## 4. Critic：同样拿 1 分，含义可能完全不同

一道容易的题，原本平均就能拿 0.9；一道困难的题，平均只能拿 0.1。两次都拿到 1，减去基线后是 0.1 和 0.9。我们想区分的是“比当时的预期好多少”，不是给分数换个名字。

令 $G_t$ 为从位置 $t$ 开始的回报，一个 Monte Carlo 优势估计是 $\hat A_t=G_t-V_\psi(s_t)$。为什么可以减？对于固定状态、与当前动作无关的基线 $b(s)$：

$$
\mathbb E_{a\sim\pi_\theta}[b(s)\nabla_\theta\log\pi_\theta(a|s)]
=b(s)\nabla_\theta\sum_a\pi_\theta(a|s)=0.
$$

满足可交换求导与求和等条件时，这一项不改变期望梯度。Actor loss 中应把基线当成固定目标，不让梯度经 advantage 误流回 Critic。推导可对照 [Spinning Up 的 baseline 小节](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#baselines-in-policy-gradients)。

合适的基线可以降方差，但任意基线不保证都有帮助，REINFORCE 也不是没有 Critic 就无法训练。使用 GAE 和不精确的 value bootstrap 时，还涉及偏差与方差的取舍，不能用上面的零期望等式包办所有情况。详见 [GAE 原论文](https://arxiv.org/abs/1506.02438)和[本站逐步计算](../deep-rl/actor-critic-gae.md)。

## 5. PPO clipping 管的是另一件事

PPO 的比值是当前策略除以**这批数据的采样策略**，不是除以 Reference：

$$
\rho_t=\frac{\pi_\theta(a_t|s_t)}{\pi_{\rm old}(a_t|s_t)},\qquad
L_t=\min\left(\rho_t\hat A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right).
$$

如果 $\hat A_t=2$、$\epsilon=0.2$，$\rho_t$ 从 1.2 增到 1.5，这一项仍为 2.4。它停止继续奖励这个方向，却**没有把实际比值强制拉回 1.2**。其他样本共享参数，多轮更新也会互相影响。

[PPO 官方教学实现](https://spinningup.openai.com/en/latest/algorithms/ppo.html)明确提醒：clipping 后策略仍可能走得过远，所以还会监控 KL 并提前停止更新。不要把 PPO-Clip 说成严格执行了 TRPO 的 KL 约束。

## 6. 真正实现时，我会先检查这些

| 检查项 | 容易发生的错误 | 怎么查 |
| --- | --- | --- |
| 比值的分母 | 每次更新都重算旧概率，或误用 Reference | 同一 rollout 的 old log-prob 固定，首次同策略比值应接近 1 |
| 回答 mask | prompt、padding 也混进 loss | 手算一个两条、不同长度的 batch |
| KL 的单位 | token 均值与序列求和混用 | 同时记录长度、序列和与 token 均值 |
| 回报尺度 | 奖励乘十倍却把原 $\beta$ 当成同样约束 | 明确归一化位置，再比较系数 |
| Critic 目标 | 回报与优势混淆，或 bootstrap 越过终止 | 用短轨迹核对 value target 与 advantage |

继续读[数据由谁采出来](on-off-policy.md)，再看[GRPO、DPO、RLVR 的取舍](after-rlhf.md)。这三个问题分别是：目标是什么，梯度怎么估，数据从哪里来；它们需要一起说清楚。
