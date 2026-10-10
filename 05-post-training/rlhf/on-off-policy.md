# On-policy、off-policy：旧回答还能拿来训练吗？

**中文** · [English](on-off-policy.en.md)

> 阅读时间：约 12 分钟 · 难度：必修 · 最近审阅：2026-10-09

生成一批长回答很贵。刚收集完就丢掉，当然心疼；但一直反复使用，又会遇到一个问题：**生成这批回答的模型，已经不是眼下正在更新的模型了。**

这不是数据过了几天的问题。就算刚过去十秒，只要参数或采样规则变了，行为分布就可能不同。

<span id="on-policyoff-policy-offline-rl"></span>

## 三个容易混在一起的问题

| 问题 | 在问什么 | 一个例子 |
|---|---|---|
| On-policy / off-policy | 学习所针对的策略，和产生数据的策略是否一致 | 用旧策略的回答更新新策略，需要考虑分布差异 |
| Online / offline | 学习过程中还能不能取得新交互数据 | 固定日志上的 offline RL 无法现场补采样 |
| On / off distribution | 测试输入与训练输入的分布是否不同 | 即使每轮重新 rollout，也可能没见过新的业务领域 |

它们不是同义词。SAC 一边和环境交互、一边使用 replay buffer，是 **online、off-policy** 的例子。[Spinning Up 的 SAC 文档](https://spinningup.openai.com/en/latest/algorithms/sac.html)可以看到完整流程。标准离线 DPO 使用固定偏好对，但不宜因此把它直接当作带环境转移和 Bellman backup 的 offline RL。

## 为什么 PPO 叫 on-policy，却能训练好几个 epoch？

先想清一批数据的生命周期：

```text
策略 v7 生成回答 → 保存回答、reward、v7 的 log-prob
                         ↓
                 更新为 v8，再更新为 v9
                         ↓
               结束这批更新，重新采样
```

最初的数据由 rollout policy $\pi_{\mathrm{old}}$ 生成。更新开始后，$\pi_\theta$ 就变了，但分母仍然使用生成数据时的概率：

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)}.
$$

PPO 允许对这一批数据做多轮 minibatch 更新，同时用 surrogate objective、clipping，或实现里的 KL early stopping 控制变化。它通常被归为 on-policy，也常被口头称为 near-on-policy：这是强调**频繁刷新 rollout、限制旧数据复用**的工作方式，不是保证每次更新后分布完全一致。[PPO 论文](https://arxiv.org/abs/1707.06347)与 [Spinning Up 实现说明](https://spinningup.openai.com/en/latest/algorithms/ppo.html)分别给出了方法和具体训练循环。

不要每更新一步就把分母改成最新模型的 log-prob。那样的 ratio 已经不是「相对这批数据的生成策略」了。这里的 old policy 也不是用于 KL anchor 的 Reference。

## GRPO 也是旧模型采样，为什么还叫 on-policy？ {#grpo-data-lifecycle}

因为模型一更新，刚才的参数就成了“旧模型”。如果把这也理解成不能用旧数据，那么连一次 rollout 后的第二步更新都没法做了。

这里真正要看的是：**这批回答离当前策略有多远，准备用多久，什么时候换一批。**原始 GRPO 没有要求每做一次梯度更新就重新生成所有回答。它先采样一组回答，再在这组数据上做有限次更新。[DeepSeekMath 的 Algorithm 1](https://arxiv.org/html/2402.03300v3)把采样步骤和内部的 $\mu$ 次更新分开写了。

| 顺序 | 做什么 | 哪些东西别悄悄换掉 |
|---|---|---|
| 1. 采样 | 保存策略 v7；每个问题生成 $G$ 个回答 | 记录生成版本、采样设置和对应 log-prob |
| 2. 打分 | 在同一个问题的回答组内算 reward、均值和 advantage | 别把不同问题的回答拼成一组，也别在过滤后假定原来的组均值仍为 0 |
| 3. 更新 | 用这些回答把模型更新为 v8、v9 | Ratio 的分母仍对应 v7；这批 advantage 和训练 mask 保持明确、一致 |
| 4. 刷新 | 结束这批更新，用新的策略生成下一批 | 这是刷新 old policy，不是自动刷新 Reference |

例如 v7 生成某个 token 时的概率是 0.4，更新后变成 0.6，ratio 就是 1.5。如果每步都把“旧概率”覆盖成当前的 0.6，ratio 又变回 1——程序能跑，但已经量不出相对采样策略的变化了。

Reference 的更新节奏是另一件事。原论文在外层迭代设置 Reference，内部再多次采样、更新；它不是每个 optimizer step 都跟着 actor 走。原算法还出现了 replay buffer，不过那一步用于**更新 reward model**，不能据此说 actor 默认从任意历史回答池里抽样。

今天的实现也有不同选择。[TRL GRPO 文档](https://huggingface.co/docs/trl/grpo_trainer)提供 `num_iterations` 来控制每次生成后的更新次数，也有不同 loss 和 rollout 设置（2026-10-09 核对）。因此读一份训练配置时，光看到 `GRPO` 不够，还得看实际的数据循环。

### 同步、复用、异步，怎么选？ {#stale-rollout-checks}

最容易排查的起点是同步训练：生成完一批，再更新，再同步参数。代价是生成和训练可能互相等。增加更新次数能摊薄生成成本；异步流水线能减少等待，但更容易用到落后多步的回答。

| 选择 | 想省什么 | 需要盯住什么 |
|---|---|---|
| 每批更新少、及时刷新 | 先把分布差异压小，方便定位问题 | 生成开销、GPU 等待时间 |
| 同一批多更新几次 | 减少每次更新分摊的 rollout 成本 | Ratio 分布、实际进入平坦分支的比例、KL 和验证集效果 |
| 异步生成与训练 | 让生成和更新重叠 | 策略版本差、队列年龄、过期样本处理、采样端与训练端概率是否一致 |

比较时不要只报“每秒更新多少步”。还要在相同算力或时间预算下，看最终质量、有效训练 token 数，以及有多少回答因为过期被丢掉。复用次数没有适合所有任务的最优值。

Importance weighting 能修正特定假设下的分布差异，但不会改变数据的来历，也不会把任意旧数据自动变成 on-policy。PPO/GRPO 的逐 token ratio 和 clipping 更不能保证精确还原整条轨迹的分布。下面用一个可以手算的例子，把能修正什么、修不回什么分开看。

## 一个两动作例子：为什么要做重要性加权 {#reweighting}

先不用语言模型。假设只做一次选择，动作 A 的奖励为 1，B 为 3。旧策略 $\mu$ 以 $0.9/0.1$ 的概率选择 A/B；目标策略 $\pi$ 则各选一半。

| 动作 | 旧策略 $\mu$ | 目标策略 $\pi$ | reward | 权重 $\pi/\mu$ |
|---|---|---|---|---|
| A | 0.9 | 0.5 | 1 | $5/9$ |
| B | 0.1 | 0.5 | 3 | 5 |

旧策略的期望 reward 是 $1.2$，目标策略是 $2$。直接平均旧日志，当然不会自动变成目标策略的表现。若目标策略有概率选择的动作，旧策略也都有非零概率，即满足支持条件，就可以换测度：

$$
\begin{aligned}
\mathbb E_{a\sim\pi}[r(a)]
&=\sum_a\mu(a)\frac{\pi(a)}{\mu(a)}r(a)\\
&=\mathbb E_{a\sim\mu}\left[\frac{\pi(a)}{\mu(a)}r(a)\right].
\end{aligned}
$$

这里动作后的 reward 机制也假定不变。对一次随机决策，这是一个精确的期望恒等式；有限样本估计仍然有误差。

下面刻意造 45 个 A 和 5 个 B，让频数恰好等于期望比例，方便核算；不是说真实采样每次都会这样。

```python
import math

behavior = {"A": 0.9, "B": 0.1}
target = {"A": 0.5, "B": 0.5}
rewards = {"A": 1.0, "B": 3.0}
actions = ["A"] * 45 + ["B"] * 5
weights = [target[action] / behavior[action] for action in actions]
logged_mean = sum(rewards[action] for action in actions) / len(actions)
estimate = sum(weight * rewards[action] for weight, action in zip(weights, actions)) / len(actions)
effective_size = sum(weights) ** 2 / sum(weight ** 2 for weight in weights)
assert math.isclose(logged_mean, 1.2)
assert math.isclose(estimate, 2.0)
assert math.isclose(effective_size, 18.0)
print(round(logged_mean, 2), round(estimate, 2), round(effective_size, 2))
```

虽然有 50 条记录，权重的有效样本量指标（ESS）只有 18：少数 B 扛了很大一部分权重。这是**权重集中程度的诊断**，不是严格保证「相当于 18 个独立样本」，更不是置信区间。

## 三个修不回来的地方

### 1. 从未采到的动作，不能靠除法补出来

如果旧策略从来不选 B，我们只观察到 A 的 reward 是 1。B 的 reward 可以是 0，也可以是 3，这两个世界都符合手里的日志，却对应不同的目标策略期望：$0.5$ 或 $2$。

在分母加一个很小的数能让程序不报错，却不能创造缺失的观测。需要新探索、额外假设或更保守的决策。更多推导见 [offline RL 与 OPE](../deep-rl/offline-and-ope.md)，以及 [Levine 等人的 offline RL 综述](https://arxiv.org/abs/2005.01643)。

### 2. 单个 token 的小偏差，乘起来也不小

完整轨迹的重要性权重通常含有各步 ratio 的乘积。哪怕每步只有 $1.1$，20 步后也是 $1.1^{20}\approx6.73$。不是说真实 ratio 总会同向变化，而是提醒我们：长序列可能放大估计方差。

用 log-prob 相加能改善数值计算，却不会消除统计方差。PPO 的逐 token surrogate 也不能简单解释成「精确修正了整条旧轨迹的分布」。

### 3. 裁剪权重改变了估计

上例若把权重截到 2，五个 B 的贡献被压小，估计变成 $(45\times5/9+5\times2\times3)/50=1.1$，不再是 2。这说明 bias–variance tradeoff 很具体，不是写一句「加 clipping 更稳定」就结束了。

这只是普通重要性权重裁剪的例子，**不是 PPO 的完整 clipped objective**。PPO 还会根据 advantage 的正负取最小值，详见 [PPO clipping](ppo-clipping.md)。

## 放回 LLM：该保存哪些信息？

| 信息 | 为什么要保存 | 常见误区 |
|---|---|---|
| Rollout policy 版本 | 知道是哪次参数生成的回答 | 用当前模型重算，然后假装是旧 log-prob |
| 实际采样规则与概率 | temperature、top-p 等会改变行为分布 | 把未处理的模型概率直接当作 sampler 概率 |
| Token、mask、终止原因 | 区分 prompt、padding、EOS 与长度截断 | 把截断当自然结束，或把 padding 算入 loss |
| Reward / verifier 版本 | 打分规则变了，目标也可能变了 | 混合不同版本分数却不做标记 |
| 轨迹年龄与队列延迟 | 异步生成可能落后于训练很多步 | 只看墙上过去几秒，不看策略更新了几次 |

特别是 top-p 等截断采样：一些 token 的实际采样概率会变成零。若你要声称做了精确 importance correction，就必须说明实际行为概率、目标分布与支持条件；不能默认为未截断 softmax。工程上采用近似也可以，但要明确近似在哪，并检查它的影响。

<span id="reward-model-critic"></span>

## Reward Model、Critic 和日志分别告诉你什么？

| 对象 | 回答的问题 | 不能替代什么 |
|---|---|---|
| Reward Model / verifier | 按这套标准，这个结果得几分？ | 不是产生样本的行为概率 |
| Critic | 从这个状态按所评估的策略继续，预期回报多少？ | 不是外部的「正确答案判官」 |
| 行为日志 | 某个策略实际选了什么、随后发生什么？ | 没发生过的反事实不能直接读出来 |

回到开头：旧回答不是不能用，而是要知道它们来自哪里、覆盖了什么，以及你愿意承担哪种偏差。数据复用能省生成成本；为了省这笔钱，把估计前提弄丢了，最后可能连「训练变好了没有」都说不清。

下一篇看 [Reference 与 Critic](reference-and-critic.md)，把 reference KL、old-policy ratio 和 baseline 彻底分开。
