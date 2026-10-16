# Q-learning、PPO、DPO：到底在学什么？

**中文** · [English](methods-map.en.md)

> 阅读时间：约 10–13 分钟 · 最近审阅：2026-10

这些名字经常一起出现，但并不是一代替换一代。Q-learning 先估计动作的价值，PPO 直接调整策略，DPO 则从成对偏好中学习。先分清它们拿到什么数据、更新什么东西，比背年份有用。

只想看关系，读开头的图和最后的表；想算明白，接着看三个小例子。方法的来历放在[RL 发展史](../../learn/rl-timeline.md)，训练成本和失败原因另见[为什么 RL 还是难训](why-rl-is-hard.md)。

## 三条路线，不是一条升级路线 {#three-routes}

<figure class="worked-update" aria-label="三类方法各自学习什么">
<ol>
<li><small>Q-learning</small><strong>先估值，再选动作</strong><span>一次转移：状态、动作、奖励、下一状态。更新 Q 值，再依据 Q 选动作。</span></li>
<li><small>Policy gradient / PPO</small><strong>试过以后，直接改策略</strong><span>策略产生轨迹。估计哪些动作比预期好，再调整它们的概率。</span></li>
<li><small>离线 DPO</small><strong>拿一对回答来比较</strong><span>同一问题下的偏好对。更新回答之间相对 reference 的概率比。</span></li>
</ol>
<figcaption>这是学习机制的区别，不是性能排名。前两类还可以结合成 Actor–Critic；离线 DPO 也可以放进外层反复收集偏好的流程。</figcaption>
</figure>

比如让模型写一个排序函数：你可以给一份正确实现让它模仿，这是 SFT；可以提供两个实现和偏好标签，用 DPO；也可以让当前模型不断生成代码、跑测试，再按结果更新策略，这是在线 RL 的一种设置。三者面对的任务相同，监督和训练过程却不同。

## Q-learning：把这一步接到未来 {#q-update}

设一个两步任务：先选路线，再到终点。当前认为动作 $a$ 值 2 分；执行后拿到 1 分，下一状态两个动作的估值分别是 3 和 4。折扣 $\gamma=0.9$，学习率 $\alpha=0.5$。

$$
y=r+\gamma(1-d)\max_{a'}Q(s',a'),\qquad
Q(s,a)\leftarrow Q(s,a)+\alpha\bigl(y-Q(s,a)\bigr).
$$

$d$ 表示任务真正终止。未终止时，target 是 $1+0.9\times4=4.6$，新估值是 $2+0.5(4.6-2)=3.3$。如果这一步已经终止，target 就只是 1，新估值变成 1.5。**同样的奖励，是否还有未来，会改变这次更新。**

这里的 4 不是已经拿到的奖励，而是模型自己的估计，这就是 bootstrapping。target 用下一步的最大 Q，不要求收集数据时真的选了那个动作；这也是 Q-learning 属于 off-policy 方法的原因。能用旧转移，不代表旧数据可以无限重复、还一定稳定。

[Watkins 与 Dayan（1992）](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf)给出的收敛保证有条件：有限表格、充分访问各状态动作、合适的递减学习率等。换成神经网络、有限数据和不断变化的采样分布，不能直接沿用这个保证。DQN 的 replay 和 target network 是应对部分训练问题的设计，不是把困难全部消掉。继续看 [Fitted Q](fitted-q.md) 和 [DQN](dqn.md)。

## PPO 与 GRPO：改概率之前，先估计好坏 {#policy-update}

策略梯度（policy gradient）不一定先学一个 Q 表。它可以直接提高表现较好的动作的概率，但“好”最好是相对某个基准，而不是只看原始奖励。这个差值或估计量就是 advantage。

PPO 的常见 Actor–Critic 实现用 value model 和 GAE 估计 advantage。设某个已采样 token 的旧概率是 0.20，新概率是 0.26，概率比 $\rho=1.3$。若 advantage 为 2，裁剪范围为 $[0.8,1.2]$：

$$
L_{\mathrm{clip}}=\min\bigl(\rho A,\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)A\bigr)
=\min(2.6,2.4)=2.4.
$$

这是要**最大化**的 surrogate objective；写成 loss 时通常取负。这个样本继续提高概率，在裁剪分支上不会继续获益，但其他样本仍可改变共享参数。裁剪不是把真实概率比硬锁在范围内，也不是保证 KL 不会变大。出处是 [PPO（2017），§3](https://arxiv.org/pdf/1707.06347)；负 advantage 的另一半情况在 [PPO clipping](../rlhf/ppo-clipping.md)。

GRPO 的一个关键变化是 advantage 的估计方式：同一问题采多个回答，用组内标准化后的奖励代替 value model 的估值，代价是每题要采样多次；整组同分时还要避免 $0/0$（分母加小量或跳过该组）。奖励 $[0,0,1,1]$ 怎样变成 $[-1,-1,1,1]$、同分组为什么没有相对信号却不等于完全不更新，见 [LLM bridge 的算例](llm-bridge.md#group-signal)；组大小、稀疏奖励和长任务的代价见[同一篇](llm-bridge.md#sparse-groups)。[DeepSeekMath（2024），§4.1](https://arxiv.org/html/2402.03300v3)介绍了这条路线。

## DPO：不跑在线 rollout，也能学偏好 {#dpo-pair}

普通离线 DPO 用固定的 $(x,y_w,y_l)$：同一输入、偏好的回答、不偏好的回答。它借助 KL 正则化奖励最大化问题与偏好模型之间的关系，直接写出策略的训练目标；不需要在这段训练循环中不断生成回答、拟合独立 reward model。**这不等于数据免费，也不等于它在做 Q-learning。** 见 [DPO（2023），§3–4](https://arxiv.org/html/2305.18290v3)。

把一对回答相对 reference 的变化压成一个数：

$$
m=\beta\left[\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right],\qquad
\ell=-\log\sigma(m).
$$

看一个自拟的两回答小分布，$\beta=1$：reference 分别给两者 0.5、0.5；当前策略给 0.6、0.4。于是 $m=\log1.5$，loss 为 $-\log0.6\approx0.5108$，小于初始的 $-\log0.5\approx0.6931$。真实语言模型用完整回答的条件 log-prob，不是把这两个回答单独做一次 softmax。

还有一个容易误会的地方：**loss 下降，不保证 chosen 的绝对概率上升。** 另设一个三回答例子：保持 $\beta=1$，固定 reference 为 $(0.2,0.2,0.6)$，当前策略从同一分布变成 $(0.15,0.05,0.80)$，前两项是 chosen / rejected。chosen 概率降低了，但 rejected 降得更多；相对 reference 的比值之比是 3，DPO loss 仍从 $\log2$ 降到 $\log(4/3)\approx0.2877$。这个反例说明目标约束了什么，不是在模拟一次真实梯度更新。

偏好标签如果只奖励篇幅或讨喜语气，模型也可能学到这些。DPO 省掉在线 RL 的部分环节，并没有替你解决标签标准、分布覆盖或独立评估。更完整的推导在 [DPO、GRPO 与 RLVR](../rlhf/after-rlhf.md)。

## 三种“旧模型”，别混在一起 {#model-roles}

| 名称 | 为什么留着它 | 一般怎样更新 |
| --- | --- | --- |
| Q-learning / DQN 的 target network | 给 bootstrap target 一个相对稳定的估值来源 | 周期复制或缓慢更新；表格 Q-learning 不一定有独立网络 |
| PPO 的 old / behavior policy | 说明样本从哪个分布来，构造概率比 | 随采样批次变化；异步时还要区分实际 behavior 与更新所用的 proximal policy |
| DPO / RLHF 的 reference policy | 为偏好优化或 KL 正则化提供参照 | 普通设置通常在一阶段内固定；不等于上一轮 rollout 模型 |

这三种角色都可能是“复制一份模型”，但用途不同。缓存 log-prob 可以节省某些前向计算；共享参数、offload 和并行方式又会改变成本。不能仅靠“有几个模型”估显存。

## 该用哪个，先看手里有什么 {#choose}

| 你手里有什么 | 可以先试什么 | 仍然要付出的代价 |
| --- | --- | --- |
| 可靠的示范 | SFT baseline | 示范覆盖、输入输出一致性、独立 eval |
| 同一输入下质量可靠的偏好对 | 离线 DPO baseline | 偏好采集、reference 计算、分布变化 |
| 可以反复交互、评分相对可靠的任务 | PPO / GRPO 等在线方法 | rollout、验证、探索、更新稳定性 |
| 离散环境转移，希望先学动作价值 | Q-learning / DQN 及其扩展 | 探索、估值误差、数据覆盖与 bootstrap 稳定性 |

这张表是实验起点，不是强制规则。PPO 不专属于人类偏好，GRPO 不专属于数学；RLVR 说的是奖励可验证，不是规定只能用哪一个 optimizer。下一篇[训练难点](why-rl-is-hard.md)接着算：选了算法以后，时间和预算到底花在哪里？

本页三个数值例子可以运行：`python 05-post-training/deep-rl/code/rl_tradeoffs.py`。程序只检查教学算例，不训练模型，也不测 GPU 性能。
