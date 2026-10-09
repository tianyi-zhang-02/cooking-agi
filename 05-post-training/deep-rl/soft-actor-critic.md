# SAC：为什么奖励之外还要熵？

**中文** · [English](soft-actor-critic.en.md)

> 阅读时间：约 10–12 分钟 · 最近审阅：2026-10

确定性 Actor 太早认准一条路，可能错过别的选择。SAC 不只是给动作加噪声，而是把保留选择余地写进训练目标。**它优化的是另一种价值定义，Critic 的 target 也必须跟着改。**

## 熵项到底在鼓励什么

最大熵目标通常写为：

$$
J(\pi)=\mathbb E\left[\sum_t\gamma^t
\left(r_t+\alpha\mathcal H(\pi(\cdot\mid s_t))\right)\right].
$$

$\alpha$ 控制回报与随机性的相对权重。离散两动作的 Q 为 $[0,1]$ 时，固定 Q 下最大化 $\sum_ap_aQ_a+\alpha H(p)$，最优分布是 $p_a\propto\exp(Q_a/\alpha)$。$\alpha$ 小时更集中，大时更接近均匀。

<div class="drl-lab" data-drl-lab="entropy"><p>离散教学例子：Q=[0,1]，α=0.5 时，高价值动作概率约 0.881。它不是连续 SAC 训练模拟。</p></div>

连续动作使用的是微分熵，可以为负，并且依赖坐标尺度。不要把离散熵图里的数值范围直接搬到连续高斯策略。

## Softmax 为什么会从熵目标里出现？

固定一个状态和每个动作的 $Q_a$，暂时只优化离散分布 $p$。设 $Z=\sum_a\exp(Q_a/\alpha)$，$p^*_a=\exp(Q_a/\alpha)/Z$，就可以把目标改写成：

$$
\sum_a p_aQ_a+\alpha H(p)
=\alpha\log Z-\alpha D_{\rm KL}(p\|p^*).
$$

KL 不小于零，因此最优是 $p=p^*$。这不是凭空把 softmax 塞进来，而是熵正则化问题自己的解。对应的 soft value 是 $\alpha\log\sum_a\exp(Q_a/\alpha)$。

Q 为 $[0,1]$、$\alpha=0.5$ 时，高价值动作概率约 0.8808，期望 Q 约 0.8808，熵约 0.3653，合起来的 soft value 约 1.0635。它超过最大 Q 的 1，是因为目标额外包含熵收益，不是发现了一个奖励大于 1 的动作。

连续动作时，求和变成积分，归一化可能算不出来；SAC 用参数化策略近似改进，而不是把每个连续动作枚举进一个巨大 softmax。

## 自动温度的方向，可以用一个符号检查

设要求策略熵至少达到 $H_{\rm target}$，固定策略时考虑温度目标 $L_\alpha=\alpha(H-H_{\rm target})$，对 $\alpha\ge0$ 做下降。当熵低于目标，导数为负，更新会增大 $\alpha$，增强鼓励随机性的力度；熵太高则反过来。

用采样的 $-\log\pi(a\mid s)$ 估计熵时，这一步要停止策略梯度。实践中常优化 $\log\alpha$ 以保证温度为正；有的实现采用同方向但不同缩放的 surrogate，比较代码时要说明具体 loss。连续微分熵目标可以是负数，不能只凭负号判断是不是写错。

## 常用的双 Q 版本在算什么

这里用常见的无独立 V-network 的 SAC 版本；早期论文的网络安排并不完全相同。下一动作从当前随机策略采样，target Q 使用慢更新参数，$d$ 表示真实终止：

$$
y=r+\gamma(1-d)
\left[\min_iQ_{\bar\phi_i}(s',a')-\alpha\log\pi_\theta(a'\mid s')\right],
\quad a'\sim\pi_\theta(\cdot\mid s').
$$

这里的 soft Q 包含当前奖励，以及从下一状态开始的奖励与熵；当前状态的熵放在策略目标里，不要两边重复加。Critic 拟合上面停止梯度的 target，Actor 则最小化：

$$
L_\pi=\mathbb E_{s\sim\mathcal D,\,a\sim\pi_\theta}
\left[\alpha\log\pi_\theta(a\mid s)-\min_iQ_{\phi_i}(s,a)\right].
$$

第一项鼓励分布不要过早收缩，第二项鼓励高 Q。这里的状态从 replay 采样，是实际算法用来改进策略的目标，不应直接当成开头 $J(\pi)$ 在真实状态访问分布下的精确梯度。奖励整体乘 100、$\alpha$ 却不变，也会改变两项的比例。

## 重参数化与 tanh：最常漏掉的一项

重参数化把随机动作写成“固定噪声经过可微变换”。先把数据流画出来，再看变量变换：

<div class="drl-flow" aria-label="SAC Actor 的可微采样路径">
<span>状态 s<br><small>输出 μ、log σ</small></span><b>→</b><span>固定噪声 ε<br><small>u = μ + σε</small></span><b>→</b><span>动作 a<br><small>tanh 与密度校正</small></span><b>→</b><span>Q 与 log π<br><small>两条路径都到 Actor</small></span>
</div>

令 $u=\mu_\theta(s)+\sigma_\theta(s)\epsilon$，$\epsilon\sim\mathcal N(0,I)$，再用 $a=\tanh u$ 限制范围。这样可以通过动作向 Actor 反传。图里的“固定噪声”只指这一次求导时把采到的 $\epsilon$ 当常数，不是整场训练反复使用同一份噪声。

但动作密度不再是原高斯密度，必须做 change of variables：

$$
\log\pi(a\mid s)=\log\mathcal N(u;\mu,\sigma)
-\sum_j\log(1-\tanh^2u_j).
$$

实际实现用数值稳定的等价式，别在 tanh 饱和处直接取 $\log0$。若再按环境边界缩放动作，还要计入缩放的 log-determinant。它对固定 Actor 梯度可能是常数，但会影响熵数值和温度目标的一致性。

稳定的单维 Jacobian 项可以写成：

$$
\log(1-\tanh^2u)=2\left(\log2-u-\operatorname{softplus}(-2u)\right).
$$

例如 $u=0$ 时结果是 0；$u$ 很大时直接计算 $1-\tanh^2u$ 容易舍入成零，而右侧不用先做两项几乎相等的数相减。多维动作要沿动作维度求和，不是把 batch 维也一起压掉。

## 用一条 transition 核对 3 个更新

设 $r=1,\gamma=0.9,\alpha=0.2$，采样的下一动作有 $\log\pi(a'\mid s')=-0.7$，target Q 的较小值为 4。非终止时：

$$
y=1+0.9\left[4-0.2(-0.7)\right]=4.726.
$$

这里增加的 0.126 来自 soft-value 定义，不是任务真的多发了奖励。这个数字也不能拿来证明 SAC 比 TD3 更好。

| 更新什么 | 读取哪些量 | 哪些地方不能有梯度 |
| --- | --- | --- |
| 两个 Q | Replay 的 $(s,a)$ 与 soft target | Target Q、target 中的动作和 log-prob 整条分支 |
| Actor | 当前状态、重参数化动作、当前两个 Q、log-prob | 不更新 Critic 参数，但保留 Q → 动作 → Actor |
| 温度 α（若启用） | 策略熵与目标熵的差 | 这一步把策略的 log-prob 当固定观测 |

策略梯度那篇的 score-function 更新，把已经采到的动作当作固定样本；这里则让梯度穿过动作回到 Actor。区别在估计方法，不单在动作是否连续：连续策略也可以使用 score-function。两条路径不能照搬彼此的 detach 位置。

## 自动温度，不等于不用做选择

自动调 $\alpha$ 是让策略接近指定 target entropy。你依然要定义目标熵、动作尺度和奖励尺度，还要检查更新符号。观察 $\alpha$ 曲线、log probability、action saturation 和 Critic 值域，比只看一个 return 曲线更容易定位问题。

| 选择 | 收益 | 代价 |
| --- | --- | --- |
| 固定温度 | 简单、容易分析 | 对 reward scale 敏感 |
| 自动温度 | 跟随策略熵调整 | 增加目标和优化器，错误符号会持续放大问题 |
| 双 Critic 的 min | 抑制部分过估计 | 也可能保守低估 |

## 适合什么，不适合什么

SAC 常用于可重用数据的连续控制。它不是“所有 RL 的升级版”：超大离散 token 空间需要不同设计；极稀疏奖励下，熵也未必带来有意义的探索；真实系统还可能禁止随机试错。

参考：[SAC 原始论文](https://arxiv.org/abs/1801.01290) · [Algorithms and Applications](https://arxiv.org/abs/1812.05905) · [双 Q 版本公式与伪代码](https://spinningup.openai.com/en/latest/algorithms/sac.html)。接下来：[学环境模型，先在脑中试几步](model-based.md)。
