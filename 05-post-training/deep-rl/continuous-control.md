# DDPG 到 TD3：连续动作怎么选？

**中文** · [English](continuous-control.en.md)

> 阅读时间：约 8–10 分钟 · 最近审阅：2026-10

离散动作能枚举，方向盘角度却不能一一试完。DDPG 的想法是多学一个网络 $\mu_\theta(s)$，直接给出动作，再沿 Critic 的斜率改它。TD3 则针对这条路容易放大的估值误差做了几处修正。

## Actor 不再输出类别，而是一个坐标

Critic 输入 $(s,a)$，输出一个标量 Q；Actor 输入 $s$，输出有界的连续动作。Actor 的目标是让 Critic 认为这个动作更有价值：

$$
\nabla_\theta J\approx
\mathbb E_{s\sim\mathcal D}
[\nabla_aQ_\phi(s,a)|_{a=\mu_\theta(s)}\,\nabla_\theta\mu_\theta(s)].
$$

这是基于 replay 状态的实际更新形式。离开策略数据分布的理论问题并没有因为写成梯度就消失。Critic 参数在 Actor 这一步固定，但 **Q 对动作的梯度不能断**。

例如 $Q(s,a)=-(a-0.7)^2$，Actor 当前输出 0.2，斜率是 $-2(0.2-0.7)=1$，会把动作往 0.7 推。若这个山峰只是 Q 的拟合错误，Actor 也会很认真地往假山峰爬。

## DDPG 的一轮更新

先从 replay 采样，用 target Actor 生成下一动作，再由 target Critic 得到 bootstrap target。更新当前 Critic 后，以 $-Q_\phi(s,\mu_\theta(s))$ 更新 Actor，并慢慢同步 target 参数。

```text
replay (s, a, r, s')
    → target actor(s') → target critic(s', a') → detached target
    → critic(s, a) → regression update
    → actor(s) → critic(s, actor(s)) → actor update
```

行为采样时往确定性动作上加噪声。这种 action noise 是探索机制；不是 SAC 的随机策略熵，也不是下面的 target smoothing。

## TD3 的 3 个修正为什么配套

| 修正 | 做什么 | 主要针对什么 |
| --- | --- | --- |
| 两个 Critic | target 使用较小的 Q | 减少过高估值被 Actor 利用 |
| 延迟 Actor 更新 | Critic 多更新几步，再更新 Actor | 避免追着暂时不准的 Q 跑 |
| Target policy smoothing | 下一动作附近加截断噪声 | 不让尖锐、脆弱的 Q 峰决定 target |

TD3 target 可以写成：

$$
\tilde a'=\mathrm{clip}(\mu_{\bar\theta}(s')+\mathrm{clip}(\epsilon,-c,c)),
\qquad
y=r+\gamma(1-d)\min_{i=1,2}Q_{\bar\phi_i}(s',\tilde a').
$$

这里 $d$ 表示真实终止，$\epsilon\sim\mathcal N(0,\sigma^2I)$ 是平滑噪声，$c$ 限制噪声幅度；外层 clip 再把动作限制到环境允许的范围。两套 Q 并不统计独立，取 min 也可能低估。Actor 通常通过第一套当前 Critic 更新。

## 把 Actor 梯度一直算回参数

继续用 $Q(a)=-(a-0.7)^2$，让 Actor 输出 $a=\tanh\theta$。当 $a=0.2$：

$$
\frac{dQ}{d\theta}=\frac{dQ}{da}\frac{da}{d\theta}
=-2(a-0.7)\cdot(1-a^2)=1\times0.96=0.96.
$$

梯度上升提高 $\theta$，动作向 0.7 移动。如果动作已接近 ±1，$1-a^2$ 很小，就可能进入饱和区。动作边界和缩放既影响环境含义，也影响优化；把不同单位的控制量直接当同一尺度，不是中性的选择。

固定 Critic 参数，不代表固定 Q 对动作的输出。可以暂时关闭 Critic 参数的梯度累积，但必须保留 Q → action → Actor 的计算图；否则 Actor 根本不知道往哪里走。

## 两个 Q 取 min，会不会又太保守？

会。设两个 Q 对同一个真实零价值动作，各自有独立的 ±1 零均值误差。四种组合的 min 是 $[-1,-1,-1,1]$，平均为 −0.5。它抑制了乐观噪声，却引入了保守倾向。实际两个 Critic 误差相关，数值不会照这个例子走，但“取小的一定更准”显然不成立。

再看 smoothing。对一个二次峰 $Q(a)=-(a-a^*)^2$，加入零均值、方差为 $\sigma^2$ 的噪声且暂不考虑边界裁剪：

$$
\mathbb E_\epsilon[Q(a+\epsilon)]=-(a-a^*)^2-\sigma^2.
$$

例子说明平滑考虑附近动作，而不是只信某个点。真实网络里的尖峰形状可能复杂，截断噪声和动作 clip 也会改变分布；噪声尺度需要相对动作范围解释，不能把一份环境的数值原样复制到另一份。

## 把一轮更新拆开看

先把一轮训练的开关列出来。以 Actor 每 2 次 Critic 更新才更新一次为例，这只是用于说明的频率：

| 分支 | 用的网络 | 梯度到哪里 | 什么时候做 |
| --- | --- | --- | --- |
| 构造下一步 target | Target Actor + 两个 Target Critic | 整条分支不反传 | 每次 Critic 更新 |
| 拟合当前 Q | 两个当前 Critic | 各自 Critic 参数 | 每次更新 |
| 改当前动作 | 当前 Actor + 第一个当前 Critic | 穿过 Q 对动作的导数到 Actor | 每 2 次更新 |
| 慢同步 | 当前参数 → Target 参数 | 不是 optimizer 反传 | 通常跟随延迟 Actor 更新 |

慢同步常写成 $\bar\phi\leftarrow(1-\tau)\bar\phi+\tau\phi$。有些代码用 $\rho$ 表示旧参数的权重，恰好是 $1-\tau$；看到接近 1 的系数，不要先判定同步很快。

再算一个 target：$r=1,\gamma=0.9$，同一个平滑后的下一动作在两套 Target Critic 中分别得 4 和 6。非终止时 $y=1+0.9\times4=4.6$，终止时 $y=1$。若当前两个 Q 为 3 和 5，对固定 target 的平方误差分别为 2.56 和 0.16；这还没轮到 Actor 更新。

| 容易混的 3 种“噪声” | 加在哪里 | 为了什么 |
| --- | --- | --- |
| 行为探索噪声 | 与环境交互时的动作 | 去收集不同的数据 |
| Target smoothing 噪声 | 训练时下一动作的 target 分支 | 不让局部尖峰支配 bootstrap |
| Q 估计误差 | 学到的价值函数里 | 这是要处理的误差，不是主动探索机制 |

连续控制、交互很贵、可以重用历史 transition 时，这条路线有吸引力。代价是 replay、多个 target、更新频率和动作尺度都需要对齐。每交互 1 步训练多少次，叫 update-to-data ratio；提高它可以省交互，也可能把旧数据拟合过头。

<details markdown="1">
<summary>把整个 Critic 调用包在 no_grad 里，再更新 Actor，行吗？</summary>

不行。Actor 需要 Q 对动作的导数。可以临时冻结 Critic 参数，避免为它累积梯度，但仍保留从 Q 到动作、再到 Actor 的计算图。真正应 no_grad 的是构造 Critic bootstrap target 的那一支。

</details>

参考：[DDPG](https://arxiv.org/abs/1509.02971) · [TD3](https://arxiv.org/abs/1802.09477) · [TD3 算法说明](https://spinningup.openai.com/en/latest/algorithms/td3.html)。下一篇：[SAC 为什么把随机性放进目标？](soft-actor-critic.md)
