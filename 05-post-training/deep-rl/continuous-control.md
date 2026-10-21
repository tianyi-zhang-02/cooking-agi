# DDPG 到 TD3：连续动作怎么选？

**中文** · [English](continuous-control.en.md)

> 阅读时间：约 10–12 分钟 · 最近审阅：2026-10

离散动作能枚举，方向盘角度却不能一一试完。DDPG 的想法是多学一个网络 $\mu_\theta(s)$，直接给出动作，再沿 Critic 的斜率改它。TD3 则针对这条路容易放大的估值误差做了几处修正。

## Actor 不再输出类别，而是一个坐标

Critic 输入 $(s,a)$，输出一个标量 Q；Actor 输入 $s$，输出有界的连续动作。Actor 的目标是让 Critic 认为这个动作更有价值：

$$
\nabla_\theta J\approx
\mathbb E_{s\sim\mathcal D}
[\nabla_aQ_\phi(s,a)|_{a=\mu_\theta(s)}\,\nabla_\theta\mu_\theta(s)].
$$

这是基于 replay 状态的实际更新形式。replay 里的状态并不来自当前 Actor 会访问的分布；这个 off-policy 的理论问题，不会因为写成梯度就消失。Critic 参数在 Actor 这一步固定，但 **Q 对动作的梯度不能断**。

例如 $Q(s,a)=-(a-0.7)^2$，Actor 当前输出 0.2，斜率是 $-2(0.2-0.7)=1$，会把动作往 0.7 推。若这个高值只是 Q 的拟合错误，Actor 仍会朝那里更新，利用的就是估值误差。

## DDPG 的一轮更新

先从 replay 采样，构造 $y=r+\gamma(1-d)Q_{\bar\phi}(s',\mu_{\bar\theta}(s'))$，其中 $d$ 只表示真实终止。更新当前 Critic 后，以 $-Q_\phi(s,\mu_\theta(s))$ 更新 Actor，并慢慢同步 target 参数。

```text
replay (s, a, r, s', d)
    → target actor(s') → target critic(s', a') → r + γ(1-d)Q' → detached target
    → critic(s, a) → regression update
    → actor(s) → critic(s, actor(s)) → actor update
```

行为采样时往确定性动作上加噪声。这种 action noise 是探索机制；不是 SAC 的随机策略熵，也不是下面的 target smoothing。

Replay 保存实际执行的动作，包括探索噪声和边界裁剪的结果。若只是外部时间上限截断而非任务终止，仍从截断前最后的 $s'$ bootstrap，不能换成 reset 后的新状态；DDPG 和 TD3 都遵守这个边界。

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

例子说明平滑考虑附近动作，而不是只信某个点。若把峰写成 $Q(a)=-k(a-a^*)^2$，平滑后会多扣 $k\sigma^2$：峰越尖（$k$ 越大）扣得越多，所以窄而尖的峰不容易主导 target。真实网络里的尖峰形状可能复杂，截断噪声和动作 clip 也会改变分布；噪声尺度需要相对动作范围解释，不能把一份环境的数值原样复制到另一份。

## 不训练 actor，也能在连续动作上取 max 吗？

DDPG 的 Actor 其实就是一个“近似求 argmax 的函数”：在每个状态替 Critic 找一个高分动作。Q-learning 的 target 里那个 $\max_{a'}Q(s',a')$，还有两种不单独训练网络的做法。

**采样取最大。** 需要取 max 时，采 $N$ 个动作，用 Critic 逐个打分，取最高的，这叫 random shooting。CEM 在此基础上多迭代几轮：保留最好的几个，拟合新的高斯，再采样（单步手算见[规划与 LQR](planning-and-control.md)）。

[QT-Opt（Kalashnikov 等，2018）](https://arxiv.org/abs/1806.10293)在视觉抓取中用 CEM 代替 Actor：每轮采 64 个、留 6 个，做两轮；同一个过程既算训练 target，也在真实机器人上选动作。作者认为这避开了 actor–critic 的不稳定，并说 CEM 在低维问题上对局部最优有一定的鲁棒性。

按候选数量计，取一次 max 要评估 $64\times2=128$ 个动作的 Q 值，batch 里每条样本的 target 都需要这样的搜索。同一轮的候选可以批量前向，下一轮则依赖上一轮的精英样本；因此 128 个候选评估不等于 128 次串行网络调用，实际延迟还取决于批处理与硬件。采到的最好值不会超过这条 Q 曲线自己的最大值，样本少时 target 偏低；而相对真实价值，Q 本身的估值误差照样会被取 max 放大。

**让 Q 本身好求 max。** [NAF（Gu 等，2016）](https://arxiv.org/abs/1603.00748)把 Q 写成状态价值减去一个对动作二次的项：

$$
Q(s,a)=V(s)-\tfrac12\big(a-\mu(s)\big)^\top P(s)\big(a-\mu(s)\big),\qquad P(s)=L(s)L(s)^\top .
$$

$L(s)$ 是网络输出的下三角矩阵，对角线取指数，保证 $P(s)$ 正定。于是 Q 在 $a=\mu(s)$ 处取到最大值 $V(s)$，target 里的 max 不用搜。放弃的是表达力：每个状态下，Q 对动作只有一个峰。论文自己也承认二次形式比一般网络受限；QT-Opt 的作者则认为这类形状不适合抓取：伸向物体的动作价值高，伸进物体之间空隙的动作价值低，Q 对动作有好几个峰。

手算一次。一维动作 $a\in[-1,1]$，自拟的 Q 曲线 $Q(a)=\max\big(1-8(a-0.5)^2,\;0.8-2(a+0.5)^2\big)$ 有两个峰：右边窄而高，在 $a=0.5$ 取到真正的最大值 1；左边宽而矮，在 $a=-0.5$ 取到 0.8。

- **8 个随机样本**：假设一次均匀采样恰好得到 −0.85、−0.6、−0.35、−0.1、0.15、0.3、0.75、0.95，Q 依次是 0.555、0.78、0.755、0.48、0.02、0.68、0.5、−0.62。最好的是 $a=-0.6$，落在左边的矮峰上；离真峰最近的 0.3 只有 0.68，因为右峰太窄。
- **CEM**：第 1 轮就用这 8 个样本，保留最好的 2 个（−0.6、−0.35），新高斯的均值 −0.475、标准差 0.125。要找到比 0.8 更好的动作，需满足 $1-8(a-0.5)^2>0.8$，即 $a\in(0.5-\sqrt{0.2/8},\,0.5+\sqrt{0.2/8})$，区间端点约为 0.3419 和 0.6581；所以 $a>0.34$ 只是一个粗略的必要条件。这个区间的下界离新均值超过 6.5 个标准差，下一轮从该高斯采少量样本时极难命中，但概率不为零。若后续样本仍集中在左峰，搜索可能继续向 −0.5 附近收缩；第一轮并不保证收敛到那里。方差下限、重启和混入均匀样本都会改变后续搜索。
- **NAF 式二次 Q**：假设一个对动作二次的函数在 $[-1,1]$ 上等权拟合这条曲线（教学假设；实际训练靠 Bellman 回归，数据分布决定它偏向哪个峰），拟合出的峰在 $a\approx-0.23$，那里真实 Q 只有约 0.65，比两个峰都低。
- **Actor 梯度上升**：DDPG 沿 $\partial Q/\partial a$ 爬坡。从 $a=0$ 出发，左边那段更高、斜率为 $-2$，一路爬到 −0.5；从 $a=0.2$ 出发，右边那段更高、斜率为 4.8，爬到 0.5。

| 方法 | 每次取 max 的计算 | 找到的动作 | 那里的真实 Q |
| --- | --- | --- | --- |
| 8 个随机样本 | 8 个候选 Q 评估，可批量计算 | −0.6 | 0.78 |
| CEM，每轮 8 个、留 2 个 | 每轮 8 个候选 Q 评估，可批量计算 | 约 −0.5（若继续集中于左峰） | 约 0.8 |
| NAF 式二次 Q | 1 次前向，直接读出 $\mu(s)$ | 约 −0.23 | 约 0.65 |
| 学到的 Actor | 1 次前向，另需训练 Actor | 取决于起点：−0.5 或 0.5 | 0.8 或 1 |
| 真正的最大值 | — | 0.5 | 1 |

这个例子只说明机制，不比较性能：哪种方法都没有自动找到窄峰。动作只有几维、训练规模大而调参昂贵时，采样或 CEM 省掉了一个难调的网络，代价是推理时的大量 Q 调用；价值对动作接近单峰（比如“到达某个位置”）时，NAF 一类做法最便宜；动作有几十维、控制频率又高时，采样需要的样本数随维度迅速增加，学一个 Actor（DDPG、TD3）更划算，但要接受它也只做局部爬坡，并且会追着 Critic 的误差走。

## 把一轮更新拆开看

先把一轮训练的开关列出来。以 Actor 每 2 次 Critic 更新才更新一次为例，这只是用于说明的频率：

| 分支 | 用的网络 | 梯度到哪里 | 什么时候做 |
| --- | --- | --- | --- |
| 构造下一步 target | Target Actor + 两个 Target Critic | 整条分支不反传 | 每次 Critic 更新 |
| 拟合当前 Q | 两个当前 Critic | 各自 Critic 参数 | 每次更新 |
| 改当前动作 | 当前 Actor + 第一个当前 Critic | 穿过 Q 对动作的导数到 Actor | 每 2 次更新 |
| 慢同步 | 当前参数 → Target 参数 | 不是 optimizer 反传 | 通常跟随延迟 Actor 更新 |

Target 网络先复制对应当前网络的初始参数。慢同步常写成 $\bar\phi\leftarrow(1-\tau)\bar\phi+\tau\phi$，Target Actor 也同样同步。有些代码用 $\rho$ 表示旧参数的权重，恰好是 $1-\tau$；看到接近 1 的系数，不要先判定同步很快。

再算一个 target：$r=1,\gamma=0.9$，同一个平滑后的下一动作在两套 Target Critic 中分别得 4 和 6。非终止时 $y=1+0.9\times4=4.6$，终止时 $y=1$。若当前两个 Q 为 3 和 5，对固定 target 的平方误差分别为 2.56 和 0.16；这还没轮到 Actor 更新。

| 容易混的 3 种“噪声” | 加在哪里 | 为了什么 |
| --- | --- | --- |
| 行为探索噪声 | 与环境交互时的动作 | 去收集不同的数据 |
| Target smoothing 噪声 | 训练时下一动作的 target 分支 | 不让局部尖峰支配 bootstrap |
| Q 估计误差 | 学到的价值函数里 | 这是要处理的误差，不是主动探索机制 |

连续控制、交互很贵、可以重用历史 transition 时，这条路线有吸引力。代价是 replay、多个 target、更新频率和动作尺度都需要对齐。这里把 update-to-data ratio 定义为每新增一条环境 transition，做多少轮 Critic minibatch 更新；同一批次更新两个 Q 算一轮，不把 Actor 的 optimizer step 另加进去，batch size 也要单独报告。提高这个比率可以省交互，也可能把旧数据拟合过头。

<details markdown="1">
<summary>把整个 Critic 调用包在 no_grad 里，再更新 Actor，行吗？</summary>

不行。Actor 需要 Q 对动作的导数。可以临时冻结 Critic 参数，避免为它累积梯度，但仍保留从 Q 到动作、再到 Actor 的计算图。真正应 no_grad 的是构造 Critic bootstrap target 的那一支。

</details>

参考：[DDPG](https://arxiv.org/abs/1509.02971) · [TD3](https://arxiv.org/abs/1802.09477) · [TD3 算法说明](https://spinningup.openai.com/en/latest/algorithms/td3.html)。下一篇：[SAC 为什么把随机性放进目标？](soft-actor-critic.md)
