# RLHF：PPO clipping 的四种更新情况

**中文** · [English](ppo-clipping.en.md)

> 阅读时间：约 2 分钟 · 难度：必修 · 最近审阅：2026-09

<details class="interview" markdown="1">
<summary>快速记忆：$t$、$\rho_t$、$A_t$ 分别是什么？</summary>

- $t$ 是 **time step**；在语言模型里通常就是第 $t$ 个生成 token 的位置，不是第 $t$ 次 optimizer update。
- $s_t=(x,y_{<t})$ 是 prompt 加已生成前缀，$a_t=y_t$ 是这一位置实际采样出的 token。
- $\rho_t$（有些材料写 $r_t$）是新旧策略给**同一个已采样 token**的 probability ratio，不是 reward。
- $A_t$ 决定应该提高还是降低这个 token 的概率；$\rho_t$ 表示概率已经改变了多少。

> **$A_t$ 决定方向，$\rho_t$ 报告步幅，clip 只阻止正确方向走得过头。**

</details>

GRPO 改变了 advantage 的来源，但常继续使用 PPO-style clipped surrogate。对 rollout
中第 $t$ 个已采样 token，定义当前策略与 rollout policy 的 token-level probability ratio：

$$
\rho_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\text{old}}(a_t\mid s_t)}.
$$

其中 $\rho_t>1$ 表示当前 Actor 提高了该 token 的概率，$\rho_t<1$ 表示降低了概率；
$\rho_t$ 始终为正，所以不能直接比较「ratio 和 advantage 的正负」。应该比较的是
$\rho_t-1$ 与 $A_t$ 的方向。

PPO 最大化下面的 clipped surrogate（代码里通常最小化它的负数）：

$$
\min\left(
\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t
\right).
$$

括号顺序很重要：先把 $\rho_t$ 裁到 $[1-\epsilon,1+\epsilon]$，再乘 $A_t$，最后与
未裁剪的 $\rho_tA_t$ 取 `min`。例如 $\epsilon=0.2$ 时，裁剪区间是 $[0.8,1.2]$。
但这**不等于**把实际 ratio 永远硬限制在这个区间。

<!-- widget:tx-ppo-clip -->

| Advantage | Ratio 相对 1 | 当前策略做了什么 | 方向与 clipping |
| --- | --- | --- | --- |
| $A_t>0$ | $\rho_t>1$ | 提高了好 token 的概率 | 方向正确；超过 $1+\epsilon$ 后停止继续奖励 |
| $A_t>0$ | $\rho_t<1$ | 降低了好 token 的概率 | 方向错误；不做下界 clipping，保留梯度把概率拉高 |
| $A_t<0$ | $\rho_t<1$ | 降低了坏 token 的概率 | 方向正确；低于 $1-\epsilon$ 后停止继续奖励 |
| $A_t<0$ | $\rho_t>1$ | 提高了坏 token 的概率 | 方向错误；不做上界 clipping，保留梯度把概率压低 |

因此可以用一个符号判断方向：

$$
(\rho_t-1)A_t
\begin{cases}
>0, & \text{更新方向正确，越过对应边界才 clip},\\
<0, & \text{更新方向错误，不 clip，让梯度纠正。}
\end{cases}
$$

`min` 之所以能产生这种单边效果，是因为 $A_t$ 的符号会改变大小关系：

$$
\ell_t=
\begin{cases}
\min(\rho_t,1+\epsilon)A_t, & A_t>0,\\
\max(\rho_t,1-\epsilon)A_t, & A_t<0.
\end{cases}
$$

正 advantage 只裁上界，负 advantage 只裁下界。进入 clipped plateau 后，这一项的
局部梯度为零；它只是停止鼓励策略继续走远，**不会主动把已经越界的 ratio 拉回区间**。
这也不是 gradient clipping：被裁的是 policy objective 中的 probability ratio。

<details markdown="1">
<summary><b>深挖</b>：old policy、Reference 与 Reward Model 为什么不是一回事？</summary>

- **Old / rollout policy** 提供 $\rho_t$ 的分母；一批 rollout 收集完后，它的 log-probability 在这批 PPO 更新中固定，用来限制局部更新。
- **Reference Model** 是长期冻结的 SFT 副本，用 KL 约束 Actor 在整个训练过程中不要累计漂移太远。
- **Reward Model** 在 PPO 阶段通常冻结，负责给完整回答打分；它训练时用的是 chosen/rejected 的 Bradley–Terry loss，不使用这套四象限 clipping。

所以 PPO clipping 和 reference KL 是两把不同的尺子，Reward Model training 又是前一个阶段的另一种目标。

</details>

标准 GRPO 常把同一 response 的 group-normalized sequence-level advantage 应用到它的
生成 token；实现会在 token aggregation、KL placement 和 clipping 范围上有所变化。
上表描述的是经典 PPO-style clipped objective 的局部行为。
