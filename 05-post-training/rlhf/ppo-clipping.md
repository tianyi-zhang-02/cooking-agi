# PPO clipping：为什么不能一直往上加？

**中文** · [English](ppo-clipping.en.md)

> 阅读时间：约 9 分钟 · 难度：必修 · 最近审阅：2026-10-09

模型生成一个回答，得到不错的反馈，于是我们提高它生成这个回答的概率。这很合理。

但 PPO 会拿同一批回答训练好几轮。如果一份好结果每轮都推动模型大幅改变，旧样本很快就可能不再适合指导现在的模型。Clipping 的做法是：当某个方向已经改了不少，这一项就不再鼓励它继续沿同一方向变化。

先看它对一次更新做了什么。

## 答对以后，概率怎么改？ {#worked-update}

把任务缩小到一个 token：问“2 + 3 等于几？”，只允许回答“5”或“6”。答对得 1 分，答错得 0 分。

假设采样时，模型有 40% 的概率答对，Critic 也估计能拿 0.4 分。这次答了“5”，拿到 1 分，比预期好 $1-0.4=0.6$。这个差就是本例的 advantage。

训练了几步后，回答“5”的概率变成 60%。相对采样时的 40%，概率比值是 $0.6/0.4=1.5$。如果继续直接用“比值 × advantage”，这一项是 $1.5\times0.6=0.9$，概率越高，值还会越大。

PPO 加了一条裁剪分支。设 $\epsilon=0.2$，正 advantage 对应的这条分支最多使用比值 1.2，然后取两者较小值：

$$
\min(1.5\times0.6,\;1.2\times0.6)=0.72.
$$

现在再把“5”的概率推高，这一项也不增加了。不过，**实际概率仍然可以是 60%**。Clipping 不是把它强制降回 48%，只是停止追加这份激励。

反过来，如果采到的是“6”，它低于预期，advantage 为 $0-0.4=-0.4$。这时训练希望降低它的概率，而不是提高。下面会看到，裁剪方向也跟着反过来。

## 放回语言模型里

上面只看了一个 token，并暂时省去 KL 奖励项。长回答的处理仍然是逐个生成位置计算：状态 $s_t$ 是问题加已生成前缀，动作 $a_t$ 是当前位置采样出的 token。

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)}.
$$

$$
\begin{aligned}
c_t&=\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon),\\
\ell_t&=\min(\rho_t\hat A_t,c_t\hat A_t).
\end{aligned}
$$

这里 $t$ 是生成位置，不是第几轮训练。分母来自**生成这批回答时的旧策略**；同一批数据更新期间，它保持不变。Reference 是另一份训练参照，不放在这个分母里。

PPO 最大化这些采样项组成的目标，代码通常最小化它的负数。上一节的 0.72 只是一个采样项，并不是整批训练的平均得分。

## 看正负两个方向

图中用 $A=1$ 和 $A=-1$ 展示方向。可以切换正负，再拖动 $\rho$；纵轴数值因此不直接对应刚才的 $A=0.6$。

<!-- widget:tx-ppo-clip -->

| Advantage | 概率怎么变 | 裁剪后的效果 |
|---|---|---|
| $A_t>0$ | 概率提高 | 比值超过 $1+\epsilon$，停止追加激励 |
| $A_t>0$ | 概率降低 | 保留把概率往上拉的梯度 |
| $A_t<0$ | 概率降低 | 比值低于 $1-\epsilon$，停止追加激励 |
| $A_t<0$ | 概率提高 | 保留把概率往下压的梯度 |

所以它并不是把上下两端一律截平。**沿 advantage 指示的方向改得太多，才停止继续鼓励；往反方向走，仍然要纠正。**

<details markdown="1">
<summary>为什么一个 min 就能得到这两种裁剪方向？</summary>

当 $A_t$ 为负，乘法会反转大小关系。把式子分开写就能看出来：

$$
\begin{aligned}
A_t>0:\quad\\
\ell_t=\min(\rho_t,1+\epsilon)A_t,\\[6pt]
A_t<0:\quad\\
\ell_t=\max(\rho_t,1-\epsilon)A_t.
\end{aligned}
$$

正 advantage 只裁上界，负 advantage 只裁下界。$A_t=0$ 时，这个采样项为零。进入平坦区后，这一项对 ratio 的局部梯度为零；边界处则涉及分段函数的不可导点。

</details>

## 被 clip 的 token，还有梯度吗？ {#clipped-token-gradients}

先别急着回答“没有”。代码里算过 `clamp`，不等于最终 loss 选中了那条平坦的分支。还要看 `min`，以及 advantage 的正负。

把旧概率固定为 0.5，$\epsilon=0.2$。下面列出 4 个独立的采样位置，使用代码实际最小化的 loss：$L_t=-\ell_t$。

| 当前概率 | Ratio | Advantage | $L_t$ | 对当前 log-prob 的梯度 |
|---|---|---|---|---|
| 0.7 | 1.4 | +1 | −1.2 | **0**：好方向已经走得够远 |
| 0.3 | 0.6 | +1 | −0.6 | **−0.6**：好 token 的概率反而降了，要拉回来 |
| 0.3 | 0.6 | −1 | +0.8 | **0**：坏方向的概率已经降了不少 |
| 0.7 | 1.4 | −1 | +1.4 | **+1.4**：坏 token 的概率反而升了，要压下去 |

4 个 ratio 都在区间外，只有 2 项没有梯度。这里的梯度是对 **log-prob** 求的，不是直接对 logit 或模型参数求的；后者还要继续走计算图。旧 log-prob 和 advantage 在这次更新中都作为常量。

<details markdown="1">
<summary>用 PyTorch 跑一下这 4 种情况</summary>

只需要 CPU。这里把选中 token 的 log-prob 当作叶子变量，是为了单独检查 loss 这一层；真实模型中，它来自整份词表 logits 的 log-softmax。

```python
import math
import torch

current_log_probs = torch.tensor(
    [0.7, 0.3, 0.3, 0.7], dtype=torch.float64
).log().requires_grad_()
old_log_probs = torch.full_like(current_log_probs, math.log(0.5))
advantages = torch.tensor([1., 1., -1., -1.], dtype=torch.float64)
ratios = (current_log_probs - old_log_probs).exp()
direct = ratios * advantages
clipped = ratios.clamp(0.8, 1.2) * advantages
losses = -torch.minimum(direct, clipped)
losses.sum().backward()
torch.testing.assert_close(
    current_log_probs.grad,
    torch.tensor([0., -0.6, 0., 1.4], dtype=torch.float64),
)
print(losses.detach().tolist(), current_log_probs.grad.tolist())
```

若把 `sum()` 改成 `mean()`，梯度会再除以 4，但哪些项为零不会变。例子刻意避开 0.8、1.2 两个不可导边界，方便用有限差分复核。

</details>

### 这一项不动，不代表模型不动 {#other-gradient-paths}

再往前想一步：即使某个 token 的 policy loss 真进了平坦区，它的概率也不一定保持不变。

| 还有谁在更新？ | 一个小例子 |
|---|---|
| KL 或 entropy 等其他 loss | 当前二选一概率为 0.7/0.3，Reference 为 0.5/0.5。Policy loss 可以已经平了，但 KL 仍可能把分布往 Reference 拉 |
| 其他 token、其他样本 | 两个不同问题共用同一个参数。第一个样本不再提供梯度，第二个仍然可以改变这个参数，进而改变第一个问题的输出 |

这不是说每个实现都有 KL 和 entropy bonus；有没有、权重多大，要看实际配置。想看具体的梯度，可以展开下面的二选一例子。

<details markdown="1">
<summary>Policy 项为 0，其他路径还能给出多大的梯度？</summary>

[配套 CPU 脚本](../code/policy_gradient_checks.py)分别检查了这两种情况：

- **其他 loss：**二选一的当前概率为 0.7/0.3，旧策略和 Reference 都为 0.5/0.5，采到第一个动作且 advantage 为 +1。令总 loss 为 policy loss 加 $0.1\,\mathrm{KL}(\pi\|\pi_{\rm ref})$，再减 $0.01H(\pi)$。对第一个动作 logit 求导，policy 项为 0，总梯度约为 0.0196。这里用的是两个动作上**精确求和的 KL**，不是 sampled KL estimator。
- **共享参数：**两个不同问题的选中动作都由同一个 scalar logit 决定，当前概率均为 0.7，旧概率均为 0.5，advantage 分别为 +1 和 −1。第一项平了，第二项没有；对两项取平均后，共享 logit 的梯度为 0.21。一次更新仍然会降低第一个动作的概率。

</details>

所以比较准确的说法是：**进入平坦区的那一项，不再通过这条 policy-loss 分支提供局部梯度。**不要把它说成“这个 token 被冻结了”。如果实现换了 loss 或裁剪方式，也要重新检查计算图，不能只看到一个 `clamp` 就套用结论。

## Clipping 没有替我们解决什么？

它不知道评分标准是否合理。如果奖励模型偏爱啰嗦的答案，advantage 也可能鼓励模型变啰嗦。这里的“好方向”只是相对当前估计而言，不保证对用户真的更好。

它也没有给整次参数更新设下严格的 KL 上限：其他样本会共享参数，多轮更新还会累积影响。[Spinning Up 的 PPO 说明](https://spinningup.openai.com/en/latest/algorithms/ppo.html)因此还介绍了按 KL 提前停止更新的办法。PPO clipping 和 gradient clipping 也不同，后者直接处理梯度。

GRPO 经常沿用这类裁剪目标，只是 advantage 的来源不同。想继续看这条区别，读 [GRPO、DPO、RLVR](after-rlhf.md)；如果不清楚旧策略和 Reference 的关系，回到 [Reference 与 Critic](reference-and-critic.md)。
