# GRPO、DPO、RLVR：它们到底改了哪一环？

**中文** · [English](after-rlhf.en.md)

> 阅读时间：约 12 分钟 · 难度：必修 · 最近审阅：2026-10

模型已经能回答问题了，接下来怎么让它答得更好？先看你手里有什么：是已经整理好了回答对，并标出了更偏好哪一份；还是能让模型生成新回答，再逐一打分？

标准离线 DPO 对应前一种做法。PPO 和 GRPO 常用于后一种，只是把分数变成训练信号的方式不同。RLVR 则在回答另一个问题：**能不能用程序检查结果、给出奖励？** 所以它可以和 PPO 或 GRPO 搭配，不是排在它们后面的“下一代算法”。

先看下面的对照图，就能分清它们解决的问题。想再往下理解，我们会跟着一组回答看 reward 怎样变成训练信号，再算一对偏好回答的 DPO loss。推导与代码可以按需展开。

## 先把三个问题分开

| 要决定什么 | 可以怎么选 | 不要混淆什么 |
|---|---|---|
| 反馈从哪里来 | 人的偏好、Reward Model、程序检查、环境结果 | RLVR 主要回答这一层 |
| 怎么构造更新信号 | Critic 估计的 advantage、组内相对 reward、偏好对 | GRPO 和 DPO 的信号构造不同 |
| 训练时怎么拿数据 | 当前策略生成新回答、复用旧 rollout、固定偏好数据 | 算法名字不能代替数据流程 |

<section class="method-map" lang="zh-CN" aria-labelledby="method-map-title" id="method-comparison">
  <header class="method-map-header"><span>放在一起看</span><strong id="method-map-title">怎么更新，和谁来打分，是两回事</strong></header>
  <p class="method-map-label"><span>01</span> 模型怎么学？</p>
  <div class="method-map-methods">
    <div class="method-map-card"><strong>PPO</strong><p>生成 → 打分 → 更新</p><dl><dt>比较什么</dt><dd>实际回报与价值估计的差别。</dd><dt>典型做法</dt><dd>训练一个价值模型（Critic），帮助估计优势。</dd></dl></div>
    <div class="method-map-card"><strong>GRPO</strong><p>同一道题，生成一组回答</p><dl><dt>比较什么</dt><dd>每个回答相对组内其他回答的表现。</dd><dt>省掉什么</dt><dd>不再单独训练 Critic，但仍要生成回答和打分。</dd></dl></div>
    <div class="method-map-card"><strong>离线 DPO</strong><p>已有回答对，知道更偏好哪一个</p><dl><dt>比较什么</dt><dd>两个回答相对参考模型的概率变化。</dd><dt>数据怎么来</dt><dd>用固定偏好数据训练，不在每次更新时重新采样回答。</dd></dl></div>
  </div>
  <p class="method-map-label"><span>02</span> 好不好，谁说了算？</p>
  <div class="method-map-feedback">
    <div><strong>PPO / GRPO</strong><p>可以由奖励模型打分，也可以用测试程序等验证结果。后者就是 RLVR 的思路，不限定必须用哪一种更新方法。</p></div>
    <div><strong>离线 DPO</strong><p>使用回答之间的偏好标签。标签可以来自人工或其他评审，但质量仍要检查。</p></div>
  </div>
  <p class="method-map-note">显存不能只按“几个模型”来算，还要看权重是否共享、哪些结果能缓存、哪些部分能卸载。</p>
</section>

比如，「GRPO + 程序验证奖励」完全说得通。RLVR 也能用带 Critic 的方法，不能把它直接等同于「只剩两个模型」。权重共享、缓存和卸载还会改变实际显存占用。

## GRPO：同一道题，先比较这一组回答

经典 PPO 通常训练一个 Critic，估计当前状态的预期回报。GRPO 换了一种办法：给同一个 prompt 生成多份回答，用这组回答的 reward 构造相对信号，不再训练独立的 Critic。[DeepSeekMath 原论文](https://arxiv.org/abs/2402.03300)介绍了这一路径；它并不要求 reward 一定来自程序。

### 四份回答，只有一份答对 {#group-example}

比如，同一道代码题，模型生成了四份实现，只有第二份通过当前测试集。若通过得 1 分、否则得 0 分，reward 就是 `[0, 1, 0, 0]`。这是教学例子，不是实验结果；通过当前测试，也不等于证明程序在所有输入上都正确。

采用组内总体标准差（分母为组大小 $G$），暂时省略稳定项 $\epsilon$：

$$
\begin{aligned}
\bar r &= \frac{1}{G}\sum_i r_i,\\
s &= \sqrt{\frac{1}{G}\sum_i(r_i-\bar r)^2},\\
\hat A_i &= \frac{r_i-\bar r}{s+\epsilon}.
\end{aligned}
$$

均值是 $0.25$，标准差约为 $0.433$。通过测试的那份实现得到约 $1.732$ 的 advantage，其余各为 $-0.577$。这不是逐行认定哪些代码有用，而是**按当前评分标准，这一组里哪个结果比平均更好**。

```python
import math

rewards = [0.0, 1.0, 0.0, 0.0]
mean_reward = sum(rewards) / len(rewards)
variance = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
deviation = math.sqrt(variance)
advantages = [(reward - mean_reward) / deviation for reward in rewards]
assert math.isclose(advantages[1], math.sqrt(3))
assert math.isclose(sum(advantages), 0, abs_tol=1e-12)
print([round(advantage, 3) for advantage in advantages])
```

若四份全错，或四份全对，减去均值后都为零。这一组没有提供区分回答的 policy-gradient 信号。加了 KL 等其他项时，总梯度不一定为零，不能把两件事混为一谈。

### Reward、advantage、loss：三个数各管什么？ {#reward-to-update}

刚才通过测试的回答拿到了 1 分，为什么训练时又出现了 `1.732`？前者记录测试结果，后者描述这份回答在当前组里的相对表现。真正更新参数时，还要看模型给这份回答的概率变了多少。

<figure class="worked-update" lang="zh-CN" id="grpo-signal-path">
  <figcaption>沿用刚才的四份回答 · 只看通过测试的那份。</figcaption>
  <ol>
    <li><small>1 · 奖励（reward）</small><strong>测试通过，得 1 分</strong><span>这是当前测试的评分，还没有和其他回答比较。</span></li>
    <li><small>2 · 优势（advantage）</small><strong>组内比较：≈ 1.732</strong><span>在 [0, 1, 0, 0] 中高于平均。换一组回答，这个数也可能变。</span></li>
    <li><small>3 · 进入更新</small><strong>还要看概率比</strong><span>用当前模型与采样时的旧策略比较，再结合 clipping 等设置计算目标。</span></li>
  </ol>
</figure>

只挑这份回答中的一个 token 来算：给定同样的 prompt 和此前的回答前缀，旧策略给它 `0.20` 的概率，当前模型给 `0.26`，概率比就是 `1.3`。不是拿 reward 去除，也不是拿 Reference 的概率作分母。

假设 clipping 范围是 `[0.8, 1.2]`。这里的 advantage 为正，裁剪目标取 `1.3 × 1.732` 与 `1.2 × 1.732` 中较小的一项，约为 **2.078**。这是要最大化的单 token policy 目标；改成最小化的 loss 时取负号。它不是整个 batch 的 loss，也没包括 KL 等其他项。[DeepSeekMath §4.1.1–4.1.2](https://arxiv.org/html/2402.03300v3#S4.SS1)

<details markdown="1">
<summary>算一下：继续提高概率，这个目标还会变大吗？</summary>

```python
import math

def clipped_policy_term(ratio, advantage, epsilon=0.2):
    clipped_ratio = min(max(ratio, 1 - epsilon), 1 + epsilon)
    return min(ratio * advantage, clipped_ratio * advantage)

advantage = math.sqrt(3)
old_probability, current_probability = 0.20, 0.26
ratio = current_probability / old_probability
objective = clipped_policy_term(ratio, advantage)
loss_term = -objective
assert math.isclose(objective, 1.2 * math.sqrt(3))
assert math.isclose(clipped_policy_term(1.6, advantage), objective)
print(round(objective, 3), round(loss_term, 3))
```

把概率比从 `1.3` 提到 `1.6`，这一个正 advantage 项的值不再增加。但 clipping 不是把整个模型的变化锁死：其他 token、共享参数以及其他损失项仍然会影响更新。负 advantage 也不能照搬“超过上界就不动”的解释，见 [PPO clipping](ppo-clipping.md)。

</details>

这里有两个容易认错的角色。**旧策略（old policy）**生成这批回答，留下的 log-prob 用于概率比；**参考模型（Reference）**在启用 KL 约束时提供比较基准。两者可能在某个时刻权重相同，作用却不同。用 log-prob 实现时，概率比是 `exp(current_logp - old_logp)`，不是两个 log-prob 相除。采样设置若改变了生成分布，也要确认这些概率对应哪种分布。

再回头看那 1 分：它只说明当前测试通过了。Advantage 给出相对信号，概率比与裁剪决定它如何进入目标；三者都不保证下一次生成一定答对。验证效果，仍要看没有用于训练的新题。

### 省了 Critic，不等于训练免费了

| 设计选择 | 得到什么 | 付出什么 / 要检查什么 |
|---|---|---|
| 每题采样多份回答 | 不训练 Critic，也能构造相对信号 | rollout、打分和长回答仍然花钱 |
| 减去组内均值 | 同一题内比较，减少题目难度差异的干扰 | 全对、全错都可能没信号；组太小，估计也不稳 |
| 除以组内标准差 | 调整不同组的信号尺度 | 也会改变不同 prompt 的相对权重，不是纯数值修补 |
| 同一回答的 token 共用 outcome advantage | 终局 reward 能进入 token loss | 不代表找到了真正有贡献的推理步骤 |

这里最后一行很重要：答对了一道题，不等于中间每一步都对。把一个序列的 reward 传给所有 token，是一种训练估计方式，不是逐步因果归因。

[TRL 的 GRPO 文档](https://huggingface.co/docs/trl/grpo_trainer)列出了 reward scaling、KL 和 loss aggregation 的不同选择。读代码时要把这些设置记下来：同样叫 GRPO，组内标准差、长度归一化和 KL 的处理也可能不同。组内相对信号是否有用，要和采样成本一起看。

## DPO：把偏好比较直接写进策略的 loss

假设你已经有同一个 prompt 的一对回答：$y_w$ 更受偏好，$y_l$ 较差。标准离线 DPO 用这些偏好对训练策略，不需要在每个训练 step 里再做 rollout、训练独立的 RM 或 Critic。**这不代表数据从来不需要生成，也不代表偏好标签免费。**

比如问“用一句话解释过拟合”。一个回答是“模型连训练样本里的偶然细节也学了进去，遇到新数据反而表现变差”，另一个说“模型还没学会训练数据”。标注者选前者。DPO 直接学习这组**相对偏好**，不必先让一个奖励模型分别给两句话打分。学到这一对，也不代表遇到任何新问题都会选对。

### 为什么 loss 里会出现 Reference？

这里的 Reference 是**冻结的参考模型，不是一条标准答案**。它会给数据里的两份回答分别算概率，作为比较的起点；并不需要再生成一份回答供我们模仿。哪份更好，仍由偏好标签决定，不由 Reference 决定。

沿用上面的“解释过拟合”：把同一条 prompt 和同一份回答分别交给当前模型、参考模型，就能比较这份回答相对起点变得更可能还是更不可能。DPO 再比较 preferred 和 rejected 的这两个变化。[原论文 §4](https://arxiv.org/html/2305.18290v3#S4)

<details markdown="1">
<summary>这个比较为什么能写成 DPO loss？</summary>

在 $\beta>0$ 的 KL 正则化奖励目标下，若把优化范围看成所有满足支持条件的策略分布，并假定配分函数 $Z(x)$ 有限，最优策略满足：

$$
\begin{gathered}
\pi^*(y\mid x)=\frac{\pi_{\mathrm{ref}}(y\mid x)}{Z(x)}\\
{}\times\exp\!\left(\frac{r(x,y)}{\beta}\right).
\end{gathered}
$$

把它反过来写，得到 $r(x,y)=\beta\log[\pi^*(y\mid x)/\pi_{\mathrm{ref}}(y\mid x)]+\beta\log Z(x)$。再放入 Bradley–Terry 偏好模型：比较同一 prompt 的两个回答时，$\log Z(x)$ 抵消了。这就得到 DPO 的形式：

$$
\begin{aligned}
\delta_w&=\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)},\\
\delta_l&=\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}.
\end{aligned}
$$

这里 $\delta_w,\delta_l$ 就是两份回答的相对 log-prob 变化。把缩放后的分差记为 $g$，对偏好数据集 $\mathcal D$ 取平均：

$$
\begin{aligned}
g&=\beta(\delta_w-\delta_l),\\
\mathcal L_{\mathrm{DPO}}&=-\mathbb E_{\mathcal D}\log\sigma(g).
\end{aligned}
$$

这段推导来自 [DPO 原论文](https://arxiv.org/abs/2305.18290)。它依赖奖励目标、偏好模型和支持条件；并不是证明了「所有人的真实偏好都长这样」，也不保证有限神经网络和有限数据能找到全局最优。

</details>

### 算一对回答的 loss {#dpo-example}

假设当前模型给 preferred / rejected 的序列 log-prob 分别是 $-2,-4$，Reference 都是 $-3$。这些是假设数字，数值越大，表示模型给这份回答的概率越高：

<figure class="worked-update worked-update--pairs" lang="zh-CN" id="dpo-score-changes">
  <figcaption>同一份回答，在参考模型和当前模型下的 log-prob。</figcaption>
  <ol>
    <li><small>更受偏好的回答</small><strong>−3 → −2</strong><span>参考模型 → 当前模型<br>相对变化：+1</span></li>
    <li><small>较差的回答</small><strong>−3 → −4</strong><span>参考模型 → 当前模型<br>相对变化：−1</span></li>
  </ol>
</figure>

两种变化再相减，分差是 $1-(-1)=2$。取 $\beta=0.2$，送进 sigmoid 的值就是 $0.4$，loss 约为 $0.513$。如果当前模型和 Reference 相同，两边的变化都为零，loss 就是 $\log 2\approx0.693$；这不要求它原本给两份回答的概率相等。

<details markdown="1">
<summary>用 Python 核对这个数</summary>

```python
import math

chosen_logp, rejected_logp = -2.0, -4.0
reference_chosen, reference_rejected = -3.0, -3.0
beta = 0.2
relative_gap = (chosen_logp - reference_chosen) - (rejected_logp - reference_rejected)
preference_logit = beta * relative_gap
loss = math.log1p(math.exp(-preference_logit))
assert math.isclose(preference_logit, 0.4)
assert math.isclose(loss, 0.5130152523999526)
assert loss < math.log(2)
print(round(loss, 4))
```

</details>

这里用了温和的数值，便于手算；生产实现应使用数值稳定的 `logsigmoid` / `softplus`。标准序列 log-prob 是回答 token 的 log-prob 之和，要正确处理 prompt、padding 和 EOS。随意改成 token 均值，就不再是完全相同的目标。[TRL DPO 文档](https://huggingface.co/docs/trl/dpo_trainer)可用来对照实际的数据格式和损失实现。

DPO 能提高一对回答的相对分差，但这不保证 chosen 的绝对概率一定上升；降低 rejected 的概率也能改善这个 loss。它也无法从一份固定数据里发现根本没有出现过的新失败。

## RLVR：程序给分，程序也有盲区

RLVR 把可验证结果用于 reward，例如答案匹配、编译结果或测试通过情况。它说的是**反馈怎么来**，不是一种唯一的 policy update 算法。[DeepSeek-R1](https://arxiv.org/abs/2501.12948)是使用规则奖励的公开实例之一。

「程序是确定的」和「程序检查得完整」是两回事。假设任务是排序，模型永远输出 `[1, 2, 3]`：

| 检查输入 | 固定输出 | 这个检查能告诉我们什么 |
|---|---|---|
| `[3, 1, 2]` | `[1, 2, 3]` | 这一例通过，不能据此确认会排序 |
| `[8, 4]` | `[1, 2, 3]` | 暴露了硬编码答案 |
| `[2, 2, 1]` | `[1, 2, 3]` | 暴露了重复元素处理错误 |

现实里的漏洞还可能在答案提取、空测试集、异常返回值和超时处理。**优化器只收到你给的分数，不会自动补全你没写出来的需求。**确定性 verifier 能减少一些主观打分误差，却不能消灭 specification gaming。[Anthropic 的受控研究](https://www.anthropic.com/research/reward-tampering)也提醒我们：奖励和真实目标之间的缝隙可能被利用；这不是所有训练都会发生的必然结果。

所以至少分开训练检查器与独立评估：保留未用于训练的题目和边界用例，检查新的输入类型，必要时加入人工抽查。只换一道题、却保留同一个有漏洞的检查逻辑，并不够独立。

## 真要选方法，先看手里有什么

| 眼下的条件 | 值得先试什么 | 比较时别忘了什么 |
|---|---|---|
| 已有质量不错的静态偏好对 | 标准 DPO 作为基线 | 标注一致性、数据覆盖、held-out preference accuracy 与任务结果 |
| 能生成新回答，并能稳定打分 | PPO 或 GRPO | 相同生成预算、reward 质量、Critic 成本或组内有效信号 |
| 任务有可靠的自动验证办法 | 把 verifier 接入合适的 RL 方法 | 检查器覆盖、投机解、独立测试与推理成本 |
| 多轮环境中有延迟反馈 | 明确状态、终止和 credit assignment，再选更新方法 | 不要把整段轨迹粗暴当成互不相关的回答 |

不用急着问哪个名字更新。先画出数据怎么来、奖励由谁给、loss 怎么算、谁负责最后验收。算法换了以后，这四件事能说清楚，才算真的知道自己换了什么。

接着读 [on-policy / off-policy](on-off-policy.md)，看看旧数据还能怎么用；再用[评估清单](evaluation-and-review.md)检查「分数涨了」能不能支持你的结论。
