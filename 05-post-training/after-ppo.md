# PPO、GRPO 与 DPO：反馈怎样变成一次更新

**中文** · [English](after-ppo.en.md)

> 阅读时间：约 18 分钟 · 类型：教学 · 最近审阅：2026-10-09

<span id="ppo"></span>

## 先看反馈怎样进入训练 {#_1}

这些算法有的换了 advantage 的估计方式，有的换了数据来源或优化目标，并不都能理解成“删掉 PPO 的一个部件”。读它们时，先问：反馈从哪里来，要采多少回答，最后怎样计算梯度？

PPO、GRPO 和 DPO 回答的是同一个问题：已经有了 SFT 模型，怎样利用回答质量或人类偏好，继续提高好回答的概率、降低差回答的概率？区别在于反馈怎样进入训练：

- **PPO**：用回答的 reward、Critic 的 value baseline 和 clipped policy update；
- **GRPO**：同一 prompt 采样一组回答，用组内相对奖励代替 Critic；
- **DPO**：直接读取 chosen/rejected pair，用一个偏好分类损失更新 policy。

### 先用同一道题建立直觉 {#_2}

Prompt 都是“计算 $17\times24$”：

| 方法 | 它看到了什么 | 它怎样得到更新方向 |
| --- | --- | --- |
| PPO | Actor 回答 388，reward $R=0$，Critic 预测 $V=0.6$ | $A=R-V=-0.6$，降低这次轨迹的概率 |
| GRPO | 同题回答 `[408, 388, 408（含过程）, 428]`，reward `[1,0,1,0]` | 组内标准化后约为 $A=[1,-1,1,-1]$，提高相对好的回答、降低相对差的回答 |
| DPO | 固定数据中 `chosen=408, rejected=388` | 让当前 policy 相对于 Reference 更偏爱 chosen |

这三个方法可以各用一句话记住：PPO 问“实际结果比 Critic 预期高多少”；GRPO 问“这次结果比同题其他回答高多少”；DPO 不估 advantage，而是直接问“相对于 Reference，chosen 是否比 rejected 提升得更多”。

## PPO 里，谁在学习，谁在打分？ {#ppo_1}

经典 PPO-RLHF 概念图里有四种角色，其中两个需要训练：

| 模型 | 训不训 | 干什么 |
| --- | --- | --- |
| Policy | 训 | 要产出的那个模型 |
| Critic | **训** | 估状态价值，给优势函数当基线 |
| Reward | 冻 | 给回复打分 |
| Reference | 冻 | KL 锚点，防止跑太远 |

Critic 也要训练，除了前向和反向计算，还要保存优化器状态。它可以是独立的 value model，也可以和 Actor 共享 backbone、只增加 value head。所以表里列的是四种职责，不一定是四个常驻显存的完整模型。接下来的问题是：不用 Critic，能不能也得到有用的比较标准？

## 不用 Critic，baseline 从哪里来？ {#critic}

Critic 估计状态价值，既可以提供降低方差的 baseline，也参与 TD / GAE 的 bootstrapping。若任务适合整条回答的终局评分，可以考虑用采样统计构造相对信号；多步、延迟奖励的情况则要重新考虑 credit assignment。

| 算法 | 基线从哪来 | 代价 |
| --- | --- | --- |
| **GRPO** | 同一个提示采一组回答，用组内均值（并除以标准差） | 每个提示要采多个回答，采样成本上去了 |
| **RLOO** | 留一法：某个样本的基线 = 其余样本奖励的均值 | 同样要多采样；且把整个回复当**一个动作**，放弃 token 级信用分配 |
| **REINFORCE++** | 全局 batch 的统计量做优势归一化 | 基线更粗，但保留了 PPO 的剪切与 KL 稳定化 |

这些方法不再依赖独立训练的 value model，但并没有消除估计误差。省下的训练状态，要和额外采样、信号方差及任务结构一起比较。

RLOO 的基线写出来是这样，$k$ 是同一提示下的采样数：

$$\hat A_i = r_i - \frac{1}{k-1}\sum_{j \neq i} r_j$$

它的立论是：RLHF 的起点是训好的 SFT 模型，不是随机初始化的网络，所以 PPO 里那些为不稳定训练准备的机制（GAE、逐 token 的价值估计）未必必要。

### GRPO 的组相对 advantage {#grpo-advantage}

对同一个 prompt 采样 $G$ 个回答，得到奖励 $R_1,\ldots,R_G$。常见的 response-level advantage 是：

$$
\begin{aligned}
\mu_R&=\frac{1}{G}\sum_{i=1}^{G}R_i,\\
\sigma_R&=\sqrt{\frac{1}{G}\sum_i(R_i-\mu_R)^2},\\
\hat A_i&=\frac{R_i-\mu_R}{\sigma_R+\varepsilon}.
\end{aligned}
$$

如果奖励为 $[1,0,1,0]$，那么 $\mu_R=0.5$、$\sigma_R=0.5$，忽略很小的 $\varepsilon$ 后，优势约为 $[1,-1,1,-1]$。这里用总体标准差；代码库若用样本标准差，数值尺度会不同。奖励可以来自 Reward Model、人类、数学 verifier、单元测试、代码执行或格式检查器。

常见 GRPO 实现把同一回答的 sequence-level advantage 广播给该回答里的 token，再使用 PPO-style clipped objective：

$$
\rho_{i,t}(\theta)=
\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}
{\pi_{\mathrm{old}}(o_{i,t}\mid q,o_{i,<t})},
$$

$$
\begin{aligned}
c_{i,t}&=\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon),\\
s_{i,t}&=\min(\rho_{i,t}\hat A_i,\;c_{i,t}\hat A_i),\\
J_{\mathrm{GRPO}}&=\frac{1}{G}\sum_i\frac{1}{|o_i|}\sum_t s_{i,t}.
\end{aligned}
$$

这里 $G$ 是同一 prompt 的回答数，$|o_i|$ 是第 $i$ 条回答的有效 token 数；$J$ 是要最大化的策略目标，代码通常最小化 $-J$。[DeepSeekMath 原论文 §4.1](https://arxiv.org/html/2402.03300v3#S4.SS1)还加入相对于 Reference 的 KL 项，后来的配方也有设 $\beta=0$ 的。GRPO 主要省掉的是 learned value baseline，不是 rollout、reward 或 policy ratio。具体框架的 loss reduction、KL 系数和更新次数都要单独看，不能只读算法名。

| | PPO baseline | GRPO baseline |
| --- | --- | --- |
| 来源 | Critic 学出的 $V_\phi(s_t)$ | 同 prompt 的组内 reward statistics |
| 粒度 | 可随生成前缀/token 改变 | 常见形式对整条 response 给一个相对 advantage |
| 主要成本 | 训练 value function | 每个 prompt 要做多次 rollout |
| 典型风险 | Critic 拟合不准或训练不稳 | 组内缺少奖励差异，采样没有学习信号 |

## Loss 是 0，为什么模型还能学？ {#zero-loss-gradient}

一组回答有好有坏，训练日志里的 policy loss 却是 0，看起来像没有信号。先别急着改代码：**相互抵消的是几个数，不一定是它们对参数的梯度。**

把问题缩到两个单 token 回答 A、B。采样时各有 50% 的概率，奖励分别是 1、0；为便于手算，使用优势 `[1,-1]`。只用一个参数 $\theta$ 控制概率：$p(A)=\sigma(\theta)$，$p(B)=1-p(A)$。旧概率固定为 0.5。

刚开始 $\theta=0$，两个 ratio 都为 1，尚未进入裁剪的平坦区。要最小化的策略 loss 是：

$$\begin{aligned}
\mathcal L(\theta)&=-\tfrac12\big[2p(A)-2(1-p(A))\big]\\
&=1-2\sigma(\theta),\\
\mathcal L(0)&=0,\qquad \mathcal L'(0)=-0.5.
\end{aligned}$$

Loss 为 0，斜率却不是 0。用学习率 0.2 做一次普通梯度下降，$\theta$ 变成 0.1，A 的概率约为 **0.525**。这只是手工构造的两动作检查，不是一次语言模型训练实验。

这里最容易写错的是分母。`logp - old_logp.detach()` 在刚采样后数值上可以为 0，但分子仍可导；写成 `logp - logp` 则把同一条计算路径减掉，梯度也一起消失。只做一次更新时可以用当前值的 detached 副本；如果复用这批数据多次，必须保留当初的副本，不能每次重算分母。

<details markdown="1">
<summary>用几行 PyTorch 看看这个零 loss</summary>

```python
import torch

theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
logits = torch.stack((theta, torch.zeros_like(theta)))
logp = logits.log_softmax(dim=-1)
old_logp = logp.detach().clone()
advantages = torch.tensor([1.0, -1.0], dtype=torch.float64)
loss = -((logp - old_logp).exp() * advantages).mean()
loss.backward()
assert abs(loss.item()) < 1e-12
assert abs(theta.grad.item() + 0.5) < 1e-12
```

这段只检查第一次更新，省略尚未生效的 clipping。配套 [CPU 脚本](code/policy_gradient_checks.py)另用完整 `min + clamp` 目标验证相同结果，并检查梯度、mask 和归一化。没有调用模型、采样服务或 GPU。

</details>

### 哪些条件变了，初始 loss 就未必是 0？ {#zero-loss-assumptions}

上面的抵消依赖于**完整组、组内中心化、相同 ratio 和所用的平均方式**，不是 GRPO 的运行断言：

| 情况 | 会发生什么 |
| --- | --- |
| 同一组奖励完全相同 | 所有 advantage 都是 0，这次 policy-gradient 项确实没有信号；与“正负项抵消”不同 |
| 把完整组拆到不同 minibatch / rank | 局部日志不一定均值为 0；不要要求每个小批次都显示 0 |
| 两条回答长度分别为 1、3，优势为 `[1,-1]` | ratio 都为 1 时，先各自平均再平均得到 loss 0；按 4 个有效 token 平均得到 0.5 |
| 当前策略等于 old，但不等于 Reference | 策略项可能是 0，KL 项不一定是 0；每轮刷新 old 并不会同时重置 Reference |
| 换了采样温度、推理后端，或用了异步旧轨迹 | 权重文件相同也不保证保存的 log-prob 与训练端一致，先查 ratio 为什么偏离 1 |

长度算例对应的是两种不同的目标，不能靠改日志把它们“修”成相同值。[Dr. GRPO](https://arxiv.org/html/2503.20783v1#S3)讨论了长度与组内标准差带来的权重变化；[TRL 的 loss types](https://huggingface.co/docs/trl/grpo_trainer#loss-types)也区分多种聚合方式。检查配置中的 `loss_type` 和实际 mask，比猜初始 loss 应该是多少更有用。

日志至少把 policy loss、KL、有效 token 数、组内奖励方差和梯度范数分开看。Loss 接近 0 既不证明模型没学，也不证明已经收敛。接下来若想问“被 clip 的 token 还会动吗”，看[完整梯度算例](rlhf/ppo-clipping.md#clipped-token-gradients)；数据为何叫 old，则看[一组回答的生命周期](rlhf/on-off-policy.md#grpo-data-lifecycle)。

## 换成组内比较以后，要注意什么？ {#critic_1}

省去一个模型，不等于省去了判断好坏这件事。现在比较标准来自这一组回答，新的问题也就出在这里。

**问题一：组内奖励没有差异时，基线不含信息。**
一组回答全对或全错，例如 $R=[0,0,0,0]$，减去组内均值后 advantage 都是 0；带有 $\varepsilon$ 的实现避免了除零，但创造不出相对信号。这一组对 policy-gradient 项**不产生有效梯度**。任务偏简单或偏难时，这种组占比会很高。

**问题二：归一化引入偏置。**
除以组内标准差会改变不同 prompt 的相对权重；按回答长度平均又会改变不同长度的 token 权重。[Dr. GRPO](https://arxiv.org/abs/2503.20783)分析了这些偏置。是否改善你自己的任务，仍要按相同采样与评估预算比较，不能把去掉归一化当成普遍正确的修复。

**问题三：同样的 ratio 上界，对低概率 token 意味着什么？**
PPO 在部分分支截平 surrogate 的收益，不是把实际概率比硬限制在区间里。相同乘法上界对低概率 token 对应的绝对增量更小；放宽上界可能改变探索，但熵下降有多种原因，不应只凭一条曲线归因。

**问题四：序列级的损失稀释长回答。**
先在每条回答内部平均、再对回答平均时，一个 1000 token 的回答和一个 50 token 的回答总权重相同。前者每个 token 的 loss 权重只有后者的 1/20；实际梯度大小还取决于 advantage 和模型。若重要的推理步骤集中在长回答里，这种加权方式值得检查，但回答长本身不代表推理更好。

DAPO 围绕采样、clipping、token 权重和超长回答做了组合调整。柔性长度惩罚没有取消生成长度上限，也不等于去掉组内标准差。详细算例见 [DAPO：采样、长度和 clipping](dapo.md)。

## DPO：直接用偏好对训练 {#rl}

DPO 换了一个做法：在标准的离线训练阶段，**不需要显式 Reward Model、Critic 或在线 rollout loop**。

每条数据是 $(x,y_w,y_l)$：prompt、chosen 和 rejected。定义两条回答相对于冻结 Reference 的 log-probability 变化：

$$
\Delta_w=\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)},
$$

$$
\Delta_l=\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}.
$$

DPO 优化：

$$
\mathcal L_{\mathrm{DPO}}
=-\log\sigma\!\left(\beta[\Delta_w-\Delta_l]\right).
$$

不用死记展开式。它只要求：

**相对于 Reference，让 chosen 的提升大于 rejected 的提升。**

为什么不用单独训练 Reward Model？Bradley–Terry 偏好模型写成

$$
\begin{gathered}
\Delta r=r(x,y_w)-r(x,y_l),\\
P(y_w\succ y_l\mid x)=\sigma(\Delta r).
\end{gathered}
$$

而带 KL 约束的奖励最大化，其最优 policy 与 reward 满足

$$
r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+C(x).
$$

代回偏好概率后，$C(x)$ 在同一个 prompt 的 reward difference 中抵消，reward difference 就能直接用 policy/reference log-ratio 表示。因此奖励概念没有消失，而是被**隐式吸收进 policy objective**。完整推导见 [RLHF 的三个阶段](rlhf/after-rlhf.md)。

### DPO 和 SFT 到底差在哪 {#dpo-sft}

如果 A 是 chosen、B 是 rejected：

$$
\mathcal L_{\mathrm{SFT}}=-\log\pi_\theta(A\mid x)
$$

只告诉模型“A 值得模仿”；DPO 同时看到 A 和 B，学习的是“A 相对 B 更受偏好”。chosen 不必是唯一完美答案，原始二元偏好也通常只提供 $A>B$ 的顺序，而不直接告诉模型好多少。

$\beta$ 同时参与 Reference 约束的理论关系和 preference logit 的尺度。把它机械记成“越大越保守”容易出错：不同推导约定、loss 实现和数据尺度会影响观察到的训练行为，面试中最好先写清公式和代码库定义，再讨论调大调小。

标准 DPO 的典型代价是使用**固定的离线偏好对**。策略在训练中一直变，数据却由过去的策略产生，分布错配会逐渐扩大；它也不能主动发现当前 policy 的新失败模式。Online DPO 和迭代式数据刷新可以缓解这一点，所以“DPO 只能离线”并不是算法家族的绝对边界。真正关键的是数据是否跟随当前 policy 更新，以及反馈是否能形成闭环。

两个后续修补：

- **IPO** — 使用不同的偏好目标，使相对 log-ratio 拟合有限的目标间隔；不是简单在 DPO 后面加一个通用正则项。[原论文](https://arxiv.org/abs/2310.12036)
- **KTO** — 可以用非成对的 desirable / undesirable 标签；数据形式更灵活，但成本仍取决于实际标注流程，不能保证一定更便宜。

## RLVR：让程序检查答案 {#_3}

数学题可以对答案，代码可以跑测试。这类任务的奖励**不需要学**，写个检查器就行。

这一步删掉的是 Reward Model。确定性检查器消除了一个主要攻击面——learned Reward Model 的近似误差——但检查器仍可能写错目标或留下漏洞。可验证奖励缩小了 reward hacking 的空间，并不自动保证 objective 正确。

局限也明显：只适用于能写出检查器的任务。

## 一张表收尾 {#_4}

| | 典型训练数据 | 显式 Reward | Critic | 训练时 rollout | 核心取舍 |
| --- | --- | --- | --- | --- | --- |
| PPO | prompt + 当前/近期 policy 的回答 | RM、规则或环境 | 需要 | 需要 | 反馈灵活、可在线探索；系统最复杂 |
| GRPO | prompt + 同题一组回答 | RM、verifier 或环境 | 不需要 | 需要，而且每题多次 | 省 Critic；依赖组内差异并增加生成成本 |
| DPO | 固定 chosen/rejected pairs | 不显式调用，reward difference 隐含在 loss 中 | 不需要 | 标准离线训练不需要 | 简单稳定；受偏好数据覆盖和分布错配限制 |

再看扩展方法，就不必只数“少了几个模型”：RLOO 改的是 baseline 的估计，REINFORCE++ 使用 batch statistics，RLVR 改的是奖励来源，DAPO 则调整了采样、clipping、token weighting 和超长回答的处理。这些选择不全在同一层，也不一定互斥。

## 怎么选 {#_5}

<details class="interview" markdown="1">
<summary>第一步：奖励能不能被自动验证？</summary>

**能验证时，优先考虑 verifier / RLVR。** 数学答案、代码测试、结构约束等可以由程序稳定判断，不必先拟合一个 Reward Model。但仍要审计 verifier 是否只检查最终答案、是否存在测试漏洞，以及它是否真的代表产品目标。

</details>

<details class="interview" markdown="1">
<summary>第二步：训练时能不能持续从当前 policy 采样？</summary>

只有静态 chosen/rejected pairs 时，DPO 系通常最直接，但要接受覆盖不足与 distribution mismatch。能够持续 rollout、获得 reward 或环境反馈时，PPO / GRPO 一类 online method 能发现当前 policy 的新失败；代价是生成计算、系统复杂度和训练方差都更高。

</details>

<details class="interview" markdown="1">
<summary>第三步：真正的资源或优化瓶颈是什么？</summary>

Critic 的显存、计算或训练稳定性是瓶颈时，考虑 GRPO、RLOO 等 critic-free 方法。长回答被稀释、低概率 token 难以恢复或 entropy 持续下降时，再考虑 DAPO 的 token-level loss、Clip-Higher、动态采样和长度处理。不要先挑算法名，再回头勉强适配数据。

</details>

顺序是：**先看 feedback 是否可靠，再看能否形成 online loop，最后看系统瓶颈。**

## 选择算法时检查什么 {#_6}

<details class="interview" markdown="1">
<summary>1. Reward 是 learned 还是 verified？分别监控什么？</summary>

Learned Reward Model 要监控 reward hacking、长度偏好、风格捷径以及与人工判断的相关性是否随 policy 漂移。Verifier 要审计 specification gap：测试是否覆盖真实要求、只验最终答案是否奖励了猜测、模型能否钻格式或执行环境的漏洞。

</details>

<details class="interview" markdown="1">
<summary>2. GRPO 中有多少组全对或全错？</summary>

直接记录 zero-variance group rate。组内 reward 完全相同时，标准化 advantage 为 0，这些 rollout 不提供相对梯度。比例过高时，可以调整题目难度、增加组大小、改善 reward 分辨率，或用 dynamic sampling 过滤无信息组；但过滤也会改变训练数据分布，需要同时监控。

</details>

<details class="interview" markdown="1">
<summary>3. Reward 与长度归一化引入了什么偏置？</summary>

同时画 reward、正确率、response length、entropy 和每组 reward variance 的训练曲线。按组标准差归一化会让低方差组获得不同尺度的权重；按 response 长度平均则可能稀释长序列 token。若长度持续增长但正确率不涨，模型可能在优化归一化或 reward 的捷径。

</details>

<details class="interview" markdown="1">
<summary>4. 长回答的 token 梯度是否被短回答压过？</summary>

先确认 loss 是 per-sequence mean 还是全 batch 的 per-token aggregation。若每条 response 总权重相同，1000-token 回答中的单个 token 通常比 50-token 回答中的 token 分到更小权重。需要长推理时，可考虑 token-level loss aggregation，并分别报告按 response 和按 token 的指标。

</details>

<details class="interview" markdown="1">
<summary>5. DPO 偏好数据来自哪个 policy？为什么重要？</summary>

记录数据生成 checkpoint、sampling temperature、解码约束和标注时间。当前 policy 离数据生成 policy 越远，固定 pair 越难覆盖它现在会犯的错。用 held-out prompt 检查覆盖范围，并在明显漂移后重新采样、重新标注或切换到 iterative / online preference loop。

</details>

<details class="interview" markdown="1">
<summary>6. DPO 能完全替代 PPO / GRPO 吗？</summary>

不能一概而论。标准 DPO 用固定偏好对，便宜、稳定，适合高质量离线数据已经覆盖目标行为的场景；PPO / GRPO 能从当前 policy rollout 并接收新 reward，更适合 exploration、环境交互和多步可验证任务，但训练更贵、更不稳定。关键区别是 feedback loop，而不只是 loss 名字。

</details>

## 面试时能否两分钟讲清楚 {#_7}

<details class="interview" markdown="1">
<summary>请写出 PPO、GRPO 与 DPO 最关键的三个式子。</summary>

PPO 的核心方向是 $\hat A_t\approx G_t-V_\phi(s_t)$，再用 probability ratio 的 clipped surrogate 限制单次更新。GRPO 把 learned value baseline 换成 $\hat A_i=(R_i-\mu_R)/(\sigma_R+\varepsilon)$。DPO 使用 $-\log\sigma\!\left(\beta[\Delta_w-\Delta_l]\right)$，其中 $\Delta$ 是当前 policy 相对 Reference 的 log-probability 变化。

</details>

<details class="interview" markdown="1">
<summary>为什么 GRPO 省掉 Critic，却没有省掉 rollout？</summary>

它仍要让当前或近期 policy 对同一 prompt 生成多个回答，并给每个回答计算 reward；否则没有组内相对 baseline。它省掉的是训练 $V_\phi(s)$ 的 Critic。若同组全对或全错，$R_i-\mu_R=0$，所以这一组没有相对 policy-gradient signal。

</details>

<details class="interview" markdown="1">
<summary>DPO 为什么不需要显式 Reward Model？</summary>

带 KL 约束的最优策略满足 $r(x,y)=\beta\log[\pi^*(y\mid x)/\pi_{\mathrm{ref}}(y\mid x)]+C(x)$。把它代入 Bradley–Terry 偏好概率时，同一 prompt 的 $C(x)$ 抵消，于是 reward difference 可以直接写成 chosen/rejected 的 policy/reference log-ratio。Reward 没消失，而是隐含在 DPO loss 中。

</details>

<details class="interview" markdown="1">
<summary>两分钟结论：DPO 能完全替代 PPO 吗？</summary>

不能。DPO 适合高质量、覆盖充分的固定偏好对，训练像监督学习一样简单稳定；它不会自动探索当前 policy 的新失败。PPO / GRPO 可以持续从当前 policy 采样并接收环境、Reward Model 或 verifier 的反馈，更适合需要 exploration 和多步结果的任务，但 rollout 更贵，优化也更难稳定。Online DPO 说明真正的分界在数据是否随 policy 更新，而不是算法名称本身。

</details>

## 继续阅读 {#_8}

- [RLHF 的三个阶段，和后来发生了什么](rlhf/)：四个模型各自在干嘛，DPO 的推导
- [数据与反馈](../01-data-and-feedback/)：偏好标签本身的质量问题
- [Evaluation](../07-evaluation/)：怎么判断对齐之后真的变好了

## 快速学习：PPO、GRPO、DPO 在删什么 {#ppogrpodpo}

<details class="interview" markdown="1">
<summary>先看数据是否 online、奖励是否可验证，再选算法</summary>

**快速记忆**：PPO 有 Actor + Critic + RM + Reference；GRPO 用同 prompt 的组相对 baseline 删 Critic；DPO 用静态 chosen/rejected pairs 连显式 RM 与 rollout loop 一起删。

**面试回答**

> DPO 便宜稳定，但主要学习离线偏好对，不能自动探索当前 policy 的新失败。GRPO 仍需 online rollout，只是用组内相对 reward 估 advantage。奖励可验证且需要多步探索时 online RL 更自然；只有静态偏好数据时 DPO 更合适。

<details markdown="1">
<summary><b>深挖</b>：GRPO 组内全对或全错为什么没有梯度信号？</summary>

组标准化 advantage 使用 $(r_i-\bar r)/s_r$。若整组 reward 相同，去均值后全部为 0；实现即使给分母加 epsilon，也没有相对优劣可学。高比例的 homogeneous groups 说明 rollout 难度或采样多样性需要调整。

</details>
</details>

## 参考论文 {#_9}

- [PPO](https://arxiv.org/abs/1707.06347) — 剪切目标与信任域
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — GRPO
- [DAPO](https://arxiv.org/abs/2503.14476) — 针对 GRPO 四个失效模式的四条改进
- [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783) — Dr. GRPO，归一化项引入的长度偏置
- [Back to Basics](https://arxiv.org/abs/2402.14740) — RLOO
- [REINFORCE++](https://arxiv.org/abs/2501.03262) — 去 Critic 但保留 PPO 的稳定化技巧
- [DPO](https://arxiv.org/abs/2305.18290) · [IPO](https://arxiv.org/abs/2310.12036) · [KTO](https://arxiv.org/abs/2402.01306)

## 中文导读 {#_10}

- [大模型中的强化学习](https://zhuanlan.zhihu.com/p/693582342) — 知乎 @大家好我是爱因，专栏《机器学习小王子》。本章按「删掉了什么」这一条轴组织，只覆盖主干；那篇是百科式的全景，从 MDP 要素、贝尔曼方程、MC/TD/GAE 一路铺到各算法的细节，想要更全的地图从它开始。
