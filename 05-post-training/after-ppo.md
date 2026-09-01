# PPO 之后：每个算法都在删掉它的一部分

**中文** · [English](after-ppo.en.md)

> 阅读时间：约 11 分钟 · 类型：教学 · 最近审阅：2026-09

## 先看这些算法在简化什么

PPO 之后出现的一长串算法，看起来像各自独立的发明，其实是同一件事的不同程度：**把 PPO 的某个部件删掉，然后处理删掉之后冒出来的问题。** 搞清楚每个删了什么、代价是什么，这张表就不用背了。

PPO、GRPO 和 DPO 回答的是同一个问题：已经有了 SFT 模型，怎样利用回答质量或人类偏好，继续提高好回答的概率、降低差回答的概率？区别在于反馈怎样进入训练：

- **PPO**：用回答的 reward、Critic 的 value baseline 和 clipped policy update；
- **GRPO**：同一 prompt 采样一组回答，用组内相对奖励代替 Critic；
- **DPO**：直接读取 chosen/rejected pair，用一个偏好分类损失更新 policy。

### 先用同一道题建立直觉

Prompt 都是“计算 $17\times24$”：

| 方法 | 它看到了什么 | 它怎样得到更新方向 |
| --- | --- | --- |
| PPO | Actor 回答 388，reward $R=0$，Critic 预测 $V=0.6$ | $A=R-V=-0.6$，降低这次轨迹的概率 |
| GRPO | 同题回答 `[408, 388, 408（含过程）, 428]`，reward `[1,0,1,0]` | 组内标准化后 $A=[1,-1,1,-1]$，提高相对好的回答、降低相对差的回答 |
| DPO | 固定数据中 `chosen=408, rejected=388` | 让当前 policy 相对于 Reference 更偏爱 chosen |

这三个方法可以各用一句话记住：PPO 问“实际结果比 Critic 预期高多少”；GRPO 问“这次结果比同题其他回答高多少”；DPO 不估 advantage，而是直接问“相对于 Reference，chosen 是否比 rejected 提升得更多”。

## 先看 PPO 有什么可删的

经典 PPO-RLHF 概念图里有四种角色，其中两个需要训练：

| 模型 | 训不训 | 干什么 |
| --- | --- | --- |
| Policy | 训 | 要产出的那个模型 |
| Critic | **训** | 估状态价值，给优势函数当基线 |
| Reward | 冻 | 给回复打分 |
| Reference | 冻 | KL 锚点，防止跑太远 |

**Critic 是额外成本最大的角色之一**：它跟随当前 policy 训练，还要保存优化器状态。工程上它可以是独立 value model，也可以和 Actor 共享 backbone、只增加 value head；这里的四项首先是概念角色，不保证是四个独立常驻显存的完整模型。后面的故事仍然大多围绕如何省掉 Critic 展开。

## 主线：删掉 Critic

Critic 存在的唯一理由是**给优势函数提供一个基线**，用来降方差。既然只是要一个基线，那不一定非得学一个网络出来。

| 算法 | 基线从哪来 | 代价 |
| --- | --- | --- |
| **GRPO** | 同一个提示采一组回答，用组内均值（并除以标准差） | 每个提示要采多个回答，采样成本上去了 |
| **RLOO** | 留一法：某个样本的基线 = 其余样本奖励的均值 | 同样要多采样；且把整个回复当**一个动作**，放弃 token 级信用分配 |
| **REINFORCE++** | 全局 batch 的统计量做优势归一化 | 基线更粗，但保留了 PPO 的剪切与 KL 稳定化 |

三者的共同点是：**基线从"学出来的"变成"采样估出来的"。** 这一步省掉了一个在训的全尺寸模型，但也埋下了后面所有问题的种子。

RLOO 的基线写出来是这样，$k$ 是同一提示下的采样数：

$$\hat A_i = r_i - \frac{1}{k-1}\sum_{j \neq i} r_j$$

它的立论是：RLHF 的起点是训好的 SFT 模型，不是随机初始化的网络，所以 PPO 里那些为不稳定训练准备的机制（GAE、逐 token 的价值估计）未必必要。

### GRPO 的组相对 advantage

对同一个 prompt 采样 $G$ 个回答，得到奖励 $R_1,\ldots,R_G$。常见的 response-level advantage 是：

$$
\mu_R=\frac{1}{G}\sum_{i=1}^{G}R_i,\qquad
\sigma_R=\sqrt{\frac{1}{G}\sum_i(R_i-\mu_R)^2},\qquad
\hat A_i=\frac{R_i-\mu_R}{\sigma_R+\varepsilon}.
$$

如果奖励为 $[1,0,1,0]$，那么 $\mu_R=0.5$、$\sigma_R=0.5$，优势就是 $[1,-1,1,-1]$。奖励可以来自 Reward Model、人类、数学 verifier、单元测试、代码执行或格式检查器。

常见 GRPO 实现把同一回答的 sequence-level advantage 广播给该回答里的 token，再使用 PPO-style clipped objective：

$$
\rho_{i,t}(\theta)=
\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}
{\pi_{\mathrm{old}}(o_{i,t}\mid q,o_{i,<t})},
$$

$$
L_{\mathrm{GRPO}}=
\frac{1}{G}\sum_i\frac{1}{|o_i|}\sum_t
\min\!\left(
\rho_{i,t}\hat A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_i
\right),
$$

并通常加上相对于冻结 Reference 的 KL 约束。GRPO 因而没有删掉 rollout、reward、policy ratio 或稳定化约束；它主要删掉的是 learned value baseline。

| | PPO baseline | GRPO baseline |
| --- | --- | --- |
| 来源 | Critic 学出的 $V_\phi(s_t)$ | 同 prompt 的组内 reward statistics |
| 粒度 | 可随生成前缀/token 改变 | 常见形式对整条 response 给一个相对 advantage |
| 主要成本 | 训练 value function | 每个 prompt 要做多次 rollout |
| 典型风险 | Critic 拟合不准或训练不稳 | 组内缺少奖励差异，采样没有学习信号 |

## 删掉 Critic 之后冒出来的问题

这一节才是重点。**采样基线和学出来的基线，失效方式不一样。**

**问题一：组内奖励没有差异时，基线不含信息。**
一组回答全对或全错，例如 $R=[0,0,0,0]$，减去组内均值后 advantage 都是 0；带有 $\varepsilon$ 的实现避免了除零，但创造不出相对信号。这一组对 policy-gradient 项**不产生有效梯度**。任务偏简单或偏难时，这种组占比会很高。

**问题二：归一化引入偏置。**
除以组内标准差看着是标准化，其实给不同的组加了不同的权重；按序列长度归一化则会系统性地偏向某个长度方向。Dr. GRPO 的论点就是这个：这些归一化项让模型倾向于**产出越来越长但不一定更对**的回答。去掉它们，token 效率明显改善。

**问题三：剪切上下限对称，低概率 token 提不上来。**
PPO 的剪切把概率比限制在 $[1-\epsilon, 1+\epsilon]$。对一个当前概率很低的 token，上限 $1+\epsilon$ 允许的绝对提升量非常小——它几乎没有翻身机会。长期后果是熵坍塌：模型越来越确定，多样性越来越低。

**问题四：序列级的损失稀释长回答。**
损失在样本级平均时，一个 1000 token 的回答和一个 50 token 的回答权重一样，于是长回答里每个 token 拿到的梯度被摊薄了。推理任务恰恰依赖长回答。

DAPO 的四条改进正好对着这四个问题：动态采样过滤掉全对全错的组、剪切上下限拆开（Clip-Higher）、损失改成 token 级、超长回答用柔性惩罚而不是硬截断。

## 另一条线：连 RL 循环一起删

DPO 走得更远——在标准的离线训练阶段，**不需要显式 Reward Model、Critic 或在线 rollout loop**。

每条数据是 $(x,y_w,y_l)$：prompt、chosen 和 rejected。定义两条回答相对于冻结 Reference 的 log-probability 变化：

$$
\Delta_w=\log\pi_\theta(y_w\mid x)-\log\pi_{\mathrm{ref}}(y_w\mid x),
$$

$$
\Delta_l=\log\pi_\theta(y_l\mid x)-\log\pi_{\mathrm{ref}}(y_l\mid x).
$$

DPO 优化：

$$
\mathcal L_{\mathrm{DPO}}
=-\log\sigma\!\left(\beta[\Delta_w-\Delta_l]\right).
$$

不用死记展开式。它只要求：

$$
\boxed{\text{相对于 Reference，chosen 的提升要大于 rejected 的提升。}}
$$

为什么不用单独训练 Reward Model？Bradley–Terry 偏好模型写成

$$
P(y_w\succ y_l\mid x)=\sigma\!\left(r(x,y_w)-r(x,y_l)\right).
$$

而带 KL 约束的奖励最大化，其最优 policy 与 reward 满足

$$
r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+C(x).
$$

代回偏好概率后，$C(x)$ 在同一个 prompt 的 reward difference 中抵消，reward difference 就能直接用 policy/reference log-ratio 表示。因此奖励概念没有消失，而是被**隐式吸收进 policy objective**。完整推导见 [RLHF 的三个阶段](rlhf-pipeline.md)。

### DPO 和 SFT 到底差在哪

如果 A 是 chosen、B 是 rejected：

$$
\mathcal L_{\mathrm{SFT}}=-\log\pi_\theta(A\mid x)
$$

只告诉模型“A 值得模仿”；DPO 同时看到 A 和 B，学习的是“A 相对 B 更受偏好”。chosen 不必是唯一完美答案，原始二元偏好也通常只提供 $A>B$ 的顺序，而不直接告诉模型好多少。

$\beta$ 同时参与 Reference 约束的理论关系和 preference logit 的尺度。把它机械记成“越大越保守”容易出错：不同推导约定、loss 实现和数据尺度会影响观察到的训练行为，面试中最好先写清公式和代码库定义，再讨论调大调小。

标准 DPO 的典型代价是使用**固定的离线偏好对**。策略在训练中一直变，数据却由过去的策略产生，分布错配会逐渐扩大；它也不能主动发现当前 policy 的新失败模式。Online DPO 和迭代式数据刷新可以缓解这一点，所以“DPO 只能离线”并不是算法家族的绝对边界。真正关键的是数据是否跟随当前 policy 更新，以及反馈是否能形成闭环。

两个后续修补：

- **IPO** — DPO 在偏好数据上过拟合得很快。IPO 加了一个正则项，让模型不靠"提前停止"这类技巧也能收敛。
- **KTO** — 不要求成对偏好，只要求把样本标成"好"或"坏"。数据采集成本低一个档次，代价是信号更粗。

## 还有一条：把奖励模型换成程序

数学题可以对答案，代码可以跑测试。这类任务的奖励**不需要学**，写个检查器就行。

这一步删掉的是 Reward Model。确定性检查器消除了一个主要攻击面——learned Reward Model 的近似误差——但检查器仍可能写错目标或留下漏洞。可验证奖励缩小了 reward hacking 的空间，并不自动保证 objective 正确。

局限也明显：只适用于能写出检查器的任务。

## 一张表收尾

| | 典型训练数据 | 显式 Reward | Critic | 训练时 rollout | 核心取舍 |
| --- | --- | --- | --- | --- | --- |
| PPO | prompt + 当前/近期 policy 的回答 | RM、规则或环境 | 需要 | 需要 | 反馈灵活、可在线探索；系统最复杂 |
| GRPO | prompt + 同题一组回答 | RM、verifier 或环境 | 不需要 | 需要，而且每题多次 | 省 Critic；依赖组内差异并增加生成成本 |
| DPO | 固定 chosen/rejected pairs | 不显式调用，reward difference 隐含在 loss 中 | 不需要 | 标准离线训练不需要 | 简单稳定；受偏好数据覆盖和分布错配限制 |

扩展算法可以继续用“删掉什么”定位：RLOO 用 leave-one-out baseline 替代 Critic；REINFORCE++ 用 batch statistics；RLVR 用 verifier 替代 learned Reward Model；DAPO 则不再只删组件，而是针对 GRPO 的采样、clipping、token weighting 和超长回答逐项修补。

## 怎么选

**先问奖励能不能验证。** 能写检查器就写，这一步的收益比换算法大得多。

**再问能不能 online 采样。** 不能（只有静态偏好数据）→ DPO 系，并接受分布错配。能 → GRPO 系。

**最后才问删哪个部件。** 显存是瓶颈就去 Critic；输出很长、要防熵坍塌，就把 DAPO 那几条修补一起上。

顺序别反过来——**先挑算法再回头适配数据，是这类项目最常见的返工来源。**

## 选择算法时检查什么

1. 我的奖励是学出来的还是验证出来的？如果是学出来的，我有没有在盯 reward hacking？
2. 用了组相对基线的话，有多少比例的组是全对或全错？那部分数据是白采的。
3. 我的归一化项引入了什么偏置？回答长度在训练过程中是不是单调变长？
4. 长回答里每个 token 拿到的梯度，和短回答比是不是被摊薄了？
5. 如果用 DPO：我的偏好数据是哪个策略产生的？和现在的策略差多远？
6. 如果有人说“DPO 完全替代 PPO”，他是否忽略了 online exploration、环境交互和当前 policy 的数据闭环？

## 面试时能否两分钟讲清楚

1. 不看笔记写出 PPO 的 $A\approx G-V$、GRPO 的组标准化 advantage 和 DPO loss。
2. 解释为什么 GRPO 省掉 Critic，却没有省掉 rollout；全对或全错的一组为什么学不到东西。
3. 解释 DPO 为什么不需要显式 Reward Model，以及“reward 被隐式吸收”到底指哪条等式。
4. 回答 DPO 能不能完全替代 PPO：固定偏好对便宜稳定，但不会自动探索当前 policy 的新失败；需要在线采样、环境反馈或多步可验证结果时，PPO/GRPO 一类 online method 更自然，也更贵、更难稳定。

## 继续阅读

- [RLHF 的三个阶段，和后来发生了什么](rlhf-pipeline.md)：四个模型各自在干嘛，DPO 的推导
- [数据与反馈](../01-data-and-feedback/)：偏好标签本身的质量问题
- [Evaluation](../07-evaluation/)：怎么判断对齐之后真的变好了

## 参考论文

- [PPO](https://arxiv.org/abs/1707.06347) — 剪切目标与信任域
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — GRPO
- [DAPO](https://arxiv.org/abs/2503.14476) — 针对 GRPO 四个失效模式的四条改进
- [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783) — Dr. GRPO，归一化项引入的长度偏置
- [Back to Basics](https://arxiv.org/abs/2402.14740) — RLOO
- [REINFORCE++](https://arxiv.org/abs/2501.03262) — 去 Critic 但保留 PPO 的稳定化技巧
- [DPO](https://arxiv.org/abs/2305.18290) · [IPO](https://arxiv.org/abs/2310.12036) · [KTO](https://arxiv.org/abs/2402.01306)

## 中文导读

- [大模型中的强化学习](https://zhuanlan.zhihu.com/p/693582342) — 知乎 @大家好我是爱因，专栏《机器学习小王子》。本章按「删掉了什么」这一条轴组织，只覆盖主干；那篇是百科式的全景，从 MDP 要素、贝尔曼方程、MC/TD/GAE 一路铺到各算法的细节，想要更全的地图从它开始。
