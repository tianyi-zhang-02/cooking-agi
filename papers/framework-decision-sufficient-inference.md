# 非平稳环境下的决策充分推断:一个统一框架(草稿)

> 工作名:Decision-Sufficient Inference under Non-stationarity
> 状态:框架草稿,未验证。所有标为"猜想"的陈述都还没有证明。

## 0. 一句话

一个决策者在世界会变的情况下,必须**记住什么、发现什么、去问什么、怎样被打分**,这四个问题可以用同一个量度量:

$$
\delta_{\mathcal D}(\phi_t)
= \sup_{L\in\mathcal D}\ \mathbb E\Big[\, r\big(\bar b(\phi_t),L\big) - r\big(b_t,L\big)\Big]
$$

即"用压缩状态 $\phi_t$ 代替完整 belief $b_t$ 做决策,在你关心的那族损失 $\mathcal D$ 上平均多亏多少"。它是 Le Cam deficiency 限制到 $\mathcal D$ 上的版本。

整个框架的主张是:可辨识性、变点检测、信息价值、校准、非平稳 regret,都是这一个量在不同切片下的读数。

## 1. 设定与记号

**环境。** 离散时间,有限集合:

| 符号 | 含义 |
| --- | --- |
| $z_t\in\mathcal Z$ | 隐藏 regime(决定当前动力学) |
| $x_t\in\mathcal X$ | 隐藏状态 |
| $a_t\in\mathcal A$ | 行动(含 act / probe / wait / abstain) |
| $o_t\in\mathcal O$ | 观测 |

动力学:

$$
z_{t+1}\sim H(\cdot\mid z_t),\qquad
x_{t+1}\sim T_{z_t}(\cdot\mid x_t,a_t),\qquad
o_{t+1}\sim O_{z_{t+1}}(\cdot\mid x_{t+1},a_t).
$$

$H$ 是 regime 切换核(几何 hazard 是特例)。观测允许依赖上一步行动,这样"probe"才能有意义。

**历史与 belief。** $h_t=(o_1,a_1,\dots,a_{t-1},o_t)$。记 $\theta_t=(z_t,x_t)\in\Theta=\mathcal Z\times\mathcal X$,完整 belief

$$
b_t=P(\theta_t\mid h_t)\in\Delta(\Theta).
$$

$b_t$ 是最优控制的充分统计量(Åström 1965),这是已知的;它的问题是太大,且维护它需要知道 $T,O,H$。

**决策族。** $\mathcal D$ 是一族一步损失 $L:\Theta\times\mathcal A\to[0,1]$。给定 belief $b$,贝叶斯风险与贝叶斯行动:

$$
r(b,L)=\min_{a}\langle b,\,L(\cdot,a)\rangle,\qquad
a^*_L(b)\in\arg\min_a\langle b,\,L(\cdot,a)\rangle.
$$

$r(\cdot,L)$ 是单纯形上的分片线性凹函数;每个 $L$ 把单纯形切成若干多面体胞腔,每个胞腔对应一个最优行动。

**状态统计量。** $\phi:\text{histories}\to\mathcal S$,并要求**递归性**:存在更新映射 $U$ 使

$$
\phi(h_{t+1})=U\big(\phi(h_t),\,a_t,\,o_{t+1}\big).
$$

给定 $\phi(h_t)=s$,粗化 belief 为 $\bar b(s)=P(\theta_t\mid\phi(h_t)=s)=\mathbb E[b_t\mid\phi(h_t)=s]$。

## 2. 核心对象:决策受限 deficiency

**定义 2.1(静态 $\mathcal D$-充分)。** $\phi$ 在时刻 $t$ 是 $\mathcal D$-充分的,若 $\delta_{\mathcal D}(\phi_t)=0$。

**定义 2.2(递归 $\mathcal D$-充分状态)。** $\phi$ 递归且在每个时刻 $\mathcal D$-充分。

**命题 2.3(充分性的几何刻画)。** 由 $r(\cdot,L)$ 的凹性,$r(\bar b(s),L)\ge\mathbb E[r(b_t,L)\mid s]$,等号成立当且仅当 $\phi$ 的这条纤维内几乎所有 belief 共享同一个 $L$-最优行动。因此:

> $\phi$ 静态 $\mathcal D$-充分 $\iff$ 对每个 $L\in\mathcal D$,$\phi$ 的每条纤维落在 $L$ 的单个最优行动胞腔内(允许边界处平局)。

推论:**最小静态 $\mathcal D$-充分统计量就是贝叶斯行动向量** $b\mapsto\big(a^*_L(b)\big)_{L\in\mathcal D}$。这给出一个谱:

- $\mathcal D=\{L\}$ 单个损失:最小统计量退化为一个行动,几乎不含信息;
- $\mathcal D$ = 全体损失:最小统计量是完整后验;
- $\mathcal D=\{\ell(g(\theta),a)\}$ 只通过 $g(\theta)$ 依赖 $\theta$:$g(\theta)$ 的后验就充分,$\theta$ 中 $g$ 之外的部分可丢弃。

**命题 2.4(递归版本)。** 递归 $\mathcal D$-充分的最粗划分存在且唯一:它是"贝叶斯行动向量划分"在更新映射下的最粗闭包(与 bisimulation 的最粗划分同构)。有限视野、有限观测时可达 belief 集有限,可用划分细化算法精确计算。

**命题 2.5(有限情形可计算)。** $\Theta,\mathcal A,\mathcal D$ 有限时,$\delta_{\mathcal D}(\phi)$ 是一个有限维线性规划;先验取 sup 的 Le Cam 版本同样是 LP。

**Le Cam 与 Blackwell 的关系。** 原始 Le Cam deficiency 是 $\delta(\mathcal E,\mathcal F)=\inf_K\sup_\theta\|K\circ P_\theta-Q_\theta\|_{TV}$,其随机化准则说 $\delta\le\varepsilon$ 当且仅当在所有 $[0,1]$ 损失上贝叶斯风险差不超过 $\varepsilon$。定义 2.1 把"所有损失"换成 $\mathcal D$,把"所有先验"换成真实先验(simulator 已知时可算);需要分布无关版本时改回 sup。

## 3. 五个侧面

### 3.1 可辨识性 → $\mathcal D$-等价类上的可辨识性

**定义 3.1.** regime $z\sim_{\mathcal D}z'$,若对所有 $L\in\mathcal D$ 和所有 $\mathcal X$ 上的 belief $\beta$,把 $z$ 换成 $z'$ 不改变贝叶斯行动:$a^*_L(\beta\otimes\delta_z)=a^*_L(\beta\otimes\delta_{z'})$(在平局意义下)。

**猜想 3.2.** 最小递归 $\mathcal D$-充分状态只需跟踪商空间 $\mathcal Z/\!\sim_{\mathcal D}$ 上的后验。分不清但在 $\mathcal D$ 上后果相同的 regime,不构成不可辨识。

这是"部分可辨识性"的严格版本,也是"decision-sufficient belief 不需要重建完整 regime"的理论依据。

### 3.2 变点检测 → 决策相关的变点

要检测的不是"观测分布变了",而是"$\mathcal D$-等价类变了"。这里有一个必须诚实处理的细节:

- **需要检测什么**由 $\mathcal D$ 决定(等价类是否跨越);
- **能多快检测**由观测流的可分辨性决定,而观测流依赖于行动策略(persistent excitation)。

**猜想 3.3(决策相关的检测下界)。** 对两个不 $\mathcal D$-等价的 regime 类 $[z],[z']$,任何策略 $\pi$ 下的检测延迟满足

$$
\text{delay}\ \gtrsim\ \frac{\log\gamma}{\max_{\pi}\ \mathrm{KL}\big(\text{obs law under }[z],\pi\ \big\|\ \text{obs law under }[z'],\pi\big)},
$$

其中 $\gamma$ 是平均误报间隔约束。分母对策略取 max,就是"最优探测"的贡献;若 max 为 0,则该切换不可检测,只能靠 probe 改变观测律。

**推论(如果 3.3 成立)。** 可以事先算出哪些 regime 切换根本不需要检测(同一等价类内),哪些必须靠主动探测才能检测(被动 KL 为零)。这是整个框架里解释力最强的一条。

### 3.3 Blackwell / 信息价值 → deficiency 的下降量

一次探测 $q$ 的价值定义为它能让 $\delta_{\mathcal D}$ 降多少减去成本:

$$
\mathrm{VOI}_{\mathcal D}(q)=\delta_{\mathcal D}(\phi_t)-\mathbb E_{o\sim q}\big[\delta_{\mathcal D}(\phi_{t+1})\big]-c(q).
$$

当 $\mathcal D=\{\text{log loss}\}$ 时退化为 EIG。Blackwell 定理保证:**没有任何标量信息度量能对所有 $\mathcal D$ 同时正确**,所以 EIG 型 reward 在 $\mathcal D$ 不是 log loss 时会系统性地问错问题——最能降熵的观测未必翻转最优行动。

### 3.4 校准 → 对 $\mathcal D$ 校准

不要求全局校准,只要求:对每个 $L\in\mathcal D$,按预测 $p$ 采取贝叶斯行动后,实际损失等于预测损失:

$$
\mathbb E\big[L(\theta,a^*_L(p))\big]=\mathbb E\big[\langle p,L(\cdot,a^*_L(p))\rangle\big].
$$

这已有名字:**decision calibration**(Zhao 等 2021)和 **omniprediction**(Gopalan 等 2022,一个预测器同时对一族损失最优)。二者是 $\mathcal D$-充分性在学习理论中的化身,但都是**静态、单步、无行动、无非平稳**。

训练上的含义:用 log loss 拟合真实后验是严格 proper scoring,最优解唯一且为真后验——这是"SFT on exact posterior"的理论理由;但对 $\mathcal D$ 校准的要求更弱,允许更小的模型。

### 3.5 非平稳 regret → deficiency 的时间积分

**猜想 3.4(deficiency 控制 regret)。** 若 $\phi$ 在每步满足 $\delta_{\mathcal D}(\phi_t)\le\varepsilon$,且递归更新在自身纤维上精确(或误差 $\le\eta$),则基于 $\phi$ 行动相对于贝叶斯最优(BAPOMDP)策略的动态 regret 满足

$$
\mathrm{Regret}_T\ \le\ \varepsilon T+C\cdot\eta\cdot T\cdot(\text{horizon})
$$

(一步损失情形;折扣多步情形应有 AIS 型的 $\varepsilon/(1-\gamma)$ 形式)。

变点之后的 recovery half-life 就是 $\delta_{\mathcal D}(\phi_t)$ 回落到 $\varepsilon$ 以下的时间,并可分解为

$$
\text{检测延迟(3.2)}+\text{重新辨识的代价(3.1)}+\text{探测的信息成本(3.3)}+\text{利用阶段的损失}.
$$

各项可加性与紧性未证。

## 4. 结果路线图

| 编号 | 陈述 | 难度 | 状态 |
| --- | --- | --- | --- |
| R1 | 有限情形最小递归 $\mathcal D$-充分状态的精确刻画与算法;状态大小随 $\mathcal D$ 单调变化的曲线 | 低 | 命题 2.3–2.5 基本给出,需实现验证 |
| R2 | deficiency 控制 regret(猜想 3.4) | 中 | 需 AIS 型论证 |
| R3 | 决策相关的变点下界(猜想 3.3) | 中高 | 需处理策略依赖的 KL |
| R4 | 序贯 omniprediction:有行动、有状态更新、regime 会变时的存在性与学习保证 | 高 | 开放 |

R1 是地基;R2 把评估指标和目标焊死;R3 是解释力最强的一条;R4 是接到 post-training 的那一条。

## 5. R1 的具体设定

目标:不用任何 LLM,把"决策充分的状态到底长什么样"精确算出来。

**环境。**
- $\mathcal Z=\{z_1,z_2,z_3\}$,其中 $z_1,z_2$ 观测律不同但在主任务损失下后果相同($\mathcal D$-等价),$z_3$ 与二者不等价;
- $\mathcal X$ 取 4 个状态;
- $\mathcal A=\{\text{act}_1,\text{act}_2,\text{wait},\text{probe}\}$,probe 让观测噪声降低但有成本;
- $\mathcal O$ 取 3 个观测值,噪声依赖 regime;
- $H$:几何 hazard,$h=0.1$;
- 视野 $H=4$,可达 belief 树规模 $(|\mathcal A||\mathcal O|)^4=20736$,可精确枚举。

**决策族。** 一族参数化损失 $L_\lambda=L_{\text{task}}+\lambda L_{\text{risk}}$,$\lambda\in\{0,0.5,1,2,5\}$,加一个 $L_{\text{cost}}$ 惩罚 probe/wait。令 $\mathcal D_k$ 为前 $k$ 个损失,$k=1,\dots,6$。

**算法。**
1. 对每个 $L\in\mathcal D_k$ 计算单纯形上的最优行动胞腔;
2. 枚举可达 belief 树,给每个节点贴上贝叶斯行动向量;
3. 划分细化(bisimulation 最小化)得到最粗递归 $\mathcal D_k$-充分划分;
4. 对若干候选压缩(只跟踪 $x$ 的后验、只跟踪 $z$ 的后验、MAP、top-2、量化 belief)计算 $\delta_{\mathcal D_k}$。

**输出。**
- 曲线一:最小递归充分状态的类数 vs $k$。预期单调不减,从接近 $|\mathcal A|$ 增长到接近可达 belief 数;
- 曲线二:各候选压缩的 $\delta_{\mathcal D_k}$ vs $k$;
- 验证 3.1:$z_1,z_2$ 在最小状态中是否确实被合并。

**证伪条件。** 若曲线一不单调,或 $z_1,z_2$ 在 $\mathcal D$-等价时仍未被合并,命题 2.3/2.4 的递归版本有错。

## 6. 与现有文献的关系

| 文献 | 有什么 | 缺什么 |
| --- | --- | --- |
| Blackwell 1951/53;Le Cam 1964;Torgersen 1991 | 实验比较、充分性、deficiency | 静态、无行动 |
| Åström 1965;Striebel 1965 | belief 是最优控制充分统计量 | 不压缩 |
| AIS(Subramanian 等 2022);value equivalence(Grimm 等 2020) | 序贯压缩状态的价值损失上界 | 要求预测充分,比决策充分强 |
| decision calibration(Zhao 等 2021);omnipredictors(Gopalan 等 2022) | 对一族损失同时最优的预测器 | 静态、单步、无行动、无非平稳 |
| Lorden 1971;Moustakides 1986;Adams & MacKay 2007 | 变点检测最优性与递归实现 | 不管决策,假设变化后分布已知 |
| Besbes–Gur–Zeevi 2014;Duff 2002;Ross 等 2008(BAPOMDP) | 非平稳 regret 预算;贝叶斯自适应最优性 | 不管状态表示 |
| Petrie 1969;Allman–Matias–Rhodes 2009 | HMM 可辨识性 | 无行动、无部分可辨识 |
| Foster & Vohra 1998;Błasiok 等 2023 | 无假设在线校准;校准距离 | 单步、不带决策 |

框架的新处在交集:把"决策充分"这个恰到好处的弱要求放进序贯 + 主动 + 非平稳的设定。

## 7. 与应用和 post-training 的接口

- **post-training**:训练目标是让模型学一个递归 $\mathcal D$-充分状态,即最小化 $\delta_{\mathcal D}$(序贯、决策感知的 omniprediction);评估用 $\delta_{\mathcal D}$ 及其分解。模型是否为 LLM 不影响理论,只影响 $\phi$ 的参数化和 $U$ 的实现。
- **quant / search / personalization**:各自对应不同的 $\mathcal D$ 和不同的非平稳预算(突变 vs 漂移)。同一套定理,换参数。
- **可解释性**:模型表现差时,$\delta_{\mathcal D}$ 的分解指出是没记住(3.1)、没发现(3.2)、没去问(3.3)还是没校准(3.4)。

## 8. 风险与开放问题

1. **$\mathcal D$ 的选择是软肋。** 太小则结果平凡,太大则退化为经典理论。R1 的曲线一必须把"$\mathcal D$ 的大小如何决定状态的大小"变成结论而非假设。
2. **3.3 的策略依赖 KL** 可能没有干净的闭式;退路是对固定策略族陈述。
3. **R4 可能做不到通用版本**;退到有限 $\Theta$、有限 $\mathcal D$ 仍是完整故事。
4. **递归充分与静态充分的差距**:一个静态充分但不可递归的统计量在实践中很常见(比如只存当前 MAP),需要量化"为了递归性额外要付多少状态"。
5. **多步损失**:目前 $\mathcal D$ 只含一步损失;多步目标下"贝叶斯行动向量"要换成价值函数的等价类,与 value equivalence 的关系需要厘清。

## 9. 参考文献

- Blackwell, D. (1951). Comparison of experiments. *Proc. 2nd Berkeley Symp.*
- Blackwell, D. (1953). Equivalent comparisons of experiments. *Ann. Math. Statist.*
- Le Cam, L. (1964). Sufficiency and approximate sufficiency. *Ann. Math. Statist.*
- Torgersen, E. (1991). *Comparison of Statistical Experiments.* Cambridge UP.
- Åström, K. J. (1965). Optimal control of Markov processes with incomplete state information. *J. Math. Anal. Appl.*
- Lorden, G. (1971). Procedures for reacting to a change in distribution. *Ann. Math. Statist.*
- Moustakides, G. V. (1986). Optimal stopping times for detecting changes in distributions. *Ann. Statist.*
- Adams, R. P., & MacKay, D. J. C. (2007). Bayesian online changepoint detection. arXiv:0710.3742.
- Petrie, T. (1969). Probabilistic functions of finite state Markov chains. *Ann. Math. Statist.*
- Allman, E. S., Matias, C., & Rhodes, J. A. (2009). Identifiability of parameters in latent structure models with many observed variables. *Ann. Statist.*
- Savage, L. J. (1971). Elicitation of personal probabilities and expectations. *JASA.*
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA.*
- Foster, D. P., & Vohra, R. V. (1998). Asymptotic calibration. *Biometrika.*
- Błasiok, J., Gopalan, P., Hu, L., & Nakkiran, P. (2023). A unifying theory of distance from calibration. *STOC.*
- Zhao, S., Kim, M. P., Sahoo, R., Ma, T., & Ermon, S. (2021). Calibrating predictions to decisions: a novel approach to multi-class calibration. *NeurIPS.*
- Gopalan, P., Kalai, A. T., Reingold, O., Sharan, V., & Wieder, U. (2022). Omnipredictors. *ITCS.*
- Subramanian, J., Sinha, A., Seraj, R., & Mahajan, A. (2022). Approximate information state for approximate planning and reinforcement learning in partially observed systems. *JMLR.*
- Grimm, C., Barreto, A., Singh, S., & Silver, D. (2020). The value equivalence principle for model-based reinforcement learning. *NeurIPS.*
- Besbes, O., Gur, Y., & Zeevi, A. (2014). Stochastic multi-armed-bandit problem with non-stationary rewards. *NeurIPS.*
- Duff, M. O. (2002). *Optimal Learning: Computational Procedures for Bayes-Adaptive Markov Decision Processes.* PhD thesis, UMass Amherst.
- Ross, S., Chaib-draa, B., & Pineau, J. (2008). Bayes-adaptive POMDPs. *NeurIPS.*
