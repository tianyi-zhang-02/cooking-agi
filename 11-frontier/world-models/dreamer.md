# 世界模型：在想象里学（Dreamer）

**中文** · [English](dreamer.en.md)

> 阅读时间：约 12 分钟 · 最近审阅：2026-10

机械臂每推一次杯子，都要有人把杯子摆回去；推掉了，杯子还可能摔碎。能不能先从记录下来的交互里学一个模型，让策略在模型“想象”出的轨迹上练成千上万次，再上真机？Dreamer 系列就是这么做的。

它学习一个紧凑的潜在动力学模型（latent dynamics model），actor-critic 只在想象出的 latent 轨迹上学习。这样可以减少真实交互，但策略也会继承、甚至利用模型的错误。

## Dreamer 把计算放在训练时

[Model-based RL 那篇](../../05-post-training/deep-rl/model-based.md)里的 MPC，每次决策都要用模型搜索一串动作。Dreamer 走另一条路：模型只在训练时用来“做梦”，部署时直接从 actor 采样动作。DreamerV3 论文写明，和环境交互时不做前瞻规划（[Hafner 等，DreamerV3](https://arxiv.org/abs/2301.04104)，“Critic learning”一节）。它就是[规划那篇](../../05-post-training/deep-rl/planning-and-control.md)表格里的 “Latent imagination” 一行。

来历：[PlaNet（2018）](https://arxiv.org/abs/1811.04551)提出同时有确定性和随机部分的 latent 动力学，并在 latent 空间里在线规划；[Dreamer（2019）](https://arxiv.org/abs/1912.01603)去掉在线规划，改为把学到的价值估计沿想象轨迹反传，训练 actor-critic。

## RSSM：一条确定的记忆，加一个随机的猜测

Dreamer 的世界模型叫 RSSM（recurrent state-space model）。按 DreamerV3 论文式 (1)：

$$
h_t=f_\phi(h_{t-1},z_{t-1},a_{t-1}),\qquad
z_t\sim q_\phi(z_t\mid h_t,x_t),\qquad
\hat z_t\sim p_\phi(\hat z_t\mid h_t)
$$

奖励、继续标志和重建图像都从模型状态 $(h_t,z_t)$ 预测：$\hat r_t,\hat c_t,\hat x_t\sim p_\phi(\cdot\mid h_t,z_t)$。

| 符号 | 是什么 | 在推杯子的例子里 |
| --- | --- | --- |
| $x_t$ | 第 $t$ 步的观测 | 相机拍到的一帧 |
| $a_{t-1}$ | 上一步的动作 | “往前推 1 cm” |
| $h_t$ | 确定性的循环状态，由 $f_\phi$（一个 RNN）更新 | 记住推过几次、杯子大概在哪 |
| $z_t$ | 随机 latent，从后验 $q_\phi$ 采样，用到了当前观测 | 看过这一帧之后，判断杯子有没有在晃 |
| $\hat z_t$ | 从先验 $p_\phi$ 采样，没有看观测 | 闭着眼猜：推完以后杯子大概会怎样 |
| $\hat r_t,\hat c_t$ | 预测的奖励，和回合是否继续（$c=0$ 表示结束） | 杯子到了目标区给奖励；掉下桌就结束 |

后验和先验的区别，是“想象”能成立的关键。训练时有真实画面，可以用后验；想象时没有观测，只能靠先验一步步往前猜。所以训练要让先验尽量猜中后验。为什么既要 $h$ 又要 $z$？直观地说，确定性部分稳定地携带记忆，随机部分表达“同一个动作，结果可能不同”。

## 世界模型怎么训练：重建、预测、互相靠拢

DreamerV3 的世界模型损失（论文式 2–3），$\phi$ 是世界模型的全部参数：

$$
\mathcal L(\phi)=\mathbb E_{q_\phi}\Big[\sum_t\big(\beta_{\rm pred}\mathcal L_{\rm pred}+\beta_{\rm dyn}\mathcal L_{\rm dyn}+\beta_{\rm rep}\mathcal L_{\rm rep}\big)\Big],\qquad
\beta_{\rm pred}=1,\ \beta_{\rm dyn}=1,\ \beta_{\rm rep}=0.1
$$

$$
\begin{aligned}
\mathcal L_{\rm pred}&=-\ln p_\phi(x_t\mid z_t,h_t)-\ln p_\phi(r_t\mid z_t,h_t)-\ln p_\phi(c_t\mid z_t,h_t)\\
\mathcal L_{\rm dyn}&=\max\big(1,\ \mathrm{KL}[\operatorname{sg}(q_\phi(z_t\mid h_t,x_t))\,\|\,p_\phi(z_t\mid h_t)]\big)\\
\mathcal L_{\rm rep}&=\max\big(1,\ \mathrm{KL}[q_\phi(z_t\mid h_t,x_t)\,\|\,\operatorname{sg}(p_\phi(z_t\mid h_t))]\big)
\end{aligned}
$$

- $\mathcal L_{\rm pred}$：重建观测，预测奖励和继续标志。重建让 $z_t$ 留住输入里的信息。
- $\mathcal L_{\rm dyn}$：$\operatorname{sg}$ 是 stop-gradient。后验当成固定目标，只训练先验去猜中它。
- $\mathcal L_{\rm rep}$：反过来，先验固定，让编码器产生更容易预测的表示。
- $\max(1,\cdot)$ 叫 free bits：KL 已经低于 1 nat 时，这一项变成常数、没有梯度，模型把力气留给预测损失。

用教学数字看分工。某一步 KL 是 0.6 nat，两项都被截成常数 1，不产生梯度。KL 是 2.3 nat 时，两个 KL 数值相同，只是梯度流向不同：dynamics 项贡献 $1\times2.3=2.3$，representation 项只有 $0.1\times2.3=0.23$，“先验去追后验”的权重是反方向的 10 倍。这种不对称来自 [DreamerV2](https://arxiv.org/abs/2010.02193) 的 KL balancing：同一个 KL 上，先验方向权重 0.8、后验方向 0.2（§2.1），免得表示被还没训好的先验拉偏。

## 在想象里训练 actor 和 critic

actor 和 critic 只看模型状态 $s_t=\{h_t,z_t\}$。从回放数据里取真实序列，编码成起点；然后 actor 选动作，先验预测下一个状态，各个头预测奖励和继续标志，一路滚出 $T$ 步。DreamerV3 正文写 $T=16$（超参表里记为 15），折扣 $\gamma=0.997$，$\lambda=0.95$。

<!-- paper-figure:dreamerv3-training -->

想象只有十几步，之后的回报交给 critic。λ-return 把两者混在一起：

$$
R^\lambda_t=r_{t+1}+\gamma\,c_{t+1}\big[(1-\lambda)\,v_{t+1}+\lambda R^\lambda_{t+1}\big],\qquad R^\lambda_T=v_T
$$

- $r_{t+1},c_{t+1}$：到达下一个想象状态 $s_{t+1}$ 时预测的奖励和继续标志；$c=0$ 会截掉之后的全部回报。
- $v_{t+1}$：critic 对 $s_{t+1}$ 的价值估计。
- $\lambda$：越接近 1，越相信想象出的奖励序列；越接近 0，越早交给 critic。

论文式 (5) 把 $r,c,v$ 的下标都写成 $t$；这里写成 $t+1$，强调它们都属于“下一步”。可以用 $\lambda=0$ 检查：目标变成 $r_{t+1}+\gamma c_{t+1}v_{t+1}$，正是一步 TD 目标。

critic 用最大似然拟合这个目标，$\mathcal L(\psi)=-\sum_t\ln p_\psi(R^\lambda_t\mid s_t)$，输出是指数间隔分箱上的类别分布。actor 的损失（论文式 6–7）：

$$
\mathcal L(\theta)=-\sum_t\Big[\operatorname{sg}\Big(\frac{R^\lambda_t-v_\psi(s_t)}{\max(1,S)}\Big)\log\pi_\theta(a_t\mid s_t)+\eta\,\mathrm H[\pi_\theta(\cdot\mid s_t)]\Big],\qquad
S=\operatorname{EMA}\big(\operatorname{Per}(R^\lambda,95)-\operatorname{Per}(R^\lambda,5),\,0.99\big)
$$

第一项是 REINFORCE 式的策略梯度：优势为正，就提高这个动作的概率。DreamerV3 对离散和连续动作都用这个估计器；Dreamer v1 则让价值梯度穿过想象轨迹反传。$S$ 是回报第 5 到第 95 百分位的范围，只把大回报缩小，小于 1 的不放大；$\eta=3\times10^{-4}$ 是固定的熵系数。

## 手算一次：3 步想象轨迹的 λ-return

为了好算，这里用 $\gamma=0.9$、$\lambda=0.5$，这是教学设定，不是论文的值。起点 $s_0$ 是回放来的真实状态：杯子离桌沿 6 cm，机械臂贴着杯子。actor 每步推 1 cm，模型想象出 3 步：

| 想象状态 | 预测奖励 $r$ | 继续 $c$ | critic 价值 $v$ | $R^\lambda$ |
| --- | --- | --- | --- | --- |
| $s_0$（起点） | — | — | 0.6 | 0.695 |
| $s_1$ | 0 | 1 | 0.5 | 1.044 |
| $s_2$ | 0 | 1 | 0.6 | 1.72 |
| $s_3$（到达目标区） | 1 | 1 | 0.8 | 0.8 |

从后往前算：

- $R^\lambda_3=v_3=0.8$；
- $R^\lambda_2=r_3+\gamma c_3[(1-\lambda)v_3+\lambda R^\lambda_3]=1+0.9\times0.8=1.72$；
- $R^\lambda_1=0+0.9\times(0.5\times0.6+0.5\times1.72)=0.9\times1.16=1.044$；
- $R^\lambda_0=0+0.9\times(0.5\times0.5+0.5\times1.044)=0.9\times0.772\approx0.695$。

起点的优势是 $0.695-0.6=0.095$，为正，actor 会稍微提高“推 1 cm”的概率。这里 $S<1$，分母取 1。

<!-- widget:step-lambda-return -->

换一个条件看 $\lambda$：$\lambda=0$ 时 $R^\lambda_0=r_1+\gamma v_1=0.45$，完全相信 critic；$\lambda=1$ 时变成 $0.9^2\times1+0.9^3\times0.8\approx1.393$，完全相信想象出的奖励，只在末端用一次 $v_3$。$\lambda=0.5$ 落在两者之间。

再换一个条件看模型错误。假设模型在第 2 步错误地预测杯子掉了，给出 $r_2=-1$、$c_2=0$。那么 $R^\lambda_1=-1$，$R^\lambda_0=0.9\times(0.5\times0.5+0.5\times(-1))=-0.225$，优势变成 $-0.825$：actor 学会躲开一个其实安全的动作。反过来，如果模型漏掉了真实的掉落，actor 就会学着去推。**actor 只看得到模型给的数字，看不到真实世界。**

## DreamerV2、V3 改了什么

| 版本 | 主要改动 | 作者报告的结果 |
| --- | --- | --- |
| [DreamerV2](https://arxiv.org/abs/2010.02193)（2020） | 随机 latent 从高斯换成 32 个各有 32 类的类别变量；KL balancing | 第一个完全在单独训练的世界模型里学行为、在 Atari 55 个任务上达到人类水平的 agent；单张 V100 不到 10 天跑完 2 亿环境步 |
| [DreamerV3](https://arxiv.org/abs/2301.04104)（2023；[Nature 2025](https://www.nature.com/articles/s41586-025-08744-2)） | 一套固定超参；symlog 变换、KL balance 加 free bits、类别分布混入 1% 均匀分布、按百分位归一化回报、symexp twohot 输出头 | 一套配置跑 150 多个任务；摘要称它是“the first algorithm to collect diamonds in Minecraft from scratch without human data or curricula”（第一个不用人类数据和课程、从零在 Minecraft 里拿到钻石的算法），每次 Minecraft 运行都在 1 亿环境步内找到钻石 |

这些改动大多为了“不调参也能跑”：奖励尺度可能差几个数量级，所以用 $\operatorname{symlog}(x)=\operatorname{sign}(x)\ln(|x|+1)$ 压缩目标；1% 的均匀混合让类别分布永远不会变成确定的，KL 不会突然暴涨。论文的算力对比是 VPT 用 720 张 GPU 训练 9 天拿到钻石，DreamerV3 用 1 张 GPU 训练 9 天；但 DreamerV3 用的是 MineRL 比赛的动作空间，里面有抽象的合成动作。

## Dreamer 4：换成视频世界模型，只用离线数据

Google DeepMind 的 [Dreamer 4](https://arxiv.org/abs/2509.24527)（Hafner、Yan、Lillicrap，2025-09）保留“在想象里训练”，把 RSSM 换成可扩展的视频世界模型；论文附录 F 说，RSSM 推理很轻，但难以扩展到多样的数据。

<!-- paper-figure:dreamer4-world-model -->

- **世界模型**：因果 tokenizer（400M）加交互式 dynamics transformer（1.6B），共约 2B（§4）。训练目标 shortcut forcing 让每一帧有自己的噪声水平，网络还以步长为条件，直接预测干净的 latent；推理时每帧 4 次前向（§3.2）。这类逐帧扩散配方在[能交互的视频](interactive-video.md)一篇展开。
- **数据**：只用 VPT 的承包商数据集，2,541 小时、360p、20 FPS、带键鼠动作的游戏录像。
- **agent**：插入带任务 embedding 的 agent token，其他模态不能注意到它们，免得模型对未来的预测被任务而不是动作左右。先用行为克隆学策略头和奖励头，再在世界模型生成的 rollout 里做 RL；这时 transformer 冻结，只更新策略头和价值头（§3.3）。
- **RL 目标**：价值头仍拟合上面的 λ-return（$\gamma=0.997$）；策略头改用 PMPO，只看优势的正负、不看大小，所以不必归一化回报，另加一个指向行为克隆策略的 KL（§3.3）。

论文还报告（表 1）：一张 H100 上 640×360、9.6 秒上下文、21 FPS；人类玩家借它完成了 16 个交互任务中的 14 个（Oasis-large 5 个）。图 7 中，2,500 小时视频只有 100 小时带动作标签，动作条件的准确度就达到全部带标签时的 80% 以上。

## 0.7% 说明了什么，没说明什么

主结果（§4.1，图 3）在真实的 Minecraft 里评估：每回合 60 分钟，随机世界、空背包起步，统计 1000 个回合。石镐及之前的里程碑成功率超过 90%，铁镐 29%，钻石 0.7%，即 1000 回合里大约 7 回合。作为参照，论文说有经验的人类玩家平均要 20 分钟、约 24,000 次键鼠操作拿到一颗钻石；同样离线的对照里，VPT（finetuned）最远到木棍（53%），基于 Gemma 3 的 VLA 最远到铁镐（11%）。

| 这个数字说明 | 这个数字不说明 |
| --- | --- |
| 训练阶段完全不和环境交互，策略也能走到钻石；论文称这是第一次 | 已经能稳定拿钻石：99% 以上的回合没拿到 |
| 想象训练对越难的里程碑帮助越大（相对同一模型的行为克隆版本） | 训练没用人类数据：数据就是人类玩家的录像和动作，任务奖励也来自数据集里的事件标注 |
| 比 VPT 的离线 agent 少用约 100 倍数据（2.5K 小时，VPT 还额外用了 27 万小时网络视频） | 可以和 DreamerV3 的钻石直接比：V3 是在线交互约 1.4K 小时、64×64 图像加背包状态、带抽象合成动作 |
| 评估时用一串固定的任务提示引导 agent（表 6） | agent 自己规划出了整条科技树：提示的顺序是作者给的 |

论文写了训练用 256 到 1024 块 TPU-v5p，但没给训练时长或总算力，所以“省数据”不能直接换算成“省钱”。

## 和 MuZero、MPC 放在一起看

[MuZero](https://arxiv.org/abs/1911.08265) 有 3 个函数：表示 $s^0=h_\theta(o_1,\dots,o_t)$，动力学 $r^k,s^k=g_\theta(s^{k-1},a^k)$，预测 $p^k,v^k=f_\theta(s^k)$。损失只要求预测奖励、价值和 MCTS 给出的策略，论文强调隐藏状态不必对应任何环境状态。放到推杯子里：MuZero 的模型不必知道杯子在哪，只要对“推 5 cm 之后值多少”猜得准。

| | Dreamer | MuZero | MPC（model-based 那篇） |
| --- | --- | --- | --- |
| 模型预测什么 | 下一 latent 状态、奖励、继续；还要重建观测 | 只有奖励、价值和策略 | 下一状态（常是低维）和奖励 |
| 计算花在哪 | 训练时在想象里学 actor；部署时直接采样 | 每次决策做 MCTS | 每次决策搜索动作序列 |
| 模型状态有没有含义 | 能解码回画面，可以拿来检查 | 不要求有环境语义 | 通常就是物理状态 |
| 代表结果 | DreamerV3 一套超参跑 150 多个任务 | MuZero Reanalyze 在 57 个 Atari 游戏上中位人类归一化分数 731%（2 亿帧） | 那篇的小车例子：模型增益错了，重规划能救回一部分 |

这几条路线的训练目标各有侧重：Dreamer 重建输入，MuZero 只预测和决策有关的量，[下一篇](jepa.md)的 JEPA 预测 embedding。

## 读 Dreamer 类结果时查什么

| 查什么 | 为什么重要 | 本篇里的例子 |
| --- | --- | --- |
| 真实交互量或数据量 | “省数据”要有明确的基数 | DreamerV3：Minecraft 1 亿环境步；Dreamer 4：2,541 小时离线录像 |
| 在线还是离线 | 在线训练能用真实反馈纠正模型，离线不能 | V3 在线；Dreamer 4 训练全离线，评估在真实游戏里 |
| 想象步数 | 越长越依赖模型，误差越会累积 | V3 约 16 步，之后交给 critic |
| 观测和动作空间 | 换了动作空间，任务难度就变了 | V3：64×64、背包状态、抽象合成动作；Dreamer 4：360×640、原始键鼠 |
| 报告的到底是什么 | 成功率要配上回合长度、回合数和引导方式 | Dreamer 4：60 分钟回合、1000 回合、任务提示序列 |
| 算力 | 交互少不等于便宜 | V3：1 张 GPU、9 天；Dreamer 4：给了芯片数，没给时长 |

## 局限

- **误差会滚雪球。** 想象越长越依赖模型，[model-based 那篇](../../05-post-training/deep-rl/model-based.md)里误差累积的界同样适用。Dreamer 用短想象加 critic 兜底，代价是远处的回报全靠 critic 估得准。
- **策略会钻模型的漏洞。** [Ha 和 Schmidhuber（2018）](https://arxiv.org/abs/1803.10122)在 VizDoom 里发现，agent 学会了让“梦里”的怪物不开火的对抗性策略；调高采样温度 $\tau$、让梦更难预测，反而更难被钻空子（§4.5，表 2）：$\tau=0.10$ 时梦里得分 2086，真实只有 193；$\tau=1.15$ 时迁移最好，真实得分 1092 ± 556。
- **重建会把容量花在无关细节上。** DreamerV3 论文自己说，复杂 3D 环境里有很多控制用不上的细节，需要更强的正则；它的办法是 free bits 加较小的表示损失。干脆不重建像素的做法见[下一篇 JEPA](jepa.md)。
- **记忆短。** Dreamer 4 的上下文只有 9.6 秒，论文也说它远不是游戏的完整复制品：记忆短，背包预测不准（§6）。

## 这条线接下来看什么

只列能改变判断的结果，不做预测：

1. **离线想象训练能不能更可靠？** Google DeepMind 2025 年 9 月报告的钻石成功率是 0.7%（1000 个 60 分钟回合）。只用离线数据而成功率明显更高，或者在 Minecraft 之外（论文动机里提到机器人）也成立，“只在模型里学”才不只是一次首次演示。
2. **能不能主要靠没有动作标签的视频？** Dreamer 4 图 7 显示，100 小时动作标签加 2,500 小时视频就够学动作条件；论文把用通用网络视频预训练列为后续方向。主要靠无标签网络视频、再在想象里训练成功的 agent，会改变对数据瓶颈的判断。
3. **记忆能不能补上？** 论文 v1 的上下文是 9.6 秒，作者自己点名记忆短、背包预测不准。能长期记住背包和地图、并因此提高后期里程碑成功率的结果，才说明这个短板补上了。
4. **算力账能不能对齐？** DreamerV3 报告 1 张 GPU 9 天，VPT 是 720 张 GPU 9 天；Dreamer 4 v1 只给了 256–1024 块 TPU-v5p，没给时长。补上完整的算力数字，才能判断“少 100 倍数据”是否也意味着更便宜。

继续读：[预测表示，不预测像素（JEPA）](jepa.md)。
