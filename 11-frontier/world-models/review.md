# 世界模型：复习题

**中文** · [English](review.en.md)

> 阅读时间：约 4 分钟 · 最近审阅：2026-10

## 面试常见问题

<details class="interview" markdown="1">
<summary>别人说“世界模型”，你先问哪几个问题，才知道他指的是哪一种？</summary>

先问四件事：预测什么，在什么空间预测，以什么为条件（有没有动作），拿来做什么。按这四问，常见用法能分成 RL 里的学习型模拟器（Dreamer、MuZero）、JEPA 一类的表示空间预测、可交互视频（Genie 3）、机器人和驾驶的世界基础模型（Cosmos、GAIA）、3D 空间生成（Marble），以及“LLM 或视频模型内部有没有隐含的世界模型”。World Labs 2026-06 的分类按输出更简洁：renderer 输出观测，simulator 输出状态，planner 输出动作。

同叫世界模型，学的东西可以完全不同：Dreamer 重建输入，JEPA 预测 embedding，MuZero 只预测奖励、价值和策略，隐藏状态不必对应任何真实状态。见[导读](README.md)。

</details>

<details class="interview" markdown="1">
<summary>DreamerV3 的 RSSM 由哪些部分组成？actor 怎么在“想象”里训练？</summary>

确定性循环状态 $h_t=f_\phi(h_{t-1},z_{t-1},a_{t-1})$ 负责携带记忆，随机的类别 latent $z_t$ 表达不确定性；编码器给后验 $q(z_t\mid h_t,x_t)$，动力学给不看观测的先验 $p(z_t\mid h_t)$，另有奖励、继续标志和解码头。损失是预测项、$L_{\rm dyn}$（先验追后验）和 $0.1\,L_{\rm rep}$（让编码器好预测）之和，两个 KL 都用 free bits 截在 1 nat。

想象时从回放的真实状态出发，用先验滚 16 步；critic 拟合 λ-return，actor 的策略梯度用“优势除以回报 5%–95% 百分位范围（至少取 1）”加权。actor 只看得到模型给的数字。Dreamer 4（2025-09）把世界模型换成 2B 的视频模型，只用离线数据在 Minecraft 拿到钻石，但只在 0.7% 的回合里。见 [Dreamer](dreamer.md)。

</details>

<details class="interview" markdown="1">
<summary>JEPA 为什么不预测像素？“塌缩”是什么，有哪些防法？</summary>

LeCun 2022 的理由是：像素级的未来（树叶、纹理）本身不可预测，在表示空间预测可以丢掉这些细节。代价是多了一条捷径：编码器对任何输入都输出同一个常数，预测误差就是 0，这就是塌缩（I-JEPA §2 的定义）。

三类防法：VICReg 的方差和协方差约束；I-JEPA、V-JEPA 2 用的 EMA 目标编码器加 stop-gradient，经验上有效，但损失里没有一项明确禁止塌缩；LeJEPA（2025-11）的 SIGReg，要求 embedding 的随机一维投影像标准正态。V-JEPA 2-AC 再加一个动作条件预测器，用 CEM 最小化预测 embedding 到目标图像 embedding 的 L1 距离来规划。见 [JEPA](jepa.md)。

</details>

<details class="interview" markdown="1">
<summary>Genie 怎么从没有动作标签的视频里学会“可控”？局限在哪？</summary>

潜在动作模型（LAM）的编码器看过去和下一帧，输出一个连续向量，再量化到只有 8 个代码的码本；解码器只看历史和这个代码，却必须重建下一帧，所以代码被迫装下“最主要的变化”。动力学模型以这些代码为条件预测下一帧的 token；推理时丢掉编码器，玩家直接选 0–7。可控性用 $\Delta_t$PSNR 衡量：分别用推断出的动作和随机动作生成，比较哪个离真实帧更近。

局限：代码的含义要靠人事后试玩；8 个代码表达粗，“推 3.5 厘米”可能正好落在两个代码的边界上；目标不区分变化是不是智能体造成的（这一点是推断）；原版只有 16 帧记忆、约 1 FPS。DreamGen 用 latent action model 或 IDM 给机器人视频补伪动作，是同一个思路。见[交互视频](interactive-video.md)。

</details>

<details class="interview" markdown="1">
<summary>可交互世界模型和 Sora、Veo 这类视频生成器有什么本质区别？看演示该问什么？</summary>

交互模型必须先交出第 $t$ 帧，才知道下一个按键，所以只能因果生成。24 fps 时一帧约 41.7 ms，每帧 $K$ 个采样步就是每步最多约 $41.7/K$ ms，这是开源系统普遍蒸馏到 2–4 步的原因；它还要在分钟级会话里记住看不见的东西。离线生成器拿到完整提示，再交付整段。

看演示要问：帧率和分辨率在什么硬件上测，按键到响应的延迟，能保持一致多久，有哪些控制，看不见的东西是否还在、还在变（WRBench、WorldMemBench 报告这里常失败），开放到什么程度。截至 2026-10-15，Genie 3 没有论文和 API，架构也没有公开。

</details>

<details class="interview" markdown="1">
<summary>一个世界模型评测器和真机成功率的 Pearson 相关是 0.86，能直接拿它挑上线的策略吗？</summary>

不能直接用。Pearson 衡量的是这批策略上的线性关联，在 $\hat R=aR+b$（$a>0$）下不变，所以绝对成功率可能整体偏低（Veo 评测器论文自述如此），不能拿来判断是否过门槛；它还常被差策略和好策略之间的大差距撑高。教学例子里 Pearson 0.98、Spearman 0.9、MMRV 0.04，按模型挑出的第一名仍然是错的；只看前三名，Spearman 降到 0.5。另外，Veo 论文里的 0.86 是针对一个策略、在几条 OOD 轴上算的。

实际做法：同时报告排序指标和“第一名是否选对”，留出从没参与挑选的策略，定期回真机抽检，防止策略专门钻评测器的空子。见[机器人与汽车](robotics-driving.md)。

</details>

<details class="interview" markdown="1">
<summary>探针从模型里读出了“棋盘状态”，能说明模型有世界模型吗？</summary>

只能说明信息读得出。还要看：沿探针方向干预，输出是否按预期改变（Othello-GPT 做了）；行为是否一致，Vafa 等 2024 的出租车模型下一步合法率约 1.00，序列压缩的 precision 却只有 0.10–0.50，遇到 10% 绕路时合法路线从 0.99 跌到 0.08；分布外是否用对了规律，轨道模型 $R^2$ 超过 0.9999，推出的却是非牛顿的力定律；结果分布是否校准。

Richens 等的定理说，能完成多步目标的智能体，其策略里一定能提取出一个世界模型；但前提是完全可观测，而且定理不说明这个模型被使用。见[它真的懂世界吗](do-they-understand.md)。

</details>

<details class="interview" markdown="1">
<summary>明天你读到一条新的“世界模型突破”，会怎样判断它？</summary>

按顺序问：它是哪一种世界模型，输出观测、状态还是动作，接不接动作；谁在什么时候测的，是有方法的论文、只有博客的公告、厂商自报的数字，还是二手报道；比的条件是否一致，包括硬件、采样次数和提示，比如 Physics-IQ Verified 排行榜前列的条目用了 best-of-N 和定制提示；能保持多久，有没有测看不见的东西是否还在变化。

再问它在下游闭环里是否真的有用，而不只是画面好看（WorldArena 的 perception–functionality gap）；能否复现，有没有论文、权重或 API；哪些没公开，比如 Genie 3、GAIA-4 都没有公开架构。复述时写成“截至某日，某机构自报”，而不是一句永久的结论。

</details>

## 自检

<div class="taste-check">
  <strong>试着用自己的话解释：</strong>
  <ol>
    <li>为什么交互式视频模型只能因果生成？24 fps、每帧 4 个采样步时，每步最多有多少毫秒？</li>
    <li>RSSM 的 $L_{\rm dyn}$ 和 $L_{\rm rep}$ 各把梯度送给谁？KL 低于 1 nat 时，free bits 起什么作用？</li>
    <li>JEPA 去掉 EMA 目标编码器和 stop-gradient 后，最省事的“解”是什么？SIGReg 为什么能罚掉它？</li>
    <li>5 个策略的算例里，Pearson 0.98 为什么还会选错第一名？只比前三名时 Spearman 是多少？</li>
    <li>杯子真实掉落率 30%，一个总说“掉”的模型和一个总说“不掉”的模型，期望 Brier 分数各是多少？为什么单看一段视频分不出来？</li>
  </ol>
</div>

## 参考论文

- [World Models](https://arxiv.org/abs/1803.10122)：Ha & Schmidhuber 2018，在模型的“梦”里训练控制器
- [DreamerV3](https://arxiv.org/abs/2301.04104)：RSSM 与想象训练，一套配置跑 150 多个任务
- [Dreamer 4](https://arxiv.org/abs/2509.24527)：视频世界模型加离线想象训练
- [MuZero](https://arxiv.org/abs/1911.08265)：只预测奖励、价值和策略的模型（对照）
- [A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf)：LeCun 2022，JEPA 与能量式规划
- [I-JEPA](https://arxiv.org/abs/2301.08243)：图像 JEPA，EMA 目标编码器
- [V-JEPA 2](https://arxiv.org/abs/2506.09985)：含 V-JEPA 2-AC 的 embedding 空间规划
- [LeJEPA](https://arxiv.org/abs/2511.08544)：用 SIGReg 防塌缩
- [Genie](https://arxiv.org/abs/2402.15391)：从无标签视频学 latent action
- [Genie 3 博客](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/)：截至 2026-10 只有官方博客，没有论文
- [GameNGen](https://arxiv.org/abs/2408.14837)：扩散模型做实时 DOOM
- [Cosmos World Foundation Model Platform](https://arxiv.org/abs/2501.03575)：NVIDIA 的世界基础模型平台
- [DreamGen](https://arxiv.org/abs/2505.12705)：视频世界模型生成机器人训练数据
- [Evaluating Gemini Robotics Policies in a Veo World Simulator](https://arxiv.org/abs/2512.10675)：Veo 评测器，Pearson 与 MMRV
- [DreamZero](https://arxiv.org/abs/2602.15922)：world action model 直接当策略
- [Cosmos Policy](https://arxiv.org/abs/2601.16163)：视频模型后训练成策略
- [Emergent World Representations (Othello-GPT)](https://arxiv.org/abs/2210.13382)：探针与干预
- [Evaluating the World Model Implicit in a Generative Model](https://arxiv.org/abs/2406.03689)：Myhill–Nerode 式一致性指标
- [What Has a Foundation Model Found?](https://arxiv.org/abs/2507.06952)：归纳偏置探针与轨道模型
- [General agents contain world models](https://arxiv.org/abs/2506.01622)：能力蕴含可提取的世界模型
- [Physics-IQ](https://arxiv.org/abs/2501.09038) 与 [Physics-IQ Verified](https://arxiv.org/abs/2606.18943)：真实视频物理基准及其审核版
- [WorldArena](https://arxiv.org/abs/2602.08971)：具身世界模型的三种用途分别测
- [WRBench](https://arxiv.org/abs/2606.20545) 与 [WorldMemBench](https://arxiv.org/abs/2610.19063)：记忆与状态推进
- [A Definition and Roadmap for World Models](https://arxiv.org/abs/2607.06401)：2026 年的定义与路线图
- [A Functional Taxonomy of World Models](https://www.worldlabs.ai/blog/taxonomy-of-world-models)：renderer / simulator / planner
