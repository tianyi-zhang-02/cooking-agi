# RLHF：后来发生了什么（GRPO、DPO、RLVR）

**中文** · [English](after-rlhf.en.md)

> 阅读时间：约 2 分钟 · 难度：必修 · 最近审阅：2026-09

## 后来发生了什么

四个模型、两个在训练，其中 Critic 还是个全尺寸网络。这个成本是后续所有工作要削的东西。

![各方法需要同时在显存里的模型数](../assets/rlhf-model-count.svg)

**GRPO：把 Critic 去掉。** 对同一个提示采样一组回答，用组内的相对好坏当基线：

$$\hat A_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

既然只需要一个基线来降方差，那组内均值就够了，不必专门训一个网络去估它。省掉一个在训的全尺寸模型。

**DPO：连 RL 循环一起去掉。** 关键推导是：带 KL 约束的奖励最大化问题有闭式最优解，反解之后奖励可以用策略本身表示，于是偏好损失可以**直接对策略求**：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

不用训奖励模型，不用采样，不用 Critic。标准 DPO 通常使用**离线**偏好对：策略在训练中改变，数据分布却不跟着改变，因此不能主动发现当前策略的新失败模式。Online DPO 可以重新采样来缓解这个问题。真正的分界不是算法名字，而是数据是否跟随当前策略更新、反馈来自 preference、Reward Model、verifier 还是环境。

**RLVR：奖励换成一个程序。** 数学题可以对答案，代码可以跑测试。这类任务的奖励不需要学，写一个检查器就行。学出来的奖励模型消失了，reward hacking 的空间也随之大幅收窄——**能被钻空子的是被拟合出来的奖励，不是被验证出来的**。
