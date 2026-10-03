# 数学与量化复习：把会用的公式，变成讲得清的推理

**中文** · [English](README.en.md)

> 阅读时间：约 6 分钟 · 覆盖表与阅读路线 · 最近审阅：2026-10

这部分是我自己也会回来翻的复习笔记。希望遇到一道换了说法的题，不只是觉得“好像见过”，而是知道该定义什么、先用哪条性质、答案怎样检查。

先从[概率复习路线](probability/study-guide.md)开始也完全可以。想把数学、随机过程和金融基础连起来，就按下面分区走；不用一口气读完。

## 和“绿皮书”怎样对应

这里暂按 Xinfeng Zhou 的 *A Practical Guide to Quantitative Finance Interviews* 整理主题。[书目介绍](https://books.google.com/books?id=RosxmAYFFosC)列出的主要范围包括逻辑题、微积分、线代、概率、随机过程、金融与编程。

**这是一份独立的知识点复习，不是原书题目与答案的转载。** 目前覆盖主要方向，但还没有对照你手里版本的完整目录、每个小节和每道习题逐项验收，所以不写“全书 100% 覆盖”。阅读时也会明确区分完整推导、证明思路和只介绍的进阶结论。

| 范围 | 到哪里复习 | 当前深度 |
| --- | --- | --- |
| 解题与表达 | [证明工具箱](probability/proof-toolbox.md)、[综合复习](review.md) | 建模、切入点、边界检查与自述 |
| 逻辑与 brain teasers | [不变量、抽屉、归纳、双计数](methods/README.md) | 证明 + 变式；不收谜底题库 |
| 微积分 | [连续概率](probability/continuous-calculus.md)、[微积分与优化](methods/calculus-optimization.md) | 积分、换元、MVT、Taylor、凸性；KKT 入门 |
| 线性代数 | [投影、最小二乘、PSD、SVD](methods/linear-algebra.md) | 关键推导；谱定理与 SVD 的存在性引用而不完整证明 |
| 概率基础 | [公理到极限的路线](probability/study-guide.md) | 事件、条件、期望、不等式的证明；CLT 给证明骨架 |
| 组合与随机变量 | [计数](probability/counting.md)、[分布](probability/distribution-toolkit.md)、[联合与顺序统计](probability/joint-and-order.md) | 推导、例题、反例 |
| 随机过程 | [Markov 链](probability/markov-chains.md)、[Poisson](processes/README.md)、[鞅](processes/martingales.md)、[动态规划](processes/dynamic-programming.md) | 有限状态递推、有界停止与 Wald 证明、有限期决策 |
| 随机微积分 | [Brownian motion 与 Itô](processes/brownian-ito.md) | 二次变差推导、Itô 直觉与例子；不替代严格课程 |
| 金融 | [无套利与二叉树](finance/README.md)、[期权](finance/options-and-greeks.md)、[组合与风险](finance/portfolio-and-market.md) | 复制推导、BS 推导路线、Greeks 与基本风险概念 |
| 算法与数值 | [数值计算与流式算法](methods/numerical-and-coding.md)、[已有算法专题](../interview/leetcode.md) | 方法、复杂度、可运行示例；C++ 语言专题仍未完整展开 |
| 额外拓展 | [统计推断](methods/statistics.md) | MLE、Bayes、区间、检验、回归、时间依赖与验证 |

## 先按用途选路线

<div class="lesson-recipe">
  <div><span>1 · 找回基础</span><strong>事件、计数、分布、期望：先把随机对象说清楚</strong></div>
  <div><span>2 · 把证明补上</span><strong>条件、不等式、极限：每个等号都有理由</strong></div>
  <div><span>3 · 看懂过程与模型</span><strong>递推、停止、优化：把一个大问题拆成小问题</strong></div>
  <div><span>4 · 检查结论靠不靠谱</span><strong>统计误差、数值误差、模型假设：别只盯最后的数字</strong></div>
</div>

- **先补概率：**[事件证明](probability/event-proofs.md) → [计数](probability/counting.md) → [期望](probability/expectation-proofs.md) → [条件期望](probability/conditioning-proofs.md) → [分布工具](probability/distribution-toolkit.md)。
- **准备研究类数学讨论：**[不等式](probability/inequalities.md) → [LLN / CLT](probability/limits.md) → [线代](methods/linear-algebra.md) → [统计推断](methods/statistics.md) → [数值方法](methods/numerical-and-coding.md)。
- **补等待、决策与金融模型：**[Markov 链](probability/markov-chains.md) → [Poisson](processes/README.md) → [鞅](processes/martingales.md) → [动态规划](processes/dynamic-programming.md) → [Itô](processes/brownian-ito.md) → [无套利](finance/README.md)。

这些是学习路线，不代表某家公司固定考什么。金融部分先掌握核心模型即可；是否需要更深的衍生品内容，要看具体岗位。

## 每章怎么用，才不只是“看懂了”

1. **先说模型。** 什么是随机的？哪些独立？样本空间是什么？
2. **自己推关键一步。** 不必默写全文，先找那一个让问题变简单的等式。
3. **换一个条件。** 放回改不放回、公平改偏置、固定次数改随机停止。
4. **做一次检查。** 小规模、极端参数、单位、概率范围、另一种解法。
5. **用代码抓错。** 代码检查具体例子，不替代证明；模拟还要有误差条。

提示和解答都能在原地展开。不加打卡、掌握度、排行榜；复习可以按自己的节奏来。

## 接下来仍值得细化的内容

- 不同版本的原书小节对应表，尤其是书中具体编程语言与题型；需要确切目录才能核对。
- 线代中谱定理、SVD 的完整存在性证明；随机过程中的一般可选停止和弱收敛。
- 更深入的数理统计、时间序列与数值线代；目前是能继续读讲义的基础层。
- 专门的利率模型、奇异期权与完整 C++ 专题，不在这版声称完成。

如果只剩一点时间，去[综合复习：定理条件与变式](review.md)。如果基础还不稳，回[概率路线](probability/study-guide.md)，慢一点更划算。
