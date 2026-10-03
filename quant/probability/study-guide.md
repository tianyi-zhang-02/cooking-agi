# 概率复习：把公式重新推一遍

**中文** · [English](study-guide.en.md)

想接着复习线代、统计、随机过程和金融数学？看[完整分区与覆盖表](../README.md)。本页保留概率主线，不把所有内容塞成一张长清单。

> 阅读时间：约 4 分钟 · 基础与证明导读 · 最近审阅：2026-10

有些公式，看见时认识，真让自己从头解释却会卡住。我想把这部分补回来：不只记住答案，还知道第一步为什么这样写，换个条件以后还能不能用。

这组笔记从概率公理开始，接到期望、条件、不等式、极限和 Markov 链。例子都是通用的数学练习，不对应某家公司的题库。证明能短就短；需要更多背景的地方，会明确标出这里只给思路。

## 先选一条路线

| 你现在想做什么 | 建议怎么读 |
| --- | --- |
| 基础有点生疏 | [公理与随机变量](README.md) → [分布](distributions.md) → [条件概率](conditional.md) |
| 公式会用，证明不会写 | [事件证明](event-proofs.md) → [期望与方差](expectation-proofs.md) → [条件期望](conditioning-proofs.md) |
| 想补常用定理 | [不等式](inequalities.md) → [大数定律与 CLT](limits.md) |
| 看到连续变量或“等多久”就卡住 | [连续概率与微积分](continuous-calculus.md) → [Markov 链](markov-chains.md) |
| 想练自己找切入点 | [证明技巧](proof-toolbox.md) → [练习与反例](practice.md) |
| 想把常见题型补齐 | [计数与抽样](counting.md) → [分布工具](distribution-toolkit.md) → [联合分布与顺序统计](joint-and-order.md) |

每次挑一篇就好。先不看证明，自己写出“已知什么、要证什么”，哪一步走不下去，再展开看。

## 这些章节怎样接起来

<div class="lesson-recipe">
  <div><span>1 · 事件</span><strong>把重叠的情况拆开，才知道概率怎么加</strong></div>
  <div><span>2 · 随机变量</span><strong>把“发生没发生”变成数字，开始算平均与波动</strong></div>
  <div><span>3 · 条件与上界</span><strong>先固定一部分信息；算不准时，先控制最坏能有多大</strong></div>
  <div><span>4 · 极限与过程</span><strong>重复很多次会怎样？一步接一步又该记住什么？</strong></div>
</div>

这里有两条很值得连起来的线：

- **指示变量 → Markov 不等式 → Chebyshev → 弱大数定律。** “一个事件发生的概率”可以转成“一个非负变量的期望”来控制。
- **全概率 → 条件期望 → 首步分析。** 先问下一步发生什么，再把剩下的问题交回同一种状态。

## 常用定理，先看条件

| 定理或工具 | 先检查什么 | 证明的入口 |
| --- | --- | --- |
| 并集上界（union bound） | 事件有限或可数；不要求独立 | [去掉重复部分](event-proofs.md) |
| 期望线性性 | 这里用有限绝对期望；不要求独立 | [展开联合分布](expectation-proofs.md) |
| 尾和公式（tail-sum） | 非负整数变量；结果允许为无穷 | [把数值写成一层层指示变量](expectation-proofs.md) |
| 全期望（tower property） | 这里假设可积 | [先分组，再加回去](conditioning-proofs.md) |
| 全方差 | 二阶矩有限 | [组内波动 + 组间波动](conditioning-proofs.md) |
| Markov / Chebyshev | 非负 / 有限方差；阈值为正 | [逐点比较，再取期望](inequalities.md) |
| Cauchy–Schwarz / Jensen | 二阶矩 / 凸性及可积条件 | [平方非负 / 支撑直线](inequalities.md) |
| 弱大数定律 | 本篇证明用 iid、有限方差 | [样本均值的方差随 n 下降](limits.md) |
| 中心极限定理（CLT） | 本篇用 iid、有限且非零方差 | [标准化后研究分布](limits.md) |
| 首次到达时间 | 状态信息够用；期望有限或另外论证 | [首步分类，写边界条件](markov-chains.md) |

“条件充分”不等于“少一条就一定错”。例如，有限方差让这里的大数定律证明非常短，但它不是所有版本都必须有的条件。

## 证明卡住时，先问这 4 句

1. 我现在操作的是事件、随机变量，还是一个确定的数？
2. 这个等号来自定义，还是用了某条定理？条件满足吗？
3. 有没有重复计数、漏掉平局，或把条件概率当成普通概率？
4. 如果把独立、非负、有限方差拿掉，能不能造个反例？

每篇都有就地展开的证明或提示，没有打卡和“掌握度”。[练习页](practice.md)另附一个只用 Python 标准库的小检查脚本：穷举和精确分数能帮忙抓错，但不能替代一般性的证明。

## 想继续读原始讲义

[MIT 6.041 讲义目录](https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/pages/lecture-notes/)适合按主题查；[Harvard Stat 110 的 Markov 链讲义](https://stat110.hsites.harvard.edu/resource/markov-chains)适合接着看状态、转移矩阵和长期行为。

现在开始：[事件证明：为什么可以这样拆](event-proofs.md)。
