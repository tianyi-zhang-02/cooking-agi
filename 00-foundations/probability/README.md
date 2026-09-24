# 概率：从三条公理出发

**中文** · [English](README.en.md)

> 阅读时间：约 5 分钟 · 难度：入门 · 最近审阅：2026-09

<div class="lesson-recipe">
  <div><span>解决什么问题</span><strong>概率到底是什么，哪些是定义，哪些是推出来的</strong></div>
  <div><span>前置知识</span><strong>集合的交、并、补</strong></div>
  <div><span>核心机制</span><strong>样本空间 Ω、事件、概率函数 P，外加三条公理</strong></div>
  <div><span>常见错误</span><strong>把事件和随机变量混着写，比如写 P(X)</strong></div>
</div>

## 三样东西

拿「明天下不下雨」当例子：

- **样本空间 Ω**：所有可能结果的集合。这里 $\Omega = \{\text{下雨}, \text{不下雨}\}$。
- **事件**：Ω 的一个子集。$A = \{\text{下雨}\}$，它的补是 $A^c = \{\text{不下雨}\}$。
- **概率 P**：给每个事件配一个数的函数，要满足下面三条公理。

注意「明天下雨」是一个**事件**，不是随机变量。这个区别现在分清楚，后面会省很多事。

## 三条公理

Kolmogorov 在 1933 年把概率定义成满足这三条的函数 P：

1. $P(A) \ge 0$，对任何事件 A；
2. $P(\Omega) = 1$；
3. 如果 $A_1, A_2, \dots$ 两两不相交，那么 $P(A_1 \cup A_2 \cup \cdots) = P(A_1) + P(A_2) + \cdots$。

**定义就这么多。** 概率论里别的东西，要么是从这三条推出来的定理，要么是另外加上去的定义（条件概率、独立、随机变量）。

完整的对象叫概率空间 $(\Omega, \mathcal{F}, P)$。$\mathcal{F}$ 是「允许谈概率的事件」组成的集合：离散情况下就是所有子集，连续情况下是区间以及由区间做可数次并、补得到的集合。面试里基本不用管它。

## 平时怎么算，和定义是两回事

- **古典概型**：Ω 有限、每个结果等可能时，$P(A) = |A| / |\Omega|$。这是公理 3 的推论，不是公理。骰子、扑克、硬币都在这里。
- **连续情形**：$P(a \le X \le b) = \int_a^b f(x)\,dx$。密度 f 是让 P 满足公理的工具，它本身不是概率。
- **长期频率**：重复 n 次，A 出现的比例会趋于 $P(A)$。这是大数定律，一条定理；它也是为什么这三条公理「选得对」的直觉来源。

## 六个马上能推出来的结论

每条都不超过三行，建议先自己推一遍再看：

1. $P(A^c) = 1 - P(A)$。A 和 $A^c$ 不相交，并起来是 Ω，所以 $P(A) + P(A^c) = P(\Omega) = 1$。刚才那个例子，$P(\text{下雨}) = 0.8$ 就有 $P(\text{不下雨}) = 0.2$——答案你本来就知道，现在它是从公理推出来的。
2. $P(\varnothing) = 0$。$\varnothing$ 是 Ω 的补，用第 1 条。
3. 若 $A \subseteq B$，则 $P(A) \le P(B)$。把 B 拆成 A 和 $B \setminus A$ 两块不相交的部分：$P(B) = P(A) + P(B \setminus A) \ge P(A)$。
4. $P(A) \le 1$。$A \subseteq \Omega$，用第 3 条。
5. $P(A \cup B) = P(A) + P(B) - P(A \cap B)$。把 $A \cup B$ 拆成 $A \setminus B$、$A \cap B$、$B \setminus A$ 三块不相交的部分，再分别写出 $P(A)$ 和 $P(B)$，交集那块被数了两次，减掉一次。
6. 并集上界 $P(A_1 \cup \cdots \cup A_n) \le \sum_i P(A_i)$。第 5 条给出 $P(A \cup B) \le P(A) + P(B)$，再归纳。「至少有一个发生」的题里天天用。

## 事件不是随机变量

**随机变量是一个函数** $X: \Omega \to \mathbb{R}$，把每个结果映成一个数。想把事件变成随机变量，最常用的办法是指示变量：

$$X = \mathbf{1}_A = \begin{cases} 1 & \text{下雨} \\ 0 & \text{不下雨} \end{cases}$$

这时 $P(X = 1) = P(A) = 0.8$，$P(X = 0) = 0.2$，X 服从 Bernoulli(0.8)。下面这一行撑起了一半的概率面试题：

$$\mathbb{E}[\mathbf{1}_A] = 1 \cdot P(A) + 0 \cdot P(A^c) = P(A)$$

**指示变量的期望就是事件的概率。** 这是「事件和机会」通往「期望计数」的那座桥：问「平均有几个……」的题，基本都是把个数拆成一堆指示变量的和，再用期望的线性性。顺手还有方差：

$$\mathrm{Var}(\mathbf{1}_A) = P(A)\,\bigl(1 - P(A)\bigr) = 0.8 \times 0.2 = 0.16$$

## 什么记号配什么

| 对象 | 用什么 | 例子 |
| --- | --- | --- |
| 事件 | P | $P(A)$、$P(A \cap B)$、$P(A \mid B)$ |
| 随机变量 | E、Var，或者某个取值的 P | $\mathbb{E}[X]$、$\mathrm{Var}(X)$、$P(X = k)$ |

X 是随机变量的时候写 $P(X)$，在面试里读起来是个危险信号。小事，现在改掉最省事。

下一篇把随机变量画出来：[把分布画出来](distributions.md)。

## 面试常见问题

<details class="interview" markdown="1">
<summary>掷两枚均匀硬币，A =「至少一个正面」。P(A) 是多少？用两种方法算。</summary>

$\Omega = \{HH, HT, TH, TT\}$，$A = \{HH, HT, TH\}$。直接数：$3/4$。走补集：$A^c = \{TT\}$，$1 - 1/4 = 3/4$。

</details>

<details class="interview" markdown="1">
<summary>接上题，指示变量 1_A 的期望和方差是多少？</summary>

$\mathbb{E}[\mathbf{1}_A] = P(A) = 3/4$，$\mathrm{Var}(\mathbf{1}_A) = 3/4 \times 1/4 = 3/16$。

</details>

<details class="interview" markdown="1">
<summary>只用三条公理，推出 P(A ∪ B) = P(A) + P(B) − P(A ∩ B)。</summary>

$A \cup B$ 拆成三块不相交的部分：$A \setminus B$、$A \cap B$、$B \setminus A$。由公理 3，$P(A \cup B)$ 是三块之和；同样 $P(A) = P(A \setminus B) + P(A \cap B)$，$P(B) = P(B \setminus A) + P(A \cap B)$。代进去，$A \cap B$ 被多数了一次，减掉即可。

</details>

<details class="interview" markdown="1">
<summary>「明天下雨」是随机变量吗？为什么不能写 P(X)？</summary>

不是，它是事件，Ω 的一个子集。随机变量是从 Ω 到实数的函数，比如指示变量 $\mathbf{1}_A$。事件配 P，随机变量配 E 和 Var，要谈某个取值就写 $P(X = k)$。

</details>
