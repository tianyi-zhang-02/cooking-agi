# 概率练习：自己选第一步

**中文** · [English](practice.en.md)

> 阅读时间：按题选做 · 前置：[复习路线](study-guide.md) · 最近审阅：2026-10

这些题不拼难度。更重要的是：你能不能选对对象，说明用了什么条件，再把证明走完。每题先有提示，解答单独展开；可以只做自己最不熟的几题。

## 1 · 两个独立事件的补集还独立吗

已知 $A,B$ 独立。证明 $A^c$ 与 $B$ 独立，也证明 $A^c$ 与 $B^c$ 独立。允许事件概率为 0。

<details markdown="1">
<summary>提示</summary>

别先除以事件概率。把交集写成差集，用独立的乘积定义。

</details>

<details markdown="1">
<summary>解答</summary>

$P(A^c\cap B)=P(B)-P(A\cap B)=(1-P(A))P(B)=P(A^c)P(B)$。因此 $A^c,B$ 独立。再对第二个事件取补，得到 $P(A^c\cap B^c)=P(A^c)P(B^c)$。全程没有除法，所以零概率情况也覆盖了。

</details>

## 2 · “两两独立，所以三个相乘”哪里错了

两枚独立公平硬币。$A$ 为第一枚正面，$B$ 为第二枚正面，$C$ 为两枚结果相同。验证两两独立，并判断三者是否相互独立。

<details markdown="1">
<summary>提示</summary>

枚举 HH、HT、TH、TT。分别数两事件和三事件交集。

</details>

<details markdown="1">
<summary>解答</summary>

三个事件概率都是 $1/2$，每个两两交集都只有 HH，概率为 $1/4$。但三重交集也只有 HH，概率是 $1/4$，并非 $(1/2)^3=1/8$。相互独立要求每个有限子集都能因子分解，不只检查所有两两组合。

</details>

## 3 · 同生日的人对，平均有几对

假设 $n$ 人的生日独立，均匀分布在 365 天，忽略闰日。求同生日的人对数的期望。它与“至少一对同生日”的概率一样吗？

<details markdown="1">
<summary>提示</summary>

给每一对人放一个指示变量，而不是先求完整的碰撞分布。

</details>

<details markdown="1">
<summary>解答</summary>

一共 $\binom n2$ 对，每对同生日概率为 $1/365$，期望为 $\binom n2/365$。它是个数的期望，可以超过 1，不是至少一对的概率。若 $N$ 是对数，$\mathbf1_{\{N\ge1\}}\le N$，因此该概率至多为 $\min(1,\mathbb E N)$。

</details>

## 4 · 几何等待的方差，不背公式

独立试验成功率为 $p\in(0,1]$，$T$ 包含第一次成功那次。已知 $\mathbb E[T]=1/p$，推 $\operatorname{Var}(T)$。

<details markdown="1">
<summary>提示</summary>

写 $T=1+IT'$：$I$ 表示首步失败，$T'$ 是独立的后续等待。对两边平方，再取期望。

</details>

<details markdown="1">
<summary>解答</summary>

记 $q=1-p$、$m=1/p$、$s=\mathbb E[T^2]$。由于 $I^2=I$，$s=1+2qm+qs$，所以 $s=(2-p)/p^2$，方差为 $s-m^2=q/p^2$。

二阶矩有限不是凭空假设：由几何 PMF，$\sum_{k\ge1}k^2pq^{k-1}$ 收敛（$p=1$ 时恒为 1；否则用比值判别法）。因此移项合法。

</details>

## 5 · 两层随机性，不要只留平均概率

公平选一枚正面率为 $0.2$ 或 $0.8$ 的硬币，用同一枚独立抛 10 次。求正面次数的均值与方差。

<details markdown="1">
<summary>提示</summary>

先给定硬币类型。全方差里的第二项不能丢。

</details>

<details markdown="1">
<summary>解答</summary>

条件均值为 2、8，条件方差都为 1.6。均值为 5；全方差为 $1.6+((2-5)^2+(8-5)^2)/2=10.6$。不是 Binomial($10,0.5$)。它与[条件期望章节](conditioning-proofs.md)是同一个例子，试着不看原文推完。

</details>

## 6 · 均匀变量的最大值

$U_1,\ldots,U_n$ 独立且均匀分布在 $(0,1)$。令 $M=\max_iU_i$，求 CDF 和期望。

<details markdown="1">
<summary>提示</summary>

最大值不超过 x，等价于每一个都不超过 x。期望可用非负变量的尾积分。

</details>

<details markdown="1">
<summary>解答</summary>

对 $0\le x\le1$，$P(M\le x)=\prod_iP(U_i\le x)=x^n$。区间外 CDF 分别为 0、1。于是

$$
\mathbb E[M]=\int_0^1(1-x^n)\,dx=\frac n{n+1}.
$$

独立用在乘积那一步。若所有 $U_i$ 都复制同一个变量，最大值仍均匀，期望为 $1/2$。

</details>

## 7 · 等 H、HH、HTH：模式不同，答案会怎样变

公平硬币独立抛掷，分别求首次出现 H、HH、HTH 的期望等待时间。不要直接取窗口概率的倒数。

<details markdown="1">
<summary>提示</summary>

H 是几何等待；HH 用 0、H 两个未完成状态；HTH 用 0、H、HT。每次回退保留哪些后缀？

</details>

<details markdown="1">
<summary>解答</summary>

H 为 2。HH 的递推是 $E_0=1+(E_0+E_H)/2$，$E_H=1+E_0/2$，解得 6。HTH 的方程见 [Markov 链](markov-chains.md)，得到 10。对应窗口概率的倒数为 2、4、8，后两项都不对。非重叠块给出有限性保证。

</details>

## 8 · 检查一句“样本多了就不会偏”

有人说：“我的样本均值收敛了，所以它就是我要估计的真实总体均值。”哪里缺了条件？

<details markdown="1">
<summary>提示</summary>

大数定律让样本平均接近哪个分布的均值？稳定与无偏是一回事吗？

</details>

<details markdown="1">
<summary>解答</summary>

即使样本是 iid，也只保证靠近抽样分布的均值。若只从总体某个子群抽样，样本量增加不会自动修正选择偏差。先定义目标总体与采样过程，再谈误差。这里没有说相关数据不能分析，而是不能原封不动套用 iid 证明。

</details>

## 用代码查错，不用代码代替证明

[精确检查脚本](code/proof_checks.py)只用标准库：枚举牌对与硬币，验证全期望、全方差，并自动建立模式等待的状态方程，用分数求解。它不运行随机模拟，也不调用外部服务。

在仓库根目录运行：

~~~bash
python3 quant/probability/code/proof_checks.py
~~~

可以把硬币概率改成 $1/3$，或把目标模式换成 HHH。**有限例子全对，不代表定理已被证明**；它主要帮助发现漏平局、漏状态和错误的数值例子。

回到[证明技巧](proof-toolbox.md)或[复习路线](study-guide.md)，挑一处卡住的步骤补上即可。
