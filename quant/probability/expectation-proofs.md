# 期望与方差：哪些地方真的需要独立

**中文** · [English](expectation-proofs.en.md)

> 阅读时间：约 8 分钟 · 前置：[随机变量与分布](distributions.md) · 最近审阅：2026-10

“期望可以直接加”没错，但下一句“方差也直接加”就未必了。这篇把几条经常一起用的公式拆开，看看独立究竟在哪一步出现。

## 1 · 线性性：联合分布不必拆成乘积

**定理。** 若 $\mathbb E|X|,\mathbb E|Y|<\infty$，常数 $a,b$ 满足：

$$
\mathbb E[aX+bY]=a\mathbb E[X]+b\mathbb E[Y].
$$

以离散变量为例，联合 PMF 记作 $p(x,y)$：

$$
\begin{aligned}
\mathbb E[aX+bY]
&=\sum_{x,y}(ax+by)p(x,y)\\
&=a\sum_x x\sum_y p(x,y)+b\sum_y y\sum_xp(x,y)\\
&=a\mathbb E[X]+b\mathbb E[Y].
\end{aligned}
$$

中间只用了求和的线性性和边缘化，**没有用** $p(x,y)=p_X(x)p_Y(y)$。绝对可积保证重排合法；连续情形换成联合密度积分，有相同推导。更一般的版本来自积分的线性性。

非负变量也有允许 $+\infty$ 的版本，但不要拿 $\infty-\infty$ 当数字算。

## 2 · 函数的期望：不用先求新分布

离散情形下，只要 $\mathbb E|g(X)|<\infty$，就有：

$$
\mathbb E[g(X)]=\sum_xg(x)P(X=x).
$$

这常叫 LOTUS。证明就是把映射到同一个 $z=g(x)$ 的结果归到一起：

$$
\sum_z zP(g(X)=z)=\sum_z z\sum_{x:g(x)=z}P(X=x)=\sum_xg(x)P(X=x).
$$

比如 $X$ 等概率取 $-1,0,1$，$\mathbb E[X]=0$，但 $\mathbb E[X^2]=2/3$。**把平均代进非线性函数，通常不是同一件事。**

## 3 · 指示变量与尾和公式

若 $N$ 是非负整数，逐个结果都有：

$$
N=\sum_{k=1}^{\infty}\mathbf1_{\{N\ge k\}}.
$$

当 $N=3$，右边就是 $1+1+1+0+\cdots$。所以：

$$
\mathbb E[N]=\sum_{k=1}^{\infty}P(N\ge k).
$$

无穷求和不是靠有限线性性“直接延长”。这里各项非负，用单调收敛定理（或 Tonelli）交换求和与期望，结果可以为无穷。非负连续变量同理有 $\mathbb E[X]=\int_0^\infty P(X>t)\,dt$。

**例子：第一次成功要等多久？** 独立重复试验，每次成功率 $p\in(0,1]$，$T$ 把成功那次也算进去。$P(T\ge k)=(1-p)^{k-1}$，于是：

$$
\mathbb E[T]=\sum_{k=1}^{\infty}(1-p)^{k-1}=\frac1p.
$$

<details markdown="1">
<summary>同一道题，用首步分析再推一次</summary>

记 $m=\mathbb E[T]$。先花一次机会；失败后剩余等待时间与原来同分布，所以 $m=1+(1-p)m$，解得 $m=1/p$。上面的尾和已经证明它有限，因此这里的移项有依据。

若不先检查有限性，写出一个期望递推式并不自动证明它有有限解。

</details>

## 4 · 方差：把平方展开，不要漏交叉项

假设二阶矩有限。由定义展开：

$$
\operatorname{Var}(X)=\mathbb E[(X-\mathbb E X)^2]=\mathbb E[X^2]-(\mathbb E X)^2.
$$

对和展开同一个平方：

$$
\operatorname{Var}(X+Y)=\operatorname{Var}(X)+\operatorname{Var}(Y)+2\operatorname{Cov}(X,Y).
$$

其中 $\operatorname{Cov}(X,Y)=\mathbb E[XY]-\mathbb E[X]\mathbb E[Y]$。独立可推出 $\mathbb E[XY]=\mathbb E[X]\mathbb E[Y]$，所以协方差为零；但零协方差不必独立。

| 情形 | 和的方差 | 原因 |
| --- | --- | --- |
| $Y=X$ | $4\operatorname{Var}(X)$ | 两份完全同向 |
| $Y=-X$ | $0$ | 正好抵消 |
| $X,Y$ 独立 | $\operatorname{Var}(X)+\operatorname{Var}(Y)$ | 交叉项为零 |

<details markdown="1">
<summary>给一个“不相关，但不独立”的例子</summary>

让 $X$ 等概率取 $-1,0,1$，$Y=X^2$。则 $\mathbb E[X]=\mathbb E[X^3]=0$，所以 $\operatorname{Cov}(X,Y)=0$。但知道 $X=0$ 就知道 $Y=0$，而 $P(Y=0)=1/3$，显然不是独立。

</details>

## 5 · 再推一个熟悉的结果

独立 Bernoulli($p$) 变量 $I_1,\ldots,I_n$ 的和 $S$ 服从 Binomial($n,p$)。由于 $I_i^2=I_i$，每项方差为 $p-p^2$。于是：

$$
\mathbb E[S]=np,\qquad \operatorname{Var}(S)=np(1-p).
$$

期望那一步不需要独立；方差这一步用了它。若 $I_i$ 全是同一枚硬币的结果复制，期望仍是 $np$，方差却是 $n^2p(1-p)$。

## 接下来

下一篇：[先分组再平均——条件期望](conditioning-proofs.md)。对照阅读：[MIT 6.041 讲义目录中的第 5、7、11 讲](https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/pages/lecture-notes/)。
