# 联合分布：先画范围，再动积分

**中文** · [English](joint-and-order.en.md) · [复习总览](../README.md)

> 阅读时间：约 10 分钟 · 前置：密度、条件概率、基本积分 · 最近审阅：2026-10

只知道 X 和 Y 各自长什么样，通常还不知道它们一起怎样变化。两个人各自均匀到达，可能完全独立，也可能约好同一时刻到达。

## 1 · 联合、边缘、条件，分别在固定什么

设联合密度 f(x,y)。把 y 积掉是边缘密度；固定 X=x 后再归一化，是条件密度：

$$
f_X(x)=\int f(x,y)\,dy,\qquad
f_{Y\mid X=x}(y)=\frac{f(x,y)}{f_X(x)},\quad f_X(x)>0.
$$

这里不是直接给零概率事件套初等条件概率，而是密度形式的条件分布。独立等价于联合密度几乎处处可分解为两个边缘密度的乘积。

取 $f(x,y)=2$ 在 $0<x<y<1$，其他地方为 0。三角形面积为 1/2，总概率正好为 1。对固定 x，y 从 x 到 1，所以 $f_X(x)=2(1-x)$。

<details markdown="1">
<summary>Y 给定 X=x 后是什么分布？</summary>

条件密度为 $1/(1-x)$，范围 x<y<1，即 Uniform(x,1)。所以 $E[Y\mid X=x]=(1+x)/2$。Y 的分布会随 x 变，二者不独立；联合密度“在支持集上恒定”不等于独立。

</details>

## 2 · 两个变量相加：卷积从哪儿来

独立连续变量 Z=X+Y：

$$
f_Z(z)=\int_{-\infty}^{\infty}f_X(x)f_Y(z-x)\,dx.
$$

给定总和 z，X=x 时 Y 就要在 z−x 附近；把所有合法 x 加起来。两个独立 Uniform(0,1) 相加，合法区间长度先增加、再减少：

$$
f_Z(z)=
\begin{cases}
z,&0<z<1,\\
2-z,&1\le z<2,\\
0,&\text{otherwise}.
\end{cases}
$$

检验总面积为 1，且 $E[Z]=1$。若不独立，积分里要用 $f_{X,Y}(x,z-x)$，不能硬乘边缘密度。

## 3 · 换变量：Jacobian 不是装饰

X、Y 独立 Exp(λ)。令 S=X+Y、R=X/(X+Y)。逆变换为 X=RS、Y=(1−R)S，范围 S>0、0<R<1，Jacobian 绝对值为 S：

$$
f_{S,R}(s,r)=\lambda^2s e^{-\lambda s}\,\mathbf1_{\{s>0,\ 0<r<1\}}.
$$

它分解成 Gamma(2,λ) 的密度与 Uniform(0,1) 的密度，所以 S 与 R 独立。换变量后丢掉 S，就会连总概率都算不对。

多对一变换还要把所有逆分支的贡献相加。比如 Z=X²，不能只保留正平方根，除非 X 本来只在正半轴。

## 4 · 第 k 小：先数有几个落在左边

设 n 个 iid、具有密度 f 和 CDF F 的样本，$X_{(k)}$ 为第 k 小。它≤x 等价于至少 k 个样本≤x：

$$
P(X_{(k)}\le x)=\sum_{j=k}^n\binom njF(x)^j[1-F(x)]^{n-j}.
$$

<details markdown="1">
<summary>把 CDF 换成密度</summary>

让一个样本落在很窄的区间，k−1 个在左侧，n−k 个在右侧，得到：

$$
f_{X_{(k)}}(x)=\frac{n!}{(k-1)!(n-k)!}
F(x)^{k-1}[1-F(x)]^{n-k}f(x).
$$

更严格地可直接对上面的有限和求导，项会相消。iid 是关键：不同分布的样本不能这样统一数。

</details>

Uniform(0,1) 时它是 Beta(k,n+1−k)，所以 $E[X_{(k)}]=k/(n+1)$。最大值平均 n/(n+1)，最小值平均 1/(n+1)。用 Beta 积分即可验证，而不是凭“均匀间隔”猜答案。

## 5 · 二维正态：这里不相关才等于独立

对联合正态的 X,Y，标准差非零，相关系数 |ρ|<1：

$$
Y\mid X=x\sim N\left(
\mu_Y+\rho\frac{\sigma_Y}{\sigma_X}(x-\mu_X),
\ \sigma_Y^2(1-\rho^2)\right).
$$

把联合密度关于 y 配方就能读出均值和方差。也可以写成 $Y=\mu_Y+\rho(\sigma_Y/\sigma_X)(X-\mu_X)+\sigma_Y\sqrt{1-\rho^2}Z$，其中 Z 独立于 X。

ρ=0 时条件分布不随 x 改变，故独立。**X、Y 各自正态还不够，必须联合正态。** |ρ|=1 是退化情形，要单独处理。

## 6 · 一个容易忽略的抽样模型

“随机取圆的一条弦”还没定义分布：均匀取两个端点、均匀取中点，会得到不同答案。先说清采样程序，再写所谓等可能结果。这也是几何概率里最重要的建模习惯。

<details markdown="1">
<summary>练习：两个独立 Uniform(0,1)，E[|X−Y|] 是多少？</summary>

先在 y<x 的三角形积分，再用对称乘 2：$2\int_0^1\int_0^x(x-y)\,dy\,dx=1/3$。求 P(X+Y≤1/2) 则是边长 1/2 的直角三角形面积，等于 1/8。面积法成立依赖单位正方形上的联合均匀密度。

</details>

继续：[统计推断](../methods/statistics.md)。参考：[MIT 18.05 概率与统计讲义](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/pages/classes-reading-and-in-class-materials/)。
