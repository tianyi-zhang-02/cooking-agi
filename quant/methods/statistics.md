# 统计推断：从样本到结论，中间还差什么

**中文** · [English](statistics.en.md) · [复习总览](../README.md)

> 阅读时间：约 11 分钟 · 前置：条件概率、方差、CLT、最小二乘 · 最近审阅：2026-10

概率题通常给模型，让你算数据会怎样；统计反过来：给有限数据，问模型参数或结论是否站得住。最大的问题不只是算错，还包括把数据收集过程当成透明的。

## 1 · 估计量也会波动

对 iid 样本，均值 μ、方差 σ² 有限，$\bar X$ 无偏且方差 σ²/n。估计量的均方误差拆成：

$$
E[(\hat\theta-\theta)^2]=\operatorname{Var}(\hat\theta)+[E(\hat\theta)-\theta]^2.
$$

证明就是加减 $E\hat\theta$ 后展开，交叉项期望为 0。无偏并不自动最好；稍有偏差但方差小很多，MSE 可能更低。

样本方差为什么除以 n−1？用恒等式 $\sum(X_i-\bar X)^2=\sum(X_i-\mu)^2-n(\bar X-\mu)^2$，取期望得到 $(n-1)\sigma^2$。n>1 时除以 n−1 无偏。

## 2 · MLE：先写似然，再谈最优

n 次 iid Bernoulli，s 次成功。似然 $L(p)=p^s(1-p)^{n-s}$。内部情形 0<s<n：

$$
\ell'(p)=\frac{s}{p}-\frac{n-s}{1-p}=0
\quad\Rightarrow\quad\hat p=\frac sn.
$$

二阶导为负，所以是唯一内部最大值。s=0 或 n 时最优在边界，不能用一个无效的内部求导把它漏掉。

正态均值未知时，MLE 为样本均值；方差也未知时，MLE 方差分母是 n，不是 n−1。**最大似然与无偏是不同目标。**

Fisher information 衡量似然对参数的局部敏感度。Bernoulli 每个样本的信息量是 $1/[p(1-p)]$，p∈(0,1)。在适当正则条件下，Cramér–Rao 给无偏估计量方差下界；参数相关支持集等情况不能直接套。这里介绍结论，不给一般证明。

## 3 · Bayes：数据之外的假设要摆在台面上

若 p 的先验是 Beta(α,β)，后验为 Beta(α+s,β+n−s)，均值 $(\alpha+s)/(\alpha+\beta+n)$。

例：先验 Beta(2,2)，10 次中 7 次成功，后验 Beta(9,5)，后验均值 9/14，而 MLE 是 0.7。样本很少时，两者不同不意味着谁计算错了。

<details markdown="1">
<summary>基础例子：阳性到底说明多少？</summary>

患病率 1%，敏感度 90%，特异度 95%。阳性后患病概率为 $0.9\cdot0.01/[0.9\cdot0.01+0.05\cdot0.99]=2/13$，约 15.4%。可以想成 10,000 人：90 个真阳性和 495 个假阳性。这里只是概率教学模型，不是医学判断。

</details>

观察机制同样重要。例如标准 Monty Hall：主持人知道答案、必开一个空门、必给换门机会，则换门赢 2/3。若主持人随机开门，恰好没开到奖品，在相应模型下条件概率会不同。条件里必须包含“消息如何产生”。

## 4 · 置信区间：不是参数有 95% 概率在里面

正态 iid、未知方差、n>1 时：

$$
\frac{\bar X-\mu}{S/\sqrt n}\sim t_{n-1}.
$$

这给出精确 t 区间。非正态数据一般依赖大样本近似，不能只看 n 就无视重尾和相关性。置信度描述的是重复抽样的覆盖率；贝叶斯可信区间描述给定模型与数据的后验概率。

一个样本中的 S 描述单次观测的分散，S/√n 描述均值估计的标准误（standard error）。它们不是同一个误差条。

## 5 · 假设检验：显著不代表有用

p-value 是**在零假设及抽样模型成立时**，得到当前统计量这样极端或更极端结果的概率。它不是“零假设为真的概率”。

| 概念 | 该问什么 |
| --- | --- |
| Type I error | 零假设真时，误拒绝的概率是否受控？ |
| Type II error / power | 哪种备择、哪个效应大小下，会漏掉多少？ |
| Effect size | 差了多少，是否有实际意义？ |
| Multiple testing | 是否挑了很多次，只展示最好的一次？ |

做 m 个检验，Bonferroni 把每个显著性水平设为 α/m，靠并集上界控制至少一次误拒绝概率，不需要检验独立。它可能保守；反复偷看数据后临时停下也会改变错误率。

## 6 · 回归：拟合恒等式与统计保证分开

线性模型 $y=X\beta+\varepsilon$。满列秩且 $E[\varepsilon\mid X]=0$ 时，OLS 条件无偏；若再有 $\operatorname{Var}(\varepsilon\mid X)=\sigma^2I$：

$$
\operatorname{Var}(\hat\beta\mid X)=\sigma^2(X^\top X)^{-1}.
$$

代入 $\hat\beta=\beta+(X^\top X)^{-1}X^\top\varepsilon$ 即可推导。不需要正态性才能无偏；精确小样本 t 检验通常还要额外分布假设。异方差、相关误差、遗漏变量要分别处理，不能靠“多一点样本”全部抹平。

## 7 · 拓展到研究：抽样方式也是模型的一部分

时间序列先按时间切训练与测试，防止未来信息泄漏；自相关下 iid 标准误可能太乐观。普通 bootstrap 独立重采样观测，不能原样保留时间依赖，需要考虑适当的块重采样等方法。

AR(1) 模型 $X_t=\phi X_{t-1}+\varepsilon_t$，独立零均值噪声、方差 σ²、|φ|<1 时存在平稳解，方差 $\sigma^2/(1-\phi^2)$，滞后 k 的相关为 φᵏ。由递推与独立性可以直接核对。|φ|≥1 不可套这套平稳公式。

<details markdown="1">
<summary>复习时自己讲：为什么样本翻倍，偏差可能完全没改善？</summary>

抽样一直漏掉同一类人、标签一直偏向某个行为，估计量会更稳定地估计错误目标。标准误变小不意味着选择偏差变小。要说清目标总体、采样机制、估计对象和验证数据。

</details>

参考：[MIT 18.05 推断讲义](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/pages/classes-reading-and-in-class-materials/)、[MIT 回归讲义](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/resources/mit18_s096f13_lecnote6/)。
