# Scaling：学习率（LR）、batch size 和 schedule 怎么跟着放大

**中文** · [English](hyperparameters.en.md)

> 阅读时间：约 13 分钟 · 最近审阅：2026-10

在一个窄的小模型上搜索到最好的学习率（LR），原样搬到宽 8 倍的大模型上，常见的结果是训练不稳定，或者明显不是最优。原因不止一个：标准参数化（SP）下，最优学习率会随宽度移动；有用的 batch size 上限会随 loss 下降和数据量增长而变大；cosine schedule 的形状还依赖训练总长度。

放大时要分开处理这三件事。先看临界 batch size：超过它以后，batch 每翻一倍，省下的步数越来越少，处理的数据却越来越多。选择学习率时，可以用按规模拟合的经验公式作为起点，也可以用最大更新参数化（µP）一类方法，让学习率在不同宽度之间迁移。至于 schedule，可以让它对准训练长度，也可以用 warmup-stable-decay（WSD），推迟到训练后期再决定何时结束。

先读[一次训练](../deep-dives/training-step.md)和[从 SGD 到 AdamW](../deep-dives/optimizers.md)会更顺：要用到 optimizer step、梯度累积和 Adam 的更新式。教学预算沿用[第 1 篇](compute-optimal.md)的 $C=10^{22}$ FLOPs。

## Batch size：多大以后就不划算

一次更新用多少 token，决定两件事：要走多少步（墙钟时间），以及总共要看多少数据（算力）。[McCandlish 等（2018-12）](https://arxiv.org/abs/1812.06162)给了一个简单模型：训练到某个固定 loss，步数 $S$ 和处理的样本数 $E=BS$ 满足

$$
\left(\frac{S}{S_{\min}}-1\right)\left(\frac{E}{E_{\min}}-1\right)=1,\qquad B_{\rm crit}=\frac{E_{\min}}{S_{\min}}.
$$

$S_{\min}$ 是 batch 无限大时最少要走的步数，$E_{\min}$ 是 batch 很小时最少要看的数据量。整理一下就是 $S=S_{\min}(1+B_{\rm crit}/B)$、$E=E_{\min}(1+B/B_{\rm crit})$。他们论证 $B_{\rm crit}$ 大约等于梯度噪声尺度，简化版是 $\mathcal B_{\rm simple}={\rm tr}(\Sigma)/|G|^2$：单个样本梯度的方差之和，除以真实梯度长度的平方。噪声相对信号越大，越值得用大 batch 把它平均掉。

教学数字：$S_{\min}=10{,}000$ 步，$B_{\rm crit}=2\text{M}$ token，于是 $E_{\min}=200$ 亿 token。

| batch（token） | 步数 $S$ | 处理 token $E$ | 相对最少步数 | 相对最少数据 |
| --- | --- | --- | --- | --- |
| 0.5M | 50,000 | 250 亿 | 5.0 倍 | 1.25 倍 |
| 1M | 30,000 | 300 亿 | 3.0 倍 | 1.5 倍 |
| 2M | 20,000 | 400 亿 | 2.0 倍 | 2.0 倍 |
| 4M | 15,000 | 600 亿 | 1.5 倍 | 3.0 倍 |
| 16M | 11,250 | 1800 亿 | 1.1 倍 | 9.0 倍 |

在 $B_{\rm crit}$ 以下，batch 翻倍，步数接近减半，数据量增加不多，相当于用更多卡缩短训练时间；在它以上，batch 从 4M 加到 16M 只省 25% 的步数，处理的数据量却变成原来的 3 倍。正好在 $B_{\rm crit}$ 时，步数和数据量都是各自最小值的 2 倍。

$B_{\rm crit}$ 不是常数。Kaplan 等拟合出 $B_{\rm crit}(L)\approx2\times10^{8}/L^{1/0.21}$ token（$L$ 是 WebText2 上的 loss），loss 每降约 13%，临界 batch 翻一倍；loss 从 3 降到 2，它从约 110 万涨到约 740 万 token。所以很多大训练前期用较小的 batch，后面逐步加大：Llama 3 405B 从 4M token 起，训到 252M token 时翻到 8M，2.87T token 时再翻到 16M；DeepSeek-V3 在前 469B token 里把 batch size 从 3072 逐步加到 15360（报告没写单位；按它 4K 的预训练序列长度看应是序列数，约 1260 万到 6290 万 token）。

它主要跟着什么变？[Zhang 等（2024-10，ICLR 2025）](https://arxiv.org/abs/2410.21676)在 C4 上训 8500 万到 12 亿参数的模型，把临界 batch 定义为比线性加速多花 20% 步数时的 batch，再把模型大小和数据量拆开看：两者按 Chinchilla 比例一起放大时，临界 batch 约按 $D^{0.47}$ 增长；固定数据量只放大模型，指数只剩 0.087；固定 302M 的模型只加数据，临界 batch 照样变大。所以增长主要来自训练数据量 $D$，而不是模型大小。[Bergsma 等（2025-05，NeurIPS 2025）](https://arxiv.org/abs/2505.13738)得到相近的结论：最优 batch 约 $\propto D^{0.4}$，临界 batch 约 $\propto D^{0.5}$，都与模型大小无关。一个 batch 怎样分到多张卡上，见[多卡训练](../../06-systems/distributed-training.md)。

## 学习率：跟着 batch 和规模一起变

batch 变了，学习率通常也要跟着变。两条常见的换算规则：

| 规则 | 出处 | batch 乘以 $\kappa$ 时 | 适用范围 |
| --- | --- | --- | --- |
| 线性 | [Goyal 等（2017-06）](https://arxiv.org/abs/1706.02677) | 学习率乘 $\kappa$，前期配 warmup | SGD；他们在 ImageNet 上把 batch 用到 8192 张图 |
| 平方根 | [Malladi 等（2022-05）](https://arxiv.org/abs/2205.10287) | 学习率乘 $\sqrt\kappa$；$1-\beta_1$、$1-\beta_2$ 乘 $\kappa$；$\epsilon$ 除以 $\sqrt\kappa$ | Adam / RMSprop；推导假设梯度噪声远大于梯度本身，$\kappa$ 越大误差界越松 |

例：AdamW 原来 batch 1M token、学习率 $3\times10^{-4}$、$\beta_2=0.95$，batch 加到 4M（$\kappa=4$）。平方根规则给出学习率 $6\times10^{-4}$、$\beta_2=1-4\times0.05=0.80$；线性规则给出 $1.2\times10^{-3}$。$\beta_2=0.80$ 在大模型预训练里很少见，这正说明规则推导的是“噪声主导”的小 batch 区间，batch 已经接近临界值时不能照搬。

另一条路是直接拟合。[DeepSeek LLM（2024-01）](https://arxiv.org/abs/2401.02954)在 $10^{17}$ 到 $2\times10^{19}$ FLOPs 上做网格搜索，把 loss 比最好值高不超过 0.25% 的配置都算“接近最优”，拟合出

$$
\eta_{\rm opt}=0.3118\,C^{-0.1250},\qquad B_{\rm opt}=0.2920\,C^{0.3271}.
$$

这里的 $C=M\cdot D$，$M$ 是每个 token 不含 embedding 的 FLOPs（含 attention 的计算），$B$ 以 token 计。代入 $C=10^{22}$：$\eta_{\rm opt}=0.3118\times10^{-2.75}\approx5.5\times10^{-4}$，$B_{\rm opt}=0.2920\times10^{7.196}\approx4.6\text{M}$ token。论文自己强调：接近最优的区域很宽，所以这只是起点；同样的算力下，模型和数据分配不同，最优区域也会挪。Bergsma 等则认为按 $C$ 拟合 batch 有误导性，因为它实际主要取决于 $D$。

[Power Lines 的 §2.4](https://arxiv.org/html/2505.13738v2) 用 AdamW 的时间尺度 $\tau=B/(\eta\lambda D)$ 来联系 batch 和 weight decay。这里 $\eta$ 是峰值学习率，$\lambda$ 是 weight decay。**“batch 加倍，weight decay 也加倍”的前提是峰值学习率不变**，同时固定 $N$、$D$，以及按训练进度定义的 schedule 形状。论文在所测的 batch 区间里发现这个关系有用；超过临界 batch 后，最优时间尺度也开始偏移。如果学习率也变了，要在固定 $D$ 时保持 $\tau$，应按下面的比例换算：

$$
\frac{\lambda_{\rm new}}{\lambda_{\rm old}}=
\frac{B_{\rm new}/B_{\rm old}}{\eta_{\rm new}/\eta_{\rm old}}.
$$

接着上面的例子，batch ×4、学习率 ×2，weight decay 就应 ×2，而不是 ×4。这只是保持了公式中的时间尺度，不保证新配置就是最优。模型或数据规模改变时，论文把最优 $\tau$ 拟合为约 $(D/N)^{-0.5}$，并不是一直固定它。Qwen3 报告（2025-05）也说为每个预训练阶段拟合了预测学习率 schedule 和 batch size 的 scaling law，但没有公开系数。

## Schedule：cosine 要先定长度，WSD 可以最后再定

cosine 是最常见的形状：warmup 之后，学习率按

$$
\eta(t)=\eta_{\min}+\tfrac12(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

从 $\eta_{\max}$ 降到 $\eta_{\min}$，这个退火形状来自 [SGDR](https://arxiv.org/abs/1608.03983)。Chinchilla 衰减 10 倍，即 $\eta_{\min}=0.1\eta_{\max}$：$t/T=0.5$ 时学习率是 $0.55\eta_{\max}$，$t/T=0.8$ 时是 $0.19\eta_{\max}$。麻烦在于 $T$ 必须事先定好。计划训 1000 亿 token、却在 500 亿停下，此时学习率还有 $0.55\eta_{\max}$，没退下来，loss 比一开始就按 500 亿规划的 run 高。Chinchilla 附录 B 发现，周期比实际训练长出 25% 以上就明显变差，衰减 10 倍也比只衰减 5 倍好。第 1 篇讲的 Kaplan 估计偏差，就有一部分来自这里。

warmup-stable-decay（WSD）把 schedule 拆成三段：warmup，长时间保持 $\eta_{\max}$，最后一小段快速降下来。[MiniCPM（2024-04）](https://arxiv.org/abs/2404.06395)提出这个名字，发现衰减阶段占总 token 的 10% 就够，2.5% 不够；[Hägele 等（2024-05，NeurIPS 2024）](https://arxiv.org/abs/2405.18392)发现 20% 长的 cooldown 能追平 cosine，$1-\sqrt{\cdot}$ 形状的下降比线性好。[Wen 等（2024-10）](https://arxiv.org/abs/2410.05192)假设 loss 地形像一条河谷，用它解释为什么会这样：大学习率让参数在两侧谷壁之间来回振荡，同时沿谷底前进，所以 loss 看起来降得慢；快速衰减把振荡压下去，之前沿谷底走过的距离才显现出来。

WSD 的好处是“训多久”可以最后再定。教学例子：要比较 250 亿、500 亿、1000 亿 token 三个训练长度。

| 做法 | 怎么跑 | 一共训练多少 token |
| --- | --- | --- |
| cosine | 三个长度各跑一次 | 250 + 500 + 1000 = 1750 亿 |
| WSD，cooldown 占 20% | 一条常数学习率主干跑到 800 亿，在 200 亿、400 亿、800 亿处各分出一支，分别衰减 50 亿、100 亿、200 亿 | 800 + 50 + 100 + 200 = 1150 亿 |

省约三分之一；要比较的长度越多，省得越多。Hägele 等估计，Chinchilla 的扫描如果用 cooldown 复用 run，总算力可以从约 $5.59\times10^{23}$ 降到 $2.36\times10^{23}$ FLOPs。

大模型的实际配方多半介于两者之间：

| 模型 | 学习率形状 | 出处 |
| --- | --- | --- |
| DeepSeek LLM 7B / 67B | 2000 步 warmup；训完 80% token 后降到峰值的 31.6%，90% 后降到 10% | 2024-01 报告 §2.3 |
| Llama 3 405B | 8000 步 warmup 到 $8\times10^{-5}$，cosine 在 120 万步内降到 $8\times10^{-7}$；最后 4000 万 token 线性降到 0 | 2024-07 报告 §3.4 |
| DeepSeek-V3 | 2000 步 warmup 到 $2.2\times10^{-4}$，保持到 10T token，再用 4.3T token 的 cosine 降到 $2.2\times10^{-5}$；最后 500B token 先在这个值保持 333B，再降到 $7.3\times10^{-6}$ 训 167B | 2024-12 报告 §4.2 |
| Kimi K2 | 报告称之为 WSD：500 步 warmup，前 10T token 保持 $2\times10^{-4}$，之后 5.5T token 的 cosine 降到 $2\times10^{-5}$ | 2025-07 报告 |
| OLMo 2 | warmup 后 cosine 降到峰值的 10%（7B 在 4T token 处截断），mid-training 再线性降到 0 | 2024-12 报告 |

## µP：让学习率在宽度之间可迁移

标准参数化（SP）指框架默认的初始化、所有层共用一个学习率。在它之下，最优学习率会随宽度移动。[Tensor Programs V（Yang 等，2022-03，下称 TP5）](https://arxiv.org/abs/2203.03466)用 SGD 在 CIFAR-10 上训 MLP，看到宽度从 256 增到 8192，最优学习率移动了大约一个数量级，用最窄模型的最优学习率去训最宽的模型，效果很差甚至发散；Transformer 上也是如此。

为什么会移？看一个隐藏层 $h=Wx$，$W$ 是 $n\times n$ 矩阵，输入 $x$ 的每个分量是 $O(1)$。按 [Adam 原论文的算法 1](https://arxiv.org/abs/1412.6980)，从零初始化的一、二阶矩出发，不计 weight decay，第一步经偏差修正后的更新是 $\Delta W_{ij}=-\eta g_{ij}/(|g_{ij}|+\epsilon)$。在非零梯度的大小远大于 $\epsilon$ 时，可近似为 $-\eta\,{\rm sign}(g_{ij})$。batch 为 1 时 $g_{ij}=\delta_ix_j$（$\delta_i$ 是 loss 对 $h_i$ 的梯度），于是

$$
(\Delta W x)_i\approx-\eta\,{\rm sign}(\delta_i)\sum_{j=1}^{n}|x_j|\approx-\eta\,{\rm sign}(\delta_i)\,n\,\mathbb E|x|.
$$

这里保留 $\operatorname{sign}(\delta_i)$ 来区分更新方向；当 $\delta_i\ne0$ 且上述近似成立时，变化幅度是 $|\Delta h_i|\approx\eta n\mathbb E|x|$。若 $\delta_i=0$，这一行的更新为零。下面的手算和代码都采用忽略 $\epsilon$ 的理想化更新。

关键在于更新和 $x$ 是对齐的：$n$ 项同号相加，大小随 $n$ 线性增长。如果更新方向和 $x$ 无关，正负会互相抵消，大小只按 $\sqrt n$ 增长。$x_j$ 都取 ±1 时手算：$\eta=0.01$、$n=512$ 时 $|\Delta h_i|=5.12$；同样的学习率放到 $n=4096$，变成 40.96，激活一步之内的变化大了 8 倍。要让它保持 5.12，隐藏层学习率得乘 $512/4096$，即 $0.00125$。这就是 µP 对 Adam 规定隐藏层学习率 $\propto1/\text{fan\_in}$ 的直觉。用高斯输入复算一次：

```python
import random

def one_step_change(width, lr, seed=0):
    rng = random.Random(seed)
    x = [rng.gauss(0, 1) for _ in range(width)]          # 上一层激活，每个分量 O(1)
    update = [-lr * (1 if xj > 0 else -1) for xj in x]   # Adam 第一步：-lr * sign(delta * x_j)，取 delta > 0
    return abs(sum(u * xj for u, xj in zip(update, x)))  # |(ΔW x)_i|

for width in (512, 4096):
    print(width, round(one_step_change(width, 0.01), 2),            # SP：所有宽度同一个学习率
          round(one_step_change(width, 0.01 * 512 / width), 2))     # µP：隐藏层学习率乘 base/width
# 512 4.17 4.17
# 4096 32.78 4.1
```

高斯输入时 $\mathbb E|x|=\sqrt{2/\pi}\approx0.80$，预测值 $0.01\times512\times0.80\approx4.1$、$0.01\times4096\times0.80\approx32.7$，和输出一致；不对齐的随机方向只有 $\eta\sqrt n$，即 0.23 和 0.64。

TP5 表 3 给出 Adam 下 µP 和 SP 的区别（只写随宽度的缩放，常数可以另加乘子）：

| 参数 | 初始化方差 | Adam 学习率 | 和 SP 的区别 |
| --- | --- | --- | --- |
| 输入层权重（embedding）和所有 bias | $1/\text{fan\_in}$ | 不随宽度变 | 无 |
| 隐藏层权重（attention 投影、MLP） | $1/\text{fan\_in}$ | $\propto1/\text{fan\_in}$ | 学习率随宽度缩小 |
| 输出层权重（logits） | $1/\text{fan\_in}^2$ | $\propto1/\text{fan\_in}$ | 初始化更小，学习率随宽度缩小 |
| attention logit | 用 $q^\top k/d$ | 不适用 | SP 用 $q^\top k/\sqrt d$ |

### 手算一次迁移

代理模型宽 512，在上面扫出全局 Adam 学习率 $\eta=0.01$（以及输出乘子等）。目标模型宽 4096，宽度比 $m=8$。按上表，并像原文式 4 那样把宽度因子写成相对代理模型的 $m=n/n_0$：embedding、bias 和 norm 的增益仍用学习率 0.01；隐藏层矩阵用 $0.01/8=0.00125$，初始化标准差仍按 $1/\sqrt{\text{fan\_in}}$；输出层的初始化方差取 $1/(\text{fan\_in}\cdot m)$，比 SP 再小 8 倍，学习率 $0.00125$；attention logit 用 $\sqrt{d_0}\,q^\top k/d$（$d$ 是 head 维度，$d_0$ 是代理模型的 head 维度；如果放宽时只加 head 数、head 维度不变，这一项不变）。$m=1$ 时这些因子都是 1，代理模型就是普通的 SP 模型，照常训练。原文还给了两种等价写法：表 8 让输出层初始化方差不随宽度变，改在前向乘一个 $1/\text{fan\_in}$ 的乘子，所有向量类参数共用一套规则，更容易实现，输入、输出 embedding 也能共享权重；表 9 是早先 TP4 论文的写法。

[MiniCPM](https://arxiv.org/abs/2404.06395) 用的是同一类做法：在 0.009B 的模型上搜索，二维权重的学习率按 $1/(d_m/d_{\rm base})$ 缩放、初始化标准差按 $1/\sqrt{d_m/d_{\rm base}}$ 缩放；模型从 0.04B 增到 0.5B，最优基础学习率一直在 0.01 附近，在 2.1B 上复核仍是 0.01（自报）。TP5 作者报告，用 4000 万参数的代理模型调出的超参数训练 6.7B 的 GPT-3，结果超过了原版 6.7B 公布的数字，调参花费约为预训练总算力的 7%（作者自报）。

边界也要记清：

- **不能迁移的。** TP5 表 1 把 dropout、weight decay 这类正则化列为不可迁移；batch size、训练长度、序列长度和深度上的迁移只有经验验证。
- **深度。** [Tensor Programs VI（2023-10，TP6）](https://arxiv.org/abs/2310.02244)给出每个残差块只有一层时的 Depth-µP，并指出 Transformer 这种多层块在深度趋于无穷时有根本性的限制。
- **不只 µP。** [Everett 等（2024-07，ICML 2024）](https://arxiv.org/abs/2407.05872)训练了数万个模型（最大 26.8B），发现他们比较的四种参数化都能做到超参数迁移，不只 µP；他们为 SP 设计的逐层学习率规则还胜过 µP；Adam 的 $\epsilon$ 也要随宽度正确缩放，否则梯度会下溢。
- **换了优化器要重新推。** Muon 按矩阵形状缩放更新，让逐元素的更新 RMS 不随矩阵大小变化，是另一条思路，见 [Muon](../deep-dives/muon.md)。

## 开训前的检查表

| 想做的事 | 怎么做 | 观察什么 | 不能据此得出 |
| --- | --- | --- | --- |
| 定 batch | 小规模测几个 batch 达到同一 loss 的步数和 token 数 | batch 翻倍时步数是否接近减半 | 临界 batch 在整个训练里不变 |
| 定学习率 | 在代理模型上按 2 倍间隔扫 5–7 个点，确认最优点不在边界 | loss 对学习率曲线的谷底位置和宽度 | SP 下的最优值能直接用于宽 8 倍的模型 |
| 验证 µP 实现 | 同一学习率下训 3 个宽度的短 run，记录各层激活的大小 | 各层激活的 RMS 是否随宽度保持稳定 | 宽度能迁移，深度也能迁移 |
| 选 schedule | 训练长度没定时用 WSD，在几个长度上分支衰减 | cooldown 前后 loss 的落差 | 常数学习率阶段的 loss 代表最终效果 |

最后：[复习题](review.md)。
