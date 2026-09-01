# 归一化：BatchNorm、LayerNorm 与 RMSNorm

**中文** · [English](normalization.en.md)

> 阅读时间：约 12 分钟 · 难度：必修 · 最近审阅：2026-09

<div class="lesson-recipe">
  <div><span>解决什么问题</span><strong>让深层网络中的数值尺度与梯度路径更可控</strong></div>
  <div><span>前置知识</span><strong>残差连接 · 均值与方差 · Jacobian</strong></div>
  <div><span>核心机制</span><strong>沿哪条轴统计，以及 Norm 放在残差分支的哪里</strong></div>
  <div><span>常见错误</span><strong>忘掉 final LN，或笼统地说 BatchNorm 一定泄漏未来</strong></div>
</div>

## 30 秒建立 mental model

| 概念 | 一句话记忆 | 面试关键词 |
| --- | --- | --- |
| 为什么归一化 | 让每个 sublayer 看到尺度可预测的输入，使优化对参数尺度不那么敏感 | stable activations · conditioning · larger learning rate |
| 为什么不用 BatchNorm | 一个 token 的表示不应依赖 batch 里的其他样本或未来位置 | variable length · padding · train/eval mismatch · causality |
| Pre-LN vs Post-LN | Pre-LN 把 Norm 移出残差主干，留下 identity gradient path | $I+J_fJ_{\mathrm{LN}}$ · final LN |
| RMSNorm | 保留 re-scaling，去掉 re-centering | RMS only · no mean subtraction · cheaper reduction |

> **Normalization 不是为了让所有表示永久保持“均值 0、方差 1”，而是为了控制送进每个子层的数值尺度，并让深网络更容易优化。**

## 核心区别：归一化轴不同

**BatchNorm 沿着「一批样本」求统计量，LayerNorm 沿着「一个样本自己的特征」求。**

剩下的差别，能不能处理变长、batch size 能不能是 1、能不能自回归生成、训练和推理是否一致，全都能从这句话推出来。

![三种归一化各自沿哪条轴](../assets/norm-axes.svg)

## 三个公式

**BatchNorm**（对每个特征 $j$，在批次维上统计）：

$$\mu_j = \frac{1}{N}\sum_{i=1}^{N} x_{ij}, \qquad \sigma_j^2 = \frac{1}{N}\sum_{i=1}^{N}(x_{ij}-\mu_j)^2$$

$$\hat{x}_{ij} = \frac{x_{ij}-\mu_j}{\sqrt{\sigma_j^2+\epsilon}}, \qquad y_{ij} = \gamma_j \hat x_{ij} + \beta_j$$

**LayerNorm**（对每个样本 $i$，在特征维上统计）：

$$\mu_i = \frac{1}{d}\sum_{j=1}^{d} x_{ij}, \qquad y_{ij} = \gamma_j\,\frac{x_{ij}-\mu_i}{\sqrt{\sigma_i^2+\epsilon}} + \beta_j$$

**RMSNorm**（LayerNorm 去掉减均值，也去掉 $\beta$）：

$$y_{ij} = \gamma_j\,\frac{x_{ij}}{\sqrt{\frac{1}{d}\sum_{k} x_{ik}^2+\epsilon}}$$

三种写法里，$\gamma$ 和 $\beta$ 都是**按特征维**走的，长度都是 $d$。真正变的只有一件事：统计量沿哪条轴算。

## 四个概念：先会答，再深挖

<details class="interview" markdown="1">
<summary>1. 为什么深层网络需要 normalization？</summary>

**快速学习**

残差网络不断做

$$
x_{\ell+1}=x_\ell+f_\ell(x_\ell).
$$

如果每层分支输出的尺度都没有约束，残差相加会让不同深度的激活尺度逐渐漂移；反向传播时，不同方向的曲率和梯度尺度也可能相差很大。结果不是一定单调爆炸，而是**同一个学习率很难同时适合所有层和所有方向**。

Normalization 给每个 sublayer 一个尺度更可预测的输入，降低训练对初始化和参数尺度的敏感度，通常改善有效条件数（effective conditioning），因此更容易使用较大的学习率并训练更深的网络。

**面试版回答**

> 残差流会持续累积各层更新；若 sublayer 输入尺度不断漂移，激活和梯度的数值范围会越来越难控制，优化问题也会变得病态。LayerNorm / RMSNorm 把每个 token 送入子层前的尺度拉回稳定范围，改善梯度传播和优化条件，因此深层 Transformer 更容易训练。

<details markdown="1">
<summary><b>深挖</b>：残差尺度、condition number 与“稳定”到底指什么？</summary>

如果暂时假设各层残差更新近似不相关，

$$
\operatorname{Var}(x_L)
\approx
\operatorname{Var}(x_0)+
\sum_{\ell=0}^{L-1}\operatorname{Var}\big(f_\ell(x_\ell)\big).
$$

这解释了为什么残差流的尺度可能随深度增长。但真实网络里各项相关、权重也会适应，所以它不是“方差必然线性增长”的定理。更准确的说法是：**深度带来了 scale drift，Norm 负责让每个分支面对一个受控的输入尺度。**

从优化角度看，如果 Hessian 在不同方向的特征值跨度很大，一个学习率会在高曲率方向发散、在低曲率方向又走得太慢。Normalization 不能保证全局 Hessian 一定漂亮，但它减少了层与层之间的尺度差异，通常让梯度对参数扰动更平滑、有效 conditioning 更好。

还要注意：Pre-LN 并没有让残差流 $x_\ell$ 自身始终保持单位方差；它只是先把 $\operatorname{Norm}(x_\ell)$ 送进 Attn / FFN。因此末尾仍需要 final LN 把交给输出头的尺度收回来。

</details>

</details>

<details class="interview" markdown="1">
<summary>2. 为什么 Transformer 通常不用 BatchNorm？</summary>

**快速学习**

1. **Variable length + padding**：批内长度不同，统计量可能被 padding 污染；即使做 mask，后部位置的有效样本也很少。
2. **依赖 batch composition**：训练时换一组同行样本，同一个 token 的输出也会改变；小 batch 的统计量尤其噪。
3. **Train / inference mismatch**：训练用当前 batch statistics，推理通常改用 running statistics。自回归解码常是小 batch 或 batch size 1，不能依赖现场批统计量。
4. **可能破坏 causality**：如果实现把 time 轴也纳入统计，未来 token 会通过均值和方差影响过去 token，绕过 causal attention mask。

LayerNorm 对每个 token 自己的 hidden features 做统计，与 batch、序列中其他位置和训练/推理模式都无关。

**面试版回答**

> Transformer 使用 LayerNorm，是因为 LN 对每个 token 独立地沿 hidden dimension 归一化；它不依赖 batch size、序列长度或其他 token，训练与推理一致。BatchNorm 会引入 batch composition、padding 和 running statistics 问题；若统计还跨 time 轴，则会把未来 token 泄漏到过去位置。

<details markdown="1">
<summary><b>深挖</b>：BatchNorm 什么时候真的会泄漏未来？</summary>

必须看张量布局和 reduction axes，不能笼统说“BN 一定泄漏”。

- 若对 $X\in\mathbb R^{B\times T\times d}$ 的每个位置 $t$，只沿 $B$ 统计，那么当前位置不会直接读取同一序列的未来 token；但输出仍依赖同 batch 的其他样本，而且每个位置的有效样本数不同。
- 若像常见的 sequence <code>BatchNorm1d</code> 用法那样，对每个 channel 同时沿 $B$ 和 $T$ 统计，那么 $\mu_j,\sigma_j$ 包含 $t'>t$ 的 token。此时位置 $t$ 的归一化结果已经带有未来信息，causal mask 只限制 attention，挡不住这条旁路。
- 如果先把 $B\times T$ 展平再做 BN，也会出现同样的问题。

所以最准确的说法是：

> **BatchNorm 的统计轴一旦包含 time，就破坏自回归因果性；即使不包含 time，它仍有 batch dependence、padding、small-batch 和 train/eval mismatch，因此依然不是语言模型的好默认选择。**

推理时 batch size 1 并不意味着 BatchNorm 一定报错：在 eval mode 它可以使用 running statistics。真正的问题是这些统计来自训练分布，不是当前 token 自己，而且 train / inference 执行的是两套不同函数。

</details>

</details>

<details class="interview" markdown="1">
<summary>3. Pre-LN 和 Post-LN 放在哪里？为什么 Pre-LN 更稳？</summary>

**先会写结构**

<pre><code>Post-LN（2017 原版）             Pre-LN（现代常见）
x = LN(x + Attn(x))             x = x + Attn(LN(x))
x = LN(x + FFN(x))              x = x + FFN(LN(x))
                                 ...
                                 x = LN(x)   ← final LN</code></pre>

**面试版回答**

> Post-LN 把 LayerNorm 放在残差相加之后，因此残差主干的梯度每层都要穿过一次 LN。Pre-LN 把 LN 移进分支，主干保留直接的 identity path；每个 block 的 Jacobian 都显式包含 identity 项，所以深层梯度传播更稳定。代价是残差流本身不被逐层归一化，标准 Pre-LN 架构需要在所有 block 后补 final LN。

<details markdown="1">
<summary><b>深挖</b>：用 Jacobian 看 identity path</summary>

忽略多分支细节，Post-LN 写成

$$
x_{\ell+1}=\operatorname{LN}\big(x_\ell+f_\ell(x_\ell)\big).
$$

它的 Jacobian 是

$$
\frac{\partial x_{\ell+1}}{\partial x_\ell}
=
J_{\operatorname{LN}}\big(I+J_{f_\ell}\big).
$$

跨很多层时，梯度反复乘上 $J_{\operatorname{LN}}$。它不代表梯度必然每层缩小，但会打断纯粹的 identity highway，使深层训练更依赖初始化、residual scaling 和 learning-rate warmup。

Pre-LN 写成

$$
x_{\ell+1}=x_\ell+f_\ell\big(\operatorname{LN}(x_\ell)\big),
$$

于是

$$
\frac{\partial x_{\ell+1}}{\partial x_\ell}
=
I+J_{f_\ell}J_{\operatorname{LN}}.
$$

即使分支梯度很小，$I$ 仍提供一条直接路径。更严谨的说法是 Pre-LN **显著降低**深层训练对 warmup 的依赖，而不是保证所有配置都完全不需要 warmup。

Pre-LN 也有 trade-off：残差流尺度会增长，后期层的增量相对主干可能越来越小，产生“effective depth”不足的问题。final LN 解决输出尺度，不自动解决所有表达效率问题。

</details>

</details>

<details class="interview" markdown="1">
<summary>4. RMSNorm 为什么只做 re-scaling 也能工作？</summary>

**快速学习**

LayerNorm 做两件事：

$$
x\longmapsto x-\mu
\quad\text{（re-centering）},
\qquad
x\longmapsto \frac{x}{\sqrt{\operatorname{Var}(x)+\epsilon}}
\quad\text{（re-scaling）}.
$$

RMSNorm 去掉减均值，通常也不使用 $\beta$，只保留

$$
x\longmapsto
\frac{x}{\sqrt{\frac{1}{d}\sum_jx_j^2+\epsilon}}\odot\gamma.
$$

它仍能稳定现代 LLM，说明很多场景里最关键的是控制向量长度和子层输入尺度，而不是强制每个 token 的特征均值为 0。

**面试版回答**

> RMSNorm 保留了 LayerNorm 的尺度控制和可学习 gain，但去掉 mean subtraction 与 bias。它少一次均值归约，计算和通信更简单；实践上质量通常不受明显影响，说明 re-scaling 往往比 re-centering 更关键。

<details markdown="1">
<summary><b>深挖</b>：能否由 RMSNorm 成功推出“均值完全没用”？</summary>

不能。RMSNorm 的成功是很强的经验信号，但不是数学证明。LayerNorm 的分母是标准差，

$$
\sqrt{\frac{1}{d}\sum_j(x_j-\mu)^2+\epsilon},
$$

RMSNorm 的分母是二阶矩，

$$
\sqrt{\frac{1}{d}\sum_jx_j^2+\epsilon}.
$$

当特征均值接近 0 时，两者非常接近；当均值偏移明显时，两者不同。模型可以通过前后线性层、bias-free 设计和训练过程适应这种差异。

工程上 RMSNorm 的主要收益是少算一次 mean reduction，并减少相关的同步与内存流量。实现时平方和通常使用 fp32 accumulation，避免 bf16 / fp16 的数值误差。

</details>

</details>

## 怎样选择归一化方法

| 场景 | 用什么 | 为什么 |
| --- | --- | --- |
| CNN 图像分类，batch 够大且固定 | **BatchNorm** | 批统计量稳定，顺带带来正则化效果，通常还更快收敛 |
| 任何 Transformer / 语言模型 | **LayerNorm / RMSNorm** | 变长、batch 可能为 1、生成必须确定 |
| RNN / LSTM | **LayerNorm** | 时间步之间批统计量不可比 |
| batch size 很小（检测、分割、大模型微调） | **GroupNorm / LayerNorm** | BatchNorm 在小批次下统计量噪声太大 |
| 强化学习、在线学习 | **LayerNorm** | 数据分布随策略变化，滑动平均会一直滞后 |
| GAN 判别器 | 常用 **InstanceNorm / LayerNorm** | 避免同批次样本互相泄漏信息 |

一句判据：**只要「同一个输入在不同批次里应该得到同一个输出」是硬要求，就不要让 normalization 使用 batch statistics。**

现代大模型常进一步选择 RMSNorm：少一次 mean reduction，计算、同步和内存流量更简单，实践中通常能保持相近质量。

<details markdown="1">
<summary><b>补充</b>：为什么 internal covariate shift 不是完整解释？</summary>

原论文的说法是减少 internal covariate shift。后来 [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604) 用实验反驳了这个解释——他们在 BN 之后人为注入分布噪声，训练依然又快又稳。

更稳妥的表述是：归一化降低了模型对参数尺度的敏感度，并经常让 loss landscape 与梯度更平滑，从而改善优化；这不是对任意网络全局条件数的无条件保证。

还有一个常被忽略的角度：归一化把权重的**尺度**自由度消掉了。$\text{Norm}(\alpha W x) = \text{Norm}(Wx)$，所以权重整体放大不改变输出——只改变有效学习率。这也是为什么归一化层通常不做 weight decay。

</details>

## 动手验证

[`../code/norm_compare.py`](../code/norm_compare.py) 用同一批激活值分别跑三种归一化，打印各自沿哪条轴统计、以及把 batch size 降到 1 时 BatchNorm 怎么塌掉。

图由 [`../code/make_norm_figures.py`](../code/make_norm_figures.py) 生成。

## 自检

<div class="taste-check">
  <strong>如果真的理解了，你应该能解释：</strong>
  <ol>
    <li>对 $B\times T\times d$ 的激活，LayerNorm 沿哪条轴统计？</li>
    <li>为什么“BatchNorm 一定泄漏未来”不够严谨？什么实现会真的泄漏？</li>
    <li>写出 Pre-LN 与 Post-LN，并指出 final LN 在哪里。</li>
    <li>为什么 Pre-LN 的 Jacobian 里有一条 identity path？</li>
    <li>RMSNorm 去掉了什么？它的成功能否证明 re-centering 永远没用？</li>
  </ol>
</div>

## 继续阅读

归一化让每层输入的尺度可控，但深度真正可行还差另一半——[残差连接](residual-connections.md)。
