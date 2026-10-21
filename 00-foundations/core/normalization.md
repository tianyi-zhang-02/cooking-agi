# 归一化：BatchNorm、LayerNorm 与 RMSNorm

**中文** · [English](normalization.en.md)

> 阅读时间：约 17 分钟 · 最近审阅：2026-10

拿向量 `[1, 3]` 看归一化会比较直观。暂时忽略 epsilon 和可学习参数：LayerNorm 先减均值 2，再除以标准差 1，得到 `[-1, 1]`；RMSNorm 不减均值，而是除以均方根 √5。两者都改变尺度，却没有做同一件事。下面再把这笔计算放回一批序列里，看清到底沿哪个轴统计。

## 核心区别：归一化轴不同 {#_1}

**BatchNorm 沿着「一批样本」求统计量，LayerNorm 沿着「一个样本自己的特征」求。**

先看统计轴，再看 running statistics 与 train / eval 模式；只看轴还不够，三者合起来才能判断变长、batch size 为 1 和自回归生成时的行为。

![三种归一化各自沿哪条轴](../assets/norm-axes.svg)

图中每行是一条 feature vector。放到 `[B,T,d]` 序列里，逐 token 归一化对应 `LayerNorm(d)` / `RMSNorm(d)`；写成 `(T,d)` 就会把时间维也算进去。**是不是逐 token，要看 `normalized_shape`，不能只看层的名字。** [LayerNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html) 和 [RMSNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.RMSNorm.html) 的接口都允许归一化多个末尾维度。

## 三个公式 {#_2}

**BatchNorm**（对每个特征 $j$，在批次维上统计）：

$$\mu_j = \frac{1}{N}\sum_{i=1}^{N} x_{ij}, \qquad \sigma_j^2 = \frac{1}{N}\sum_{i=1}^{N}(x_{ij}-\mu_j)^2$$

$$\hat{x}_{ij} = \frac{x_{ij}-\mu_j}{\sqrt{\sigma_j^2+\epsilon}}, \qquad y_{ij} = \gamma_j \hat x_{ij} + \beta_j$$

**LayerNorm**（对每个样本 $i$，在特征维上统计）：

$$\mu_i = \frac{1}{d}\sum_{j=1}^{d} x_{ij}, \qquad \sigma_i^2 = \frac{1}{d}\sum_{j=1}^{d}(x_{ij}-\mu_i)^2, \qquad y_{ij} = \gamma_j\,\frac{x_{ij}-\mu_i}{\sqrt{\sigma_i^2+\epsilon}} + \beta_j$$

**RMSNorm**（按均方根缩放，常见写法只保留可学习增益 $\gamma$）：

$$y_{ij} = \gamma_j\,\frac{x_{ij}}{\sqrt{\frac{1}{d}\sum_{k} x_{ik}^2+\epsilon}}$$

这些写法中的 $\gamma$，以及使用时的 $\beta$，都是**按特征维**学习的，长度为 $d$。BatchNorm 与 LayerNorm 要先分清统计轴；LayerNorm 与 RMSNorm 则沿同一特征轴，但分母分别来自方差和原始二阶矩。RMSNorm 不是“假设均值已经等于 0 的 LayerNorm”。

把开头的 `[1, 3]` 整体加 10，得到 `[11, 13]`。忽略 epsilon，LayerNorm 仍输出 `[-1, 1]`；RMSNorm 的分母从 $\sqrt{5}$ 变成 $\sqrt{145}$，输出当然也变了。它没有去掉这次平移。至于 Norm 放在子层前还是后，那是 **Pre-Norm / Post-Norm 的另一个选择**，不能与使用哪种归一化公式混为一谈。[RMSNorm 原论文](https://arxiv.org/abs/1910.07467)讨论的正是去掉 re-centering 后的尺度归一化。

## 同一批数据，train 和 eval 为什么不一样？ {#train-eval-statistics}

上面的 BatchNorm 公式用二维输入举例；在 `[N,C,L]` 上，`BatchNorm1d` 对每个 channel 沿 `N,L` 统计。默认训练用当前输入的统计量，eval 用积累下来的 running statistics；`track_running_stats=False` 时 eval 也用当前输入。它们是统计状态，不是由优化器学习的 gamma / beta。[PyTorch 2.14 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm1d.html)明确区分了这些行为（2026-10 核对）。

先别急着归因于“训练和测试的数据分布不同”。**就算喂同一批 `[1,3]`，结果也可能不同。** 令 `eps=0`、不加 affine，并用 `momentum=1` 让 running statistics 直接采用这一批的值：

<figure class="worked-update worked-update--pairs" id="normalization-train-eval" lang="zh-CN">
<figcaption>同样输入 [1, 3]，分母不同</figcaption>
<ol>
<li><span>训练前向</span><strong>[-1, 1]</strong><p>均值 2；平方偏差之和为 2，除以样本数 2，方差为 1。</p></li>
<li><span>随后切到 eval</span><strong>约 [-0.707, 0.707]</strong><p>存储的方差用了样本数减一作分母，得到 2；这次除的是 √2。</p></li>
</ol>
</figure>

这里有两个分母：训练前向用 $v_b=\sum(x-\mu)^2/m$；更新 running variance 用 $v_u=\sum(x-\mu)^2/(m-1)$。$m$ 是每个 channel 参与统计的值数。若 `momentum=a`，更新为 $v_{\rm run}\leftarrow(1-a)v_{\rm run}+av_u$，均值也按同样的加权方式更新。这不是优化器的 momentum。[原始 BN 论文 §3.1](https://proceedings.mlr.press/v37/ioffe15.pdf)也区分了训练统计量与推理时的估计量。

<details markdown="1">
<summary>用几行代码复算（PyTorch 2.8 / CPU）</summary>

```python
import torch

def batchnorm_train_eval_example():
    inputs = torch.tensor([[1.0], [3.0]])
    norm = torch.nn.BatchNorm1d(1, eps=0, momentum=1, affine=False)
    training = norm(inputs)
    stored_variance = norm.running_var.clone()
    norm.eval()
    evaluation = norm(inputs)
    return training, evaluation, stored_variance

training, evaluation, stored_variance = batchnorm_train_eval_example()
print(training.flatten().tolist())
print(evaluation.flatten().tolist(), stored_variance.item())
```

默认 `momentum=0.1` 时还会混入之前的统计量，上面的数字就不适用了。这个例子故意选 `momentum=1`，只是为了单独看清方差估计的区别。代码在 PyTorch 2.8 / CPU 上运行过；2026-10 对照 2.14 文档，BatchNorm 的这些行为没有变。

</details>

### 梯度累积，并没有把 BN 的 batch 变大 {#batchnorm-accumulation}

一次喂 `[0,2,10,12]`，均值是 6；拆成 `[0,2]` 和 `[10,12]` 两次前向，每次均值分别是 1 和 11。梯度可以攒到一起再更新，**已经发生的两次归一化却不会合成一次**。BN 的 running statistics 也是每次前向更新，不是等到 `optimizer.step()` 才更新。

从 running mean 为 0、running variance 为 1 开始，用 `momentum=0.1`：

| 喂数据的方式 | 最后 running mean | 最后 running variance |
| --- | ---: | ---: |
| 4 个值一起前向 | 0.6 | 4.3667 |
| 分两次、每次 2 个值 | 1.19 | 1.19 |

第一行的样本方差是 $104/3$；第二行每批都是 2，但更新了两次。输入的归一化结果、后续 loss 和梯度也可能不同。因此不能把“有效 batch size 一样”当成带 BN 训练等价的证明。

还有一个常见混淆：`torch.no_grad()` 关闭梯度记录，不会切换 BN 模式；训练模式下仍会更新 running statistics。要用已有统计量做验证，通常是 `.eval()` 加 `no_grad()`，分别控制两件事。[Autograd 文档](https://docs.pytorch.org/docs/2.14/notes/autograd.html#evaluation-mode-nn-module-eval)

## 四个概念：先会答，再深挖 {#_3}

需要先补输入特征的 min–max 与 Z-score，可以看[预处理和网络 Norm 的区别](../deep-dives/generalization.md)。

<details class="interview" markdown="1">
<summary>1. 为什么深层网络需要 normalization？</summary>

**快速学习**

残差网络不断做

$$
x_{\ell+1}=x_\ell+f_\ell(x_\ell).
$$

如果每层分支输出的尺度都没有约束，残差相加会让不同深度的激活尺度逐渐漂移；反向传播时，不同方向的曲率和梯度尺度也可能相差很大。结果未必是单调爆炸，更常见的麻烦是**同一个学习率很难同时适合所有层和所有方向**。

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

1. **Variable length + padding**：普通 `BatchNorm1d` 不接收 padding mask，填充值会混入统计。若自己实现有效 token 的 pooled statistics，长度 `[3,1]` 会贡献 4 个 token；只有“每个位置单独统计”的实现，才会在后部位置只剩一个有效样本。
2. **依赖 batch composition**：训练时换一组同行样本，同一个 token 的输出也会改变；小 batch 的统计量尤其噪。
3. **Train / inference mismatch**：训练用当前 batch statistics，推理通常改用 running statistics。自回归解码常是小 batch 或 batch size 1，不能依赖现场批统计量。
4. **可能破坏 causality**：若用当前序列计算统计量，并把 time 轴也算进去，未来 token 会影响过去 token，绕过 causal attention mask。使用冻结的 running statistics 没有这条旁路。

Transformer 常用的 `LayerNorm(d)` 只统计每个 token 的 hidden features，不依赖 batch、其他位置或 train/eval 模式。`LayerNorm((T,d))` 则不是这个行为：修改最后一个 token，也会影响第一个 token 的输出。

**面试版回答**

> Transformer 通常沿每个 token 的 hidden dimension 使用 LN，训练与推理的归一化方式相同。BN 还要处理 batch composition、padding 和 running statistics；如果从当前序列跨 time 统计，就会读到未来位置。关键是统计轴和模式，而不只是层的名字。

<details markdown="1">
<summary><b>深挖</b>：BatchNorm 什么时候真的会泄漏未来？</summary>

必须看张量布局和 reduction axes，不能笼统说“BN 一定泄漏”。

- 若对 $X\in\mathbb R^{B\times T\times d}$ 的每个位置 $t$，只沿 $B$ 统计，那么当前位置不会直接读取同一序列的未来 token；但输出仍依赖同 batch 的其他样本；有 padding 时，每个位置的有效样本数还可能不同。
- 若像常见的 sequence <code>BatchNorm1d</code> 训练前向那样，从当前输入同时沿 $B$ 和 $T$ 统计，那么 $\mu_j,\sigma_j$ 包含 $t'>t$ 的 token。此时位置 $t$ 的归一化结果已经带有未来信息，causal mask 只限制 attention，挡不住这条旁路。
- 如果先把 $B\times T$ 展平，再用当前输入做 BN 统计，也会出现同样的问题。

所以最准确的说法是：

> **从当前序列跨 time 计算统计量，会让早期位置依赖未来。使用固定 running statistics 的 eval 不会；但 BN 的训练 / 推理差异，以及批次和 padding 的处理，仍要额外考虑。**

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

即使分支梯度很小，$I$ 仍提供一条直接路径。更严谨的说法是 Pre-LN **显著降低**深层训练对 warmup 的依赖，而不是保证所有配置都完全不需要 warmup。[Xiong 等（2020）](https://arxiv.org/abs/2002.04745)用 mean-field 分析得出：初始化时，Post-LN 靠近输出层的参数期望梯度较大，warmup 能避免初期更新过猛；Pre-LN 的梯度在初始化时表现良好，他们的实验里去掉 warmup 也能训练。这是论文设定下的结论，不是所有配置都“必须 / 不必 warmup”的定理。

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

它在现代 LLM 中效果很好，这提示在很多场景里，控制向量长度和子层输入尺度比强制每个 token 的特征均值为 0 更重要。

**面试版回答**

> RMSNorm 保留了 LayerNorm 的尺度控制和可学习 gain，但去掉 mean subtraction 与 bias。它少一次均值归约，计算和通信更简单；实践上质量通常不受明显影响，说明 re-scaling 往往比 re-centering 更关键。

<details markdown="1">
<summary><b>深挖</b>：能否由 RMSNorm 成功推出“均值完全没用”？</summary>

不能。RMSNorm 的成功是很强的经验信号，但不是数学证明。LayerNorm 的分母是标准差，

$$
\sqrt{\frac{1}{d}\sum_j(x_j-\mu)^2+\epsilon},
$$

RMSNorm 的分母是二阶矩加 epsilon 后的平方根，

$$
\sqrt{\frac{1}{d}\sum_jx_j^2+\epsilon}.
$$

判断“均值小不小”，要和向量本身的尺度比。由 $\mathbb E[x^2]=\operatorname{Var}(x)+\mu^2$ 可知，只有 $\mu^2\ll\operatorname{Var}(x)+\epsilon$ 时，两个分母才接近；输出还差了一步减均值。

比如 `[0.000001,0.000003]` 的均值已经很小，但忽略 epsilon，LN 仍输出 `[-1,1]`，RMSNorm 却约为 `[0.447,1.342]`。把输入一起缩小，并不会让这个区别消失。均值恰好为 0、epsilon 和 affine 设置也一致时，两个公式才重合；模型如何适应非零均值，仍是训练和架构的问题，不能只从公式推断质量。

工程上 RMSNorm 的主要收益是少算一次 mean reduction，并减少相关的同步与内存流量。实现时平方和通常使用 fp32 accumulation，避免 bf16 / fp16 的数值误差。

</details>

</details>

## 现代 LLM 还在哪儿加 norm：QK-norm 与前后两次 norm {#qk-norm}

Pre-LN 管住了送进子层的 $x$，但 attention 里还有一个会自己涨起来的量：logit $q\cdot k/\sqrt{d_k}$。$x$ 被归一化了，$q=xW_Q$、$k=xW_K$ 的长度却还跟着投影权重走。

看一个小例子。这里只比较分数差，先省略 $\sqrt{d_k}$ 这个固定常数。$q=[2,1]$，两个 key 是 $[1,0]$ 和 $[0,1]$，分数是 2 和 1，softmax 约为 `[0.731, 0.269]`。训练中如果 $W_Q$、$W_K$ 各自整体变大 3 倍，$q=[6,3]$，key 变成 $[3,0]$ 和 $[0,3]$，分数是 18 和 9。分数差从 1 变成 9，softmax 约为 `[0.99988, 0.00012]`，几乎是 one-hot，经 softmax 传回分数的梯度也变得很小（推导见[多头注意力](multi-head-attention.md#sqrtd_k)）。[ViT-22B](https://arxiv.org/abs/2302.05442) 报告过这种情况：attention logit 变得极大，注意力熵接近 0，训练几千步后 loss 发散。

**QK-norm** 在点积之前，对每个 head 的 $q$ 和 $k$ 各做一次 RMSNorm（各带一组可学习的 gain）：

$$
\hat q=\operatorname{RMSNorm}(q),\qquad
\hat k=\operatorname{RMSNorm}(k),\qquad
s=\frac{\hat q\cdot\hat k}{\sqrt{d_k}}.
$$

前面讲过，忽略 epsilon 时 RMSNorm 对整体缩放不变：$[6,3]$ 和 $[2,1]$ 归一化后相同，所以权重整体放大 3 倍不再改变分数。gain 为 1 时 $\|\hat q\|=\|\hat k\|=\sqrt{d_k}$，于是 $|s|\le\sqrt{d_k}$；$d_k=128$ 时上界约 11.3。这阻止了投影范数整体放大带来的 logit 膨胀，但不代表分数只能靠 gain 变大：即使 gain 固定，q、k 的方向更接近也会抬高分数；学习 gain 还可以改变上界。这两种变化都没有“一定很慢”的保证。RoPE 是旋转，不改变向量长度，所以这个上界和 norm 放在 RoPE 前后无关；[Gemma 3 的 Transformers 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py)放在 RoPE 之前。

[Qwen3 技术报告](https://arxiv.org/abs/2505.09388)写明去掉 Qwen2 的 QKV bias、引入 QK-Norm，是为了训练稳定；[Gemma 3 技术报告](https://arxiv.org/abs/2503.19786) §2 用 QK-norm 替换了 Gemma 2 的 soft-capping。soft-capping 用 $c\tanh(s/c)$ 把分数压在 $(-c,c)$ 里，是事后截住；QK-norm 从源头限制 $q$、$k$ 的长度。

**前后两次 norm。** Gemma 3 在每个子层的前后各放一个 RMSNorm：

$$
h=x+\operatorname{Norm}_{\text{post}}\big(\operatorname{Attn}(\operatorname{Norm}_{\text{pre}}(x))\big),
$$

FFN 也一样。这里的 post-norm 只作用在**分支输出**上，加回残差之后不再归一化，所以主干仍保留上面 Pre-LN 的 identity path。它和 2017 年那种把“残差加和”整体归一化的 Post-LN 名字相同、位置不同，可以理解为限制每个子层写回残差流的更新尺度。

## 怎样选择归一化方法 {#_4}

| 场景 | 用什么 | 为什么 |
| --- | --- | --- |
| CNN 图像分类，有足够统计样本 | **BatchNorm** 是常见基线 | 按 channel 统计；有效样本还包括空间位置，不只看 batch size |
| 自回归 Transformer | 常用 **LayerNorm / RMSNorm** | 按 token 的特征统计，不把未来位置或同行样本混进来 |
| RNN / LSTM | 可比较 **LayerNorm** | 每个时间步独立归一化；仍要检查放在门控前还是后 |
| 小 batch 的检测、分割 | 可比较 **GroupNorm / LayerNorm** | 不依赖批统计，但分组和归一化轴仍需与架构匹配 |
| 强化学习、在线学习 | **LayerNorm** 可作为基线 | 不维护批均值；分布变化并不意味着 BatchNorm 在所有 RL 方法中都不可用 |
| 不允许跨样本耦合的目标 | 使用样本内的统计量 | 先看目标约束，再选 LN、GN 或其他方案，不能只按任务名字决定 |

一句判据：**只要「同一个输入在不同批次里应该得到同一个输出」是硬要求，就不要让 normalization 使用 batch statistics。**

现代大模型常进一步选择 RMSNorm。少做均值中心化可以简化计算，但实际速度和同步次数还取决于融合 kernel 与张量切分，不能只数公式里的求和号。

<details markdown="1">
<summary><b>补充</b>：为什么 internal covariate shift 不是完整解释？</summary>

原论文的说法是减少 internal covariate shift。后来 [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604) 用实验反驳了这个解释——他们在 BN 之后人为注入分布噪声，训练依然又快又稳。

更稳妥的表述是：归一化降低了模型对参数尺度的敏感度，并经常让 loss landscape 与梯度更平滑，从而改善优化；这不是对任意网络全局条件数的无条件保证。

还有一个值得分清的性质：忽略 epsilon、统计量非零且 $\alpha>0$，并从当前输入重算统计量时，$\text{Norm}(\alpha Wx)=\text{Norm}(Wx)$。这是对输入整体正向缩放的不变性，不是说任意改动权重都不影响网络。使用固定 running statistics 的 BatchNorm eval 不满足这个前提。保留 epsilon 时通常只有近似关系；Norm 自己的 $\gamma$ 是否做 weight decay，又是单独的优化选择，不能从这个等式直接推出。

</details>

## 动手验证 {#_5}

[`../code/norm_compare.py`](../code/norm_compare.py) 用同一批激活比较统计轴，再检查三个不同情况：`[1,C]` 的 BatchNorm 训练前向会报错；eval 使用 running statistics 可以运行；`[1,C,L]` 在 `L>1` 时也能训练，但统计会跨时间。不要把这三件事都概括成“小 batch 不能用”。

图由 [`../code/make_norm_figures.py`](../code/make_norm_figures.py) 生成。

## 自检 {#_6}

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

## 继续阅读 {#_7}

归一化让每层输入的尺度可控；深度能训练，还要靠[残差连接](residual-connections.md)。两篇都读过后，接着看 [FFN 与 SwiGLU](ffn-and-gates.md)，再把零件拼成[原版 Transformer](vanilla-transformer.md)。

## 30 秒建立 mental model {#30-mental-model}

| 概念 | 一句话记忆 | 面试关键词 |
| --- | --- | --- |
| 为什么归一化 | 让每个 sublayer 看到尺度可预测的输入，使优化对参数尺度不那么敏感 | stable activations · conditioning · larger learning rate |
| 为什么不用 BatchNorm | 一个 token 的表示不应依赖 batch 里的其他样本或未来位置 | variable length · padding · train/eval mismatch · causality |
| Pre-LN vs Post-LN | Pre-LN 把 Norm 移出残差主干，留下 identity gradient path | $I+J_fJ_{\mathrm{LN}}$ · final LN |
| RMSNorm | 保留 re-scaling，去掉 re-centering | RMS only · no mean subtraction · cheaper reduction |

> **Normalization 要控制的是送进每个子层的数值尺度，让深网络更容易优化；它并不要求所有表示永久保持“均值 0、方差 1”。**
