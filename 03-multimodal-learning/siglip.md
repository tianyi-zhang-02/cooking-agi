# SigLIP：图文配对，不一定要用 softmax

**中文** · [English](siglip.en.md)

> 核对：2026-10-11 · 先讲 SigLIP（2023）的目标，再讲 SigLIP 2（2025）的扩展。手算与代码均为本站教学例子，不是预训练实验结果。

在 [CLIP](clip.md) 里，给定一张图，要从一批文字里选出配对的那句。SigLIP 换了一个问法：**这一张图和这一句话，配不配？** 对每一对分别计算 loss，不要求先把一整行分数归一化成和为 1 的分布。

两者仍然可以是 image encoder 和 text encoder 各算一个向量，再比较相似度。换了目标，不等于变成了会聊天的多模态 LLM。[SigLIP 原论文](https://arxiv.org/abs/2303.15343)

## 先拿两张图片试一下 {#two-pairs}

两张图分别是自行车和咖啡，对应两句描述。设模型已经给出了这张 **logit** 表；它包括相似度的缩放与偏置，不是原始余弦：

| | “一辆自行车” | “一杯咖啡” |
| --- | ---: | ---: |
| 自行车图 | 2 | −2 |
| 咖啡图 | −2 | 2 |

Sigmoid 把 2 映射到约 0.881，把 −2 映射到约 0.119。四个位置各有一次判断：对角线上希望分数高，另外两个位置希望低。

CLIP 则会问“在这两个候选里哪个更合适？”第一行的 softmax 约为 `[0.982, 0.018]`。现在再放入一句同样合适、logit 也是 2 的描述，第一项降到约 0.495；SigLIP 对固定配对的 sigmoid 值仍是 0.881。**这里固定了模型与已有 logits**，不是说训练时增加负例不会改变参数。

这个区别也不是“sigmoid 更准确”：两种数值回答不同问题，都不能未经校准就当成真实世界的正确率。

## 双塔没有消失，变的是分数怎么进入目标 {#architecture}

<figure class="worked-update" lang="zh-CN">
<ol>
<li><small>01 / 两路编码</small><strong>图和文字分别变成向量</strong><span>编码、投影并归一化。候选向量仍可提前计算，查询时再比较。</span></li>
<li><small>02 / 成对打分</small><strong>缩放相似度，再加偏置</strong><span>z = scale × cosine + bias。负配对比正配对多，bias 的作用不能直接忽略。</span></li>
<li><small>03 / 配对监督</small><strong>每一对贡献一个 logistic loss</strong><span>正例希望 z 变大，负例希望 z 变小。各项共享 encoder 参数，梯度最终一起更新它们。</span></li>
</ol>
<figcaption>逐对计算 loss，不等于每对拥有独立参数，也不等于训练数据相互独立。</figcaption>
</figure>

有 $B$ 个配对，归一化向量为 $u_i,v_j$，令 $z_{ij}=\exp(a)u_i^\top v_j+b$；正例 $y_{ij}=+1$，负例为 $-1$。$a$ 和 $b$ 可学习，指数让 scale 保持为正。

每行只有一个正例、计算 batch 内所有图文组合时，目标是：

$$L=\frac{1}{B}\sum_{i=1}^{B}\sum_{j=1}^{B}\operatorname{softplus}(-y_{ij}z_{ij}).$$

注意是除以 $B$，不是 $B^2$。[官方训练实现](https://github.com/google-research/big_vision/blob/0127fb6b337ee2a27bf4e54dea79cff176527356/big_vision/trainers/proj/image_text/siglip.py#L289)先求和每行的配对 loss，再对行平均。如果用默认对全部元素求平均的 BCE，梯度尺度会少一个 $B$；固定 batch 时可以重新处理这个尺度，但不能一边改变 reduction，一边声称其他训练条件完全相同。

## 把 loss 算出来 {#loss-example}

上表四项都是 $\log(1+e^{-2})\approx0.1269$，所以 $L\approx4\times0.1269/2=0.2539$。下面只用 Python 标准库，避免直接算大数的 `exp(1000)`：

```python
import math


def pair_loss(logit, positive):
    if not math.isfinite(logit) or not isinstance(positive, bool):
        raise ValueError("expected a finite logit and a boolean label")
    margin = -logit if positive else logit
    return max(margin, 0.0) + math.log1p(math.exp(-abs(margin)))


def siglip_loss(logits):
    batch_size = len(logits)
    if batch_size == 0 or any(len(row) != batch_size for row in logits):
        raise ValueError("expected a nonempty square score matrix")
    return math.fsum(
        pair_loss(score, row_index == column_index)
        for row_index, row in enumerate(logits)
        for column_index, score in enumerate(row)
    ) / batch_size


scores = [[2.0, -2.0], [-2.0, 2.0]]
assert math.isclose(siglip_loss(scores), 2 * math.log1p(math.exp(-2)))
```

这段默认对角线是正例，用来核对公式，不是生产训练实现。两张几乎一样的图片、重复描述或一图多段有效文字，需要另行定义多正例与采样规则；sigmoid 本身不会识别假负例。

<details markdown="1">
<summary>再看一步：梯度和 bias 在做什么？</summary>

对二值标签 $q\in\{0,1\}$，一个未归一化配对 loss 的导数是 $\sigma(z)-q$。正例得分太低，梯度下降会提高它；负例得分太高，会降低它。整个目标还要乘上 $1/B$。

假设所有相似度暂时为零，只学习共同 bias。每行 1 个正例、$B-1$ 个负例，令梯度为零得到 $B\sigma(b)-1=0$，即 $b=-\log(B-1)$（$B>1$）。这是解释类别比例的简化推导，不是在指定官方模型的初始化值，也不是实际训练的最终 bias。负例采样比例变了，计算结果也会变。

</details>

## 更容易分块，不等于负例免费 {#distributed}

不需要全局 softmax 分母，就可以把配对项分块处理、累加 loss。在这个自拟算例中，把 128 个配对均分到四个数据并行进程（rank），每个进程 32 个：只在本地比较，每张图看到 31 个负例；若目标是整个 128 配对的池，就需要看到另外 96 条文字的信息。怎么交换 embedding、在哪里计算、怎样回传梯度，仍是工程问题。

同理，4 次本地 batch 的梯度累积，不会自动产生跨 batch 的配对。删掉负例或调整正负比例，改变的是训练目标，不只是省显存的实现。更大的对比池也可能带来更多假负例。

## SigLIP 2 多了什么 {#siglip2}

[SigLIP 2 论文](https://arxiv.org/html/2502.14786v1)在 sigmoid loss 的基础上扩展了多语言数据与训练配方，结合图像描述生成、自蒸馏、掩码预测等任务，并提供不同的图像尺寸处理路线。**这些阶段并非每个变体都完全相同**；论文里的 NaFlex 路径没有采用标准变体的自蒸馏和掩码预测阶段。

<!-- paper-figure:siglip2-recipe -->

| 想解决的问题 | 理解这类训练时可以这样看 | 实际选模型时查什么 |
| --- | --- | --- |
| 整张图对得上，但局部语义弱 | 全局匹配之外，让局部表示也得到监督 | 下游取 pooled vector，还是 patch features |
| 中英文效果差很多 | 语言覆盖是训练数据与配方问题 | 对应 checkpoint 的语言与任务评估 |
| 长截图缩成方形，细节损失大 | 保留宽高比、调整视觉 token 预算 | 是否真的是 NaFlex，以及配套 processor |

接入时用 checkpoint 对应的 processor，不要沿用 CLIP 的 resize / normalize / tokenizer 配置。固定尺寸与 NaFlex 的输入处理不是随便替换的参数。[官方模型发布说明](https://google-research.github.io/big_vision/big_vision/configs/proj/image_text/README_siglip2.html)和 [Transformers 的 SigLIP2 接口](https://huggingface.co/docs/transformers/en/model_doc/siglip2)可用来核对具体模型。

## 检索和 VLM，接的是不同输出 {#downstream}

做图文检索时，通常关心可比较的全局向量：候选离线编码、查询在线编码，再接 ANN 或精确打分。升级 encoder 后，旧库向量和新查询向量一般不能混用；要规划重建索引与切换。

接入生成式 VLM 时，常用视觉分支的 token / patch features，再经过 projector 或 resampler 交给 LLM。还需要连接模块、训练数据和生成目标，不能只插入一个 SigLIP encoder 就说模型已经会视觉问答。详细流程见[图片怎样进入 LLM](vision-to-language.md)。机器人用的 VLA 模型也这样接：OpenVLA 把 DINOv2 和 SigLIP 两路特征一起送进 Llama 2，再输出动作，见[机器人学习：VLA](../13-embodied/robot-learning/vla.md)。

选择 CLIP、SigLIP 或 SigLIP 2，可以先比较你真正需要的切片：中英文、细小文字、相似物体、重复描述、低清晰度，以及离线编码与在线查询成本。名称更新，不代表你的任务一定提升；先固定数据切分和检索协议，再看差别。
