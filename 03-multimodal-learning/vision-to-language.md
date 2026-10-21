# 图片怎样进入语言模型

**中文** · [English](vision-to-language.en.md)

> 阅读时间：约 8 分钟 · 最近审阅：2026-10

CLIP 能说“这张图更像这句描述”，但不会接着写答案。要让语言模型读图，我们需要把视觉信息变成它能利用的输入，再用合适的目标教它回答。

这里选一条常见路线来讲：**ViT 提取视觉特征，投影层（projector）连接两侧，再由 decoder-only LLM 生成回答**。不是所有视觉语言模型都这样搭。先读 [CLIP](clip.md) 和 [Decoder-only](../00-foundations/core/decoder-only.md)，会更容易跟上。

## 图片不是一个词，也不是只能变成一个向量

[ViT](https://arxiv.org/abs/2010.11929) 把图片切成 patch，展平并线性投影，加上位置信息，再交给 Transformer。

形状账在 [ViT](vit.md) 里算过：$224\times224$ 的 RGB 图按 $16\times16$ 切成 196 块，每块 768 个原始数值。这里要记住的是，投影后这 196 个向量可以保留成序列，不必立刻压成一个全局向量。原始 ViT 分类还会加入 class token，算序列长度时要另加 1。

```mermaid
flowchart TD
    A["图片"] --> B["切 patch + 位置编码"]
    B --> C["Vision encoder：P 个视觉特征"]
    C --> D["Projector：视觉维度 → LLM 维度"]
    D --> E["视觉向量 + 问题的 token embeddings"]
    E --> F["Decoder-only LLM"]
    F --> G["逐 token 生成回答"]
```

视觉向量通常是连续特征，不必先变成词表里的离散 ID。这里的 projector 可以改变每个向量的维度；**普通逐 token 线性层或 MLP 不会自动减少 token 数量**。压缩长度需要另加 pooling、resampler 或其他机制。

## 对齐维度，还不等于对齐含义

把 $d_v$ 维投影成 $d_{LM}$ 维，只解决了“矩阵能接上”。如果 LLM 从没学过这些向量的意义，单靠维度正确不会读图。

原始 [LLaVA 的训练设计](https://arxiv.org/abs/2304.08485)提供了一个清楚的例子：

| 阶段 | 哪些参数更新 | 要解决什么 |
| --- | --- | --- |
| 图文对齐 | projector；视觉 encoder 和 LLM 冻结 | 让视觉特征成为语言模型能使用的条件 |
| 视觉指令微调 | projector 和 LLM；视觉 encoder 冻结 | 学习结合图片和问题给出回答 |

这是这篇论文采用的训练方案，不代表 VLM 都要冻结视觉分支。读其他方案时，仍要看用了什么数据、训练了哪些模块，以及算力是否够用。

## 训练时，loss 算在哪里

在这个回答生成的教学设置中，输入含图片、问题和真实答案前缀；目标是预测答案中的下一个 token。视觉位置和问题不直接计入 answer-only loss，但它们是有效上下文，仍会影响答案，梯度也可能经过它们流向可训练模块。

拿一条自拟样本数一遍：一张图变成 196 个视觉位置，问题“红色方块在圆的左边吗？”假设切成 8 个 token，答案“是”加结束标记共 2 个 token，为了凑齐 batch 再补 2 个 padding。

| 位置 | 个数 | 作为预测目标计入 loss？ | 答案位置能通过 attention 读到它吗？ |
| --- | ---: | --- | --- |
| 视觉位置 | 196 | 否 | 能 |
| 问题 token | 8 | 否 | 能 |
| 答案与结束标记 | 2 | 是 | 能读到自己及之前的位置 |
| Padding | 2 | 否 | 否 |

208 个位置里只有 2 个算分，所以下面公式的分母 $M=2$，不是 208，也不是去掉 padding 后的 206。逐位置的 label 怎样写，见 [VLM 微调的 mask 算例](vlm-finetuning.md)。

Padding 要另看：它只是为了把样本补齐，应同时排除它的 loss，并阻止有效答案位置通过 attention 读到它。**Loss mask 决定哪些预测算分，attention mask 决定可以看哪些输入；二者不能互相替代。**正确屏蔽后，padding 输入位置不应成为答案的信息或梯度路径。这里说的是这个输入位置，不是说共享的 embedding / output 参数永远不会从其他位置收到梯度。[PyTorch 的 padding mask 说明](https://docs.pytorch.org/docs/2.8/generated/torch.nn.MultiheadAttention.html)

$$L=-\frac{1}{M}\sum_{t\in\mathcal A}\log p_\theta(a_t\mid I,q,a_{<t}),\qquad M=|\mathcal A|.$$

$\mathcal A$ 是计分的答案位置。它与“图片和哪句话配对”的对比 loss 不是同一个问题。需要检查 shift 和 mask 时，回到[一次语言模型训练](../00-foundations/deep-dives/training-step.md)。

## 设计上最值得问的 3 个取舍

- **细节与长度**：在固定 patch 大小、单尺度的例子里，边长翻倍，patch 数变成 4 倍；只看这些 token 的 dense attention 分数矩阵，元素数变成 16 倍（算例见 [ViT](vit.md)）。完整运行成本还取决于文本长度、attention 实现和压缩方式，不能直接说“整体慢 16 倍”。
- **冻结与适配**：少训练一些模块，通常能减少它们的梯度和优化器状态，但不等于没有前向计算或 activation 成本。是否损失领域适配能力，要用实验回答。
- **描述与证据**：流畅 caption 可能漏掉题目需要的小字、位置或关系。保留 OCR 文本或更高分辨率不是万能答案，要看目标任务究竟缺哪种信息。

## 别只问它答得像不像

用自己制作、答案可核对的小图做成对测试。例如同样的问题“红色方块在圆的左边吗”，只交换两个形状的位置，正确答案就应该改变。保持 prompt 和解码设置不变，再比较。去图、只改一处证据、遮住关键区域、图文冲突这几种对照各能说明什么、不能说明什么，统一列在[多模态首页的对照表](README.md#modality-checks)；空间问题再加两种：给一张不匹配的图，看模型会不会盲从问题里的暗示；问图里根本看不见的属性，看它能不能承认信息不足，而不是给出更长的解释。

这是一组建议的教学实验，不是任何模型的实测结论。按小字、计数、位置关系等类型分别记录，比一个总正确率更容易找到问题。

接下来可以把这些检查接到[对照实验与切片](../07-evaluation/ablation-and-slices.md)，再考虑是否真的需要继续训练模型。同样是“视觉编码器 + 连接模块 + LLM”，机器人的 VLA 模型把输出从文字换成了动作，见[机器人学习：VLA](../13-embodied/robot-learning/vla.md)。
