# ViT：图片怎么变成 Transformer 的输入？

**中文** · [English](vit.en.md)

> 阅读时间：约 10 分钟 · 最近审阅：2026-10

看一张路口照片，你会注意到车、人和红绿灯。但模型刚收到的只是像素数值。ViT 的第一步不是先找出这些物体，而是把图片分成小块，让每一块变成一个向量。之后，注意力才让不同位置交换信息。

这里先把“切块”和“理解”分开。读完这一篇，再看 [CLIP](clip.md)，就能区分图像编码器的结构和训练它的目标。

## 先算清楚，到底有多少个 token

假设图片是 $224\times224$ 的 RGB 图像，每块是 $16\times16$，不重叠、不补边：

$$N=\frac{224}{16}\frac{224}{16}=196,\qquad P^2C=16^2\times3=768.$$

196 是块数，768 是每块原始数值的个数。它们不是同一个维度。下面沿用 PyTorch 常见的通道在前格式；其他框架也可能把通道放在最后。

| 步骤 | 一批图片的形状 | 这一步做了什么 |
| --- | --- | --- |
| 输入 | $B\times3\times224\times224$ | 保留高、宽和颜色通道 |
| 切块并展平 | $B\times196\times768$ | 每个位置对应一块像素 |
| 共享线性投影 | $B\times196\times D$ | 所有块使用同一组权重 |
| 加 CLS 和位置向量 | $B\times197\times D$ | 多一个汇总位置，并标记各块在哪里 |
| Transformer 编码器 | $B\times197\times D$ | 更新每个位置的表示，不在这里缩短序列 |

**投影不一定是压缩。** $D$ 是模型的隐藏维度：选 768 时维度不变，选 1024 时反而增加。它学的是哪些像素组合值得保留，不是把图片查表换成词表 ID。

## 用一张 4 × 4 的“小图片”走一遍

下面是自拟的单通道数值图，不是训练结果。每个 $2\times2$ 小块按行展平，再从左上到右下排列。

```text
 1  2 |  3  4       左上 → [1, 2, 5, 6]
 5  6 |  7  8       右上 → [3, 4, 7, 8]
------+------       左下 → [9, 10, 13, 14]
 9 10 | 11 12       右下 → [11, 12, 15, 16]
13 14 | 15 16
```

人为选两个投影：第一个算块内均值，第二个算“右列均值减左列均值”。左上块就变成 `[3.5, 1.0]`。真实模型会学习权重，不需要我们提前规定“检测左右差异”。

```python
def patchify_gray(image, patch_size):
    if not isinstance(patch_size, int) or patch_size <= 0 or not image or not image[0]:
        raise ValueError("Expected a nonempty image and a positive integer patch size")
    height, width = len(image), len(image[0])
    if any(len(row) != width for row in image):
        raise ValueError("Rows must have equal widths")
    if height % patch_size or width % patch_size:
        raise ValueError("This example does not pad or crop partial patches")
    return [
        [image[top + row][left + column]
         for row in range(patch_size) for column in range(patch_size)]
        for top in range(0, height, patch_size)
        for left in range(0, width, patch_size)
    ]

image = [[1 + 4 * row + column for column in range(4)] for row in range(4)]
patches = patchify_gray(image, 2)
projection_rows = [[0.25, 0.25, 0.25, 0.25], [-0.5, 0.5, -0.5, 0.5]]
embeddings = [[sum(value * weight for value, weight in zip(patch, weights))
               for weights in projection_rows] for patch in patches]
assert embeddings == [[3.5, 1.0], [5.5, 1.0], [11.5, 1.0], [13.5, 1.0]]
```

这里也能看到一个限制：只有这两个特征时，一些不同的小块会得到相同结果。例如 `[1, 2, 5, 6]` 和 `[2, 3, 4, 5]` 都输出 `[3.5, 1.0]`。后面的注意力无法从这两个数中恢复被丢掉的差异。

实际实现通常用 kernel size 和 stride 都等于 patch size 的卷积完成切块投影；没有重叠时，它与逐块使用同一个线性映射对应。读 [ViT 官方实现](https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py) 时，可以顺着 `embedding`、reshape、`cls`、`Transformer` 看形状变化。

## CLS 不是标签，位置向量也不是装饰

CLS 起初是一个可学习的向量，不携带“这张图是汽车”的答案。经过多层注意力，它可以汇总其他块的信息；分类头再根据它预测类别。也可以设计全局平均池化的读出方式，不能把 CLS 当成所有视觉模型必须有的部件。

位置则解决另一个问题：同样几块内容，上下左右换了位置，图片含义可能不同。原始 ViT 使用可学习的位置向量。它们加在内容向量上，而不是另塞一列坐标。注意力仍然会看所有块，不因此变成只看相邻块的卷积。[原论文 §3](https://arxiv.org/abs/2010.11929)

## 一个 block 里发生什么

省略 dropout，把整条序列记为 $X$，Pre-LN block 可以写成：

$$U=X+\operatorname{Attention}(\operatorname{LN}(X)),\qquad
Y=U+\operatorname{MLP}(\operatorname{LN}(U)).$$

注意力在位置之间交换信息，MLP 在每个位置上变换特征，残差保留原来的路径。于是左上块的输出不再只依赖左上像素，也能参考远处的路牌。要拆到 Q/K/V，可以接着读[多头注意力](../00-foundations/core/multi-head-attention.md)。

这还是 encoder：各块可以双向交流，不像生成文字时必须遮住未来 token。把 encoder 中的一层叫 block，并不意味着模型“不再使用 encoder”。

## 分辨率、块大小，各自在改什么

先固定隐藏维度和层数，只算 patch token，不计 CLS：

| 图片 / patch | token 数 | 单头完整注意力分数的元素数 |
| --- | --- | --- |
| $224\times224$ / $16\times16$ | 196 | 38,416 |
| $448\times448$ / $16\times16$ | 784 | 614,656 |
| $448\times448$ / $32\times32$ | 196 | 38,416 |

第二行保留更细的空间采样，但 token 变为 4 倍，完整分数矩阵变为 16 倍。第三行把 token 数降回来，却把更多像素放进同一块；**账面成本相同，不代表小字、边缘和物体关系的识别能力相同。**

这张表不是整网显存或耗时预测。MLP、投影、batch size、内核是否物化分数矩阵都会改变实际成本。更高分辨率还要求处理位置向量：原始 ViT 微调时对 patch 网格做二维插值，CLS 单独处理。插值让形状接得上，不保证新尺寸下效果不变。

## 微调骨干和只训练分类头，不是一回事

| 设置 | 更新什么 | 主要在测什么 |
| --- | --- | --- |
| Linear probe | 冻结骨干，只训练线性头 | 已有特征能否线性区分类别 |
| Fine-tuning | 更新骨干及任务头，具体冻结范围须说明 | 模型适应新任务后的能力 |
| 特征提取 | 全部冻结，不训练新头 | 给检索或其他模型提供表示 |

原始 ViT 的迁移学习不是一律冻结骨干：论文更换为零初始化的线性分类头后进行微调，另有冻结表示的少样本评测。看结果时先确认协议，不能把这两类数字直接放在一起比较。[原论文 §3.2、§4](https://arxiv.org/abs/2010.11929)

ViT 解释了“图片怎么编码”，但没有规定编码后用来干什么。[CLIP](clip.md) 用图文配对训练表示；[视觉到语言](vision-to-language.md)继续把视觉特征接到生成模型。结构相似，目标不同，学出来的能力也不能画等号。
