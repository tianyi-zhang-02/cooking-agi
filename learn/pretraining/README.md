# 预训练：从数据走到一次可靠的更新

**中文** · [English](README.en.md)

理解了 Transformer 的前向计算，还差一步才算理解训练：同一段文本怎样变成标签，哪些 token 计入 loss，梯度怎样汇总，更新以后又凭什么说模型变好了？这一章按这个顺序往下走。

## 先把单卡上的一小步弄清楚

如果你还没写过完整训练循环，可以先跑 [PyTorch 导读里的两个样本](../../00-foundations/pytorch/README.md)，看清前向、反向和更新的区别，再回到语言模型的 token 目标。基础不必一口气补完；遇到梯度尺度或泛化问题，再读[初始化](../../00-foundations/deep-dives/activation-and-initialization.md)、[优化器](../../00-foundations/deep-dives/optimizers.md)和[训练诊断](../../00-foundations/deep-dives/generalization.md)。

设想两段长度不同的文本：一段有 2 个有效目标，loss 总和是 4；另一段有 6 个有效目标，loss 总和是 6。

按有效 token 计算，平均 loss 是 $(4+6)/(2+6)=1.25$。如果先分别求平均，再平均这两个数，就会得到 $(2+1)/2=1.5$。数据没有变，优化的权重却变了。梯度累积、多卡训练和 padding 都会把这个小问题带回来。

先把目标、mask 和分母说清楚，再谈怎样更快地计算它。

| 阅读顺序 | 带着什么问题读 | 读完检查什么 |
| --- | --- | --- |
| 1. [语言模型目标](../../00-foundations/deep-dives/language-model-objective.md) | 在哪个位置预测哪个 token？ | 为什么训练可以并行，生成通常不行？ |
| 2. [预训练流程](../../00-foundations/deep-dives/pretraining-pipeline.md) | 文本如何清洗、去重、混合与 packing？ | EOS、attention mask 和 loss mask 是一回事吗？ |
| 3. [一次训练](../../00-foundations/deep-dives/training-step.md) | 一个 batch 怎样改变参数？ | 改了 batch 切分以后，目标是否相同？ |
| 4. [多 token 预测](../../00-foundations/deep-dives/multi-token-prediction.md) | 多加的预测头拿到了什么监督？ | 标签是否越过文档边界或提前泄漏答案？ |

## 然后才是精度、显存与多卡

这些优化不是同一层的事。BF16 改变数字怎样表示；重算改变哪些中间量要保存；FSDP 改变状态放在哪张卡；TP 则拆开一次矩阵计算。它们都可能“省显存”，但不是互相替代的开关。

| 遇到的现象 | 先检查 | 接着读 |
| --- | --- | --- |
| loss 出现 Inf / NaN | 数值范围、缩放与梯度裁剪顺序 | [精度与显存](../../00-foundations/deep-dives/precision-and-memory.md) |
| 权重能放下，反向却 OOM | 激活、优化器状态、临时张量、序列长度 | [精度与显存](../../00-foundations/deep-dives/precision-and-memory.md) |
| 增加卡数后结果对不上 | 数据分片、有效 token 数、同步与随机性 | [多卡训练](../../06-systems/distributed-training.md) |
| 利用率高，吞吐却不理想 | 有效 token 吞吐、重算、通信与等待 | [多卡训练](../../06-systems/distributed-training.md) |

## 本章目录

<!-- widget:study-atlas -->

## 怎样用起来

先用固定的小 batch 跑通前向、反向与更新，保存参数变化；然后只改一个条件，例如累积步数或精度。比较输出、梯度与一次更新是否在合理容差内一致，之后再扩大规模。

手算数字用于检查机制，不是硬件性能报告。实际显存和速度还要记录模型版本、序列长度、有效 batch、dtype、硬件与计时边界。

下一步可以读 [SFT 与后训练](../../05-post-training/README.md)，或者转到[生成与推理](../inference/README.md)。
