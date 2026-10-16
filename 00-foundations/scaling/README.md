# Scaling：开训之前，先把模型、数据和学习率（LR）算一遍

**中文** · [English](README.en.md)

> 阅读时间：约 5 分钟 · 草稿，待审核

一次预训练开始之前，有几件事必须先定：模型多大，训多少 token，学习率（LR）和 batch size 取多少，学习率怎么随时间变。这些一旦定下，几天到几个月的算力就花出去了，中途很难改。scaling law 的用处，是先用一组便宜的小 run 把这些数估出来。**这组笔记的结论是：训练算力固定时，参数和数据大致同比例放大（Chinchilla 的约 20 tokens/parameter）；要上线的模型因为推理成本，通常训得远比这个久；学习率和 batch size 也有随规模变化的规律，用最大更新参数化（µP）一类方法可以把小模型上调好的学习率迁移到大模型。这些都是在特定数据和配方上拟合的经验曲线，离拟合范围越远，越要当成区间看。**

> **待审核草稿。** 这一篇还没有经过站主审阅，事实、取舍和措辞都可能修改；引用前请回到文中链接的一手资料核对。

## 开训前要定的几个数

| 要定的数 | 主要由什么决定 | 在哪一篇 | 常见的错 |
| --- | --- | --- | --- |
| 训练算力 $C$ | GPU 数、峰值算力、模型浮点利用率（MFU）、天数；$C\approx6ND$ | 第 1 篇 | 用 sparse 峰值算预算 |
| 参数量 $N$ 和训练 token 数 $D$ | IsoFLOP 拟合、预期推理量、不重复数据量 | 第 1、2 篇 | 照搬“20”，不看数据和推理量 |
| batch size | 临界 batch size，随 loss 下降和数据量增长 | 第 3 篇 | 为了多用卡，把 batch 加到临界值的好几倍 |
| 学习率 | 按规模拟合的公式，或 µP 迁移 | 第 3 篇 | 标准参数化（SP）下把小模型的最优学习率直接搬到大模型 |
| 学习率 schedule | cosine 对准训练长度，或 warmup-stable-decay（WSD） | 第 3 篇 | cosine 周期按计划长度设，却提前停下 |

## 一个贯穿的教学例子

假设你有 64 张 H100 SXM5，跑大约 4.6 天，按 BF16 dense 峰值 989.4 TFLOPS、40% MFU 算，约为 $C=10^{22}$ FLOPs（MFU 的口径见[性能：roofline 与 MFU](../../12-infra/performance.md#mfu)）。三篇依次回答：

1. 只看训练算力，最优大约是 90 亿参数、1800 亿 token；这个答案怎样从几个小 run 拟合出来，又能外推多远。
2. 如果模型上线后要处理 $10^{12}$ 个推理 token，最优变成约 46 亿参数、4500 亿 token；如果只有 1000 亿个不重复 token，重复训练还值多少。
3. 这个 run 的 batch size 和学习率取多少，schedule 选哪种，以及怎样把宽 512 的代理模型上扫出的学习率搬到宽 4096 的模型上。

文中所有手算数字都是教学用的，不是任何模型的实测。

## 时间线

| 日期 | 工作 | 一句话 |
| --- | --- | --- |
| 2018-12 | [McCandlish 等](https://arxiv.org/abs/1812.06162) | 梯度噪声尺度预测临界 batch size |
| 2020-01 | [Kaplan 等](https://arxiv.org/abs/2001.08361) | loss 对参数、数据、算力呈幂律；$N_{\rm opt}\propto C^{0.73}$ |
| 2022-03 | [Chinchilla](https://arxiv.org/abs/2203.15556) | 400 多个模型；参数和数据同比例放大，约 20 tokens/parameter |
| 2022-03 | [µP / µTransfer](https://arxiv.org/abs/2203.03466) | 用 4000 万参数代理模型的超参数训练 6.7B 的 GPT-3 |
| 2023-02 | [LLaMA](https://arxiv.org/abs/2302.13971) | 按推理成本选模型，7B 训 1T token |
| 2023-03 | [GPT-4 技术报告](https://arxiv.org/abs/2303.08774) | 自报用不超过万分之一算力的模型预测了最终 loss |
| 2023-05 | [数据受限的 scaling](https://arxiv.org/abs/2305.16264) | 重复 4 遍左右几乎等于新数据 |
| 2023-12 | [Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448) | 把推理 FLOPs 算进总账 |
| 2024-01 | [DeepSeek LLM](https://arxiv.org/abs/2401.02954) | 按算力拟合最优学习率和 batch size；数据质量改变分配 |
| 2024-04 | [MiniCPM](https://arxiv.org/abs/2404.06395)、[Chinchilla 复现](https://arxiv.org/abs/2404.10102) | WSD schedule；第三种方法重拟后也是约 20 |
| 2024-06 | [Porian 等](https://arxiv.org/abs/2406.19146)、[Pearce 和 Song](https://arxiv.org/abs/2406.12907) | 解释 Kaplan 与 Chinchilla 的分歧 |
| 2024-07 | [Llama 3](https://arxiv.org/abs/2407.21783) | 自己拟合最优训练 token 数 $D^*\propto C^{0.53}$，405B 训 15.6T；小模型远超算力最优 |
| 2024-10 | [Zhang 等](https://arxiv.org/abs/2410.21676) | 临界 batch size 主要随数据量增长 |
| 2025-05 | [Power Lines](https://arxiv.org/abs/2505.13738) | batch size 和 weight decay 的 scaling law |
| 2026-04 | [DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | 两个 MoE 模型都在 32T 以上 token 上预训练，大部分参数用 Muon 更新 |

## 哪些比较确定，哪些还在变

每格都对应一条带日期的公开记录；“接下来看什么”写的是什么样的结果会改变判断，不是预测。

| 比较确定 | 还在变 | 接下来看什么 |
| --- | --- | --- |
| 训练算力固定时，参数和数据大致同比例放大：Chinchilla 三种方法给出 0.46–0.50，复现和重拟后约 20 tokens/parameter | 比例随数据质量、tokenizer 和配方移动：DeepSeek LLM 在三份数据上得到 0.450–0.578 | 公开 IsoFLOP 原始数据和置信区间，而不只是一条拟合线 |
| 要上线的模型训得远超算力最优：Llama 3 8B 每参数约 1900 token，Qwen3 的小模型更多 | 极端比例下 loss 怎样变平：Sardana 等发现用常规比例拟合会高估收益 | 每参数上千 token 区间的系统测量 |
| 标准参数化下最优学习率随宽度移动，µP 能让它跨宽度迁移 | 跨深度、batch 和训练长度的迁移；Everett 等发现按逐层规则设学习率的标准参数化也能迁移 | 大规模、公开对比 µP 与其他参数化的训练报告 |
| cosine 要对准训练长度；WSD 的短 cooldown 能追平它 | 临界 batch size 主要跟着数据量还是跟着 loss 变：Zhang 等、Bergsma 等和 Kaplan、DeepSeek LLM 的拟合方式不同 | 同一配方下，比较按 $C$ 和按 $D$ 拟合 batch 的预测误差 |

## 这一组怎么读

先读这些会更顺：[预训练流程](../deep-dives/pretraining-pipeline.md)（$C\approx6ND$ 和数据配比）、[一次训练](../deep-dives/training-step.md)、[从 SGD 到 AdamW](../deep-dives/optimizers.md)。相邻的几篇：[Muon](../deep-dives/muon.md) 讲按矩阵形状缩放更新；[性能：roofline 与 MFU](../../12-infra/performance.md#mfu) 讲怎样把 GPU 小时换成有效 FLOPs；[测试时计算](../../11-frontier/test-time-compute/README.md)讲推理时多花算力和放大预训练怎样比较；总参数和激活参数见 [MoE](../moe/README.md)。

1. [给定算力，模型多大、数据多少](compute-optimal.md)：$C\approx6ND$，幂律，用几个小 run 手拟一条线，Kaplan 与 Chinchilla 为什么不同，20 tokens/parameter 的条件，小模型拟合能信多少
2. [为什么现在的模型都训练得比 Chinchilla 久](beyond-chinchilla.md)：把推理算进总账，过度训练，数据重复几遍还值，开源模型的实际配比，这些修正的边界
3. [学习率、batch size 和 schedule 怎么跟着放大](hyperparameters.md)：临界 batch size，学习率换算与拟合，cosine 与 WSD，µP 和一次手算的学习率迁移
4. [复习题](review.md)：面试题和自检
