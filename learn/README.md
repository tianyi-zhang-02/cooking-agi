# 基础与原理

**中文** · [English](README.en.md)

这里主要讲模型怎么工作、为什么这样训练，以及怎样判断结果。先用例子理解，再按需要往推导和实现里走。不用读完所有基础，也不用追完每个新模型；只关心某个方向，直接选读相关章节。

- **刚开始学**：先看小例子和计算过程，推导可以第二遍再读。
- **已经有基础**：从你想解决的问题进入，缺哪个概念再回去补。
- **准备动手做**：读实现约定、跑小实验，再看工程实践中的模块怎么接起来。

近期补充：[Muon](../00-foundations/deep-dives/muon.md)解释矩阵更新与 AdamW 的区别；[DFlash 与 MTP](../00-foundations/deep-dives/dflash.md)讲草拟、验证和加速条件，也补了 DFlash 2 的变化。想入门可以先跳过；已经在做训练或推理，可以直接读对应问题。

## 从基础开始

如果还不熟悉向量、loss 和梯度，先读[从线性模型到神经网络](../00-foundations/from-linear-to-neural.md)。下面 1–3 步讲清模型怎样计算和学习，第 5 步教你检查结果；后训练和推理优化可以按需要再读，不是入门的必修清单。代码也不用留到最后，可以一起跑 [PyTorch 小实验](../00-foundations/pytorch/README.md)。

<nav class="study-route" aria-label="建议阅读主线">
<a href="../00-foundations/core/tokenization.md"><small>01 / 输入</small><strong>文本怎样变成向量</strong><span>Token、词表与序列</span></a>
<a href="../00-foundations/core/README.md"><small>02 / 模型</small><strong>Transformer 在算什么</strong><span>Attention、残差、Norm 与 FFN</span></a>
<a href="pretraining/README.md"><small>03 / 训练</small><strong>数据怎样改变参数</strong><span>目标、梯度与一次更新</span></a>
<a href="../05-post-training/README.md"><small>04 / 调整行为</small><strong>示范、偏好与奖励</strong><span>SFT、蒸馏与强化学习</span></a>
<a href="../07-evaluation/README.md"><small>05 / 检验</small><strong>怎样知道真的变好了</strong><span>对照、指标与具体错误</span></a>
<a href="inference/README.md"><small>06 / 生成</small><strong>怎样把回答算出来</strong><span>采样、缓存与推理成本</span></a>
</nav>

## 带着问题来，可以这样读

不用为了一个疑问读完整个板块。先读每条路线的第一篇，能回答最后一栏的问题就可以先停；后面的文章是需要时继续走的路，不是必须打卡的顺序。

| 你现在想弄明白什么 | 阅读顺序 | 最后动手看什么 |
| --- | --- | --- |
| 公式懂了，训练代码还是写不对 | [张量与存储](../00-foundations/pytorch/tensors-and-storage.md) → [自动求导](../00-foundations/pytorch/autograd.md) → [训练循环](../00-foundations/pytorch/training-loop.md) | 参数有没有更新，验证 loss 的分母对不对 |
| 想微调，先准备什么 | [预训练数据流程](../00-foundations/deep-dives/pretraining-pipeline.md) → [SFT](../05-post-training/sft-and-its-ceiling.md) → [LoRA / QLoRA](../05-post-training/lora-and-qlora.md) | 标签、loss mask、数据切分与显存 |
| 新架构名字很多，分不清差别 | [组件阅读地图](../00-foundations/deep-dives/README.md) → [模型报告精读](../00-foundations/model-families/README.md) | 改的是序列、缓存、专家，还是网络深度 |
| 图片接上了，模型真的看了吗 | [图文如何配对](../03-multimodal-learning/clip.md) → [视觉特征进入 LLM](../03-multimodal-learning/vision-to-language.md) → [VLM 微调与评估](../03-multimodal-learning/vlm-finetuning.md) | 换掉图片里的事实，答案会不会跟着变 |
| RL 公式会推，实验却不稳定 | [价值估计与 GAE](../05-post-training/deep-rl/actor-critic-gae.md) → [PPO clipping](../05-post-training/rlhf/ppo-clipping.md) → [实验排错](../05-post-training/deep-rl/experiments.md) | reward、advantage、更新幅度各出了什么变化 |
| 检索分数涨了，回答却没有更好 | [双塔检索](../04-search/dual-encoder.md) → [混合检索与重排](../04-search/hybrid-and-reranking.md) → [指标稳健性](../07-evaluation/metric-robustness.md) | 固定候选池后，哪些样本真的改善了 |
| 换 embedding 模型，需要改哪些地方？ | [Qwen Embedding 与 BGE-M3](../04-search/embedding-models.md) → [向量索引](../04-search/vector-indexes.md) | 输入与 pooling 怎么对齐，什么时候要重建索引 |

<span id="area-coding"></span>
<span id="subject-coding"></span>
<span id="subject-algorithms"></span>
<span id="subject-ml-exercises"></span>
<span id="subject-system-design"></span>

**想复习和写题？** 去[面试准备](../interview/README.md)，那里按 ML / LLM 基础问答、ML Coding、Python 与 LeetCode 分开整理。**想设计和搭系统？** 去[工程实践](../practice/README.md)，先做系统设计，再看公开项目和教学实现。这里保留原理讲解中的小实验，不把所有练习混进目录。

<details markdown="1">
<summary>只对一个方向感兴趣，需要补哪些基础？</summary>

| 方向 | 先会这些就能开始 | 第一轮可以先跳过 |
| --- | --- | --- |
| 多模态 | 向量相似度、attention、交叉熵 | 分布式训练、完整 RL 推导 |
| SFT / 偏好学习 | next-token loss、梯度、数据切分 | 从头预训练、MoE 部署 |
| RL / 后训练 | 概率期望、log-prob、基本求导 | 机器人控制；需要时再补连续动作方法 |
| 推理与部署 | decoder 前向、张量形状、KV cache | 奖励模型和偏好优化 |
| 检索 / RAG / Agent | embedding、检索指标、基本 API 调用 | 全部模型家族与训练配方 |
| Python / 算法 | 函数、循环、列表与字典 | 大模型知识；这条路线可以单独读 |

这不是给读者分等级。做 RAG 的人不必先学完 SAC，做训练的人也可以暂时不研究服务调度；碰到具体问题，再沿文内链接补就好。

</details>

<span id="01"></span>
<span id="02"></span>
<span id="03"></span>
<span id="04"></span>

## 全部章节

按三个方向整理，和左侧目录一致：**理解模型 → 训练与检验 → 推理与应用**。这是内容类型，不是难度等级。每篇只放在一个主要位置，相关主题用链接接起来，不重复放一份。

点标题读导读，展开小节看全部文章。模型家族放在组件原理之后；技术问答与手写练习在面试准备，设计题与项目拆解在工程实践。学到一半想动手，随时可以过去，不必读完整个板块。

<!-- widget:study-atlas -->

## 阅读与更新

右上角 **EN / 中文** 可以直接切换同一篇文章。中文保留必要的英文术语，英文版也有完整的例子、推导和代码。

顶部搜索或按 `/` 找概念；“我的阅读”保留收藏和最近阅读，数据只在当前浏览器。想知道哪些内容还在补，见[覆盖与待补清单](coverage.md)。参考资料帮我们查漏，本站的章节仍按理解顺序来组织。
