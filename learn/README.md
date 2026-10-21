# 基础与原理

**中文** · [English](README.en.md)

这里讲模型的结构、训练方法、评估与推理。每个主题从例子讲起，再展开推导和实现。

- **刚开始学：**沿着下面的入门顺序读，先弄懂一次计算。
- **已有基础：**在问题索引或章节目录里找需要的内容。
- **想动手做：**配合文中的小实验，再到[工程实践](../practice/README.md)看完整系统。

不确定该选哪一部分，可以先看[阅读路线](choose-your-path.md)；主要为面试复习，直接去[面试准备](../interview/README.md)。

近期补充（2026-10）：

- [前沿观察](../11-frontier/README.md)：新增的第四部分。[世界模型](../11-frontier/world-models/README.md)和[递归自我改进（RSI）](../11-frontier/self-improvement/README.md)两组已审阅；[测试时计算（test-time compute）](../11-frontier/test-time-compute/README.md)、[用 RL 训练 agent](../11-frontier/agentic-rl/README.md)、[机制可解释性](../11-frontier/interpretability/README.md)三组是待审核草稿。
- 两个新板块，都是待审核草稿：[AI 基础设施（AI infra）](../12-infra/README.md)，从芯片、GPU、CUDA 讲到框架和集群；[感知、机器人与自动驾驶](../13-embodied/README.md)。
- 2026-10-16 又加了三组待审核草稿：[Scaling law 与超参数](../00-foundations/scaling/README.md)，开训前怎样定模型大小、数据量和学习率（LR）；[扩散语言模型（dLLM）](../11-frontier/diffusion-lm/README.md)，放在前沿观察里；[量化](../12-infra/quantization/README.md)，放在 AI 基础设施里。
- 强化学习补了三篇：[学什么样的模型](../05-post-training/deep-rl/model-types.md)、[规划和学习怎样互相喂](../05-post-training/deep-rl/planning-and-learning.md)、[离线策略评估（OPE）的估计量](../05-post-training/deep-rl/ope-estimators.md)。

待审核草稿的开头有提示框，侧栏标「（待审核）」；引用其中的事实前，请回到文中链接的一手资料核对。

## 从基础开始

向量、loss 和梯度还不熟，可以先读[从线性模型到神经网络](../00-foundations/from-linear-to-neural.md)。下面的第 1–3 步是模型计算与训练的基础，第 5 步讲评估；后训练和推理优化可以稍后再看。阅读时可以配合 [PyTorch 小实验](../00-foundations/pytorch/README.md)。

<nav class="study-route" aria-label="建议阅读主线">
<a href="../00-foundations/core/tokenization.md"><small>01 / 输入</small><strong>文本怎样变成向量</strong><span>Token、词表与序列</span></a>
<a href="../00-foundations/core/README.md"><small>02 / 模型</small><strong>Transformer 在算什么</strong><span>Attention、残差、Norm 与 FFN</span></a>
<a href="pretraining/README.md"><small>03 / 训练</small><strong>数据怎样改变参数</strong><span>目标、梯度与一次更新</span></a>
<a href="../05-post-training/README.md"><small>04 / 调整行为</small><strong>示范、偏好与奖励</strong><span>SFT、蒸馏与强化学习</span></a>
<a href="../07-evaluation/README.md"><small>05 / 检验</small><strong>怎样知道真的变好了</strong><span>对照、指标与具体错误</span></a>
<a href="inference/README.md"><small>06 / 生成</small><strong>怎样把回答算出来</strong><span>采样、缓存与推理成本</span></a>
</nav>

## 带着问题来，可以这样读

从左边找到接近你的问题，按中间一栏选读。右边列出了阅读时值得留意的细节。

| 你想弄明白什么 | 相关阅读 | 重点看什么 |
| --- | --- | --- |
| FSDP、vLLM、SGLang 都听过，却分不清关系 | [训练框架](../06-systems/training-frameworks.md) → [推理引擎](../06-systems/inference-frameworks.md) | 训练状态、请求调度与算子分别由谁负责 |
| 想知道模型怎么成为一个完整产品 | [ML 系统全貌](../practice/ml-system/README.md) → [数据到模型](../practice/ml-system/data-to-model.md) → [上线与反馈](../practice/ml-system/serving-and-feedback.md) | 输入输出、版本、失败处理与团队交接 |
| 公式懂了，训练代码还是写不对 | [张量与存储](../00-foundations/pytorch/tensors-and-storage.md) → [自动求导](../00-foundations/pytorch/autograd.md) → [训练循环](../00-foundations/pytorch/training-loop.md) | 参数有没有更新，验证 loss 的分母对不对 |
| 想微调，先准备什么 | [预训练数据流程](../00-foundations/deep-dives/pretraining-pipeline.md) → [SFT](../05-post-training/sft-and-its-ceiling.md) → [LoRA / QLoRA](../05-post-training/lora-and-qlora.md) | 标签、loss mask、数据切分与显存 |
| 新架构名字很多，分不清差别 | [组件阅读地图](../00-foundations/deep-dives/README.md) → [模型报告精读](../00-foundations/model-families/README.md) | 改的是序列、缓存、专家，还是网络深度 |
| 图片接上了，模型真的看了吗 | [图文如何配对](../03-multimodal-learning/clip.md) → [视觉特征进入 LLM](../03-multimodal-learning/vision-to-language.md) → [VLM 微调与评估](../03-multimodal-learning/vlm-finetuning.md) | 换掉图片里的事实，答案会不会跟着变 |
| RL 公式会推，实验却不稳定 | [价值估计与 GAE](../05-post-training/deep-rl/actor-critic-gae.md) → [PPO clipping](../05-post-training/rlhf/ppo-clipping.md) → [实验排错](../05-post-training/deep-rl/experiments.md) | reward、advantage、更新幅度各出了什么变化 |
| 检索分数涨了，回答却没有更好 | [双塔检索](../04-search/dual-encoder.md) → [混合检索与重排](../04-search/hybrid-and-reranking.md) → [指标稳健性](../07-evaluation/metric-robustness.md) | 固定候选池后，哪些样本真的改善了 |
| 换 embedding 模型，需要改哪些地方？ | [Qwen Embedding 与 BGE-M3](../04-search/embedding-models.md) → [向量索引](../04-search/vector-indexes.md) | 输入与 pooling 怎么对齐，什么时候要重建索引 |
| `nvidia-smi` 显示 100%，生成还是很慢 | [AI 基础设施](../12-infra/README.md) → [GPU 架构](../12-infra/gpu-architecture.md) → [性能：roofline 与 MFU](../12-infra/performance.md) | 每从显存搬 1 字节做几次运算，瓶颈在带宽还是算力（待审核草稿） |
| 车和机器人怎样从像素走到动作 | [感知与几何](../13-embodied/perception/README.md) → [自动驾驶](../13-embodied/driving/README.md) → [机器人学习](../13-embodied/robot-learning/README.md) | 每一步交给下一步的是什么（待审核草稿） |
| “世界模型”“自我改进”到底指什么 | [前沿](../11-frontier/README.md) → [世界模型](../11-frontier/world-models/README.md) → [递归自我改进](../11-frontier/self-improvement/README.md) | 同一个词说的是哪件事，证据走到了哪一步 |

<span id="area-coding"></span>
<span id="subject-coding"></span>
<span id="subject-algorithms"></span>
<span id="subject-ml-exercises"></span>
<span id="subject-system-design"></span>

相关练习集中在[面试准备](../interview/README.md)：ML / LLM 问答、ML Coding、Python 和 LeetCode。系统设计与项目拆解放在[工程实践](../practice/README.md)。本页保留原理讲解和配套小实验。

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

这些是阅读所需的前置知识。例如，做 RAG 可以暂时跳过 SAC，做训练也可以之后再研究服务调度。

</details>

<span id="01"></span>
<span id="02"></span>
<span id="03"></span>
<span id="04"></span>

## 全部章节

分为**理解模型、训练与检验、推理与应用、前沿观察**四个部分，与左侧目录一致。点标题看导读，展开小节找文章。相关主题会在文中互相链接。

<!-- widget:study-atlas -->

## 阅读与更新

想了解这些技术的发展过程，可以看[编年史](technology-timeline.md)。

右上角 **EN / 中文** 可以直接切换同一篇文章。中文保留必要的英文术语，英文版也有完整的例子、推导和代码。

顶部搜索或快捷键 `/` 可以查概念；「我的阅读」里有收藏和最近读过的文章，记录保存在当前浏览器。更新进度见[覆盖与待补清单](coverage.md)。
