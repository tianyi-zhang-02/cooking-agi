# 基础与原理

**中文** · [English](README.en.md)

模型怎样学习，一次回答又是怎样算出来的？这里按模型、训练、评估和推理整理笔记。可以从小例子读起，也可以直接找需要的推导和实现。

<nav class="study-route reading-entry" aria-label="选择阅读方式">
<a href="#_1"><strong>从头开始</strong><span>按顺序读，先弄懂一次计算和更新。</span></a>
<a href="#_2"><strong>带着问题来</strong><span>从具体问题找到相关章节。</span></a>
<a href="#_3"><strong>直接找章节</strong><span>展开目录，选自己感兴趣的主题。</span></a>
</nav>

想动手做，可以配合文中的小实验，再到[工程实践](../practice/README.md)看完整系统。不确定从哪里开始，先看[阅读路线](choose-your-path.md)；主要为面试复习，可以去[面试准备](../interview/README.md)。

## 从基础开始 {#_1}

概念区分成七条学习线。每条线按阅读顺序排了几站，每站讲清楚一个概念；文章顶部会标出它在哪条线的第几站、先读哪一篇、下一站去哪。向量、loss 和梯度还不熟，从第一条线的第一站[从线性模型到神经网络](../00-foundations/from-linear-to-neural.md)开始。阅读时可以配合 [PyTorch 小实验](../00-foundations/pytorch/README.md)。

<!-- widget:track-map -->

想把数学和工程连起来，可以从[矩阵梯度与训练系统](../00-foundations/deep-dives/matrix-gradients-to-systems.md)开始，再沿[每次一个问题的选读路线](../practice/runtime-reading/README.md#small-steps)走到 attention、多卡、推理与 NeMo RL。

## 带着问题来，可以这样读 {#_2}

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
| `nvidia-smi` 显示 100%，生成还是很慢 | [AI 基础设施](../12-infra/README.md) → [GPU 架构](../12-infra/gpu-architecture.md) → [性能：roofline 与 MFU](../12-infra/performance.md) | 每从显存搬 1 字节做几次运算，瓶颈在带宽还是算力 |
| 车和机器人怎样从像素走到动作 | [感知与几何](../13-embodied/perception/README.md) → [自动驾驶](../13-embodied/driving/README.md) → [机器人学习](../13-embodied/robot-learning/README.md) | 每一步交给下一步的是什么 |
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

## 全部章节 {#_3}

完整目录和左侧目录一致：先是上面的七条学习线，再是**延伸阅读**（模型报告、组件设计、MoE、Looped 模型和经典深度强化学习）和**还在变化**（前沿方向）。点标题看导读，展开小节找文章。不在主线上的文章都在这里，地址没有变。

<!-- widget:study-atlas -->

<details markdown="1">
<summary>此前新增的专题（2026-10）</summary>

- [前沿观察](../11-frontier/README.md)：新增的第四部分。[世界模型](../11-frontier/world-models/README.md)和[递归自我改进（RSI）](../11-frontier/self-improvement/README.md)，以及[测试时计算（test-time compute）](../11-frontier/test-time-compute/README.md)、[用 RL 训练 agent](../11-frontier/agentic-rl/README.md)、[机制可解释性](../11-frontier/interpretability/README.md)，五组都已审阅。
- 两个新板块：[AI 基础设施（AI infra）](../12-infra/README.md)，从芯片、GPU、CUDA 讲到框架和集群；[感知、机器人与自动驾驶](../13-embodied/README.md)。
- 2026-10-16 又加了三组：[Scaling law 与超参数](../00-foundations/scaling/README.md)，开训前怎样定模型大小、数据量和学习率（LR）；[扩散语言模型（dLLM）](../11-frontier/diffusion-lm/README.md)，放在前沿观察里；[量化](../12-infra/quantization/README.md)，放在 AI 基础设施里。
- 强化学习补了三篇：[学什么样的模型](../05-post-training/deep-rl/model-types.md)、[规划与学习怎样配合](../05-post-training/deep-rl/planning-and-learning.md)、[离线策略评估（OPE）的估计量](../05-post-training/deep-rl/ope-estimators.md)。

这些专题在 2026-10-21 完成了首轮一手资料核对，并去掉草稿标记；后续修订继续记录在各篇笔记里。

</details>

## 阅读与更新 {#_4}

想了解这些技术的发展过程，可以看[编年史](technology-timeline.md)。

右上角 **EN / 中文** 可以直接切换同一篇文章。中文保留必要的英文术语，英文版也有完整的例子、推导和代码。

顶部搜索或快捷键 `/` 可以查概念；「我的阅读」里有收藏和最近读过的文章，记录保存在当前浏览器。更新进度见[覆盖与待补清单](coverage.md)。
