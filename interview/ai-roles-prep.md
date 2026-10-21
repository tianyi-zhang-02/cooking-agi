# RE / MLE 面试：从哪里开始练？

**中文** · [English](ai-roles-prep.en.md)

> 来源核对：2026-10-11 · 公开经验与本站自拟练习，不是公司原题或统一招聘标准。

看论文时觉得懂了，打开空白编辑器却写不出来，很常见。先挑一个小模块写一遍，比继续收藏十份清单更容易发现自己缺什么。这页把公开求职复盘里提到的准备方向，接到站内已有的讲解和实现；不要求你每一项都做完。

## 先看岗位做什么，再决定练多深 {#routes}

RE 是 Research Engineer，MLE 是 Machine Learning Engineer，RS 是 Research Scientist。同一个 title，在不同团队可能做很不一样的事。先看职位描述，再向 recruiter 确认轮次、编码环境、是否允许查文档和使用 AI。不要从“某人没被考过”推导出“这个不用准备”。

| 目标工作 | 先准备的内容 | 再往下练 | 时间紧时先放一放 |
| --- | --- | --- | --- |
| RE / MLE，偏模型和训练 | 算法、[ML 实现](#implement)、训练排错、数据与评估、项目选择 | 数据与目标、分布式训练、实验设计 | 每个模型家族的细节 |
| RS，偏研究 | 已要求的 coding 与实现能力，加上研究问题、证据、个人贡献和后续实验 | 自己方向的论文、推导与失效条件，不必照搬 LLM 题单 | 一口气实现所有热门结构 |
| Inference / serving / ML systems | 算法、基础编码、生成循环与 cache、显存估算、调度 | [请求链路与性能诊断](#inference)，再选一层深入，拿出性能证据 | 背每个框架的全部类名 |
| 通用 SWE | 算法、[读代码](code-comprehension.md)、测试、项目；按通知准备系统设计 | [Python 与 LeetCode](leetcode.md) | 与工作无关的模型推导 |

刚准备实习或 new grad，可以先写 MLP、attention、一个训练循环，再讲清一个自己的项目。已有经验的读者，则可以多练性能取舍、故障恢复和跨团队接口。这个顺序是本站的练习建议，不是不同职级的录用门槛。

## ML Coding：每次留下一个能检查的结果 {#implement}

每次只挑一行：先写输入输出和最小测试，再实现，最后改一个条件。下面链接里既有短讲解，也有可运行代码；“有章节”不代表关掉页面后就能独立完成。

| 要实现什么 | 站内入口 | 写完至少检查一次 |
| --- | --- | --- |
| 两层 MLP 的 forward / backward | [手写模块](../00-foundations/hand-write-kit.md#_5)，配套 `interview_kit.py` | mean loss 的分母；float64 中心差分对照解析梯度，避开 ReLU 拐点 |
| Multi-head causal attention | [多头注意力](../00-foundations/core/multi-head-attention.md)，[NumPy 实现](../00-foundations/code/attention_numpy.py) | 标出 `[B,H,T,D]`；修改未来 token，不能影响更早输出 |
| 稳定 softmax 与交叉熵 | [softmax 算例](../00-foundations/hand-write-kit.md#softmax-numerics)、[交叉熵](../00-foundations/pytorch/cross-entropy.md) | 大 logits、被屏蔽的类别、全 mask；不要先取极小概率再取 log |
| Temperature、top-k、top-p 与 EOS | [解码策略](../00-foundations/core/decoding.md)、[手写实现](../00-foundations/hand-write-kit.md)、[生成何时停止](../00-foundations/core/decoder-only.md#1-assistant) | top-p 保留跨过阈值的 token；各序列独立停止，另有长度上限 |
| 增量解码与 KV cache | [从零实现实验](../00-foundations/code/README.md)、[KV 原理](../00-foundations/deep-dives/kv-cache-and-inference.md) | 固定权重、关闭 dropout，同一 token 序列的逐位置 logits 与全前向在容差内一致；先别比较随机采样文本 |
| 简化 MoE | [Router](../00-foundations/moe/router.md)、[负载均衡](../00-foundations/moe/load-balancing.md) | 同一 token 去多个 expert 后怎样加权；门控归一化、容量限制、空 expert |
| 故意弄错训练循环，再修好 | [PyTorch 训练循环](../00-foundations/pytorch/training-loop.md)、[数据与目标](../practice/post-training/data-and-objectives.md) | label shift、梯度清零、padding loss mask、`train/eval` 各改一处；观察 loss 与梯度，而不只看能否运行 |
| 小型 decoder-only Transformer | [从零实现实验](../00-foundations/code/README.md)、[GPT 家族](../00-foundations/model-families/gpt.md) | attention / FFN / 残差 / norm 接起来；先让小 batch 的训练 loss 下降，再检查泛化 |

最后一行要分清版本：本站 `model.py` 用的是 **RMSNorm、RoPE、GQA 和 SwiGLU**，不是 GPT-2。若练 GPT-2 风格，应明确 learned absolute positions、LayerNorm 的 pre-norm 位置、GELU MLP 和最终 LayerNorm；不能只改文件名。可以跟 [Karpathy 的 GPT-2 实现视频](https://www.youtube.com/watch?v=l8pRSuU81PU)做，再对照两种结构。本站目前没有第二份完整的 GPT-2 复现代码。

## 小算法和张量操作，也要能写 {#small-problems}

[五组短练习](coding-checks.md)补的是容易被“大模型题单”挤掉的基本功：

| 练习 | 不只检查答案，还检查什么 |
| --- | --- |
| kNN：循环、向量化、分块 | 邻居和投票一致；同距离怎么处理；中间张量有多大 |
| 二分的两个边界 | 第一个 ≥ target、最后一个 ≤ target；空数组和重复元素 |
| Broadcasting | `[B,1] - [B]` 为什么变成 `[B,B]`，而不是逐样本误差 |
| 任务依赖图 | 拓扑顺序、环、可同时开始的任务；“提交了”不等于“完成了” |
| 两状态 Markov chain | 多步概率、平稳分布、首次到达时间是三个不同问题 |

算法主线仍在 [Python 与 LeetCode](leetcode.md)。Markov chain 在这里是一个小型概率练习，不重新展开整套 Quant 备考。

## 能解释机制，比只背术语更有用 {#concepts}

| 主题 | 可以这样检查理解 | 继续读 |
| --- | --- | --- |
| 初始化 | 输入宽度翻倍，为什么权重尺度通常要变？推导用了哪些独立性假设？ | [激活与初始化](../00-foundations/deep-dives/activation-and-initialization.md) |
| 位置编码 | 绝对位置与 RoPE 分别怎样进入计算？长度变长，不只改一个 max length 吧？ | [位置与上下文](../00-foundations/deep-dives/position-and-context.md) |
| PPO / GRPO | advantage 来自哪里？同组奖励全相等时还剩什么信号？ | [PPO 之后](../05-post-training/rlhf/after-rlhf.md)、[稀疏奖励组](../05-post-training/deep-rl/llm-bridge.md#sparse-groups)、[多步 agent 的信用分配（credit assignment）](../11-frontier/agentic-rl/credit-and-training.md) |
| DP / TP / PP | 拆的是样本、层内计算还是层？每种方案在哪儿通信？ | [分布式训练](../06-systems/distributed-training.md)、[框架与训练栈](../06-systems/training-frameworks.md) |
| 对比学习 | 正负样本怎样构造？换 batch size 后 loss 的归一化有没有变？ | [CLIP](../03-multimodal-learning/clip.md)、[SigLIP](../03-multimodal-learning/siglip.md) |

这些是复习问题，不是只适用于某几家公司的考点。做 CV、语音、机器人或其他方向时，换成对应的数据、模型和评价问题；不要拿一份 LLM 清单替代岗位本身。

## Inference：从请求讲到 GPU，再讲回来 {#inference}

先读 [vLLM / SGLang 的完整 stack](../06-systems/inference-frameworks.md#engine-map)，再用同一条请求往下追。练习目标不是背类名，而是能说明：谁持有什么状态，数据交给谁，慢了应该测哪里。

| 一层层追问 | 要落到的具体东西 |
| --- | --- |
| 这套服务要服务谁？ | 输入与输出长度、到达率、TTFT / ITL 目标、质量和预算 |
| 请求进来后谁决定下一轮算什么？ | token 预算、等待队列、KV 容量、取消与抢占 |
| 有多少东西放在 GPU 上？ | [权重、KV、workspace 分账](../00-foundations/deep-dives/kv-cache-and-inference.md#inference-budget)，不是只用参数量乘精度 |
| 两个请求共享前缀，复用了什么？ | token 前缀与模型配置兼容、缓存归属、跨副本命中，以及租户隔离 |
| 为什么吞吐高了，体验却差了？ | [长 prefill 与 decode 争抢](../06-systems/llm-serving.md#token-budget)；看尾延迟，不只看 tokens/s |
| GPU 利用率低，先改什么？ | 分开检查 CPU 输入、队列、执行、HBM 带宽、通信和输出缓冲；先[取 trace](../12-infra/performance.md#profiling)，再改一个变量 |
| 什么时候拆 prefill / decode？ | 在测出的干扰之外，算上 KV 传输、两端队列与失败恢复成本 |

准备 inference 方向，至少挑一层准备到能讲细节：调度策略、KV cache 管理，或者你拿得出的性能证据。传统系统设计也别丢；轮次还是要问清楚。

## 项目、研究讨论和 BQ，别拖到最后 {#projects}

拿一个自己的项目，准备一张表就够开始：问题与 baseline、你的改动、比较过的替代方案、实验结果、还没解决的事。把个人工作与团队成果分开。早期项目和参与较少的论文也重新读一遍；记不清时先查记录，不现场补剧情。[Andrey 的复盘](https://reyzharkov.com/blog/posts/interviews-2025-ml-research-engineer-uk)里就有被追问旧项目和论文细节的经历。

接着练三种追问，例子由本站自拟：

- **结果涨了，真是方法有效吗？** 固定数据切分和评价集，排查泄漏，报告 seed 波动、算力和调参预算；不要把一次最高分当稳定收益。配合[实验与发布](../practice/post-training/experiments-and-release.md)。
- **预算砍半怎么办？** 先决定保留哪个假设检验，再缩数据、模型或实验数量，讲清哪些结论因此无法支持。不是所有项目都只能回答“上 LoRA”。
- **结果与预期相反呢？** 区分实现错误、测量问题和假设不成立；给一个能区分它们的后续实验，而不只说继续调参。

BQ 可以准备合作分歧、失败、模糊需求各一个真实故事：当时有哪些选择，你怎么处理，结果与后来改变了什么。不要求把所有经历包装成成功。项目范围与职位有关；校园项目也能讲清责任，不需要编出跨团队领导经历。

## 免费资料怎么接着用 {#resources}

不必把它们全部刷完。下面分别解决“需要练什么”和“怎样动手”。

| 资料 | 适合拿来做什么 |
| --- | --- |
| [Alisa：Notes on the Industry Job Search](https://alisawuffles.github.io/blog/job-search/) · 2026-06 | PhD 求职视角；把 ML coding、技术讨论和研究讨论区分开，不只准备论文介绍 |
| [Silvia：ML Job Interviews](https://silviasapora.github.io/blog/ml-interviews.html) · 2026-06 | ML PhD 的 RS 求职经历；实现和 debugging 清单。她的题目范围不等于所有人的必修课 |
| [Andrey：ML/Research Engineer interviews 2025](https://reyzharkov.com/blog/posts/interviews-2025-ml-research-engineer-uk) · 2026-03 发布 | 复盘的是 2025 年英国 / 远程求职，有经验工程师的视角；别把发布时间当面试发生年份 |
| [Stanford CS336 Assignment 1](https://github.com/stanford-cs336/assignment1-basics) | 从学生版接口写实现、接测试；是系统性作业，不是现成答案。正在修课时遵守课程协作规则 |
| [Karpathy：Zero to Hero](https://karpathy.ai/zero-to-hero.html) → [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) | 先练从零实现，再练训练与效率；视频关掉后重新写，比只跟着敲更能发现缺口 |
| [TinyZero](https://github.com/Jiayi-Pan/TinyZero) → [verl](https://github.com/volcengine/verl) | 有余力再做 RL 实验。TinyZero 已声明不再积极维护，建议新实验用 verl；其 0.5B 结果只描述该模型与设置，不是小模型能力上限 |

这几篇作者的背景差别很大。我们借用它们发现遗漏，练习、例子和站内组织另写，不搬运公司题目、薪资推断或成功门槛。朋友的个人建议也只能作为补充，不写成某家公司的内部标准。

如果只打算从一件事开始：关掉补全，计时写一个 attention 或 MLP，给它加测试，再解释哪一步最不确定。平时当然可以用 AI 辅助学习；这一遍只是想知道，哪些已经是你自己的能力。
