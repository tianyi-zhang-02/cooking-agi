# 用强化学习训练 agent（agentic RL）：从一问一答到多步任务

**中文** · [English](README.en.md)

> 阅读时间：约 7 分钟 · 最近审阅：2026-10

用 RLVR 训数学题时，一条样本是一个 prompt、一个回答、一个分数。现在要训一个修 bug 的 agent：它读文件、改代码、跑测试，看了报错再改，几十轮之后提交，隐藏测试决定它得 1 分还是 0 分。**算法骨架没有变：采样、打分、按 advantage 更新策略。变的是一条样本的样子：它是一段在沙盒里跑出来的交互，奖励多半只在最后给一次，工具返回的文字混在序列里却不是模型的动作，跑一条可能要几分钟到几十分钟。于是难点从“选哪个 loss”移到了三处：环境和奖励靠不靠得住，一条轨迹的成败怎么分给其中的几十个动作，rollout 怎么跑得起。**

## 从一问一答到多步任务，变了什么

| | 单轮 RLVR | 多轮 agent RL | 在哪篇展开 |
| --- | --- | --- | --- |
| 一条样本 | prompt 加一个回答 | prompt，加上多轮的（动作，观察） | 第 2 篇 |
| 状态 | prompt 和已生成的 token | 上下文，加上上下文之外的世界：文件、数据库、网页 | 第 1 篇 |
| 观察 | 没有 | 工具输出、测试日志、网页、用户回复，都不是模型采样出来的 | 第 2 篇 |
| 奖励 | 回答之后由检查器打分 | 多半在回合结束时，看环境变成了什么样 | 第 1 篇 |
| 检查器在哪 | 模型碰不到 | 常常和 agent 在同一个沙盒里 | 第 1 篇 |
| 长度 | 一轮 | 几十到上千次工具调用；Kimi K3 的技术报告（2026-07）说，它的长程个人助理任务里一次 rollout 可能有数千次工具调用、数百万 token 上下文 | 第 2 篇 |
| 一次 rollout 的成本 | 一次生成，加一次便宜的检查 | 起容器、执行命令、跑测试；DeepSWE（2025-07）生成限时 20 分钟、测试限时 5 分钟 | 第 1、2 篇 |
| 怎么结束 | 生成结束符 | 提交、失败、预算用完、工具崩溃、超时，各不相同 | 第 2 篇 |

<!-- paper-figure:taubench-overview -->

为什么工具输出不进梯度，可以从轨迹的概率直接看出来。记 $h_k$ 为第 $k$ 个动作之前的全部历史，$a_k$ 为第 $k$ 个动作（一段 token），$o_k$ 为环境随后返回的观察：

$$
\log p_\theta(\tau)=\sum_{k=1}^{T}\log\pi_\theta(a_k\mid h_k)+\sum_{k=1}^{T}\log P_{\rm env}(o_k\mid h_k,a_k)
$$

第二个求和是环境的转移，和参数 $\theta$ 无关，求梯度时直接消失，所以 policy gradient 只剩第一项：

$$
\nabla_\theta J(\theta)=\mathbb E_{\tau}\Big[R(\tau)\sum_{k=1}^{T}\nabla_\theta\log\pi_\theta(a_k\mid h_k)\Big]
$$

举个数：一条两轮的轨迹，策略选 $a_1$ 的概率 0.5，工具返回 $o_1$ 的概率 0.9，选 $a_2$ 的概率 0.2，$o_2$ 是确定的，整条轨迹概率 $0.5\times0.9\times0.2\times1=0.09$。改动 $\theta$ 只能改变 0.5 和 0.2；那个 0.9 是环境的事。前提是环境里没有用同一套权重：如果子 agent 也用正在训练的模型，它的输出就和 $\theta$ 有关了；Kimi K2.5（2026-02）的做法是冻结子 agent，把它们的输出当作观察。实现里把工具输出的 token 从 loss 里 mask 掉，做的就是这件事。实际训练会把 $R(\tau)$ 换成减过基线的 advantage，再加上概率比和 clipping，见[接到 LLM](../../05-post-training/deep-rl/llm-bridge.md)。

## 时间线

只收一手来源能核对的事件。各团队报告的分数都是自报，评测设置各不相同。

| 日期 | 事件 | 机构 | 一句话 |
| --- | --- | --- | --- |
| 2024-06 | [τ-bench](https://arxiv.org/abs/2406.12045) | Sierra | LLM 扮演用户，按结束时的数据库状态评分 |
| 2025-01-22 | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | DeepSeek | 局限一节说，评估耗时太长，还没在软件工程任务上大规模用 RL |
| 2025-01-23 | [Computer-Using Agent](https://openai.com/index/computer-using-agent/) | OpenAI | 说 CUA 把 GPT-4o 的视觉能力和通过 RL 得到的推理结合起来 |
| 2025-02-02 | [deep research](https://openai.com/index/introducing-deep-research/) | OpenAI | 说它用端到端 RL 在困难的浏览和推理任务上训练 |
| 2025-02-24 | [Claude 3.7 Sonnet 系统卡](https://www-cdn.anthropic.com/9ff93dfa8f445c932415d335c88852ef47f1201e.pdf) | Anthropic | RL 训练中的 reward hacking，让模型在 agent 编码环境里给测试写特判 |
| 2025-02-25 | [SWE-RL](https://arxiv.org/abs/2502.18449) | Meta | 单轮设置：按补丁和参考补丁的相似度给奖励 |
| 2025-03-12 | [Search-R1](https://arxiv.org/abs/2503.09516) | Jin 等 | 多轮检索 RL，检索回来的 token 不进 loss |
| 2025-04-16 | [o3 与 o4-mini](https://openai.com/index/introducing-o3-and-o4-mini/) | OpenAI | 说用 RL 训练模型使用工具，以及判断何时使用 |
| 2025-05-16 | [codex-1](https://openai.com/index/introducing-codex/) | OpenAI | 说在多种环境的真实编码任务上用 RL 训练 |
| 2025-06-20 | [Kimi-Researcher](https://moonshotai.github.io/Kimi-Researcher/) | Moonshot AI | 端到端 agentic RL；HLE Pass@1 从 8.6% 到 26.9%（自报） |
| 2025-07-02 | [DeepSWE](https://www.together.ai/blog/deepswe) | Agentica、Together AI | Qwen3-32B 只用 RL，SWE-bench Verified Pass@1 42.2%（自报） |
| 2025-07-22 | [Qwen3-Coder](https://qwenlm.github.io/blog/qwen3-coder/) | Qwen | 长程 Agent RL，2 万个独立环境并行 |
| 2025-11-21 | [Natural emergent misalignment](https://www.anthropic.com/research/emergent-misalignment-reward-hacking) | Anthropic | 在生产编码环境里学会钻空子，泛化到其他不对齐行为 |
| 2025-12-02 | [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) | DeepSeek | 1,800 多个环境、8.5 万个 prompt；后训练算力超过预训练成本的 10% |
| 2026-02-17 | [GLM-5](https://arxiv.org/abs/2602.15763) | Z.ai | 1 万多个可验证代码环境；生成与训练解耦的异步 agent RL |
| 2026-06 | [GLM-5.2](https://z.ai/blog/glm-5.2) | Z.ai | 长任务从组相对改成 critic PPO |
| 2026-07 | [Kimi K3 技术报告](https://github.com/MoonshotAI/Kimi-K3) | Moonshot AI | 百万 token 的 agentic RL；训练和评测共创建 5,121 万个沙盒 |
| 2026-08-18 | [Pacing model development](https://openai.com/index/pacing-model-development-cyber-capabilities/) | OpenAI | 因 Hugging Face 事件，以及新模型可能达到其网络安全 Critical 阈值的证据，暂停面向部署模型的 RL 训练两周 |

几家实验室公开的内容很不一样。OpenAI 的产品发布说明了 agent 是用 RL 训的，但没有给算法和环境细节。Anthropic 的系统卡不讲训练算法，却持续报告训练环境里的钻空子和防护：从 Claude 3.7 Sonnet 的特判，到 Claude Opus 5.5（2026-09-22）用分类器筛查训练回合。Google DeepMind 的 [Gemini 2.5 技术报告](https://arxiv.org/abs/2507.06261)（2025-07）说 RL 环境里加入了需要多步动作和工具使用的任务。给出算法、环境数量和基础设施细节的，主要是开放权重模型的团队：Moonshot AI、DeepSeek、Qwen、Z.ai，以及 Agentica 这样的开源项目。

## 哪些已经比较确定，哪些还在变

每格都对应一条带日期的公开记录；“接下来看”一栏写的是什么样的结果会改变判断，不是预测它会发生。

| 比较确定 | 还在变 | 接下来看什么信号 |
| --- | --- | --- |
| 结果奖励加可执行检查，足以用 RL 把 agent 训起来：DeepSWE（2025-07）只用 RL 把 Qwen3-32B 的 SWE-bench Verified Pass@1 训到 42.2%（自报）；OpenAI 说 deep research（2025-02）和 codex-1（2025-05）用 RL 训练 | 奖励不可验证的任务怎么给分：Kimi K2（2025-07）用模型自评的 rubric 奖励；Gaia2（2026-02）报告 RL 策略骗过了做软检查的 LLM judge | 用 judge 做奖励、并由独立人工评估确认收益的 agent RL 结果 |
| 工具输出留在上下文里但不进 loss：Search-R1（2025-03）的消融里，加 mask 平均 exact match 0.431，不加 0.343 | 信用分配的粒度：episode 级组相对（DeepSWE 2025-07、GLM-5 2026-02）、按状态分组（GiGPO 2025-05）、critic PPO（GLM-5.2 2026-06） | 长代码任务上、同样 rollout 预算下的公开对比 |
| 检查器会被钻，学会的钻法还会泛化：Claude 3.7 Sonnet 系统卡（2025-02）、OpenAI 的 Baker 等（2025-03）、Anthropic 的生产环境实验（2025-11） | 怎么防：训练前评审环境（Anthropic 2026-04 说没能拦下全部）、训练中用分类器筛查（Claude Opus 5.5 系统卡，2026-09）、隐藏验证器（Kimi K3，2026-07） | 在固定的一组环境上、跨模型代际报告的钻空子比例 |
| 环境和评分是主要成本：DeepSeek-R1（2025-01）因评估太慢没在软件工程上大规模用 RL；DeepSWE 每次迭代并行 512 个容器 | 环境规模：代码任务从数千个（SWE-Gym，2024-12）到上万个可验证环境（GLM-5，2026-02），Kimi K3（2026-07）训练和评测中创建的沙盒以千万计；真实环境和合成环境各占多少 | “少而精”对“多而粗”的消融；在一种 agent 外壳里训练、换到另一种外壳里用的掉点 |

## 这一组怎么读

需要补基础的话：policy gradient、GRPO 和 critic 在 LLM 上怎么用，见[接到 LLM](../../05-post-training/deep-rl/llm-bridge.md)；测试当奖励会漏掉什么，见[可验证奖励](../../05-post-training/verifiable-rewards.md)；终止和截断，见 [MC 与 TD](../../05-post-training/deep-rl/returns-and-td.md)；数据变旧怎么修正，见[异步 RL](../../05-post-training/async-policy-learning.md)；RLHF 的基本角色，见 [RLHF](../../05-post-training/rlhf/README.md)。agent 本身的结构见 [Agents](../../10-agents/README.md)，多 agent 什么时候值得拆见[一个 agent 还是多个](../../10-agents/single-vs-multi-agent.md)；用自产数据训练、为什么卡在验证器，见[生成、验证、再训练](../self-improvement/generate-verify-train.md)。

1. [环境与奖励从哪里来](environments-and-rewards.md)：代码沙盒、浏览、模拟用户、工具 API 和电脑操作环境；四种奖励来源；已经测到的钻空子；测试只抓住一部分失败时，期望奖励和真实成功率差多少
2. [多步回合的信用分配与训练系统](credit-and-training.md)：3 轮回合、4 条轨迹的组相对 advantage；哪些 token 有梯度；GiGPO、turn 级 critic 和逐轮奖励；长上下文、截断和异步 rollout
3. [复习题](review.md)：面试题和自检
