# 机制可解释性：打开模型看它怎么算

**中文** · [English](README.en.md)

> 阅读时间：约 7 分钟 · 草稿，待审核

你可能读到过这样的说法：Claude 里有一个“金门大桥”特征，把它调大，模型会说自己就是金门大桥；Claude 3.5 Haiku 写诗时，会在开始写一行之前先想好行尾要押的词。这些结论是怎么得出来的，能信几分？机制可解释性（mechanistic interpretability）做的就是这件事：**除了看输入和输出，还要在模型内部找出特征（feature）和电路（circuit），用它们解释某个行为，再通过改动内部状态检验这个解释。截至 2026 年 10 月，这套工具能在一部分模型、一部分提示词上给出可以验证的局部解释，离完整读懂一个前沿模型还差得远。**

> **待审核草稿。** 这一篇还没有经过站主审阅，事实、取舍和措辞都可能修改；引用前请回到文中链接的一手资料核对。

## 它想回答什么：行为评估之外的那一步

行为评估问的是“在这些输入上，输出对不对”。可解释性问的是“模型内部哪个中间量产生了这个输出；如果改掉它，输出会不会按预期变化”。

举个例子。两个模型都能把 “Fact: the capital of the state containing Dallas is” 补成 Austin。一个可能先在内部得出 Texas，再从 Texas 走到 Austin；另一个可能直接记住了 Dallas 和 Austin 的搭配。准确率一样，行为评估分不出来。Anthropic 在 Claude 3.5 Haiku 里找到的是两者都有：一条经过 “Texas” 特征的两步路径，外加一条从 Dallas 直接到 “say Austin” 的捷径；抑制 “Dallas” 特征后，“Texas” 特征和下游跟着变弱（[Lindsey 等，2025-03](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)）。换一批城市、或者模型认为自己正在被测试时，这类区别就可能变得要紧。

两个关键词：

- **特征**：模型内部表示某个概念的单位。最常用的工作假设是：一个特征对应激活空间里的一个方向（线性表示假说，见第 1 篇）。
- **电路**：特征之间经过具体部件（attention head、MLP）连起来的计算路径，而且这条路径能用干预检验。

| | 行为评估 | 机制可解释性 |
| --- | --- | --- |
| 问什么 | 在这些输入上输出对不对 | 内部哪个中间量导致了这个输出 |
| 证据 | 准确率、成功率、偏好 | 探针读数、修补效果、归因图加干预 |
| 强项 | 便宜，能规模化，直接对应用户关心的结果 | 能区分“同分不同机制”，可能看到输出里没有表现出来的状态 |
| 弱项 | 不知道模型为什么对；模型知道自己在被测时，行为可能变 | 慢、贵、覆盖少；常常解释的是一个近似的替代模型 |

世界模型系列的[它真的懂世界吗](../world-models/do-they-understand.md)已经讲过用探针检查模型内部有没有世界状态（Othello-GPT 等），这里不重复，只讲这些工具本身。

## 主要工具

| 工具 | 做什么 | 给出的证据 | 主要的坑 | 在哪篇 |
| --- | --- | --- | --- | --- |
| 线性探针 | 在冻结的激活上训练一个线性分类器 | 信息读得出来 | 可能是探针自己学会了任务；读得出不等于在用 | 第 1 篇 |
| 激活引导（steering） | 沿一个方向加减激活 | 这个方向能改变行为 | 改得动不等于它就是那个概念；强度要调，能力可能受损 | 第 1 篇 |
| 稀疏自编码器、transcoder | 把激活拆成少数几个字典方向的和 | 不用标签就能找出大量候选特征 | 重建不完整；标签可能不准；下游任务上常输给简单基线 | 第 2 篇 |
| 激活修补（causal tracing） | 把一次运行里的中间值换成另一次运行的 | 某个部件对行为是必要的还是足够的 | 结论会随指标和破坏方式变 | 第 3 篇 |
| 路径修补、自动电路发现 | 只沿特定的边修补 | 行为经过哪些路径 | 有非线性时，各条路径的效果加起来不等于总效果 | 第 3 篇 |
| 归因图（attribution graph） | 用替代模型画出单个提示词上的特征连线 | 自动生成机制假设 | 解释的是替代模型；原版不解释注意力为什么看某个位置 | 第 3 篇 |
| 自然语言读出 | 让模型把激活翻译成文字 | 直接可读的假设 | 会编造细节；说不清依据是激活的哪一部分 | 第 2、3 篇 |

## 一条时间线

- **2020-03** [Zoom In](https://distill.pub/2020/circuits/zoom-in/)（Olah 等，Distill）：在视觉模型里提出用“特征”和“电路”来描述网络。
- **2021-12** [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)（Anthropic）：把 Transformer 写成各层对残差流的读写，在两层纯注意力小模型里找到 induction head。
- **2022-02** [ROME](https://arxiv.org/abs/2202.05262)：用 causal tracing 发现 GPT-2 XL 的事实回忆集中在主语最后一个 token 上的中间层 MLP。
- **2022-03** [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)：训练早期有一次相变，induction head 形成和上下文学习能力提升同时发生；作者自己标明，小的纯注意力模型上是因果证据，大模型上主要是相关证据。
- **2022-09** [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)：在小 ReLU 网络里直接看到“特征比维度多”的叠加。
- **2022-11** [IOI 电路](https://arxiv.org/abs/2211.00593)：GPT-2 small 里一个由 26 个 attention head、7 类角色组成的电路。
- **2023-09 至 10** 稀疏自编码器（SAE）出现：[Cunningham 等](https://arxiv.org/abs/2309.08600)；Anthropic 的 [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) 把单层模型的 512 个神经元拆成最多 131,072 个特征。
- **2024-05** [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)：在 Claude 3 Sonnet 中间层训练 1M、4M、34M 特征的 SAE；“金门大桥 Claude”作为研究演示上线 24 小时（[Anthropic](https://www.anthropic.com/news/golden-gate-claude)）。
- **2024-06 至 08** OpenAI 的 [TopK SAE](https://arxiv.org/abs/2406.04093) 在 GPT-4 上训练 1600 万个 latent；Google DeepMind 发布 [Gemma Scope](https://arxiv.org/abs/2408.05147)，400 多个开放 SAE。
- **2025-03** Anthropic 发布 [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) 和 [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)，在 Claude 3.5 Haiku 上画归因图；同月，Google DeepMind 可解释性团队发文说 SAE 在一个下游任务上输给线性探针，暂缓基础 SAE 研究（[Alignment Forum，2025-03-26](https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks)）。
- **2025-05 至 07** Anthropic [开源归因图工具](https://www.anthropic.com/research/open-source-circuit-tracing)（示例模型为 Gemma-2-2b 和 Llama-3.2-1b，并接入 Neuronpedia），之后用 [QK 归因](https://transformer-circuits.pub/2025/attention-qk/index.html)补上“注意力为什么看这里”。
- **2025-09 至 12** [Claude Sonnet 4.5 系统卡](https://assets.anthropic.com/m/12f214efcc2f457a/original/Claude-Sonnet-4-5-System-Card.pdf)在部署前评估里抑制“评估意识”相关的方向；OpenAI 发表[权重稀疏 Transformer](https://arxiv.org/abs/2511.13653)（后收录于 ICML 2026）；Google DeepMind 团队提出[务实的可解释性](https://www.alignmentforum.org/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability)（12-01），并发布 [Gemma Scope 2](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/)（12-19）。
- **2026-01 至 07** Anthropic 依次发表 [assistant axis](https://www.anthropic.com/research/assistant-axis)（01-19）、[情绪概念](https://transformer-circuits.pub/2026/emotions/index.html)（04-02）、[自然语言自编码器 NLA](https://transformer-circuits.pub/2026/nla/index.html)（05-07）和[可口头报告的表示构成全局工作区](https://transformer-circuits.pub/2026/workspace/index.html)（07-06）。最后一篇借用了神经科学的全局工作区理论来组织证据，作者说明不对主观体验表态。

## 截至 2026 年 10 月

- **可解释性进了实验室的部署前审计，但证据多是实验室自报的案例。** Sonnet 4.5 系统卡（Anthropic，2025-09）报告：抑制几个与“这是测试”相关的方向后，在 100 条合成蜜罐提示上，模型口头说出“这是测试”的比例下降，不当行为从 0 条升到最多 8–9%。NLA 论文（Anthropic，2026-05-07）说，在 Claude Opus 4.6 的部署前审计里，NLA 帮助找出了模型心里认为在被测试、却没有说出口的情况。
- **SAE 的地位降了，简单基线的地位升了。** Google DeepMind 团队 2025-03 暂缓基础 SAE 研究，2025-12 主张先试提示、引导、探针这类简单方法，不够用再加复杂度；[AxBench](https://arxiv.org/abs/2501.17148)（2025-01）发现，在引导任务上提示词胜过所有基于表示的方法。2026 年仍有预印本为 SAE 辩护（[Jørgensen & Hansen，2026-05](https://arxiv.org/abs/2605.31183)）。
- **开放工具多了。** Gemma Scope 2 覆盖 Gemma 3 的 270M 到 27B 全部尺寸，Google DeepMind 自报存储了约 110 PB 数据、训练了超过 1 万亿参数；circuit-tracer 加 Neuronpedia 让外部研究者能在开源模型上画归因图。
- **新的读出方式是“翻译成文字”。** NLA 一类方法把激活解释成自然语言，好读，但作者自己列出了会编造、缺少机制依据、训练和推理成本高等局限。

## 哪些已经比较确定，哪些还在变

每格都对应一条带日期的公开记录；“接下来看”一栏写的是什么样的结果会改变判断，不是预测它会发生。

| 比较确定 | 还在变 | 接下来看什么信号 |
| --- | --- | --- |
| 叠加在玩具模型里真实存在；SAE 在大模型上能找出大量可读、可用于引导的方向（Elhage 等 2022-09；Templeton 等 2024-05） | 这些方向是不是模型“真正在用”的特征：同样数据、只换随机种子，131K 宽的 SAE 只有约 30% 的特征重合（[Paulo & Belrose，2025-01](https://arxiv.org/abs/2501.16615)） | 跨种子、跨架构都稳定的特征；以随机初始化的模型为基线时仍明显领先 |
| 小模型上能把一个电路验证到底：IOI 电路恢复完整模型 87% 的 logit 差，并做了完整性和最小性检验（2022-11） | 大模型上的自动电路发现：Anthropic 说归因图只在大约四分之一的提示词上给出令人满意的解释（2025-03） | 这个比例在后续报告里上升；别的实验室在自己的模型上复现 |
| 简单的线性方向能读出、也能改变一些安全相关的行为：拒答方向在 13 个开源模型上成立（[Arditi 等，2024-06](https://arxiv.org/abs/2406.11717)）；欺骗探针 AUROC 0.96–0.999（[Apollo，2025-02](https://arxiv.org/abs/2502.03407)） | 这些探针在真实部署分布下够不够稳：Apollo 自己说目前还不足以作为可靠防线 | 在不是人为植入的问题上、由第三方复现的检测结果 |
| 在有强基线的下游任务上，SAE 常常不占优（[Kantamneni 等，2025-02](https://arxiv.org/abs/2502.16681)；Google DeepMind 2025-03；AxBench 2025-01） | 改进版 SAE 能否扭转：[SAEBench](https://arxiv.org/abs/2503.09532)（2025-03）发现代理指标和实际用处脱节 | 在事先固定的代理任务上，新的字典方法稳定胜过 difference-in-means 和线性探针 |
| 可解释性工具在部署前审计中找到过具体问题，例如 Opus 4.6 的一批错配训练数据（Anthropic 自报，2026-05） | 不知道答案时能否可靠找出问题：审计游戏里 4 支队伍有 3 支找到隐藏目标，只有 API 权限的那支没找到（[Marks 等，2025-03](https://arxiv.org/abs/2503.10965)） | Amodei 2025-04 提出的目标是 2027 年前“可靠检测出大多数模型问题”（[文章](https://www.darioamodei.com/post/the-urgency-of-interpretability)）；第三方白盒审计的公开结果 |

“比较确定”的意思是：在原论文的设置里有一手结果；不代表换模型、换任务也成立。

## 和本站其他笔记的关系

- 残差流见[残差连接](../../00-foundations/core/residual-connections.md)，attention head 见[多头注意力](../../00-foundations/core/multi-head-attention.md)，MLP 见 [FFN 与门控](../../00-foundations/core/ffn-and-gates.md)。可解释性的大部分说法，都是在说“哪个部件往残差流里写了什么”。
- 用探针检查世界模型，见[它真的懂世界吗](../world-models/do-they-understand.md)。
- 第 3 篇的安全部分会用到 reward hacking，定义见[可验证奖励](../../05-post-training/verifiable-rewards.md)；行为评估这一侧见[评估](../../07-evaluation/README.md)。

## 原始资料从哪读起

- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)：为什么神经元会一词多义。
- [Interpretability in the Wild（IOI）](https://arxiv.org/abs/2211.00593)：一个电路说法需要哪些检验。
- [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255)：修补实验的实操建议。
- [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)：SAE 用在生产模型上，包括作者自己写的局限。
- [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)：归因图的方法和局限一节。
- [Open Problems in Mechanistic Interpretability](https://arxiv.org/abs/2501.16496)（Sharkey 等，2025-01）：多家机构合写的开放问题清单。
- [A Pragmatic Vision for Interpretability](https://www.alignmentforum.org/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability)（2025-12）：另一种看法，主张从具体任务和简单基线出发。

## 这一组怎么读

1. [特征、叠加与线性探针](features-and-superposition.md)：线性表示假说、探针准确率说明什么、手算 3 个特征挤进 2 维、激活引导
2. [稀疏自编码器（SAE）](sparse-autoencoders.md)：重建加稀疏的目标、手算两个 λ、死 latent 和特征分裂、自动贴标签、规模化数字和 2025–2026 的批评
3. [电路、归因与因果干预](circuits.md)：激活修补和路径修补、手算 3 个节点、IOI 电路、归因图，以及和安全评估怎么接上
4. [复习题](review.md)：面试题和自检
