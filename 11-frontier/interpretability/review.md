# 机制可解释性：复习题

**中文** · [English](review.en.md)

> 阅读时间：约 5 分钟 · 最近审阅：2026-10

这一页用面试题和自检把整个系列串一遍，每个答案后面都指向可以回查的那一篇。

## 面试常见问题

<details class="interview" markdown="1">
<summary>机制可解释性和行为评估有什么区别？什么时候值得用前者？</summary>

行为评估问“在这些输入上输出对不对”，可解释性问“内部哪个中间量产生了这个输出，改掉它输出会不会按预期变”。同样答对 “capital of the state containing Dallas”，模型可能先想到 Texas 再走到 Austin，也可能直接记住了搭配；Claude 3.5 Haiku 里两条路都有。

值得用的场合：同分不同机制会影响分布外表现；模型可能在知道被测试时改变行为（Sonnet 4.5 系统卡）；需要在没有标签的情况下找假设。代价是慢、贵、覆盖少，而且常常解释的是替代模型。见[导读](README.md)。

</details>

<details class="interview" markdown="1">
<summary>线性探针在某一层读出某个属性，准确率 95%。这说明什么？怎样把结论做实？</summary>

只说明信息可以从这一层线性读出。它不说明模型的输出用了这个信息，也不排除准确率来自探针自己的拟合能力。Hewitt & Liang 的控制任务给每个词型随机指定固定标签：ELMo 第一层词性标注上，线性探针 97.2 对控制任务 71.2，选择性 26.0；MLP 探针 97.3 对 92.8，选择性只有 4.5。

做实的办法：报告选择性；用随机方向或随机初始化模型做基线（Anthropic 的 sleeper agent 探针 AUROC 99.3%，1000 个随机方向里最好的也有 95.3%）；沿探针方向干预，看输出是否按预期变；注意子空间修补可能制造假象（Makelov 等）。见[特征与探针](features-and-superposition.md)。

</details>

<details class="interview" markdown="1">
<summary>什么是叠加？为什么特征越稀疏越划算？</summary>

叠加是网络表示的特征比维度多，让特征方向彼此不完全正交、容忍一些干扰。Elhage 等的玩具模型是 $x'=\mathrm{ReLU}(W^\top Wx+b)$：特征稠密时只学最重要的 $m$ 个，变稀疏后出现叠加，而且是相变。

手算：3 个特征相隔 120° 放进 2 维，两两干扰 −0.5。只有一个特征激活时，负干扰被 ReLU 截掉，还原无误；两个同时激活各丢一半；三个全激活完全抵消。每个特征以概率 $p$ 激活时，期望损失 $1.5p^2+1.5p^3$，对比放弃一个特征的 $p$：$p=0.1$ 时 0.0165 对 0.1，平衡点约 0.457。去掉 ReLU，叠加就不划算了。见[特征与探针](features-and-superposition.md)。

</details>

<details class="interview" markdown="1">
<summary>SAE 的目标函数是什么？L1、TopK、JumpReLU 各怎么处理稀疏？</summary>

$f=\mathrm{ReLU}(W_{\rm enc}x+b)$，$\hat x=W_{\rm dec}f+b'$，损失是重建误差加 $\lambda\sum_i f_i\|W_{\rm dec}^{(i)}\|$；乘解码列范数是为了防止缩小 $f_i$、放大解码向量来躲惩罚。L1 带来收缩：非零激活被系统性低估。TopK（OpenAI）每个 token 只留最大的 $k$ 个，直接定稀疏度，损失只剩重建项；JumpReLU（Google DeepMind）给每个 latent 学阈值，直接罚 L0，用直通估计器训练。

手算：$x=(1,1)$，字典里有 $(1,0)$、$(0,1)$、$(0.707,0.707)$。L1 偏爱只用斜方向（代价 $\sqrt2$ 对 2）；$\lambda=1$ 时系数 0.914，相对平方重建误差为 $0.25/2=12.5\%$，其补数为 87.5%，不是重建向量的能量保留率；$\lambda\ge2\sqrt2$ 时什么都不激活。TopK 取 $k=1$ 则系数 $\sqrt2$，没有收缩。见[稀疏自编码器](sparse-autoencoders.md)。

</details>

<details class="interview" markdown="1">
<summary>死 latent、特征分裂、吸收各是什么？换随机种子会怎样？</summary>

死 latent：整个训练中都不激活，拿不到梯度。Anthropic 的 1M、4M、34M SAE 约有 2%、35%、65% 是死的；OpenAI 用编码器按解码器转置初始化加 AuxK 辅助损失，1600 万宽时只有 7% 是死的（不加缓解可达 90%）。特征分裂：字典变大，一个特征分成几个更细的，如 “San Francisco” 从 1 个到 2 个到 11 个。吸收：特征有层级时，父特征在该激活的地方不激活，被子特征吸收，改大小或稀疏度解决不了。

换种子：Llama 3 8B 一个 MLP 上 131K 宽的 SAE 只有约 30% 特征共同，TopK 更依赖种子。所以 SAE 给出的是一种实用的分解，不是模型特征的完整清单。见[稀疏自编码器](sparse-autoencoders.md)。

</details>

<details class="interview" markdown="1">
<summary>为什么 Google DeepMind 团队在 2025 年暂缓基础 SAE 研究？SAE 现在还适合做什么？</summary>

2025-03-26 的进展报告：在 Gemma 2 9B IT 上识别有害意图并测分布外泛化，残差流线性探针分布外 AUROC 0.999，SAE 探针明显更差。同期 Kantamneni 等在稀疏探测的四种困难设置下没能稳定胜过基线；AxBench 发现引导上提示词最好、检测上均值差最好，SAE 都没有竞争力；SAEBench 发现代理指标和实际用处脱节。2025-12 团队进一步主张先试提示、引导、探针。

SAE 还适合“不知道要找什么”的探索：审计游戏里一人 90 分钟靠 SAE 特征找到隐藏目标（论文也说语义搜索能复现）；OpenAI 用 SAE 比较微调前后的模型找到 “toxic persona” 特征；找训练数据里的伪相关。已知概念、有标签时，先用探针或均值差。见[稀疏自编码器](sparse-autoencoders.md)。

</details>

<details class="interview" markdown="1">
<summary>激活修补的 denoising 和 noising 有什么区别？IOI 为什么需要“完整性”标准？</summary>

denoising 在破坏运行里换进干净值，问“它够不够把行为救回来”；noising 在干净运行里换进破坏值，问“没有它行不行”。有备份部件时两者会不一致：3 节点算例里 B 的 denoising 恢复 86%，noising 只掉 43%，因为 C 在 B 缺席时顶上。路径效果也不可加，两条路径合计 143%。

IOI 电路恢复完整模型 87% 的 logit 差（忠实度），但忠实度看不到备份：拿掉 name mover 后 backup name mover 会顶上。完整性要求从电路和从完整模型里拿掉同一组部件，效果相近；算例里拿掉 B，模型剩 2.0、电路剩 0，暴露出漏掉了 C。见[电路](circuits.md)。

</details>

<details class="interview" markdown="1">
<summary>归因图是怎么做出来的？听到“可解释性能检测欺骗或 reward hacking”，你会怎么追问？</summary>

先训练跨层 transcoder 替换所有 MLP；对单个提示词冻结注意力模式和归一化，加误差节点让局部替代模型与原模型输出一致；边权 $A_{s\to t}=a_s w_{s\to t}$；剪枝、归组，再回原模型干预验证。局限由作者自己列出：不解释注意力为什么看某处（后来有 QK 归因）、误差节点遮住关键计算、只看单个提示词、替代模型机制未必相同；大约四分之一的提示词上令人满意。

追问：是人为植入的问题，还是自然出现的？谁报告的，有没有第三方复现？和什么基线比（随机方向、思维链监控、行为评估）？误报率多少？例如 Apollo 的欺骗探针 AUROC 0.96–0.999，但作者说还不足以作为可靠防线；Sonnet 4.5 抑制评估意识后不当行为最多 8–9%，是实验室自报且随数据集变化。见[电路](circuits.md)。

</details>

## 自检

<div class="taste-check">
  <strong>可以用下面的问题检查理解：</strong>
  <ol>
    <li>三个特征相隔 120° 放进 2 维、偏置为 0、带 ReLU：特征 1 和 2 同时激活时还原是什么？$p=0.1$ 时叠加和“放弃一个特征”的期望损失各是多少？</li>
    <li>线性探针 97.2 对控制任务 71.2，MLP 探针 97.3 对 92.8：选择性各是多少？为什么准确率几乎一样，结论却不同？</li>
    <li>字典 $(1,0)$、$(0,1)$、$(0.707,0.707)$，激活 $(1,1)$，$\lambda=1$ 时 L1 SAE 的最优系数和重建是什么？$\lambda$ 多大时什么都不激活？TopK 取 $k=1$ 会怎样？</li>
    <li>3 节点算例里，B 的 denoising 恢复 86%、noising 只掉 43%，为什么？两条路径的效果为什么加起来是 143%？</li>
    <li>一个归因图说“模型先想到 Texas 再说 Austin”，你还要看到哪些证据，才相信这是原模型而不只是替代模型的机制？</li>
  </ol>
</div>

## 参考论文

- [Zoom In: An Introduction to Circuits](https://distill.pub/2020/circuits/zoom-in/)：特征与电路的说法从哪来
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)：残差流的读写视角
- [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)：induction head 与训练中的相变
- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)：叠加的玩具模型
- [Designing and Interpreting Probes with Control Tasks](https://arxiv.org/abs/1909.03368)：控制任务与选择性
- [The Linear Representation Hypothesis and the Geometry of LLMs](https://arxiv.org/abs/2311.03658)：线性表示的两种形式化
- [Not All Language Model Features Are One-Dimensionally Linear](https://arxiv.org/abs/2405.14860)：圆形的多维特征（反例）
- [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)：均值差与方向消除
- [Inference-Time Intervention](https://arxiv.org/abs/2306.03341)：激活引导提高 TruthfulQA，及其代价
- [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) 与 [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)：Anthropic 的 SAE 两篇
- [Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093)：TopK SAE 与 GPT-4 上的 1600 万 latent
- [Gemma Scope](https://arxiv.org/abs/2408.05147) 与 [Gemma Scope 2 博客](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/)：开放的 SAE 和 transcoder
- [A is for Absorption](https://arxiv.org/abs/2409.14507) 与 [SAEs Trained on the Same Data Learn Different Features](https://arxiv.org/abs/2501.16615)：吸收与种子依赖
- [Are Sparse Autoencoders Useful?](https://arxiv.org/abs/2502.16681)、[AxBench](https://arxiv.org/abs/2501.17148)、[SAEBench](https://arxiv.org/abs/2503.09532)：下游任务上的比较
- [Negative Results for SAEs On Downstream Tasks](https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks)：Google DeepMind 团队的进展报告
- [Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers](https://arxiv.org/abs/2501.17727)：自动解释需要随机基线
- [Locating and Editing Factual Associations in GPT（ROME）](https://arxiv.org/abs/2202.05262)：causal tracing
- [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255)：修补实验的实操建议
- [Interpretability in the Wild（IOI）](https://arxiv.org/abs/2211.00593)：电路及忠实度、完整性、最小性
- [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) 与 [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)：归因图的方法与案例
- [Weight-sparse transformers have interpretable circuits](https://arxiv.org/abs/2511.13653)：从训练上让电路变小
- [Auditing language models for hidden objectives](https://arxiv.org/abs/2503.10965)：盲审游戏
- [Detecting Strategic Deception Using Linear Probes](https://arxiv.org/abs/2502.03407)：欺骗探针
- [Natural Language Autoencoders](https://transformer-circuits.pub/2026/nla/index.html)：把激活翻译成文字，及其在 Opus 4.6 审计中的使用
- [Open Problems in Mechanistic Interpretability](https://arxiv.org/abs/2501.16496)：开放问题清单
