# 语言模型目标、训练与生成

**中文** · [English](language-model-objective.en.md)

> 阅读时间：约 9 分钟 · 最近审阅：2026-10

一条答案有 20 个有效 token，另一条有 200 个。如果 loss 按所有 token 平均，后者就贡献十倍的位置数；如果先分别求平均再按样本平均，两条答案权重一样。**还没有换模型，“平均”怎么做，就已经改变了模型主要向谁学习。**[下面算一遍](#cross-entropy)。

<span id="_1"></span>

## 训练和生成为什么节奏不同？ {#_2}

训练与生成使用同一个参数化分布：

$$p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t})$$

区别只在“未来知不知道”。训练时整条真实序列已经摊在桌上；生成时下一个 token 还没出现，模型只能写一个、看一个，再继续写。下表“主要瓶颈”一行为什么这样分，可以用 [roofline](../../12-infra/performance.md#roofline) 算出来。

| | 训练 / prefill | autoregressive decode |
| --- | --- | --- |
| 输入 | 一整段已知 token | 最新生成的一个或少量 token |
| attention | causal mask 下并行 | 查询历史 KV cache |
| 主要瓶颈 | compute、activation memory | memory bandwidth、cache、串行步数 |
| 误差来源 | 数据与目标 | 还多了 sampling 与错误累积 |

## Cross-entropy 没有替你决定“什么重要” {#cross-entropy}

单个位置的 loss：

$$\ell_t=-\log p_\theta(y_t\mid x_{\le t})$$

这里 $y_t=x_{t+1}$ 是位置 $t$ 之后的真实 token。对 logits 的梯度是 $p-e_{y_t}$（$e_{y_t}$ 是 $y_t$ 的 one-hot 向量），公式很干净。但把所有位置加起来以后，频繁 token 和长样本天然拥有更多“投票权”。所以数据混合、sample weighting 与 loss masking 不是训练脚本边角料——它们就在定义模型到底该重视谁。

回到开头的两条答案。设短答案（20 个 token）每个 token 的平均 loss 是 1.0，长答案（200 个 token）是 2.0：

| 怎么平均 | 算式 | 结果 |
| --- | --- | ---: |
| 所有 token 一起平均 | $(20\times1.0+200\times2.0)/220=420/220$ | 约 1.91 |
| 每条先平均，再按样本平均 | $(1.0+2.0)/2$ | 1.5 |

按 token 平均时，长答案占 $200/220\approx91\%$ 的权重，结果几乎就是它自己的 2.0；按样本平均时，两条各占一半。哪种合适，要看你想让模型更重视什么，公式不会替你选。梯度累积时，几个 microbatch 之间也有同样的问题，见[一次训练](training-step.md#token-mean)。

Perplexity 是平均 token negative log-likelihood 的指数：

$$\text{PPL}=\exp\!\left(\frac{1}{N}\sum_t \ell_t\right)$$

先代两个数。每个 token 的平均 NLL 是 2 nats 时，PPL 是 $e^2\approx7.39$，大致像每一步都在 7.39 个同样可能的 token 里挑。模型若对词表里 $V$ 个 token 一律给 $1/V$，每步 NLL 都是 $\ln V$，PPL 正好等于 $V$。

不同 tokenizer 的 token 单位不同，perplexity 不能直接横向比较。看一个教学用的例子（不是实测）：同一段 100 bytes 的文本，两个模型给整段的总 NLL 都是 100 nats，只是 tokenizer 切法不同：

| tokenizer 切成 | 每 token NLL | PPL | bits per byte（bpb） |
| --- | ---: | ---: | ---: |
| 25 个 token | 4 nats | $e^4\approx54.6$ | $100/(100\ln2)\approx1.44$ |
| 50 个 token | 2 nats | $e^2\approx7.39$ | $100/(100\ln2)\approx1.44$ |

两个模型给整段文本的概率一样，PPL 却差了 7 倍多：切得越碎，每个 token 越好猜。把总 NLL 换成 bits，再除以 byte 数，单位就固定在同一段 bytes 上；[GPT-2 报告 §3.1](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) 也按字符、byte 或词这类固定单位报告结果。bpb 也有边界：按 [UTF-8 的编码规则](https://www.rfc-editor.org/rfc/rfc3629#section-3)，ASCII 字符占 1 byte，大多数中日韩汉字占 3 bytes。同样的内容换一种语言写，byte 数就变了，跨语言比较 bpb 仍要小心。

## 训练用真实前缀，生成靠自己的输出 {#_3}

训练时模型总在真实前缀上预测（teacher forcing）；生成时它必须在自己的输出上继续。这种错位常被称为 exposure bias，例子见 [Seq2Seq](../core/seq2seq.md)。某一步的小概率错误可能把后续上下文带到训练数据很少覆盖的区域。

这里很容易条件反射地说“那就上 RL”。先别急，检查问题到底来自：

1. 训练数据没覆盖目标行为；
2. loss mask 或模板错误；
3. sampling 策略不合适（区分 1 和 3：让模型给参考答案打分；参考答案的 log-likelihood 很低，多半是训练没覆盖；概率不低、采样结果却差，再查 temperature、top-p 等设置）；
4. 长程目标无法由逐 token likelihood 表达。

第 4 类最直接指向 sequence-level preference 或 RL objective；如果问题正是模型自己的输出把上下文带离训练分布，在模型自己生成的前缀上训练（on-policy 数据、RL 等）也值得考虑。

## 缓存必须和完整前向等价 {#cache}

增量解码必须与一次性前向在数值容差内一致。位置、mask 或 K/V 顺序稍有错误，模型仍可能写出通顺的句子，logits 却已经变了；所以要逐步比较 logits，不能只看文本像不像。逐项排查清单见 [KV cache 一篇](kv-cache-and-inference.md#cache-checks)，[`../code/test_model.py`](../code/test_model.py) 有对应的等价性测试。

<span id="_4"></span><span id="factorization"></span>

<span id="self-check"></span>

## Post-Training 到底改变了什么 {#post-training}

SFT、preference learning 与 RL 并没有换一种模型。它们在同一个自回归分布上改变：训练样本来自哪里、哪些 token 计入 loss、不同输出如何被比较、回报如何分配给整条 trajectory。

继续读 [Post-Training](../../05-post-training/)。
