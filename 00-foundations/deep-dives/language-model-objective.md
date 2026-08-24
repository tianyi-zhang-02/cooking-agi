# 语言模型目标、训练与生成

**中文** · [English](language-model-objective.en.md)

> 阅读时间：约 10 分钟 · 难度：进阶 · 最近审阅：2026-08

<div class="lesson-recipe advanced">
  <div><span>这次要拆什么</span><strong>同一个模型为何有训练与生成两条执行路径</strong></div>
  <div><span>需要先会</span><strong>causal LM · cross-entropy · attention mask</strong></div>
  <div><span>真正的主角</span><strong>loss weighting · teacher forcing · KV cache</strong></div>
  <div><span>最后要能判断</span><strong>问题该修数据、采样、SFT，还是 sequence-level objective</strong></div>
</div>

## 问题上桌：一个模型，为什么跑出两种节奏

训练与生成使用同一个参数化分布：

$$p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t})$$

区别只在“未来知不知道”。训练时整条真实序列已经摊在桌上；生成时下一个 token 还没出现，模型只能写一个、看一个，再继续写。

| | 训练 / prefill | autoregressive decode |
| --- | --- | --- |
| 输入 | 一整段已知 token | 最新生成的一个或少量 token |
| attention | causal mask 下并行 | 查询历史 KV cache |
| 主要瓶颈 | compute、activation memory | memory bandwidth、cache、串行步数 |
| 误差来源 | 数据与目标 | 还多了 sampling 与错误累积 |

## 拆解一：Cross-entropy 没有替你决定“什么重要”

单个位置的 loss：

$$\ell_t=-\log p_\theta(y_t\mid x_{\le t})$$

对 logits 的梯度仍是 $p-y$，公式很干净。但把所有位置加起来以后，频繁 token 和长样本天然拥有更多“投票权”。所以数据混合、sample weighting 与 loss masking 不是训练脚本边角料——它们就在定义模型到底该重视谁。

Perplexity 是平均 token negative log-likelihood 的指数：

$$\text{PPL}=\exp\!\left(\frac{1}{N}\sum_t \ell_t\right)$$

不同 tokenizer 的 token 单位不同，perplexity 不能直接横向比较。

## 拆解二：训练时有人递答案，生成时只能自己接着写

训练时模型总在真实前缀上预测；生成时它必须在自己的输出上继续。某一步的小概率错误可能把后续上下文带到训练数据很少覆盖的区域。

这里很容易条件反射地说“那就上 RL”。我会先忍一下，检查问题到底来自：

1. 训练数据没覆盖目标行为；
2. loss mask 或模板错误；
3. sampling 策略不合适；
4. 长程目标无法由逐 token likelihood 表达。

只有第 4 类才真正指向 sequence-level preference 或 RL objective。

## 正确性底线：Cache 不是“能跑就行”，它必须等价

增量解码必须与一次性前向在数值容差内等价。需要同时保证：

- 新 query 使用正确的绝对位置；
- cache 中 K/V 的 token 顺序不变；
- causal mask 允许 query 看见全部过去和自己；
- 每层读到同一个 cache position；
- batch reorder 后 cache 也同步 reorder。

[`../code/test_model.py`](../code/test_model.py) 把 full forward 与 token-by-token decode 对齐，是比“生成看起来正常”更强的正确性证据。

## 接到下一锅：Post-Training 到底改了哪里

SFT、preference learning 与 RL 并不是另一套模型学。它们在同一个自回归分布上改变：训练样本来自哪里、哪些 token 计入 loss、不同输出如何被比较、回报如何分配给整条 trajectory。

继续读 [Post-Training](../../05-post-training/)。

## 自检

<div class="taste-check advanced">
  <strong>遇到生成质量问题时，先问：</strong>
  <ol>
    <li>训练目标没覆盖这个行为，和 sampling 选坏了，证据上怎样区分？</li>
    <li>为什么 perplexity 在不同 tokenizer 之间不能直接横比？</li>
    <li>怎样用 full forward 与 incremental decode 验证 KV cache，而不是只看文本像不像？</li>
  </ol>
</div>
