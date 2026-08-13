# 语言模型目标、训练与生成

**中文** · [English](language-model-objective.en.md)

> 阅读时间：约 10 分钟 · 难度：进阶 · 最近审阅：2026-08

## 同一个模型，两条执行路径

训练与生成使用同一个参数化分布：

$$p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t})$$

区别在于训练时整条真实序列已知，生成时未来 token 尚不存在。

| | 训练 / prefill | autoregressive decode |
| --- | --- | --- |
| 输入 | 一整段已知 token | 最新生成的一个或少量 token |
| attention | causal mask 下并行 | 查询历史 KV cache |
| 主要瓶颈 | compute、activation memory | memory bandwidth、cache、串行步数 |
| 误差来源 | 数据与目标 | 还多了 sampling 与错误累积 |

## Cross-entropy 到底优化什么

单个位置的 loss：

$$\ell_t=-\log p_\theta(y_t\mid x_{\le t})$$

对 logits 的梯度仍是 $p-y$。频繁 token 和长样本天然贡献更多训练位置，因此数据混合、sample weighting 与 loss masking 会直接改变模型学到的行为。

Perplexity 是平均 token negative log-likelihood 的指数：

$$\text{PPL}=\exp\!\left(\frac{1}{N}\sum_t \ell_t\right)$$

不同 tokenizer 的 token 单位不同，perplexity 不能直接横向比较。

## Teacher forcing 与 exposure gap

训练时模型总在真实前缀上预测；生成时它必须在自己的输出上继续。某一步的小概率错误可能把后续上下文带到训练数据很少覆盖的区域。

这不意味着必须用 RL。先判断问题来自：

1. 训练数据没覆盖目标行为；
2. loss mask 或模板错误；
3. sampling 策略不合适；
4. 长程目标无法由逐 token likelihood 表达。

只有第 4 类才真正指向 sequence-level preference 或 RL objective。

## KV cache 的不变量

增量解码必须与一次性前向在数值容差内等价。需要同时保证：

- 新 query 使用正确的绝对位置；
- cache 中 K/V 的 token 顺序不变；
- causal mask 允许 query 看见全部过去和自己；
- 每层读到同一个 cache position；
- batch reorder 后 cache 也同步 reorder。

[`../code/test_model.py`](../code/test_model.py) 把 full forward 与 token-by-token decode 对齐，是比“生成看起来正常”更强的正确性证据。

## 从基础接到 Post-Training

SFT、preference learning 与 RL 并不是另一套模型学。它们在同一个自回归分布上改变：训练样本来自哪里、哪些 token 计入 loss、不同输出如何被比较、回报如何分配给整条 trajectory。

继续读 [Post-Training](../../05-post-training/)。
