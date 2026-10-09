# Engram：把常见局部模式交给查表

**中文** · [English](engram.en.md)

读到一个常见短语时，模型既要识别这串词，也要判断它在这句话里是什么意思。Engram 把前一部分做成可学习的查表模块，再用上下文决定查到的信息值不值得用。它不是替用户保存聊天记录，也不是把 RAG 改了个名字。

## 一次 lookup 经过哪些步骤？

```text
最近几个 token ─→ 规范化 ID ─→ 多组 n-gram hash ─→ embedding 表
                                                       ↓ 拼接
当前 hidden state ─→ 与查表 key 匹配 ─→ gate × value ─→ 局部卷积 ─→ residual
```

先对 token ID 做规范化映射，让某些书写变体共享查表空间。这条分支不意味着主模型的 tokenizer 输入全部被替换。然后取不同长度的后缀 n-gram，用多组 hash 找到表中条目，拼接成向量 $e_t$。表中的向量通过训练学习，不是人工写好的答案。

## 有上下文，为什么还要查表？

可以把它理解成一种计算分工：重复出现的局部模式使用便宜、确定的地址读取；需要长距离关系与消歧的部分仍交给模型。查到同一个局部模式，也不代表一定采用同样的信息。

例如英文 `the bank` 可能出现在河岸，也可能出现在银行相关文本里。表的局部输入相同，周围的 hidden state 不同。门让两个上下文对同一条目有不同的使用程度。这只是帮助理解消歧的例子，不是论文实验结果。

一种简化的单分支写法是：

$$
k_t=W_Ke_t,\quad v_t=W_Ve_t,\qquad
\alpha_t=\sigma\left(\frac{\operatorname{RMSNorm}(h_t)^\top
\operatorname{RMSNorm}(k_t)}{\sqrt d}\right),\quad u_t=\alpha_tv_t.
$$

随后用 causal depthwise convolution 增加局部交互，并保留 $u_t$ 的通路：

$$
y_t=u_t+\operatorname{SiLU}\big(\operatorname{Conv}_{\rm causal}(\operatorname{RMSNorm}(u))_t\big).
$$

最后注入 residual。这里省去了论文多分支 residual 设置的展开；不能把这个式子直接当作完整 checkpoint 实现。卷积必须因果，否则训练时会偷看后面的 token。

## Hash 冲突是什么样？

假设教学用的 bigram hash 是 $(3a+b)\bmod m$。下面不是作者的 hash 算法，只用于观察冲突。

```python
def bigram_bucket(tokens, table_size):
    if len(tokens) != 2 or type(table_size) is not int or table_size <= 0:
        raise ValueError("Expected two IDs and a positive table size")
    if any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("Expected nonnegative integer token IDs")
    return (3 * tokens[0] + tokens[1]) % table_size

assert bigram_bucket([1, 2], 5) == bigram_bucket([2, 4], 5) == 0
assert bigram_bucket([1, 2], 7) == 5
assert bigram_bucket([2, 4], 7) == 3
```

一个表里撞上了，另一个表里可能没撞上；拼接多组读取结果，能减少单次冲突造成的混淆，但不保证完全无冲突。训练更新共享条目时，也会影响映射到该条目的其他 n-gram。

## 省的是计算，不是所有资源

下面只算一个假想表：100 万个条目，每个 64 维，BF16 每维 2 bytes。

| 项目 | 计算 | 结果 |
| --- | --- | --- |
| 整张表的参数 | $10^6\times64\times2$ | 128 MB，约 122 MiB |
| 每 token 读 2 个条目 | $2\times64\times2$ | 256 bytes |
| 每秒 10 万 token 的原始读取量 | $10^5\times256$ | 25.6 MB/s |

这不是实际部署成本：还要加多层、多种 n-gram、更多 hash heads、地址处理、传输与训练状态。随机读取的有效带宽也不等于顺序内存带宽。

确定性地址带来的机会是提前算好并预取条目；风险是 CPU/GPU 传输或随机读成为瓶颈。参数量很大但每步只读少量参数，与“免费增加知识”是两回事。

## 怎么验证确实值得加？

先保证 same prefix 的读取与输出不受未来 token 改动影响。再检查规范化、哈希版本、词表与表权重是否随 checkpoint 一起保存；只存主模型权重不够。

模型实验应比较相同训练预算下的基线与加表版本，区分常见短语、低频组合、上下文消歧和长距离任务。还可以屏蔽查表分支观察回退幅度，或记录 gate 是否饱和。不要只展示被记住的短语案例。

| 常见误解 | 更准确的说法 |
| --- | --- |
| 会自动记住新用户 | 表是训练得到的参数，不是每用户可写数据库 |
| 能给出原文出处 | embedding 不等于可追溯文档，需要额外检索与引用系统 |
| gate 能保证忽略错误条目 | gate 也会学错，仍需消歧和冲突测试 |
| 替代 attention | 它补局部模式记忆，不负责全部长距离关系 |

核对日期：2026-10-08。依据 [Engram 论文](https://arxiv.org/abs/2601.07372) 与[作者仓库](https://github.com/deepseek-ai/Engram)。本页测试的是 hash 和资源算例，未进行完整模型训练。
