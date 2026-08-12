# 基础：模型到底在算什么？

**中文** · [English](README.en.md)

## 先用一句话讲清楚

从线性回归到 Transformer，**最后一步从来没变过**——永远是一个线性分类器。变的只是它前面那个「坐标变换」有多复杂。

## 用做菜来理解

想象你要从一堆食材里挑出「能做甜品的」。

- **线性回归 / 逻辑回归**：你只能按一条直线切——比如「含糖量 > 10 就是甜品原料」。盐和糖分得开，但「番茄能做甜品也能做菜」这种情况你切不出来。
- **手工特征工程**：你自己想出新指标，比如「含糖量 × 酸度」，在这个新指标上再切一刀。有用，但你得先猜到该造哪个指标。
- **神经网络**：你不再猜。网络自己学出该看哪些指标，然后在那个学出来的坐标系上，还是切一刀直线。
- **Transformer**：同样是切一刀，只不过前面那套「学出该看什么」的机制，变成了几十层注意力。

所以整个深度学习的故事可以压缩成一句话：**把特征工程交给梯度下降**。

## 这一章有什么

### [从线性模型到神经网络](from-linear-to-neural.md)

为什么逻辑回归的边界仍然是直线？sigmoid 到底解决了什么问题（不是表达力，是梯度）？加了交互项算不算非线性？隐藏层在几何上做了什么？

配一份能跑的 XOR 对照实验：同样的数据，逻辑回归 50%，33 参数的「深度线性网络」还是 50%，9 参数的单隐藏层 98%。

### [Transformer 架构](transformer.md)

2017 年的 encoder-decoder 原版，和今天的 decoder-only 差在哪。三处注意力分别在干什么，post-norm 为什么必须配 warmup，位置编码从正弦走到 RoPE，KV cache 是怎么回事。

### [`code/`](code/)：手搓一遍

不调 `nn.MultiheadAttention`，不调 `F.scaled_dot_product_attention`，只用 `nn.Linear` 和裸张量运算写两版 Transformer，附带能证明它是对的验证脚本（因果性、KV cache 等价性、RoPE 相对性）。

跑起来是这样的——学会「把序列反转」之后，cross-attention 自己长出了反对角线：

```
        1  2  3  4  5  6  7  8   <- source (encoder)
  BOS                        @
    1                     @
    2                  @
    3               @
    4            #  :
    5         @
    6      @
    7   @
  ^ decoder step
```

## 为什么要从这里开始

后面几章讲的是**系统**：数据怎么进来、记忆怎么组织、检索找什么、怎么评估。那些讨论都建立在一个前提上——你知道模型内部那个「表示」是什么东西。

比如：

- [记忆](../02-memory/)里说「重要信息被压扁成一个平均值」，压扁的就是这里的 $h$。
- [Post-Training](../05-post-training/)改的是这个坐标变换，不是最后那个线性层。
- [Evaluation](../07-evaluation/)之所以不能只看一个总分，部分原因是同一个 $h$ 在不同 slice 上的可分性完全不同。

## 从哪里继续读

- [从线性模型到神经网络](from-linear-to-neural.md)
- [Transformer 架构](transformer.md)
- 读完可以接 [Post-Training](../05-post-training/)：这些参数后来是怎么被继续改的。

## 起始论文

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原版 Transformer
- [Layer Normalization](https://arxiv.org/abs/1607.06450)
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — pre-norm 为什么能去掉 warmup
- [RoFormer](https://arxiv.org/abs/2104.09864) — RoPE
- [GQA](https://arxiv.org/abs/2305.13245) — 分组查询注意力
