# MoE：为什么要让模型变稀疏

**中文** · [English](README.en.md)

> 阅读时间：约 2 分钟 · 难度：进阶 · 最近审阅：2026-09

<div class="lesson-recipe">
  <div><span>解决什么问题</span><strong>参数想要更多，每个 token 的计算又不想跟着涨</strong></div>
  <div><span>前置知识</span><strong>Transformer block · FFN · softmax</strong></div>
  <div><span>核心机制</span><strong>把 FFN 换成 N 个 expert，router 只让每个 token 走其中 k 个</strong></div>
  <div><span>常见错误</span><strong>以为 8×7B 就是 56B；以为激活参数少就省显存</strong></div>
</div>

## 一个 dense block 的参数大多在 FFN

一个 Transformer block 就两块：attention 和 FFN。FFN 占了其中大约三分之二的参数，而且**每个 token 都要把这些参数从头到尾算一遍**。前向一次，每个参数折合大约 2 次浮点运算——在 dense 模型里，参数量和每个 token 的计算量就这样绑在了一起：想多加参数，就得多付计算。

## MoE 把 FFN 换成 N 个 expert

MoE 层把这一个 FFN 拆成 $N$ 个形状一样的 expert FFN，前面加一个很小的 router。每个 token 只去打分最高的 $k$ 个 expert，输出是它们的加权和：

$$y = \sum_{i \in \mathrm{TopK}(x)} g_i(x)\, E_i(x)$$

attention、embedding、norm 都不动，还是所有 token 共用一份。

<!-- widget:tx-moe -->

## 总参数和激活参数

总参数跟着 $N$ 涨，每个 token 的计算只跟着 $k$ 涨。几个公开模型的官方数字：

| 模型 | 每层 expert | 每个 token 走几个 | 总参数 | 激活参数 |
| --- | --- | --- | --- | --- |
| Mixtral 8x7B | 8 | 2 | 46.7B | 12.9B |
| DeepSeek-V3 | 256 个 routed + 1 个 shared | 8 + 1 | 671B | 37B |
| Qwen3-235B-A22B | 128 | 8 | 235B | 22B |
| gpt-oss-120b | 128 | 4 | 116.8B | 5.1B |

Mixtral 8x7B 不是 56B：复制成 8 份的只有 FFN，attention 和 embedding 各只有一份，加起来 46.7B。

## 稀疏省的是计算，不是显存

哪个 token 去哪个 expert，事先并不知道，所以**全部 expert 都得待在显存里**（或者分片摊在多张卡上）。显存按总参数算，计算按激活参数算。这是 MoE 部署时最主要的代价，后面单独有一篇讲。

## 这一组怎么读

1. [Router 怎样选 expert](router.md)：打分、top-k、归一化，以及早期为什么要加噪声（图能点）
2. [负载均衡](load-balancing.md)：放着不管就会塌缩；辅助 loss、capacity，以及只调 bias 的办法（图能自己跑）
3. [细粒度专家与共享专家](fine-grained-and-shared.md)：DeepSeekMoE 的两个改动，以及各家怎么选
4. [训练和推理的系统代价](systems.md)：expert parallelism、all-to-all、显存和解码
5. [复习题](review.md)：面试题和自检
