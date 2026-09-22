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

一个 Transformer block 里有两样东西：attention 和 FFN。FFN 大约占一个 block 三分之二的参数，而且**每个 token 都要把 FFN 的全部参数算一遍**。前向时每个参数大约贡献 2 次浮点运算，所以 dense 模型的参数量和每个 token 的计算量是绑死的：想要更多参数，就得付更多计算。

## MoE 把 FFN 换成 N 个 expert

Mixture-of-Experts（MoE）层把这一个 FFN 换成 $N$ 个形状相同的 expert FFN，再加一个很小的 router。每个 token 只被送到 router 打分最高的 $k$ 个 expert，输出是它们的加权和：

$$y = \sum_{i \in \mathrm{TopK}(x)} g_i(x)\, E_i(x)$$

attention、embedding 和 norm 都不变，仍然是所有 token 共用。

<!-- widget:tx-moe -->

## 总参数和激活参数

总参数随 $N$ 增长，每个 token 的计算只随 $k$ 增长。几个公开模型的官方数字：

| 模型 | 每层 expert | 每个 token 走几个 | 总参数 | 激活参数 |
| --- | --- | --- | --- | --- |
| Mixtral 8x7B | 8 | 2 | 46.7B | 12.9B |
| DeepSeek-V3 | 256 个 routed + 1 个 shared | 8 + 1 | 671B | 37B |
| Qwen3-235B-A22B | 128 | 8 | 235B | 22B |
| gpt-oss-120b | 128 | 4 | 116.8B | 5.1B |

Mixtral 8x7B 不是 56B：被复制成 8 份的只有 FFN，attention 和 embedding 只有一份，所以总参数是 46.7B。

## 稀疏省的是计算，不是显存

任何一个 token 都可能被分到任何一个 expert，所以**全部 expert 都得放在显存里**（或者分片放在多张卡上）。显存看的是总参数，每个 token 的计算看的是激活参数。这也是 MoE 在部署时的主要代价，后面单独讲。

## 这一组怎么读

1. [Router 怎样选 expert](router.md)：打分、top-k、归一化，以及为什么早期要加噪声（可以动手点）
2. [负载均衡](load-balancing.md)：不管它就会塌缩；辅助 loss、capacity 和只调 bias 的办法（可以动手跑）
3. [细粒度专家与共享专家](fine-grained-and-shared.md)：DeepSeekMoE 的两个改动，以及各家怎么选
4. [训练和推理的系统代价](systems.md)：expert parallelism、all-to-all、显存和解码
5. [复习题](review.md)：面试题和自检
