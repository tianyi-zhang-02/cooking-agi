# MoE：为什么要让模型变稀疏

**中文** · [English](README.en.md)

> 阅读时间：约 3 分钟 · 难度：进阶 · 最近审阅：2026-10-09

假设一层存了 8 个同样大小的 FFN，每个 token 只使用其中 2 个。这能让不同 token 用到不同参数，而不必每次把 8 个都算一遍。不过，8 份参数仍然要存，token 也可能需要跨设备搬运。MoE 的收益和代价，都要从“存了多少”和“这次用了多少”分开算。

## 一个 dense block 的参数大多在 FFN

以模型宽度 d、FFN 中间宽度 4d 的标准 MHA block 为例，忽略 bias 和 norm：attention 的投影约有 4d² 个参数，FFN 约有 8d²，占两者合计的三分之二。换成 GQA 或不同 FFN 宽度，这个比例也会变。Dense FFN 的矩阵通常对每个 token 都执行，每个矩阵权重约对应一次乘加，也就是按常见口径算的 2 FLOPs；扩宽 FFN 就会增加逐 token 计算。

## MoE 把 FFN 换成 N 个 expert

MoE 层把这一个 FFN 拆成 $N$ 个形状一样的 expert FFN，前面加一个很小的 router。每个 token 只去打分最高的 $k$ 个 expert，输出是它们的加权和：

$$y = \sum_{i \in \mathrm{TopK}(x)} g_i(x)\, E_i(x)$$

attention、embedding、norm 都不动，还是所有 token 共用一份。

<!-- widget:tx-moe -->

## 总参数和激活参数

固定 expert 宽度时，总 expert 参数随 N 增长，主要 expert 计算随 k 增长。Router 还要为 N 个专家打分，attention、共享部分与通信也不能漏算。下表是具体公开型号的参数口径，不是整个家族的统一配置：

| 模型 | 每层 expert | 每个 token 走几个 | 总参数 | 激活参数 |
| --- | --- | --- | --- | --- |
| Mixtral 8x7B | 8 | 2 | 46.7B | 12.9B |
| DeepSeek-V3 | 256 个 routed + 1 个 shared | 8 + 1 | 671B | 37B |
| Qwen3-235B-A22B | 128 | 8 | 235B | 22B |
| gpt-oss-120b | 128 | 4 | 116.8B | 5.1B |

Mixtral 8x7B 不是 56B：复制成 8 份的只有 FFN，attention 和 embedding 各只有一份，加起来 46.7B。

## 稀疏省的是计算，不是显存

少算几个 expert 不等于只存几个 expert。权重仍要完整保存在 GPU、CPU 或其他存储中；全 GPU 常驻按总参数分配存储，offload 可以减少 GPU 占用，却增加搬运和等待。与 dense 模型比较时，还要固定精度、目标质量与 batch，不能仅凭“稀疏”判断谁更省显存。

参数来源见[复习页的模型报告](review.md)，系统取舍见[训练和推理的代价](systems.md)。

## 这一组怎么读

1. [Router 怎样选 expert](router.md)：打分、top-k、归一化，以及早期为什么要加噪声（图能点）
2. [负载均衡](load-balancing.md)：为什么可能失衡；辅助 loss、capacity，以及调 bias 的办法（图能自己跑）
3. [细粒度专家与共享专家](fine-grained-and-shared.md)：DeepSeekMoE 的两个改动，以及各家怎么选
4. [训练和推理的系统代价](systems.md)：expert parallelism、all-to-all、显存和解码
5. [复习题](review.md)：面试题和自检
