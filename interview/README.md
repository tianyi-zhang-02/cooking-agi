# 技术面：怎么用这一组笔记

**中文** · [English](README.en.md)

> 阅读时间：约 2 分钟 · 最近审阅：2026-09

> **先读这个**：面试的形式和重点变得很快，这里的内容有时效性；这里只整理**公开的基础知识**，不写任何一家公司的面试题或流程。

技术面问的东西看起来五花八门，但大半在反复问同一件事：你是不是真的理解了机制，能不能从公式写到代码，再讲清楚为什么。这一组笔记就是按这个顺序排的。

## 算法题

- [LeetCode 怎么刷](leetcode.md) —— 刷什么、我的方法、题型清单，以及刷到什么程度就够

## 系统设计

- [系统设计](system-design.md) —— 没有固定公式，关键是 trade-off；公开资源，加一个推荐系统的例子

## 考前快速过一遍

这一组和“学习”那边讲的是同一批知识，呈现方式不一样：那边讲透、能动手玩；这边只留题和答案，给考前用。重复是故意的。

- [考前速查：按知识板块过一遍考题](questions.md) —— 自动从站内笔记里抽出来的面试题和自检题，点开就是答案，可以随机抽题
- [Transformer 进阶速查](transformer-followups.md) —— KV cache、MQA / GQA / MLA、RoPE、MoE、FlashAttention 的追问，一问一答

## 建议的顺序

1. [面试基础题：大半在问同一件事](../00-foundations/interview-basics.md) —— 先把主线立起来
2. [白板上会让你手写的那七个](../00-foundations/hand-write-kit.md) —— 检查自己能不能不看资料写出来
3. [ML 数学面试主线：从概率到估计量](../00-foundations/ml-math-interview.md) —— 补上概率与统计这条线
4. [Transformer 交互图解](../00-foundations/transformer-lab.md) —— KV cache、GQA / MLA、RoPE、MoE、FlashAttention 这类追问，用能拖能点的图自测

## 被追问到细节时

- [多头注意力：从公式到实现](../00-foundations/core/multi-head-attention.md)
- [归一化：BatchNorm、LayerNorm 与 RMSNorm](../00-foundations/core/normalization.md)
- [Decoder-only：自回归生成](../00-foundations/core/decoder-only.md)
- [解码策略：temperature、top-k、top-p](../00-foundations/core/decoding.md)
- [Post-Training 总览](../05-post-training/README.md) —— SFT、偏好学习、RLHF 与之后的算法

想按岗位来安排顺序，见[首页的路线图](../README.md)。
