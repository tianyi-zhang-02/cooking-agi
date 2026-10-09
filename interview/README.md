# 面试准备

**中文** · [English](README.en.md)

看过不一定讲得清，讲得清也不一定写得出来。这里把技术准备分成 3 条线：ML / LLM 基础问答、ML Coding、Python 与 LeetCode。可以只练其中一项，不必先刷完一整份清单。

内容来自公开知识和自拟练习，不收集公司的非公开面试题。平时想检查自己学得怎么样，也可以来这里练。

## 先选你想练的部分 {#prep-routes}

<nav class="study-route prep-route" aria-label="面试复习路线">
<a href="basics/README.md"><small>01 / 讲清原理</small><strong>ML / LLM 基础问答</strong><span>概率、优化、模型机制与取舍</span></a>
<a href="../learn/ml-exercises/README.md"><small>02 / 写出实现</small><strong>ML Coding</strong><span>PyTorch、训练循环、Attention 与 loss</span></a>
<a href="leetcode.md"><small>03 / 解决问题</small><strong>Python 与 LeetCode</strong><span>容器、复杂度与可迁移的算法方法</span></a>
</nav>

| 你现在卡在哪里 | 先做什么 | 做完检查什么 |
| --- | --- | --- |
| 名词都见过，追问就说不清 | [基础问答](basics/README.md)：选一个概念，用小例子解释 | 能否说清前提；换个条件，结论还成立吗 |
| 公式会写，张量 shape 总不对 | [ML Coding](../learn/ml-exercises/README.md)：从一个小实现开始 | 输出、梯度、mask、数值稳定性 |
| 算法题能看懂答案，自己想不到 | [LeetCode 方法](leetcode.md)：按解题方法练，不按题号背 | 为什么适用，时间和空间成本是多少 |

## ML 问答与手写

这两项可以一起练，但先分清自己缺什么。比如 attention：解释为什么要缩放点积，属于[基础问答](basics/README.md)；写出 causal mask、核对 softmax 的维度，属于 [ML Coding](../learn/ml-exercises/README.md)。实现中不明白的步骤，再回[基础与原理](../learn/README.md)细读。

## Python 与算法

语法不熟，先开 [Python 常用写法](python.md)，边读边跑。算法从复杂度、Hash Map、双指针、二分、DFS / BFS 开始，再到回溯和 DP，重点是能迁移的 Easy / Medium 方法，不以刷完 Hard 为目标。

例如练 [DFS / BFS](algorithms/traversal.md) 时，别只记模板：分别用栈和队列走一个小图，看看 visited 在什么时候更新，哪一种才能保证无权图的最短路。

## 系统设计

系统设计放在[工程实践](../practice/README.md)，和项目拆解一起看。准备这类面试，可以直接做 [Feed、RAG、长期记忆设计题](../learn/system-design/README.md)：先问需求，再比较方案，最后解释失败时怎么办。它不只是画架构图，也不是多报几个框架名字。

## 怎么复习

1. 先自己讲或写，卡住了再读对应段落。
2. 用一个小输入验证；把假设、边界条件和复杂度写在旁边。
3. 换一个条件再试。比如加入 padding，或者数据不能一次装进内存，原来的方法还行吗？

第一次学一个主题，不用在问答页硬撑，回[基础与原理](../learn/README.md)按顺序读更省力。找工作的经历、心态和准备节奏在[求职](../career/README.md)，不和技术练习混在一起。
