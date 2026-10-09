# ML Coding：PyTorch 与手写实现

<span id="pytorch-ml"></span>

**中文** · [English](README.en.md)

这里有两条线：一条用 PyTorch 把数据、梯度和训练循环接起来；另一条把 attention、loss 和采样写成小函数。可以先跑通，再拆开核对，不必只靠白板记忆。

## 想练哪一块？

| 练什么 | 入口 | 做完检查什么 |
| --- | --- | --- |
| 写对训练代码 | [PyTorch：从张量到训练](../../00-foundations/pytorch/README.md) | 数据有没有别名，梯度是否累积，loss 分母是否正确 |
| 从公式写到代码 | [白板手写](../../00-foundations/hand-write-kit.md) | shape、mask、边界条件和数值稳定性对不对 |
| 排查 shape 和广播 | [张量运算](../../00-foundations/pytorch/operations-and-shapes.md) | 每个维度的含义，广播有没有改变原意 |
| 确认梯度正确 | [自动求导](../../00-foundations/pytorch/autograd.md) | 计算图有没有被切断，梯度是否流向预期参数 |

想练概念表达和数学推导，去 [ML / LLM 基础问答](../../interview/basics/README.md)。这里只专注实现，不把“能讲”和“能写”算成同一件事。

## 一道题练三次

以 attention 为例：第 1 次说清 Q、K、V 怎样参与计算；第 2 次写一个带 causal mask 的小实现；第 3 次把 padding、长序列或 cache 加进去，判断原来的代码哪里不够了。

卡住了就回[基础与模型](../../00-foundations/study-guide.md)补一补，不用硬背整页。做题是为了看看自己哪里还没懂，不是猜某家公司会考什么。

## 自测

<details><summary>代码输出对了，为什么还不算讲清楚？</summary><p>还要解释在哪些输入和假设下成立。小例子可能没碰到空输入、mask 方向、精度或复杂度问题。</p></details>

<details><summary>一道题刚做完，要怎样检查自己不是只记住答案？</summary><p>改一个约束：规模增大、数据有缺失、允许重复或不能看未来。先预测哪些结论会变，再验证。</p></details>
