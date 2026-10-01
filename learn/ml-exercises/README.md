# ML 问答与手写：会说，也能做

**中文** · [English](README.en.md)

看答案觉得懂了，自己动手时却写不出来，这很常见。这里可以先不看题解，讲一个例子、写几行代码，再回来对照。

## 想练哪一块？

| 练什么 | 入口 | 做完检查什么 |
| --- | --- | --- |
| 把基础讲清楚 | [基础问答](../../00-foundations/interview-basics.md) | 有没有说清假设，而不只是报术语 |
| 从公式写到代码 | [白板手写](../../00-foundations/hand-write-kit.md) | shape、mask、边界条件和数值稳定性对不对 |
| 算清概率与估计 | [ML 数学](../../00-foundations/ml-math-interview.md) | 分母是什么，估计量依赖什么样本 |
| 解释架构取舍 | [Transformer 追问](../../interview/transformer-followups.md) | 改了哪个瓶颈，又多付了什么代价 |
| 随机抽一题复习 | [站内问答集](../../interview/questions.md) | 换一个条件后，还能不能推下去 |

## 一道题练三次

以 attention 为例：第 1 次说清 Q、K、V 怎样参与计算；第 2 次写一个带 causal mask 的小实现；第 3 次把 padding、长序列或 cache 加进去，判断原来的代码哪里不够了。

卡住了就回[基础与模型](../../00-foundations/study-guide.md)补一补，不用硬背整页。做题是为了看看自己哪里还没懂，不是猜某家公司会考什么。

## 自测

<details><summary>代码输出对了，为什么还不算讲清楚？</summary><p>还要解释在哪些输入和假设下成立。小例子可能没碰到空输入、mask 方向、精度或复杂度问题。</p></details>

<details><summary>一道题刚做完，要怎样检查自己不是只记住答案？</summary><p>改一个约束：规模增大、数据有缺失、允许重复或不能看未来。先预测哪些结论会变，再验证。</p></details>
