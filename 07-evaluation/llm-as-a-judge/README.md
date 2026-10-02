# LLM-as-a-Judge：怎么评，才有用？

**中文** · [English](README.en.md)

换了一个 prompt，回答更流畅，平均分也涨了。可用户的问题真的解决得更好了吗？

LLM-as-a-Judge，就是让模型按给定标准检查另一个模型的输出。它能帮我们读很多答案，但**打出一个分数，和知道这个分数值得相信，是两件事**。

不用先记一堆指标。我们从一条客服回答开始，走完「给材料 → 分项判断 → 做决定 → 检查 judge」这条路。例子里的规则和判断都是手写的，不调用模型。

## 先走一遍：这条回答，能不能发给用户？

<div data-judge-lab="walkthrough"><p>这是虚构的退货问答：一个回答可能同时做到切题、友好，却编错了退货政策。不能因为 3 项过了 2 项，就把事实错误放过去。交互版可以更换回答、隐藏政策、比较两种放行规则。</p></div>

可以这样试：先保留「热情，但编错政策」，走到第 3 步。它为什么能通过“多数项通过”的规则？再打开必要条件，看看**判断没变，决定为什么变了**。最后换一条完整回答，隐藏政策：这时不是模型答错了，而是我们没给 judge 足够的判断依据。

这里故意把 3 件事拆开了：

- **被测模型**写答案。
- **Judge**按 rubric 给出分项判断，并说明依据。
- **决策规则**决定接受、拦截还是转人工。它不是模型临场拍脑袋定的。

同一个 judge 结果可以对应不同用途：离线比较两版 prompt，和自动把回答发给用户，对误判的容忍度就不一样。单条通过，也不能直接推出整个系统值得上线。

## 这组笔记怎么读

<div class="judge-route" aria-label="分成 3 段的阅读路线">
<section><span class="jr-number">01 · 定义</span><h3>先说清楚，什么叫好</h3><p>把“回答不错”拆成能检查的条件；决定要类别、分数还是比较结果。</p><ul><li><a href="criteria.md">评分标准：评什么，凭什么？</a></li><li><a href="scoring.md">评分方式：通过、打分、二选一</a></li></ul><small>写出一份有边界例子的 rubric。</small></section>
<section><span class="jr-number">02 · 验证</span><h3>再检查，裁判靠不靠谱</h3><p>不只看平均分。换顺序、藏证据、对照人工，看看它具体错在哪里。</p><ul><li><a href="probability-scores.md">概率分数：20 次评分说明什么？</a></li><li><a href="bias-and-workflow.md">偏差检查：哪些改动不该影响结果？</a></li><li><a href="calibration.md">人工校准：误放、误拦与复核</a></li></ul><small>拿出分项错误、样本数和不确定性。</small></section>
<section><span class="jr-number">03 · 使用</span><h3>最后，放回实际任务</h3><p>任务变了，证据也得变。把输出校验、失败处理和报告接起来。</p><ul><li><a href="case-studies.md">RAG、Agent、Memory 怎么评？</a></li><li><a href="implementation.md">跑通一个 Python 小实验</a></li></ul><small>交出可复查的记录，而不只是一个总分。</small></section>
</div>

**第一次接触**：按 01 → 02 → 03 读。**已经在用 judge**：先看[偏差检查](bias-and-workflow.md)和[人工校准](calibration.md)。**想自己搭一个**：带着一条 rubric 去跑[最小实现](implementation.md)，再回来补不理解的部分。

每篇解决一个具体问题。这里没有背完才准往下走的清单，也不用一次读完。

## 能用规则判的，为什么还要问模型？

比如“输出能不能解析为 JSON”，解析器比一句“我觉得格式正确”更有用；“agent 到底有没有保存草稿”，先查最终状态。

但“摘要漏掉了哪项限制”“回答有没有回应用户真正的问题”，不总能用一条正则写清楚。这是 LLM judge 可以帮忙的地方。**确定性检查负责能直接核对的事实，语义判断补上不容易写成规则的部分。**

| 你在检查什么 | 先用什么 | 别混淆 |
| --- | --- | --- |
| 格式、数值、测试、最终状态 | Parser、测试、状态断言 | 测试通过只覆盖测试检查过的行为 |
| 含义、遗漏、材料支持、偏好 | 明确的 rubric + LLM / 人工判断 | 理由流畅不等于判断正确 |
| Judge 自己是否可靠 | 独立人工对照、边界例子、扰动测试 | 几个模型同意，也不是自动获得真值 |

[MT-Bench / Chatbot Arena](https://arxiv.org/abs/2306.05685)记录了位置、篇幅等偏差；[JudgeBench](https://arxiv.org/abs/2410.12784)则直接考查 judge 能不能分清较难题目的正确与错误答案。模型很会回答，不代表它在你的任务上就很会判卷。

还有一个容易混的地方：**同样的分数，拿来观测和拿来训练，风险不同。**一旦分数成了 reward，系统会主动寻找让分数上涨的办法。那更需要独立验收，不能既用同一位 teacher 产标签，又只用它证明 student 变好了。[偏差与流程](bias-and-workflow.md)里会接着讲。

## 这些词，各管哪件事？

| 名称 | 放回刚才的例子 |
| --- | --- |
| Criterion · 评估维度 | 政策有没有编错？申请步骤有没有说？ |
| Rubric · 分档与判定规则 | 哪些情况算 pass、fail、unknown，边界怎么判 |
| Evidence · 判断依据 | 商店的退货政策 |
| Reference · 参考答案 | 这道题的一份可接受答案；不一定是唯一表述 |
| Few-shot examples · 评分示范 | 另外几道已经评过的题，展示如何应用标准 |
| Verdict · 判断结果 | 每一项的 pass、fail 或 unknown |
| Meta-evaluation · 评估裁判 | 对照人工，检查 judge 有没有漏掉错误 |

Few-shot 决定给不给示范，reference-based 决定给不给当前题的参考答案。它们可以一起用，也可以分开用。[下一篇会展开](criteria.md)。

## 想继续往下看

- [G-Eval](https://arxiv.org/abs/2303.16634)：为什么要写 rubric，以及评分 token 概率怎么参与加权。
- [Prometheus 2](https://arxiv.org/abs/2405.01535)：专门训练 evaluator 的思路；换成它也要做任务校准。
- [Judging the Judges](https://arxiv.org/abs/2406.07791)：位置偏差该怎么测。
- [Replacing Judges with Juries](https://arxiv.org/abs/2404.18796)：多个 judge 组成 panel 的一种做法，不是正确性的保证。
- [Confident AI Blog](https://www.confident-ai.com/blog)：可以继续找实践例子；选了框架，仍然得自己定义“好”。

本组所有互动都用虚构数据，不上传输入、不调用模型。分数和判断只是用来解释机制，不是任何真实模型的成绩。

接着读：[评分标准](criteria.md) · [评估总览](../README.md)
