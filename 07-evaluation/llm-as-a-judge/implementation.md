# 最小实现：先把一次评估完整地跑通

**中文** · [English](implementation.en.md)

先不接大型平台，也不需要 API key。我们用几条虚构的 judge 输出，跑通解析、校验和汇总。重点不是让模型真的评分，而是避免最常见的事故：**错误输出被当成通过，失败调用从分母里消失。**

## 本地跑一下

在仓库根目录：

~~~bash
python3 07-evaluation/llm-as-a-judge/code/judge_eval.py
~~~

[查看完整 Python 实现](code/judge_eval.py) · [查看示例 JSONL](code/example-results.jsonl)

程序只用标准库，读取本地文件，不联网。输出包括总数、可判定数、弃权、坏格式、调用错误，以及 confusion matrix。示例里故意有错误放行、错误拦截、unknown、无效 JSON 和一次调用失败。

修改一条 judge_raw 再运行，看看覆盖率和错误数怎么变。**这里没有训练、没有真实 judge，也没有模型性能结果。**

## 输入和输出分别是什么

输入记录里同时保存人工标签与 judge 原始输出，是为了演示事后 meta-evaluation。真正发送给 judge 时，**不要把 human 标签一起传进去**。

~~~json
{
  "case_id": "toy-01",
  "group_id": "conversation-01",
  "slice": "rag",
  "human": "pass",
  "allowed_evidence_ids": ["policy-1"],
  "judge_raw": "{\"criterion_id\":\"policy-grounding-v1\",\"verdict\":\"pass\",\"evidence_ids\":[\"policy-1\"],\"reason\":\"期限与条件均受条款支持。\"}"
}
~~~

实际请求只发送 task、candidate、criterion 和允许的 evidence；有意使用 reference-based 时，再加入 reference。Human 标签留在独立的评分对照表里。

程序检查 verdict 枚举、criterion_id、字段类型、未知证据 ID，以及重复 case_id。它不会验证「引用了 policy-1 就一定推理正确」，那仍需要语义复核。Judge 输出里多塞一个字段也会拒绝，避免下游误用没有约定过的数据。

## 4 种状态不要揉成一个分数

| 状态 | 意思 | 后续处理 |
| --- | --- | --- |
| decided | 有合法 pass/fail | 才进入条件准确率和 confusion matrix |
| unknown | 合法输出，但证据不足以判断 | 计入弃权，按预定规则复核 |
| invalid | JSON 或 schema 不合约定 | 单独记 evaluator error，不自动给 candidate 判失败 |
| error | 调用没有成功返回 | 记基础设施失败；保留尝试记录 |

Demo 的 coverage 分母是**所有输入 case**，其余错误也显示出来。正式接 API 后还应单独记录每次重试；不要只保留最终成功的那一次，就说调用成功率 100%。

## 真要接模型，在哪儿加？

~~~text
冻结 cases / evidence
→ 确定性检查
→ 构建不含 human 标签的 judge request
→ 调用你选择的 provider adapter
→ 保存 raw response + versions + cost + attempt
→ validate / unknown / error routing
→ 按 case_id 与人工对照表连接
→ 分项统计 + slices + 人工抽查
~~~

把 API adapter 放在「调用」那一步，别把模型 SDK、rubric 和聚合函数全塞进一个 prompt。超时和限流可以做有上限的重试；重试次数、最终错误都保留。开发阶段先用少量脱敏样本，并限制预算。

如果并行评分，结果按 case_id 对齐，不能按返回顺序拼。做缓存时把 candidate、evidence、rubric、模型与采样配置都算进 cache key；不然换了评分规则，还可能读到旧分。

## 上线前的一页验收单

- Rubric 有明确边界、hard gates 和 unknown 语义。
- Development、calibration、holdout 没有串题，也没有泄漏人工标签。
- 顺序、长度、参考答案冲突、注入、缺证据都测过。
- 人工对照里能看到误放、误拦、coverage 和每个 slice 的样本数。
- 随机抽看 pass、fail、unknown 和 parser error，不只看模型挑出来的「精彩解释」。
- 换 judge 或 prompt 时先并行重评固定样本；发生漂移能追到版本并回退。
- 报告说明哪些只是离线证据，还不能推出在线体验改善。

这套小实现不是生产框架，缺少 API、持久化、访问控制和正式的不确定性估计。它的用处是让评估流程先有一个可检查的骨架。

<details markdown="1">
<summary>练习：把一个不存在的 evidence ID 加进示例输出，会怎样？</summary>

它应进入 invalid，而不是因为原始 verdict 写着 pass 就放行。再试把 verdict 改成 unknown：这次是合法弃权，不是 schema 错误。这两个路径要能在报告里区分出来。

</details>

回到：[整组导读](README.md) · [校准与统计](calibration.md)
