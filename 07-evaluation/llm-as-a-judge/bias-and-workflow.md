# LLM-as-a-Judge：偏差与稳健流程

**中文** · [English](bias-and-workflow.en.md)

## 常见偏差

- **Position bias**：pairwise 中偏爱先出现或后出现的答案。
- **Verbosity bias**：把更长、更详细误判为更好。
- **Self-enhancement bias**：judge 偏爱与自己模型家族相似的输出。
- **Style bias**：被标题、Markdown、语气和自信程度影响。
- **Reference anchoring**：参考答案不完整时，judge 可能错误惩罚其他有效解法。
- **Scale bias**：长期偏爱某几个分数，导致评分范围实际没有被使用。
- **Evidence leakage**：candidate 中的提示注入或自我评价影响 judge。

## 一个更稳健的最小流程

```text
1. 先用确定性规则处理能明确验证的条件
2. 每个 judge 只判断一个 criterion
3. 写出有行为锚点的 rubric
4. 有 reference 就提供，但允许其他等价答案
5. 用少量人工样本校准 prompt 和阈值
6. Pairwise 交换顺序；pointwise 检查分数分布
7. 保存 verdict、rationale、evidence 和 evaluator version
8. 按任务、用户和 failure type 做 slice analysis
9. 定期重新测 judge 与人的一致性
```
