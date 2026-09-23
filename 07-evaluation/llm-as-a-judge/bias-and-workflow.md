# LLM-as-a-Judge：偏差与稳健流程

**中文** · [English](bias-and-workflow.en.md)

## 常见偏差

- **Position bias**：pairwise 中偏爱先出现或后出现的答案。
- **Verbosity bias**：把更长、更详细误判为更好。
- **Self-enhancement bias**：judge 偏爱与自己模型家族相似的输出。
- **Style bias**：被标题、Markdown、语气和自信程度影响。
- **Reference anchoring**：参考答案不完整时，judge 可能把其他同样有效的解法判错。
- **Scale bias**：总是集中给某几个分数，其他档位几乎用不上。
- **Evidence leakage**：被评回答里的提示注入或自我吹捧干扰了 judge。

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
