# LLM-as-a-Judge：criterion 怎么写

**中文** · [English](criteria.en.md)

一个好 criterion 通常有五个特点。

## 1. 一次只判断一件事

“这是不是一个高质量回答”太模糊。可以拆成 correctness、faithfulness、relevance、completeness 和 style。

如果必须把多个维度合成总分，也应该先分别判断，再明确怎样组合。

## 2. 能指向可观察证据

不要只写“helpful”。应说明 helpful 在当前任务中表现为什么，例如：是否给出可执行步骤、是否覆盖用户明确约束、是否避免无关内容。

## 3. 写清作用范围

Faithfulness 可以指“只依据给定 context”，也可以指“与现实事实一致”。这两个任务需要的证据完全不同。

## 4. 给出边界和反例

说明什么情况一定失败、什么情况仍然可以通过，以及信息不足时是否允许 `unknown`。

## 5. 让不同分数有行为锚点

`4 = good`、`3 = okay` 没有太大帮助。更好的 rubric 是：

```text
5  完全正确，覆盖所有必要约束，没有实质遗漏
4  核心正确，只有不影响使用的小遗漏
3  部分正确，但遗漏一个重要要求或需要用户补充修正
2  存在主要错误，只有少量内容可用
1  错误、无关，或违反关键约束
```

## 常见 criterion

不同任务应该选择不同组合，而不是默认把所有指标都跑一遍。

### RAG / 问答

- **Answer correctness**：答案是否正确？
- **Faithfulness / groundedness**：结论是否能由提供的证据支持？
- **Answer relevance**：是否直接回答用户问题？
- **Context relevance**：检索证据是否与问题有关？
- **Completeness**：是否覆盖完成任务需要的关键点？

### Agent

- **Task completion**：真实任务是否完成？
- **Tool correctness**：工具选择与参数是否正确？
- **State integrity**：执行后的外部状态是否符合预期？
- **Policy compliance**：是否遵守权限和不可违反的约束？
- **Efficiency**：是否存在无进展的循环、重复搜索或成本失控？
- **Recovery quality**：工具失败后是否正确解释、重试或升级给人？

### Personalization / Model Experience

- **Intent fit**：是否符合当前意图，而不是只匹配长期偏好？
- **Memory appropriateness**：使用的记忆是否相关、仍然有效且允许被纠正？
- **Breadth**：结果是否过度重复或把用户困在一个狭窄主题？
- **Control**：用户能否理解并干预关键决定？
- **Longitudinal improvement**：纠正是否在后续 session 中真正生效？

### 开放式生成

- **Correctness**、**coherence**、**instruction following**、**tone**、**clarity**、**originality**。

这些维度越主观，越需要明确 rubric 和人工校准。
