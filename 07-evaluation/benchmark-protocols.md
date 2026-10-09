# 选 benchmark 之前，先把要比较的事写清楚

**中文** · [English](benchmark-protocols.en.md)

假如我们要选一个读长报告的助手，只看知识问答总分，很可能选不到真正需要的模型。知识、找证据、跨段推理、引用和拒答是不同能力；一个漂亮的平均分没法替它们全部作证。

这篇不列“最新最强”的排行榜。它给一套可复用的测试方法：先选问题，再定协议，然后解释分数。核对日期：2026-10-08；旧 benchmark 可用于历史可比性，但不能因此叫它当前最有区分度的测试。

## 1. 每种测试回答什么问题？

| 想知道什么 | 可以参考的测试 | 还需要补什么 |
| --- | --- | --- |
| 学科知识与选择题能力 | [MMLU](https://github.com/hendrycks/test) | 开放式回答、引用、真实任务与数据污染检查 |
| 中文学科知识 | [C-Eval](https://github.com/hkust-nlp/ceval) | 中文交流、行业语言、混合语言和领域任务 |
| 代码能否通过测试 | [LiveCodeBench](https://livecodebench.github.io/) | 依赖环境、仓库级修改、安全与维护性 |
| 长上下文中的查找、关联、汇总 | [RULER](https://arxiv.org/abs/2404.06654) | 自然文档的歧义与真实工作流程 |
| 更复杂的真实长文任务 | [LongBench v2](https://arxiv.org/abs/2412.15204) | 你的文件分布、成本与用户要求 |
| 对话是否有用 | 人审或校准后的 judge | 格式偏好、事实、执行结果要分开 |

选择题正确率不能直接代表聊天好用；代码通过测试也不保证没有安全问题。HELM 的重要思路是把[场景与多个评价维度](https://arxiv.org/abs/2211.09110)一起看，而不是找一个数字替全部行为排名。

## 2. 同名 benchmark，也可能不是同一场考试

最少保存下面这份实验清单。`model_revision` 和 `dataset_revision` 应是可定位的版本，不只写 `latest`。

```text
model / model_revision / tokenizer_revision
dataset / dataset_revision / split / item_ids
prompt_template / few_shot_examples / chat_template
context_limit / truncation_policy / output_budget
temperature / top_p / seed / samples_per_item
tools / network_access / retries / timeouts
answer_extraction / scorer_version / invalid_output_policy
hardware / engine_version / dtype / quantization
```

例如旧模型只输出选项，新模型先推理 8,000 token 再选答案。新模型可能确实更有用，但这不是固定计算预算的比较。可以同时报告“相同预算”和“各自推荐配置”两组，别混在一张不加说明的表里。

超时、解析失败、拒答怎么计分也要事先定。只在成功返回的题目上算准确率，会奖励那些把难题变成超时的系统。[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 能统一部分协议，但仍需锁定 commit、task 配置和运行参数。

## 3. Needle 测试：找到一条记录只是开始

先做一个很小的自拟任务：在很多条记录中找到 `target` 对应的值。我们把同一条证据移动到开头、中间、结尾，其他记录不变。

```python
def lookup_case(record_count, position, include_evidence=True):
    if type(record_count) is not int or record_count < 2:
        raise ValueError("Expected at least two record slots")
    if type(position) is not int or not 0 <= position < record_count:
        raise ValueError("Position must be inside the record slots")
    if type(include_evidence) is not bool:
        raise ValueError("Evidence flag must be boolean")
    records = [f"item-{index:04d}: value-{index:04d}" for index in range(record_count - 1)]
    evidence = "target: maple-47" if include_evidence else "unrelated: birch-82"
    records.insert(position, evidence)
    return "\n".join(records), "maple-47" if include_evidence else None

contexts = [lookup_case(9, position)[0] for position in (0, 4, 8)]
assert all(context.count("target: maple-47") == 1 for context in contexts)
assert all(set(context.splitlines()) == set(contexts[0].splitlines()) for context in contexts)
assert lookup_case(9, 4, False)[1] is None
```

这里只移动**记录槽位**，不是宣称文本正好有多少模型 token。实际运行前要用对应 tokenizer 计数，并确认截断没有把证据删掉。重复 filler 很多遍是容易控制的合成测试，不代表自然长文。

| 加一层难度 | 自拟测试 | 要排除的假解释 |
| --- | --- | --- |
| 位置 | 同一证据换位置 | 是否只擅长靠近末尾的线索？ |
| 多证据 | 查 `target` 的别名，再查别名对应值 | 是否只复制了局部字符串？ |
| 汇总 | 多处记录共同决定总数 | 是否能整合而不漏项、重复计数？ |
| 干扰 | 加相似 ID、过期值或引用 | 是否分得清目标与干扰项？ |
| 无答案 | 删除唯一证据 | 是否仍然猜出那个熟悉的答案？ |

RULER 扩展了单 needle 的测试范围，LongBench v2 则提供更接近真实长文任务的另一类证据。不要把某一张全绿的 needle 图当作“这个模型充分理解了全部上下文”。

## 4. 用一张结果表区分能力和开销

测试长度、证据位置和任务难度的组合；每格要有多个内容不同的样本。下面只是记录格式，不填虚构结果：

| 长度 / 位置 / 任务 | 样本数 | 正确率 | 无答案误答率 | 输入/输出 token | TTFT / 总延迟 |
| --- | --- | --- | --- | --- | --- |
| 短 / 中间 / 单证据 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 长 / 中间 / 多证据 | 待测 | 待测 | 待测 | 待测 | 待测 |

这里 TTFT 是首 token 延迟。缓存命中、prefill、decode 和工具等待会影响不同部分，应单独记录。输出几乎为空的失败请求，可能延迟很低，但没有完成任务。

部署声称支持的最大长度、这套任务上还能保持质量的长度、以及满足延迟预算的长度，往往是三个不同的数。

## 5. 分数差两题，值得换模型吗？

200 道题，旧模型对 160 道，新模型对 164 道，只能先说样本上高了 2 个百分点。还不知道差异来自哪类题、是否稳定，也没算额外成本。

保存每题成对结果，用[配对比较](metric-robustness.md#paired-comparison)估计不确定性；同一原始文档派生的题应按文档等合理单位分组，而不是假装每题独立。多次采样测生成波动，多次独立训练才测训练波动。

提前约定主要指标、不能退化的切片与可接受成本。测试后才挑一个涨得最大的子集，很容易讲出一个并不稳定的故事。区间包含零不是等价证明，统计显著也不自动意味着值得上线。

## 6. 数据污染与评估更新

老题可能进入预训练、SFT、合成数据甚至 prompt 示例。检查完全重复与近似重复有用，但没找到重复不等于证明干净。时间切分也要核对实际训练和数据更新时间；换个日期标签不能消除答案泄漏。

以持续更新的测试为例，固定某个发布快照和日期窗口，避免把不同时间段的题目混成一个可比较的分数。对于模型声称的 cutoff，还应保留不确定性，不把它当可审计的全部训练清单。

最终留一套未用于改 prompt、调 judge 或选模型的 holdout。更新 benchmark 时，新旧系统一起重跑；否则“涨分”也可能只是试卷变了。这里的代码生成测试夹具，不会调用模型，更没有声称已经完成这些真实评估。
