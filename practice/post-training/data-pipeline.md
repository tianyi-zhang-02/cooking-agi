# 从一条示范到一个训练 batch

**中文** · [English](data-pipeline.en.md)

> 原创教学项目 · 核对：2026-10-09。示例是合成数据；校验程序只检查结构与计算，不代替内容审核。

给训练器一个 `messages` 字段，很容易启动 SFT。但几天后发现模型总引用旧资料，想回查是哪批数据教的，只有对话文本就不够用了。

我们分开保留两份东西：**能追溯的原始记录**，以及**训练器实际吃到的样本**。前者方便审核和重做，后者方便检查模板与 loss。

## 原始记录：不要过早丢掉来源

下面把一条 JSONL 记录展开显示。项目只处理一轮文字问答；多轮对话、工具调用或图片需要扩展 schema，不能直接套这个校验器。

```json
{
  "sample_id": "demo-001",
  "group_id": "registration-deadline",
  "split": "train",
  "question": "When does registration close?",
  "evidence": [
    {"id": "policy-7", "revision": "v2", "text": "Registration closes on Wednesday."}
  ],
  "answer": "Registration closes on Wednesday [policy-7].",
  "citations": ["policy-7"],
  "answerable": true,
  "label_source": "synthetic-demonstration",
  "review_status": "approved"
}
```

`revision` 让我们能找到当时的资料，`label_source` 区分人工示范与模型生成，`group_id` 把同一问题的改写放在一起。`approved` 只是流程状态，不证明答案正确；示范依然要抽查引用是否支持结论。

如果问题不可回答，明确记 `answerable: false`，写合适的拒答示范，引用列表为空。不是把答案留成空字符串，也不是随便选一篇资料充数。

## 先划分，再变成训练格式

先去重、确定分组，再划分 train / dev / test，然后分别做改写、模板化和 tokenization。先生成一堆近义改写再随机按行切分，会让测试集看起来比实际容易。

按问题分组并不能解决所有泄漏。如果目标是泛化到新文档，需要按文档或文档家族隔离；如果是未来数据，需要时间边界。选哪种取决于你要验证什么。正文 hash 能找完全重复，不能替代语义近重复检查。

| 产物 | 保留什么 | 可以复查什么 |
| --- | --- | --- |
| 审核记录 | 原始问题、证据版本、答案、来源、分组 | 这条示范为什么能进训练 |
| 训练视图 | `messages`，另外保留 sample ID 对照表 | 系统指令和证据到底怎么拼接 |
| tokenized 缓存 | token、目标 mask、长度、预处理指纹 | 同一句话为什么变成这些监督位置 |
| 运行清单 | 数据快照、模板、tokenizer revision、配置 | 更换依赖后是否还在做同一个实验 |

训练视图可使用 `system` 指令、包含问题和证据的 `user` 消息、作为示范的 `assistant` 消息。来源元数据留在旁路文件，不要不加区分地全塞进模型上下文。证据要有清晰边界，并按不可信资料处理，不能让其中的文字覆盖系统指令。

模板也不是随意添加几个角色名。[Transformers 的聊天模板文档](https://huggingface.co/docs/transformers/chat_templating)说明了模型特定控制 token 的作用。训练使用完整示范，不额外加一个待生成的 assistant 开头；若先格式化成字符串再 tokenize，要避免重复添加 special tokens。

## 看一条样本的 labels，而不只看 JSON

设玩具 token 序列为 `[11, 12, 21, 22, 2]`：前两个表示上下文，后两个是答案，`2` 是结束 token。目标 mask 为 `[0, 0, 1, 1, 1]`。

```text
input_ids      11    12    21    22     2
labels       -100  -100    21    22     2
attention       1     1     1     1     1
```

在常见 causal-LM loss 接口中，模型内部把 `logits[:, :-1]` 与 `labels[:, 1:]` 对齐。于是位置 1 的输出预测 token 21；并不是同一位置“偷看”答案后预测自己。接自定义 loss 时要确认谁负责 shift，不能重复移位。

用户 token 不作为 target，不代表模型看不见它们。`attention_mask` 控制可见或有效位置，`labels=-100` 是这里采用的忽略 loss 约定。更完整的手算见[数据与目标](data-and-objectives.md)。

还有一个很容易漏掉的情况：pad 与 EOS 可能使用同一个 token ID。若只按 `input_ids == pad_id` 屏蔽，就会连真实结束 token 一起抹掉。下面的校验程序按**实际长度**补 pad，因此保留真实 EOS 的监督。

## 长度预算与 packing：省掉的是什么

假设输入共 1,600 token，答案 500 token，而上限是 2,048，至少超了 52 token。机械保留前 2,048 个 token 会截掉答案尾部，可能恰好丢掉引用和结束标记。先决定缩短哪些证据或历史，再重新生成样本并验证支持关系；保不住目标就剔除或另分一桶，不静默裁掉。

| 做法 | 好处 | 代价与检查 |
| --- | --- | --- |
| batch 内 padding | 边界清楚，容易排错 | 长短差距大时浪费计算；按长度分桶会改变取样顺序 |
| packing 多条样本 | 减少填充，让计算更多用于实际 token | 要确认 attention 隔离、position IDs 和样本边界处的 labels |
| 按 token 预算组 batch | 每批计算规模更接近 | 样本数量会变；要重新检查 loss 分母和累积逻辑 |

长度 3 和 5 的两条样本，padding 到 5 需要 10 个槽位，其中 8 个是真实 token，利用率是 80%。把它们拼起来可用 8 个槽位，但正确的独立样本训练还需要边界语义：第二条不能注意第一条，第一条的末尾也不能被训练成预测第二条开头。

只重置 position IDs，不保证所有 attention 后端都自动隔离。TRL 的 packing 行为依赖策略与后端；按锁定版本验证。下面代码**只实现易检查的 padding，不假装实现高效 packing**。

## 跑一次，再故意把数据弄错

[标准库实现](code/training_contracts.py)包含记录校验和教学 collator。在仓库根目录运行：

```bash
python3 practice/post-training/code/training_contracts.py
python3 -m unittest discover -s site/tests -p 'test_training_contracts.py'
```

可以改三处试试：把引用 ID 换成不存在的资料、把全部 target mask 改成 0、把 pad ID 设成与 EOS 一样。前两种应报错，最后一种应保留真实 EOS，只屏蔽新补的 pad。测试还会拒绝跨集合的同组样本。

这些检查抓不住“资料存在，但日期读错了”。那部分仍需要内容审核、反例和固定评估。不要把 schema 校验通过写成数据质量已经过关。

下一步先[手算 loss](data-and-objectives.md)，再考虑[多卡是否改变了这个目标](distributed-training.md)。
