# Qwen 精读：一个家族，多种规模与推理模式

**中文** · [English](qwen.en.md)

> 类型：模型家族精读 · 最近核对：2026-10-10

Qwen 1–2.5 的 tokenizer、GQA、数据和后训练变化见[早期版本精读](qwen-early.md)。本篇以初版 Qwen3 为基线，再对照 Qwen3-2507、Qwen3-Next、Qwen3.5、Qwen3.6 和已核对的 Qwen3.8 型号。核对截至 2026-10-10，不是完整产品目录或“最新版”保证；模型卡描述不等于已验证的服务能力或完整训练配方。

<div class="lesson-recipe advanced">
  <div><span>核心问题</span><strong>怎样让一个模型家族覆盖不同预算与 workload？</strong></div>
  <div><span>重点组件</span><strong>Dense / MoE · GQA · QK-norm · multilingual</strong></div>
  <div><span>训练主线</span><strong>大模型四阶段后训练；小模型另走 strong-to-weak distillation</strong></div>
  <div><span>读完要会</span><strong>区分架构稀疏、推理时计算和小模型蒸馏</strong></div>
</div>

## 一句话定位

读 Qwen3，我更关心的是：**同一家族怎样适应不同任务和预算？** 它有 dense 和 MoE、不同参数规模，也支持 thinking 与 non-thinking。把这些版本放在一起看，才能分清哪些差异来自模型大小，哪些来自架构或推理方式。

这里的模式切换主要指初版 Qwen3 的 hybrid checkpoints。家族名不能代替型号：例如 [Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) 只支持 non-thinking，而 [Thinking-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507) 只支持 thinking。这是两份不同权重，不是同一个 checkpoint 的两种请求设置。

### 初版 Qwen3：先看配置，再看规模名 {#original-qwen3-config}

[2025 年 Qwen3 报告 §2](https://arxiv.org/html/2505.09388v1#S2)列出的 GQA 头数如下；不要把它套到 Next、3.5 或 3.8：

| 初版型号 | Q heads / KV heads | KV 相对同形状 MHA |
| --- | --- | --- |
| Qwen3-0.6B / 1.7B | 16 / 8 | 1/2 |
| Qwen3-4B / 8B | 32 / 8 | 1/4 |
| Qwen3-14B | 40 / 8 | 1/5 |
| Qwen3-32B | 64 / 8 | 1/8 |
| Qwen3-30B-A3B | 32 / 4 | 1/8 |
| Qwen3-235B-A22B | 64 / 4 | 1/16 |

这些比例只替换同一模型的 KV 头数，不能横向当作型号之间的缓存比。初版相对 Qwen2.5 去掉 QKV bias，并加 QK-norm 来稳定训练；QK-norm 作用于 Q/K 向量，不是替换 softmax，也不减少缓存头数。两款初版 MoE 都是 128 个专家、每 token 选 8 个，**没有 shared expert**。

特别看 [Qwen3-30B-A3B 的配置](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json)：`hidden_size=2048`、32 个 Q heads，但 `head_dim=128`，不是 $2048/32=64$。Q 投影宽度可以不同于 residual hidden width，不能总用 `hidden_size / num_attention_heads` 猜 head dimension。

[Qwen3-8B 配置](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json)中的 `vocab_size=151936` 是模型矩阵尺寸；报告的 tokenizer 词表数是 151,669。其 [tokenizer 配置](https://huggingface.co/Qwen/Qwen3-8B/blob/main/tokenizer_config.json)还定义了 `<think>` / `</think>` 的 ID 与模板。三者要一起核对，不能只看“仍是 byte-level BPE”。上下文也有不同口径：8B 卡片区分 32,768 native 与用 YaRN 扩展到 131,072；默认配置的 `max_position_embeddings=40960` 既不是预训练序列长度的证明，也不表示已经启用 128K 扩展。

### 模式开关到底改了什么？ {#qwen-thinking-switches}

以 [初版 Qwen3-8B 的接口](https://huggingface.co/Qwen/Qwen3-8B#switching-between-thinking-and-non-thinking-mode)为例，`enable_thinking=False` 是传给 chat template 的硬开关；`/think` 与 `/no_think` 是它开启时的软指令，不能覆盖硬关闭。当 `add_generation_prompt=True` 且硬关闭时，当前模板把一个空的 `<think>…</think>` 段放进 **prompt**；它不是模型新生成的推理。因此检查“有没有 think 标签”时，要先分开输入模板、生成 continuation 和服务端解析后的字段。

这不涉及每次请求重新训练权重，也不靠切换 MoE 路由数量实现。后训练让模型学会协议，推理时模板和采样控制它如何作答；`max_new_tokens` 限的是整个 continuation，不是独立的“思考额度”。预算用尽而截断也不保证已生成最终答案。

后来的型号又多了两个开关（截至 2026-10）。[Qwen3.8-27B 模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)用 `reasoning_effort` 调推理深度，可选 `xhigh`（默认）、`medium`、`low`，示例里直接作为请求参数传入。`preserve_thinking` 决定多轮对话里是否保留之前各轮的 thinking 块：Qwen3.8 默认保留，[Qwen3.6](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) 默认不保留、只留最近一轮。前者改变这一次回答花多少推理，后者改变下一轮输入里带着多少旧推理。它们和 `enable_thinking` 一样，只改请求和模板，不改权重。

## 先把三种“省计算”分开

<figure class="worked-update" id="qwen-compute-choices" lang="zh-CN">
  <figcaption>三个选择，不是三个先后执行的步骤</figcaption>
  <ol>
    <li><strong>模型规模</strong><span>选更小或更大的权重。小模型也可以通过蒸馏学习大模型的输出。</span></li>
    <li><strong>Dense 还是 MoE</strong><span>改变每个 token 调用的参数范围；总权重仍需要存放。</span></li>
    <li><strong>推理预算</strong><span>在支持的型号上选择 thinking / non-thinking，再设置生成预算。</span></li>
  </ol>
</figure>

- **参数规模**影响模型容量，也决定部署时最基本的资源需求。
- **MoE**让总参数增长得比每 token 激活计算更快，但会引入路由与通信。
- **Thinking budget**控制一次请求花多少计算来生成推理过程；这既不等于换成 MoE，也不等于增加参数量。

把三件事混在一起，就很容易把“更大”“更稀疏”和“想得更久”都笼统叫成 scaling。

## 真正值得抓住的三件事

1. **不同版本面向不同任务，不只是参数量不同。** 小 dense 模型适合低成本部署，大 MoE 提高容量，thinking mode 把额外计算留给复杂问题。
2. **初版 hybrid checkpoints 可以切换 thinking。** 这是具体 checkpoint 的接口，不是整个家族的永久承诺；评价时必须分别报告质量、长度与延迟。
3. **大模型也可以帮助训练小模型。** 通过蒸馏，小模型能学习大模型的输出和推理过程，不必完全独立地学习这些能力。

## 取舍表

| 选择 | 得到什么 | 付出什么 |
| --- | --- | --- |
| Dense + MoE 产品线 | 同一生态覆盖不同成本区间 | 训练、服务和评估矩阵更复杂 |
| Thinking mode | 复杂问题可使用更多 test-time compute | 延迟、token 成本与输出稳定性变化 |
| 多语言扩展 | 更广的语言与跨语言迁移 | 数据配比和长尾语言评估更难 |
| 家族内蒸馏 | 小模型继承大模型部分能力 | 上限受 teacher 与蒸馏数据约束 |

## 后训练不是所有型号都走同一条线

[Qwen3 报告 §4](https://arxiv.org/html/2505.09388v1#S4)把大模型路线分成四步：long-CoT cold start、reasoning RL、thinking mode fusion、general RL。小模型使用 strong-to-weak distillation，不能把它简单写成所有模型的“第五步”。

例如，学生已经生成了一段不太理想的前缀。只让它模仿老师事先写好的答案，和让老师在**学生自己的前缀**上提供分布，是两种不同训练输入。前者对应这里的 off-policy 阶段，后者是 on-policy distillation 要解决的问题。区别在于训练时遇到了谁产生的上下文，不是老师“在线”还是“离线部署”。

### Qwen3-Next：同名家族里的架构转弯 {#qwen3-next-boundary}

[Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)已经换成 Gated DeltaNet 与 Gated Attention 的 3:1 混合布局；MoE 是 512 个 routed experts、每 token 选 10 个，另有 1 个 shared expert，不能套用初版的 128/8、无 shared expert。这个 Instruct checkpoint 只支持 non-thinking。模型卡还描述了 MTP；训练中预测后续多个位置，与推理引擎是否使用对应的 speculative decoding 路径，是两件事，不能仅凭模块存在就推导实际加速。

## Qwen3.5：线性状态与完整历史各负责什么？

以 [Qwen3.5-397B-A17B 的官方配置](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)为例：每组 3 层 Gated DeltaNet 后接 1 层 Gated Attention，另有视觉编码器；MoE 使用 512 个 routed experts，每 token 激活 10 个，加 1 个 shared expert。这些数字属于这个型号，不要套到所有 Qwen3.5。[配置文件](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json)把语言模型放在 `text_config` 下；完整注意力的 Q/KV heads 与 DeltaNet 的 linear heads 也是不同字段，不能用一个顶层 head 数替代。

它支持通过 `enable_thinking` 关闭 thinking，但不应沿用初版的斜杠软指令。自托管示例用 `chat_template_kwargs` 包裹该参数，官方托管示例则直接传参数；“兼容同一种 API”不保证额外字段的嵌套位置一致。

直观地看，递归状态把历史压进固定形状的矩阵；完整注意力还会查历史 KV。两者混用，不等于整个模型有恒定大小的缓存或线性的完整序列计算量。单步 decode 与整段 prefill 也要分开算。

“按 key 修正记忆”同样不等于毫无干扰地删掉一条记录。两个 key 很接近时，改动一个方向可能影响另一个方向。[Gated DeltaNet 的推导与数值例子](../deep-dives/gated-deltanet.md)解释了这个限制，也区分了训练参数与 forward 里的快速状态。

## 2026 更新：Qwen3.8 不是一种统一结构

Qwen3.5 和 3.8 之间还有 Qwen3.6：据 [Qwen 的 GitHub 发布记录](https://github.com/QwenLM/Qwen3.8)，MoE 的 Qwen3.6-35B-A3B 发布于 2026-04-16，dense 的 Qwen3.6-27B 发布于 2026-04-22，两者都沿用 3:1 的 DeltaNet / attention 布局并带视觉编码器。

下面按 2026-10-10 实际读取的官方模型卡整理（Qwen3.6 一行读取于 2026-10-16），只列影响理解的差异，不代表检查了全部 2026 发布。链接指向可变页面；复现时仍应固定 checkpoint revision、模板和推理引擎版本。

| 公开型号 | token mixer / FFN | 输入与部署边界 |
| --- | --- | --- |
| [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | 3 层 Gated DeltaNet + 1 层 Gated Attention；MoE 256 个专家，每 token 8 个 routed + 1 个 shared | 公开视觉语言模型；默认 thinking，可按请求关闭，不支持斜杠软指令 |
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 3 层 Gated DeltaNet + 1 层 Gated Attention 的重复布局；dense FFN | 公开视觉语言模型；支持按请求关闭 thinking，不是纯文本 Qwen3 的换名版本 |
| [Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) | 同类混合 attention，MoE；总量 2.4T、激活 95B | 公开权重是文本、thinking-only 模型；托管 Max 的视觉、non-thinking 与工具功能不能归到这份权重 |
| [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | Gated DeltaNet + QSA；另有 gated residual、n-gram embedding 与 MoE | 公开视觉语言模型，可关闭 thinking；QSA 选 micro-block，模型卡将其定位为新架构预览 |

27B 型号仍有完整注意力层，所以“混合线性状态”不等于所有历史缓存恒定。2.4T 型号的 active 数字描述每 token 使用的计算子集，不是把其他权重从内存账里删掉。Flash-Next 则还多了一笔 n-gram 表的存储与读取成本。

### Flash-Next 的参数账，为什么不能只看 6B？

模型卡在语言模型部分列出 125B 主体、51B n-gram embedding 与 4B MTP，合计约 180B；这不是包含视觉编码器的完整部署账。主体每 token 激活约 6B。这些口径回答不同问题：

- 6B 帮助理解主计算路径，不包含完整的驻留与通信成本。
- 51B 表参数通过局部地址读取，不能等同于每 token 执行 51B dense matmul；但表仍需要存储、传输与缓存。
- MTP 的训练与推理使用方式单独算；不能因为附带这个模块就声称解码必然加速。

若只给这笔语言模型参数小计做粗略的 BF16 原始权重估算，180B × 2 bytes 约 360 GB（十进制）。量化、offload、共享与额外缓冲都会改变实际需求；这个数字不是可部署的显存预算。

QSA 模型卡给出的预算是 512 个 micro-block、对应 2,048 个 token。**1/16 只适用于单个 query 面对 32,768 个可见位置时，按 2,048 个被选位置估算的主读取比例；不是整段 causal prefill 的比例。** 假设第 $t$ 个 query 恰好读取 $\min(t, k)$ 个位置，那么整段读取数为 $\sum_{t=1}^{S}\min(t,k)$，dense 则是 $S(S+1)/2$。

```python
def causal_read_pairs(tokens, budget=None):
    if type(tokens) is not int or tokens < 0:
        raise ValueError("tokens must be a non-negative integer")
    if budget is not None and (type(budget) is not int or budget <= 0):
        raise ValueError("budget must be a positive integer")
    capped = tokens if budget is None else min(tokens, budget)
    return capped * (capped + 1) // 2 + (tokens - capped) * capped


sequence_length, token_budget = 32768, 2048
decode_fraction = min(sequence_length, token_budget) / sequence_length
prefill_fraction = (
    causal_read_pairs(sequence_length, token_budget)
    / causal_read_pairs(sequence_length)
)
assert decode_fraction == 1 / 16
assert abs(prefill_fraction - 0.121092) < 1e-6
```

在这个理想化的 token-budget 算例里，整段比例约为 0.121092（12.11%），不是 6.25%。它忽略 micro-block 对齐、实际选块、indexer 和 kernel 实现，因此不是对 QSA 实际执行量的测量。无论哪个比例，都不是整模型加速比：MoE、状态更新和其他计算仍在。

同样，其深度门控不能直接写成 [Block AttnRes](../deep-dives/attention-residuals.md)，n-gram 方案不能只因用了 hash 就等同于 [Engram](../deep-dives/engram.md)。相似动机不意味着同一实现。

### 模型卡上的成绩该怎么读？

先对齐 harness、工具权限、上下文、生成预算、采样次数与 benchmark 修订。比如 27B 卡片注明部分代码评估修正了任务、重新评测基线；2.4T 页面也展示托管 Max 的结果。不能把这些表格直接解释为“下载权重，用随便一个脚本就会得到相同分数”。

更实用的比较是固定一个自己的任务集，同时记录成功率、失败类别、总生成 token、工具调用和延迟。公开架构能帮助预测成本在哪，却不能代替这个测量。本页没有重跑上述型号的性能实验，也不把模型卡当作公开了全部训练数据和 recipe。

## 我会怎样使用这个家族

如果研究问题包含**动态计算预算**，Qwen 是很好的对象：同一个任务可以比较 dense / MoE；对确实支持双模式的 checkpoint，还能固定权重比较 thinking / non-thinking。关键是别只报最终准确率；至少同时记录生成 token、延迟、激活参数与失败类型。

## 自检

- MoE 的“稀疏”和 thinking mode 的“多算一会儿”有什么本质区别？
- 为什么同一个模型切换模式后，评估协议也必须改变？
- 蒸馏让模型家族获得了什么，又可能把哪些 teacher 偏差传下去？
- 如果一个小模型在 benchmark 上接近大模型，你还会检查哪些成本和泛化维度？

## 原始资料

本次读了 Qwen3 报告的架构、预训练与后训练章节，以及上文所链接的配置、tokenizer 模板和模型卡相关段落；没有逐行审核推理 kernel、下载权重或复现训练。2026 型号的结构与接口描述来自官方卡片/配置，不代表独立验证了性能，也不推断未公开的训练细节。


- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Qwen3 官方代码与模型](https://github.com/QwenLM/Qwen3)
- [Gated Delta Networks](https://arxiv.org/abs/2412.06464)
