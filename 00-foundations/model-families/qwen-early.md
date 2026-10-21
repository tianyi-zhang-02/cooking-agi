# Qwen 1–2.5：哪些改了架构，哪些改了训练？

**中文** · [English](qwen-early.en.md)

早期 Qwen 不只是新版的背景知识。它很适合练习一件事：看到一代模型更好，先拆开 tokenizer、架构、数据与后训练，别把全部变化都归功于某个新组件。本篇讲历史版本；Qwen3（2025）及之后的型号见 [Qwen 精读](qwen.md)。

## 先认准讨论的是哪个版本

| 版本 | 本篇抓住的变化 | 不能顺手推断 |
| --- | --- | --- |
| Qwen | 大词表、decoder、RoPE / RMSNorm / SwiGLU，base 与 chat 分开 | chat 行为全部来自预训练 |
| Qwen1.5 | 多语言、对齐与标准化使用接口 | 家族内所有尺寸的实现都完全相同 |
| Qwen2 | GQA、长上下文；dense 与 MoE 两类 | MoE 的 active 参数就是部署所需全部内存 |
| Qwen2.5 | 延续主要 block，明显扩充数据与后训练 | 提升必然来自新的 attention 公式 |

依据 [Qwen 报告](https://arxiv.org/abs/2309.16609)、[Qwen1.5 官方介绍](https://qwenlm.github.io/blog/qwen1.5/)、[Qwen2 报告](https://arxiv.org/abs/2407.10671)与 [Qwen2.5 报告](https://arxiv.org/abs/2412.15115)。发布日期、后续补充型号、API 别名是不同信息，旧博客更新过的列表不能都当成首发阵容。

先把时间线对齐：[Qwen1.5](https://qwenlm.github.io/blog/qwen1.5/) 发布于 **2024-02-04**，[Qwen2](https://qwenlm.github.io/blog/qwen2/) 是 **2024-06-07**，[Qwen2.5](https://qwenlm.github.io/blog/qwen2.5/) 是 **2024-09-19**。技术报告可能晚于模型发布，后续还有新增尺寸和专用版本。看旧资料时，先确认它描述的是哪一次发布，再比较配置和结果。

## Qwen：从输入到 loss，看完整一次

文本经过 byte-level BPE 变成 ID，embedding 进入 causal decoder，各层包含 attention 与 FFN，最后投影到词表并预测下一个 token。RoPE 处理位置，RMSNorm 控制表示尺度，SwiGLU 是 FFN 的门控激活。它们不是同一类“注意力优化”。

原报告采用 untied input embedding 与 output projection。假设词表 $V=150000$、hidden size $d=4096$，单个矩阵就有 $Vd=614400000$ 个参数；额外保留一份 BF16 权重约 1.14 GiB，还没算梯度和优化器。这是自拟预算，不是某个 Qwen 型号的精确配置。

**tokenizer 的词条数不等于模型矩阵的行数。** [Qwen2 报告 §2.1](https://arxiv.org/html/2407.10671v1#S2.SS1)写的是 151,643 个普通 token 加 3 个控制 token，并明确说 embedding 会扩到更大的尺寸；[Qwen2.5 报告 §2](https://arxiv.org/html/2412.15115v1#S2)把控制 token 扩为 22 个。[Qwen2-7B 配置](https://huggingface.co/Qwen/Qwen2-7B/blob/main/config.json)中的 `vocab_size=152064` 则用于模型张量形状，不能解释成有 152,064 个普通词条。核对 token ID 映射、added tokens、模板和矩阵尺寸时，要分别看对应文件；矩阵行数相同也不保证协议可互换。

大词表可能让一段文字用更少 token 表示，但同时增加 embedding 与输出分类的成本。比较中文与英文吞吐时，应同时记字符数、token 数、batch 与生成长度；“每秒 token 多”不自动表示“每秒处理的原文多”。

## Qwen1.5：接口也是实验的一部分

这一代向标准 Transformers 接口靠拢，使用 tokenizer 中的 chat template 与 `generate`，而不是假定仍有旧版专用 `chat` 方法。旧依赖版本不用背；这里值得学的是：**同样的消息列表可能被编码成不同输入**。

做一次升级前，保存旧、新 tokenizer 对同一 messages 的 token IDs，检查 role、turn boundary、assistant 起始位置、EOS 与 padding。别手动拼好模板后，又让库重复套一层。按 [Transformers 的模板说明](https://huggingface.co/docs/transformers/chat_templating)，`apply_chat_template(..., tokenize=True)` 可直接得到 ID；若先用 `tokenize=False` 得到文本，再单独 tokenize，应设 `add_special_tokens=False`，避免重复添加 special tokens。

接口类名也不是发布日期标签：Qwen1.5-7B 的配置已经使用 `Qwen2ForCausalLM` / `model_type="qwen2"`，但它不是 Qwen2-7B；下面的 Q/KV 数量就不同。

| 观察到的现象 | 先检查什么 |
| --- | --- |
| 模型不断续写用户发言 | assistant generation prefix 是否缺失 |
| 回答结束后还接着生成新一轮 | 停止 token / stop 规则是否匹配 |
| 换版本后 loss 异常下降 | 是否错误计入 prompt 或模板 token，标签有没有 shift 错 |

这些是诊断路径，不是说原模型存在这些错误。Base checkpoint 也不能直接当作已经学会 chat protocol 的模型来评价。

## Qwen2：GQA 的账可以单独算

GQA 让多个 query heads 共享 K/V heads。以报告中 7B 型号的 28 个 Q heads、4 个 KV heads 为例，固定其他条件，相比 28 个 KV heads，原始 KV 存储是 $4/28=1/7$。这个缓存比例不是整模型 7 倍加速：28 路 query 和 FFN 都没有少算，道理和 [Llama 的 GQA 算例](llama.md#gqa-cache)一样。

也不能把 GQA 的分界写成“Qwen1.5 全是 MHA，Qwen2 才有 GQA”。以下是本次读取的具体配置，而非首发阵容：

| 配置文件 | Q heads / KV heads | 相对同尺寸 MHA 的 KV 比例 |
| --- | --- | --- |
| [Qwen1.5-7B](https://huggingface.co/Qwen/Qwen1.5-7B/blob/main/config.json) | 32 / 32 | 1（MHA） |
| [Qwen1.5-32B](https://huggingface.co/Qwen/Qwen1.5-32B/blob/main/config.json) | 40 / 8 | 1/5 |
| [Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B/blob/main/config.json) | 28 / 4 | 1/7 |
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json) | 28 / 4 | 1/7 |

每一行的比较都固定层数、head dimension、序列长度和 dtype，不是跨型号内存比。套用 [Llama 一节的缓存公式](llama.md#gqa-cache)，下面只算单请求、28 层、8,192 token、BF16 的 Qwen2-7B KV 张量：

```python
def kv_cache_bytes(*, batch, layers, tokens, kv_heads, head_dim, bytes_per_value=2):
    dimensions = (batch, layers, kv_heads, head_dim, bytes_per_value)
    if any(type(value) is not int or value <= 0 for value in dimensions):
        raise ValueError("cache dimensions must be positive integers")
    if type(tokens) is not int or tokens < 0:
        raise ValueError("tokens must be a non-negative integer")
    return 2 * batch * layers * tokens * kv_heads * head_dim * bytes_per_value


qwen2_gqa = kv_cache_bytes(batch=1, layers=28, tokens=8192, kv_heads=4, head_dim=128)
qwen2_mha = kv_cache_bytes(batch=1, layers=28, tokens=8192, kv_heads=28, head_dim=128)
assert qwen2_gqa == 448 * 2**20
assert qwen2_mha == 7 * qwen2_gqa
```

结果是 448 MiB；假想的 28-KV-head 版本是 3,136 MiB。这只是 KV 张量本身，假定运行时没有把共享 KV 复制成 28 份；权重、激活和框架开销都不在内。

Qwen2 还给出 57B-A14B MoE 与 dense 路线。读 MoE 配置时，分别记总参数、激活专家与 shared experts，再算通信和驻留权重。长上下文部分用到 DCA / YaRN，机制见[位置与上下文](../deep-dives/position-and-context.md)，不是只改一个最大长度常数。

## Qwen2.5：架构相近，训练配方仍然很重要

报告中的公开 dense 型号延续 GQA、SwiGLU、RoPE 与 RMSNorm 主体。预训练语料扩充、筛选与混合方式改变，后训练则分别讨论 SFT、offline DPO 与 online GRPO。不要把这些写成“用了 RL”一句带过。

| 阶段 | 输入信号 | 为什么不能互相代替 |
| --- | --- | --- |
| Pretraining | 混合语料的 next-token 目标 | 主要积累语言与知识模式 |
| SFT | 任务和示范输出 | 教模型怎样按要求作答 |
| Offline preference | 事先比较的回答对 | 偏好优化受现有回答覆盖限制 |
| Online generation / reward | 当前策略采样与评分 | 接近当前行为，但增加采样和奖励设计成本 |

这张表帮助追踪信号，不表示每个尺寸、API 与专用模型有一份完全相同的 recipe。Coder、Math、VL 和后来的 1M 版本也不能只当成上下文参数不同的同一 checkpoint。

## 做一个不会混淆因素的小实验

假如升级后一个结构化抽取任务更好，先固定 50–100 个自拟或有授权的样本：中文、英文、混合语言，短输入与长输入都要有。输出 schema 固定，检查字段正确性与无法确定时的行为，不只检查 JSON 能否解析。

1. 用各自正确的模板比较型号，记录精确 revision、tokenizer、量化、采样参数和截断。
2. 单独比较 base / instruct，别把后训练差异误归为 architecture。
3. 在共同支持的上下文长度与相同输出预算下比较，再另做长上下文实验。
4. 报告错误类别与成本。若同时改变了数据、规模和模板，只能说整套版本的效果不同，不能认定某个组件是原因。

核对日期：2026-10-10。阅读范围是上述报告的 tokenizer、架构与训练章节、Qwen1.5 接口介绍和所链接的具体配置；不是逐一验证全部历史型号。配置链接指向可变的 `main`，复现实验时仍须固定 revision。本篇没有复跑各代完整 benchmark；预算与诊断例子是教学用例。
