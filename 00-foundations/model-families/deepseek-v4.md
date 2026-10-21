# DeepSeek-V4：压缩历史以后，还能读到什么？

**中文** · [English](deepseek-v4.en.md)

把上下文加长，只是允许模型接收更多输入。真正难的是：历史越来越长，哪些信息还要保留，下一步应该读哪些，以及这些取舍会不会损失任务需要的细节。

这篇接着 [V2 / V3 / R1](deepseek.md) 往下读。先看注意力（attention），再看残差和训练；不要把三个层面的变化混成一个“模型更强”。2026-10-10 核对范围是报告 v1 与 V4-Pro / V4-Flash preview 的公开配置和推理源码，不是对所有后续版本的概括，也不声称复现官方性能。

> **之后的版本（截至 2026-10）：** 2026-08-13 [V4-Pro 正式版](https://api-docs.deepseek.com/news/news260813)上线，API 名称不变，并给 V4-Pro 和 V4-Flash 加了 low / high / max 三档推理强度。2026-09-10 发布的 [V4.1-Flash](https://api-docs.deepseek.com/news/news260910) 换成了新的 Causal Encoder–Decoder 结构：据[模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)，它是 552B 主干参数的 MoE，prefill 时每 token 激活 8B、decode 时激活 16B，原生支持图像输入；注意力改用 CSA2，每层固定为 Full、Reindex、Reuse 三种模式之一，模型卡没有提 HCA。9 月的发布说明表示 V4-Flash 将停用，原 API 名暂时转到 V4.1-Flash，`deepseek-v4-pro` 从 9 月 14 日起也会转过去，直到 V4.1-Pro 上线。不过，2026-10-21 核对时，[官方模型表](https://api-docs.deepseek.com/quick_start/pricing/)仍把 `deepseek-v4-pro` 标为 V4-Pro-0813，两份资料并不一致，不能只凭公告断言当前路由。同一张表把 Flash 的推荐名改为 `deepseek-flash`，由 2026-10-17 更新的 [V4.1-Flash-1017](https://api-docs.deepseek.com/news/news261017) 提供；旧名 `deepseek-v4-flash` 仍可调用，也转到这个版本。下面的 CSA / HCA 分析描述的是 V4 权重；比较 API 结果前，还要确认实际服务的模型和版本。

## 1. 两种注意力，各省哪一笔？

[V4 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)介绍 CSA / HCA 混合架构：前者压缩后再稀疏选择，后者更重地压缩、读取压缩后的历史。它们交错安排在不同层，不是每层都先 CSA、再 HCA。两者都保留近处的滑动窗口。

但不能把 Pro 和 Flash 画成完全一样。以下数值来自当天读取的配置；链接中的 `main` 会变化：

| 配置项 | [V4-Pro config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json) | [V4-Flash config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json) |
| --- | --- | --- |
| 主干层数 `num_hidden_layers` | 61 | 43 |
| 前两层 attention | HCA + 局部窗口 | 仅滑动窗口（SWA） |
| CSA `index_topk` | 1024 | 512 |
| 每个 MoE 层的 routed experts | 384 | 256 |

两者的 CSA / HCA 压缩比例都是 4 / 128，局部窗口为 128；每 token 选 6 个 routed experts，另有 1 个 shared expert。前三个 MoE 层采用 token-ID hash 路由，而非内容 top-k；之后的 affinity 使用 `sqrtsoftplus`，不能直接照搬 V3 的 sigmoid。两者都配置 1 个额外 MTP 层；`compress_ratios` 最后的 `0` 对应这个纯 SWA 的 MTP 层，不是多出来的主干层。[配置解释所用源码：Gate / Transformer](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/model.py)

下面这张图只画历史的读法，省略投影、归一化和位置处理：

<figure class="worked-update worked-update--pairs" id="v4-history-paths" lang="zh-CN">
  <figcaption>两种层配置：都保留局部窗口，远处的历史读法不同</figcaption>
  <ol>
    <li><strong>CSA 层</strong><span>历史状态 → 较轻压缩 → 选择部分条目。把选中条目与近期原始条目一起交给本层 attention。</span></li>
    <li><strong>HCA 层</strong><span>历史状态 → 较重压缩 → 读取全部压缩条目。它们同样与局部窗口一起参与本层 attention。</span></li>
  </ol>
</figure>

图里两条路代表**不同层的配置**，不是一个 head 把两套都跑一遍。CSA 的实际压缩有重叠：相邻块的两个投影参与合成，但仍然每 m 个位置产出一个条目，不是每 2m 个。细节见[报告 §2.3](https://arxiv.org/html/2606.19348v1#S2.SS3)。

这里的主 attention 使用 shared-KV MQA，不是直接复用 V3 的 MLA 缓存公式。局部条目与压缩条目进入同一个归一化，分母还含可学习的 attention sink；不能分别 softmax 后直接相加。位置处理等细节仍以源码为准。

## 2. 算存储和算读取，要分开

用一个缩小的记账例子想一遍：已处理 1,024 个位置，轻压缩比例 4，重压缩比例 128，稀疏读取 32 个条目，局部窗口 128。`history` 包含当前 query 所在位置；这是处理完该位置之后的账，而不是当前 token 之前的前缀长度。

| 项目 | 轻压缩 + 选择 | 重压缩 + 全读 |
| --- | --- | --- |
| 完成的压缩条目 | 256 | 8 |
| 当前 query 读的压缩条目 | 最多 32 | 8 |
| 加上局部窗口的主 attention 条目数 | 最多 160 | 最多 136 |
| 没算进去的部分 | 压缩器、indexer、投影、临时状态 | 压缩器、投影、临时状态 |

这里沿用 V4 的压缩比和窗口，但把 top-k 改成 32，不能当作实际 Pro / Flash 配置或延迟预测。原始 token 和压缩条目不是等量的信息，不能因为后者少就说它同样准确。并且**只读 32 个，不代表只存 32 个**：后续 query 可能选择别的条目。局部窗口与压缩块即使覆盖了同一段文本，也保留为不同条目；表中不是去重后的原始 token 数，也不计 sink 标量。

```python
def attention_ledger(history, light_ratio, heavy_ratio, selected, window):
    values = (history, light_ratio, heavy_ratio, selected, window)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("Expected positive integer counts")
    if heavy_ratio <= light_ratio:
        raise ValueError("Heavy compression must use a larger ratio")
    light_entries = history // light_ratio
    heavy_entries = history // heavy_ratio
    local_entries = min(history, window)
    return {
        "light_stored": light_entries,
        "heavy_stored": heavy_entries,
        "light_main_reads": min(selected, light_entries) + local_entries,
        "heavy_main_reads": heavy_entries + local_entries,
        "light_tail": history % light_ratio,
        "heavy_tail": history % heavy_ratio,
    }

assert attention_ledger(1024, 4, 128, 32, 128)["light_main_reads"] == 160
assert attention_ledger(1024, 4, 128, 32, 128)["heavy_main_reads"] == 136
assert attention_ledger(9, 4, 128, 32, 128)["light_tail"] == 1
```

这里按已完成块记账，`light_tail` / `heavy_tail` 只表示当前未完成块的 token 数，不是完整的压缩器状态；CSA 还需要保留前一块的重叠状态。CSA 的 indexer 还会给可见的压缩候选打分，top-k 只限制后续主 attention 的读取，不能推出总计算是常数。历史翻倍，压缩缓存仍会增长；HCA 的主 attention 读取也会增加。

算字节数还要区分论文方案和演示代码：报告 §2.3.4 讨论 RoPE 维度用 BF16、其余 KV 维度用 FP8；所读 `inference/model.py` 的演示缓存则以 BF16 保存并模拟量化。条目数不能直接证明实际显存占用。[报告](https://arxiv.org/html/2606.19348v1#S2.SS3.SSS4) · [推理源码](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/model.py)

## 3. 两个容易藏住的错误

**未来泄漏。** 用从 0 开始的位置，query 在 8，块宽 4：已经处理 9 个位置，有两个完成条目，不能读取包含 8–11 的完整压缩条目。对 CSA，第二个条目融合了 4–7 和前一块 0–3，不是只含 4–7；重叠不改变完成时间。所读源码按 `(query_position + 1) // block_width` 判断完成数量，因此在位置 7 就可以读取刚完成的第二个条目。测试应覆盖位置 3、4、7、8 等边界；改掉未来 token，不应改变当前位置的输出。当前位置及附近内容还由因果局部窗口保留。

**不可逆的信息损失。** 假设一种简化压缩只保留均值，`[1, 9]` 和 `[5, 5]` 都变成 5。问“最大值是什么”，压缩之后就无法区分。V4 不是简单求均值，这个反例只是说明：压缩必须结合训练和任务验证，不能凭压缩比例推断质量。

实测可以同时放单条证据、多条证据、精确数字、相似干扰和无答案样本。测试设计见 [Benchmark 与长上下文测试](../../07-evaluation/benchmark-protocols.md)。

## 4. mHC 约束残差；Muon 改参数更新

mHC 把残差状态写成多条流，形式为：

$$
X_{\ell+1}=B_\ell X_\ell+C_\ell F_\ell(A_\ell X_\ell).
$$

这里 $X\in\mathbb{R}^{n\times d}$ 有 $n$ 条残差流；$A\in\mathbb{R}^{1\times n}$ 聚合输入，$F$ 仍处理 $d$ 维向量，$C\in\mathbb{R}^{n\times1}$ 把结果送回各流。$B\in\mathbb{R}^{n\times n}$ 的非负元素、行和、列和受到约束；报告用有限次 Sinkhorn 归一化近似实现。[报告 §2.2](https://arxiv.org/html/2606.19348v1#S2.SS2)

用一个**固定**矩阵算：

$$
B=\begin{bmatrix}0.8&0.2\\0.2&0.8\end{bmatrix},
\quad X=\begin{bmatrix}2\\10\end{bmatrix},
\quad BX=\begin{bmatrix}3.6\\8.4\end{bmatrix}.
$$

两条流混合后，和仍是 12，平方和从 104 变成 83.52。一般地，对非负、行列和都为 1 的固定 $B$，由凸性：

$$
\sum_i\left(\sum_j B_{ij}x_j\right)^2
\leq \sum_{i,j}B_{ij}x_j^2
=\sum_jx_j^2.
$$

但这**不是整个模型的收缩证明**：还有 $CF(AX)$，而实际映射也依赖输入。有限次归一化还要检查误差，不能把“理论目标”当浮点实现的恒等式。

Muon 则处理参数更新矩阵，属于 optimizer，不是 attention，也不是给残差矩阵做同一个约束。两者都可能关系到稳定性，但修改的位置完全不同。V4 也不是所有参数都用 Muon：embedding、输出 head、RMSNorm，以及 mHC 的静态 bias / gating factors 等仍用 AdamW。[报告 §2.4](https://arxiv.org/html/2606.19348v1#S2.SS4)

## 5. 后训练：先训练专家，再合并能力

[V4 官方说明](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)区分领域专家的 SFT / GRPO 与之后的 on-policy distillation。此处“专家”是老师模型，不是 MoE 的某个 FFN expert；OPD 也不是直接平均老师权重。

报告的 OPD 在学生自己生成的轨迹上向相应老师对齐，使用 full-vocabulary reverse KL；这不同于原始 R1-Distill 对筛选好的文本做 SFT。[报告 §5.1.2](https://arxiv.org/html/2606.19348v1#S5.SS1.SSS2)

可以用一个自拟的两词词表理解分布比较。学生给 `[0.8, 0.2]`，老师给 `[0.5, 0.5]`：

$$
D_{\mathrm{KL}}(p_s\Vert p_t)
=0.8\log(1.6)+0.2\log(0.4)\approx0.193.
$$

这里用自然对数，单位为 **nats**，且只算**某个前缀下**的 KL；把方向反过来会得到约 0.223 nats，不是同一个目标。完整训练还要决定前缀从哪里来、哪个老师负责、mask、分母和更新方式。学生生成的前缀与老师生成的前缀，会让训练遇到不同状态；不能换掉数据来源，却还把它叫同一个实验。更详细的训练取舍见[知识蒸馏](../../05-post-training/distillation.md)。

## 6. 如果要复现，先检查什么？

| 检查 | 一个小测试 | 失败通常意味着什么 |
| --- | --- | --- |
| 因果性 | 改未来 token，比较此前输出 | 压缩块边界或 mask 错 |
| Prefill / decode 一致性 | 整段与逐步运行比较 | 尾部缓存、位置或状态恢复错 |
| mHC 约束 | 行列和、最小元素、残差映射误差 | 归一化或精度问题 |
| 压缩质量 | 同长度下换证据类型、位置与干扰 | 总分掩盖了具体损失 |
| 部署成本 | 同请求和并发下记录 TTFT、延迟、显存 | FLOPs 不能解释全部延迟 |
| 后训练 | 固定任务、老师版本与生成预算 | 模型、数据和推理预算混在一起 |

本文执行的是记账与数学小测试，不是模型复现。部署前还要核对[官方 inference 说明](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/README.md)、权重精度、message encoder 和运行时版本；不要只复制模型托管页自动生成的一行启动命令。
