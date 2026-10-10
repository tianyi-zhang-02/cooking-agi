# MLA 与稀疏注意力：少存一点，还是少看一点？

**中文** · [English](latent-and-sparse-attention.en.md)

> 最近审阅：2026-10 · 前置：[KV cache](kv-cache-and-inference.md)、[RoPE](position-and-context.md)

一段历史很长，推理成本会从两个地方涨：存下来的状态越来越多，每次生成还要读这些状态。压缩状态与减少读取范围，处理的是不同问题。

MLA 主要回答“每个 token 留下多少缓存”；稀疏注意力回答“当前 query 需要看哪些位置”。它们可以组合，但不能互相替代。

## 先看一条压缩后的表示

先省略位置编码和多头，使用列向量。把某个位置的 hidden state 压成 $c_j=W_{\mathrm{down}}h_j$，再定义

$$
k_j=U_Kc_j,\qquad v_j=U_Vc_j.
$$

如果 $c_j$ 比拼起来的 K/V 小，就可能只缓存 $c_j$。这是一种学到的参数化，不是对任意已训练模型的 K/V 做无损压缩。压缩维度会限制可表达的变换。

例如 $c=[2,-1]$，设

$$
\begin{gathered}
U_K=\begin{bmatrix}1&0\\0&2\\1&1\end{bmatrix},\\
k=[2,-2,1]^\top,\\
q=[1,2,-1]^\top.
\end{gathered}
$$

直接点积 $q^\top k=-3$。另一种算法先算 $U_K^\top q=[0,3]^\top$，再与 $c$ 点积，还是 -3。没有必要为了这一项分数先还原出三维 key。

## “吸收矩阵”是交换线性计算的顺序

对一个 head 的内容分数：

$$
q^\top k_j=q^\top U_Kc_j=(U_K^\top q)^\top c_j.
$$

加权 values 也有类似关系：

$$
\sum_j\alpha_jU_Vc_j
=U_V\left(\sum_j\alpha_jc_j\right).
$$

先在 latent 空间聚合，再投影出去。多个 heads 有各自的投影和 attention 权重，不能把它们全混成一条共享的概率分布。

```python
latent = [2.0, -1.0]
key_projection = [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]
query = [1.0, 2.0, -1.0]
key = [sum(weight * value for weight, value in zip(row, latent))
       for row in key_projection]
absorbed_query = [sum(row[column] * query_value
                      for row, query_value in zip(key_projection, query))
                  for column in range(len(latent))]
direct_score = sum(first * second for first, second in zip(query, key))
latent_score = sum(first * second for first, second in zip(absorbed_query, latent))
assert direct_score == latent_score == -3.0
```

这段代码只检查线性恒等式，省略了 softmax 的缩放与位置分支，不是完整 MLA。[DeepSeek-V2](https://arxiv.org/abs/2405.04434)给出了 MLA 的联合 KV 压缩及推理时的投影处理。

## 为什么 RoPE 要单独处理？

<span id="rope-absorption"></span>

不是“MLA 不能用 RoPE”，而是**给展开后的整个 key 做 RoPE，会妨碍刚才那种省计算的重排**。注意力照样能算，只是不一定还能只靠一条压缩后的 query 去读全部历史。

先看矩阵的顺序。位置 $t$ 的 query 乘旋转矩阵 $R_t$，历史位置 $j$ 的 key 乘 $R_j$，分数变成

$$
(R_tq_t)^\top R_jU_Kc_j
=q_t^\top R_t^\top R_jU_Kc_j.
$$

没有旋转时，$U_K^\top q_t$ 算一次就够了。现在若把右侧全部挪到 query 上，得到的却是 $U_K^\top R_j^\top R_tq_t$：它随历史位置 $j$ 改变。我们失去的是“一次投影，复用到所有历史位置”的便利，不是位置编码本身失效。[DeepSeek-V2 §2.1](https://arxiv.org/html/2405.04434v5)据此把内容和位置分开处理。

### 旋转和投影，为什么不能随便换顺序？ {#rotation-order}

用一个两维例子就能看出来。向量是 `[1, 1]`，投影把第二维放大 2 倍，旋转则逆时针转 90°：

| 计算顺序 | 第一步 | 第二步 |
| --- | --- | --- |
| 先投影，再旋转 | `[1, 2]` | `[-2, 1]` |
| 先旋转，再投影 | `[-1, 1]` | `[-1, 2]` |

结果不同。若 query 为 `[1, 0]`，连 attention 的点积分数都会从 -2 变成 -1。这只是展示矩阵顺序的反例；实际 RoPE 的角度由位置和频率决定，并非每个 token 都转 90°。

<details markdown="1">
<summary>用几行 Python 检查这个反例</summary>

```python
def project_two(vector):
    first, second = vector
    return [first, 2 * second]

def quarter_turn(vector):
    first, second = vector
    return [-second, first]

rotated_key = quarter_turn(project_two([1, 1]))
swapped_key = project_two(quarter_turn([1, 1]))
assert rotated_key == [-2, 1]
assert swapped_key == [-1, 2]
```

这里刻意让投影是方阵，才能直接比较两种顺序。真正的 $U_K$ 往往还是长方形矩阵：latent 和 key 的维度不同，连“直接交换”都可能没有定义。特殊结构可以允许某些重排，但不能把它当成任意投影都满足的性质。

</details>

### 分开以后，完整分数怎么算？ {#decoupled-score}

MLA 的内容 key 仍由 latent 线性投影得到；位置 key 走另一条较小的 RoPE 分支。对一个 head，设内容维度为 $d_C$，位置维度为 $d_R$：

$$
\text{score}_{t,j}=
\frac{(U_K^\top q_t^{C})^\top c_j+(q_t^{R})^\top k_j^{R}}
{\sqrt{d_C+d_R}}.
$$

这里的 $q_t^R$、$k_j^R$ 已经做过旋转。**两项先相加，再对可见的历史位置做一次 softmax**，不是内容和位置各算一套概率。比如两条历史的内容分数都是 1，位置分数分别为 0、2；若总维度为 4，缩放后的分数是 0.5、1.5，权重约为 0.269、0.731。位置分支改变了我们更关注哪一条历史。

内容分支可以保留投影重排；缓存里额外留下较小的位置 key。也可以选择缓存完整旋转 key，计算仍然正确，但省缓存的收益会变。具体 kernel 如何处理，不能只看公式就宣布速度更快。

一组**自拟配置**：8 个 heads、每 head 的 K/V 各 64 维，MHA 每 token 每层存 1024 个标量；若 latent 为 128 维、共享位置 key 为 32 维，则需存 160 个标量。这里不含其他缓存、对齐与并行复制。数字只比较状态大小，不代表速度或质量提升。

MLA 也不是 GQA 的另一个名字。GQA 让多个 queries 共享 KV heads；MLA 改变 KV 的低维表示及计算方式。实际是否更快还取决于 kernel、batch 和硬件。

## Decode 时，哪些东西真要留下来？

按 DeepSeek-V2 的分支关系画出来，会比一句“压缩 KV”清楚：

```text
当前 hidden state h
  ├─ KV 下投影 ─→ c_KV ───────────────→ 写入历史缓存
  │               └─ 内容 K/V 的线性投影（可重排计算）
  ├─ 位置 key 投影 ─→ RoPE ─→ k_R ───→ 写入历史缓存
  └─ Query 路径 ─→ 当前 query ─────────→ 读缓存，算本步输出
```

位置 key 来自 hidden state 的单独投影，不是给还原后的整个内容 key 再转一圈；多个 heads 共享这条位置 key。实现可以把两条输入投影合成一次矩阵乘法，但输出仍有不同用途。

| 中间量 | 以后的位置还要用吗？ | 怎么处理 |
| --- | --- | --- |
| 历史 $c_j^{KV}$ | 要，用来算内容分数和聚合 values | 缓存 |
| 历史位置 key $k_j^R$ | 要，用来算相对位置项 | 缓存 |
| 当前 query 的 latent / 投影 | 常规自回归 attention 不需要历史 Q | 本步使用，不属于 KV cache |
| 展开的每头内容 K/V | 数学上会用到，不一定要物化 | 用线性重排避免反复展开 |

这里有两个容易混的点。第一，query 压缩不等于历史 KV 省显存；它们压的是不同对象。第二，矩阵吸收靠的是**线性关系**：在 $c$ 形成之前做归一化可以，在 $U_Vc$ 后面随意插一个非线性函数，就不能再把投影提到求和外面。

两条历史的玩具例子足够看清第二点：values 为 2 和 -2，各占一半。先平均再 ReLU 得到 0；先各自 ReLU 再平均得到 1。因而不能把任意 MLP 都称作“可吸收的 up projection”。

## 稀疏注意力：某些位置根本不参与

设当前 query 原本能看 8 个历史位置。全注意力对它们全部归一化；稀疏方案只选择集合 $\mathcal S$：

$$
o=\sum_{j\in\mathcal S}
\frac{e^{s_j}}{\sum_{k\in\mathcal S}e^{s_k}}v_j.
$$

这通常改变了计算结果，不像 FlashAttention 那样只是重新组织同一个 dense attention。

取所有分数为 0，只有第 2 个位置的 value 为 8，其他为 0。全注意力输出为 1；如果选出的集合漏掉第 2 个位置，输出就是 0。哪怕读取过程快了很多，丢失的证据也不会自动回来。

| 选择方式 | 为什么可能有效 | 容易漏什么 |
| --- | --- | --- |
| 局部窗口 | 相邻内容常有关系 | 很远处的关键事实 |
| 固定全局位置 | 给信息留下跨段通路 | 不在预设位置上的证据 |
| 内容驱动选块 | 按 query 挑相关历史 | 选择器没识别出的少数重要信息 |
| 压缩摘要加局部/选块 | 同时保留粗粒度与细粒度 | 摘要损失、选择成本 |

不能先完整算出所有 dense scores，再称 top-k 省掉了这部分计算。选择器本身也有开销，而且不规则的小读取可能不利于 GPU。

## NSA 的设计可以怎样理解？

[Native Sparse Attention](https://arxiv.org/abs/2502.11089)组合压缩、选择和局部窗口路径，强调训练方式与块状硬件计算配合。它不是“只保留最大的几个 attention 权重”那么简单。

阅读这类论文时，先画出四件事：选择器看到什么、哪些块保留细节、哪些只有压缩表示、是否严格 causal。尤其不要让选择器或压缩块提前包含未来 token；attention 主路径 mask 正确，并不能救选择路径的信息泄漏。

本篇讲机制，不把 NSA 与其他名字相近的稀疏架构当成同一种实现。

### 三条路径，各留住什么？

| 路径 | 读什么 | 留下什么 | 代价与遗漏 |
| --- | --- | --- | --- |
| Compression | 历史块的压缩表示 | 较粗的全局信息 | 压缩会丢细节；块数仍随长度增长 |
| Selection | 挑选块里的原始 K/V | 与当前 query 相关的细节 | 选择和块读取有开销，也可能漏选 |
| Sliding window | 最近的一段 K/V | 连贯的局部上下文 | 看不到窗口以外的完整细节 |

```text
压缩历史 ─→ compression attention ─────────→ o_comp
                      └─ 相关性分数 ─→ 选块 ─→ o_select
最近历史 ──────────────────────────────────→ o_window
                   三路输出分别乘 gate，再相加
```

这不是“先压缩，再挑选，最后加窗口”的单条流水线。选择会借用压缩分数，但三个 attention 分支各有自己的归一化。NSA 用独立 sigmoid gates 混合输出，不能擅自写成三个 gate 必须加起来为 1。

压缩块与选择块也未必一样大。只有块划分对齐时，才能直接把一组压缩分数当作选块分数；否则要做映射。GQA 下的选块还涉及同组 query heads 的信息聚合，不等于先把这些 heads 的 K/V 随便相加。

### 一个不用训练就能检查的错误：未来信息漏进来了

从 0 编号，当前 query 在位置 5。若每块 4 个位置，块 `[0,1,2,3]` 已经完整出现；块 `[4,5,6,7]` 还含未来的 6、7。即使之后只读 4、5，**也不能先把整块 4–7 压成摘要去帮忙选块**。

还有一种更隐蔽的写法：先对全部分数 softmax，再把未来位置乘 0。未来虽然不再直接贡献 value，却已经影响了分母。

下面把正确与错误的写法放在一起。两个可见位置的 value 是 2、4，未来 value 是 999；分数先全设为 0。

```python
import math

def masked_scalar_attention(scores, values, visible, mask_before=True):
    if not (len(scores) == len(values) == len(visible)) or not any(visible):
        raise ValueError("Expected equal lengths and a visible position")
    included = [score for score, keep in zip(scores, visible) if keep or not mask_before]
    shift = max(included)
    weights = [math.exp(score - shift) if keep or not mask_before else 0.0
               for score, keep in zip(scores, visible)]
    return sum(weight * value for weight, value, keep in zip(weights, values, visible) if keep) / sum(weights)

values, visible = [2.0, 4.0, 999.0], [True, True, False]
assert masked_scalar_attention([0, 0, 0], values, visible) == 3.0
assert masked_scalar_attention([0, 0, 0], values, visible, mask_before=False) == 2.0
assert masked_scalar_attention([0, 0, 20], values, visible) == 3.0
assert masked_scalar_attention([0, 0, 20], values, visible, mask_before=False) < 0.001
```

正确输出一直是 3；错误版本连未来的**分数**变了都会跟着变。这个小测试只检查 attention 归一化，不是 NSA 实现。完整模型还应做 future-perturbation test：固定前缀、改动后缀，在关闭 dropout 等随机因素后，前缀 logits 应保持不变。压缩、选块、位置编码和 attention 主路径都要通过。

### 选块和合并输出，实际算一遍

先取压缩块和选择块完全对齐的特例。两个共享 KV 的 query heads 对 3 个块分别给出概率 $[0.6,0.3,0.1]$ 与 $[0.05,0.45,0.5]$。单独选 top-1，会分别读第 0、2 块；组内相加后得到 $[0.65,0.75,0.6]$，共同选第 1 块。共享选择减少不同 heads 读取集合的并集，但也引入取舍：某一个 head 最喜欢的块未必保留。

```python
def shared_block_choice(head_probabilities, count):
    import math

    if not head_probabilities or not head_probabilities[0]:
        raise ValueError("Expected nonempty head distributions")
    width = len(head_probabilities[0])
    if type(count) is not int or not 1 <= count <= width:
        raise ValueError("Invalid selection count")
    for head in head_probabilities:
        if len(head) != width or any(not math.isfinite(value) or value < 0 for value in head):
            raise ValueError("Expected finite nonnegative distributions of equal size")
        if not math.isclose(sum(head), 1.0, abs_tol=1e-9):
            raise ValueError("Each head must sum to one")
    totals = [sum(head[index] for head in head_probabilities) for index in range(width)]
    return sorted(range(width), key=lambda index: (-totals[index], index))[:count]

assert shared_block_choice([[0.6, 0.3, 0.1], [0.05, 0.45, 0.5]], 1) == [1]
assert abs(0.2 * 2 + 0.7 * 6 + 0.4 * 4 - 6.2) < 1e-12
```

代码只处理块对齐的情况。一般情况下，压缩长度 $l$、步幅 $d$ 和选择块长 $l'$ 不同。原论文在 $d$ 能整除两种块长时，把相关压缩块分数聚合为

$$
\begin{gathered}
p^{\mathrm{sel}}_t[j]=\\
\sum_{u=0}^{l'/d-1}\sum_{v=0}^{l/d-1}
p^{\mathrm{cmp}}_t[(l'/d)j+u+v],
\end{gathered}
$$

然后再按共享 KV 的 query 组求和、选块。这里必须跟论文的块编号和边界一致；不能把重叠块的概率直接当作不重叠原始 token 的概率。[NSA §3.3](https://arxiv.org/html/2502.11089v1#S3.SS3)

三条路径各自完成 attention 后，输出为 $g_c o_c+g_s o_s+g_w o_w$。自拟数值 $o=[2,6,4],g=[0.2,0.7,0.4]$ 得到 6.2，超过三个输出中的最大值 6。这没违反公式，因为独立 sigmoid gates 不要求和为 1。擅自归一化 gates 会换成另一种模型。

**训练时发生什么？** 压缩分支本身进入最终输出，因此压缩器有可微的训练路径；离散选中的下标却不会因为叫“端到端训练”就变得处处可微。不要把 NSA 的共享压缩分数，与 DSA 的独立 indexer / KL 训练混写。读实际 kernel 还应核对块布局、padding、梯度、因果边界与各分支的 softmax，而不是只测 forward shape。

## DSA：先用便宜的 indexer 选位置

DeepSeek-V3.2 的 [DeepSeek Sparse Attention](https://arxiv.org/abs/2512.02556) 在 MLA 旁边增加了 lightning indexer。它不是先算完整 MLA attention 再删掉小权重，而是用较便宜的打分选出 top-k 位置，然后主 attention 才读取被选中的 latent K/V。

简化记号下，query 位置 $t$ 对历史位置 $s$ 的 index 分数是：

$$
\begin{gathered}
I_{t,s}=\sum_h w^I_{t,h}\operatorname{ReLU}\big((q^I_{t,h})^\top k^I_s\big),\\
\mathcal S_t=\operatorname{TopK}_{s\le t}(I_{t,s}).
\end{gathered}
$$

Indexer 和主 attention 各自有投影。小 head 数与低精度计算减少筛选成本；选中的 latent 条目由主 MLA 的 query heads 共享。$I$ 是选位置的分数，不直接代替主 attention 的 softmax 权重。

### Indexer 学什么？

| 阶段 | 主模型 | Indexer 的监督 |
| --- | --- | --- |
| Dense warm-up | 冻结其余参数，保留 dense attention | 对齐聚合、归一化后的主 attention 分布 |
| Sparse training | 主模型按 LM loss 更新 | 在选中集合上继续对齐 attention 分布 |

论文的 indexer 输入会 detach，indexer 使用自己的 KL 监督。不能把 hard top-k 想象成普通可微的 softmax，然后认为 LM loss 会直接穿过选位操作训练所有 index 分数。

### 一次错选会丢掉什么？

用已有的 `masked_scalar_attention` 做一个小实验。主 attention 认为第三个位置最重要，但 indexer 恰好把它排到 top-2 外。

```python
def selected_positions(index_scores, count):
    if type(count) is not int or not 0 < count <= len(index_scores):
        raise ValueError("Expected a valid selection count")
    if any(not math.isfinite(score) for score in index_scores):
        raise ValueError("Expected finite index scores")
    return sorted(range(len(index_scores)), key=lambda position: (-index_scores[position], position))[:count]

chosen = selected_positions([3, 2, 1, 0], 2)
attention_scores = [0, 0, 4, 0]
values = [0, 0, 10, 0]
dense_output = masked_scalar_attention(attention_scores, values, [True] * 4)
sparse_output = masked_scalar_attention(attention_scores, values, [position in chosen for position in range(4)])
assert dense_output > 9
assert sparse_output == 0
```

这不是 DSA 效果差的证据，只说明选择器的 recall 也是质量瓶颈。只量主 kernel 快了多少，不足以评价系统。

### 复杂度不要只算一半

长度为 $L$、每个 query 选 $k$ 个位置，主 attention 的 pair 数由约 $L^2$ 降到 $Lk$。但对每个 query 扫描历史的 indexer 在 prefill 仍有二次项；因此不能笼统说整个 DSA 是线性复杂度。短序列时，筛选与不规则读取的额外成本还可能抵消节省。

少读位置也不等于少存全部历史。后续 query 可能选到别的位置，MLA 历史 cache 仍要保留，另外还存在 indexer 状态。NSA 的 block 选择与 DSA 的 token 选择，计算路径和硬件取舍并不相同。

实现检查要包含 causal selection、top-k 边界、cache 与位置编码。[V3.2-Exp 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) 记录过 indexer 与 MLA 使用不同 RoPE layout 的修正：shape 都正确，也可能算错旋转。本文核对到 2026-10-08，讲的是 V3.2 路径，不把它自动套到后续 DeepSeek 版本。

## 怎样比较才不容易自我误导？

MLA 要测 latent 维度、质量、KV 大小和 decode 速度；稀疏注意力还要专门测远距离证据、少数关键 token、选择开销与端到端速度。

固定上下文长度、有效 batch、dtype 和硬件；记录预填充与解码各自的耗时。短上下文可能不足以抵消选择开销，长上下文也可能在质量上暴露新问题。

压缩、稀疏、低精度和更好的 kernel 是不同的改动。先一次改一项，才知道节省来自哪里，以及代价是什么。
