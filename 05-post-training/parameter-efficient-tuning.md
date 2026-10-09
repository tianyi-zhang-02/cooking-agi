# Prompt、Prefix、Adapter：冻结模型后，还能改哪里？

**中文** · [English](parameter-efficient-tuning.en.md)

> 最近审阅：2026-10 · 前置：[模型适配](model-adaptation.md)、[LoRA / QLoRA](lora-and-qlora.md)

假设已经有一个模型，希望它把客服记录整理成固定格式。数据和 loss 暂时相同，只比较允许哪些参数变化。这样才能分清：效果差异来自更新位置，还是来自换了训练任务。

PEFT 是一类参数高效微调方法，不是一种 loss。它可以配合 SFT，也可能配合其他目标；“冻结基座”也不等于“不需要穿过基座反向传播”。

## 先画出四个可以动的位置

```text
输入 embedding： [可学习的 soft prompt] + [文本]
每层 attention： [可学习的 prefix K/V] + [文本的 K/V]
层内表示：       h → 小型非线性 adapter → h'
线性权重：       W0 → W0 + 低秩更新 BA
```

这些只是机制示意。实际插入位置、共享方式和参数化随实现不同，不能只看方法名就知道训练了什么。

## Prompt Tuning：在输入前学一小段向量

设模型宽度 $d$，加入 $m$ 个可学习向量 $P\in\mathbb R^{m\times d}$：

$$
H_0=[P;E(x)],\qquad \hat y=f_{\theta_0}(H_0).
$$

只有 $P$ 更新，$\theta_0$ 冻结。这不是人写的一段提示词；soft prompt 通常没有一一对应的可读文本。[Prompt Tuning](https://arxiv.org/abs/2104.08691)研究了这种适配方式。

取 $m=16,d=512$，输入 prompt 有 8192 个参数。但模型仍要处理这些额外位置，梯度仍要从输出传回输入。训练参数少，不代表中间激活也按相同比例缩小。

decoder 场景下，固定 prompt 的部分状态可以预计算；具体能缓存什么要看任务、位置和实现。不能因为有这项优化，就忽略它占用的上下文和 attention 工作。

## Prefix Tuning：在每层提供额外的 Key / Value

简化为每层直接学习 $m$ 对 prefix K/V：

$$
\operatorname{Attn}\left(Q,[K_P;K_x],[V_P;V_x]\right).
$$

正常 token 的 query 可以读这些位置。它不是只在输入前放一段 embedding，影响模型的入口更深。[Prefix-Tuning](https://arxiv.org/abs/2101.00190)的训练参数化还可以使用额外网络，不能把最终 prefix 状态大小当作所有实现的训练参数总量。

若有 $L=6$ 层、每层 K/V 总宽度均为 $d=512$，直接存储的 prefix 为 $2Lmd=98,304$ 个标量。GQA 下应使用实际 KV 宽度，不要把 query 宽度硬代进去。

它不需要生成这些虚拟位置的普通输出，但它们仍参与 attention 和缓存。初始化、prefix 长度和层间参数化都值得做对照，而不是只调学习率。

## Adapter：在层内加一条非线性小支路

一种瓶颈 adapter 写作

$$
h'=h+W_{\mathrm{up}}\phi(W_{\mathrm{down}}h),
$$

其中 $W_{\mathrm{down}}\in\mathbb R^{r\times d}$，$W_{\mathrm{up}}\in\mathbb R^{d\times r}$。它先压窄，再变换，再加回输入。[Houlsby 等人的 adapter](https://arxiv.org/abs/1902.00751)是这种思路的经典工作，具体放置位置可以不同。

忽略 bias，单个模块需要 $2dr$ 个参数。$d=512,r=8$ 时是 8192 个。因为中间有非线性，通常不能像一个线性 LoRA 增量那样直接合并进原矩阵，部署会多一段计算。

压得过窄可能表达不够；每层都插则增加开销。近似恒等初始化有助于起步，但两边矩阵都设成零可能使这条支路学不动，初始化也要核对梯度。

## 同一小模型，四种方法的账怎样列？

仍取 $L=6,d=512,m=16,r=8$。下表只是参数化计算，不是效果排行榜：

| 方法 | 本例约定 | 可训练参数 / 状态规模 |
| --- | --- | ---: |
| Prompt | 输入处一组 $m\times d$ | 8,192 |
| Prefix | 每层直接学习 K/V，KV 宽度为 $d$ | 98,304 |
| Adapter | 每层 1 个瓶颈模块，不计 bias | 49,152 |
| LoRA | 每层只改一个 $d\times d$ 矩阵 | 49,152 |

Adapter 与 LoRA 数量相同，不代表函数空间相同。若把 LoRA 加到 Q、K、V、O 和 FFN 的多个矩阵，参数自然更多；比较时必须列出 target modules。

## 冻结权重后，梯度为什么还要穿过模型？

Prompt 在最前面，loss 在最后面，需要通过链式法则求 $\partial\mathcal L/\partial P$。如果把整个模型放进 no-grad，通常也就切断了这条路。冻结权重应控制参数是否更新，不是随意切断输入到 loss 的计算图。

Adapter / LoRA 同样需要经过前后算子传播梯度。可以避免保存某些不需要的量，也能使用 checkpointing，但节省多少取决于图结构，详见[精度与显存](../00-foundations/deep-dives/precision-and-memory.md)。

## 选择之前，先做一个公平的小实验

固定数据、目标、tokenizer、训练 tokens 与调参预算。分别记录任务质量、通用能力回归、可训练参数、峰值训练显存、推理延迟和每任务存储。

如果任务只是固定格式，先比较一个好的普通 prompt；如果需要大幅改变能力或补大量领域知识，不要假定极小的适配器一定足够。多任务部署还要测试切换时的缓存失效、批处理和版本绑定。

最后的选择应该能解释：需要改变什么行为、在哪一层施加变化最合适、实际节省了什么。不是谁的 checkpoint 最小，就一定最好。
