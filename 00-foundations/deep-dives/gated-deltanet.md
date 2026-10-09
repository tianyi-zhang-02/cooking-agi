# Gated DeltaNet：先忘一点，再按误差更新记忆

**中文** · [English](gated-deltanet.en.md)

KV cache 会为历史 token 保留记录。循环模型换了一种做法：每读一个 token，就更新一个固定形状的状态。这样 decode 不必反复扫描完整历史，但也失去了逐条回看的能力。Gated DeltaNet 研究的是：这块有限的状态应该怎么写、怎么改、什么时候忘。

## 把状态当成一个小的关联记忆

设 $S\in\mathbb R^{d_v\times d_k}$。key $k$ 和 query $q$ 是 $d_k$ 维列向量，value $v$ 是 $d_v$ 维。读记忆就是 $Sq$。

最直接的写法是 $S_t=S_{t-1}+v_tk_t^\top$。但同一个 key 多次出现，会不断叠加，而不是把旧值更新成新值。例如一维状态里先写 2，再写 6，最后读出 8；如果希望记住最新的 6，就写错了。

Delta rule 先读出当前 key 对应的旧值，再写入差值：

$$
S_t=S_{t-1}+\beta_t(v_t-S_{t-1}k_t)k_t^\top.
$$

在 $\|k_t\|_2=1$ 时，更新后沿这个 key 读到的值为

$$
S_tk_t=(1-\beta_t)S_{t-1}k_t+\beta_tv_t.
$$

也就是在旧值与新值之间插值。$\beta=1$ 时完全替换该方向上的值；对与 $k_t$ 正交的方向，本次修正不产生影响。**相近但不正交的 key 仍会互相干扰**，这不是一个精确字典。

## 为什么再加一个遗忘门？

只修正当前 key，不方便快速清理其他过时内容。Gated DeltaNet 先用 $\alpha_t$ 衰减状态，再对衰减后的读取误差做修正：

$$
\bar S_t=\alpha_tS_{t-1},\qquad
S_t=\bar S_t+\beta_t(v_t-\bar S_tk_t)k_t^\top.
$$

等价地：

$$
S_t=\alpha_tS_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top,\qquad o_t=S_tq_t.
$$

别漏掉误差里的 $\alpha$。先忘了部分旧值，就必须针对**忘过之后**的预测来纠错。

从优化角度也能得到这个式子：把 $\bar S_t$ 当起点，对局部平方损失 $\frac12\|Sk_t-v_t\|^2$ 做一次步长为 $\beta_t$ 的梯度更新。这里更新的是一次 forward 里的快速状态，不是给主模型参数偷偷做了一次 optimizer step；主模型仍由外层训练学习投影与门。

## 手算两格记忆

取 $S=[2,10]$、$k=[1,0]^\top$、新值 $v=6$，$\alpha=0.5,\beta=0.25$。

| 步骤 | 结果 |
| --- | --- |
| 衰减全部状态 | $[1,5]$ |
| 用 key 读取 | $1$ |
| 新旧误差 | $6-1=5$ |
| 沿 key 写入修正 | $0.25\times5[1,0]=[1.25,0]$ |
| 新状态 | $[2.25,5]$ |

第二格没有被本次 delta 修正，却仍受全局遗忘影响。如果错误地用衰减前的 2 算误差，会得到 `[2,5]`，与公式不一致。

```python
import math

def gated_delta_step(state, key, value, decay, write):
    if not key or len(state) != len(value) or not state:
        raise ValueError("Invalid state, key, or value shape")
    if any(len(row) != len(key) for row in state):
        raise ValueError("State rows must match key dimension")
    scalars = [decay, write, *key, *value, *(entry for row in state for entry in row)]
    if any(not math.isfinite(entry) for entry in scalars):
        raise ValueError("Expected finite inputs")
    if not 0 <= decay <= 1 or not 0 <= write <= 1:
        raise ValueError("This example uses gates in [0, 1]")
    if not math.isclose(sum(entry * entry for entry in key), 1.0, abs_tol=1e-9):
        raise ValueError("This example assumes a unit key")
    decayed = [[decay * entry for entry in row] for row in state]
    prediction = [sum(entry * key_entry for entry, key_entry in zip(row, key)) for row in decayed]
    return [[entry + write * (target - old) * key_entry
             for entry, key_entry in zip(row, key)]
            for row, target, old in zip(decayed, value, prediction)]

state = gated_delta_step([[2.0, 10.0]], [1.0, 0.0], [6.0], 0.5, 0.25)
assert state == [[2.25, 5.0]]
assert gated_delta_step([[2.0, 10.0]], [1.0, 0.0], [6.0], 1.0, 1.0) == [[6.0, 10.0]]
```

这是可复算的递归式，不包括输入投影、短卷积、归一化、输出门和 GPU kernel。0/1 端点用于观察极限行为；论文与具体模型的门参数化要另查。某些扩展允许更大的 write 范围，不能直接套用本例的插值解释。

## Decode 递归，训练不必逐 token 写 Python 循环

每一步都可写成 $S_t=S_{t-1}A_t+B_t$。连续两步得到

$$
S_{t+1}=S_{t-1}A_tA_{t+1}+B_tA_{t+1}+B_{t+1}.
$$

组合是有序的，矩阵不能随意交换。论文使用分块与结构化变换，把块内工作变成适合硬件的矩阵运算。它不是把所有依赖删掉，也不是把上面的慢循环直接当高吞吐训练实现。

实际实现应该让单步递归、分块 prefill、保存状态后继续运行三种路径对齐。连续两个独立样本必须重置状态；同一请求接着 decode 则不能重置。把这些边界弄反，模型会在样本间泄漏或忘掉前缀。

## 固定状态到底省了什么？

假想每层 16 个 head，每个状态是 $128\times128$，FP32 保存：共 1 MiB，与已经读了多少 token 无关。另一个假想 attention 层有 16 个 KV head，head dimension 128，BF16 保存 8,192 个位置的 K 和 V：共 64 MiB。

这不是同能力模型的公平性能对比，只说明两个缓存项的增长方式。循环状态也占内存，训练还要保存或重算激活；混合架构里的完整 attention 层仍有随长度增长的 KV cache。

| 想得到的性质 | 相应代价 |
| --- | --- |
| 固定形状的 decode 状态 | 历史压缩，不能保证精确回忆每个 token |
| Delta 定向纠错 | 相关 key 有干扰，容量受维度与训练限制 |
| 全局遗忘 | 清理快，但也可能抹掉仍有用的信息 |
| 与 attention 混用 | 保留部分精确读取，同时带回其缓存和计算成本 |

核对日期：2026-10-08。依据 [Gated Delta Networks](https://arxiv.org/html/2412.06464v2) 与[作者代码](https://github.com/NVlabs/GatedDeltaNet)。完整训练和 GPU 速度未复现。接着读 [Qwen 家族](../model-families/qwen.md)，看它怎样与 attention、FFN / MoE 配合。
