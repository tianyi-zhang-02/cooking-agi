# MoE：细粒度专家与共享专家

**中文** · [English](fine-grained-and-shared.en.md)

> 阅读时间：约 6 分钟 · 难度：进阶 · 最近审阅：2026-10

## 把 expert 切细

[DeepSeekMoE（2024）](https://arxiv.org/abs/2401.06066)把专家中间层缩窄：原来 $N$ 个，现在有 $mN$ 个，每个 token 从激活 $K$ 个改成 $mK$ 个。这里保持的是主要的 expert 矩阵乘法预算，不是整个模型的运行时间。

论文里的例子：16 个 expert 选 2 个，只有 120 种组合；切成 64 个选 8 个，组合数是

$$\binom{64}{8} = 4{,}426{,}165{,}368$$

组合更多，给专家分工留下了更多可能；至于有没有学出有用的分工，还得看训练结果。这也不是把一个训练好的大 expert 切几刀，就能无损变成几个小 expert。

## 先把“预算一样”算明白

用一组小尺寸算账：hidden size 为 128，SwiGLU expert 的中间维度为 256。忽略 bias，每个 expert 有 $3df=98,304$ 个权重，每 token 主要做同样数量的乘加（MAC）。这里 1 次 MAC 不等于 1 FLOP；若乘和加各算一次，约为 2 FLOPs。

| 自拟方案 | 每个 expert 的宽度 | 总 expert 数 | 每 token 激活 | Expert 总参数 | 激活参数 / 主要 MAC |
| --- | --- | --- | --- | --- | --- |
| 较宽的 experts | 256 | 8 | 2 | 786,432 | 196,608 |
| 切成 4 倍数量 | 64 | 32 | 8 | 786,432 | 196,608 |
| 其中 1 个改为 shared | 64 | 31 routed + 1 shared | 7 routed + 1 shared | 786,432 | 196,608 |

```python
def expert_budget(width, intermediate, routed, top_k, shared=0):
    if min(width, intermediate, routed, top_k) < 1 or shared < 0 or top_k > routed:
        raise ValueError("Invalid expert configuration")
    per_expert = 3 * width * intermediate
    return per_expert * (routed + shared), per_expert * (top_k + shared)

coarse = expert_budget(128, 256, 8, 2)
fine = expert_budget(128, 64, 32, 8)
with_shared = expert_budget(128, 64, 31, 7, shared=1)
assert coarse == fine == with_shared == (786432, 196608)
```

第三行很容易算错：**共享专家也要算钱**。如果保留 8 个 routed 再额外加 1 个 shared，激活部分就变成 221,184，比前两行多 12.5%。拿它跟原方案比，不能把所有收益都归因于“共享”这个设计。

这张表没算 router、激活函数、attention 和通信。小矩阵更碎、派发次数更多，可能让相同 MAC 跑得更慢；总参数一样，也不代表每张卡的峰值显存一样。

## 共享专家

第二个改动：留出 $K_s$ 个**所有 token 都要经过**的 shared expert，让共享计算有一条固定路径，减少 routed experts 重复学习的需要。这是设计意图，不是给专家规定好了知识分区；不能保证“语法一定在 shared，数学一定在某个 routed expert”。

论文报告 DeepSeekMoE 16B 用大约 40% 的计算量，做到了和 LLaMA2 7B 相当的效果。DeepSeek-V3 把这两个改动都留了下来：61 层里前 3 层是 dense，其余每层 1 个 shared expert 加 256 个 routed expert，每个 token 挑 8 个，每个 expert 的隐藏维度只有 2048。

## 不是每家都这么做

| 模型 | 每层 routed expert | 每个 token 走几个 | shared expert |
| --- | --- | --- | --- |
| DeepSeek-V3 | 256 | 8 | 1 个 |
| Qwen3-235B-A22B / 30B-A3B | 128 | 8 | 没有 |
| Llama 4 Maverick | 128 | 1 | 1 个（与 dense 层交替） |
| Llama 4 Scout | 16 | 1 | 1 个 |
| gpt-oss-120b / 20b | 128 / 32 | 4 | 没有 |

[Qwen3 技术报告](https://arxiv.org/abs/2505.09388)明确说明，相比报告中的 Qwen2.5-MoE，Qwen3-MoE 不再使用 shared experts。Llama 4 Maverick 则每个 token 走 1 个 shared expert，外加 128 个 routed expert 里的 1 个。比较时记下具体 checkpoint，别只记家族名。

## 怎么选

想试细粒度专家，先固定总 expert 参数和激活预算，再看验证 loss、实际吞吐与负载分布。想试共享专家，就明确它是从原预算里拿出一份，还是额外增加一份。

不要只看组合数大不大。一个方案如果把常见 token 分得很好，却让低频领域退步，或者让跨卡通信拖慢整个 step，也不一定值得换。架构提供选择空间，实验才告诉我们这些选择有没有被用好。
