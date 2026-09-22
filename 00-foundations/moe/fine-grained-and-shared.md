# MoE：细粒度专家与共享专家

**中文** · [English](fine-grained-and-shared.en.md)

> 阅读时间：约 2 分钟 · 难度：进阶 · 最近审阅：2026-09

## 把 expert 切细

DeepSeekMoE（2024）的第一个改动：把 $N$ 个 expert 各切成 $m$ 份更小的，一共 $mN$ 个，每个 token 激活 $mK$ 个。每个 token 的计算量不变，但可以选择的组合一下子多了很多。

论文里的例子：16 个 expert 选 2 个，只有 120 种组合；切成 64 个选 8 个，组合数是

$$\binom{64}{8} = 4{,}426{,}165{,}368$$

组合越多，每个 expert 就越能只学一小块专门的东西，而不是被迫什么都学一点。

## 共享专家

第二个改动：单独拿出 $K_s$ 个**所有 token 都会经过**的 shared expert，专门放通用知识。这样 routed expert 就不必每个都重复学一遍语法、常见词这类东西，冗余更少。

论文报告 DeepSeekMoE 16B 用大约 40% 的计算量，达到和 LLaMA2 7B 相当的效果。DeepSeek-V3 延续了这两个改动：61 层里前 3 层是 dense，其余每层 1 个 shared expert 加 256 个 routed expert，每个 token 选 8 个，每个 expert 的隐藏维度只有 2048。

## 不是每家都这么做

| 模型 | 每层 routed expert | 每个 token 走几个 | shared expert |
| --- | --- | --- | --- |
| DeepSeek-V3 | 256 | 8 | 1 个 |
| Qwen3-235B-A22B / 30B-A3B | 128 | 8 | 没有 |
| Llama 4 Maverick | 128 | 1 | 1 个（与 dense 层交替） |
| Llama 4 Scout | 16 | 1 | 1 个 |
| gpt-oss-120b / 20b | 128 / 32 | 4 | 没有 |

Qwen3 的 MoE 明确去掉了 shared expert（它的上一代 Qwen2.5-MoE 是有的）；Llama 4 则是每个 token 走 1 个 shared expert 加 128 个 routed expert 里的 1 个。

## 怎么选

细粒度让专长分得更开，代价是 router 要做更多选择、跨卡通信更碎。shared expert 让通用知识有一个固定去处，代价是这部分参数每个 token 都要算。两种设计都在当前的大模型里用着，没有一个统一答案；看到一个新 MoE，先问这两个问题：expert 切得多细，有没有 shared expert。
