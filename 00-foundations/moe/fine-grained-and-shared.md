# MoE：细粒度专家与共享专家

**中文** · [English](fine-grained-and-shared.en.md)

> 阅读时间：约 2 分钟 · 难度：进阶 · 最近审阅：2026-09

## 把 expert 切细

DeepSeekMoE（2024）的第一个改动：把 $N$ 个 expert 每个再切成 $m$ 份小的，一共 $mN$ 个，每个 token 激活 $mK$ 个。计算量没变，能挑的组合却一下子多了很多。

论文里的例子：16 个 expert 选 2 个，只有 120 种组合；切成 64 个选 8 个，组合数是

$$\binom{64}{8} = 4{,}426{,}165{,}368$$

组合多了，每个 expert 才有机会只钻一小块，而不是什么都得会一点。

## 共享专家

第二个改动：单独留出 $K_s$ 个**所有 token 都要经过**的 shared expert，专门装通用知识。语法、常见词这类东西有了固定去处，routed expert 就不用每个都重学一遍。

论文报告 DeepSeekMoE 16B 用大约 40% 的计算量，做到了和 LLaMA2 7B 相当的效果。DeepSeek-V3 把这两个改动都留了下来：61 层里前 3 层是 dense，其余每层 1 个 shared expert 加 256 个 routed expert，每个 token 挑 8 个，每个 expert 的隐藏维度只有 2048。

## 不是每家都这么做

| 模型 | 每层 routed expert | 每个 token 走几个 | shared expert |
| --- | --- | --- | --- |
| DeepSeek-V3 | 256 | 8 | 1 个 |
| Qwen3-235B-A22B / 30B-A3B | 128 | 8 | 没有 |
| Llama 4 Maverick | 128 | 1 | 1 个（与 dense 层交替） |
| Llama 4 Scout | 16 | 1 | 1 个 |
| gpt-oss-120b / 20b | 128 / 32 | 4 | 没有 |

Qwen3 的 MoE 明确把 shared expert 去掉了（上一代 Qwen2.5-MoE 还有）；Llama 4 则是每个 token 走 1 个 shared expert，外加 128 个 routed expert 里的 1 个。

## 怎么选

切细了，专长分得更开，代价是 router 要挑更多次、跨卡通信更碎。shared expert 给通用知识安排了固定去处，代价是这部分参数每个 token 都得算。两种设计现在都有人用，没有标准答案。看到一个新的 MoE，先问两句：expert 切得多细？有没有 shared expert？
