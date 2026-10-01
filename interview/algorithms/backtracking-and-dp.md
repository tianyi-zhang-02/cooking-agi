# 回溯与 DP：是在找路径，还是在重复解同一个问题

**中文** · [English](backtracking-and-dp.en.md) · [方法地图](../leetcode.md)

> 阅读时间：约 8 分钟 · 最近审阅：2026-09

这两类题都可能画出递归树，但关注点不一样。回溯要把合法选择走出来；DP 要把重复子问题的结果存下来。它们可以结合，不能只靠“有没有递归”来区分。

## 1. 回溯：选择、往下走、撤销

枚举 `[1, 2]` 的子集，结果是 `[]、[1]、[1,2]、[2]`。路径 path 只表示当前选择；start 保证只往后选，不把 `[1,2]` 和 `[2,1]` 都算成一个子集的不同答案。

```python
def subsets(values):
    result = []
    path = []

    def extend(start):
        result.append(path.copy())
        for index in range(start, len(values)):
            path.append(values[index])
            extend(index + 1)
            path.pop()

    extend(0)
    return result
```

假设输入元素互不相同。存答案必须 copy，不然所有答案都指向同一个会继续变化的 path。辅助空间 O(n)，输出空间和时间 O(n·2ⁿ)；输出每一个子集本身就不便宜。存在重复元素时，需要额外去重规则，这个模板不自动处理。

## 2. 3 种题，3 种“下一步”

| 问题 | 下一层怎么走 | 如何避免重复 |
| --- | --- | --- |
| 子集 / 不可重复使用的组合 | 从 index+1 继续 | 按下标单向前进 |
| 元素可重复使用的组合 | 可继续用当前 index | 仍保持选择顺序；剪枝需满足数值前提 |
| 排列 | 每层尝试所有尚未使用的下标 | used 记录当前路径使用情况，回退时撤销 |

**剪枝不是想当然地跳过。** 若都是正数，剩余目标已经小于 0，可以停；有负数时，这个理由不成立。全局 visited 也不能随便加，因为不同路径可能是不同输出。

## 3. DP：先把状态说成一句人话

House Robber 的简化模型：一排非负收益，不能取相邻两项。`best(index)` 表示“从 index 开始，最多还能拿多少”。两种选择：跳过当前，或拿当前并跳过下一项。

```python
from functools import cache

def rob_memo(values):
    @cache
    def best(index):
        if index >= len(values):
            return 0
        return max(best(index + 1), values[index] + best(index + 2))

    return best(0)
```

没有缓存时，很多 index 被反复计算；加 [cache](https://docs.python.org/3/library/functools.html#functools.cache) 后，每个状态只解一次。时间 O(n)，缓存与调用栈 O(n)。输入在计算中不能改变；这里 cache 放在函数内部，每次调用重新创建，避免串到另一组输入。

## 4. 同一个递推，换成迭代

从左往右：one_back 是已经处理的前缀最优值，two_back 是再少一项的前缀最优值。新值只有“跳过”或“拿了加上 two_back”两种选择。

```python
def rob_iterative(values):
    two_back = one_back = 0
    for value in values:
        current = max(one_back, two_back + value)
        two_back, one_back = one_back, current
    return one_back
```

`[2, 7, 9, 3, 1]` 的前缀最优值依次是 `2, 7, 11, 11, 12`。时间 O(n)，辅助空间 O(1)。压缩状态后只返回最优值；如果要还原具体选了哪些位置，通常要保留更多信息或重新计算。

## 5. recursive / iterative 的取舍

| 写法 | 优势 | 代价 |
| --- | --- | --- |
| 递归 + memoization | 接近问题定义，只访问需要的状态 | 函数调用、cache、递归深度限制 |
| 表格 bottom-up | 顺序明确，方便检查所有状态 | 可能计算用不到的状态，要排好依赖顺序 |
| 滚动变量 | 内存小、实现简短 | 只适合有限历史依赖，还原路径更麻烦 |

分析 DP 时用“**状态数 × 每个状态的转移成本**”，再加 key 构造和输出成本。不是所有 DP 都 O(n)。例如二维表可能 O(nm)，每个状态还遍历 k 种选择，就可能 O(nmk)。

## 6. 少量例题与自测

- [Subsets](https://leetcode.com/problems/subsets/)：练 path 的复制和撤销。
- [Permutations](https://leetcode.com/problems/permutations/)：区分“顺序不同算不算不同答案”。
- [House Robber](https://leetcode.com/problems/house-robber/)：递归 → cache → 表格 → 滚动变量。
- [Coin Change](https://leetcode.com/problems/coin-change/)：把状态写成“凑到某金额的最少硬币数”，再算状态与转移。

<details class="interview" markdown="1">
<summary>为什么不能给所有回溯直接加 cache，就变成多项式时间？</summary>

缓存要有完整、可复用的状态；当前路径可能影响合法选择和最终输出。即使重复计算能省掉，枚举全部子集仍有指数数量的输出，缓存不能消掉输出成本。

</details>

<details class="interview" markdown="1">
<summary>贪心“每次取当前最大的收益”能解 House Robber 吗？</summary>

不能。`[4, 5, 4]` 取中间 5 不如取两端得到 8。DP 保留互斥选择的最优结果，而不是只看当前最大。

</details>
