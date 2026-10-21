# 回溯与 DP：是在找路径，还是在重复解同一个问题

**中文** · [English](backtracking-and-dp.en.md) · [方法地图](../leetcode.md)

> 阅读时间：约 12 分钟 · 最近审阅：2026-10

看到一道题能用递归，不代表接下来就知道该写回溯还是 DP。可以先想输出：题目是要你列出每一种选法，还是只要一个最优值？如果两条不同的选择路线，最后都在问同一个问题，这个问题的答案能不能只算一次？下面分别用子集和“不选相邻元素”来说明。

## 1. 回溯：选择、往下走、撤销

先列出 `[1, 2]` 的所有子集。空集也算，所以结果是 `[]、[1]、[1, 2]、[2]`。我们用 `path` 放目前选了哪些数，用 `start` 指明接下来从哪个位置开始选。

```text
[]
├─ 选 1 → [1]
│          └─ 再选 2 → [1, 2]
└─ 选 2 → [2]
```

每到一个节点，就已经得到一个子集，不必等到最底下才记答案。每次只往后选：选了 1 之后还能接着选 2；从 2 开始的那一支却不会回头选 1，所以 `[2, 1]` 不会和 `[1, 2]` 重复。

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

这几行最容易混的是 `copy` 和 `pop`。它们不是一回事：

- `copy` 是给答案留一份当前选择的副本。否则所有答案都指向同一个列表，后面改 `path`，前面存的答案也跟着变。
- `pop` 是撤销刚才的选择。走完 `[1, 2]`，先退回 `[1]`；这一支走完再退回 `[]`，才能从空集开始选 2。不撤销，下一支就会带着上一支的东西。

这里假设输入元素互不相同。有重复值时，要另加去重规则。n 个元素有 2ⁿ 个子集，所有子集总共要保存 n·2ⁿ⁻¹ 个元素（n≥1），所以输出空间与时间是 O(n·2ⁿ)。不算输出，当前路径和递归栈只需要 O(n)。

## 2. 3 种题，3 种“下一步”

| 问题 | 下一层怎么走 | 如何避免重复 |
| --- | --- | --- |
| 子集 / 不可重复使用的组合 | 从 index+1 继续 | 按下标单向前进 |
| 元素可重复使用的组合 | 可继续用当前 index | 仍保持选择顺序；剪枝需满足数值前提 |
| 排列 | 每层尝试所有尚未使用的下标 | used 记录当前路径使用情况，回退时撤销 |

剪枝就是提前判断“这条路走下去也不会有答案”，然后不再往下试。比如用正数凑目标和，已经超了，就没必要再加；但允许负数时，后面可能把和减回来，这样剪就会漏解。类似地，不能一看见重复节点就加全局 `visited`：两条不同路径，可能正是题目要的两个答案。

## 3. DP：先把状态说成一句人话

动态规划怎么从递归里来？用 House Robber 看一遍。先不想题目里的故事：给一排非负数 `[2, 7, 9, 3, 1]`，选一些数，不能相邻，希望总和最大。例如选 2、9、1 得到 12。

先定义 `best(index)`：**只看从 index 开始的这一段，最多还能选出多少收益。** 前面的选择已经处理完，不计入这个返回值。

面对当前位置，只有两种情况：

- 不选它：剩下的问题是 `best(index + 1)`。
- 选它：拿到 `values[index]`，下一项不能选，剩下是 `best(index + 2)`。

两边取较大值，就得到递推式。它是把题目允许的选择逐项写出来得到的，不用先猜一个公式、再去想它的意思。

```text
best(0)
├─ 不选位置 0 → best(1)
│                ├─ 不选位置 1 → best(2)
│                └─ 选位置 1   → 7 + best(3)
└─ 选位置 0   → 2 + best(2)
```

图里 `best(2)` 出现了两次。虽然前面选法不同，但接下来都是在问：`[9, 3, 1]` 这一段最多能拿多少？它的答案不会因为之前选了什么而改变，所以可以只算一次，存下来复用。这就是记忆化搜索（memoization）。

<!-- widget:algo-rob -->

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

走到数组外，已经没有东西可选，返回 0；这就是递归终止条件。加上 [cache](https://docs.python.org/3/library/functools.html#functools.cache) 后，每个位置只计算一次，每次比较两个选择，时间 O(n)，缓存和调用栈各占 O(n)。输入不能在计算中改变；缓存放在外层函数内部，每次处理新输入时也会重新创建。

为什么这里只记一个下标就够？因为“不能选相邻项”已经体现在跳过下一项这个动作里。如果规则改成“最多选两项”，同一段数组还会受剩余名额影响，就得把名额也放进状态。**能不能复用，要看余下的问题是否真的相同，不是看函数参数长得像不像。**

## 4. 同一个递推，换成迭代

写迭代版时，先不急着压缩成两个变量，把刚才的结果填进一张表。`best[index]` 仍表示“从这里开始的最大收益”，含义没有换。它依赖右边两个位置，所以**从右往左填**：

<div class="algorithm-trace" markdown="1">

| 位置 | 当前值 | 不选 | 选当前 | 最多收益 |
| --- | --- | --- | --- | --- |
| 5、6 | 已越界 | — | — | 0 |
| 4 | 1 | 0 | 1 + 0 | 1 |
| 3 | 3 | 1 | 3 + 0 | 3 |
| 2 | 9 | 3 | 9 + 1 | 10 |
| 1 | 7 | 10 | 7 + 3 | 10 |
| 0 | 2 | 10 | 2 + 10 | 12 |

</div>

```python
def rob_table(values):
    best = [0] * (len(values) + 2)
    for index in range(len(values) - 1, -1, -1):
        best[index] = max(best[index + 1], values[index] + best[index + 2])
    return best[0]
```

多留的两个 0 对应递归里的越界返回。先把这张表算对，出错时也容易定位：如果位置 2 没算出 10，就看它是否正确读到了位置 3 和 4。

### 最后再省空间

算每一格时，其实只读右边两格，其他结果之后都用不到。用 `one_ahead` 和 `two_ahead` 保存它们，仍然从右往左走：

```python
def rob_iterative(values):
    one_ahead = two_ahead = 0
    for value in reversed(values):
        current = max(one_ahead, value + two_ahead)
        one_ahead, two_ahead = current, one_ahead
    return one_ahead
```

比如处理 9 之前，两个变量分别是 3 和 1；这一轮算出 `max(3, 9 + 1) = 10`，再把它们换成 10 和 3，供下一轮使用。Python 的这次同时赋值先计算右侧，再写到左侧，不会提前覆盖旧的 `one_ahead`。

表格版时间和辅助空间都是 O(n)，滚动变量版时间仍是 O(n)，辅助空间降到 O(1)；`reversed` 也不会复制整个列表。两版只是保存结果的方式不同。如果还要知道具体选了哪些位置，保留表格往往更省事，不能只拿一个最优值就还原完整选择。

## 5. 递归与迭代的取舍

| 写法 | 优势 | 代价 |
| --- | --- | --- |
| 递归 + 记忆化 | 接近问题定义，只访问需要的状态 | 函数调用、缓存、递归深度限制 |
| 从已知结果逐步填表 | 顺序明确，方便检查所有状态 | 可能计算用不到的状态，要排好依赖顺序 |
| 滚动变量 | 内存小、实现简短 | 只适合有限历史依赖，还原路径更麻烦 |

分析 DP 时用“**状态数 × 每个状态的转移成本**”，再加 key 构造和输出成本。不是所有 DP 都 O(n)。例如二维表可能 O(nm)，每个状态还遍历 k 种选择，就可能 O(nmk)。

### 用英语讲一遍这个解法

英语里，**recursion** 和 **iteration** 是名词；描述写法时，用 **a recursive solution** 或 **an iterative solution**。不必翻译整段笔记，先把状态、选择和停止条件讲清楚：

> I define `best(index)` as the maximum sum I can get from that index onward without taking adjacent elements. I either skip the current element, or take it and skip the next one. The base case is zero once the index reaches the end.

接着说为什么要缓存，以及它的成本：

> Different choices can lead to the same subproblem, so I cache each result. There are O(n) states, and each takes O(1) work. That gives O(n) time and O(n) auxiliary space, including the recursion stack.

如果被问到能不能省空间，再解释从右往左填表，以及只保留后两个状态。不用一开口就报最优复杂度，却说不清它从哪里来。

## 6. 少量例题与自测

- [子集](https://leetcode.com/problems/subsets/)：练当前选择的复制和撤销。
- [全排列](https://leetcode.com/problems/permutations/)：区分“顺序不同算不算不同答案”。
- [打家劫舍](https://leetcode.com/problems/house-robber/)：递归 → 记忆化 → 表格 → 滚动变量。
- [零钱兑换](https://leetcode.com/problems/coin-change/)：先说明“凑到这个金额，最少要几枚硬币”，再写状态和转移。

<details class="interview" markdown="1">
<summary>为什么不能给所有回溯直接加 cache，就变成多项式时间？</summary>

缓存要有完整、可复用的状态；当前路径可能影响合法选择和最终输出。即使重复计算能省掉，枚举全部子集仍有指数数量的输出，缓存不能消掉输出成本。

</details>

<details class="interview" markdown="1">
<summary>贪心“每次取当前最大的收益”能解 House Robber 吗？</summary>

不能。`[4, 5, 4]` 取中间 5 不如取两端得到 8。DP 保留互斥选择的最优结果，而不是只看当前最大。

</details>

## 7. 换一种讲法再看

- [Hello 算法：初探动态规划](https://www.hello-algo.com/chapter_dynamic_programming/intro_to_dynamic_programming/)：用爬楼梯把搜索、记忆化、填表和空间优化连起来，可以和本页的例子对照。
- [代码随想录：动态规划理论基础](https://github.com/youngyangyang04/leetcode-master/blob/master/problems/%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%E7%90%86%E8%AE%BA%E5%9F%BA%E7%A1%80.md)：写完以后，可以按状态含义、递推、初值和遍历顺序检查，尤其别省掉手算一张小表。

不同资料可能从前往后定义状态，也可能像本页这样从后往前。两种都可以；改写代码时先确认每个变量的含义，不要只照着换循环方向。
