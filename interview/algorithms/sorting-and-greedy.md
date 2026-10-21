# 排序、区间与贪心：这一步为什么不会后悔

**中文** · [English](sorting-and-greedy.en.md) · [方法地图](../leetcode.md)

> 阅读时间：约 6 分钟 · 最近审阅：2026-10

排序本身不是答案，它给后面的扫描创造顺序。贪心也不只是“选个看起来最好的”：你得说明这个局部选择不会把更好的答案排除掉。

## 1. 排序换来了什么

对于区间，按起点排序后，只需要比较当前区间和最后一个合并结果，不必与所有旧区间逐个比较。对于不重叠活动选择，常见的是按**结束时间**排序：结束越早，给后面留下的空间越多。

两者排序 key 不同，因为目标不同。`sorted(intervals, key=lambda interval: interval[0])` 看起点；`key=lambda interval: interval[1]` 看终点。别把“区间题”当成一种固定模板。

## 2. 合并区间：只盯着最后一段

这里输入合法的闭区间 `[start, end]`，start≤end；端点相接也合并。

```python
def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged
```

`[[5,7], [1,3], [2,4]]` 排成 `[1,3]、[2,4]、[5,7]`，扫描后得到 `[[1,4], [5,7]]`。当前区间若连最后一段都碰不到，就更碰不到前面已结束的段。

时间 O(n log n)，排序副本 O(n)，输出最坏 O(n)，不修改输入。若输入已按起点排序，可以省掉排序，扫描 O(n)。半开区间 `[start,end)` 是否合并相接部分要看题目语义，别忘了边界规则。

## 3. 贪心：用可达前沿压缩选择

Jump Game 的输入是非负整数，每个数表示从该位置最多往前跳几步。只问能不能到末尾，不问具体路径或最少跳几次。

```python
def can_jump(jumps):
    if not jumps:
        return False
    farthest = 0
    for index, jump in enumerate(jumps):
        if index > farthest:
            return False
        farthest = max(farthest, index + jump)
        if farthest >= len(jumps) - 1:
            return True
    return False
```

`[2,3,1,1,4]`：处理位置 0 后，最远可达 2；位置 1 在已知可达范围内，把前沿扩到 4，成功。`[3,2,1,0,4]`：前沿一直停在 3，位置 4 不可达。

<!-- widget:algo-jump -->

**不变量**：已经处理过的可达位置，能覆盖从 0 到 farthest 的整段位置。因为每次允许跳“至多”那么远，中间不会凭空出现洞。时间 O(n)、空间 O(1)。如果规则变成必须恰好跳给定步数、可以后退或存在复杂障碍，这个证明就不成立。

## 4. 贪心怎么讲得让人相信

- **不变量**：像 Jump Game，说明留下的摘要包含所有后续需要的信息。
- **交换论证**：例如最多安排几个互不重叠活动。把一个最优方案的第一个活动换成结束最早的，不会更晚结束，所以后面的活动仍能安排；递归应用到剩下的问题。
- **反例**：面额 `[1,3,4]` 凑金额 6，先拿最大面额得到 `4+1+1`，不如 `3+3`。没有证明时先找这种小反例。

### 用五个活动走一遍交换论证

活动用半开区间 `[start, end)` 表示：一个活动在 6 结束，下一个可以从 6 开始。手上有 A `[0,7)`、B `[1,3)`、C `[2,4)`、D `[4,6)`、E `[6,8)`，问最多能排几个互不重叠的活动。按结束时间排好，只记“上一个已选活动在什么时刻结束”：

| 按结束时间的顺序 | 区间 | 能不能接在上一个后面 | 已选 |
| --- | --- | --- | --- |
| B | `[1,3)` | 还没选过，选 | B |
| C | `[2,4)` | 2 早于 3，冲突，跳过 | B |
| D | `[4,6)` | 4 不早于 3，选 | B、D |
| A | `[0,7)` | 0 早于 6，跳过 | B、D |
| E | `[6,8)` | 6 不早于 6，选 | B、D、E |

结果是 3 个。C、D、E 也是一种 3 个的排法。把它的第一项 C 换成结束最早的 B：B 在 3 结束，比 C 的 4 还早，D 照样能从 4 开始，数量没变。换完以后，剩下的问题是“从时刻 3 以后最多还能排几个”，贪心在这个更小的问题上做同样的选择，同样的论证再用一次。完整证明见 [Kleinberg–Tardos 贪心讲义的区间调度一节](https://www.cs.princeton.edu/~wayne/kleinberg-tardos/pdf/04GreedyAlgorithmsI.pdf)。

换一个看起来也合理的规则，就会出错。按最早开始选，会先拿 A：它占住 0 到 7，其余活动最晚也在 6 开始，全都和它冲突，只能排 1 个。按最短时长选也有反例：`[0,4)`、`[4,8)`、`[3,5)` 里先拿最短的 `[3,5)`，另外两个都会被挡住，只排 1 个，而最优是 2 个。

排序 O(n log n)，扫描 O(n)，只多存一个结束时刻。如果活动有不同的权重，要最大化总权重而不是个数，这个交换就不成立了：换进来的 B 可能权重很小。那时要按结束时间排序后做 DP，见 [同一套讲义的动态规划部分](https://www.cs.princeton.edu/~wayne/kleinberg-tardos/pdf/06DynamicProgrammingI.pdf)。

合并区间是排序扫描；并不是凡是“排序后扫一遍”都在做最优化贪心。能准确说出自己在证明什么，比把名字报对更重要。

## 5. 例题和第一遍的范围

- [Merge Intervals](https://leetcode.com/problems/merge-intervals/)：练闭区间重叠、嵌套、空输入。
- [Jump Game](https://leetcode.com/problems/jump-game/)：只存可达前沿，解释为什么足够。
- [Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)：维护此前最小价格，算今天卖出的最好利润；不能先卖后买。

第一遍先把常见扫描、查找、遍历、状态方法练熟。[任务依赖练习](../coding-checks.md#dependencies)补了拓扑排序与并行执行的区别；Trie 和 Union-Find 可按需要另读，不必为了“覆盖完整”直接跳去 Hard。

<details class="interview" markdown="1">
<summary>为什么按起点排序适合合并，按终点排序适合最多活动数？</summary>

合并要整理覆盖范围，起点顺序让重叠局部化；活动选择要给后面留时间，最早结束有交换论证。目标不同，排序依据不同。

</details>

<details class="interview" markdown="1">
<summary>Jump Game 模板能直接返回最少跳跃次数吗？</summary>

不能。这一版只维护是否可达。最少次数需要额外维护当前层的边界和下一层可达边界，或按层搜索；不要把另一个输出要求当成同一个算法已经解决了。

</details>
