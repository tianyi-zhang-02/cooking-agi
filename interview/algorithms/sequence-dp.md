# 序列 DP：先说明白，你在给哪一段做总结

**中文** · [English](sequence-dp.en.md) · [方法地图](../leetcode.md)

> 最近审阅：2026-10-09 · 先读：[回溯与 DP](backtracking-and-dp.md)

`CAB` 和 `ACB` 里，`AB` 都按同样顺序出现，但可以跳过字符。这是在找子序列，不是在找一段连续文本。最长公共子序列（LCS）长度为 2；`CB` 也是一个答案，所以“返回长度”和“返回唯一序列”不是一回事。

序列 DP 容易混，是因为状态常常只差几个字：**以这个位置结尾**，还是**只看前这么多项**？先把这一句写完整，递推式通常就有了落点。

## LCS：两个前缀之间，最多能配上多少

$L(i,j)$ 表示第一个串前 $i$ 个字符、第二个串前 $j$ 个字符的 LCS 长度。空前缀的答案是 0。

- 末尾字符相等：可以在两个更短前缀的 LCS 后接上这个字符，得到 $L(i-1,j-1)+1$。
- 末尾不同：公共子序列不能同时使用这两个不同的末尾，至少跳过一个，取 $\max(L(i-1,j),L(i,j-1))$。

相等时为什么不需要额外比较跳过的两项？在任意最优匹配中，如果这个相等的末尾没有成对出现，可以将最后一对相同字符移到这两个末尾而不破坏此前顺序；若两边末尾都未使用，则还可以追加一对。因此总能选择一个使用这对末尾的最优解。

| 第一个串的前缀 \ 第二个串的前缀 | 空 | A | AC | ACB |
| --- | --- | --- | --- | --- |
| 空 | 0 | 0 | 0 | 0 |
| C | 0 | 0 | 1 | 1 |
| CA | 0 | 1 | 1 | 1 |
| CAB | 0 | 1 | 1 | 2 |

右下角是 2。表格中的 1 并不都来自同一个公共子序列；每个格子只保存它自己的最佳长度。

## 只求长度，两行就够

```python
def lcs_length(first, second):
    previous = [0] * (len(second) + 1)
    for first_char in first:
        current = [0]
        for column, second_char in enumerate(second, start=1):
            if first_char == second_char:
                current.append(previous[column - 1] + 1)
            else:
                current.append(max(previous[column], current[-1]))
        previous = current
    return previous[-1]
```

`previous[column]` 是上方，`current[-1]` 是左方，`previous[column-1]` 是左上方。保留两行让这三种依赖很清楚，先别急着压成一行再保存“旧对角线”。

长度分别为 $n,m$，时间 $O(nm)$，辅助空间 $O(m)$。可以把较短的串放第二个参数来减少空间。若要恢复一条公共子序列，保留全表，从右下角沿用到的转移回溯；有并列时，制定一个规则即可，不必假装答案唯一。

## LIS：状态不是“前面最好的”，而是“在这里结尾的”

最长严格递增子序列（LIS）的一种直接写法：

$$
D(i)=1+\max\bigl(\{D(j):j<i,\ a_j<a_i\}\cup\{0\}\bigr).
$$

$D(i)$ 必须以 $a_i$ 结尾。这样才能判断新的数能否接上；最终答案才是所有 $D(i)$ 的最大值。双循环时间 $O(n^2)$、空间 $O(n)$，空数组返回 0。

`[3,5,6,2,4]` 对应 `D=[1,2,3,1,2]`，答案 3。若只存“前面最长为 3”，却忘了结尾是 6，就可能误以为 4 可以直接接上。

## 用最小尾值压缩候选，但别把它当成答案序列

对每种长度，只留最小结尾：长度相同、结尾更小的递增序列，未来至少一样容易延长。`tails[length-1]` 就表示这个最小结尾。

| 新值 | tails |
| --- | --- |
| 3 | [3] |
| 5 | [3,5] |
| 6 | [3,5,6] |
| 2 | [2,5,6] |
| 4 | [2,4,6] |

最后的 `[2,4,6]` **不是**原数组的子序列：6 在 2 和 4 前面。它是分别代表三种长度的摘要，长度 3 才是答案。

```python
def lis_length(values):
    tails = []
    for value in values:
        position = bisect_left(tails, value)
        if position == len(tails):
            tails.append(value)
        else:
            tails[position] = value
    return len(tails)
```

先 `from bisect import bisect_left`。`bisect_left` 找第一个不小于当前值的位置，因此相等值不会把长度加一。对于非递减子序列，要改成 `bisect_right` 并相应改变题目条件；不是“两个函数都能用”。[Python bisect 文档](https://docs.python.org/3/library/bisect.html)

每个元素做一次二分，时间 $O(n\log n)$，空间 $O(n)$。替换只是维护最小尾值，不是在原数组里交换元素。要恢复真实序列，还需记录每个长度对应的原下标与 predecessor。

## 相似题，别用同一套边界

| 问题 | 状态里的关键条件 | 不能照搬的地方 |
| --- | --- | --- |
| LCS | 两个前缀，允许跳字 | 不相等时取上方 / 左方最大值 |
| 最长公共子串 | 当前两处结尾，要求连续 | 不相等时该格归零，不是沿用最大值 |
| LIS | 一个序列，按原顺序且严格递增 | 相等元素不能延长 |
| 编辑距离 | 两个前缀之间的最少操作数 | 空前缀边界是另一边的长度，不是全 0 |

试一下 `[2,2,2]`：严格 LIS 是 1，非递减版本是 3。再试空串与反序字符串。代码在 [deeper_patterns.py](../code/deeper_patterns.py)，小输入用子序列枚举检查；推导练习可接着看 [MIT 动态规划讲义](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/resources/lecture-notes/)。
