# 双指针、滑动窗口、二分：为什么能往这边走

**中文** · [English](pointers-and-search.en.md) · [方法地图](../leetcode.md)

> 阅读时间：约 7 分钟 · 最近审阅：2026-09

这 3 种方法都在缩小范围，但理由不一样。别只记 `left += 1`，要能说出这一步排除了什么。

## 1. 双指针：有序性替你排除组合

有序数组 `[1, 3, 4, 7]`，目标 8：最左加最右刚好是 8。如果和太小，把 right 往左挪只会更小，所以 left 可以前进；和太大则反过来。

```python
def sorted_pair(values, target):
    left, right = 0, len(values) - 1
    while left < right:
        total = values[left] + values[right]
        if total == target:
            return [left, right]
        if total < target:
            left += 1
        else:
            right -= 1
    return None
```

前提是升序，返回 0-based 下标。时间 O(n)、辅助空间 O(1)。优点是不需要字典；无序数组不能直接套，先排序又会付出时间和原下标维护成本。

另一个常见形状是**读写指针**：read 扫输入，write 指向下一个保留位置。适合原地删除、去重；不变量是 `values[:write]` 已经是处理过部分的正确结果。

## 2. 滑动窗口：维护一段合法区间

最长无重复子串，读到一个重复字符时，把左边界移到它上次出现位置的后一格。但边界不能往回走。

```python
def longest_unique(text):
    last_seen = {}
    left = best = 0
    for right, character in enumerate(text):
        left = max(left, last_seen.get(character, -1) + 1)
        last_seen[character] = right
        best = max(best, right - left + 1)
    return best
```

`"abba"`：第二个 b 把 left 推到 2；最后的 a 上次在 0，不能让 left 从 2 退回 1。答案是 2，不是 3。

时间平均 O(n)，空间 O(min(n, 字符种类数))。这版利用“上次位置”直接跳；另一种窗口写法会用计数器，在 `while` 中移除左端元素，直到条件重新满足。**最长合法窗口是在恢复合法后更新答案；最短满足窗口通常在仍满足时更新，再继续收缩。** 两者不要照抄同一处更新位置。

## 3. 不是所有子数组问题都能滑

若所有数都为正，窗口向右扩张只会增大和，收缩只会减小；可以找“和至少 target 的最短子数组”。有负数就不一定：`[1, -1, 3]`，target = 3。总和是 3，删除第一个 1 后变 2；如果立刻停止收缩，会漏掉单独的 `[3]`。

换个条件，就换方法或重新证明单调性。统计有负数的“和恰好等于 target”，先看[前缀和 + hash map](hash-and-prefix.md)。

## 4. 二分：先写边界含义

下面找第一个 `>= target` 的位置，不存在就返回 n。区间统一用 `[left, right)`：left 左边都小于 target，right 及右边都大于等于 target。

```python
def lower_bound(values, target):
    left, right = 0, len(values)
    while left < right:
        middle = (left + right) // 2
        if values[middle] < target:
            left = middle + 1
        else:
            right = middle
    return left
```

`[1, 2, 2, 5]` 找 2 返回 1，找 6 返回 4。时间 O(log n)，空间 O(1)。查是否存在还要验证 `index < len(values)` 且 `values[index] == target`。现成接口是 [bisect_left](https://docs.python.org/3/library/bisect.html)。

“二分答案”用的是同一条逻辑：若某个容量可行，更大容量也可行，就二分第一个可行容量。总复杂度是**判定成本 × 二分次数**，不是一律 O(log n)。

## 5. 少量例题，反复换条件

| 练习 | 重点 |
| --- | --- |
| [Two Sum II](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/) | 证明为什么可以丢掉一端；题目输出是 1-based |
| [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/) | left 不回退 |
| [Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/) | 正数前提，什么时候更新最短答案 |
| [Search Insert Position](https://leetcode.com/problems/search-insert-position/) | 空区间、重复值、越过末尾 |

<details class="interview" markdown="1">
<summary>为什么窗口里有 while，仍然可以 O(n)？</summary>

每个元素最多进入一次、离开一次。只有在每次维护计数或和是 O(1) 时，这个证明才成立；每次重新扫描窗口就不行。

</details>

<details class="interview" markdown="1">
<summary>“有序数组才能二分”完整吗？</summary>

不完整。真正需要的是单调的判定，让答案边界一侧都为假、另一侧都为真。有序数组只是最常见的情况。

</details>
