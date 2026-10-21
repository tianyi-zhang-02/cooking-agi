# 双指针、滑动窗口、二分：为什么能往这边走

**中文** · [English](pointers-and-search.en.md) · [方法地图](../leetcode.md)

> 阅读时间：约 12 分钟 · 最近审阅：2026-10

这几种题写起来往往只有十几行，难的是想明白：为什么这里该移动左边，而不是右边？移动以后，会不会把答案漏掉？先用小数组走几步，再回来看代码，边界就没那么难记了。

## 1. 双指针：有序性替你排除组合

在有序数组 `[1, 3, 5, 8, 12]` 里找两个数，和等于 11。最直接的做法是把所有组合试一遍。但数组已经排好序，有些组合其实不用试。

让 `left` 从最左边开始，`right` 从最右边开始：

| 现在选的两个数 | 和 | 下一步为什么这样走 |
| --- | --- | --- |
| 1 和 12 | 13，太大 | 1 已经是剩下最小的数，换成 3 或 5 只会更大。12 不可能配出答案，右边往里走 |
| 1 和 8 | 9，太小 | 8 已经是剩下最大的数，换成 5 或 3 只会更小。1 不可能配出答案，左边往里走 |
| 3 和 8 | 11 | 找到了，返回这两个位置 |

<!-- widget:algo-two-pointers -->

所以每次移动都不用猜方向：它排除的是一个**已经不可能参与答案的端点**。这个理由成立，靠的就是有序性。

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

这段代码返回从 0 开始的下标，上面的结果是 `[1, 3]`。两个指针总共向内移动不超过 n−1 次，所以时间是 O(n)，辅助空间是 O(1)。无序数组不能直接套；如果题目要原下标，先排序还得保留元素原来的位置，也可以改用 [hash map](hash-and-prefix.md)。

双指针不一定要从两端出发。原地去重时，常见的是**读写指针**：一个往前读，一个记录下次往哪里写。例如 `[1, 1, 3]`，读到第二个 1 时不写，读到 3 时把它写到位置 1。最后只看前两项 `[1, 3]`；后面的旧值不属于结果。

## 2. 滑动窗口：维护一段合法区间

以“最长无重复子串”为例。子串要求连续，不能从中间跳过一个字符。我们用 `[left, right]` 表示正在看的这一段，再用集合记录里面有哪些字符。

读 `"abba"` 时，窗口这样变化：

| 读到哪里 | 发生了什么 | 调整后的窗口 | 目前最长 |
| --- | --- | --- | --- |
| 位置 0 的 a | 没见过，放进来 | `a` | 1 |
| 位置 1 的 b | 没重复，继续扩大 | `ab` | 2 |
| 位置 2 的 b | 重复了。先移走 a，仍然重复；再移走旧的 b | `b` | 2 |
| 位置 3 的 a | 当前窗口里没有 a，可以放进来 | `ba` | 2 |

<!-- widget:algo-sliding-window -->

注意第三步：不是碰到重复字符就把整个窗口清空，而是从左边逐个移走，直到这次加入不会重复。这样留下的仍是**以当前位置结尾、最长的合法子串**。

```python
def longest_unique_window(text):
    window = set()
    left = best = 0
    for right, character in enumerate(text):
        while character in window:
            window.remove(text[left])
            left += 1
        window.add(character)
        best = max(best, right - left + 1)
    return best
```

这里必须用 `while`，因为移走一个字符未必够。答案也要等窗口重新没有重复后再更新。集合查询、增删按平均 O(1) 算；左右边界各走一遍，整体平均 O(n)，辅助空间 O(min(n, 字符种类数))。

用英语讲，不用把每一行都念出来，说明窗口保持什么条件就好：

> I maintain a sliding window with no repeated characters. If the next character is already in the window, I move the left boundary until that duplicate is removed, then add the new character. Each character is added and removed at most once, so the algorithm takes O(n) time on average, assuming average O(1) set operations.

“窗口没有重复字符”就是这里的不变量。解释复杂度时，别漏掉集合操作的平均时间前提。

### 看懂逐个移走，再看直接跳过去

上面的写法足够用了。如果还记得每个字符上次出现在哪，就能直接把左边界跳到重复位置的后一格：

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

这里的 `last_seen` 记录整段已读文本，不只是当前窗口。所以最后读到 a 时，它上次的位置虽然是 0，却早已在窗口外：`left` 已经是 2，不能退回 1。`max` 就是在防止这个回退；否则会把有重复的 `"bba"` 算进去。

两版的时间和空间复杂度相同。区别是第一版把收缩过程写出来，第二版利用“上次出现的位置”省掉逐个移走的步骤。先写清楚第一版，再理解这一步优化，比直接记住 `max` 更容易。

## 3. 不是所有子数组问题都能滑

再看一个容易混淆的要求：在正数数组 `[2, 1, 4, 3]` 中，找和至少为 7 的最短连续子数组。

- 扩到 `[2, 1, 4]`，和是 7，先记下长度 3；移走 2 后变成 5，暂时不能再缩。
- 加入 3，窗口是 `[1, 4, 3]`，和是 8；移走 1 后，`[4, 3]` 的和仍是 7，答案改成 2。
- 再移走 4，和只剩 3，这时停止收缩。

和上一题不同，这里是在**窗口仍满足要求时记答案，再试着缩短**。上一题则要先消除重复，才能记下合法的长度。不要把更新答案的位置当成通用模板。

为什么这里敢停止收缩？因为每个数都为正，继续移走只会让和更小。有负数就不一定了：把目标改成 3，`[1, -1, 3]` 的总和是 3，移走第一个 1 后变成 2，但再移走 −1，反而又回到 3。如果在和小于目标时立刻停下，就会漏掉单独的 `[3]`。

这不代表有负数的题一概不能用窗口，而是这套“和不够就扩、够了就缩”的理由不再成立。统计有负数的“和恰好等于 target”，可以看[前缀和 + hash map](hash-and-prefix.md)；它解决的是计数问题，不要直接当成“最短长度”的解法。

## 4. 二分：先写边界含义

二分不只是“找一个等于目标的数”。下面要找的是：升序数组中，第一个 `>= target` 的位置。没有就返回数组长度 n，相当于插在末尾。

先分清两边已经知道什么：`left` 左边都小于目标；从 `right` 开始的现有元素都大于等于目标。中间 `[left, right)` 的元素还没排除完。**最后要找的位置可以恰好是 `right`，甚至是 n；它不是必须留在数组里的某个元素。**

例如 `[1, 3, 3, 7, 9]` 找第一个不小于 3 的位置：

| left / right | middle 及其值 | 这次能确定什么 |
| --- | --- | --- |
| 0 / 5 | 位置 2，值是 3 | 它符合，但左边可能还有。保留这个边界，`right = 2` |
| 0 / 2 | 位置 1，值是 3 | 继续往左找，`right = 1` |
| 0 / 1 | 位置 0，值是 1 | 它太小，连它一起排除，`left = 1` |

<!-- widget:algo-binary-search -->

两边相遇在 1，这就是答案。`right = middle` 和 `left = middle + 1` 并不对称：前者遇到的数可能就是答案，后者遇到的数已经确定不够大。

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
| [两数之和 II](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/) | 解释为什么能排除一端；注意题目下标从 1 开始 |
| [无重复字符的最长子串](https://leetcode.com/problems/longest-substring-without-repeating-characters/) | 左边界为什么不能回退 |
| [长度最小的子数组](https://leetcode.com/problems/minimum-size-subarray-sum/) | 正数前提，什么时候更新最短答案 |
| [搜索插入位置](https://leetcode.com/problems/search-insert-position/) | 空数组、重复值、插在末尾 |

<details class="interview" markdown="1">
<summary>为什么窗口里有 while，仍然可以 O(n)？</summary>

每个元素最多进入一次、离开一次。只有在每次维护计数或和是 O(1) 时，这个证明才成立；每次重新扫描窗口就不行。

</details>

<details class="interview" markdown="1">
<summary>“有序数组才能二分”完整吗？</summary>

不完整。真正需要的是单调的判定，让答案边界一侧都为假、另一侧都为真。有序数组只是最常见的情况。

</details>

## 6. 换一种讲法再看

- [灵茶山艾府的基础算法视频与题单](https://github.com/EndlessCheng/codeforces-go/blob/master/leetcode/README.md)：想跟着推一遍双指针、滑动窗口和二分，可以从对应视频开始，不用顺着整份题单刷。
- [labuladong 的滑动窗口讲解](https://labuladong.online/zh/algo/essential-technique/sliding-window-framework/)：适合对照“何时扩、何时缩、何时记答案”。其中部分例题更难，先把无重复子串写熟就好。
- [Hello 算法：二分查找](https://www.hello-algo.com/chapter_searching/binary_search/)：可以对着图比较不同区间写法。本页只用左闭右开这一种，避免把几套边界混在一起。
