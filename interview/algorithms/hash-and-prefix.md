# Hash map 与前缀和：到底要记住什么

**中文** · [English](hash-and-prefix.en.md) · [方法地图](../leetcode.md)

> 阅读时间：约 7 分钟 · 最近审阅：2026-09

如果每来一个数，都要回头扫描之前的数，就问一句：**能不能把以后要查的信息先存起来？** Hash map 的重点不是背 `dict`，而是选对 key 和 value。

## 1. 先决定存什么

| 以后要问的问题 | key | value / 容器 |
| --- | --- | --- |
| 这个数见过吗？ | 数值 | set 就够 |
| 上次在哪儿？ | 数值 / 字符 | 下标 |
| 出现了几次？ | 数值 / 字符 | 频次 |
| 哪些词是同一组？ | 规范化后的签名 | 单词列表 |
| 之前有几个前缀能与当前配对？ | 前缀和 | 出现次数 |

Python 的 key 要可哈希：整数、字符串、只含可哈希元素的 tuple 常用；list 不行。分组异位词时，可以用排序后的字符串当 key；长度 m 的词，排序要 O(m log m)，别把造 key 的时间漏掉。[哈希表原理](https://algs4.cs.princeton.edu/34hash/)

## 2. Two Sum：先查，再存

例子：`[4, 1, 7]`，target = 8。读到 7 时，需要的是 1；字典告诉我们 1 在下标 1，于是答案是 `[1, 2]`。

```python
def two_sum(values, target):
    seen = {}
    for index, value in enumerate(values):
        complement = target - value
        if complement in seen:
            return [seen[complement], index]
        seen[value] = index
    return None
```

**不变量**：查找时，字典里只有当前下标之前的数。这样 `[3]`、target = 6 不会误把自己用两次；`[3, 3]` 则可以配成一对。时间平均 O(n)，空间 O(n)。优势是不用排序、保留原下标；代价是额外字典空间。

## 3. dict 常用写法，差别在哪里

```python
counts = {"pear": 2}
exists = "pear" in counts
missing_count = counts.get("apple", 0)
counts["pear"] = counts.get("pear", 0) + 1
removed = counts.pop("apple", 0)
pairs = list(counts.items())
```

`exists == True`，`missing_count == 0`，`removed == 0`，`pairs == [('pear', 3)]`。`get` 不会把缺失 key 插进去；`defaultdict` 的 `mapping[key]` 会触发默认值创建。`setdefault(key, [])` 会在缺失时插入，适合分组；`keys()`、`values()`、`items()` 返回视图，不是独立快照。[Python mapping 接口](https://docs.python.org/3/library/stdtypes.html#mapping-types-dict)

别用 `if seen.get(value)` 判断是否存在：下标 0 会被当成 False。用 `value in seen`。遍历字典时也别直接改变它的大小。

## 4. 前缀和：把一段相加改成两端相减

定义 prefix[0] = 0，prefix[right] 是前 right 个元素的和。那么左闭右开区间 `[left, right)` 的和，就是 `prefix[right] - prefix[left]`。

若目标是统计“和等于 target 的连续子数组”，当前前缀为 prefix，只需要数之前出现过多少次 `prefix - target`。

```python
def count_subarrays(values, target):
    frequencies = {0: 1}
    prefix = 0
    total = 0
    for value in values:
        prefix += value
        total += frequencies.get(prefix - target, 0)
        frequencies[prefix] = frequencies.get(prefix, 0) + 1
    return total
```

| 读入 `[1, -1, 1]`，target = 1 | 当前前缀 | 查哪个旧前缀 | 新增答案 |
| --- | --- | --- | --- |
| 1 | 1 | 0 | 1 |
| −1 | 0 | −1 | 0 |
| 1 | 1 | 0 | 2 |

合计 3：两个单独的 1，以及整段。这里存的是**频次**，不是一个下标或 set。初始 `{0: 1}` 代表数组开始前的空前缀；先查再存，避免 target = 0 时把空子数组算进去。

时间平均 O(n)，空间 O(n)。可以处理负数；相比窗口，多用了字典，但不依赖“窗口越长和越大”。前缀和很适合静态区间查询；原数组频繁改值时，不能指望已有前缀仍然有效。

## 5. 练什么、怎么迁移

- [Two Sum](https://leetcode.com/problems/two-sum/)：value → index，想清楚更新顺序。
- [Group Anagrams](https://leetcode.com/problems/group-anagrams/)：设计同一组共享的 key；分析造 key 成本。
- [Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)：prefix → count，练有负数的连续区间。

<details class="interview" markdown="1">
<summary>把 frequencies 换成 set，会漏什么？</summary>

同一个前缀和可能出现多次，每次对应一个不同起点。set 只能告诉你“有”，不能告诉你有几个。`[0, 0]`、target = 0 的答案是 3，不是 2。

</details>

<details class="interview" markdown="1">
<summary>Two Sum 改成有序输入，而且只要值对，还一定要用字典吗？</summary>

不一定。可以用左右指针，时间 O(n)、辅助空间 O(1)。如果先排序，要算 O(n log n)，而且原下标会丢失，除非额外保存。

</details>
