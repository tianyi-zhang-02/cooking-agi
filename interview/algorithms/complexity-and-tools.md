# 复杂度与 Python：先知道工具在做什么

**中文** · [English](complexity-and-tools.en.md) · [方法地图](../leetcode.md)

> 阅读时间：约 6 分钟 · 最近审阅：2026-09

选对容器通常比把一行代码写短更有用。先问：我要查找、计数、按顺序处理，还是不断取出最小值？

## 1. 复杂度不是数 for 的个数

`for` 里套 `while`，不一定是 O(n²)。如果两个指针都只往前走，每个最多移动 n 次，总共仍是 O(n)。反过来，一个 `for` 每次都做 `values[:index]`，复制长度从 0 到 n−1，累加就是 O(n²)。

这里默认整数运算、单次比较是 O(1)；遇到很大的 Python 整数、长字符串或 tuple key，要另算运算、比较或哈希的长度成本。

| 常见形状 | 时间 | 为什么 |
| --- | --- | --- |
| 扫一遍 | O(n) | 每个元素处理固定次数 |
| 每次砍半 | O(log n) | 区间长度 n → n/2 → n/4 |
| 排序 | O(n log n) 最坏 | 不是“调用一个函数所以 O(1)” |
| 所有元素两两配对 | O(n²) | 大约 n(n−1)/2 对 |
| 枚举子集并复制答案 | O(n·2ⁿ) | 2ⁿ 个输出，总长度也要算 |

## 2. average 和 amortized 别混着说

- **Average / expected**：`dict` 查找通常 O(1)，依赖哈希分布等条件；极端碰撞可到 O(n)。
- **Amortized**：`list.append` 偶尔扩容很贵，但一长串追加摊下来，每次 O(1)。不是说每次追加都只花固定时间。

输入通常不计入辅助空间；新建的字典、递归栈要算；返回全部答案的空间单独说，不要藏起来。

## 3. 容器速查

| 工具 | 操作 | 成本与限制 |
| --- | --- | --- |
| `list` | `append` / `pop()` / `[-1]` | 末尾追加摊还 O(1)，末尾删除与索引 O(1) |
| `list` | `pop(0)` / `insert(0, x)` / `x in values` | O(n)；不要拿它当 BFS 队列 |
| `dict` / `set` | 查找、插入、删除 | 通常平均 O(1)；set 不保留频次 |
| `deque` | `append` / `popleft` | 两端操作 O(1)，中间索引不是 O(1) |
| `heapq` | `heappush` / `heappop` | O(log n)；`heap[0]` 取最小 O(1) |
| `heapq` | `heapify` | O(n)，不是逐个 push 的 O(n log n) |
| `bisect` | `bisect_left` / `bisect_right` | 查位置 O(log n)；插进 list 仍需 O(n) |

成本针对常见 Python 实现和上面的基本操作假设。接口见官方 [dict / set](https://docs.python.org/3/library/stdtypes.html#mapping-types-dict)、[deque / Counter](https://docs.python.org/3/library/collections.html)、[heapq](https://docs.python.org/3/library/heapq.html)、[bisect](https://docs.python.org/3/library/bisect.html)。

## 4. 几个经常用的 function

```python
from collections import Counter, defaultdict, deque
from bisect import bisect_left, bisect_right
import heapq

counts = Counter("banana")
groups = defaultdict(list)
groups["fruit"].append("pear")
queue = deque(["start"])
queue.append("next")
first = queue.popleft()
values = [1, 2, 2, 5]
left = bisect_left(values, 2)
right = bisect_right(values, 2)
heap = [5, 1, 3]
heapq.heapify(heap)
smallest = heapq.heappop(heap)
```

结果：`counts['a'] == 3`；`first == 'start'`；`left == 1`、`right == 3`，所以 2 出现了 `right-left == 2` 次；`smallest == 1`。

`enumerate(values)` 给下标和值；`zip(first, second)` 默认在短的那边结束；`sorted(values, key=...)` 返回新列表；`values.sort()` 原地改、返回 `None`。最大堆的负数法兼容较老 Python；Python 3.14 起也有公开的 `_max` 接口，先确认运行版本。

## 5. 递归、切片和输出都要算空间

一棵长成链的树：没有显式列表，递归深度仍可达 n，调用栈就是 O(n)。平衡二叉树的高度才是 O(log n)。

`values[left:right]` 会复制这一段；`''.join(parts)` 也要创建输出字符串。原地修改不一定代表 O(1) 空间，例如排序内部还可能使用辅助缓冲。

## 6. 合上笔记，试一下

<details class="interview" markdown="1">
<summary>两个指针只增加，循环看起来嵌套，为什么可能还是 O(n)？</summary>

数整个过程的移动次数，不是把循环机械相乘。两个指针各最多增加 n 次，且每步更新 O(1)，总时间 O(n)。如果每步又复制窗口或重新求和，要加上那部分成本。

</details>

<details class="interview" markdown="1">
<summary>bisect 找位置只要 O(log n)，为什么维护有序 list 不是每次 O(log n)？</summary>

找位置和搬元素是两件事。把元素插到中间，后面的元素需要移动，最坏 O(n)。

</details>

下一步：[Hash map：到底要记住什么](hash-and-prefix.md)。
