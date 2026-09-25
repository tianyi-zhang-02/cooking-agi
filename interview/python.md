# 技术面：刷题用得上的 Python 语法

**中文** · [English](python.en.md)

> 阅读时间：约 8 分钟 · 最近审阅：2026-09

> **先读这个**：这里只整理公开的语言知识和我自己刷题时常用的写法，不写任何一家公司的面试题。

用 Python 刷题，省下来的时间主要来自几样东西：lambda 配合排序、推导式、几种解包写法，以及标准库里现成的数据结构。这一篇按用到的频率排，最后是面试里常被追问的语言细节。

## lambda：一行写完的小函数

`lambda 参数: 表达式` 就是一个没有名字的函数。它只能写一个表达式，算出来的值自动返回：

```python
double = lambda x: x * 2      # 等价于 def double(x): return x * 2
```

单独这么写意义不大，它真正的用处是当作参数传给别的函数，最常见的是排序的 `key`。`sorted` 会先对每个元素算一遍 `key(元素)`，再按算出来的值排：

```python
pairs = [("amy", 90), ("bob", 85), ("cat", 90)]
sorted(pairs, key=lambda p: p[1])           # 按分数升序
sorted(pairs, key=lambda p: (-p[1], p[0]))  # 分数降序，同分按名字升序
max(pairs, key=lambda p: p[1])              # 分数最高的那一个
```

按多个条件排序，就让 key 返回一个 tuple：tuple 先比第一个元素，相同再比第二个。要某一项降序，数字取负号就行；字符串不能取负，只能用 `reverse=True`，或者分两次排序。

下面可以换不同的 key，看结果怎么变。留意同分的人：Python 的排序是**稳定**的，key 相同的元素会保持原来的相对顺序。

<!-- widget:tx-py-sort -->

`sorted(x)` 返回一个新列表，原列表不变；`x.sort()` 在原地排序，返回的是 `None`。所以写成 `x = x.sort()` 会把 x 变成 None，这是很常见的 bug。

还有一个坑：在循环里建 lambda，它记住的是变量本身，不是当时的值，见文末第 2 题。

## 推导式：一行建出列表、字典和集合

```python
squares = [x * x for x in nums]                 # 列表
evens   = [x for x in nums if x % 2 == 0]       # 带过滤条件
index   = {v: i for i, v in enumerate(nums)}    # 字典：值 -> 下标
seen    = {c for c in s}                        # 集合
grid    = [[0] * n for _ in range(m)]           # m 行 n 列的二维数组
total   = sum(x * x for x in nums)              # 生成器表达式，不建中间列表
```

最后一行用的是圆括号，这叫生成器表达式：它不会先把所有元素放进内存，而是用一个算一个，数据大的时候更省内存。

## 解包、enumerate、zip 和切片

```python
a, b = b, a                     # 交换两个变量，不需要临时变量
first, *rest = nums             # first 是第一个元素，rest 是剩下的列表
for i, x in enumerate(nums):    # 同时拿到下标和值
    ...
for x, y in zip(xs, ys):        # 两个列表按位置配对，长度按短的那个算
    ...
s[::-1]                         # 反转字符串或列表
nums[i:j]                       # 左闭右开，返回一个新的副本
nums[-1]                        # 最后一个元素
```

## 标准库里现成的数据结构

很多题型都有现成的工具，记住下面这张表能省不少代码：

| 需要什么 | 用什么 | 要点 |
| --- | --- | --- |
| 计数 | `collections.Counter` | `Counter(s).most_common(k)` 取出现次数最多的 k 个 |
| 分组、建图 | `collections.defaultdict(list)` | 访问不存在的 key 时，自动建一个空列表 |
| 队列、BFS | `collections.deque` | `append` 和 `popleft` 都是 O(1)；list 的 `pop(0)` 是 O(n) |
| 最小堆、Top K | `heapq` | 只有最小堆；要最大堆，就把数字取负再放进去 |
| 在有序数组里找位置 | `bisect` | `bisect_left` 找第一个大于等于 x 的位置，O(log n) |
| 记忆化递归 | `functools.cache` | 加在递归函数上，参数相同的调用直接返回缓存结果 |
| 排列组合 | `itertools` | `permutations`、`combinations`、`product`、`accumulate` |
| 无穷大 | `math.inf` 或 `float("inf")` | 求最小值时拿来当初始值 |

```python
import heapq
heap = []
heapq.heappush(heap, 3)
smallest = heapq.heappop(heap)    # 弹出最小的元素
heapq.heappush(heap, -x)          # 最大堆：存负数，取出来再取负
heapq.nlargest(k, nums)           # 直接拿最大的 k 个
```

## 面试里常被追问的语言细节

- **二维数组别写成 `[[0] * n] * m`。** 外面的 `* m` 复制的是同一个内层列表的引用，改一个格子，每一行都会跟着变。要写成 `[[0] * n for _ in range(m)]`。
- **默认参数不要用可变对象。** `def f(x, acc=[])` 里的 `[]` 只在定义函数时创建一次，之后每次调用都共用这一个列表。一般写成 `acc=None`，函数里再判断 `if acc is None: acc = []`。
- **`is` 和 `==` 不一样。** `==` 比较值是否相等，`is` 比较是不是同一个对象。判断 None 要用 `is None`。
- **浅拷贝和深拷贝。** `x[:]`、`list(x)`、`x.copy()` 只复制最外层；里面还套着列表的话，要用 `copy.deepcopy`。
- **负数的整除和取余。** Python 的 `//` 向下取整：`-7 // 2` 是 `-4`，`-7 % 2` 是 `1`。C++ 和 Java 是向零取整，`-7 / 2` 得 `-3`，换语言时容易踩坑。
- **字符串不能原地修改。** 在循环里用 `s += c` 拼接，每次都会新建一个字符串；先把片段放进列表，最后 `"".join(parts)`。
- **递归深度有上限。** 默认大约 1000 层，DFS 遇到很长的链会报 `RecursionError`。可以用 `sys.setrecursionlimit` 调高，或者改写成用栈的迭代版本。
- **`*args` 和 `**kwargs`。** 前者把多出来的位置参数收成一个 tuple，后者把多出来的关键字参数收成一个 dict。
- **生成器。** 函数里用了 `yield` 就成了生成器：每次取值时算出一个就停下，下次从停下的地方继续。适合处理很大的数据，或者没有尽头的序列。

更多题型怎么刷，见 [LeetCode 怎么刷](leetcode.md)。

## 面试常见问题

<details class="interview" markdown="1">
<summary>grid = [[0] * 3] * 3，改了 grid[0][0]，为什么三行都变了？</summary>

`* 3` 复制的是引用：三行其实是同一个列表。改其中一个，从三个名字看到的都是同一个对象。正确写法是 `[[0] * 3 for _ in range(3)]`，每次循环都新建一行。

</details>

<details class="interview" markdown="1">
<summary>fs = [lambda: i for i in range(3)]，为什么 fs[0]() 返回的是 2？</summary>

lambda 记住的是变量 `i` 本身，调用时才去看 `i` 的值；循环结束时 `i` 已经是 2，所以三个函数都返回 2。这叫延迟绑定（late binding）。要记住当时的值，就用默认参数把它固定下来：`lambda i=i: i`。

</details>

<details class="interview" markdown="1">
<summary>sorted 和 list.sort 有什么区别？Python 的排序稳定吗？</summary>

`sorted` 接受任何可迭代对象，返回一个新列表；`list.sort` 只能用在列表上，原地排序，返回 `None`。两者都是稳定排序（Timsort），key 相同的元素保持原来的相对顺序，时间复杂度 O(n log n)。

</details>

<details class="interview" markdown="1">
<summary>heapq 怎么当最大堆用？Top K 问题怎么做？</summary>

`heapq` 只提供最小堆，存进去时取负数，取出来再取负，就相当于最大堆。求最大的 k 个元素，可以维护一个大小为 k 的最小堆：新元素比堆顶大就替换堆顶，最后堆里就是最大的 k 个，时间 O(n log k)。数据不大时直接 `heapq.nlargest(k, nums)`。

</details>

<details class="interview" markdown="1">
<summary>-7 // 2 等于多少？-7 % 2 呢？</summary>

`-7 // 2 = -4`，`-7 % 2 = 1`。Python 的整除向下取整，并且保证 `a == (a // b) * b + a % b`。C++ 和 Java 向零取整，`-7 / 2` 得 `-3`，余数是 `-1`。

</details>

<details class="interview" markdown="1">
<summary>列表推导式和生成器表达式有什么区别？</summary>

列表推导式 `[...]` 一次把所有元素算出来放进内存；生成器表达式 `(...)` 用到一个才算一个，几乎不占额外内存，但只能遍历一次。只是求和、取最大值这类一次性用完的场景，用生成器更省。

</details>

<details class="interview" markdown="1">
<summary>为什么默认参数不能写成空列表？</summary>

默认值只在定义函数时求值一次。`def f(acc=[])` 里的那个列表会被之后所有调用共用，上一次调用往里加的东西，下一次还在。改成 `acc=None`，在函数体里 `if acc is None: acc = []`。

</details>
