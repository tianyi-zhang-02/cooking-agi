# 技术面：刷题用得上的 Python 语法

**中文** · [English](python.en.md)

> 阅读时间：约 10 分钟 · 最近审阅：2026-10

> **先读这个**：这里只整理公开的语言知识和我自己刷题时常用的写法，不写任何一家公司的面试题。

用 Python 刷题，省下来的时间主要来自几样东西：lambda 配合排序、推导式、几种解包写法，以及标准库里现成的数据结构。这一篇按用到的频率排，最后是面试里常被追问的语言细节。

下面的代码块可以各自运行。如果语法看得懂，却经常遇到“改了副本，原数据也变了”，接着读 [变量、拷贝与函数调用](python-objects.md)：那里会把对象之间的关系拆开讲。

## lambda：一行写完的小函数

`lambda 参数: 表达式` 就是一个没有名字的函数。它只能写一个表达式，算出来的值自动返回：

```python
double = lambda number: number * 2
assert double(3) == 6
```

单独这么写意义不大，它真正的用处是当作参数传给别的函数，最常见的是排序的 `key`。`sorted` 会先对每个元素算一遍 `key(元素)`，再按算出来的值排：

```python
pairs = [("amy", 90), ("bob", 85), ("cat", 90)]
by_score = sorted(pairs, key=lambda pair: pair[1])
by_score_then_name = sorted(pairs, key=lambda pair: (-pair[1], pair[0]))
best = max(pairs, key=lambda pair: pair[1])
assert by_score == [("bob", 85), ("amy", 90), ("cat", 90)]
assert by_score_then_name == [("amy", 90), ("cat", 90), ("bob", 85)]
assert best == ("amy", 90)
```

按多个条件排序，可以让 key 返回 tuple：先比第一项，相同再比第二项。上面的 `(-分数, 名字)` 就是分数降序、名字升序。注意，`reverse=True` 会反转整个 key 的顺序，不是只反转其中一项。

如果主排序条件是字符串，又希望次排序条件保持升序，可以先排次要条件，再稳定地排主要条件。比如先按组名降序，同组按编号升序：

```python
records = [("beta", 3), ("alpha", 2), ("beta", 1), ("alpha", 1)]
by_number = sorted(records, key=lambda record: record[1])
result = sorted(by_number, key=lambda record: record[0], reverse=True)
assert result == [("beta", 1), ("beta", 3), ("alpha", 1), ("alpha", 2)]
```

第二次排序只比较组名，同组元素保留第一次排好的编号顺序。[排序指南](https://docs.python.org/3.14/howto/sorting.html#sort-stability-and-complex-sorts)

下面可以换不同的 key，看结果怎么变。留意同分的人：Python 的排序是**稳定**的，key 相同的元素会保持原来的相对顺序。

<!-- widget:tx-py-sort -->

`sorted(x)` 返回一个新列表，原列表不变；`x.sort()` 在原地排序，返回的是 `None`。所以写成 `x = x.sort()` 会把 x 变成 None，这是很常见的 bug。

还有一个坑：在循环里建 lambda，它记住的是变量本身，不是当时的值，见文末第 2 题。

## 推导式：一行建出列表、字典和集合

```python
numbers = [3, 2, 3]
squares = [number * number for number in numbers]
evens = [number for number in numbers if number % 2 == 0]
positions = {value: index for index, value in enumerate(numbers)}
seen = {character for character in "banana"}
grid = [[0] * 3 for row in range(2)]
total = sum(number * number for number in numbers)
assert squares == [9, 4, 9] and evens == [2]
assert positions == {3: 2, 2: 1} and seen == {"b", "a", "n"}
grid[0][0] = 1
assert grid[1] == [0, 0, 0] and total == 22
```

`positions` 遇到重复的值会覆盖旧下标，所以 3 对应最后一次出现的位置。`grid` 每次新建一行，不会让两行共用同一个列表。

`sum` 里的生成器表达式用到一个平方才计算一个，不额外建立平方列表。不过输入 `numbers` 本来就已在内存中，它不会因此消失。

## 解包、enumerate、zip 和切片

```python
left, right = 2, 5
left, right = right, left
numbers = [10, 20, 30]
first, *rest = numbers
indexed = list(enumerate(numbers))
paired = list(zip(["amy", "bob"], [90]))
assert (left, right) == (5, 2)
assert first == 10 and rest == [20, 30]
assert indexed == [(0, 10), (1, 20), (2, 30)]
assert paired == [("amy", 90)]
assert numbers[1:3] == [20, 30] and numbers[-1] == 30
assert "abc"[::-1] == "cba"
```

`zip` 默认在最短输入结束时停下；如果长度不一致意味着数据有问题，Python 3.10+ 可用 `zip(..., strict=True)`，迭代到不匹配处会报错。普通 list 的切片是浅拷贝，右边界不包含；空列表不能取 `[-1]`，也不能解包出 `first`。[zip 文档](https://docs.python.org/3.14/library/functions.html#zip)

## 标准库里现成的数据结构

很多题型都有现成的工具，记住下面这张表能省不少代码：

| 需要什么 | 用什么 | 要点 |
| --- | --- | --- |
| 计数 | `collections.Counter` | `Counter(s).most_common(k)` 取出现次数最多的 k 个 |
| 分组、建图 | `collections.defaultdict(list)` | 用 `groups[key]` 读取缺失 key 时建空列表；`get` 不触发 |
| 队列、BFS | `collections.deque` | `append` 和 `popleft` 都是 O(1)；list 的 `pop(0)` 是 O(n) |
| 最小堆、Top K | `heapq` | 默认小顶堆；负数法可实现最大堆，兼容较老版本；Python 3.14+ 也有 `_max` 接口 |
| 在有序数组里找位置 | `bisect` | `bisect_left` 找第一个大于等于 x 的位置，O(log n) |
| 记忆化递归 | `functools.cache` | 参数必须可哈希；缓存没有大小上限，别忽略内存和外部状态 |
| 排列组合 | `itertools` | `permutations`、`combinations`、`product`、`accumulate` |
| 无穷大 | `math.inf` 或 `float("inf")` | 求最小值时拿来当初始值 |

```python
import heapq

heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
smallest = heapq.heappop(heap)
assert smallest == 1 and heap == [3]
numbers = [2, 9, 4]
negative_heap = [-number for number in numbers]
heapq.heapify(negative_heap)
assert -heapq.heappop(negative_heap) == 9
assert heapq.nlargest(2, numbers) == [9, 4]
```

例如 `groups.get("missing")` 返回 `None`，不会插入新 key。缓存则会保留参数与返回值的引用；被缓存的函数如果依赖会变化的外部数据，就可能返回过期结果。[defaultdict](https://docs.python.org/3.14/library/collections.html#collections.defaultdict) · [cache](https://docs.python.org/3.14/library/functools.html#functools.cache)

## 面试里常被追问的语言细节

- **二维数组别写成 `[[0] * n] * m`。** 外面的 `* m` 复制的是同一个内层列表的引用，改一个格子，每一行都会跟着变。要写成 `[[0] * n for _ in range(m)]`。
- **默认参数不要用可变对象。** `def f(x, acc=[])` 里的 `[]` 只在定义函数时创建一次，之后每次调用都共用这一个列表。一般写成 `acc=None`，函数里再判断 `if acc is None: acc = []`。
- **`is` 和 `==` 不一样。** `==` 比较值是否相等，`is` 比较是不是同一个对象。判断 None 要用 `is None`。
- **浅拷贝和深拷贝。** 普通 list 的切片和 `copy()` 只复制外层。是否复制内部对象，要看你接下来会改哪一层；不是遇到嵌套就必须 `deepcopy`。见[副本为什么会影响原数据](python-objects.md#shallow-copy)。
- **负数的整除和取余。** Python 的 `//` 向下取整：`-7 // 2` 是 `-4`，`-7 % 2` 是 `1`。C++ 和 Java 是向零取整，`-7 / 2` 得 `-3`，换语言时容易踩坑。
- **字符串不可变。** 反复拼接可能反复复制已积累的前缀，不能依赖某个解释器的优化来保证线性耗时。拼接一批片段时，通常直接用 `"".join(parts)`。[字符串操作](https://docs.python.org/3.14/library/stdtypes.html#common-sequence-operations)
- **递归深度有上限。** 常见默认值约 1000 层，以 `sys.getrecursionlimit()` 为准。DFS 遇到长链可能报 `RecursionError`，优先考虑显式栈；盲目提高限制不能保证安全。对照例子见 [DFS / BFS 图解](algorithms/traversal.md)。
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

`sorted` 接受可迭代对象，返回新列表；`list.sort` 原地排序，返回 `None`。稳定性是两者的保证：key 相同就保留原顺序。CPython 的排序能利用已有的有序片段；在 key 计算和比较为常数成本时，最坏时间为 O(n log n)，已排序输入可以更快。长字符串比较、昂贵的 key 函数也要单独计入成本。

</details>

<details class="interview" markdown="1">
<summary>heapq 怎么当最大堆用？Top K 问题怎么做？</summary>

`heapq` 默认使用小顶堆。存入时取负、取出再取负，可以实现数值最大堆，兼容较老 Python；3.14 起也提供公开的 `_max` 接口。求最大的 k 个元素（1≤k≤n），维护大小不超过 k 的小顶堆：新元素比堆顶大就替换。扫描时间 O(n log(k+1))，有序输出还需 O(k log k)。详见 [Top K 的淘汰线](algorithms/stack-heap-links.md) 和 [官方接口](https://docs.python.org/3/library/heapq.html)。

</details>

<details class="interview" markdown="1">
<summary>-7 // 2 等于多少？-7 % 2 呢？</summary>

`-7 // 2 = -4`，`-7 % 2 = 1`。Python 的整除向下取整，并且保证 `a == (a // b) * b + a % b`。C++ 和 Java 向零取整，`-7 / 2` 得 `-3`，余数是 `-1`。

</details>

<details class="interview" markdown="1">
<summary>列表推导式和生成器表达式有什么区别？</summary>

列表推导式 `[...]` 保存全部结果；生成器表达式 `(...)` 按需计算，不建这份结果列表。生成器仍保留执行状态，也可能让输入对象继续占着内存；它耗尽以后不会自动从头再来。要多次遍历同一批结果，保存列表可能更合适。见[短路与迭代器](python-objects.md#iteration-and-truth)。

</details>

<details class="interview" markdown="1">
<summary>为什么默认空列表不会在每次调用时重新创建？</summary>

默认值只在定义函数时求值一次。`def f(acc=[])` 里的那个列表会被之后所有调用共用，上一次调用往里加的东西，下一次还在。改成 `acc=None`，在函数体里 `if acc is None: acc = []`。

</details>
