# 单调队列：窗口滑走了，最大值怎么办

**中文** · [English](monotonic-queue.en.md) · [方法地图](../leetcode.md)

> 最近审阅：2026-10-09 · 先读：[栈、堆与链表](stack-heap-links.md)

`[4,1,3,5,2,5]`，每次看连续 3 个数，最大值依次是 `[4,5,5,5]`。每移动一次都重新求 max，当然能做，但长度为 $n$、窗口为 $k$ 时要 $O(nk)$。

只记一个最大值又不够：4 离开窗口后，要知道谁接班。单调队列保留的，就是**仍有可能接班的下标**。

## 什么情况下，一个数可以永久删掉

现在已有一个较早的 1，后来来了 3。只要未来窗口里还包含这个 1，就一定也包含后来的 3；3 更大，还更晚过期。这个 1 永远不会再成为最大值，可以删掉。

这依赖窗口右端只向前走、宽度固定。不是说 1 从数据里消失了，而是它不再可能影响我们要的答案。如果你要窗口总和、第二大值或所有并列最大值的位置，不能随便扔。

队列存下标而不是数值，才能判断有没有过期。我们维持两条性质：下标从左到右增加，对应的值从左到右严格减小。

## 每一步有两种不同的删除

1. 从队首移除离开窗口的下标。
2. 从队尾移除不大于新值的下标。
3. 加入新下标；窗口凑齐以后，队首就是最大值。

| 新下标和值 | 处理后的队列：下标(值) | 本窗口最大值 |
| --- | --- | --- |
| 0: 4 | 0(4) | 窗口未满 |
| 1: 1 | 0(4), 1(1) | 窗口未满 |
| 2: 3 | 0(4), 2(3) | 4 |
| 3: 5 | 3(5) | 5 |
| 4: 2 | 3(5), 4(2) | 5 |
| 5: 5 | 5(5) | 5 |

最后一个 5 到来时，旧 5 也被删掉。求最大值时留更新的那个就够了；若要求最早出现的位置，就改用严格小于来删队尾，并重新核对并列规则。

## Python 实现

```python
def window_maxima(values, width):
    if type(width) is not int or not 1 <= width <= len(values):
        raise ValueError("width must be an integer between 1 and input length")
    candidates = deque()
    maxima = []
    for index, value in enumerate(values):
        while candidates and candidates[0] <= index - width:
            candidates.popleft()
        while candidates and values[candidates[-1]] <= value:
            candidates.pop()
        candidates.append(index)
        if index >= width - 1:
            maxima.append(values[candidates[0]])
    return maxima
```

先 `from collections import deque`。这里拒绝空输入和不在 `1…len(values)` 之间的窗口宽度，比较的值应当可排序且不含 NaN。不要换成 `list.pop(0)`：列表头删需要移动后面的元素；`deque` 适合两端操作。[Python collections 文档](https://docs.python.org/3/library/collections.html#collections.deque)

`index - width` 是刚刚过期的下标，当前窗口左端是 `index - width + 1`。这是最容易错一位的地方。

## 两个 while 会不会变成平方时间

每个下标进队一次，最多出队一次：要么从头过期，要么从尾被更好的候选淘汰。同一个下标不会被删两次，所以总队列操作是 $O(n)$，辅助空间 $O(k)$，输出另占 $O(n-k+1)$。

数组严格递减时，队列会保留整个窗口；严格递增时，常常只剩最新一个。不能拿后一种输入推断空间永远是 $O(1)$。

## 和堆、单调栈有什么不同

| 方法 | 保留什么 | 代价或限制 |
| --- | --- | --- |
| 逐窗 max | 整个窗口 | 最直观，$O(nk)$ |
| 单调队列 | 尚未被支配的窗口候选 | $O(n)$，依赖窗口向前移动 |
| 懒删除堆 | 值与下标 | 只弹过期堆顶，埋在堆里的旧项可能一直留下，最坏 $O(n)$ 空间 |
| 单调栈 | 等待未来答案的元素 | 适合“右边第一个更大值”；没有自动处理窗口到期 |

懒删除堆若不定期重建或支持索引删除，不能顺手写成 $O(k)$ 空间。需要任意位置删除、分位数等查询时，再考虑更强的数据结构。

## 换个条件试试

把输入改成 `[5,5,5]`、`k=2`，答案仍是 `[5,5]`。再试 `k=1`、`k=n`、全负数和递减输入。你应该能说清每个下标为什么被删，而不只是背两段 while。

代码与小数组穷举测试在 [deeper_patterns.py](../code/deeper_patterns.py) 和 [test_deeper_algorithm_patterns.py](../../site/tests/test_deeper_algorithm_patterns.py)。这是一种窗口方法的教学演示，不代表需要把复习重心转向 Hard 题。
