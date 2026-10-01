# 栈、堆与链表：给不同的等待方式选容器

**中文** · [English](stack-heap-links.en.md) · [方法地图](../leetcode.md)

> 阅读时间：约 8 分钟 · 最近审阅：2026-09

栈记“最近还没处理完的事”；堆维护“当前优先级最高的事”；链表让你直接重接相邻节点。名字都熟，真正容易错的是把一种容器当另一种用。

## 1. Stack：最近打开的，最先关上

括号匹配时，左括号入栈；右括号必须与栈顶匹配。遍历结束后栈要空。`list.append` / `list.pop()` 就够了，别从头部删除。n 个括号最多各进出一次，时间 O(n)，空间 O(n)。

```text
输入 "([])"
读 (：栈底 → (
读 [：栈底 → ( → [（栈顶）
读 ]：匹配 [，弹出；剩下 (
读 )：匹配 (，弹出；栈空
```

这里展示的是字符顺序，实际字符串可写成 `"([])"`。如果题目不只允许括号，要先明确普通字符怎么处理。

## 2. Monotonic stack：等待更大的那个

每天的温度，想知道还要几天才更暖。栈里存**还没找到答案的下标**，对应温度保持不增。新温度更高时，它就是栈顶那天遇到的第一个更高温度。

```python
def next_warmer(temperatures):
    waits = [0] * len(temperatures)
    pending = []
    for index, temperature in enumerate(temperatures):
        while pending and temperatures[pending[-1]] < temperature:
            earlier = pending.pop()
            waits[earlier] = index - earlier
        pending.append(index)
    return waits
```

`[30, 40, 35, 50] → [1, 2, 1, 0]`。读到 50 时，先解决 35，再解决 40；40 之前一直没遇到更暖的，所以距离是 2。

时间 O(n)，辅助空间 O(n)，输出 O(n)。虽然有内层 while，每个下标只压栈、弹栈各一次。`<` 表示严格更高；改成 `<=` 就会把同温也当成答案。优势是避免为每一天向后重扫；限制是问题得有这种可维护的顺序关系，不是所有“下一个”都能套。

## 3. Heap：保留最大的 k 个，为什么用小顶堆

因为保留集合里**最小的那个是淘汰线**。新值连它都不如，就不用留；比它大，就替换它。

```python
import heapq

def largest_values(values, count):
    if count <= 0:
        return []
    heap = []
    for value in values:
        if len(heap) < count:
            heapq.heappush(heap, value)
        elif value > heap[0]:
            heapq.heapreplace(heap, value)
    return sorted(heap, reverse=True)
```

`[5, 1, 4, 2]` 取 2 个，结果 `[5, 4]`。这里保留重复值，k 超过 n 时返回全部。

令 m = min(k, n)，k>0 时，扫描时间 O(n log(m+1))，最后排序 O(m log m)，空间 O(m)。k=1 仍需要 O(n) 扫描，别被 `log 1=0` 绕进去。若只要第 k 大，不必排序，最后看堆顶即可（前提 1≤k≤n）。

`heapreplace` 一定弹出旧堆顶再插入；`heappushpop` 会在加入新值后的集合里弹最小。两者不总等价。堆也不是完全有序的数组，保证的是父节点不大于孩子；别直接把 heap 当排序结果。[heapq 文档](https://docs.python.org/3/library/heapq.html)

## 4. Linked list：先保住后路，再改指向

```python
class ListNode:
    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node


def reverse_list(head):
    previous = None
    current = head
    while current is not None:
        following = current.next
        current.next = previous
        previous = current
        current = following
    return previous
```

`1 → 2 → 3` 反转时，先保存原来的 next，否则一改指针就找不到后面的节点。循环开始时 previous 是已反转前缀的头，current 是尚未处理部分的头。时间 O(n)，辅助空间 O(1)，**会修改原链表**；输入需无环。用递归也能反转，但调用栈变 O(n)。

删头节点、合并两条链时，dummy node 可以减少“第一个节点”特判。删除一个已知节点通常还需要它的前驱；从头找前驱依然 O(n)，不能笼统说“链表删除都 O(1)”。

## 5. 快慢指针：判断有没有环

```python
def has_cycle(head):
    slow = fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False
```

判断的是节点身份 `is`，不是节点的值相等。无环时 fast 会到末尾；有环时，两者在环内的相对距离每轮减少 1（模环长），最终相遇。时间 O(n)，空间 O(1)。这里仅检测是否有环，不是寻找环入口。

## 6. 例题与自测

- [Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)：stack 的后进先出。
- [Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)：存下标，不只是温度。
- [Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)：堆大小为什么是 k。
- [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)：从身份而不是值来判断。

<details class="interview" markdown="1">
<summary>为什么 Top K 不是看到一个大数就覆盖数组里的某个位置？</summary>

必须知道保留集合中最小值在哪里，否则每次都要重新扫描。小顶堆让淘汰线一直在堆顶。n 不大、要完整排序时，直接 sorted 也完全合理。

</details>

<details class="interview" markdown="1">
<summary>反转链表时省略 following = current.next 会怎样？</summary>

把 current.next 改成 previous 后，原来通向剩余链表的指针就丢了。每轮先保存、再重连、再前进，顺序不能随便改。

</details>
