# Stacks, heaps, and linked lists: choose how work waits

[中文](stack-heap-links.md) · **English** · [Pattern map](../leetcode.en.md)

> Reading time: ~8 min · Last reviewed: 2026-09

A stack remembers the most recently unfinished task. A heap keeps the current priority accessible. A linked list lets you reconnect neighboring nodes. The common mistake is using one as though it were another.

## 1. Stack: close the most recently opened item

For bracket matching, push opening brackets. Each closing bracket must match the top. At the end the stack must be empty. `list.append` / `list.pop()` suffice; don't remove from the front. O(n) time and O(n) space for n brackets.

```text
Input: ([])
Read (: push (
Read [: push [ above (
Read ]: match and pop [
Read ): match and pop (; stack empty
```

If other characters are allowed, define how they should be handled first.

## 2. Monotonic stack: wait for something larger

For each day's temperature, find the wait for a warmer day. Store **unresolved indices**, with nonincreasing temperatures. A new warmer value resolves the top: no earlier warmer day could have been skipped.

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

`[30, 40, 35, 50] → [1, 2, 1, 0]`. Reading 50 resolves 35, then 40. The wait from 40 is 2.

O(n) time, O(n) auxiliary space, O(n) output. Each index is pushed and popped at most once despite the inner loop. `<` means strictly warmer; `<=` would accept equal temperatures. This avoids rescanning suffixes but needs the appropriate order relationship: not every “next” problem qualifies.

## 3. Heap: why use a min-heap for the largest k?

The **smallest retained value is the cutoff**. A smaller new value is irrelevant; a larger one replaces the cutoff.

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

`[5, 1, 4, 2]`, count 2 returns `[5, 4]`. Duplicates remain; k larger than n returns everything.

For k>0, let m = min(k, n). Scanning costs O(n log(m+1)), final sorting O(m log m), and space O(m). k=1 still requires an O(n) scan; don't misread log 1 as zero total work. For just the kth largest, skip sorting and read the heap top, assuming 1≤k≤n.

`heapreplace` always removes the old minimum before inserting; `heappushpop` removes the minimum of the set including the new value. They aren't interchangeable in general. A heap isn't fully sorted: it maintains parent-child order, not a sorted array. [heapq documentation](https://docs.python.org/3/library/heapq.html)

## 4. Linked lists: save the next link before changing it

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

For `1 → 2 → 3`, save the old next link before rewiring, or lose the remaining list. At each loop entry, previous heads the reversed prefix and current heads the unprocessed suffix. O(n) time, O(1) auxiliary space; **mutates the list** and assumes no cycle. Recursive reversal instead uses O(n) call-stack space.

A dummy node simplifies deleting the head or merging lists. Deleting a known node often still needs its predecessor. Finding that predecessor from the head is O(n), so “linked-list deletion is O(1)” needs qualifications.

## 5. Slow and fast pointers: detecting a cycle

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

Compare node identity with `is`, not equal values. Without a cycle fast reaches the end. Inside a cycle, the relative gap changes by one modulo cycle length each iteration, so they meet. O(n) time, O(1) space. This detects a cycle; it doesn't locate its entry.

## 6. Practice and self-checks

- [Valid Parentheses](https://leetcode.com/problems/valid-parentheses/): last in, first out.
- [Daily Temperatures](https://leetcode.com/problems/daily-temperatures/): store indices, not just temperatures.
- [Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/): why retain k elements?
- [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/): identity, not value.

<details class="interview" markdown="1">
<summary>Why not replace an arbitrary retained element whenever a large value arrives?</summary>

You need the retained minimum, or you'd scan the retained set each time. A min-heap keeps that cutoff accessible. For small data or a fully sorted result, simply sorting can be a good choice.

</details>

<details class="interview" markdown="1">
<summary>What happens if reversal omits following = current.next?</summary>

Once current.next points backward, you've lost the old link to the remaining suffix. Save, reconnect, then advance; order matters.

</details>
