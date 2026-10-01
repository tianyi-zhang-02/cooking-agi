# Complexity and Python: know what the tool does

[中文](complexity-and-tools.md) · **English** · [Pattern map](../leetcode.en.md)

> Reading time: ~6 min · Last reviewed: 2026-09

Choosing the right container usually helps more than shortening a line. Are you looking something up, counting, processing in order, or repeatedly extracting a minimum?

## 1. Complexity isn't a count of loops

A `while` inside a `for` isn't automatically O(n²). If both pointers only advance, each moves at most n times: O(n) overall. Conversely, one loop copying `values[:index]` each time copies 0 through n−1 elements in total: O(n²).

We assume constant-cost integer operations and comparisons. Very large integers, long strings, and tuple keys require accounting for arithmetic, comparison, and hashing lengths.

| Shape | Time | Reason |
| --- | --- | --- |
| One scan | O(n) | Constant work per element |
| Halve the range | O(log n) | n → n/2 → n/4 |
| Sort | O(n log n) worst case | One function call isn't constant work |
| All pairs | O(n²) | About n(n−1)/2 pairs |
| Produce and copy every subset | O(n·2ⁿ) | Count and total length of outputs matter |

## 2. Average and amortized are different

- **Average / expected:** `dict` lookup is usually O(1) under hashing assumptions. Extreme collisions can make it O(n).
- **Amortized:** an individual `list.append` can trigger an expensive resize, but a long sequence costs O(1) per append amortized.

Normally exclude input space, include extra containers and recursion frames, and report output storage separately when it dominates.

## 3. Container cheat sheet

| Tool | Operations | Cost and limits |
| --- | --- | --- |
| `list` | `append` / `pop()` / `[-1]` | Amortized O(1) append; O(1) tail removal and indexing |
| `list` | `pop(0)` / `insert(0, x)` / membership | O(n); avoid it as a BFS queue |
| `dict` / `set` | Lookup, insertion, deletion | Usually average O(1); a set loses counts |
| `deque` | `append` / `popleft` | O(1) at the ends; middle indexing isn't O(1) |
| `heapq` | `heappush` / `heappop` | O(log n); minimum at `heap[0]` in O(1) |
| `heapq` | `heapify` | O(n), unlike n separate pushes |
| `bisect` | `bisect_left` / `bisect_right` | O(log n) search, but list insertion remains O(n) |

These bounds describe usual Python implementations with the assumptions above. Official APIs: [dict / set](https://docs.python.org/3/library/stdtypes.html#mapping-types-dict), [deque / Counter](https://docs.python.org/3/library/collections.html), [heapq](https://docs.python.org/3/library/heapq.html), [bisect](https://docs.python.org/3/library/bisect.html).

## 4. Functions you'll actually use

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

Results: `counts['a'] == 3`, `first == 'start'`, `left == 1`, `right == 3`. The count of 2 is `right-left == 2`; `smallest == 1`.

`enumerate(values)` supplies index and value; `zip(first, second)` stops at the shorter input by default; `sorted(values, key=...)` returns a new list; `values.sort()` mutates and returns `None`. Negating numeric priorities gives a max-heap on older Python versions. Python 3.14 also exposes `_max` APIs; check the runtime first.

## 5. Recursion, slices, and outputs use space too

A tree shaped like a chain can need n recursive frames even without an explicit list: O(n) stack space. O(log n) height requires a balanced binary tree.

`values[left:right]` copies that slice; `''.join(parts)` allocates the output string. “In place” doesn't always mean O(1) auxiliary space: sorting may still use temporary storage internally.

## 6. Close the notes and try

<details class="interview" markdown="1">
<summary>Why can nested loops with advancing pointers take O(n)?</summary>

Count movements over the entire run. Each pointer advances at most n times. With O(1) updates, the total is O(n). Copying or summing each window adds work and changes that conclusion.

</details>

<details class="interview" markdown="1">
<summary>If bisect finds a position in O(log n), why isn't a sorted-list insertion O(log n)?</summary>

Finding a position and moving elements are separate costs. Inserting in the middle shifts the tail, taking O(n) in the worst case.

</details>

Next: [Hash maps: what should you remember?](hash-and-prefix.en.md).
