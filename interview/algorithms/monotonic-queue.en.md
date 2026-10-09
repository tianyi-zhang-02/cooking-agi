# Monotonic queues: what replaces a maximum when it leaves?

[中文](monotonic-queue.md) · **English** · [Pattern map](../leetcode.en.md)

> Last reviewed: 2026-10-09 · Prerequisite: [Stacks, heaps, and lists](stack-heap-links.en.md)

For `[4,1,3,5,2,5]`, the maxima of consecutive windows of width 3 are `[4,5,5,5]`. Recomputing max for each window works, but costs $O(nk)$ for input length $n$ and width $k$.

Keeping just the current maximum isn't enough: when 4 leaves, who replaces it? A monotonic queue keeps the **indices that could still become the maximum**.

## When an element can be discarded permanently

Suppose an earlier 1 is followed by 3. Every future window still containing that 1 must also contain the later 3. The 3 is larger and expires later, so the 1 can never win again.

This depends on a forward-moving, fixed-width window. The 1 hasn't disappeared from the data; it has stopped affecting this query. For sums, second-largest values, or every tied maximum's position, that deletion may be invalid.

Store indices, not just values, to detect expiry. Indices increase from front to back; their values strictly decrease.

## Two different reasons to remove an index

1. Remove expired indices from the front.
2. Remove values no larger than the new value from the back.
3. Append the new index; once the window is full, its maximum is at the front.

| New index and value | Queue after processing: index(value) | Window maximum |
| --- | --- | --- |
| 0: 4 | 0(4) | Not full |
| 1: 1 | 0(4), 1(1) | Not full |
| 2: 3 | 0(4), 2(3) | 4 |
| 3: 5 | 3(5) | 5 |
| 4: 2 | 3(5), 4(2) | 5 |
| 5: 5 | 5(5) | 5 |

The last 5 removes the older 5 too. The newer copy is sufficient for the maximum value. To return the earliest tied position, remove only strictly smaller values and check the tie-breaking contract again.

## Python implementation

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

Import `deque` from `collections` first. This contract rejects empty input and widths outside `1…len(values)`. Values must be orderable and not NaN. Don't replace the queue with `list.pop(0)`: removing a list's first element shifts the rest; `deque` is designed for end operations. [Python collections documentation](https://docs.python.org/3/library/collections.html#collections.deque)

`index - width` has just expired. The current left boundary is `index - width + 1`. Keep those two positions distinct.

## Do the two while loops make this quadratic?

Each index enters once and leaves at most once, either through expiry at the front or domination at the back. It cannot be removed twice. Total queue operations are $O(n)$, with $O(k)$ auxiliary space and a separate $O(n-k+1)$ result.

Decreasing input can retain the full window. Increasing input often retains only the newest index; that does not make the worst-case space $O(1)$.

## How this differs from a heap or monotonic stack

| Method | What survives | Cost or limitation |
| --- | --- | --- |
| Recompute max | Entire window | Simplest; $O(nk)$ |
| Monotonic queue | Undominated window candidates | $O(n)$; needs forward-moving windows |
| Lazy-deletion heap | Values and indices | Expired items buried below the root may remain; worst-case $O(n)$ space |
| Monotonic stack | Items awaiting a future answer | Useful for next-greater-element queries; no automatic window expiry |

A lazy heap is not automatically $O(k)$ space. You need a bound such as rebuilding or indexed deletion. Arbitrary deletions or quantile queries may justify a richer structure.

## Change a condition

For `[5,5,5]` and `k=2`, the result remains `[5,5]`. Also try `k=1`, `k=n`, negative values, and decreasing input. Explain why each removed index is no longer needed, rather than memorizing two while loops.

Code and exhaustive small-array checks: [deeper_patterns.py](../code/deeper_patterns.py), [test_deeper_algorithm_patterns.py](../../site/tests/test_deeper_algorithm_patterns.py). This teaches a transferable window technique; it doesn't change the notebook into a Hard-problem collection.
