# Pointers, windows, and binary search: justify the move

[中文](pointers-and-search.md) · **English** · [Pattern map](../leetcode.en.md)

> Reading time: ~7 min · Last reviewed: 2026-09

All three narrow a search, but for different reasons. Don't just memorize `left += 1`; explain what that step rules out.

## 1. Two pointers: order eliminates combinations

In `[1, 3, 4, 7]` with target 8, the endpoints already sum to 8. If the sum is too small, moving right inward only decreases it, so discard the current left endpoint. Reverse the reasoning when the sum is too large.

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

Requires ascending input; returns zero-based indices. O(n) time, O(1) auxiliary space. It avoids a map but cannot be applied directly to unsorted input. Sorting adds cost and complicates original indices.

Another shape is **read/write pointers**: read scans, write marks the next retained position. Useful for in-place filtering and deduplication. The invariant is that `values[:write]` is the correct retained portion of the processed input.

## 2. Sliding windows: maintain a valid interval

For the longest unique substring, a repeated character moves the left boundary past its previous occurrence. Never move the boundary backward.

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

In `"abba"`, the second b moves left to 2. The final a previously appeared at 0: don't move left back to 1. The answer is 2, not 3.

Average O(n) time, O(min(n, alphabet size)) space. This version jumps using the last position. A count-based version removes leftmost elements in a `while` until validity returns. **For a longest valid window, update after restoring validity. For a shortest satisfying window, usually record while it still satisfies the condition, then shrink.** Don't copy the update position blindly.

## 3. Not every subarray problem supports a window

Positive values make expansion increase the sum and contraction decrease it. That supports finding the shortest subarray with sum at least target. Signed values break this: for `[1, -1, 3]` and target 3, removing the first 1 drops the sum to 2. Stopping there misses the one-element `[3]`.

When conditions change, choose a different method or re-prove monotonicity. For counting signed subarrays with an exact sum, try [prefix sums plus a map](hash-and-prefix.en.md).

## 4. Binary search: define the boundaries first

Find the first position `>= target`, or n if absent. Use `[left, right)`: everything left of left is smaller than target; everything at or beyond right is at least target.

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

For `[1, 2, 2, 5]`, target 2 returns 1; target 6 returns 4. O(log n) time, O(1) space. To test existence, also check `index < len(values)` and `values[index] == target`. The library equivalent is [bisect_left](https://docs.python.org/3/library/bisect.html).

“Binary search on the answer” uses the same idea: if feasibility holds for a capacity and every larger capacity, search for the first feasible one. Time is **predicate cost × number of search steps**, not automatically O(log n).

## 5. A few problems, many changed conditions

| Practice | Focus |
| --- | --- |
| [Two Sum II](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/) | Justify discarding an endpoint; the problem returns one-based indices |
| [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/) | Never move left backward |
| [Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/) | Positive-input assumption and answer-update timing |
| [Search Insert Position](https://leetcode.com/problems/search-insert-position/) | Empty intervals, duplicates, past-the-end results |

<details class="interview" markdown="1">
<summary>Why can a window with an inner while loop still take O(n)?</summary>

Each element enters and leaves at most once. This assumes O(1) bookkeeping per change; rescanning the window invalidates the argument.

</details>

<details class="interview" markdown="1">
<summary>Is “binary search requires a sorted array” the whole story?</summary>

No. It needs a monotone predicate separating false from true. A sorted array is just the familiar case.

</details>
