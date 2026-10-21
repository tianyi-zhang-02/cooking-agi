# Sorting, intervals, and greedy choices: why won't you regret this step?

[中文](sorting-and-greedy.md) · **English** · [Pattern map](../leetcode.en.md)

> Reading time: ~6 min · Last reviewed: 2026-10

Sorting creates useful order; it isn't the solution by itself. Greedy doesn't mean “pick what looks best.” You need to explain why the local choice doesn't exclude a better answer.

## 1. What sorting buys

Sorting intervals by start lets you compare only with the last merged interval, rather than every old interval. For selecting the most nonoverlapping activities, sorting by **finish time** is useful: an earlier finish leaves room for later activities.

Different goals need different keys. `sorted(intervals, key=lambda interval: interval[0])` sorts by start; `key=lambda interval: interval[1]` by finish. “Interval problem” doesn't identify one universal template.

## 2. Merging intervals: watch the last segment

Assume valid closed intervals `[start, end]`, start≤end; touching endpoints merge.

```python
def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged
```

`[[5,7], [1,3], [2,4]]` sorts to `[1,3], [2,4], [5,7]` and merges to `[[1,4], [5,7]]`. If the current interval cannot reach the last merged segment, it cannot reach any earlier completed segment either.

O(n log n) time, O(n) sorted-copy space, and up to O(n) output, without input mutation. Already start-sorted input can be scanned in O(n). For half-open intervals `[start,end)`, whether touching intervals should combine depends on the problem contract.

## 3. Greedy: summarize choices with a reachable frontier

Jump Game uses nonnegative maximum forward jump lengths. We only ask whether the end is reachable, not for a path or minimum jump count.

```python
def can_jump(jumps):
    if not jumps:
        return False
    farthest = 0
    for index, jump in enumerate(jumps):
        if index > farthest:
            return False
        farthest = max(farthest, index + jump)
        if farthest >= len(jumps) - 1:
            return True
    return False
```

For `[2,3,1,1,4]`, position 0 reaches 2; reachable position 1 extends the frontier to 4. For `[3,2,1,0,4]`, the frontier stays at 3, leaving position 4 unreachable.

<!-- widget:algo-jump -->

**Invariant:** processed reachable positions cover the entire interval from 0 to farthest. Jumps are “up to” a length, so no holes appear between. O(n) time, O(1) space. Exact-length jumps, backward moves, or complex obstacles break this argument.

## 4. Make a greedy argument convincing

- **Invariant:** explain why the retained summary captures everything future choices need.
- **Exchange argument:** for maximum nonoverlapping activity count, replace an optimal schedule's first activity with the earliest-finishing one. It finishes no later, so the remaining activities still fit. Repeat on the remaining problem.
- **Counterexample:** coins `[1,3,4]`, amount 6. Largest-first yields `4+1+1`, worse than `3+3`. Look for small counterexamples before trusting intuition.

### Walk the exchange argument through five activities

Activities are half-open intervals `[start, end)`: if one ends at 6, the next can start at 6. Take A `[0,7)`, B `[1,3)`, C `[2,4)`, D `[4,6)`, and E `[6,8)`, and ask for the largest set of nonoverlapping activities. Sort by finish time and remember only when the last chosen activity ends:

| Order by finish time | Interval | Does it fit after the last choice? | Chosen |
| --- | --- | --- | --- |
| B | `[1,3)` | Nothing chosen yet; take it | B |
| C | `[2,4)` | 2 is before 3: conflict, skip | B |
| D | `[4,6)` | 4 is not before 3: take it | B, D |
| A | `[0,7)` | 0 is before 6: skip | B, D |
| E | `[6,8)` | 6 is not before 6: take it | B, D, E |

The result is 3. C, D, E is another schedule of 3. Replace its first activity, C, with the earliest-finishing B: B ends at 3, before C's 4, so D can still start at 4 and the count is unchanged. After the swap, the remaining problem is “how many more fit after time 3?” Greedy makes the same choice on that smaller problem, and the same argument applies again. For the full proof, see [the interval-scheduling section of the Kleinberg–Tardos greedy slides](https://www.cs.princeton.edu/~wayne/kleinberg-tardos/pdf/04GreedyAlgorithmsI.pdf).

A rule that sounds just as reasonable can fail. Earliest start takes A first: it occupies 0 to 7, every later activity starts by 6 and conflicts, so only 1 fits. Shortest duration has a counterexample too: among `[0,4)`, `[4,8)`, and `[3,5)`, taking the shortest `[3,5)` blocks the other two, giving 1 where the optimum is 2.

Sorting costs O(n log n) and the scan O(n), storing only one finish time. If activities carry different weights and you maximize total weight instead of the count, this exchange breaks: the swapped-in B might be worth very little. Sort by finish time and use DP instead; see [the dynamic-programming slides from the same series](https://www.cs.princeton.edu/~wayne/kleinberg-tardos/pdf/06DynamicProgrammingI.pdf).

Interval merging is a sorted scan. Not every sort-and-scan algorithm is a greedy optimization. Explain what you're proving rather than merely naming a technique.

## 5. Practice and first-pass scope

- [Merge Intervals](https://leetcode.com/problems/merge-intervals/): closed endpoints, nesting, empty input.
- [Jump Game](https://leetcode.com/problems/jump-game/): explain why a frontier is sufficient.
- [Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/): retain the earlier minimum price and evaluate selling today; buying must precede selling.

Start with scanning, lookup, traversal, and state patterns. The [task-dependency exercise](../coding-checks.en.md#dependencies) adds topological sorting and distinguishes it from parallel execution. Trie and Union-Find can be studied as needed; completeness is not a reason to jump straight to Hard problems.

<details class="interview" markdown="1">
<summary>Why sort by start for merging and finish for activity count?</summary>

Merging organizes coverage, and start order makes overlap local. Activity selection preserves time for future choices, with an exchange argument for earliest finish. Different goals justify different orders.

</details>

<details class="interview" markdown="1">
<summary>Does this Jump Game template return the minimum number of jumps?</summary>

No, it tracks reachability only. Minimum jumps needs additional current-layer and next-layer boundaries, or a layered search. A different output requirement isn't automatically solved by the same template.

</details>
