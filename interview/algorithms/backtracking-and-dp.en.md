# Backtracking and DP: paths or repeated subproblems?

[中文](backtracking-and-dp.md) · **English** · [Pattern map](../leetcode.en.md)

> Reading time: ~8 min · Last reviewed: 2026-09

Both may produce recursion trees, but they focus on different things. Backtracking explores legal choices; DP reuses repeated subproblem results. They can be combined, so the presence of recursion doesn't distinguish them.

## 1. Backtracking: choose, explore, undo

The subsets of `[1, 2]` are `[], [1], [1,2], [2]`. path holds current choices. start enforces increasing indices so `[1,2]` and `[2,1]` don't duplicate the same subset.

```python
def subsets(values):
    result = []
    path = []

    def extend(start):
        result.append(path.copy())
        for index in range(start, len(values)):
            path.append(values[index])
            extend(index + 1)
            path.pop()

    extend(0)
    return result
```

Assumes distinct input values. Copy each answer or all outputs would reference the same mutable path. Auxiliary space O(n); output space and time O(n·2ⁿ). Emitting every subset is inherently expensive. Duplicate inputs require additional deduplication rules.

## 2. Three problems, three next steps

| Problem | Next level | Avoiding duplicates |
| --- | --- | --- |
| Subsets / combinations without reuse | Continue at index+1 | Increasing indices |
| Combinations allowing reuse | May continue at the same index | Preserve selection order; pruning needs numeric assumptions |
| Permutations | Try every unused index | Path-local used state, undone on return |

**Pruning needs a reason.** With positive values, a negative remaining target can end a branch; with negative values that argument fails. Global visited may also be wrong because different paths can be distinct outputs.

## 3. DP: define the state in plain language

House Robber's simplified model: nonnegative rewards in a row; adjacent positions can't both be selected. `best(index)` means “the maximum available from index onward.” Skip this position, or take it and skip the next.

```python
from functools import cache

def rob_memo(values):
    @cache
    def best(index):
        if index >= len(values):
            return 0
        return max(best(index + 1), values[index] + best(index + 2))

    return best(0)
```

Without caching, indices are recomputed repeatedly. [cache](https://docs.python.org/3/library/functools.html#functools.cache) solves each state once: O(n) time and O(n) cache plus call-stack space. Don't mutate input during computation. The nested cache is recreated per outer call to avoid mixing inputs.

## 4. The same recurrence, implemented iteratively

Moving left to right, one_back is the best processed-prefix result and two_back is the result one position earlier. Skip the new value or add it to two_back.

```python
def rob_iterative(values):
    two_back = one_back = 0
    for value in values:
        current = max(one_back, two_back + value)
        two_back, one_back = one_back, current
    return one_back
```

For `[2, 7, 9, 3, 1]`, prefix optima are `2, 7, 11, 11, 12`. O(n) time, O(1) auxiliary space. State compression returns only the optimum; reconstructing selected positions generally needs extra information or recomputation.

## 5. Recursive vs iterative tradeoffs

| Form | Strength | Cost |
| --- | --- | --- |
| Recursion + memoization | Mirrors the definition; visits needed states | Calls, cache, recursion limits |
| Bottom-up table | Explicit order and inspectable state values | May compute unused states; needs dependency order |
| Rolling variables | Small memory, concise implementation | Requires limited historical dependencies; harder reconstruction |

Analyze DP as **number of states × transition cost per state**, plus key construction and output. DP isn't automatically O(n): an n×m table with k choices per state may take O(nmk).

## 6. A few problems and self-checks

- [Subsets](https://leetcode.com/problems/subsets/): path copying and undoing.
- [Permutations](https://leetcode.com/problems/permutations/): does order distinguish answers?
- [House Robber](https://leetcode.com/problems/house-robber/): recursion → cache → table → rolling variables.
- [Coin Change](https://leetcode.com/problems/coin-change/): define “fewest coins for this amount,” then count states and transitions.

<details class="interview" markdown="1">
<summary>Why doesn't adding cache make every backtracking problem polynomial?</summary>

Caching needs a complete reusable state. The current path may affect legality and output. Even if repeated work disappears, all-subset enumeration still has exponentially many outputs.

</details>

<details class="interview" markdown="1">
<summary>Can “take the largest available reward” solve House Robber?</summary>

No. In `[4, 5, 4]`, choosing 5 loses to taking both ends for 8. DP compares mutually exclusive possibilities instead of choosing by immediate size alone.

</details>
