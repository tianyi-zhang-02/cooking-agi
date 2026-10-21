# Backtracking and DP: paths or repeated subproblems?

[中文](backtracking-and-dp.md) · **English** · [Pattern map](../leetcode.en.md)

> Reading time: ~12 min · Last reviewed: 2026-10

Recognizing recursion doesn't yet tell you whether to use backtracking or DP. Start with the output: do you need every selection, or just an optimum? If two different sequences of choices lead to the same remaining question, can its answer be computed once? We'll use subsets and a non-adjacent-selection problem to make those questions concrete.

## 1. Backtracking: choose, explore, undo

List every subset of `[1, 2]`, including the empty set: `[], [1], [1, 2], [2]`. Let `path` hold the current selection and `start` mark the next position we may choose from.

```text
[]
├─ take 1 → [1]
│           └─ take 2 → [1, 2]
└─ take 2 → [2]
```

Every node is already a subset, so record it without waiting for a leaf. After choosing 1, only choose from later positions. This also prevents counting `[2, 1]` as another version of `[1, 2]`.

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

The two operations most easily confused here are `copy` and `pop`:

- `copy` saves an independent answer. Without it, all results reference the same list, and later edits change previously saved answers too.
- `pop` undoes the latest choice. After `[1, 2]`, return to `[1]`; after finishing that branch, return to `[]` before choosing 2. Without undoing, the next branch inherits choices from the previous one.

Input values are assumed distinct; duplicates require a separate deduplication rule. There are 2ⁿ subsets containing n·2ⁿ⁻¹ elements in total for n≥1, so output space and time are O(n·2ⁿ). Excluding output, the current path and call stack use O(n) space.

## 2. Three problems, three next steps

| Problem | Next level | Avoiding duplicates |
| --- | --- | --- |
| Subsets / combinations without reuse | Continue at index+1 | Increasing indices |
| Combinations allowing reuse | May continue at the same index | Preserve selection order; pruning needs numeric assumptions |
| Permutations | Try every unused index | Path-local used state, undone on return |

Pruning means recognizing that a branch cannot produce a valid answer and stopping early. When adding positive values toward a target sum, overshooting is enough to stop. With negative values, later choices might bring the sum back down. Likewise, a global `visited` set can be wrong when two different paths are exactly the two answers the problem wants.

## 3. DP: define the state in plain language

Strip away the story in House Robber: given nonnegative values `[2, 7, 9, 3, 1]`, select a maximum-sum set of non-adjacent positions. Choosing 2, 9, and 1 gives 12.

Define `best(index)` as **the maximum available from index onward**. Earlier choices have already been handled; their rewards are not included in this return value.

There are only two possibilities at the current position:

- Skip it: the remaining result is `best(index + 1)`.
- Take it: gain `values[index]`, skip its neighbor, and add `best(index + 2)`.

Taking the larger result gives the recurrence. We get the formula by enumerating the legal choices, rather than guessing a formula and trying to assign it a meaning.

```text
best(0)
├─ skip index 0 → best(1)
│                 ├─ skip index 1 → best(2)
│                 └─ take index 1 → 7 + best(3)
└─ take index 0 → 2 + best(2)
```

`best(2)` appears twice. Despite different earlier choices, both calls ask exactly the same remaining question: what is the best selection from `[9, 3, 1]`? Its answer doesn't depend on how we reached it, so compute it once and reuse it. This is memoization.

<!-- widget:algo-rob -->

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

Beyond the array, nothing remains to select, so return 0. With [cache](https://docs.python.org/3/library/functools.html#functools.cache), each position is evaluated once, comparing two choices: O(n) time, O(n) cache space, and O(n) call-stack space. Don't mutate the input during computation. The nested cache is recreated for each new input.

Why is the index enough? Skipping the next position after taking one already enforces the adjacency rule. If the rule also allowed at most two selections, the remaining allowance would affect the answer and would have to join the state. **Reuse is valid when the remaining problem is genuinely the same, not merely when calls look similar.**

## 4. The same recurrence, implemented iteratively

Before compressing anything into two variables, write down a table. `best[index]` still means the maximum available from this index onward. Since each entry depends on the next two entries, **fill from right to left**:

<div class="algorithm-trace" markdown="1">

| Index | Value | Skip | Take | Best result |
| --- | --- | --- | --- | --- |
| 5, 6 | Past the end | — | — | 0 |
| 4 | 1 | 0 | 1 + 0 | 1 |
| 3 | 3 | 1 | 3 + 0 | 3 |
| 2 | 9 | 3 | 9 + 1 | 10 |
| 1 | 7 | 10 | 7 + 3 | 10 |
| 0 | 2 | 10 | 2 + 10 | 12 |

</div>

```python
def rob_table(values):
    best = [0] * (len(values) + 2)
    for index in range(len(values) - 1, -1, -1):
        best[index] = max(best[index + 1], values[index] + best[index + 2])
    return best[0]
```

The extra two zeros implement the recursive base case. A table also makes debugging easier: if index 2 doesn't produce 10, check that it read the right results from indices 3 and 4.

### Compress space last

Each entry reads only the next two entries; older results won't be needed again. Keep them in `one_ahead` and `two_ahead`, still processing right to left:

```python
def rob_iterative(values):
    one_ahead = two_ahead = 0
    for value in reversed(values):
        current = max(one_ahead, value + two_ahead)
        one_ahead, two_ahead = current, one_ahead
    return one_ahead
```

Before processing 9, the saved results are 3 and 1. Compute `max(3, 9 + 1) = 10`, then replace them with 10 and 3 for the next iteration. Python evaluates the right-hand side before assigning either variable, so the old `one_ahead` isn't overwritten too early.

The table uses O(n) time and auxiliary space. Rolling variables keep O(n) time but reduce auxiliary space to O(1); `reversed` doesn't copy the list either. These implementations store the same results differently. If you also need the selected positions, retaining the table is often simpler: the optimum alone doesn't reconstruct the choices.

## 5. Recursive vs iterative tradeoffs

| Form | Strength | Cost |
| --- | --- | --- |
| Recursion + memoization | Mirrors the definition; visits needed states | Calls, cache, recursion limits |
| Bottom-up table | Explicit order and inspectable state values | May compute unused states; needs dependency order |
| Rolling variables | Small memory, concise implementation | Requires limited historical dependencies; harder reconstruction |

Analyze DP as **number of states × transition cost per state**, plus key construction and output. DP isn't automatically O(n): an n×m table with k choices per state may take O(nmk).

### Explaining this solution aloud

The method is **recursion**; the implementation is **a recursive solution**. Likewise, **iteration** is the noun, and **an iterative solution** describes the implementation. Start with the state, choices, and stopping condition:

> I define `best(index)` as the maximum sum I can get from that index onward without taking adjacent elements. I either skip the current element, or take it and skip the next one. The base case is zero once the index reaches the end.

Then explain what caching saves and what it costs:

> Different choices can lead to the same subproblem, so I cache each result. There are O(n) states, and each takes O(1) work. That gives O(n) time and O(n) auxiliary space, including the recursion stack.

If asked to reduce space, explain the right-to-left table and why only the next two states need to be kept. An optimal complexity claim is less useful than being able to justify it.

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

## 7. Further explanations

- [Hello Algo: introduction to dynamic programming](https://www.hello-algo.com/chapter_dynamic_programming/intro_to_dynamic_programming/) (Chinese, with an English edition available): the climbing-stairs example connects search, memoization, tabulation, and space optimization.
- [leetcode-master: dynamic-programming foundations](https://github.com/youngyangyang04/leetcode-master/blob/master/problems/%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%E7%90%86%E8%AE%BA%E5%9F%BA%E7%A1%80.md) (Chinese): useful checks for state meaning, recurrence, initialization, and iteration order, including a hand-computed table.

Some explanations define prefix states; this note uses suffix states. Either can work. Before changing loop direction, check what every variable means rather than mixing conventions.
