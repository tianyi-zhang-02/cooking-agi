# Knapsack DP: loop direction decides how often an item can be used

[中文](knapsack.md) · **English** · [Pattern map](../leetcode.en.md)

> Last reviewed: 2026-10-09 · Prerequisite: [Backtracking and DP](backtracking-and-dp.en.md)

Capacity is 5. Three items have weight–value pairs `(2,5), (3,7), (4,8)`. With each item available once, the first two give value 12. Taking the largest value first selects the weight-4 item and gets only 8.

The missing ingredient isn't a clever sorting key. A choice changes the remaining capacity. DP records that condition instead of repeatedly exploring the same continuation.

## Define two dimensions before compressing to one row

$F(i,c)$ is the best value using the first $i$ items with total weight at most $c$. Item $i$ has weight $w_i$ and value $v_i$:

$$
F(i,c)=
\begin{cases}
F(i-1,c),&c<w_i,\\
\max\{F(i-1,c),F(i-1,c-w_i)+v_i\},&c\ge w_i.
\end{cases}
$$

The alternatives skip or take the item. Both read the **previous row**, so the current item cannot be used twice. Taking nothing is allowed, giving $F(0,c)=0$.

| Items considered | Capacity 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| None | 0 | 0 | 0 | 0 | 0 | 0 |
| (2,5) | 0 | 0 | 5 | 5 | 5 | 5 |
| Then (3,7) | 0 | 0 | 5 | 7 | 7 | 12 |
| Then (4,8) | 0 | 0 | 5 | 7 | 8 | 12 |

After compression, `best[c]` still means “at most c,” not “exactly c.” A larger capacity may hold a lighter combination.

## Why 0/1 knapsack updates capacities downward

```python
def knapsack_once(items, capacity):
    if type(capacity) is not int or capacity < 0:
        raise ValueError("capacity must be a nonnegative integer")
    best = [0] * (capacity + 1)
    for weight, value in items:
        if type(weight) is not int or weight <= 0:
            raise ValueError("weights must be positive integers")
        for room in range(capacity, weight - 1, -1):
            best[room] = max(best[room], best[room - weight] + value)
    return best[capacity]
```

Items are `(positive integer weight, numeric value)` pairs; capacity is a nonnegative integer. Each item is available at most once, and selecting nothing is allowed. Negative values are simply not selected. The function returns the value, not the chosen items.

With only `(2,5)` and capacity 4, the answer is 5. Updating upward writes `best[2]=5` and then reads that new value for `best[4]`, producing 10: it reused the same item. Updating downward leaves `best[c-weight]` untouched in the current pass, preserving the previous-row dependency.

For $n$ items and integer capacity $C$, time is $O(nC)$ and auxiliary space $O(C)$. This is pseudopolynomial: the numeric value of $C$ may greatly exceed the number of bits needed to encode it. A small item count alone doesn't make a huge-capacity DP cheap. [MIT pseudopolynomial algorithms notes](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/resources/mit6_006s20_lec18/)

## Why repetition reverses the direction

For coins `[1,3,4]` and amount 6, the minimum is two coins: `3+3`. Reading a state updated in the same pass is now desirable because a denomination may be reused.

```python
def minimum_coins(coins, amount):
    if type(amount) is not int or amount < 0:
        raise ValueError("amount must be a nonnegative integer")
    if any(type(coin) is not int or coin <= 0 for coin in coins):
        raise ValueError("coins must be positive integers")
    best = [0] + [float("inf")] * amount
    for coin in coins:
        for subtotal in range(coin, amount + 1):
            best[subtotal] = min(best[subtotal], best[subtotal - coin] + 1)
    return -1 if best[amount] == float("inf") else best[amount]
```

`best[subtotal]` is the minimum number of coins making **exactly** that amount. Only `best[0]=0`; all other amounts begin unreachable, represented by infinity. Return -1 for an unreachable target. Positive integer denominations avoid zero-weight ambiguities; repeated denominations don't change this minimum-value problem.

For $d$ denominations and amount $A$, time is $O(dA)$ and space $O(A)$. This is not greedy change-making: changing denominations can invalidate “take the largest coin first.”

## Similar-looking problems need different initialization

| Question | Initial states | Transition |
| --- | --- | --- |
| Best value within capacity | Every capacity starts at 0 | max |
| Best value filling capacity exactly | Capacity 0 is 0; others negative infinity | Extend reachable states only |
| Fewest coins making an exact amount | Amount 0 is 0; others positive infinity | min |
| Number of ways | Amount 0 is 1; others 0 | Add, with an explicit order convention |

For coin combinations, iterate denominations outside and increasing amounts inside. Reversing those loops counts ordered sequences. With `[1,2]` and amount 3, the combinations are `1+1+1` and `1+2`; ordered sequences also include `2+1`. This isn't merely code style.

## Recovering a solution requires more history

A one-row DP retains the optimal value. To recover items, keep the two-dimensional table and backtrack: taking an item reduces capacity; skipping it only moves to the previous row. Overwriting a one-row DP while recording a casual predecessor pointer doesn't guarantee that historical choices remain valid.

Test no items, capacity 0, all items too heavy, and the single-item `(2,5)` / capacity-4 counterexample. Code and exhaustive checks: [deeper_patterns.py](../code/deeper_patterns.py). The important question is which layer every read refers to, not which loop you memorized.
