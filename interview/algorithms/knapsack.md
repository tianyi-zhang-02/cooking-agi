# 背包 DP：循环方向，其实是在决定能拿几次

**中文** · [English](knapsack.en.md) · [方法地图](../leetcode.md)

> 最近审阅：2026-10-09 · 先读：[回溯与 DP](backtracking-and-dp.md)

容量 5，有三件物品：重量和价值分别是 `(2,5)、(3,7)、(4,8)`。每件最多拿一次，最优是前两件，价值 12。按价值最大先拿，会选重量 4 的那件，反而只有 8。

这里不缺一个聪明的排序 key，而是每次选择都会改变剩余容量。DP 把这个剩余条件记下来，避免反复枚举同一种后续问题。

## 先写二维状态，再压成一行

$F(i,c)$ 表示只考虑前 $i$ 件物品、总重量不超过 $c$ 时的最大价值。第 $i$ 件重量为 $w_i$、价值为 $v_i$：

$$
F(i,c)=
\begin{cases}
F(i-1,c),&c<w_i,\\
\max\{F(i-1,c),F(i-1,c-w_i)+v_i\},&c\ge w_i.
\end{cases}
$$

两项分别是不拿、拿这件。两种都引用**上一行**，所以当前物品不会用两次。允许什么都不拿，因此 $F(0,c)=0$。

| 已考虑物品 | 容量 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| 无 | 0 | 0 | 0 | 0 | 0 | 0 |
| (2,5) | 0 | 0 | 5 | 5 | 5 | 5 |
| 再加 (3,7) | 0 | 0 | 5 | 7 | 7 | 12 |
| 再加 (4,8) | 0 | 0 | 5 | 7 | 8 | 12 |

压成一维时，`best[c]` 仍表示“不超过 c”，不是“刚好用满 c”。大容量格里可以放更轻的组合。

## 为什么 0/1 背包必须倒着更新

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

物品列表提供 `(正整数重量, 数值价值)`，容量是非负整数。每件至多用一次，可以不选，负价值自然不会被选。函数不返回具体选择。

只给一件 `(2,5)`、容量 4，正确答案是 5。若从小到大更新，先把 `best[2]` 写成 5，随后 `best[4]` 又读这个新值，得到 10：同一件物品被拿了两次。倒着遍历时，`best[c-weight]` 还没被本轮修改，读到的才是上一行。

$n$ 件、整数容量 $C$：时间 $O(nC)$、辅助空间 $O(C)$。这是伪多项式复杂度：$C$ 的数字值可能远大于输入里表示它需要的位数。容量特别大，不能只看物品数量说“DP 很快”。[MIT 伪多项式讲义](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/resources/mit6_006s20_lec18/)

## 可以重复拿时，顺序为什么反过来

硬币 `[1,3,4]`，凑金额 6，最少用 2 枚：`3+3`。现在读当前行的结果恰好是想要的，因为一种面额允许多次使用。

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

`best[subtotal]` 表示**恰好**凑到该金额的最少枚数。只有 `best[0]=0`；其他金额初始不可达，用无穷大表示。不可达时返回 -1。正整数面额避免零重量循环之类的歧义；重复面额不影响这个最小值问题。

有 $d$ 种面额、金额 $A$，时间 $O(dA)$，空间 $O(A)$。这仍不是最少硬币的贪心算法：面额一改，“总拿最大的”就可能失效。

## 几个相似题目，初始化却不一样

| 题目在问什么 | 状态初值 | 转移 |
| --- | --- | --- |
| 不超过容量的最大价值 | 所有容量都是 0 | max |
| 恰好用满的最大价值 | 0 容量为 0，其余为负无穷 | 只从可达状态转移 |
| 恰好凑钱的最少枚数 | 0 金额为 0，其余为正无穷 | min |
| 凑法数量 | 0 金额为 1，其余为 0 | 加法，还要明确是否区分顺序 |

数硬币组合时通常先遍历面额，再向上遍历金额；若先金额再面额，会数出有序序列。对 `[1,2]` 凑 3，组合只有 `1+1+1、1+2` 两种，有序序列还包括 `2+1`。这不是一个无关紧要的代码风格选择。

## 想拿回方案，需要再留什么

一维数组只保留最优值，足够回答“值是多少”。若要输出物品，可以保留二维表并从末格倒推：选中当前物品时减去重量，否则只退一行。不要边覆盖一维 DP 边随手记一个 predecessor，就假定历史选择还存在。

先检验空物品、容量 0、全部装不下，以及 `(2,5)` / 容量 4 的单物品反例。代码与穷举测试在 [deeper_patterns.py](../code/deeper_patterns.py)。背包的重点不是背循环，而是每次读取的是上一层还是当前层。
