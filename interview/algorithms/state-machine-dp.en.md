# State-machine DP: profit alone doesn't tell you what you may do next

[中文](state-machine-dp.md) · **English** · [Pattern map](../leetcode.en.md)

> Last reviewed: 2026-10-09 · Prerequisite: [Backtracking and DP](backtracking-and-dp.en.md)

This is an algorithm exercise, not investment advice: prices are given for each day, at most one unit may be held, and buying is forbidden on the day after selling. Find the best final profit without a position.

“Best profit through yesterday” is insufficient. It might come from selling yesterday or from resting for a day. The amounts could match while **today's legal actions differ**. That distinction requires another state.

## Separate states by what tomorrow permits

At the end of each day:

| State | Meaning | Tomorrow's options |
| --- | --- | --- |
| holding | Own one unit; profit includes its purchase cost | Hold or sell |
| sold | Sold today | Rest; buying is forbidden |
| resting | Own nothing and did not sell today | Rest or buy |

Before any day, `resting=0`; the other states are negative infinity because they are impossible. Don't invent an initial completed sale.

```text
resting ── buy ─────→ holding
holding ── sell ────→ sold
sold ──── cooldown → resting
resting ── stay ────→ resting
holding ── hold ────→ holding
```

Each arrow advances a day. You cannot traverse several in one day. The diagram shows allowed connections; the equations define the update.

## Every transition reads yesterday, not today's new values

For price $p_t$ on day $t$, let $H,S,R$ be the best profit in the three states:

$$
\begin{aligned}
H_t&=\max(H_{t-1},R_{t-1}-p_t),\\
S_t&=H_{t-1}+p_t,\\
R_t&=\max(R_{t-1},S_{t-1}).
\end{aligned}
$$

Holding comes from continuing to hold or buying after a day when buying was permitted. It cannot come from yesterday's `sold` state. Resting follows either another rest or yesterday's sale and the required cooldown.

Why are these states sufficient? Histories at the same day and state allow the same continuations, so a lower-profit history cannot beat a higher-profit one from there. If the number of transactions is limited, the remaining count must also be part of that argument and state.

## Trace an example before memorizing variables

Prices: `[1,2,3,0,2]`.

| Time | holding | sold | resting |
| --- | --- | --- | --- |
| Before starting | −∞ | −∞ | 0 |
| Day 1, price 1 | −1 | −∞ | 0 |
| Day 2, price 2 | −1 | 1 | 0 |
| Day 3, price 3 | −1 | 2 | 1 |
| Day 4, price 0 | 1 | −1 | 2 |
| Day 5, price 2 | 1 | 3 | 2 |

The answer is `max(sold,resting)=3`. One path buys at 1, sells at 2, rests at 3, buys at 0, and sells at 2. The positive holding state on day 4 is net profit after purchasing a current position, not a final result with no position.

```python
def cooldown_profit(prices):
    holding = float("-inf")
    sold = float("-inf")
    resting = 0
    for price in prices:
        holding, sold, resting = (
            max(holding, resting - price),
            holding + price,
            max(resting, sold),
        )
    return max(resting, sold)
```

Python evaluates the right-hand side before assigning the targets, preserving yesterday's values. Updating variables one line at a time can accidentally read today's state. Inputs are finite nonnegative prices; an empty list returns 0. Time is $O(n)$ and auxiliary space $O(1)$. The result is profit, not a recovered action sequence.

## Changed constraints may require changed states

| New condition | What changes | What not to assume |
| --- | --- | --- |
| No cooldown | Merge non-holding states; allow buying the day after a sale | Three states aren't always necessary |
| Transaction fee | Charge once when buying or selling | Don't charge both |
| At most two transactions | Add remaining or completed transaction count | Equal profit doesn't mean equal future options |
| Multiple cooldown days | Track waiting time or appropriate older states | Not just a larger constant in this template |
| Return the action sequence | Store predecessor states | Three rolling values cannot recover full history |

This is DP on a known deterministic sequence, not reinforcement learning. Prices are inputs and transitions are given; no policy is learned. Bellman-style reasoning alone doesn't make an exercise RL.

## Test examples that expose illegal transitions

For `[1,2,3,0,2]` the answer is 4 **without** cooldown and 3 with it. This detects implementations that silently drop the rule. Also test decreasing prices, a single day, and repeated prices.

Code: [deeper_patterns.py](../code/deeper_patterns.py). Tests enumerate all legal buy, sell, and wait actions for short sequences rather than copying the same recurrence as an “independent” oracle. The public [Stock with Cooldown](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/) exercise is useful for learning how to add state that changes future choices.
