# Dynamic programming and optimal stopping: continue or accept?

[中文](dynamic-programming.md) · **English** · [Review map](../README.en.md)

> Reading time: ~7 min · Prerequisites: conditional expectation, states, recurrences · Last reviewed: 2026-10

Waiting problems specify the rule. Decision problems ask you to choose it. Separate immediate reward from the value of future choices.

## 1 · Bellman's equation is a decomposition

For finite horizon, state s, action a, immediate reward r and next state S′, maximizing expected total reward gives:

$$
V_t(s)=\max_{a\in A(s)}
E[r_t(s,a,S')+V_{t+1}(S')\mid s,a].
$$

Specify terminal V_T and work backward. The state must determine future transitions, feasible actions, and rewards. If history matters, retain the relevant history rather than assuming it away.

## 2 · At most two die rolls: when should you stop?

Roll a fair six-sided die. Accept the result or discard it and roll again; the last roll must be accepted. Before the last roll, expected value is 3.5.

After seeing x on the first roll, compare x with 3.5. Accept 4,5,6 and reroll 1,2,3:

$$
V_2=\frac{3\cdot3.5+4+5+6}{6}=\frac{17}{4}.
$$

The threshold comes from remaining option value, not a vague desire to try again.

<details markdown="1">
<summary>What if three rolls are allowed, or rerolling costs money?</summary>

With three rolls, continuation is worth V₂=4.25, so initially accept only 5 or 6. Then $V_3=(4\cdot4.25+5+6)/6=14/3$.

A cost c for each reroll subtracts c from that continuation branch; all subsequent decisions use the same cost-aware recurrence. Costs change the policy, not merely final accounting.

</details>

## 3 · What optimal substructure means

Given a state, if a strategy's continuation is not optimal for that state, replace it with a better continuation. This improves overall expected reward, supporting backward induction.

The argument depends on the objective and state. Maximizing a median, imposing pathwise risk constraints, or retaining history-dependent restrictions may require more than an immediate reward plus expected future value.

## 4 · Relative ranks: a sequential-selection model

Consider n candidates in uniformly random order, relative ranks only, no recall, and success defined as selecting the overall best. Skip r≥1 candidates, then take the first new record. A fallback if no new record appears does not affect the probability of selecting the best.

The best appears at j>r with probability 1/n. To reach it, the best of the first j−1 must lie among the first r:

$$
P_r=\frac rn\sum_{j=r+1}^n\frac1{j-1}.
$$

For large n and r/n≈x this becomes $-x\log x$, maximized near x=1/e. This explains the 37% approximation within the threshold family. Proving global optimality requires additional stopping analysis, not just this derivative.

## 5 · Do not turn a toy model into a life rule

The approximation relies on fixed n, random order, relative ranks, no recall, and a winner-takes-all objective. Hiring, relationships, and personal choices rarely meet all of these. Use the model to practice assumptions and recursion, not to manufacture precision for real decisions.

Continue: [Brownian motion and Itô](brownian-ito.en.md). For implementation basics, see [recursion and dynamic programming](../../interview/algorithms/backtracking-and-dp.en.md).
