# Sequence DP: say exactly what each state summarizes

[中文](sequence-dp.md) · **English** · [Pattern map](../leetcode.en.md)

> Last reviewed: 2026-10-09 · Prerequisite: [Backtracking and DP](backtracking-and-dp.en.md)

`AB` appears in order in both `CAB` and `ACB`, with characters skipped. This is a subsequence, not a contiguous substring. Their longest common subsequence (LCS) has length 2. `CB` is another answer, so returning a length is not the same as returning a unique sequence.

Sequence states often differ by just a few words: **ending at this position**, or **using only this prefix**? Write that sentence before the recurrence.

## LCS: how much can two prefixes match?

$L(i,j)$ is the LCS length of the first $i$ characters of one string and the first $j$ of the other. Any empty-prefix boundary is 0.

- Equal final characters: append that character to the LCS of the shorter prefixes, giving $L(i-1,j-1)+1$.
- Different final characters: a common subsequence cannot use both as its final pair. Skip at least one and take $\max(L(i-1,j),L(i,j-1))$.

Why is the diagonal enough when the final characters match? If an optimal matching uses one of those final characters but not the pair, move its last matching pair to both final positions without disrupting earlier order. If it uses neither, a pair can be appended. Thus an optimum using the equal final pair always exists.

| First-string prefix \ Second-string prefix | Empty | A | AC | ACB |
| --- | --- | --- | --- | --- |
| Empty | 0 | 0 | 0 | 0 |
| C | 0 | 0 | 1 | 1 |
| CA | 0 | 1 | 1 | 1 |
| CAB | 0 | 1 | 1 | 2 |

The bottom-right answer is 2. Cells containing 1 need not describe the same common subsequence; each retains only its own best length.

## Two rows are enough for the length

```python
def lcs_length(first, second):
    previous = [0] * (len(second) + 1)
    for first_char in first:
        current = [0]
        for column, second_char in enumerate(second, start=1):
            if first_char == second_char:
                current.append(previous[column - 1] + 1)
            else:
                current.append(max(previous[column], current[-1]))
        previous = current
    return previous[-1]
```

`previous[column]` is above, `current[-1]` is left, and `previous[column-1]` is diagonal. Two rows make those dependencies explicit; a one-row optimization also needs to preserve the old diagonal.

For lengths $n,m$, time is $O(nm)$ and auxiliary space $O(m)$. Pass the shorter string second to reduce space. To reconstruct one LCS, retain the whole table and backtrack from the bottom-right along valid transitions. Ties need a rule, not a claim that the answer is unique.

## LIS: ending here is not the same as best so far

A direct recurrence for the longest strictly increasing subsequence (LIS) is:

$$
D(i)=1+\max\bigl(\{D(j):j<i,\ a_j<a_i\}\cup\{0\}\bigr).
$$

$D(i)$ must end at $a_i$. That tells us whether another value can extend it. Only the final answer takes the maximum over all $D(i)$. Two loops cost $O(n^2)$ time and $O(n)$ space; empty input returns 0.

For `[3,5,6,2,4]`, `D=[1,2,3,1,2]` and the answer is 3. Remembering only “the earlier best length was 3,” without its endpoint 6, could incorrectly allow 4 to extend it.

## Keep minimal tails, but don't mistake them for a solution

For each length, keep the smallest possible final value. Of two increasing subsequences of equal length, the one ending lower is at least as easy to extend. `tails[length-1]` stores that minimum endpoint.

| New value | tails |
| --- | --- |
| 3 | [3] |
| 5 | [3,5] |
| 6 | [3,5,6] |
| 2 | [2,5,6] |
| 4 | [2,4,6] |

The final `[2,4,6]` is **not** a subsequence of the input: 6 occurs before 2 and 4. It summarizes three different lengths. Its length, 3, is the answer.

```python
def lis_length(values):
    tails = []
    for value in values:
        position = bisect_left(tails, value)
        if position == len(tails):
            tails.append(value)
        else:
            tails[position] = value
    return len(tails)
```

Import `bisect_left` from `bisect`. It finds the first value not smaller than the new one, so equal values do not extend the sequence. For a nondecreasing subsequence, use `bisect_right` and change the stated ordering requirement. These are not interchangeable choices. [Python bisect documentation](https://docs.python.org/3/library/bisect.html)

One binary search per element gives $O(n\log n)$ time and $O(n)$ space. Replacing a tail updates a summary; it does not reorder the input. Reconstructing a real sequence also requires original indices and predecessor links.

## Similar problems need different boundaries

| Problem | Defining state condition | What cannot be reused unchanged |
| --- | --- | --- |
| LCS | Two prefixes; skipping allowed | On mismatch, take the larger upper / left state |
| Longest common substring | Ends at these two positions; contiguous | On mismatch, reset this cell to zero |
| LIS | Original order; strictly increasing | Equality cannot extend a sequence |
| Edit distance | Fewest operations between prefixes | Empty-prefix boundaries equal the other prefix's length, not zero |

For `[2,2,2]`, strict LIS is 1; the nondecreasing version is 3. Also test empty and reverse-ordered strings. Code: [deeper_patterns.py](../code/deeper_patterns.py), checked against subsequence enumeration on small inputs. For more derivation practice, see [MIT dynamic programming notes](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/resources/lecture-notes/).
