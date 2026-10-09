# Primality and sieves: one number or a whole range?

[中文](primes-and-sieves.md) · **English**

> Reading time: about 10 minutes · Prerequisites: divisibility, loops, arrays · Reviewed: 2026-10-09

Testing whether 97 is prime and listing every prime below a million call for different starting points. For one number, look for a divisor. For a range, eliminate many composites at once.

This is an optional algorithm-review note, not a prerequisite for ML. Read sections 1–3 for boundaries and correctness, then continue for complexity and repeated queries. All code uses the Python standard library.

## 1. Why is the square root enough? {#trial-division}

A prime is an integer greater than 1 whose only positive divisors are 1 and itself. Zero, one, and negative integers are not prime. Two is the only even prime.

If $n$ is composite, write $n=ab$ with $1<a\le b$. Then $a^2\le ab=n$, so $a\le\sqrt n$. **If no nontrivial factor appears up to the square root, a larger factor cannot exist on its own: it would need a smaller partner.**

This is not specific to even numbers. Testing 91 with 2, 3, 5, and 7 reaches $91=7\times13$; testing through 90 is unnecessary. For 49, the divisor 7 itself must be included. Stopping strictly before the square root is wrong.

```python
def is_prime(number):
    if type(number) is not int:
        raise TypeError("number must be an integer")
    if number < 2:
        return False
    if number == 2:
        return True
    if number % 2 == 0:
        return False
    for divisor in range(3, isqrt(number) + 1, 2):
        if number % divisor == 0:
            return False
    return True
```

Import `isqrt` with `from math import isqrt`. [`math.isqrt`](https://docs.python.org/3.12/library/math.html#math.isqrt) returns the exact integer floor of the square root without floating-point approximation. Python's `range` excludes its right endpoint, hence `+ 1`. After eliminating even numbers, trying only odd divisors is safe. Some candidates, such as 9 and 15, are composite themselves; checking them adds work but does not break correctness.

The function accepts Python `int` strictly: it does not treat `True` as 1 or silently convert `7.0`. Worst-case cost is $O(\sqrt n)$ divisions and a constant number of extra integers. For now, treat integer arithmetic as constant cost; section 5 revisits large-integer bit complexity.

## 2. For a range, share the work {#sieve}

Calling `is_prime` separately for every integer from 2 through 30 repeatedly discovers multiples of 2. The Sieve of Eratosthenes reverses the process: keep flags and use each prime to cross out its later multiples.

| Current value | Action | Reason |
| --- | --- | --- |
| 2 | Cross out 4, 6, 8, …, 30 | All have factor 2 |
| 3 | Start at 9; cross out 9, 12, 15, …, 30 | 6 was handled by 2; crossing out 12 again is harmless |
| 4 | Skip | Already known composite |
| 5 | Start at 25; cross out 25 and 30 | Smaller multiples already have a smaller prime factor |
| Beyond $\sqrt{30}$ | Stop marking and read the flags | Every composite must have a prime factor at most its square root |

The remaining numbers are `2, 3, 5, 7, 11, 13, 17, 19, 23, 29`. Trace the table on paper and notice which numbers each step eliminates for the first time.

```python
def prime_flags(limit):
    if type(limit) is not int:
        raise TypeError("limit must be an integer")
    if limit < 0:
        raise ValueError("limit must be non-negative")
    flags = bytearray([1]) * (limit + 1)
    flags[0] = 0
    if limit >= 1:
        flags[1] = 0
    for prime in range(2, isqrt(limit) + 1):
        if flags[prime]:
            for multiple in range(prime * prime, limit + 1, prime):
                flags[multiple] = 0
    return flags
```

The contract covers **0 through `limit`, inclusive**. A flag of 1 means prime. A `bytearray` uses one byte per position, not a one-bit bitset, and avoids representing each flag as a separate Python integer object.

Explicit assignments make the marking process visible. For larger inputs, consider slices, odd-only storage, or bit arrays, while accounting for temporary allocations rather than judging only by code length. [Princeton's course implementation](https://introcs.cs.princeton.edu/java/14array/PrimeSieve.java.html) is another classic sieve reference. Our interface explicitly includes the upper endpoint; check a problem's contract before substituting it.

## 3. Why start at the square, and why are surviving numbers prime?

When processing prime $p$, the numbers $2p,3p,\ldots,(p-1)p$ all have a prime factor smaller than $p$ and have already been handled. Start at $p^2$. For 7, smaller primes already eliminate 14, 21, 28, 35, and 42; 49 is the first number that needs 7.

Use this invariant: **before processing $p$, every composite whose smallest prime factor is less than $p$ has been marked zero.**

1. If `flags[p]` is still 1, $p$ cannot be composite: a smaller prime factor would already have eliminated it.
2. Processing $p$ marks composites whose smallest prime factor is $p$. None is smaller than $p^2$, so starting there misses none of them.
3. Once $p$ passes $\sqrt N$, every composite at most $N$ has been reached through its smallest prime factor. Actual primes were never marked because every marked number has two factors greater than 1.

Two common errors follow directly: starting at `p` deletes the prime itself; excluding the square-root boundary can leave perfect squares such as 49 incorrectly flagged.

## 4. Complexity is not just counting nested loops

Separate trial-division tests have the worst-case upper bound:

$$
\sum_{n=2}^{N}O(\sqrt n)=O(N^{3/2}).
$$

The sieve does not scan the entire array for every outer value. Only primes $p$ start an inner loop, with roughly $N/p$ writes each. Total marking is bounded by:

$$
\sum_{\substack{p\le\sqrt N\\p\text{ is prime}}}\frac{N}{p}
=N\sum_{p\le\sqrt N}\frac1p.
$$

The number-theoretic result that the sum of prime reciprocals grows as $\log\log x+O(1)$ gives $O(N\log\log N)$ time. Initialization separately costs $O(N)$. **The two loops alone do not establish that bound.** Without using the prime-reciprocal result, bounding by the harmonic sum over all integers gives the valid but looser $O(N\log N)$. This note does not prove the number-theoretic theorem; [CMU’s algorithm analysis](https://www.cs.cmu.edu/~scandal/cacm/node8.html) also states the sieve bound.

| Need | Starting point | Time and space |
| --- | --- | --- |
| Test one moderately sized integer | Trial division | Worst-case $O(\sqrt n)$ divisions; a constant number of extra integers |
| List all primes through $N$ | Sieve | $O(N\log\log N)$ time and $O(N)$ bytes of flags |
| Repeated queries within the same bound | Sieve once, then look up | Same preprocessing; each later lookup is $O(1)$ |
| Query a narrow interval far from zero | Consider a segmented sieve | Avoid storing flags through the full upper endpoint, but still need base primes |

Returning a list of primes adds output storage. If only a count is needed, use `sum(flags)`. If more membership queries follow, keeping the flags avoids rebuilding the same information.

## 5. Boundaries, segments, and large integers

Check small inputs first: `prime_flags(0)` gives a byte array containing `[0]`, `prime_flags(1)` gives `[0, 0]`, and `prime_flags(2)` first retains 2. To count primes **strictly below** $n$, return 0 for $n\le2$; otherwise count `prime_flags(n - 1)`, not `prime_flags(n)`.

A segmented sieve is useful for a closed interval $[L,R]$. First find base primes through $\sqrt R$. For each prime $p$, start crossing out at:

$$
\max\left(p^2,\left\lceil\frac{L}{p}\right\rceil p\right).
$$

For $0\le L\le R$, integer code is `max(prime * prime, ((left + prime - 1) // prime) * prime)`. In `[20, 30]`, prime 3 starts at 21 and prime 5 starts at 25. The $p^2$ condition also prevents deleting the prime itself when it lies inside the interval. Explicitly clear 0 and 1 if present. Keeping a base sieve and segment flags uses $O(\sqrt R+R-L+1)$ space, not just the interval width.

Python integers do not overflow like fixed-width integers, but their arithmetic is not free. For a $B$-bit input, trial division on the order of $\sqrt n$ has an exponentially growing worst-case count in $B$, and each large-integer operation costs something too. Do not use this code for cryptographic-size primality testing. Choose scale-appropriate algorithms and verified implementations. This note does not implement Miller–Rabin or present a probabilistic test as an unconditional proof.

## 6. One extension: factorization

Primality testing stops after finding a divisor. Factorization keeps dividing and shrinks the remaining value. For 84:

```text
84 → divide by 2 → 42 → divide by 2 again → 21
21 → divide by 3 → 7
Append the remaining 7: [2, 2, 3, 7]
```

Why can the remaining 7 be appended directly? Once the trial divisor's square exceeds the remainder, a remainder greater than 1 cannot be composite: it would need a smaller factor that should already have been processed. Compare against the **shrinking remainder**, not the original 84.

[number_theory.py](../code/number_theory.py) includes `is_prime`, `prime_flags`, and `prime_factors`. Factoring 1 returns an empty list; zero and negative inputs raise errors because signed factorization is outside this interface. Worst-case cost remains $O(\sqrt n)$ trial divisions, plus output storage. Run from the repository root:

```bash
python3 interview/code/number_theory.py
```

```text
Primes through 30: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
49 is prime: False
Factors of 84: [2, 2, 3, 7]
```

For review, explain why both algorithms involve a square root: trial division uses factor pairs, while the sieve uses that fact to establish which composites must already have been removed. Test 1, 2, and 49, then trace the sieve through 30. For a public exercise, try [Count Primes](https://leetcode.com/problems/count-primes/) and remember that it excludes the upper endpoint.

Return to the [algorithm overview](../leetcode.en.md). For preprocessing versus per-query cost and output space, revisit [complexity and Python tools](complexity-and-tools.en.md).
