# String matching: reuse what a failed comparison already told you

[中文](string-matching.md) · **English** · [Pattern map](../leetcode.en.md)

> Last reviewed: 2026-10-09 · Prerequisite: [Pointers and windows](pointers-and-search.en.md)

Find `abab` in `ababab`. The starting positions are 0 and 2. Jumping four characters after the first match misses the overlapping second one.

The useful question is: **can the characters we already checked help the next attempt?** In everyday code, `find` or `in` is usually the sensible starting point. KMP teaches how to reuse matching information; it isn't a reason to implement every string operation yourself.

## What brute force repeats

For text length $n$ and pattern length $m$, trying each start costs $O(nm)$ in the worst case. Searching for `aaaab` in `aaaa…a` repeatedly verifies the same run of `a` characters before failing.

KMP never rewinds the text pointer. It keeps the length of the pattern prefix matched so far. On a mismatch, it shortens that prefix and retries **the same current text character**.

## What the prefix table stores

`prefix[index]` is the length of the longest equal proper prefix and suffix of `pattern[:index+1]`. “Proper” excludes the entire string; the two pieces may overlap.

| Pattern prefix | Longest matching ends | Length |
| --- | --- | --- |
| a | empty | 0 |
| ab | empty | 0 |
| aba | a | 1 |
| abab | ab | 2 |
| ababa | aba | 3 |
| ababac | empty | 0 |

For the last `c`, the previous matched length is 3. Pattern index 3 expects `b`, so fall back to length 1, which also expects `b`. Fall back to 0, expecting `a`. None matches; only then is the result 0. A mismatch does not mean “always restart from zero.”

```python
def prefix_lengths(pattern):
    prefix = [0] * len(pattern)
    matched = 0
    for index in range(1, len(pattern)):
        while matched and pattern[index] != pattern[matched]:
            matched = prefix[matched - 1]
        if pattern[index] == pattern[matched]:
            matched += 1
        prefix[index] = matched
    return prefix
```

Any shorter prefix that could still work must also be a suffix of the part just matched. Otherwise it contradicts characters already read. The prefix table gives the next possible length on that chain.

## Carry matching information into the next search step

```python
def kmp_positions(text, pattern):
    if not pattern:
        return list(range(len(text) + 1))
    prefix = prefix_lengths(pattern)
    matched = 0
    positions = []
    for index, character in enumerate(text):
        while matched and character != pattern[matched]:
            matched = prefix[matched - 1]
        if character == pattern[matched]:
            matched += 1
        if matched == len(pattern):
            positions.append(index - len(pattern) + 1)
            matched = prefix[matched - 1]
    return positions
```

The input is a Python string. Results include overlapping matches. Here an empty pattern matches every boundary `0…len(text)`; confirm the required convention for a particular problem.

Trace `ababab` / `abab`:

| Text position | What happens | Retained matched |
| --- | --- | --- |
| 0–2 | Match a, b, a | 1, 2, 3 |
| 3 | Full match; record start 0 | Fall back to prefix[3] = 2 |
| 4 | The retained ab is followed by a | 3 |
| 5 | Full match again; record start 2 | 2 |

Not resetting to zero preserves overlaps. `matched` is a **length**, and therefore the next pattern index to compare, not the index of the last matched character.

## Why a while loop inside a for loop is still linear

Reading a new character increases `matched` by at most 1. Every fallback strictly decreases it. Across the scan, decreases cannot outgrow the increases that made them possible. Apply that accounting separately to preprocessing and searching: $O(n+m)$ time, $O(m)$ working space, plus $O(z)$ for $z$ reported matches.

This is amortized analysis, not a claim that every character is compared only once. If only the first match matters, return immediately instead of collecting results.

## Where the template stops applying

- This finds a contiguous substring, not a subsequence that may skip characters. See [sequence DP](sequence-dp.en.md).
- Define case handling, Unicode normalization, and what counts as a character. Python indices are not displayed-glyph indices; normalization can also change the mapping to original offsets.
- Ordinary rolling hashes need collision handling. KMP uses exact equality. Searching many patterns at once calls for a different data-structure comparison.
- Try `aaaa` / `aa` → `[0,1,2]`, then empty strings, a longer pattern, and no match.

Runnable code: [deeper_patterns.py](../code/deeper_patterns.py). Tests enumerate small strings and compare with direct slicing. Further reading: [Princeton substring search](https://algs4.cs.princeton.edu/53substring/). Its DFA presentation and this prefix-table presentation express the same reuse after a mismatch.
