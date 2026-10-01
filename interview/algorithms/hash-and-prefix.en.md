# Hash maps and prefix sums: what should you remember?

[中文](hash-and-prefix.md) · **English** · [Pattern map](../leetcode.en.md)

> Reading time: ~7 min · Last reviewed: 2026-09

If every new value makes you scan all earlier ones, ask: **can I store the information I'll need to query later?** The important part isn't memorizing `dict`; it's choosing keys and values.

## 1. Decide what to store

| Future question | Key | Value / container |
| --- | --- | --- |
| Have I seen this? | Value | A set is enough |
| Where was it last seen? | Value / character | Index |
| How many occurrences? | Value / character | Count |
| Which words belong together? | Canonical signature | List of words |
| How many earlier prefixes match this one? | Prefix sum | Count |

Python keys must be hashable. Integers, strings, and tuples of hashable values work; lists don't. For anagrams, a sorted string is a possible signature. Sorting a word of length m costs O(m log m): include key-construction time. [Hash-table background](https://algs4.cs.princeton.edu/34hash/)

## 2. Two Sum: query before inserting

For `[4, 1, 7]` with target 8, reaching 7 means looking for 1. The map says 1 appeared at index 1; return `[1, 2]`.

```python
def two_sum(values, target):
    seen = {}
    for index, value in enumerate(values):
        complement = target - value
        if complement in seen:
            return [seen[complement], index]
        seen[value] = index
    return None
```

**Invariant:** the map contains only earlier indices at lookup time. `[3]` cannot reuse itself for target 6, while `[3, 3]` can form a pair. Average O(n) time, O(n) space. It avoids sorting and preserves original indices, at the cost of a map.

## 3. Common dict operations

```python
counts = {"pear": 2}
exists = "pear" in counts
missing_count = counts.get("apple", 0)
counts["pear"] = counts.get("pear", 0) + 1
removed = counts.pop("apple", 0)
pairs = list(counts.items())
```

`exists == True`, `missing_count == 0`, `removed == 0`, and `pairs == [('pear', 3)]`. `get` doesn't insert a missing key; indexing a `defaultdict` can create its default. `setdefault(key, [])` inserts only when absent. `keys()`, `values()`, and `items()` return views, not independent snapshots. [Python mapping API](https://docs.python.org/3/library/stdtypes.html#mapping-types-dict)

Don't test existence with `if seen.get(value)`: a stored index of 0 is falsey. Use `value in seen`. Don't change a dictionary's size while iterating over it.

## 4. Prefix sums: subtract boundaries instead of summing intervals

Define prefix[0] = 0 and prefix[right] as the sum of the first right elements. The half-open interval `[left, right)` sums to `prefix[right] - prefix[left]`.

To count subarrays summing to target, query how often `prefix - target` occurred earlier.

```python
def count_subarrays(values, target):
    frequencies = {0: 1}
    prefix = 0
    total = 0
    for value in values:
        prefix += value
        total += frequencies.get(prefix - target, 0)
        frequencies[prefix] = frequencies.get(prefix, 0) + 1
    return total
```

| Read `[1, -1, 1]`, target = 1 | Current prefix | Earlier prefix needed | New answers |
| --- | --- | --- | --- |
| 1 | 1 | 0 | 1 |
| −1 | 0 | −1 | 0 |
| 1 | 1 | 0 | 2 |

Total: 3, the two individual 1s and the entire array. Store **counts**, not just an index or a set. `{0: 1}` represents the empty prefix before the input. Query before inserting to avoid counting an empty subarray when target is zero.

Average O(n) time, O(n) space. It handles negative values without relying on sums growing with window size. Prefix sums suit static range queries; updates to the original array invalidate later prefixes.

## 5. Practice and transfer

- [Two Sum](https://leetcode.com/problems/two-sum/): value → index; explain update order.
- [Group Anagrams](https://leetcode.com/problems/group-anagrams/): design a shared key and count its construction cost.
- [Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/): prefix → count; include signed values.

<details class="interview" markdown="1">
<summary>What breaks if frequencies becomes a set?</summary>

Equal prefix sums at different positions represent different starts. A set loses multiplicity. `[0, 0]` with target 0 has 3 subarrays, not 2.

</details>

<details class="interview" markdown="1">
<summary>If Two Sum's input is sorted and we only need values, do we still need a map?</summary>

No. Two pointers take O(n) time and O(1) auxiliary space. Sorting first would cost O(n log n), and original indices are lost unless retained separately.

</details>
