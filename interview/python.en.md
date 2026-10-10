# Interviews: the Python you actually use in coding problems

[中文](python.md) · **English**

> Reading time: ~10 min · Last reviewed: 2026-10

> **Read this first**: this covers public language knowledge and the idioms I use myself when practising problems. It contains no interview questions from any company.

A few Python tools save a lot of typing: sort keys, comprehensions, unpacking, and standard-library containers. Start with the syntax you need, then use the questions at the end to check the details.

Each code block runs on its own. If the syntax is familiar but changing a copy still changes your original data, continue with [names, copies, and function calls](python-objects.en.md). That page follows the objects behind the names.

## lambda: a small function on one line

`lambda args: expression` is a function without a name. It holds a single expression, and its value is returned automatically:

```python
double = lambda number: number * 2
assert double(3) == 6
```

On its own that is not very useful. The point is to pass it to another function, most often as a sort `key`. `sorted` first computes `key(element)` for every element, then orders by those values:

```python
pairs = [("amy", 90), ("bob", 85), ("cat", 90)]
by_score = sorted(pairs, key=lambda pair: pair[1])
by_score_then_name = sorted(pairs, key=lambda pair: (-pair[1], pair[0]))
best = max(pairs, key=lambda pair: pair[1])
assert by_score == [("bob", 85), ("amy", 90), ("cat", 90)]
assert by_score_then_name == [("amy", 90), ("cat", 90), ("bob", 85)]
assert best == ("amy", 90)
```

For multiple criteria, return a tuple: compare the first item, then the next on a tie. Here, `(-score, name)` gives descending scores and ascending names. `reverse=True` reverses the ordering of the whole key, not just one field.

For a descending string field with an ascending tie-breaker, sort the secondary field first. Then use a stable sort on the primary field. Here that means descending group names, with ascending numbers inside each group:

```python
records = [("beta", 3), ("alpha", 2), ("beta", 1), ("alpha", 1)]
by_number = sorted(records, key=lambda record: record[1])
result = sorted(by_number, key=lambda record: record[0], reverse=True)
assert result == [("beta", 1), ("beta", 3), ("alpha", 1), ("alpha", 2)]
```

The second sort compares group names only, preserving the numeric order within each group. [Sorting guide](https://docs.python.org/3.14/howto/sorting.html#sort-stability-and-complex-sorts)

Try different keys below and watch the result change. Look at the people with equal scores: Python's sort is **stable**, so elements with equal keys keep their original relative order.

<!-- widget:tx-py-sort -->

`sorted(x)` returns a new list and leaves the original alone; `x.sort()` sorts in place and returns `None`. So `x = x.sort()` turns x into None, a very common bug.

One more trap: a lambda created in a loop remembers the variable, not its value at the time. See question 2 at the end.

## Comprehensions: build a list, dict or set in one line

```python
numbers = [3, 2, 3]
squares = [number * number for number in numbers]
evens = [number for number in numbers if number % 2 == 0]
positions = {value: index for index, value in enumerate(numbers)}
seen = {character for character in "banana"}
grid = [[0] * 3 for row in range(2)]
total = sum(number * number for number in numbers)
assert squares == [9, 4, 9] and evens == [2]
assert positions == {3: 2, 2: 1} and seen == {"b", "a", "n"}
grid[0][0] = 1
assert grid[1] == [0, 0, 0] and total == 22
```

Repeated values overwrite earlier entries in `positions`, so 3 maps to its last index. Each row in `grid` is a new list; changing one row does not change the other.

The generator expression passed to `sum` calculates one square at a time without building a list of squares. The input `numbers` is already in memory, though; using a generator does not make it disappear.

## Unpacking, enumerate, zip and slicing

```python
left, right = 2, 5
left, right = right, left
numbers = [10, 20, 30]
first, *rest = numbers
indexed = list(enumerate(numbers))
paired = list(zip(["amy", "bob"], [90]))
assert (left, right) == (5, 2)
assert first == 10 and rest == [20, 30]
assert indexed == [(0, 10), (1, 20), (2, 30)]
assert paired == [("amy", 90)]
assert numbers[1:3] == [20, 30] and numbers[-1] == 30
assert "abc"[::-1] == "cba"
```

`zip` stops at the shortest input by default. If unequal lengths indicate bad data, Python 3.10+ offers `zip(..., strict=True)`, which raises when iteration reaches the mismatch. An ordinary list slice is a shallow copy and excludes the right endpoint. An empty list cannot supply `[-1]` or the required `first` item. [zip documentation](https://docs.python.org/3.14/library/functions.html#zip)

## Ready-made data structures in the standard library

Many problem types have a tool waiting for them. Knowing this table saves a lot of code:

| You need | Use | Notes |
| --- | --- | --- |
| Counting | `collections.Counter` | `Counter(s).most_common(k)` gives the k most frequent |
| Grouping, building a graph | `collections.defaultdict(list)` | `groups[key]` creates a list for a missing key; `get` does not |
| A queue, BFS | `collections.deque` | `append` and `popleft` are O(1); a list's `pop(0)` is O(n) |
| Min-heap, top K | `heapq` | min-heap by default; negation works on older versions, and Python 3.14+ also has `_max` APIs |
| A position in a sorted array | `bisect` | `bisect_left` finds the first index with value ≥ x, in O(log n) |
| Memoised recursion | `functools.cache` | arguments must be hashable; the cache is unbounded, so consider memory and external state |
| Permutations and combinations | `itertools` | `permutations`, `combinations`, `product`, `accumulate` |
| Infinity | `math.inf` or `float("inf")` | a starting value when looking for a minimum |

```python
import heapq

heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
smallest = heapq.heappop(heap)
assert smallest == 1 and heap == [3]
numbers = [2, 9, 4]
negative_heap = [-number for number in numbers]
heapq.heapify(negative_heap)
assert -heapq.heappop(negative_heap) == 9
assert heapq.nlargest(2, numbers) == [9, 4]
```

For example, `groups.get("missing")` returns `None` without inserting a key. Caching keeps references to arguments and return values; a cached function that depends on changing external data may return stale results. [defaultdict](https://docs.python.org/3.14/library/collections.html#collections.defaultdict) · [cache](https://docs.python.org/3.14/library/functools.html#functools.cache)

## Language details interviewers like to probe

- **Do not build a 2D array as `[[0] * n] * m`.** The outer `* m` copies a reference to the same inner list, so changing one cell changes every row. Write `[[0] * n for _ in range(m)]`.
- **Do not use a mutable default argument.** In `def f(x, acc=[])` the `[]` is created once, when the function is defined, and every later call shares it. Write `acc=None` and check `if acc is None: acc = []` inside.
- **`is` and `==` differ.** `==` compares values; `is` asks whether two names refer to the same object. Test for None with `is None`.
- **Shallow versus deep copy.** Ordinary list slicing and `copy()` copy the outer layer. Whether nested objects need copying depends on what you will mutate; nesting alone does not require `deepcopy`. See [why a copy can affect the original](python-objects.en.md#shallow-copy).
- **Integer division and modulo with negatives.** Python's `//` rounds down: `-7 // 2` is `-4` and `-7 % 2` is `1`. C++ and Java round towards zero, so `-7 / 2` is `-3`, which trips people up when switching languages.
- **Strings are immutable.** Repeated concatenation may repeatedly copy a growing prefix; interpreter optimizations are not a portable linear-time guarantee. For a batch of fragments, use `"".join(parts)`. [String operations](https://docs.python.org/3.14/library/stdtypes.html#common-sequence-operations)
- **Recursion depth is limited.** A common default is about 1000 frames; check `sys.getrecursionlimit()`. A long DFS chain may raise `RecursionError`. Prefer an explicit stack over blindly raising the limit, which is not a safety guarantee. See the [DFS / BFS walkthrough](algorithms/traversal.en.md).
- **`*args` and `**kwargs`.** The first collects extra positional arguments into a tuple, the second collects extra keyword arguments into a dict.
- **Generators.** A function that uses `yield` becomes a generator: each time a value is requested it computes one and pauses, then resumes where it stopped. Good for very large data or sequences with no end.

For choosing and reviewing problems, see [LeetCode practice](leetcode.en.md).

## Common interview questions

<details class="interview" markdown="1">
<summary>grid = [[0] * 3] * 3. After changing grid[0][0], why did all three rows change?</summary>

`* 3` copies references: the three rows are one and the same list. Change it through any of the three names and you see the same object. The right way is `[[0] * 3 for _ in range(3)]`, which builds a new row on every iteration.

</details>

<details class="interview" markdown="1">
<summary>fs = [lambda: i for i in range(3)]. Why does fs[0]() return 2?</summary>

The lambda remembers the variable `i`, not its value, and looks the value up when it is called; by then the loop has finished and `i` is 2, so all three return 2. This is late binding. To keep the value from that moment, fix it with a default argument: `lambda i=i: i`.

</details>

<details class="interview" markdown="1">
<summary>What is the difference between sorted and list.sort? Is Python's sort stable?</summary>

`sorted` returns a new list from an iterable; `list.sort` sorts in place and returns `None`. Both guarantee stability. CPython's sort exploits existing ordered runs: with constant-cost keys and comparisons, worst-case time is O(n log n), while ordered input can be faster. Account separately for expensive keys or long-string comparisons.

</details>

<details class="interview" markdown="1">
<summary>How do you use heapq as a max-heap? How do you solve top K?</summary>

`heapq` uses a min-heap by default. Negating numeric values on push and pop implements a max-heap on older versions; Python 3.14 also exposes `_max` APIs. For the k largest values (1≤k≤n), maintain at most k elements and replace the minimum when a larger value arrives. Scanning costs O(n log(k+1)); sorted output adds O(k log k). See [the Top-K cutoff](algorithms/stack-heap-links.en.md) and [official API](https://docs.python.org/3/library/heapq.html).

</details>

<details class="interview" markdown="1">
<summary>What is -7 // 2? And -7 % 2?</summary>

`-7 // 2 = -4` and `-7 % 2 = 1`. Python's integer division rounds down and guarantees `a == (a // b) * b + a % b`. C++ and Java round towards zero: `-7 / 2` is `-3` and the remainder is `-1`.

</details>

<details class="interview" markdown="1">
<summary>What is the difference between a list comprehension and a generator expression?</summary>

A list comprehension `[...]` stores all results. A generator expression `(...)` computes on demand without building that result list. It still retains execution state and may keep its input alive; once exhausted, it does not restart. A stored list may be preferable when results need several passes. See [short-circuiting and iterators](python-objects.en.md#iteration-and-truth).

</details>

<details class="interview" markdown="1">
<summary>Why is a default empty list not recreated for each call?</summary>

A default value is evaluated once, when the function is defined. The list in `def f(acc=[])` is shared by every later call, so whatever one call adds is still there for the next. Use `acc=None` and `if acc is None: acc = []` in the body.

</details>
