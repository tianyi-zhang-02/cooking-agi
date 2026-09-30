# Interviews: the Python you actually use in coding problems

[中文](python.md) · **English**

> Reading time: ~8 min · Last reviewed: 2026-09

> **Read this first**: this covers public language knowledge and the idioms I use myself when practising problems. It contains no interview questions from any company.

Writing problems in Python saves time mostly through a handful of things: lambda with sorting, comprehensions, a few kinds of unpacking, and the ready-made data structures in the standard library. This page goes in order of how often they come up, and ends with the language details interviewers like to probe.

## lambda: a small function on one line

`lambda args: expression` is a function without a name. It holds a single expression, and its value is returned automatically:

```python
double = lambda x: x * 2      # same as: def double(x): return x * 2
```

On its own that is not very useful. The point is to pass it to another function, most often as a sort `key`. `sorted` first computes `key(element)` for every element, then orders by those values:

```python
pairs = [("amy", 90), ("bob", 85), ("cat", 90)]
sorted(pairs, key=lambda p: p[1])           # by score, ascending
sorted(pairs, key=lambda p: (-p[1], p[0]))  # score descending, ties by name
max(pairs, key=lambda p: p[1])              # the highest score
```

To sort by several criteria, have the key return a tuple: tuples compare their first elements, then the second on a tie, and so on. To make one criterion descending, negate it if it is a number; strings cannot be negated, so use `reverse=True` or sort twice.

Try different keys below and watch the result change. Look at the people with equal scores: Python's sort is **stable**, so elements with equal keys keep their original relative order.

<!-- widget:tx-py-sort -->

`sorted(x)` returns a new list and leaves the original alone; `x.sort()` sorts in place and returns `None`. So `x = x.sort()` turns x into None, a very common bug.

One more trap: a lambda created in a loop remembers the variable, not its value at the time. See question 2 at the end.

## Comprehensions: build a list, dict or set in one line

```python
squares = [x * x for x in nums]                 # list
evens   = [x for x in nums if x % 2 == 0]       # with a filter
index   = {v: i for i, v in enumerate(nums)}    # dict: value -> index
seen    = {c for c in s}                        # set
grid    = [[0] * n for _ in range(m)]           # an m-by-n 2D array
total   = sum(x * x for x in nums)              # generator expression, no list built
```

The last line uses parentheses. That is a generator expression: instead of putting every element in memory first, it produces them one at a time, which saves memory on large inputs.

## Unpacking, enumerate, zip and slicing

```python
a, b = b, a                     # swap without a temporary variable
first, *rest = nums             # first element, and a list of the rest
for i, x in enumerate(nums):    # index and value together
    ...
for x, y in zip(xs, ys):        # pair two lists by position; stops at the shorter
    ...
s[::-1]                         # reverse a string or list
nums[i:j]                       # half-open [i, j), returns a new copy
nums[-1]                        # the last element
```

## Ready-made data structures in the standard library

Many problem types have a tool waiting for them. Knowing this table saves a lot of code:

| You need | Use | Notes |
| --- | --- | --- |
| Counting | `collections.Counter` | `Counter(s).most_common(k)` gives the k most frequent |
| Grouping, building a graph | `collections.defaultdict(list)` | a missing key gets an empty list automatically |
| A queue, BFS | `collections.deque` | `append` and `popleft` are O(1); a list's `pop(0)` is O(n) |
| Min-heap, top K | `heapq` | min-heap only; for a max-heap, push negated numbers |
| A position in a sorted array | `bisect` | `bisect_left` finds the first index with value ≥ x, in O(log n) |
| Memoised recursion | `functools.cache` | put it on a recursive function; repeated calls return the cached result |
| Permutations and combinations | `itertools` | `permutations`, `combinations`, `product`, `accumulate` |
| Infinity | `math.inf` or `float("inf")` | a starting value when looking for a minimum |

```python
import heapq
heap = []
heapq.heappush(heap, 3)
smallest = heapq.heappop(heap)    # pops the smallest element
heapq.heappush(heap, -x)          # max-heap: store negatives, negate on the way out
heapq.nlargest(k, nums)           # the k largest, directly
```

## Language details interviewers like to probe

- **Do not build a 2D array as `[[0] * n] * m`.** The outer `* m` copies a reference to the same inner list, so changing one cell changes every row. Write `[[0] * n for _ in range(m)]`.
- **Do not use a mutable default argument.** In `def f(x, acc=[])` the `[]` is created once, when the function is defined, and every later call shares it. Write `acc=None` and check `if acc is None: acc = []` inside.
- **`is` and `==` differ.** `==` compares values; `is` asks whether two names refer to the same object. Test for None with `is None`.
- **Shallow versus deep copy.** `x[:]`, `list(x)` and `x.copy()` copy only the outer layer; if there are lists inside, use `copy.deepcopy`.
- **Integer division and modulo with negatives.** Python's `//` rounds down: `-7 // 2` is `-4` and `-7 % 2` is `1`. C++ and Java round towards zero, so `-7 / 2` is `-3`, which trips people up when switching languages.
- **Strings cannot be changed in place.** `s += c` in a loop builds a new string every time; collect the pieces in a list and finish with `"".join(parts)`.
- **Recursion depth is limited.** The default limit is about 1000 frames, so a DFS down a long chain raises `RecursionError`. Raise it with `sys.setrecursionlimit`, or rewrite the search iteratively with a stack.
- **`*args` and `**kwargs`.** The first collects extra positional arguments into a tuple, the second collects extra keyword arguments into a dict.
- **Generators.** A function that uses `yield` becomes a generator: each time a value is requested it computes one and pauses, then resumes where it stopped. Good for very large data or sequences with no end.

For how to practise problem types, see [how to grind LeetCode](leetcode.en.md).

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

`sorted` takes any iterable and returns a new list; `list.sort` works only on lists, sorts in place and returns `None`. Both are stable (Timsort), so elements with equal keys keep their relative order, and both run in O(n log n).

</details>

<details class="interview" markdown="1">
<summary>How do you use heapq as a max-heap? How do you solve top K?</summary>

`heapq` only provides a min-heap: push negated values and negate them again when you pop, and it behaves as a max-heap. For the k largest elements, keep a min-heap of size k and replace the top whenever a new element is larger; at the end the heap holds the k largest, in O(n log k). For small inputs just use `heapq.nlargest(k, nums)`.

</details>

<details class="interview" markdown="1">
<summary>What is -7 // 2? And -7 % 2?</summary>

`-7 // 2 = -4` and `-7 % 2 = 1`. Python's integer division rounds down and guarantees `a == (a // b) * b + a % b`. C++ and Java round towards zero: `-7 / 2` is `-3` and the remainder is `-1`.

</details>

<details class="interview" markdown="1">
<summary>What is the difference between a list comprehension and a generator expression?</summary>

A list comprehension `[...]` computes every element at once and keeps them in memory; a generator expression `(...)` computes each one only when it is needed, using almost no extra memory, but can be iterated only once. For one-pass uses such as a sum or a maximum, the generator is the lighter choice.

</details>

<details class="interview" markdown="1">
<summary>Why can a default argument not be an empty list?</summary>

A default value is evaluated once, when the function is defined. The list in `def f(acc=[])` is shared by every later call, so whatever one call adds is still there for the next. Use `acc=None` and `if acc is None: acc = []` in the body.

</details>
