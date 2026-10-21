# LeetCode: learn the pattern, not just the answer

[中文](leetcode.md) · **English**

> Reading time: ~5 min · Last reviewed: 2026-10

A solution can look obvious while you're reading it, yet a week later you may not know where to start. These notes focus on that missing step: why a method fits, what its variables remember, and whether it still works when a condition changes.

The common patterns in [Top Interview 150](https://leetcode.com/studyplan/top-interview-150/) are a starting point. We focus on reusable Easy / Medium techniques, not 150 individual solutions or Hard puzzles. The links are public practice problems, not a collection of company interview questions.

## 1. Follow patterns, not problem numbers

If you're new to practice, start with the first two notes: get comfortable with Python's tools, then learn to store information you've already seen. Otherwise, jump to the pattern you're stuck on. Each note starts with a small worked example; continue to the code and complexity analysis when you need them.

| Order | Note | What to take away |
| --- | --- | --- |
| 1 | [Complexity and Python tools](algorithms/complexity-and-tools.en.md) | Average vs amortized costs, recursion space, dict / deque / heapq |
| 2 | [Hash maps and prefix sums](algorithms/hash-and-prefix.en.md) | Choosing keys and values, and when counts matter |
| 3 | [Pointers, windows, and binary search](algorithms/pointers-and-search.en.md) | Why each move can safely discard possibilities |
| 4 | [DFS / BFS: recursive or iterative](algorithms/traversal.en.md) | Order, stacks, queues, visited states, shortest-path assumptions |
| 5 | [Stacks, heaps, and linked lists](algorithms/stack-heap-links.en.md) | Waiting for an answer, retaining Top K, changing pointers safely |
| 6 | [Backtracking and dynamic programming](algorithms/backtracking-and-dp.en.md) | Paths vs states, memoization vs loops |
| 7 | [Sorting, intervals, and greedy choices](algorithms/sorting-and-greedy.en.md) | What sorting removes and why a local choice is safe |

Don't memorize every pattern at once. Once the basics feel familiar, choose the next note by the question you still can't explain:

| Where you get stuck | Read next | The extra step in the reasoning |
| --- | --- | --- |
| String matching keeps restarting | [String matching · KMP](algorithms/string-matching.en.md) | Which matched information survives a mismatch |
| The window maximum expires | [Monotonic queues](algorithms/monotonic-queue.en.md) | Why a candidate can be discarded permanently |
| BSTs feel like a recursion template | [Binary search trees](algorithms/binary-search-trees.en.md) | Ancestor bounds, inorder rank, and tree height |
| DP loop direction keeps going wrong | [Knapsack and coin change](algorithms/knapsack.en.md) | Reading the previous layer versus reusing this layer |
| LCS, LIS, and substrings blur together | [Sequence DP](algorithms/sequence-dp.en.md) | What prefixes, endpoints, and contiguity constrain |
| One best value loses a rule | [State-machine DP](algorithms/state-machine-dp.en.md) | Which history changes the next legal actions |

For an optional number-theory topic, [primality and sieves](algorithms/primes-and-sieves.en.md) compares testing one number with preprocessing a range, including the square-root boundary. It is not part of the first-pass essentials.

For syntax, start with our contributor's [Python notes](python.en.md). These pages focus on what each line does for the algorithm; use both together.

## 2. Ask 5 questions before coding

1. **What does the input guarantee?** Sorted? Negative values? Duplicates? Cycles?
2. **What is the output?** Existence, one solution, an optimum, or every solution?
3. **What work repeats?** Searching from the beginning, or summing the same interval?
4. **What information must survive?** Indices, counts, a window, a path, or a state's best result?
5. **What can be discarded?** Why can't that discard hide a valid answer? Start your invariant there.

| Signal in the problem | Consider | Check first |
| --- | --- | --- |
| Seen before / how many / at which position | set / dict / Counter | Counts and original indices may matter |
| Contiguous subarray or substring | Window / prefix sum | Negative values can break sum monotonicity |
| Sorted, or feasible from some boundary onward | Binary search | Can you state a monotone predicate? |
| Connectivity, levels, fewest steps | DFS / BFS | Ordinary BFS needs equal edge costs |
| Every combination | Backtracking | The output itself may be exponential |
| Many paths lead to the same subproblem | DP / memoization | Does the state capture everything affecting the future? |
| Only the largest few values matter | Min-heap | Is a full sort necessary? |

These signals narrow the choices; they don't decide for you. “Shortest” doesn't automatically imply BFS: fewest graph steps and fewest coins for an amount lead to different reasoning. Check the actual constraints.

## 3. A review session

- **Explain before coding:** describe brute force, then the repeated work you will avoid.
- **Trace 4–6 elements:** write down variables instead of trusting a mental sketch.
- **Close the solution and rewrite:** check empty input, duplicates, and extreme orderings.
- **Change a condition a few days later:** sorted to unsorted, positive to signed, tree to cyclic graph. Explain what breaks.

Don't worry about hitting a problem count. Being able to rewrite a solution days later and explain what changes under a different constraint matters more than remembering its number.

If your interviews are in English but you study in Chinese, practice explaining a little in English too. The Chinese pages label key terms in English. After an example, put the notes aside and explain what the state stores, how the algorithm moves, why it is correct, and what it costs. A few notes include short spoken explanations to adapt, not memorize. Use the language link for the full Chinese or English page.

## 4. Run the examples

The templates in [patterns.py](code/patterns.py) and [deeper_patterns.py](code/deeper_patterns.py) run on Python 3.10+ with no third-party packages. Use **Python 3.11+** to run all the tests below: the tests read TOML configuration using [tomllib](https://docs.python.org/3.11/library/tomllib.html), which joined the standard library in 3.11. Run these commands from the repository root:

```bash
python interview/code/patterns.py
python -m unittest discover -s site/tests -p 'test_algorithm_patterns.py'
python interview/code/deeper_patterns.py
python -m unittest discover -s site/tests -p 'test_deeper_algorithm_patterns.py'
```

Tests compare templates against small exhaustive examples, and check that note templates match the tested code. These are learning examples, not a claim to match every LeetCode contract. Adapt the `Solution` wrapper, zero- vs one-based indices, and mutation rules when submitting.

## 5. Where to start today

- Twenty minutes: Two Sum → `get` vs `in` → rewrite without the answer.
- Mixing up DFS and BFS: open the [traversal lab](algorithms/traversal.en.md), predict the next node, then step forward.
- Memorizing without transferring: try each note's changed-condition question before adding more problems.

You don't need a new problem every session. Resolving something you couldn't explain in an old problem is useful practice too. Leave time for projects and life outside coding.

## 6. Other explanations to compare

Sometimes a different explanation makes an idea click. The resources below are primarily in Chinese; Hello Algo also offers an English edition. Choose by the question you're trying to answer, rather than treating all four as required reading.

| What you want to understand | Another resource |
| --- | --- |
| How a stack, queue, or tree changes step by step | [Hello Algo](https://www.hello-algo.com/): follow the diagrams before returning to the code |
| Why a familiar template works | [EndlessCheng's foundational algorithm lessons](https://github.com/EndlessCheng/codeforces-go/blob/master/leetcode/README.md): work through a two-pointer, binary-search, or DP lesson; the accompanying lists also contain advanced problems |
| How related problems use the same technique differently | [labuladong's explanations](https://labuladong.online/zh/algo/intro/core-intro/): compare adaptations of a pattern; some material and tools have paid access |
| Where to start debugging a DP solution | [leetcode-master's DP foundations](https://github.com/youngyangyang04/leetcode-master/blob/master/problems/%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%E7%90%86%E8%AE%BA%E5%9F%BA%E7%A1%80.md): check state meaning, initialization, recurrence, and iteration order against a small hand-computed example |

We work through and test our own examples and code, with links to the references. Variable names, interval conventions, and state definitions differ between sources; borrow the reasoning without splicing incompatible templates together. If something here still doesn't make sense, tell us which step lost you.
