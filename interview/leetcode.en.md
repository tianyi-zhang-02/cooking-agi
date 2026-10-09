# LeetCode: learn the pattern, not just the answer

[中文](leetcode.md) · **English**

> Reading time: ~4 min · Last reviewed: 2026-09

A solution can look obvious while you're reading it and disappear from memory a week later. This is a notebook to come back to: **recognize the structure, choose a tool, explain why it works, then write it yourself.**

The common patterns in [Top Interview 150](https://leetcode.com/studyplan/top-interview-150/) are a starting point, not a list of 150 solutions to copy. We focus on transferable Easy / Medium techniques, not Hard puzzles. Links are public practice problems, not company interview reports or claims about anyone's completed problems.

## 1. Follow patterns, not problem numbers

Each note has small sections: when to use it → a hand-worked example → Python → complexity → where it fails → self-checks.

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

This isn't a keyword classifier. “Shortest” could mean BFS or DP; structure and constraints decide.

## 3. A review session

- **Explain before coding:** describe brute force, then the repeated work you will avoid.
- **Trace 4–6 elements:** write down variables instead of trusting a mental sketch.
- **Close the solution and rewrite:** check empty input, duplicates, and extreme orderings.
- **Change a condition a few days later:** sorted to unsorted, positive to signed, tree to cyclic graph. Explaining what breaks matters more than remembering an ID.

Don't worry about hitting a problem count. Being able to rewrite a solution days later and explain what changes under a different constraint matters more than remembering its number.

## 4. Run the examples

The core templates live in [patterns.py](code/patterns.py), require Python 3.10+, and use only the standard library. From the repository root:

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

Aim to explain it, rewrite it, and adapt it. Coding practice is one part of preparation; leave time for projects and life outside it.
