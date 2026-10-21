# Five short exercises: calculate, implement, check

[中文](coding-checks.md) · **English**

These exercises need neither a large model nor a GPU. Define the inputs, expected results, and boundaries before comparing implementations. They are original practice problems, not questions attributed to a company.

The reference implementation, [prep_checks.py](code/prep_checks.py), needs only NumPy. From the repository root:

```bash
python interview/code/prep_checks.py
python -m unittest discover -s site/tests -p 'test_interview_preparation.py'
```

## kNN: what does vectorization actually save? {#knn}

Take three points, `[[0,0], [2,0], [0,3]]`, with labels `[0,1,1]`. The query `[1,0]` has squared distances `[1,1,10]`; `[0,2]` has distances `[4,8,1]`. Write a loop first, then calculate the same distances with NumPy. The neighbor order should agree.

Our convention breaks distance ties by original row index and voting ties by the smaller numerical label. The first query's 1-NN prediction is therefore 0; the second is 1. Other conventions are possible, but changing implementations should not silently change them.

For queries `[Q,D]` and points `[N,D]`, direct broadcasting produces differences `[Q,N,D]`. Summing squared differences over D gives `[Q,N]`. **Fewer Python loops does not mean fewer distance calculations—or less memory.**

| Implementation | Distance work | Main extra storage, excluding inputs and final neighbors |
| --- | --- | --- |
| Loop over queries, points, and features | $O(QND)$ | One distance row, $O(N)$ |
| Broadcast everything | $O(QND)$ | Differences $O(QND)$, plus distances and sorting indices |
| Process C queries at a time | $O(QND)$ | $O(CND+CN)$; C controls the peak, not the intended neighbors |

To handle distance ties reproducibly, the reference performs a stable sort of each entire row. That adds $O(QN\log N)$ time; final neighbor indices take $O(Qk)$ space. Computing distances is only part of the kNN cost. A partial-selection alternative needs an explicit rule for sorting the chosen k neighbors and breaking ties at the cutoff.

A scale check: 1,000 queries, 10,000 points, and 128 features produce 1.28 billion difference elements. The float64 differences alone occupy about **10.24 GB**, before output and temporary arrays. Blocks of 10 queries reduce one difference tensor to about 102.4 MB. These are element counts, not measured process peaks.

<details markdown="1">
<summary>Can we avoid the three-dimensional differences?</summary>

Yes: $\|q-x\|^2=\|q\|^2+\|x\|^2-2q^\top x$ reduces the main intermediate to `[Q,N]`, which can also be blocked. But subtracting large, nearly equal quantities can lose precision for nearby vectors with large coordinates. Clipping small negative results to zero cannot recover a lost distance ordering. Compare against direct differences on small inputs, then consider centering, higher precision, or another implementation.

This exercise does not implement an ANN index. Exact search, tree indexes, and approximate retrieval have different operating conditions. See [scikit-learn's neighbor algorithms](https://scikit-learn.org/stable/modules/neighbors.html) and the [retrieval route](../04-search/README.en.md).

</details>

Try duplicate points and `k` equal to 1, N, and N+1. The first two need defined results; this implementation rejects the last. Empty queries return empty results; an empty point set cannot support a valid k. The exercise assumes finite float64 inputs of ordinary scale; it does not claim exact results for arbitrarily extreme coordinates.

## Binary search: a match is not a boundary {#bounds}

Finding any 2 in `[1,2,2,4]` is not enough. The first value ≥ 2 is at index 1; the last value ≤ 2 is at index 2.

Both searches maintain a half-open interval `[lower, upper)`. The first discards values `< target` to find the `bisect_left` position. The second discards values `≤ target` to find `bisect_right`, then subtracts one. [Python's bisect documentation](https://docs.python.org/3/library/bisect.html) specifies these partition conditions.

| target | First value ≥ target | Last value ≤ target |
| --- | ---: | ---: |
| 0 | 0 | -1: absent |
| 2 | 1 | 2 |
| 3 | 3 | 2 |
| 5 | 4: absent | 3 |

The reference `search_bounds` returns `(0,-1)` for an empty list. Check sentinels before indexing: Python's `values[-1]` means the last element, not an absent result. Inputs must already be sorted ascending. Search takes $O(\log N)$ time and $O(1)$ extra space; sorting and insertion have separate costs.

## Broadcasting: valid code can compute the wrong loss {#broadcast}

Two predictions are `[[1],[3]]` and their targets are `[1,2]`. Direct subtraction gives:

```python
import numpy as np

predictions = np.array([[1.0], [3.0]])
targets = np.array([1.0, 2.0])
wrong = predictions - targets
right = predictions - targets[:, None]
assert wrong.shape == (2, 2)
assert right.shape == (2, 1)
assert np.mean(wrong ** 2) == 1.5
assert np.mean(right ** 2) == 0.5
```

The first compares every prediction with every target: `[[0,-1],[2,1]]`. The second aligns examples: `[[0],[1]]`. NumPy checks the numbers here; in PyTorch autograd, both versions would support backpropagation and yield scalar losses. A training loop that runs without errors will not expose the difference.

[NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) compares trailing dimensions: sizes must match, or one must be 1. It checks shapes, not which axis represents examples. For `[B,H,T,D]`, matrix multiplication, and masks, continue with [tensor operations and shapes](../00-foundations/pytorch/operations-and-shapes.en.md).

## Task dependencies: what can start now? {#dependencies}

Consider a small data workflow: read the data, then run cleaning and auditing independently, and start training only after both finish.

```text
             → clean ─┐
read ────────         ├→ train
             → audit ─┘
```

Kahn's algorithm tracks each task's unfinished predecessors: its indegree. Take tasks with indegree zero, and reduce successor indegrees after completion. Removing every task yields a topological order. If tasks remain but none is ready, there is a cycle.

The reference `dependency_layers` returns `[[read], [clean,audit], [train]]`. Its input maps tasks to predecessors; it separately builds successor lists and deduplicates repeated dependencies. Time and space are $O(V+E)$, including the output.

**These layers describe dependencies, not an optimal parallel schedule.** If independent tasks take 1 and 10 seconds, successors of the short task need not wait for the long one. A real executor releases dependencies as individual tasks finish and accounts for worker limits, resources, failure, retries, and idempotency. Python's [`TopologicalSorter.get_ready()` / `done()`](https://docs.python.org/3/library/graphlib.html) illustrates the distinction: call `done()` when a task finishes, not when it is submitted to a worker.

Add a dependency from `read` to `train`: the algorithm must reject the cycle. Add an independent `report` task: it can appear in the first layer. This implementation also requires every predecessor to be declared, so a typo cannot silently become an already-completed node.

## Markov chains: three different questions {#markov}

Suppose states A and B have transition matrix

$$P=\begin{bmatrix}0.8&0.2\\0.4&0.6\end{bmatrix}.$$

**Rows are the current state; columns are the next state**, with each row summing to 1. The same table can answer a finite-step, stationary, or waiting-time question—but those are not interchangeable.

| Question | Calculation | Result |
| --- | --- | --- |
| Starting at A, where are we after two steps? | $[1,0]P^2$ | $[0.72,0.28]$ |
| Which distribution is unchanged by a transition? | $\pi P=\pi,\ \pi_A+\pi_B=1$ | $[2/3,1/3]$ |
| Starting at A, how long until the first visit to B? | $h_B=0,\ h_A=1+0.8h_A$ | $h_A=5$ |

For the first question, paths A→A→B and A→B→B give $0.8\times0.2+0.2\times0.6=0.28$. For the second, solve $0.2\pi_A=0.4\pi_B$ and normalize. For the third, spend one step, then with probability 0.8 return to A and start waiting again. Stop upon reaching B; its subsequent transitions do not enter the hitting time.

The reference `markov_example` checks the three results with matrix multiplication and linear equations. It is a fixed two-state example, not a general solver for all chains. More generally, non-target states satisfy $h=\mathbf1+Qh$, where $Q$ is $P$ restricted to the non-target rows and columns (here $Q=[0.8]$). Solving $(I-Q)h=\mathbf1$ for finite expected times requires appropriate reachability and finiteness conditions.

Stationarity does not guarantee convergence from every initial distribution. With `[[0,1],[1,0]]`, `[0.5,0.5]` is stationary, but starting at A alternates forever. If a starting state cannot reach the target, its hitting time is not finite. To add actions to states, continue with [MDPs and Bellman equations](../05-post-training/deep-rl/mdp-bellman.en.md).

## Change one assumption {#next}

The number of problems you have solved is a poor measure of mastery. With ten times as much kNN data, how would you block it? When a task fails, when may its successors run? How do duplicates change a binary-search boundary? Predict the effect, then test it rather than memorizing this page.

Return to the [RE / MLE preparation route](ai-roles-prep.en.md), or continue with [attention, sampling, and MLP implementations](../00-foundations/hand-write-kit.en.md).
