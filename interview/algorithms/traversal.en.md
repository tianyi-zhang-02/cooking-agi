# DFS / BFS: recursive or iterative?

[中文](traversal.md) · **English** · [Pattern map](../leetcode.en.md)

> Reading time: ~9 min · Last reviewed: 2026-09

Think of rooms and corridors. DFS follows one route before returning; BFS checks rooms one step away, then two steps away. **DFS / BFS describe exploration order. Recursive / iterative describe implementation. These are different choices.**

## 1. Walk the same graph first

Neighbors are visited left to right. Predict the next node before stepping. “Discovered” means registered; “processing” means examining its neighbors.

<div class="traversal-lab" data-traversal-lab data-language="en">
<p>Without JavaScript: A connects to B, C; B to D, E; C to F. DFS discovery order is A B D E C F; BFS order is A B C D E F.</p>
</div>

This is a teaching trace, not a benchmark. Try the cycle to see visited prevent rediscovery, then the chain and wide tree to compare the queue with the DFS path.

## 2. Recursive DFS: the call stack remembers where to return

Use an adjacency list with integer or string nodes. Missing keys have no outgoing edges; start is a valid vertex. These functions explore only the part reachable from start.

```python
def dfs_recursive(graph, start):
    seen = set()
    order = []

    def visit(node):
        seen.add(node)
        order.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in seen:
                visit(neighbor)

    visit(start)
    return order
```

Register on entry to stop cycles. Calling `visit(B)` doesn't discard A's frame: it waits to continue with its next neighbor after B returns.

- **Strength:** tree recursion, postorder aggregation, and backtracking map naturally to function inputs and outputs.
- **Cost:** calls have overhead and Python limits recursion depth. For deep chains, iteration is usually safer than just raising that limit.
- **Tree example:** maximum depth is `1 + max(left depth, right depth)`; an empty tree returns 0. Define the return value before coding.

## 3. Iterative DFS: manage the pending work yourself

For a tree without cycles or shared children, a simple version pops from a list's end and appends the right child before the left, so the left is processed first.

General graphs have repeated paths. A simple stack with mark-on-push works for reachability, but its order need not match recursive DFS exactly. To simulate recursion faithfully, store **the node and its remaining neighbors** in each frame:

```python
def dfs_iterative(graph, start):
    seen = {start}
    order = [start]
    frames = [(start, iter(graph.get(start, [])))]
    while frames:
        node, neighbors = frames[-1]
        neighbor = next(neighbors, None)
        if neighbor is None:
            frames.pop()
        elif neighbor not in seen:
            seen.add(neighbor)
            order.append(neighbor)
            frames.append((neighbor, iter(graph.get(neighbor, []))))
    return order
```

`iter` creates a cursor; `next(neighbors, None)` returns None at exhaustion. None is not a node in this example. Popping an exhausted frame mirrors returning from a function. More bookkeeping than a simple stack, but entry/exit semantics are preserved. Avoiding Python recursion limits does **not** eliminate stack memory.

## 4. BFS: a queue processes increasing distances

```python
from collections import deque

def bfs_distances(graph, start):
    distances = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in graph.get(node, []):
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances
```

`distances` doubles as visited. **Register on enqueue**, not only on dequeue, so multiple parents don't queue the same child repeatedly. To return a path, store `parent[neighbor] = node` and reconstruct backward from the goal.

BFS finds a minimum-edge path because FIFO processes closer nodes first. With unequal edge weights, fewer edges need not mean lower cost. This isn't a general weighted shortest-path algorithm. [DFS, BFS, and shortest paths](https://algs4.cs.princeton.edu/41graph/)

## 5. Complexity and tradeoffs

| Method | Time | Auxiliary space, excluding output | Natural use |
| --- | --- | --- | --- |
| Recursive DFS on a tree | O(n) | O(h) call stack | Postorder computation, path backtracking |
| Frame-based iterative DFS on a tree | O(n) | O(h) explicit stack | Deep trees, entry/exit control |
| BFS on a tree | O(n) | O(w) queue | Levels, nearest equal-cost target |
| DFS / BFS on a general graph | O(V + E) | O(V) | Reachability, equal-cost shortest paths |

h is height; w is maximum layer width. General-graph space includes visited, but not the O(V+E) input adjacency list. The examples also return O(V) order or distance data. Tree-specific code can omit visited; cyclic graphs cannot. For a single reachable component, V and E refer to the explored portion.

A grid is also a graph: cells are vertices, up/down/left/right moves are edges. Flood fill on an R×C grid usually costs O(RC) time and worst-case O(RC) auxiliary space. Marking cells in place can replace visited but mutates input.

## 6. Practice the same method in different forms

- [Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/): recursive return values, then rewrite with BFS.
- [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/): capture `len(queue)` at each level and process exactly that many nodes.
- [Number of Islands](https://leetcode.com/problems/number-of-islands/): each new unvisited land cell starts a component flood fill.

<details class="interview" markdown="1">
<summary>Why doesn't DFS guarantee a shortest path? Does replacing recursion with a loop help?</summary>

DFS may reach the goal through a long branch before checking a one-edge alternative. An explicit stack changes the implementation, not the depth-first ordering. Use BFS for minimum steps with equal edge costs.

</details>

<details class="interview" markdown="1">
<summary>When is global visited valid, and when isn't it?</summary>

Reachability needs each node processed once. Enumerating all simple paths needs path-local membership with backtracking because distinct paths can be distinct answers. If a state includes keys or remaining budget, visited must include those too.

</details>
