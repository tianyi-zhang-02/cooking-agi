# DFS / BFS：递归和迭代到底差在哪

**中文** · [English](traversal.en.md) · [方法地图](../leetcode.md)

> 阅读时间：约 9 分钟 · 最近审阅：2026-09

把图想成房间和走廊。DFS 先沿一条路走到底，再回来；BFS 先看一步能到的房间，再看两步能到的。**DFS / BFS 是探索顺序，recursive / iterative 是实现方式，不是同一组分类。**

## 1. 先用同一张图走一遍

图中邻居固定按从左到右访问。先猜下一步是谁，再点按钮；“发现”表示已登记，“处理”表示正在查看它的邻居。

<div class="traversal-lab" data-traversal-lab data-language="zh">
<p>无 JavaScript 也能读：A 连 B、C；B 连 D、E；C 连 F。DFS 的首次访问顺序是 A B D E C F；BFS 是 A B C D E F。</p>
</div>

这只是教学图，不是性能测试。切换“有环”后，看 visited 怎样阻止节点被反复加入；切换“长链”和“宽树”，比较等待队列和 DFS 路径的差别。

## 2. Recursive DFS：让调用栈记住回来去哪

图用 adjacency list 表示，节点用整数或字符串；不存在的 key 当作无出边，start 作为一个有效起点。下面只遍历 start 可达的部分。

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

进函数时登记，避免沿环走回来。调用 `visit(B)` 时，A 的函数没有消失，只是等 B 返回后接着看下一个邻居。

- **好处**：写树的递归、后序汇总、回溯很自然；函数的输入输出就是子问题。
- **代价**：有调用开销，也受 Python 递归深度限制。长链别只靠调大限制，改迭代通常更稳。
- **树题怎么想**：最大深度是 `1 + max(左子树深度, 右子树深度)`，空树是 0。先定义返回值，再填代码。

## 3. Iterative DFS：自己管理那一摞任务

对没有环和共享孩子的树，简单写法是：从 list 尾部 `pop()` 一个节点，处理后把右孩子、左孩子依次 `append`，这样左孩子会先出来。

一般图有重复路径。只求可达节点时，“入栈登记 visited”的简单栈足够，但访问顺序未必与递归完全一致。若要严格模拟递归，就让每个栈帧保存**节点 + 还没访问完的邻居**：

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

`iter` 是一个逐个读取的游标；`next(neighbors, None)` 读不到时返回 None。这里节点不使用 None。帧耗尽就退栈，等价于函数返回。这版比简单栈多几行，但可以保留递归的进入／退出语义；不会受 Python 递归层数限制，也**不等于不占栈空间**。

## 4. BFS：队列保证按距离一层层看

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

`distances` 同时当 visited；**入队时登记**，而不是出队后才登记。这样多个父节点指向同一个孩子时，不会重复入队。若想返回路径，再存 `parent[neighbor] = node`，找到目标后倒着还原。

普通 BFS 能找最少边数路径，因为先进先出的队列先处理距离小的节点。边权不同时，“边少”不代表“代价低”；别把这个模板直接当通用最短路。[DFS / BFS 与最短路说明](https://algs4.cs.princeton.edu/41graph/)

## 5. 复杂度、优势和限制

| 方法 | 时间 | 辅助空间，不含输出 | 更自然的用途 |
| --- | --- | --- | --- |
| 树的 recursive DFS | O(n) | O(h) 调用栈 | 后序计算、路径回溯 |
| 树的 frame-based iterative DFS | O(n) | O(h) 显式栈 | 深树，控制进入／退出 |
| 树的 BFS | O(n) | O(w) 队列 | 层序、等边代价的最近目标 |
| 一般图的 DFS / BFS | O(V + E) | O(V) | 连通性、等边代价最短路 |

h 是高度，w 是最大层宽。图的 O(V) 包括 visited；输入 adjacency list 本身 O(V+E) 另算。上面示例还返回 O(V) 的 order 或 distance。树专用写法可省 visited；不能把这个节省套到有环的图。只搜一个连通部分时，V / E 指访问到的部分。

二维网格也能当图：每个格子是节点，上下左右是邻居。R×C 网格的 flood fill 通常 O(RC) 时间、最坏 O(RC) 辅助空间；可以原地改格子代替 visited，但会修改输入。

## 6. 选几题练“换个壳还是同一个方法”

- [Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)：递归返回值，然后用 BFS 重写。
- [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)：每层开始记录 `len(queue)`，恰好处理这一层的节点。
- [Number of Islands](https://leetcode.com/problems/number-of-islands/)：每找到一个未访问的陆地，flood fill 一个连通分量。

<details class="interview" markdown="1">
<summary>DFS 为什么不保证找到最短路？递归改循环会改变这个结论吗？</summary>

DFS 可以先沿很长的支路碰到目标，即使另一个邻居一步就能到。换成显式栈只改变实现，不改变深度优先的顺序；要按步数找最近目标，用等边代价下的 BFS。

</details>

<details class="interview" markdown="1">
<summary>同一个节点，什么时候能全局 visited，什么时候不能？</summary>

只求可达性时，同一节点处理一次够了。找所有简单路径时，不同路径到同一节点可能是不同答案，要用当前路径集合并回退。状态若还有钥匙、剩余预算等，visited 的 key 也必须包含这些信息。

</details>
