# BST：不是“左孩子小、右孩子大”就够了

**中文** · [English](binary-search-trees.en.md) · [方法地图](../leetcode.md)

> 最近审阅：2026-10-09 · 先读：[DFS / BFS](traversal.md)

根是 8，右孩子是 10，而 10 的左孩子是 6。每对父子看起来都对：10 大于 8，6 小于 10。但 6 在 8 的右子树里，它必须大于 8。这棵树不是二叉搜索树。

BST 的约束作用于**整棵子树**，不是只检查相邻的两个节点。本篇采用严格顺序，不允许重复 key；实际容器也可以把同一个 key 的次数存在节点里。

## 搜索为什么可以舍掉半边

在节点 8 找 6：6 小于 8，右子树全部大于 8，不可能藏着 6，继续找左子树即可。省掉的是一棵子树，不保证刚好省掉一半节点。

设树高为 $h$，搜索和普通插入花 $O(h)$。平衡时 $h=O(\log n)$，按顺序插入却可能形成长链，变成 $O(n)$。普通 BST 不会自动平衡；AVL、红黑树通过额外维护来控制高度。[MIT 二叉树讲义](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/resources/lecture-notes/)

这里的代码假定输入是无环、无共享孩子的正常二叉树，key 是可比较整数；它不是不可信图结构的验证器。

## 验证时，把祖先的范围带下去

根没有上下界。向左走，收紧上界；向右走，收紧下界。前面的 6 接到的范围是 $(8,10)$，因此立即失败。

`None` 表示没有边界，不用某个猜测的“大整数”充当无穷大。

```python
@dataclass
class TreeNode:
    value: int
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

```

```python
def valid_bst(root):
    pending = [(root, None, None)]
    while pending:
        node, lower, upper = pending.pop()
        if node is None:
            continue
        if lower is not None and node.value <= lower:
            return False
        if upper is not None and node.value >= upper:
            return False
        pending.append((node.right, node.value, upper))
        pending.append((node.left, lower, node.value))
    return True
```

`TreeNode` 需要 `from dataclasses import dataclass`。所有节点都要检查，时间 $O(n)$；显式 DFS 栈最坏 $O(h)$，退化时仍可能到 $O(n)$。迭代避免 Python 递归深度限制，但没有凭空省掉状态。

## 中序遍历为什么是排序后的顺序

先左子树，再根，再右子树。左边全小于根，右边全大于根；对子树重复同样的论证，就得到严格递增序列。

```text
        8
       / \
      3   10
     / \
    1   6

inorder: 1 → 3 → 6 → 8 → 10
```

因此第 3 小是 6。无需先生成整个数组：每弹出一个待访问节点，就消耗一个名额。

```python
def kth_smallest(root, rank):
    if type(rank) is not int or rank < 1:
        raise ValueError("rank must be a positive integer")
    ancestors = []
    current = root
    while current is not None or ancestors:
        while current is not None:
            ancestors.append(current)
            current = current.left
        current = ancestors.pop()
        rank -= 1
        if rank == 0:
            return current.value
        current = current.right
    raise ValueError("rank exceeds the number of nodes")
```

这个函数假定输入已经是合法 BST；不会每次查询都先做一次 $O(n)$ 验证。`rank` 从 1 开始，空树或名次越界抛出异常。时间 $O(h+rank)$，最坏 $O(n)$；栈 $O(h)$。

如果频繁查询第几小，可以在每个节点维护子树大小。查询时比较左子树大小与目标名次，沿一条路径走，代价 $O(h)$；但插入、删除、旋转后都要维护大小，不能只给查询函数多加一个字段。

## 删除一个节点时，哪部分必须保住

| 情况 | 做法 | 要保留的关系 |
| --- | --- | --- |
| 没有孩子 | 断开父节点指向它的边 | 其他子树不变 |
| 一个孩子 | 让该孩子接替它的位置 | 孩子原有整棵子树不能丢 |
| 两个孩子 | 用右子树最小 key 替换，再删除那个后继节点 | 替换值仍在左右子树之间 |

最后一种要删除后继的**原节点**，否则重复 key 破坏严格顺序；若节点同时存 value，也要成对处理。这里只讲维护原理，不把它包装成完整平衡树实现。

## 怎么选容器，怎么检查

| 需求 | 合理的起点 |
| --- | --- |
| 只问某个 key 存不存在 | hash set / dict，平均常数查找，但不提供有序范围 |
| 数据基本不变，常做二分 | 排序数组；查找 $O(\log n)$，插入移动最坏 $O(n)$ |
| 经常插删，还要有序查询 | 平衡搜索树；实现与维护更复杂 |

手测 3 类树：空树、全向右的长链、父子都对但跨祖先边界的反例。再问：允许重复 key 时，你选“计数”“固定放左边”还是“固定放右边”？三种约定的验证条件不能混用。

完整教学代码在 [deeper_patterns.py](../code/deeper_patterns.py)。测试还会用 3,000 个节点的长链确认迭代版本不依赖递归深度，并对不同插入顺序核对排名。
