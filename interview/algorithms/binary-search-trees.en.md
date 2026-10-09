# BSTs: checking each child against its parent is not enough

[中文](binary-search-trees.md) · **English** · [Pattern map](../leetcode.en.md)

> Last reviewed: 2026-10-09 · Prerequisite: [DFS / BFS](traversal.en.md)

The root is 8, its right child is 10, and 10's left child is 6. Every parent–child comparison looks fine: 10 exceeds 8; 6 is below 10. But 6 belongs to 8's right subtree and must exceed 8. This is not a BST.

The constraint applies to **entire subtrees**, not just adjacent nodes. This note uses strict ordering and rejects duplicate keys. A real container might instead store a count at each key.

## Why search can discard a subtree

Searching for 6 at node 8, go left. Every key on the right exceeds 8, so none can be 6. You discard a subtree, not necessarily half the nodes.

Search and ordinary insertion cost $O(h)$ for height $h$. A balanced tree has $h=O(\log n)$; sorted insertions can create a chain with $O(n)$ height. An ordinary BST doesn't balance itself. AVL and red–black trees maintain additional structure to control height. [MIT binary-tree notes](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/resources/lecture-notes/)

The code assumes an actual tree without cycles or shared children and with comparable integer keys. It does not validate arbitrary untrusted graph structures.

## Pass ancestor bounds into each subtree

The root starts without bounds. Going left tightens the upper bound; going right tightens the lower one. The misplaced 6 inherits $(8,10)$ and fails immediately.

`None` represents an absent bound; no guessed “large enough” integer is required.

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

Import `dataclass` from `dataclasses`. Validation visits every node: $O(n)$ time and $O(h)$ DFS stack space, potentially $O(n)$ for a degenerate tree. Iteration avoids Python's recursion-depth limit, not the need to retain state.

## Why inorder produces sorted keys

Visit the left subtree, then the root, then the right subtree. All left keys are smaller and all right keys larger. Applying the same argument recursively produces a strictly increasing sequence.

```text
        8
       / \
      3   10
     / \
    1   6

inorder: 1 → 3 → 6 → 8 → 10
```

The third-smallest key is 6. No complete output array is necessary: each visited node consumes one remaining place.

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

This assumes a valid BST; it doesn't perform $O(n)$ validation on every query. Rank is one-based. Empty input or an out-of-range rank raises an exception. Time is $O(h+rank)$, at worst $O(n)$, with an $O(h)$ stack.

For frequent rank queries, maintain each node's subtree size. Comparing the left size with the requested rank then follows one path in $O(h)$. Insertion, deletion, and rotations must maintain those sizes; adding a field to the query alone is insufficient.

## What deletion must preserve

| Case | Action | Relationship to preserve |
| --- | --- | --- |
| No children | Disconnect the parent edge | Other subtrees stay intact |
| One child | Replace the node with that child | Keep the child's entire subtree |
| Two children | Replace the key with the right subtree's minimum, then delete that successor | Replacement lies between the remaining subtrees |

The last case must remove the successor's **original node**; otherwise the duplicate violates strict ordering. If keys have associated values, move them together. This explains the invariant, not a complete balanced-tree implementation.

## Choose a container, then test its contract

| Need | Reasonable starting point |
| --- | --- |
| Membership only | Hash set / dict: average constant-time lookup, no ordered ranges |
| Mostly static data with searches | Sorted array: $O(\log n)$ search, worst-case $O(n)$ insertion shifts |
| Frequent updates and ordered queries | Balanced search tree, with more maintenance complexity |

Test an empty tree, a right-only chain, and a tree satisfying local comparisons but violating an ancestor bound. If duplicates are allowed, choose counts, left placement, or right placement explicitly; their validation rules differ.

Full teaching code: [deeper_patterns.py](../code/deeper_patterns.py). Tests include a 3,000-node chain to check independence from recursion depth, and rank checks across different insertion orders.
