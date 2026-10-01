from collections import deque
from functools import cache
import heapq


def two_sum(values, target):
    seen = {}
    for index, value in enumerate(values):
        complement = target - value
        if complement in seen:
            return [seen[complement], index]
        seen[value] = index
    return None


def count_subarrays(values, target):
    frequencies = {0: 1}
    prefix = 0
    total = 0
    for value in values:
        prefix += value
        total += frequencies.get(prefix - target, 0)
        frequencies[prefix] = frequencies.get(prefix, 0) + 1
    return total


def sorted_pair(values, target):
    left, right = 0, len(values) - 1
    while left < right:
        total = values[left] + values[right]
        if total == target:
            return [left, right]
        if total < target:
            left += 1
        else:
            right -= 1
    return None


def longest_unique(text):
    last_seen = {}
    left = best = 0
    for right, character in enumerate(text):
        left = max(left, last_seen.get(character, -1) + 1)
        last_seen[character] = right
        best = max(best, right - left + 1)
    return best


def lower_bound(values, target):
    left, right = 0, len(values)
    while left < right:
        middle = (left + right) // 2
        if values[middle] < target:
            left = middle + 1
        else:
            right = middle
    return left


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


class ListNode:
    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node


def reverse_list(head):
    previous = None
    current = head
    while current is not None:
        following = current.next
        current.next = previous
        previous = current
        current = following
    return previous


def has_cycle(head):
    slow = fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False


def next_warmer(temperatures):
    waits = [0] * len(temperatures)
    pending = []
    for index, temperature in enumerate(temperatures):
        while pending and temperatures[pending[-1]] < temperature:
            earlier = pending.pop()
            waits[earlier] = index - earlier
        pending.append(index)
    return waits


def largest_values(values, count):
    if count <= 0:
        return []
    heap = []
    for value in values:
        if len(heap) < count:
            heapq.heappush(heap, value)
        elif value > heap[0]:
            heapq.heapreplace(heap, value)
    return sorted(heap, reverse=True)


def subsets(values):
    result = []
    path = []

    def extend(start):
        result.append(path.copy())
        for index in range(start, len(values)):
            path.append(values[index])
            extend(index + 1)
            path.pop()

    extend(0)
    return result


def rob_memo(values):
    @cache
    def best(index):
        if index >= len(values):
            return 0
        return max(best(index + 1), values[index] + best(index + 2))

    return best(0)


def rob_iterative(values):
    two_back = one_back = 0
    for value in values:
        current = max(one_back, two_back + value)
        two_back, one_back = one_back, current
    return one_back


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def can_jump(jumps):
    if not jumps:
        return False
    farthest = 0
    for index, jump in enumerate(jumps):
        if index > farthest:
            return False
        farthest = max(farthest, index + jump)
        if farthest >= len(jumps) - 1:
            return True
    return False


if __name__ == "__main__":
    graph = {"A": ["B", "C"], "B": ["D", "E"], "C": ["F"], "D": [], "E": [], "F": []}
    print("Two Sum:", two_sum([4, 1, 7], 8))
    print("Prefix count:", count_subarrays([1, -1, 1], 1))
    print("DFS recursive:", dfs_recursive(graph, "A"))
    print("DFS iterative:", dfs_iterative(graph, "A"))
    print("BFS distances:", bfs_distances(graph, "A"))
    print("DP:", rob_memo([2, 7, 9, 3, 1]), rob_iterative([2, 7, 9, 3, 1]))
