from bisect import bisect_left
from collections import deque
from dataclasses import dataclass


def prefix_lengths(pattern):
    prefix = [0] * len(pattern)
    matched = 0
    for index in range(1, len(pattern)):
        while matched and pattern[index] != pattern[matched]:
            matched = prefix[matched - 1]
        if pattern[index] == pattern[matched]:
            matched += 1
        prefix[index] = matched
    return prefix


def kmp_positions(text, pattern):
    if not pattern:
        return list(range(len(text) + 1))
    prefix = prefix_lengths(pattern)
    matched = 0
    positions = []
    for index, character in enumerate(text):
        while matched and character != pattern[matched]:
            matched = prefix[matched - 1]
        if character == pattern[matched]:
            matched += 1
        if matched == len(pattern):
            positions.append(index - len(pattern) + 1)
            matched = prefix[matched - 1]
    return positions


def window_maxima(values, width):
    if type(width) is not int or not 1 <= width <= len(values):
        raise ValueError("width must be an integer between 1 and input length")
    candidates = deque()
    maxima = []
    for index, value in enumerate(values):
        while candidates and candidates[0] <= index - width:
            candidates.popleft()
        while candidates and values[candidates[-1]] <= value:
            candidates.pop()
        candidates.append(index)
        if index >= width - 1:
            maxima.append(values[candidates[0]])
    return maxima


@dataclass
class TreeNode:
    value: int
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


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


def knapsack_once(items, capacity):
    if type(capacity) is not int or capacity < 0:
        raise ValueError("capacity must be a nonnegative integer")
    best = [0] * (capacity + 1)
    for weight, value in items:
        if type(weight) is not int or weight <= 0:
            raise ValueError("weights must be positive integers")
        for room in range(capacity, weight - 1, -1):
            best[room] = max(best[room], best[room - weight] + value)
    return best[capacity]


def minimum_coins(coins, amount):
    if type(amount) is not int or amount < 0:
        raise ValueError("amount must be a nonnegative integer")
    if any(type(coin) is not int or coin <= 0 for coin in coins):
        raise ValueError("coins must be positive integers")
    best = [0] + [float("inf")] * amount
    for coin in coins:
        for subtotal in range(coin, amount + 1):
            best[subtotal] = min(best[subtotal], best[subtotal - coin] + 1)
    return -1 if best[amount] == float("inf") else best[amount]


def lcs_length(first, second):
    previous = [0] * (len(second) + 1)
    for first_char in first:
        current = [0]
        for column, second_char in enumerate(second, start=1):
            if first_char == second_char:
                current.append(previous[column - 1] + 1)
            else:
                current.append(max(previous[column], current[-1]))
        previous = current
    return previous[-1]


def lis_length(values):
    tails = []
    for value in values:
        position = bisect_left(tails, value)
        if position == len(tails):
            tails.append(value)
        else:
            tails[position] = value
    return len(tails)


def cooldown_profit(prices):
    holding = float("-inf")
    sold = float("-inf")
    resting = 0
    for price in prices:
        holding, sold, resting = (
            max(holding, resting - price),
            holding + price,
            max(resting, sold),
        )
    return max(resting, sold)


def main():
    assert kmp_positions("ababab", "abab") == [0, 2]
    assert window_maxima([4, 1, 3, 5, 2, 5], 3) == [4, 5, 5, 5]
    root = TreeNode(8, TreeNode(3, TreeNode(1), TreeNode(6)), TreeNode(10))
    assert valid_bst(root) and kth_smallest(root, 3) == 6
    assert knapsack_once([(2, 5), (3, 7), (4, 8)], 5) == 12
    assert minimum_coins([1, 3, 4], 6) == 2
    assert lcs_length("CAB", "ACB") == 2
    assert lis_length([3, 5, 6, 2, 4]) == 3
    assert cooldown_profit([1, 2, 3, 0, 2]) == 3
    print("All deeper-pattern examples passed.")


if __name__ == "__main__":
    main()
