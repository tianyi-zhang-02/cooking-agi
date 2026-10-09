# 字符串匹配：失败之后，哪些比较不用重做

**中文** · [English](string-matching.en.md) · [方法地图](../leetcode.md)

> 最近审阅：2026-10-09 · 先读：[双指针与窗口](pointers-and-search.md)

在 `ababab` 里找 `abab`，答案是下标 0 和 2。第一次匹配结束后，如果直接跳过 4 个字符，就漏掉了重叠的第二次。

字符串匹配真正需要想清楚的是：**已经看过的字符，能不能帮下一次尝试省点事？** 日常代码直接用 `find`、`in` 往往更合适；学 KMP 是为了看懂如何复用这段信息，不是要求所有字符串问题都手写 KMP。

## 从暴力法看到重复工作

设文本长 $n$，模式长 $m$。暴力法逐个试起点，最坏每个起点比较 $m$ 次，时间 $O(nm)$。例如长串 `aaaa…a` 中找 `aaaab`，每次都在最后一个字母失败，前面那串 `a` 被反复比较。

KMP 不把文本指针退回去，而是保留“到这里为止，匹配了模式的多长前缀”。失败时，缩短这个已匹配前缀，再拿**当前同一个字符**继续试。

## 前缀表到底记什么

`prefix[index]` 记录 `pattern[:index+1]` 的最长相等真前缀、真后缀的长度。“真”指不能拿整个字符串与自己比较；前缀和后缀可以重叠。

| 模式前缀 | 最长相等的两端 | 长度 |
| --- | --- | --- |
| a | 空 | 0 |
| ab | 空 | 0 |
| aba | a | 1 |
| abab | ab | 2 |
| ababa | aba | 3 |
| ababac | 空 | 0 |

看最后一个 `c`：此前匹配长度是 3，先试模式下标 3 的 `b`，不等；沿前缀表退到长度 1，还是要 `b`；再退到 0，要 `a`，仍不等，结果才是 0。不是失败一次就无条件归零。

```python
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
```

为什么可以只沿这条链找？一个还可能成功的较短前缀，必须也是刚才那段已匹配内容的后缀；否则已经读过的字符就对不上。前缀表正好列出了下一段可能成立的长度。

## 搜索时，把信息留给下一轮

```python
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
```

输入是 Python 字符串，返回所有起点，允许重叠。这里约定空模式匹配 `0…len(text)` 的每个边界；这不是每道题都采用的约定。

手走 `ababab` / `abab`：

| 读到的文本位置 | 此刻发生什么 | 保留的 matched |
| --- | --- | --- |
| 0–2 | 依次匹配 a、b、a | 1、2、3 |
| 3 | 匹配完整，记录起点 0 | 回到 prefix[3] = 2 |
| 4 | 已保留的 ab 后面又来了 a | 3 |
| 5 | 再次完整，记录起点 2 | 2 |

匹配成功后不归零，就是为了保留重叠。`matched` 是**长度**，同时也是下一次要比较的模式下标；不要把它误当成最后一个已匹配字符的位置。

## while 套在 for 里，为什么仍是线性的

每次读入新字符，`matched` 至多加 1；每次回退都会严格减小。回退次数不可能长期超过此前累积的增长次数。前缀表和搜索各算一次这笔账，总时间 $O(n+m)$，额外工作空间 $O(m)$，另加 $O(z)$ 存放 $z$ 个匹配位置。

这是摊还分析，不是说每个字符只比较一次。若只要第一次匹配，可以立即返回，不必保留所有结果。

## 哪些地方不能直接套

- 找的是连续子串，不是允许跳字的子序列；后者看[序列 DP](sequence-dp.md)。
- 大小写、Unicode 组合字符、用户眼里的一个“字”要先定义清楚。Python 的字符串下标不是屏幕上的字形编号；归一化还可能改变原始下标映射。
- 普通滚动哈希要考虑碰撞；KMP 不靠哈希判断相等。很多模式同时查，则是另一个数据结构选择问题，不要把单模式模板硬套上去。
- 先试 `aaaa` / `aa` → `[0,1,2]`，再试空串、模式更长、完全不匹配。

代码在 [deeper_patterns.py](../code/deeper_patterns.py)，测试用小字符串逐一枚举，与直接切片比较的答案核对。进一步阅读：[Princeton 字符串查找讲义](https://algs4.cs.princeton.edu/53substring/)；其中 DFA 写法与本文的前缀表写法表示同一种失败后复用思路。
