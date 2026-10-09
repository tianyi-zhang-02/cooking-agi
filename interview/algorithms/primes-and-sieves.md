# 质数判断与筛法：查一个数，还是查一整段？

**中文** · [English](primes-and-sieves.en.md)

> 阅读时间：约 10 分钟 · 前置知识：整除、循环、数组 · 最近审阅：2026-10-09

判断 97 是不是质数，和找出 100 万以内的所有质数，看起来只是数据多了，实际值得用两种办法。前者问“谁能整除它”，后者更适合一次划掉很多合数。

这篇是算法复习的补充，不是学习 ML 的前置要求。先看第 1–3 节，把边界和证明讲清楚；需要算复杂度或处理多次查询，再往后读。代码都只用 Python 标准库。

## 1. 为什么试到平方根就够了？ {#trial-division}

质数是大于 1、正因子只有 1 和自身的整数。0、1 和负数都不是质数，2 是唯一的偶质数。

如果 $n$ 是合数，可以写成 $n=ab$，其中 $1<a\le b$。于是 $a^2\le ab=n$，所以 $a\le\sqrt n$。换句话说：**如果一个非平凡因子都没在平方根以内找到，更大的因子也不可能凭空出现，它得有一个较小的搭档。**

这不是只对偶数成立。以 91 为例，试除 2、3、5、7，到 7 时找到 $91=7\times13$；不用继续试到 90。以 49 为例，必须试到 7 本身，不能把平方根漏掉。

```python
def is_prime(number):
    if type(number) is not int:
        raise TypeError("number must be an integer")
    if number < 2:
        return False
    if number == 2:
        return True
    if number % 2 == 0:
        return False
    for divisor in range(3, isqrt(number) + 1, 2):
        if number % divisor == 0:
            return False
    return True
```

文件顶部需要 `from math import isqrt`。[`math.isqrt`](https://docs.python.org/3.12/library/math.html#math.isqrt)返回整数平方根向下取整的精确值，不用浮点数近似。Python 的 `range` 右端不包含，所以写 `+ 1`。先排除 2 的倍数后，只试奇数就不会漏解；奇数里虽有 9、15 这样的合数，试它们只会多做工作，不影响正确性。

实现严格接收 Python `int`，不把 `True` 当作 1，也不把 `7.0` 偷偷转成 7。对一个数，最坏需要 $O(\sqrt n)$ 次试除，额外保存常数个整数。这里暂按整数运算为常数成本；大整数的位运算代价留到第 5 节。

## 2. 要查一整段，就把重复工作合起来 {#sieve}

如果对 2 到 30 每个数都调用一次 `is_prime`，2 的倍数会被一遍遍判断。埃拉托斯特尼筛法（Sieve of Eratosthenes）反过来做：保留一个标记数组，遇到质数就划掉它的后续倍数。

| 当前走到 | 要做的事 | 为什么 |
| --- | --- | --- |
| 2 | 划掉 4、6、8、…、30 | 它们都有因子 2 |
| 3 | 从 9 开始划掉 9、12、15、…、30 | 6 已被 2 划掉；重复划掉 12 没关系 |
| 4 | 跳过 | 已知不是质数，不必再筛一遍 |
| 5 | 从 25 开始，划掉 25、30 | 更小的倍数已有较小的质因子 |
| 超过 $\sqrt{30}$ | 停止筛选，读取剩下的标记 | 每个合数都应有一个不大于平方根的质因子 |

最后留下 `2, 3, 5, 7, 11, 13, 17, 19, 23, 29`。可以跟着表在纸上划一遍，看看每一步新排除了哪些数。

```python
def prime_flags(limit):
    if type(limit) is not int:
        raise TypeError("limit must be an integer")
    if limit < 0:
        raise ValueError("limit must be non-negative")
    flags = bytearray([1]) * (limit + 1)
    flags[0] = 0
    if limit >= 1:
        flags[1] = 0
    for prime in range(2, isqrt(limit) + 1):
        if flags[prime]:
            for multiple in range(prime * prime, limit + 1, prime):
                flags[multiple] = 0
    return flags
```

约定是返回 **0 到 `limit`，包含右端点**的标记；`flags[number] == 1` 表示质数。`bytearray` 每个位置占 1 字节，不是压到 1 bit 的 bitset，但比用 Python 整数对象表达状态更直接。

代码选择逐个赋值，方便对应筛选过程。大规模时可以考虑切片、只存奇数或位数组，不过要同时计算临时分配，不能只看最短的代码。筛法的经典实现也可以对照 [Princeton 的课程代码](https://introcs.cs.princeton.edu/java/14array/PrimeSieve.java.html)；本篇接口明确包含右端点，不能不看边界就替换进题目。

## 3. 为什么从平方开始？为什么剩下的都是质数？

假设现在轮到质数 $p$。$2p,3p,\ldots,(p-1)p$ 都有一个小于 $p$ 的质因子，之前已经处理过。所以可以从 $p^2$ 开始。以 7 为例：14、21、28、35、42 已分别有更小的质因子，49 才是第一次必须靠 7 划掉的数。

可以用这个不变量（invariant）证明外层循环：**处理 $p$ 之前，所有最小质因子小于 $p$ 的合数已经被标成 0。**

1. 如果 `flags[p]` 仍为 1，$p$ 不可能是合数。否则它有一个小于 $p$ 的质因子，早该被划掉。
2. 处理 $p$ 时，把最小质因子为 $p$ 的合数划掉。它们都不小于 $p^2$，所以没有漏掉。
3. 当 $p$ 走过 $\sqrt N$，任何不超过 $N$ 的合数都已经有机会被最小质因子划掉。真正的质数从未被划，因为我们只标记具有两个大于 1 的因子的数。

这也解释了两个典型错误：从 `p` 开始会把质数本身划掉；外层漏掉平方根，会把 49 之类的完全平方数误留在表里。

## 4. 复杂度不只是“有两层循环”

逐个做试除，最坏时间上界为：

$$
\sum_{n=2}^{N}O(\sqrt n)=O(N^{3/2}).
$$

筛法不是每个外层数都扫描整个数组。内层只对质数 $p$ 启动，大约写 $N/p$ 次。总标记量可以上界为：

$$
\sum_{\substack{p\le\sqrt N\\p\text{ is prime}}}\frac{N}{p}
=N\sum_{p\le\sqrt N}\frac1p.
$$

用质数倒数和增长为 $\log\log x+O(1)$ 这个数论结果，得到 $O(N\log\log N)$ 时间；数组初始化另需 $O(N)$。**两层循环本身推不出这个结果**。如果暂时不使用质数倒数和的结论，用所有整数的调和和上界，也能正确得到较松的 $O(N\log N)$。这里不展开该数论定理的证明；筛法的这个时间界也可对照 [CMU 的算法分析](https://www.cs.cmu.edu/~scandal/cacm/node8.html)。

| 需求 | 合适的起点 | 时间与空间 |
| --- | --- | --- |
| 判断单个中小整数 | 试除 | 最坏 $O(\sqrt n)$ 次除法，常数个额外整数 |
| 列出 $N$ 以内全部质数 | 筛法 | $O(N\log\log N)$ 时间，$O(N)$ 字节标记 |
| 同一上界内反复问是否为质数 | 一次筛表，之后查表 | 预处理同上，之后每次 $O(1)$ 查下标 |
| 只查很高位置的一小段 | 考虑分段筛 | 不必保存从 0 到整个上界的表；仍要基础质数 |

如果真要输出所有质数，返回的列表还有自己的空间成本。只要数量，可以 `sum(flags)`；如果还要多次查询，不要为了省这一行代码把标记表丢掉。

## 5. 边界、分段筛和大整数

先检查小边界：`prime_flags(0)` 返回 `[0]` 的字节数组，`prime_flags(1)` 是 `[0, 0]`，`prime_flags(2)` 才会把 2 留下。若题目问“严格小于 $n$ 的质数个数”，$n\le2$ 返回 0，否则统计 `prime_flags(n - 1)`；不是 `prime_flags(n)`。

分段筛（segmented sieve）适合只关心闭区间 $[L,R]$ 的情况。先筛出 $\sqrt R$ 以内的基础质数，再对每个 $p$ 从下面这个位置划：

$$
\max\left(p^2,\left\lceil\frac{L}{p}\right\rceil p\right).
$$

整数代码可以写成 `max(prime * prime, ((left + prime - 1) // prime) * prime)`，这里限定 $0\le L\le R$。例如查 `[20, 30]`，$p=3$ 从 21 开始；$p=5$ 从 25 开始，不能误删 5 本身。区间包含 0 或 1 时还要单独置零。存储基础筛表和区间标记需要 $O(\sqrt R+R-L+1)$ 空间；不是只有区间宽度。

“Python 整数不会像固定宽度整数那样溢出”不等于运算免费。一个 $B$ bit 的整数，$\sqrt n$ 量级的试除次数随 $B$ 指数增长，每次大整数运算也有成本。密码学大小的质数测试不该直接用这篇代码；需要适合规模的算法和经过验证的实现。本篇不实现 Miller–Rabin，也不把概率性检验写成无条件确定证明。

## 6. 再多一步：顺手分解质因子

判断质数一旦遇到因子就返回；分解则继续除，并把剩余部分缩小。以 84 为例：

```text
84 → 除以 2 → 42 → 再除以 2 → 21
21 → 除以 3 → 7
剩下的 7 记入结果：[2, 2, 3, 7]
```

为什么剩下的 7 可以直接保留？当当前试除数的平方已经超过剩余值，剩余值若大于 1，就不能再是合数，否则应该还有一个没处理的小因子。这里循环比较的是**不断缩小的剩余值**，不是原始 84。

[number_theory.py](../code/number_theory.py)包含 `is_prime`、`prime_flags` 和 `prime_factors`。`prime_factors(1)` 返回空列表，0 和负数则报错；负数的符号分解不在这个接口里。最坏时间仍是 $O(\sqrt n)$ 次试除，另存输出因子。可从仓库根目录运行：

```bash
python3 interview/code/number_theory.py
```

```text
Primes through 30: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
49 is prime: False
Factors of 84: [2, 2, 3, 7]
```

复习时试着解释：为什么两种算法都出现平方根？前者利用因子成对出现，后者把这个结论用于“哪些合数一定已被划掉”。再用 1、2、49 检查边界，用 30 手走筛法。想做一道公开练习，可以看 [Count Primes](https://leetcode.com/problems/count-primes/)，注意它不包含右端点。

接回[算法总览](../leetcode.md)；需要区分预处理代价、单次查询和输出空间，可以回看[复杂度与 Python 工具](complexity-and-tools.md)。
