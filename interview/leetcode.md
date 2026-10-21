# LeetCode：把方法练熟，不只记题

**中文** · [English](leetcode.en.md)

> 阅读时间：约 5 分钟 · 最近审阅：2026-10

有些题看答案时觉得“这不难啊”，隔几天再写，还是不知道从哪里下手。这组笔记想补的就是中间这一步：为什么想到这个方法，代码里的变量在记什么，换个条件以后还能不能用。

选题参考 [Top Interview 150](https://leetcode.com/studyplan/top-interview-150/)，主要讲 Easy / Medium 中经常能用到的方法，不按 150 道题逐题整理答案，Hard 也暂时不展开。这里的链接都是公开练习，不是公司原题合集。

## 1. 按方法走，不按题号走

刚开始刷题，可以从前两篇看起：先熟悉 Python 工具，再练“把见过的信息存起来”。已经刷过一些，就直接找自己卡住的题型。每篇先拿小例子把过程走清楚，代码和复杂度放在后面，需要时再往下看。

| 顺序 | 小节 | 主要讲什么 |
| --- | --- | --- |
| 1 | [复杂度与 Python 工具](algorithms/complexity-and-tools.md) | average / amortized、递归栈、dict / deque / heapq 怎么选 |
| 2 | [Hash map 与前缀和](algorithms/hash-and-prefix.md) | key 存什么、value 存什么，为什么有时要存频次 |
| 3 | [双指针、窗口与二分](algorithms/pointers-and-search.md) | 每次移动，凭什么能排除一部分答案 |
| 4 | [DFS / BFS：递归还是迭代](algorithms/traversal.md) | 访问顺序、栈与队列、visited、最短路的前提 |
| 5 | [栈、堆与链表](algorithms/stack-heap-links.md) | 等一个未来答案、维护 Top K、安全地改指针 |
| 6 | [回溯与动态规划](algorithms/backtracking-and-dp.md) | 路径和状态怎么区分，记忆化和循环怎么互换 |
| 7 | [排序、区间与贪心](algorithms/sorting-and-greedy.md) | 排序省掉了什么，局部选择为什么不会后悔 |

第一遍不用把所有方法一起背。基础模板熟悉后，按自己卡住的问题往下读：

| 我卡在这里 | 接着看 | 这次多想一步 |
| --- | --- | --- |
| 字符串匹配总是从头重来 | [字符串匹配 · KMP](algorithms/string-matching.md) | 失败后哪些已匹配信息还能复用 |
| 窗口最大值离开后不知道谁接班 | [单调队列](algorithms/monotonic-queue.md) | 一个候选为什么可以永久删除 |
| BST 只会写递归模板 | [二叉搜索树](algorithms/binary-search-trees.md) | 祖先范围、中序排名与树高 |
| DP 的循环方向老是反 | [背包与凑钱](algorithms/knapsack.md) | 读上一层，还是允许复用当前层 |
| LCS、LIS 与子串搞混 | [序列 DP](algorithms/sequence-dp.md) | 前缀、结尾、连续，分别限制什么 |
| 只存一个最优值就丢了规则 | [状态机 DP](algorithms/state-machine-dp.md) | 哪段历史会改变后续可选动作 |

还有一篇按需看的[质数判断与筛法](algorithms/primes-and-sieves.md)：一个数用试除，一段数用筛表，重点是平方根边界和预处理何时值得。它不在第一轮必刷清单里。

语法不熟，先看朋友整理的 [Python 语法](python.md)。这里更关心“这行代码在算法里起什么作用”，两篇可以来回查。

## 2. 读完题，先问 5 个问题

1. **输入有什么保证？** 已排序？可能有负数？有重复？图里有环吗？
2. **输出是什么？** 存在与否、一个答案、最优值，还是所有答案？
3. **重复做的事是什么？** 每次从头找一个数，还是每次重新算一段和？
4. **要留下什么信息？** 下标、频次、窗口、路径，还是某个状态的最优结果？
5. **能扔掉什么？** 为什么扔掉后不会漏解？这就是不变量和正确性的起点。

| 题目里的信号 | 优先考虑 | 先确认的前提 |
| --- | --- | --- |
| “见过吗 / 几次 / 哪个位置” | set / dict / Counter | 要不要保留重复次数、原下标 |
| 连续子数组或子串 | 窗口 / 前缀和 | 有负数时，和的单调性可能消失 |
| 有序，或“从这里开始都可行” | 二分 | 能否写出单调判定 |
| 连通、层数、最少几步 | DFS / BFS | 普通 BFS 的边代价需相同 |
| 列出全部组合 | 回溯 | 输出本身可能指数级 |
| 很多路径通往相同子问题 | DP / memoization | 状态是否包含所有影响未来的信息 |
| 只关心最大的几个 | 小顶堆 | 是否真的需要全部排序 |

表里的提示只能帮你缩小范围，不能看到“最短”两个字就决定用 BFS。先看状态怎么表示、每次转移的代价是否相同。例如，最少硬币也能把金额当作节点、加一枚硬币当作一条边，用 BFS 求最少步数；也可以用 DP 复用子问题。再根据金额范围、硬币取值和空间开销选实现。

## 3. 一轮怎么复习

- **先讲再写**：用一两句话说暴力法，再说准备省掉哪部分重复工作。
- **手走 4～6 个元素**：把变量写下来，别只在脑子里觉得“应该对”。
- **关掉答案再写一次**：写完自己挑空输入、重复值、极端顺序试一试。
- **隔几天换个条件**：有序变无序、正数加负数、树变带环图。说说哪里会失效。

不用追求刷了多少题。隔几天还能写出来、能解释换个条件后哪里要改，就比记住题号更有用。

如果面试用英语，平时也留一点时间用英语讲。中文正文会在关键术语旁标出英文。看懂一个例子后，试着不看笔记，用英语讲清“状态存什么、下一步怎么走、为什么不会漏、复杂度是多少”。几篇里也放了简短的英文讲法，可以先对照，再换成自己的话；需要整篇英文时，点页面上的 English 就能切换。

## 4. 代码可以直接跑

核心模板在 [patterns.py](code/patterns.py) 和 [deeper_patterns.py](code/deeper_patterns.py)，用 Python 3.10+ 就能运行，不依赖第三方库。要连同下面的测试一起跑，请用 **Python 3.11+**：测试会读取 TOML 配置，用到 3.11 才加入标准库的 [tomllib](https://docs.python.org/3.11/library/tomllib.html)。在仓库根目录运行：

```bash
python interview/code/patterns.py
python -m unittest discover -s site/tests -p 'test_algorithm_patterns.py'
python interview/code/deeper_patterns.py
python -m unittest discover -s site/tests -p 'test_deeper_algorithm_patterns.py'
```

测试会拿小输入枚举答案，检查窗口、前缀和、DP 等模板；也会核对正文代码和测试文件是否一致。样例用于学习方法，不宣称覆盖 LeetCode 所有输入规格。提交时注意题目要求的 `Solution`、下标从 0 还是 1 开始，以及是否允许修改输入。

## 5. 今天先复习哪块

- 只有 20 分钟：Hash map 的 Two Sum → `get` / `in` 的区别 → 关掉答案写一次。
- 总搞混 DFS / BFS：打开[遍历小实验](algorithms/traversal.md)，先预测下一个节点，再点下一步。
- 模板会背但一变就不会：看每篇的“换个条件”，找反例，不急着加题量。

不必每次都刷新题。把一道旧题里没想明白的地方补上，也算一次有效的复习。算法只是准备的一部分，给项目和休息也留点时间。

## 6. 还有哪些中文讲解值得对照

同一个概念，换个人讲，可能就通了。下面几份资料各有侧重，不需要全看完；按眼下的问题选一篇就好。

| 你现在想弄明白什么 | 可以接着看 |
| --- | --- |
| 栈、队列、树到底是怎样一步步变化的 | [Hello 算法](https://www.hello-algo.com/)：先看对应章节的图，再回来看代码 |
| 会写模板，但不知道为什么这样写 | [灵茶山艾府的基础算法课](https://github.com/EndlessCheng/codeforces-go/blob/master/leetcode/README.md)：跟着双指针、二分、DP 等专题推一遍；题单里也有进阶题，不用全做 |
| 几道题很像，却不知道怎么放在一起理解 | [labuladong 的算法讲解](https://labuladong.online/zh/algo/intro/core-intro/)：可以对照同一种方法在不同题目里怎么改；部分内容和工具有付费入口 |
| DP 写出来了，但结果不对，不知道从哪查 | [代码随想录的 DP 基础](https://github.com/youngyangyang04/leetcode-master/blob/master/problems/%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%E7%90%86%E8%AE%BA%E5%9F%BA%E7%A1%80.md)：逐项核对状态、初值、递推和遍历顺序，再手算一个小例子 |

本站的例子和代码会自己推一遍、跑测试，也会保留参考链接。不同资料用的变量名、区间和状态定义可能不同，借鉴思路时别把几套写法直接拼在一起。如果哪篇还是没讲清楚，欢迎指出具体卡在哪一步。
