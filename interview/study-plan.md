# 按计划学习

**中文** · [English](study-plan.en.md)

想系统学一遍，可以先用一个小练习判断从哪里开始，再把阅读和动手安排在一起。这页介绍怎么选内容、安排时间，以及保存记录，方便下次接着学。

[面试准备首页](README.md) · [快速复习](speedrun.md)

## 先确定从哪里开始 {#build}

如果 ML 接触不多，从 [起步路线的一次预测](speedrun-start.md#prediction) 开始。算完后问自己：输入是什么，模型给出什么，loss 又比较了什么？

- 看不懂列表和函数：先做 [Python 小例子](speedrun-start.md#first-two)，再回来。
- 会写代码，但没学过训练：跳过语法，从预测与 loss 接到 [一次参数更新](speedrun-start.md#update)。
- 这些都很熟：直接做下面的[自查练习](#refresh)。

按主题选起点就好。比如做过系统开发、但没接触过模型训练，可以跳过 Python，从预测和 loss 开始。

## 已有基础，用练习查漏补缺 {#refresh}

挑一个和目标岗位有关的小任务，先不查答案。做不出来时，分清是原理没懂、代码写不出，还是不会检查结果，再回看对应章节。

| 试一下 | 哪里卡住，就从哪里补 |
| --- | --- |
| 写出一个 batch 的平均 loss，换一种分批方式再算 | [聚合与分母](code-comprehension.md#aggregation)：你到底在平均什么 |
| 写一个 causal attention，标出各步 shape | [ML Coding](../learn/ml-exercises/README.md)：shape、mask 和测试 |
| 用一个小图说明 DFS 和 BFS 的访问顺序 | [遍历方法](algorithms/traversal.md)：栈、队列和 visited |
| 画出一条 LLM 请求从入队到返回 token 的路径 | [推理框架拆解](../06-systems/inference-frameworks.md)：调度、缓存与执行 |
| 讲一个项目，说明为什么没选另一个方案 | [项目准备](../career/prepare.md)：约束、个人贡献和证据 |

能独立完成的部分可以跳过。只记得结论时，就用一个具体输入验证它，看看自己能否解释中间每一步。

## 每次安排一个练习 {#sessions}

可以先留 **30–60 分钟有效学习时间**，用于阅读、写代码或复习，休息不计入。这个时长是一次安排的参考，不代表一个单元一定能在一小时内学完。

日期自己定，每次时长也可以不同，比如周二 25 分钟、周六 70 分钟。不用每天学，也不要求连续打卡，用自己习惯的日历或备忘录排就行。

| 顺序 | 先读一点 | 这次动手做什么 | 留下什么，方便下次继续 |
| --- | --- | --- | --- |
| 1 · 输入与预测 | [一次预测](speedrun-start.md#prediction) | 手算两个样本的预测和 loss，再用 Python 核对 | 一个可运行的函数，写明输入和预期输出 |
| 2 · 更新参数 | [一次更新](speedrun-start.md#update) | 改一次学习率，观察 loss；解释步子太大时发生什么 | 参数更新前后的数值和一个疑问 |
| 3 · 写成训练循环 | [张量与 PyTorch](../00-foundations/pytorch/README.md) → [训练循环](../00-foundations/pytorch/training-loop.md) | 用一个小 batch 跑前向、反向和更新；区分 train / eval | 一个能重复跑的脚本和一次检查结果 |
| 4 · 练一个算法方法 | [Hash Map](algorithms/hash-and-prefix.md) 或 [DFS / BFS](algorithms/traversal.md) | 自己写一次，补重复值或成环的例子，解释复杂度 | 一个边界测试，以及方法适用的条件 |
| 5 · 接到目标方向 | [按岗位选练习](ai-roles-prep.md) | 从下面选一条分支，只做其中一个练习 | 一个实现、实验记录或设计草图 |
| 6 · 合起来讲一次 | [代码理解](code-comprehension.md) 与 [项目准备](../career/prepare.md) | 给代码改一个要求；用一个项目解释自己做过的选择 | 这次没讲清的两处，下次从其中一处继续 |

1–3 适合刚接触 ML 的读者；算法也可以单独并行练，不必等模型部分全部学完。每个单元都可以拆开，或者在已经会做时跳过。

## 按岗位继续准备 {#roles}

各方向先练什么、往下练什么，见[按岗位选练习](ai-roles-prep.md#routes)的表。下面只给每个方向一个接着做的练习：

- <span id="mle-research-engineer"></span>**MLE / Research Engineer**：从 [ML 手写练习](../learn/ml-exercises/README.md) 里选一个实现，例如稳定的 softmax 或 attention；先做小输入，再检查输出、梯度和边界。做 LLM 接着看[后训练实践](../practice/post-training/README.md)，做推荐与检索选[推荐系统拆解](../practice/recommender-systems/README.md)。
- <span id="inference-ml-systems"></span>**Inference / ML Systems**：沿[一条推理请求](../06-systems/inference-frameworks.md)看每个模块的输入和输出，再挑一个约束：显存不够、排队太久，或者长请求拖慢短请求。写下两种方案，说明各自改变哪一段、还需要测什么。做[系统设计练习](../learn/system-design/README.md)时，先定 workload 与延迟目标，再谈组件。
- <span id="coding-swe"></span>**通用 Coding / SWE**：先把 [Python 与 LeetCode](leetcode.md) 和[读代码](code-comprehension.md)放进安排，每次练一个方法，再改一个条件看看还能不能用。本站主要从 ML 与 AI 系统的角度组织内容，具体要补哪些语言、后端或系统知识，看岗位和面试说明。
- <span id="research-scientist"></span>**Research Scientist**：挑一篇自己做过或认真读过的论文，讲清问题、对照实验和结论的范围，再提出一个能区分两种解释的后续实验。[评估基础](../07-evaluation/README.md)可以帮你检查“指标涨了”是否支持自己的解释。相关实现也要练，研究方向的专业积累这条路线替代不了。

不管哪个方向，算法和读代码都要留时间，不能只练架构图。

## 按方向挑一组练习 {#routes}

四个方向，每个方向两组，每组三个练习。「起步」适合刚接触这个方向，「复习」适合学过、想检查自己能不能独立做对。时间只是一次练习的估计，可以拆开做。

### ML 基础与 coding

**起步 · 从 shape 到 attention**：会写一点 Python，但还没独立写过模型。

| 练习 | 先读 | 怎么做 | 时间 |
| --- | --- | --- | --- |
| 画出一次 broadcasting | [运算与维度：代码能跑，不代表算对了](../00-foundations/pytorch/operations-and-shapes.md) | 先手算两个张量相加的 shape，再用三个小数组验证。 | 25 分钟 |
| 走完一次 attention | [多头注意力：从公式到实现](../00-foundations/core/multi-head-attention.md) | 用两三个 token 标出 Q、K、V 和输出的形状，不急着写完整模型。 | 40 分钟 |
| 写一个小 kNN | [五组短练习：从会算到写对](coding-checks.md) | 先用循环算距离，再做向量化；检查重复点和平局。 | 30 分钟 |

**复习 · 实现、排错、补测试**：学过模型原理，想检查自己能不能独立写对。

| 练习 | 先读 | 怎么做 | 时间 |
| --- | --- | --- | --- |
| 不看答案写 attention | [多头注意力：从公式到实现](../00-foundations/core/multi-head-attention.md) | 标清 shape，加 causal mask，再验证未来 token 不会影响前面的输出。 | 45 分钟 |
| 找出能运行的 shape 错误 | [五组短练习：从会算到写对](coding-checks.md) | 比较 [B,1] 和 [B] 的相减结果，写一个会抓住错配的测试。 | 25 分钟 |
| 修一次评估聚合 | [代码理解：读懂、改对，还要能验证](code-comprehension.md) | 先读懂输入和输出，再用不同 batch size 检查汇总结果。 | 35 分钟 |

### LLM 后训练

**起步 · 先看清训练信号**：知道模型在预测 token，想弄懂后训练具体改了什么。

| 练习 | 先读 | 怎么做 | 时间 |
| --- | --- | --- | --- |
| 给一条对话标 labels | [SFT：一条对话怎样变成训练信号](../05-post-training/sft-and-its-ceiling.md) | 写出 user 和 assistant 的 token、shift 后的 labels，以及哪些位置算 loss。 | 30 分钟 |
| 分清三种更新 | [Q-learning、PPO、DPO：到底在学什么？](../05-post-training/deep-rl/methods-map.md) | 分别说清 Q-learning、PPO、DPO 需要什么数据、改什么量。 | 30 分钟 |
| 检查一组训练样本 | [为什么后训练总在改数据？](../01-data-and-feedback/data-as-supervision.md) | 挑几个样本，写下它们在奖励什么行为，以及什么行为还测不到。 | 25 分钟 |

**复习 · 从目标到训练检查**：做过训练，想把数据、目标和工程选择连起来。

| 练习 | 先读 | 怎么做 | 时间 |
| --- | --- | --- | --- |
| 审核 mask 与 shift | [SFT：一条对话怎样变成训练信号](../05-post-training/sft-and-its-ceiling.md) | 用一条多轮对话验证 loss mask、结束符和 label shift，写出错误的影响。 | 30 分钟 |
| 讲清 reference 和 old policy | [Q-learning、PPO、DPO：到底在学什么？](../05-post-training/deep-rl/methods-map.md) | 分别解释它们的用途、什么时候更新，以及混用会改掉什么目标。 | 35 分钟 |
| 选一次训练并行方案 | [训练框架怎么选：从 DDP 到 FSDP2](../06-systems/training-frameworks.md) | 给定显存和卡数，比较 DDP 与 FSDP，并列出换框架后的正确性检查。 | 40 分钟 |

<span id="ml"></span>

### 推理与 ML 系统 {#inference-ml-systems_1}

**起步 · 一条请求怎么跑起来**：会调用模型 API，想看懂服务背后发生了什么。

| 练习 | 先读 | 怎么做 | 时间 |
| --- | --- | --- | --- |
| 画出 prefill 和 decode | [KV cache：少算了什么，又多存了什么？](../00-foundations/deep-dives/kv-cache-and-inference.md) | 用一小段 prompt，画出第一次计算和后续每个 token 用到的缓存。 | 30 分钟 |
| 追踪一条生成请求 | [vLLM 和 SGLang：生成引擎在忙什么？](../06-systems/inference-frameworks.md) | 从入口画到 scheduler、KV cache 和模型执行；每层写一句职责。 | 35 分钟 |
| 先分清训练和推理 | [训练框架怎么选：从 DDP 到 FSDP2](../06-systems/training-frameworks.md) | 比较 DDP、FSDP、vLLM 各自解决什么问题，别把它们当同一层的替代品。 | 30 分钟 |

**复习 · 从架构追到瓶颈**：接触过部署，想练系统拆解和取舍。

| 练习 | 先读 | 怎么做 | 时间 |
| --- | --- | --- | --- |
| 拆解 vLLM 与 SGLang | [vLLM 和 SGLang：生成引擎在忙什么？](../06-systems/inference-frameworks.md) | 用同一个请求比较调度和缓存，再挑一个差异深入解释。 | 40 分钟 |
| 估一次 KV cache 显存 | [KV cache：少算了什么，又多存了什么？](../00-foundations/deep-dives/kv-cache-and-inference.md) | 写明层数、KV heads、长度、batch 和精度，检查 GQA 如何改变估算。 | 30 分钟 |
| 设计一个有意义的压测 | [vLLM 和 SGLang：生成引擎在忙什么？](../06-systems/inference-frameworks.md) | 固定输入输出长度分布，区分 TTFT、decode 延迟和吞吐，再说怎么定位瓶颈。 | 35 分钟 |

### 算法与数据结构

**起步 · 练能迁移的方法**：刚开始刷题，先掌握几种常见结构。

| 练习 | 先读 | 怎么做 | 时间 |
| --- | --- | --- | --- |
| 想清 hash map 存什么 | [Hash map 与前缀和：到底要记住什么](algorithms/hash-and-prefix.md) | 用 Two Sum 解释为什么先查再存，再加一个重复元素的测试。 | 25 分钟 |
| 写清二分的边界 | [双指针、滑动窗口、二分：为什么能往这边走](algorithms/pointers-and-search.md) | 找第一个 ≥ target 的位置；测空数组、重复值和找不到的情况。 | 30 分钟 |
| 同一张图走 DFS 和 BFS | [DFS / BFS：递归和迭代到底差在哪](algorithms/traversal.md) | 画出栈和队列的变化，解释什么时候标记 visited。 | 30 分钟 |

**复习 · 换条件，再验一遍**：常见题型做过，想检查边界、复杂度和迁移能力。

| 练习 | 先读 | 怎么做 | 时间 |
| --- | --- | --- | --- |
| 用前缀和数子数组 | [Hash map 与前缀和：到底要记住什么](algorithms/hash-and-prefix.md) | 写出 map 的含义，检查负数、零和重复前缀；说明为什么不是滑动窗口。 | 30 分钟 |
| 把递归 DFS 改成迭代 | [DFS / BFS：递归和迭代到底差在哪](algorithms/traversal.md) | 比较访问顺序、状态保存和额外空间，再用带环图验证。 | 30 分钟 |
| 排一组有依赖的任务 | [五组短练习：从会算到写对](coding-checks.md) | 输出拓扑顺序并检测环，区分依赖层和真正的并行调度。 | 35 分钟 |

## 留下记录，下次继续 {#resume}

结束前留三行就够了：**这次做了什么、哪里还不清楚、下次先试什么**。比如：

> causal mask 能跑了，但还没测 padding。下次构造两条长度不同的序列，先手写预期 mask。

这三行写在自己的代码或笔记里就行。想之后再读的文章，点标题下的「收藏本页」；右上角「我的阅读」里能找到收藏和最近读过的文章，再打开时会回到上次读到的位置。

隔了一段时间再回来，先跑一下上次的例子。还能解释清楚，就往下做；忘了哪一步，再回头补。

## 备份与恢复 {#backups}

收藏、阅读记录和阅读位置都存在当前浏览器里。换电脑、换浏览器，或清理网站数据前，先导出一份：

1. 打开右上角「我的阅读」→「设置」，在「备份」一行点「导出」，会下载一个 JSON 文件。
2. 在另一台设备上打开同一个位置，点「导入」，选这个文件。
3. 两边的记录会合并：同一条记录保留较新的一份，已有的记录不会被删掉。

- **网站更新不会清空记录。** 记录存在浏览器里，不在源码中，不需要 fork 或 clone；正常更新网页、`git pull` 都不影响。但换了域名、协议或本地端口，需要导入备份。
- **浏览器数据也可能丢。** 清理网站数据、关闭无痕窗口，或浏览器自动清理存储，都可能删掉记录。备份文件另存一份更稳妥，比如放进自己的 iCloud / Drive 文件夹；本站不连接云盘，也不自动同步设备。
- **导入不会上传文件。** 处理都在本地完成。深色/浅色主题不在备份里；「最近读过」只记录本站页面，不读取浏览器历史。
- **以前的备份也能导入。** 旧版学习台导出的文件、以前只含收藏的文件都可以导入。旧文件里的学习计划和备注会原样保留在备份里，只是网站上不再显示。
- **章节改名不影响记录。** 记录按固定 ID 匹配，不按标题、顺序或语言匹配。暂时找不到的章节也会保留记录。

[回到面试准备首页](README.md) · [只练某一项](README.md#prep-routes) · [完整知识库](../learn/README.md)
