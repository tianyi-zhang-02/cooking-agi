# 为什么 RL 还是难训：信号、成本与稳定性

**中文** · [English](why-rl-is-hard.en.md)

> 阅读时间：约 13–17 分钟 · 最近审阅：2026-10 · 系统文档核对：2026-10-11

训练脚本跑通了，GPU 也一直在忙，reward 却不怎么涨；换一个采样温度，曲线又完全不同。做 RL 时，这几件事可以同时发生，并不矛盾。

RL 不是新概念，也不是完全没有共同框架。**采样、拿反馈、估计更新方向、改策略**这套思路很成熟。难的是：换一个模型、任务、奖励或预算，同一套配置未必还好用。算法有框架，可靠的配方却不能直接照搬。

这篇主要讲 LLM 后训练。先[分清效率](#efficiency)，再[算一轮成本](#rollout-budget)，最后看[排错顺序](#diagnosis)。经典算法的关系见[方法地图](methods-map.md)；历史沿革见[RL 发展史](../../learn/rl-timeline.md)。这里的预算与时间均为教学假设，不是某个框架的实测结果。

## 和 SFT 相比，多出来的困难在哪里 {#moving-data}

固定数据上的 SFT 也会很难，但至少这轮更新通常不会改变下一批数据的内容。在线 RL 不一样：更新策略以后，模型会去到不同的状态、写出不同的回答，接着又用这些新数据训练。

例如一个代码模型经常过不了编译，你先修这一类错误很合理。等编译基本通过，剩下的可能是边界条件和隐藏测试。继续喂同样的简单任务，采样花了钱，能区分回答好坏的反馈却越来越少。难度分布需要跟着能力变化，但不能为了保持训练 reward 好看，只留下容易题。

<figure class="worked-update" aria-label="一次更新如何改变下一轮数据">
<ol>
<li><small>本轮策略</small><strong>生成代码，再运行测试</strong><span>记录回答、采样概率、失败类型和花费。不是只留一个总分。</span></li>
<li><small>本轮更新</small><strong>哪些尝试值得增加概率？</strong><span>标签有噪声、advantage 有误差，更新本身也可能过大。</span></li>
<li><small>下一轮采样</small><strong>遇到的错误又变了</strong><span>题目不变，回答分布仍会变。再检查难度、覆盖和独立评估。</span></li>
</ol>
<figcaption>这个循环让模型有机会学到示范之外的行为，也让训练分布、估计误差和系统成本互相影响。</figcaption>
</figure>

因此，“造数据和写 eval 很重要”是对的，但不是把它们做好，RL 就自然稳定了。数据决定练什么，反馈决定奖励什么；探索、信用分配和优化决定模型能否利用这些反馈。训练与推理系统则决定我们实际执行的是不是想要的算法。

## Token efficiency 先说清分母 {#efficiency}

“更省 token”至少可能指四件不同的事，不能混着比较。

| 你想省什么 | 应该比较什么 | 常见误判 |
| --- | --- | --- |
| 交互样本：sample efficiency | 到达同一固定 eval 目标所需的环境交互、尝试或标注数 | 少做 optimizer step，不等于少交互 |
| 训练 token | 同一质量目标下，生成、打分、训练分别处理了多少 token | 只算进入 loss 的 token，漏掉被丢弃的 rollout |
| 使用时的 token | 同一测试集与质量要求下，总生成 token、采样次数、工具调用 | 每条回答变短，但需要更多次尝试才能成功 |
| 计算与时间 | GPU-hours、成本、wall-clock，以及达到目标的时间 | tokens/s 上升，最终质量或总预算反而变差 |

同样一个 token，在 prefill、逐步 decode、reward model 前向和反向传播中的成本不同；decode 的成本怎样随 batch 和 KV cache 变化，见[性能：roofline 与 MFU](../../12-infra/performance.md#decode-batching)。比较不同长度和不同模型时，也不能把 token 数直接换成 FLOPs。

一个实用的记录方式是：固定评估题、采样预算和判分器，然后画 **质量—累计 rollout token**、**质量—GPU-hours** 两条曲线，再分别报告输出长度与成功率。它比单看 reward 或 GPU utilization 更接近你真正想知道的事。

## 算一笔 rollout 账 {#rollout-budget}

假设一批有 64 个问题，每题采样 8 个回答，每个回答恰好生成 2,048 token：

$$
N_{\mathrm{generated}}=64\times8\times2048=1{,}048{,}576.
$$

如果筛选后只保留四分之一的组，那么训练保留了 262,144 个输出 token，另有 786,432 个没有进入这次训练。但它们**已经生成过了**，还可能跑过 verifier。这里只数输出 token，尚未计入 prompt prefill、评分、reference 前向、反向和重试。

<figure class="rl-budget" aria-label="教学预算：生成的输出中四分之一被保留">
<div class="rl-budget__bar"><span class="rl-budget__kept">25%</span><span class="rl-budget__discarded">75%</span></div>
<figcaption><strong>保留 262,144 · 未保留 786,432</strong><br>颜色只表示 token 去向，不表示梯度质量或模型效果。去掉劣质样本仍可能值得，但必须把花掉的预算算进去。</figcaption>
</figure>

如果只靠同样的筛选，在固定保留率、等长回答的假设下，凑够原先 64 组的保留量平均需要采样 256 组。真实保留率会随模型变化，长短回答也不一致，不能把这个倍数当作恒定吞吐。

为什么会有这么多没用上的组？可能是题太难、整组都失败；也可能是题太容易、整组都成功。对组内相对奖励，二者都可能缺少区分。并不是“只留下成功回答”就解决了问题。[DAPO（2025），§3.2](https://arxiv.org/html/2503.14476v2)的 dynamic sampling 处理这类组，但补采样要付成本；更多原理见[稀疏组算例](llm-bridge.md#sparse-groups)。

## 为什么加卡、加并行也不一定省钱 {#system-cost}

一轮同步流程可以先按阶段拆开记账：

$$
T_{\mathrm{round}}=T_{\mathrm{rollout}}+T_{\mathrm{verify}}+T_{\mathrm{update}}+T_{\mathrm{sync}}.
$$

假设四项是 60、20、30、10 秒，总共 120 秒。如果资源足够独立、阶段可以充分重叠，稳态流水线的理想下界由最慢阶段决定，这个例子是 60 秒一批。**不是第一批只需 60 秒，也不是直接保证两倍加速。** 排队、填充与排空、网络竞争、权重传输，以及额外占用的 GPU 都要计入。

<figure class="worked-update" aria-label="LLM RL 中不同系统部件的职责">
<ol>
<li><small>Rollout · 推理侧</small><strong>生成轨迹与行为 log-prob</strong><span>vLLM / SGLang 等引擎处理生成与缓存；上层 worker 还要记录版本、采样设置和终止原因。</span></li>
<li><small>Reward · 环境侧</small><strong>给每次尝试可靠反馈</strong><span>测试沙箱、工具或判分模型。超时与任务失败要区分，反馈必须对回原轨迹。</span></li>
<li><small>Update · 训练侧</small><strong>从轨迹计算一次更新</strong><span>训练后端处理前后向和通信；controller 协调批次、权重同步、检查点与评估。</span></li>
</ol>
<figcaption>FSDP 负责训练状态分片等职责，不负责定义 reward；推理引擎也不会自动决定 advantage。框架把部件接起来，任务语义仍要自己确认。</figcaption>
</figure>

FSDP、NCCL 这些训练侧部件各管哪一层，见[框架这一层的多卡地图](../../12-infra/frameworks.md#nccl-map)。

同步容易追踪“哪版模型生成、哪版模型训练”，但会等慢请求。异步减少等待，却让样本落后于当前策略。[AReaL（2025），§4–5](https://arxiv.org/html/2505.24298v2)把 generation 与 training 解耦，同时控制数据陈旧度并调整更新目标；这不是单纯多开几个 worker。其[当前文档](https://areal-ai.io/docs/en/algorithms/async.html)也把陈旧度与吞吐作为取舍，而不是越异步越好。

另一个细节更隐蔽：权重相同，训练后端和推理后端的概率也未必完全相同，精度、kernel 与采样变换都可能影响计算。应区分实际 behavior log-prob、训练侧重算的 old log-prob，以及当前策略 log-prob。[verl 的 rollout correction 文档](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html)专门讨论这类分布差异。重要性加权、裁剪和过滤各有条件与方差 / 偏差代价，不能保证修好所有不匹配。

## 稳定性不是只把 learning rate 调小 {#stability}

| 现象 | 可能卡在哪里 | 下一步先查什么 |
| --- | --- | --- |
| Reward 很久不动 | 题太难、反馈太稀疏，或评分器总给同分 | 按题目切片看成功率和组内方差，人工读失败样本 |
| Reward 涨，独立 eval 不涨 | 利用评分漏洞、分布变窄或评估预算变了 | 固定预算重测，检查没有参与训练的用例 |
| 回答越来越像 | 探索不足、熵下降、奖励偏好单一形式 | 同题多次采样比较，查看 entropy 与独立正确率 |
| 长回答越来越多 | 真有收益、长度被误奖，或终止规则有问题 | 按长度分桶评估；记录 EOS、截断和工具循环 |
| Loss / ratio 突然异常 | 版本、log-prob、mask、归约分母或数值不匹配 | 固定同一小批数据，逐 token 对比更新前后的量 |
| 异步更快但结果更差 | 数据过旧，或丢弃策略改变了任务分布 | 按 policy lag、长度和任务类型分桶对比 |

这些是待检验的假设，不是看到一条曲线就能下结论。比如回答变长，可能是长思维链（CoT）真带来了收益，也可能只是长度被误奖；R1-Zero 里长度为什么会自己涨，见[长思维链是怎样训练出来的](../../11-frontier/test-time-compute/long-chains.md)。只减小学习率可能延后出问题，却没有修正 reward 对错轨迹、跨样本串数据或错误分母。

近年的改法也在处理不同环节。[DAPO §3](https://arxiv.org/html/2503.14476v2)讨论探索、采样、token 级归约与过长回答；[GSPO（2025），§4–5](https://arxiv.org/html/2507.18071v1)改变序列与 token 的加权 / 裁剪方式，并在其 Qwen3 MoE 设置中评估稳定性。它们提供了可检验的设计，不意味着已经找到适用于所有模型、奖励和预算的最优配置。细节接到 [DAPO](../dapo.md) 与[策略概率比](../policy-ratios.md)。

## 数据配置，就是训练问题的一部分 {#data-contract}

同一段代码、同一个算法名，下面任何一项变了，都可能是在做不同的实验：

| 需要固定的东西 | 为什么不是“杂活” |
| --- | --- |
| Prompt 来源、去重、任务比例、难度 | 决定模型把尝试花在哪些问题上；独立验证集不能被课程采样过程用来调题 |
| Prompt / response / tool observation mask | 决定哪些 token 属于策略动作、哪些位置进入 loss |
| Verifier 版本、超时、重试、部分分 | 决定失败是什么意思；基础设施故障不应悄悄被当作模型失败 |
| Group ID、采样策略、policy 版本、行为 log-prob | 决定比较的对象和概率比是否对得上；多版本轨迹可能还需逐段记录 |
| EOS、截断、上下文预算 | 决定一次任务什么时候结束；超时不一定等于环境真正终止 |
| Token / sequence 平均、分布式全局分母 | 决定长短回答、不同 rank 的样本究竟占多大权重 |

检查点也不只是权重：恢复 optimizer、scheduler、RNG、数据游标，并为未完成 rollout 和队列设计明确的恢复或丢弃规则。能重新启动，不代表继续了同一条训练轨迹。具体检查见[后训练数据流水线](../../practice/post-training/data-pipeline.md#supervision-smoke-test)与[训练框架](../../06-systems/training-frameworks.md)。

## 有些困难，靠换框架解决不了 {#limits}

**长任务的信用分配。** 最后测试没过，错在第一步读题、第三次工具调用，还是最后输出格式？终局奖励到任务结束时才给出，定位错误很粗；它是否便宜，仍取决于测试或判分模型的成本。过程奖励更细，却增加标注和验证难度；Critic 可以提供估计，也会把自身误差带进更新。更长的推理文本本身不等于更准确的信用分配。多轮 agent 里的做法，如按状态再分组的 GiGPO 和 turn 级 critic，见[Agent RL 的信用分配](../../11-frontier/agentic-rl/credit-and-training.md)。

**探索的成本。** 如果初始策略从来不产生有用尝试，再可靠的 verifier 也只能不断判失败。示范、课程或更强的探索可以帮助，但会改变数据、预算和先验，应该计入比较。RL 不是一定只能重排已有答案，也不是给任何基座加奖励就会自动出现新能力。

**反馈与真实需求的差距。** 代码通过一组测试，不代表满足所有使用场景；开放问答更难写出完整评分器。模型会接触到评估者没想到的情况。用同一个 judge 造数据、发 reward、做最终评估，尤其容易把共同盲点当成进步。训练反馈与独立 eval 的职责要分开，详见[数据作为监督](../../01-data-and-feedback/data-as-supervision.md)。

因此，当前可做的是把失败拆小、做受控比较，而不是宣布“RL 已经有万能配方”或“RL 没用”。这是一组公开机制与工程限制的梳理，不是对所有实验室最新训练方案的完整调查。

## 自己做实验，先按这个顺序排查 {#diagnosis}

1. **先定怎样算变好。** 留出独立数据，固定生成预算和判分标准；保留原模型与简单 SFT / DPO baseline，方法不同也要记录各自的数据和计算成本。
2. **用少量固定轨迹查正确性。** 手算 reward、mask、advantage、概率比和归约分母；确认一次更新方向正确。验证恢复时不会重复发奖励或错配轨迹。
3. **再跑小规模同步训练。** 看真实尝试与错误切片，不只看总分；用多个随机种子确认结果不是偶然。
4. **一次只改一类问题。** 增大 group、改难度、换奖励、换更新方式，分别对比。同一评估预算下的质量和累计成本都记录。
5. **正确性站稳后再扩大并行。** 加异步、低精度或新后端时，重新核对 log-prob、样本分布与最终质量，而不是只检查速度。

如果质量停住，下一步不一定是更复杂的 loss。也许需要更可靠的测试、不同难度的数据；如果信号清楚但训练仍崩，才继续定位估计、更新与系统问题。**难点是这些环节互相影响，不是谁可以永远排在第一。**

算例可运行 `python3 05-post-training/deep-rl/code/rl_tradeoffs.py`。它检查预算与目标函数的局部计算，不模拟完整训练，也不能据此声称某种方法收敛更快。
