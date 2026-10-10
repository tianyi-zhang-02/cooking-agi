# 后训练的基础设施：采样、数值、上下文

**中文** · [English](post-training-infrastructure.en.md)

> 最近审阅：2026-10 · 前置：[RL 与语言模型](deep-rl/llm-bridge.md)、[PPO clipping](rlhf/ppo-clipping.md)

<span id="_1"></span>

## 先跟着一条回答走一遍 {#_2}

给模型一道乘法题，它调用计算器，读到结果，再写出答案。到这里，我们有了一次运行记录。但要用它训练，还得知道：哪些 token 是模型选的？生成时用的哪版权重？它是真的完成了，还是被长度限制叫停了？

这些信息不是事后可有可无的日志。丢了它们，可能连 loss 应该算在哪些位置都说不清。下面先讲 rollout 记录什么，再看吞吐、log-prob 和上下文怎样影响训练。算例均为教学假设，不是某个框架的实测性能。

## Rollout 不等于一条完整答案 {#_3}

**Rollout 是让策略实际运行，收集动作、观察和结果的过程。** 可以一次跑一个任务，也可以批量跑；可以收集完整任务，也可以只收集一段。它本身不规定一定使用最新策略，也不规定一定有 Reward Model。

| 名称 | 在计算器任务里指什么 | 容易混淆的地方 |
| --- | --- | --- |
| Completion / 一段生成 | 一次模型调用生成的 token 序列 | 工具调用也可能是一段生成，不一定是最终答案 |
| Assistant turn / 助手回合 | 助手的一次发言或工具请求 | 具体边界取决于对话模板与运行协议 |
| Episode / 完整任务 | 从题目到成功、失败或定义好的终点 | 可能包含多次模型调用和环境返回 |
| Fragment / 轨迹片段 | 收集器暂时截下的一段过程 | 不一定走到了任务终点 |

模型生成了结束标记，可能只是这一轮发言结束；工具还没返回，整个任务就还没有完成。反过来，若任务本来只要求一条回答，一次 completion 也可以就是整个 episode。

<figure class="worked-update worked-update--pairs">
<figcaption>同一段对话，模型动作和环境观察要分开记录。</figcaption>
<ol>
<li><small>模型生成</small><strong>请求计算 → 写出答案</strong><span>两段 assistant 输出都可以含策略动作。保留采样 token、对应 log-prob 和生成版本。</span></li>
<li><small>环境返回</small><strong>计算器返回 12</strong><span>结果会进入后续上下文，但不是策略采到的动作。不能直接给它套 policy loss。</span></li>
</ol>
</figure>

### 一条能用于训练的记录 {#rollout-record}

想象记录中有这些片段，长度只是虚构数字，不代表实际 tokenizer 的切分：

| 片段 | token 数 | 是否作为策略动作计算 loss | 原因 |
| --- | ---: | --- | --- |
| 用户题目 | 6 | 否 | 是输入条件 |
| assistant 工具请求 | 4 | 是，按该任务约定 | 是模型实际生成的选择 |
| 计算器结果 | 3 | 否 | 来自环境，不是策略采样 |
| assistant 最终回答 | 5 | 是 | 是模型生成的答案 |
| padding | 2 | 否 | 只是对齐批次 |

输入共有 20 个位置，但有效动作只有 9 个。policy loss 的分母不能随手写成 20。工具结果虽然不直接产生 policy loss，仍会影响后续动作；“不算它的 loss”不等于“从上下文里删掉”。角色 / 控制标记是否为采样动作，要按真实生成协议决定，不能仅凭文本长相。

记录至少包含下列几类信息，字段名随实现而定：

| 信息 | 为什么留它 |
| --- | --- |
| prompt / episode ID、token IDs、角色边界、有效动作 mask | 对齐 logits、动作和后续回放 |
| 行为策略（behavior policy）版本、采样参数、动作 log-prob | 知道这条数据怎样被采出来 |
| 停止原因、最终 observation、工具状态与错误 | 区分任务终止、收集截断和运行失败 |
| reward 来源 / 版本、评分状态 | 未评分不是零分；工具超时也不自动等于模型答错 |
| 需要时的 old / reference log-prob、value、return / advantage | 明确由哪个模型、在哪一步算出 |

reward 可以来自测试、环境、人工或学习出的评分模型。Reference、value、advantage 不一定都在生成时计算，甚至不一定都需要；不要为了凑一张大表而混淆算法角色。PPO 的 rollout buffer 通常只服务一轮有限的多次更新，不等于历史经验可以无限重放。[PPO 原论文](https://arxiv.org/abs/1707.06347)讨论的是采样与多轮 minibatch 更新交替进行。

<details markdown="1">
<summary>小检查：20 个位置，为什么只平均 9 个？</summary>

下面已经按“被预测的 token”对齐，不再演示 logits shift。数值是任意的逐位置教学 loss。真实实现还要核对下一 token 的移位与采样协议。

```python
def action_mean(values, mask):
    if len(values) != len(mask) or any(flag not in (0, 1) for flag in mask):
        raise ValueError("aligned values and binary mask required")
    count = sum(mask)
    if count == 0:
        raise ValueError("no policy actions")
    return sum(value for value, flag in zip(values, mask) if flag) / count

mask = [0] * 6 + [1] * 4 + [0] * 3 + [1] * 5 + [0] * 2
values = [99.0] * 6 + [2.0] * 4 + [99.0] * 3 + [4.0] * 5 + [99.0] * 2
assert len(mask) == 20 and sum(mask) == 9
assert abs(action_mean(values, mask) - 28 / 9) < 1e-12
```

</details>

### 到了终点，还是只是采样时间用完了？ {#rollout-boundaries}

假设当前一步 reward 为 0.2，下一状态估值为 0.8，折扣 $\gamma=0.9$。做一步 TD target 时：

- 任务真的终止：没有后续回报，target = 0.2。
- 只是外部收集时间到了，任务本可继续：保留 bootstrap，target = `0.2 + 0.9 × 0.8 = 0.92`。

这不是让你一律给长度截断补 value。若任务定义为“最多 10 步，没完成就失败”，预算耗尽本身就是终点；如果只是收集器每 10 步暂停一次，就不是同一种情况。[Gymnasium 的时间限制说明](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)明确区分 termination 与 truncation；有限时域任务还需要让观察包含剩余时间等必要信息。

bootstrap 要使用被截断轨迹的最后状态，不能误用环境 reset 后新任务的第一帧。若做 GAE，暂停收集的边界通常还要截断 advantage 递推，不把下一个 episode 的误差串过来；**bootstrap mask 和递推边界 mask 不一定相同**。没有 Critic 的方法，则按自己的截断和奖励约定处理，不必硬加一个 value 字段。

## 采样慢，先分清时间花在哪里 {#rl}

一批数据可能依次经历生成、工具等待、评分、训练、权重同步。只盯 GPU utilization，很容易把“GPU 在等工具”误判成“模型太小”。

用一组假设时间算账：采样 30 秒，评分 10 秒，训练 20 秒，同步 5 秒。完全串行需要 65 秒。如果前三段有独立资源、能在不同批次间理想重叠，而每批同步的 5 秒不能重叠，稳态间隔下界才是 `max(30, 10, 20) + 5 = 35` 秒，约 1.86×。这不是“异步必然快一倍”：首批仍要经过完整流程，资源争用、长尾任务和网络都会增加时间。

| 方式 | 省心在哪里 | 需要付出的代价 |
| --- | --- | --- |
| 同步采样，再更新 | 版本和批次边界清楚，容易排查 | 各阶段可能互相等待；也可以复用同一批 GPU |
| 限制队列长度的异步 | 能重叠部分生成与训练 | 需要处理策略延迟、丢弃规则和回压 |
| 深队列、持续采样 | 容纳突发数据和慢任务 | 可能更旧、更占内存，不保证有效更新更快 |

要同时记录采样 tokens/s、真正参与 loss 的 tokens/s、队列等待分位数、版本落后程度、超时 / 丢弃比例，以及等 wall time 的任务质量。更高吞吐若主要来自收集无效回答，就没有省到学习成本。异步方法的目标差异见 [异步策略学习](async-policy-learning.md)。

## 同一份权重，log-prob 也可能对不上 {#_4}

先把三件事拆开：模型原始分布、温度 / top-p 等处理后的采样分布、训练框架重算出的分布。它们可能相同，也可能不同。引擎返回的字段叫 `logprob`，不代表已经告诉你是哪一种。

例如原始概率为 `[0.8, 0.2]`。用温度 2 采样，概率变成 `[2/3, 1/3]`。若新模型给第二项概率 0.25，相对实际行为分布的比率是 `0.25 / (1/3) = 0.75`；若错用原始 0.2 作分母，就变成 1.25，连增减方向都反了。top-p 截掉某些动作后，还会引入支持集问题，不能假装普通 importance sampling 能恢复从来没采过的部分。

若选择以实际 behavior policy $b$ 为基准，一个动作的比率为：

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{b(a_t\mid s_t)}.
$$

也有实现用训练端重算的 old policy 作为 PPO anchor，再另外校正 rollout 与 old 的差异。**分母不是永远来自推理引擎，也不是永远能由训练端替换。** 要按目标和实现看这几个分布怎样对应。[verl 的 rollout correction 文档](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html)区分了这两类设计；此处核对的是 2026-09-23 更新的文档，不保证未来配置名称不变。

统一采样语义后，再查 kernel、精度、模板、位置、mask 等差异。同一批 token、同一权重在两侧评分，按有效动作统计差值的分位数、极端值和长度切片，不要求所有硬件都 bitwise 一致。比率偏离大也不一定全因数据旧了：对齐错误和数值差异同样可能造成。

<details markdown="1">
<summary>为什么只看平均差值不够？</summary>

两个动作的 log-prob 差分别是 +0.7 和 −0.7，平均恰好为 0。取指数后，比率却约为 2.01 和 0.50，平均也不再是 1。这是指数非线性的简单例子，不是实际误差分布。

还要分清监控对象：原始 ratio 超出 clipping 区间，不等于梯度一定被裁掉。PPO 的 active branch 与 advantage 正负有关。详情见 [PPO clipping](rlhf/ppo-clipping.md)；不要只凭一个 clip fraction 就断言训练已经失效。

</details>

## 上下文删掉什么，训练就看不到什么 {#_5}

继续用工具任务。用户最早说“不要改动原文件”，几轮之后，为了腾出窗口，系统把这句话删了。最终模型改动了文件，问题就不只是 reward 给得不好：它做决定时根本没看到约束。

| 处理方式 | 能省下什么 | 要验证什么 |
| --- | --- | --- |
| 保留最近若干轮 | 实现简单、长度可控 | 目标与早期约束是否被挤掉 |
| 固定保留目标，裁剪重复工具输出 | 少一次摘要模型调用 | 裁剪规则是否误删证据 |
| 摘要旧历史，必要时检索原记录 | 可保留更长任务的主要信息 | 摘要错误、遗漏和检索失败 |

这些做法同时影响任务完成率、答案质量和延迟，不只是“让上下文装得下”。更换压缩策略改变了策略看到的观察，不一定改变底层环境的转移规则；丢掉必要历史还可能让观察不再满足 Markov 条件。把模板、摘要器 / 检索器版本、原始工具记录和外部状态一起管理，才能比较实验并恢复任务。

## 先跑通一小批，再扩大规模 {#_6}

不必一开始就搭最复杂的异步系统。先固定一小批任务，追踪它们从采样到 loss 的每一步：动作数能对上，终止原因能解释，reward 能复核，再去优化等待时间。这里没有“算法一定比工程次要”的排序；需要排除的，是代码实际上没有执行你以为的那个目标。

这些改动也不是都不改变目标函数。更换截断奖励、数据筛选或采样处理，可能改变训练分布、估计量甚至任务定义。把变动记录下来，比笼统地叫它“提速优化”更有用。

## 放大训练前，我会检查这些 {#_7}

1. 从一条原始记录，能重新对齐 token、角色、policy mask 和下一 token 的 logits 吗？
2. 未评分、工具失败、任务失败、长度截断，是否被悄悄合成了同一个零分？
3. old、current、reference 与实际 behavior distribution 分别是什么？采样参数是否留档？
4. 同步改成异步后，采样吞吐、数据年龄和等时间质量是否一起报告？
5. 恢复时除了模型权重，队列、采样器、工具与上下文状态是否也有明确处理？

## 继续阅读 {#_8}

- [实验与调试](deep-rl/experiments.md)：从小实验检查训练循环。
- [Checkpoint 与恢复](../practice/post-training/checkpoint-and-resume.md)：验证恢复后的下一步，而不只是能加载文件。
- [蒸馏](distillation.md)：学生 rollout 怎样接入 teacher 监督。
- [同一个基座能走多远](same-base-different-posttraining.md)：具体报告里的算法与系统变化。

## 如果只记住一件事 {#systems-problem}

Rollout 不只是“生成一点训练数据”。它要留下足够的信息，让训练知道模型在什么条件下作了哪些选择。先把这条记录讲清楚，再看怎样跑得更快。
