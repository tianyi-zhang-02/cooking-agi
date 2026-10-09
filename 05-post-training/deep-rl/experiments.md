# RL 实验与排错：先确认学对了，再看学得快不快

**中文** · [English](experiments.en.md)

> 阅读时间：约 6 分钟 · 最近审阅：2026-10

RL 的 loss 很容易“正常下降”，行为却越来越差。因为采样、target 和策略都在变。先做一个小到可以预测答案的环境，比直接跑大 benchmark 更容易发现问题。

## 一条完整训练循环长什么样

<div class="drl-flow" aria-label="训练与评估分开">
<span>环境采样<br><small>记录策略版本与边界</small></span><b>→</b><span>构造 target<br><small>return / TD / GAE</small></span><b>→</b><span>更新网络<br><small>固定 target 与 old policy</small></span>
</div>

采样完成后先检查 shape、mask、reward scale，再更新。评估用独立环境和固定设置，不把评估轨迹偷偷塞回训练。Checkpoint 不只是网络权重：optimizer、target network、随机数状态、计数器，以及必要时 replay buffer 都影响能否续跑。

## 先跑 2 个 CPU 小程序

在仓库根目录运行；第一个只依赖 Python 标准库，第二个需要 PyTorch：

```bash
python3 05-post-training/deep-rl/code/rl_checks.py
python3 05-post-training/deep-rl/code/torch_updates.py
```

[算术与边界检查](code/rl_checks.py)验证 Bellman、return、GAE、Double DQN 和 ESS；[PyTorch 检查](code/torch_updates.py)验证 Actor 梯度方向、target 无梯度、连续动作的梯度路径，并训练一个两动作 bandit。

Bandit 奖励是 $[1,3]$。初始概率各 0.5，固定随机种子，采样动作并用 REINFORCE 更新 logits。预期是第二个动作概率上升，而不是要求每一步 reward 都单调增长。**它只检查一阶段机制，不是 DQN / SAC 的完整复现或性能 benchmark。**

## 每个阶段留一个可检查的量

| 阶段 | 至少记录 | 能发现什么 |
| --- | --- | --- |
| 数据 | episode length、termination / truncation、reward 分布 | 提前结束、空奖励、错误 reset |
| Value | target、预测值、TD error、绝对值范围 | 发散或 target 量纲错 |
| Policy | entropy、动作比例、KL / ratio（适用时） | 过早塌缩、更新过猛、无变化 |
| 优化 | gradient norm、学习率、更新次数 | 梯度断开、更新频率错 |
| 评估 | 多 seed return、成功率、失败类型 | 平均数掩盖的不稳定性 |

Loss 和 reward 用不同坐标轴也要标清。高 Q、低真实 return 可能是过估计；低 entropy 可能是学会，也可能是卡死。解释要结合行为，而不是凭一个 scalar 下结论。

## 比较算法，先把预算说清楚

环境交互数、梯度更新数、wall-clock 和硬件不是同一个预算。每个训练 seed 可能有多个评估 episode；episode 不能当成独立训练 seed 来夸大样本量。报告 seed 间差异和合适的区间，不只挑最好的一条曲线。

调参用 validation 环境或划分，最终 test 尽量少碰。平滑曲线保留原始数据，并说明窗口；失败运行也要计入。多任务汇总时说明尺度归一化，不能让 reward 大的环境自动决定总排名。

## 两条曲线均值一样，结论也一样吗？

假设各做 5 次独立训练，分别算出每个模型在独立评估回合里的平均 return：

| 方法 | 5 次训练的评估均值 | 再取平均 | 中位数 |
| --- | --- | --- | --- |
| A | 10，10，10，10，50 | 18 | 10 |
| B | 16，17，18，19，20 | 18 | 18 |

只报均值，两者相同；只挑最佳运行，A 很亮眼；需要可重复的表现时，B 更稳定。这里没有一个脱离任务的“正确汇总法”：要先说关心典型表现、失败风险，还是充分调参后的最好结果，并报告相应的不确定性。五个 seed 只是算例，不是足够统计功效的保证。

每个训练 seed 下再跑 100 个评估回合，有助于估计那一个训练结果的表现，却不能把 5 个模型说成 500 次独立训练。重采样或区间估计要尊重层级：环境任务、训练 seed、评估回合，不是一堆可以随便混的独立行。

## 一次可以复查的消融怎么安排？

例如想检查 GAE 的 $\lambda$，先固定环境、奖励、网络、采样量、优化设置和评估规则，只改 $\lambda$。然后才单独比较 Critic 容量或 rollout 长度。

| 记录 | 一个具体约定 |
| --- | --- |
| 主问题 | 同样交互预算下，改变估计长度能否改善稳定性？ |
| 比较对象 | $\lambda=0$、一个中间值、$\lambda=1$；不是同时换三个网络 |
| 主要结果 | 预先选定预算点的独立评估表现及跨 seed 差异 |
| 辅助诊断 | Advantage 分布、Critic 误差、KL、梯度尺度 |
| 异常处理 | 保留失败运行，说明提前停止和排除条件 |

若训练 loss 下降、任务变差，保留 rollout 看模型实际在干什么。数字能定位方向，却不能代替观察：机器人可能不动也能拿稳定小分，语言模型可能把回答写长来讨好 judge。

## 续训测试为什么也算算法正确性？

在小环境中做两组：一组连续训练，另一组在同一步保存并恢复。比较接下来的采样、target、参数和计数器。确定性设置下可以期待严格对齐；非确定性设备或并行环境下，先定义可接受的差异，不承诺所有配置逐 bit 相同。

只保存 Actor 权重，可能漏掉 optimizer 动量、Critic、target network、replay、随机数状态和探索日程。恢复后策略还能输出动作，不代表训练真的从同一个状态继续。应该把“能加载”和“能按原训练语义恢复”分开测试。

## 一个有顺序的排错清单

先随机策略与简单基线；再单条 transition 手算；再固定小 batch 过拟合；再检查随机性与边界；最后扩到完整训练。一次改一个因素。

如果单 batch 都拟合不了，先查代码。如果 train return 好、eval 很差，查分布与评估配置。如果 reward 涨、任务没改善，查 reward specification。三种情况不该用同一种“再加训练步数”处理。

## 扩展实验怎么做

给 bandit 加奖励噪声，比较带 baseline 与不带 baseline 的多 seed 梯度方差；把 GAE 的终止改成截断，看 bootstrap 与 trace 是否分别正确；改变 Double DQN 的两个排序，检查选动作和估值是否真的分离。

参考：[Deep RL 的统计评估问题](https://arxiv.org/abs/2108.13264)。最后接到[语言模型与工具交互](llm-bridge.md)。
