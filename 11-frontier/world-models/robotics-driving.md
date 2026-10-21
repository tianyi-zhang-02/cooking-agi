# 世界模型：给机器人和汽车当模拟器、评测器与策略

**中文** · [English](robotics-driving.en.md)

> 阅读时间：约 12 分钟 · 最近审阅：2026-10

手上有 5 个推杯子策略的 checkpoint，想知道哪个最好。每个在真机上试 50 次，就是 250 次试验，还得有人复位杯子、盯着别让它摔下桌。能不能先让世界模型“演”一遍？**在机器人和自动驾驶里，世界模型干三份不同的活：造训练数据、给策略打分、直接当策略。三份活要验证的东西不一样；当评测器时，和真机成功率相关系数高，也不保证能挑对最好的那一个。**

## 一个模型，三份活

[Cosmos 论文](https://arxiv.org/abs/2501.03575)（NVIDIA，2025-01）把世界基础模型 $\mathcal W$ 写成：根据过去的观测和当前的“扰动”预测下一个状态（图 3）。

$$
\hat x_{t+1}=\mathcal W(x_{0:t},\,c_t)
$$

$x_{0:t}$ 是到时刻 $t$ 为止的观测，通常是视频帧；$c_t$ 是当前的扰动，可以是机器人动作、自车轨迹或一段文字。同一个预测器，用法不同，就是不同的活：

| 活 | 输入 → 输出 | 替代了什么 | 必须验证什么 | 例子 |
| --- | --- | --- | --- | --- |
| 数据引擎 | 场景 + 指令 → 合成视频，再推出动作 | 采更多真机示范 | 用合成数据训出的策略在真机上是否变好 | DreamGen、Cosmos |
| 策略评测器 | 策略的动作 → 模型里的 rollout → 成功与否 | 真机评测 | 模型打分的排序是否和真机一致 | WorldEval、WorldGym、Veo 评测器、dWorldEval、Ctrl-World、1X |
| 策略本身 | 当前画面 + 任务 → 预测未来 → 动作 | 单独训练的 VLA 策略 | 闭环成功率、延迟、泛化 | DreamZero、Cosmos Policy、1X NEO、Praxis-1 |

[WorldArena](https://arxiv.org/abs/2602.08971)（Shang 等，2026-02 论文）正是按数据引擎、策略评测器和规划器三种用途，分别测了 14 个具身世界模型，摘要的结论是“high visual quality does not necessarily translate into strong embodied task capability”，即画面好看不等于具身任务上有用，作者称之为 perception–functionality gap。[World-in-World](https://arxiv.org/abs/2510.18135)（2025-10，ICLR 2026 Oral）也报告：闭环任务成功更取决于可控性，而不是画质。

## 活一：造数据

[DreamGen](https://arxiv.org/abs/2505.12705)（NVIDIA GEAR 等，2025-05 论文）摘要里的流程是：先把一个 image-to-video 模型适配到目标机器人，生成熟悉或全新任务、不同场景的视频；视频没有动作标签，就用 latent action model 或逆动力学模型（inverse dynamics model，IDM）推出伪动作。这和[交互视频一篇](interactive-video.md)里 Genie 的 latent action 是同一个思路。论文把这种合成数据叫 neural trajectories，报告人形机器人只用一个环境里一个抓放任务的遥操作数据，就学会了 22 种新行为。

<!-- paper-figure:dreamgen-overview -->

代价也要算。[DreamGen 论文](https://arxiv.org/abs/2505.12705)报告：生成 24 万条 RoboCasa 样本，用 1,500 张 L40 跑了 54 小时，即 $1500\times54=81{,}000$ GPU 小时，平均每条约 $81{,}000/240{,}000\approx0.34$ GPU 小时，每条约 20 分钟的 GPU 时间。真机采一条示范贵不贵，要和这个数放在一起比。

NVIDIA 的 Cosmos 是这条线上最完整的开放系列。下表只列我们在公开资料里核对到的内容，日期都是 NVIDIA 的发布时间：

| 时间 | 版本 | 公开内容 |
| --- | --- | --- |
| 2025-01 | Cosmos WFM 平台 | 开放的 diffusion 与自回归世界基础模型（4B–14B）、tokenizer、数据整理流程；定位为物理 AI 的“世界数字孪生” |
| 2025-03 | [Cosmos-Transfer1](https://arxiv.org/abs/2503.14492) / [Reason1](https://arxiv.org/abs/2503.15558) | 控制图 → 视频；物理推理 VLM（7B / 56B） |
| 2025-06-11 | [Cosmos-Predict2](https://github.com/nvidia-cosmos/cosmos-predict2) | 2B / 14B Video2World，外加一个动作条件模型 |
| 2025-10-28 | [Predict2.5 / Transfer2.5](https://arxiv.org/abs/2511.00062) | 统一的 flow 模型（2B / 14B），在 2 亿条整理过的片段上训练 |
| 2026-01-06 | [CES 更新](https://forums.developer.nvidia.com/t/nvidia-cosmos-announcements-at-ces-2026/356629) | Cosmos Reason 2、Predict 2.5、Transfer 2.5-2B |
| 2026-03-16 / 05-31 | [Cosmos 3](https://arxiv.org/abs/2606.02800) | GTC 宣布，5 月 31 日发布；双塔 mixture-of-transformers：自回归 VLM 推理塔，加一个处理文本、图像、视频、音频和动作的 diffusion 生成塔 |
| 2026-07-15 | Cosmos 3 Edge | 4B，基于 Nemotron |

Cosmos 3 的规模，NVIDIA 自己的两处说法不一致：[开发者博客](https://developer.nvidia.com/blog/develop-physical-ai-reasoning-world-and-action-models-with-nvidia-cosmos-3)写 Nano 16B、Super 64B，[Hugging Face 合集](https://huggingface.co/collections/nvidia/cosmos3)把 Super 列为 65B、Edge 4B。NVIDIA 2026-05-31 的[新闻稿](https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-3-the-open-frontier-foundation-model-for-physical-ai)称它在 Physics-IQ、PAI-Bench 等多个榜单上位列开源模型第一，属公司自报；第三方维护的 [Physics-IQ Verified 排行榜](https://github.com/google-deepmind/physics-IQ-benchmark)在 2026-10-15 读取时，Cosmos3-Super（multiframe）为 50.8 ± 2.2。

另一个关于成本的数据点：Meta 在 [V-JEPA 2 论文](https://arxiv.org/abs/2506.09985)（2025-06）里把 Cosmos 放进同一个 CEM 规划器当世界模型，每个动作要 4 分钟，抓放成功率 0%；V-JEPA 2-AC 每个动作 16 秒。这是 Meta 的测量，只说明那次实验里的 Cosmos 不适合当这种规划器的内核，不代表它当数据引擎没有价值；细节见 [JEPA 一篇](jepa.md)。

## 活二：当评测器

不部署就估计新策略的效果，RL 里叫 off-policy evaluation，见[固定数据下的 RL](../../05-post-training/deep-rl/offline-and-ope.md)。世界模型给了另一条路：让策略在模型里行动，数一数成功了几次。下面是把世界模型当评测器的几项工作；有数字的，都是各自作者在自己选的策略和任务上测的：

| 系统（机构，时间） | 报告的对照结果 | 一起读的条件 |
| --- | --- | --- |
| [WorldEval](https://arxiv.org/abs/2505.19017)（Midea / ECNU，2025-05） | 与真实结果 r = 0.942 | 作者自选的策略和任务 |
| [WorldGym](https://arxiv.org/abs/2506.00613)（Stanford、NYU、GDM，2025-06） | 按任务 r = 0.78 | 同上 |
| [1X World Model](https://www.1x.tech/discover/redwood-ai-world-model)（1X，2025-06-16） | 用动作条件视频加价值模型给人形机器人策略排序 | 没有公布相关系数 |
| [Ctrl-World](https://arxiv.org/abs/2510.10125)（Stanford、Tsinghua，2025-10） | 多视角机器人世界模型，用来评测并改进策略 | 我们没有核对到具体相关数字 |
| [Veo 评测器](https://arxiv.org/abs/2512.10675)（Google DeepMind，2025-12，v2 2026-01） | OOD 研究里 Pearson 0.86、MMRV 0.06 | 见下文 |
| [dWorldEval](https://arxiv.org/abs/2604.22152)（Li 等，2026-04） | r ≈ 0.91–0.93 | 离散 diffusion 世界模型 |
| [PiL-World](https://arxiv.org/abs/2606.05773)（2026-06） | 成功率估计误差从 63.2% 降到 12.0% | 只有 3 个双臂任务 |

Veo 评测器这一篇我们读了原文。它把 Veo 2 文生视频模型在机器人数据上微调，评测 ALOHA 2 双臂平台上的策略（4 个相机，§2、§3.1）；摘要写共做了 1,600 多次真机评测，涉及 8 个 checkpoint、5 个任务。Pearson 0.86 和 MMRV 0.06 是针对最强的那一个策略，在背景、小干扰物、大干扰物、物体这几条 OOD 泛化轴上比较预测成功率和真实成功率（§4.1）；正文没写点数，要回图 8 去数。论文自述的局限：预测的绝对成功率低于真实值（§3.2、§4.1）；接触密集、特别是小物体的交互难模拟（§7）；每次 rollout 8 秒，分钟级的多视角一致生成仍是待达成的目标（§7）；成功与否目前靠人工判定（§7）。

<!-- paper-figure:veo-eval-ood -->

Veo 评测器论文 §3.2 给出的 MMRV 定义如下（指标出自 Li 等 2024）：

$$
\text{MMRV}=\frac1n\sum_{i=1}^n\max_j\;|R_i-R_j|\cdot\mathbb 1\big[(\hat R_i<\hat R_j)\neq(R_i<R_j)\big]
$$

$R_i$ 是策略 $i$ 的真机成功率，$\hat R_i$ 是世界模型预测的成功率。每个策略只记“和它排反了的对手里，真实差距最大的那一个”，再对所有策略取平均。0 表示排序完全一致。

### 算一次：相关 0.98，第一名还是选错了（教学例子）

5 个推杯子策略，数字是编的：

| 策略 | 真机成功率 $R$ | 模型预测 $\hat R$ | 真机名次 | 模型名次 |
| --- | --- | --- | --- | --- |
| A | 0.10 | 0.05 | 5 | 5 |
| B | 0.25 | 0.20 | 4 | 4 |
| C | 0.45 | 0.30 | 3 | 3 |
| D | 0.70 | 0.55 | 2 | 1 |
| E | 0.80 | 0.50 | 1 | 2 |

- **Pearson**：$\bar R=0.46$，$\bar{\hat R}=0.32$；交叉项之和 $\sum(R-\bar R)(\hat R-\bar{\hat R})=0.239$，两个平方和分别是 0.347 和 0.173，所以 $r=0.239/\sqrt{0.347\times0.173}\approx0.98$。
- **Spearman**：只有 D、E 名次互换，名次差的平方和 $\sum d^2=2$，$\rho=1-6\sum d^2/(n(n^2-1))=1-12/120=0.9$。
- **MMRV**：只有 (D, E) 这一对排反，真实差 0.10；D、E 各记 0.10，其余为 0，$\text{MMRV}=0.2/5=0.04$。

三个指标都好看，可按模型挑出的最好策略是 D，真机上 E 比它高 10 个百分点。相关性主要是 A、B 这些差策略和好策略之间的大差距撑起来的。只看真正要选的前三名 C、D、E，Spearman 变成 $1-6\times2/(3\times8)=0.5$。

<!-- widget:step-evaluator-rank -->

还有两点：

- **绝对值。** Pearson 在 $\hat R=aR+b$（$a>0$）下不变：一个把所有成功率都报成真实值一半的模型，$r=1$。上表的模型也普遍低估，和 Veo 评测器论文自述的一样。要用评测器判断“是否达到 70% 的上线门槛”，相关系数帮不上忙。
- **真机这一侧也有噪声。** 若每个策略只试 50 次，D、E 成功率之差的标准误约为 $\sqrt{0.7\times0.3/50+0.8\times0.2/50}\approx0.086$。差 0.10 只有约 1.2 个标准误，真机自己的排名也不稳。两个策略怎么比，见[分数涨了，究竟能说明什么](../../07-evaluation/metric-robustness.md)；一致、相关、校准各管什么，见 [judge 校准一篇](../../07-evaluation/llm-as-a-judge/calibration.md)。

| 相关系数高能说明 | 不能说明 |
| --- | --- |
| 在这批策略里，模型分数大体随真实成功率升降 | 最好的那个一定排第一 |
| 用它粗筛掉明显差的 checkpoint，风险较小 | 两个接近的 checkpoint 谁更好 |
| 在验证过的任务和场景里可用 | 换任务、换机器人、换策略家族后仍成立 |
| 线性关系强 | 绝对成功率准，能拿来判断是否达标 |

### 被评测的策略会钻评测器的空子

[Model-based RL 一篇](../../05-post-training/deep-rl/model-based.md)讲过：规划越用力，越可能找到模型的漏洞。评测也一样。从 20 个 checkpoint 里挑世界模型分数最高的，被挑中的往往是模型高估得最多的那个；如果再拿世界模型分数当奖励去训练，策略会主动寻找“模型判成功、真机失败”的动作。1X 在 2026-01-12 就自己写明，它生成的视频对任务完成过于乐观。验证评测器时，至少留一组从没参与挑选的策略，并定期回真机抽检。

## 活三：直接当策略（world action model）

思路是先在模型里“想象”接下来几秒应该看到什么，再从想象的画面里读出动作；或者让模型同时生成视频和动作。

```text
方式 A（1X NEO）：当前画面 + 任务 → 14B 视频模型生成未来视频 → 逆动力学模型读出动作
方式 B（DreamZero、Cosmos Policy）：当前画面 + 任务 → 同一个模型联合生成未来画面和动作
```

- **DreamZero**（NVIDIA 等，2026-02-17 [论文](https://arxiv.org/abs/2602.15922)）：14B 的自回归 video-diffusion“World Action Model”，以 7 Hz 闭环控制。摘要报告：在真机上，对新任务和新环境的泛化是最先进 VLA 的 2 倍以上；10–20 分钟来自其他机器人或人类的纯视频示范，能让未见任务的表现相对提升 42% 以上；换到新机体只需 30 分钟 play 数据。NVIDIA 在 2026-03-16 称 GR00T N2 基于 DreamZero 的研究（公司自报）。
- **Cosmos Policy**（NVIDIA 与 Stanford，2026-01-22 [论文](https://arxiv.org/abs/2601.16163)）：把 Cosmos-Predict2-2B 后训练成策略，动作、未来状态和价值都编码成 latent 帧。仿真里 LIBERO 平均 98.5%、RoboCasa 67.1%；真机 ALOHA 平均得分 93.6，高于 π0.5 的 88.6；但在分布外，π0.5 更好（92.5 对 89.3）。
- **1X 世界模型当 NEO 的策略**（1X，2026-01-12 [博客](https://www.1x.tech/discover/world-model-self-learning)）：14B 视频主干先用 900 小时第一人称人类视频 mid-train，再用 70 小时 NEO 数据微调；生成 5 秒视频要 11 秒，IDM 再约 1 秒。拉纸巾任务单次生成成功率 30%，并行生成 8 次为 45%。1X 自己列出的局限：生成视频对完成任务过于乐观，3D 落地弱，评测不是盲测。
- **Runway Praxis-1**（2026-09-30 [公告](https://runway.com/research/introducing-praxis-1)）：宣布为开放权重的 world action model；截至 2026-10-15，我们只核对到这篇公告。

算一下时间：7 Hz 意味着每个控制步约 $1000/7\approx143$ ms。1X 生成 5 秒未来要 11 秒，比实时慢 $11/5=2.2$ 倍，加上 IDM 约 12 秒才得到一段动作。我们的推断是：这适合慢节奏、能等的任务，不适合需要马上反应的场景。Cosmos Policy 的对比则提醒，“用视频模型当策略泛化更好”要分条件说：分布内赢，分布外输。另外，Physical Intelligence 在 2026-04-16 介绍 [π0.7](https://www.pi.website/blog/pi07) 时提到用一个“轻量世界模型”生成视觉子目标，但没有给细节。

## 驾驶：GAIA 与 Waymo World Model

驾驶里先分清开环和闭环。开环是回放录好的传感器数据，看策略在每一帧会怎么开；可策略一旦开得和录像不同，后面的画面就对不上了。闭环是策略的动作驱动世界模型生成下一帧传感器数据，策略再看新画面做决定；这样才能看到“偏离之后会发生什么”。

| 时间 | 系统（机构） | 公开内容 |
| --- | --- | --- |
| 2023-09 | [GAIA-1](https://arxiv.org/abs/2309.17080)（Wayve） | 视频、文本、动作 token；6.5B 世界模型加 2.6B 解码器；4,700 小时伦敦驾驶数据 |
| 2025-03 | [GAIA-2](https://arxiv.org/abs/2503.20523)（Wayve） | 8.4B latent diffusion，多相机、可控 |
| 2025-12-02 | [GAIA-3](https://wayve.ai/thinking/gaia-3/)（Wayve） | 15B latent diffusion，定位为评测端到端驾驶系统 |
| 2026-02-06 | [Waymo World Model](https://waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation)（Waymo） | 在 Genie 3 上后训练，输出摄像头和 lidar；可用驾驶动作、场景布局和语言控制；能把行车记录仪或手机视频转成仿真；没有公布保真度指标 |
| 2026-06-10 | [Oasis 3](https://decart.ai/publications/introducing-oasis-3-first-interactive-world-model-for-physical-ai)（Decart） | 三路同步相机的驾驶与机器人世界模型，512×768、22 FPS、延迟低于 200 ms（公司自报） |
| 2026-08-03 | [GAIA-4](https://wayve.ai/thinking/gaia-4/)（Wayve） | 闭环：AI Driver 的动作驱动下一帧传感器数据；新增雷达输出（Wayve 称业内首个）；唯一的数字是“重建忠实度好 2.5 倍”（公司自报）；没有论文和参数量 |

截至 2026-10-15，我们在公开资料里没有找到 GAIA-3/4、Waymo World Model 或 1X 评测器与真实结果的相关系数。Tesla 在 ICCV 2025 和 CVPR 2026 的演讲里介绍过自家用来做闭环评测的“神经世界模拟器”（[演讲视频](https://www.youtube.com/watch?v=2V0ddkF1Svo)，内容据视频自动字幕整理），但没有发表论文，也没有公布和真实结果的对照；详见[驾驶系列的端到端一篇](../../13-embodied/driving/end-to-end.md)。

## 风险清单

| 风险 | 具体会怎样 | 例子 | 怎么查 |
| --- | --- | --- | --- |
| 仿真到真实的差距 | 画面逼真，接触、摩擦、小物体却不对 | Veo 评测器自述小物体接触难模拟；WorldArena 的 perception–functionality gap | 按失败类型分组和真机对照，不只看总相关 |
| 评测器被钻空子 | 策略学会“模型判成功、真机失败”的动作 | 1X 自述生成视频过于乐观；Ha & Schmidhuber 的 VizDoom 对抗策略 | 留出从未参与挑选的策略，定期真机抽检 |
| 失败模式覆盖不全 | 训练数据里少见的失败，模型演不出来 | 8 秒 rollout 看不到更慢的失败，比如杯子一点点滑出桌边 | 列出真机失败清单，逐类检查模型能否复现 |
| 绝对值不准 | 排序对了，门槛判断错了 | Veo 评测器预测成功率整体偏低 | 另做校准，或只拿来排序 |
| 算力和延迟 | 生成比真机还慢、还贵 | DreamGen 用 1,500 张 L40 跑 54 小时；1X 11 秒生成 5 秒；Cosmos 规划每个动作 4 分钟 | 把生成成本和真机成本记在同一本账上 |
| 不透明 | 外界无法独立检验 | Waymo、GAIA-4 未公布保真度指标或架构 | 只把公开验证过的用途当结论 |

## 这条线接下来看什么

1. **评测器能否跨机器人、跨任务、跨策略家族成立？** Veo 评测器（Google DeepMind，2025-12）验证在一个平台（ALOHA 2）和 8 个 checkpoint 上；WorldEval（2025-05）和 dWorldEval（2026-04）各在作者选的设置上。能改变判断的结果：在多种机体和留出的策略家族上，同时报告相关、MMRV 和“第一名是否选对”。
2. **驾驶世界模型会不会公布保真度和闭环验证？** Waymo（2026-02）和 Wayve 的 GAIA-4（2026-08）都没有公布与真实结果的相关系数。能改变判断的结果：公开的“仿真里的事故或接管率”与真实道路数据的对照。
3. **World action model 和 VLA 谁在什么条件下更好？** Cosmos Policy（2026-01）分布内赢、分布外输给 π0.5；DreamZero（2026-02）报告 2 倍以上的泛化；Runway 在 2026-09-30 宣布 Praxis-1 开放权重。能改变判断的结果：同一套真机任务、同样数据量下的第三方对比。
4. **延迟和成本能否降到闭环可用？** DreamZero 自报 7 Hz；1X 自报生成比实时慢 2.2 倍。能改变判断的结果：在写明的硬件上实时运行，成功率不降。

接着读[它真的懂世界吗](do-they-understand.md)；想回顾世界模型怎么拿来规划，回到 [Model-based RL](../../05-post-training/deep-rl/model-based.md)。
