# 自动驾驶：端到端驾驶与闭环评测

**中文** · [English](end-to-end.en.md)

> 阅读时间：约 13 分钟 · 最近审阅：2026-10

两个驾驶模型在开环评测上的 L2 误差一模一样；放进闭环仿真，一个不到两秒就压线，另一个始终留在车道里。为什么会这样？

开环评测每一步都把车放回日志里的位置，模型自身的小偏差不会累积；闭环里，这些偏差会改变它下一步看到的世界。随着端到端模型能力提升，闭环或接近闭环的评测仍不可缺少；上路之前，还需要一份安全论证。

## “端到端”到底指什么

共同点是：规划或驾驶目标的梯度能传回感知，中间输出不再是必须的接口。差别在于内部还留多少结构、有没有语言模型参与。

| 系统 | 时间 | 输入 → 输出 | 内部结构 | 来源性质 |
| --- | --- | --- | --- | --- |
| [UniAD](https://arxiv.org/abs/2212.10156) | arXiv 2022-12，CVPR 2023 Best Paper | 多相机 → 规划轨迹 | 跟踪、建图、运动预测、占用、规划用 query 串起来 | 论文与 [README](https://github.com/OpenDriveLab/UniAD) |
| [EMMA](https://waymo.com/blog/2024/10/introducing-emma/)（Waymo） | 2024-10 | 原始相机图像 → 以文本形式输出轨迹、3D 物体、道路图 | 基于 Gemini 的多模态大模型，用任务提示切换 | Waymo 博客；自述不用 lidar 和 radar、推理成本高，未称已部署 |
| Waymo Foundation Model | 2025-12-09 | 相机、lidar、radar → 轨迹、物体、地图、验证信号 | 传感器融合编码器加 Gemini 训练的 Driving VLM，大 Teacher 蒸馏成车载 Student | [Waymo 博客](https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving)，没有给参数量 |
| Tesla FSD v12–v14 | 2023-12 起 | 相机等 → 车辆控制 | Tesla 没有发表架构论文 | 股东信和演讲，见下文 |
| [Alpamayo](https://www.nvidia.com/en-us/solutions/autonomous-vehicles/alpamayo/llm-info/)（NVIDIA） | [R1](https://arxiv.org/abs/2511.00088) 2025-10；2 Super 权重 2026-08-04 | 相机 → 推理文本和轨迹 | Cosmos 推理 VLM 加 diffusion 轨迹解码器；2 Super 为 32B 加 2.3B action expert | 公司自报；AlpaSim 分数没有外部定义 |

<!-- paper-figure:uniad-pipeline -->

下面只按 Tesla 自己的文件和演讲、按日期列，数字都是公司自报；这些一手资料都没有给出 v12–v14 的逐层结构：

- [2023 年第四季度股东信](https://digitalassets.tesla.com/tesla-contents/image/upload/IR/TSLA-Q4-2023-Update.pdf)：V12 于 2023 年 12 月底开始推送，用 AI 来影响车辆控制，而不是把每种驾驶行为都手写出来，并称用了端到端训练。
- 2025 年 10 月 ICCV 演讲的[文字版](https://x.com/aelluswamy/status/1981644831790379245)（Elluswamy，2025-10-24）：输入是各相机的像素、车速等运动学信号、音频、地图和导航，输出开车的控制命令。
- [2025 年第四季度股东信](https://assets-ir.tesla.com/tesla-contents/IR/TSLA-Q4-2025-Update.pdf)：FSD 跑在用客户车和 Robotaxi 真实数据训练的端到端 foundation model 上。
- 2026 年第一、二季度股东信（SEC 副本：[Q1](https://www.sec.gov/Archives/edgar/data/0001318605/000162828026026551/exhibit991.htm)、[Q2](https://www.sec.gov/Archives/edgar/data/1318605/000162828026049213/exhibit991.htm)）：v14.3 升级了针对长尾的 RL 阶段，推理延迟最多降 20%；v14 lite 把 AI4 上 v14 的驾驶行为蒸馏到 AI3 硬件。
- 2026 年 6 月 3 日 CVPR 自动驾驶研讨会（WAD）的主题演讲（[日程](https://cvpr2026.wad.vision/)、[视频](https://www.youtube.com/watch?v=2V0ddkF1Svo)）：据视频自动字幕，Elluswamy 说网络在两颗冗余的 AI4 芯片上各跑一份；这一句没有文字稿可核对。

## 模仿之外：RL 与 self-play

多数端到端模型先在人类驾驶日志上做模仿学习，也会遇到[模仿与奖励](../../05-post-training/deep-rl/imitation-and-rewards.md)里讲的误差累积问题：日志里几乎只有“人在车道中央”的状态，没有“偏了以后怎么回来”的示范。早期的应对方法之一是 [ChauffeurNet](https://arxiv.org/abs/1812.03079)：在训练数据里人为合成扰动，让模型见到偏离的状态。

另一条路是让模型在模拟器里自己开、按结果学：

- Apple 的 [GigaFlow](https://arxiv.org/abs/2502.03349)（[ICML 2025](https://proceedings.mlr.press/v267/cusumano-towner25a.html)）在批量模拟器里只用 self-play 做 RL。作者报告共模拟 16 亿公里、仿真中平均 17.5 年才出一次事故，在 CARLA、nuPlan、Waymax 上达到最好成绩；自述局限是只在仿真里，感知被抽象掉了。
- Elluswamy 在 2025 年 10 月 ICCV 演讲的[文字版](https://x.com/aelluswamy/status/1981644831790379245)里写，Tesla 的神经世界模拟器可以拿历史数据验证新的驾驶模型、合成新的对抗场景来测 corner case，还能在闭环里做大规模 RL。他也写道，开环预测上的 loss 不一定对应真实路上的表现，评测是最难的一关。

RL 绕开了“日志里没有偏离状态”的问题，但换来了新问题：模拟器不像真实世界的地方，可能被策略利用，这和 [model-based 一篇](../../05-post-training/deep-rl/model-based.md)里规划器钻模型漏洞是同一件事。

## 算一次：同样的开环分数，完全不同的闭环结局

下面全是教学设定。直道，车道宽 3.5 m，车宽 1.8 m。为单独观察横向偏移，把车身近似为始终与车道平行、宽度固定的占用区间，忽略车长和航向变化对轮廓的影响。这样，车身中心向一侧偏 $(3.5-1.8)/2=0.85$ m 时车身边界触线；偏到 1.75 m 时中心到达车道线，再偏才越线。车宽不是轮距，题中也没有给轮胎宽度，不能据此判断车轮何时压线。车速 $v=10$ m/s。日志里人类一直在车道中央直行。两个策略都有同样的小偏差 $\varepsilon=0.02$ rad（约 1.15°）：

- **策略 A：沿自车当前航向外推，再多偏 $\varepsilon$。** 它只看自车状态，不看车道线。
- **策略 B：看车道线，往中心拉。** 计划航向 $=\varepsilon-g\,y$，$y$ 是车身中心向左的偏移（m），$g=0.1$ rad/m。

**开环**，即 nuScenes 式的评测：每一帧都把日志里的状态喂给模型（$y=0$，航向 0），让它输出未来 3 s 轨迹，再和日志比。两个策略在这个状态下都输出航向 0.02 的直线，横向偏移 $10\times t\times0.02=0.2t$：1、2、3 s 的 L2 误差都是 0.2、0.4、0.6 m，平均 0.4 m。下一帧又回到日志状态，误差从零重来。

**闭环**：每次执行计划的前 0.5 s（5 m），再从车实际到达的状态重新规划。用小角度近似 $\sin\theta\approx\theta$：

$$
\text{A:}\ \theta_k=\theta_{k-1}+\varepsilon,\qquad
\text{B:}\ \theta_k=\varepsilon-g\,y_{k-1},\qquad
y_k=y_{k-1}+v\,\Delta t\,\theta_k .
$$

| 时间 (s) | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 |
| --- | --- | --- | --- | --- | --- | --- |
| A 的航向 (rad) | 0.02 | 0.04 | 0.06 | 0.08 | 0.10 | 0.12 |
| A 的横向偏移 (m) | 0.1 | 0.3 | 0.6 | 1.0 | 1.5 | 2.1 |
| B 的横向偏移 (m) | 0.100 | 0.150 | 0.175 | 0.188 | 0.194 | 0.197 |

A 的航向每步加 0.02，偏移按 $0.05\,k(k+1)$ 增长：按上面的简化轮廓，在 1.5 s 到 2.0 s 之间车身边界触线，3 s 时车身中心已在邻车道。B 每步 $y_k=0.5\,y_{k-1}+0.1$，收敛到 $\varepsilon/g=0.2$ m，在这一近似下一直留在车道里。

| 指标 | 策略 A | 策略 B |
| --- | --- | --- |
| 开环 L2（1 / 2 / 3 s） | 0.2 / 0.4 / 0.6 m | 0.2 / 0.4 / 0.6 m |
| 闭环 3 s 后横向偏移 | 2.1 m | 0.197 m |
| 车身边界触线（简化轮廓） | 1.5–2.0 s 之间 | 不触线 |

开环分不出它们，原因是日志里根本没有 $y\neq0$ 的状态，“偏了以后怎么办”从来没被问过。再换一个条件：一个完全不看路、只按当前速度和航向匀速外推的策略，在这条直道上开环 L2 是 0，满分。A 其实就是这种“自车状态外推器”加了一点偏差。

<!-- widget:step-open-closed-loop -->

## nuScenes 开环指标为什么不再可信

nuScenes 开环规划评测比较模型预测的自车轨迹和日志轨迹，报告 1、2、3 s 的 L2 误差和碰撞率。几项一手结果：

- [Li 等](https://arxiv.org/abs/2312.03031)（CVPR 2024）统计出 nuScenes 有 73.9% 是直行；摘要说“These models tend to rely predominantly on the ego vehicle's status for future path planning”，即这些模型主要靠自车状态做规划，继续刷这个榜未必能得出可信的结论。
- [AD-MLP](https://arxiv.org/abs/2305.10430) 的摘要说，只输入过去轨迹和速度、不看图像或 lidar 的 MLP，平均 L2 反而低约 20%，只是碰撞率不如用感知的方法。同一个 AD-MLP 放进 Bench2Drive 闭环测试，成功率是 0（README v0.0.4 的表）。
- 实现也不统一。UniAD 自己的 [README](https://github.com/OpenDriveLab/UniAD) 报 nuScenes 平均 L2 0.90 m，[SparseDrive](https://github.com/swc-17/SparseDrive) README 的对比表里 UniAD 是 0.73 m；碰撞率的算法各篇也不一样。
- [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) 的 README（2025-02-05）说这个指标没有说服力，维护者已停止报告。

## 闭环和准闭环各测什么

| 评测 | 车会不会偏离日志 | 其他车会不会反应 | 主要指标 | 主要局限 |
| --- | --- | --- | --- | --- |
| nuScenes 开环 | 不会，每步回到日志 | 不适用 | L2、碰撞率 | 直行为主、自车状态捷径、实现不一 |
| NAVSIM v1 | 只在一次 4 s 模拟里 | 不会 | PDMS | 不看误差累积和其他车的反应 |
| NAVSIM v2 | 第二阶段从扰动后的起点出发 | 不会 | EPDMS | 起点是预先算好的，画面靠重建渲染 |
| nuPlan | 会 | CLS-R 会 | CLS-R / CLS-NR / OLS | 输入是给定的物体状态，不测感知 |
| CARLA / Bench2Drive | 会，全程自己开 | 会，由脚本控制 | 驾驶分 DS、成功率 | 画面和交通来自游戏引擎 |
| 学出来的世界模型 | 会 | 取决于模型 | 各家自定 | 截至 2026-10 没有公开能核对的保真度数值 |

最后一行指 Waymo World Model（2026-02）、Wayve GAIA-4（2026-08）和 Tesla 的神经世界模拟器（据演讲自动字幕）这类生成式模拟器；纯重建为什么在车偏离日志后失效，见[三维重建与神经渲染](../perception/reconstruction.md)，世界模型怎样当评测器，见[世界模型：机器人与驾驶](../../11-frontier/world-models/robotics-driving.md)。

### NAVSIM：给一条计划，模拟 4 秒

[NAVSIM v1](https://arxiv.org/abs/2406.15349)（NeurIPS 2024）把模型给出的计划交给 LQR 控制器，在俯视图里不带反应地模拟 4 s，再按规则打分：

$$
\text{PDMS}=\text{NC}\times\text{DAC}\times\frac{5\,\text{EP}+5\,\text{TTC}+2\,\text{C}}{12}
$$

NC 是没有自己负责的碰撞，取 0、1/2 或 1；DAC 是没有驶出可行驶区域，取 0 或 1；EP 是行驶进度，在 0 到 1 之间；TTC 是碰撞时间没有低于阈值，C 是舒适，都取 0 或 1（[指标文档](https://github.com/autonomousvision/navsim/blob/main/docs/metrics.md)）。算一次：NC=DAC=1，EP=0.8，TTC=C=1，PDMS $=(4+5+2)/12\approx0.917$，论文里写成 91.7。若像第 2 篇的候选 A 那样离前车太近、TTC=0，PDMS 降到 $(4+0+2)/12=0.5$；若驶出路面、DAC=0，其他项再好也是 0。这和第 2 篇的结构一样：硬约束相乘，软项加权平均。

论文 navtest 上的 PDMS：人类 94.8，TransFuser 84.0，UniAD 83.4，只用自车状态的 MLP 65.6，匀速外推 20.6。在这里，只看自车状态的方法明显落后，nuScenes L2 上看不出这一点。论文也写明了自己的局限：不模拟其他车的反应，不看误差累积，不含红绿灯和停车标志规则，建议同时跑 CARLA。

放回我们的例子：A 和 B 在日志起点给出的是同一条计划，4 s 后横向偏移 $0.2\times4=0.8$ m，PDMS 自然相同。v1 一样分不出它们，这正是“不看误差累积”的意思。

[NAVSIM v2](https://arxiv.org/abs/2506.04218)（CoRL 2025）的“伪仿真”分两个阶段：第一阶段照常给计划打分；第二阶段从一批预先算好、用 3D Gaussian splatting 渲染的扰动起点出发再打分，按起点与第一阶段终点的接近程度加权，最后两段相乘。它的 EPDMS 在乘法项里加了行驶方向和红绿灯，把舒适项换成历史舒适，并新增车道保持和扩展舒适两个加权项（[指标文档](https://github.com/autonomousvision/navsim/blob/main/docs/metrics.md)）。摘要报告，它与闭环结果的 $R^2$ 为 0.8，最好的开环指标是 0.7。按这个设计，如果第二阶段的起点里有“已经偏左 0.8 m、车头也歪了”这样的状态，A 会继续偏，B 会拉回来，车道保持和可行驶区域两项才有机会把它们分开。

<!-- paper-figure:navsim2-pseudosim -->

### CARLA 与 Bench2Drive：让车全程自己开

[CARLA 排行榜 2.0](https://leaderboard.carla.org/evaluation_v2_0/) 的驾驶分是路线完成率乘以违规惩罚，每发生一次违规就乘一次对应系数：撞行人 0.50，撞车 0.60，撞静止物体 0.65，闯红灯 0.70，闯停车标志 0.80。

$$
\text{DS}=\text{RC}\times\prod_j p_j^{\,n_j}
$$

算一次：完成 90% 路线，撞车一次、闯红灯一次，$\text{DS}=90\times0.60\times0.70=37.8$。另一个模型开到一半就停下、没有任何违规，$\text{DS}=50\times1=50$，反而更高。这不只是假想：CARLA 官网写明，参赛者发现提前结束路线比跑完全程得分更高，所以 2025 年 3 月起的 2.1 版把违规扣分从连乘改成线性，2.1 和 2.0 的分数不能直接比；官网也显示 2025 年没有办挑战赛。读 DS 时要和路线完成率、违规次数一起看。

[Bench2Drive](https://arxiv.org/abs/2406.03877)（NeurIPS 2024）在 CARLA 0.9.15 里用 44 类场景、220 条路线分项测试闭环能力，数据集是 13,638 段片段、200 万帧。[README](https://github.com/Thinklab-SJTU/Bench2Drive) v0.0.4（2026-08-06）汇总的 DS 从 UniAD 的 38.69 到 LEAD 的 95.59；这是维护者汇总的表，各方法的训练设置不一定相同。

## 安全论证：几份标准各管什么

评测分数回答“它开得怎么样”，安全论证回答“凭什么相信它在设计运行范围内足够安全”。ISO 标准正文收费，页面也拒绝了我们的自动抓取，所以 ISO 和 SAE 几项只按标准名称和通常的定位来写，没有读到原文：

| 标准 | 管什么 | 和本篇的关系 |
| --- | --- | --- |
| ISO 26262（道路车辆功能安全，第 2 版 2018） | 电子电气系统出故障时引起的危险 | 传感器坏了、芯片出错；上文 Tesla 演讲提到的冗余芯片，按我们的理解属于这一类考虑 |
| ISO 21448:2022（预期功能安全，SOTIF） | 没有任何部件坏，但功能本身不够用、或可预见的误用引起的危险 | 策略 A 没有坏，只是学得不够，按我们的理解更接近这一类 |
| ISO/PAS 8800:2024（道路车辆的安全与人工智能） | 道路车辆里 AI 部件的安全 | 端到端网络正是这样的部件 |
| [UL 4600](https://www.ul.com/news/ul-4600-edition-3-updates-incorporate-autonomous-trucking)（第 3 版 2023-03-17） | 以安全论证为核心、按目标而不规定具体技术；不设固定的路测里程，要求组合分析、仿真、封闭场地和公共道路测试 | 告诉你一份论证需要什么证据，不规定用什么模型 |
| SAE J3016 | 驾驶自动化 0–5 级的分类 | 说清楚是辅助驾驶还是无人驾驶 |

端到端模型让“按部件逐项论证”变难：没有一个叫“感知”的模块可以单独出测试报告。Waymo 的做法是在车上另设一层独立的验证，并让同一个基础模型同时充当驾驶员、模拟器和评审（[博客](https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving)）；它也公开了安全方法论（[Webb 等 2020](https://arxiv.org/abs/2011.00054)）。另一条思路是把“什么算安全”写成形式化规则，例如 [RSS](https://arxiv.org/abs/1708.06374)。

## 截至 2026-10：部署与公开证据

上路运营是最终的闭环测试。各家的乘车量和城市数见[导读](README.md)；这里只列能拿来判断“闭环里表现如何”的公开证据。

> **截至 2026-10-15（变化很快，引用前请核对日期）**
>
> - **Waymo 的 [Safety Impact](https://waymo.com/safety/impact/)（数据截至 2026-06，公司自报）：** 2.713 亿英里 rider-only 里程；与同类城市道路上的人类驾驶基准相比，重伤及以上的事故少 95%（少 55 起），致伤事故少 82%。这是 Waymo 自己做的相对比较，重伤事故的绝对数很小，且只覆盖 5 个成熟市场。
> - **召回暴露的长尾（NHTSA 召回文件）：** Waymo [26E026](https://static.nhtsa.gov/odi/rcl/2026/RCLRPT-26E026-6527.pdf)（2026-05-06，3,791 辆）在较高车速道路上遇到积水车道，可能减速但不停车；Waymo 26E035（2026-06，3,871 辆）可能高速驶入封闭的高速公路施工区（读自[汇总站](https://vehiclesafetyrecalls.com/recall/26E035000)）；Zoox [26E044](https://static.nhtsa.gov/odi/rcl/2026/RCLRPT-26E044-2993.pdf)（2026-07-16，105 辆无人车）在事故现场可能识别不了浓烟。
>
> 积水、施工区、浓烟都是训练日志里少见的情况，开环 L2 测不出来。

## 读端到端论文时检查什么

| 先看什么 | 应该找到什么 | 缺了它，不能推出什么 |
| --- | --- | --- |
| 用的是哪个 benchmark 和版本 | NAVSIM v1 还是 v2，CARLA 2.0 还是 2.1 | 跨版本的分数可比 |
| 有没有去掉自车状态的消融 | 去掉后分数掉多少 | 模型真的用到了感知 |
| 闭环里的路线完成率和违规次数 | 和 DS 一起报告 | DS 高就是开得好 |
| 有没有和规则基线比 | PDM-Closed、匀速外推 | 学出来的规划器更好 |
| 数字是作者自报还是第三方 | 官方榜单或独立复现 | 结论能推广到别的设置 |
| 训练和测试的城市、路线是否分开 | 明确的划分 | 能泛化到新城市 |

参考：[Parting with Misconceptions（PDM）](https://arxiv.org/abs/2306.07962) · [DAgger](https://arxiv.org/abs/1011.0686)。上一篇：[预测与规划](prediction-planning.md)；返回[导读](README.md)。
