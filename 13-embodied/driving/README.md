# 自动驾驶：从传感器到一条轨迹

**中文** · [English](README.en.md)

> 阅读时间：约 7 分钟 · 草稿，待审核

一辆车上有 6–8 个摄像头，常常还有 lidar 和 radar，每秒产生几十帧数据；可车每隔几十毫秒只需要交出一样东西：未来几秒的一条轨迹，或者直接是方向盘、油门和刹车。中间发生了什么？**经典做法把这段路拆成感知、预测、规划、控制几层，每层的接口都能单独检查；端到端做法把它们训练成一个网络，中间输出变成可选的辅助头。不管哪种，最难的都不是在录好的日志上开得像人，而是证明它自己开的时候不出事。**

> **待审核草稿。** 这一篇还没有经过站主审阅，事实、取舍和措辞都可能修改；引用前请回到文中链接的一手资料核对。

## 一个场景，四个问题

先用一个场景把问题摆出来，第 2 篇的规划算例用的就是它：你在双车道上以 12 m/s（约 43 km/h）行驶，本车道前方 30 米有辆车只开 8 m/s，左车道后方有车正在靠近，右边是路缘。系统依次要回答：

1. 周围有什么、在哪里？这是感知，第 1 篇讲多相机怎样变成一张俯视图。
2. 前车和后车接下来几秒可能怎么走？这是预测，第 2 篇讲为什么要给出几种可能。
3. 我走哪一条？这是规划，第 2 篇用 4 条候选轨迹算一次。
4. 怎样让车真的沿这条轨迹走？这是控制，已在[规划与 LQR](../../05-post-training/deep-rl/planning-and-control.md)讲过。

## 经典的模块化栈

| 模块 | 输入 | 输出 | 常见方法 | 在哪里讲 |
| --- | --- | --- | --- | --- |
| 感知 | 多相机图像、lidar 点云、radar | 鸟瞰图特征、3D 框、车道线、占用网格 | Lift-splat、BEVFormer、BEVFusion、占用网络 | [第 1 篇](bev-occupancy.md)，相机几何见[感知系列](../perception/camera-geometry.md) |
| 跟踪与状态估计 | 每一帧的检测结果 | 带 ID 和速度的目标轨迹 | Kalman 滤波、数据关联 | [状态估计与跟踪](../perception/tracking.md) |
| 预测 | 目标的历史轨迹、地图 | 每个目标的几条未来轨迹及概率 | MTR、Wayformer、MotionLM | [第 2 篇](prediction-planning.md) |
| 规划 | 预测、地图、路线、自车状态 | 自车未来几秒的轨迹 | 搜索、采样后打分、优化 | [第 2 篇](prediction-planning.md) |
| 控制 | 目标轨迹、自车状态 | 方向盘、油门、刹车 | PID、LQR、MPC | [规划与 LQR](../../05-post-training/deep-rl/planning-and-control.md) |

拆开的好处是每个接口都能读、能单独测、能单独替换，出了事能查到是哪一层错的。代价是接口由人来定：感知只交出框，规划就看不到框描述不了的东西，比如路面上一块形状古怪的碎片；上游犯的错一层层传下去，下游也没法要求上游“为了规划而优化”。占用网格就是为前一个问题补上的，第 1 篇会讲。

## 端到端：把接口换成梯度

“端到端”在不同系统里指的东西不一样，至少有三种：

- **仍有中间任务，但一起训练。** [UniAD](https://arxiv.org/abs/2212.10156)（CVPR 2023 Best Paper）用 query 把跟踪、建图、运动预测、占用和规划串成一个网络，以规划为最终目标；各任务通过 query 相连，可以一起训练。
- **一个网络从像素到动作。** Tesla 在 [2023 年第四季度股东信](https://digitalassets.tesla.com/tesla-contents/image/upload/IR/TSLA-Q4-2023-Update.pdf)里说，V12 用 AI 来影响车辆控制（方向盘、踏板、转向灯等），而不是把每种驾驶行为都手写出来；Tesla 的 Ashok Elluswamy 在 2025 年 10 月 ICCV 的演讲中说，现在是一个大网络把像素映射到下一个动作，显式感知只作为可选的辅助输出（据演讲视频的自动字幕转述）。
- **大模型加结构化输出。** Waymo 2025-12-09 的[博客](https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving)描述的 Foundation Model 保留了物体、道路图等结构化输出，加了一个用 Gemini 训练的 Driving VLM，再把车上跑不动的大 Teacher 模型蒸馏成车载 Student，并在车上另设一层独立的验证。

| 对比 | 模块化栈 | 端到端模型 |
| --- | --- | --- |
| 接口 | 人定义：框、轨迹、地图 | 学出来的特征，结构化输出可选 |
| 训练信号 | 每个模块各有标签 | 规划或驾驶目标的梯度贯穿全网 |
| 出错后定位 | 能查到哪一层 | 要靠辅助头、探针、仿真重放 |
| 主要风险 | 接口丢信息，误差逐层传递 | 学到捷径：开环分数好看，闭环失败 |
| 主要验证方式 | 分模块测试加系统测试 | 闭环仿真、路测和安全论证 |

真实系统大多在两者之间。第 3 篇会说明，为什么“开环分数好看、闭环失败”在端到端模型上格外要紧。

## 这三篇怎么接起来

- **第 1 篇**回答“东西在哪”：多相机图像怎样变成一张米制的俯视网格，占用网格比框多了什么。
- **第 2 篇**回答“别人会怎么走、我走哪条”：预测为什么要给几种可能，候选轨迹怎样按代价挑，规则规划器为什么仍是强基线。
- **第 3 篇**回答“怎样判断一个把这些都做了的模型好不好”：为什么开环分数会骗人，闭环和准闭环评测各测什么，上路前要有什么安全论证。

## 先读什么

- 感知系列的[相机几何](../perception/camera-geometry.md)（本系列第 1 篇直接用它）、[状态估计与跟踪](../perception/tracking.md)和[检测与分割](../perception/detection.md)。
- MPC 和规划器钻模型漏洞：[Model-based RL](../../05-post-training/deep-rl/model-based.md)；LQR 和约束：[规划与 LQR](../../05-post-training/deep-rl/planning-and-control.md)；行为克隆的误差累积：[模仿与奖励](../../05-post-training/deep-rl/imitation-and-rewards.md)，第 3 篇的开环、闭环对比建立在它上面。
- 世界模型当驾驶模拟器：[世界模型：机器人与驾驶](../../11-frontier/world-models/robotics-driving.md)。

想做自动驾驶方向的话，一条顺序是：感知系列第 1 篇（相机几何）→ 感知系列第 2 篇（状态估计与跟踪）→ 本系列第 1、2、3 篇，最后补感知系列第 5 篇[三维重建与神经渲染](../perception/reconstruction.md)。

## 截至 2026-10：领域状态与部署

> **截至 2026-10-15 的状态（变化很快，引用前请核对日期）**
>
> - **学出来的模拟器成了闭环测试的主要方向。** Tesla 在 2025 年 10 月 ICCV 和 2026 年 6 月 CVPR 的演讲中介绍神经世界模拟器（据演讲视频的自动字幕）；[Waymo World Model](https://waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation)（2026-02-06）；Wayve [GAIA-4](https://wayve.ai/thinking/gaia-4/)（2026-08-03）。详见[世界模型系列](../../11-frontier/world-models/robotics-driving.md)。
> - **nuScenes 开环 L2 和碰撞率基本退场。** [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) 的维护者已停止报告这两项；2026 年的论文多报告 NAVSIM v1/v2、HUGSIM、nuPlan 和 Bench2Drive。原因见第 3 篇。
> - **开放权重的驾驶 VLA 出现了。** NVIDIA 的 [Alpamayo 2 Super](https://www.nvidia.com/en-us/solutions/autonomous-vehicles/alpamayo/llm-info/)（34B）在 2026-08-04 发布权重，许可证为 OpenMDW-1.1；它的性能数字都是 NVIDIA 自报。

下面的运营数字全部是公司自报，除非另有说明；它们各自截至不同日期，不能直接横向比较。

| 公司 | 截至 | 公开说法 | 性质 |
| --- | --- | --- | --- |
| Waymo | 2026-04-29 | 每周超过 50 万次完全自动驾驶乘车（Pichai 在 Alphabet 第一季度[财报稿](https://www.sec.gov/Archives/edgar/data/0001652044/000165204426000043/googexhibit991q12026.htm)里的说法） | 公司自报；截至 2026-10-15 我们没找到更新的周数据 |
| Waymo | 2026-10-15 | [更新页](https://waymo.com/updates/)列出 15 个向公众服务的美国城市 | 公司自述 |
| Waymo | 2026-06 | 累计 2.713 亿英里 rider-only 里程（[Safety Impact](https://waymo.com/safety/impact/)） | 公司自报 |
| Tesla | 2026-07-22 | 2026 年第二季度[股东信](https://www.sec.gov/Archives/edgar/data/1318605/000162828026049213/exhibit991.htm)说 Robotaxi 已在 7 个大都市区运营：湾区有安全员；Austin、Dallas、Houston、Miami、Orlando、Tampa 标为 “Ramping Unsupervised” | Tesla 自述；信中没有说任何城市已全部无人监督 |
| Tesla | 2026-09-17 | 加州 DMV [名单](https://www.dmv.ca.gov/portal/vehicle-industry-services/autonomous-vehicles/autonomous-vehicle-testing-permit-holders/)上，Tesla Robotaxi LLC 只持有带安全员的测试许可 | 监管名单 |
| Zoox | 2026-07-31 | NHTSA 依 Part 555 给予豁免，可商业部署，每 12 个月最多 2,500 辆（[联邦公报](https://public-inspection.federalregister.gov/2026-15485.pdf)） | 监管文件 |
| 百度 Apollo Go | 2026 Q1 / Q2 | Q1 完成 320 万次全无人乘车（[季度新闻稿](https://www.sec.gov/Archives/edgar/data/0001329099/000119312526228123/d156481dex991.htm)）；Q2 约 100 万次，李彦宏在 2026-08-18 的电话会上归因于部分国内城市的“运营调整” | 公司自报；Q2 数字据报道来自电话会记录 |

## 这一组怎么读

1. [从环视相机到鸟瞰图（BEV）与占用](bev-occupancy.md)：lift-splat 手算、BEV query、稀疏 query、camera–lidar 融合、占用网格和它的标签
2. [预测与规划](prediction-planning.md)：minADE / minFDE 手算、联合与边缘预测、搜索与采样规划器、4 条候选轨迹按代价挑一条、PDM-Closed
3. [端到端驾驶与闭环评测](end-to-end.md)：端到端的几种含义、模仿与 self-play、开环与闭环手算、NAVSIM / Bench2Drive / CARLA、安全标准、部署情况
