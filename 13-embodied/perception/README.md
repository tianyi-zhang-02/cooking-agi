# 感知与几何：从像素到三维

**中文** · [English](README.en.md)

> 阅读时间：约 7 分钟 · 最近审阅：2026-10

车的前视相机拍到一帧画面。里面有什么？它们在三维里的哪儿？和上一帧看到的是不是同一个？这三个问题分别交给检测、几何和跟踪。**这些岗位的底子不是某一个网络，而是两门老手艺：几何（像素和三维之间怎么换算）和估计（带着不确定度把多次观测合起来）。2026 年的大模型已经能直接输出相机、深度和点图，但读懂、检查和使用这些输出，还是靠这两样。**

## 为什么离不开几何和估计

- **一个框只活在像素里。** 检测器说“行人在 (860, 690)”，规划器要的是“前方 10 m、偏左 1 m”。中间要用内参、外参，再加一个约束：地面、第二台相机或深度模型（第 1 篇）。
- **外参的小误差会放大。** 多台相机要靠外参放进同一个车体坐标，才能拼成鸟瞰图。第 1 篇算过：相机俯仰差 1°，靠地面测出的 10 m 会变成约 8.9 m 或 11.3 m。
- **检测每帧独立，规划要的是轨迹。** 位置、速度、不确定度和身份要跨帧维持，这是滤波和数据关联（第 2 篇）。
- **新模型输出的就是几何量。** VGGT、Depth Anything 3、MapAnything 这类前馈模型一次前向给出相机内外参、深度和点图（第 4 篇）；3D Gaussian splatting 做驾驶仿真，先要有准确的相机模型和位姿（第 5 篇）。

公开的岗位描述也是这样写的。我们在 2026-10-15 读了 13 份车、机器人和人形机器人方向的公开岗位描述，粗略统计（只看有没有提到，不看分量）：几乎每一份都要求把东西部署到真车、真机器人或生产代码里；约 8 份提到实时或面向硬件的优化；约 5 份点名三维几何视觉、SLAM/VIO、传感器融合或 BEV。这里只汇总知识领域，不点名公司，也不涉及任何招聘流程或面试内容。

## 五篇怎么分工

| 篇 | 回答什么问题 | 用哪个例子走一遍 | 站内前置 |
| --- | --- | --- | --- |
| 1 一个三维点怎样变成像素 | 内参、外参、畸变各做什么；什么时候一个 homography 就够；三角化怎样找回深度 | 车体坐标里的一点投到像素 (860, 690)，再用地面和第二台相机找回 10 m | 线性代数 |
| 2 状态估计与跟踪 | 怎样跨帧维持位置和身份；ByteTrack 为什么要用低分框；滤波和优化两种 SLAM | 一维 Kalman 的一次预测和修正；3×3 IoU 矩阵的 Hungarian 配对 | 概率（条件概率、高斯）；第 1 篇 |
| 3 2026 年的检测与分割 | AP 怎么算；DETR 和 YOLO26 怎样去掉 NMS；闭集实时、开放词表和 SAM 3 怎么选；哪些权重能商用 | 3 个真值、5 个预测，一路算到 AP@[.5:.95] | ViT、CLIP |
| 4 冻结骨干与几何基础模型 | 自监督特征为什么在稠密任务上强；相对深度和度量深度；什么是点图；一次前向怎样给出相机、深度和点 | 点 (1, 0.5, 5) m 放大 2 倍仍落在同一像素；用一两个已知距离定尺度；从点图反推焦距 | ViT、CLIP；第 1 篇 |
| 5 三维重建与神经渲染 | SfM 和 bundle adjustment 解决什么；NeRF 和 3DGS 的区别；3DGS 为什么成了驾驶仿真的基础设施，又在哪里失效 | 两台相机折中一个点的位置；四个 Gaussian 叠成一个像素；换道后才露出的行人 | 第 1、4 篇；世界模型系列 |

## 站内前置和外部教材

- [ViT](../../03-multimodal-learning/vit.md) 和 [CLIP](../../03-multimodal-learning/clip.md)：第 3、4 篇的骨干和图文对齐。
- 概率：第 2 篇用到条件概率和高斯分布。
- [世界模型怎样给机器人和汽车当仿真器](../../11-frontier/world-models/robotics-driving.md)：第 5 篇讲重建式仿真时的对照。
- 免费教材：[Szeliski《Computer Vision》第 2 版](https://szeliski.org/Book/)、[Barfoot《State Estimation for Robotics》第 2 版](https://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser24.pdf)；Hartley & Zisserman 只有[样章](https://www.robots.ox.ac.uk/~vgg/hzbook/)免费，其中包括对极几何一章。
- 课程：CMU [16-385](http://16385.courses.cs.cmu.edu/)（相机模型、两视图几何、跟踪）和 [16-822](https://geometric3d.github.io/)（射影几何、基础矩阵、基于学习的 SfM/SLAM），两门都有 2026 年秋季版；[Stanford CS231n](https://cs231n.stanford.edu/) 2026 年春季版覆盖检测、分割和三维视觉。

## 按岗位怎么读

- **感知 / CV 岗：** 1 → 3 → 4 → 2，需要重建或仿真时再读 5。
- **自动驾驶岗：** 1 → 2，然后进[自动驾驶系列](../driving/README.md)（从[环视相机到鸟瞰图与占用](../driving/bev-occupancy.md)开始），涉及仿真时回到 5。
- **机器人 / 操作岗：** 1，然后进[机器人学习系列](../robot-learning/README.md)；做移动机器人再加读 2。

## 截至 2026-10：这两年变了什么

下面是 2024 年到 2026 年 10 月的主要变化，都会过时，引用时连日期一起引。

- **分割接受“概念”，不只接受点击。** [SAM 3](https://arxiv.org/abs/2511.16719)（arXiv 2025-11-20）给一个名词短语或示例，返回图像和视频里所有匹配实例的掩码和 ID；SAM 3.1（2026-03-27）加入多目标跟踪。许可证也变了：SAM 1、2 是 Apache-2.0，SAM 3 用自定义的 SAM License（[仓库](https://github.com/facebookresearch/sam3)）。
- **自监督骨干到了 7B，卖点变成稠密特征。** [DINOv3](https://arxiv.org/abs/2508.10104)（2025-08-13）是在 LVD-1689M 上训练的 ViT-7B，用 Gram anchoring 防止长时间训练中稠密特征退化；冻结骨干时作者报告 COCO 检测 66.1 mAP、ADE20k 63.0 mIoU。
- **深度从单张图走到任意多视角，并带上尺度。** [Depth Anything 3](https://arxiv.org/abs/2511.10647)（2025-11-13）从任意数量的视角预测一致的深度和光线；[MoGe](https://github.com/microsoft/MoGe) 的第 2 版加入度量尺度，MoGe-3 权重 2026-08-18 发布。很多最强的深度权重是非商用许可。
- **前馈三维重建成了主线。** [VGGT](https://github.com/facebookresearch/vggt) 获 CVPR 2025 最佳论文；CVPR 2026 最佳论文给了 D4RT（动态 4D 重建），SAM 3D 获荣誉提名（[CVPR 2026 闭幕页](https://cvpr.thecvf.com/Conferences/2026/News/Closing)）。
- **3D Gaussian splatting 成了驾驶仿真的生产基础设施。** NVIDIA Omniverse NuRec 2026-03-25 正式可用（[NVIDIA 论坛公告](https://forums.developer.nvidia.com/t/nvidia-omniverse-nurec-now-generally-available/364767)）；[Instant NuRec](https://arxiv.org/abs/2607.14203)（2026-07-15）作者报告用约 1.5 s 从 10–20 s 的多相机日志建出场景。Waymo 持相反看法：它基于 Genie 3 的生成式 World Model（2026-02-06）声称能处理离开记录路线的反事实场景，但没有给定量对比（公司自报，[Waymo 博客](https://waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation/)）。
- **实时检测分成两派。** DETR 式的 [RF-DETR](https://arxiv.org/abs/2511.09554)（ICLR 2026）声称是第一个在 COCO 上超过 60 AP 的实时检测器，DEIMv2 接上了 DINOv3 特征；Ultralytics 的 [YOLO26](https://docs.ultralytics.com/models/yolo26/)（2026-01-14 发布）加了不需要 NMS 的推理头（文档里预测默认仍走带 NMS 的头），YOLO27 截至 2026-10 只是预告。
- **开放词表检测并进分割模型，并换到更难的考场。** SAM 3 论文直接报 COCO、LVIS 的框 AP，并称之为零样本结果；[RF100-VL](https://arxiv.org/abs/2505.20612) 显示视觉语言检测器在一些医学数据集上零样本低于 2%。
- **Benchmark 文化变了。** COCO 框 AP 仍出现在每篇检测论文里，但只是和 RF100-VL、ODinW13、LVIS 或 SA-Co 并列的一列；Papers with Code 据报道在 2025 年 7 月停止服务。
- **用权重前先查许可证。** DUSt3R、MASt3R、CUT3R 是 CC BY-NC-SA 4.0；Fast3R 和 VGGT-Ω 是 FAIR 非商用研究许可；π^3 权重和 CoTracker3 是 CC BY-NC 4.0；原版 3DGS 代码是 Inria 非商用许可；Ultralytics 模型是 AGPL-3.0 或企业许可。检测模型的许可表在第 3 篇，深度和三维模型的在第 4 篇。

## 这一组怎么读

1. [一个三维点怎样变成像素](camera-geometry.md)：针孔模型、内外参、畸变、对极约束和三角化，用同一个点手算一遍
2. [状态估计与跟踪](tracking.md)：Kalman filter、Hungarian 配对、ByteTrack 的低分框，以及滤波和优化两种 SLAM
3. [2026 年的检测与分割](detection.md)：手算 AP，NMS 和一对一匹配，闭集实时、开放词表和 SAM 3 怎么选，常见权重的许可证
4. [冻结骨干与几何基础模型](foundation-models.md)：DINOv3、Depth Anything 3、VGGT，以及一次前向输出相机、深度和点图
5. [三维重建与神经渲染（NeRF、3DGS）](reconstruction.md)：从 SfM 到 Gaussian splatting 仿真器，以及重建在哪里不够用
