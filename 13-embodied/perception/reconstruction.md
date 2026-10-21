# 感知与几何：三维重建与神经渲染（NeRF、3DGS）

**中文** · [English](reconstruction.en.md)

> 阅读时间：约 13 分钟 · 最近审阅：2026-10

车队录下一段 20 秒的多相机日志。你改了规划器，想知道新规划器在同一时刻会不会换道，换道后相机看到什么。直接回放录像不行：车一旦开得和录像不同，画面就对不上了（开环和闭环的区别见[世界模型一篇](../../11-frontier/world-models/robotics-driving.md)）。能不能先把这段路重建成三维场景，再从新位置渲染？**能，但只在日志看到过的地方可信。3D Gaussian splatting（3DGS）把重建做到了实时渲染、可以按物体编辑，所以成了驾驶仿真的底座；车开到日志没看到的位置，重建就没有数据可用，这正是生成式世界模型想补的缺口。**

需要先读本系列的[相机几何](camera-geometry.md)和[冻结骨干与几何基础模型](foundation-models.md)：这里沿用那两篇的针孔投影、pointmap 和尺度不确定。

## 第一步：SfM 先求相机和稀疏点

重建要先知道每张图是从哪里、用什么相机拍的。structure-from-motion（SfM）只靠图片本身求出这些。[Schönberger 和 Frahm 的 CVPR 2016 论文](https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf)随论文开源了 COLMAP，后面要讲的 3DGS 论文，输入相机就是用它标定的。增量 SfM 的流程是：

```text
提取特征并两两匹配 → 几何验证，去掉对不上的图对
→ 选一对图做初始两视图重建
→ 循环：用 PnP 注册一张新图 → 三角化新点 → bundle adjustment → 过滤外点
```

Bundle adjustment（BA）是这一循环里的总调整。论文式 (1) 写的是：同时调整所有相机参数 $P_c$ 和三维点 $X_k$，让重投影误差最小：

$$
E=\sum_j\rho_j\Big(\big\|\pi(P_c,X_k)-x_j\big\|_2^2\Big)
$$

$x_j$ 是第 $j$ 个观测，也就是点 $X_k$ 在相机 $c$ 里被检测到的像素位置；$\pi$ 是投影函数；$P_c$ 包括位姿、焦距和畸变；$\rho_j$ 是稳健损失，用来压低错配的影响，COLMAP 在局部 BA 里用的是 Cauchy 函数。求解一般用 Levenberg–Marquardt，并利用 Schur 补先解相机、再回代求点。

[相机几何](camera-geometry.md)一篇的三角化，最后一步就是最小化一个点的重投影误差；BA 把所有相机也一起放进来调。算一次（教学例子，只看水平方向）。沿用那一篇的教学相机，$f=1000$、$c_x=960$；两台相机朝向相同、都朝 $+z$，相机 1 在原点，相机 2 在 $x=1$ m。为了能手算，固定点的深度 $Z=5$ m，只求它的横坐标 $X$。相机 1 看到 $u_1=1166$，单看它会推出 $X=(1166-960)\times5/1000=1.03$；相机 2 看到 $u_2=960$，单看它会推出 $X=1.00$。两个观测带噪声，互相矛盾。代入投影公式，总误差是：

$$
E(X)=(200X-206)^2+(200X-200)^2
$$

$X=1.03$ 时 $E=0+36=36$，$X=1.00$ 时 $E=36+0=36$，$X=1.015$ 时两项残差都是 3 px，$E=18$ 最小。BA 做的就是这种折中，只是同时动所有相机、所有点、焦距和畸变。如果相机 2 的观测其实是错配，落在 $u_2=1080$，平方损失会把点拖到 $X=1.315$，相机 1 的残差变成 57 px；稳健损失让大残差的代价增长得更慢，错配就拖不动整体。

还有一件 BA 改不了的事：把所有相机中心和所有点一起放大 $k$ 倍（再加上整体旋转、平移），每个像素的重投影都不变。上例里相机 2 移到 $x=k$、点移到 $(kX,kZ)$，$u_2=1000(kX-k)/(kZ)+960$ 和原来一样。只靠图片，SfM 的结果只到尺度，这和[上一篇的单张深度](foundation-models.md)是同一个 $k$。驾驶日志的相机一般装在标定好的支架上（[Instant NuRec](https://arxiv.org/abs/2607.14203) 的输入就是标定过的多相机），基线长度已知，尺度就定了。

上一篇的前馈模型也能接到这里。[VGGT 的 README](https://github.com/facebookresearch/vggt)（2025-06-02 一条）提供脚本，把预测结果存成 COLMAP 格式，可选再跑一次 BA，然后直接交给 gsplat 等 Gaussian splatting 库。

## 第二步：从稀疏点到能渲染的场景

SfM 只给出相机和一组稀疏点，还不能从新视角渲染一张完整的图。NeRF 和 3DGS 都在补这一步，而且[3DGS 论文](https://arxiv.org/abs/2308.04079)（Inria 与 MPII，2023-08）§2 指出，两者算像素颜色的公式其实是同一个：沿着视线把一串半透明的东西从前往后叠起来。

$$
C=\sum_{i=1}^{N}T_i\,\alpha_i\,\mathbf c_i,\qquad T_i=\prod_{j=1}^{i-1}(1-\alpha_j)
$$

$\mathbf c_i$ 是第 $i$ 个元素的颜色，$\alpha_i$ 是它挡住光的比例，$T_i$ 是光走到它面前时还剩多少（透射率）。区别在 $\alpha_i$ 从哪里来：

$$
\alpha_i^{\rm NeRF}=1-e^{-\sigma_i\delta_i},\qquad
\alpha_i^{\rm 3DGS}=o_i\exp\!\Big(-\tfrac12(\mathbf p-\boldsymbol\mu_i)^\top\Sigma_i^{-1}(\mathbf p-\boldsymbol\mu_i)\Big)
$$

- [NeRF](https://arxiv.org/abs/2003.08934)（2020-03）用一个 MLP 把位置和视角方向映射成密度 $\sigma$ 和颜色。渲染一个像素要沿射线取很多样本，每个样本查一次 MLP，$\delta_i$ 是相邻样本的间距。它需要已知位姿的图片，并且每个场景单独优化。
- 3DGS 把场景存成一大堆显式的三维 Gaussian：位置、协方差 $\Sigma^{\rm 3D}=RSS^\top R^\top$（旋转乘缩放）、不透明度 $o_i$，以及用球谐系数表示、随视角变化的颜色。渲染时把每个 Gaussian 投到屏幕上，得到二维的均值 $\boldsymbol\mu_i$ 和协方差 $\Sigma_i$；上式在像素位置 $\mathbf p$ 处取值，再乘不透明度。初始 Gaussian 来自 SfM 的稀疏点；训练时最小化渲染图和真实图之间的 $(1-\lambda)L_1+\lambda L_{\text{D-SSIM}}$（$\lambda=0.2$），并不断复制、拆分、删除 Gaussian 来调整密度。

3DGS 快在渲染方式。屏幕先切成 16×16 像素的 tile，每个 Gaussian 按“tile 编号加深度”做一次 GPU 基数排序，然后每个 tile 内的像素从前往后叠加，透射率快用完就停。不用沿射线反复查网络，也不用逐像素排序。

<!-- paper-figure:3dgs-pipeline -->

## 算一次：四个 Gaussian 合成一个像素

教学例子。一个像素被四个 Gaussian 覆盖，已经按深度从近到远排好。$g$ 是该 Gaussian 在这个像素处的取值 $\exp(-\tfrac12m^2)$，$m$ 是像素到 Gaussian 中心的 Mahalanobis 距离：像素正好在中心时 $g=1$；$m^2=2\ln2\approx1.39$ 时 $g=0.5$。

| 编号 | 深度 | 颜色 (R, G, B) | 不透明度 $o$ | 本像素的 $g$ | $\alpha=o\,g$ |
| --- | --- | --- | --- | --- | --- |
| 1 | 4 m | (1, 0, 0) 红 | 0.5 | 1.0 | 0.5 |
| 2 | 6 m | (0, 1, 0) 绿 | 0.8 | 0.5 | 0.4 |
| 3 | 9 m | (0, 0, 1) 蓝 | 1.0 | 0.5 | 0.5 |
| 4 | 20 m | (1, 1, 1) 白 | 0.9 | 1.0 | 0.9 |

从前往后一个个叠：

| 步 | 进来时的 $T$ | 权重 $w=T\alpha$ | 出去时的 $T$ | 累计颜色 |
| --- | --- | --- | --- | --- |
| 1 | 1 | 0.5 | 0.5 | (0.5, 0, 0) |
| 2 | 0.5 | 0.2 | 0.3 | (0.5, 0.2, 0) |
| 3 | 0.3 | 0.15 | 0.15 | (0.5, 0.2, 0.15) |
| 4 | 0.15 | 0.135 | 0.015 | (0.635, 0.335, 0.285) |

四个权重加起来 0.985，等于 $1-0.015$；剩下的 1.5% 留给背景（这里设为黑色）。最终颜色 (0.635, 0.335, 0.285)，偏红。

<!-- widget:step-alpha-blend -->

再看三件事：

- **顺序很重要。** 如果排序出错，第 1 个和第 4 个对调，最远的白色 Gaussian 最先叠，顺序变成白、绿、蓝、红，权重变成 0.9、0.04、0.03、0.015，颜色是 (0.915, 0.94, 0.93)，几乎全白：远处的东西透到了前面。剩下的透射率还是 0.015，因为 $\prod(1-\alpha_j)$ 和顺序无关；颜色却和顺序有关。3DGS 每个 tile 只排一次序，不逐像素排，论文 §6 承认这时混合是近似的，Gaussian 缩到像素大小时误差可以忽略；局限一节里提到的“popping”，部分原因就是 Gaussian 的前后顺序突然对调。
- **同样的权重也能渲染深度，但结果可能落在空处。** $\sum w_iz_i=0.5\times4+0.2\times6+0.15\times9+0.135\times20=7.25$，除以 0.985 约为 7.36 m。这个位置上什么都没有，它是几个表面的加权平均。把这样的深度当成激光雷达读数用，要先想清楚。
- **数值细节。** 论文附录写明：$\alpha<1/255$ 的项直接跳过，$\alpha$ 上限截到 0.99，累计不透明度快超过 0.9999 时停止。上例累计到 0.985，四个都会叠进去。

上面每一个量对 $o$、$g$（也就是位置和协方差）和颜色都可导，训练就是让这样算出的颜色贴近真实照片。

## NeRF 还是 3DGS

| | NeRF（2020） | 3DGS（2023） |
| --- | --- | --- |
| 场景存成什么 | 一个 MLP：位置和视角方向 → 密度和颜色 | 一大堆显式的各向异性 Gaussian |
| 一个像素怎么算 | 沿射线采样，每个样本查一次 MLP | 投影、按 tile 排序、从前往后叠 |
| 渲染速度 | 3DGS 论文指出，NeRF 要靠代价高的随机采样找样本，带来噪声和计算开销 | 1080p 实时：arXiv v1 写 ≥30 fps，项目页写 ≥100 fps，两处说法不一 |
| 显存 | 主要是网络权重 | 3DGS 论文自述明显高于 NeRF 类方法；未优化的原型训练大场景时峰值可超过 20 GB |
| 挪走一辆车 | 场景藏在网络权重里，没有现成的“这辆车”单元 | 一辆车就是一组 Gaussian，可以整组移动 |
| 相机模型 | 逐射线采样，畸变相机只是换一组射线方向 | 光栅化假设理想针孔，见下一节的 3DGUT |

## 为什么 3DGS 成了驾驶仿真的底座

按时间看几块拼图（数字都是作者或厂商自报）：

1. **快。** [Street Gaussians](https://arxiv.org/abs/2401.01339)（2024-01，ECCV 2024）报告在 KITTI 和 Waymo 上训练约 30 分钟后，以 1066×1600 分辨率 135 FPS 渲染；[OmniRe](https://arxiv.org/abs/2408.16760)（2024-08，ICLR 2025）报告约 60 Hz 的仿真。闭环仿真每一步都要重新渲染，速度是前提。
2. **能按物体编辑。** Street Gaussians 把背景和每辆车分成不同的 Gaussian 组，车跟着跟踪到的位姿走；OmniRe 把车辆、行人（用 SMPL 人体模型驱动的 Gaussian）和骑车人组织成场景图。挪车、删车、插入新车，是造“假如”场景的基础。
3. **像真实的相机。** [3DGUT](https://arxiv.org/abs/2412.12507)（NVIDIA，2024-12，CVPR 2025）用 unscented transform 代替原版 3DGS 的 EWA splatting 投影近似，支持带畸变和卷帘快门的相机，以及反射、折射这类二次光线，同时保持光栅化的速度。
4. **建场景只要几秒。** [Instant NuRec](https://arxiv.org/abs/2607.14203)（NVIDIA，2026-07-15）是前馈模型：把 10–20 秒的多相机日志直接变成分层的静态和动态 3DGS，外加天空立方体贴图和每个相机的 ISP 校正，约 1.5 秒；通过 3DGUT 支持非针孔相机，能接 AlpaSim 做闭环仿真。作者报告在 Waymo Open 上 PSNR 比最强基线高 2.01 dB；摘要没说测试视角是否在日志轨迹上。
5. **成了产品。** NVIDIA Omniverse NuRec 在 2026-03-25 正式发布（[开发者论坛公告](https://forums.developer.nvidia.com/t/nvidia-omniverse-nurec-now-generally-available/364767)）：参考容器、基于 gsplat 的渲染器、NCore 数据格式、基于 Cosmos 的 Fixer 模型（清理新视角里的伪影），以及生成未观测区域资产的 Asset Harvester。更早的 2025-10-28 [NVIDIA 技术博客](https://developer.nvidia.com/blog/accelerating-av-simulation-with-neural-reconstruction-and-world-foundation-models/)说 CARLA 已接入 NuRec 的渲染接口，并列出 Foretellix、MathWorks、Mcity 等合作方（公司自报）。
6. **进了评测基准。** [NAVSIM v2](https://arxiv.org/abs/2506.04218)（2025-06，CoRL 2025）的伪仿真分两段：先给规划打分，再在 3DGS 渲染的扰动起点上评测，按离第一段终点的远近加权。[HUGSIM](https://arxiv.org/abs/2412.01718)（2024-12）的摘要称它是建在 3DGS 上的实时闭环仿真器，我们只核对了摘要。

不是每家都公开用的是什么。Wayve 的 [PRISM-1](https://wayve.ai/thinking/prism-1/)（2024-06）做视频的四维重建，给它的 Ghost Gym 重仿真器用，但没说是 Gaussian 还是 NeRF；Wayve 自己写到骑车人很难重建，新场景要靠生成模型 GAIA-1。截至 2026-10-15，我们没有找到 Waymo 公开的 3DGS 流程。

还有一条不用逐场景优化的线：[pixelSplat](https://arxiv.org/abs/2312.12337)（2023-12，图像对）、[MVSplat](https://arxiv.org/abs/2403.14627)（2024-03）、[DepthSplat](https://arxiv.org/abs/2410.13862)（2024-10，作者称 12 张 512×960 的图 0.6 秒）、[AnySplat](https://arxiv.org/abs/2505.23716)（2025-05，未标定的图集一次前向，同时预测内外参）。Instant NuRec 是这条线在驾驶日志上的版本，Depth Anything 3 也能直接输出 3DGS。

## 车开出日志之外，重建就没有数据

用俯视图算一次（教学例子，$x$ 向右、$z$ 向前，单位米）。日志里的车在红灯前缓慢跟车，相机从 $(0,0)$ 挪到 $(0,5)$。右前方停着一辆面包车，车尾在 $z=15$ 处，横跨 $1.5\le x\le3.5$。面包车前面 $(3,20)$ 站着一个行人，准备过马路。相机到行人的视线，在 $z=15$ 处的横坐标是 $x_c+(3-x_c)(15-z_c)/(20-z_c)$：

| 相机位置 $(x_c,z_c)$ | 视线在 $z=15$ 处的 $x$ | 被面包车挡住吗 |
| --- | --- | --- |
| (0, 0)，日志起点 | 2.25 | 挡住 |
| (0, 5)，日志终点 | 2.0 | 挡住 |
| (−3.5, 0)，规划器改成向左换道 3.5 m | 1.375 | 看得见 |

日志全程这个值在 2.25 和 2.0 之间，行人从没出现在任何一帧里，重建出的场景里也就没有代表他的 Gaussian。换道后的新视角本该看见他，渲染出来的却是空路面，或者一团没被约束过的 Gaussian。3DGS 论文自己的[局限一节](https://arxiv.org/html/2308.04079v1#S7.SS4)也写到：“In regions where the scene is not well observed we have artifacts …”，即观测不充分的区域会出伪影。Waymo 在 2026-02-06 的[博客](https://waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation/)里说的正是这件事：模拟路线和原始驾驶差得太多时，纯重建方法（例如 3DGS）会因为缺观测而出现画面崩坏。

<!-- paper-figure:3dgs-sparse-views -->

| 做法 | 例子 | 补上了什么 | 代价 |
| --- | --- | --- | --- |
| 只在日志附近评测 | NAVSIM v2 在 3DGS 渲染的扰动起点上评测，按离第一段终点的远近加权 | 我们的理解：起点离已观测区域近，渲染更可信 | 测不到大幅偏离 |
| 用生成模型清理和补全新视角 | NuRec 的 Fixer（基于 Cosmos）和 Asset Harvester | 去掉新视角里的伪影，补上没观测到的物体 | 补出来的内容没有被观测过 |
| 整个换成生成式世界模型 | Waymo World Model（2026-02-06，在 Genie 3 上后训练，输出相机和 lidar）；Wayve [GAIA-4](https://wayve.ai/thinking/gaia-4/)（2026-08-03） | 任何位置都能给出看起来合理的画面 | 两家都没有公布能核对的保真度数值（GAIA-4 只给了一个没说基线的“2.5 倍”）；可能凭空多出或漏掉东西 |

Waymo 的博客称它的模型因为生成能力强，能保持真实和一致，但没有给定量对比，这属于公司自报。还要看清一点：上例的行人在这段日志里从来没被看见过，任何方法都无法从这段日志知道那里有人。生成模型能画出一个合理的行人，也能画出空路面；哪一个对，这段日志核对不了。重建给的是“看到过的地方忠实”，生成给的是“哪里都像样”，两者回答的是不同的问题。世界模型怎样当仿真器和评测器，以及它们自己的验证难题，见[世界模型：给机器人和汽车当模拟器](../../11-frontier/world-models/robotics-driving.md)；为什么规划器必须闭环评测，见[端到端驾驶与闭环评测](../driving/end-to-end.md)。

## 动手之前

| 先试什么 | 应该观察什么 | 不能据此推出什么 |
| --- | --- | --- |
| 把例子里第 1 和第 4 个 Gaussian 的顺序对调 | 颜色从 (0.635, 0.335, 0.285) 变成接近白色，剩余透射率仍是 0.015 | 排序错误总会被看出来（深度接近的 Gaussian 对调，差别可能很小） |
| 在日志轨迹上的留出帧算 PSNR | 重建在原路线附近的质量 | 向左换道 3.5 m 后的画面一样好 |
| 把渲染相机从日志路线逐步横移 | 被遮挡的区域开始露出，伪影增多 | 生成模型补出来的内容就是真实的 |
| 只用图片跑 SfM 和 BA | 相机和点的相对位置 | 米制尺度（要靠标定基线、GPS/IMU 或激光雷达） |

## 截至 2026 年 10 月

- **许可。** 原版 3DGS 代码是 Inria/MPII 许可，仅限非商业研究和评估；[gsplat](https://arxiv.org/abs/2409.06765) 是 Apache-2.0（作者报告比原版最多少 10% 训练时间，显存少到四分之一）；3DGUT 的代码 3dgrut 是 Apache-2.0；OmniRe 所在的 drivestudio 是 MIT；Street Gaussians 用自定义的 Project Registration License v1.0（2026）。NuRec 的 Fixer 用 NVIDIA Open Model License；Instant NuRec 和 NuRec 参考容器的许可我们没有核对。
- **产品节点。** NuRec 正式发布于 2026-03-25，Instant NuRec 论文发布于 2026-07-15，Waymo World Model 发布于 2026-02-06，GAIA-4 发布于 2026-08-03。
- **没公开的。** Waymo 没有公开自己的 3DGS 流程；Wayve 没说 PRISM-1 用的是哪种表示。

回到[本系列导读](README.md)；想把相机和点云的来历补全，读上一篇[冻结骨干与几何基础模型](foundation-models.md)。
