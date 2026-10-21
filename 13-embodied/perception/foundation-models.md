# 感知与几何：冻结骨干与几何基础模型

**中文** · [English](foundation-models.en.md)

> 阅读时间：约 13 分钟 · 最近审阅：2026-10

给一台仓储机器人加两样能力：分出地面上哪里能走，估计前方障碍物有多远。标注图只有几千张，训不起一个大骨干，于是打算冻结一个预训练 ViT，只在上面训一个小头。该选 CLIP 式的图文模型，还是 DINO 这类自监督模型？深度模型吐出来的数，能不能直接当米用？**按作者报告的冻结骨干对比，像素级任务上 DINO 系特征明显好于 CLIP 式特征；单张图的相对深度天生少一个尺度，要从别处补一个已知距离；2025 年以来的几何基础模型一次前向就能给出相机、深度和点云，但不少最强的 checkpoint 不能商用。**

先读过 [ViT](../../03-multimodal-learning/vit.md)、[CLIP](../../03-multimodal-learning/clip.md) 和本系列的[相机几何](camera-geometry.md)会更顺：这里默认你知道内参 $K$、外参 $[R\mid t]$ 和针孔投影。

## 冻结骨干，只训一个头

冻结骨干是指预训练网络的权重一个不动，只把它当特征提取器。图片切成 patch，ViT 给每个 patch 输出一个向量。分类只用一个全局向量（CLS token 或池化结果）；分割、深度这类 dense 任务要给每个像素一个答案，用的是每个 patch 自己的向量，也就是 patch features。

比较骨干时常用 linear probing：在冻结的 patch features 上只训一个线性层。分数高，说明需要的信息在特征里，而且线性就读得出来。[DINOv3 论文](https://arxiv.org/abs/2508.10104)（Meta，2025-08）§6.1.2 就是这样比的：分割看 ADE20k 的 mIoU，越高越好；深度看 NYUv2 的 RMSE，越低越好。

## 为什么自监督特征在像素级任务上更强

先看数。DINOv3 论文表 3（冻结骨干、线性头、作者自报）：ADE20k 上，用图文对比训练的 SigLIP 2 g/16 是 42.7 mIoU，PE Core G/14 是 38.9；自监督的 DINOv2 g/14 是 49.5，DINOv3 7B/16 是 55.9。NYUv2 深度的 RMSE 依次是 0.494、0.590、0.372、0.309。论文自己的概括是：ADE20k 上 DINOv3 比自监督基线高 6 mIoU 以上，比图文这类弱监督基线高 13 以上。

为什么会这样，各篇论文是这样解释的：

| 解释 | 出处 | 读的时候要打的折扣 |
| --- | --- | --- |
| 图文训练只能留下文字说得出的信息：caption 只是图片的近似，像素级的细节在这种监督下未必会被学出来 | [DINOv2 论文](https://arxiv.org/abs/2304.07193)（2023-04）§1 | 这是动机，不是对照实验 |
| DINO 系的目标里本来就有 patch 级的项：整图级的 DINO loss 之外还有 iBOT loss，遮住学生网络的一部分 patch，教师看完整的图，让学生在被遮的位置预测教师的 patch 特征 | DINOv2 §4；DINOv3 §4.2 | CLIP 式模型也在补局部监督：[SigLIP 2](../../03-multimodal-learning/siglip.md#siglip2) 加了自蒸馏和 masked prediction，但在 DINOv3 表 3 里仍然落后 |
| 信息未必缺，只是不在最后一层：PE 论文发现，它这个图文对比模型最好的特征在中间层，要靠语言或空间对齐取出来；把 SAM 2 蒸馏进 PE Core 得到的 PE Spatial，ADE20k 从 38.9 升到 49.3 | [PE 论文](https://arxiv.org/abs/2504.13181)（2025-04）；DINOv3 表 3 | 49.3 借了 SAM 2 的分割监督，已经不是纯图文模型 |
| 同一批数据上做受控对比，视觉自监督随规模涨得比 CLIP 好，到 7B 还没饱和 | [Web-SSL](https://arxiv.org/abs/2504.01017)（2025-04） | 同一张 DINOv3 表里，Web-DINO 7B/14 的 ADE20k 只有 42.7；“自监督”本身不保证 dense 特征好，数据整理和训练配方同样要紧 |

还有两条整体折扣。第一，这些数都出自 Meta 的论文，而被比较的 PE 也是 Meta 的模型；评测用的是线性头和固定分辨率（分割统一为 1024 个 patch token），全量微调以后差距还剩多少，这张表回答不了。第二，要用文字查询（“找出所有托盘”）时，光有视觉特征不够：要么用 CLIP 式模型，要么像 DINOv3 的 dino.txt 那样，冻结视觉骨干，另训一个文本编码器去对齐它（LiT 式训练，DINOv3 §5.3）。

## 训得越久，局部特征反而会变差

自监督也不是天然就给好的 dense 特征。DINOv3 §4.1 报告：训练 ViT-7B 时，ImageNet 分类精度一路上涨，Pascal VOC 上的线性分割却在约 20 万步后开始下降；同时，一个 patch 和其余 patch 的余弦相似度图越来越乱，无关的 patch 也变得很像。整图任务和像素任务在争同一份表示。

论文的修补办法叫 Gram anchoring。把一张图的 $P$ 个 patch 特征各自归一化，排成 $P\times d$ 的矩阵 $X$；$XX^\top$ 是所有 patch 两两之间的余弦相似度，叫 Gram 矩阵。训练时让学生网络的 Gram 矩阵贴近一个更早 checkpoint（称为 Gram teacher）的：

$$
\mathcal L_{\rm Gram}=\big\|X_SX_S^\top-X_GX_G^\top\big\|_F^2
$$

$X_S$、$X_G$ 分别是学生和 Gram teacher 的特征，$\|\cdot\|_F$ 是矩阵元素平方和再开方。把所有特征一起乘同一个正交矩阵 $Q$，$(XQ)(XQ)^\top=XX^\top$ 不变，所以这个 loss 只管“谁和谁像”，不管特征具体落在哪里。论文在 100 万步之后才加上这一项，Gram teacher 每 1 万步更新一次。

小算一次（教学例子）：3 个 patch，a、b 属于同一个托盘，c 是地面。早期特征 a=(1,0)、b=(1,0)、c=(0,1)，两两相似度 ab=1、ac=0、bc=0。训久之后变成 a=(1,0)、b=(0.8,0.6)、c=(0.6,0.8)，相似度 ab=0.8、ac=0.6、bc=0.96：地面和托盘上的 b 反而比 a、b 彼此还像。对角线都是 1，非对角每项出现两次：

$$
\mathcal L_{\rm Gram}=2\big[(0.8-1)^2+(0.6-0)^2+(0.96-0)^2\big]=2\times1.3216\approx2.64
$$

梯度会把 c 推离 b，却不要求任何一个向量回到原来的坐标。

## 深度：相对深度少一个尺度

针孔相机把相机坐标系里的点 $(X,Y,Z)$ 投到像素 $(u,v)$：

$$
u=f\frac{X}{Z}+c_x,\qquad v=f\frac{Y}{Z}+c_y
$$

$f$ 是以像素计的焦距（这里 $f_x=f_y=f$），$(c_x,c_y)$ 是主点。沿用[相机几何](camera-geometry.md)里的教学相机：$f=1000$、$(c_x,c_y)=(960,540)$。点 $P=(1,\,0.5,\,5)$ m 落在 $u=1000\times1/5+960=1160$、$v=1000\times0.5/5+540=640$。把整个场景绕相机中心放大 $k=2$ 倍，$P$ 变成 $(2,1,10)$，仍然落在 $(1160,640)$：分子分母同乘 $k$，抵消了。**同一张图，对应所有“深度乘 $k$”的场景。** 图里没有任何像素能区分它们，模型只能靠学到的先验（车一般多大、门一般多高）去猜米数。

所以很多模型只输出相对深度。设一张图里有三个像素：路面、前车、远处楼房，真实深度 4、8、20 m。一个只差尺度的模型输出 $(1,2,5)$，也就是真实值除以 4。现在激光雷达告诉你前车在 8 m：$k=8/2=4$，三个像素变成 $(4,8,20)$。**一个已知距离就把尺度定下来了。** 它定不了形状：模型如果把楼房说成 4 而不是 5，用前车定出 $k=4$ 后楼房是 16 m，差的 4 m 还在。

更常见的是仿射不变输出，尺度 $s$ 和偏移 $t$ 都不知道：$d=s\hat d+t$。[Marigold](https://arxiv.org/abs/2312.02145) 和第一版 [MoGe](https://arxiv.org/abs/2410.19115) 就是仿射不变的。设模型输出 $\hat d=(0,1,4)$。只知道前车 8 m，方程 $8=s+t$ 有无数组解：$(s,t)=(4,4)$ 给出 $(4,8,20)$，$(2,6)$ 给出 $(6,8,14)$，两组都和这一个观测吻合。再知道路面 4 m，才解出 $t=4$、$s=4$，楼房是 20 m。**仿射输出要两个已知距离。** [Depth Anything 3 的 README](https://github.com/ByteDance-Seed/Depth-Anything-3) 指出 Depth Anything 2 是 disparity（逆深度）输出：这时 $s$、$t$ 作用在 $1/d$ 上，未知数还是两个，但不能按深度直接线性换算。

<!-- widget:step-depth-scale -->

已知距离从哪里来：

- 稀疏激光雷达：[Depth Anything V2 的 README](https://github.com/DepthAnything/Depth-Anything-V2)（2024-12-22 一条）提到 Prompt Depth Anything 用低分辨率激光雷达去“提示”模型，得到米制深度。
- 已知的相机高度：假设点在平坦路面上，离地高度就是那个已知距离。[相机几何](camera-geometry.md)一篇用离地 1.5 m 的相机算出过 10 m，也算了俯仰偏 1° 时这个数会偏多少。
- 双目基线、多帧加 IMU，或者另一个米制模型。[Depth Anything 3](https://arxiv.org/abs/2511.10647) 的 Nested 版本就是用多视图模型估位姿和深度，再用单目米制模型给尺度（README）。

直接输出米制深度的单目模型还要面对焦距。由 $u-c_x=fX/Z$，把 $f$ 和每个点的 $Z$ 同时乘 2，像素不变：长焦拍一个沿光轴拉长的场景，和短焦拍原场景，得到同一张图。不知道焦距，米制深度就少一个条件。所以 [Depth Pro](https://arxiv.org/abs/2410.02073)（Apple，2024-10）同时估计焦距，[Metric3D v2](https://arxiv.org/abs/2404.15506) 先把图像变换到一个规范相机空间，[UniDepth](https://arxiv.org/abs/2403.18913) 带一个自己估计相机的模块，[MoGe-2](https://arxiv.org/abs/2507.02546) 也加上了米制尺度。Depth Anything 3 的单目米制版本则把焦距交给用户：[README](https://github.com/ByteDance-Seed/Depth-Anything-3) 给的换算是“米制深度 = 焦距（像素）× 网络输出 / 300”，焦距填错，所有距离按同一比例错。

## Pointmap：每个像素直接给一个三维点

深度图每个像素只给一个数；pointmap 每个像素给一个三维点。[DUSt3R 论文](https://arxiv.org/abs/2312.14132)（Naver Labs Europe，2023-12）§3 的定义是一张 $W\times H\times3$ 的点阵，像素和场景点一一对应。有深度 $D$ 和内参 $K$ 时：

$$
X_{u,v}=D_{u,v}\,K^{-1}\begin{bmatrix}u\\ v\\ 1\end{bmatrix}
$$

接着上面的例子：像素 $(1160,640)$、深度 5 m，$K^{-1}[1160,640,1]^\top=((1160-960)/1000,\,(640-540)/1000,\,1)=(0.2,\,0.1,\,1)$，乘 5 得 $(1,\,0.5,\,5)$，正好回到 $P$。

反过来，pointmap 里藏着相机。已知点 $(1,0.5,5)$ 落在 $u=1160$，假设主点在图像中心，$f=(u-c_x)Z/X=200\times5/1=1000$。DUSt3R §3.3 就是这样对所有像素做稳健拟合来求焦距的（假设主点居中、像素是正方形）。两张图时更有用：DUSt3R 把两张图的 pointmap 都放在第一张图的相机坐标系里，同一个三维点在两张图里的位置就是对应关系；把同一张图在两个坐标系里的点云对齐（Procrustes 有闭式解，更稳健的是 PnP 加 RANSAC），就得到相对位姿。它的输出只到一个未知尺度：训练 loss 先把预测和真值各自按“所有点到原点的平均距离”归一化，消掉的正是上一节的 $k$。

<!-- paper-figure:dust3r-arch -->

## 一次前向拿到相机、深度和点

传统做法是先匹配特征、跑 structure-from-motion 求相机，再做稠密重建，每一步都是迭代优化，[下一篇](reconstruction.md)会拆开讲。下面这条线把它变成网络的一次前向：

| 时间 | 模型 | 输入 → 输出 | 多张图怎样进同一个坐标系 |
| --- | --- | --- | --- |
| 2023-12 | [DUSt3R](https://arxiv.org/abs/2312.14132) | 两张未标定的图 → 两张 pointmap | 超过两张要另跑全局对齐（优化） |
| 2024-06 | [MASt3R](https://arxiv.org/abs/2406.09756) | DUSt3R 加稠密局部特征和快速匹配 | 同上 |
| 2025-01 | [Fast3R](https://arxiv.org/abs/2501.13928) | N 张图 → pointmap | 一次前向，不需要全局对齐；作者称一次可处理 1000 多张 |
| 2025-03 | [VGGT](https://arxiv.org/abs/2503.11651) | 1 到几百张图 → 内外参、深度图、pointmap、三维点轨迹 | 一次前向；作者称重建不到 1 秒；CVPR 2025 最佳论文（[README](https://github.com/facebookresearch/vggt)） |
| 2025-07 | [$\pi^3$](https://arxiv.org/abs/2507.13347) | 不指定参考视图，对输入顺序置换等变 | 一次前向 |
| 2025-09 | [MapAnything](https://arxiv.org/abs/2509.13414) | 图像，外加可选的内参、位姿、深度 → 米制深度、射线图、位姿和尺度因子 | 一次前向；一个模型覆盖 12 类以上重建任务 |
| 2025-11 | [Depth Anything 3](https://arxiv.org/abs/2511.10647) | 任意张图，有无位姿都行 → 一致的深度和射线 | 普通 DINO 式 Transformer，只用一个 depth-ray 目标 |
| 2025-12 | [D4RT](https://arxiv.org/abs/2512.08924) | 一段视频 → 深度、时空对应和相机，场景里的东西可以在动 | 查询式解码器，可以问任意时刻、任意点；CVPR 2026 最佳论文（[闭幕页](https://cvpr.thecvf.com/Conferences/2026/News/Closing)） |

<!-- paper-figure:vggt-arch -->

几处值得停下来看：

- **输出是冗余的。** VGGT 同时给深度、相机和 pointmap，[官方 README](https://github.com/facebookresearch/vggt) 写明：用深度加相机反投影出的点，通常比 pointmap 分支直接给的点更准。冗余可以拿来做一致性检查。
- **“深度加射线”是一种因子化。** 三维点等于相机中心加上“沿射线的距离乘射线方向”。Depth Anything 3 的论点是只预测这一个 depth-ray 目标就够；[MapAnything 的 README](https://github.com/facebookresearch/map-anything) 把 `ray_directions`、`depth_along_ray`、`depth_z`、`camera_poses` 和 `pts3d` 分开输出。
- **z 深度和沿射线的距离不是一回事。** 上例像素的射线方向是 $(0.2,0.1,1)$，长度 $\sqrt{1.05}\approx1.025$。z 深度 5 m，沿射线的距离就是 $5\times1.025\approx5.12$ m。离主点越远，两者差得越多。把 z 深度当成沿射线的距离去反投影，边缘的点会被成片拉近（上例算出的 z 只有 $5/1.025\approx4.88$ m）；反过来把沿射线的距离当 z 深度用，就会被成片推远（5.12 m）。
- **尺度从哪来，要逐个模型查。** DUSt3R 只到尺度；[CUT3R](https://arxiv.org/abs/2501.12387) 和 MapAnything 输出米制；Depth Anything 3 的米制靠 Metric 和 Nested 两个版本。

所有对比数字都是作者自报。例如 Depth Anything 3 的[摘要](https://arxiv.org/abs/2511.10647)写的是“surpassing prior SOTA VGGT by an average of 44.3% in camera pose accuracy …”，即相机位姿平均比此前最好的 VGGT 高 44.3%，几何高 25.1%；这是在作者自己新建的基准上测的。

## 能不能商用：常见 checkpoint 的许可

截至 2026-10-15，按各仓库的许可文件和 Hugging Face 许可标签整理（这不是法律意见；许可会改，上线前读原文）：

| 模型 | 许可 | 能否商用（粗读） |
| --- | --- | --- |
| DINOv2 | 代码和权重都是 Apache-2.0 | 可以 |
| DINOv3 | 自定义 DINOv3 License（2025-08-19） | Meta 博客称可商用；禁止 ITAR 和军事用途 |
| Web-SSL / Web-DINO | 仓库和权重都是 CC BY-NC 4.0 | 不可以 |
| Perception Encoder | 代码和模型都是 Apache-2.0 | 可以 |
| Depth Anything V2 | Small 为 Apache-2.0；Base/Large/Giant 为 CC BY-NC 4.0 | 只有 Small |
| Depth Anything 3 | Small/Base/Metric-L/Mono-L 为 Apache-2.0；Large/Giant/Nested 为 CC BY-NC 4.0 | 只有前一组 |
| Depth Pro | 代码为 Apple 示例代码许可；权重为 Apple ML Research Model 许可，仅限研究 | 权重不可以 |
| MoGe-2 / MoGe-3 | 代码和权重都是 MIT | 可以 |
| UniDepth | CC BY-NC 4.0 | 不可以 |
| DUSt3R / MASt3R / CUT3R | CC BY-NC-SA 4.0 | 不可以 |
| Fast3R、VGGT-Ω | FAIR Noncommercial Research License | 不可以 |
| VGGT | 原版 1B 权重 CC BY-NC 4.0；2025-07-29 起另有 VGGT-1B-Commercial，自定义许可，禁止军事用途，需申请 | 只有 Commercial 版 |
| $\pi^3$ | 代码 BSD-3；权重 CC BY-NC 4.0 | 权重不可以 |
| MapAnything | 代码 Apache-2.0；权重分 `map-anything`（CC BY-NC 4.0）和 `map-anything-apache`（Apache-2.0） | 选 apache 版 |
| D4RT | 项目页没有给出代码、权重或许可 | — |

许可确实会变：VGGT 在 2025-07-29 补发了可商用的 checkpoint（[README](https://github.com/facebookresearch/vggt)），Segment Anything 从 SAM 3 起换成了自定义许可（见[检测与分割](detection.md)）。同一个模型家族里，小号和大号的许可也常常不同。

## 动手之前

| 先试什么 | 应该观察什么 | 不能据此推出什么 |
| --- | --- | --- |
| 把一张相对深度图整体乘 $k$，用同一个 $K$ 投回图像 | 每个点的像素位置都不变 | 模型的相对深度是对的 |
| 用一个激光雷达点定出 $k$ | 所有像素都有了米数 | 其他像素的形状误差也被修好了 |
| 仿射不变输出只给一个已知距离 | $(s,t)$ 有无数组解 | 随便取一组就能用 |
| 把 `depth_z` 当成沿射线距离去反投影 | 图像边缘的点被成片拉近 | 是模型在边缘不准 |
| 冻结骨干上的 linear probe 分数高 | 需要的信息线性可读 | 全量微调后排名不变 |

## 截至 2026 年 10 月

- 继任者：[VGGT-Ω](https://arxiv.org/abs/2605.15195)（2026-05）放大了 VGGT，同时处理静态和动态场景，arXiv 页面的备注是 CVPR 2026 oral。截至 2026-10-15，我们没有找到 DINOv4 或 Depth Anything 4 的公开发布；没找到不等于不存在。
- 下游用法：DINOv3 已被当作现成骨干，例如 DEIMv2 检测器和 SAM 3D Body 的 checkpoint；DINO-WM 拿冻结的 DINOv2 patch 特征当世界模型的表示（见 [JEPA 一篇](../../11-frontier/world-models/jepa.md)）。

接着读[三维重建与神经渲染](reconstruction.md)：相机和点云有了以后，怎样变成一个能从新视角渲染的场景。也可以回到[本系列导读](README.md)。
