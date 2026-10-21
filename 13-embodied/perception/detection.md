# 感知与几何：2026 年的检测与分割

**中文** · [English](detection.en.md)

> 阅读时间：约 11 分钟 · 最近审阅：2026-10

给机器人或车选检测器时，论文常把 YOLO26、RF-DETR、SAM 3 等模型的 COCO AP 列在一起。这个数是怎么算出来的？该选哪一类模型？权重能不能商用？

COCO AP 汇总的是整条 precision–recall 曲线，而不是某个分数阈值下的 precision：在每个类别、每个 IoU 阈值下完成匹配，按置信度排序累积 precision 和 recall，再对插值后的 precision 在 101 个 recall 点上取平均，最后对 IoU 阈值和有效类别取平均。下面用一个小例子逐步计算。

实际选型时，还要比较闭集实时模型、开放词表模型和可提示模型（如 SAM 3）。闭集模型面对新类别或新领域时通常需要标注和微调；开放词表或可提示模型可以先用提示测试，不必为每个新查询重训，但不保证无需领域适配。速度取决于模型、输入、硬件和部署配置，不能仅按类别断言谁更快。许可证也要先查，不能只按 AP 排名选择。

## 检测器输出什么，IoU 怎么算

前置知识是 [ViT](../../03-multimodal-learning/vit.md) 和 [CLIP](../../03-multimodal-learning/clip.md)：现在的检测器多用 Transformer 骨干，开放词表检测靠图文对齐。

一个检测器对每张图输出一组预测：框 $(x_1,y_1,x_2,y_2)$、类别和分数。实例分割再给每个实例一张掩码，视频模型再给一个跨帧的 ID。框和真值重合得好不好，用 IoU（intersection over union）衡量：

$$
\mathrm{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}.
$$

例如真值 G1 = (0, 0, 10, 10)，预测 P1 = (1, 0, 11, 10)：交集是 x 方向 [1, 10]、y 方向 [0, 10]，面积 $9\times10=90$；并集 $100+100-90=110$；IoU $=0.818$。

## 手算 AP：3 个真值、5 个预测

教学数据：一张图、一个类别、3 个真值，模型给了 5 个预测。

| 框 | 坐标 (x1, y1, x2, y2) | 分数 | 和哪个真值重叠最多 | IoU |
| --- | --- | --- | --- | --- |
| G1 | (0, 0, 10, 10) | 真值 | — | — |
| G2 | (20, 0, 30, 10) | 真值 | — | — |
| G3 | (40, 0, 50, 10) | 真值 | — | — |
| P1 | (1, 0, 11, 10) | 0.95 | G1 | 0.818 |
| P2 | (0, 1, 10, 11) | 0.90 | G1 | 0.818 |
| P3 | (22, 1, 32, 11) | 0.80 | G2 | 0.563 |
| P4 | (60, 0, 70, 10) | 0.60 | 无 | 0 |
| P5 | (41, 1, 51, 10) | 0.40 | G3 | 0.743 |

按 COCO 官方评测代码（[pycocotools 的 cocoeval.py](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)），在本例没有 crowd 或 ignore 标注的情况下，配对规则是：预测按分数从高到低排，每个预测去拿还没被占用、IoU 达到阈值 $t$ 且最大的那个真值；拿不到就是误检（FP）。没被拿走的真值是漏检，拉低 recall。

取 $t=0.5$。P1 拿走 G1；P2 也和 G1 重叠，但 G1 已被占，P2 是重复框，记 FP；P3 拿 G2；P4 是背景，FP；P5 拿 G3。

| 按分数第 k 个 | 结果 | 累计 TP | 累计 FP | precision | recall |
| --- | --- | --- | --- | --- | --- |
| P1 | TP | 1 | 0 | 1.00 | 0.33 |
| P2 | FP | 1 | 1 | 0.50 | 0.33 |
| P3 | TP | 2 | 1 | 0.67 | 0.67 |
| P4 | FP | 2 | 2 | 0.50 | 0.67 |
| P5 | TP | 3 | 2 | 0.60 | 1.00 |

先用插值 precision–recall 曲线下的面积来理解 AP。对每个 recall 水平，取达到同等或更高 recall 时的最大 precision，得到一条不随 recall 上升而增大的包络线。于是 recall 在 1/3 以内 precision 是 1.0，到 2/3 是 0.667，到 1 是 0.6：

$$
\mathrm{AP}_{50}=\tfrac13\times1.0+\tfrac13\times0.667+\tfrac13\times0.6=0.756.
$$

cocoeval.py 不直接算面积，而是在 recall = 0, 0.01, …, 1 这 101 个点上取插值后的 precision 再平均；这个例子四舍五入后同样得到 0.756。

COCO 的主指标 AP 还要在 10 个 IoU 阈值 0.50, 0.55, …, 0.95 上再平均，多类别时再对类别平均（常叫 mAP）：

| IoU 阈值 | 算 TP 的预测 | AP |
| --- | --- | --- |
| 0.50、0.55 | P1、P3、P5 | 0.756 |
| 0.60、0.65、0.70 | P1、P5 | 0.467 |
| 0.75、0.80 | P1 | 0.333 |
| 0.85、0.90、0.95 | 无 | 0 |

10 个数平均是 0.358（101 点版本 0.359）。P3 在 0.5 时算命中，但在 10 个阈值里只占 2 个：AP@[.5:.95] 奖励框得准。cocoeval.py 的默认设置还有：每张图最多算 100 个预测，并按面积分小（< 32² 像素）、中（32²–96²）、大（> 96²）三档另报。每次只改一个条件再算（我用一个按 cocoeval.py 逻辑写的小脚本复算过）：

| 改一个条件 | 结果 | 说明什么 | 不能据此推出 |
| --- | --- | --- | --- |
| 用 NMS 删掉重复框 P2 | AP50 升到 0.917 | 重复框会实打实扣分 | NMS 总是对的（见下一节） |
| 不删 P2，只把它的分数降到 0.30 | AP50 同样是 0.917 | 排在所有 TP 之后的误检不扣 AP | AP 高说明部署时误检少（部署要定一个分数阈值，看的是那一点的 precision） |
| 把 P3 改成 (21, 0, 31, 10)，IoU 0.818 | AP50 不变，平均 AP 从 0.358 升到 0.489 | AP@[.5:.95] 看定位精度 | AP50 一样的两个模型定位一样好 |

<!-- widget:step-coco-ap -->

## 重复框：NMS 还是一对一匹配

传统检测器会对同一个物体给出好几个框。非极大值抑制（NMS）在推理后处理：按分数排序，保留最高的，删掉和它 IoU 超过阈值（例如 0.5）的其他框，再重复。P1 和 P2 的 IoU 是 0.681，P2 被删，AP50 从 0.756 升到 0.917。

NMS 的问题也在这里：如果 P2 其实框的是站在 G1 身后、和它 IoU 0.68 的另一个人，NMS 就删掉了一个真阳性。阈值是手调的，拥挤场景里很难两头顾。

[DETR（2020）](https://arxiv.org/abs/2005.12872)把检测写成集合预测，摘要说这样做“effectively removing the need for many hand-designed components like a non-maximum suppression procedure …”，即不再需要 NMS 这类手工设计的组件。具体做法是：解码器固定输出 $N$ 个槽位（论文用 $N=100$），训练时用 Hungarian 算法在预测和真值（不够 $N$ 个就补“无物体” $\varnothing$）之间找一对一配对：

$$
\hat\sigma=\arg\min_{\sigma}\sum_{i=1}^{N}\mathcal L_{\rm match}\big(y_i,\hat y_{\sigma(i)}\big),\qquad
\mathcal L_{\rm match}=-\mathbb 1_{\{c_i\neq\varnothing\}}\,\hat p_{\sigma(i)}(c_i)+\mathbb 1_{\{c_i\neq\varnothing\}}\,\mathcal L_{\rm box}\big(b_i,\hat b_{\sigma(i)}\big).
$$

$c_i$ 是真值类别，$\hat p_{\sigma(i)}(c_i)$ 是配上的那个槽位给这个类别的概率，$\mathcal L_{\rm box}$ 是 L1 损失和 generalized IoU 损失的加权和。没配上的槽位被训练去预测 $\varnothing$，所以模型自己学会不出重复框。放到上面的例子里：G1 只会配给 P1、P2 中代价更低的一个，另一个被推向“无物体”。这和[跟踪那篇](tracking.md)用的是同一个 Hungarian 算法，只是那里用来在运行时配轨迹，这里用来在训练时分配监督目标。

<!-- paper-figure:detr-set-prediction -->

论文的消融也说明了这一点：对第一层解码器的输出做 NMS 有帮助，到最后几层反而略微降低 AP，因为 NMS 删掉了真阳性。DETR 的短板是收敛慢；[Deformable DETR](https://arxiv.org/abs/2010.04159) 作者报告训练轮数少 10 倍，之后 [RT-DETR](https://arxiv.org/abs/2304.08069)、[D-FINE](https://arxiv.org/abs/2410.13842)、[DEIM](https://arxiv.org/abs/2412.04234) 和 [RF-DETR](https://arxiv.org/abs/2511.09554) 把这条线做到了实时。

YOLO 这条线在 2026 年也有了不需要 NMS 的推理路径。[YOLO26 论文（2026-06-02）](https://arxiv.org/abs/2606.03748)用双头：一对多的头提供密集监督，一对一的头直接输出、不需要 NMS；训练中损失权重逐步移向一对一的头。双头分配的做法沿用自 YOLOv10。论文把一对一的头当默认；但 2026-10 读到的 [Ultralytics 文档](https://docs.ultralytics.com/models/yolo26/)写的是，预测和验证默认走一对多的头再做 NMS（精度略高），要 NMS-free 得设 `nms=False`。部署前先确认自己跑的是哪个头。

## 闭集实时、开放词表、可提示：三类怎么选

| 类别 | 输入 | 输出 | 代表（作者报告） | 适合什么 |
| --- | --- | --- | --- | --- |
| 闭集实时 | 图像；类别在训练时定死 | 固定类别的框（可加掩码） | YOLO26：COCO 40.9–57.5 mAP，T4 TensorRT 1.7–11.8 ms（n 到 x，速度按 NMS-free 头测）；RF-DETR：Nano 48.0 AP，2XL 是第一个过 60 AP 的实时检测器 | 类别固定、有标注、延迟预算紧 |
| 开放词表检测 | 图像 + 文字类别名 | 框 | [Grounding DINO](https://arxiv.org/abs/2303.05499)：COCO 零样本 52.5 AP；YOLOE-26x：LVIS minival 40.6 AP（文字提示） | 新类别没有标注时先摸底 |
| 可提示概念分割 | 图像或视频 + 名词短语或示例框 | 所有匹配实例的掩码和 ID | [SAM 3](https://arxiv.org/abs/2511.16719)：COCO 框 AP 56.4、LVIS 框 AP 53.6（[README](https://github.com/facebookresearch/sam3)）；H200 上一张有 100 多个目标的图约 30 ms | 自动标注、长尾概念、在视频里找“所有的 X” |

开放词表这条线的前身包括 [GLIP](https://arxiv.org/abs/2112.03857)、[OWL-ViT](https://arxiv.org/abs/2205.06230) / [OWLv2](https://arxiv.org/abs/2306.09683)、[YOLO-World](https://arxiv.org/abs/2401.17270) 和 [YOLOE](https://arxiv.org/abs/2503.07465)。SAM 3（2025-11-20）把它和分割合在一起：给一个短名词短语（如“黄色校车”）或示例框，返回图像或视频里所有匹配实例的掩码和 ID。它由一个 DETR 式检测器和一个带记忆的视频跟踪器组成，共用一个骨干；一个 presence head 单独判断“这个概念在不在图里”，把识别和定位拆开（论文摘要）。

<!-- paper-figure:sam3-arch -->

数字要连着条件看。同一个 SAM 3，在 COCO 上框 AP 为 56.4（论文表 1 把它列为零样本结果）；在 [RF100-VL](https://arxiv.org/abs/2505.20612) 的 100 个多领域数据集上，零样本平均只有 15.2 AP，其中医学类为 2.6。用每类约 10 个标注实例微调（10-shot）后，平均 AP 为 36.5；全量微调后为 61.6（论文 v2 表 35、36，作者报告）。

论文附录 B 也说明了限制：零样本时难以泛化到细粒度的领域外概念，如飞机型号和医学术语，在热成像这类小众领域尤其如此。提示只能是简单名词短语；要处理较长的指代描述，还需配合一个多模态大模型。

怎么选，下面是设计上的取舍，不是哪篇论文的结论：

- 类别固定、能标数据、延迟预算紧：微调一个闭集实时模型，先查许可证。
- 新类别、还没有标注：用开放词表模型或 SAM 3 摸底和预标注，但一定在自己的验证集上量；RF100-VL 说明零样本在某些领域可能接近 0。需要上车或上机器人时，可以用这些标注再训一个小的闭集模型。
- 视频里要“所有的 X”并带 ID：SAM 3 或 SAM 3.1。

## Benchmark 现在各管什么

| Benchmark | 测什么 | 2025–2026 的角色 |
| --- | --- | --- |
| [COCO（2014）](https://arxiv.org/abs/1405.0312) | 80 类检测和分割，AP@[.5:.95] | 每篇检测论文仍然报，但只是其中一列；RF-DETR 论文认为专用检测器在架构、学习率调度和数据增强上对 COCO 隐式过拟合 |
| [LVIS（2019）](https://arxiv.org/abs/1908.03195) | 1000 多类的长尾实例分割；只在 1–10 张训练图里出现的类叫 rare | 开放词表的主考场：看 rare 类 AP（APr），因为这些类几乎没有框标注 |
| [ODinW13](https://arxiv.org/abs/2204.08790) | ELEVATER 的 35 个“野外”检测集中常用的 13 个 | 零样本和少样本迁移；SAM 3 零样本 61.0 |
| RF100-VL（2025） | 100 个多领域检测集，分零样本、少样本、全量 | 越来越常和 COCO 并列；Grounding DINO、Qwen2.5-VL 在难的医学集上零样本低于 2% |
| SA-Co（2025-11） | 概念分割，含负样本提示，指标 cgF1 | SAM 3 引入；SA-Co/Gold 掩码 cgF1 人类 72.8、SAM 3 54.1（README） |

注意 RF-DETR 和 RF100-VL 都出自 Roboflow 及合作者，“COCO 不够用”这个论点和他们的产品方向一致；论点本身有数据支持，但读的时候要知道这层关系。

排行榜也变了。Papers with Code 据报道在 2025 年 7 月停止服务，2026-10-15 访问 paperswithcode.com 会被重定向到 Hugging Face 的 Trending Papers；据报道，原来的 SOTA 榜没有被继承。同一天，Hugging Face 的 49 个 [“official benchmark” 数据集](https://huggingface.co/api/datasets?filter=benchmark:official)里没有一个是检测 benchmark。Roboflow 维护一个 COCO val 排行榜，会自己重跑各家模型推理，但它也是 RF-DETR 的出品方（[方法说明](https://leaderboard.roboflow.com/methodology/)）。结论：别指望一张公共榜替你选型，在自己的数据上量。

## 能不能商用：常见 checkpoint 的许可证

下表是 2026-10-15 读到的仓库或 Hugging Face 许可证，只是阅读记录，不是法律意见。代码和权重的许可证可能不同，训练数据还有自己的条款。

| 模型 | 许可证 | 要注意什么 |
| --- | --- | --- |
| SAM、SAM 2 / 2.1 | Apache-2.0 | SA-1B 数据集本身是研究用许可 |
| SAM 3、SAM 3.1 | 自定义 SAM License（2025-11-19） | 免版税、允许修改；禁止 ITAR、军事和武器用途；下载需申请 |
| RF-DETR Nano / S / M / L 及分割模型 | Apache-2.0 | — |
| RF-DETR Atto / Femto / Pico / XL / 2XL | PML 1.0 | 部署前读 PML 条款 |
| YOLO11、YOLO26（含 YOLOE-26） | AGPL-3.0 或 Ultralytics Enterprise | 按 AGPL-3.0 的条件用，或另购 Enterprise 许可 |
| YOLOv12（非 Ultralytics 团队）、YOLOE（清华） | AGPL-3.0 | 按 AGPL-3.0 的条件用 |
| YOLO-World | GPL-3.0 | 可另行申请商业许可 |
| RT-DETR、D-FINE、DEIM | Apache-2.0 | D-FINE 提醒：用 Objects365 预训练的权重可能受该数据集条款约束 |
| DEIMv2 | 自定义 DEIMv2 License（2026） | 条款区分商用和非商用，部署前读 |
| Grounding DINO、OWL-ViT / OWLv2 | Apache-2.0 | — |
| Grounding DINO 1.5、DINO-X | 只提供 API | 拿不到权重 |
| Florence-2、GLIP | MIT | — |
| DINOv3（DEIMv2 用的骨干） | 自定义 DINOv3 License | Meta 博客称允许商用；禁止 ITAR 和军事用途 |

各行出处：[SAM](https://github.com/facebookresearch/segment-anything)、[SAM 2](https://github.com/facebookresearch/sam2)、[SAM 3](https://github.com/facebookresearch/sam3)、[RF-DETR](https://github.com/roboflow/rf-detr)、[YOLO26 文档](https://docs.ultralytics.com/models/yolo26/)、[YOLO11 文档](https://docs.ultralytics.com/models/yolo11/)、[YOLOv12](https://arxiv.org/abs/2502.12524)、[D-FINE](https://github.com/Peterande/D-FINE)、[DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)、[Grounding DINO 1.5 API](https://github.com/IDEA-Research/Grounding-DINO-1.5-API)、[DINO-X API](https://github.com/IDEA-Research/DINO-X-API)、[Florence-2](https://arxiv.org/abs/2311.06242)、[DINOv3](https://github.com/facebookresearch/dinov3)。

## 截至 2026-10

- **YOLO27 还只是预告。** Ultralytics 文档写明“no launch date has been set”，权重和代码未公开，初步数字可能变（[文档](https://docs.ultralytics.com/models/yolo27/)，公司自报）。YOLO26 的文档模型表后来加了语义分割和深度头，论文摘要里没有。
- **SAM 3.1（2026-03-27）** 加入多目标联合跟踪，见[跟踪那篇](tracking.md)。2026-10-15 检索没有找到 “SAM 4”，也可能只是还没被索引。
- **同一模型，不同来源的数字可能不同。** SAM 3 的 SA-Co 概念数，README 写 27 万、论文 v2 写 20.7 万；引用时注明来源版本。
- DEIMv2 的新许可证何时替换了原许可证，目前查不到。

下一篇：[冻结骨干与几何基础模型](foundation-models.md)，讲 DINOv3、Depth Anything 3 和 VGGT 这些检测器背后的骨干与三维模型。
