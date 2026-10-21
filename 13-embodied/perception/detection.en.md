# Perception & geometry: detection and segmentation in 2026

[中文](detection.md) · **English**

> Reading time: ~11 min · Last reviewed: 2026-10

You need a detector for a robot or a car. Papers show a row of COCO AP numbers, and the names run from YOLO26 and RF-DETR to SAM 3. How is that number computed? Which kind of model should you pick? Can you ship the weights commercially? **A detection score comes from matching predicted boxes one-to-one to ground truth and averaging precision over several IoU thresholds. The real choice in 2026 is between fast closed-set real-time models that you fine-tune on your own labels, and slower open-vocabulary or promptable models (such as SAM 3) that need no retraining; the licence often decides for you before the AP does.**

## What a detector outputs, and how IoU is computed

Prerequisites: [ViT](../../03-multimodal-learning/vit.en.md) and [CLIP](../../03-multimodal-learning/clip.en.md). Most current detectors use Transformer backbones, and open-vocabulary detection relies on image–text alignment.

For each image a detector outputs a set of predictions: a box $(x_1,y_1,x_2,y_2)$, a class and a score. Instance segmentation adds a mask per instance, and video models add an ID that persists across frames. How well a box overlaps the ground truth is measured by IoU (intersection over union):

$$
\mathrm{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}.
$$

For example, ground truth G1 = (0, 0, 10, 10) and prediction P1 = (1, 0, 11, 10): the intersection spans [1, 10] in x and [0, 10] in y, area $9\times10=90$; the union is $100+100-90=110$; IoU $=0.818$.

## AP by hand: 3 ground-truth boxes, 5 predictions

Teaching data: one image, one class, 3 ground-truth boxes, and 5 predictions from the model.

| Box | Coordinates (x1, y1, x2, y2) | Score | Ground truth it overlaps most | IoU |
| --- | --- | --- | --- | --- |
| G1 | (0, 0, 10, 10) | ground truth | — | — |
| G2 | (20, 0, 30, 10) | ground truth | — | — |
| G3 | (40, 0, 50, 10) | ground truth | — | — |
| P1 | (1, 0, 11, 10) | 0.95 | G1 | 0.818 |
| P2 | (0, 1, 10, 11) | 0.90 | G1 | 0.818 |
| P3 | (22, 1, 32, 11) | 0.80 | G2 | 0.563 |
| P4 | (60, 0, 70, 10) | 0.60 | none | 0 |
| P5 | (41, 1, 51, 10) | 0.40 | G3 | 0.743 |

The matching rule in COCO's official evaluation code ([cocoeval.py in pycocotools](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)): sort predictions by score, highest first; each prediction takes the not-yet-taken ground-truth box with the largest IoU at or above the threshold $t$; if there is none, it is a false positive (FP). Ground-truth boxes left untaken are misses and lower recall.

Take $t=0.5$. P1 takes G1. P2 also overlaps G1, but G1 is taken, so P2 is a duplicate and counts as FP. P3 takes G2. P4 is background, FP. P5 takes G3.

| k-th by score | Result | Cumulative TP | Cumulative FP | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| P1 | TP | 1 | 0 | 1.00 | 0.33 |
| P2 | FP | 1 | 1 | 0.50 | 0.33 |
| P3 | TP | 2 | 1 | 0.67 | 0.67 |
| P4 | FP | 2 | 2 | 0.50 | 0.67 |
| P5 | TP | 3 | 2 | 0.60 | 1.00 |

AP is the area under the precision–recall curve. First flatten the curve: replace each point's precision with the largest precision seen at any higher recall. Precision is then 1.0 for recall up to 1/3, 0.667 up to 2/3, and 0.6 up to 1:

$$
\mathrm{AP}_{50}=\tfrac13\times1.0+\tfrac13\times0.667+\tfrac13\times0.6=0.756.
$$

cocoeval.py does not compute the area directly; it reads the flattened precision at 101 recall points, 0, 0.01, …, 1, and averages them. In this example that also gives 0.756.

COCO's headline AP then averages over 10 IoU thresholds, 0.50, 0.55, …, 0.95, and over classes when there are several (often called mAP):

| IoU threshold | Predictions that count as TP | AP |
| --- | --- | --- |
| 0.50, 0.55 | P1, P3, P5 | 0.756 |
| 0.60, 0.65, 0.70 | P1, P5 | 0.467 |
| 0.75, 0.80 | P1 | 0.333 |
| 0.85, 0.90, 0.95 | none | 0 |

The mean of the 10 values is 0.358 (0.359 with the 101-point version). P3 is a hit at 0.5 but counts at only 2 of the 10 thresholds: AP@[.5:.95] rewards tight boxes. cocoeval.py's defaults also count at most 100 predictions per image and report small (< 32² pixels), medium (32²–96²) and large (> 96²) objects separately. Now change one condition at a time (I recomputed each row with a small script that follows cocoeval.py's logic):

| Change one condition | Result | What it shows | What you can NOT conclude |
| --- | --- | --- | --- |
| Remove the duplicate P2 with NMS | AP50 rises to 0.917 | Duplicates cost real points | NMS is always right (see the next section) |
| Keep P2 but lower its score to 0.30 | AP50 is also 0.917 | An FP ranked below every TP costs no AP | High AP means few false positives in deployment (deployment fixes one score threshold, and precision at that point is what counts) |
| Change P3 to (21, 0, 31, 10), IoU 0.818 | AP50 unchanged; mean AP rises from 0.358 to 0.489 | AP@[.5:.95] measures localisation | Two models with the same AP50 localise equally well |

<!-- widget:step-coco-ap -->

## Duplicate boxes: NMS or one-to-one matching

Classic detectors produce several boxes for one object. Non-maximum suppression (NMS) post-processes the output: sort by score, keep the top box, delete other boxes whose IoU with it exceeds a threshold (say 0.5), and repeat. P1 and P2 have IoU 0.681, so P2 is deleted and AP50 rises from 0.756 to 0.917.

That is also where NMS fails: if P2 were actually boxing a second person standing behind G1 at IoU 0.68, NMS would delete a true positive. The threshold is hand-tuned and hard to get right in both directions in crowded scenes.

[DETR (2020)](https://arxiv.org/abs/2005.12872) treats detection as set prediction, which, its abstract says, streamlines the pipeline, "effectively removing the need for many hand-designed components like a non-maximum suppression procedure …". The decoder always outputs $N$ slots (the paper uses $N=100$). During training the Hungarian algorithm finds a one-to-one matching between predictions and ground truth, padded with "no object" $\varnothing$ up to $N$:

$$
\hat\sigma=\arg\min_{\sigma}\sum_{i=1}^{N}\mathcal L_{\rm match}\big(y_i,\hat y_{\sigma(i)}\big),\qquad
\mathcal L_{\rm match}=-\mathbb 1_{\{c_i\neq\varnothing\}}\,\hat p_{\sigma(i)}(c_i)+\mathbb 1_{\{c_i\neq\varnothing\}}\,\mathcal L_{\rm box}\big(b_i,\hat b_{\sigma(i)}\big).
$$

$c_i$ is the ground-truth class, $\hat p_{\sigma(i)}(c_i)$ is the probability the matched slot assigns to that class, and $\mathcal L_{\rm box}$ is a weighted sum of an L1 loss and the generalized IoU loss. Unmatched slots are trained to predict $\varnothing$, so the model itself learns not to emit duplicates. In the example above, G1 is matched to whichever of P1 and P2 is cheaper, and the other is pushed towards "no object". This is the same Hungarian algorithm as in the [tracking note](tracking.en.md); there it matches tracks at run time, here it assigns training targets.

<!-- paper-figure:detr-set-prediction -->

The paper's ablation shows the same thing: NMS helps the output of the first decoder layer, but at the last layers it slightly lowers AP because it removes true positives. DETR's weakness is slow convergence; [Deformable DETR](https://arxiv.org/abs/2010.04159) reports 10× fewer training epochs, and [RT-DETR](https://arxiv.org/abs/2304.08069), [D-FINE](https://arxiv.org/abs/2410.13842), [DEIM](https://arxiv.org/abs/2412.04234) and [RF-DETR](https://arxiv.org/abs/2511.09554) then brought this line to real time.

The YOLO line also gained an NMS-free inference path in 2026. The [YOLO26 paper (2026-06-02)](https://arxiv.org/abs/2606.03748) uses two heads: a one-to-many head gives dense supervision, and a one-to-one head produces the output directly with no NMS; during training the loss weight shifts progressively towards the one-to-one head. The dual-assignment idea comes from YOLOv10. The paper treats the one-to-one head as the default, but the [Ultralytics docs](https://docs.ultralytics.com/models/yolo26/), read in October 2026, say prediction and validation default to the one-to-many head followed by NMS (slightly more accurate), and NMS-free inference needs `nms=False`. Before deploying, check which head you are actually running.

## Closed-set real-time, open-vocabulary, promptable: which to pick

| Kind | Input | Output | Examples (author-reported) | Good for |
| --- | --- | --- | --- | --- |
| Closed-set real-time | Image; classes fixed at training time | Boxes (optionally masks) for fixed classes | YOLO26: 40.9–57.5 COCO mAP at 1.7–11.8 ms on T4 TensorRT (n to x, speed measured with the NMS-free head); RF-DETR: Nano 48.0 AP, 2XL the first real-time detector above 60 AP | Fixed classes, labelled data, tight latency budget |
| Open-vocabulary detection | Image + class names as text | Boxes | [Grounding DINO](https://arxiv.org/abs/2303.05499): 52.5 AP zero-shot on COCO; YOLOE-26x: 40.6 AP on LVIS minival with text prompts | Surveying new classes before you have labels |
| Promptable concept segmentation | Image or video + noun phrase or exemplar box | Masks and IDs for every matching instance | [SAM 3](https://arxiv.org/abs/2511.16719): 56.4 box AP on COCO, 53.6 box AP on LVIS ([README](https://github.com/facebookresearch/sam3)); about 30 ms for an image with 100+ objects on an H200 | Auto-labelling, long-tail concepts, finding "every X" in video |

Earlier open-vocabulary work includes [GLIP](https://arxiv.org/abs/2112.03857), [OWL-ViT](https://arxiv.org/abs/2205.06230) / [OWLv2](https://arxiv.org/abs/2306.09683), [YOLO-World](https://arxiv.org/abs/2401.17270) and [YOLOE](https://arxiv.org/abs/2503.07465). SAM 3 (2025-11-20) merges it with segmentation: given a short noun phrase (such as "yellow school bus") or an exemplar box, it returns masks and IDs for every matching instance in an image or video. It combines a DETR-style detector and a memory-based video tracker that share one backbone; a presence head separately decides "is this concept in the image at all", decoupling recognition from localisation (paper abstract).

<!-- paper-figure:sam3-arch -->

Read numbers together with their conditions. The same SAM 3 scores 56.4 box AP on COCO (Table 1 of the paper lists it as a zero-shot result); on the 100 multi-domain datasets of [RF100-VL](https://arxiv.org/abs/2505.20612) its zero-shot average is only 15.2 AP, with 2.6 on the medical group; fine-tuning on about 10 labelled instances per class (10-shot) gives 36.5 on average, and full fine-tuning 61.6 (paper v2, Tables 35 and 36, author-reported). Appendix B of the paper says so itself: zero-shot, it struggles with fine-grained out-of-domain concepts (such as aircraft types and medical terms), especially in niche domains like thermal imagery; prompts must be simple noun phrases, and long referring expressions need a multimodal LLM alongside.

How to choose. The following are design trade-offs, not the conclusion of any paper:

- Fixed classes, labels available, tight latency budget: fine-tune a closed-set real-time model, and check its licence first.
- New classes with no labels yet: survey and pre-label with an open-vocabulary model or SAM 3, but always measure on your own validation set; RF100-VL shows zero-shot can be close to 0 in some domains. When the model must run on the car or robot, those labels can train a small closed-set model.
- "Every X" in video, with IDs: SAM 3 or SAM 3.1.

## What each benchmark is for now

| Benchmark | What it measures | Role in 2025–2026 |
| --- | --- | --- |
| [COCO (2014)](https://arxiv.org/abs/1405.0312) | 80-class detection and segmentation, AP@[.5:.95] | Still reported by every detection paper, but as one column; the RF-DETR paper argues that specialist detectors implicitly overfit COCO through architecture, learning-rate schedules and augmentation |
| [LVIS (2019)](https://arxiv.org/abs/1908.03195) | Long-tail instance segmentation over 1000+ classes; classes that appear in only 1–10 training images are called rare | The main test for open vocabulary: look at rare-class AP (APr), since these classes have almost no box labels |
| [ODinW13](https://arxiv.org/abs/2204.08790) | A commonly used 13 of ELEVATER's 35 "in the wild" detection datasets | Zero-shot and few-shot transfer; SAM 3 scores 61.0 zero-shot |
| RF100-VL (2025) | 100 multi-domain detection datasets, in zero-shot, few-shot and full regimes | Increasingly reported next to COCO; Grounding DINO and Qwen2.5-VL score below 2% zero-shot on hard medical sets |
| SA-Co (2025-11) | Concept segmentation, including negative prompts, scored with cgF1 | Introduced with SAM 3; SA-Co/Gold mask cgF1 is 72.8 for humans and 54.1 for SAM 3 (README) |

Note that RF-DETR and RF100-VL both come from Roboflow and collaborators, and the "COCO is not enough" argument fits their product direction. The argument is backed by data, but read it knowing that relationship.

Leaderboards changed too. Papers with Code reportedly shut down in July 2025; on 2026-10-15, paperswithcode.com redirects to Hugging Face's Trending Papers, and reportedly the old SOTA leaderboards were not carried over. On the same day, none of Hugging Face's 49 ["official benchmark" datasets](https://huggingface.co/api/datasets?filter=benchmark:official) was a detection benchmark. Roboflow runs a COCO val leaderboard and re-runs each model's inference itself, but it is also the maker of RF-DETR ([methodology](https://leaderboard.roboflow.com/methodology/)). Bottom line: do not expect a public leaderboard to pick your model; measure on your own data.

## Can you ship it: licences of common checkpoints

The table records the repository or Hugging Face licences read on 2026-10-15. It is a reading log, not legal advice. Code and weights may carry different licences, and training data has its own terms.

| Model | Licence | What to watch |
| --- | --- | --- |
| SAM, SAM 2 / 2.1 | Apache-2.0 | The SA-1B dataset itself has a research-only licence |
| SAM 3, SAM 3.1 | Custom SAM License (2025-11-19) | Royalty-free, modifications allowed; bans ITAR, military and weapons uses; download requires a request |
| RF-DETR Nano / S / M / L and segmentation models | Apache-2.0 | — |
| RF-DETR Atto / Femto / Pico / XL / 2XL | PML 1.0 | Read the PML terms before deploying |
| YOLO11, YOLO26 (including YOLOE-26) | AGPL-3.0 or Ultralytics Enterprise | Use under AGPL-3.0's conditions, or buy an Enterprise licence |
| YOLOv12 (not by Ultralytics), YOLOE (Tsinghua) | AGPL-3.0 | Use under AGPL-3.0's conditions |
| YOLO-World | GPL-3.0 | A commercial licence can be requested separately |
| RT-DETR, D-FINE, DEIM | Apache-2.0 | D-FINE warns that weights pretrained on Objects365 may be bound by that dataset's terms |
| DEIMv2 | Custom DEIMv2 License (2026) | Terms separate commercial from non-commercial use; read before deploying |
| Grounding DINO, OWL-ViT / OWLv2 | Apache-2.0 | — |
| Grounding DINO 1.5, DINO-X | API only | No weights available |
| Florence-2, GLIP | MIT | — |
| DINOv3 (the backbone DEIMv2 uses) | Custom DINOv3 License | Meta's blog says commercial use is allowed; bans ITAR and military uses |

Sources per row: [SAM](https://github.com/facebookresearch/segment-anything), [SAM 2](https://github.com/facebookresearch/sam2), [SAM 3](https://github.com/facebookresearch/sam3), [RF-DETR](https://github.com/roboflow/rf-detr), [YOLO26 docs](https://docs.ultralytics.com/models/yolo26/), [YOLO11 docs](https://docs.ultralytics.com/models/yolo11/), [YOLOv12](https://arxiv.org/abs/2502.12524), [D-FINE](https://github.com/Peterande/D-FINE), [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2), [Grounding DINO 1.5 API](https://github.com/IDEA-Research/Grounding-DINO-1.5-API), [DINO-X API](https://github.com/IDEA-Research/DINO-X-API), [Florence-2](https://arxiv.org/abs/2311.06242), [DINOv3](https://github.com/facebookresearch/dinov3).

## As of October 2026

- **YOLO27 is only a preview.** Ultralytics' docs say "no launch date has been set", weights and code are not public, and preliminary numbers may change ([docs](https://docs.ultralytics.com/models/yolo27/), vendor-reported). The YOLO26 docs' model table later added semantic-segmentation and depth heads, which the paper abstract does not mention.
- **SAM 3.1 (2026-03-27)** adds joint multi-object tracking; see the [tracking note](tracking.en.md). A search on 2026-10-15 found no "SAM 4", though it may simply not be indexed yet.
- **The same model can have different numbers in different sources.** SAM 3's SA-Co concept count is 270K in the README and 207K in paper v2; cite the version you used.
- When DEIMv2's new licence replaced the original could not be established.

Next: [frozen backbones and geometry foundation models](foundation-models.en.md), on DINOv3, Depth Anything 3 and VGGT, the backbones and 3D models behind these detectors.
