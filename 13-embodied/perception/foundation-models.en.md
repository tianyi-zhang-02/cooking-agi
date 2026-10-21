# Perception & geometry: frozen backbones and geometry foundation models

[中文](foundation-models.md) · **English**

> Reading time: ~13 min · Last reviewed: 2026-10

You want to give a warehouse robot two skills: segment the traversable floor and estimate the distance to an obstacle ahead. You have only a few thousand labelled images and cannot afford to train a large backbone, so you plan to freeze a pretrained ViT and train a small task head. Should you pick a CLIP-style image–text model or a self-supervised model such as DINO? Can you interpret a depth model's outputs directly in metres?

In the frozen-backbone comparisons reported by the authors, DINO-family features clearly outperform CLIP-style features on pixel-level tasks. Depth raises a separate issue: the output convention matters. One known distance fixes a scale-only depth output, whereas affine-invariant output also has an unknown shift; inverse-depth output must be calibrated in inverse depth. Since 2025, geometry foundation models have also been able to return cameras, depth, and points in one forward pass, but many of the strongest checkpoints cannot be used commercially. The sections below examine these results, their assumptions, and their limits.

It helps to have read [ViT](../../03-multimodal-learning/vit.en.md), [CLIP](../../03-multimodal-learning/clip.en.md) and this series' [camera geometry](camera-geometry.en.md) note first: this note assumes you know the intrinsics $K$, the extrinsics $[R\mid t]$ and pinhole projection.

## Freeze the backbone, train only a head

A frozen backbone means none of the pretrained weights change; the network is used only as a feature extractor. The image is cut into patches and the ViT outputs one vector per patch. Classification uses a single global vector (the CLS token or a pooled one). Dense tasks such as segmentation and depth need an answer for every pixel, so they use each patch's own vector: the patch features.

Backbones are often compared by linear probing: train only a linear layer on the frozen patch features. A high score means the needed information is in the features and can be read out linearly. The [DINOv3 paper](https://arxiv.org/abs/2508.10104) (Meta, 2025-08) compares backbones exactly this way in §6.1.2: segmentation is scored by mIoU on ADE20k (higher is better), depth by RMSE on NYUv2 (lower is better).

## Why self-supervised features are stronger on pixel-level tasks

Start with the numbers. In Table 3 of the DINOv3 paper (frozen backbone, linear head, author-reported), on ADE20k the image–text contrastive SigLIP 2 g/16 scores 42.7 mIoU and PE Core G/14 scores 38.9; the self-supervised DINOv2 g/14 scores 49.5 and DINOv3 7B/16 scores 55.9. Depth RMSE on NYUv2 is 0.494, 0.590, 0.372 and 0.309 in the same order. The paper's own summary: on ADE20k, DINOv3 beats the self-supervised baselines by more than 6 mIoU and the weakly supervised (image–text) baselines by more than 13.

Why? Here is how the papers themselves explain it:

| Explanation | Source | Limitation to consider |
| --- | --- | --- |
| Image–text training keeps only what text can express: a caption only approximates the image, and pixel-level detail may never be learned under that supervision | [DINOv2 paper](https://arxiv.org/abs/2304.07193) (2023-04) §1 | This is a motivation, not a controlled experiment |
| DINO-family objectives contain a patch-level term: besides the image-level DINO loss there is the iBOT loss, which masks some patches for the student while the teacher sees the whole image, and asks the student to predict the teacher's patch features at the masked positions | DINOv2 §4; DINOv3 §4.2 | CLIP-style models are adding local supervision too: [SigLIP 2](../../03-multimodal-learning/siglip.en.md#siglip2) adds self-distillation and masked prediction, yet still trails in DINOv3's Table 3 |
| The information may be present but not in the last layer: the PE paper finds that the best features of its image–text contrastive model sit in intermediate layers and need language or spatial alignment to extract; PE Spatial, made by distilling SAM 2 into PE Core, rises from 38.9 to 49.3 on ADE20k | [PE paper](https://arxiv.org/abs/2504.13181) (2025-04); DINOv3 Table 3 | The 49.3 borrows SAM 2's segmentation supervision, so it is no longer a pure image–text model |
| In a controlled comparison on the same data, visual self-supervision scales better than CLIP and has not saturated at 7B | [Web-SSL](https://arxiv.org/abs/2504.01017) (2025-04) | In the same DINOv3 table, Web-DINO 7B/14 scores only 42.7 on ADE20k; "self-supervised" alone does not guarantee good dense features, data curation and training recipe matter as much |

Two broader limitations also matter. First, all these numbers come from Meta papers, and PE, the comparison model, is also Meta's. The evaluation uses a linear head and a fixed resolution (segmentation at 1024 patch tokens), so the table cannot tell you how much of the gap survives full fine-tuning. Second, if you need text queries ("find every pallet"), visual features alone are not enough: either use a CLIP-style model, or do what DINOv3's dino.txt does and train a text encoder to align with the frozen vision backbone (LiT-style training, DINOv3 §5.3).

## Train longer and local features get worse

Self-supervision does not give good dense features for free either. DINOv3 §4.1 reports that while training a ViT-7B, ImageNet classification accuracy keeps rising, but linear segmentation on Pascal VOC starts falling after about 200k iterations; at the same time, the cosine-similarity map between one patch and all others gets noisier, and unrelated patches start to look alike. The image-level task and the pixel-level task are competing for the same representation.

The paper's fix is called Gram anchoring. Normalise each of an image's $P$ patch features and stack them into a $P\times d$ matrix $X$; $XX^\top$ then holds the cosine similarity of every pair of patches and is called the Gram matrix. During training, the student's Gram matrix is pulled towards that of an earlier checkpoint (the Gram teacher):

$$
\mathcal L_{\rm Gram}=\big\|X_SX_S^\top-X_GX_G^\top\big\|_F^2
$$

$X_S$ and $X_G$ are the student's and the Gram teacher's features, and $\|\cdot\|_F$ is the square root of the sum of squared entries. Multiplying all features by the same orthogonal matrix $Q$ leaves $(XQ)(XQ)^\top=XX^\top$ unchanged, so this loss only constrains "who is similar to whom", not where the features sit. The paper adds this term only after 1M iterations and refreshes the Gram teacher every 10k iterations.

One small calculation (a teaching example): three patches, a and b on the same pallet, c on the floor. Early features a=(1,0), b=(1,0), c=(0,1) give pairwise similarities ab=1, ac=0, bc=0. After long training they become a=(1,0), b=(0.8,0.6), c=(0.6,0.8), with ab=0.8, ac=0.6, bc=0.96: the floor now looks more like the pallet patch b than a and b look like each other. The diagonal is all 1, and each off-diagonal entry appears twice:

$$
\mathcal L_{\rm Gram}=2\big[(0.8-1)^2+(0.6-0)^2+(0.96-0)^2\big]=2\times1.3216\approx2.64
$$

The gradient pushes c away from b without asking any vector to return to its old coordinates.

## Depth: distinguish scale from shift {#depth-relative-depth-is-missing-a-scale}

A pinhole camera maps a point $(X,Y,Z)$ in camera coordinates to pixel $(u,v)$:

$$
u=f\frac{X}{Z}+c_x,\qquad v=f\frac{Y}{Z}+c_y
$$

$f$ is the focal length in pixels (here $f_x=f_y=f$) and $(c_x,c_y)$ is the principal point. Reuse the teaching camera from [camera geometry](camera-geometry.en.md): $f=1000$, $(c_x,c_y)=(960,540)$. The point $P=(1,\,0.5,\,5)$ m lands at $u=1000\times1/5+960=1160$ and $v=1000\times0.5/5+540=640$. Scale the whole scene about the camera centre by $k=2$: $P$ becomes $(2,1,10)$ and still lands on $(1160,640)$, because the $k$ in numerator and denominator cancels. **One image corresponds to every scene whose depths are multiplied by $k$.** No pixel in the image can tell them apart; a model can only guess metres from learned priors (how big cars usually are, how tall doors usually are).

That is why many models output only relative depth, but “relative” does not always mean a single unknown scale. Start with the scale-only case, with no unknown shift. Take three pixels in one image: the road, the car ahead and a distant building, with true depths 4, 8 and 20 m. A model that is off only by scale outputs $(1,2,5)$, the true values divided by 4. Now the lidar says the car is at 8 m: $k=8/2=4$, and the three pixels become $(4,8,20)$. **In this scale-only setting, a single known distance fixes the scale.** It does not fix shape: if the model had said 4 instead of 5 for the building, $k=4$ from the car gives 16 m, and the 4 m error remains.

Other models produce affine-invariant output, where both the scale $s$ and the shift $t$ are unknown: $d=s\hat d+t$. [Marigold](https://arxiv.org/abs/2312.02145) and the first [MoGe](https://arxiv.org/abs/2410.19115) are affine-invariant. Suppose the model outputs $\hat d=(0,1,4)$. Knowing only that the car is at 8 m, the equation $8=s+t$ has infinitely many solutions: $(s,t)=(4,4)$ gives $(4,8,20)$ and $(2,6)$ gives $(6,8,14)$, and both agree with that one observation. Only when you also know the road is at 4 m do you get $t=4$, $s=4$ and a building at 20 m. **With noiseless data and a valid affine relation, two known distances paired with distinct predicted values determine scale and shift.** With noise, fit more correspondences; an output that gives only depth ordering without this affine relation cannot use the same conversion. The [Depth Anything 3 README](https://github.com/ByteDance-Seed/Depth-Anything-3) points out that Depth Anything 2 outputs disparity (inverse depth): then $s$ and $t$ act on $1/d$, there are still two unknowns, but you cannot convert linearly in depth.

<!-- widget:step-depth-scale -->

Where can a known distance come from?

- Sparse lidar: the [Depth Anything V2 README](https://github.com/DepthAnything/Depth-Anything-V2) (entry of 2024-12-22) mentions Prompt Depth Anything, which uses low-resolution lidar to "prompt" the model and get metric depth.
- A known camera height: assume the point is on a flat road, and the camera's height above it is the known distance. The [camera geometry](camera-geometry.en.md) note works out 10 m with a camera 1.5 m above the ground, and how far that number moves if the pitch is off by 1°.
- A stereo baseline, several frames plus an IMU, or another metric model. The Nested version of [Depth Anything 3](https://arxiv.org/abs/2511.10647) uses a multi-view model for pose and depth and a monocular metric model for scale (README).

A monocular model that outputs metric depth directly also has to deal with focal length. From $u-c_x=fX/Z$, doubling $f$ and every point's $Z$ leaves the pixels unchanged: a long lens filming a scene stretched along the optical axis produces the same image as a short lens filming the original. Without the focal length, metric depth is missing one condition. That is why [Depth Pro](https://arxiv.org/abs/2410.02073) (Apple, 2024-10) also estimates the focal length, [Metric3D v2](https://arxiv.org/abs/2404.15506) first transforms images into a canonical camera space, [UniDepth](https://arxiv.org/abs/2403.18913) has a module that estimates its own camera, and [MoGe-2](https://arxiv.org/abs/2507.02546) adds metric scale. The monocular metric version of Depth Anything 3 leaves the focal length to the user: its [README](https://github.com/ByteDance-Seed/Depth-Anything-3) gives "metric depth = focal length (pixels) × network output / 300", so a wrong focal length scales every distance by the same wrong factor.

## Pointmap: a 3D point for every pixel

A depth map gives each pixel one number; a pointmap gives each pixel a 3D point. The [DUSt3R paper](https://arxiv.org/abs/2312.14132) (Naver Labs Europe, 2023-12) defines it in §3 as a $W\times H\times3$ array with a one-to-one mapping between pixels and scene points. Given depth $D$ and intrinsics $K$:

$$
X_{u,v}=D_{u,v}\,K^{-1}\begin{bmatrix}u\\ v\\ 1\end{bmatrix}
$$

Continuing the example: pixel $(1160,640)$ at depth 5 m gives $K^{-1}[1160,640,1]^\top=((1160-960)/1000,\,(640-540)/1000,\,1)=(0.2,\,0.1,\,1)$; times 5 that is $(1,\,0.5,\,5)$, which is $P$ again.

Conversely, a pointmap can be used to estimate camera parameters. The point $(1,0.5,5)$ lands at $u=1160$; assuming the principal point is at the image centre, $f=(u-c_x)Z/X=200\times5/1=1000$. DUSt3R §3.3 estimates the focal length this way, with a robust fit over all pixels (assuming a centred principal point and square pixels).

With two images, it can also estimate the relationship between their views. DUSt3R expresses both pointmaps in the first camera's coordinate frame, so the image locations of the same 3D point give a correspondence. Aligning the same image's points in the two frames (Procrustes in closed form, or the more robust PnP with RANSAC) gives the relative pose. Its output is only defined up to an unknown scale: the training loss first normalises prediction and ground truth by "the average distance of all points to the origin", which removes exactly the $k$ of the previous section.

<!-- paper-figure:dust3r-arch -->

## Cameras, depth and points in one forward pass

The classical route matches features, runs structure-from-motion to get cameras, then does dense reconstruction, and every step is an iterative optimisation; [the next note](reconstruction.en.md) takes it apart. The line below turns it into one forward pass of a network:

| Date | Model | Input → output | How several images end up in one frame |
| --- | --- | --- | --- |
| 2023-12 | [DUSt3R](https://arxiv.org/abs/2312.14132) | Two uncalibrated images → two pointmaps | More than two needs a separate global alignment (optimisation) |
| 2024-06 | [MASt3R](https://arxiv.org/abs/2406.09756) | DUSt3R plus dense local features and fast matching | Same |
| 2025-01 | [Fast3R](https://arxiv.org/abs/2501.13928) | N images → pointmaps | One forward pass, no global alignment; the authors report 1000+ images in one pass |
| 2025-03 | [VGGT](https://arxiv.org/abs/2503.11651) | One to hundreds of images → intrinsics and extrinsics, depth maps, pointmaps, 3D point tracks | One forward pass; the authors report reconstruction in under 1 second; CVPR 2025 Best Paper ([README](https://github.com/facebookresearch/vggt)) |
| 2025-07 | [$\pi^3$](https://arxiv.org/abs/2507.13347) | No reference view; permutation-equivariant in the input order | One forward pass |
| 2025-09 | [MapAnything](https://arxiv.org/abs/2509.13414) | Images plus optional intrinsics, poses or depth → metric depth, ray maps, poses and a scale factor | One forward pass; one model covers 12+ reconstruction tasks |
| 2025-11 | [Depth Anything 3](https://arxiv.org/abs/2511.10647) | Any number of images, posed or not → consistent depth and rays | A plain DINO-style transformer with a single depth-ray target |
| 2025-12 | [D4RT](https://arxiv.org/abs/2512.08924) | One video → depth, spatio-temporal correspondence and cameras, with things in the scene allowed to move | A query-based decoder can ask about any point at any time; CVPR 2026 Best Paper ([closing page](https://cvpr.thecvf.com/Conferences/2026/News/Closing)) |

<!-- paper-figure:vggt-arch -->

A few things worth pausing on:

- **The outputs are redundant.** VGGT returns depth, cameras and a pointmap; its [official README](https://github.com/facebookresearch/vggt) says that points unprojected from depth plus cameras are usually more accurate than those from the pointmap branch. The redundancy can be used as a consistency check.
- **"Depth plus ray" is a factorisation.** A 3D point equals the camera centre plus "distance along the ray times ray direction". Depth Anything 3 argues that predicting this one depth-ray target is enough; the [MapAnything README](https://github.com/facebookresearch/map-anything) outputs `ray_directions`, `depth_along_ray`, `depth_z`, `camera_poses` and `pts3d` separately.
- **z-depth and distance along the ray are not the same thing.** The example pixel's ray direction is $(0.2,0.1,1)$, of length $\sqrt{1.05}\approx1.025$. A z-depth of 5 m is a distance along the ray of $5\times1.025\approx5.12$ m. The further from the principal point, the bigger the gap. Unproject z-depth as if it were distance along the ray and the points near the image edges are pulled in wholesale (in the example the recovered z is only $5/1.025\approx4.88$ m); use distance along the ray as z-depth and they are pushed away (5.12 m).
- **Check where the scale comes from, model by model.** DUSt3R is up to scale; [CUT3R](https://arxiv.org/abs/2501.12387) and MapAnything output metric; Depth Anything 3 gets metric scale from its Metric and Nested versions.

All comparison numbers are author-reported. For example, the Depth Anything 3 [abstract](https://arxiv.org/abs/2511.10647) reports "surpassing prior SOTA VGGT by an average of 44.3% in camera pose accuracy …", and 25.1% on geometry, measured on a benchmark the authors built themselves.

## Can you ship it? Licences of popular checkpoints

As of 2026-10-15, from each repository's licence file and the Hugging Face licence tags (this is not legal advice; licences change, so read the original text before shipping):

| Model | Licence | Commercial use (rough reading) |
| --- | --- | --- |
| DINOv2 | Code and weights Apache-2.0 | Yes |
| DINOv3 | Custom DINOv3 License (2025-08-19) | Meta's blog says commercial use is allowed; ITAR and military uses banned |
| Web-SSL / Web-DINO | Repo and weights CC BY-NC 4.0 | No |
| Perception Encoder | Code and models Apache-2.0 | Yes |
| Depth Anything V2 | Small Apache-2.0; Base/Large/Giant CC BY-NC 4.0 | Small only |
| Depth Anything 3 | Small/Base/Metric-L/Mono-L Apache-2.0; Large/Giant/Nested CC BY-NC 4.0 | First group only |
| Depth Pro | Code under Apple's sample-code licence; weights under the Apple ML Research Model licence, research only | Weights: no |
| MoGe-2 / MoGe-3 | Code and weights MIT | Yes |
| UniDepth | CC BY-NC 4.0 | No |
| DUSt3R / MASt3R / CUT3R | CC BY-NC-SA 4.0 | No |
| Fast3R, VGGT-Ω | FAIR Noncommercial Research License | No |
| VGGT | Original 1B weights CC BY-NC 4.0; since 2025-07-29 also VGGT-1B-Commercial, custom licence, no military use, application required | Commercial checkpoint only |
| $\pi^3$ | Code BSD-3; weights CC BY-NC 4.0 | Weights: no |
| MapAnything | Code Apache-2.0; weights as `map-anything` (CC BY-NC 4.0) and `map-anything-apache` (Apache-2.0) | Use the apache variant |
| D4RT | The project page gives no code, weights or licence | — |

Licences really do change: VGGT added a commercial checkpoint on 2025-07-29 ([README](https://github.com/facebookresearch/vggt)), and Segment Anything moved to a custom licence from SAM 3 on (see [detection and segmentation](detection.en.md)). Within one model family, the small and large sizes often carry different licences.

## Before you build on it

| What to try | What you should observe | What you can NOT conclude |
| --- | --- | --- |
| Multiply a relative depth map by $k$ and project it back with the same $K$ | Every point lands on the same pixel | That the model's relative depth is correct |
| Fix $k$ with one lidar point | Every pixel now has a value in metres | That shape errors at other pixels are fixed too |
| Give affine-invariant output only one known distance | Infinitely many $(s,t)$ solutions | That any one of them will do |
| Unproject `depth_z` as if it were distance along the ray | Points near the image edges pulled in wholesale | That the model is inaccurate at the edges |
| A high linear-probe score on a frozen backbone | The needed information is linearly readable | That the ranking survives full fine-tuning |

## As of October 2026

- Successors: [VGGT-Ω](https://arxiv.org/abs/2605.15195) (2026-05) scales VGGT up and handles static and dynamic scenes; its arXiv comment says CVPR 2026 oral. As of 2026-10-15 we found no public release of a DINOv4 or a Depth Anything 4; not finding one does not prove there is none.
- Downstream use: DINOv3 is already used as an off-the-shelf backbone, for example by the DEIMv2 detector and the SAM 3D Body checkpoints; DINO-WM uses frozen DINOv2 patch features as a world-model representation (see the [JEPA note](../../11-frontier/world-models/jepa.en.md)).

Next, read [3D reconstruction and neural rendering](reconstruction.en.md): once you have cameras and points, how do they become a scene you can render from new viewpoints? Or go back to the [series guide](README.en.md).
