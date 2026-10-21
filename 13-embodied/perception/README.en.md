# Perception & geometry: from pixels to 3D

[中文](README.md) · **English**

> Reading time: ~7 min · Last reviewed: 2026-10

A car's front camera captures a frame. Which objects are present? How far away are they? Is this pedestrian the same person seen in the previous frame? Detection, geometry, and tracking address these questions.

Two foundations help explain the methods. Geometry connects pixels to 3D coordinates; estimation combines observations while accounting for uncertainty. Models may predict camera parameters, depth, and pointmaps directly, but these foundations still matter when checking their outputs and passing them to the rest of the system.

## Why geometry and estimation cannot be skipped

- **A box lives only in pixels.** The detector says "pedestrian at (860, 690)"; the planner needs "10 m ahead, 1 m to the left". Getting there takes the intrinsics, the extrinsics and one more constraint: the ground, a second camera or a depth model (note 1).
- **Small extrinsic errors get amplified.** Several cameras can only be merged into a bird's-eye view once their extrinsics put them in one vehicle frame. Note 1 works it out: a 1° pitch error turns a ground-plane estimate of 10 m into about 8.9 m or 11.3 m.
- **Detection is per frame; planning needs tracks.** Position, velocity, uncertainty and identity must persist across frames, which is filtering and data association (note 2).
- **New models output geometric quantities.** Feed-forward models such as VGGT, Depth Anything 3 and MapAnything return camera intrinsics and extrinsics, depth and pointmaps in one forward pass (note 4); driving simulation with 3D Gaussian splatting first needs accurate camera models and poses (note 5).

Public job descriptions say the same. On 2026-10-15 we read 13 public job descriptions for cars, robots and humanoids and made a rough tally (whether an area is mentioned at all, not how much weight it gets): nearly every one asks for shipping to real vehicles, real robots or production code; about 8 mention real-time or hardware-aware optimisation; about 5 name 3D geometric vision, SLAM/VIO, sensor fusion or BEV. This series only aggregates knowledge areas; it names no company and covers no hiring process or interview content.

## How the five notes divide the work

| Note | Question it answers | Example worked through | Prerequisites on this site |
| --- | --- | --- | --- |
| 1 How a 3D point becomes a pixel | What intrinsics, extrinsics and distortion each do; when one homography is enough; how triangulation recovers depth | A point in the vehicle frame projects to pixel (860, 690); the ground and a second camera recover 10 m | Linear algebra |
| 2 State estimation and tracking | How to keep position and identity across frames; why ByteTrack uses low-score boxes; filter-based and optimisation-based SLAM | One Kalman predict-and-correct step in 1D; Hungarian matching on a 3×3 IoU matrix | Probability (conditional probability, Gaussians); note 1 |
| 3 Detection and segmentation in 2026 | How AP is computed; how DETR and YOLO26 drop NMS; choosing between closed-set real-time, open-vocabulary and SAM 3; which weights you can ship | 3 ground-truth boxes and 5 predictions, all the way to AP@[.5:.95] | ViT, CLIP |
| 4 Frozen backbones and geometry foundation models | Why self-supervised features do well on dense tasks; relative vs metric depth; what a pointmap is; how one forward pass returns cameras, depth and points | The point (1, 0.5, 5) m scaled up 2× lands on the same pixel; fixing scale with one or two known distances; recovering the focal length from a pointmap | ViT, CLIP; note 1 |
| 5 3D reconstruction and neural rendering | What SfM and bundle adjustment solve; NeRF vs 3DGS; why 3DGS became driving-simulation infrastructure, and where it fails | Two cameras compromising on one point's position; four Gaussians blended into one pixel; a pedestrian who only appears after a lane change | Notes 1 and 4; the world-models series |

## Prerequisites on this site and outside textbooks

- [ViT](../../03-multimodal-learning/vit.en.md) and [CLIP](../../03-multimodal-learning/clip.en.md): the backbones and image–text alignment behind notes 3 and 4.
- Probability: note 2 uses conditional probability and the Gaussian distribution.
- [How world models serve robots and cars as simulators](../../11-frontier/world-models/robotics-driving.en.md): the counterpoint when note 5 discusses reconstruction-based simulation.
- Free textbooks: [Szeliski, *Computer Vision*, 2nd ed.](https://szeliski.org/Book/) and [Barfoot, *State Estimation for Robotics*, 2nd ed.](https://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser24.pdf); for Hartley & Zisserman only the [sample chapters](https://www.robots.ox.ac.uk/~vgg/hzbook/) are free, including the chapter on epipolar geometry.
- Courses: CMU [16-385](http://16385.courses.cs.cmu.edu/) (camera models, two-view geometry, tracking) and [16-822](https://geometric3d.github.io/) (projective geometry, the fundamental matrix, learning-based SfM/SLAM), both offered in Fall 2026; [Stanford CS231n](https://cs231n.stanford.edu/), Spring 2026, covers detection, segmentation and 3D vision.

## Reading paths by role

- **Perception / CV roles:** 1 → 3 → 4 → 2, then 5 when you need reconstruction or simulation.
- **Autonomous-driving roles:** 1 → 2, then the [autonomous-driving series](../driving/README.en.md) (starting with [from surround cameras to BEV and occupancy](../driving/bev-occupancy.en.md)), and back to 5 when simulation comes up.
- **Robotics / manipulation roles:** 1, then the [robot-learning series](../robot-learning/README.en.md); add 2 for mobile robots.

## As of October 2026: what changed in the last two years

These are the main changes from 2024 to October 2026. All of them will age; cite them with their dates.

- **Segmentation takes a concept, not just a click.** [SAM 3](https://arxiv.org/abs/2511.16719) (arXiv 2025-11-20) takes a noun phrase or an exemplar and returns masks and IDs for every matching instance in images and video; SAM 3.1 (2026-03-27) adds multi-object tracking. The licence changed too: SAM 1 and 2 are Apache-2.0, while SAM 3 uses a custom SAM License ([repository](https://github.com/facebookresearch/sam3)).
- **Self-supervised backbones reached 7B parameters, and dense features became the selling point.** [DINOv3](https://arxiv.org/abs/2508.10104) (2025-08-13) is a ViT-7B trained on LVD-1689M; Gram anchoring keeps dense features from degrading during long training. With a frozen backbone the authors report 66.1 mAP on COCO detection and 63.0 mIoU on ADE20k.
- **Depth went from one image to any number of views, with scale.** [Depth Anything 3](https://arxiv.org/abs/2511.10647) (2025-11-13) predicts consistent depth and rays from any number of views; the second version of [MoGe](https://github.com/microsoft/MoGe) adds metric scale, and MoGe-3 weights were released on 2026-08-18. Many of the strongest depth checkpoints have non-commercial licences.
- **Feed-forward 3D reconstruction became the main line.** [VGGT](https://github.com/facebookresearch/vggt) won the CVPR 2025 Best Paper; the CVPR 2026 Best Paper went to D4RT (dynamic 4D reconstruction), with an honourable mention for SAM 3D ([CVPR 2026 closing page](https://cvpr.thecvf.com/Conferences/2026/News/Closing)).
- **3D Gaussian splatting became production infrastructure for driving simulation.** NVIDIA Omniverse NuRec became generally available on 2026-03-25 ([NVIDIA forum announcement](https://forums.developer.nvidia.com/t/nvidia-omniverse-nurec-now-generally-available/364767)); for [Instant NuRec](https://arxiv.org/abs/2607.14203) (2026-07-15), the authors report building a scene from a 10–20 s multi-camera log in about 1.5 s. Waymo takes the opposite view: it says its Genie-3-based generative World Model (2026-02-06) handles counterfactuals that leave the logged route, without a quantitative comparison (vendor-reported, [Waymo blog](https://waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation/)).
- **Real-time detection split into two camps.** The DETR-style [RF-DETR](https://arxiv.org/abs/2511.09554) (ICLR 2026) claims to be the first real-time detector above 60 AP on COCO, and DEIMv2 plugs in DINOv3 features; Ultralytics' [YOLO26](https://docs.ultralytics.com/models/yolo26/) (released 2026-01-14) added an NMS-free inference head (though its docs still default prediction to the head that uses NMS), and YOLO27 is only a preview as of October 2026.
- **Open-vocabulary detection is merging into segmentation models and being tested on harder ground.** The SAM 3 paper reports box AP on COCO and LVIS directly and calls these zero-shot results; [RF100-VL](https://arxiv.org/abs/2505.20612) shows vision-language detectors below 2% zero-shot on some medical datasets.
- **Benchmark culture changed.** COCO box AP still appears in every detection paper, but as one column next to RF100-VL, ODinW13, LVIS or SA-Co; Papers with Code reportedly shut down in July 2025.
- **Check the licence before using weights.** DUSt3R, MASt3R and CUT3R are CC BY-NC-SA 4.0; Fast3R and VGGT-Ω use the FAIR Noncommercial Research License; π^3 weights and CoTracker3 are CC BY-NC 4.0; the original 3DGS code has an Inria non-commercial licence; Ultralytics models are AGPL-3.0 or an Enterprise licence. The licence table for detection models is in note 3, and the one for depth and 3D models is in note 4.

## Reading order

1. [How a 3D point becomes a pixel](camera-geometry.en.md): the pinhole model, intrinsics and extrinsics, distortion, the epipolar constraint and triangulation, worked by hand on one point
2. [State estimation and tracking](tracking.en.md): the Kalman filter, Hungarian matching, ByteTrack's low-score boxes, and filter-based vs optimisation-based SLAM
3. [Detection and segmentation in 2026](detection.en.md): AP by hand, NMS vs one-to-one matching, choosing between closed-set real-time, open-vocabulary and SAM 3, and licences of common weights
4. [Frozen backbones and geometry foundation models](foundation-models.en.md): DINOv3, Depth Anything 3, VGGT, and one forward pass that outputs cameras, depth and pointmaps
5. [3D reconstruction and neural rendering](reconstruction.en.md): from SfM to Gaussian-splatting simulators, and where reconstruction is not enough
