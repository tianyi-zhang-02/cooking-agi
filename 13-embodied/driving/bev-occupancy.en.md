# Autonomous driving: from surround cameras to bird's-eye view and occupancy

[中文](bev-occupancy.md) · **English**

> Reading time: ~12 min · Last reviewed: 2026-10

The planner wants to ask "is there anything in the cell 8 m ahead and 2 m to the left?", but 6–8 cameras deliver pixels in their own views, and in a single image nobody knows how far each pixel is from the car. **Putting every camera's features into one top-down grid (bird's-eye view, BEV) gives planning one shared metric frame; the hard part is always depth: lift-splat spreads each pixel's features along its line of sight according to a depth distribution, BEV queries go the other way and let each cell fetch features from the images, and occupancy grids refine the output to "is this piece of space taken", without first deciding what it is.**

## Why a top-down map

Camera images are perspective views: distant cars are small, near ones large, and a car straddling the border between two cameras gets cut in half. Planning needs a metric, top-down frame shared by all sensors: which way the lane runs, how many metres to the lead car, whether the next cell over is drivable. A BEV grid divides the area around the ego car into cells, each holding a feature vector; detection, map and planning heads all work on this grid. A lidar point cloud already carries 3D coordinates and can be dropped straight in; what cameras lack is depth.

Two common settings: [Lift-Splat-Shoot](https://arxiv.org/abs/2008.05711) (Philion and Fidler, ECCV 2020) covers ±50 m around the ego car at 0.5 m per cell, a 200×200 grid; [BEVFormer](https://arxiv.org/abs/2203.17270) (ECCV 2022) uses ±51.2 m at 0.512 m per cell, also 200×200. With 64 float16 channels per cell, one BEV feature map is about $200\times200\times64\times2\approx5.1$ MB, a teaching estimate.

Under the pinhole camera model, one pixel corresponds to one ray leaving the camera centre, and every point on that ray projects to the same pixel. From one image alone, you cannot tell where along the ray the thing in that pixel is. For the projection itself, see [camera geometry](../perception/camera-geometry.en.md).

## Work it once: how one pixel spreads into bird's-eye cells

All numbers below are a teaching setup. The camera sits at the ego origin and faces forward (the $x$ axis). One pixel lies left of the image centre, and its ray satisfies $y=0.25\,d$: $d$ is depth along the optical axis and $y$ is the lateral distance to the left. Look only at the ground plane and ignore height for now.

The LSS paper (§3.1): for each pixel the network outputs a context feature $c\in\mathbb R^C$ and a distribution $\alpha$ over $|D|$ discrete depths; the feature placed at the point at depth $d$ is their product:

$$
c_d=\alpha_d\,c,\qquad \sum_{d\in D}\alpha_d=1 .
$$

Take only 4 depths, $D=\{4,8,12,16\}$ m, network outputs $\alpha=(0.1,0.6,0.2,0.1)$ and $c=(2,1)$. Here $C=2$, and the two channels can be read as how much it looks like "car" and like "road surface"; real channels have no such tidy meaning, it just makes the example easy to read. BEV cells are 8 m × 8 m (the paper uses 0.5 m; they are enlarged here so that two points land in one cell).

| Depth $d$ | $\alpha_d$ | Top-down position $(x,y)$ | Feature $c_d=\alpha_d c$ | Cell it lands in |
| --- | --- | --- | --- | --- |
| 4 m | 0.1 | (4, 1) | (0.2, 0.1) | $x\in[0,8)$ |
| 8 m | 0.6 | (8, 2) | (1.2, 0.6) | $x\in[8,16)$ |
| 12 m | 0.2 | (12, 3) | (0.4, 0.2) | $x\in[8,16)$ |
| 16 m | 0.1 | (16, 4) | (0.2, 0.1) | $x\in[16,24)$ |

That is the **lift**: one pixel becomes 4 points with features along its ray. Then the **splat**: the paper assigns each point to its nearest pillar (a vertical cell of unlimited height) and simply adds the features within a cell (sum pooling). All 4 points have $y$ in $[0,8)$, so:

- cell $[0,8)$: $(0.2,0.1)$
- cell $[8,16)$: $(1.2+0.4,\;0.6+0.2)=(1.6,0.8)$
- cell $[16,24)$: $(0.2,0.1)$

The three cells add up to $(2.0,1.0)=c$. Because $\sum\alpha_d=1$, the total feature from this pixel is unchanged; it is only distributed over distances along the ray. If the car really is at 8 m, 80% of its "car" feature lands in the right cell, 10% in a nearer cell and 10% in a farther one. The familiar "smear along the line of sight" in camera BEV comes from exactly this.

<!-- paper-figure:lss-lift -->

At the paper's real resolution (0.5 m cells, one depth per 1 m), neighbouring depth points on the same ray rarely share a cell; the adding happens mostly between neighbouring pixels and where two cameras' fields of view overlap.

### Change one condition: a sharper, flatter or misplaced distribution

- **Sharper.** $\alpha=(0,1,0,0)$: all of the feature lands in $[8,16)$. This amounts to estimating one depth and then projecting; the paper likens it to pseudo-lidar.
- **Flatter.** $\alpha=(0.25,0.25,0.25,0.25)$: each depth point gets $(0.5,0.25)$, so $[8,16)$ receives $(1.0,0.5)$ and the other two cells $(0.5,0.25)$ each. Every position on the ray has the same representation; the paper likens it to OFT. In the paper's words, the network can therefore choose between placing the feature at one location and spreading it along the whole ray when depth is ambiguous.
- **Misplaced.** The car is really at 16 m, but $\alpha$ is still $(0.1,0.6,0.2,0.1)$: the right cell $[16,24)$ gets only 10%, and most lands in $[8,16)$. In the top-down map the car appears about 8 m too close. Camera depth error turns directly into position error in BEV; this is the most direct motivation for fusing cameras with lidar.

<!-- widget:step-lift-splat -->

## The paper's real settings and cost

The LSS paper's settings (§4.1): depths limited to between 4 m and 45 m, one every 1 m; input images resized and cropped to 128×352; a 200×200 BEV grid. An $H\times W$ feature map produces $|D|\cdot HW$ points (here $|D|=41$), times the number of cameras, so the point count is large. Section 4.2 does the sum pooling with a "cumsum trick": sort points by cell index, take a cumulative sum over features, and subtract at each segment boundary; the authors derive this layer's analytic gradient, which doubles training speed. The paper reports the full forward pass at 35 Hz on a Titan V, an author measurement on 2020 hardware.

The **shoot** in the name is planning: the network outputs a cost map in BEV, "shoots" 1,000 template trajectories (from K-means over expert trajectories) onto it, sums the cost along each, trains this as classification, and at test time takes the lowest-cost one. It is a learned version of the "candidate trajectories plus scoring" in note 2.

## Pulling instead: BEV queries

[BEVFormer](https://arxiv.org/abs/2203.17270) does not push outward from pixels; it pulls from cells into the images. It gives every BEV cell a learnable query. Per §3.3 and §4.2 of the paper: each cell is raised into a vertical pillar, with 4 reference points spaced evenly between −5 m and 3 m in height; the camera intrinsics and extrinsics project these points into the images, and features are taken only from the cameras they hit; around each projected point, deformable attention samples 4 more points. A separate temporal self-attention fuses in the previous BEV, aligned for ego motion. The abstract reports 56.9 NDS on the nuScenes test set, 9.0 above the previous best.

<!-- paper-figure:bevformer-arch -->

Use the same camera as before: the cell centred at $(12,3)$ satisfies $3=0.25\times12$, so it lies exactly on the ray from before and projects to the same pixel. $(4,1)$, $(8,2)$ and $(16,4)$ all project to the same column too. If each cell just copied that pixel's feature, all 4 cells would get the identical thing: the ambiguity along the ray has not gone away, it has moved. What separates them: the same cell is seen by another camera, or at the next time step from another angle; reference points at different heights project to different pixel rows, and a car body and the road surface look different at different heights; and the attention weights themselves are learned.

| Comparison | Lift-splat (push) | BEV query (pull) |
| --- | --- | --- |
| Direction | Each pixel pushes features to several depths along its ray | Each BEV cell fetches features from the images |
| How depth is handled | Explicitly predicts a depth distribution $\alpha$ | No depth prediction; relies on multiple views, multiple heights and attention |
| What compute grows with | Pixels × depth bins | Cells × reference points × sampling points |
| Typical failure | Wrong depth distribution, features smear along the ray | Cells projecting to the same pixel stay confused; needs time and other views |

Tesla presented the same family of ideas at AI Day on 2021-08-19: an output-space raster tiled with positional encodings attends to image features to turn image space into BEV, followed by a feature queue and a video module for temporal fusion ([video](https://www.youtube.com/watch?v=j0z4FweCy4M), from auto-captions of the talk). That describes Tesla's earlier modular stack.

## Not filling the grid: sparse queries

Most cells in a 200×200 grid are empty road. Another line of work builds no dense BEV and uses only a set of object queries. The abstract of [PETR](https://arxiv.org/abs/2203.05625) (ECCV 2022) says it encodes the positions of 3D coordinates into image features, and object queries attend directly to these "3D-position-aware" features to detect objects; it reports 50.4 NDS and 44.1 mAP on nuScenes. Per its abstract, [StreamPETR](https://arxiv.org/abs/2303.11926) carries object queries forward frame by frame for temporal fusion; [Sparse4D](https://arxiv.org/abs/2211.10581) samples sparsely in space and time. [SparseDrive](https://github.com/swc-17/SparseDrive) connects this line to planning: fully sparse perception plus a parallel motion planner (per its README).

The gain is that compute grows with the number of queries, not the grid area; the cost is that you do not get a dense drivable-area map for free. Things that are not "objects", such as the shape of the road or a pile of gravel, need a separate map head or occupancy head.

## Cameras plus lidar: BEVFusion

[BEVFusion](https://arxiv.org/abs/2205.13542) (ICRA 2023) puts camera features and lidar features into the same BEV and fuses them there, serving several tasks. Its abstract reports 1.3% higher mAP and NDS on nuScenes 3D detection, 13.6% higher mIoU on BEV map segmentation, 1.9x lower computation cost, and, after optimising BEV pooling, more than 40x lower latency for the view transformation. Back in the example above: lidar measures 16 m directly, so the feature no longer depends on $\alpha$ guessing right.

Industry holds both positions. Waymo's December 2025 Foundation Model fuses camera, lidar and radar in a Sensor Fusion Encoder ([blog](https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving)); Tesla's [Q1 2025 shareholder letter](https://digitalassets.tesla.com/tesla-contents/image/upload/IR/TSLA-Q1-2025-Update.pdf) describes a vision-only architecture built on end-to-end neural networks.

## Occupancy: ask whether it is taken before asking what it is

A 3D box is usually 3 numbers for the centre, 3 for size and 1 for heading, plus a class. A ladder lying across the lane, an overhanging branch or irregularly shaped cargo are poorly described by boxes, and a detector may simply miss things not in its class list. The abstract of [Occ3D](https://arxiv.org/abs/2304.14365) (NeurIPS 2023) defines occupancy prediction as estimating the occupancy state and semantics of every location in a scene, to make up for boxes' weakness on fine geometry and out-of-vocabulary objects.

The [CVPR 2023 occupancy prediction challenge](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction) used Occ3D-nuScenes: 0.4 m voxels in a 200×200×16 grid, 18 classes (one of them "free", meaning empty), cameras only at inference, ranked by mIoU. Work it out: horizontally $200\times0.4=80$ m square, vertically $16\times0.4=6.4$ m tall; each frame has $200\times200\times16=640{,}000$ voxels, each needing a class. Per the abstract of the [technical report](https://arxiv.org/abs/2307.01492), the winner, FB-OCC, reached 54.19% mIoU.

mIoU computes intersection over union per class, then averages over classes. Say for "car" the model predicts 120 voxels, the ground truth has 100, and 80 overlap:

$$
\text{IoU}=\frac{80}{120+100-80}=\frac{80}{140}\approx0.571 .
$$

Rare and common classes weigh the same, so doing badly on a few small classes pulls mIoU down noticeably.

Tesla presented occupancy networks twice in 2022, both in talks, paraphrased here from auto-captions. CVPR 2022 WAD keynote (2022-06-20, [video](https://www.youtube.com/watch?v=jPCV4GKX9Dw)): the video streams of all 8 cameras together output voxel occupancy probabilities and occupancy flow, with queries carrying 3D positional encodings attending to the multi-camera image features; the talk gives 8 cameras at 1.2 MP and 36 fps, and the network running within 10 ms (vendor-reported). AI Day 2022 (2022-09-30, [video](https://www.youtube.com/watch?v=ODSJsviD_SU)): inputs are 12-bit raw photon-count images with no ISP processing; a fixed voxel grid is not precise enough for planning, so each voxel carries features and an MLP can be queried at any point; training uses auto-labels rather than human labelling.

### Where the labels come from

At 640,000 voxels per frame, nobody labels them by hand. Some approaches:

- **Turn sparse lidar into dense voxels.** The Occ3D abstract lists three steps: voxel densification, occlusion reasoning and image-guided voxel refinement, producing "visibility-aware" labels. Per its abstract, [SurroundOcc](https://arxiv.org/abs/2303.09551) (ICCV 2023) generates dense ground truth from multi-frame lidar plus Poisson reconstruction.
- **Offboard auto-labelling.** An onboard model can only see the past; offboard labelling can look at the whole sequence, past and future frames, with bigger models. Waymo's [3D Auto Labeling](https://arxiv.org/abs/2103.05073) (Qi et al., 2021) detects objects offboard from entire point-cloud sequences.

Labels made this way inherit lidar's blind spots: voxels that were occluded or never scanned have no reliable ground truth. Evaluation therefore often scores only "visible" voxels: the CVPR 2023 challenge README states that mIoU is computed only on voxels visible in the current camera view, and unobserved voxels are left out. When reading other results, check which convention was used.

## What they became inside end-to-end stacks

In the [text version](https://x.com/aelluswamy/status/1981644831790379245) of his October 2025 ICCV talk, Tesla's Elluswamy wrote that besides control commands the end-to-end model can produce interpretable intermediate tokens, such as 3D geometry; the text does not mention occupancy or objects. That the same model can be asked to output occupancy, objects and so on comes only from the talk video's auto-captions ([video](https://www.youtube.com/watch?v=IRu-cPkpiFk)), with no transcript to check against. Waymo's Foundation Model still has a World Decoder that outputs other road users' behaviour, maps and trajectories. [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) 0.0.4 (2026-08) added an occupancy modality. BEV and occupancy have not disappeared: they became auxiliary supervision, debugging views, and intermediate results that safety checks can read.

## What to look at when reproducing and debugging

| What to try first | What to observe | What you can NOT conclude |
| --- | --- | --- |
| Replace the predicted depth distribution with lidar depth | Objects stop smearing along the line of sight; distance errors shrink | Camera depth estimation is already good enough |
| Randomly drop one camera during training | Less degradation when a camera is missing at test time; the LSS paper reports the model trained this way was also best with all 6 cameras | Redundant cameras can be removed |
| Add noise to the extrinsics in training and testing | The LSS paper reports: with little test noise, training without noise is best; with heavy noise, training with noise holds up better | Training noise cancels real calibration drift |
| Compute mAP / mIoU by distance band | Whether far ranges are clearly worse | A good average means long range is reliable |
| Evaluate on visible voxels only vs all voxels | How far apart the two scores are | The model guessed right where nothing was visible |

References: [Lift-Splat-Shoot](https://arxiv.org/abs/2008.05711) · [BEVFormer](https://arxiv.org/abs/2203.17270) · [BEVFusion](https://arxiv.org/abs/2205.13542) · [Occ3D](https://arxiv.org/abs/2304.14365). Next: [motion forecasting and planning](prediction-planning.en.md). For camera projection see [camera geometry](../perception/camera-geometry.en.md); for reconstruction used as driving simulation see [3D reconstruction and neural rendering](../perception/reconstruction.en.md).
