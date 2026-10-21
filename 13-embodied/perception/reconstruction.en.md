# Perception & geometry: 3D reconstruction and neural rendering

[中文](reconstruction.md) · **English**

> Reading time: ~13 min · Last reviewed: 2026-10

Your fleet recorded a 20-second multi-camera log. You changed the planner and want to know whether it would change lanes at the same moment, and what the cameras would see afterwards. Replaying the recording does not work: once the car drives differently from the log, the images no longer match (see the [world-models note](../../11-frontier/world-models/robotics-driving.en.md) for open versus closed loop). Could you first reconstruct this stretch of road as a 3D scene, then render it from new positions?

Yes, but its trustworthy range is limited to regions observed in the log. 3D Gaussian splatting (3DGS) supports real-time rendering and object-level editing, making it useful for driving simulation. When the car moves into an unobserved region, reconstruction lacks the necessary data. Generative world models try to supply the missing scene content.

Read this series' [camera geometry](camera-geometry.en.md) and [frozen backbones and geometry foundation models](foundation-models.en.md) first: this note reuses their pinhole projection, pointmaps and scale ambiguity.

## Step one: SfM recovers cameras and sparse points

Before reconstructing anything you need to know where each image was taken from and with what camera. Structure-from-motion (SfM) recovers this from the images alone. The [CVPR 2016 paper by Schönberger and Frahm](https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf) released COLMAP as open source alongside the paper, and the 3DGS paper discussed below calibrates its input cameras with it. The incremental SfM pipeline is:

```text
extract features and match image pairs → geometric verification drops pairs that do not agree
→ pick one pair for an initial two-view reconstruction
→ loop: register a new image with PnP → triangulate new points → bundle adjustment → filter outliers
```

Bundle adjustment (BA) is the overall correction inside this loop. Equation (1) of the paper adjusts all camera parameters $P_c$ and 3D points $X_k$ together to minimise the reprojection error:

$$
E=\sum_j\rho_j\Big(\big\|\pi(P_c,X_k)-x_j\big\|_2^2\Big)
$$

$x_j$ is observation $j$, the pixel where point $X_k$ was detected in camera $c$; $\pi$ is the projection function; $P_c$ includes pose, focal length and distortion; $\rho_j$ is a robust loss that limits the influence of mismatches, and COLMAP uses the Cauchy function in local BA. BA is usually solved with Levenberg–Marquardt, using the Schur complement to solve for the cameras first and then back-substitute the points.

Triangulation in the [camera geometry](camera-geometry.en.md) note ends by minimising one point's reprojection error; BA brings all the cameras into the adjustment too. One calculation (a teaching example, horizontal direction only). Reuse that note's teaching camera, $f=1000$ and $c_x=960$; two cameras face the same way, along $+z$, with camera 1 at the origin and camera 2 at $x=1$ m. To keep it hand-computable, fix the point's depth at $Z=5$ m and solve only for its horizontal coordinate $X$. Camera 1 sees $u_1=1166$, which on its own implies $X=(1166-960)\times5/1000=1.03$; camera 2 sees $u_2=960$, which on its own implies $X=1.00$. The two observations are noisy and contradict each other. Plugging in the projection formula, the total error is:

$$
E(X)=(200X-206)^2+(200X-200)^2
$$

At $X=1.03$, $E=0+36=36$; at $X=1.00$, $E=36+0=36$; at $X=1.015$ both residuals are 3 px and $E=18$ is the minimum. BA makes this kind of compromise, only it moves every camera, every point, the focal lengths and the distortion at once. If camera 2's observation were actually a mismatch at $u_2=1080$, the squared loss would drag the point to $X=1.315$ and leave camera 1 with a 57 px residual. A robust loss makes the cost of large residuals grow more slowly to reduce outlier influence; it does not guarantee automatic correction ([Ceres documentation: LossFunction](https://github.com/ceres-solver/ceres-solver/blob/master/docs/source/nnls_modeling.rst)).

This example has only two contradictory, equally weighted observations. Applying the same robust loss to both gives the objective no information about which one is the mismatch. The result also depends on the loss and its scale, observation weights, and initialization. Reliable outlier identification generally needs more consistent views, geometric constraints, or explicit outlier checks; a robust loss does not replace match validation.

There is one thing BA cannot fix: scale all camera centres and all points together by $k$ (plus any global rotation and translation) and every reprojection stays the same. In the example, moving camera 2 to $x=k$ and the point to $(kX,kZ)$ gives $u_2=1000(kX-k)/(kZ)+960$, unchanged. From images alone, SfM results are only defined up to scale; it is the same $k$ as single-image depth in [the previous note](foundation-models.en.md). Driving logs usually come from cameras on a calibrated rig ([Instant NuRec](https://arxiv.org/abs/2607.14203) takes a calibrated multi-camera rig as input), so the baseline is known and so is the scale.

The feed-forward models from the previous note plug in here too. The [VGGT README](https://github.com/facebookresearch/vggt) (entry of 2025-06-02) provides a script that saves predictions in COLMAP format, optionally runs BA once more, and hands the result straight to gsplat or other Gaussian splatting libraries.

## Step two: from sparse points to a renderable scene

SfM gives cameras and a sparse set of points, which is not enough to render a full image from a new viewpoint. NeRF and 3DGS both fill this gap, and the [3DGS paper](https://arxiv.org/abs/2308.04079) (Inria and MPII, 2023-08) points out in §2 that their formula for a pixel's colour is essentially the same: stack a sequence of semi-transparent things along the line of sight, front to back.

$$
C=\sum_{i=1}^{N}T_i\,\alpha_i\,\mathbf c_i,\qquad T_i=\prod_{j=1}^{i-1}(1-\alpha_j)
$$

$\mathbf c_i$ is the colour of element $i$, $\alpha_i$ the fraction of light it blocks, and $T_i$ how much light is left when the ray reaches it (the transmittance). The difference is where $\alpha_i$ comes from:

$$
\alpha_i^{\rm NeRF}=1-e^{-\sigma_i\delta_i},\qquad
\alpha_i^{\rm 3DGS}=o_i\exp\!\Big(-\tfrac12(\mathbf p-\boldsymbol\mu_i)^\top\Sigma_i^{-1}(\mathbf p-\boldsymbol\mu_i)\Big)
$$

- [NeRF](https://arxiv.org/abs/2003.08934) (2020-03) uses an MLP that maps position and viewing direction to a density $\sigma$ and a colour. Rendering a pixel takes many samples along the ray, one MLP query per sample, with $\delta_i$ the spacing between samples. It needs images with known poses and optimises each scene separately.
- 3DGS stores the scene as a large set of explicit 3D Gaussians: position, covariance $\Sigma^{\rm 3D}=RSS^\top R^\top$ (rotation times scale), opacity $o_i$, and a view-dependent colour stored as spherical-harmonic coefficients. To render, each Gaussian is projected to the screen, giving a 2D mean $\boldsymbol\mu_i$ and covariance $\Sigma_i$; the formula above is evaluated at the pixel position $\mathbf p$ and multiplied by the opacity. The initial Gaussians come from the SfM sparse points; training minimises $(1-\lambda)L_1+\lambda L_{\text{D-SSIM}}$ ($\lambda=0.2$) between rendered and real images, and keeps cloning, splitting and removing Gaussians to adjust the density.

3DGS is fast because of how it renders. The screen is first cut into 16×16-pixel tiles, all Gaussians are sorted once by "tile ID plus depth" with a GPU radix sort, and then the pixels in each tile blend front to back and stop when the transmittance is nearly used up. There is no repeated network query along a ray and no per-pixel sort.

<!-- paper-figure:3dgs-pipeline -->

## One calculation: four Gaussians make one pixel

A teaching example. One pixel is covered by four Gaussians, already sorted from near to far. $g$ is the Gaussian's value at this pixel, $\exp(-\tfrac12m^2)$, where $m$ is the Mahalanobis distance from the pixel to the Gaussian's centre: $g=1$ when the pixel sits exactly at the centre, and $g=0.5$ when $m^2=2\ln2\approx1.39$.

| # | Depth | Colour (R, G, B) | Opacity $o$ | $g$ at this pixel | $\alpha=o\,g$ |
| --- | --- | --- | --- | --- | --- |
| 1 | 4 m | (1, 0, 0) red | 0.5 | 1.0 | 0.5 |
| 2 | 6 m | (0, 1, 0) green | 0.8 | 0.5 | 0.4 |
| 3 | 9 m | (0, 0, 1) blue | 1.0 | 0.5 | 0.5 |
| 4 | 20 m | (1, 1, 1) white | 0.9 | 1.0 | 0.9 |

Blend them front to back, one at a time:

| Step | $T$ coming in | Weight $w=T\alpha$ | $T$ going out | Accumulated colour |
| --- | --- | --- | --- | --- |
| 1 | 1 | 0.5 | 0.5 | (0.5, 0, 0) |
| 2 | 0.5 | 0.2 | 0.3 | (0.5, 0.2, 0) |
| 3 | 0.3 | 0.15 | 0.15 | (0.5, 0.2, 0.15) |
| 4 | 0.15 | 0.135 | 0.015 | (0.635, 0.335, 0.285) |

The four weights add up to 0.985, which is $1-0.015$; the remaining 1.5% goes to the background (black here). The final colour, (0.635, 0.335, 0.285), is reddish.

<!-- widget:step-alpha-blend -->

Three more things to notice:

- **Order matters.** If the sort goes wrong and Gaussians 1 and 4 swap, the farthest, white Gaussian is blended first and the order becomes white, green, blue, red; the weights become 0.9, 0.04, 0.03 and 0.015, and the colour is (0.915, 0.94, 0.93), almost pure white: the far object shows through in front. The leftover transmittance is still 0.015, because $\prod(1-\alpha_j)$ does not depend on order; the colour does. 3DGS sorts once per tile, not per pixel, and §6 of the paper acknowledges that blending is then approximate, with the error becoming negligible as Gaussians shrink to pixel size; the "popping" mentioned in its limitations section comes partly from Gaussians suddenly swapping their front-to-back order.
- **The same weights can render depth, but the result may land in empty space.** $\sum w_iz_i=0.5\times4+0.2\times6+0.15\times9+0.135\times20=7.25$, and dividing by 0.985 gives about 7.36 m. Nothing is there; it is a weighted average of several surfaces. Think twice before treating such a depth as a lidar reading.
- **Numerical details.** The paper's appendix states that terms with $\alpha<1/255$ are skipped, $\alpha$ is clamped to at most 0.99, and blending stops before the accumulated opacity would exceed 0.9999. The example accumulates to 0.985, so all four are blended.

Every quantity above is differentiable in $o$, $g$ (and so in position and covariance) and colour; training makes the colour computed this way match the real photo.

## NeRF or 3DGS

| | NeRF (2020) | 3DGS (2023) |
| --- | --- | --- |
| What the scene is stored as | One MLP: position and viewing direction → density and colour | A large set of explicit anisotropic Gaussians |
| How one pixel is computed | Sample along the ray, one MLP query per sample | Project, sort per tile, blend front to back |
| Rendering speed | The 3DGS paper notes that NeRF needs costly random sampling to find its samples, with resulting noise and compute | Real-time at 1080p: arXiv v1 says ≥30 fps, the project page says ≥100 fps; the two sources disagree |
| Memory | Mostly the network weights | The 3DGS paper says its memory use is significantly higher than NeRF-based methods; peak GPU memory can exceed 20 GB when the unoptimised prototype trains large scenes |
| Moving one car | The scene lives in the network weights; there is no ready-made "this car" unit | A car is a group of Gaussians that can be moved as a whole |
| Camera model | Per-ray sampling; a distorted camera is just a different set of ray directions | Rasterisation assumes an ideal pinhole; see 3DGUT in the next section |

## Why 3DGS became the base layer of driving simulation

The pieces, in date order (all numbers are author- or vendor-reported):

1. **Speed.** [Street Gaussians](https://arxiv.org/abs/2401.01339) (2024-01, ECCV 2024) reports rendering at 135 FPS at 1066×1600 after about 30 minutes of training on KITTI and Waymo; [OmniRe](https://arxiv.org/abs/2408.16760) (2024-08, ICLR 2025) reports simulation at about 60 Hz. Closed-loop simulation re-renders at every step, so speed is a precondition.
2. **Editable by object.** Street Gaussians splits the background and each vehicle into separate Gaussian groups, with the vehicles following their tracked poses; OmniRe organises vehicles, pedestrians (Gaussians driven by the SMPL body model) and cyclists into a scene graph. Moving, deleting and inserting vehicles is the basis for building "what if" scenarios.
3. **Realistic cameras.** [3DGUT](https://arxiv.org/abs/2412.12507) (NVIDIA, 2024-12, CVPR 2025) replaces the EWA splatting projection approximation of the original 3DGS with an unscented transform, supporting distorted and rolling-shutter cameras and secondary rays such as reflection and refraction, while keeping rasterisation speed.
4. **Scenes in seconds.** [Instant NuRec](https://arxiv.org/abs/2607.14203) (NVIDIA, 2026-07-15) is a feed-forward model: it turns a 10–20 s multi-camera log directly into layered static and dynamic 3DGS plus a sky cubemap and per-camera ISP corrections in about 1.5 s; it supports non-pinhole cameras through 3DGUT and works with AlpaSim for closed-loop simulation. The authors report PSNR 2.01 dB above the strongest baseline on Waymo Open; the abstract does not say whether the test views lie on the logged trajectory.
5. **A product.** NVIDIA Omniverse NuRec became generally available on 2026-03-25 ([developer forum announcement](https://forums.developer.nvidia.com/t/nvidia-omniverse-nurec-now-generally-available/364767)): a reference container, gsplat-based renderers, the NCore data format, the Cosmos-based Fixer model (which cleans artefacts out of novel views), and Asset Harvester, which generates assets for areas that were never observed. An earlier [NVIDIA technical blog](https://developer.nvidia.com/blog/accelerating-av-simulation-with-neural-reconstruction-and-world-foundation-models/) (2025-10-28) says CARLA integrates NuRec's rendering APIs and lists Foretellix, MathWorks, Mcity and others as partners (vendor-reported).
6. **Inside benchmarks.** [NAVSIM v2](https://arxiv.org/abs/2506.04218) (2025-06, CoRL 2025) runs pseudo-simulation in two stages: score the plan first, then evaluate from 3DGS-rendered perturbed start states, weighted by proximity to the first stage's endpoint. The abstract of [HUGSIM](https://arxiv.org/abs/2412.01718) (2024-12) describes it as a real-time closed-loop simulator built on 3DGS; we checked only the abstract.

Not everyone discloses what they use. Wayve's [PRISM-1](https://wayve.ai/thinking/prism-1/) (2024-06) does 4D reconstruction from video for its Ghost Gym resimulator but does not say whether it uses Gaussians or NeRF; Wayve itself writes that cyclists are hard to reconstruct and that new scenarios come from the generative model GAIA-1. As of 2026-10-15 we found no public description of a Waymo 3DGS pipeline.

There is also a line that skips per-scene optimisation: [pixelSplat](https://arxiv.org/abs/2312.12337) (2023-12, image pairs), [MVSplat](https://arxiv.org/abs/2403.14627) (2024-03), [DepthSplat](https://arxiv.org/abs/2410.13862) (2024-10; the authors report 12 views at 512×960 in 0.6 s), and [AnySplat](https://arxiv.org/abs/2505.23716) (2025-05; uncalibrated image collections in one forward pass, also predicting intrinsics and extrinsics). Instant NuRec is this line applied to driving logs, and Depth Anything 3 can output 3DGS directly.

## Off the logged path, reconstruction has no data

One top-down calculation (a teaching example, $x$ to the right, $z$ forward, in metres). The logged car is creeping forward at a red light, and its camera moves from $(0,0)$ to $(0,5)$. A van is parked ahead on the right, its rear at $z=15$, spanning $1.5\le x\le3.5$. In front of the van, at $(3,20)$, a pedestrian is about to cross. The line of sight from the camera to the pedestrian crosses $z=15$ at $x_c+(3-x_c)(15-z_c)/(20-z_c)$:

| Camera position $(x_c,z_c)$ | Sight line's $x$ at $z=15$ | Blocked by the van? |
| --- | --- | --- |
| (0, 0), start of the log | 2.25 | Blocked |
| (0, 5), end of the log | 2.0 | Blocked |
| (−3.5, 0), the planner now changes lanes 3.5 m to the left | 1.375 | Visible |

Throughout the log this value stays between 2.25 and 2.0, so the pedestrian never appears in any frame, and the reconstructed scene contains no Gaussians for them. The new viewpoint after the lane change should see the pedestrian, but the render shows empty road or a blob of unconstrained Gaussians. The 3DGS paper's own [limitations section](https://arxiv.org/html/2308.04079v1#S7.SS4) says as much: "In regions where the scene is not well observed we have artifacts …". Waymo's [blog post](https://waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation/) of 2026-02-06 makes exactly this point: when the simulated route differs too much from the original drive, purely reconstructive methods (such as 3DGS) break down visually because observations are missing.

<!-- paper-figure:3dgs-sparse-views -->

| Approach | Example | What it adds | Cost |
| --- | --- | --- | --- |
| Evaluate only near the log | NAVSIM v2 evaluates from 3DGS-rendered perturbed start states, weighted by proximity to the first stage's endpoint | Our reading: start states stay close to observed regions, so the renders are more trustworthy | Large deviations cannot be tested |
| Clean up and complete novel views with generative models | NuRec's Fixer (Cosmos-based) and Asset Harvester | Removes artefacts from novel views and fills in unobserved objects | What it fills in was never observed |
| Replace the whole thing with a generative world model | Waymo World Model (2026-02-06, post-trained from Genie 3, outputs camera and lidar); Wayve [GAIA-4](https://wayve.ai/thinking/gaia-4/) (2026-08-03) | Plausible-looking images from any position | Neither has published fidelity numbers you can check (GAIA-4 gives only a "2.5x" improvement with no stated baseline); may add or drop objects out of nowhere |

Waymo's blog says its model maintains realism and consistency through its generative ability, but provides no quantitative comparison; that is a company claim. The pedestrian in the example was never seen in this log, so no method can infer from the log alone that someone is standing there. A generative model can draw a plausible pedestrian or an empty road; the log cannot tell us which is correct.

Reconstruction aims to reproduce observed regions faithfully, while generation attempts to produce plausible views even where observations are missing. They address different questions. For world models as simulators and evaluators, and the difficulty of validating them, see [world models for robots and cars](../../11-frontier/world-models/robotics-driving.en.md). For why planners must be evaluated in closed loop, see [end-to-end driving and closed-loop evaluation](../driving/end-to-end.en.md).

## Before you build on it

| What to try | What you should observe | What you can NOT conclude |
| --- | --- | --- |
| Swap Gaussians 1 and 4 in the example | The colour goes from (0.635, 0.335, 0.285) to nearly white, while the leftover transmittance stays 0.015 | That a sorting error will always be visible (swapping Gaussians at similar depths may change little) |
| Compute PSNR on held-out frames on the logged trajectory | Reconstruction quality near the original route | That the view after a 3.5 m lane change is just as good |
| Shift the render camera sideways from the logged path step by step | Occluded regions start to show and artefacts grow | That what a generative model fills in is real |
| Run SfM and BA on images alone | Relative positions of cameras and points | Metric scale (that needs a calibrated baseline, GPS/IMU or lidar) |

## As of October 2026

- **Licences.** The original 3DGS code is under the Inria/MPII licence, non-commercial research and evaluation only; [gsplat](https://arxiv.org/abs/2409.06765) is Apache-2.0 (the authors report up to 10% less training time and 4× less memory than the original); 3dgrut, the 3DGUT code, is Apache-2.0; drivestudio, which hosts OmniRe, is MIT; Street Gaussians uses a custom Project Registration License v1.0 (2026). NuRec's Fixer is under the NVIDIA Open Model License; we have not checked the licences of Instant NuRec or the NuRec reference container.
- **Milestones.** NuRec general availability on 2026-03-25, the Instant NuRec paper on 2026-07-15, the Waymo World Model on 2026-02-06, GAIA-4 on 2026-08-03.
- **Not disclosed.** Waymo has not published its own 3DGS pipeline; Wayve has not said which representation PRISM-1 uses.

Go back to the [series guide](README.en.md), or read the previous note, [frozen backbones and geometry foundation models](foundation-models.en.md), to see where the cameras and point clouds come from.
