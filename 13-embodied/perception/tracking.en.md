# Perception & geometry: state estimation and tracking

[中文](tracking.md) · **English**

> Reading time: ~10 min · Draft, pending review

A detector produces a pile of boxes every frame, each frame on its own. A planner needs something else: "the car ahead is at 12.75 m, moves about 2 m per frame, and here is how uncertain that is", and "this pedestrian is the same one as in the last frame". **Tracking splits into two jobs: predict with a motion model, then correct with a measurement weighted by both uncertainties (the Kalman filter); and decide which measurement belongs to which track (data association). The first has a closed-form optimal answer under linear-Gaussian assumptions; many identity switches come from the second.**

> **Draft, pending review.** The site owner has not reviewed this note yet; facts, emphasis and wording may change. Check the primary sources linked in the text before relying on it.

## Write "what you believe" as a Gaussian

Prerequisites: conditional probability and the Gaussian distribution, and projection from [note 1](camera-geometry.en.md).

Tracking does not store a number; it stores a distribution. The mean is the most likely position and the variance says how unsure you are. Under linear models with Gaussian noise the distribution stays Gaussian, so only two quantities need updating. Take the one-dimensional case of eq. (3.1) in [Barfoot, *State Estimation for Robotics*, 2nd ed.](https://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser24.pdf) (the book writes the known input as $v_k$; here it is $u_k$):

$$
x_k=x_{k-1}+u_k+w_k,\quad w_k\sim\mathcal N(0,Q);\qquad
y_k=x_k+n_k,\quad n_k\sim\mathcal N(0,R).
$$

$x_k$ is the true position at frame $k$, $u_k$ is a known displacement (for example the motion model's velocity times the frame interval), and $w_k$ is what the motion model misses, with variance $Q$. $y_k$ is the position the detector reports, with noise variance $R$.

## One dimension by hand: predict once, correct once

The Kalman filter takes two steps per frame (the one-dimensional case of Barfoot eq. 3.131; $\check{}$ marks a prediction, $\hat{}$ a corrected estimate):

$$
\check x_k=\hat x_{k-1}+u_k,\qquad \check P_k=\hat P_{k-1}+Q
$$

$$
K_k=\frac{\check P_k}{\check P_k+R},\qquad
\hat x_k=\check x_k+K_k\,(y_k-\check x_k),\qquad
\hat P_k=(1-K_k)\,\check P_k
$$

Teaching setup: track the car ahead along the lane. Last frame's estimate is $\hat x=10$ m with $\hat P=1$ m². The car moves $u=2$ m per frame (20 m/s at 10 Hz), $Q=0.5$. This frame the detector reports $y=13$ m with $R=0.5$.

1. Predict: $\check x=12$, $\check P=1+0.5=1.5$. Prediction makes you less certain.
2. Gain: $K=1.5/(1.5+0.5)=0.75$. The prediction is less certain than the measurement, so the measurement gets more weight.
3. Correct: $\hat x=12+0.75\times(13-12)=12.75$, $\hat P=0.25\times1.5=0.375$.

$y_k-\check x_k$ is the innovation: "what you saw" minus "what you expected". Check it another way: the corrected estimate is the precision-weighted average (precision = inverse variance) of prediction and measurement,

$$
\hat x=\frac{12/1.5+13/0.5}{1/1.5+1/0.5}=12.75,\qquad
\hat P=\Big(\frac{1}{1.5}+\frac{1}{0.5}\Big)^{-1}=0.375.
$$

The corrected variance is smaller than either the prediction's or the measurement's: two independent sources together are more certain than either one. Now change one condition at a time:

| Change one condition | Result | What it shows | What you can NOT conclude |
| --- | --- | --- | --- |
| $R$ from 0.5 to 4.5 (say, a poor box after occlusion) | $K=0.25$, $\hat x=12.25$, $\hat P=1.125$ | The less you trust the measurement, the more you lean on the prediction | The filter knows how good the detector is ($R$ is your assumption; set it wrong and the filter stays just as confident) |
| No detection this frame | Predict only: 12, 1.5; another missing frame: 14, 2.0 | Uncertainty grows by $Q$ every frame | The track is still reliable (after a few frames it is only extrapolation) |
| Set $Q$ to 0 (trust the motion model completely) | $K$ over 4 frames: 0.67, 0.40, 0.29, 0.22 | $P$ keeps shrinking and the filter listens less and less | Smaller $Q$ is more accurate (when the car ahead brakes, the estimate lags) |
| Measurement $y=18$ (maybe a different car) | Without a gate, the estimate is pulled to 16.5 | Innovation 6, $6^2/S=36/2=18$ | A gate prevents wrong matches (two targets close together can both pass it) |

In the last row, $S=\check P+R=2$ is the innovation's variance. $e^2/S$ is the normalised innovation squared (NIS); when the model is right it follows a $\chi^2$ distribution with 1 degree of freedom, whose 95% quantile is about 3.84. Barfoot, Section 5.1.2, uses NIS to test whether a filter is consistent; using it to screen measurements one at a time is gating: 18 is far above 3.84, so this measurement is rejected.

<!-- widget:step-kalman-1d -->

## From one dimension to SORT: a box has a state too

[SORT (2016)](https://arxiv.org/abs/1602.00763) writes each target's state as $[u,v,s,r,\dot u,\dot v,\dot s]$: box centre, area, aspect ratio, and the rates of change of the first three; the aspect ratio is treated as constant. The motion model is linear constant velocity and ignores other targets and camera motion. The equations are exactly the one-dimensional ones with numbers replaced by matrices (Barfoot eq. 3.131):

$$
\check P_k=A\hat P_{k-1}A^{\top}+Q,\qquad
K_k=\check P_kC^{\top}\big(C\check P_kC^{\top}+R\big)^{-1},\qquad
\hat P_k=(I-K_kC)\,\check P_k
$$

$A$ encodes "new position = old position + velocity × one frame", and $C$ picks out the observable part, $[u,v,s,r]$. A new track's velocity has never been observed, so SORT initialises it at 0 with a large variance. The paper also says plainly that the constant-velocity model predicts real motion poorly; in its experiments a target missing for one frame is deleted ($T_{\rm Lost}=1$) and gets a new ID if it reappears.

When you track in 3D but measure in pixels, the measurement model is note 1's projection, which is nonlinear. The extended Kalman filter (EKF) linearises once at the current estimate; Barfoot, Section 4.3.4, points out that its main weakness is exactly this single linearisation without iterating to convergence.

## Data association: 3 tracks, 3 detections

Teaching setup: two pedestrians, T1 and T2, walking right side by side, and a third, T3, farther away. To keep it hand-computable, every box is 100 px tall with aligned top and bottom edges and 100 px wide, so IoU depends only on the horizontal overlap: with an overlap of $o$ px, $\mathrm{IoU}=o/(200-o)$.

- Kalman-predicted boxes (x ranges): T1 [100, 200], T2 [30, 130], T3 [400, 500].
- This frame's detections: D1 [80, 180], D2 [140, 240], D3 [410, 510].

For example, T1 and D1 overlap on [100, 180], so $o=80$ and IoU $=80/120=0.667$. All of them:

| Track \ detection | D1 | D2 | D3 |
| --- | --- | --- | --- |
| T1 | 0.667 | 0.429 | 0 |
| T2 | 0.333 | 0 | 0 |
| T3 | 0 | 0 | 0.818 |

**Greedy** matching takes the largest IoU each time: first T3–D3 (0.818), then T1–D1 (0.667). T2 is left with D2 at IoU 0 and stays unmatched; D2 is treated as a new object. If both people actually walked faster than predicted, D1 is T2 and D2 is T1: greedy gave T2's detection to T1, so T2 breaks and T1's box jumps onto someone else.

The **Hungarian algorithm** ([Kuhn 1955](https://doi.org/10.1002/nav.3800020109)) finds the global optimum: a one-to-one matching with the smallest total cost $1-\mathrm{IoU}$. A 3×3 case can be done by hand: subtract each row's minimum from that row, then each column's minimum from that column:

```text
cost 1 − IoU              subtract row minima        subtract column minima (only D2 needs 0.238)
     D1    D2    D3          D1    D2    D3          D1    D2    D3
T1  0.333 0.571 1.000       0     0.238 0.667       0     0     0.667
T2  0.667 1.000 1.000       0     0.333 0.333       0     0.095 0.333
T3  1.000 1.000 0.182       0.818 0.818 0           0.818 0.580 0
```

Now pick one zero per row and column. T2 has only one zero, at D1, so T2–D1, T1–D2, T3–D3, with total cost 1.420, lower than the greedy set's 1.515. I recomputed it with SciPy's `linear_sum_assignment` and got the same pairs. For 3×3 you can also enumerate all 6 permutations to check; the point of the Hungarian algorithm is that with many targets you do not enumerate $n!$ of them.

After matching, apply a gate: pairs with too little IoU are rejected outright. SORT uses a tuned $\mathrm{IoU}_{\min}$; ByteTrack's implementation details reject matches with IoU below 0.2. T2–D1 at 0.333 passes.

The Hungarian algorithm only maximises total IoU; it does not guarantee the right answer. If T2 were actually occluded and D1 really were T1, it would get this wrong. Motion direction and appearance features (Re-ID) help some; [DanceTrack (2021)](https://arxiv.org/abs/2111.14690) deliberately collects group-dance videos with similar clothing and complex motion, so appearance barely separates people and methods have to rely on motion.

## ByteTrack: do not throw low-score boxes away yet

Continue the example. In the next frame T2 walks behind a pole. The detector puts D4 at T2's predicted position, but with a score of only 0.35; it also puts D5, score 0.25, on a poster. The common practice keeps only boxes above a score threshold ([ByteTrack](https://arxiv.org/abs/2110.06864) uses 0.5 in its illustration), so D4 is dropped and T2 goes unmatched; under SORT's settings it is deleted, and the person gets a new ID on reappearing.

<!-- paper-figure:bytetrack-byte -->

ByteTrack's association method, BYTE, keeps almost every box and runs two rounds split by a score threshold $\tau$ (Algorithm 1 in the paper):

```text
every frame:
  detect → split boxes by score threshold τ into D_high and D_low
  predict every track (including temporarily lost ones) with the Kalman filter
  round 1: tracks ↔ D_high, similarity from IoU or Re-ID, Hungarian matching
  round 2: tracks left from round 1 ↔ D_low, IoU only
  unmatched D_low: delete as background
  unmatched tracks: mark as lost, keep for 30 frames, can still be recovered
  unmatched D_high: start new tracks
```

The paper's default is $\tau=0.6$, and lost tracks are kept for 30 frames. Round 2 uses IoU only because low-score boxes often come from occlusion or motion blur, where appearance features are unreliable. In our example, T2's predicted box overlaps D4 heavily, so round 2 reconnects them; D5 has no track nearby and is deleted as background.

The authors compare on MOT17 validation (the second half of each training video) with the same detector: replacing SORT's association with BYTE moves MOTA from 74.6 to 76.6 and IDF1 from 76.9 to 79.3, and cuts identity switches from 291 to 159 (Table 2 of the paper, author-reported). In the paper's words, MOTA is computed from false positives, misses and identity switches and leans towards detection quality; IDF1 measures identity preservation; HOTA balances detection, association and localisation.

| Situation | What BYTE does | Why |
| --- | --- | --- |
| An occluded target's score drops | Round 2 reconnects it to its track | The low-score box overlaps the predicted box |
| A low-score false positive far from every track | Deleted | Round 2 finds no track for it |
| A low-score false positive right next to a track | May be attached to that track by mistake | Round 2 looks only at IoU |
| Large camera motion and low frame rate (BDD100K) | Re-ID beats IoU by a wide margin in round 1 (Table 1 of the paper) | Motion cues fail and predicted boxes are off |

Common pedestrian-tracking benchmarks: [MOT17](https://motchallenge.net/data/MOT17/) reuses all MOT16 sequences with more accurate annotations, has 7 training and 7 test sequences, and ships three public detectors (DPM, Faster R-CNN, SDP); [MOT20](https://arxiv.org/abs/2003.09003) has 8 very crowded sequences; DanceTrack is described above.

## Filter or optimise: the same problem in SLAM

Replace the state with the ego vehicle's pose (and a map) and you have visual odometry and SLAM. It is still "predict, then correct with measurements", but done in two ways.

**Filters** keep only the current state and its covariance; each measurement is used once and then folded into the covariance. [MSCKF](https://doi.org/10.1109/ROBOT.2007.364024) (Mourikis & Roumeliotis 2007) is EKF-based vision-aided inertial navigation: it keeps several past camera poses in the state, leaves 3D feature points out, builds constraints from "one feature seen from several poses", and its cost grows linearly with the number of features (paper abstract).

**Optimisation** keeps all states and raw measurements inside a window and repeatedly solves a nonlinear least-squares problem. [VINS-Mono (2017)](https://arxiv.org/abs/1708.03852) fuses pre-integrated IMU measurements and feature observations with tightly coupled nonlinear optimisation over a sliding window, marginalises old states into a prior to keep the cost constant, and uses loop closure plus a 4-degree-of-freedom pose graph for global consistency. [ORB-SLAM3 (2020)](https://arxiv.org/abs/2007.11898) is feature-based, uses maximum-a-posteriori estimation even during IMU initialisation, and can bring keyframes from long ago, or from a previous mapping session, into bundle adjustment; the authors report 3.5 cm average accuracy on EuRoC with stereo-inertial input. [DSO (2016)](https://arxiv.org/abs/1607.02565) takes another route: no keypoints, it minimises photometric error directly.

In the linear-Gaussian case both give the same answer for the newest state: Barfoot, Section 3.3, derives the Kalman filter from the batch maximum-a-posteriori solution, and it is exactly the forward pass of the Rauch–Tung–Striebel smoother. The differences come from nonlinearity:

| | Filtering (EKF, MSCKF) | Sliding-window or batch optimisation (VINS-Mono, ORB-SLAM3) |
| --- | --- | --- |
| What stays in the state | Current state and covariance (MSCKF also keeps a few poses) | All states and raw measurements in the window |
| Linearisation | Each measurement is linearised once, at the estimate of that moment | Re-linearised at every iteration |
| Old information | Folded into the covariance and cannot be revised | Marginalised into a prior; loop closure can correct it via a pose graph or BA |
| What to watch | Linearisation errors stay in the estimate (Barfoot, Section 4.3.4) | Heavy computation, hard to run in real time on constrained platforms (VINS-Mono, related work) |

The line is not absolute. VINS-Mono's related-work section mentions an MSCKF extension that uses an inverse filter with iterative re-linearisation, making it equivalent in effect to optimisation-based methods.

## As of October 2026

- **Video segmentation models now track too.** [SAM 3](https://arxiv.org/abs/2511.16719) (2025-11) detects and tracks every instance of a noun phrase in video and assigns IDs. SAM 3.1 (2026-03-27) adds joint multi-object tracking; Meta reports going from 16 to 32 FPS on one H100 with a medium number of objects (vendor-reported, [Meta blog](https://ai.meta.com/blog/segment-anything-model-3/)).
- **Point tracking is its own line.** [TAPIR (2023)](https://arxiv.org/abs/2306.08637) is Apache-2.0; [CoTracker3 (2024-10)](https://arxiv.org/abs/2410.11831) is CC BY-NC 4.0 ([model card](https://huggingface.co/facebook/cotracker3), licence read on 2026-10-15) and cannot be used commercially as is. [VGGT](https://arxiv.org/abs/2503.11651) also outputs 3D point tracks.
- This note does not go into how MOTA, IDF1 and HOTA are computed, and does not cover point-tracking models after CoTracker3.

Next: [detection and segmentation in 2026](detection.en.md). In autonomous driving these tracks feed forecasting and planning; see the [autonomous-driving series](../driving/README.en.md).
