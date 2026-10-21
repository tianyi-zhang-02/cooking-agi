# Perception & geometry: how a 3D point becomes a pixel, and back

[中文](camera-geometry.md) · **English**

> Reading time: ~10 min · Last reviewed: 2026-10

In the front camera, a pedestrian's foot lands on pixel (860, 690). How far away is the pedestrian? That one pixel cannot tell you: projection throws depth away, and a pixel corresponds only to a ray. **To get depth back you must add one more constraint: the point lies on a known plane, a second view sees it, or a learned depth prior fills it in.** This note follows a single point through forward projection, back-projection, distortion, two views and triangulation. Linear algebra is all you need.

## Which frames does a point pass through?

A point passes through three coordinate frames:

- **World frame** (here, the vehicle frame): x forward, y left, z up, in metres.
- **Camera frame**: origin at the optical centre, x right, y down, z forward along the optical axis. This is the usual convention in vision; with another convention the signs below change.
- **Pixel frame**: u to the right, v down, origin at the top-left corner of the image.

The extrinsics $[R\mid t]$ take world coordinates to camera coordinates, and the intrinsics $K$ project camera coordinates onto pixels:

$$
\lambda\begin{bmatrix}u\\v\\1\end{bmatrix}=K\,[R\mid t]\begin{bmatrix}X\\Y\\Z\\1\end{bmatrix},\qquad
K=\begin{bmatrix}f_x&s&c_x\\0&f_y&c_y\\0&0&1\end{bmatrix}.
$$

$f_x,f_y$ are focal lengths in pixels, $(c_x,c_y)$ is the principal point, and $s$ is the skew term for pixel axes that are not perpendicular, usually set to 0. $\lambda$ is the point's depth $Z_c$ in the camera frame: multiplying the homogeneous pixel vector by this scale makes it equal to the right-hand side. Dividing out that scale during projection discards depth. [Zhang 2000](https://doi.org/10.1109/34.888718), Section 2, uses this pinhole model (he writes $K$ as $A$; the [technical-report version](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf) is free to read). The [Hartley & Zisserman sample chapter 9](https://www.robots.ox.ac.uk/~vgg/hzbook/) writes the camera matrix as $P=K[R\mid t]$.

$R$ rotates the world axes onto the camera axes; $t=-RC$, where $C$ is the optical centre's position in the world frame. Reading $t$ directly as "where the camera is mounted" is an easy mistake.

## Work it once by hand: from vehicle frame to pixel

Mount the camera at $C=(2,0,1.5)$ in the vehicle frame: 2 m ahead of the rear axle, 1.5 m above the ground, optical axis pointing straight ahead. The image is 1920×1080, $f_x=f_y=1000$, $(c_x,c_y)=(960,540)$, $s=0$. These are teaching values, not any real car.

The vehicle's "forward, left, up" must become the camera's "right, down, forward": camera x = −vehicle y, camera y = −vehicle z, camera z = vehicle x. So

$$
R=\begin{bmatrix}0&-1&0\\0&0&-1\\1&0&0\end{bmatrix},\qquad
t=-RC=\begin{bmatrix}0\\1.5\\-2\end{bmatrix}.
$$

The pedestrian's foot is at $X_w=(12,1,0)$ in the vehicle frame: 12 m ahead, 1 m to the left, on the ground.

1. Extrinsics: $X_c=RX_w+t=(-1,0,12)+(0,1.5,-2)=(-1,\,1.5,\,10)$, i.e. 1 m left of the camera, 1.5 m below it, 10 m ahead. That matches intuition.
2. Divide by depth: $(X_c/Z_c,\;Y_c/Z_c)=(-0.1,\;0.15)$. These are normalised coordinates.
3. Intrinsics: $u=1000\times(-0.1)+960=860$, $v=1000\times0.15+540=690$.

The pixel is (860, 690). Now change one condition at a time:

| Change one condition | Pixel becomes | What it shows | What you can NOT conclude |
| --- | --- | --- | --- |
| Pedestrian twice as far: $X_w=(22,1,0)$ | (910, 615) | The offset from the image centre shrinks as $1/Z_c$ | Box size alone gives distance (you also need the object's true size) |
| Double the focal length: $f=2000$ | (760, 840) | The offset scales with focal length | A telephoto lens keeps depth in a single image |
| Same spot, a point 0.5 m above the ground | (860, 640) | A higher point has a smaller v | A pixel higher in the image is farther away (true only if every point is on the ground) |
| Camera pitch changes by 1° (say, over a speed bump) | v about 672 or 708 | A small extrinsic error moves the pixel by about 18 px | Calibrating once at the factory is enough |

## Backwards: a pixel gives only a ray

Back-projection first removes the intrinsics with $K^{-1}$:

$$
\begin{bmatrix}x_n\\y_n\\1\end{bmatrix}=K^{-1}\begin{bmatrix}u\\v\\1\end{bmatrix}
=\begin{bmatrix}(860-960)/1000\\(690-540)/1000\\1\end{bmatrix}
=\begin{bmatrix}-0.1\\0.15\\1\end{bmatrix},\qquad
X_c=Z_c\begin{bmatrix}x_n\\y_n\\1\end{bmatrix}.
$$

Any positive $Z_c$ projects to the same pixel. There are three ways to pin it down:

- **A known plane.** Assume the point is on the ground. The camera is 1.5 m up and its y axis points down, so the ground is $Y_c=1.5$; then $0.15\,Z_c=1.5$ and $Z_c=10$ m, which is correct. But the point 0.5 m above the ground sits at v=640, and the same assumption gives $Z_c=1.5/0.1=15$ m, 50% too far. The plane assumption holds only for points that really are on the plane.
- **A second view.** The point is where the two rays meet; see triangulation below.
- **A learned prior.** A monocular depth network learns from many images "how far things usually are". Relative depth cannot be read directly in metres: scale-only outputs preserve depth ratios, while other outputs also have an unknown shift or are defined in inverse depth. Metric depth gives metres directly. These distinctions and models are covered in [note 4](foundation-models.en.md).

The 1° pitch change from the table turns 10 m into about 8.9 m or 11.3 m under the ground assumption (depending on the direction). A system that measures distance from the ground plane must either estimate camera pitch online or use a cue that does not depend on the ground.

## Lens distortion: worse towards the edges

Real lenses are not ideal pinholes. Zhang 2000, Section 3.3, keeps only the first two radial-distortion terms, applied to normalised coordinates:

$$
\breve x=x\,(1+k_1r^2+k_2r^4),\qquad \breve y=y\,(1+k_1r^2+k_2r^4),\qquad r^2=x^2+y^2.
$$

Take $k_1=-0.2$, $k_2=0$ (barrel distortion, a teaching value). Our point has $r^2=0.0325$, a factor of 0.9935, so the pixel moves by only about 0.65 px and 1 px. A point near the right edge, $(0.9,\,0.5)$, has $r^2=1.06$ and a factor of 0.788: u moves from 1860 to about 1669, roughly 190 px.

So undistort first, then apply the pinhole equations. Wide-angle and fisheye lenses need their own camera models; [ORB-SLAM3](https://arxiv.org/abs/2007.11898), for example, supports both pinhole and fisheye models. The original 3D Gaussian splatting rasteriser assumes an ideal pinhole; NVIDIA's [3DGUT](https://arxiv.org/abs/2412.12507) adds distorted and rolling-shutter cameras (the paper's claims here are qualitative).

Calibration means estimating $K$, the distortion coefficients and the extrinsics. Zhang's method only needs the camera to see a planar pattern from at least two different orientations; it solves in closed form, then refines by maximum-likelihood nonlinear optimisation. Multiple sensors also need their time offsets calibrated: at 20 m/s, a 10 ms offset is a 0.2 m position error. [Furgale et al. 2013](https://doi.org/10.1109/IROS.2013.6696514) calibrate time and space together.

## Two views: the epipolar constraint turns a 2D search into a line

Add a second camera with the same $K$, shifted 0.5 m to the right and not rotated: a stereo pair. In the first camera's frame, the second camera has $R=I$ and $t=(-0.5,0,0)$. The same point is $(-1.5,\,1.5,\,10)$ in its frame, at pixel (810, 690).

Without knowing depth, where can the first image's point land in the second image? Its ray projects to a line in the second image, the epipolar line. With the essential matrix:

$$
E=[t]_\times R,\qquad
[t]_\times=\begin{bmatrix}0&-t_z&t_y\\t_z&0&-t_x\\-t_y&t_x&0\end{bmatrix},\qquad
\hat x'^{\top}E\,\hat x=0,
$$

where $\hat x,\hat x'$ are normalised coordinates in the two images ([HZ chapter 9](https://www.robots.ox.ac.uk/~vgg/hzbook/), eq. 9.11). With our numbers, $[t]_\times$ has only 0.5 in row 2, column 3 and −0.5 in row 3, column 2, so $E\hat x=(0,\;0.5,\;-0.075)$. Its dot product with $\hat x'=(-0.15,\,0.15,\,1)$ is $0.15\times0.5-0.075=0$: the constraint holds.

$E\hat x$ is itself the epipolar line in the second image: $0.5\,y'-0.075=0$, i.e. $y'=0.15$, pixel row $v'=690$. You only need to search along row 690, so a 2D search becomes 1D.

This is the simplest case. With rotation, epipolar lines are tilted. Under pure forward translation the epipole $e'=K't$ sits exactly at the principal point (HZ eq. 9.3), and all epipolar lines radiate from the image centre; a monocular camera on a car driving straight is close to this. Without $K$, write the fundamental matrix in pixel coordinates, $F=K'^{-\top}EK^{-1}$, and the constraint becomes $x'^{\top}Fx=0$. HZ chapter 9 gives $F$ 7 degrees of freedom and $E$ 5, and notes that [Longuet-Higgins 1981](https://doi.org/10.1038/293133a0) introduced the essential matrix to computer vision (a similar matrix appeared earlier in the photogrammetry literature); $F$ generalises it by dropping the assumption that the cameras are calibrated.

In practice $E$ or $F$ is estimated from matched points. Each match gives one linear equation in the 9 entries of $F$; with 8 or more matches you can solve linearly (the eight-point algorithm) and then enforce rank 2. [Hartley 1997](https://doi.org/10.1109/34.601246) states in its abstract that a simple normalisation of the matched coordinates (translation and scaling) beforehand makes the eight-point results comparable to the best iterative algorithms of the time. Wrong matches are removed with robust methods such as RANSAC ([Barfoot, 2nd ed.](https://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser24.pdf), Section 5.4.1).

## Triangulation: getting depth back

The 3D point is where the two rays meet. A stereo pair with identical intrinsics and only a horizontal shift is the easiest case: depth is the same in both cameras, so the homogeneous relation in HZ eq. 9.6 becomes the ordinary equality $x'=x+Kt/Z$. Substituting $t=(-b,0,0)$ gives

$$
d=u-u'=\frac{f\,b}{Z}\quad\Longrightarrow\quad Z=\frac{f\,b}{d}.
$$

Our point has disparity $d=860-810=50$ px, so $Z=1000\times0.5/50=10$ m, matching where we started.

Now the error. Differentiating $Z=fb/d$ with respect to $d$ gives $|\partial Z/\partial d|=Z^2/(fb)$. For a small disparity error, $|\Delta d|\ll d$, the first-order approximation is $|\Delta Z|\approx Z^2|\Delta d|/(fb)$. At 10 m, a 1 px disparity error gives approximately 0.2 m of depth error; at 40 m the disparity is only 12.5 px, and the same 1 px error gives approximately 3.2 m. Both are local approximations: the exact finite change is $\Delta Z=-fb\,\Delta d/[d(d+\Delta d)]$, so positive and negative 1 px errors do not have symmetric effects. **For fixed baseline, focal length and small disparity error, the first-order depth error grows with the square of distance.** This explains why stereo and multi-view ranging get worse far away. A longer baseline $b$ or focal length $f$ helps, at a price: the two cameras share less field of view up close, and the two images look more different, so matching gets harder.

<!-- widget:step-pixel-ray -->

The general case (rotation, more than two views) has no formula this tidy. With noise the two rays do not meet. The usual approach is a linear method for a first estimate, then minimising reprojection error: the 3D point should project back as close as possible to the observed pixel in every image. Putting the camera poses into the same least-squares problem is bundle adjustment (Barfoot, 2nd ed., Section 10.1), where [note 5](reconstruction.en.md) picks up.

## When one homography is enough

HZ eq. 9.7 expresses the second image's homogeneous pixel coordinates as two terms. The scale convention matters: if $x=(u,v,1)^\top$ and $x'=(u',v',1)^\top$ both have third component 1, and the point has depths $Z,Z'$ in the two cameras, then

$$
\frac{Z'}{Z}x'=K'RK^{-1}x+\frac{K't}{Z},
\qquad x'\sim K'RK^{-1}x+\frac{K't}{Z}.
$$

Here $\sim$ means equality up to a nonzero scale. To recover $x'$ with third component 1, divide the right-hand side by its third component. Its first term depends only on rotation and intrinsics; the second changes with depth $Z$. Which term vanishes, or becomes predictable, decides which relation you need:

| Case | Relation between the two images | Degrees of freedom | Examples |
| --- | --- | --- | --- |
| All points lie on one plane | Point to point: $x'\sim Hx$, $H$ is 3×3 | 8 | The ground, a calibration board, a document; Zhang's calibration uses $H=K[r_1\;r_2\;t]$ |
| The camera only rotates, $t=0$ | Point to point: $x'\sim K'RK^{-1}x$, independent of depth | Set by $K,K',R$ | Panoramas taken by turning in place |
| Translation and a scene with relief | Point to line: $x'^{\top}Fx=0$ | 7 for $F$, 5 for $E$ | Stereo, a monocular camera on a moving car, a camera on a robot arm |

The ground-plane ranging above uses the first row: points on the ground map one-to-one to pixels, and the mapping breaks as soon as a point leaves the ground.

## As of October 2026: the geometry has not aged, only who computes it

In 2025–2026, feed-forward 3D models output camera intrinsics and extrinsics, depth and pointmaps in a single forward pass. [VGGT](https://arxiv.org/abs/2503.11651) won the CVPR 2025 Best Paper, and its authors report reconstruction in under one second; the CVPR 2026 Best Paper went to [D4RT](https://arxiv.org/abs/2512.08924), which reconstructs dynamic scenes in 4D ([CVPR 2026 closing page](https://cvpr.thecvf.com/Conferences/2026/News/Closing)). What these models output is still this note's $K$, $[R\mid t]$ and depth; reading, checking and using those outputs still takes this note's geometry. Details are in [note 4](foundation-models.en.md).

This layer is also something public job descriptions spell out: of the 13 public job descriptions for cars and robots we read on 2026-10-15, about 5 name 3D geometric vision, SLAM/VIO, sensor fusion or BEV, and some ask for the pinhole camera model and epipolar geometry by name (an aggregate of knowledge areas only; no company is named).

Further reading: the [HZ sample chapters](https://www.robots.ox.ac.uk/~vgg/hzbook/) (chapter 9 on epipolar geometry is free), [Szeliski, 2nd ed.](https://szeliski.org/Book/) (free download for personal use), and the course pages of CMU [16-385](http://16385.courses.cs.cmu.edu/) and [16-822](https://geometric3d.github.io/). Next: [state estimation and tracking](tracking.en.md).
