# World models: predict representations, not pixels (JEPA)

[中文](jepa.md) · **English**

> Reading time: ~12 min · Last reviewed: 2026-10

If the robot arm pushes 5 cm, will the cup fall? Ask a pixel-level video model and it must also draw the shadow's texture, the reflections on the table, even where the shards land. Most of that detail cannot be predicted, and none of it decides whether the cup falls. A JEPA (joint-embedding predictive architecture) works differently: it first encodes the future into a vector and asks only that the vector be predicted.

This can leave out unpredictable detail, but the encoder must be prevented from mapping everything to the same vector. Without a decoder, you also cannot see the frames it predicts.

## Where pixel prediction spends its effort

Imagine a frame with only 100 "pixels" (a teaching setup): 4 describe the cup's position and pose; 96 are shadow and table texture that flicker randomly with the lighting, each +1 or −1 with equal probability, independent of the push. Train a deterministic predictor with mean squared error:

- For the 96 random pixels, the best prediction is the mean 0, with expected error 1 each, 96 in total, and no amount of training lowers it;
- For the 4 cup pixels, a right guess costs 0 and a wrong one (drawing "stayed" when it "fell") costs 1 each, 4 in total.

So the total loss is 96 with the cup right and 100 with it wrong: the thing that actually decides the outcome is 4% of the loss, and the loss curve alone barely shows whether it was learned. For the 96 pixels the model outputs a grey average, the same problem as "averaging two paths into a wall" in the [model-based note](../../05-post-training/deep-rl/model-based.en.md); a generative model such as a diffusion model can draw sharp textures, but must then spend capacity learning their distribution.

Now suppose an encoder compresses the frame into two numbers: (how far the cup moved along the table, whether it tipped). With the error computed only on those, a right and a wrong cup differ by 0 versus 2, and the whole loss is about the cup. This is exactly [LeCun (2022)](https://openreview.net/forum?id=BZ5a1r-kVsf)'s argument against generative architectures (§4.5–4.6): pixel-level futures such as leaves and textures are inherently unpredictable, and a JEPA can drop them.

Two caveats. First, "the encoder keeps only cup information" is an assumption here; a real encoder is learned, and nothing guarantees it keeps "whether it tipped". Second, if the frames themselves are the product, as in a game or a simulation meant for people, pixels are required; see [video you can act in](interactive-video.en.md).

## The objective: measure error in embedding space

LeCun 2022 writes it as an energy (§4.4, Eqs. 10–12):

$$
E_w(x,y,z)=D\big(s_y,\ \mathrm{Pred}(s_x,z)\big),\qquad F_w(x,y)=\min_z E_w(x,y,z)
$$

$x$ is the visible part and $y$ the part to predict; two encoders give $s_x$ and $s_y$; the predictor $\mathrm{Pred}$ guesses $s_y$ from $s_x$; $z$ is an optional latent variable for what $x$ alone cannot settle, such as which way the cup tips; $D$ is a distance. Low energy means $x$ and $y$ are compatible.

The loss of [I-JEPA](https://arxiv.org/abs/2301.08243) (2023) on images (§3):

$$
\mathcal L=\frac1M\sum_{i=1}^{M}\sum_{j\in B_i}\big\|\hat s_{y_j}-s_{y_j}\big\|_2^2
$$

[V-JEPA 2](https://arxiv.org/abs/2506.09985) (2025) uses L1 on video, computed only on masked patches (§2.1, Eq. 1):

$$
\min_{\theta,\phi,\Delta_y}\ \big\|P_\phi(\Delta_y,E_\theta(x))-\operatorname{sg}(E_{\bar\theta}(y))\big\|_1
$$

The components in these three formulations correspond as follows:

| Role | LeCun 2022 | I-JEPA | V-JEPA 2 | In a video of the cup push |
| --- | --- | --- | --- | --- |
| Visible part | $x\to s_x$ | One large context block, into the context encoder $f_\theta$ (a ViT) | Unmasked patches, into $E_\theta$ | The parts of the frames not masked |
| Part to predict | $y\to s_y$ | Target encoder $f_{\bar\theta}$ sees the whole image; $M=4$ target blocks $B_i$ are taken | Masked patches, into $E_{\bar\theta}$, with stop-gradient | The masked space-time blocks |
| Predictor | $\mathrm{Pred}(s_x,z)$ | A narrow ViT $g_\phi$, with positional mask tokens saying which block to guess | $P_\phi$; $\Delta_y$ are learnable mask tokens | Guesses the embeddings where things are masked |
| Distance | $D$ | Squared L2 | L1 | — |

Nowhere in training does any term ask for pixels to be reconstructed.

## Collapse: a constant encoder also reaches zero loss

Two clips: in A the arm pushes 2 cm and the cup stays; in B it pushes 5 cm and the cup falls. The encoder gives each clip's "future" one number.

- A good encoder: $s_y(A)=0$, $s_y(B)=2$; the predictor also guesses 0 and 2, loss $|0-0|+|2-2|=0$.
- A collapsed encoder: outputs 0.7 whatever the input; the predictor always guesses 0.7, loss $|0.7-0.7|+|0.7-0.7|=0$.

The losses are equal, yet the second says nothing about whether the cup falls. If the target encoder is trained by the same loss, prediction loss alone gives no reason to prefer the first; pulling all outputs together is in fact a shortcut to a lower loss. I-JEPA defines collapse as the encoder outputting a constant regardless of input (§2); LeCun 2022 says that without extra constraints the energy surface goes flat (§4.5). Three kinds of fix follow.

### Fix one: require different inputs to differ

LeCun 2022 takes VICReg as an example (§4.5.1): a variance term penalises each embedding dimension's within-batch standard deviation when it falls below a threshold, and a covariance term encourages decorrelation between dimensions.

For this hand calculation, use the batch standard deviation with divisor $n$, omit the stabilising term $\epsilon$, and give the one-dimensional variance penalty unit weight. Set the threshold to 1: the collapsed encoder's $\{0.7,0.7\}$ has standard deviation 0 and penalty $\max(0,1-0)=1$; the good encoder's $\{0,2\}$ has standard deviation 1 and penalty 0. Adding the prediction loss, collapse scores 1 and the good encoder 0, so collapse is no longer optimal in this example.

The [VICReg paper](https://arxiv.org/html/2105.04906v3) also uses a threshold of 1. Its [official implementation](https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py) uses sample variance with divisor $n-1$ and adds $\epsilon=10^{-4}$ before taking the square root. The scores 1 and 0 above belong to the simplified example, not the full training loss with both branches and all term weights.

### Fix two: an EMA target encoder plus stop-gradient

I-JEPA and V-JEPA 2 use this one. The target encoder's parameters $\bar\theta$ receive no gradient; they are an exponential moving average of the context encoder's parameters, $\bar\theta\leftarrow m\bar\theta+(1-m)\theta$, with $m$ rising from 0.996 to 1.0 in I-JEPA (Appendix A). One step: $\bar\theta=1$, $\theta$ suddenly becomes 0, and after the update $\bar\theta=0.996$; even if $\theta$ stays at 0, $\bar\theta$ is still about 0.67 after 100 steps. The target moves slowly.

I-JEPA calls the EMA target encoder essential (§3), and V-JEPA 2 also says stop-gradient plus EMA prevents collapse. But no term in the loss explicitly forbids collapse; this is a design that works empirically, and LeJEPA's [title](https://arxiv.org/abs/2511.08544) calls such techniques heuristics: "Without the Heuristics".

### Fix three: require embeddings to look standard Gaussian

[LeJEPA](https://arxiv.org/abs/2511.08544) (Balestriero, LeCun, 2025-11) argues theoretically that downstream prediction risk is lowest when embeddings follow an isotropic Gaussian. Its SIGReg projects embeddings onto many random one-dimensional directions and uses the Epps–Pulley statistic to test whether each projection looks like $N(0,1)$. A collapsed encoder projects to a single point in every direction, with variance 0, so it cannot look standard Gaussian and is penalised. The abstract reports no need for stop-gradient or teacher-student, a single trade-off hyperparameter, and 79% ImageNet-1K linear probe accuracy with a ViT-H/14.

| Approach | Source | How it prevents collapse | What needs tuning |
| --- | --- | --- | --- |
| Variance and covariance terms | LeCun 2022 (VICReg) | Penalise per-dimension standard deviation below a threshold; encourage decorrelation | Threshold and term weights |
| EMA target plus stop-gradient | I-JEPA, V-JEPA 2 | Target encoder follows slowly and receives no gradient | Momentum schedule |
| SIGReg | LeJEPA | Random 1-D projections must look like $N(0,1)$ | One trade-off hyperparameter |
| Prediction plus Gaussian regulariser | [LeWorldModel](https://arxiv.org/abs/2603.19312) | Two loss terms, trained end to end from pixels | Loss hyperparameters cut from 6 to 1 (versus the only earlier end-to-end alternative) |
| Contrastive learning | [VL-JEPA](https://arxiv.org/abs/2512.10942) | Bi-directional InfoNCE | — |

## V-JEPA 2-AC: give the predictor actions

The JEPAs above only learn representations and take no action, so they cannot yet answer "what happens if I push 5 cm". Meta FAIR's V-JEPA 2 (2025-06) first pretrains the encoder on more than one million hours of video, then freezes it and trains a block-causal transformer predictor of about 300M parameters; that is V-JEPA 2-AC (§3.1):

- **Inputs**: interleaved actions $a_k$, end-effector states $s_k$ and frame features $z_k$, predicting the next frame's features $z_{k+1}$; $z_k$ is a 16×16×1408 ViT-g feature map.
- **Where actions come from**: the dataset records a 7-dimensional end-effector state per frame (position, orientation, gripper), and $a_k$ is the state difference between adjacent frames.
- **Data**: under 62 hours of DROID robot video, about 23k trajectories including both successes and failures. "Unlabeled" in the paper means no reward, task-type or success annotations are used.

The loss has two terms, both L1 in embedding space:

$$
\mathcal L_{\rm tf}(\phi)=\frac1T\sum_{k=1}^{T}\big\|\hat z_{k+1}-z_{k+1}\big\|_1,\qquad
\mathcal L_{\rm rollout}(\phi)=\big\|P_\phi(a_{1:T},s_1,z_1)-z_{T+1}\big\|_1
$$

The teacher-forcing term feeds real features at every step, with $T=15$; the rollout term makes the model continue from its own predictions, with $T=2$ in practice. The total loss is the sum of the two.

<!-- paper-figure:vjepa2-ac-training -->

## Planning with embedding distances

Encode the goal image as $z_g$. Planning means finding an action sequence whose predicted future embedding is closest to $z_g$ (§3.2, Eq. 5):

$$
E(\hat a_{1:T};z_k,s_k,z_g)=\big\|P(\hat a_{1:T};s_k,z_k)-z_g\big\|_1
$$

The paper minimises this energy with [CEM](../../05-post-training/deep-rl/planning-and-control.en.md): sample action sequences from zero-mean, unit-variance Gaussians, refit the mean and variance to the lowest-energy batch, repeat a few times and return the mean; execute only the first action and re-plan after seeing the new frame, which is MPC.

Section 4.1 describes the action region as an “L1 ball of radius 0.075” and reports a maximum end-effector displacement of about 13 cm per action, because large actions are out of the models' training distribution. That norm terminology differs from the current public implementation: the [MPC code](https://github.com/facebookresearch/vjepa2/blob/main/notebooks/utils/mpc_utils.py) clips translation coordinates individually, and the [official example](https://github.com/facebookresearch/vjepa2/blob/main/notebooks/energy_landscape_example.ipynb) limits each to $\pm0.075$. In metre-valued coordinates, this is an L∞ constraint, with a Euclidean displacement bound of $\sqrt{3}\times0.075\ \text{m}\approx13\ \text{cm}$, not the displacement bound of a three-dimensional L1 ball.

<!-- paper-figure:vjepa2-ac-planning -->

Compute one round with the cup (teaching numbers). Replace the 16×16×1408 feature map with 2 numbers: (how many cm the cup moved along the table, how far it tipped, 0–1). The current $z_k=(0,0)$; in the goal image the cup has moved 3 cm and stands upright, $z_g=(3,0)$. With a planning horizon of 1, compare 3 candidates:

| Candidate action | Predicted embedding | Energy $\lVert\hat z-z_g\rVert_1$ |
| --- | --- | --- |
| Push 2 cm | (2, 0) | 1 |
| Push 3 cm | (3, 0.2) | 0.2 |
| Push 5 cm | (4, 1) | 2 |

The planner picks push 3 cm, executes it, takes a new frame, re-encodes and plans again; at no point is a single frame of the cup drawn.

Change one condition: the training data contain almost no falling cups, and the predictor returns (3, 0.1) for "push 5 cm", energy 0.1, lower even than push 3 cm. The planner now picks push 5 cm, the cup really falls, and re-planning cannot undo it. This is the "planner exploits the model" problem from the [model-based note](../../05-post-training/deep-rl/model-based.en.md); confining actions to a small range is one way the paper keeps away from regions the model has not seen.

Reported results (Table 2): zero-shot deployment on two Franka arms in two labs, neither environment in DROID; 10 trials per task, averaged over the two labs.

| Task | V-JEPA 2-AC | Octo |
| --- | --- | --- |
| Reach a target position | 100% | 100% |
| Grasp cup | 65% | 15% |
| Grasp box | 25% | 0% |
| Pick-and-place cup | 80% | 15% |
| Pick-and-place box | 65% | 10% |

Table 3 swaps the world model inside the same CEM planner for Cosmos (an action-conditioned video model based on latent diffusion), on one RTX 4090 with a planning horizon of 1: Cosmos needs 4 minutes per action (80 samples) and gets 0% on pick-and-place; V-JEPA 2-AC needs 16 seconds per action (800 samples) and gets 80% and 50% on cup and box in Lab 2.

Keep in mind when reading these numbers: pick-and-place is given two sub-goal images besides the final goal (the object grasped, the object near the target), in an order chosen by people; the planning horizon is 1, so each step looks only one action ahead; and the camera position was settled by manual trial and error (§4.3).

## What is new on this line in 2026

- **V-JEPA 2.1** (Meta FAIR, arXiv 2026-03-15, [link](https://arxiv.org/abs/2603.14482)): both visible and masked tokens contribute to the loss, deep self-supervision across several layers, and tokenizers shared by images and video. The authors report in the abstract a real-robot grasping success rate 20 points above V-JEPA 2-AC.
- **LeWorldModel** (Maes et al., including LeCun and Balestriero, 2026-03-13): trained end to end from pixels with only two losses, next-embedding prediction and a Gaussian regulariser. The authors report about 15M parameters, training on one GPU in a few hours, and planning up to 48× faster than foundation-model-based world models on 2D and 3D control tasks.
- **H-JEPA** (Zhang, Terver, Rabbat, LeCun, Balestriero, 2026-10-05, [link](https://arxiv.org/abs/2610.06805)): a hierarchy of action-conditioned JEPAs, each level predicting farther ahead in its own latent space; planning runs top-down, and an upper level's predictions become sub-goals for the level below. The abstract says that when factors in the data change on separate timescales, higher levels drop fast, unpredictable detail and keep slower, task-relevant state. On Visual AntMaze a three-level hierarchy raises success from 18% to 73% with less planner compute. In the cup example, the upper level handles "the cup ends 3 cm from the edge" and the lower level each 1 cm push.
- **AMI Labs**: the company LeCun co-founded announced on 2026-03-10 a $1.03B round and a mission centred on world models: understanding the world, persistent memory, reasoning and planning, controllability and safety ([AMI Labs](https://amilabs.xyz/updates)). This is funding and self-description, not technical results. As of 2026-10-15 its updates page lists no model, paper or code; H-JEPA's author affiliations include Advanced Machine Intelligence, but that is a paper, not a model release from the company.

## Compared with pixel video prediction and Dreamer-style models

| | JEPA | Pixel video prediction | Dreamer-style latent model |
| --- | --- | --- | --- |
| Predicts what | Embeddings of masked or future parts | Future frames, or latents that decode back into frames | Next latent state, reward, continuation, while reconstructing observations |
| Needs a decoder? | No, so you also cannot see the frames it predicts | Yes, frames are the output | Reconstruction supplies the training signal; no decoding during imagination |
| Good for | Representation learning; planning toward a goal image | Viewing by people, playable worlds, synthetic data, policy evaluation | Training a policy in imagination |
| Typical failure | Representation collapse; embeddings that drop what the task needs; goals must be images | Slow and expensive, Cosmos needs 4 minutes per action inside CEM; realistic frames are not correct physics | Policies exploit model flaws; reconstruction spends capacity on irrelevant detail |

The lines are not always this clean: [Dreamer 4](dreamer.en.md)'s world model is itself a video model; [DINO-WM](https://arxiv.org/abs/2411.04983) (2024-11) uses frozen DINOv2 patch features as its representation and plans with CEM, close to a JEPA except that the encoder is off the shelf.

## Limits and open questions

- **Planning does not reach far.** The V-JEPA 2-AC paper lists three limits itself (§4.3): sensitivity to camera position; error accumulation over long horizons, with the search space of action sequences growing exponentially in the horizon; and goals that can only be given as images.
- **You cannot see what it is thinking.** Without a decoder, embeddings can only be checked with probes. Joseph et al. (2026-02, ICML 2026) found physical variables becoming decodable abruptly at about one-third depth in V-JEPA 2; Punzo et al. (2026) compared V-JEPA, VideoMAE and a diffusion generator and found that what can be decoded depends more on the benchmark and probe than on the pretraining objective. More in [do they actually understand the world?](do-they-understand.en.md)
- **Collapse prevention keeps changing recipes.** From EMA plus stop-gradient to SIGReg and LeWM's Gaussian regulariser; LeJEPA's main results are on images, and LeWM is a control model of about 15M parameters, so at V-JEPA 2's scale of a million hours of video this note found no matching result.

## What to watch on this line next

Only results that would change the picture, not predictions:

1. **Will AMI Labs release a model?** As of 2026-10-15 its updates page has only funding and mission. A public model that outsiders can reproduce and evaluate is what would let this approach be tested beyond Meta's papers.
2. **Can JEPA planning leave lab robot arms?** V-JEPA 2-AC (Meta, 2025-06) relies on human-provided sub-goal images for pick-and-place, uses a planning horizon of 1, and was tested only in the authors' two labs; H-JEPA (2026-10) raised AntMaze from 18% to 73% with hierarchical planning in simulation, and on real DROID videos evaluated only offline planning fidelity. Long tasks on real robots without sub-goal images, reproduced by teams other than the authors, would change the picture.
3. **Does heuristic-free collapse prevention scale?** LeJEPA (2025-11) reports 79% linear probe on ImageNet; LeWM (2026-03) has about 15M parameters. If SIGReg-like methods matched results at the scale of a million hours of video, EMA plus stop-gradient would stop being a requirement.
4. **Do embeddings hold the physics that planning needs?** Joseph et al. (2026-02) found physical variables in V-JEPA 2's middle layers, while Punzo et al. (2026) found the conclusions depend heavily on probe and benchmark. Only a result that is consistently ahead of, or consistently behind, pixel models across many probes and benchmarks would answer whether "predicting representations" really learns more physics.

Read on: [video you can act in](interactive-video.en.md), or go back to the [guide](README.en.md).
