# 机器人学习：VLA 模型与仿真到现实

**中文** · [English](vla.en.md)

> 阅读时间：约 13 分钟 · 草稿，待审核

手上有一个能看图回答问题的视觉语言模型（VLM），想让它直接开机械臂：输入两路相机画面和一句“把杯子放进水槽”，输出 50 Hz 的关节指令。最直接的办法是把动作也写成文字 token，可一算延迟就会发现跟不上。**截至 2026 年 10 月，常见做法是让 VLM 负责看懂画面和指令，旁边接一个连续的 flow-matching“动作专家”，一次生成一整块动作；人形机器人再把慢的推理和快的控制拆成两三层。仿真里训出的策略要上真机，靠的是随机化和在线适应；评测则在从接近满分的仿真榜单转向真机对比。**

> **待审核草稿。** 这一篇还没有经过站主审阅，事实、取舍和措辞都可能修改；引用前请回到文中链接的一手资料核对。

先读：[从示范中学动作](imitation.md)（多峰动作、动作块），[LLaVA 与 DeepSeek-VL](../../03-multimodal-learning/vlm-designs.md)（图片怎样接进语言模型）。

## 动作写成什么：四种接法

VLA（vision-language-action）这个词出自 [RT-2](https://arxiv.org/abs/2307.15818v1)（Google DeepMind，2023-07）。区别各家 VLA，先看动作从模型的哪里出来、写成什么。

**1. 动作当文本 token。** RT-2 的动作是 8 个数：末端 6 个自由度的位移和转动、夹爪开合、一个“结束”标志；连续的维度均匀切成 256 个 bin，整条动作写成一串整数 token，再和网页上的看图问答数据混在一起微调（§3.2）。55B 的版本放在多 TPU 的云服务上，控制频率 1–3 Hz，5B 版本约 5 Hz（§3.3）。[OpenVLA](https://arxiv.org/abs/2406.09246v1)（Stanford 等，2024-06）沿用这个思路：Llama 2 7B 加 DINOv2 和 SigLIP 两路视觉特征；7 维动作，每维在训练数据 1%–99% 分位之间切 256 个 bin，占用 Llama 词表里最少用的 256 个 token；在一张 RTX 4090 上约 6 Hz（没做编译和投机解码）。

<!-- paper-figure:openvla-arch -->

**2. 先压缩，再变 token。** [FAST](https://arxiv.org/abs/2501.09747v1)（Physical Intelligence、UC Berkeley、Stanford，2025-01）的观察是：高频控制下相邻两步几乎一样，逐步逐维的 token 大多“抄上一个就对”，next-token 的学习信号很弱。原文是“this marginal information approaches zero as the control frequency of the training signal increases …”，即训练信号的控制频率越高，每个新 token 带来的边际信息越趋近于零（[§IV](https://arxiv.org/html/2501.09747v1#S4)）。它对每一维做离散余弦变换（DCT），量化后用 BPE 压缩。作者报告的每 1 秒动作块 token 数：7 维、20 Hz 的收拾餐桌从 140 降到 28；14 维、50 Hz 的叠 T 恤从 700 降到 53（表 I）。π0-FAST 能追平扩散版 π0，训练时间最多少 5 倍（作者）。

**3. 连续的动作专家。** [π0](https://arxiv.org/abs/2410.24164v1)（2024-10）= PaliGemma 3B + 约 3 亿参数的 flow-matching 动作专家；动作块长 50，最高 50 Hz；推理时积分 10 步，图像、文字和机器人状态组成的前缀只算一次、KV 缓存起来，每步只重算动作那一段（§IV）。[π0.5](https://arxiv.org/abs/2504.16054v1)（2025-04）先用文字预测下一个子任务，再由 3 亿参数的专家出 50 步的动作块；[π0.7](https://arxiv.org/abs/2604.15483v1)（2026-04）换成 Gemma 3 4B 加 8.6 亿参数的专家，提示里加入质量、速度等元数据和子目标图片。NVIDIA 的 [GR00T N1](https://arxiv.org/abs/2503.14734v1)（2025-03）是 VLM（System 2）加 diffusion transformer 动作模块（System 1）端到端联合训练；Hugging Face 的 [SmolVLA](https://arxiv.org/abs/2506.01844v1)（2025-06）总共约 4.5 亿参数，其中约 1 亿是 flow-matching 专家。直接给 VLM 接一个随机初始化的连续专家也有代价：PI 的 [knowledge insulation](https://arxiv.org/abs/2505.23705v1)（2025-05）报告这会明显拖慢训练、损害知识迁移；正文（§1、§5.2）的解释是专家的梯度回传会破坏 VLM 主干，做法是把主干和这部分梯度隔开。

<!-- paper-figure:pi0-overview -->

**4. 不用 token，也不用扩散。** [OpenVLA-OFT](https://arxiv.org/abs/2502.19645v1)（Stanford，2025-02）把 OpenVLA 改成并行解码（双向注意力，一次前向就出整块）、连续动作、L1 回归。作者报告 LIBERO 平均成功率从 76.5% 升到 97.1%（最好的配置用了额外输入），动作生成吞吐提高 26 倍（块长 8 时测得）。

| 接法 | 代表 | 出一块 50 步 × 7 维的动作，要几次串行前向 | 多峰怎么表达 |
| --- | --- | --- | --- |
| 逐维 bin 的 token | RT-2、OpenVLA | 350 次（OFT 论文：块长 $K$、维数 $D$ 要 $KD$ 次） | 每个 token 在 256 个 bin 上做 softmax |
| 压缩后的 token | π0-FAST | 作者说通常 30–60 次 | 每个 token 在 BPE 词表上做 softmax（单数据集实验里词表为 1,024） |
| flow-matching 专家 | π0、π0.5、π0.7、GR00T、SmolVLA | 前缀 1 次 + 专家 10 次 | 噪声采样，见上一篇 |
| 并行解码 + 回归 | OpenVLA-OFT | 1 次 | 不表达，输出点估计 |

## 算一次：一块动作要等多久（教学数字）

下面的单步耗时都是**教学假设**，只为让量级能手算，不是任何模型的实测：

- 前缀：图像、指令和机器人状态过一遍 VLM，$T_{\rm prefix}=40$ ms，三种接法相同；
- 自回归解码：约 30 亿参数的主干每生成一个 token，$c_{\rm tok}=10$ ms；
- flow 一步：约 3 亿参数的动作专家（前缀 KV 已缓存）每积分一步，$c_{\rm flow}=3$ ms；
- 动作块：50 步 × 7 维，50 Hz 下正好 1 秒的动作；FAST 压缩后设为 40 个 token（作者说通常 30–60 个）。

$$
T_{\rm AR}=T_{\rm prefix}+N_{\rm tok}\,c_{\rm tok},\qquad
T_{\rm flow}=T_{\rm prefix}+K\,c_{\rm flow}
$$

$N_{\rm tok}$ 是要串行生成的 token 数，$K$ 是 flow 的积分步数。代进去：

| 接法 | 串行步数 | 出一块的时间 | 和 1 秒的动作比 |
| --- | --- | --- | --- |
| 逐维 bin 的 token | $N_{\rm tok}=50\times7=350$ | $40+350\times10=3{,}540$ ms | 生成比执行慢 3.5 倍，跟不上 |
| FAST（设 40 个 token） | $N_{\rm tok}=40$ | $40+40\times10=440$ ms | 平均跟得上，但每块要等近半秒 |
| flow-matching 专家 | $K=10$ | $40+10\times3=70$ ms | 余量很大 |

逐维 token 比 flow 慢约 $3540/70\approx51$ 倍，FAST 慢约 $440/70\approx6$ 倍。如果逐维 token 不做动作块、每次只出一步，$40+7\times10=110$ ms，最多约 9 Hz，和 RT-2（1–3 Hz）、OpenVLA（约 6 Hz）实测的个位数 Hz 在同一个量级。

再对照真实测量。π0 论文（[v1 附录 D，表 I](https://arxiv.org/abs/2410.24164v1)）在 RTX 4090、3 路相机下测得：图像编码 14 ms，前缀前向 32 ms，10 步 flow 共 27 ms，机载合计 73 ms；移动机器人经 WiFi 放到机外推理时再加 13 ms 网络。FAST 论文（§VI-E）报告：同样在 RTX 4090 上，扩散版 π0 出 1 秒动作块在 100 ms 以内，π0-FAST 每块约 750 ms，因为要解码 30–60 个 token，而且每个 token 都要过 20 亿参数的完整主干，不是 3 亿参数的专家。作者也说，在他们测的静态操作任务上，这个延迟没有拖累表现。

<!-- widget:step-vla-latency -->

这个算例**不说明**的事：

- 串行步数和策略好坏是两回事：π0-FAST 的串行步数多得多，作者报告它在成功率上追平了 π0。
- 单步耗时随 GPU、量化、batch 和实现变化很大；比较时要写清硬件和配置。
- flow 方案里前缀占了大头：教学数字里是 $40/70\approx57\%$，π0 的实测里图像编码加前缀是 $46/73\approx63\%$。想再快，先压前缀。
- 真实系统会让推理和执行重叠：π0 在 50 Hz 机器人上每执行 25 步（0.5 秒）就推理下一块；SmolVLA 的异步推理把任务完成时间从 13.75 秒降到 9.7 秒（[论文 §4.6](https://arxiv.org/abs/2506.01844v1)，作者报告）。

## 快慢分层：为什么人形机器人拆成两三层

站稳和抓握要毫秒级的反馈，“把杯子放进水槽”这句话的含义却几秒才变一次。一个几十亿参数的模型跑不到 200 Hz，于是拆层：慢层理解场景和指令，快层把它变成关节指令。

| 系统（时间） | 慢层 | 快层 | 备注 |
| --- | --- | --- | --- |
| Figure [Helix](https://www.figure.ai/news/helix)（2025-02） | S2：基于一个开源、开放权重的 7B VLM，7–9 Hz | S1：8000 万参数的 transformer，200 Hz，控制 35 自由度的上半身 | 两块低功耗嵌入式 GPU；约 500 小时遥操作数据（公司自报）；Helix 本身不开放 |
| Figure [Helix 02](https://www.figure.ai/news/helix-02)（2026-01） | S2 | S1 输出全身关节目标（200 Hz）；新增 S0：1000 万参数、1 kHz 的全身控制器，在仿真里用 1,000 小时以上重定向的人体动作训练 | 4 分钟自主清空并重装洗碗机，S0 替换了 109,504 行 C++（公司自报） |
| NVIDIA [GR00T N1](https://arxiv.org/abs/2503.14734v1)（2025-03） | VLM（System 2） | diffusion transformer 动作模块（System 1） | 两部分联合训练；在 Fourier GR-1 上演示 |
| Google DeepMind [Gemini Robotics 2](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/)（2026-07-30） | ER 2：规划几分钟长的任务；[模型卡](https://deepmind.google/models/model-cards/gemini-robotics-er-2/)写明基于 Gemini 3.5 Flash | 同一个 VLA checkpoint 既做 Apptronik Apollo 2 的全身控制，也控制双臂的 Franka Duo | 博客给的 11 个成功率在 32%–92% 之间，如拧下灯泡 92%、用簸箕 32%（公司自报）；DeepMind 自己说速度还要改进，多指灵巧操作仍难 |
| [π0.5](https://arxiv.org/abs/2504.16054v1)（2025-04） | 同一个模型先用文字预测子任务 | 再由动作专家出 50 步动作块 | 一个模型里的两步，不是两个模型 |

算一下层间节奏：S2 取 8 Hz，每 125 ms 更新一次；这期间 200 Hz 的 S1 跑 $125/5=25$ 次，1 kHz 的 S0 在 S1 的每一步里再跑 5 次，合计 125 次。代价是：慢层看错了，错误会持续整整一个慢周期；层与层之间传什么（隐向量、文字子任务还是关节目标），各家公开得都不多。

## 在 VLA 上接 RL：学示范里没有的东西

示范只能教到示范者的水平，速度也被示范者限住；摔了、卡住之后怎么接着干，示范里也很少有。2025–26 年，Physical Intelligence 开始在 VLA 上用机器人自己的经验做 RL（以下均为公司自报）：

- **[π\*0.6 / Recap](https://www.pi.website/blog/pistar06)（2025-11-17）。** 先训一个价值函数，用它估计每段经验的 advantage（这一步比平均好多少，见 [Actor–Critic 与 GAE](../../05-post-training/deep-rl/actor-critic-gae.md)），再让 π0.6 以这个信号为条件学习，预训练阶段属于离线 RL（见 [Offline RL 与 OPE](../../05-post-training/deep-rl/offline-and-ope.md)）；数据包括示范、机器人自己的经验和遥操作员的纠正。PI 报告在部分最难的任务上吞吐量翻倍以上、失败率降到一半或更低，并演示从早上 5:30 到晚上 11:30 连续做咖啡。
- **[RL tokens](https://www.pi.website/research/rlt)（2026-03-19）。** 先训练 VLA 输出一个“RL token”，之后冻结 VLA，在这个 token 上训一个小的 actor-critic 做在线 RL，由它修改 VLA 给出的动作。PI 报告最精细的阶段最多快 3 倍，最少只用 15 分钟真机数据；测了 4 个精细任务（拧 M3 螺丝、扎扎带、插网线、插电源线），RL 只针对其中的关键阶段。
- **[π0.7](https://www.pi.website/blog/pi07)（2026-04-16）。** PI 说这个通用模型在叠衣服、做咖啡、折箱子上追平了用 RL 训出的 π\*0.6 专用模型；[论文](https://www.pi.website/download/pi07.pdf)结论同时承认，没见过的任务成功率只有 60%–80%。

在仿真里训练腿足或全身控制器，常见的 RL 算法是 PPO 和 SAC，见[策略该改多大](../../05-post-training/deep-rl/trust-region.md)和 [SAC](../../05-post-training/deep-rl/soft-actor-critic.md)；Helix 02 的 S0 也是在仿真里训的。

## 评测：分数到 97% 以后

- **[LIBERO](https://arxiv.org/abs/2306.03310v1)（UT Austin 等，2023-06）**：共 130 个仿真操作任务。VLA 论文常用其中 4 个评测套件（Spatial、Object、Goal、Long），各 10 个任务；OpenVLA-OFT 报告的 97.1% 是这 40 个任务的平均。分数挤到这么高，强方法之间已经很难拉开差距；“LIBERO 接近天花板”是我们的判断，不是哪篇论文的结论。
- **[SIMPLER](https://arxiv.org/abs/2405.05941v1)（2024-05）**：在仿真里复刻真实的桌面和机器人，用来评测在真机数据上训出的策略；作者报告仿真和真机结果强相关。
- **[RoboArena](https://arxiv.org/abs/2506.18123v1)（2025-06）**：在 DROID 平台上做分布式、双盲的真机成对比较，7 家学术机构、600 多个成对 episode、7 个策略。

为什么高分段难比，可以算一下。OFT 的评测协议是每个套件 500 次（10 个任务 × 50 次），4 个套件共 2,000 次。成功率 0.97 时，标准误是 $\sqrt{0.97\times0.03/2000}\approx0.0038$，95% 区间约 ±0.75 个百分点；两个方法差 0.5 个点，差值的标准误约 $\sqrt2\times0.0038\approx0.0054$，还不到 1 个标准误。（这里把 2,000 次当成独立试验；同一任务内的试验相关时，区间还要更宽。）怎样比较两个分数，见[分数涨了，究竟能说明什么](../../07-evaluation/metric-robustness.md)。

用世界模型代替真机来评测，是另一条路，单独在[世界模型给机器人和汽车当评测器](../../11-frontier/world-models/robotics-driving.md)一篇里讲。

## 仿真到现实：差距从哪来，怎么补

仿真便宜、能并行、不怕摔，但摩擦、质量、延迟、画面都和真机不一样。先用一个能手算的例子看三种补法。

教学例子：机械臂把冰球推出去，冰球以初速 $v$ 滑行 $d=v^2/(2\mu g)$ 后停下，目标 $d^\ast=0.5$ m，$g=9.8\ \text{m/s}^2$。真实桌面的摩擦系数 $\mu$ 在 0.3 到 0.7 之间，事先不知道。

- **只在仿真里按 $\mu=0.5$ 调。** $v^2=2\times0.5\times9.8\times0.5=4.9$。到了真桌面：$\mu=0.3$ 时滑 $4.9/5.88\approx0.83$ m，多出 0.33 m；$\mu=0.7$ 时只滑 $4.9/13.72\approx0.36$ m。
- **域随机化（domain randomization）。** 训练时让 $\mu$ 在 0.3 到 0.7 之间随机，策略不知道这一次是多少，只能选一个各种情况都过得去的 $v$。若目标是让最坏误差最小，就让两端误差相等：$\frac{v^2}{2g}\left(\frac1{0.3}+\frac1{0.7}\right)=1$，得 $v^2\approx4.12$，两端分别滑 0.70 m 和 0.30 m，误差都是 0.20 m。更稳了，但很少正中。
- **在线适应（RMA 的思路）。** 先轻推一下，看冰球减速多快（减速度等于 $\mu g$），估出 $\mu$ 再出手。量到减速度 $2.94\ \text{m/s}^2$，得 $\hat\mu=0.3$，取 $v^2=2\times0.3\times9.8\times0.5=2.94$，正好滑 0.5 m；剩下的误差只来自估计误差。

对应的论文：

- [Tobin 等](https://arxiv.org/abs/1703.06907v1)（OpenAI、UC Berkeley，2017-03）随机化渲染，让真实世界看起来只是又一种变化；只用仿真图像训练的物体检测器在真实世界里定位精度 1.5 cm（作者）。动力学随机化见 [Peng 等（2017-10）](https://arxiv.org/abs/1710.06537)。
- [OpenAI 的魔方](https://arxiv.org/abs/1910.07113v1)（2019-10）用自动域随机化（ADR）：策略在一个范围上表现够好，就把随机范围再放宽；只在仿真里训练，机械手完成了魔方还原（还原步骤由 Kociemba 求解器给出，学到的是手上的操作）。
- [RMA](https://arxiv.org/abs/2107.04034v1)（UC Berkeley、CMU、Facebook AI，2021-07）：基础策略加一个适应模块，适应模块从最近的状态和动作历史里在线估计一个编码环境条件的隐向量；只在仿真里训练，作者报告不经微调就让 A1 四足机器人走沙地、台阶、草地等地形。腿足机器人 sim-to-real 的另一篇经典工作是 [Hwangbo 等（2019-01）](https://arxiv.org/abs/1901.08652)。

截至 2026 年 10 月，仿真器在往 GPU 并行上集中：

| 时间 | 系统 | 是什么 | 注意 |
| --- | --- | --- | --- |
| 2025-02 | [MuJoCo Playground](https://arxiv.org/abs/2502.08844v1)（UC Berkeley、Google DeepMind 等） | 基于 MJX 的环境集，带批量渲染 | 作者报告单张 GPU 几分钟训完，状态输入和像素输入都零样本上真机 |
| 2025-09-29 | [Newton](https://www.linuxfoundation.org/press/linux-foundation-announces-contribution-of-newton-by-disney-research-google-deepmind-and-nvidia-to-accelerate-open-robot-learning)（Disney Research、Google DeepMind、NVIDIA，交给 Linux Foundation） | 基于 NVIDIA Warp 和 OpenUSD 的 GPU 物理引擎，多种求解器 | 1.0 版在 2026-03-10 发布到 [PyPI](https://pypi.org/project/newton/) |
| 2025-11 | [Isaac Lab](https://arxiv.org/abs/2511.04831v1)（NVIDIA） | Isaac Gym 的后继：GPU 物理、渲染、执行器与传感器模型、随机化工具 | 按 NVIDIA 2026-03-16 的[新闻稿](https://nvidianews.nvidia.com/news/nvidia-and-global-robotics-leaders-take-physical-ai-to-the-real-world)，Isaac Lab 3.0（early access）建在 Newton 1.0 和 PhysX 上 |
| 持续更新 | [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) | Google DeepMind 和 NVIDIA 用 Warp 重写的 MuJoCo，属于 Newton 项目 | Apache-2.0；[文档](https://mujoco.readthedocs.io/en/latest/mjwarp/)说它目前不可微（MJX 可微） |

## 哪些模型能下载（截至 2026-10-15）

| 模型 | 权重 | 许可与条件 | 来源 |
| --- | --- | --- | --- |
| OpenVLA | 开放 | 代码 MIT；README 说模型沿用 Llama 2 Community License，Hugging Face 模型卡却标 MIT | [GitHub](https://github.com/openvla/openvla) |
| Octo | 开放 | GitHub 仓库和 Hugging Face 标 MIT（项目页没写） | [项目页](https://octo-models.github.io/) |
| π0、π0-FAST、π0.5（openpi） | 开放 | 代码 Apache-2.0；另附 Gemma 使用条款（LICENSE_GEMMA.txt），仓库没说明它适用于哪些文件 | [GitHub](https://github.com/Physical-Intelligence/openpi) |
| SmolVLA | 开放（代码、权重、数据） | 模型卡标 Apache-2.0 | [博客](https://huggingface.co/blog/smolvla) |
| GR00T N1.5、N1.6 | 开放 | 非商用（NVIDIA OneWay Noncommercial License，见模型卡和开发者页面） | [N1.6](https://research.nvidia.com/labs/gear/gr00t-n1_6/)、[开发者页面](https://developer.nvidia.com/isaac/gr00t/llm-info.md) |
| GR00T N1.7 | 开放（2026-04-17 起 early access） | 说法冲突：发布博客称可商用，模型卡写 NVIDIA Open Model License，仓库里的 LICENSE 文件却和 N1.6 的非商用许可一字不差 | [博客](https://huggingface.co/blog/nvidia/gr00t-n1-7)、[开发者页面](https://developer.nvidia.com/isaac/gr00t/llm-info.md) |
| π0.6、π\*0.6、π0.7 | 没有公开说明 | PI 在 2026-10-13 开放托管 API 内测：候补名单、未定价、对象是未命名的最新模型 | [PI 博客](https://www.pi.website/blog/api-beta) |
| Gemini Robotics 2 VLA、On-Device 2；Figure Helix、Helix 02 | 不开放 | Gemini 的 VLA 只给 early-access 合作方，ER 2 在 Google AI Studio 可用；Helix 不对外 | [DeepMind 博客](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/)、[Figure](https://www.figure.ai/news/helix) |
| GR00T N2 | 未发布 | 2026-03-16 预告“年底前”推出，截至 2026-10-15 没找到发布 | [NVIDIA 新闻稿](https://nvidianews.nvidia.com/news/nvidia-and-global-robotics-leaders-take-physical-ai-to-the-real-world) |
| LeRobot（工具库） | 开放 | Apache-2.0；内含 ACT、Diffusion Policy、π0 / π0.5、SmolVLA、GR00T 等策略 | [GitHub](https://github.com/huggingface/lerobot) |

GR00T N1.6 的骨干也有冲突：NVIDIA 研究页写的是内部的 Cosmos-2B 变体，GitHub 仓库和开发者页面写的是 Eagle。

## 人形机器人：官方说了什么（截至 2026-10-15）

Tesla Optimus 只看 Tesla 自己的季度致股东信（SEC 存档的 Exhibit 99.1）：

| 信（日期） | Tesla 写了什么 |
| --- | --- |
| [2025 Q3](https://www.sec.gov/Archives/edgar/data/0001318605/000162828025045861/exhibit991.htm)（2025-10-22） | 第一代 Optimus 产线正在安装，为“volume production”（量产）做准备 |
| [2025 Q4 / 全年](https://www.sec.gov/Archives/edgar/data/0001318605/000162828026003837/exhibit991.htm)（2026-01-28） | 计划 2026 年一季度发布 Gen 3，称它是第一款为大规模生产设计的版本；计划 2026 年底前开始生产，远期产能每年 100 万台 |
| [2026 Q1](https://www.sec.gov/Archives/edgar/data/0001318605/000162828026026551/exhibit991.htm)（2026-04-22） | 弗里蒙特的第一代产线按每年 100 万台设计，替换 Model S / X 产线；得州超级工厂在准备第二代产线，远期设计产能每年 1000 万台 |
| [2026 Q2](https://www.sec.gov/Archives/edgar/data/0001318605/000162828026049213/exhibit991.htm)（2026-07-22） | S / X 产线已拆除，正在装第一代 Optimus 产线，预计很快开始生产；首批机器人用于“Optimus Academy”，采集训练数据；展望一栏的措辞从“volume production”变成“production in 2026” |

截至 2026-10-15，Tesla 的一手资料里没找到 Gen 3 / V3 的正式发布；2026 Q3 的信还没出。关于 V3 发布和周产量的说法只见于报道，这里不采用。

其他公司（公司自报，除非另注）：

- **Figure**：BMW 2026-02-27 的[新闻稿](https://www.press.bmwgroup.com/global/article/detail/T0455864EN/bmw-group-to-deploy-humanoid-robots-in-production-in-germany-for-the-first-time?language=en)确认，Figure 02 在斯帕坦堡工厂参与生产了 3 万多辆 X3（新闻稿没说用了几台）；Figure 在 2026-04-29 称已交付 350 台以上 Figure 03（[Figure 博客](https://www.figure.ai/news/ramping-figure-03-production)），说这些机器人分给内部研发、数据采集和商业用例开发，没有点名客户，也没给现场部署台数。
- **Agility**：Digit 在 GXO 的仓库搬运了 10 万个以上周转箱（[2025-11-20](https://www.agilityrobotics.com/content/digit-moves-over-100k-totes)）。
- **Boston Dynamics**：2026-01-05 发布量产版电动 Atlas，称 2026 年的部署已全部排满，交给 Hyundai 和 Google DeepMind（[博客](https://bostondynamics.com/blog/boston-dynamics-unveils-new-atlas-robot-to-revolutionize-industry/)）；之后没找到确认已交付的一手资料。
- **1X**：2026-04-30 的[工厂公告](https://www.1x.tech/discover/neo-factory)说 NEO 尚未开始向客户交付，首批先给内部做家庭测试，承诺 2026 年内开始交付；之后没找到确认交付的官方消息。

## 接下来看什么

- NVIDIA 是否按预告在年底前发布 GR00T N2，以及权重和许可；π0.6 / π0.7 是否开放权重，还是只走 API。
- 有没有不由模型作者自己做的真机对比，比如更多策略进入 RoboArena 这类双盲评测。
- Tesla 2026 Q3 的信怎样描述 Optimus 的生产。

回到[系列导读](README.md)；动作头的细节在[上一篇](imitation.md)；世界模型直接当策略（DreamZero、1X）见[前沿板块](../../11-frontier/world-models/robotics-driving.md)。

参考：[RT-2](https://arxiv.org/abs/2307.15818v1) · [OpenVLA](https://arxiv.org/abs/2406.09246v1) · [FAST](https://arxiv.org/abs/2501.09747v1) · [π0](https://arxiv.org/abs/2410.24164v1) · [OpenVLA-OFT](https://arxiv.org/abs/2502.19645v1) · [RMA](https://arxiv.org/abs/2107.04034v1)
