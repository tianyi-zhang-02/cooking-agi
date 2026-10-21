# 机器人学习：从示范到通用策略

**中文** · [English](README.en.md)

> 阅读时间：约 6 分钟 · 最近审阅：2026-10

给机械臂录 100 条示范，训一个策略，它能学会绕开杯子去拿盒子吗？换一张桌子、换一句指令、换一台机器人呢？**这个系列只有两篇：第一篇讲怎样从示范里学动作，为什么“平均一下”会撞上障碍，动作块和数据怎么取舍；第二篇讲怎样把视觉语言模型变成机器人策略（VLA），以及在仿真里学到的东西怎样搬到真机。**

## 两篇怎么分工

| 篇 | 回答什么 | 手算的例子 | 读完应该能说清 |
| --- | --- | --- | --- |
| [从示范中学动作](imitation.md) | 行为克隆在机器人上多出哪些问题；生成式动作头和动作块各解决什么；数据从哪来 | 左右绕障：MSE 选 0 撞上杯子；flow matching 只走 1 步也会退回 0，要多走几步才分到两边；块长从 1 到 50 的抖动和反应延迟 | 为什么现在的策略一次输出一段动作，而且用 diffusion / flow matching |
| [VLA 模型与仿真到现实](vla.md) | VLM 怎样变成机器人策略；人形机器人为什么拆快慢层；RL 怎样接在 VLA 后面；怎样评测；仿真到真机怎样补差距 | 一块 50 步 × 7 维的动作：逐 token 解码约 3.5 秒，flow-matching 专家约 70 毫秒（教学数字）；推冰球时摩擦系数不确定 | 各家 VLA 的动作从哪里出来、为什么；哪些模型开放；人形机器人公司官方说了什么 |

一条动作从数据走到电机，大致经过这几步；两篇分别讲前半和后半：

```text
示范：遥操作、手持夹爪、第一人称人类视频
→ 策略：画面 + 指令 + 机器人状态 → 一段动作（action chunk）
→ 动作头：回归 / 离散 token / diffusion 或 flow matching
→ 执行：20–50 Hz 发给控制器，执行一部分后重新推理
→ 评测：仿真基准、真机双盲对比
→ 改进：在机器人自己的经验上做 RL；在仿真里随机化后上真机
```

## 先读什么

| 先读 | 为什么 |
| --- | --- |
| [模仿与奖励](../../05-post-training/deep-rl/imitation-and-rewards.md) | 行为克隆、误差累积、DAgger、因果混淆；第一篇默认你读过 |
| [Model-based RL](../../05-post-training/deep-rl/model-based.md) | “把两条路平均成墙”和 MPC 只执行第一步，第一篇的两个例子都从这里接着讲 |
| [Qwen-Image：flow matching](../../03-multimodal-learning/image-generation.md) | flow matching 的目标函数和一维算例，第一篇沿用它的记号 |
| [LLaVA 与 DeepSeek-VL](../../03-multimodal-learning/vlm-designs.md) | 图片怎样接进语言模型；第二篇默认你知道 VLM 的基本结构 |
| [Actor–Critic 与 GAE](../../05-post-training/deep-rl/actor-critic-gae.md)、[策略该改多大（PPO）](../../05-post-training/deep-rl/trust-region.md)、[Offline RL 与 OPE](../../05-post-training/deep-rl/offline-and-ope.md) | 第二篇讲 RL 接在 VLA 后面、仿真里训练控制器时用到 |

可选：[一个三维点怎样变成像素](../perception/camera-geometry.md)（相机几何，看懂多视角输入有帮助）；[世界模型给机器人和汽车当模拟器、评测器与策略](../../11-frontier/world-models/robotics-driving.md)（这里不重复）。

## 截至 2026-10 的状态

下面每条截至 2026-10-15，出处在同一句里。“公司自报”指数字由公司自己发布，没有独立核查。

- **VLA 的常见配方**：预训练 VLM 加一个连续的“动作专家”（flow matching 或 diffusion），一次输出约 50 步、最高 50 Hz。例如 [π0](https://arxiv.org/abs/2410.24164v1) 是 PaliGemma 3B 加 3 亿参数的专家，[π0.7](https://arxiv.org/abs/2604.15483v1) 是 Gemma 3 4B 加 8.6 亿参数的专家；GR00T N1.x、SmolVLA 也是这个路子。
- **人形机器人拆快慢层**：Figure 的 [Helix](https://www.figure.ai/news/helix) 用 7–9 Hz 的 7B VLM 加 200 Hz 的 8000 万参数策略，[Helix 02](https://www.figure.ai/news/helix-02) 再加一个 1 kHz 的全身控制器（公司自报）；GR00T 和 [Gemini Robotics 2](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/)（2026-07-30）也这样拆。
- **RL 开始跑在 VLA 上**：Physical Intelligence 的 [π\*0.6 / Recap](https://www.pi.website/blog/pistar06)（2025-11）和 [RL tokens](https://www.pi.website/research/rlt)（2026-03）都报告了吞吐量或速度的提升（公司自报）。
- **数据押在第一人称人类视频**：NVIDIA 的 [EgoScale](https://arxiv.org/abs/2602.16710v1) 用了 20,854 小时；Figure 的 [Index](https://www.figure.ai/news/introducing-index) 称有 1,600 万条以上视频；Generalist 的 [GEN-0](https://generalistai.com/blog/gen-0) 称有 27 万小时的操作数据、每周约增加 1 万小时，由“上千台采集设备和机器人”采集，但没说人和机器人各占多少（均为公司自报）。
- **视频世界模型进入策略本身**：例如 14B 的 [DreamZero](https://arxiv.org/abs/2602.15922v1)（2026-02）；NVIDIA 在 2026-03 预告的 [GR00T N2](https://nvidianews.nvidia.com/news/nvidia-and-global-robotics-leaders-take-physical-ai-to-the-real-world) 基于它，截至 2026-10-15 没找到正式发布。细节见[前沿板块的世界模型一篇](../../11-frontier/world-models/robotics-driving.md)。
- **开放程度分两边**：[openpi](https://github.com/Physical-Intelligence/openpi)（π0、π0-FAST、π0.5）、GR00T N1.x、OpenVLA、Octo、SmolVLA 有开放权重；Gemini Robotics 的 VLA、Figure Helix、π0.6 / π0.7 没有。PI 在 2026-10-13 开放了托管 API 内测（[PI 博客](https://www.pi.website/blog/api-beta)）。许可细节见第二篇的表。
- **仿真往 GPU 并行集中**：[Newton](https://www.linuxfoundation.org/press/linux-foundation-announces-contribution-of-newton-by-disney-research-google-deepmind-and-nvidia-to-accelerate-open-robot-learning) 在 2025-09 交给 Linux Foundation，1.0 版在 2026-03-10 发布；NVIDIA 2026-03-16 的[新闻稿](https://nvidianews.nvidia.com/news/nvidia-and-global-robotics-leaders-take-physical-ai-to-the-real-world)说 Isaac Lab 3.0（early access）建在 Newton 1.0 和 PhysX 上；MuJoCo Warp 也属于 Newton 项目。
- **评测转向真机**：[RoboArena](https://arxiv.org/abs/2506.18123v1) 做双盲的真机成对比较；OpenVLA-OFT 在 LIBERO 常用的 4 个评测套件上报告平均 97.1%，我们的判断是这个仿真基准已经接近天花板。
- **人形机器人有真实部署，但规模小**：BMW 确认 Figure 02 参与生产了 3 万多辆 X3（没说用了几台）；Tesla 2026-07-22 的致股东信说第一代 Optimus 产线正在安装，“in anticipation of production in 2026”，首批机器人先用来采集训练数据。按官方说法逐家整理的列表在[第二篇末尾](vla.md)。

### 人行道配送机器人

同样是在真实世界里跑的机器人，下面三家里送得最多的 Starship 自报已经送了 1,000 万单。这里主要用公司自己的资料（数字均为公司自报），措辞也照原样保留：

- **Coco Robotics**：官网 [About 页面](https://www.cocodelivery.com/about)（2026-10-15 读取）列出 50 万次配送、生产了 1,000 台机器人（页面写作 500K、1K），并说两位创始人在 UCLA 读书时开始做这个项目；2025-06 和 2026-07-15 的[新闻稿](https://www.webull.com/news/15231685580170240)（PR Newswire 稿件的转载）写的都是“over 500,000”，一年多没变。合作平台包括 Uber Eats（洛杉矶，2024-08 起，据 [The Robot Report](https://www.therobotreport.com/uber-eats-rolls-out-coco-sidewalk-delivery-robots-in-la/)）、芬兰的 [Wolt](https://www.cocodelivery.com/blog/wolt-and-coco-launch-robot-deliveries-in-turku)，以及 2026-10-05 宣布的英国 [Deliveroo](https://www.cocodelivery.com/blog/coco-robotics-deliveroo-uk-partnership)。**说法不一致的地方要留着看**：About 页面上 CEO 的简介说公司做的是“human-piloted”（由人驾驶）的人行道机器人，而同一页的其他介绍和 2025 年以来的[新闻稿](https://www.prnewswire.com/news-releases/coco-robotics-raises-80m-to-expand-autonomous-delivery-and-ai-platform-302478405.html)都称“autonomous”（自主）；[Coco 2 页面](https://www.cocodelivery.com/coco2)写的是端到端神经网络加人工监督，需要时由“Pilot Terminal”介入。Coco 没有公布多少里程是自主、多少是远程驾驶，我们也没找到可靠媒体给出这个比例。
- **Starship**：2026-04-28 称完成 1,000 万次自主配送，有 3,000 台以上机器人，在 8 个国家运营（[新闻稿](https://www.starship.xyz/press/autonomous-delivery-moves-into-the-mainstream-as-starship-technologies-passes-10-million-deliveries/)），并把它描述为无需人主动监督的 Level 4；2026-06-04 宣布逐步结束美国大学校园业务，把 1,200 台以上校园机器人转去服务超市（[新闻稿](https://www.starship.xyz/press/starship-technologies-doubles-down-on-grocery/)）。远程协助的比例，2026 年的公告里没有。
- **Serve Robotics**：2025 年年报（[10-K](https://www.sec.gov/Archives/edgar/data/1832483/000183248326000010/patr-20251231.htm)）说截至 2025-12-31 车队超过 2,000 台，并把机器人描述为能“在一段时间内”无人介入运行的 Level 4，配有远程监督员；2026 年二季度的日均活跃机器人是 792 台（[8-K 附件 99.1](https://www.sec.gov/Archives/edgar/data/1832483/000183248326000034/serv-20260630_ex991earning.htm)）。

这些都是公司自己的说法。“自主”在不同公司的资料里含义不同，比较时先问：一个远程操作员同时看几台，什么时候接管，接管次数公开了没有。汽车上的同类问题见[自动驾驶系列](../driving/README.md)。

## 这一组不讲什么

- PID、LQR、MPC 这些底层控制：见[规划再拆一层](../../05-post-training/deep-rl/planning-and-control.md)。
- 相机几何、跟踪、三维重建：见[感知与几何系列](../perception/README.md)。
- 自动驾驶：见[自动驾驶系列](../driving/README.md)。
- 任何公司的招聘流程或面试题：这里不收集。

## 阅读顺序

1. [从示范中学动作](imitation.md)：左右绕障的均方误差，flow matching 的采样步数，动作块的长度取舍，ALOHA、UMI、DROID、Open X-Embodiment、AgiBot World 和第一人称人类视频
2. [VLA 模型与仿真到现实](vla.md)：从文本动作 token 到 FAST 再到 flow-matching 动作专家，延迟手算，快慢分层，VLA 上的 RL，评测，仿真到现实，开放程度表和人形机器人现状
