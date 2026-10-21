# 世界模型：能交互的视频（Genie 与可玩世界）

**中文** · [English](interactive-video.en.md)

> 阅读时间：约 12 分钟 · 最近审阅：2026-10

给视频模型一句“机械臂把杯子往桌边推 5 厘米”，能拿到一段几秒的视频。可如果你想推到一半停下、改成只推 2 厘米，再转头看看桌子另一边，就需要一个能接收动作、实时给出下一帧、还记得看不见的东西的模型。

交互式世界模型除了要保证单帧画质，还要同时满足因果生成、延迟预算和记忆这三条约束。截至 2026 年 10 月，这条线上备受关注的 Genie 3 仍没有公开论文和架构。

## 从“生成一段”到“按一下出一帧”

离线视频生成器拿到完整提示，一次交付整段视频；交付之前，任何一帧都还可以改。交互式模型必须先把第 $t$ 帧交给玩家，然后才知道玩家下一步按什么。所以它只能以“过去的画面 + 到目前为止的动作”为条件生成下一帧，这就是因果（causal）生成。

用本系列的例子（教学设定）：机械臂正把杯子推向桌边。离线生成时，“推 2 厘米”和“推 5 厘米”是两条提示，得到两段互不相干的视频。交互时，前 10 帧完全一样，到第 11 帧玩家才决定继续推还是停下；模型生成前 10 帧时，不能“偷看”这个决定。

| | 离线生成（Sora、Veo 一类） | 交互式世界模型（Genie 3 一类） |
| --- | --- | --- |
| 条件什么时候给 | 开始前一次给完 | 每一帧都可能来新动作 |
| 生成时能否参考后面的帧 | 交付前整段都能改 | 不能，后面取决于还没按的键 |
| 延迟要求 | 等几十秒也能接受 | 每帧都要在帧间隔内出来 |
| 长度 | 几秒到十几秒的片段 | 按会话算，分钟级 |
| 记忆 | 片段内一致即可 | 转头再转回来，东西还该在 |

粗略地说，Genie 3、GWM、Odyssey 这类系统加上了动作条件、实时因果生成和持久性，Sora、Veo 则是离线的“提示到片段”生成器。两边的说法也不一样：

- OpenAI 在 2024-02 的 [Sora 技术报告](https://openai.com/index/video-generation-models-as-world-simulators/)里，把扩大视频生成规模称为通向物理世界通用模拟器的一条有希望的路，同时列出玻璃碎裂、吃东西不改变物体状态等失败。[Sora 2 发布页](https://openai.com/index/sora-2/)（2025-09-30）说，有用的模拟器要能模拟失败，比如投篮不中应该弹框，而不是瞬移进篮。据 OpenAI [帮助页](https://help.openai.com/en/articles/20001152-what-to-know-about-the-sora-discontinuation)，Sora App 已于 2026-04-26、API 于 2026-09-24 关闭。
- Google 的 [Veo 3 技术报告](https://storage.googleapis.com/deepmind-media/veo/Veo-3-Tech-Report.pdf)描述的是对音频和视频 latent 的联合 diffusion，没有自称世界模型。Google 2026 年的视频模型 [Gemini Omni](https://deepmind.google/models/gemini-omni/) 页面也没用这个词，并把 Genie 3 另列为专门模型。

## 延迟预算：24 fps 时每帧约 42 毫秒

先算一笔账。设目标帧率为 fps，生成一帧要跑 $K$ 个采样步（diffusion 去噪步或 MaskGIT 解码步），则：

$$
\text{每个采样步最多可用的时间}\approx\frac{1000\ \text{ms}}{\text{fps}\times K}
$$

这里忽略了解码成像素、网络传输和排队，所以是上限。24 fps 时一帧只有 $1000/24\approx41.7$ ms。若每帧 25 步（[Genie](https://arxiv.org/abs/2402.15391) 推理时每帧用 25 个 MaskGIT 步），每步约 1.7 ms；蒸馏到 4 步，每步约 10.4 ms；3 步约 13.9 ms。Genie 论文（2024-02）§5 自己写明它大约只有 1 FPS。

这解释了 2025–26 年的开源系统为什么走到同一个方向：逐帧因果的 diffusion Transformer，用 diffusion forcing 或 self-forcing 训练，再蒸馏到 2–4 个采样步。这是我们对多篇工作的归纳，不是某一篇论文的结论。例如 Skywork 的 [Matrix-Game 2.0](https://arxiv.org/abs/2508.13009)（2025-08）用 3 步；Dreamer 4（2025-09）每帧做 4 次前向；Overworld 的 [Waypoint-1](https://huggingface.co/blog/waypoint-1) 先 diffusion forcing、再 self-forcing。Diffusion forcing 让每一帧有自己的噪声水平，细节见 [Dreamer 一篇](dreamer.md)。

帧率不等于响应延迟。系统可以流水线式地稳定输出 24 fps，按键后的画面却晚几帧才反应。Google 在 2026-01-29 介绍 [Project Genie](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/project-genie/) 时就写明角色可能有延迟；Decart 在 2026-06-10 自报 [Oasis 3](https://decart.ai/publications/introducing-oasis-3-first-interactive-world-model-for-physical-ai) 的延迟低于 200 ms（公司自报）。“先出声不等于全程不卡”的道理，[Omni 流式一篇](../../03-multimodal-learning/omni-streaming.md)讲过。

硬件要和数字写在一起。以下帧率都是发布方自己报告的：Matrix-Game 2.0 在一张 H100 上 25 FPS；[Matrix-Game 3.0](https://arxiv.org/abs/2604.08995) 的 720p、40 FPS 用了 8 张 GPU 跑 DiT、再加 1 张跑 VAE（论文 §3.4）；[Waypoint-1.5-1B 模型卡](https://huggingface.co/Overworld/Waypoint-1.5-1B)写的是一张 RTX 5090 上 720p、56 FPS；[Dreamer 4](https://arxiv.org/abs/2509.24527) 论文表 1 是一张 H100 上 640×360、21 FPS。分辨率和 GPU 数都不同，不能直接排名次。

## Genie 的配方：没有动作标签，也能学会“按键”

网上有海量游戏视频，却几乎没有记录玩家按了什么键。[Genie](https://arxiv.org/abs/2402.15391)（Google DeepMind，2024-02，ICML 2024 Best Paper）的做法是：从相邻两帧的变化里，自己学出一套离散“动作”。论文 §2.1 把模型分成 3 部分，都由时空（ST）Transformer 块搭成：

```text
视频帧 x_1..x_T
 ├─ 1. 视频 tokenizer（VQ-VAE，时间上因果）→ 每帧的离散 token z_1..z_T
 ├─ 2. 潜在动作模型 LAM
 │     编码器：看 x_1..x_t 和 x_{t+1} → 连续向量 ã_t → 量化成 8 个代码之一
 │     解码器：只看历史和 ã_t → 必须重建 x_{t+1}
 └─ 3. 动力学模型（MaskGIT Transformer）
       输入 z_1..z_{t-1} 和 ã_1..ã_{t-1}（stop-gradient）→ 预测下一帧的 token
推理：LAM 只留码本；玩家输入 0–7 的整数 → 查码本 → 动力学模型生成下一帧
```

关键在 LAM 的解码器：它看不到未来帧，只能靠动作代码补上“从过去到未来变了什么”。码本又只有 $|A|=8$ 个代码（论文说这个大小是为了让人能玩），装不下整张图，只能装最主要的那个变化。论文没有给 LAM 写编号的损失公式，只有文字描述。

<!-- paper-figure:genie-lam -->

### 用 8 个代码走一遍（教学例子）

假设训练完后，码本里的 8 个向量如下。数字是编的，只为能手算；真实码本维度更高，每个代码是什么意思，也要靠人事后试玩才知道。

| 代码 | 向量 | 试玩后人给它贴的标签 |
| --- | --- | --- |
| 0 | (0, 0) | 不动 |
| 1 | (0.5, 0) | 往右推一点 |
| 2 | (1.2, 0) | 往右推一大段 |
| 3 | (−0.5, 0) | 往左一点 |
| 4 | (−1.2, 0) | 往左一大段 |
| 5 | (0, 0.6) | 抬起 |
| 6 | (0, −0.6) | 放下 |
| 7 | (0.8, 0.8) | 斜向上 |

两帧之间杯子被推了 2 厘米，LAM 编码器输出 $\tilde a=(0.45,0.05)$。按平方距离找最近的代码：到代码 1 是 $0.05^2+0.05^2=0.005$，到代码 0 是 $0.45^2+0.05^2=0.205$，到代码 2 是 $0.75^2+0.05^2=0.565$，所以量化成代码 1。推 5 厘米时编码器输出 $(1.1,0.1)$：到代码 2 是 $0.01+0.01=0.02$，到代码 1 是 $0.36+0.01=0.37$，选代码 2。

再试推 3.5 厘米，编码器输出 $(0.85,0)$：到代码 1 和代码 2 的平方距离都是 $0.35^2=0.1225$，正好落在边界上。玩家手里只有 8 个键，“推 3.5 厘米”不在其中。码本小，好玩也好学，但能表达的动作就粗；这是设计取舍。

还有一点是我们的推断，不是论文结论：这个目标只要求代码说明“过去到未来最主要的变化”，并不区分变化是不是智能体造成的。如果训练视频里镜头经常自己晃，某个代码可能就学成了“镜头晃”。

可控性怎么量？论文 §3 定义：

$$
\Delta_t\text{PSNR}=\text{PSNR}(x_t,\hat x_t)-\text{PSNR}(x_t,\hat x'_t)
$$

$\hat x_t$ 用从真实视频推出的 latent action 生成，$\hat x'_t$ 用随机 latent action 生成，论文报告 $t=4$。PSNR 越高，生成帧越接近真实帧。教学数字：前者 27.0 dB、后者 24.5 dB，差 2.5 dB，说明代码确实左右了画面；如果动力学模型无视代码，两者会差不多，差值接近 0。这个指标只说明“动作有影响”，不说明影响在物理上对不对。

规模方面，论文 §3 从 5500 万个平台游戏片段中过滤出 3 万小时（680 万段 16 秒片段，10 FPS、160×90）。摘要称 11B，§3.1 拆成 10.1B 动力学、2 亿 tokenizer 和 3 亿 LAM。§5 写明的局限：可能幻想出不真实的未来，只有 16 帧记忆，约 1 FPS。同样“从无标签视频里推动作”的思路，后来也被用来给机器人视频补伪动作，见[机器人与汽车一篇](robotics-driving.md)里的 DreamGen。

## 从 Genie 到 Genie 3，以及同期的其他系统

| 日期 | 系统（机构） | 发布方公开的内容 |
| --- | --- | --- |
| 2024-08 | [GameNGen](https://arxiv.org/abs/2408.14837)（Google） | 把 Stable Diffusion v1.4 微调成实时 DOOM 引擎，一张 TPU 上 20 FPS |
| 2024-10-31 | [Oasis](https://oasis-model.github.io/)（Decart + Etched） | 类 Minecraft 的实时世界模型，DiT + ViT VAE，Diffusion Forcing；开放 5 亿参数权重 |
| 2024-12-04 | [Genie 2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)（Google DeepMind） | 以一张图为提示、可操控的 3D 世界，最长约一分钟保持一致；改成自回归 latent diffusion，动作控制用 classifier-free guidance |
| 2025-02-19 | [WHAM / Muse](https://www.nature.com/articles/s41586-025-08600-3)（Microsoft Research + Ninja Theory） | 1.6B，用 7 年以上的 Bleeding Edge 对局训练；Nature 论文 |
| 2025-04-04 | [WHAMM](https://www.microsoft.com/en-us/research/articles/whamm-real-time-world-modelling-of-interactive-environments/)（Microsoft Research） | 实时 Quake II 演示，640×360 下 10+ fps |
| 2025-08-05 | [Genie 3](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/)（Google DeepMind） | 文本提示，实时 720p、约 24 fps；几分钟内保持一致，能回忆约一分钟前的画面细节；可用文字触发世界事件（改天气、加物体或角色）；当时只对少量学者和创作者做研究预览 |
| 2025-11-13 | [SIMA 2](https://deepmind.google/blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/)（Google DeepMind） | 基于 Gemini 的智能体，在 Genie 3 生成的世界里测试，并在其中自我改进 |
| 2026-01-29 | Project Genie（Google DeepMind / Google Labs） | Genie 3、Nano Banana Pro 和 Gemini 组成的网页原型；只对美国 18 岁以上的 Google AI Ultra 订阅用户开放；每次生成 60 秒；Google 自己写明世界可能不遵循提示或物理、角色可能延迟、可触发事件尚未加入 |
| 2026-02-06 | [Waymo World Model](https://waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation)（Waymo） | 在 Genie 3 上做驾驶后训练，输出摄像头和 lidar；没有公布保真度指标 |
| 2026-05-19 | [Project Genie 扩展](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/project-genie-expands/)（Google） | 加入 Street View 落地（仅美国地点），并逐步向全球 Ultra 用户开放 |

Google DeepMind 的 Genie 3 博客（2025-08-05）说：每一帧都以不断变长的会话历史为条件自回归生成，一致性是“涌现”的，不靠 NeRF 或 Gaussian Splatting 这类显式 3D 表示。同一篇博客列出的局限有：智能体能直接执行的动作有限，多个独立智能体的交互难模拟，真实地点的地理不精确，文字常常糊掉（除非写在提示里），一次会话只能撑几分钟。

**截至 2026-10-15，我们在公开资料里没有找到 Genie 3 的论文或技术报告、API，也没有找到 Genie 4；它的架构、参数量和训练数据都没有公开。** 所以外界无法判断它的一致性靠长上下文、显式记忆还是别的机制，也不该拿 Genie 1 的结构去猜。

同一时期的其他玩家（帧率、分辨率均为发布方自报）：

- **Runway**：[GWM-1](https://runway.com/research/introducing-runway-gwm-1)（2025-12-11）是基于 Gen-4.5 的自回归实时世界模型家族（Worlds / Avatars / Robotics）；[GWM Worlds 2](https://runway.com/research/introducing-gwm-worlds-2)（2026-09-03，研究预览）自报实时 720p、24 fps，带 48 kHz 音频。
- **Odyssey**：[Odyssey-2](https://odyssey.systems/introducing-odyssey-2)（2025-10-27）是因果、可用文字引导的交互视频，20 fps；[Odyssey-2 Pro](https://odyssey.systems/the-gpt-2-moment-for-world-models)（2026-01-23）720p、22 FPS，开放 API；[Odyssey-3](https://odyssey.systems/introducing-odyssey-3)（2026-09-15）是带动作解码器的基础世界模型。
- **Skywork**：Matrix-Game 2.0（2025-08）开源 1.8B；Matrix-Game 3.0（2026-03/04）5B、带记忆，仓库为 MIT 许可；2026-10-15 [仓库](https://github.com/SkyworkAI/Matrix-Game)更新了 Matrix-Game 4.0（游戏引擎加因果视频渲染器），我们只看了仓库说明，没有读到论文。
- **Overworld**：Waypoint-1（2026-01-20）是 2.3B 的帧因果 rectified-flow 模型，自报在 RTX 5090 上 30–60 FPS；Waypoint-1 / 1.5 以 Apache-2.0 发布。
- **Tencent**：[HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay)（论文名 WorldPlay，即 HY-World 1.5，2025-12-16）是开源流式 video-diffusion 世界模型，720p、24 FPS；WorldPlay-5B 于 2026-01-06 发布。
- 另有 Decart 的 [Lucy 2](https://decart.ai/publications/lucy-2-introducing-sota-video-generation-in-realtime)（2026-01-26，公司自报 1080p、30 fps）和 Robbyant 的 [LingBot-World](https://arxiv.org/abs/2601.20540)（2026-01-28，开源，16 fps）。

## 看演示时先问这几件事

| 先问什么 | 为什么 | 本篇里的例子 |
| --- | --- | --- |
| 帧率和分辨率在什么硬件上测？ | 8+1 张 GPU 和一张消费级显卡不是一回事 | Matrix-Game 3.0 的 40 FPS 用 8+1 张 GPU；Waypoint-1.5-1B 写明 RTX 5090 |
| 按键到画面变化要多久？ | 帧率高不代表响应快 | Project Genie 自述角色可能延迟 |
| 能连续保持一致多久？ | 几秒好看不难，长会话会漂 | Genie 1 记 16 帧；Genie 3 自述几分钟；Project Genie 每次 60 秒 |
| 有哪些控制？ | 只有方向键，和能加物体、改天气差很多 | Genie 1 只有 8 个 latent action；Genie 3 有可提示事件，Project Genie 首发时还没加入 |
| 看不见的东西还在吗、还在变吗？ | 静态回忆和动态演化是两种能力 | WRBench、WorldMemBench 的发现见下一节 |
| 开放到什么程度？ | 只有视频的演示，别人无法复现和测试 | Genie 3 无论文、无 API；Matrix-Game、Waypoint、HY-WorldPlay 开放权重 |

## 局限：记忆、物理和漂移

**记忆。** 公开的数字在变长，但都不长：Genie 1 记 16 帧；Genie 2 约一分钟；Google DeepMind 称 Genie 3 能回忆约一分钟前的画面细节、几分钟内保持一致；Dreamer 4 的上下文是 9.6 秒。2026 年几个基准把问题问得更细：

- [WRBench](https://arxiv.org/abs/2606.20545)（Lu 等，USTC 与 X-Humanoid，2026-06 论文，23 个模型的 9,600 段视频）发现：镜头移开再看回来，模型往往把场景恢复成离开时的样子，而不是让正在发生的事接着推进。放到杯子例子里：转头时杯子正滑向桌边，转回来时它应该已经掉在地上，模型却常让它还停在桌边。
- [WorldMemBench](https://arxiv.org/abs/2610.19063)（Zhou 等，2026-10 论文）的结论是：静态回忆好，不等于能可靠地推演状态变化。[MBench](https://arxiv.org/abs/2606.00793)（2026-06 论文）也报告了系统性的记忆失败。

**控制和稳定性。** [WorldMark](https://arxiv.org/abs/2604.21686)（Xu 等，2026-04 论文）给 10 个模型用同一套 WASD 动作词表、500 个测试例，发现响应最快的模型往往最不稳定，画面最好看的模型方向准确率排最后。[WBench](https://arxiv.org/abs/2605.25874)（美团 LongCat 与复旦，2026-05 论文，289 例、20 个模型）发现没有一个模型在所有维度上都强。

**物理和漂移。** Google 在 2026-01-29 写明 Project Genie 的世界可能不遵循提示或物理。自回归生成的每一帧都以自己生成的过去为条件，小错误会被带下去；这和 [Model-based RL](../../05-post-training/deep-rl/model-based.md) 里误差随 rollout 累积是同一类问题。Dreamer 4 论文就认为 v-prediction 的高频输出误差会在长时间逐帧 rollout 中累积，因此改成直接预测干净的 latent（见 [Dreamer 一篇](dreamer.md)）。

<!-- paper-figure:gamengen-drift -->

看不见的东西怎么留住，有几种思路。下面的取舍是我们的分析，不是各家的结论：

- 只靠越来越长的历史：Genie 3 博客描述的就是以会话历史为条件、一致性涌现。不需要额外结构，但历史越长越贵，超出窗口就会忘。
- 把带位姿的历史帧当空间记忆：World Labs 的 [RTFM](https://www.worldlabs.ai/blog/rtfm)（2025-10-16，一张 H100）这样做。回到原位时能找回对应画面，但还在变化的物体要另外处理。
- 直接生成显式 3D：World Labs 的 [Marble](https://www.worldlabs.ai/blog/marble-world-model) 输出 Gaussian splats 和 mesh。静态几何不会忘，正在滑落的杯子却不在这套表示里。

**被智能体钻空子。** SIMA 2 在 Genie 3 的世界里自我改进。[Ha & Schmidhuber 2018](https://arxiv.org/abs/1803.10122) 早就发现，VizDoom 智能体在“梦”里学会了让怪物不开火的对抗策略，调高采样温度后梦才更难被利用（§4.5）。把交互世界模型当训练场，进步要回到真实环境里核对。

## 这条线接下来看什么

1. **Genie 3 会不会公开技术细节？** 截至 2026-10-15，检索不到 Genie 3 的论文、API 或开放权重，Project Genie 是面向订阅用户的原型。能改变判断的结果：一份写明一致性靠什么实现（长上下文还是显式记忆）、在什么硬件上实时的论文或技术报告。
2. **分钟级记忆能否覆盖看不见、还在变的东西？** WRBench（2026-06）和 WorldMemBench（2026-10）都报告了失败。能改变判断的结果：某个模型在这类测试的动态演化部分（不只是静态回忆）稳定通过，并被第三方复现。
3. **快和稳能否兼得？** WorldMark（2026-04）发现响应最快的往往最不稳。能改变判断的结果：在共享动作词表、写明硬件的条件下，同一个模型既快又稳。
4. **在生成世界里学到的本事，能否带回真实环境？** SIMA 2（2025-11）在 Genie 3 世界里自我改进；Waymo（2026-02）基于 Genie 3 的模型没有公布保真度指标。能改变判断的结果：公开数据把生成世界里的进步和真实任务表现放在一起对照。

接着读[机器人与汽车：模拟器、评测器与策略](robotics-driving.md)，或者跳到[它真的懂世界吗](do-they-understand.md)。
