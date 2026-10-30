# AGI 大锅烩

**中英双语的 ML / LLM 笔记：算例、分步图解、代码练习和面试准备。**

<!-- repository-only -->

<span id="si-学习笔记"></span>
<span id="agi-学习笔记"></span>

<p align="center"><a href="https://tianyi-zhang-02.github.io/cooking-agi/index.zh.html"><img src=".github/media/hero.zh.svg" width="100%" alt="AGI 大锅烩一览：分步动画、论文原图、前沿观察、中英双语、技术编年史与全文搜索"></a></p>

<p align="center"><a href="https://tianyi-zhang-02.github.io/cooking-agi/index.zh.html"><b>在线阅读 →</b></a> &nbsp;·&nbsp; <a href="README.md">English</a> &nbsp;·&nbsp; <a href="#contributors">贡献者</a> &nbsp;·&nbsp; <a href="CONTRIBUTING.md">加入我们</a></p>

<!-- /repository-only -->

每个主题都围绕一个能跟着算、能跑、能回看的例子展开，旁边配上论文原图和出处。可以从零学起，也可以面试前集中复习，或者看看自己不熟的方向。**免费开放，也会一直免费。**

## 从哪里开始

| 你想做什么 | 从这里开始 |
| --- | --- |
| 从头学 ML | [起步路线](interview/speedrun-start.md) |
| 按计划学习 | [按计划学习](interview/study-plan.md) |
| 快面试了 | [快速复习](interview/speedrun.md) |
| 看看哪里不熟 | [代码练习](interview/coding-checks.md) · [岗位指南](interview/ai-roles-prep.md) |
| 对某个方向好奇 | [专题入口](learn/choose-your-path.md#topics) · [编年史](learn/technology-timeline.md) |

网站分五个板块，不用按顺序读。

- **[基础与原理](learn/README.md)**：模型怎么工作、怎么训练、怎么评估。
- **[面试准备](interview/README.md)**：ML / LLM 复习、ML Coding、Python 和常见算法方法。
- **[工程实践](practice/README.md)**：推荐、RAG 和后训练系统，从数据到上线。
- **[编年史](learn/technology-timeline.md)**：电力、芯片、网络、软件与 AI 的发展，附原始资料。
- **[求职记录](career/README.md)**：实习和 new-grad 求职中试过的做法。

<span id="interview-coach"></span>

### 在 AI 助手里练面试

Interview coach skill 在 Claude Code 或 Codex 里对照这些笔记模拟面试：一次问一道题，根据你的回答追问，并在本地记下需要复习的地方。

```bash
npx skills add https://github.com/tianyi-zhang-02/cooking-agi --skill cooking-agi-interview
```

用法见 [skills/README.md](skills/README.md)。由读者 [@Juggernaut0825](https://github.com/Juggernaut0825) 贡献。

## 内容一览

<!-- repository-only -->

<a href="https://tianyi-zhang-02.github.io/cooking-agi/learn/coverage.html"><picture><source media="(prefers-color-scheme: dark)" srcset="https://tianyi-zhang-02.github.io/cooking-agi/static/readme-overview-zh-dark.svg"><img src="https://tianyi-zhang-02.github.io/cooking-agi/static/readme-overview-zh-light.svg" width="100%" alt="笔记的复核与图解数量、按板块列出的主题，以及现在和接下来在做的事；随网站每次部署更新"></picture></a>

<!-- /repository-only -->

- **模型怎么工作**：[Transformer 交互图解](https://tianyi-zhang-02.github.io/cooking-agi/00-foundations/transformer-lab.html) · [BERT](00-foundations/core/bert.md) · [CLIP 与多模态](03-multimodal-learning/clip.md)
- **模型怎么学**：[训练基础](learn/pretraining/README.md) · [Deep RL](05-post-training/deep-rl/README.md) · [Post-training](05-post-training/README.md) · [LLM-as-a-Judge](07-evaluation/llm-as-a-judge/README.md)
- **模型怎么用起来**：[推理](learn/inference/README.md) · [搜索与检索](04-search/README.md) · [Agents 与工具](10-agents/README.md)
- **正在变化的方向**：[前沿观察](11-frontier/README.md) · [AI 基础设施](12-infra/README.md) · [感知、机器人与自动驾驶](13-embodied/README.md)
- **动手练习**：[ML Coding](learn/ml-exercises/README.md) · [Python](interview/python.md) · [算法方法](interview/leetcode.md) · [系统设计](learn/system-design/README.md)

每个板块写了什么、哪些已经审过，见[内容覆盖](learn/coverage.md)；接下来做什么，见[后续计划](community/roadmap.md)；网站怎样更新，见[更新与阅读记录](community/updates.md)。

<span id="contributors"></span>

## 贡献者与审稿人（Contributors / Reviewers）

<!-- repository-only -->

<picture><source media="(prefers-color-scheme: dark)" srcset="https://tianyi-zhang-02.github.io/cooking-agi/static/readme-people-zh-dark.svg"><img src="https://tianyi-zhang-02.github.io/cooking-agi/static/readme-people-zh-light.svg" width="100%" alt="按提交数排列的贡献者、各领域审稿人和读者的下一站；每天更新"></picture>

<!-- /repository-only -->

提交数量不能代表贡献的全部，提问、审阅和纠错同样重要。[补充署名](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=add-me-to-the-crew.yml) · [分享下一站](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=next-stop.yml)

## 加入我们！

哪里没讲明白、哪条信息过时了，或者想看一个还没有的主题，都可以告诉我们。不用先找到答案，才能提问题。

- **提反馈**：开一个 [issue](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=note-feedback.yml)，附上相关段落。每篇文章末尾也有反馈入口。
- **一起改**：小改动直接提 PR；想写新文章，先开[内容提案](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=proposal.yml)。
- **分享给朋友**：如果这些笔记帮到了你，欢迎把链接发给朋友。

这些是一边学、一边改的笔记，不是官方课程。我们分享公开知识和准备方法，不收录针对某家公司的面试题。笔记采用 CC BY 4.0，代码采用 MIT，详见[许可条款](LICENSE.md)。

<details markdown="1">
<summary>修改、审核与隐私说明</summary>

具体流程见[贡献指南](CONTRIBUTING.md)、[写作规范](EDITORIAL.md)和[板块审核人](community/README.md)。社区贡献的 PR 需要非作者的相关 CODEOWNER 审核，并通过自动检查。请使用公开资料，用自己的话解释；AI 辅助的部分需要说明并人工核对。Issue 是公开的，请不要附私人进度备份、公司资料或未脱敏的日志。

</details>

## 本地阅读与检查

不需要数据库。在仓库根目录运行，建议使用 Python 3.12：

```bash
python3 -m pip install markdown pygments numpy
python3 site/build.py --serve
```

然后打开 <http://localhost:8000>。只运行 `python3 site/build.py` 会把网站生成到 `_site/`。

<details markdown="1">
<summary>提交前的检查</summary>

```bash
python3 site/collaboration.py
python3 -m unittest discover -s site/tests
python3 site/leakcheck.py
python3 site/paritycheck.py --strict
python3 site/build.py
```

部分实验还需要 PyTorch；没有安装时，相关测试会跳过。

</details>

## 为什么整理这些

刚开始准备找工时，我的资料散在课程、论文、文档和一堆收藏里；论文里看着眼熟的东西，真要自己讲、自己写，又不太会了。整理笔记帮我发现了这些缺口，公开出来，也能让别人帮忙指出剩下的。

希望开始学一个方向，不必先有相关背景，也不必刚好认识能给你一份资料清单的人。能少花一点东翻西找的时间，学完还有空做点自己喜欢的事，我就觉得这个站值得继续做下去。

<span id="changelog"></span>

## 更新记录

<details markdown="1">
<summary>主分支上的每次改动，按日期</summary>

| 日期 | 大更新 | 小更新 |
| --- | --- | --- |
| 2026-10-29 | 搜索和页面提速：索引按语言拆开，搜索结果直接跳到匹配的小节，文件内容没变就继续用缓存（[#128](https://github.com/tianyi-zhang-02/cooking-agi/pull/128)）· README 卡片合成一张内容一览，路线图和更新说明按新的优先级重写（[#127](https://github.com/tianyi-zhang-02/cooking-agi/pull/127)） | — |
| 2026-10-26 | — | 后训练笔记的中英文润色 |
| 2026-10-24 | 扩散语言模型和自回归模型的笔记补充算例 | 中英通读后的两处小修正（[#126](https://github.com/tianyi-zhang-02/cooking-agi/pull/126)） |
| 2026-10-22 | — | 手机上的导航和「我的阅读」更好用 · build 恢复正常：PyTorch 例子只在装了 PyTorch 的环境里运行（[#125](https://github.com/tianyi-zhang-02/cooking-agi/pull/125)） |
| 2026-10-21 | 全部 323 篇笔记中英文逐篇审查：改正错误，补上算例（[#119](https://github.com/tianyi-zhang-02/cooking-agi/pull/119)）· 润色中英文讲解，订正技术细节（[#124](https://github.com/tianyi-zhang-02/cooking-agi/pull/124)） | 去掉草稿标记（[#120](https://github.com/tianyi-zhang-02/cooking-agi/pull/120)）和难度标签（[#121](https://github.com/tianyi-zhang-02/cooking-agi/pull/121)）· Decoder-only 加上 prefill 与 decode 的分步动画（[#122](https://github.com/tianyi-zhang-02/cooking-agi/pull/122)）· 世界树的生长动画去掉逐笔模糊（[#123](https://github.com/tianyi-zhang-02/cooking-agi/pull/123)） |
| 2026-10-20 | — | 编年史：连上 Genie 3，补完缺口复查剩下的候选（[#117](https://github.com/tianyi-zhang-02/cooking-agi/pull/117) [#118](https://github.com/tianyi-zhang-02/cooking-agi/pull/118)） |
| 2026-10-19 | 编年史：每个节点一个立体小物件，新增星图视图和有出处的连线（[#113](https://github.com/tianyi-zhang-02/cooking-agi/pull/113)）；补上 24 个关键工作（[#115](https://github.com/tianyi-zhang-02/cooking-agi/pull/115)）和 15 个被低估的基础工作（[#116](https://github.com/tianyi-zhang-02/cooking-agi/pull/116)） | 夜空更省资源，世界树做了缓存，页面更快（[#114](https://github.com/tianyi-zhang-02/cooking-agi/pull/114)） |
| 2026-10-17 | 面试练习 skill：在 Claude Code、Codex 里对照笔记模拟面试，笔记和题目来自网站新开的只读数据接口（[#103](https://github.com/tianyi-zhang-02/cooking-agi/pull/103)，读者 @Juggernaut0825 贡献）· README 改版：首页动图，内容一览、路线图和贡献者卡片随网站更新（[#104](https://github.com/tianyi-zhang-02/cooking-agi/pull/104) [#105](https://github.com/tianyi-zhang-02/cooking-agi/pull/105)）· README 改成站点的夜空风格，贡献者做成剧组卡片（[#109](https://github.com/tianyi-zhang-02/cooking-agi/pull/109)）· 编年史补上 2024–2026 年的节点（[#108](https://github.com/tianyi-zhang-02/cooking-agi/pull/108)） | 贡献者页面从网站移到 GitHub 首页（[#104](https://github.com/tianyi-zhang-02/cooking-agi/pull/104)）· 卡片图标与折叠（[#106](https://github.com/tianyi-zhang-02/cooking-agi/pull/106)）· 贡献者卡片一人一条（[#107](https://github.com/tianyi-zhang-02/cooking-agi/pull/107)）· 首页底部加上下一页链接（[#110](https://github.com/tianyi-zhang-02/cooking-agi/pull/110)）· 去掉每篇的阅读、收藏计数和学习小伙伴的实时人数（[#111](https://github.com/tianyi-zhang-02/cooking-agi/pull/111) [#112](https://github.com/tianyi-zhang-02/cooking-agi/pull/112)） |
| 2026-10-16 | 全部笔记逐篇审查：改正错误，面试页不再写具体公司的招聘流程（[#99](https://github.com/tianyi-zhang-02/cooking-agi/pull/99)）；事实更新到 2026 年 10 月（[#100](https://github.com/tianyi-zhang-02/cooking-agi/pull/100)）；重复的解释合到一处，补上到新板块的链接（[#101](https://github.com/tianyi-zhang-02/cooking-agi/pull/101)）· Scaling、扩散语言模型和量化三组草稿（[#102](https://github.com/tianyi-zhang-02/cooking-agi/pull/102)） | 删除未发布的 quant 笔记（[#98](https://github.com/tianyi-zhang-02/cooking-agi/pull/98)） |
| 2026-10-15 | 前沿观察、AI 基础设施和具身智能三块，配论文原图和 26 张分步动画（[#97](https://github.com/tianyi-zhang-02/cooking-agi/pull/97)） | — |
| 2026-10-14 | LeetCode 笔记加上分步动画讲解（[#78](https://github.com/tianyi-zhang-02/cooking-agi/pull/78)）· 四个章节加上分步图解和论文原图（[#83](https://github.com/tianyi-zhang-02/cooking-agi/pull/83)）· 全部笔记的中英文通读一遍，逐句核对事实、算例和表达（[#84](https://github.com/tianyi-zhang-02/cooking-agi/pull/84) [#85](https://github.com/tianyi-zhang-02/cooking-agi/pull/85) [#87](https://github.com/tianyi-zhang-02/cooking-agi/pull/87) [#88](https://github.com/tianyi-zhang-02/cooking-agi/pull/88) [#89](https://github.com/tianyi-zhang-02/cooking-agi/pull/89) [#90](https://github.com/tianyi-zhang-02/cooking-agi/pull/90)）· 训练与系统的 16 篇笔记加上分步图解（[#94](https://github.com/tianyi-zhang-02/cooking-agi/pull/94)） | 收藏、阅读记录和备份合并到「我的阅读」，去掉学习台，练习移到「按计划学习」（[#81](https://github.com/tianyi-zhang-02/cooking-agi/pull/81)）· README 宣传动图换成新版（[#79](https://github.com/tianyi-zhang-02/cooking-agi/pull/79)）· README 加上路线图、主题覆盖和更新记录（[#82](https://github.com/tianyi-zhang-02/cooking-agi/pull/82)）· 中文段落里的换行不再显示成空格（[#86](https://github.com/tianyi-zhang-02/cooking-agi/pull/86)）· 复核清单与更新记录（[#91](https://github.com/tianyi-zhang-02/cooking-agi/pull/91)）· 面试准备首页改成普通目录（[#93](https://github.com/tianyi-zhang-02/cooking-agi/pull/93)）· 私有仓库归档后，泄漏检查成为唯一一道闸门（[#92](https://github.com/tianyi-zhang-02/cooking-agi/pull/92)）· 门控注意力的例子改用不相加为 1 的门（[#95](https://github.com/tianyi-zhang-02/cooking-agi/pull/95)） |
| 2026-10-13 | 编年史：技术世界树，143 个有来源的节点，之后改成竖向年表（[#65](https://github.com/tianyi-zhang-02/cooking-agi/pull/65) [#71](https://github.com/tianyi-zhang-02/cooking-agi/pull/71)）· 学习台与阅读记录（[#64](https://github.com/tianyi-zhang-02/cooking-agi/pull/64) [#67](https://github.com/tianyi-zhang-02/cooking-agi/pull/67)） | 阅读面板：今天、本周、累计（[#66](https://github.com/tianyi-zhang-02/cooking-agi/pull/66) [#70](https://github.com/tianyi-zhang-02/cooking-agi/pull/70)）· 每页一键反馈（[#72](https://github.com/tianyi-zhang-02/cooking-agi/pull/72)）· 安全加固（[#73](https://github.com/tianyi-zhang-02/cooking-agi/pull/73)）· 学习小伙伴（[#74](https://github.com/tianyi-zhang-02/cooking-agi/pull/74)）· 匿名计数（[#75](https://github.com/tianyi-zhang-02/cooking-agi/pull/75) [#76](https://github.com/tianyi-zhang-02/cooking-agi/pull/76)）· vLLM / SGLang 对照（[#68](https://github.com/tianyi-zhang-02/cooking-agi/pull/68)）· README 动图（[#69](https://github.com/tianyi-zhang-02/cooking-agi/pull/69)） |
| 2026-10-11 | 双语入口、README 与学习路线；工程练习和更多算例（[#61](https://github.com/tianyi-zhang-02/cooking-agi/pull/61) [#62](https://github.com/tianyi-zhang-02/cooking-agi/pull/62)） | 训练、推理、SigLIP 与 ML 系统指南（[#63](https://github.com/tianyi-zhang-02/cooking-agi/pull/63)） |
| 2026-10-09 | 双语学习路线全面整理，例子逐个核对（[#60](https://github.com/tianyi-zhang-02/cooking-agi/pull/60)） | — |
| 2026-10-05 | Deep RL 双语课程与交互实验（[#58](https://github.com/tianyi-zhang-02/cooking-agi/pull/58)） | 求职笔记改得更自然（[#59](https://github.com/tianyi-zhang-02/cooking-agi/pull/59)） |
| 2026-10-03 | 数学与 Quant 复习：证明和算例（[#56](https://github.com/tianyi-zhang-02/cooking-agi/pull/56)，后来下架） | 讨论区下线，回到学习笔记（[#57](https://github.com/tianyi-zhang-02/cooking-agi/pull/57)） |
| 2026-10-02 | LLM-as-a-Judge 双语课程与交互评估（[#55](https://github.com/tianyi-zhang-02/cooking-agi/pull/55)） | — |
| 2026-10-01 | 导航、阅读工具与社区页（[#45](https://github.com/tianyi-zhang-02/cooking-agi/pull/45)） | 讨论区上线与排版（[#53](https://github.com/tianyi-zhang-02/cooking-agi/pull/53) [#54](https://github.com/tianyi-zhang-02/cooking-agi/pull/54)） |
| 2026-09-30 | 双语学习路线与章节自测（[#42](https://github.com/tianyi-zhang-02/cooking-agi/pull/42)） | CLIP / VLM 笔记与双语 README（[#44](https://github.com/tianyi-zhang-02/cooking-agi/pull/44)）· 刷题常用的 Python（[#40](https://github.com/tianyi-zhang-02/cooking-agi/pull/40)） |
| 2026-09-25 | — | 移除 AI Infra 板块；补改跳过的笔记，增加协作审稿人（[#37](https://github.com/tianyi-zhang-02/cooking-agi/pull/37) [#38](https://github.com/tianyi-zhang-02/cooking-agi/pull/38)） |
| 2026-09-24 | Quant 专区：概率系列（[#36](https://github.com/tianyi-zhang-02/cooking-agi/pull/36)，后来下架） | — |
| 2026-09-23 | 精读、Agents、模型选择、求职与 AI Infra（[#33](https://github.com/tianyi-zhang-02/cooking-agi/pull/33) [#34](https://github.com/tianyi-zhang-02/cooking-agi/pull/34)） | 贡献者页与致谢（[#30](https://github.com/tianyi-zhang-02/cooking-agi/pull/30) [#31](https://github.com/tianyi-zhang-02/cooking-agi/pull/31) [#32](https://github.com/tianyi-zhang-02/cooking-agi/pull/32)）· ML Systems 推荐阅读（[#35](https://github.com/tianyi-zhang-02/cooking-agi/pull/35)） |
| 2026-09-22 | RLHF 九篇系列；MoE、Looped Transformer 与 Agents 板块，七个交互图（[#24](https://github.com/tianyi-zhang-02/cooking-agi/pull/24) [#25](https://github.com/tianyi-zhang-02/cooking-agi/pull/25)） | 每页只用一种语言（[#18](https://github.com/tianyi-zhang-02/cooking-agi/pull/18)）· 夜空设计与首页（[#19](https://github.com/tianyi-zhang-02/cooking-agi/pull/19) [#20](https://github.com/tianyi-zhang-02/cooking-agi/pull/20) [#21](https://github.com/tianyi-zhang-02/cooking-agi/pull/21) [#22](https://github.com/tianyi-zhang-02/cooking-agi/pull/22)）· 评估长文拆短（[#23](https://github.com/tianyi-zhang-02/cooking-agi/pull/23)）· 求职笔记（[#26](https://github.com/tianyi-zhang-02/cooking-agi/pull/26) [#27](https://github.com/tianyi-zhang-02/cooking-agi/pull/27) [#28](https://github.com/tianyi-zhang-02/cooking-agi/pull/28) [#29](https://github.com/tianyi-zhang-02/cooking-agi/pull/29)） |
| 2026-09-21 | 网站重做：求职路线、知识板块与快速复习（[#17](https://github.com/tianyi-zhang-02/cooking-agi/pull/17)） | — |
| 2026-08-27 | — | 面试基础题、白板手撕清单和参考实现（[#15](https://github.com/tianyi-zhang-02/cooking-agi/pull/15) [#16](https://github.com/tianyi-zhang-02/cooking-agi/pull/16)） |
| 2026-08-26 | 后训练系列：四段递进，从 PPO 讲到之后的算法（[#11](https://github.com/tianyi-zhang-02/cooking-agi/pull/11) [#13](https://github.com/tianyi-zhang-02/cooking-agi/pull/13) [#14](https://github.com/tianyi-zhang-02/cooking-agi/pull/14)） | 补一条中文导读（[#10](https://github.com/tianyi-zhang-02/cooking-agi/pull/10)） |
| 2026-08-24 | — | 补回解码与 RLHF 流程（[#8](https://github.com/tianyi-zhang-02/cooking-agi/pull/8)） |
| 2026-08-14 | — | 多头注意力、归一化与残差连接（[#7](https://github.com/tianyi-zhang-02/cooking-agi/pull/7)） |
| 2026-08-12 | 基础章节、阅读顺序和第一版网站（[#3](https://github.com/tianyi-zhang-02/cooking-agi/pull/3)） | 中文润色、侧栏分组（[#4](https://github.com/tianyi-zhang-02/cooking-agi/pull/4)）· 架构图与入门 / 进阶分层（[#6](https://github.com/tianyi-zhang-02/cooking-agi/pull/6)） |
| 2026-08-10 | 第一版 AI 基础设施学习指南（[#1](https://github.com/tianyi-zhang-02/cooking-agi/pull/1)） | 重画图解（[#2](https://github.com/tianyi-zhang-02/cooking-agi/pull/2)） |

</details>
