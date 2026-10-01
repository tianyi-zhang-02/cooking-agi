# AGI 学习笔记
**AGI Study Notes · ML 基础、语言模型与工程实践**

[**在线阅读 →**](https://tianyi-zhang-02.github.io/cooking-agi/) · **中文** / [English](README.en.md) · [学习导航](learn/README.md) · [加入我们](CONTRIBUTING.md)

[![Build](https://github.com/tianyi-zhang-02/cooking-agi/actions/workflows/site.yml/badge.svg?branch=main)](https://github.com/tianyi-zhang-02/cooking-agi/actions/workflows/site.yml)

学 AI 时，找资料往往就花掉不少时间。这里把读过的资料和自己的理解整理在一起，从具体问题讲起，配上图解、小实验和复习卡，尽量说清楚模型怎么工作、为什么这样设计。

这份中英双语笔记由大家一起维护，还在慢慢补。希望能帮你少绕点路，准备起来更轻松些，也有时间做学习之外喜欢的事。

## 从哪里开始

| 你想做什么 | 阅读入口 |
| --- | --- |
| 系统补基础 | [学习与复习导读](00-foundations/study-guide.md) → [大模型学习地图](00-foundations/README.md) |
| 先动手看看 | [Transformer 交互图解](https://tianyi-zhang-02.github.io/cooking-agi/00-foundations/transformer-lab.html) · [CLIP 图文对齐](03-multimodal-learning/clip.md) |
| 准备 ML 实习或 new-grad | [代码与题解](interview/README.md) · [求职记录](career/README.md) |
| 读论文或补一篇笔记 | [论文](papers/README.md) · [写作规范](EDITORIAL.md) · [内容提案](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=proposal.yml) |

**建议在网站阅读。** 中英文切换、交互图和复习卡在那里可直接使用；这个仓库保存正文、实验代码与站点源码。

## 内容地图

学习笔记分成 **基础与模型、训练与应用、代码与题解、系统设计**。想看找工作的经历和心态，可以去“求职”；想读公开项目、了解实际实现，可以去“工程实践”。

| 板块 | 内容与入口 |
| --- | --- |
| 01 · 基础与模型 | [概率与统计](quant/probability/README.md) · [大模型基础](00-foundations/README.md) · [模型家族](00-foundations/model-families/README.md) |
| 02 · 训练与应用 | [Post-training](05-post-training/README.md) · [评估](07-evaluation/README.md) · [多模态](03-multimodal-learning/README.md) · [数据、记忆、检索与 Agents](learn/README.md) |
| 03 · 代码与题解 | [Python](interview/python.md) · [LeetCode 方法](interview/leetcode.md) · [ML 问答与手写](learn/ml-exercises/README.md) |
| 04 · 系统设计 | [推荐 Feed、RAG 和带记忆的助手](learn/system-design/README.md)：带约束的设计练习 |
| 工程实践 | [Twitter / X 推荐系统](practice/recommender-systems/README.md)：先看架构，再拆召回、排序、评估和工程实现 |
| 求职 | [经历、心态和准备节奏](career/README.md)：与技术题解分开 |
| 论文 | [论文笔记](papers/README.md)：回到原文看方法、证据与限制 |

每篇尽量按 **问题 → 例子与图解 → 原理 → 验证 → 取舍 → 复习** 来讲。不用一开始就吃透所有公式，先跟着例子走，再回来补不明白的地方。

## 加入我们！

不用等到能写一整章才参与。改一个错误、补一张图，或者告诉我们“这里没看懂”，都很有帮助。

- **提问或纠错**：[开 issue](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=note-feedback.yml)，附上页面和具体段落。
- **补充内容**：小改动直接提 PR；新文章或目录调整先开[内容提案](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=proposal.yml)，避免重复劳动。
- **参与审核**：看[板块分工与审核约定](community/README.md)；常规合并需要非作者的相关 CODEOWNER 审核和自动检查。
- **认识贡献者**：[幕后](contributors.md)记录一起参与的朋友；地图位置只收自愿提供的国家或地区。

[贡献指南](CONTRIBUTING.md) · [写作规范](EDITORIAL.md)

只分享公开知识、可复现实验和公开项目笔记。不上传公司内部资料、未公开的面试题、凭据或他人隐私。AI 可以协助写作，但需要核对来源、验证内容，并在 PR 中说明使用方式。

## 本地阅读与检查

静态站点使用 Python 构建，无需数据库。以下命令在仓库根目录运行，建议 Python 3.12：

```bash
python3 -m pip install markdown pygments
python3 site/build.py --serve
```

打开 <http://localhost:8000>。仅构建时运行 `python3 site/build.py`，输出在 `_site/`。

<details markdown="1">
<summary>提交前的检查</summary>

```bash
python3 site/collaboration.py
python3 -m unittest discover -s site/tests
python3 site/leakcheck.py
python3 site/paritycheck.py
python3 site/build.py
```

部分教学实验另外需要 NumPy 或 PyTorch，依赖写在对应章节。复习进度只保存在当前浏览器，不上传到服务器。

</details>

## 为什么整理这些

最开始整理这些，是因为我想从学术研究走向企业里的 ML 工作。学过的东西不少，但要把它们串起来，在面试和项目里讲清楚，还是走了不少弯路。

目前的内容更贴近机器学习、语言模型，以及 MLE / Research Scientist 岗位的准备，也欢迎只是想了解 AI 的朋友。这里的经验不一定适合所有人，也不会分享具体公司的面试原题。软件开发、前后端和 AI 基础设施等我不太熟悉的方向，会优先推荐更有经验的作者。

内容还不完整，解释也会继续改。哪里不对、还想看什么，欢迎直接告诉我们。
