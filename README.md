# AGI 学习笔记
**AGI Study Notes · ML 基础、语言模型与工程实践**

[**在线阅读 →**](https://tianyi-zhang-02.github.io/cooking-agi/) · **中文** / [English](README.en.md) · [学习导航](learn/README.md) · [加入我们](CONTRIBUTING.md)

[![Build](https://github.com/tianyi-zhang-02/cooking-agi/actions/workflows/site.yml/badge.svg?branch=main)](https://github.com/tianyi-zhang-02/cooking-agi/actions/workflows/site.yml)

学 AI 时，找资料往往就花掉不少时间。这里把读过的资料和自己的理解整理在一起，从具体问题讲起，配上图解和小实验，尽量说清楚模型怎么工作、为什么这样设计。

这份中英双语笔记由大家一起维护，还在慢慢补。希望能帮你少绕点路，准备起来更轻松些，也有时间做学习之外喜欢的事。

## 从哪里开始

| 你想做什么 | 阅读入口 |
| --- | --- |
| 系统补基础 | [学习与复习导读](00-foundations/study-guide.md) → [大模型学习地图](00-foundations/README.md) |
| 先动手看看 | [Transformer 交互图解](https://tianyi-zhang-02.github.io/cooking-agi/00-foundations/transformer-lab.html) · [CLIP 图文对齐](03-multimodal-learning/clip.md) |
| 准备 ML 实习或 new-grad | [面试准备](interview/README.md) · [求职记录](career/README.md) |
| 练系统设计或看项目实现 | [工程实践](practice/README.md)：设计题、推荐系统、RAG 与后训练 |
| 补一篇笔记 | [写作规范](EDITORIAL.md) · [内容提案](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=proposal.yml) |

**建议在网站阅读。** 可以随时切换中英文，也能直接操作交互图；这个仓库保存正文、实验代码与站点源码。

只想看某个方向，不必从头读。先看该板块的导读，再选你需要的章节。已经写到哪里、还在补什么，放在[内容覆盖与待办](learn/coverage.md)，不把待补的内容藏起来。

## 内容地图

网站按你想做的事分成四个板块：学原理、准备技术面、做项目，以及读求职经历。每篇文章有一个主要位置，相关内容互相链接，不用在几个目录里找重复的文章。

| 板块 | 内容与入口 |
| --- | --- |
| [基础与原理](learn/README.md) | 模型与多模态、训练与评估、推理与应用；从入门到进阶按需阅读 |
| [面试准备](interview/README.md) | [ML / LLM 基础问答](interview/basics/README.md) · [ML Coding](learn/ml-exercises/README.md) · [Python](interview/python.md) · [LeetCode 方法](interview/leetcode.md) |
| [工程实践](practice/README.md) | [系统设计](learn/system-design/README.md) · [推荐系统](practice/recommender-systems/README.md) · [RAG](practice/rag/README.md) · [后训练项目](practice/post-training/README.md) |
| [求职](career/README.md) | 实习和 new-grad 的经历、心态与准备节奏，不混入技术题解 |

笔记里会穿插小例子、推导和代码，帮你看清每一步发生了什么。不用一开始就吃透所有公式；先跟着一个例子走，再回来补不明白的地方。

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
python3 -m pip install markdown pygments numpy
python3 site/build.py --serve
```

打开 <http://localhost:8000>。仅构建时运行 `python3 site/build.py`，输出在 `_site/`。

<details markdown="1">
<summary>提交前的检查</summary>

```bash
python3 site/collaboration.py
python3 -m unittest discover -s site/tests
python3 site/leakcheck.py
python3 site/paritycheck.py --strict
python3 site/build.py
```

部分教学实验另外需要 PyTorch，依赖写在对应章节；未安装时，相关测试会明确跳过。收藏、最近阅读和阅读位置只保存在当前浏览器，不上传到服务器。

</details>

## 为什么整理这些

最开始整理这些，是因为我想从学术研究走向企业里的 ML 工作。学过的东西不少，但要把它们串起来，在面试和项目里讲清楚，还是走了不少弯路。

目前的内容更贴近机器学习、语言模型，以及 MLE / Research Scientist 岗位的准备，也欢迎只是想了解 AI 的朋友。这里的经验不一定适合所有人，也不会分享具体公司的面试原题。软件开发、前后端和 AI 基础设施等我不太熟悉的方向，会优先推荐更有经验的作者。

内容还不完整，解释也会继续改。哪里不对、还想看什么，欢迎直接告诉我们。
