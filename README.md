# AGI 学习笔记
**AGI Study Notes · ML 基础、语言模型与工程实践**

[**在线阅读 →**](https://tianyi-zhang-02.github.io/cooking-agi/) · **中文** / [English](README.en.md) · [学习路线](00-foundations/study-guide.md) · [加入我们](CONTRIBUTING.md)

[![Build](https://github.com/tianyi-zhang-02/cooking-agi/actions/workflows/site.yml/badge.svg?branch=main)](https://github.com/tianyi-zhang-02/cooking-agi/actions/workflows/site.yml)

把分散的资料串起来：从一个问题出发，配上图解、小实验和复习卡，尽量讲清模型为什么这样设计、哪里有用、又有什么限制。

这份中英双语笔记由社区一起维护，仍在持续补充。希望你能少花些时间找资料，把精力留给真正想学的东西，也给生活留一点余地。

## 从哪里开始

| 你想做什么 | 阅读入口 |
| --- | --- |
| 系统补基础 | [学习与复习导读](00-foundations/study-guide.md) → [大模型学习地图](00-foundations/README.md) |
| 先动手看看 | [Transformer 交互图解](https://tianyi-zhang-02.github.io/cooking-agi/00-foundations/transformer-lab.html) · [CLIP 图文对齐](03-multimodal-learning/clip.md) |
| 准备 ML 实习或 new-grad | [技术面准备](interview/README.md) · [求职记录](career/README.md) |
| 读论文或补一篇笔记 | [论文](papers/README.md) · [写作规范](EDITORIAL.md) · [内容提案](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=proposal.yml) |

**建议在网站阅读。** 中英文切换、交互图和复习卡在那里可直接使用；这个仓库保存正文、实验代码与站点源码。

## 内容地图

学习笔记分为 **Quant Researcher** 和 **AI / ML Engineer** 两区。工程实践、求职和论文各有入口，不把所有内容塞进一条路线。

| 板块 | 内容与入口 |
| --- | --- |
| 概率与基础 | [概率、分布与条件概率](quant/probability/README.md) · [Token、向量与 Transformer](00-foundations/README.md) |
| 大模型与多模态 | [模型家族](00-foundations/model-families/README.md) · [MoE](00-foundations/moe/README.md) · [CLIP 与视觉语言模型](03-multimodal-learning/README.md) |
| 训练与评估 | [Post-training](05-post-training/README.md) · [评估与 LLM-as-a-judge](07-evaluation/README.md) |
| 数据、记忆与检索 | [反馈与目标](01-data-and-feedback/README.md) · [记忆](02-memory/README.md) · [检索](04-search/README.md) |
| Agents 与系统 | [Agent 结构](10-agents/README.md) · [可观测性与人工介入](06-systems/README.md) · [模型体验](08-model-experience/README.md) · [Personal AGI](09-personal-agi/README.md) |
| 从理解到实践 | [工程实践](practice/README.md)（整理中）· [面试准备](interview/README.md) · [求职](career/README.md) · [论文](papers/README.md) |

每篇尽量沿着 **问题 → 例子与图解 → 原理 → 验证 → 取舍 → 复习** 来讲。公式不是终点：能讲出前提、用小实验检查一次，才算往前走了一步。

## 加入我们！

不用先写一整章。一个纠错、一张更清楚的图，或者一句“这里没看懂”，都能帮到下一位读者。

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

这份笔记最初来自从学术研究转向 industry 的准备过程：学过不少东西，但把它们串起来、在面试或项目里讲清楚，仍然走过弯路。

目前更适合关注 ML、language models，以及 MLE / Research Scientist 方向的读者，也欢迎只是好奇 AI 的朋友。它不是万能求职路线，更不是具体面试题库；SDE、前后端和不熟悉的 infra 方向会优先引用更有经验的作者。

内容还不完整，解释也会继续改。哪里不对、还想看什么，欢迎直接告诉我们。
