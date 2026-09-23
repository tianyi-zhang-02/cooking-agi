<div align="center">

<img src="site/static/og.png" alt="AGI 学习笔记 · 从一条直线，到会读世界的系统" width="760">

<h1>AGI 学习笔记</h1>

**从一条直线，到会读世界的系统**

一份公开的中英双语学习笔记：模型从什么数据里学、怎么记住和检索、如何使用工具，<br>
以及我们凭什么相信一次指标上涨真的代表系统变好了。

[![在线阅读](https://img.shields.io/badge/在线阅读-cooking--agi-E8A672?style=flat-square)](https://tianyi-zhang-02.github.io/cooking-agi/)
[![build](https://github.com/tianyi-zhang-02/cooking-agi/actions/workflows/site.yml/badge.svg)](https://github.com/tianyi-zhang-02/cooking-agi/actions/workflows/site.yml)
[![bilingual](https://img.shields.io/badge/中文-English-8a8a8a?style=flat-square)](README.en.md)

[**开始阅读**](https://tianyi-zhang-02.github.io/cooking-agi/) ·
[English](README.en.md) ·
[幕后船员](contributors.md) ·
[一起写](CONTRIBUTING.md)

</div>

---

## 为什么写这些

很多概念第一次看懂并不难，难的是把它们连起来：模型从什么数据里学，怎样记住和检索信息，如何使用工具，我们又凭什么相信一次指标上涨真的代表系统变好了。

我关心的不只是"怎样训练一个更大的模型"，而是数据、表征、记忆、搜索、反馈、训练和评估怎样一起工作，最终让 AI **更准确地理解需求、找到证据，并在得到反馈后修正自己**。

每篇尽量按同一个顺序写：先说问题从哪儿来，用最小的例子建立直觉，再进公式、实现和工程取舍，最后留下它的假设、边界，和还没想明白的地方。整体从今天的 LLM 系统往回看，历史只在它能解释当前设计的时候出现。

这些是我现阶段的理解，会一直改。

## 从哪里开始

| 我想… | 去哪儿 |
| --- | --- |
| 系统地学一遍 | [大模型基础](00-foundations/)，按板块往下读；[Transformer 交互图解](00-foundations/transformer-lab.md)里的图能直接拖 |
| 看懂不同模型为什么这样设计 | [模型家族精读](00-foundations/model-families/)：用同一组问题比较 Llama、Qwen、DeepSeek、Gemma |
| 准备面试 | [技术面](interview/)面试前快速复习；[求职笔记](career/)是我自己找 MLE / RE 的记录 |
| 读一篇论文 | [精读](papers/)：它 claim 了什么、证据够不够、什么能推翻它 |
| 知道谁写的 | [幕后船员](contributors.md) |

## 里面有什么

| 板块 | 讲什么 |
| --- | --- |
| [大模型基础](00-foundations/) | 从线性模型到 Transformer：注意力、归一化、残差、MoE、looped transformer |
| [Post-training](05-post-training/) | SFT、RLHF、PPO 和它的近亲，以及对齐到底在对齐什么 |
| [评估](07-evaluation/) | 指标稳不稳、LLM-as-a-judge 怎么用才不骗自己 |
| [数据与检索](01-data-and-feedback/) | 数据从哪儿来、反馈怎么收、检索怎么建 |
| [系统与多模态](06-systems/) | 一次请求真正经过哪些环节，可观测性与人在回路 |
| [Agents](10-agents/) | 这个词的来历、几种结构、不同场景，以及模型怎么选 |
| [AI Infra](open-source/) | 在 NeMo RL 里做贡献：零散 PR 怎样长成对系统的理解 |
| [求职](career/) | 心态、要准备什么、我自己的时间线和复盘 |
| [精读](papers/) | 一篇一篇拆论文 |

交互图（Transformer、KV cache、MoE 路由、PPO 裁剪、吃豆人迷宫、模型路由……）都在 [`site/static/tx-lab.js`](site/static/tx-lab.js) 里，手写 SVG，没用图表库。

## 站点怎么跑

全静态，构建脚本就一个 Python 文件：

```bash
pip install markdown pygments
python3 site/build.py            # 构建到 _site/
python3 site/build.py --serve    # 本地预览
```

提交前跑两个检查：

```bash
python3 site/paritycheck.py      # 中英两版结构是否一致
python3 site/leakcheck.py        # 不该公开的东西有没有混进来
```

## 一起写

哪里讲错了、没讲清楚，或者缺一个更好的例子，都欢迎指出——改一个错字也算贡献。这个仓库就是站点源码，怎么参与见 [CONTRIBUTING.md](CONTRIBUTING.md)，那儿也写了几条规矩：只聊技术不聊政治、别把这儿当广告位、AI 起草可以但你自己得先读一遍。

想让[世界地图](https://tianyi-zhang-02.github.io/cooking-agi/contributors.html)上多亮一块，开个 [issue](../../issues/new?template=add-me-to-the-crew.yml) 告诉我你在哪儿。

这里只放基础原理、推导、公开论文、可复现实验和开源项目笔记；来自具体公司或招聘过程的内容不会出现在这儿。
