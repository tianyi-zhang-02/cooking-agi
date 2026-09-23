<div align="center" markdown="1">

<img src="site/static/og.png" alt="AGI 学习笔记 · 从一条直线，到会读世界的系统" width="760">

<h1>AGI 学习笔记</h1>

**从基础开始，一起学懂 AI**

这里记录我学习 AI 时整理的知识、读过的论文，以及准备面试时的思考。<br>
有基础原理，也有可以动手试的交互图。欢迎一起看、一起讨论。

[![在线阅读](https://img.shields.io/badge/在线阅读-cooking--agi-E8A672?style=flat-square)](https://tianyi-zhang-02.github.io/cooking-agi/)
[![build](https://github.com/tianyi-zhang-02/cooking-agi/actions/workflows/site.yml/badge.svg)](https://github.com/tianyi-zhang-02/cooking-agi/actions/workflows/site.yml)
[![bilingual](https://img.shields.io/badge/中文-English-8a8a8a?style=flat-square)](README.en.md)

[**开始阅读**](https://tianyi-zhang-02.github.io/cooking-agi/) ·
[English](README.en.md) ·
[幕后](contributors.md) ·
[加入我们！](CONTRIBUTING.md)

</div>

---

## 为什么写这些

学 AI 的时候，我经常遇到一种情况：单看一个概念好像懂了，换到论文或代码里，又说不清它到底在做什么。于是我开始边学边记，试着把这些零散的知识连起来。

我比较关注 representation learning、LLM post-training 和 model / agent evaluation。读论文或做实验时，我常会问：模型从什么信号里学到了东西？一个分数变高了，实际表现是不是也更好了？

这里尽量先用例子把问题讲明白，再看公式和代码。既会介绍现在常用的方法，也会回头看看早期模型，理解一些设计是怎么来的。

这些都只是我目前的理解，有些章节还没写完，也难免有讲错的地方。我会继续补充，也欢迎你来挑错。

## 从哪里开始

| 想看什么 | 可以从这里开始 |
| --- | --- |
| 系统地学一遍 | [大模型基础](00-foundations/)，按板块往下读；[Transformer 交互图解](00-foundations/transformer-lab.md)里的图能直接拖 |
| 看懂不同模型为什么这样设计 | [模型家族精读](00-foundations/model-families/)：用同一组问题比较 Llama、Qwen、DeepSeek、Gemma |
| 准备面试 | [技术面](interview/)用来复习基础；[求职笔记](career/)记录我的准备方法和心态变化 |
| 读一篇论文 | [论文](papers/)：作者提出了什么结论、证据够不够、怎样检验 |
| 知道谁写的 | [幕后](contributors.md) |

## 里面有什么

| 板块 | 讲什么 |
| --- | --- |
| [大模型基础](00-foundations/) | 从线性模型到 Transformer：注意力、归一化、残差、MoE、looped transformer |
| [Post-training](05-post-training/) | SFT、RLHF、PPO 等方法怎么做，又各自适合什么情况 |
| [评估](07-evaluation/) | 怎么设计评估、理解指标，以及用 LLM-as-a-judge 辅助判断 |
| [数据与检索](01-data-and-feedback/) | 数据从哪儿来、反馈怎么收、检索怎么建 |
| [系统与多模态](06-systems/) | 一次请求经过哪些环节，怎么排查问题，什么时候需要人来确认 |
| [Agents](10-agents/) | 这个词的来历、几种结构、不同场景，以及模型怎么选 |
| [AI Infra](open-source/) | 在 NeMo RL 里做贡献：从具体改动出发，逐步看懂整套系统 |
| [求职](career/) | 心态、要准备什么、我自己的时间线和复盘 |
| [论文](papers/) | 记录读论文时的理解、疑问和实验思路 |

交互图（Transformer、KV cache、MoE 路由、PPO 裁剪、吃豆人迷宫、模型路由……）都在 [`site/static/tx-lab.js`](site/static/tx-lab.js) 里，手写 SVG，没用图表库。

## 站点怎么跑

想在本地打开网站，可以运行下面的命令。网站是静态的，用 Python 脚本生成：

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

## 加入我们！

如果你也在学这些，欢迎来聊聊！哪段看不懂、哪里写错了，或者有想了解的话题，都可以提 issue。有更好的例子或自己的学习笔记，也欢迎分享。

想直接修改内容，可以提 PR；具体怎么操作，见[贡献指南](CONTRIBUTING.md)。

[幕后](https://tianyi-zhang-02.github.io/cooking-agi/contributors.html)记录了参与过的朋友。想在地图上也留下一个位置，可以[告诉我们你所在的国家或地区](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=add-me-to-the-crew.yml)，不用提供具体地址。

这里分享公开的技术知识和学习心得，请不要上传公司内部资料、未公开的面试内容或其他敏感信息。
