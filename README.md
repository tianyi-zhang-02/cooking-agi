> 希望这些笔记能帮你少花点时间找资料，多留点时间做自己喜欢的事。

<div align="center" markdown="1">

<img src="site/static/og.png" alt="AGI 学习笔记" width="760">

<h1>AGI 学习笔记</h1>

学过的基础、读过的论文，还有准备找工作时的一些记录。<br>
中英文都有，部分内容配了可以动手玩的交互图。

[![在线阅读](https://img.shields.io/badge/在线阅读-cooking--agi-E8A672?style=flat-square)](https://tianyi-zhang-02.github.io/cooking-agi/)
[![build](https://github.com/tianyi-zhang-02/cooking-agi/actions/workflows/site.yml/badge.svg)](https://github.com/tianyi-zhang-02/cooking-agi/actions/workflows/site.yml)
[![bilingual](https://img.shields.io/badge/中文-English-8a8a8a?style=flat-square)](README.en.md)

[**在线阅读**](https://tianyi-zhang-02.github.io/cooking-agi/) · [English](README.en.md) · [幕后](contributors.md) · [参与贡献](CONTRIBUTING.md)

</div>

## 关于这份笔记

去年开始，我决定试试 industry。之前本科基本都在做科研，真正开始准备找工作，才发现有不少东西要补。准备了挺久，也走了一些弯路，所以想把学过的东西整理出来，方便自己回头看，也希望能帮到有类似需要的人。

这里主要记录 ML 和 language models 的基础、post-training、evaluation、论文阅读，以及我的面试准备方法和找工过程中的一些想法。**不是面经题库，不会放具体公司的面试题。** 很多章节还没写完，也难免有理解不到位的地方，我会继续补充。

## 适合谁看

- 在找 **ML 相关实习或 new-grad 岗位**，想整理一下基础和准备思路；
- 想转到 **ML / LLM 方向**，不知道先从哪些内容开始；
- 不一定在找工作，只是想了解模型怎么工作。

我的准备和面试经历主要集中在 **MLE 和 Research Scientist** 岗位，不太适合给 SDE 面试建议。AI infra 等我还不熟悉的方向，会整理一些自己觉得不错的资料，方便大家去看更有经验的人怎么讲。这里不是一份适合所有人的路线，按自己的需要参考就好。

## 可以看些什么

内容分成两个主要入口：[学习笔记](00-foundations/)讲原理、公式和教学演示；[工程实践](practice/)记录具体问题是怎么发现、实现和验证的。求职和论文笔记继续单独整理。

建议在[网站](https://tianyi-zhang-02.github.io/cooking-agi/)上阅读，可以切换中英文，也能直接操作交互图。想先随便看看，可以从 [Transformer 图解](https://tianyi-zhang-02.github.io/cooking-agi/00-foundations/transformer-lab.html)开始。

| 板块 | 内容 |
| --- | --- |
| [大模型基础](00-foundations/) | Tokenization、RNN / LSTM、Transformer、MoE 与模型家族 |
| [Post-training](05-post-training/) | SFT、RLHF、PPO 等方法的原理和适用场景 |
| [评估](07-evaluation/) | 评估设计、指标与 LLM-as-a-judge |
| [数据与检索](01-data-and-feedback/) | 数据、反馈信号与检索 |
| [系统与多模态](06-systems/) | 请求处理、问题排查与人工介入 |
| [Agents](10-agents/) | 常见结构、使用场景与模型选择 |
| [工程实践](practice/) | 从 NeMo RL 开源贡献看问题定位、实现、测试与取舍 |
| [面试准备](interview/) | ML 基础复习、代码练习和系统设计资料 |
| [求职](career/) | 准备过程、踩过的坑和心态变化 |
| [论文](papers/) | 读论文时的理解、疑问和实验思路 |

## 加入我们！

这份笔记正在从个人记录变成大家一起维护的知识库。你可以只改一小处，也可以认领一个喜欢的板块；不用一上来就承诺很多时间。[参与方式、板块分工和审核约定 →](community/README.md)

发现错误、哪段没看懂，或者有想看的内容，都欢迎[提 issue](https://github.com/tianyi-zhang-02/cooking-agi/issues)。有自己的笔记或更好的例子，也欢迎提 PR，我也想跟着大家多学一点。

怎么参与可以看[贡献指南](CONTRIBUTING.md)，参与过的朋友会出现在[幕后](contributors.md)。想在地图上留下一个位置，也可以[告诉我你所在的国家或地区](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=add-me-to-the-crew.yml)，不用提供具体地址。

这里只分享公开的技术知识和学习心得，请不要上传公司内部资料、未公开的面试内容或其他敏感信息。

## 本地运行

<details markdown="1">
<summary>展开查看安装、预览和检查命令</summary>

在仓库根目录运行：

```bash
pip install markdown pygments
python3 site/build.py          # 构建到 _site/
python3 site/build.py --serve  # 本地预览
```

提交前运行：

```bash
python3 site/paritycheck.py  # 检查中英两版结构
python3 site/leakcheck.py    # 检查敏感内容
```

</details>
