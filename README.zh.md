# AGI 学习笔记

**先看懂，再跟着例子算一遍，最后动手试试。**

[**在线阅读 →**](https://tianyi-zhang-02.github.io/cooking-agi/index.zh.html) · [English](README.md) · [幕后](contributors.md) · [加入我们](CONTRIBUTING.md)

[![Build](https://github.com/tianyi-zhang-02/cooking-agi/actions/workflows/site.yml/badge.svg?branch=main)](https://github.com/tianyi-zhang-02/cooking-agi/actions/workflows/site.yml)

学 AI 时，资料不少，找起来、串起来却挺花时间。这里把 ML 和 LLM 的笔记整理成中英文两版，配上图解、算例和小实验。希望不管你是在补基础、做项目，还是准备面试，都能少花些时间到处翻资料。

第一次打开网站时默认英文。点顶部的 **中文 / EN**，直接切换同一篇笔记，不用再开菜单。中文里会附上常用英文术语，方便对照。浏览器会记住你的选择；直接打开某种语言的文章链接，不会被强行切到另一版。

## 从哪里开始

按你今天想做的事选就好，不用从头读到尾。

- **[基础与原理](learn/README.md)**：从模型组件讲到训练、评估与推理。刚接触语言模型，可以先看[学习导读](00-foundations/study-guide.md)。
- **[面试准备](interview/README.md)**：复习 ML / LLM 基础，练 ML Coding、Python，以及能举一反三的算法方法。
- **[工程实践](practice/README.md)**：看看推荐、RAG 和后训练项目怎么搭，聊数据、评估、checkpoint，以及方案怎么选。
- **[求职记录](career/README.md)**：实习和 new-grad 的准备、选择，也记录下次想换个做法的地方。

先跟着小例子理解大意；想深挖，再往下看推导，或者展开实现细节。图解里的主要信息直接可见，不用每一步都点一遍才知道在讲什么。

## 内容地图

基础部分有三条路线，对哪一块感兴趣，就从哪一块读：

- **模型怎么工作**：[Transformer 交互图解](https://tianyi-zhang-02.github.io/cooking-agi/00-foundations/transformer-lab.html) · [BERT](00-foundations/core/bert.md) · [CLIP 与多模态](03-multimodal-learning/clip.md)。
- **模型怎么学**：[训练基础](learn/pretraining/README.md) · [Deep RL](05-post-training/deep-rl/README.md) · [Post-training](05-post-training/README.md) · [LLM-as-a-Judge](07-evaluation/llm-as-a-judge/README.md)。
- **模型怎么用起来**：[推理](learn/inference/README.md) · [搜索与检索](04-search/README.md) · [Agents 与工具](10-agents/README.md)。

想动手练习，可以直接去 [ML Coding](learn/ml-exercises/README.md)、[Python](interview/python.md)、[算法方法](interview/leetcode.md)或[系统设计](learn/system-design/README.md)。

**还在继续补：**有些讲解和参考资料还需要核对。[内容覆盖与待办](learn/coverage.md)会区分“已经写了”和“已经审过”，不会把两者混为一谈。小例子跑通了，也不代表复现了完整模型训练或 benchmark。

## 加入我们！

发现错误，或者读到一句“这到底在说什么”，都可以[开个 issue](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=note-feedback.yml)，告诉我们是哪一页、哪里卡住了。补一个好懂的例子，和写一篇新文章一样欢迎。

小改动直接提 PR；想加新文章或调整目录，先开[内容提案](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=proposal.yml)，避免大家重复做。具体流程见[贡献指南](CONTRIBUTING.md)、[写作规范](EDITORIAL.md)和[板块审核人](community/README.md)。常规合并需要非作者的相关 CODEOWNER 审核，以及自动检查通过。

请使用公开资料，用自己的话讲清楚。不上传公司内部资料、未公开的面试题、凭据或他人隐私。可以用 AI 辅助，但要自己核对来源和例子，并在 PR 里说明。一起参与的朋友会出现在[幕后](contributors.md)；是否为地图提供国家或地区，由你自己决定。

## 本地阅读与检查

这个仓库包含笔记、例子和静态站点源码，不需要数据库。在仓库根目录运行，建议使用 Python 3.12：

```bash
python3 -m pip install markdown pygments numpy
python3 site/build.py --serve
```

打开 <http://localhost:8000>。只想构建、不启动预览，可以运行 `python3 site/build.py`，结果在 `_site/`。

<details markdown="1">
<summary>提交前的检查</summary>

```bash
python3 site/collaboration.py
python3 -m unittest discover -s site/tests
python3 site/leakcheck.py
python3 site/paritycheck.py --strict
python3 site/build.py
```

部分实验还需要 PyTorch，依赖写在对应章节；没有安装时，相关测试会明确跳过。收藏、最近阅读和阅读位置只保存在这个浏览器里，不会上传。

</details>

## 为什么整理这些

最开始是我想从学术研究走向企业里的 ML 工作。准备时，资料散在论文、课程、文档和一堆收藏里。把它们放在一起，才慢慢看清哪些地方真的懂了，哪些还得补。

所以目前的内容比较贴近 MLE 和研究岗位，尤其是语言模型方向。它不是适合所有人的求职路线，也不会收集具体公司的面试原题。SDE、前后端，以及我们不熟悉的基础设施问题，会优先推荐更懂这些方向的作者。

希望你能少翻一会儿资料，学完之后，也有时间做点别的喜欢的事。
