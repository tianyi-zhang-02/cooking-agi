# 按目标找路线

**中文** · [English](choose-your-path.en.md)

按你现在的目标找资料：基础入门、复习面试、读论文、动手做项目、补一个专题，或者安排求职准备。每条路线都列了起点和建议练习。

<nav class="study-route goal-route" aria-label="按目标选择阅读路线">
<a href="#getting-started"><small>先建立理解</small><strong>我刚开始学 ML / LLM</strong><span>从一个预测问题，到 Transformer</span></a>
<a href="#interviews"><small>准备技术面</small><strong>我需要复习和练代码</strong><span>基础问答、ML Coding 与算法</span></a>
<a href="#research"><small>读懂研究</small><strong>我想看懂论文和实验</strong><span>机制、比较条件、复现与局限</span></a>
<a href="#projects"><small>动手做东西</small><strong>我想搭一个能用的项目</strong><span>先跑通，再讨论设计与取舍</span></a>
<a href="#topics"><small>只看感兴趣的</small><strong>我只想补一个专题</strong><span>后训练、评估、RAG、多模态、推理……</span></a>
<a href="#applications"><small>安排求职准备</small><strong>我想理一理准备的顺序</strong><span>岗位方向、项目追问和面试复盘</span></a>
</nav>

已经熟悉的部分可以跳过。这些笔记主要覆盖 ML / LLM 和 MLE / research 相关内容，其他岗位需要另补对应领域的知识。

## 刚开始学：先弄懂一次计算 {#getting-started}

会一点 Python 就能开始。向量和梯度不熟，先读[从线性模型到神经网络](../00-foundations/from-linear-to-neural.md)，不用先学完全部数学。

1. [Tokenization](../00-foundations/core/tokenization.md)：一句话怎样变成模型的输入。
2. [Transformer 图解](../00-foundations/transformer-lab.md)：沿着一小段输入，看 Attention、FFN 和残差怎样接起来。
3. [训练基础](pretraining/README.md)：预测错了以后，loss 怎样让参数发生变化。

**读完试一试：**拿一句短句，讲清输入怎样变成预测，以及 loss 怎样参与参数更新。再用 [PyTorch 小实验](../00-foundations/pytorch/README.md)对照计算过程。模型家族、MoE 和分布式训练可以之后再看。

## 准备技术面：先找出自己卡在哪 {#interviews}

在[面试准备](../interview/README.md)里按需要选一块。已有基础可以直接看[快速复习](../interview/speedrun.md)，想系统补课可以看[按计划学习](../interview/study-plan.md)。路线中的小时数是参考，可以按自己的时间拆开学。

先看岗位描述和已经收到的面试说明，不要因为标题都叫 MLE，就默认每一轮考的一样。各岗位先练什么、时间紧时先放什么，见[按岗位选练习](../interview/ai-roles-prep.md#routes)。还没有具体面试，也可以拿一道概念题、一个小实现试试。

1. **说不清原理**：去 [ML / LLM 基础问答](../interview/basics/README.md)。试着不看笔记回答，再用一个例子解释。
2. **原理懂，代码写不出来**：去 [ML Coding](ml-exercises/README.md)。重点核对 shape、mask、梯度和边界条件。
3. **写题慢或方法想不到**：去 [Python](../interview/python.md) 和 [LeetCode 方法](../interview/leetcode.md)，按方法练，不按题号背。

**练习建议：**独立完成一道题，解释思路，再换几个输入测试。需要准备设计题，接着看 [Feed / RAG / Memory](system-design/README.md)；项目追问参考[准备什么](../career/prepare.md)。

## 读研究：弄清它到底改了什么 {#research}

这条路线适合了解 Transformer、想进一步读论文或模型报告的人。重点是理解作者改了什么、实验能支持什么结论；Research Scientist / Research Engineer 的岗位要求仍以具体团队为准。

1. [架构组件](../00-foundations/deep-dives/README.md)：先找到论文改的是哪个部件，不熟的机制单独补。
2. [模型家族](../00-foundations/model-families/README.md)：带着同一组问题比较设计，而不是记版本名字。
3. [评估基础](../07-evaluation/README.md)：看比较对象、数据、指标和消融，分清结果支持什么。

**读完试一试：**用一页纸概括问题、方法、实验结果和待验证的疑问。能复算的例子自己算一遍；只做了局部验证，就说明验证范围。对后训练感兴趣，可以继续看[原理](../05-post-training/README.md)和[实验实践](../practice/post-training/README.md)。

## 做项目：先让一条路径走通 {#projects}

没做过完整应用，先从[本站与阅读清单](../practice/web-lab/README.md)开始。它不需要 GPU：看懂构建与 CI/CD，再连起页面、API 和数据库。

已经会写基本程序，可以直接选自己想做的项目：

- [推荐系统](../practice/recommender-systems/README.md)：沿着一次请求，看召回、排序、过滤与评估。
- [RAG 知识库](../practice/rag/README.md)：从资料的版本和权限，到检索、引用和答案检查。
- [后训练实验](../practice/post-training/README.md)：从要修的错误，到 SFT、数据 batch、多卡选择、checkpoint 和发布判断。

**动手试一试：**跑通一个最小例子，再改一个条件。例如，删掉支持答案的文档，或中断训练后恢复，观察系统会怎样处理。准备设计面试，可以先做[系统设计题](system-design/README.md)，再对照项目实现检查方案。

## 只想补一个专题，直接过去 {#topics}

下面按主题列出入口，各专题内有更详细的阅读顺序。

<div class="worked-table" markdown="1">

| 你关心的内容 | 从这里进入 |
| --- | --- |
| 模型基础与架构 | [核心机制](../00-foundations/core/README.md) · [组件与模型家族](../00-foundations/model-families/README.md) |
| 数据与训练 | [数据和反馈](../01-data-and-feedback/README.md) · [预训练与训练工程](pretraining/README.md) |
| 强化学习与后训练 | [Deep RL](../05-post-training/deep-rl/README.md) · [SFT / RLHF / 蒸馏](../05-post-training/README.md) |
| 评估是否可信 | [指标与实验](../07-evaluation/README.md) · [LLM-as-a-Judge](../07-evaluation/llm-as-a-judge/README.md) |
| 图片怎样接进语言模型 | [多模态](../03-multimodal-learning/README.md) |
| 检索与长期状态 | [搜索与检索](../04-search/README.md) · [记忆](../02-memory/README.md) |
| 工具与智能体 | [Agents](../10-agents/README.md) · [Personal AGI](../09-personal-agi/README.md) |
| 推理速度、成本与可靠性 | [推理](inference/README.md) · [AI 系统](../06-systems/README.md) · [模型体验](../08-model-experience/README.md) |
| 芯片、GPU、CUDA 与集群 | [AI 基础设施（AI infra）](../12-infra/README.md) |
| 感知、机器人与自动驾驶 | [从像素到动作](../13-embodied/README.md) |
| 还在变化的方向 | [前沿](../11-frontier/README.md)：世界模型、递归自我改进（RSI）、测试时计算（test-time compute）、用 RL 训练 agent、机制可解释性和扩散语言模型（dLLM） |

</div>

只想查某个词，点顶部搜索或按 `/`。中文、英文术语都可以试试。

## 求职准备：先安排眼前需要做的事 {#applications}

1. [到底要准备什么](../career/prepare.md)：按岗位和自己的短板安排精力。
2. [一次面试复盘](../career/one-interview.md)：面完留下具体问题，再调整下一轮准备。

可以先为最近一轮面试列出要补的内容，再安排具体练习。个人经历放在 [25–26 intern 求职](../career/journey.md)和 [26–27 new grad 求职](../career/new-grad-2026-2027.md)，仅供参考。

## 想看全貌，或以后接着读 {#all-notes}

全部内容分在[基础与原理](README.md)、[面试准备](../interview/README.md)、[工程实践](../practice/README.md)、[编年史](technology-timeline.md)和[求职记录](../career/README.md)中。路线只是不同的阅读顺序，指向同一份文章。

右上角 **中文 / EN** 切换同一篇。点标题下的「收藏本页」收藏文章，之后在「我的阅读」里找回收藏和最近读过的页面；记录只在当前浏览器，换设备前可以在「设置」里导出备份。内容还在继续补，哪些已写、哪些待核对，见[覆盖与待办](coverage.md)。
