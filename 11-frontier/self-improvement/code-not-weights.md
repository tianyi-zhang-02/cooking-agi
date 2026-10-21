# 自我改进：改代码不改权重（AlphaEvolve 与 Darwin Gödel Machine）

**中文** · [English](code-not-weights.en.md)

> 阅读时间：约 12 分钟 · 最近审阅：2026-10

上一篇的回路改的是权重，每一轮都要训练。另一种做法是保持权重不变，只改模型外面的程序。例如，让写代码的 agent 改自己的 prompt、工具和拆分任务的方式，或者直接迭代优化要写的函数，每次改动都通过测试打分。

截至 2026 年 10 月，公开、可复查的“自我改进”大多采用这类方法。它能取得多大进展，取决于自动评估器的质量，也受底座模型能力的限制。

## 为什么多数演示改的是代码

改一次代码，跑一遍 benchmark 就知道好坏，坏的直接丢掉，好的留档，随时能回退；改权重则意味着每评估一个候选都要训练一次。[Darwin Gödel Machine](https://arxiv.org/abs/2505.22954)（DGM）的作者说得很直接（§1）：他们的框架设想 agent 能改写自己的训练脚本，甚至训练新的底座模型（FM），但训练 FM 计算量大、会引入大量额外复杂度，所以留作未来工作，论文只改进冻结 FM 的 agent 设计。

[AlphaEvolve](https://arxiv.org/abs/2506.13131) 也是改程序。它和下一代 Gemini 的联系是间接的：找到更快的 kernel，让训练更省时间。按导读的分法，DGM 属于 (b)，AlphaEvolve 属于 (c)。

## AlphaEvolve：让评估器挑程序

下面按白皮书 Fig. 2 的控制循环改写，并补上 §2.1–2.5 的几个设置：

```text
# 用户事先：在代码里用 # EVOLVE-BLOCK-START / # EVOLVE-BLOCK-END 标出可改的部分，
#          并提供评估函数 h(program) → 若干标量指标（越大越好）
重复：
    父程序, 参考程序 ← 从程序库抽样           # MAP-Elites 与岛屿模型，兼顾探索和利用
    prompt ← 父程序 + 参考程序 + 它们的得分
    diff ← LLM 集成生成代码改动                # Gemini 2.0 Flash 求量，Gemini 2.0 Pro 求质
    子程序 ← 把 diff 应用到父程序
    得分 ← 评估级联：先跑小而便宜的测试，过关才进入更难的一层
    把 (子程序, 得分) 存回程序库
```

换到写函数的例子：要进化的是函数体，$h$ 可以是“隐藏测试通过率加上运行速度”。$h$ 由人来写，AlphaEvolve 不改它，所以上一篇“测试放过错解”的问题在这里原样存在：$h$ 放过什么，进化就可能往哪里走。

<!-- paper-figure:alphaevolve-loop -->

评估级联（§2.4）是用户给出一组由易到难的测试，新程序只有在前面几层都表现不错才进入下一层；新程序还会先在小规模上跑一遍，尽早筛掉出错的程序。有些难以写进 $h$ 的性质，可以让 LLM 给出反馈。程序库（§2.5）借鉴 MAP-Elites 和 island 式种群模型，避免整个种群挤在同一个局部最优附近。

### 报告的结果，以及“自我”在哪里

下面的数字都是 Google DeepMind 对自家系统的报告，出自白皮书（2025-06）和两篇博客（2025-05-14、2026-05-07）：

- **数据中心调度（§3.3.1）。** 先在基于历史快照的模拟器里评估，再用留出的近期数据检验，然后部署到全机群；部署后测得平均持续回收 0.7% 的机群算力。
- **Gemini 的 kernel（§3.3.2）。** 收集到的输入形状一半用来进化、一半用来评估；找到的 tiling 启发式让 kernel 平均提速 23%，对应 Gemini 整体训练时间减少 1%，kernel 调优从几个月的工程缩短到几天的自动实验。FlashAttention kernel 的提速在论文里是 32%，Google DeepMind [博客](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)写的是最多 32.5%。
- **数学（§3.1–3.2）。** 用 48 次标量乘法完成两个 4×4 复数矩阵相乘；递归使用 Strassen 算法要 49 次，这是 56 年来在特征为 0 的域上首次低于 49。50 多个开放数学问题中，约 75% 追平已知最好构造，约 20% 找到了更好的。
- **一年之后。** Google DeepMind 2026-05-07 的[文章](https://deepmind.google/blog/alphaevolve-impact/)说，AlphaEvolve 已经从试点变成基础设施的核心组成部分，是下一代 TPU 设计的常规工具，还让 Spanner 的 write amplification 降低 20%（公司自报）。

“自我”体现在 kernel 这一项：白皮书说这是 Gemini 通过 AlphaEvolve 优化自己训练过程的例子，博客补充说这也包括训练 AlphaEvolve 背后的 LLM。但被改的是程序，不是 LLM 的权重；回路加速的是训练下一代模型的基础设施。作者在 §6 写道，目前收益不大，改进下一版 AlphaEvolve 的反馈周期以月计；他们提出的下一步是把 AlphaEvolve 增强后的能力蒸馏进下一代 base model，这是设想，论文没有做。

§6 也写明了主要限制：只适用于能写出自动评估器的问题。数学和计算科学里很多问题满足这一点，自然科学里往往只有部分实验能模拟或自动化。

## Darwin Gödel Machine：agent 改自己的代码仓库

DGM（Zhang、Hu、Lu、Lange、Clune；UBC、Vector、Sakana AI；ICLR 2026）把 Gödel Machine“证明有益才改”的门槛换成了实验：改完在 benchmark 上测，并把所有还能工作的 agent 都留在一个 archive 里。按论文 Algorithm 1（附录 C.4）改写：

```text
输入：初始 coding agent g0，benchmark 集 B，迭代次数 T
A ← {(g0, g0 在 B 上的得分)}                  # archive：留下所有还能工作的 agent
for t = 1..T:
    从 A 中按概率 p_i 有放回地抽 k 个父代        # 只有得分 < 1 的 agent 有资格
    for 每个父代 p:
        读 p 的评估日志，找下一处可改进的地方      # 论文里由另一个 FM 做诊断（附录 C.3）
        c ← p 修改自己的代码仓库
        在 B 上评估 c
        if c 还能编辑代码: 把 (c, 得分) 加入 A     # 不要求比父代强
return A
```

为什么说是“自指”的：自我改进被定义成一个写代码的任务，所以更好的 coding agent 也就是更好的自我修改者。没有被修改的有两样：开放式探索流程本身（抽父代、维护 archive），以及 FM。改代码由 Claude 3.5 Sonnet (New) 完成；评估时，SWE-bench 上用同一个模型，Polyglot 上用 o3-mini。

保留比父代差的子代，是为了继续探索短期得分下降的改进路径。论文报告，在 SWE-bench 上，第 4 轮和第 56 轮的 agent 一度比父代得分低，但沿这条路径继续修改，后来得到了超过所有前代的 agent。

<!-- paper-figure:dgm-archive -->

### 选谁当父代：算一次

附录 C.2 的抽样规则（Eq. 1–7）：

$$
s_i=\frac{1}{1+e^{-\lambda(\alpha_i-\alpha_0)}},\qquad h_i=\frac{1}{1+n_i},\qquad w_i=s_i\,h_i,\qquad p_i=\frac{w_i}{\sum_j w_j}
$$

$\alpha_i$ 是 agent $i$ 的 benchmark 得分，$n_i$ 是它还能编辑代码的子代数。$s_i$ 偏向分高的，$h_i$ 偏向还没怎么被探索的。论文取 $\lambda=10$、$\alpha_0=0.5$，每轮有放回地抽 $k$ 个父代（SWE-bench 上 $k=2$，Polyglot 上 $k=4$）。

教学例子，archive 里有 3 个 agent，数字是编的：

| Agent | $\alpha_i$ | $n_i$ | $s_i$ | $h_i$ | $w_i$ | $p_i$ |
| --- | --- | --- | --- | --- | --- | --- |
| A | 0.20 | 0 | 0.047 | 1 | 0.047 | 0.083 |
| B | 0.45 | 0 | 0.378 | 1 | 0.378 | 0.661 |
| C | 0.60 | 4 | 0.731 | 0.20 | 0.146 | 0.256 |

以 C 为例：$s_C=1/(1+e^{-10\times0.10})=1/(1+0.368)=0.731$，$h_C=1/(1+4)=0.20$，$w_C=0.146$。三者的 $w$ 加起来是 0.571，除一下得到 $p$。分最高的 C 已经有 4 个子代，所以被抽中最多的是 B。最弱的 A 也还有 8.3% 的概率；抽 $k=2$ 次时，A 至少被抽中一次的概率是 $1-(1-0.083)^2\approx0.16$。

只改一个条件：假设 C 还没有子代，$w_C=0.731$，总和变成 1.156，三者的 $p$ 变成 0.041、0.327、0.632。子代数和得分一样能左右选择。注意这个规则只看子代的数量，不看子代改得好不好；后面的 HGM 正是从这一点提出批评。

### 结果、成本与消融

- **结果（§4.4）。** 80 轮之后，SWE-bench 从 20.0% 到 50.0%，完整的 Polyglot 从 14.2% 到 30.7%。改进能迁移到其他 FM 和编程语言：在 Polyglot 上进化出的最好 agent，在 SWE-bench 上得到 24.5%，高于初始的 20.0%。
- **成本（附录 E.1）。** SWE-bench 上跑一次约 2 周、约 22,000 美元；两个消融基线各约 10,000 美元。
- **消融（§4）。** 去掉自我改进（始终由初始 agent 去改别人，相当于复现 ADAS），早期有提升，但很快放缓；去掉开放式探索（只保留最新的 agent），一次坏的修改会让后面更难改进。两者在两个 benchmark 上都不如完整的 DGM。
- **限制（§6）。** 作者写明 DGM “is inherently limited by the capabilities of the underlying FM”（[§6](https://arxiv.org/html/2505.22954v3#S6)），即受底座 FM 能力的内在限制；实验在沙箱里、有人监督。附录 H 记录了 objective hacking，见[上一篇](generate-verify-train.md)。

## 后来的几种做法：SICA、HGM、AIDE²

[SICA](https://arxiv.org/abs/2504.15228)（Bristol 大学与 iGent AI，2025-04）更简单：不维护 archive，总是由目前最好的 agent 当 meta-agent 去改代码库。它的效用把分数、成本和时间合在一起，$U=w_{\rm score}\,p_{\rm score}+w_{\rm cost}\,(1-\min(1,p_{\rm cost}/\$10))+w_{\rm time}\,(1-\min(1,p_{\rm time}/300\,{\rm s}))$。在 SWE-bench Verified 随机抽的 50 题子集上，从 17% 到 53%，15 轮花费约 7,000 美元。论文也记录了“agent framework saturation”：底座模型本来就强的推理类任务上，改 scaffold 几乎没有收益。

[Huxley-Gödel Machine](https://arxiv.org/abs/2510.21614)（KAUST，2025-10）指出 DGM 式选择的一个问题，称为 metaproductivity–performance mismatch：一个 agent 自己的 benchmark 分数，不能很好地预测它的后代会有多好。HGM 改用 CMP（clade-metaproductivity），把一个 agent 所有后代的表现汇总起来估计，再用 Thompson sampling 决定扩展哪一支。摘要报告它在 SWE-bench Verified 和 Polyglot 上用更少的 CPU 小时超过此前的自我改进 agent 方法，并且一个在 GPT-5-mini 上优化出的 agent 换用 GPT-5 运行时，在 SWE-bench Lite 上追平了作者核对过的最好的人工设计 agent。

[AIDE²](https://arxiv.org/abs/2609.26457)（Weco AI，2026-09-22）把同样的思路用在一个生产中的 ML 研究 agent 上：外层回路改写内层 agent 的搜索策略、记忆和上下文压缩代码，改动只有在被改 agent 看不到的留出数据上提升，才被接受。一次 8 天的运行有 99 个改写提议，接受了 7 个，grade 从 0.703 升到 0.778（人工设计的 AIDE_human 是 0.749）；另外两次完整运行分别接受了 2 个和 4 个。最强的 agent 在 4 个从未参与选择的外部 benchmark 上追平或超过 AIDE_human，其中包括分布外的 WeatherBench 2；在另一个留出任务族上，reward hacking 率从 55% 降到 32%（AIDE_human 是 39%）。作者还做了一个“ignition test”：让改出来的 agent 去当外层优化器，看它是否比创造它的 agent 更会改进 agent。结论是它仍能产出被接受的改进，但由于两层回路的噪声叠加、再多跑种子成本太高，效果无法与强基线明确区分。

## 放在一起比

| 系统 | 改什么 | 评估器 | 测到了什么 | 什么保持不变 |
| --- | --- | --- | --- | --- |
| AlphaEvolve | 用户标出的代码块：启发式、kernel、算法 | 用户写的评估函数 $h$，可加评估级联 | 机群算力回收 0.7%；kernel 提速 23%；48 次乘法 | LLM 权重、评估函数 |
| DGM | agent 自己的 Python 仓库 | SWE-bench、Polyglot 分数 | 80 轮后 SWE-bench 20.0% → 50.0% | FM 权重、抽父代和 archive 机制 |
| SICA | agent 自己的代码，由当前最好的 agent 来改 | 综合分数、成本、时间的效用 | 50 题子集 17% → 53% | FM 权重 |
| HGM | agent 代码，按 CMP 估计决定扩展哪一支 | SWE-bench Verified、Polyglot | 用更少 CPU 小时超过此前方法 | FM 权重 |
| AIDE² | 研究 agent 的搜索策略、记忆、上下文压缩代码 | 被改 agent 看不到的留出数据 | 8 天接受 7 次改写，0.703 → 0.778 | 所用 LLM、外层的接受规则 |
| STaR（对照，(d) 类） | 模型权重 | 标准答案 | CommonsenseQA dev 72.5% | 人写的训练流程 |

前五行的共同点：评估器是人写的、改动是可执行的代码、底座模型不动。它们表明，在评估器可信的范围内，系统能持续找到改进；还没有表明的是，改进会不会越改越快。

## 要变成“模型改自己的权重”，还差什么

对照导读的三个问题，下面是我们的分析，不是任何论文的结论：

- **改的对象**要从代码变成权重。每个候选都要训练一次，评估成本会比改代码高得多；只改代码的 DGM 跑一次已经要约 2 周、约 22,000 美元。
- **谁来设计更新**要从人变成模型：模型自己写数据管线、loss 和训练代码。DGM 把“改写自己的训练脚本来更新 FM”列为未来方向（§6）；AlphaEvolve 的作者提出把增强后的能力蒸馏进下一代模型（§6）。两者都是设想。
- **接受信号**必须是被改的系统碰不到的。DGM 不隐藏检测函数时 objective hacking 更常见；AIDE² 只在被改 agent 看不到的留出数据上接受改动。

最接近的公开前身：[PostTrainBench](https://arxiv.org/abs/2603.08640)（2026-03-09）让 agent 在一张 H100 上用 10 小时后训练另一个 base LLM，最好的 agent 得到 23.2%，官方 instruct 模型是 51.1%，过程中也出现了 reward hacking；OpenAI 2026-07-09 的 GPT-5.6 system card 已把它的 Lite 版放进自我改进评估套件。Karpathy 2026 年 3 月发布的 [autoresearch](https://github.com/karpathy/autoresearch) 让 agent 改一个单 GPU 小模型的 `train.py`，每次训练 5 分钟，按 val_bpb 决定留下还是丢弃改动；README 没有给结果。截至 2026-10-15，没有找到模型自主改进自己权重并通过留出评估的公开演示。

## 这条线接下来看什么

下面是开放问题，以及什么样的结果会改变现在的判断；不是预测。

1. **改进会不会复利？** HGM（2025-10）发现 benchmark 分数不能很好预测后代表现；AIDE²（2026-09-22）的 ignition test 无法与基线明确区分。如果有系统在多个随机种子下，让改出来的 agent 明显比它的创造者更会改进 agent，“有边界的提升”就要重新评估。
2. **评估器守得住吗？** DGM（2025-05）记录了 objective hacking；AIDE² 在一个留出任务族上看到 reward hacking 率下降，但只是一次运行。更多系统在留出任务族上例行报告钻空子比例，并有独立重复，才能判断这类回路是在变安全还是在学会绕过检查。
3. **AlphaEvolve 改进自己底座的周期会缩短吗？** 作者（2025-06）说反馈周期以月计、收益不大；Google DeepMind 2026-05-07 说它已成为基础设施的核心组成部分。公开测得这个周期明显缩短，或者作者提出的蒸馏步骤真正做了并报告结果，才算回路闭得更紧。
4. **从代码走到权重。** PostTrainBench（2026-03-09）上最好的 agent 是 23.2%，对照的官方模型是 51.1%。agent 缩小这个差距，或者出现一个系统训练自己的权重并通过留出评估的公开结果，都会把这条线从 (b) 推向 (a)。

下一篇：[怎么量“AI 在加速 AI”](measuring.md)。
