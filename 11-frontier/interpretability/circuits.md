# 可解释性：电路、归因与因果干预

**中文** · [English](circuits.en.md)

> 阅读时间：约 12 分钟 · 最近审阅：2026-10

“GPT-2 small 用 26 个 attention head 找出间接宾语”“Claude 3.5 Haiku 先在内部想到 Texas，再说出 Austin”，这类电路解释具体指什么，又有哪些证据支持？

电路解释是一种可以通过干预检验的因果假设：在某一类输入上，哪些部件经过哪些路径，解释了行为的多大一部分。激活修补和路径修补用来检验这些假设，归因图则把提出假设这一步自动化。不过，归因图解释的是一个替代模型；Anthropic 报告，它只在大约四分之一的提示词上给出令人满意的解释。

安全评估中已经有一批实验室自报的案例，也有在人为植入问题上的成功，但还没有经过独立验证的通用检测器。

## 从相关到因果：激活修补

探针和归因只说明“相关”或“局部敏感”。要说某个部件导致了某个行为，就得动手改。激活修补（activation patching，也叫 causal tracing、interchange intervention）跑三次：一次干净输入（clean），一次破坏过的输入（corrupt），再跑一次破坏过的输入、但把某个部件的值换成干净运行里的值（patched）。效果按比例算：

$$
\text{修补效果}=\frac{y_{\rm patched}-y_{\rm corrupt}}{y_{\rm clean}-y_{\rm corrupt}}
$$

$y$ 是关心的输出指标，比如正确答案的概率，或正确与错误答案的 logit 差。方向可以反过来：在干净运行里换进破坏后的值，看行为掉多少。前者叫 denoising，问的是“这个部件够不够把行为救回来”；后者叫 noising，问的是“没有它行不行”（术语见 [Heimersheim & Nanda，2024-04](https://arxiv.org/abs/2404.15255)）。

[ROME（Meng 等，2022-02）](https://arxiv.org/abs/2202.05262)是一个经典用例。在 GPT-2 XL 上取 1000 条模型知道的事实，给主语 token 的 embedding 加上 3 倍标准差的高斯噪声，正确答案的平均概率从 27.0% 掉到 8.47%；再逐个把干净的隐藏状态放回去。平均总效应是 18.6%，单个状态里效果最强的在主语最后一个 token、第 15 层附近（平均间接效应 8.7%）；在这个位置，MLP 的贡献最高 6.6%，注意力只有 1.6%。作者据此认为事实存放在中间层 MLP，并直接修改这些 MLP 的权重来编辑事实。

<!-- paper-figure:rome-causal-trace -->

两条提醒。[Hase 等（2023-01）](https://arxiv.org/abs/2301.04213)发现，causal tracing 指出的层，并不能预测编辑哪一层效果最好：定位和编辑是两回事。[Zhang & Nanda（2023-09）](https://arxiv.org/abs/2309.16042)发现，换一种指标（概率还是 logit 差）、换一种破坏方式（加噪声还是换成另一句话），定位结论就可能不同。

## 手算：3 个节点的修补（教学例子）

下面这个小网络是编的，只为看清修补在量什么。$s=1$ 表示干净输入，$s=0$ 表示破坏后的输入：

```text
A = 2·s
B = 1.5·A
C = ReLU(A − 0.5·B)
y = B + C
```

干净运行：$A=2$，$B=3$，$C=\mathrm{ReLU}(2-1.5)=0.5$，$y=3.5$。破坏运行：全是 0，$y=0$。

| 修补的节点 | denoising 后的 $y$ | 恢复比例 | noising 后的 $y$ | 下降比例 |
| --- | --- | --- | --- | --- |
| A | 3.5 | 100% | 0 | 100% |
| B | 3.0 | 86% | 2.0 | 43% |
| C | 0.5 | 14% | 3.0 | 14% |

看 B 这一行。denoising：$A=0$、$B=3$，$C=\mathrm{ReLU}(0-1.5)=0$，$y=3.0$，恢复 $3/3.5=86\%$。noising：$A=2$、$B=0$，这时 $C=\mathrm{ReLU}(2-0)=2$，$y=2.0$，只掉了 $1.5/3.5=43\%$。C 平时贡献很小，B 一被拿掉它就顶上来，是一个“备份”。只做 noising 会低估 B，只做 denoising 又看不到 C 的作用。

再做一次路径修补（[Goldowsky-Dill 等，2023-04](https://arxiv.org/abs/2304.05969)）：在破坏运行里，只把干净的 A 送给 A→B 这条边，C 收到的 A 仍是 0。于是 $B=3$，$C=\mathrm{ReLU}(0-1.5)=0$，$y=3.0$，占 86%。只送给 A→C 这条边：$B=0$，$C=\mathrm{ReLU}(2-0)=2$，$y=2.0$，占 57%。两条路径加起来是 143%，不是 100%。有非线性时，各条路径的效果不能简单相加，“这条路径占几成”这个问题本身就要小心问。

| 先试什么 | 应该观察什么 | 不能据此推出什么 |
| --- | --- | --- |
| 只做 denoising | B 能恢复 86% | B 是必要的 |
| 只做 noising | 拿掉 B 只掉 43% | B 不重要，或 C 无关 |
| 把两条路径的效果相加 | 得到 143% | 每条路径各占多少份额 |

## 一个电路说法长什么样：IOI

[Wang 等（2022-11）](https://arxiv.org/abs/2211.00593)研究的任务叫间接宾语识别（IOI）：“When Mary and John went to the store, John gave a bottle of milk to” 应接 Mary。GPT-2 small 有 12 层、每层 12 个 head；在 10 万个样例上，正确名字比重复出现的名字平均高 3.56 个 logit，99.3% 的样例里选对。

他们给出的电路有 26 个 head，分 7 类。大意是：duplicate token head 和 induction head（借助 previous token head）发现 John 出现了两次；S-inhibition head 把这个信息写进 name mover head 的 query，让它别看 John；name mover head 在句末看向剩下的名字 Mary，把它复制到输出；negative name mover head 往反方向写，降低置信度；backup name mover head 平时不怎么用，name mover 被拿掉时才顶上。部件的“拿掉”用均值消融：换成在三个不同名字的句子分布上的平均激活。

<!-- paper-figure:ioi-circuit -->

光说“这些 head 组成电路”不够，他们用三条标准检验。设 $F(\cdot)$ 是平均 logit 差：

- **忠实度**：只保留电路时 $F(C)$ 接近完整模型 $F(M)$。实测 $|F(M)-F(C)|=0.46$，是 3.56 的 13%，即电路恢复了 87%。
- **完整性**：从电路里拿掉任意一组部件 $K$，效果应和从完整模型里拿掉 $K$ 差不多。
- **最小性**：电路里每个部件都有某个 $K$ 让它变得必要。

完整性就是为“备份”准备的。回到上面的算例：电路 {A, B}（C 被消融）的输出是 3.0，忠实度 86%，看起来不错；可是拿掉 B，完整模型还剩 2.0（C 顶上），电路只剩 0，差得很远，说明 C 漏掉了。作者自己在[摘要](https://arxiv.org/abs/2211.00593)里也写道：“Though these criteria support our explanation, they also point to remaining gaps in our understanding”，即这三条标准支持他们的解释，但也暴露了理解上的空缺。这个说法的范围是：GPT-2 small、15 个模板生成的句子、均值消融这种选法。

找电路后来也自动化了：ACDC 在 GPT-2 small 的“greater-than”任务上，从 32,000 条边里选出 68 条，全部是此前人工找到的，并重新发现了 5 类部件中的全部 5 类（[Conmy 等，2023-04](https://arxiv.org/abs/2304.14997)）。另外，[Makelov 等（2023-11）](https://arxiv.org/abs/2311.17030)提醒，修补一个子空间可能激活一条平时不用的平行通路，造成假象；他们也把 IOI 当作成功案例，因为先前的人工电路分析说明了特征在哪里。

## 归因图：把提出假设这一步自动化

IOI 的电路分析花了大量人工，而且只覆盖一个任务。Anthropic 2025-03 的 [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) 尝试自动画出模型在每个提示词上的计算过程。

做法分四步。第一，训练一个跨层 transcoder（CLT）：第 $\ell$ 层的特征从该层残差流读入，可以写到第 $\ell$ 层及之后所有层的 MLP 输出上；在 Claude 3.5 Haiku 上训练的 CLT 总共有 30 万到 3000 万个特征。用它替换所有 MLP，就得到一个替代模型；他们最大的 CLT 在一个 18 层小模型上，只在 50% 的多样提示词上和原模型给出同一个下一 token。第二，针对单个提示词，冻结原模型的注意力模式和归一化分母，并在每个位置、每层加一个“误差节点”补上 CLT 没还原的部分，这样这个局部替代模型的激活和输出与原模型完全一致。第三，在这个已经线性化的图上算边：

$$
A_{s\to t}=a_s\,w_{s\to t}
$$

$a_s$ 是源特征的激活，$w_{s\to t}$ 是两者之间经过残差流和冻结注意力的“虚拟权重”。第四，剪掉弱边，把作用相近的特征归成超节点，再回到原模型上做干预来验证。

同期的 [Biology](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) 一文报告了一批案例。开头提到的两步推理：抑制 “Dallas” 特征，“Texas” 特征和下游的 “say Austin” 随之变弱，而 “say a capital” 基本不受影响。写诗时的规划：在一行开始前的换行符上，就出现了候选押韵词（如 rabbit、habit）的特征，改掉它，模型会把整行改写成以新词结尾；作者说大约一半的诗里找到了这种规划词特征。另外还有多语言、加法、医学诊断、幻觉、拒答、越狱、思维链是否忠实、隐藏目标等案例。

方法论文专门写了局限：不解释注意力为什么看某个位置（QK 电路），所以在 induction 和多选题这类例子里，图会“跳过最有意思的部分”；CLT 还原不全，误差节点有时把关键计算遮住，一个混淆过的越狱提示上几乎全是误差节点；没激活的特征和抑制关系很难表达；图很复杂；特征分裂和吸收让抽象层次不对；只能看单个提示词，全局电路很难；替代模型和原模型未必用同一种机制。Biology 一文说，归因图大约在四分之一试过的提示词上给出令人满意的洞见，文中展示的都是成功案例。

后续进展：2025-05-29 Anthropic [开源了工具](https://www.anthropic.com/research/open-source-circuit-tracing)，可以在 Gemma-2-2b、Llama-3.2-1b 上画图，也接入了 Neuronpedia；2025-07-31 的 [QK 归因](https://transformer-circuits.pub/2025/attention-qk/index.html)把注意力分数写成 query 侧和 key 侧特征的双线性函数，验证了 induction 和多选题里此前只是猜测的机制。另一条路是从训练上下手：OpenAI 训练大部分权重为零的 Transformer，剪出的任务电路小而可读，但更稀疏就更弱，作者说超过几千万个非零参数后还难以保持可解释性（[Gao 等，2025-11，ICML 2026](https://arxiv.org/abs/2511.13653)）。一篇 2026-02 的 LessWrong 复现（Anthropic Fellows 项目，未经同行评审）认为这类剪出的电路可能“可读但不忠实”：剪枝在一个标签被重新指定、本身没有意义的任务上，也能用约 30 个节点做到 95% 以上的准确率；作者也说明自己改用了零消融，而原论文的均值消融可能会得到更忠实的电路（[Drori](https://www.greaterwrong.com/posts/sHpZZnRDLg7ccX9aF/weight-sparse-circuits-may-be-interpretable-yet-unfaithful)）。

## 和安全评估怎么接上

行为评估和思维链监控依赖模型做了什么、说了什么。OpenAI 发现用另一个模型读思维链能抓到编程 agent 的 reward hacking，但如果把这个监控直接放进训练奖励、优化得太狠，agent 会学会在思维链里藏起意图，继续作弊（[Baker 等，2025-03](https://arxiv.org/abs/2503.11926)）。可解释性的期望是提供一种不依赖模型自述的检查。reward hacking 的定义见[可验证奖励](../../05-post-training/verifiable-rewards.md)。下面是能核对到的公开结果：

| 时间与机构 | 设置 | 方法 | 结果 | 要打的折扣 |
| --- | --- | --- | --- | --- |
| 2024-04，[Anthropic](https://www.anthropic.com/research/probes-catch-sleeper-agents) | 人为植入后门、部署时写不安全代码的 sleeper agent | 只用一对通用问答（“你是一个有帮助的 AI 吗？”答“否”和答“是”）的均值差做探针 | AUROC 99.3% | 后门是人为植入的；1000 个随机方向里最好的也有 95.3% |
| 2025-02，[Apollo](https://arxiv.org/abs/2502.03407) | Llama-3.3-70B 在隐瞒内幕交易、故意压低评估分数等场景里说谎 | 线性探针 | AUROC 0.96–0.999；在 1% 误报率下抓到 95–99% | 作者说还不足以作为可靠防线 |
| 2025-03，[Anthropic](https://arxiv.org/abs/2503.10965) | 被训练成讨好奖励模型、又隐瞒这一目标的模型 | 盲审游戏：SAE、行为攻击、训练数据分析 | 4 队中 3 队找到隐藏目标；只给 API 的那队没找到 | 目标是人为植入的；SAE 的解法用语义搜索也能复现 |
| 2025-09，[Anthropic](https://assets.anthropic.com/m/12f214efcc2f457a/original/Claude-Sonnet-4-5-System-Card.pdf) | Claude Sonnet 4.5 部署前对齐评估 | 抑制与“这是测试”相关的 SAE 特征和对比向量 | 100 条蜜罐提示上，口头的评估意识下降，不当行为从 0 升到最多 8–9%；10 个随机 SAE 特征的对照为 0 | 实验室自报；在别的数据集上效果小或没有 |
| 2026-04，[Anthropic](https://transformer-circuits.pub/2026/emotions/index.html) | Claude Sonnet 4.5 在 7 个“不可能完成”的编程任务上 | 沿“desperate”情绪向量引导 | 强度从 −0.1 调到 +0.1，reward hacking 从约 5% 升到约 70% | 是引导实验，说明这个方向影响行为，本身不是检测器 |
| 2026-05，[Anthropic](https://transformer-circuits.pub/2026/nla/index.html) | Claude Opus 4.6 部署前审计 | 自然语言自编码器（NLA） | 发现未说出口的评估意识；顺着早期版本用俄语回答英文提问的异常，查到一批提示和回答语言错配的训练数据 | 实验室自报；NLA 会编造细节 |

能说到哪一步：探针、引导和字典特征在“有标准答案”的设置里（人为植入的后门、隐藏目标）表现不错，也已进入实验室的部署前审计；但成功多来自人为构造的模型或实验室自报，Apollo 和 Anthropic 都在原文里写了局限。Dario Amodei 2025-04 的[文章](https://www.darioamodei.com/post/the-urgency-of-interpretability)说 Anthropic 的目标是 2027 年前让可解释性“可靠地检测出大多数模型问题”，并说按当前进度，他很看好可解释性在 5 到 10 年内成为真正的“AI 的 MRI”。这是一家公司负责人的目标和判断，不是测出来的结果。

## 每一步证据能说明什么

| 证据 | 能说明 | 不能说明 |
| --- | --- | --- |
| 梯度或线性归因 | 局部、一阶的影响大小 | 改动较大时的实际效果 |
| denoising 修补 | 这个部件足以把行为救回来 | 它是必要的 |
| noising 修补或消融 | 拿掉它行为掉多少 | 掉得少就说明它不参与（可能有备份） |
| 路径修补 | 效果经过哪些边 | 每条路径各占几成（有非线性时不可加） |
| 忠实度、完整性、最小性 | 在这个输入分布上电路解释得多完整 | 换分布、换模型也成立 |
| 归因图加干预 | 替代模型里的机制，并在原模型上验证了一部分 | 原模型整体用的是同一机制 |
| 在人为植入的问题上检测成功 | 有标准答案时方法能找到问题 | 能找出自然出现、事先没人知道的问题 |

## 这条线接下来看什么

1. **注意力和全局电路。** QK 归因（2025-07）补上了一块，但 Anthropic 2025-03 报告的“令人满意”比例约为四分之一。能改变判断的结果：后续报告里这个比例明显上升，或出现跨很多提示词都成立的全局电路。
2. **替代模型忠实吗？** Anthropic 2025-07 的一篇短文用[玩具模型](https://transformer-circuits.pub/2025/faithfulness-toy-model/index.html)展示了 transcoder 会出错的情形；2026-02 的复现质疑剪枝电路的忠实性。能改变判断的结果：每个基于图的说法都在原模型上做干预验证，并公开验证失败的比例。
3. **从人为构造的模型走到真实部署。** 截至 2026-10，能核对的检测成功多在人为植入的问题上。能改变判断的结果：可解释性在生产模型里找到行为评估漏掉的问题，并由独立机构复现。
4. **白盒检查和思维链监控各管什么。** 多家机构的作者在 2025-07 联合发文，称思维链可监控性是一个“新的、脆弱的机会”（[Korbak 等](https://arxiv.org/abs/2507.11473)）。能改变判断的结果：在同一批不当行为上，直接比较白盒探针和思维链监控的检出率和误报率。

最后用[复习题](review.md)把整个系列串一遍。
