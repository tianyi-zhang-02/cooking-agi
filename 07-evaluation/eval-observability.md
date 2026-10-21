# 评估接到线上：LLM 应用的可观测性

**中文** · [English](eval-observability.en.md)

> 阅读时间：约 14 分钟 · 最近审阅：2026-10

新 prompt 在离线评估集上拿了 92 分，上线两周后，“答非所问”的投诉却多了。离线集没变，变的是线上流量：新话题、新语言、更长的对话，模型提供方也可能在同一个别名后面换了版本。线上没有参考答案，每天十万条请求也不可能都让人看。**线上评估要回答一个可以算出来的问题：只评判一小部分请求，能以多大把握、多快发现多大的退化？答案往往比直觉差，对小切片尤其如此。**

## 离线评估集和线上流量，差在哪里

| 方面 | 离线评估集 | 线上流量 |
| --- | --- | --- |
| 输入 | 固定，分布已知 | 随时间漂移，不断出现长尾和新话题 |
| 参考答案 | 通常有，或有执行器能验证 | 几乎没有，只能做无参考检查 |
| 版本 | 跑评估时锁定 | prompt、索引、工具和模型别名背后的版本都可能变 |
| 上下文 | 多为单轮，上下文干净 | 多轮，带记忆和检索结果，可能被截断 |
| 规模 | 几百到几千条，可以全部评 | 每天数万条以上，只能抽样评 |
| 反馈 | 标注者给的标签 | 稀疏且有偏的用户行为：点踩、重新生成、中途放弃 |

所以线上评估不替代离线评估。它做两件离线集做不到的事：发现离线集没覆盖的失败；在没有参考答案时，估计质量有没有在变。发现的失败再送回离线集，成为以后每次改动都要跑的回归题。

还要分清监控和实验。监控回答“质量有没有变”，A/B 实验回答“是不是这个改动造成的”。发布当天曲线下跌，可能是发布引起的，也可能是同一天流量结构变了；要归因，仍需要[对照实验](ablation-and-slices.md)。

## 每条请求要记下什么

[Agent 可观测性](../06-systems/agent-observability.md)已经讲了 trace 怎样串起模型、检索和工具调用，以及怎样回放和脱敏。这里只问一件事：**为了事后能评估这条请求，必须留下哪些字段？** 也就是能让 judge 或人重新看懂这次回答的条件：哪个 prompt 模板和版本、哪个模型和参数、检索到了什么、调用了哪些工具、输出了什么、花了多少时间和 token，以及用户之后做了什么。

OpenTelemetry（OTel）为这些字段定义了 GenAI 语义约定（semantic conventions）。字段名还在变，所以先说状态，截至 2026-10-15：

- 这套约定的状态是 **Development**（OTel 以前把这一级叫 Experimental）。按 [OTel 的版本与稳定性说明](https://opentelemetry.io/docs/specs/otel/versioning-and-stability/#development)，这一级可能有 breaking change，而且“Long-term dependencies SHOULD NOT be taken against signals in Development”，即不应对它建立长期依赖。
- 核心语义约定从 [v1.42.0](https://github.com/open-telemetry/semantic-conventions/releases/tag/v1.42.0)（2026-06-12）起弃用全部 `gen_ai.*` 定义，迁到独立的 [semantic-conventions-genai 仓库](https://github.com/open-telemetry/semantic-conventions-genai)；从 v1.42.0 到目前最新的 v1.44.0（2026-08-04），核心仓库和 [opentelemetry.io 上的 GenAI 页面](https://opentelemetry.io/docs/specs/semconv/gen-ai/)都只剩“已迁移”的说明。
- 新仓库截至 2026-10-15 没有任何 release 或 tag，CHANGELOG 只有“Unreleased”，其中已攒下约 10 条 breaking 变更。

所以下表分两列：左边是最后一个带完整 GenAI 文档的核心发布 v1.41.0 里的名字（[spans](https://github.com/open-telemetry/semantic-conventions/blob/v1.41.0/docs/gen-ai/gen-ai-spans.md)、[metrics](https://github.com/open-telemetry/semantic-conventions/blob/v1.41.0/docs/gen-ai/gen-ai-metrics.md)、[events](https://github.com/open-telemetry/semantic-conventions/blob/v1.41.0/docs/gen-ai/gen-ai-events.md)；v1.41.1 没有改动这几份文档），右边注明新仓库 main 分支（2026-10-14，未发布）上已知的变化。

| 要记的内容 | v1.41.0 中的名字 | 注意（含 main 上的变化） |
| --- | --- | --- |
| 操作类型、提供方 | `gen_ai.operation.name`、`gen_ai.provider.name` | 操作取值有 `chat`、`retrieval`、`execute_tool`、`invoke_agent`、`embeddings` 等 |
| 请求的模型、实际回答的模型 | `gen_ai.request.model`、`gen_ai.response.model` | 规范示例分别是 `gpt-4` 和 `gpt-4-0613`；两者都记，才看得出别名背后换了版本 |
| 生成参数 | `gen_ai.request.temperature`、`gen_ai.request.top_p`、`gen_ai.request.max_tokens` | 只记实际发出的值 |
| Prompt 模板和版本 | `gen_ai.prompt.name` 只在属性注册表里，没挂到任何 span 上 | main 上新增 `gen_ai.prompt.version`；在发布之前用自定义属性 |
| 会话 | `gen_ai.conversation.id` | 多轮对话按会话聚类分析 |
| 系统指令、输入和输出原文 | `gen_ai.system_instructions`、`gen_ai.input.messages`、`gen_ai.output.messages` | 都是 Opt-In，规范写明默认不应采集 |
| 检索 | `retrieval` 操作的 span；`gen_ai.data_source.id`；`gen_ai.retrieval.documents` | 文档内容是 Opt-In |
| 工具调用 | `execute_tool` 操作的 span；`gen_ai.tool.name`、`gen_ai.tool.call.id` | 参数和结果 `gen_ai.tool.call.arguments`、`gen_ai.tool.call.result` 是 Opt-In |
| 结束原因 | `gen_ai.response.finish_reasons` | `length` 的比例上升，常常意味着截断 |
| Token | `gen_ai.usage.input_tokens`、`gen_ai.usage.output_tokens`；直方图 `gen_ai.client.token.usage` | main 上删掉了这个直方图，换成 `gen_ai.client.inference.usage.input_tokens`、`…output_tokens` 等按方向拆开的指标 |
| 延迟 | span 时长；指标 `gen_ai.client.operation.duration`、`gen_ai.client.operation.time_to_first_chunk` | main 上推理调用改用 `gen_ai.client.inference.duration`、`gen_ai.client.inference.time_to_first_chunk` |
| 错误 | `error.type` | 这个属性是 Stable |
| 评估结果 | 事件 `gen_ai.evaluation.result`：`gen_ai.evaluation.name`、`gen_ai.evaluation.score.value`、`gen_ai.evaluation.score.label`、`gen_ai.evaluation.explanation` | 挂在被评估的 span 下；拿不到 span 时用 `gen_ai.response.id` 关联 |
| 费用、用户反馈 | 没有标准属性 | 费用用 token 数乘以带日期的价目表在下游算；反馈用自定义事件，按 trace id 或 response id 关联 |

三条实用建议。第一，记下自己发的是哪一版约定，在面板查询和原始字段之间放一层映射；升级 instrumentation 时名字可能变（v1.41.0 规定，原来按 v1.36.0 或更早版本发数据的 instrumentation 默认保持旧名字，设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 才切到最新实验版）。第二，模型名、参数和 prompt 版本要在请求发出时记下，事后从配置推断容易对不上。第三，judge 调用本身也是 GenAI 调用，同样要记模型、参数和 rubric 版本，讲漂移时会用到。

## 线上评估怎么跑：从抽样到回归集

```text
生产流量 ──(100%)──► 错误、延迟、token、费用、结束原因：每条都算
    │
    └─(按 trace id 随机抽样)──► 待评 trace
                                 ├─ 确定性检查：schema、引用 ID、工具参数、PII 检测
                                 └─ LLM judge：是否有依据、是否回答了问题、该不该拒答
                                          │
                       分数事件 ──► 按切片聚合 ──► 告警
                                          │
            随机审计 ─┬─ judge 不确定或高风险 ─┬─ 用户负反馈
                      └──────► 人工标注队列 ◄──┘
                                          │
                      确认的失败 ──► 回归集 ──► 离线评估与 CI
```

### 先随机抽样，再按需要补抽

估计“通过率”这类比例，需要和全部流量同分布的样本。OTel 文档里最常见的 head sampling 是按 trace id 做一致的概率抽样：一条 trace 要么整条被选中，要么整条不选。按错误或慢请求保留的 tail sampling 适合排障，但会改变分布，直接拿来算比率会偏；[Agent 可观测性](../06-systems/agent-observability.md)里 10/19 和 10/100 的例子说的就是这件事。两类样本可以同时存在，只要分开标记，算比率时只用随机那一份。

抽样单位也要想清楚。同一用户或同一对话的请求互相相关；按请求抽样、按请求算区间，会高估有效样本数。

### 没有参考答案时，查什么

线上能做的是无参考检查（reference-free）。先跑确定性检查：JSON 是否合法、引用的文档 ID 是否在检索结果里、工具参数是否通过 schema、输出里有没有疑似个人信息、`finish_reasons` 是否为 `length`。这些便宜、确定，可以对 100% 流量做。

规则写不出来的部分交给 judge：回答的主张是否有当次检索结果支持（需要记下或能按引用找回当时的上下文）、是否回应了用户真正的问题、该拒答时有没有拒答。每个 judge 只看一个标准，分开打分，判断方法见[分层评估](evaluation-stack.md)和 [LLM-as-a-Judge](llm-as-a-judge/README.md)。

### 什么样的 trace 送人工

常见有四个入口：judge 拿不准或分数在阈值附近；高风险类别（钱、健康、对外发送）；用户点踩或要求转人工；以及**与 judge 结果无关的随机审计**。

前三个入口挑出的都是“看起来有问题”的样本，只能告诉你被标记的样本里有多少真有问题。要估计 judge 放过了多少坏回答，必须从 judge 判“通过”的样本里随机抽一部分给人看。没有随机审计，judge 的漏判率永远是未知数。什么时候值得让人介入，见[人工参与](../06-systems/human-in-the-loop.md)。

### 把失败变成回归题

人工确认的失败，按这几步进离线集：

1. 固定能复现它的条件（输入、上下文快照或引用、各组件版本），脱敏并确认允许复制出生产环境；
2. 写下可检查的期望行为，例如“必须引用 2026 年的政策”，而不只是一段参考答案；标注切片和失败类型，去重；
3. 一部分进 CI 回归集，一部分留作不参与调参的留出集。

修好以后不要删掉这道题；但反复对着同一批题调 prompt，回归集会慢慢变成训练集。划分方法见[人工校准](llm-as-a-judge/calibration.md)，做成能拦截回归的 CI 见[评估报告与 CI](../practice/web-lab/eval-report.md)。

## 算一次：1% 抽样能看见多大的退化？

下面全是教学假设的虚构数字：

- 每天 100,000 条请求，按 trace id 随机抽 1% 给 judge，每天评判 $n=1000$ 条。
- 在人工标注的校准样本上，人判“通过”的回答，judge 有 $s=0.95$ 的比例也判通过；人判“不通过”的，judge 有 $t=0.85$ 的比例也判不通过。
- 按人工标准，真实通过率 $p=0.90$。

先注意，“judge 和人工一致率 94%”这一个数不够用。$p=0.90$ 时一致率是 $0.90\times0.95+0.10\times0.85=0.94$，可通过率一变，一致率也跟着变。要校正线上估计，得分别知道 $s$ 和 $t$。

judge 看到的通过率 $q$ 不等于 $p$：

$$
q = s\,p + (1-t)(1-p) = 0.95\times0.90 + 0.15\times0.10 = 0.87.
$$

反过来解出 $p$，就是筛查检验里估计患病率的经典校正（[Rogan & Gladen, 1978](https://pubmed.ncbi.nlm.nih.gov/623091/)）：

$$
\hat p = \frac{\hat q + t - 1}{s + t - 1},\qquad
\mathrm{SE}(\hat p) \approx \frac{1}{s+t-1}\sqrt{\frac{\hat q(1-\hat q)}{n}}.
$$

$\hat q$ 是当天被评判样本里 judge 判通过的比例。分母 $s+t-1=0.80$ 的意思是：真实通过率变 1 个百分点，judge 看到的只变 0.8 个点，所以误差要放大 $1/0.8=1.25$ 倍。代入 $\hat q=0.87$：$\hat p=(0.87+0.85-1)/0.80=0.90$，$\mathrm{SE}(\hat p)\approx0.0106/0.80=0.0133$，95% 区间约为 90% ± 2.6 个百分点。

能发现多小的退化？比较今天和昨天，两天各 1,000 条，差值的标准误是 $\sqrt2\,\mathrm{SE}(\hat p)$。按双侧 5% 检验、80% 把握（power），最小可检测效应（MDE）近似为：

$$
\mathrm{MDE}\approx(1.96+0.84)\sqrt{2}\,\mathrm{SE}(\hat p)=2.8\times1.414\times0.0133\approx0.053.
$$

也就是说，真实通过率要从 90% 掉到约 84.7%，日环比比较才有八成把握报警。换几个条件再算：

| 方案 | 每天评判条数 | 通过率的 95% 区间 | 日环比 MDE |
| --- | --- | --- | --- |
| 全部流量，抽 1% | 1,000 | ±2.6 个百分点 | 约 5.3 个百分点 |
| 全部流量，抽 5% | 5,000 | ±1.2 | 约 2.4 |
| 占流量 2% 的切片，抽 1% | 20 | ±18（正态近似已不可靠） | 约 37 |
| 同一切片，抽 5% | 100 | ±8.2 | 约 17 |
| 同一切片单独抽 25% | 500 | ±3.7 | 约 7.4 |

```python
from math import sqrt

def judge_monitor(n, q, s=0.95, t=0.85, z=1.96, z_power=0.84):
    """n judged traces; q = judge pass rate; s, t = judge agreement on human pass / human fail."""
    j = s + t - 1                         # share of a true difference that survives the judge
    p_hat = (q + t - 1) / j               # misclassification-corrected pass rate
    se = sqrt(q * (1 - q) / n) / j        # sampling error only; s and t treated as known
    mde = (z + z_power) * sqrt(2) * se    # day over day, two-sided 5%, about 80% power
    return p_hat, z * se, mde

rows = {n: judge_monitor(n, q=0.87) for n in (1000, 5000, 20, 100, 500)}
assert abs(rows[1000][0] - 0.90) < 1e-9
assert round(rows[1000][1], 3) == 0.026 and round(rows[1000][2], 3) == 0.053
assert round(rows[5000][2], 3) == 0.024 and round(rows[20][2], 2) == 0.37
for n, (p_hat, half_width, mde) in rows.items():
    print(f"n={n:5d}  p={p_hat:.3f} ±{half_width:.3f}  MDE≈{mde:.3f}")
```

从这张表能看出三件事。

**抽样率翻 5 倍，误差只缩小 $\sqrt5\approx2.2$ 倍。** judge 的费用和抽样率成正比，精度只按平方根改善。

**小切片几乎看不见。** 一个占流量 2% 的切片，比如某种语言或某个新功能，在 1% 抽样下每天只有 20 条，正态近似已经不可靠：20 条里 17 条通过，用 Wilson 区间算，judge 通过率的 95% 区间是 64% 到 95%。假设这个切片从 90% 跌到 60%，总体通过率只降 $0.02\times30=0.6$ 个百分点，远小于 5.3 甚至 2.4 的 MDE，总体曲线几乎是平的。这就是[人群混合](metric-robustness.md#population-mix)和[加权切片](ablation-and-slices.md)讲的问题在线上的样子。

**分层抽样把预算放到关键切片上。** 对这个切片单独抽 25%（每天 500 条），其余 98% 的流量仍抽 1%（980 条），每天一共 1,480 次 judge 调用，切片的 MDE 降到约 7.4 个百分点。代价是算总体通过率时必须按真实流量份额加权：切片在样本里约占三分之一，在真实流量里只占 2%，直接把 1,480 条平均会把它的分量放大约 17 倍。

这些数只算了抽样误差，下面几项还没算进去：

- $s$ 和 $t$ 本身是从有限的校准样本估出来的。若 $t$ 来自 100 条人工判“不通过”的样本，标准误约 0.036；$\hat p$ 对 $t$ 的敏感度是 $(s-q)/(s+t-1)^2=0.125$，折合约 0.45 个百分点。这主要让绝对水平偏移；只要 $s,t$ 不变，日环比里大部分会抵消。
- judge 在不同切片上的 $s,t$ 可能不同。用全局 $s,t$ 去校正一个 judge 不熟悉的语言切片，结果可能整体偏掉。
- 同一用户或对话的请求相关，有效样本比条数少，应按用户或对话聚类估计，见[指标稳健性](metric-robustness.md)。
- 同时盯 20 个切片、5 个指标，即使什么都没变，每天也会冒出几个“显著”。

另一种做法不依赖固定的 $s,t$：每个统计周期从随机样本里再抽一小部分请人标注，用 judge 在大样本上的平均，减去 judge 与人工在小样本上的平均差。这就是 [prediction-powered inference](https://arxiv.org/abs/2301.09633)（Angelopoulos 等，2023）的均值估计；论文的设定是人工样本和被评流量来自同一分布、独立抽取（随机审计正好满足），在这个前提下区间有效，不需要假设 judge 有多准。judge 越接近人工，区间越窄。代价是每个周期都要有新鲜的人工标签。

<!-- paper-figure:ppi-intervals -->

## 哪些信号该告警，多快告警

| 信号 | 在多少流量上算 | 怎样告警 |
| --- | --- | --- |
| 错误率（按 `error.type` 分组）、格式与工具参数校验失败率 | 全部 | 分钟级阈值 |
| 延迟 p50/p95、流式首块时间 | 全部 | 分钟级，按模型和路由分开 |
| 每条请求的 token 与费用 | 全部 | 小时级；同时看每个成功任务的费用，重试变多时它会上升 |
| `length` 结束原因的比例 | 全部 | 上升常意味着截断或输出变长 |
| 拒答率、安全标记率 | 全部（规则或小分类器）或抽样（judge） | 双向看：太高是过度拒答，太低可能是护栏失效；安全标记单条也可能要人看 |
| judge 通过率（校正后） | 抽样 | 按天或按周的窗口，阈值来自上一节的 MDE |
| 点踩、重新生成的比例 | 全部，但稀疏有偏 | 看趋势；没有点踩不等于满意，见[从反馈到训练目标](../01-data-and-feedback/feedback-to-objectives.md) |

每条请求都能算的信号可以快速告警；judge 给的质量信号是抽样的、有延迟的，阈值应该从上一节的计算来，而不是拍一个整数。每个要单独盯的切片都要单独算样本量，不然它的告警只是噪声。

## Judge 自己也会漂移

接着用上一节的数字。一个月后，judge 的提供方在同一别名下更新了模型，新版本对不合格回答更宽松：$t$ 从 0.85 降到 0.75。与此同时，产品真的退化了一点，真实通过率从 90% 降到约 88.6%。judge 看到的通过率是 $0.95\times0.886+0.25\times0.114\approx0.870$，仍是 87%。两件事刚好抵消，曲线是平的。

<!-- widget:step-judge-sampling -->

能做的事：

- 固定 judge 的模型版本（记下 judge 调用的 `gen_ai.response.model`）、rubric 版本和解析规则。任何一项变了，就当作换了一把尺子，重新建基线，不要把前后曲线接成一条。
- 维护一个锚点集（anchor set）：一批固定的人工标注样本，定期重跑，judge 有任何变化时也重跑。分开报告 $s$ 和 $t$，不只报一致率。
- 锚点集里要有足够多的“不通过”样本。随机流量里只有 10% 不通过，200 条锚点只有约 20 条能用来估计 $t$，标准误约 0.08，看不出 0.85 到 0.75 的变化。可以有意多放不通过的样本，并按切片补齐。
- 固定的锚点集会过时，要从最近的随机审计样本里持续补新题，也就是[人工校准](llm-as-a-judge/calibration.md)里的 fresh audit。哪些干扰会让 judge 跑偏，见[偏差检查](llm-as-a-judge/bias-and-workflow.md)。

## 隐私和保留：哪些东西不该记

OTel GenAI 约定的默认立场是保守的：系统指令、用户输入和模型输出被视为敏感且体积大，instrumentation **默认不应采集**，由使用者主动开启；在遥测量大或涉及敏感数据的生产环境，推荐把内容放到外部存储、span 上只记引用（见 v1.41.0 [spans 规范](https://github.com/open-telemetry/semantic-conventions/blob/v1.41.0/docs/gen-ai/gen-ai-spans.md)中采集指令、输入和输出的一节）。[OTel 的敏感数据说明](https://opentelemetry.io/docs/security/handling-sensitive-data/)也要求只收集服务于观测目的的数据。

放到线上评估里，常见的坑是：

- **不该记的。** 凭证、session token、API key（常藏在工具参数、请求头和 URL 里），支付和证件信息，与评估无关的联系方式和健康信息。工具参数和返回是自由文本，只过滤顶层字段不够。
- **judge 和审核者也是数据接收方。** 把对话发给 judge，等于又发给一个模型服务，它的解释还可能原样引用个人信息；标注队列里先脱敏，按任务限定谁能看原文。
- **保留期分层。** 聚合指标长期保留，原文和完整 trace 短期保留。复制进回归集的样本会逃出原来的保留期限，复制前要脱敏并记下来源；用户要求删除时，trace、judge 事件、标注队列和回归集都要能找到。
- **先确认工具实际做了什么。** 例如截至 2026-10，Langfuse 的[许可说明](https://langfuse.com/self-hosting/license-key)把数据保留策略（Data Retention Policies）和服务端脱敏（Server-Side Data Masking）列为需要企业许可证的自托管功能；用开源版时，这些要在应用或 SDK 一侧自己做。

选工具时先分清类别：埋点与采集（OTel SDK 与 Collector）、通用 tracing 后端（如 [Jaeger](https://www.jaegertracing.io/docs/latest/)）、能按抽样率在生产 trace 上跑 judge 的 LLM 平台（如 [Langfuse](https://langfuse.com/docs/evaluation/get-started/online)）、接进 CI 的离线 eval runner（如 [promptfoo](https://www.promptfoo.dev/docs/intro/)）；这些只是举例，不是推荐。再问几个能核对的问题：它接收哪一版 GenAI 约定？内容存在哪里、谁能看、留多久？judge 抽样率能否按切片设？标注结果能否导出成回归集？

## 最后：每种信号能说明什么，不能说明什么

| 信号 | 能告诉你 | 不能告诉你 |
| --- | --- | --- |
| 确定性检查（schema、引用 ID） | 这一条明确违反了某条规则 | 通过检查的回答内容是对的 |
| 单条 judge 分数 | 按这份 rubric，这条回答大概有没有问题 | 这条回答一定对或一定错 |
| 校正后的抽样 judge 通过率 | 总体质量在 MDE 以上的变化 | 小切片的退化；judge 自己有没有变 |
| 随机人工审计 | judge 的漏判与误判率，没有偏的质量估计 | 便宜、快速、能覆盖每个切片 |
| 触发式人工审核 | 被标记样本里有多少真有问题 | 总体失败率，或 judge 漏掉了什么 |
| 点踩、重新生成 | 有一部分用户不满意，以及变化趋势 | 沉默的大多数满不满意 |
| 一条失败的完整 trace | 这次失败经过了哪些步骤，原因假设是什么 | 这种失败有多常见 |

接着读：[Agent 可观测性](../06-systems/agent-observability.md) · [人工校准](llm-as-a-judge/calibration.md) · [指标稳健性](metric-robustness.md) · [评估总览](README.md)
