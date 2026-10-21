# 分数涨了，为什么 CI 还是红的？

**中文** · [English](eval-report.en.md)

> 可运行的小项目 · Python 3.12 标准库 · 最近核对：2026-10。用六道虚构问题演示评估报告，不调用模型，不需要 API key。输入里的答案与通过标签都是为讲解手写的。

你改了一个问答系统，六道题从答对四道变成五道。看总分，似乎该上线了。点开结果却发现：原来会老实说“不知道”的一道题，现在开始编答案。

这篇不再加一个复杂的评估框架。我们做个小工具：读入两次结果，逐题对比，生成能直接打开的 HTML；如果碰到事先约定不能接受的错误，就返回失败状态，让 CI 拦下来。

## 先看这一处变化 {#example}

题目来自虚构的图书馆问答。有些给出了材料，有些故意没有。比如 q5：“材料没有提供周日的开放安排，图书馆周日开门吗？”预期是说明材料不足，而不是猜一个时间。

| 看哪里 | 旧版本 | 新版本 | 发生了什么 |
| --- | --- | --- | --- |
| 六道题整体 | 4/6 | 5/6 | 提高约 16.7 个百分点 |
| 有材料的四题 `evidence` | 2/4 | 4/4 | q1、q2 从错变对 |
| 无材料的两题 `no-evidence` | 2/2 | 1/2 | q5 开始无依据地断言周日开门 |

**改善两题、退步一题，总共多对一题。** “提高 16.7 个百分点”和“相对提高 25%”说的是不同分母；这里直接显示 4/6 → 5/6，先把数量讲清楚。

这次先定两条规则：关键题必须全部通过，整体答对的题数不能下降。q5、q6 被我们标为关键题，因此新版本虽然总分更高，仍会被拦下。真实项目要按需求定规则，不要照搬这六道题或临时改门槛让结果好看。

### CI 的绿灯，和模型真的变好，是两回事 {#gate-and-evidence}

把 q5 修好后，这套规则会放行。它告诉我们的是“这六道题满足了约定”，不是“真实用户遇到的问题都会答得更好”。

<div class="worked-table" markdown="1">

| 你想确认什么 | 这份报告能告诉你什么 | 还需要什么 |
| --- | --- | --- |
| 已知的关键错误有没有再出现 | q5、q6 这次是否通过 | 有代表性的新问题、遗漏的失败类型 |
| 这批结果哪里变了 | 两题改善、一题退步，以及原始答案 | 检查 passed 标签是否判对 |
| 换一批问题还能不能保持提升 | 六题不足以支持这个结论 | 未用于调试的数据，以及合适的配对分析 |

</div>

即使暂时把六题当作独立同分布的样本，“改善 2 题、退步 1 题”的双侧精确 McNemar 检验也得到 p = 1。这并不说明新旧模型一样，只说明这点数据还区分不了它们；本例的题目还是手写的，更不能当真实部署分布的测量。推导和另外两组对照放在[同样涨分的两种情况](../../07-evaluation/metric-robustness.md#paired-binary)，不需要先学完才能把项目跑起来。

所以比较器不自动套一个 p 值门槛。先拦住已经知道不能接受的错误，再单独研究整体收益，两个问题分别回答。

## 三份输入，分别负责什么 {#inputs}

<figure class="worked-update worked-update--pairs">
<ol>
<li><small>固定题目</small><strong>suite.json</strong><span>ID、题目、预期行为、分类与关键题标记。</span></li>
<li><small>旧结果</small><strong>baseline.json</strong><span>每道题的输出与 passed 标签，作为对照。</span></li>
<li><small>新结果</small><strong>candidate.json</strong><span>同一套题的新输出与标签，按 ID 配对，不按行号。</span></li>
<li><small>比较结果</small><strong>HTML + JSON</strong><span>看总数、分类和具体变化，再判断是否满足规则。</span></li>
</ol>
<figcaption>分类和关键题名单来自固定题集，不能由候选结果自行修改。三个文件带相同的 suite 版本；修改题目或判分规则后，应重新生成两边结果再比较。</figcaption>
</figure>

代码在 [eval_report.py](code/eval_report.py)，数据在 [eval-fixtures](code/eval-fixtures/suite.json)。一条结果很短：

```json
{"id": "q5", "passed": false, "output": "Yes, it opens every Sunday."}
```

`passed` 不是程序读完 `output` 后推断的。这个练习把**判分**和**比较**分开：比较器相信你给的标签，所以错误标签也会得到一份格式漂亮、结论错误的报告。接真实系统时，要记录输出来源、模型与 prompt 版本、judge 或人工判分规则；这里的 `run` 只起名字标识作用，不证明模型真的运行过。

## 先跑一次会被拦下的结果 {#run}

从仓库根目录运行，不用安装额外包。以下是 macOS / Linux 命令；Windows 可以指定一个新的本地输出目录。

```bash
LAB=practice/web-lab/code
RUN_DIR=$(mktemp -d)
python3 "$LAB/eval_report.py" \
  --suite "$LAB/eval-fixtures/suite.json" \
  --baseline "$LAB/eval-fixtures/baseline.json" \
  --candidate "$LAB/eval-fixtures/candidate.json" \
  --out-dir "$RUN_DIR/blocked" --lang zh
```

它会打印 `BLOCKED`，并以 **1** 退出。这是预期结果，不是脚本坏了。打开刚打印路径里的 `report.html`；报告不依赖服务器，也不会往外发请求。退步的 q5 默认展开，其他题可以点击或按 Enter 展开，比较两边答案。

接着保持题集和旧结果不变，换成修正后的教学结果：

```bash
python3 "$LAB/eval_report.py" \
  --suite "$LAB/eval-fixtures/suite.json" \
  --baseline "$LAB/eval-fixtures/baseline.json" \
  --candidate "$LAB/eval-fixtures/candidate-fixed.json" \
  --out-dir "$RUN_DIR/fixed" --lang zh
```

这次是 `PASS`，退出码 **0**，新结果为 6/6。只是例子里把 q5 的输出和标签一起改好了，**不是训练了一轮模型**。`--lang en` 生成英文界面；题目和答案作为原始数据保留，不自动翻译。

每次用一个新的输出目录，脚本拒绝覆盖已有目录，免得把上次的报告误认成这次。`report.json` 保存逐题结果、规则判断，以及三份输入的 SHA-256。哈希能帮你核对输入有没有变化，不能替你判断答案对不对。

## 遇到这些情况，不要悄悄算一个分数 {#validation}

| 试着改什么 | 工具应该怎么做 | 为什么 |
| --- | --- | --- |
| 删掉新结果中的 q5 | 退出 2，不生成新报告 | 缺题不能通过取交集被藏起来 |
| 复制一条相同 ID 的结果 | 退出 2 | 不能任意保留其中一条 |
| 把 `false` 写成字符串 `"false"` 或数字 `0` | 退出 2 | 明确只接受 JSON 布尔值，不靠隐式转换 |
| 把结果列表倒过来 | 结果不变 | 配对依据是 ID，不是文件中的位置 |
| 把答案写成 `<script>…</script>` | 报告只显示文字 | 模型输出是不可信数据，不能当 HTML 执行 |
| 改动题集的版本名，却不重跑两边结果 | 退出 2 | 拦住明显的题集版本错配 |

版本名检查不防止有人忘改或故意复用名称；输入哈希也不证明结果确实来自该题集。真正接入 pipeline 时，应让生成阶段记录实际题集内容哈希，而不是让人手填一个相同的名字。

脚本还有 2 MiB / 文件、最多 1,000 道题的教学上限，不是大规模评估系统。输入无效时不会创建输出目录；遇到磁盘写入错误则以 2 退出，可能留下不完整目录，不把它当作有效报告。先用单元测试把这些边界跑一遍：

```bash
python3 -m unittest discover -s site/tests -p 'test_eval_report.py'
```

## 接进 CI：失败时，也要能看到报告 {#ci}

在**自己的 fork** 中，把下面保存为 `.github/workflows/eval-lab.yml`。它只运行比较器，不发布网站。本例用固定结果演示流程；不是在每次 PR 上真的运行一遍 LLM。真实评估要在比较前增加独立的生成和判分步骤，并保持测试集、baseline 与门槛受审阅保护。

```yaml
name: Eval lab
on: [pull_request, workflow_dispatch]
permissions:
  contents: read
jobs:
  compare:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          persist-credentials: false
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Compare supplied results
        run: |
          python practice/web-lab/code/eval_report.py \
            --suite practice/web-lab/code/eval-fixtures/suite.json \
            --baseline practice/web-lab/code/eval-fixtures/baseline.json \
            --candidate practice/web-lab/code/eval-fixtures/candidate.json \
            --out-dir eval-output
      - name: Keep the report, including a failed comparison
        if: ${{ !cancelled() }}
        uses: actions/upload-artifact@v4
        with:
          name: eval-report
          path: eval-output/
          if-no-files-found: error
```

比较步骤退出 1，job 就会失败；上传报告仍尝试执行，但不会把红灯改绿。下载 artifact 后打开 HTML，可以知道是哪个例子触发了规则。输入错误导致没有报告时，上传步骤也会报错，先看前面的具体错误。退出 0 / 非 0 的含义和 `!cancelled()` 的行为见 [GitHub 退出码说明](https://docs.github.com/en/actions/how-tos/create-and-publish-actions/set-exit-codes)及[状态条件](https://docs.github.com/en/actions/reference/workflows-and-actions/expressions#status-check-functions)。

这些 action 标签用于说明配置，不宣称是最新版本；正式使用时按仓库策略固定完整 commit SHA 并定期更新。[artifact 文档](https://github.com/actions/upload-artifact/tree/v4)说明了文件上传和保留方式。**报告也是数据副本**：别把真实用户对话、密钥或未授权数据放进去，也不要自动公开部署。这个例子没有新 workflow 写入本站，只是一份供 fork 练习的配置。

## 为什么第一版不用数据库、图表库和 LLM judge？ {#tradeoffs}

| 选择 | 先得到什么 | 暂时做不到什么 |
| --- | --- | --- |
| 固定 JSON，不接数据库 | 能把同一份输入交给别人重跑 | 没有多用户、结果查询和大批量存储 |
| Python 生成静态 HTML | 无服务、无前端依赖，失败报告也能带走 | 不提供在线刷新或编辑判分 |
| 人工写好的通过标签 | 先把配对、计数和门槛讲清 | 不会自动评判开放式回答 |
| 原始输出转义后展示 | 避免把回答执行成网页代码 | 不负责删除输出里本就包含的敏感信息 |

HTML 转义调用 Python 标准库的 [`html.escape`](https://docs.python.org/3.12/library/html.html)；JSON 解析额外拒绝重复字段和非标准数值，避免默默覆盖，见 [`json` 的相关行为](https://docs.python.org/3.12/library/json.html)。先看清数据，再决定是否值得接更多组件。

## 跑通以后，再问两个问题 {#next}

第一，报告显示“没变”的题，是两边都对，还是两边都错？这里只说通过标签没变，不是两个答案文本相同；展开后还要看输出。关键题如果一直错，本工具仍会拦下，不会因为“没退步”就放行。

第二，非关键的一类题退步，但其他题改善抵消了它，会不会放行？**可能会。** 当前门槛没有要求每个分类都不下降，分类表先把问题呈现出来。如果这不符合你的产品要求，就在看到结果前约定分组规则，再补测试；不要看到哪版占优才选哪种指标。

接下来可以用 [LLM-as-a-Judge](../../07-evaluation/llm-as-a-judge/README.md) 讨论标签怎么来，或用[评估协议](../../07-evaluation/benchmark-protocols.md)处理数据划分、重复试验和不确定性。这个小项目解决的是“怎么让结果可检查”，不是“怎么证明模型全面更好”。

[返回小项目导读](README.md)
