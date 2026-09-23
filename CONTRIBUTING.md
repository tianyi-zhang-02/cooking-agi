# 加入我们！ · Contributing

**中文** · [English](#english)

不用先写出一整篇。改一句不顺的话、指出公式哪里不对，或者问一个好问题，都很欢迎。

## 先选一件小事

- **错字、引用、翻译**：直接 fork → 修改 → PR。
- **解释看不懂或发现问题**：用[提问与纠错](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=note-feedback.yml)，贴页面链接和具体段落。
- **新文章、交互图或目录调整**：先开[内容提案](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=proposal.yml)，避免重复劳动，再开 draft PR。
- **想长期参与**：先看[板块分工与审核约定](community/README.md)。没有认领的板块由维护者接审，不要求你随时在线。

每个 PR 尽量只做一件事。写清改了什么、依据是什么、怎么检查的。常规合并需要一位非作者的相关 CODEOWNER 批准、检查通过、讨论处理完；大范围调整才走提案，具体例外和分歧处理见上面的协作约定。

## 内容怎么写

1. 从具体问题和小例子讲起，再讲原理、假设与局限。长推导可以放折叠块；不要把主线藏起来。
2. `.md` 对应中文，`.en.md` 对应英文。尽量同 PR 更新；只写一种也欢迎，但标明缺哪部分、怎么补。结构检查只是提醒。
3. 术语统一放在 `site/glossary.tsv`，不反复手写括号解释。公式用 `$...$` 或 `$$...$$`。
4. 代码放在相应章节的 `code/`，写清运行方法；图尽量提供 SVG 或生成脚本，外部素材说明来源和使用许可。
5. 给事实和关键结论附公开来源。实验写明条件，区分已验证、假设和计划；没有结果也可以记录。
6. AI 可以协助，但你要读过、验证过，并在 PR 里说明重要使用方式。不能拿模型输出充当独立审核。
7. 不上传公司内部资料、未公开的面试内容、凭据或他人隐私；即使脱敏，也不把具体公司的内部案例搬过来。

## 目录和本地检查

原理与教学演示放在现有学习章节；实践入口在 `practice/`，开源笔记在 `open-source/`；求职在 `career/` / `interview/`，论文在 `papers/`。只调整导航时不要搬文件，避免已有链接失效。

```bash
pip install markdown pygments
python site/collaboration.py
python -m unittest discover -s site/tests
python site/leakcheck.py
python site/paritycheck.py
python site/build.py --serve
```

打开 <http://localhost:8000>。改动布局或交互时，同时看看手机宽度、键盘操作和另一种语言。只有 main 会部署，PR 检查不会发布站点。

新增导航 group 或调整 reviewer 时，修改 `site/collaboration.toml`，然后运行 `python site/collaboration.py --write`，把生成的 `.github/CODEOWNERS` 一起提交。**配置文件不授予 GitHub 权限**；新 reviewer 须本人同意，并由仓库所有者确认相应权限后再加入。审核配置、构建和协作规则由维护者审核。

## 署名与参与

代码与笔记署名来自 main 的提交和 `Co-authored-by`；合并时请保留。审阅、提问、翻译和纠错也可以在[贡献记录](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=add-me-to-the-crew.yml)里附链接，经本人同意后补到[幕后](contributors.md)。提交数不是质量排名，也不决定权限。

地图只收自愿提供的国家或地区，不要填地址。自己可以提 PR 更新或移除 `crew.toml` 的条目，维护者核实后合并。感谢记录在 `site/collaboration.toml` 的 `acknowledgements`：每人一项，包含 `login`、`zh`、`en` 和 `evidence`（本仓库 issue / PR 链接数组）。没有记录时保持空数组，不编造贡献或提前列朋友的名字。

尊重不同意见，不做人身攻击，不发广告。一般问题先沟通；严重或反复滥用才限制参与。详细规则和需要拍板时的做法，都在[协作约定](community/README.md)。

---

<a id="english"></a>

# Contributing

Fix a sentence, question a formula, or ask about something unclear. You don't need a complete article to contribute.

## Start small

- **Typos, references, translations:** fork → edit → PR.
- **Questions or corrections:** use [note feedback](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=note-feedback.yml) with a page and passage.
- **New notes, visualizations, or navigation changes:** open a [proposal](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=proposal.yml) before a draft PR to avoid duplicate work.
- **Ongoing involvement:** see [area contacts and review rules](community/README.en.md). Maintainers cover unassigned areas; constant availability isn't expected.

Keep each PR focused and explain the change, evidence, and checks. Routine merges require one relevant CODEOWNER approval from a non-author, passing checks, and resolved discussions. Proposals are for broader changes; the collaboration guide explains exceptions and disagreements.

## Writing notes

1. Start with a concrete question and example, then explain the mechanism, assumptions, and limits. Fold long derivations, not the main argument.
2. Pair Chinese `.md` with English `.en.md` where possible. One-language work is welcome with an explicit follow-up plan. Structure checks are advisory.
3. Put recurring terminology in `site/glossary.tsv`. Use `$...$` or `$$...$$` for math.
4. Put runnable examples in the section's `code/` directory with instructions. Prefer SVG or reproducible figures; credit external assets and check reuse permission.
5. Cite public evidence, state experimental conditions, and distinguish verified findings, hypotheses, and plans. Negative results are useful too.
6. Verify AI-assisted work yourself and disclose material assistance in the PR. Model output is not independent review.
7. Do not submit internal company materials, non-public interview details, credentials, or personal data, including redacted company-specific cases.

## Structure and local checks

Concepts and teaching demos stay in the learning chapters. `practice/` is the practice entry, `open-source/` holds open-source notes, `career/` and `interview/` cover preparation, and `papers/` holds paper notes. Don't move files just to reorganize navigation; preserve existing URLs.

```bash
pip install markdown pygments
python site/collaboration.py
python -m unittest discover -s site/tests
python site/leakcheck.py
python site/paritycheck.py
python site/build.py --serve
```

Visit <http://localhost:8000>. For UI changes, check narrow screens, keyboard navigation, and both languages. Only main deploys; PR checks do not publish the site.

For new navigation groups or reviewers, update `site/collaboration.toml`, run `python site/collaboration.py --write`, and include the generated `.github/CODEOWNERS` in the PR. **Configuration does not grant GitHub access.** Reviewers must consent and the owner must verify their permissions first. Maintainers review changes to collaboration rules, reviewer configuration, and builds.

## Credit and participation

Preserve main-branch authors and `Co-authored-by` trailers when merging. Reviews, questions, translations, and corrections can also receive credit on [Behind the notes](contributors.en.md): submit a [contribution record](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=add-me-to-the-crew.yml) with evidence and the person's consent. Commit counts do not determine quality or access.

Map locations are optional countries or regions, not addresses. You may propose updates or removal of your own entry in `crew.toml`; maintainers verify requests. Non-commit credits live in the `acknowledgements` array in `site/collaboration.toml`, one record per person with `login`, `zh`, `en`, and `evidence` (issue/PR URLs in this repository). Leave the array empty until there are verified, consented credits.

Be respectful; no personal attacks or advertising. Normally discuss problems first, restricting participation only for serious or repeated abuse. See the [collaboration guide](community/README.en.md) for the full rules and decision process.
