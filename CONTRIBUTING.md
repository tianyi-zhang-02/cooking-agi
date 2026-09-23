# 参与这个仓库 · Contributing

**中文** · [English below](#english)

欢迎一起完善这份笔记！发现错误、遇到看不懂的解释，或者想补充一个例子，都可以提出来。
**不必写一整篇文章才算参与**，提问和纠错也很有帮助。

贡献者名单会自动生成，不用单独登记。[幕后](contributors.md)页面有演职员表、贡献榜单和世界地图。
每位贡献者都有一个固定编号，比如 `CG 001`，按首次参与的顺序分配。

想出现在地图上，可以提一个 [issue](../../issues/new?template=add-me-to-the-crew.yml)，
填写 GitHub 用户名和国家或地区代码（ISO 3166-1 两位字母）。也可以填两个地方，贡献各计一半。
我会定期整理并更新 [`crew.toml`](crew.toml)；这个文件由我统一维护，请通过 issue 提交位置，不要直接修改。

地图亮度按贡献分为 5 档。位置是自愿填写的，不填写也会出现在贡献者名单里。

## 几条规矩

为了让这里保持一个适合学习和交流的环境，请注意：

- **讨论围绕技术和学习展开，不涉及政治话题。**
- **请勿发布广告或推广信息。**
- **可以用 AI 帮忙，但提交前请自己检查。** 确认事实、公式和链接没有问题，也读一遍文字是否通顺；
  对自己提交的内容负责。
- 违反以上约定的贡献者会被移出名单，后续 PR 也不再接受。

## 最容易上手的几件事

| 我想… | 怎么做 |
| --- | --- |
| 改一个错字或病句 | 在站点页面底部点「提交修改」，直接在 GitHub 上改，提 PR |
| 觉得某段没讲明白 | 开一个 issue，贴上那句话，说说你卡在哪一步，方便我们改进解释 |
| 补一个例子或反例 | 直接加到对应的 `.md` 里 |
| 加一篇论文笔记 | 复制 [`templates/paper-note.md`](templates/paper-note.md) 到 [`papers/`](papers/) |
| 加一个术语的中英对照 | 往 [`site/glossary.tsv`](site/glossary.tsv) 加一行，全站自动生效 |
| 给某个知识板块加考题 | 在对应笔记里加一节 `## 面试常见问题`，每道题写成 `<details><summary>问题</summary>答案</details>`；[考前速查](interview/questions.md)会自动收录，不用另外登记 |
| 改了一篇中文笔记 | 同一个 PR 里把对应的 `.en.md` 一起改掉，然后跑 `python3 site/paritycheck.py` 确认两版结构一致 |
| 写一整节新内容 | 先开 issue 聊聊放在哪一章，避免和正在写的内容重复 |

## 写作约定

每篇笔记尽量按同一个顺序展开：**先讲它是什么 → 为什么需要它 → 一个最简单的例子
→ 技术上的主要做法 → 它依赖什么假设 → 什么情况下会失效。**

写的时候可以参考下面几点：

1. **中英双语成对**。`foo.md` 是中文，`foo.en.md` 是英文。只写一种也可以合并，
   另一种可以由别人补——但请在 PR 里说明。
2. **术语不要在正文里手写括号注释**。加到 [`site/glossary.tsv`](site/glossary.tsv)，
   构建时会自动在中文页面里标注英文原文，并生成悬浮解释。
3. **公式用 LaTeX**：行内 `$...$`，独立 `$$...$$`。GitHub 和站点都能渲染。
4. **代码要能跑**。放进对应章节的 `code/` 目录，在 README 表格里加一行说明。
5. **图要能重现**。不要提交手画的示意图；写一个生成脚本放进 `code/`，
   把 `.svg` 输出到该章节的 `assets/`。参考
   [`00-foundations/code/make_figures.py`](00-foundations/code/make_figures.py)。
   这样改了模型图会跟着变，图和正文不会说两套话。
6. **入门和进阶分层，不要拆成两篇。** 主线保持五分钟能读完；推导、边角情况、
   踩过的坑放进折叠块。GitHub 和站点都能渲染它：

   ```markdown
   <details markdown="1">
   <summary><b>进阶</b>：为什么「接近 one-hot」就等于没有梯度</summary>

   （这里照常写 markdown，公式、代码、链接都行）

   </details>
   ```

   `markdown="1"` 让站点解析内部内容，GitHub 会忽略这个属性但同样能渲染。
   判断标准：**去掉这个块，主线还成立吗？** 不成立就说明它不该被折叠。
7. **不确定就写不确定**。「我不知道为什么」比编一个解释好。
8. **不提交公司实操。** 雇佣、招聘或具体公司场景里的实现、证据、复盘和案例，
   即使已经脱敏也留在私有笔记。这里只接受能由公开来源或可复现实验独立支撑的内容。

## 目录约定

数字前缀就是阅读顺序，GitHub 按字母排序，所以文件列表本身就是课程大纲。

```
00-foundations/     基础：模型在算什么
01…03               输入：数据、记忆、多模态
04…06               模型怎样做事：检索、post-training、系统
07…08               怎样判断做得好不好：评估、体验
09-personal-agi/    终点
papers/ templates/  论文笔记与模板
site/               站点构建（见下）
```

新开一章：建目录、放 `README.md`，然后在 [`site/nav.toml`](site/nav.toml) 加一个
`[[section]]` 块。章节里的文件会被自动发现，不需要逐个登记。

## 本地预览站点

```bash
pip install markdown pygments
python site/build.py --serve
```

打开 <http://localhost:8000>。笔记本身是纯 markdown，没有 front matter——
在 GitHub 上直接看和在站点上看是同一份内容。

推到 `main` 之后 GitHub Actions 会自动重新构建并部署。

---

<a name="english"></a>

# Contributing

[中文](#参与这个仓库--contributing) · **English**

These are public learning notes, not a textbook. Some of it is wrong, some of it is
explained badly, and a lot of it is missing. **Pointing at any of those counts** —
including typos.

The contributor list builds itself — nothing to sign up for. Scroll down
[Behind the notes](contributors.en.md) for the credits, the board and a world map. Everyone
gets a number too, `CG 001` and so on: first come, first numbered, and it stays yours.

Want a patch of the map lit? Open an
[issue](../../issues/new?template=add-me-to-the-crew.yml) with your GitHub login and a
country or region (ISO 3166-1 alpha-2). Two, if you split your time — they count half
each. I read them now and then and add people to [`crew.toml`](crew.toml) by hand. I keep
that file myself, so PRs that edit it don't get merged.

Brightness comes in five steps: one commit already lights a place, more burns brighter up
to a ceiling, and the first few stay clearly ahead. Skipping all of it is fine — your name
is on the board either way.

## A few rules

Not many:

- **Tech only, no politics.** These are study notes. Keep politics out of them — that much
  is easy.
- **Not an ad slot.** No promotion, no recruiting, no traffic funnelling.
- **AI drafts are fine; unread ones are not.** Draft with AI all you like. What I can't use
  is a page nobody read before sending — check the facts, the formulas, the links, the
  tone. Your name is the one on it.
- Break any of those and your name comes off for good, and I stop merging your PRs.

## Easiest ways in

| I want to… | How |
| --- | --- |
| Fix a typo or an awkward sentence | Hit "Suggest an edit" at the bottom of any page |
| Say a section didn't land | Open an issue, quote the sentence, say where you got stuck — more useful than fixing it |
| Add an example or a counterexample | Edit the `.md` directly |
| Add a paper note | Copy [`templates/paper-note.en.md`](templates/paper-note.en.md) into [`papers/`](papers/) |
| Add a term to the glossary | One line in [`site/glossary.tsv`](site/glossary.tsv); it applies site-wide |
| Add interview questions to a knowledge block | Add an `## Interview questions` section to the relevant note (`## 面试常见问题` in the Chinese file), one `<details><summary>question</summary>answer</details>` per question; the [quick review](interview/questions.en.md) picks them up automatically |
| Change a Chinese note | Update the matching `.en.md` in the same PR, then run `python3 site/paritycheck.py` to confirm the two versions still match in structure |
| Write a whole new section | Open an issue first so we don't collide |

## Conventions

Each note tries to follow the same arc: **what it is → why it's needed → the simplest
example → the main technical approaches → what it assumes → how it fails.**

1. **Bilingual pairs.** `foo.md` is Chinese, `foo.en.md` is English. One-language PRs are
   fine — just say so.
2. **Don't hand-annotate terms inline.** Add them to [`site/glossary.tsv`](site/glossary.tsv);
   the build annotates Chinese pages automatically and generates the hover glosses.
3. **LaTeX for math**: `$...$` inline, `$$...$$` display. Renders on GitHub and on the site.
4. **Code must run.** Put it in that chapter's `code/`, add a row to its README table.
5. **Figures must be reproducible.** No hand-drawn diagrams — write a generator in `code/`
   that emits `.svg` into the chapter's `assets/`. See
   [`00-foundations/code/make_figures.py`](00-foundations/code/make_figures.py). Change the
   model, rerun, and the figure follows — so the picture and the prose can't disagree.
6. **Layer depth in place; don't split into two notes.** Keep the main line to five
   minutes and put derivations, edge cases and hard-won traps in a collapsible block.
   Both GitHub and the site render it:

   ```markdown
   <details markdown="1">
   <summary><b>deeper</b>: why "close to one-hot" means "no gradient"</summary>

   Ordinary markdown in here — math, code and links all work.

   </details>
   ```

   `markdown="1"` tells the site to parse the contents; GitHub ignores the attribute
   and renders it anyway. The test: **does the main line still stand without this
   block?** If not, it shouldn't have been collapsed.
7. **Say when you're unsure.** "I don't know why this works" beats an invented explanation.
8. **No company practice notes.** Implementations, evidence, retrospectives, and cases from
   employment, recruiting, or a specific company stay private even after redaction. Public
   contributions must stand independently on public sources or reproducible experiments.

## Local preview

```bash
pip install markdown pygments
python site/build.py --serve
```

Then open <http://localhost:8000>. The notes stay pure markdown with no front matter, so
what you read on GitHub and what you read on the site are the same file.

Pushing to `main` rebuilds and redeploys via GitHub Actions.
