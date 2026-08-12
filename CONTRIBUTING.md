# 参与这个仓库 · Contributing

**中文** · [English below](#english)

这是一份公开的学习笔记，不是教科书。它一定有讲错的地方、讲得太绕的地方，
和还没写的地方——**指出任何一处都算贡献**，改一个错字也算。

每次合并之后，站点底部的贡献者列表会自动更新，按最近提交排序。不需要额外登记。

## 最容易上手的几件事

| 我想… | 怎么做 |
| --- | --- |
| 改一个错字或病句 | 在站点页面底部点「提交修改」，直接在 GitHub 上改，提 PR |
| 觉得某段没讲明白 | 开一个 issue，贴上那句话，说你卡在哪儿——这比修好它更有价值 |
| 补一个例子或反例 | 直接加到对应的 `.md` 里 |
| 加一篇论文笔记 | 复制 [`templates/paper-note.md`](templates/paper-note.md) 到 [`papers/`](papers/) |
| 加一个术语的中英对照 | 往 [`site/glossary.tsv`](site/glossary.tsv) 加一行，全站自动生效 |
| 写一整节新内容 | 先开 issue 聊聊放在哪一章，避免撞车 |

## 写作约定

每篇笔记尽量按同一个顺序展开：**先讲它是什么 → 为什么需要它 → 一个最简单的例子
→ 技术上的主要做法 → 它依赖什么假设 → 什么情况下会失效。**

具体一点的规矩：

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
6. **不确定就写不确定**。「我不知道为什么」比编一个解释好。

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

The contributor list at the bottom of every page regenerates on each build, ordered
by most recent commit. Nothing to sign up for.

## Easiest ways in

| I want to… | How |
| --- | --- |
| Fix a typo or an awkward sentence | Hit "Suggest an edit" at the bottom of any page |
| Say a section didn't land | Open an issue, quote the sentence, say where you got stuck — more useful than fixing it |
| Add an example or a counterexample | Edit the `.md` directly |
| Add a paper note | Copy [`templates/paper-note.en.md`](templates/paper-note.en.md) into [`papers/`](papers/) |
| Add a term to the glossary | One line in [`site/glossary.tsv`](site/glossary.tsv); it applies site-wide |
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
6. **Say when you're unsure.** "I don't know why this works" beats an invented explanation.

## Local preview

```bash
pip install markdown pygments
python site/build.py --serve
```

Then open <http://localhost:8000>. The notes stay pure markdown with no front matter, so
what you read on GitHub and what you read on the site are the same file.

Pushing to `main` rebuilds and redeploys via GitHub Actions.
