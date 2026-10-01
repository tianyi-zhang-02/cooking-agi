# 维护「聊聊」 / Maintaining Discussions

这里讨论学习、工作和生活，不属于求职栏目，也不做知识测验。网站文章走 PR 审核；读者留言留在 GitHub Discussions，不进入源码。

This section is separate from Career. Posts go through PR review. Reader comments live in GitHub Discussions, not in the source tree.

## 发一篇新话题 / Add a topic

每篇只围绕一个主题：说清一个想法，举少量例子，结尾留一个开放问题。相邻的想法可以互相链接，不要堆成一篇人生指南。引用经典要附原文和出处，把原句、自己的翻译和生活感悟区分开。

Keep one topic per post: one perspective, a few examples, and an open question. Link related posts rather than combining them into a life guide. Cite primary texts and distinguish quotations, your translations, and personal interpretations.

1. 在 GitHub Discussions 的 Announcements 分类发布主题和开场看法，记下真实的 discussion number。这个分类让维护者发主题，读者回复；其他想法可以先发到 General。
2. 新建 `discussions/<slug>.md` 和 `<slug>.en.md`。中文自然地说，英文独立写顺；不要编造作者的经历或读者留言，也不要搬运第三方帖子里的隐私。
3. 在 `site/discussions.toml` 加一个 `[[topic]]`：两种语言的标题、标签、摘要，以及作者、日期、slug、discussion number。可在 `site/nav.toml` 设置顺序和短标题。
4. 跑构建、双语与泄漏检查，提交 PR。审核看内容、隐私和链接；话题不要求大家意见一致。

Create the GitHub thread first, then a bilingual Markdown pair and a `[[topic]]` entry. Use the real discussion number. Both languages must point to the same number. Posts are curated through PRs; General remains available for reader suggestions.

不要因为改标题、换语言或改网址另建一条 thread。编号绑定能保留原来的留言。配置不包含 token。

Do not create another thread when renaming or translating a post. Number-based mapping preserves its replies. No tokens belong in the config.

## 站内留言 / In-page comments

当前 `embed_enabled = false`：页面提供真实的 GitHub 留言入口，不显示假的输入框或留言数。

To turn on the optional embed:

1. 仓库管理员在 <https://github.com/apps/giscus> 审核并安装 App，只选择 `cooking-agi`。不要给其他库授权。
2. 在 <https://giscus.app/> 确认 repo ID / category ID，保留 `mapping = number`。
3. 将 `site/discussions.toml` 的 `embed_enabled` 改成 `true`，本地确认中英页面、明暗主题、登录与回复，再按正常 PR 流程上线。
4. 未点击「加载留言」时不加载 giscus。点击后才连接第三方；留言公开，发言需要 GitHub 登录。无法加载或关闭 JavaScript 时，原始 GitHub 链接仍可用。

The repository owner must explicitly authorize the giscus App for this repository. Only then enable the embed and verify a real read/sign-in/reply flow. Do not claim embedded commenting is live before that check. Public comments load on demand; the direct GitHub thread is always available.

## 维护讨论 / Moderation

不同意见保留。垃圾广告、人身攻击和隐私泄露在 GitHub 处理；需要时锁定或删除违规内容。不要把薪资、学历和职级变成人的排名。站内短文是个人观点，不是心理或财务诊断。

Keep disagreement. Moderate spam, personal attacks, and privacy violations on GitHub, including locking threads where necessary. These are personal perspectives, not psychological or financial diagnoses.
