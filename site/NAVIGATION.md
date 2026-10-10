# 目录怎么扩充

导航仍然只改 `site/nav.toml`，不用给每个页面手写一份侧栏，也不用为了整理目录移动文章。

## 四层就够了

| 层级 | 用来放什么 | 例子 |
| --- | --- | --- |
| `category` | 读者这次来做什么 | 基础与原理、面试准备、工程实践、求职 |
| `group` | 一个长期的主题 | 大模型基础、Post-training、LeetCode 方法 |
| `section` | 能一起学习和复习的一章 | 核心机制、模型家族、LeetCode 方法 |
| page | 一个具体问题 | 怎样判断该用 DFS 还是 BFS？ |

基础区的 `zone` 分成理解模型、训练与检验、推理与应用。代码与题解由独立的面试准备入口组织，系统设计放在工程实践，不再把所有内容都塞回基础区。`zone` 是导航分区，不再多加一层折叠；文章顶部只显示同一区的入口，避免一长排按钮。一个主题只有一章时，不重复显示同名标题；一章只有一篇时，直接显示文章。不要为了增加层级而增加层级。

新增文章放进对应目录，在 section 的 `order` 里安排顺序；需要短标题时用 `label`。新章节增加 `[[section]]`，挂到已有 group。真的出现新的长期主题，再增加 group，并更新 `site/collaboration.toml` 的审核归属和生成的 CODEOWNERS。已有文章跨目录归类可以用 `include`，不改公开 URL。

## 让阅读顺序和文件夹解耦

`section.sequence` 决定同主题下的章节顺序，`order` 决定章内顺序。`section.home` 指向实际导读，用于返回本章；`intro_zh/en` 用一句话说明本章解决什么问题、需要什么前置知识。虚拟章节的 `include` 使用真实源文件路径，不能把同一篇重复注册到两章。

学习首页的完整目录由 `study-atlas` 组件从导航生成，不再手写一份容易漏项的文章列表。预训练和推理各有独立导读；旧 `deep-dives` 地址仍保留为两条路线的入口。`site/catalog-baseline.json` 保护重排前已经发布的地址，测试要求它们继续存在。

参考资料的已知范围放在 `site/reference-coverage.toml`。它区分有正文、部分讲解和待补，不等于全文核对结果。具体阅读进度、数学与代码待核项记在 `CONTENT_COVERAGE.md`，不能从一个目录标题推断完整覆盖。

## 读者怎样找回来

README 只负责介绍项目和给出几条阅读入口，不再维护另一份完整目录。完整的文章顺序仍由 `nav.toml` 生成。读者可在 `learn/coverage.md` / `.en.md` 看本轮改动与待补项；维护者用 `CONTENT_RELEASE_CHECKLIST.md` 核对发布条件，两者都不能把“有正文”写成“已全部验收”。

- 仓库首页使用英文 `README.md`，中文介绍在 `README.zh.md`。网站 `/` 与 `index.html` 默认英文；中文首页为 `index.zh.html`，原来的 `index.en.html` 保留跳转。其他笔记的中英文地址不改。
- 顶部 **中文 / EN** 直接切换同一篇笔记；第一次进入有一个可关闭的小提示，不遮住整页、不自动抢焦点。选择只存在当前浏览器，回到根入口时沿用；直接打开某种语言的文章不被强制跳走。没有 JavaScript 时根入口仍是英文，语言链接仍可用。

- **目录**只展开当前大板块，切板块用顶部导航，不再另放一排重复入口。主题和章节都能折叠，打开文章时自动展开当前路径。
- 学习区标出当前分区，主题旁显示篇数。“定位本章”展开并定位当前文章，“收起其他”只保留当前路径。导航状态使用 v2；旧版展开状态不会重新展开整棵目录，收藏和阅读记录不受影响。
- 搜索统一从顶部进入，也可以按 `/`；搜索框内可选当前板块或全站，用方向键选择，Enter 打开。
- 文章标题上方的章节名可以点回导读，文末也有本章目录和上一篇、下一篇。顺序来自 `nav.toml`，不额外维护一份推荐列表。
- **我的阅读**放收藏和最近打开的笔记。收藏上限 100 篇，最近记录保留 8 篇；“打开”不代表“读完”。正文标题下也能直接收藏，不必先打开侧栏。
- 长文记住上次停留的小节；重新打开时可以点「继续上次位置」，不会自动把页面跳走。中英文收藏共用，阅读位置分别保存，避免两版标题不一致时跳错。
- 侧栏在「备份与迁移」里提供收藏导出、导入。备份不含阅读历史；导入只认本站已有笔记，合并去重，不覆盖已有收藏，不接受外部链接。
- 清空阅读记录会同时清空最近打开和小节位置，不删除收藏。手机上继续使用菜单抽屉，不新增悬浮工具栏。

## 保存与降级

Quant 源文件暂时保留，但不在 `nav.toml` 注册，因此不生成页面，也不进入搜索。不要从已发布文章链接到这些未发布文件。构建时清理旧输出，避免旧页面残留。

只用当前站点的 localStorage，不请求账号、不上传阅读行为，也不跨设备同步。同一篇笔记的中英文使用同一个源文件标识，切换语言不会产生两份收藏。URL 和显示标题始终来自当前构建的目录，不信任存储里的链接。

保存不可用时，本页收藏交互仍可用，并提示离开前导出。阅读位置最多保留 100 个「笔记 × 语言」条目，只存小节标识，不存正文；旧小节已不存在时，不显示继续入口。JavaScript 不可用时仍能阅读目录、章节链接和全文；个人阅读功能隐藏。新增内容由构建自动加入目录和收藏索引，不需要修改前端列表。

## 改完检查什么

```bash
python -m unittest discover -s site/tests -p 'test_sidebar.py'
python -m unittest discover -s site/tests
python site/build.py
```

检查中文和英文、首页和深层文章、手机和电脑。手动试一次：搜索 → 打开结果 → 返回本章；收藏 → 切语言 → 取消收藏；滚动到某一节 → 重新打开 → 点继续；导出收藏 → 导入并确认没有重复。检查损坏或过大的导入文件不会清空原数据。手机打开菜单 → Tab 切换 → Escape 关闭，确认旧链接、正文例题和图解都还正常。

## Maintenance summary

`nav.toml` remains the single navigation source: category → group → chapter → note. Keep URLs stable, use `include` for cross-folder placement, and map groups to reviewers in `collaboration.toml`. Top navigation switches categories; the sidebar provides folding, bookmarks, and recently opened notes. Chapter links work without JavaScript. Reading positions are local, language-specific, and restored only on request. Bookmark backups contain source IDs only and merge against the current catalog, never imported URLs. Test both languages, keyboard use, storage failures, and narrow screens.
