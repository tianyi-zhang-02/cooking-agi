# 目录怎么扩充

导航仍然只改 `site/nav.toml`，不用给每个页面手写一份侧栏，也不用为了整理目录移动文章。

## 四层就够了

| 层级 | 用来放什么 | 例子 |
| --- | --- | --- |
| `category` | 读者这次来做什么 | 学习笔记、工程实践、求职、论文 |
| `group` | 一个长期的主题 | 大模型基础、Post-training、LeetCode 方法 |
| `section` | 能一起学习和复习的一章 | 核心机制、模型家族、LeetCode 方法 |
| page | 一个具体问题 | 怎样判断该用 DFS 还是 BFS？ |

学习区的 `zone` 按任务分成基础与模型、训练与应用、代码与题解、系统设计。它是导航分区，不再多加一层折叠；文章顶部只显示同一区的入口，避免一长排按钮。一个主题只有一章时，不重复显示同名标题；一章只有一篇时，直接显示文章。不要为了增加层级而增加层级。

新增文章放进对应目录，在 section 的 `order` 里安排顺序；需要短标题时用 `label`。新章节增加 `[[section]]`，挂到已有 group。真的出现新的长期主题，再增加 group，并更新 `site/collaboration.toml` 的审核归属和生成的 CODEOWNERS。已有文章跨目录归类可以用 `include`，不改公开 URL。

## 读者怎样找回来

- **目录**只展开当前大板块，切板块用顶部导航，不再另放一排重复入口。主题和章节都能折叠，打开文章时自动展开当前路径。
- 搜索统一从顶部进入，也可以按 `/`；搜索框内可选当前板块或全站，用方向键选择，Enter 打开。
- 文章标题上方的章节名可以点回导读，文末也有本章目录和上一篇、下一篇。顺序来自 `nav.toml`，不额外维护一份推荐列表。
- **我的阅读**放收藏和最近打开的笔记。收藏上限 100 篇，最近记录保留 8 篇；“打开”不代表“读完”。正文标题下也能直接收藏，不必先打开侧栏。
- 长文记住上次停留的小节；重新打开时可以点「继续上次位置」，不会自动把页面跳走。中英文收藏共用，阅读位置分别保存，避免两版标题不一致时跳错。
- 侧栏在「备份与迁移」里提供收藏导出、导入。备份不含阅读历史；导入只认本站已有笔记，合并去重，不覆盖已有收藏，不接受外部链接。
- 清空阅读记录会同时清空最近打开和小节位置，不删除收藏或章末复习记录。手机上继续使用菜单抽屉，不新增悬浮工具栏。

## 保存与降级

只用当前站点的 localStorage，不请求账号、不上传阅读行为，也不跨设备同步。同一篇笔记的中英文使用同一个源文件标识，切换语言不会产生两份收藏。URL 和显示标题始终来自当前构建的目录，不信任存储里的链接。

保存不可用时，本页收藏交互仍可用，并提示离开前导出。阅读位置最多保留 100 个「笔记 × 语言」条目，只存小节标识，不存正文；旧小节已不存在时，不显示继续入口。JavaScript 不可用时仍能阅读目录、章节链接和全文；个人阅读功能隐藏。新增内容由构建自动加入目录和收藏索引，不需要修改前端列表。

## 改完检查什么

```bash
python -m unittest discover -s site/tests -p 'test_sidebar.py'
python -m unittest discover -s site/tests
python site/build.py
```

检查中文和英文、首页和深层文章、手机和电脑。手动试一次：搜索 → 打开结果 → 返回本章；收藏 → 切语言 → 取消收藏；滚动到某一节 → 重新打开 → 点继续；导出收藏 → 导入并确认没有重复。检查损坏或过大的导入文件不会清空原数据。手机打开菜单 → Tab 切换 → Escape 关闭，确认旧链接、复习卡和图解都还正常。

## Maintenance summary

`nav.toml` remains the single navigation source: category → group → chapter → note. Keep URLs stable, use `include` for cross-folder placement, and map groups to reviewers in `collaboration.toml`. Top navigation switches categories; the sidebar provides folding, bookmarks, and recently opened notes. Chapter links work without JavaScript. Reading positions are local, language-specific, and restored only on request. Bookmark backups contain source IDs only and merge against the current catalog, never imported URLs. Test both languages, keyboard use, storage failures, and narrow screens.
