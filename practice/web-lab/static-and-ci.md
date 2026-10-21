# 从改一篇笔记，到自动检查和发布

**中文** · [English](static-and-ci.en.md)

> 最近核对：2026-10 · 以本站当前的 `site/build.py` 和 `.github/workflows/site.yml` 为例，不把仓库里使用的 action 版本称为“最新”。练习在自己的 fork 上完成。

你改了一句解释，本地看着没问题。合并后，英文入口却打不开了。靠“我记得检查过”很容易漏掉这种事；CI 的价值就是把确定能检查的部分，每次再做一遍。

**CI（continuous integration，持续集成）**是改动进入共同代码库时运行的自动检查。**CD**可能指持续交付，也可能指持续部署：前者把可发布的产物准备好，发布可以有人确认；后者还把通过门槛的版本自动上线。本站配置会在符合条件的 main 构建后尝试部署；仓库环境如要求人工批准，仍需等待批准。

## 先看本站的文件去了哪里 {#site-flow}

<figure class="worked-update worked-update--pairs">
<ol>
<li><small>源文件</small><strong>文章与目录</strong><span>Markdown 写正文，nav.toml 决定章节顺序。</span></li>
<li><small>构建</small><strong>Python 生成网页</strong><span>build.py 把正文放进模板，生成搜索索引与静态文件。</span></li>
<li><small>产物</small><strong>_site/</strong><span>这是准备发布的 HTML、CSS、JavaScript，不是运行中的 Python 服务。</span></li>
<li><small>访问</small><strong>Pages → 浏览器</strong><span>Pages 提供文件，浏览器显示页面并执行交互。</span></li>
</ol>
<figcaption>读者打开网页时，不会为这次访问重新执行 build.py。文章改了，需要重新构建并发布。</figcaption>
</figure>

[GitHub Pages 是静态托管](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)，不能把下篇的 Python API 也交给它运行。静态并不等于没有交互：搜索、主题切换、动画都可以在浏览器里完成。

## 第一步：在本机改一处，看见结果 {#local}

先 fork 仓库，再 clone **自己的 fork**。下面从仓库根目录运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install markdown pygments numpy
python site/build.py
python -m http.server 8000 --bind 127.0.0.1 --directory _site
```

Windows 可用 `.venv\Scripts\activate` 激活环境。浏览器打开 `http://127.0.0.1:8000/`；用 Ctrl+C 停止预览。这个命令只适合本地预览，不是生产服务器。不要提交 `.venv/`，在自己的 fork 里将它加入忽略列表。

改 `practice/web-lab/README.md` 中的一句正文，重新执行构建，再刷新。接着切到英文版：中文改了，英文不会自动替你改。两版都更新后，跑这些检查：

```bash
python -m unittest discover -s site/tests -p 'test_web_lab.py'
python site/collaboration.py
python site/paritycheck.py --strict
python site/leakcheck.py
```

这些命令各管一件事：阅读清单测试、审核归属配置、双语结构、已知敏感内容规则。正式工作流还会跑完整测试集。**检查通过不等于内容一定正确**，双语结构相同也不能证明翻译准确。

## 第二步：让 PR 先检查，但不发布 {#ci}

读[本站工作流](../../.github/workflows/site.yml)，不用重新抄一套。它监听向 main 的 PR、main 的 push、手动运行和定时任务。`build` 依次获取源码、安装依赖，跑协作配置检查、完整测试、泄漏检查和双语检查，然后构建，再用 `site/htmlcheck.py` 检查生成的 `_site/`；只有 main 上的非 PR 运行才上传产物。`deploy` 依赖 `build` 成功。

| 场景 | 自动做什么 | 不代表什么 |
| --- | --- | --- |
| 打开或更新 PR | 测试和构建 | PR 不是线上预览，也不会由本工作流部署 |
| main 的合格构建 | 上传 `_site/`，部署 job 等待前置条件 | 构建成功不等于 URL 已可访问 |
| 测试失败 | 当前构建停止，部署不能继续 | 不会替你修复代码或回退已经上线的旧版本 |
| main 的定时任务 | 重新检查、构建并尝试部署 | 没有 push 不代表没有运行 |

先在自己的 fork 中确认 Actions 已启用；如果要公开预览，在仓库 **Settings → Pages → Build and deployment** 选择 GitHub Actions，并检查 `github-pages` 环境的分支与批准规则。不确定会公开哪些文件，就先只跑 CI，别开发布。[官方配置说明](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)

试一次故意失败：在自己的练习分支，把 `test_web_lab.py` 里创建记录预期的 `201` 改成 `200`，开 PR，看 Actions 中哪个步骤失败。再改回正确值。这样你会看到，那盏红灯具体意味着**API 约定被一条测试检查到了**。不要把错误提交合进主分支。

`needs: build` 只保证工作流内的先后关系。要禁止失败的 PR 合并，还要由维护者配置 branch protection / ruleset 的 required checks；单写 YAML 并不会自动设置仓库合并规则。

如果把 fork 当自己的站点发布，还要核对 `site/nav.toml` 的 `repo` 和 `origin`：前者决定“改这一页”和反馈链接指向哪个仓库，后者决定页面的正式地址（canonical）和分享图地址。站内页面之间用相对路径互链，`build.py` 目前不读 `base_url`，项目路径不用另外配置。先看构建结果里的链接，别让自己的“改这一页”仍指向上游。维护者与贡献署名也要按实际情况整理，保留来源，不照搬本站的审核关系。

## 第三步：知道发布了什么，也知道怎么退回 {#delivery}

构建产物（artifact）就是准备交给托管服务的一包文件。源码和产物不是同一个东西：这次 CI 应该检查将要发布的那份产物，而不是另外手工复制一份目录。

| 需要留下的线索 | 出问题时怎么用 |
| --- | --- |
| commit SHA 与 Actions run | 找到这次上线对应哪份代码、哪些检查 |
| 构建日志与部署结果 | 区分生成失败、上传失败和发布失败 |
| 发布后的实际 URL | 做一次 smoke test：页面、语言切换、导航和静态资源是否正常 |
| 上一份已知可用的源码 | 回退有问题的改动，再通过正常流程构建发布 |

例如 CSS 请求 404，先看 Network 里的 URL：项目站通常带仓库路径，本地根路径能打开，不代表部署后的路径也对。不是每次白屏都该重新安装依赖。

在自己的仓库里，回退可以做成一个撤销特定提交的 PR，再走检查和部署。这里不教 force push 改历史。静态站回退通常比较简单；有数据库的应用还要考虑新旧 schema 是否兼容，不能只退代码。

## 不要为了演示省掉这些边界 {#boundaries}

- PR 的代码是不可信输入。不要为了“方便”给来自 fork 的代码部署密钥，也不要用 `pull_request_target` 直接执行它。
- 权限按 job 分开：测试通常只读源码，Pages 部署才需要相应写权限。第三方 action 的完整 commit SHA 比浮动标签更便于锁定执行内容；升级仍要核对版本和审阅改动。[GitHub 安全说明](https://docs.github.com/en/actions/reference/security/secure-use)
- 不把 `.env`、数据库、API key 或真实用户数据放进 `_site/`。写进前端 JavaScript 的密钥不是秘密。
- 第一轮不加 Docker 或多环境。先让失败能被发现、发布能被追踪；遇到环境不一致，再决定容器是否值得。

只想学 CI，到这里就可以停。想看真正的服务端和数据保存，接着做[阅读清单](reading-list.md)。
