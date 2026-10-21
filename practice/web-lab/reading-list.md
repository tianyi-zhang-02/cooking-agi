# 做一个刷新后还在的阅读清单

**中文** · [English](reading-list.en.md)

> 本机教学实现 · Python 3.12 标准库 + SQLite + 原生 JavaScript · 最近核对：2026-10。不需要模型 API、Node 包或数据库服务；不是公网部署模板。

在输入框写下“BERT”，点保存，再刷新。还在吗？如果只把文字加进页面，刷新就没了；写进浏览器的 localStorage，可以跨刷新保留，但换个浏览器就看不到；这次我们让 Python 接收记录，存进 SQLite。

这不是说每个清单都需要后端。单人、单浏览器的功能，用 localStorage 往往更省事。这里加 API，是为了看清前端、服务端和数据库怎么配合。

## 先运行，不急着读完代码 {#run}

从仓库根目录运行，数据库文件放在仓库外：

```bash
python3 practice/web-lab/code/server.py --db /tmp/agi-reading-list.sqlite3
```

打开 `http://127.0.0.1:8787/?lang=zh`；右上角可以切英文。地址要写 `127.0.0.1`：服务端只接受这个 Host，用 `localhost:8787` 打开会得到 403（`host_not_allowed`）。不要双击 HTML 文件，也不要把它塞进本站 `_site/` 后期待 API 自动出现。Windows 可以把 `--db` 换成自己可写的临时目录路径。

试这几步：添加“BERT”→ 改成“在读”→ 刷新 → Ctrl+C 关闭 Python → 用**同一个数据库路径**重启。记录应该仍在。`/tmp` 只是练习位置，可能被系统清理，不是长期备份；保留练习结果时选仓库外的持久目录。

[server.py](code/server.py) 是服务端，[app.js](code/app.js) 管请求和页面状态，[index.html](code/index.html) 与 [style.css](code/style.css) 管结构和样式。服务端只允许读取这三个固定静态文件，不会把整个仓库变成文件服务器。

## 点一次保存，到底发生了什么？ {#request}

<figure class="worked-update worked-update--pairs">
<ol>
<li><small>浏览器</small><strong>POST /api/notes</strong><span>发出 JSON：{"title":"BERT"}。此时还不能显示“已保存”。</span></li>
<li><small>Python</small><strong>检查输入</strong><span>标题必须是 1–120 个字符；不接受多余字段、错误类型或过大的请求体。</span></li>
<li><small>SQLite</small><strong>写入并提交</strong><span>数据库分配 ID，保存标题、状态与创建时间。</span></li>
<li><small>浏览器</small><strong>收到 201，再更新列表</strong><span>使用服务端返回的记录，不自己猜一个 ID。</span></li>
</ol>
<figcaption>前端负责交互，后端确认什么可以保存，数据库负责持久化。刷新时用 GET 重新取记录。</figcaption>
</figure>

打开开发者工具的 Network 面板，选中 `/api/notes`，看请求方法、请求体、状态码和返回值。页面底部也有简短的请求记录。想单独检查 API，可以用：

```bash
curl -i http://127.0.0.1:8787/api/notes \
  -H 'Content-Type: application/json' \
  -d '{"title":"BERT"}'
curl http://127.0.0.1:8787/api/notes
```

第一条会添加记录。不要因为想多看一次结果就重复发送 POST——这个最小版本没有创建去重，同样标题会生成不同 ID。查询结果用 GET。

## 数据和接口，约定得小一点 {#contract}

数据库只有 `notes` 一张表。`id` 是稳定标识，`title` 是标题，`status` 只接受 `to_read / reading / done`，`created_at` 是服务端写入的 UTC 时间。标题不要求唯一：两条同名记录仍可分别修改。

| 请求 | 做什么 | 成功返回 |
| --- | --- | --- |
| `GET /api/notes` | 读最近添加的 100 条，按 ID 倒序 | `200`，`{"items": [...]}` |
| `POST /api/notes` | 用 `{"title":"BERT"}` 创建 | `201`，新记录 |
| `PATCH /api/notes/1` | 用 `{"status":"done"}` 设置状态 | `200`，更新后的记录 |
| `GET /api/health` | 确认请求处理器能响应 | `200`，不证明数据库可写 |

PATCH 在这里是**设置状态**，不是“切换到下一状态”。同一个 `done` 请求执行两次，结果仍是已读；POST 创建两次却会多一条。可以先从这点理解幂等性（idempotency）：重试是否改变了原本想要的结果，取决于具体操作。

客户端的 `required` 和 `maxlength` 只改善填写体验。直接用 curl 就能绕过它们，所以 Python 还要校验。SQL 用 `?` 绑定参数，不拼接用户输入；页面用 `textContent` 显示标题，不把标题当成 HTML。两者解决的是不同问题：一个避免输入被当作 SQL，另一个避免它被当作页面代码。[SQLite 参数绑定](https://docs.python.org/3.12/library/sqlite3.html#how-to-use-placeholders-to-bind-values-in-sql-queries)

## 失败时，比“保存失败”多想一步 {#failures}

| 可以自己制造的情况 | 应该看到什么 | 为什么 |
| --- | --- | --- |
| 用 curl 发空标题 | `400`，数据库不增加记录 | 前端不是唯一入口 |
| 把状态发成 `finished` | `400`，原状态保留 | 后端限制可接受的状态 |
| 更新不存在的 ID | `404` | 不假装更新成功 |
| 服务停止后点刷新 | 页面提示无法确认结果 | 连接失败，不等于清单为空 |
| 数据库暂时无法读写 | `503`，不回传内部文件路径 | 故障不是业务上的“没有记录” |

`fetch()` 收到 HTTP 400/500 时，不会仅因为这个状态码就自动抛异常，所以代码检查 `response.ok`。网络断开、响应解析失败或超过 8 秒时，页面不会声称“肯定没保存”。服务端可能已经提交，只是响应没到：先刷新确认，再决定是否重试。[Fetch 的错误处理](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch)

按钮在请求中会暂时禁用，防止本页连续点击，但这不是跨页面、跨设备的去重保证。要做得更可靠，可以给创建操作加请求 ID，在数据库里加唯一约束，让相同 ID 的重试返回同一条记录。只把请求 ID 存在内存里，不能保证进程重启后仍能去重。

## 测试放到 CI，而不是只靠手点 {#tests}

```bash
python3 -m unittest discover -s site/tests -p 'test_web_lab.py' -v
```

测试为每个用例创建临时数据库，直接调用 WSGI 应用，不占端口：检查新增、改状态、重新创建应用后的持久化、非法输入、方法和路径限制。WSGI 是 Python Web 服务器与应用之间的接口；这里使用标准库实现，方便看清请求，而不是推荐它承担公网流量。[WSGI 文档](https://docs.python.org/3.12/library/wsgiref.html)

这些测试会被本站已有的 `unittest discover -s site/tests` 收进去，无需再新建一套部署工作流。它们不等于真实网络或浏览器测试：页面布局、请求中的提示、语言切换和服务中断后的体验，还要在浏览器检查。

## 为什么先这样选？ {#choices}

| 当前选择 | 省了什么 | 什么时候换 |
| --- | --- | --- |
| 原生 HTML / JS | 不先配置打包器和组件框架 | 复杂界面、共享状态变多时，再比较 React 等方案 |
| 标准库 WSGI | 看得见路由、JSON 和 HTTP 返回 | 业务路由变多时，用 FastAPI / Flask 等减少重复；生产服务器另选 |
| SQLite 文件 | 不需要另外运行数据库服务 | 多实例共享、高写并发等需求出现时，评估 PostgreSQL 等方案 |
| 同源提供页面和 API | 不先处理跨域与 cookie 配置 | 分开部署时再设计 CORS、认证与环境地址 |

本例刻意没有删除接口、分页、账户和数据库迁移。100 条限制是展示范围，不是数据库最多能存 100 条；生产分页不能只加一个“更多”按钮而不约定排序和游标。

**不要直接开放公网。** 服务器绑定 `127.0.0.1`，检查 Host / Origin，不提供跨域访问；这些只是在本地减少误用，不是用户认证。真正多人使用还需要身份与记录权限、HTTPS、CSRF 等请求防护、限流、备份恢复、迁移和运行监控。尤其别把端口转发出去，就当成已经完成部署。

[回到 CI/CD](static-and-ci.md) · [挑一个扩展练习](next-projects.md)
