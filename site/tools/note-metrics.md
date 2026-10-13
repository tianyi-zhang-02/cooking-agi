# 阅读与收藏计数 / Note counts

状态：前端、统计协议与本地预览已实现；默认没有连接线上服务。
Status: UI, API contract, and local preview are available; no production service is configured by default.

## 本地验证 / Local preview

```sh
python site/build.py
python site/tools/metrics_preview.py --db /tmp/cooking-note-metrics.sqlite --port 18992
```

打开 `http://127.0.0.1:18992/` 的任意正文页。页面底部会注明「本地预览 · 非线上数据」。展开统计说明，勾选参与统计，再阅读 15 秒或手动收藏。数据保存在 SQLite，停止、重启预览不会重置。不要把这个数据库放进 `_site` 或提交到 Git。

Open a note, expand “About these counts,” and opt in. A read takes 15 visible seconds; a manual save updates the save count. The SQLite file persists across restarts. Keep it outside `_site` and Git. The preview binds only to loopback; do not expose this development server publicly.

## 计数含义 / Semantics

- 中文与英文、页面改名后的重定向共用稳定的 note ID；迁移路径时沿用 `progress-id-overrides.json` 的 ID。
- 总阅读：启动统计后，同一浏览器、同一篇、每个 UTC 日最多一次；不是独立人数或完整读完人数。
- 总收藏：参与统计的浏览器当前仍保留的收藏。重复发送同一状态不会累加，取消会减去。
- 本月阅读按 UTC 自然月；本月收藏是本月新增、且尚未取消的收藏，不是累计点击量。
- 不上传旧记录、导入的备份或整个收藏列表。只统计启用后用户主动产生的动作。
- 标识每篇独立、保存在浏览器，不随学习进度导出。换设备、清空浏览器会改变标识，因此是近似计数。
- 默认不参与；尊重 DNT / GPC。关闭后不再发送新动作，不删除已计入的聚合数据。取消收藏只有在参与统计时才能同步。
- 网络断开时本地收藏仍有效。当前页面保留待重试动作；离线关页可能漏计，不声称精确一致。

Both languages use the stable note ID. Reads are deduplicated per browser/note/UTC day. Saves represent active bookmarks, and monthly saves are active bookmarks most recently added this month. Importing backups never generates count events. Opt-in is off by default; DNT/GPC suppress submissions. Per-note random identifiers are separate from progress backups. These are approximate browser counts, not user counts: cleared storage, offline exits, and device changes can cause over- or under-counting.

## API

`GET /api/note-metrics/<note-id>` returns:

```json
{"version":1,"noteId":"note-78153ce088c6824e","month":"2026-10","mode":"local","reads":1,"monthReads":1,"saves":1,"monthSaves":1}
```

数字只是协议示例，不写入页面。未取得有效数据时显示 `—`，连接成功且确实无数据时才显示 `0`。如果已显示过计数后连接中断，保留上次结果并提示暂未同步。
The numbers are contract examples, not seeded UI data. Counts show `—` until a valid response arrives; only a valid zero from the service is displayed as `0`. A later connection failure preserves the last result and shows a sync warning.

`POST` to the same URL accepts only one of:

```json
{"kind":"view","token":"0123456789abcdef0123456789abcdef"}
```

```json
{"kind":"bookmark","token":"0123456789abcdef0123456789abcdef","saved":true}
```

The server chooses the date, validates the note against the public catalog, hashes tokens, and serializes updates in a transaction. No client-supplied date, count delta, or URL is accepted. Reads retry idempotently; bookmark updates set a state rather than incrementing a counter. The local service refuses cross-origin writes and does not log request bodies, IPs, or tokens in application logs.

## 上线前 / Before production

线上实现在 [`services/counter`](../../services/counter/README.md)：Cloudflare Worker + Durable Object，实现同一协议，另外提供学习小伙伴的在线人数；部署步骤和隐私、限流说明都在那里。
The production implementation lives in [`services/counter`](../../services/counter/README.md) (Cloudflare Worker + Durable Object, same protocol, plus presence counts for the study buddy).

主站仍可放在 GitHub Pages。另选统计服务，实现同一协议，再设置 `site/static/note-metrics-config.json` 的 HTTPS endpoint。线上响应的 `mode` 才能设为 `public`，不要把本地测试数据迁到线上。服务故障不能阻止站点和本地收藏使用。

Before enabling a production endpoint, configure HTTPS, exact CORS origin allowlists, request-size limits, abuse/rate limits, persistent database backups, and hosting log retention. Do not put database credentials or admin keys in the client. Anonymous identifiers prevent casual duplicate clicks, not a determined bot creating many new identities. Production hosting, costs, retention, and abuse controls still require maintainer review; this preview is not a production deployment.
