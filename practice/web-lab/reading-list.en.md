# Build a reading list that survives a refresh

[中文](reading-list.md) · **English**

> Local teaching implementation · Python 3.12 standard library + SQLite + plain JavaScript · Reviewed 2026-10. No model API, Node packages, or separate database service. Not a public-deployment template.

Type “BERT,” save it, and refresh. Is it still there? Text added only to the page disappears. Browser localStorage survives a refresh, but another browser won't see it. This time, Python receives the record and saves it in SQLite.

Not every list needs a backend. For one person using one browser, localStorage is often simpler. We're adding an API to understand how a frontend, server, and database work together.

## Run it before reading all the code {#run}

From the repository root, keep the database outside the repository:

```bash
python3 practice/web-lab/code/server.py --db /tmp/agi-reading-list.sqlite3
```

Open `http://127.0.0.1:8787/?lang=en`; the top-right button switches to Chinese. Type `127.0.0.1` exactly: the server accepts only that Host, so `localhost:8787` returns 403 (`host_not_allowed`). Don't open the HTML file directly or copy it into this site's `_site/` and expect an API to appear. On Windows, use a writable temporary directory for `--db`.

Add “BERT,” mark it Reading, and refresh. Stop Python with Ctrl+C, then restart using **the same database path**. The record should remain. `/tmp` is a practice location that the system may clean up, not a backup; use a persistent directory outside the repo if you want to keep your work.

[server.py](code/server.py) handles requests, [app.js](code/app.js) manages requests and UI state, and [index.html](code/index.html) plus [style.css](code/style.css) provide structure and styling. Only those three static files are served; the whole repository is not exposed as a file server.

## What happens when you click Save? {#request}

<figure class="worked-update worked-update--pairs">
<ol>
<li><small>Browser</small><strong>POST /api/notes</strong><span>Send {"title":"BERT"} as JSON. It is too early to say “Saved.”</span></li>
<li><small>Python</small><strong>Validate the input</strong><span>Require 1–120 title characters; reject extra fields, wrong types, and oversized bodies.</span></li>
<li><small>SQLite</small><strong>Write and commit</strong><span>Assign an ID and store the title, status, and creation time.</span></li>
<li><small>Browser</small><strong>Receive 201, then update</strong><span>Display the returned record instead of inventing an ID locally.</span></li>
</ol>
<figcaption>The frontend handles interaction, the server validates writes, and the database persists them. A refresh fetches records again with GET.</figcaption>
</figure>

Open your browser's Network panel, select `/api/notes`, and inspect its method, request body, status, and response. The page also includes a short request trace. To try the API separately:

```bash
curl -i http://127.0.0.1:8787/api/notes \
  -H 'Content-Type: application/json' \
  -d '{"title":"BERT"}'
curl http://127.0.0.1:8787/api/notes
```

The first command creates a record. Don't repeat POST just to see the result again: this version doesn't deduplicate creation, so identical titles receive different IDs. Read results with GET.

## Keep the data and API contracts small {#contract}

There is one table, `notes`. `id` identifies a record, `title` names it, `status` accepts only `to_read / reading / done`, and `created_at` is a server-written UTC timestamp. Titles aren't unique: two identically named records can still be updated separately.

| Request | Operation | Success response |
| --- | --- | --- |
| `GET /api/notes` | Read the latest 100 additions, descending by ID | `200`, `{"items": [...]}` |
| `POST /api/notes` | Create with `{"title":"BERT"}` | `201`, the new record |
| `PATCH /api/notes/1` | Set status with `{"status":"done"}` | `200`, the updated record |
| `GET /api/health` | Check that the request handler responds | `200`; doesn't establish database writability |

This PATCH **sets a state**, rather than cycling to the next one. Sending the same `done` request twice leaves it Read; sending POST twice creates another record. That's a concrete introduction to idempotency: whether retries change the intended result depends on the operation.

Client-side `required` and `maxlength` help with form entry. A curl request bypasses them, so Python validates too. SQL binds values with `?` rather than concatenating input; the page displays titles with `textContent` rather than interpreting them as HTML. These address different problems: treating input as SQL versus treating it as page code. [SQLite parameter binding](https://docs.python.org/3.12/library/sqlite3.html#how-to-use-placeholders-to-bind-values-in-sql-queries)

## Think beyond “Save failed” {#failures}

| A failure you can try | Expected behavior | Reason |
| --- | --- | --- |
| Send an empty title with curl | `400`, no new row | The form isn't the only entry point |
| Send status `finished` | `400`, existing state unchanged | The backend limits allowed states |
| Update an absent ID | `404` | Don't pretend an update succeeded |
| Stop the server, then refresh the list | A message that the result couldn't be confirmed | Connection failure isn't an empty list |
| Make storage temporarily unavailable | `503`, without internal file paths | An outage isn't “no records” |

`fetch()` doesn't reject simply because the server returned HTTP 400 or 500, so the code checks `response.ok`. If the connection fails, parsing fails, or eight seconds elapse, the page doesn't claim that nothing was saved. The transaction may have committed before the response was lost: refresh to check before retrying. [Fetch error handling](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch)

Controls are disabled during a request to prevent repeated clicks on this page. That isn't deduplication across tabs or devices. A stronger version can attach a request ID to creation, enforce a database uniqueness constraint, and return the same record on a retry with that ID. Tracking request IDs only in memory cannot guarantee deduplication after the process restarts.

## Put tests in CI, not just in your clicking routine {#tests}

```bash
python3 -m unittest discover -s site/tests -p 'test_web_lab.py' -v
```

Each test gets a temporary database and calls the WSGI application directly, without opening a port. Tests cover creation, updates, persistence across application instances, invalid inputs, and route/method restrictions. WSGI connects Python web servers to applications. The standard-library implementation makes requests easy to follow here; it isn't a recommendation for serving public traffic. [WSGI documentation](https://docs.python.org/3.12/library/wsgiref.html)

The site's existing `unittest discover -s site/tests` includes these tests, so another deployment workflow isn't needed. They don't replace real network or browser checks: inspect layout, in-flight messages, language switching, and the experience after stopping the server too.

## Why start with these choices? {#choices}

| Choice | What it avoids initially | When to reconsider |
| --- | --- | --- |
| Plain HTML / JS | A bundler and component framework | Compare options such as React when UI and shared state become complex |
| Standard-library WSGI | Hidden routing, JSON, and HTTP behavior | FastAPI or Flask can reduce repetition as routes grow; choose a production server separately |
| A SQLite file | Operating a separate database service | Consider alternatives such as PostgreSQL for shared multi-instance storage or high write concurrency |
| Same-origin page and API | Initial cross-origin and cookie configuration | Design CORS, authentication, and environment URLs when separating deployments |

This version intentionally omits deletion, pagination, accounts, and migrations. The 100-item limit is a display boundary, not database capacity. Proper pagination needs an ordering and cursor contract, not just a More button.

**Do not expose it publicly.** It binds to `127.0.0.1`, checks Host / Origin, and doesn't enable cross-origin access. Those reduce local misuse; they are not authentication. A multi-user service needs identities and record permissions, HTTPS, request protections such as CSRF defenses, rate limits, backups and restore, migrations, and monitoring. Port forwarding is not a substitute for that work.

[Back to CI/CD](static-and-ci.en.md) · [Choose an extension](next-projects.en.md)
