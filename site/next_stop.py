"""Consent-gated, bilingual community milestones with a static HTML fallback."""

import html
import json
import re
import tomllib
from collections import Counter
from pathlib import Path

import company_icons

ROOT = Path(__file__).resolve().parents[1]
ROLES = {
    "software": ("软件工程 · SWE / SDE", "Software engineering · SWE / SDE"),
    "ml": ("机器学习工程 · MLE", "Machine learning · MLE"),
    "research": ("研究 · RS / RE", "Research · RS / RE"),
    "data": ("数据科学与分析", "Data science & analytics"),
    "product": ("产品", "Product"),
    "other": ("其他", "Other"),
}
EMPLOYMENT = {
    "internship": ("实习", "Internship"),
    "full-time": ("全职", "Full-time"),
    "contract": ("合同制", "Contract"),
    "other": ("其他", "Other"),
}
MILESTONES = {
    "offer": ("拿到 offer", "Received an offer"),
    "started": ("已入职", "Started the role"),
    "completed": ("已结束", "Completed"),
}
MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def text_field(value, limit):
    return isinstance(value, str) and 0 < len(value.strip()) <= limit


def validate(data):
    if set(data) != {"version", "companies", "records"} or data["version"] != 1:
        raise ValueError("Unsupported next-stop data format")
    companies, records = data["companies"], data["records"]
    if not isinstance(companies, dict) or not isinstance(records, list):
        raise ValueError("Expected companies object and records list")
    names = set()
    canonical_ids = set()
    for key, name in companies.items():
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", key) or not text_field(name, 80):
            raise ValueError("Invalid company ID or display name")
        normalized = " ".join(name.split()).casefold()
        if normalized in names:
            raise ValueError("Duplicate company display name; reuse the company ID")
        names.add(normalized)
        preset = company_icons.resolve(key, name)
        canonical_id = preset["id"] if preset else normalized
        if canonical_id in canonical_ids:
            raise ValueError("Duplicate company alias; reuse the canonical company ID")
        canonical_ids.add(canonical_id)
    seen = set()
    required = {"id", "start_date", "company", "role", "employment", "milestone", "consent"}
    for record in records:
        if not isinstance(record, dict) or not required <= record.keys() or record.keys() - required - {"name", "github", "note", "title", "source_issue", "owner_submission"}:
            raise ValueError("Missing or unexpected next-stop fields")
        record_id = record["id"]
        if not isinstance(record_id, str) or not re.fullmatch(r"NS-[0-9]{4,}", record_id) or record_id in seen:
            raise ValueError("Invalid or duplicate milestone ID")
        seen.add(record_id)
        if not isinstance(record["start_date"], str) or not re.fullmatch(r"20[0-9]{2}(?:-(?:0[1-9]|1[0-2]))?", record["start_date"]):
            raise ValueError("Use the start year or year-month, without inventing a missing month")
        for field, choices in (("company", companies), ("role", ROLES), ("employment", EMPLOYMENT), ("milestone", MILESTONES)):
            if not isinstance(record[field], str) or record[field] not in choices:
                raise ValueError(f"Invalid {field}: {record_id}")
        if record["consent"] is not True:
            raise ValueError("Each record needs explicit publication consent")
        if "owner_submission" in record:
            owner = tomllib.loads((ROOT / "site/nav.toml").read_text(encoding="utf-8"))["site"]["owner_login"]
            if record["owner_submission"] is not True or record.get("github") != owner or "source_issue" in record:
                raise ValueError("Direct owner submissions must identify the configured owner, without a fabricated issue")
        elif type(record.get("source_issue")) is not int or record["source_issue"] < 1:
            raise ValueError("Reader submissions need a source issue")
        if "name" in record and not text_field(record["name"], 60):
            raise ValueError("Invalid optional display name")
        if "github" in record and (not isinstance(record["github"], str) or not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?", record["github"])):
            raise ValueError("Invalid opt-in GitHub username")
        for field, limit in (("note", 400), ("title", 100)):
            if field in record and (not isinstance(record[field], dict) or set(record[field]) != {"zh", "en"} or not all(text_field(value, limit) for value in record[field].values())):
                raise ValueError(f"Optional {field} must have short Chinese and English versions")
    return data


def load():
    return validate(json.loads((ROOT / "community/outcomes.json").read_text(encoding="utf-8")))


def ordered(records):
    return sorted(records, key=lambda record: (-int(record["start_date"][:4]), -int(record["start_date"][5:] or "0"), record["id"]))


def start_date_label(record, language):
    start_date = record["start_date"]
    year = start_date[:4]
    if len(start_date) == 4:
        label = year + (" 年 · 月份未提供" if language == "zh" else " · month not shared")
    else:
        month = int(start_date[5:])
        label = f"{year} 年 {month} 月" if language == "zh" else f"{MONTHS[month - 1]} {year}"
    if record["milestone"] == "offer":
        return ("预计 " if language == "zh" else "Expected ") + label
    return label


def person_label(record, language):
    if record.get("github"):
        return "@" + record["github"]
    return record.get("name") or ("一位读者" if language == "zh" else "A reader")


def person_html(record, language):
    if not record.get("github"):
        return html.escape(person_label(record, language))
    login = html.escape(record["github"], quote=True)
    name = f'<span class="stop-display-name">{html.escape(record["name"])}</span> · ' if record.get("name") else ""
    return f'{name}<a class="stop-profile" href="https://github.com/{login}">@{login} ↗</a>'


def role_label(record, language):
    if "title" in record:
        return record["title"][language]
    language_index = 0 if language == "zh" else 1
    return f'{ROLES[record["role"]][language_index]} · {EMPLOYMENT[record["employment"]][language_index]}'


def render(page, repo, data=None):
    data = load() if data is None else validate(data)
    zh = page.lang == "zh"
    language = 0 if zh else 1
    choose = lambda chinese, english: chinese if zh else english
    escape = html.escape
    records = ordered(data["records"])
    companies = data["companies"]
    root = f"https://github.com/{escape(repo, quote=True)}"
    submit = root + "/issues/new?template=next-stop.yml"
    request_icon = root + "/issues/new?template=company-icon.yml"
    guide = page.rel("community/next-stop-guide.html" if zh else "community/next-stop-guide.en.html")
    behind = page.rel("contributors.html" if zh else "contributors.en.html")
    sibling = page.rel("community/next-stop.en.html" if zh else "community/next-stop.html")
    home = page.rel("index.zh.html" if zh else "index.html")
    prefix = "../" * page.depth
    coordinates = []
    for index, record in enumerate(records[:24]):
        company = companies[record["company"]]
        date_label = start_date_label(record, page.lang)
        label = escape(f"{person_label(record, page.lang)} · {company} · {role_label(record, page.lang)} · {date_label}", quote=True)
        initials = escape(company[:2].upper(), quote=True)
        logo = company_icons.mark(record["company"], company, prefix)
        coordinates.append(f'''<div class="crew-drifter" data-crew-id="{record['id']}"
          style="--start-x:{(19 + index * 29) % 68 + 8}%;--start-y:{(11 + index * 23) % 38 + 29}%">
          <button class="crew-pilot" type="button" aria-label="{label}" aria-expanded="false" aria-controls="stop-label-{record['id']}">
            <span class="crew-face stop-company-disc" data-initials="{initials}"><canvas class="crew-bits" width="72" height="72" aria-hidden="true"></canvas>{logo}</span>
          </button>
          <div class="crew-label" id="stop-label-{record['id']}"><span class="crew-handle">{date_label} · {choose('开始', 'Start')}</span>
            <p class="stop-orbit-person">{person_html(record, page.lang)}</p>
            <a data-stop-open-record href="#{record['id']}">{escape(company)} ↗</a><small>{escape(role_label(record, page.lang))}</small>
            <small>{MILESTONES[record['milestone']][language]}</small></div></div>''')
    if not coordinates:
        coordinates.append(f'''<div class="crew-drifter stop-invitation" data-crew-id="invitation" style="--start-x:44%;--start-y:42%">
          <button class="crew-pilot" type="button" aria-label="{choose('分享我的下一站', 'Share my next stop')}" aria-expanded="false" aria-controls="stop-invitation-label">
            <span class="crew-face" data-initials="+"><canvas class="crew-bits" width="72" height="72" aria-hidden="true"></canvas><span class="stop-disc-fallback" aria-hidden="true">+</span></span>
          </button>
          <span class="stop-invitation-caption">{choose('留个坐标', 'LEAVE A COORDINATE')}</span>
          <div class="crew-label" id="stop-invitation-label"><span class="crew-handle">{choose('还没有公开分享', 'NO UPDATES YET')}</span>
            <a href="{submit}">{choose('分享我的下一站 ↗', 'Share my next stop ↗')}</a><small>{choose('公司、岗位、开始时间，就够了。', 'A company, a role, a start date.')}</small></div></div>''')
    groups = []
    for year in sorted({record["start_date"][:4] for record in records}, reverse=True):
        cards = []
        for record in (record for record in records if record["start_date"].startswith(year)):
            start_date = record["start_date"]
            date_label = start_date_label(record, page.lang)
            note = record.get("note", {}).get(page.lang, "")
            note_html = f'<p class="stop-note">{escape(note)}</p>' if note else ""
            cards.append(f'''<li class="stop-card" id="{record['id']}" data-stop-record
              data-company="{record['company']}" data-role="{record['role']}" data-employment="{record['employment']}" data-year="{year}">
              <div class="stop-date"><span>{choose('开始时间', 'Start date')}</span><time datetime="{start_date}">{date_label}</time><span>{record['id']}</span></div>
              <div><p class="stop-person">{person_html(record, page.lang)}</p>
              <h3 class="stop-company-heading">{company_icons.identity(record['company'], companies[record['company']], prefix)}</h3>
              <p class="stop-role">{escape(role_label(record, page.lang))}</p>
              <p class="stop-meta">{MILESTONES[record['milestone']][language]} · {choose('自愿分享', 'Self-reported')}</p>
              {note_html}</div></li>''')
        groups.append(f'<section class="stop-year" data-stop-year><h2>{year}</h2><ol>{"".join(cards)}</ol></section>')
    counts = Counter(record["company"] for record in records)
    company_rows = []
    for company in sorted(counts, key=lambda key: (-counts[key], companies[key].casefold())):
        company_rows.append(f'''<li data-stop-company="{company}" data-name="{escape(companies[company], quote=True)}">
          <div><h3 class="stop-company-heading">{company_icons.identity(company, companies[company], prefix)}</h3><span><b data-stop-count>{counts[company]}</b> <span data-stop-unit>{choose('条分享', 'update' if counts[company] == 1 else 'updates')}</span></span></div>
          <div class="stop-bar" aria-hidden="true"><span style="width:{counts[company] / max(counts.values()) * 100:.2f}%"></span></div></li>''')
    def options(choices, all_label):
        return '<option value="all">' + all_label + '</option>' + ''.join(f'<option value="{key}">{values[language]}</option>' for key, values in choices.items())
    years = ''.join(f'<option>{year}</option>' for year in sorted({record['start_date'][:4] for record in records}, reverse=True))
    return f'''
<div class="next-stop" data-next-stop-root data-lang="{page.lang}">
  <section class="contributor-universe stop-universe" aria-label="{choose('下一站星空', 'Next-stop universe')}" style="background-image:url('{prefix}static/crew-sky-1bit.png')">
    <nav class="orbit-nav" aria-label="{choose('页面导航', 'Page navigation')}">
      <a href="{home}">← {choose('回到笔记', 'Notes')}</a>
      <div><a href="{behind}">{choose('幕后', 'Behind')}</a><a href="#departures">{choose('去向', 'Log')}</a>
        <button class="crew-motion" type="button" aria-pressed="false" hidden data-pause="{choose('暂停', 'Pause')}" data-play="{choose('继续', 'Resume')}">{choose('暂停', 'Pause')}</button>
        <a href="{sibling}" data-language-switch="{choose('en', 'zh')}">{choose('EN', '中文')}</a></div>
    </nav>
    <header class="orbit-title"><p>WHERE WE GO NEXT</p><h1>{choose('大家的下一站。', 'Where readers go next.')}</h1></header>
    <div class="crew-field">{''.join(coordinates)}</div>
    <footer class="orbit-footer">
      <p class="stop-sky-count">{len(records)} {choose('条自愿分享', 'self-reported updates')}{choose(' · 星空展示最近 24 条', ' · latest 24 shown in the sky') if len(records) > 24 else ''}</p>
      <p><span class="orbit-desktop">{choose('鼠标靠近，看看新去向' if records else '有新的去向，欢迎回来分享。', 'Hover over a coordinate to see an update' if records else 'Have an update? Leave a little coordinate.')}</span><span class="orbit-touch">{choose('轻点坐标，看看新去向' if records else '轻点 +，分享你的下一站', 'Tap a coordinate to see an update' if records else 'Tap + to share your next stop')}</span></p>
      <a class="orbit-more" href="#departures">{choose('往下看 ↓', 'Scroll down ↓')}</a>
    </footer>
  </section>
  <section class="crew-board stop-log" id="departures" aria-label="{choose('读者去向', 'Reader updates')}">
    <div class="board-head stop-log-head"><h2>{choose('去向记录', 'The departure log')}</h2>
      <p>{choose('实习、全职，或一次转行。读者自愿分享，不代表网站促成了录用。', 'An internship, a full-time role, or a career change. Self-reported, not a claim that the site led to a job.')}</p>
      <a class="stop-submit" href="{submit}">{choose('分享我的下一站 ↗', 'Share my next stop ↗')}</a></div>
    <div class="stop-controls" data-stop-controls hidden>
      <div class="stop-views" role="group" aria-label="{choose('展示方式', 'View')}">
        <button type="button" data-stop-view="timeline" aria-pressed="true">{choose('时间线', 'Timeline')}</button>
        <button type="button" data-stop-view="companies" aria-pressed="false">{choose('公司汇总', 'By company')}</button>
      </div>
      <div class="stop-filters"><label>{choose('岗位', 'Role')}<select data-stop-filter="role">{options(ROLES, choose('所有岗位', 'All roles'))}</select></label>
      <label>{choose('类型', 'Type')}<select data-stop-filter="employment">{options(EMPLOYMENT, choose('所有类型', 'All types'))}</select></label>
      <label>{choose('开始年份', 'Start year')}<select data-stop-filter="year"><option value="all">{choose('全部年份', 'All years')}</option>{years}</select></label></div>
    </div>
    <div class="stop-results-meta"><p data-stop-status role="status" aria-live="polite">{len(records)} {choose('条公开分享 · 按开始时间由近到远', 'published updates · newest start dates first')}</p><button type="button" data-stop-clear hidden>{choose('清除筛选', 'Clear filters')}</button></div>
    <div data-stop-timeline>{''.join(groups)}</div>
    <ol class="stop-companies" data-stop-companies hidden>{''.join(company_rows)}</ol>
    <div class="stop-empty" data-stop-empty {'hidden' if records else ''}>
      <p>{choose('还没有公开分享。收到本人同意的投稿后，会出现在这里。', 'No updates yet. Consenting submissions will appear here after review.')}</p>
    </div>
    <div class="stop-empty" data-stop-no-match hidden><h3>{choose('这个组合还没有分享。', 'No updates match these filters.')}</h3><p>{choose('换个岗位、类型或年份看看。', 'Try another role, type, or year.')}</p></div>
    <p class="stop-count-note">{choose('数量按当前筛选后的公开记录计算，同一个人可能有多段经历；不代表独立人数、录用率或公司排名。仅填年份的记录放在该年末尾，不推断具体月份。', 'Counts reflect published records after filtering, not unique people, hiring rates, or company quality. A reader may share multiple milestones. Year-only records appear last within that year; no month is inferred.')}</p>
    {company_icons.catalog(page, request_icon)}
    <details class="stop-about"><summary>{choose('分享之前，先说清楚', 'Before you share')}</summary>
      <p>{choose('可以展示 GitHub 用户名、公司、岗位和开始年月，用户名能点进个人主页。不想公开账号，也可以只用昵称。尚未入职的填预计开始时间；不确定月份就只填年份。拿到 offer、已入职和已结束会分开标注，不按收到 offer、结束或投稿的时间排序。', 'Share your GitHub username, company, role, and start month, with a link to your profile. A nickname is fine if you prefer not to link your account. For an offer, use the expected start date; share just the year if the month is unknown. Offers, started roles, and completed roles are labelled separately. Offer, completion, and submission dates do not control the order.')}</p>
      <p>{choose('投稿先由维护者确认展示范围，再通过 PR 加入。GitHub issue 是公开的，即使网页不显示姓名，issue 仍可能关联你的账号。请不要上传 offer、薪资、证件或招聘私信。', 'A maintainer reviews the display scope before adding a record through a PR. GitHub issues are public and can identify your account even when your name is hidden here. Do not upload offer letters, salary details, IDs, or private hiring messages.')}</p>
      <p>{choose('想修改或撤下？用同一个投稿入口写明记录编号即可。网页可更新，但已公开的 issue、Git 历史和外部缓存可能仍保留旧内容。', 'To correct or remove an entry, use the same form with its record ID. The page can be updated, but public issues, Git history, and external caches may retain earlier content.')}</p>
      <a href="{guide}">{choose('完整说明与维护规则 →', 'Submission and maintenance guide →')}</a>
    </details>
  </section>
  <footer class="crew-end"><a href="{behind}">← {choose('回到幕后', 'Behind the notes')}</a><a href="{guide}">{choose('投稿与展示说明', 'Sharing & privacy')}</a><a href="{home}">{choose('继续读笔记 ↗', 'Back to the notes ↗')}</a></footer>
</div>'''
