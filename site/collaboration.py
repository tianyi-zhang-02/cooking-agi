from __future__ import annotations

import argparse
import html
import re
import tomllib
from pathlib import Path
from urllib.parse import urlencode


ROOT = Path(__file__).resolve().parents[1]


def load_config(root=ROOT):
    return tomllib.loads((root / "site/collaboration.toml").read_text(encoding="utf-8"))


def validate(config, nav):
    username = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?")
    if not config.get("maintainers"):
        raise ValueError("At least one maintainer is required")
    groups = {group["id"] for group in nav["group"]}
    seen_ids, seen_groups = set(), set()
    logins = list(config["maintainers"])
    for area in config["area"]:
        if not re.fullmatch(r"[a-z][a-z0-9-]*", area["id"]) or area["id"] in seen_ids:
            raise ValueError("Invalid or duplicate area ID")
        seen_ids.add(area["id"])
        if not area.get("zh") or not area.get("en") or not area.get("groups"):
            raise ValueError("Areas need bilingual names and navigation groups")
        for group in area["groups"]:
            if group not in groups or group in seen_groups:
                raise ValueError(f"Unknown or duplicate navigation group: {group}")
            seen_groups.add(group)
        logins.extend(area["reviewers"])
    if groups != seen_groups:
        raise ValueError(f"Unassigned navigation groups: {groups - seen_groups}")
    seen_credits = set()
    repo = re.escape(nav["site"]["repo"])
    for credit in config.get("acknowledgements", []):
        logins.append(credit["login"])
        if credit["login"].lower() in seen_credits:
            raise ValueError("Combine acknowledgements for the same person")
        seen_credits.add(credit["login"].lower())
        if not credit.get("zh") or not credit.get("en") or not credit.get("evidence"):
            raise ValueError("Acknowledgements need bilingual descriptions and evidence")
        for evidence in credit["evidence"]:
            if not re.fullmatch(rf"https://github\.com/{repo}/(?:issues|pull)/[1-9]\d*(?:#[A-Za-z0-9-]+)?", evidence):
                raise ValueError("Credit evidence must link to this repository's issues or PRs")
    if any(not username.fullmatch(login) for login in logins):
        raise ValueError("Invalid GitHub username")


def area_for(config, group):
    return next(area for area in config["area"] if group in area["groups"])


def owners_for(config, area):
    return area["reviewers"] or config["maintainers"]


def codeowners(config, nav):
    fallback = " ".join(f"@{login}" for login in config["maintainers"])
    lines = [f"* {fallback}"]
    sections = sorted(nav["section"], key=lambda section: len(section["dir"].split("/")))
    for section in sections:
        if section["dir"] == ".":
            continue
        area = area_for(config, section.get("group", "reference"))
        owners = " ".join(f"@{login}" for login in owners_for(config, area))
        lines.append(f'/{section["dir"]}/ {owners}')
    for section in sections:
        area = area_for(config, section.get("group", "reference"))
        owners = " ".join(f"@{login}" for login in owners_for(config, area))
        for path in section.get("include", []):
            lines.append(f"/{path} {owners}")
            if path.endswith(".md"):
                lines.append(f"/{path[:-3]}.en.md {owners}")
    for path in (".github/", "site/", "community/", "CONTRIBUTING.md", "crew.toml"):
        lines.append(f"/{path} {fallback}")
    return "\n".join(lines) + "\n"


def github_link(repo, route, params):
    return html.escape(f"https://github.com/{repo}/{route}?{urlencode(params)}", quote=True)


def profile_links(logins):
    return ", ".join(f'<a href="https://github.com/{login}">@{login}</a>' for login in logins)


def discussion_url(repo, area_id, source=None):
    query = f'is:issue in:body "Area: {area_id}"'
    if source:
        query += f' "{source}"'
    return github_link(repo, "issues", {"q": query})


def feedback_url(repo, area_id, source=""):
    return github_link(repo, "issues/new", {
        "template": "note-feedback.yml", "area": f"Area: {area_id}", "source": source,
    })


def page_panel(page, config, repo, root=ROOT):
    area = area_for(config, page.section["group"])
    zh = page.lang == "zh"
    source = page.src.relative_to(root).as_posix()
    if source.endswith(".en.md"):
        source = source[:-6] + ".md"
    label = area["zh" if zh else "en"]
    reviewer_label = ("审核联系" if area["reviewers"] else "临时接审") if zh else (
        "Review contact" if area["reviewers"] else "Interim review contact")
    home = page.rel("community/index.html" if zh else "community/index.en.html")
    return f'''<details class="collab-panel">
<summary>{"讨论与共建" if zh else "Discuss & contribute"}</summary>
<p>{html.escape(label)} · {reviewer_label}: {profile_links(owners_for(config, area))}</p>
<nav aria-label="{"本页协作" if zh else "Page collaboration"}">
<a href="{feedback_url(repo, area['id'], source)}">{"提问 / 纠错" if zh else "Ask / correct"}</a>
<a href="{discussion_url(repo, area['id'], source)}">{"本页讨论" if zh else "Page discussions"}</a>
<a href="{discussion_url(repo, area['id'])}">{"板块讨论" if zh else "Area discussions"}</a>
<a href="{home}">{"怎么参与" if zh else "How to contribute"}</a>
</nav></details>'''


def areas_html(config, repo, zh):
    rows = []
    for area in config["area"]:
        status = ("已认领" if zh else "Assigned") if area["reviewers"] else (
            "待认领 · 临时接审" if zh else "Open · interim contact")
        rows.append(f'<tr><td>{html.escape(area["zh" if zh else "en"])}<br><code>{area["id"]}</code></td>'
                    f'<td>{status}<br>{profile_links(owners_for(config, area))}</td>'
                    f'<td><a href="{discussion_url(repo, area["id"])}">{"讨论" if zh else "Discuss"}</a></td></tr>')
    headings = ("板块", "审核联系", "讨论记录") if zh else ("Area", "Review contact", "Threads")
    return '<div class="collab-table"><table><thead><tr>' + ''.join(
        f'<th>{heading}</th>' for heading in headings) + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'


def acknowledgements_html(config, zh):
    credits = config.get("acknowledgements", [])
    if not credits:
        return '<p>' + ("审阅、提问、翻译和纠错也值得被记住。可以附上相关 issue 或 PR，申请在这里补充署名；不需要为了上榜凑提交。" if zh else
                       "Reviews, questions, translations, and corrections count too. Link the relevant issue or PR to request a credit here; no extra commits needed.") + '</p>'
    rows = []
    for credit in credits:
        evidence = " · ".join(f'<a href="{html.escape(url, quote=True)}">#{url.split("/")[-1].split("#")[0]}</a>' for url in credit["evidence"])
        rows.append(f'<li>{profile_links([credit["login"]])} — {html.escape(credit["zh" if zh else "en"])} · {evidence}</li>')
    return '<ul>' + ''.join(rows) + '</ul>'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    config = load_config()
    nav = tomllib.loads((ROOT / "site/nav.toml").read_text(encoding="utf-8"))
    validate(config, nav)
    expected = codeowners(config, nav)
    path = ROOT / ".github/CODEOWNERS"
    if args.write:
        path.write_text(expected, encoding="utf-8")
    elif not path.exists() or path.read_text(encoding="utf-8") != expected:
        raise SystemExit("CODEOWNERS is out of date. Run python site/collaboration.py --write")
    print("Collaboration config and CODEOWNERS match")


if __name__ == "__main__":
    main()
