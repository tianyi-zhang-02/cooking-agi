# Next stop: submission and display guide

[中文](next-stop-guide.md) · **English** · [Back to Next stop](next-stop.en.md)

Where you go after studying is worth remembering too. Share an internship, a full-time role, or a career change. We do not collect offer documents or turn readers' experiences into the site's performance claims.

## How do I share an update?

1. Open the [submission form](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=next-stop.yml). Add a company, role category, employment type, and **start date**, such as May 2026 or 2026-05. For an offer, use the expected start date. Share just the year if the month is unknown or you prefer not to share it. Choose whether you received an offer, started, or completed the role; you can also add the specific job title, such as ML Engineer Intern.
2. To be recognised, opt in to showing your GitHub username. A clickable `@username` appears alongside the update and in its sky detail panel, linking to your GitHub profile. A nickname or unnamed entry is fine if you prefer not to link your account. You can add a sentence about what helped or what was missing. This does not need to be a testimonial.
3. A maintainer checks your display consent and whether the same milestone already exists, then adds it through a PR. We do not collect experiences from social profiles, GitHub, or conversations without permission.

**GitHub issues are public.** An unnamed entry on the page is not fully anonymous: the issue and commit history can still identify your account. Do not use this form if you do not want the information to be public.

Do not upload offer documents, salary details, IDs, contact details, or private hiring messages. Share only your own experience, not a colleague's, classmate's, or friend's.

## What do the counts mean?

- The timeline uses the **start year and month**, newest first, not the offer, completion, or submission date. Dates for offers are marked “Expected”. Year-only entries appear last within that year; this does not assign a month or imply December.
- Company totals count **published records** matching the current filters, with ties sorted by company name. They are not company quality rankings, unique-person counts, acceptance rates, or success rates for site users.
- One reader can have different roles. When an offer becomes a started or completed role, update the existing entry instead of counting each stage again. Correct the expected start date once the actual date is known. Completing a role changes its status, not its start date.
- Role categories support browsing: SWE / SDE under software engineering, MLE under machine learning, RS / RE under research, with separate data, product, and other groups. Let the contributor confirm ambiguous roles rather than guessing from a company name.
- Updates are **self-reported**. Review checks consent, format, and duplicates; it is not employer verification and does not establish that the website led to the job.

## What if my company is not listed?

Share your update anyway: the icon library is not a company allowlist. It includes presets across tech, AI, software, and quant. Missing logos use the full company name, not initials, without blocking an update.

To add one, use the [company icon request form](https://github.com/tianyi-zhang-02/cooking-agi/issues/new?template=company-icon.yml) with the **company name, official website, and public logo or brand-resource links**. Aliases and Chinese names help with search too. The issue is assigned to **@tianyi-zhang-02**; feel free to ping me in the notes :) I’ll review the resources before adding it.

Do not submit internal files. Submitted URLs are not fetched automatically, and requests are not published automatically. Icons identify companies; they do not imply a reader works there, a partnership, or an endorsement.

## Corrections, removal, and privacy

Use the same form, choose “Correct” or “Remove”, and provide the record ID. You do not need to repeat the original experience publicly. A maintainer confirms that the request is yours before updating the page.

Public issues, Git history, forks, and external caches may retain earlier content. Removing an entry from the page does not guarantee deletion of those copies. This is why we collect as little as possible in the first place.

## How do maintainers add a record?

The source is `community/outcomes.json`, rendered by `site/next_stop.py`. Both languages use the same records. Keep the list empty until consenting submissions arrive; do not populate it with fictional outcomes.

Use a small PR for each update:

1. Check that the submission is first-party, has explicit display consent, and is not a duplicate. Reuse canonical company IDs so casing and aliases do not split one company into several.
2. Give each record a stable ID such as `NS-0001`. `start_date` stores the start year or month as `YYYY` or `YYYY-MM`; convert a submission such as May 2026 to `2026-05`. Never substitute the offer or submission date; ask the reader if unclear. `source_issue` tracks the consent source and `consent` must be `true`. The page does not link directly to that person's source issue. For the site owner's own explicitly requested update, `owner_submission: true` may replace `source_issue`, but `github` must match the configured owner. Do not use this for other readers or invent an issue number.
3. Required fields are `id`, `start_date`, `company`, `role`, `employment`, `milestone`, `consent`, and one consent source as described above. `company` references an ID in the `companies` dictionary, whose values are public display names.
4. `role`: `software / ml / research / data / product / other`; `employment`: `internship / full-time / contract / other`; `milestone`: `offer / started / completed`. An optional `title` has `zh` and `en` job titles, each at most 100 characters; otherwise the category and employment type are shown. A career change can be described in the optional note rather than replacing the employment type.
5. Add `name` and `github` only with separate opt-in for each; otherwise display “A reader”. When `github` is present, show the explicit `@username`; if a nickname is also provided, display both instead of replacing the handle. An optional `note` has both `zh` and `en` versions, each at most 400 characters. Do not add results during translation. Source data in a public repository is also public; do not store private review notes there.
6. Follow the existing review process and run the checks below before merging. Be explicit if no independent reviewer is available; passing automated checks is not independent review.

Company presets live separately in `site/company-icons.json`, with SVGs in `site/static/company-icons/`. Reuse existing IDs and add aliases and public sources. Accept only static SVGs after checking for scripts, external references, or embedded content. Use the full company name if no suitable logo is available rather than drawing an imitation. **Adding a logo does not add a career record**: icon requests and first-party career submissions are separate workflows.

```sh
python3 -m unittest discover -s site/tests
python3 site/paritycheck.py --strict
python3 site/leakcheck.py
python3 site/build.py
```

Follow the existing [community collaboration rules](README.en.md). There is no extra committee, and a company's prestige does not determine whether a submission is included.
