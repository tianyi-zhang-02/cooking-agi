# Updates and reading records {#updates}

[中文](updates.md) · **English**

This page explains how the site is updated and where your saved notes and reading records live. Future work lives in the [roadmap](roadmap.en.md).

## How the site is updated {#how}

There are no version numbers, and changes don't wait to be bundled into a release: a change goes live once it is merged into main and deployed. The home page's [changelog](../README.md#changelog) lists every merge by date; for edits to a particular article, use its source link and repository history.

The planned "v1.0.0 · Interview prep" release won't be published. Everything already built keeps working.

## Saved notes and reading records {#records}

- **Saved notes:** "Save this page" sits under a note's title; you'll find saved notes under "My reading" at the top right.
- **Reading records:** "My reading" shows which notes you read today and how often, and remembers where you stopped in each one.
- **Backups:** saved notes, reading history and reading positions share one JSON backup; importing merges, keeping the newer copy of each record.

These records stay in the current browser: no sign-up, and nothing is uploaded. Site updates don't deliberately reset records at the same browser and site address, and the backup format is versioned separately, so old backups still import.

This is not cloud sync. Export and import when changing devices, browsers or site addresses, or clearing site data. Simultaneous edits in several tabs are not guaranteed to all be saved. **Do not upload private backups to issues or the repository.**

## Known limits {#limits}

- Reading or saving a note does not mean you have learned it. There is no automatic grading or ability prediction.
- There are no accounts, cloud sync, offline caching or paid services. Export promptly if browser storage is unavailable.
- Automated checks and browser spot-checks don't mean every technical claim has been independently reviewed, or that GPU or hardware experiments were run.
