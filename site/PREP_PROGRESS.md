# Reader records: storage and compatibility

Saved notes, reading history and reading positions are user data, not content in Git. They live in one browser envelope under the stable localStorage key `cooking-agi:prep-progress` (recovery copies under `cooking-agi:prep-progress:snapshots`). A site rebuild, normal pull, or content release must not reset either key. Do not include them in general UI-preference reset handlers.

Readers see two things: a **Save this page** button under each note's title, and the **My reading** drawer at the top right (reading history, saved notes, settings and one backup row). There is no separate dashboard; `interview/dashboard.html` and its English page redirect to the interview-prep page.

The SI Study Notes rebrand changes display names only. Keep the `cooking-agi` URL path, storage keys, backup format and site identifiers, export filename prefix, and note IDs unchanged. Existing bookmarks, language preferences and backups must keep working without a migration.

## Files

| File | Role |
| --- | --- |
| `static/prep-progress-store.js` | The envelope: validate, read/write, merge, encode, recovery copies. Pure functions, tested in Node. |
| `static/prep-library-state.js` | Bookmarks and positions as workspace records; migrates the old `cooking-agi:reading:v1` and `cooking-agi:positions:v1` keys once, without deleting them. |
| `static/prep-progress.js` | Opens the store and exposes `window.PreparationProgress`: `library`, `exportBackup`, `importBackup`, `getMode`/`setMode` (the interview-prep tab), `getHealth`. |
| `static/sidebar.js` | Directory fold state and the save button(s) `[data-side-save]`. |
| `static/reader-hub.js` | The drawer: reading counts, saved list, settings, export/import. |
| `site/prep_progress.py` | Stable note IDs and `static/prep-progress-catalog.json`, the ID-to-page catalog the drawer and the counter service read. |

## What the envelope holds

The version-1 JSON envelope has `format: "cooking-agi-prep-progress"`, `version: 1`, `site: "cooking-agi"`, `entries`, `mode`, and an optional `workspace: {version: 1, records: {}, history: []}`. Each workspace record has `updatedAt` and merges independently.

| Record key | Written by | Meaning |
| --- | --- | --- |
| `bookmark-<note ID>` | save button, drawer, legacy import | `saved: true/false`. Unsaving keeps the record with `saved: false` so a merge cannot resurrect it by default. |
| `position-<lang>-<note ID>` | note pages | Last section read, so a note reopens where the reader stopped. |
| `reads-YYYYMMDD`, `reading-total`, `reader-settings` | drawer | See below. |
| `legacy-library` | migration | The old sidebar lists, kept so paths not in today's catalog are not lost. |

`mode` (`review` or `plan`) remembers which tab the interview-prep page last showed.

**Kept but no longer shown.** Earlier builds also wrote per-note `entries` (status, private memo), a `schedule` of study sessions, workspace records for topic follows and update notices, and `visit-…` records for the old sidebar's recent list (the drawer counts reads in `reads-…` instead). The store still validates, preserves, merges and exports them exactly as before, so an old backup round-trips without loss, but nothing on the site displays or edits them. Do not drop these fields from the validator without a tested migration.

### Reading history and reader settings

The reading drawer (`static/reader-hub.js`, loaded on every page after `prep-progress.js`) keeps three kinds of workspace records. They use the same envelope, merge and backup rules as the bookmarks; no new storage key is created.

| Key | Fields | Meaning |
| --- | --- | --- |
| `reads-YYYYMMDD` | `notes`: up to 200 `note-…` IDs mapped to integers 0–9999; `updatedAt` | How many times each note was read on that local calendar day. A page counts once per load after 15 seconds of visible time; reading a page is not marking it learned. |
| `reading-total` | `total`: integer 0–9,999,999; `since`: `YYYY-MM-DD` or empty; `updatedAt` | Every counted read since `since`, kept past the 90-day window. The panel never shows it below the sum of the day records still kept, and a missing record is seeded from them. |
| `reader-settings` | optional `history` (boolean, default on), `mascot` (`open`, `collapsed`, `hidden`; default `collapsed`), `starAskedOn` (`YYYY-MM-DD` or empty), `starDone` (boolean), `snoozedUntil` (UTC timestamp or empty); `updatedAt` | Whether reading is counted, and the star notice: a small toast that appears at most once a day (after 20 seconds on a page) until the reader stars, chooses "Don't show again" (`starDone`, `mascot: hidden`), or snoozes it for 12 hours ("Not now" or ×). The Star pill in the top bar is always there; clicking it also counts as starred. Choosing Show in settings clears a snooze. |

Days older than 90 are emptied rather than deleted, so a later import cannot resurrect them by default and merges stay timestamp-based. Clearing reading history empties every day record, resets `reading-total` to zero from that day, and leaves bookmarks and positions untouched. `reading-total` merges like any other record (the newer copy wins), so the all-time figure is per browser, not a sum across devices. Each day merges as one record, so the same day edited on two devices keeps the newer copy under the usual policies; counts are approximate, not an audit log. One record per active day accumulates against the 3,000-record workspace limit, which leaves years of daily use but is not unlimited. Turning counting off stops new counts without erasing old ones.

The Reading tab shows today, this week (Monday to Sunday) with last week's total, the all-time total, a week of daily bars, and a 13-week grid with the current streak, all computed from these records. On a note page a small card shows the 15-second count filling up and then that note's count for today; the numbers update in place when a read is counted, including from another tab, and roll over at local midnight.

The drawer's "Saved" tab lists the same `bookmark-…` records the save button writes. Theme stays in its separate `theme` key; the settings tab only presses the existing toggle.

## Stable IDs and chapter moves

`site/prep_progress.py` identifies a bilingual pair using its canonical Chinese source path, independent of title, navigation position, and language. The default ID is `note-` plus the first 16 hexadecimal digits of SHA-256 of that path. The build produces a public ID-to-page catalog with no user data.

A **source-path rename needs an explicit alias** in `site/progress-id-overrides.json`: map the new canonical path to the old ID. Retrieve the old ID from the previous catalog before moving the source. Do not generate a fresh ID for an existing chapter or reuse an old ID for unrelated content. The build rejects duplicate IDs. Translation pairs must retain one identity. Anchors should also remain stable; if a section is removed, an old link still reaches the note, but cannot recover the removed section.

For a split, keep the old ID on the continuation or overview and give genuinely new chapters new IDs. For a merge, preserve the older page or provide a deliberate migration; do not silently collapse independent user notes. Unknown IDs remain in storage and exports even if the current catalog has no matching page.

## Export and import

The drawer's settings tab has one **Backup** row.

- **Export** downloads `agi-notes-backup-YYYYMMDD.json`: the whole envelope (`store.serialize()`), including any kept-but-hidden fields.
- **Import** reads a local file of at most 2 MiB. A current or older envelope is parsed and validated before any write, then merged with the `newer` policy: for each record the strictly newer `updatedAt` wins, local records missing from the file stay, and nothing is deleted. The store keeps a pre-import recovery copy.
- An **old bookmarks-only file** (`{format: "cooking-agi-bookmarks", saved: [page paths]}`, exported by earlier sidebars) is recognised separately: known paths become `bookmark-…` records through the catalog; unknown or malformed entries are skipped (`legacyBookmarks` in `reader-hub.js`).
- Unreadable, oversized, or future-version files show an error and leave local records unchanged.

The parser checks the envelope, IDs, timestamps, lengths, nesting and object count, and rejects prototype-related keys. Imported titles and notes are rendered as text, never HTML; imported URLs are not followed. There is no cloud sync and no transactional multi-tab store: storage events refresh open pages, but simultaneous writes can still race, and timestamp merging depends on device clocks.

The `:snapshots` key keeps up to three recovery copies (about 2 MiB in total). Quota failure there never blocks a successful primary write. There is currently no reader-facing restore button; a maintainer can recover a copy from the browser console if a reader reports lost data.

## Future format changes

The first shipped format is version 1; there is no invented migration for an earlier format. A future writer must retain a tested migration path from version 1 before changing the version. Add immutable old-version fixtures, test migrations without modifying the input, preserve unknown records and optional fields where safe, and validate the migrated result before saving. Unsupported newer files must remain exportable but read-only in old site builds. Do not claim that this version can read arbitrary future schemas.

When editing this feature, test an old study-dashboard backup and an old bookmarks file against the new catalog, a renamed chapter with its override, a missing chapter, a repeated import, an older restore, conflicting changes, storage denial/quota failures, and malformed or newer-version data. Never equate a saved note or a counted read with mastery.

## Browser and repository boundaries

Persistence is scoped to browser profile and origin (protocol, domain, and port), not to the Git checkout; no fork or clone is needed to keep saved notes or reading history. Switching a localhost port, moving from localhost to GitHub Pages, using another browser, or deploying a fork at another origin requires importing the backup. Private browsing, clearing site data, eviction, and storage failures can remove records. The site cannot quietly save an arbitrary file on someone's computer; export is an explicit download.

Keep backups outside the checkout. `.gitignore` includes `agi-notes-backup*.json`, `agi-notes-bookmarks*.json`, the older `agi-prep-progress*.json` and `.study-progress/` as an extra safeguard, not a security boundary: differently named files and already-tracked files are not protected by those patterns. Do not commit a personal backup to a public fork; older ones can contain private memos. JSON files are not encrypted. Imports are parsed locally; no record or backup is uploaded. The public catalog fetch contains only site metadata.

## Checks

Run from the repository root:

```sh
node --test site/tests/prep-progress-store.test.cjs site/tests/prep-library-state.test.cjs site/tests/prep-progress.test.cjs
python -m unittest discover -s site/tests -p 'test_prep_progress.py'
python -m unittest discover -s site/tests -p 'test_reader_hub.py'
python -m unittest discover -s site/tests -p 'test_sidebar.py'
python site/build.py
```

For browser checks, use a separate localhost port so synthetic test notes do not alter a reader's actual records. Verify save/unsave, reopening at the saved position, language switching, export and re-import, an old bookmarks file, errors, keyboard controls and narrow layouts. Do not reset a reader's storage to prepare a test.
