# Releasing Cooking AGI

Public scope and status live in `community/updates.md` and `community/updates.en.md`. Priorities live in the paired roadmap pages. Do not keep a second full changelog in the README: link to these pages. The first proposed named release is **v1.0.0 — Study workspace**, not a claim that every AI topic is covered.

## Release scope

A release should provide a usable end-to-end flow: choose a route, create a session, follow its note, record a question, and export/restore the record. Review the associated content and navigation too. Future CV, driving, edge, simulation, and robotics proposals are not release blockers or completed features.

Before approving, compare the actual PR diff with the scope. Include all intentional changes, not just the most recent files, and preserve unrelated work. Never publish private backups, credentials, local review snapshots, private reference material, or social-platform drafts. Confirm authority to license prior contributions and third-party assets; the default license notice does not settle existing exceptions.

## Candidate checks

Run from the repository root with Python 3.12 and the CI dependencies:

```sh
python site/collaboration.py
python -m unittest discover -s site/tests
python site/leakcheck.py
python site/paritycheck.py --strict
python site/build.py
git diff --check
```

Check generated local links and anchors. Record counts, skips, errors, and the tested commit or local diff in `CONTENT_RELEASE_CHECKLIST.md`; do not reuse a previous run's numbers as evidence for new changes. Follow the browser checks in `PREP_PROGRESS.md`. Use an isolated origin and synthetic records, never someone's real records. Include an actual backup round-trip, an old backup and an old bookmarks file, mobile layouts, both themes and languages, and no-JavaScript reading links.

No GPU or hardware performance claim follows from these CPU/site tests. Semantic review, rights checks, and current primary sources remain separate tasks. An AI report is advisory, not a CODEOWNER approval.

## Merge, deploy, release

1. Read the latest remote main, existing PRs, tags, releases, and protection state. Don't append new work to a merged PR, reuse an existing tag, rewrite shared history, or sweep another author's changes into a release.
2. Open a scoped PR with the proposed version, summary, test results, known limits, and data-compatibility note. Follow the repository's human review rules and required CI. No automatic admin bypass.
3. Merge through the normal review process. Record the **actual merge SHA**, not the old branch tip. Confirm the Pages workflow for that SHA succeeds, including deploy, rather than checking an unrelated scheduled run.
4. Check the interview-prep page, the reading drawer and a paired article at the deployed URL. Create the version tag and GitHub release at that verified SHA, with explicit notes on additions, fixes, limitations, and backup compatibility. Use an explicit target SHA when creating a release. Avoid making `latest` claims for a draft or prerelease.
5. Update both public release pages and README status links to the actual release URL and date in a follow-up docs PR, then verify that deployment too. A short delay in marking a verified release is safer than claiming a pending release shipped.

PR checks, Pages deployment, and a GitHub release are three different states. If any stage is pending or fails, report that state and keep the release marked unreleased. Do not manufacture an approval, hide an exception, or treat this checklist as authorization to publish.

## Versioning and progress

Use patch versions for corrections and compatible fixes; minor versions for complete features and reviewed routes; major versions for incompatible user-facing changes. Ordinary note edits may ship between named releases. Historical release entries describe their snapshot; newer changes get a new entry rather than silently rewriting the old claims.

Progress has its own schema version. A site release must not bump it just to match a tag. Preserve stable note IDs when renaming files and keep old backup fixtures. Schema changes need migration and rollback tests, including unknown chapter records and the optional fields older builds wrote (statuses, memos, schedules), which are kept even though nothing shows them.

See `PREP_PROGRESS.md` for the data contract and recovery boundaries.

## If a release is broken

Pause the release and identify the affected commit. Use a reviewed revert or fix on main and let CI redeploy; don't move or delete a published tag to hide the problem. Update the release notes with the affected version and corrective version. For any data migration issue, preserve the original export, avoid destructive automatic recovery, and explain what was and wasn't restored. A previous code version may not understand newer data, so assess rollback compatibility before recommending it.
