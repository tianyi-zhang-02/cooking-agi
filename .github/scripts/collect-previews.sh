#!/usr/bin/env bash
# Adds the previews of open pull requests to the site that is about to deploy (site.yml, build job
# on main). Each preview is the newest `preview-pr-<N>` artifact the PR's own run uploaded; it lands
# under <dest>/pr-<N>/. Only pull requests from this repository, newest activity first, and at most
# PREVIEW_LIMIT of them: GitHub Pages caps a site at 1 GB and one copy of the site is about 120 MB.
# Needs GH_TOKEN (actions: read, pull-requests: read) and REPO (owner/name).
set -euo pipefail
dest="$1"
limit="${PREVIEW_LIMIT:-4}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$dest"

gh pr list -R "$REPO" --state open --limit 100 --json number,isCrossRepository,updatedAt \
  --jq 'map(select(.isCrossRepository | not)) | sort_by(.updatedAt) | reverse | .[].number' > "$work/open"

added=0
while read -r pr; do
  [ "$added" -lt "$limit" ] || break
  case "$pr" in ''|*[!0-9]*) continue ;; esac
  id="$(gh api "repos/$REPO/actions/artifacts?name=preview-pr-$pr&per_page=1" \
        --jq '.artifacts[0] // empty | select(.expired | not) | .id')"
  [ -n "$id" ] || continue
  gh api "repos/$REPO/actions/artifacts/$id/zip" > "$work/$pr.zip"
  mkdir -p "$work/$pr" "$dest/pr-$pr"
  unzip -q "$work/$pr.zip" -d "$work/$pr"
  tar -xf "$work/$pr/artifact.tar" -C "$dest/pr-$pr" --no-same-owner
  added=$((added + 1))
  echo "added the preview of #$pr"
done < "$work/open"
echo "$added preview(s) in this deploy"
