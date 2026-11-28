#!/usr/bin/env bash
# Keeps one comment on a pull request that links its preview (site.yml, preview and unpreview jobs).
#   preview-comment.sh live      the preview was built; the deploy on main will publish it
#   preview-comment.sh removed   the pull request closed; the next deploy drops the preview
# Needs GH_TOKEN (pull-requests: write), REPO, PR and SITE_URL; SHA for `live`.
set -euo pipefail
state="$1"
marker='<!-- pr-preview -->'
url="$SITE_URL/pr-preview/pr-$PR/"
if [ "$state" = live ]; then
  body="$marker
**Preview / 预览:** $url

Built from ${SHA:0:7}. It is live once the Pages deploy this started on main finishes (see Actions). Reading records in the preview are kept apart from the live site's. Closing the PR takes the preview down.

等 main 上这次触发的 Pages 部署跑完就能打开（进度见 Actions）。预览里的阅读记录和正式站点分开保存；PR 关闭后预览会撤下。"
else
  body="$marker
This PR is closed, so its preview has been taken down. / PR 已关闭，预览已撤下。"
fi
id="$(gh api --paginate "repos/$REPO/issues/$PR/comments" \
      --jq ".[] | select(.user.login == \"github-actions[bot]\" and (.body | startswith(\"$marker\"))) | .id" | head -n 1)"
if [ -n "$id" ]; then
  gh api -X PATCH "repos/$REPO/issues/comments/$id" -f body="$body" > /dev/null
elif [ "$state" = live ]; then
  gh api -X POST "repos/$REPO/issues/$PR/comments" -f body="$body" > /dev/null
fi
