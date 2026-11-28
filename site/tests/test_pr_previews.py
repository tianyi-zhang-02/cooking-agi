"""Pull-request previews: the preview build mode and the workflow that publishes it (RELEASING.md)."""
import base64
import hashlib
import html
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "site"))
import build

WORKFLOW = (ROOT / ".github/workflows/site.yml").read_text()
SCRIPTS = ROOT / ".github/scripts"


def job(name):
    """The text of one job in site.yml, up to the next job."""
    start = WORKFLOW.index(f"\n  {name}:\n")
    following = re.search(r"\n  [a-z]+:\n", WORKFLOW[start + 1:])
    return WORKFLOW[start:start + 1 + following.start()] if following else WORKFLOW[start:]


class PreviewBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(build.NAV), patch.dict(build.BY_SRC):
            cls.nav = build.load_nav()
            pages, cls.sections = build.discover(cls.nav)
            known = {page.url for page in pages}
            cls.page = build.BY_SRC["00-foundations/deep-dives/kv-cache-and-inference.md"]["en"]
            build.build_page(cls.page, [], cls.nav["site"]["repo"], known)
        cls.template = (build.SITE / "template.html").read_text()

    def render(self, preview):
        with patch.object(build, "PREVIEW", preview):
            return build.assemble(self.page, self.sections, self.nav, "test", self.template)

    def test_the_live_site_is_unmarked(self):
        output = self.render(None)
        self.assertNotIn('name="robots"', output)
        self.assertNotIn("preview-pill", output)
        self.assertIn('progressKey: "cooking-agi:prep-progress"', output)

    def test_a_preview_is_marked_unindexed_and_keeps_its_own_records(self):
        output = self.render(156)
        self.assertIn('<meta name="robots" content="noindex, nofollow">', output)
        self.assertIn('<a class="preview-pill" href="https://github.com/tianyi-zhang-02/cooking-agi/pull/156">'
                      "Preview · PR #156, not the live site</a>", output)
        self.assertIn('progressKey: "cooking-agi:pr-156:prep-progress"', output)
        # the page's own inline scripts still run under the Content-Security-Policy
        policy = html.unescape(re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', output).group(1))
        for script in re.findall(r"<script>(.*?)</script>", output, re.S):
            digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
            self.assertIn(f"'sha256-{digest}'", policy)

    @unittest.skipUnless(shutil.which("node"), "Node is required for the store test")
    def test_the_store_accepts_only_a_preview_key(self):
        script = """
const path = require('node:path');
const load = site => { globalThis.SITE = site; delete require.cache[require.resolve(path.resolve('site/static/prep-progress-store.js'))];
  return require(path.resolve('site/static/prep-progress-store.js')).KEY; };
console.log(JSON.stringify([load(undefined), load({progressKey: 'cooking-agi:pr-156:prep-progress'}),
  load({progressKey: 'other-site:data'}), load({progressKey: 'cooking-agi:pr-0:prep-progress'})]));
"""
        result = subprocess.run([shutil.which("node"), "-e", script], cwd=ROOT, text=True, capture_output=True, check=True)
        self.assertEqual(result.stdout.strip(), '["cooking-agi:prep-progress","cooking-agi:pr-156:prep-progress",'
                                                '"cooking-agi:prep-progress","cooking-agi:prep-progress"]')


class PreviewWorkflowTests(unittest.TestCase):
    def test_every_action_is_pinned_to_a_commit(self):
        for uses in re.findall(r"uses: (\S+)", WORKFLOW):
            self.assertRegex(uses, r"^[\w.-]+/[\w.-]+@[0-9a-f]{40}$")

    def test_every_pull_request_is_built_and_closing_one_only_unpublishes(self):
        self.assertIn("pull_request:\n    types: [opened, synchronize, reopened, closed]\n", WORKFLOW)
        self.assertIn("if: github.event.action != 'closed'", job("build"))

    def test_only_main_deploys_and_it_carries_the_open_previews(self):
        build_job = job("build")
        self.assertIn("run: bash .github/scripts/collect-previews.sh _site/pr-preview", build_job)
        collect = build_job.index("run: bash .github/scripts/collect-previews.sh")
        self.assertLess(collect, build_job.index("upload-pages-artifact"))
        self.assertIn("if: github.event_name != 'pull_request' && github.ref == 'refs/heads/main'", build_job[:collect])
        self.assertIn("if: github.event_name != 'pull_request' && github.ref == 'refs/heads/main'", job("deploy"))
        self.assertIn("pages: write", job("deploy"))
        for name in ("build", "preview", "unpreview"):
            self.assertNotIn("pages: write", job(name))

    def test_previews_are_for_this_repository_only_with_the_least_access(self):
        for name in ("preview", "unpreview"):
            text = job(name)
            self.assertIn("github.event.pull_request.head.repo.full_name == github.repository", text)
            permissions = re.search(r"permissions:\n((?:      .+\n)+)", text).group(1)
            granted = {line.split(":")[0].strip(): line.split(":")[1].split("#")[0].strip() for line in permissions.splitlines()}
            self.assertEqual(granted, {"contents": "read", "actions": "write", "pull-requests": "write"})
            self.assertIn("persist-credentials: false", text)
            # values from the event reach scripts through the environment, never spliced into the script
            for run in re.findall(r"run: \|\n((?:          .+\n)+)", text):
                self.assertNotIn("${{", run)
        preview = job("preview")
        self.assertIn('run: python site/build.py --preview "$PR"', preview)
        self.assertIn("name: preview-pr-${{ github.event.pull_request.number }}", preview)
        self.assertIn("retention-days: 14", preview)
        self.assertIn("gh workflow run site.yml -R \"$REPO\" --ref main", preview)
        self.assertIn("bash .github/scripts/preview-comment.sh live", preview)
        self.assertIn("bash .github/scripts/preview-comment.sh removed", job("unpreview"))

    def test_the_scripts_cap_previews_and_skip_forks(self):
        collect = (SCRIPTS / "collect-previews.sh").read_text()
        self.assertIn('limit="${PREVIEW_LIMIT:-4}"', collect)
        self.assertIn("select(.isCrossRepository | not)", collect)
        self.assertIn('case "$pr" in', collect)          # only numbers become paths
        self.assertIn('-C "$dest/pr-$pr"', collect)
        comment = (SCRIPTS / "preview-comment.sh").read_text()
        self.assertIn("marker='<!-- pr-preview -->'", comment)
        self.assertIn('select(.user.login == \\"github-actions[bot]\\"', comment)
        for script in ("collect-previews.sh", "preview-comment.sh"):
            subprocess.run(["bash", "-n", str(SCRIPTS / script)], check=True)


if __name__ == "__main__":
    unittest.main()
