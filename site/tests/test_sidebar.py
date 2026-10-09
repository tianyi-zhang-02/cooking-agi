import copy
from html.parser import HTMLParser
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build


class Elements(HTMLParser):
    def __init__(self, markup):
        super().__init__()
        self.elements = []
        self.feed(markup)

    def handle_starttag(self, tag, attributes):
        self.elements.append((tag, dict(attributes)))


class SidebarTests(unittest.TestCase):
    def setUp(self):
        self.old_nav = build.NAV
        self.old_sources = build.BY_SRC
        build.NAV = {
            "category": [
                {"id": "learn", "zh": "学习笔记", "en": "Study notes", "home": "learn/README.md"},
                {"id": "career", "zh": "求职", "en": "Career", "home": "career/README.md"},
            ],
            "group": [
                {"id": "basics", "zh": "基础", "en": "Basics", "category": "learn"},
                {"id": "jobs", "zh": "求职准备", "en": "Preparation", "category": "career"},
            ],
        }
        self.sections = []
        build.BY_SRC = {}
        for directory, group in [("learn/core", "basics"), ("learn/models", "basics"), ("career", "jobs")]:
            section = {"dir": directory, "group": group, "zh": directory, "en": directory, "pages": []}
            for name in ["README", "example"]:
                pair = {}
                for language in ["zh", "en"]:
                    suffix = ".en" if language == "en" else ""
                    stem = "index" if name == "README" else name
                    pair[language] = SimpleNamespace(
                        src=build.ROOT / directory / f"{name}{suffix}.md",
                        url=f"{directory}/{stem}{suffix}.html",
                        title='A & B <example> "title"', section=section, lang=language,
                        rel=lambda value: "../../" + value,
                    )
                section["pages"].append(pair)
                build.BY_SRC[f"{directory}/{name}.md"] = pair
            self.sections.append(section)
        self.page = self.sections[0]["pages"][1]["zh"]

    def tearDown(self):
        build.NAV = self.old_nav
        build.BY_SRC = self.old_sources

    def render(self, page=None):
        return build.sidebar_html(page or self.page, self.sections, build.NAV["group"])

    def test_directory_shows_current_category_but_keeps_global_reading_catalog(self):
        markup = self.render()
        elements = Elements(markup).elements
        categories = [attrs for tag, attrs in elements if tag == "li" and "data-category" in attrs]
        self.assertTrue(categories)
        for attrs in categories:
            self.assertEqual("hidden" in attrs, attrs["data-category"] != "learn")
        self.assertIn('href="../../career/example.html"', markup)
        self.assertNotIn('side-scope', markup)
        self.assertNotIn('data-side-locate', markup)
        self.assertIn('data-side-collapse', markup)
        self.assertIn('data-side-current', markup)
        self.assertIn('定位本章', markup)

    def test_current_zone_is_distinct_and_counts_are_labeled(self):
        build.NAV['zone'] = [{'id': 'models', 'zh': '模型与多模态', 'en': 'Models & multimodal'}]
        build.NAV['group'][0]['zone'] = 'models'
        markup = self.render()
        self.assertIn('zone is-current', markup)
        self.assertIn('<small>当前</small>', markup)
        self.assertIn('4 篇', markup)

    def test_only_active_ancestors_start_open(self):
        elements = Elements(self.render()).elements
        opened = [attrs.get("data-grp") or attrs.get("data-chapter")
                  for tag, attrs in elements if tag == "details" and "open" in attrs]
        self.assertEqual(opened, ["basics", "learn/core"])
        active = [attrs for tag, attrs in elements if tag == "a" and attrs.get("class") == "active"]
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["aria-current"], "page")

    def test_bilingual_links_share_bookmark_identity(self):
        versions = []
        for language in ["zh", "en"]:
            markup = self.render(self.sections[0]["pages"][1][language])
            versions.append([attrs["data-note"] for tag, attrs in Elements(markup).elements
                             if "data-note" in attrs])
        self.assertEqual(versions[0], versions[1])
        self.assertEqual(len(versions[0]), len(set(versions[0])))
        self.assertIn('example.en.html', markup)
        self.assertIn('My reading', markup)
        self.assertNotIn('我的阅读', markup)

    def test_titles_escaped(self):
        markup = self.render()
        self.assertIn('&lt;example&gt;', markup)
        self.assertNotIn('<example>', markup)

    def test_controls_have_no_javascript_fallback(self):
        elements = Elements(self.render()).elements
        controls = next(attrs for tag, attrs in elements if attrs.get("class") == "side-tools")
        self.assertIn("hidden", controls)
        self.assertTrue(any(tag == "details" for tag, attrs in elements))
        ids = [attrs["id"] for tag, attrs in elements if "id" in attrs]
        self.assertEqual(len(ids), len(set(ids)))
        for tag, attrs in elements:
            if "aria-controls" in attrs:
                self.assertIn(attrs["aria-controls"], ids)

    def test_home_does_not_repeat_global_categories(self):
        page = copy.copy(self.page)
        page.url = "index.html"
        markup = self.render(page)
        categories = [attrs for tag, attrs in Elements(markup).elements if "data-category" in attrs]
        self.assertTrue(all("hidden" in attrs for attrs in categories))
        self.assertIn('从顶部选择一个板块', markup)

    def test_future_categories_use_config_not_frontend_edits(self):
        build.NAV["category"].append({"id": "new", "zh": "新板块", "en": "New section"})
        build.NAV["group"][0]["category"] = "new"
        markup = self.render()
        self.assertIn('<div class="side-context">新板块</div>', markup)
        self.assertIn('data-category="new"><details', markup)

    def test_single_note_does_not_need_an_extra_accordion(self):
        self.sections = self.sections[:1]
        self.sections[0]["pages"] = self.sections[0]["pages"][:1]
        markup = self.render()
        self.assertIn('grp-single', markup)
        self.assertFalse(any(tag == "details" and ("data-grp" in attrs or "data-chapter" in attrs)
                             for tag, attrs in Elements(markup).elements))

    @unittest.skipUnless(shutil.which("node"), "Node is required for sidebar state tests")
    def test_reading_state_is_bounded_and_filters_untrusted_values(self):
        script = """
const assert = require('node:assert/strict');
const {cleanReading, recordVisit, importBookmarks} = require(process.argv[1]);
const known = new Set(Array.from({length: 110}, (_, index) => 'note-' + index));
assert.deepEqual(cleanReading(null, known), {saved: [], recent: []});
assert.deepEqual(cleanReading({saved: 'bad', recent: {}}, known), {saved: [], recent: []});
assert.deepEqual(cleanReading({saved: ['note-1', 'note-1', 'javascript:alert(1)', {}, null], recent: ['note-2']}, known), {saved: ['note-1'], recent: ['note-2']});
const bounded = cleanReading({saved: [...known], recent: [...known]}, known);
assert.equal(bounded.saved.length, 100);
assert.equal(bounded.recent.length, 8);
const before = {saved: ['note-0'], recent: ['note-2', 'note-1']};
const after = recordVisit(before, 'note-1');
assert.deepEqual(after, {saved: ['note-0'], recent: ['note-1', 'note-2']});
assert.deepEqual(before.recent, ['note-2', 'note-1']);
assert.equal(recordVisit(bounded, 'note-109').recent.length, 8);
const backup = entries => JSON.stringify({format: 'cooking-agi-bookmarks', version: 1, saved: entries});
const imported = importBookmarks(backup(['note-0', 'note-1', 'note-1', 'javascript:alert(1)', 'removed.md']), before, known);
assert.deepEqual(imported.state, {saved: ['note-0', 'note-1'], recent: ['note-2', 'note-1']});
assert.equal(imported.added, 1);
assert.equal(imported.unavailable, 2);
assert.deepEqual(before.saved, ['note-0']);
assert.equal(importBookmarks(backup(['note-109']), bounded, known).overflow, 1);
assert.equal(importBookmarks(backup(['note-109']), bounded, known).state.saved.length, 100);
for (const invalid of ['null', '{', '{}', JSON.stringify({format:'other', version:1,saved:[]}), backup([{}]), backup(['x'.repeat(301)]), backup(Array(1001).fill('note-1')), 'x'.repeat(131073)]) {
  assert.throws(() => importBookmarks(invalid, before, known));
}
"""
        result = subprocess.run([shutil.which("node"), "-e", script,
                                 str(build.ROOT / "site/static/sidebar.js")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
