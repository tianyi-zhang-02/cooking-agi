from html.parser import HTMLParser
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build


class ReadingElements(HTMLParser):
    def __init__(self, markup):
        super().__init__()
        self.elements = []
        self.feed(markup)

    def handle_starttag(self, tag, attributes):
        self.elements.append((tag, dict(attributes)))


class ReadingModesTests(unittest.TestCase):
    def setUp(self):
        self.chinese = SimpleNamespace(lang="zh", url="learn/example.html", rel=lambda url: "../" + url)
        self.english = SimpleNamespace(lang="en", url="learn/example.en.html", rel=lambda url: "../" + url)
        self.chinese.sibling = self.english
        self.english.sibling = self.chinese

    def test_complete_page_links_work_without_javascript(self):
        for page in (self.chinese, self.english):
            elements = ReadingElements(build.reading_controls_html(page)).elements
            links = [attrs for tag, attrs in elements if tag == "a"]
            self.assertEqual([link["href"] for link in links], ["../learn/example.html", "../learn/example.en.html"])
            current = [link for link in links if link.get("aria-current") == "page"]
            self.assertEqual(len(current), 1)
            self.assertEqual(current[0]["href"], "../" + page.url)
            self.assertEqual([link["hreflang"] for link in links], ["zh-Hans", "en"])

    def test_annotations_are_independent_and_chinese_only(self):
        chinese = build.reading_controls_html(self.chinese)
        english = build.reading_controls_html(self.english)
        self.assertIn("data-term-toggle", chinese)
        self.assertIn('class="reading-terms" hidden', chinese)
        self.assertNotIn("data-term-toggle", english)
        self.assertIn("Page language", english)
        self.assertIn("complete page", english)
        self.assertNotIn("正文语言", english)

    def test_missing_counterpart_does_not_link_to_nowhere(self):
        self.chinese.sibling = None
        markup = build.reading_controls_html(self.chinese)
        links = [attrs for tag, attrs in ReadingElements(markup).elements if tag == "a"]
        self.assertEqual(len(links), 1)
        self.assertNotIn('href="#"', markup)

    def test_annotation_switch_preserves_english_and_other_content(self):
        css = (build.SITE / "static/style.css").read_text()
        self.assertIn(':root.lang-zh[data-english-terms="off"] .term-en { display: none; }', css)
        template = (build.SITE / "template.html").read_text()
        self.assertIn("{{reading_controls}}", template)
        self.assertIn("static/reading.js", template)
        javascript = (build.SITE / "static/reading.js").read_text()
        self.assertNotIn("navigator.language", javascript)
        self.assertNotIn("location", javascript)
        self.assertNotIn("fetch(", javascript)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_preference_and_keyboard_behavior(self):
        result = subprocess.run([shutil.which("node"), "-e", r'''
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const source = fs.readFileSync(process.argv[1], "utf8");
function mount(saved, blocked = false) {
  const events = {}, changes = {}, writes = [];
  const root = { dataset: {} };
  const label = { hidden: true };
  const toggle = { checked: true, closest: () => label,
    addEventListener: (name, handler) => { changes[name] = handler; } };
  let focused = false;
  const menu = { open: false, contains: element => element === toggle,
    querySelector: () => ({ focus: () => { focused = true; } }) };
  const document = { documentElement: root, activeElement: toggle,
    querySelectorAll: selector => selector === "[data-term-toggle]" ? [toggle] : [menu],
    addEventListener: (name, handler) => { events[name] = handler; } };
  const storage = { getItem: () => { if (blocked) throw Error("blocked"); return saved; },
    setItem: (...values) => { if (blocked) throw Error("blocked"); writes.push(values); } };
  vm.runInNewContext(source, { document, localStorage: storage });
  return { root, label, toggle, menu, changes, events, writes, focused: () => focused };
}
const fresh = mount(null);
assert.equal(fresh.root.dataset.englishTerms, "on");
assert.equal(fresh.label.hidden, false);
fresh.toggle.checked = false;
fresh.changes.change();
assert.equal(fresh.root.dataset.englishTerms, "off");
assert.equal(fresh.writes[0][1], "off");
assert.equal(mount("off").toggle.checked, false);
assert.equal(mount("on").toggle.checked, true);
assert.equal(mount("invalid").toggle.checked, true);
const blocked = mount(null, true);
blocked.toggle.checked = false;
assert.doesNotThrow(() => blocked.changes.change());
assert.equal(blocked.root.dataset.englishTerms, "off");
fresh.menu.open = true;
fresh.events.keydown({ key: "Escape" });
assert.equal(fresh.menu.open, false);
assert.equal(fresh.focused(), true);
fresh.menu.open = true;
fresh.events.click({ target: fresh.toggle });
assert.equal(fresh.menu.open, true);
fresh.events.click({ target: {} });
assert.equal(fresh.menu.open, false);
''', str(build.SITE / "static/reading.js")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
