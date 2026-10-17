from html.parser import HTMLParser
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "site"))
import build
import paritycheck


class HubMarkup(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.elements = []
        self.feed(text)

    def handle_starttag(self, tag, attributes):
        self.elements.append((tag, dict(attributes)))


class PrepHubTests(unittest.TestCase):
    def test_bilingual_structure_and_study_plan_anchors(self):
        for chapter in ("README", "study-plan"):
            sources = [(ROOT / f"interview/{chapter}{suffix}.md").read_text()
                       for suffix in ("", ".en")]
            self.assertEqual(paritycheck.shape(sources[0]), paritycheck.shape(sources[1]))
            self.assertEqual(re.findall(r"\{#([^}]+)\}", sources[0]),
                             re.findall(r"\{#([^}]+)\}", sources[1]))
            for source in sources:
                self.assertFalse(paritycheck.source_issues(source))

    def test_landing_is_a_plain_directory(self):
        # The quick-review / study-plan tabs and the familiarity picker were removed on
        # 2026-10-14: readers pick the part they need themselves.
        for suffix in ("", ".en"):
            source = (ROOT / f"interview/README{suffix}.md").read_text()
            body, _ = build.render_markdown(source)
            markup = HubMarkup(body)
            self.assertFalse([attrs for _, attrs in markup.elements if "data-prep-tab" in attrs or "data-prep-panel" in attrs])
            identifiers = [attrs['id'] for _, attrs in markup.elements if 'id' in attrs]
            self.assertEqual(len(identifiers), len(set(identifiers)))
            links = [attrs['href'] for tag, attrs in markup.elements if tag == 'a']
            for target in (f'basics/README{suffix}.md', f'../learn/ml-exercises/README{suffix}.md', f'leetcode{suffix}.md',
                           f'code-comprehension{suffix}.md', f'../learn/system-design/README{suffix}.md',
                           f'../career/prepare{suffix}.md', f'ai-roles-prep{suffix}.md'):
                self.assertIn(target, links)

    def test_legacy_landing_anchors_survive(self):
        for suffix, anchors in [('', ['prep-routes', 'review', 'plan', 'ml', 'python', '_1', '_2']),
                                ('.en', ['prep-routes', 'review', 'plan', 'ml-questions-implementations',
                                         'python-algorithms', 'system-design', 'how-to-review',
                                         'code-exercises-turn-understanding-into-practice'])]:
            body, _ = build.render_markdown((ROOT / f'interview/README{suffix}.md').read_text())
            identifiers = [attrs.get('id') for _, attrs in HubMarkup(body).elements]
            for anchor in anchors:
                self.assertIn(anchor, identifiers)

    def test_hub_controls_do_not_get_inline_glossary_chips(self):
        source = '<div class="prep-hub"><p>梯度</p></div><p>梯度</p>'
        result, _ = build.annotate(source, [('梯度', 'gradient', '变化方向')])
        self.assertTrue(result.startswith('<div class="prep-hub"><p>梯度</p></div>'))
        self.assertEqual(result.count('class="term"'), 1)

    def test_only_the_hub_gets_the_hub_layout_and_assets(self):
        with patch.dict(build.NAV), patch.dict(build.BY_SRC):
            navigation = build.load_nav()
            pages, sections = build.discover(navigation)
            urls = {page.url for page in pages}
            template = (ROOT / 'site/template.html').read_text()
            self.assertEqual(sum(area['id'] == 'interview' for area in navigation['category']), 1)
            for chapter in ('README', 'study-plan', 'speedrun'):
                for language in ('zh', 'en'):
                    page = build.BY_SRC[f'interview/{chapter}.md'][language]
                    build.build_page(page, [], navigation['site']['repo'], urls)
                    result = build.assemble(page, sections, navigation, 'test', template)
                    self.assertEqual('page-prep-hub' in result, chapter == 'README')
                    self.assertNotIn('static/prep-hub.js', result)
                    self.assertEqual('static/prep-hub.css' in result, chapter == 'README')
                    if chapter == 'README':
                        self.assertNotIn('class="page-turn"', result)
                    self.assertNotIn('.md"', page.body)

if __name__ == '__main__':
    unittest.main()
