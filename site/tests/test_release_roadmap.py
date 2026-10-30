from html.parser import HTMLParser
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch
from urllib.parse import unquote, urljoin, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build
import paritycheck


class Markup(HTMLParser):
    def __init__(self, markup):
        super().__init__()
        self.elements = []
        self.feed(markup)

    def handle_starttag(self, tag, attributes):
        self.elements.append((tag, dict(attributes)))


class ReleaseRoadmapTests(unittest.TestCase):
    def test_roadmap_pairs_have_matching_sections_and_current_plan(self):
        for filename in ('roadmap', 'updates'):
            chinese = (build.ROOT / f'community/{filename}.md').read_text()
            english = (build.ROOT / f'community/{filename}.en.md').read_text()
            self.assertEqual(paritycheck.shape(chinese), paritycheck.shape(english))
            self.assertEqual(re.findall(r'\{#([^}]+)\}', chinese),
                             re.findall(r'\{#([^}]+)\}', english))
            for source in (chinese, english):
                # the interview-prep release was dropped; the site deploys on every merge
                self.assertNotIn('{#v1-0-0}', source)
                self.assertIn('#changelog', source)
                if filename == 'roadmap':
                    for planned in ('#128', 'MCP', 'Jetson', 'CV'):
                        self.assertIn(planned, source)

    def test_roadmap_belongs_to_community_not_a_top_level_subject(self):
        nav = build.load_nav()
        community = next(section for section in nav['section'] if section['dir'] == 'community')
        self.assertEqual(community['group'], 'start')
        self.assertEqual(community['order'][:3], ['README.md', 'roadmap.md', 'updates.md'])
        self.assertFalse(any(category['home'] in {'community/roadmap.md', 'community/updates.md'}
                             for category in nav['category']))

    def test_public_entry_points_link_to_the_version_and_plan(self):
        for filename, suffix in (('README.md', '.en'), ('README.zh.md', '')):
            source = (build.ROOT / filename).read_text()
            for target in ('updates', 'roadmap'):
                self.assertIn(f'(community/{target}{suffix}.md)', source)
        contribution = (build.ROOT / 'CONTRIBUTING.md').read_text()
        self.assertIn('community/roadmap.md#contribute', contribution)
        self.assertIn('community/roadmap.en.md#contribute', contribution)
        for suffix in ('', '.en'):
            source = (build.ROOT / f'community/README{suffix}.md').read_text()
            self.assertIn(f'(roadmap{suffix}.md)', source)
            self.assertIn(f'(updates{suffix}.md)', source)
            self.assertIn('<span id="review"></span>', source)

    def test_rendered_local_links_and_anchors_resolve_in_both_languages(self):
        with patch.dict(build.NAV), patch.dict(build.BY_SRC):
            nav = build.load_nav()
            pages, sections = build.discover(nav)
            known = {page.url for page in pages}
            catalog = {page.url: page for page in pages}
            for filename in ('roadmap', 'updates'):
                for language in ('zh', 'en'):
                    page = build.BY_SRC[f'community/{filename}.md'][language]
                    self.assertEqual(page.kind, 'project')
                    header = build.page_header_html(page)
                    self.assertNotIn('data-prep-note', header)
                    self.assertNotIn('page-position', header)
                    self.assertIn('项目进展' if language == 'zh' else 'Project updates', header)
                    build.build_page(page, [], nav['site']['repo'], known)
                    self.assertNotIn('.md)', page.body)
                    self.assertNotIn('<script', page.body)
                    for tag, attrs in Markup(page.body).elements:
                        if tag != 'a' or 'href' not in attrs:
                            continue
                        target = urlsplit(urljoin('https://local.test/' + page.url, attrs['href']))
                        if target.netloc != 'local.test':
                            continue
                        destination = catalog[target.path.lstrip('/')]
                        self.assertEqual(destination.lang, language, attrs['href'])
                        if target.fragment:
                            build.build_page(destination, [], nav['site']['repo'], known)
                            identifiers = {item.get('id') for _, item in Markup(destination.body).elements}
                            self.assertIn(unquote(target.fragment), identifiers, attrs['href'])

    def test_proposal_reuses_existing_form_and_has_unique_ids(self):
        source = (build.ROOT / '.github/ISSUE_TEMPLATE/proposal.yml').read_text()
        identifiers = re.findall(r'^    id: (.+)$', source, re.M)
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertIn('id: direction', source)
        self.assertIn('roadmap.en.md', source)
        for topic in ('Computer vision', 'Autonomous driving', 'Jetson', 'Physical simulation', 'Robotics'):
            self.assertIn(topic, source)


if __name__ == '__main__':
    unittest.main()
