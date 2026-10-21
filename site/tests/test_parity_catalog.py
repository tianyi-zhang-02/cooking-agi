from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'site'))
import build
import paritycheck


class ParityCatalogTests(unittest.TestCase):
    def test_scope_matches_every_published_pair(self):
        pages, _ = build.discover(build.load_nav())
        expected = {page.src for page in pages if page.lang == 'zh'}
        actual = {ROOT / 'README.zh.md' if source == ROOT / 'README.md' else source
                  for source in paritycheck.published_sources(build.load_nav(), ROOT)}
        self.assertEqual(actual, expected)
        self.assertIn(ROOT / 'README.zh.md', actual)
        self.assertIn(ROOT / 'templates/paper-note.md', actual)

    def test_root_order_includes_templates_and_excludes_private_notes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('README.md', 'README.zh.md', 'README.en.md', 'contributors.md',
                         'EDITORIAL.md', 'PRIVATE.md', 'templates/paper-note.md',
                         'templates/paper-note.en.md'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# Test')
            navigation = {'section': [
                {'dir': '.', 'order': ['README.md', 'contributors.md']},
                {'dir': '.', 'order': ['EDITORIAL.md']},
                {'dir': 'templates'},
            ]}
            self.assertEqual(
                {path.relative_to(root).as_posix() for path in paritycheck.published_sources(navigation, root)},
                {'README.md', 'contributors.md', 'EDITORIAL.md', 'templates/paper-note.md'},
            )

    def test_included_note_is_checked_once_without_including_its_neighbors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('topic/README.md', 'topic/shared.md', 'outside/selected.md',
                         'outside/private.md', 'topic/shared.en.md'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# Test')
            navigation = {'section': [
                {'dir': 'topic'},
                {'dir': 'missing', 'include': ['topic/shared.md', 'outside/selected.md', 'outside/absent.md']},
            ]}
            sources = list(paritycheck.published_sources(navigation, root))
            self.assertEqual(len(sources), 3)
            self.assertEqual({path.relative_to(root).as_posix() for path in sources},
                             {'topic/README.md', 'topic/shared.md', 'outside/selected.md'})

    def test_published_template_has_both_system_and_research_sections(self):
        chinese = (ROOT / 'templates/paper-note.md').read_text()
        english = (ROOT / 'templates/paper-note.en.md').read_text()
        self.assertEqual(paritycheck.shape(chinese), paritycheck.shape(english))
        self.assertIn('## Place in the wider system', english)
        self.assertIn('## Connections to other research', english)
        rendered, unused_toc = build.render_markdown(english)
        self.assertIn('<h2 id="connection-to-my-research-map">Connections to other research</h2>', rendered)
        rendered, unused_toc = build.render_markdown(chinese)
        self.assertIn('<h2 id="claim_1">什么结果会推翻这个结论？</h2>', rendered)


if __name__ == '__main__':
    unittest.main()
