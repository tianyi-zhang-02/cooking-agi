from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'site'))
import build
import prep_progress


class Markup(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.elements = []
        self.feed(source)

    def handle_starttag(self, tag, attributes):
        self.elements.append((tag, dict(attributes)))


class PrepProgressTests(unittest.TestCase):
    def test_store_and_import_contract(self):
        subprocess.run(['node', '--test', str(ROOT / 'site/tests/prep-progress-store.test.cjs')], check=True)

    def test_reader_api_and_library_state(self):
        subprocess.run(['node', '--test', str(ROOT / 'site/tests/prep-progress.test.cjs'),
                        str(ROOT / 'site/tests/prep-library-state.test.cjs')], check=True)

    def test_catalog_has_one_stable_id_per_pair_and_no_user_data(self):
        with patch.dict(build.NAV), patch.dict(build.BY_SRC):
            navigation = build.load_nav()
            pages, sections = build.discover(navigation)
            first = prep_progress.catalog(build.BY_SRC)
            second = prep_progress.catalog(dict(reversed(list(build.BY_SRC.items()))))
            self.assertEqual(first, second)
            self.assertEqual(len(first['notes']), len(build.BY_SRC))
            known = {page.url for page in pages}
            for identifier, entry in first['notes'].items():
                self.assertEqual(identifier, prep_progress.identifier(entry['source']))
                self.assertTrue(all(page['url'] in known for page in entry['pages'].values()))
                self.assertNotIn('status', entry)
                self.assertNotIn('note', entry)
                if 'en' in entry['pages']:
                    self.assertNotEqual(entry['pages']['zh']['url'], entry['pages']['en']['url'])

    def test_rename_override_preserves_old_identifier(self):
        identifier = prep_progress.identifier('interview/example.md')
        original = Path.read_text
        def source(path, *args, **kwargs):
            if path.name == 'progress-id-overrides.json':
                return json.dumps({'learn/renamed.md': identifier})
            return original(path, *args, **kwargs)
        with patch.object(Path, 'read_text', source):
            self.assertEqual(prep_progress.identifier('learn/renamed.md'), identifier)
            with self.assertRaises(ValueError):
                prep_progress.catalog({'interview/example.md': {}, 'learn/renamed.md': {}})

    def test_note_identity_is_shared_across_languages_and_assets_load_in_order(self):
        with patch.dict(build.NAV), patch.dict(build.BY_SRC):
            navigation = build.load_nav()
            pages, sections = build.discover(navigation)
            template = (ROOT / 'site/template.html').read_text()
            identifiers = []
            for language in ('zh', 'en'):
                page = build.BY_SRC['interview/study-plan.md'][language]
                build.build_page(page, [], navigation['site']['repo'], {item.url for item in pages})
                rendered = build.assemble(page, sections, navigation, 'test', template)
                tools = next(attrs for _, attrs in Markup(rendered).elements if attrs.get('class') == 'note-tools')
                identifiers.append(tools['data-progress-id'])
                order = [rendered.index('static/' + name) for name in
                         ('prep-progress-store.js', 'prep-library-state.js', 'prep-progress.js', 'reader-hub.js')]
                self.assertEqual(order, sorted(order))
                self.assertEqual(rendered.count('data-side-save'), 1)
                for removed in ('data-prep-memo', 'data-prep-note', 'prep-schedule', 'prep-workspace', 'prep-content-updates', 'side-reading'):
                    self.assertNotIn(removed, rendered)
                self.assertNotIn('{{', rendered)
            self.assertEqual(*identifiers)
            self.assertEqual(identifiers[0], prep_progress.identifier('interview/study-plan.md'))

    def test_private_export_names_are_ignored(self):
        names = ['agi-notes-backup-2026-10-14.json', 'agi-notes-bookmarks.json',
                 'agi-prep-progress-2026-10-11.json', '.study-progress/my-progress.json']
        result = subprocess.run(['git', 'check-ignore', *names], cwd=ROOT, text=True,
                                capture_output=True, check=True)
        self.assertEqual(len(result.stdout.splitlines()), len(names))


if __name__ == '__main__':
    unittest.main()
