from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build
import learning_views


class Elements(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.elements = []
        self.feed(source)

    def handle_starttag(self, tag, attributes):
        self.elements.append((tag, dict(attributes)))


class LearningViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outputs = {}
        cls.bodies = {}
        with patch.dict(build.NAV), patch.dict(build.BY_SRC):
            navigation = build.load_nav()
            pages, sections = build.discover(navigation)
            known = {page.url for page in pages}
            for source in (learning_views.SOURCE, '00-foundations/core/multi-head-attention.md'):
                for language in ('zh', 'en'):
                    page = build.BY_SRC[source][language]
                    build.build_page(page, build.load_glossary(), navigation['site']['repo'], known)
                    cls.bodies[source, language] = page.body
                    cls.outputs[source, language] = build.assemble(page, sections, navigation, 'test',
                                                                  (build.SITE / 'template.html').read_text())

    def test_full_note_and_existing_heading_ids_are_retained_verbatim(self):
        for language in ('zh', 'en'):
            body = self.bodies[learning_views.SOURCE, language]
            output = self.outputs[learning_views.SOURCE, language]
            self.assertIn(f'<div id="kv-full" data-learning-panel="full">{body}</div>', output)
            self.assertIn('M_{KV}', body)
            elements = Elements(output).elements
            identifiers = [attrs['id'] for _, attrs in elements if 'id' in attrs]
            self.assertEqual(len(identifiers), len(set(identifiers)))
            full = next(attrs for _, attrs in elements if attrs.get('id') == 'kv-full')
            self.assertNotIn('hidden', full)

    def test_progressive_enhancement_and_localization(self):
        for language in ('zh', 'en'):
            output = self.outputs[learning_views.SOURCE, language]
            elements = Elements(output).elements
            for identifier in ('kv-quick', 'kv-lab'):
                attrs = next(attrs for _, attrs in elements if attrs.get('id') == identifier)
                self.assertIn('hidden', attrs)
            navigation = next(attrs for _, attrs in elements if 'data-learning-nav' in attrs)
            self.assertIn('hidden', navigation)
            panels = learning_views.panels(language)
            self.assertNotIn('data-i18n', panels)
            if language == 'en':
                self.assertNotRegex(panels, r'[\u3400-\u9fff]')
            else:
                self.assertIn('刚预测出 D', panels)
            self.assertIn('?view=full', panels)
            self.assertNotRegex(panels, r'<h2\b[^>]*\bid=')
        with self.assertRaises(ValueError):
            learning_views.panels('../outside')

    def test_assets_are_only_loaded_for_the_pilot_note(self):
        for (source, _), output in self.outputs.items():
            self.assertEqual('static/learning-views.js' in output, source == learning_views.SOURCE)
            self.assertEqual('static/learning-views.css' in output, source == learning_views.SOURCE)

    def test_no_new_storage_or_progress_protocol(self):
        script = (build.SITE / 'static/learning-views.js').read_text()
        for marker in ('localStorage', 'sessionStorage', 'PreparationProgress', 'fetch(', 'XMLHttpRequest'):
            self.assertNotIn(marker, script)
        styles = (build.SITE / 'static/learning-views.css').read_text()
        self.assertIn('@media print', styles)
        self.assertIn('prefers-reduced-motion', styles)
        self.assertIn('var(--text-scale, 1)', styles)

    @unittest.skipUnless(shutil.which('node'), 'Node is required')
    def test_step_model_and_url_contract(self):
        result = subprocess.run([shutil.which('node'), '--test', str(build.SITE / 'tests/learning-views.test.cjs')],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
