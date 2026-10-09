from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'site'))
import leakcheck


class PublicSourceBoundaryTests(unittest.TestCase):
    def test_pinned_public_source_is_not_an_employer_disclosure(self):
        url = next(iter(leakcheck.PUBLIC_REFERENCE_URLS))
        line = f'Read [implementation]({url}) for cache behavior.'
        redacted = leakcheck.redact_public_reference_urls(line)
        self.assertNotIn(url, redacted)
        self.assertIn('for cache behavior.', redacted)

    def test_url_does_not_exempt_the_rest_of_the_line(self):
        url = next(iter(leakcheck.PUBLIC_REFERENCE_URLS))
        line = f'[source]({url}) private employer information: ByteDance'
        redacted = leakcheck.redact_public_reference_urls(line)
        pattern = next(rule[-1] for rule in leakcheck.RULES if rule[0] == 'employer / internal term')
        self.assertIsNotNone(re.search(pattern, redacted, re.IGNORECASE))

    def test_near_matches_are_not_allowed(self):
        url = next(iter(leakcheck.PUBLIC_REFERENCE_URLS))
        for modified in (url + '/private', url + '?private=true', url.replace('7ea635ba1575ae9ab4ae1d83d83e16a6e47fe696', 'main')):
            self.assertEqual(leakcheck.redact_public_reference_urls(modified), modified)


if __name__ == '__main__':
    unittest.main()
