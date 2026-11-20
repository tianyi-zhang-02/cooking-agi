from pathlib import Path
import re
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'site'))
import leakcheck


class PublicSourceBoundaryTests(unittest.TestCase):
    def test_new_numbered_sections_are_scanned_automatically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("00-foundations", "14-production-ml-and-recs", "18-product-b2b-and-technical-leadership", "_site"):
                (root / name).mkdir()
            targets = leakcheck.note_directories(root)
            self.assertIn("14-production-ml-and-recs", targets)
            self.assertIn("18-product-b2b-and-technical-leadership", targets)
            self.assertIn("learn", targets)
            self.assertNotIn("_site", targets)

    def test_public_attribution_does_not_require_a_profile(self):
        crew = {"crew": [{"login": "example-owner"}], "ai": [{"name": "Example tool"}]}
        self.assertEqual(leakcheck.owner_profile_issues("example-owner", {"records": []}, crew), [])

    def test_old_consent_does_not_republish_owner_career_data(self):
        outcomes = {"records": [{"github": "Example-Owner", "consent": True, "title": "Example role"}]}
        self.assertEqual(leakcheck.owner_profile_issues("example-owner", outcomes, {}),
                         [("community/outcomes.json", "owner career record")])

    def test_owner_location_and_plans_are_not_attribution(self):
        for field in ("country", "countries", "email", "plan", "employer"):
            with self.subTest(field=field):
                crew = {"crew": [{"login": "example-owner", field: "example"}]}
                self.assertTrue(leakcheck.owner_profile_issues("example-owner", {}, crew))

    def test_other_consented_contribution_records_are_not_changed(self):
        outcomes = {"records": [{"github": "another-reader", "consent": True}]}
        crew = {"crew": [{"login": "another-reader", "country": "XX"}]}
        self.assertEqual(leakcheck.owner_profile_issues("example-owner", outcomes, crew), [])

    def test_local_paths_are_caught_without_blocking_public_sources(self):
        pattern = next(rule[-1] for rule in leakcheck.RULES if rule[0] == "local personal path")
        for value in ("/Users/example/private.md", "/home/example/notes.md", r"C:\Users\example\notes.md", "file:///Users/example/private.md"):
            self.assertIsNotNone(re.search(pattern, value, re.IGNORECASE))
        for value in ("https://example.org/notes/overview", "https://github.com/twitter/the-algorithm/blob/main/home/recap/README.md"):
            self.assertIsNone(re.search(pattern, value, re.IGNORECASE))

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
