import copy
import html
import sys
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import collaboration


class CollaborationTests(unittest.TestCase):
    def setUp(self):
        self.config = collaboration.load_config()
        self.nav = tomllib.loads((collaboration.ROOT / "site/nav.toml").read_text())
        self.repo = self.nav["site"]["repo"]

    def test_all_groups_have_exactly_one_area(self):
        collaboration.validate(self.config, self.nav)
        broken = copy.deepcopy(self.config)
        broken["area"][1]["groups"].append("foundations")
        with self.assertRaises(ValueError):
            collaboration.validate(broken, self.nav)

    def test_new_group_requires_contact(self):
        self.nav["group"].append({"id": "new-topic"})
        with self.assertRaises(ValueError):
            collaboration.validate(self.config, self.nav)

    def test_unassigned_area_uses_maintainer(self):
        area = collaboration.area_for(self.config, "eval")
        area["reviewers"] = []
        self.assertEqual(collaboration.owners_for(self.config, area), self.config["maintainers"])

    def test_named_reviewer_and_protected_paths(self):
        area = collaboration.area_for(self.config, "eval")
        area["reviewers"] = ["example-reviewer"]
        output = collaboration.codeowners(self.config, self.nav)
        fallback = " ".join(f"@{login}" for login in self.config["maintainers"])
        self.assertIn("/07-evaluation/ @example-reviewer", output)
        self.assertIn(f"/.github/ {fallback}", output)
        self.assertIn(f"/site/ {fallback}", output)
        self.assertFalse(any(line.startswith("/ @") for line in output.splitlines()))

    def test_malicious_login_rejected(self):
        self.config["maintainers"] = ['bad\" onclick=\"bad']
        with self.assertRaises(ValueError):
            collaboration.validate(self.config, self.nav)

    def test_language_pairs_share_discussion(self):
        panels = []
        for lang, filename in [("zh", "README.md"), ("en", "README.en.md")]:
            page = SimpleNamespace(src=collaboration.ROOT / "07-evaluation" / filename,
                                   section={"group": "eval"}, lang=lang, rel=lambda value: "../" + value)
            panels.append(collaboration.page_panel(page, self.config, self.repo))
        url = collaboration.discussion_url(self.repo, "evaluation", "07-evaluation/README.md")
        self.assertTrue(all(url in panel for panel in panels))
        self.assertIn("community/index.en.html", panels[1])
        self.assertIn("<details", panels[0])

    def test_issue_prefill_encodes_paths(self):
        url = html.unescape(collaboration.feedback_url(self.repo, "evaluation", "test & more.md"))
        query = parse_qs(urlsplit(url).query)
        self.assertEqual(query["area"], ["Area: evaluation"])
        self.assertEqual(query["source"], ["test & more.md"])
        self.assertEqual(query["template"], ["note-feedback.yml"])

    def test_credit_requires_public_repo_evidence(self):
        self.config["acknowledgements"] = [{"login": "example", "zh": "纠错", "en": "Correction",
                                           "evidence": ["javascript:alert(1)"]}]
        with self.assertRaises(ValueError):
            collaboration.validate(self.config, self.nav)

    def test_credit_text_is_escaped(self):
        self.config["acknowledgements"] = [{"login": "example", "zh": "<script>bad</script>", "en": "Correction",
                                           "evidence": [f"https://github.com/{self.repo}/issues/1"]}]
        collaboration.validate(self.config, self.nav)
        output = collaboration.acknowledgements_html(self.config, True)
        self.assertNotIn("<script>", output)
        self.assertIn("&lt;script&gt;", output)

    def test_no_fabricated_credits(self):
        self.config["acknowledgements"] = []
        self.assertIn("no extra commits needed", collaboration.acknowledgements_html(self.config, False))


if __name__ == "__main__":
    unittest.main()
