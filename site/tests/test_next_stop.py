import copy
import re
import shutil
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import next_stop


def fixture():
    return {"version": 1, "companies": {"example": "Example <Company>"}, "records": [
        {"id": "NS-0001", "start_date": "2027", "company": "example", "role": "ml",
         "employment": "full-time", "milestone": "offer", "consent": True,
         "source_issue": 1, "note": {"zh": "测试 <script> 内容", "en": "Test <script> content"}},
        {"id": "NS-0002", "start_date": "2027-02", "company": "example", "role": "software",
         "employment": "internship", "milestone": "started", "consent": True,
         "source_issue": 2, "name": 'Test "reader"', "github": "example-reader"},
        {"id": "NS-0003", "start_date": "2026-12", "company": "example", "role": "research",
         "employment": "full-time", "milestone": "started", "consent": True, "source_issue": 3}
    ]}


class NextStopTests(unittest.TestCase):
    def render(self, language="zh", data=None):
        return next_stop.render(SimpleNamespace(lang=language, depth=1, rel=lambda path: "../" + path),
                                "example/notes", fixture() if data is None else data)

    def test_repository_data_is_valid(self):
        next_stop.load()

    def test_empty_state_does_not_invent_outcomes(self):
        for language in ("zh", "en"):
            output = self.render(language, {"version": 1, "companies": {}, "records": []})
            self.assertIn('data-stop-empty >', output)
            self.assertIn('0 ', output)
            self.assertNotIn('data-stop-record', output)
            self.assertNotIn('data-stop-company=', output)

    def test_bilingual_records_and_controls_are_equivalent(self):
        chinese, english = self.render(), self.render("en")
        for output in (chinese, english):
            self.assertEqual(output.count('data-stop-record'), 3)
            self.assertEqual(output.count('data-stop-filter='), 3)
            self.assertEqual(output.count('data-stop-view='), 2)
            self.assertIn('data-stop-count>3</b>', output)
            self.assertNotIn('/issues/1', output)
        self.assertIn('一位读者', chinese)
        self.assertIn('A reader', english)
        self.assertNotRegex(re.sub(r'<[^>]+>', '', english), r'[\u4e00-\u9fff]{3,}')

    def test_chronology_does_not_invent_months(self):
        records = next_stop.ordered(fixture()["records"])
        self.assertEqual([record["id"] for record in records], ["NS-0002", "NS-0001", "NS-0003"])
        output = self.render("en")
        self.assertIn('datetime="2027">Expected 2027 · month not shared', output)
        self.assertLess(output.index('id="NS-0002"'), output.index('id="NS-0001"'))

    def test_start_month_is_readable_in_both_languages_and_views(self):
        data = fixture()
        data["records"][1]["start_date"] = "2026-05"
        for language, label in (("zh", "2026 年 5 月"), ("en", "May 2026")):
            with self.subTest(language=language):
                output = self.render(language, data)
                self.assertIn(f'<time datetime="2026-05">{label}</time>', output)
                self.assertIn(f'<span class="crew-handle">{label} ·', output)
                self.assertNotIn('Expected May 2026', output)
                self.assertNotIn('预计 2026 年 5 月', output)
                self.assertIn('data-year="2026"', output)

    def test_progress_changes_do_not_change_start_date_order(self):
        data = fixture()
        expected = ["NS-0002", "NS-0001", "NS-0003"]
        for milestone in next_stop.MILESTONES:
            data["records"][1]["milestone"] = milestone
            self.assertEqual([record["id"] for record in next_stop.ordered(data["records"])], expected)
            label = next_stop.start_date_label(data["records"][1], "en")
            self.assertEqual(label, "Expected February 2027" if milestone == "offer" else "February 2027")

    def test_same_start_month_has_stable_order(self):
        data = fixture()
        for record in data["records"]:
            record["start_date"] = "2026-05"
        self.assertEqual([record["id"] for record in next_stop.ordered(list(reversed(data["records"])))],
                         ["NS-0001", "NS-0002", "NS-0003"])

    def test_user_content_is_escaped_and_links_are_constrained(self):
        output = self.render()
        self.assertIn('Example &lt;Company&gt;', output)
        self.assertIn('&lt;script&gt;', output)
        self.assertNotIn('<script>', output)
        self.assertIn('https://github.com/example-reader', output)
        data = fixture()
        data["records"][0]["github"] = 'javascript:alert(1)'
        with self.assertRaises(ValueError):
            next_stop.validate(data)

    def test_username_is_explicit_in_sky_and_timeline_even_with_display_name(self):
        for language in ("zh", "en"):
            output = self.render(language)
            self.assertEqual(output.count('class="stop-profile" href="https://github.com/example-reader">@example-reader ↗</a>'), 2)
            self.assertIn('Test &quot;reader&quot;', output)
            self.assertIn('aria-label="@example-reader · Example &lt;Company&gt;', output)

    def test_nickname_only_does_not_create_a_profile_link(self):
        data = fixture()
        del data["records"][1]["github"]
        output = self.render(data=data)
        self.assertNotIn('class="stop-profile"', output)
        self.assertIn('Test &quot;reader&quot;', output)

    def test_bilingual_titles_and_completed_role(self):
        data = fixture()
        data["records"][0].update(title={"zh": "机器学习工程师实习生", "en": "ML Engineer Intern <test>"}, milestone="completed")
        chinese, english = self.render(data=data), self.render("en", data)
        self.assertIn("机器学习工程师实习生", chinese)
        self.assertIn("已结束", chinese)
        self.assertIn("ML Engineer Intern &lt;test&gt;", english)
        self.assertIn("Completed", english)
        self.assertIn('data-role="ml"', english)

    def test_direct_owner_consent_does_not_require_an_invented_issue(self):
        owner = tomllib.loads((next_stop.ROOT / "site/nav.toml").read_text())["site"]["owner_login"]
        data = fixture()
        record = data["records"][0]
        del record["source_issue"]
        record.update(owner_submission=True, github=owner)
        next_stop.validate(data)
        record["source_issue"] = 1
        with self.assertRaises(ValueError):
            next_stop.validate(data)
        del record["source_issue"]
        record["github"] = "another-reader"
        with self.assertRaises(ValueError):
            next_stop.validate(data)
        record["github"] = owner
        record["owner_submission"] = "true"
        with self.assertRaises(ValueError):
            next_stop.validate(data)
        del record["owner_submission"]
        with self.assertRaises(ValueError):
            next_stop.validate(data)

    def test_rejects_missing_consent_invalid_dates_and_unexpected_fields(self):
        invalid = [
            ("consent", False), ("consent", "true"), ("source_issue", True),
            ("source_issue", -1), ("start_date", "2027-13"), ("start_date", "2027-01-02"),
            ("start_date", "2027-00"), ("start_date", "May 2026"), ("start_date", 2027),
            ("date", "2027-01"), ("role", "unknown"), ("company", "missing"),
            ("note", {"zh": "只有中文"}), ("email", "private@example.org"),
            ("title", {"en": "Missing Chinese"}), ("title", {"zh": "职位", "en": "x" * 101}),
        ]
        for field, value in invalid:
            with self.subTest(field=field, value=value):
                data = fixture()
                data["records"][0][field] = value
                with self.assertRaises(ValueError):
                    next_stop.validate(data)

    def test_rejects_duplicate_ids_and_company_aliases(self):
        data = fixture()
        data["records"].append(copy.deepcopy(data["records"][0]))
        with self.assertRaises(ValueError):
            next_stop.validate(data)
        data = fixture()
        data["companies"]["alias"] = " EXAMPLE   <COMPANY> "
        with self.assertRaises(ValueError):
            next_stop.validate(data)

    def test_plain_html_has_readable_records_without_javascript(self):
        output = self.render()
        timeline = re.search(r'<div data-stop-timeline>(.*?)</div>\s*<ol', output, re.S).group(1)
        self.assertNotIn(' hidden', timeline)
        self.assertIn('<h2>2027</h2>', timeline)
        self.assertIn('data-stop-controls hidden', output)

    def test_reuses_contributor_scene_instead_of_marketing_hero(self):
        for language in ("zh", "en"):
            output = self.render(language)
            for component in ("contributor-universe", "crew-sky-1bit.png", "orbit-nav", "orbit-title", "crew-field", "crew-motion", "orbit-footer"):
                self.assertIn(component, output)
            self.assertEqual(output.count('class="crew-drifter"'), 3)
            self.assertIn('href="#NS-0001"', output)
            self.assertNotIn('stop-astronaut', output)
            self.assertNotIn('stop-hero', output)

    def test_empty_sky_has_invitation_not_a_fictional_person(self):
        output = self.render(data={"version": 1, "companies": {}, "records": []})
        self.assertIn('data-crew-id="invitation"', output)
        self.assertIn('data-initials="+"', output)
        self.assertNotIn('data-crew-id="NS-', output)

    def test_sky_limits_tokens_but_keeps_all_timeline_records(self):
        data = fixture()
        seed = data['records'][0]
        data['records'] = [dict(seed, id=f'NS-{index:04d}') for index in range(1, 31)]
        output = self.render(data=data)
        self.assertEqual(output.count('class="crew-drifter"'), 24)
        self.assertEqual(output.count('data-stop-record'), 30)
        self.assertIn('星空展示最近 24 条', output)

    @unittest.skipUnless(shutil.which("node"), "Node is required for interaction tests")
    def test_browser_interaction_logic(self):
        result = subprocess.run([shutil.which("node"), str(Path(__file__).with_name("next-stop-behavior.cjs"))],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
