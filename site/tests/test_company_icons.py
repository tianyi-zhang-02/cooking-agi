import re
import sys
import unittest
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import company_icons
import next_stop


class CompanyIconTests(unittest.TestCase):
    def test_presets_cover_multiple_fields_without_creating_outcomes(self):
        entries, aliases = company_icons.load()
        self.assertGreaterEqual(len(entries), 70)
        self.assertEqual({entry["group"] for entry in entries}, set(company_icons.GROUPS))
        for name in ("NVIDIA", "英伟达", "HRT", "Hudson River Trading", "Service Now", "ServiceNow"):
            entry = aliases[company_icons.normalize(name)]
            self.assertIn("asset", entry)
        self.assertEqual(aliases["servicenow"]["id"], "servicenow")
        page = SimpleNamespace(lang="en", depth=1, rel=lambda path: "../" + path)
        output = next_stop.render(page, "example/notes", {"version": 1, "companies": {}, "records": []})
        self.assertEqual(output.count("data-company-preset "), len(entries))
        self.assertNotIn("data-stop-record", output)
        self.assertNotIn("data-stop-company=", output)

    def test_all_svg_assets_are_local_static_and_attributed(self):
        entries = company_icons.load()[0]
        allowed = {"svg", "title", "desc", "path", "g", "rect", "circle", "ellipse", "polygon", "polyline", "line", "defs", "linearGradient", "radialGradient", "stop", "clipPath", "mask", "style"}
        for entry in entries:
            if not entry.get("asset"):
                continue
            with self.subTest(company=entry["id"]):
                self.assertTrue(entry["source"].startswith("https://"))
                asset = company_icons.ROOT / "site/static/company-icons" / entry["asset"]
                contents = asset.read_text()
                self.assertLess(asset.stat().st_size, 100_000)
                self.assertNotRegex(contents, r"(?i)<!DOCTYPE|<!ENTITY|javascript:|@import|url\(\s*['\"]?(?!#)")
                for element in ElementTree.fromstring(contents).iter():
                    self.assertIn(element.tag.split("}")[-1], allowed)
                    for attribute in element.attrib:
                        self.assertFalse(attribute.lower().startswith("on"))
                        self.assertNotIn(attribute.split("}")[-1], {"href", "src"})

    def test_unknown_company_has_safe_full_name_fallback(self):
        result = company_icons.mark("unknown", "<Company> & friends", "../")
        self.assertIn("company-mark-text", result)
        self.assertNotIn("<img", result)
        self.assertNotIn("<Company>", result)
        self.assertIn("&lt;Company&gt; &amp; friends", result)

    def test_missing_logos_use_full_names_without_initials_or_duplication(self):
        page = SimpleNamespace(lang="en", depth=1)
        catalog = company_icons.catalog(page, "https://example.com/request")
        for entry in company_icons.load()[0]:
            if entry.get("asset"):
                continue
            with self.subTest(company=entry["id"]):
                self.assertNotIn("monogram", entry)
                mark = company_icons.mark(entry["id"], entry["name"], "../")
                self.assertIn(f'>{entry["name"]}</span>', mark)
                identity = company_icons.identity(entry["id"], entry["name"], "../")
                self.assertEqual(identity.count(entry["name"]), 1)
                self.assertIn(f'<span>{entry["name"]}<small>Text mark</small></span>', catalog)
        self.assertNotIn('>O</span>', catalog)
        self.assertNotIn('>C</span>', catalog)
        self.assertNotIn('>B</span>', catalog)

    def test_marks_are_reused_in_all_record_views(self):
        page = SimpleNamespace(lang="en", depth=1, rel=lambda path: "../" + path)
        data = {"version": 1, "companies": {"nvidia": "NVIDIA"}, "records": [{
            "id": "NS-0001", "start_date": "2027", "company": "nvidia", "role": "ml",
            "employment": "full-time", "milestone": "offer", "consent": True, "source_issue": 1,
        }]}
        output = next_stop.render(page, "example/notes", data)
        self.assertEqual(output.count('src="../static/company-icons/nvidia.svg"'), 4)
        self.assertIn("stop-company-disc", output)
        self.assertIn('data-stop-count>1</b>', output)
        data["companies"]["alias"] = "英伟达"
        with self.assertRaisesRegex(ValueError, "canonical company ID"):
            next_stop.validate(data)

    def test_catalog_has_request_link_and_bilingual_controls(self):
        for language in ("zh", "en"):
            page = SimpleNamespace(lang=language, depth=1)
            output = company_icons.catalog(page, "https://github.com/example/notes/issues/new?template=company-icon.yml")
            self.assertIn("template=company-icon.yml", output)
            self.assertIn("data-company-search", output)
            self.assertIn("data-company-no-match", output)
            self.assertIn("文字标识" if language == "zh" else "Text mark", output)

    def test_request_is_assigned_to_owner_without_automatic_publication(self):
        template = (company_icons.ROOT / ".github/ISSUE_TEMPLATE/company-icon.yml").read_text()
        self.assertRegex(template, r"assignees:\s*\n\s*- tianyi-zhang-02")
        for field in ("company", "website", "resources"):
            self.assertRegex(template, rf"id: {field}\n[\s\S]*?required: true")
