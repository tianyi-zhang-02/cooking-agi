from html.parser import HTMLParser
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build
import leakcheck
import paritycheck


class InterviewList(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.collecting = False
        self.rows = []
        self.feed(text)

    def handle_starttag(self, tag, attributes):
        if tag == 'ul' and dict(attributes).get('class') == 'interview-roles':
            self.collecting = True
        if self.collecting and tag == 'li':
            self.rows.append('')

    def handle_endtag(self, tag):
        if tag == 'ul':
            self.collecting = False

    def handle_data(self, text):
        if self.collecting and self.rows:
            self.rows[-1] += text


class CareerJourneyTests(unittest.TestCase):
    def test_roles_are_up_front_identical_in_both_languages_and_not_offers(self):
        versions = [(build.ROOT / 'career' / name).read_text() for name in ['journey.md', 'journey.en.md']]
        rows = [[row.strip() for row in InterviewList(text).rows] for text in versions]
        self.assertEqual(rows[0], rows[1])
        self.assertEqual(len(rows[0]), 9)
        self.assertEqual(len(set(rows[0])), 9)
        self.assertEqual(rows[0], [
            'LinkedInMachine Learning Engineer',
            'MillenniumAI Engineer',
            'Tower Research VenturesMachine Learning Engineer',
            'HRTAlgorithm Developer',
            'SIGQuantitative Research',
            'DatabricksData Science',
            'Amazon BedrockAI Software Development Engineer',
            'AppleAI / ML Software Development Engineer',
            'GoogleSoftware Development Engineer',
        ])
        for name in ['LinkedIn', 'HRT', 'SIG', 'Millennium', 'Amazon Bedrock', 'Apple', 'Tower Research Ventures', 'Google', 'Databricks']:
            self.assertEqual(sum(name in row for row in rows[0]), 1)
        for text in versions:
            self.assertLess(text.index('{#interview-experience}'), text.index('## ', text.index('</ul>')))
        self.assertIn('不是 offer 列表', versions[0])
        self.assertIn('not a list of offers', versions[1])
        self.assertEqual(paritycheck.shape(versions[0]), paritycheck.shape(versions[1]))

    def test_public_company_allowance_is_exact_and_scoped(self):
        line = '<li><strong>LinkedIn</strong><span>Machine Learning Engineer</span></li>'
        for source in ['career/journey.md', 'career/journey.en.md']:
            self.assertNotIn('LinkedIn', leakcheck.redact_public_career_role(line, source))
            self.assertEqual(leakcheck.redact_public_career_role(line + ' private', source), line + ' private')
            self.assertEqual(leakcheck.redact_public_career_role(line.replace('Engineer', 'Internal'), source), line.replace('Engineer', 'Internal'))
        self.assertEqual(leakcheck.redact_public_career_role(line, 'practice/private.md'), line)

    def test_grid_has_no_extra_glossary_chips(self):
        source = '<ul class="interview-roles"><li>机器学习</li></ul><p>机器学习</p>'
        output, used = build.annotate(source, [('机器学习', 'machine learning', '从数据学习')])
        self.assertIn('<li>机器学习</li>', output)
        self.assertEqual(len(used), 1)
        self.assertEqual(output.count('class="term"'), 1)


if __name__ == '__main__':
    unittest.main()
