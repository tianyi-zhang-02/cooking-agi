from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build


class InferenceGlossaryTests(unittest.TestCase):
    def setUp(self):
        self.terms = build.load_glossary()

    def test_common_systems_terms_have_matching_english(self):
        translations = {chinese: english for chinese, english, _ in self.terms}
        expected = {
            "前缀复用": "prefix reuse",
            "前缀缓存": "prefix caching",
            "连续批处理": "continuous batching",
            "分块预填充": "chunked prefill",
            "推测解码": "speculative decoding",
            "缓存命中": "cache hit",
            "缓存卸载": "cache offloading",
            "张量并行": "tensor parallelism, TP",
            "流水线并行": "pipeline parallelism, PP",
        }
        for chinese, english in expected.items():
            with self.subTest(term=chinese):
                self.assertEqual(translations[chinese], english)
                source = f'<p>{chinese}和{chinese}。</p><p>再看{chinese}。</p>'
                output, used = build.annotate(source, self.terms)
                self.assertEqual(output.count(f'<span class="term-en">（{english}）</span>'), 3)
                self.assertEqual([term["zh"] for term in used], [chinese])

    def test_longer_cache_and_prefill_terms_are_not_split(self):
        source = '<p>缓存命中率、分块预填充、吞吐量、吞吐率。</p>'
        output, used = build.annotate(source, self.terms)
        self.assertEqual([term["zh"] for term in used], ["缓存命中率", "分块预填充", "吞吐量", "吞吐率"])
        self.assertEqual(output.count('class="term-en"'), 4)

    def test_actual_pages_repeat_english_without_touching_source_structure(self):
        nav = build.load_nav()
        previous_sources = dict(build.BY_SRC)
        try:
            pages, _ = build.discover(nav)
            known = {page.url for page in pages}
            selected = [page for page in pages if page.url in {
                '06-systems/inference-frameworks.html', '06-systems/inference-frameworks.en.html',
                '06-systems/mlsys-llmsys-map.html', '06-systems/mlsys-llmsys-map.en.html',
            }]
            self.assertEqual(len(selected), 4)
            for page in selected:
                with self.subTest(page=page.url):
                    build.build_page(page, [], nav['site']['repo'], known)
                    previous_body = page.body
                    build.build_page(page, self.terms, nav['site']['repo'], known)
                    for pattern in (r'id="[^"]+"', r'href="[^"]+"', r'<pre\b.*?</pre>'):
                        self.assertEqual(re.findall(pattern, previous_body, re.S),
                                         re.findall(pattern, page.body, re.S))
                    if page.lang == 'en':
                        self.assertEqual(page.body, previous_body)
                        self.assertNotIn('class="term-en"', page.body)
                    elif page.url.endswith('inference-frameworks.html'):
                        self.assertGreater(page.body.count('（prefix reuse）'), 1)
                        self.assertIn('（prefix caching）', page.body)
                        self.assertIn('（cache offloading）', page.body)
        finally:
            build.BY_SRC.clear()
            build.BY_SRC.update(previous_sources)


if __name__ == '__main__':
    unittest.main()
