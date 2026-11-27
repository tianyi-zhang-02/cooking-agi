from collections import Counter
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import sys
import tomllib
import unittest
from unittest.mock import Mock
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build
import curriculum


class Links(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.links = []
        self.feed(text)

    def handle_starttag(self, tag, attributes):
        if tag == 'a':
            self.links.append(dict(attributes).get('href', ''))


class CurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_nav = build.NAV.copy()
        cls.old_catalog = build.BY_SRC.copy()
        cls.navigation = build.load_nav()
        build.BY_SRC.clear()
        cls.pages, cls.sections = build.discover(cls.navigation)
        cls.catalog = build.BY_SRC.copy()

    @classmethod
    def tearDownClass(cls):
        build.NAV.clear()
        build.NAV.update(cls.old_nav)
        build.BY_SRC.clear()
        build.BY_SRC.update(cls.old_catalog)

    def test_no_preexisting_published_url_is_removed_or_duplicated(self):
        expected = json.loads((build.SITE / 'catalog-baseline.json').read_text())
        urls = [page.url for page in self.pages]
        redirects = self.navigation.get('redirects', {})
        self.assertFalse(set(expected) - set(urls) - set(redirects))
        self.assertFalse({target.partition('#')[0] for target in redirects.values()} - set(urls))
        self.assertFalse(set(redirects) & set(urls))
        self.assertEqual(len(urls), len(set(urls)))

    def test_included_notes_have_exactly_one_chapter(self):
        sources = [str(pair['zh'].src.relative_to(build.ROOT))
                   for section in self.sections for pair in section['pages']]
        self.assertTrue(all(count == 1 for count in Counter(sources).values()))

    def test_every_study_note_is_reachable_from_both_atlases(self):
        study_groups = {group['id'] for group in self.navigation['group']
                        if group.get('category') == 'learn' and group['id'] != 'study'}
        for language in ('zh', 'en'):
            page = self.catalog['learn/README.md'][language]
            markup = curriculum.study_atlas(page, self.navigation, self.sections, self.catalog)
            rendered = build.rewrite_links(markup, page, 'tianyi-zhang-02/cooking-agi', {note.url for note in self.pages})
            links = set(Links(rendered).links)
            resolved = {os.path.normpath(str(page.out_rel.parent / link)) for link in links}
            for section in self.sections:
                if section['group'] in study_groups:
                    for pair in section['pages']:
                        self.assertIn(pair[language].url, resolved)
            self.assertNotIn('javascript:', markup)
            self.assertNotIn('.md"', rendered)
            self.assertTrue(all(not link.startswith('https:') for link in links))

    def test_reading_order_puts_prerequisites_before_dependents(self):
        foundation_sections = [section['dir'] for section in self.sections if section['group'] == 'foundations']
        self.assertLess(foundation_sections.index('learn/input-and-sequences'), foundation_sections.index('00-foundations/core'))
        post_sections = [section['dir'] for section in self.sections if section['group'] == 'posttrain']
        self.assertLess(post_sections.index('05-post-training/rlhf'), post_sections.index('05-post-training/experiments-and-tradeoffs'))
        self.assertEqual(self.catalog['05-post-training/sft-and-its-ceiling.md']['zh'].previous.src.name, 'README.md')
        self.assertEqual(self.catalog['05-post-training/rlhf/ppo-clipping.md']['zh'].next.src.name, 'after-rlhf.md')
        self.assertEqual(self.catalog['00-foundations/deep-dives/muon.md']['zh'].previous.src.name, 'optimizers.md')
        self.assertEqual(self.catalog['06-systems/llm-serving.md']['zh'].previous.src.name, 'kv-cache-and-inference.md')
        self.assertEqual(self.catalog['06-systems/inference-frameworks.md']['zh'].previous.src.name, 'llm-serving.md')
        self.assertEqual(self.catalog['00-foundations/deep-dives/dflash.md']['zh'].previous.src.name, 'inference-frameworks.md')
        self.assertEqual(self.catalog['03-multimodal-learning/siglip.md']['zh'].previous.src.name, 'clip.md')
        self.assertEqual(self.catalog['03-multimodal-learning/vision-to-language.md']['zh'].previous.src.name, 'siglip.md')

    def test_atlas_follows_the_tracks_then_go_deeper_and_still_moving(self):
        areas = ('reads', 'learns', 'taught', 'check', 'runs', 'world', 'beyond', 'deeper', 'moving')
        for language in ('zh', 'en'):
            page = self.catalog['learn/README.md'][language]
            markup = curriculum.study_atlas(page, self.navigation, self.sections, self.catalog)
            self.assertEqual(markup.count('class="atlas-area"'), len(areas))
            anchors = [f'id="area-{area}"' for area in areas]
            self.assertEqual([markup.count(anchor) for anchor in anchors], [1] * len(areas))
            self.assertEqual(sorted(markup.index(anchor) for anchor in anchors),
                             [markup.index(anchor) for anchor in anchors])
            for group in self.navigation['group']:
                if group.get('category') == 'learn' and group['id'] != 'study':
                    self.assertEqual(markup.count(f'id="subject-{group["id"]}"'), 1)
            self.assertNotIn(' open>', markup)
            self.assertNotIn('<button', markup)
            for moved in ('coding', 'algorithms', 'ml-exercises', 'system-design', 'interview-basics'):
                self.assertNotIn(f'id="subject-{moved}"', markup)

    def test_every_stop_is_a_study_note_listed_once_in_both_languages(self):
        study = {pair['zh'].src.relative_to(build.ROOT).as_posix()
                 for section in self.sections for pair in section['pages']}
        stops = [stop['note'] for zone in self.navigation['zone'] for stop in zone.get('stops', [])]
        self.assertEqual(len(stops), len(set(stops)))
        tracks = [zone for zone in self.navigation['zone'] if zone.get('stops')]
        self.assertEqual(len(tracks), 7)
        for zone in tracks:
            for stop in zone['stops']:
                self.assertIn(stop['note'], study)
                self.assertTrue(stop['zh'] and stop['en'])
                for need in stop.get('needs', []):
                    self.assertIn(need, self.catalog)
                for language in ('zh', 'en'):
                    self.assertIn(language, self.catalog[stop['note']])

    def test_track_map_links_every_stop_in_reading_order(self):
        for language in ('zh', 'en'):
            page = self.catalog['learn/README.md'][language]
            markup = curriculum.track_map(page, self.navigation, self.catalog)
            rendered = build.rewrite_links(markup, page, 'tianyi-zhang-02/cooking-agi', {note.url for note in self.pages})
            self.assertNotIn('.md"', rendered)
            for zone in self.navigation['zone']:
                if not zone.get('stops'):
                    self.assertNotIn(f'id="track-{zone["id"]}"', markup)
                    continue
                self.assertEqual(markup.count(f'id="track-{zone["id"]}"'), 1)
                positions = [markup.index('>' + __import__('html').escape(stop[language]) + '</a>',
                                          markup.index(f'id="track-{zone["id"]}"')) for stop in zone['stops']]
                self.assertEqual(positions, sorted(positions))

    def test_track_context_names_the_position_the_reading_before_and_the_next_stop(self):
        kv = self.catalog['00-foundations/deep-dives/kv-cache-and-inference.md']
        english = curriculum.track_context(kv['en'], self.navigation, self.catalog)
        self.assertIn('How a model runs</a> · stop 2 of 7', english)
        self.assertIn('>Decoding</a>', english)
        self.assertIn('>Attention</a>', english)
        self.assertIn('Next stop <a href="../../06-systems/llm-serving.en.html">Serving</a>', english)
        chinese = curriculum.track_context(kv['zh'], self.navigation, self.catalog)
        self.assertIn('第 2 站，共 7 站', chinese)
        first = curriculum.track_context(self.catalog['00-foundations/from-linear-to-neural.md']['en'],
                                         self.navigation, self.catalog)
        self.assertIn('stop 1 of 9', first)
        self.assertNotIn('Needs', first)
        self.assertEqual(curriculum.track_context(self.catalog['05-post-training/rloo.md']['en'],
                                                  self.navigation, self.catalog), '')

    def test_subject_overviews_keep_their_chapters_open(self):
        for source in ('learn/pretraining/README.md', 'learn/inference/README.md'):
            for language in ('zh', 'en'):
                markup = curriculum.study_atlas(self.catalog[source][language], self.navigation,
                                                self.sections, self.catalog)
                self.assertNotIn('class="atlas-area"', markup)
                self.assertIn('<h3>', markup)
                self.assertEqual(markup.count('<details'), markup.count(' open>'))

    def test_worked_diagram_keeps_labels_clean_and_prose_annotated(self):
        for diagram_class in ('worked-update', 'engine-map'):
            with self.subTest(diagram_class=diagram_class):
                diagram = f'<figure class="{diagram_class}"><ol><li>梯度</li></ol></figure>'
                rendered, used = build.annotate(diagram + '<p>梯度</p>', [('梯度', 'gradient', '变化方向')])
                self.assertTrue(rendered.startswith(diagram))
                self.assertEqual(rendered.count('class="term"'), 1)
                self.assertEqual(len(used), 1)

    def test_moved_notes_return_to_their_new_overview(self):
        pairs = [('00-foundations/deep-dives/precision-and-memory.md', 'learn/pretraining/README.md'),
                 ('00-foundations/deep-dives/attention-kernels.md', 'learn/inference/README.md')]
        for source, home in pairs:
            for language in ('zh', 'en'):
                self.assertEqual(build.chapter_home(self.catalog[source][language]), self.catalog[home][language])

    def test_known_reference_topics_remain_visible_without_false_completion(self):
        data = tomllib.loads((build.SITE / 'reference-coverage.toml').read_text())
        self.assertEqual(data['comparison'], 'browser-chapter-review-in-progress')
        topics = [topic for chapter in data['chapter'] for topic in chapter['topics']]
        names = ' '.join(topic['en'] for topic in topics)
        for name in ('GSPO', 'ASPO', 'SAO', 'DCA', 'DSA', 'Engram', 'AttnRes', 'PagedAttention', 'QLoRA'):
            self.assertIn(name, names)
        for topic in topics:
            self.assertIn(topic['state'], {'article', 'overview', 'pending'})
            if topic['state'] != 'pending':
                self.assertTrue(topic['notes'])
            for source in topic['notes']:
                self.assertIn(source, self.catalog)
                self.assertIsNotNone(self.catalog[source]['en'])

    def test_coverage_is_bilingual_and_has_real_links(self):
        data = tomllib.loads((build.SITE / 'reference-coverage.toml').read_text())
        present_states = {topic['state'] for chapter in data['chapter'] for topic in chapter['topics']}
        for language in ('zh', 'en'):
            page = self.catalog['learn/coverage.md'][language]
            markup = curriculum.reference_coverage(page, self.catalog, build.SITE / 'reference-coverage.toml')
            labels = {'article': ('有正文', 'Article available'), 'overview': ('汇总入口', 'Topic index'), 'pending': ('待补', 'To add')}
            for state in present_states:
                self.assertIn('<td>' + labels[state][language == 'en'] + '</td>', markup)
            if language == 'en':
                self.assertNotIn('待补', markup)
            rendered = build.rewrite_links(markup, page, 'tianyi-zhang-02/cooking-agi', {note.url for note in self.pages})
            for destination in Links(rendered).links:
                if not destination.startswith('https:'):
                    self.assertTrue(destination.endswith('.en.html' if language == 'en' else '.html'))
                else:
                    self.assertTrue(destination.startswith('https://arxiv.org/abs/'))

    def test_pending_coverage_has_no_fake_article_link(self):
        source = Mock()
        source.read_text.return_value = '''[[chapter]]
zh = "待写示例"
en = "Unwritten example"
topics = [{zh="一个主题", en="A topic", state="pending", notes=[]}]
'''
        for language in ('zh', 'en'):
            markup = curriculum.reference_coverage(self.catalog['learn/coverage.md'][language], self.catalog, source)
            self.assertIn('To add' if language == 'en' else '待补', markup)
            self.assertEqual(Links(markup).links, [])

    def test_built_widget_links_resolve_to_published_pages(self):
        published = {page.url for page in self.pages}
        for source in ('learn/README.md', 'learn/pretraining/README.md',
                       'learn/inference/README.md', 'learn/coverage.md',
                       '00-foundations/deep-dives/README.md'):
            for language in ('zh', 'en'):
                page = self.catalog[source][language]
                build.build_page(page, [], 'tianyi-zhang-02/cooking-agi', published)
                for destination in Links(page.body).links:
                    if destination.startswith(('https:', '#')):
                        self.assertNotIn('github.com/tianyi-zhang-02/cooking-agi/blob/main/', destination)
                        continue
                    resolved = os.path.normpath(str(page.out_rel.parent / urlsplit(destination).path))
                    self.assertIn(resolved, published, f'{source}: {destination}')

    def test_appendix_topics_keep_reading_separate_from_coverage(self):
        data = tomllib.loads((build.SITE / 'appendix-coverage.toml').read_text())
        self.assertEqual(data['scope'], 'screenshots-and-browser-directory-not-complete-reference')
        topics = [topic for chapter in data['chapter'] for topic in chapter['topics']]
        self.assertEqual(len(topics), 67)
        self.assertEqual(len({topic['id'] for topic in topics}), 67)
        questions = [topic for topic in topics if topic['id'] not in {'algorithms', 'pytorch'}]
        self.assertEqual(sum(topic['read'] == 'body' for topic in questions), 65)
        for identifier in ('checkpoint', 'training-memory', 'zero-stages',
                           'entropy-ce-kl', 'kl-estimators', 'grpo-vs-ppo'):
            topic = next(topic for topic in questions if topic['id'] == identifier)
            self.assertEqual((topic['state'], topic['read']), ('article', 'body'))
        for topic in topics:
            self.assertIn(topic['state'], {'article', 'overview', 'related', 'missing'})
            self.assertTrue(topic['gap_zh'] and topic['gap_en'])
            for target in topic['notes']:
                self.assertIn(target, self.catalog)
        kl = next(topic for topic in topics if topic['id'] == 'kl-estimators')
        self.assertEqual((kl['state'], kl['read']), ('article', 'body'))
        k3 = next(topic for topic in topics if topic['id'] == 'k3-rope')
        self.assertEqual((k3['state'], k3['read']), ('article', 'body'))
        self.assertIn('00-foundations/deep-dives/nope-and-order.md', k3['notes'])
        for identifier in ('dflash-inference', 'mtp-dflash'):
            topic = next(topic for topic in topics if topic['id'] == identifier)
            self.assertEqual((topic['state'], topic['read']), ('article', 'body'))
        for identifier in ('bert', 'bert-lstm', 'encoder-decoder-objectives', 'attention-masks',
                           'moe-routing-code', 'sequence-moe-balance', 'grpo-initial-loss',
                           'clipped-token-gradient', 'grpo-on-policy', 'rejection-sampling',
                           'reward-hacking-entropy', 'entropy-collapse', 'cross-entropy-code',
                           'sft-mask', 'sft-to-rl', 'tp-dp', 'pp-without-nvlink',
                           'collectives', 'decoder-tp', 'tp-column-row',
                           'moe-router', 'moe-changes', 'mla-rope',
                           'cross-tokenizer', 'on-policy-distillation', 'rollout',
                           'qwen-bge', 'bm25-tfidf', 'infonce-cross-entropy',
                           'clip-variants', 'rlhf-algorithms', 'importance-token-loss',
                           'muon', 'gpu-data-flow'):
            topic = next(topic for topic in topics if topic['id'] == identifier)
            self.assertEqual((topic['state'], topic['read']), ('article', 'body'))
        cache = next(topic for topic in topics if topic['id'] == 'kv-placement')
        self.assertEqual((cache['state'], cache['read']), ('article', 'body'))
        for identifier in ('softmax-implementations', 'moe-vs-dense', 'qwen-versions',
                           'prime-sieve', 'self-attention-code', 'multi-head-code',
                           'rope-code', 'bf16-fp16', 'batchnorm-layernorm'):
            topic = next(topic for topic in questions if topic['id'] == identifier)
            self.assertEqual((topic['state'], topic['read']), ('article', 'body'))

    def test_appendix_widget_languages_and_links(self):
        data = tomllib.loads((build.SITE / 'appendix-coverage.toml').read_text())
        counts = Counter(topic['state'] for chapter in data['chapter'] for topic in chapter['topics'])
        total = sum(counts.values())
        for language in ('zh', 'en'):
            page = self.catalog['learn/coverage.md'][language]
            markup = curriculum.appendix_coverage(page, self.catalog, build.SITE / 'appendix-coverage.toml')
            self.assertEqual(markup.count('<li id="coverage-'), 67)
            self.assertEqual(markup.count('<details'), 4)
            self.assertIn('参考正文已读' if language == 'zh' else 'Reference prose read', markup)
            expected_counts = (f"{total} 个入口：{counts['article']} 项已有专门讲解，{counts['overview']} 项为汇总入口，{counts['related']} 项需补充或对照，{counts['missing']} 项缺少专题讲解"
                               if language == 'zh' else
                               f"{total} entries: {counts['article']} with dedicated explanations, {counts['overview']} topic indexes, {counts['related']} needing expansion or comparison, and {counts['missing']} lacking dedicated coverage")
            self.assertIn(expected_counts, markup)
            if language == 'en':
                self.assertNotIn('待对照', markup)
            rendered = build.rewrite_links(markup, page, 'tianyi-zhang-02/cooking-agi', {note.url for note in self.pages})
            for link in Links(rendered).links:
                target = os.path.normpath(str(page.out_rel.parent / link))
                self.assertIn(target, {note.url for note in self.pages})
                self.assertEqual('.en.html' in target, language == 'en')

    def test_navigation_guide_uses_current_coverage_states(self):
        guide = (Path(__file__).resolve().parents[1] / 'NAVIGATION.md').read_text()
        self.assertIn('它区分有正文、汇总入口和待补', guide)
        self.assertNotIn('它区分有正文、部分讲解和待补', guide)

    def test_public_coverage_summary_matches_inventory(self):
        data = tomllib.loads((build.SITE / 'appendix-coverage.toml').read_text())
        topics = [topic for chapter in data['chapter'] for topic in chapter['topics']]
        counts = Counter(topic['state'] for topic in topics)
        root = build.SITE.parent
        chinese = (root / 'learn/coverage.md').read_text()
        english = (root / 'learn/coverage.en.md').read_text()
        self.assertIn(f"当前 **{counts['article']} 项有专门讲解，{counts['overview']} 项为汇总入口**", chinese)
        self.assertIn(f"**{counts['article']} dedicated explanations and {counts['overview']} topic indexes**", english)
        self.assertEqual({topic['id'] for topic in topics if topic['state'] == 'overview'}, {'algorithms', 'pytorch'})
        chapter = self.catalog['06-systems/tensor-parallel.md']
        for language in ('zh', 'en'):
                self.assertEqual(chapter[language].section['dir'], 'learn/training-resources')

    def test_appendix_reading_count_tracks_prose_not_coverage(self):
        original = (build.SITE / 'appendix-coverage.toml').read_text()
        for unread in (False, True):
            source = Mock()
            source.read_text.return_value = (original.replace('read="body"', 'read="title"', 1)
                                             if unread else original)
            data = tomllib.loads(source.read_text.return_value)
            questions = [topic for chapter in data['chapter'] for topic in chapter['topics']
                         if topic['state'] != 'overview']
            read_count = sum(topic['read'] == 'body' for topic in questions)
            for language in ('zh', 'en'):
                markup = curriculum.appendix_coverage(self.catalog['learn/coverage.md'][language], self.catalog, source)
                expected = (f'{len(questions)} 个专题的参考正文已读 {read_count} 篇' if language == 'zh'
                            else f'Reference prose read for {read_count} of {len(questions)} topics')
                self.assertIn(expected, markup)
                self.assertIn('正文已读不代表' if language == 'zh' else 'This does not certify', markup)
                self.assertEqual(markup.count('id="coverage-'), 67)

    def test_appendix_rejects_duplicate_or_invalid_states(self):
        for second in ('id="one",state="article",read="title"', 'id="two",state="complete",read="title"'):
            source = Mock()
            source.read_text.return_value = '''[[chapter]]
zh="测试"
en="Test"
topics=[
{id="one",zh="甲",en="A",state="missing",read="title",notes=[],gap_zh="待补",gap_en="Missing"},
{''' + second + ''',zh="乙",en="B",notes=[],gap_zh="待补",gap_en="Missing"}]
'''
            with self.assertRaises(ValueError):
                curriculum.appendix_coverage(self.catalog['learn/coverage.md']['zh'], self.catalog, source)


if __name__ == '__main__':
    unittest.main()
