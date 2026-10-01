import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build


class ContributorTests(unittest.TestCase):
    def collect(self, log, api=None, credits=None):
        response = io.StringIO(json.dumps(api or []))
        with patch.object(build, "git", return_value=log), \
                patch.object(build, "crew_countries", return_value={}), \
                patch.object(build, "crew_ai_collaborators", return_value=credits or {}), \
                patch.object(build.urllib.request, "urlopen", return_value=response):
            return build.contributors("example/notes")

    def test_primary_author_and_model_credits_share_one_identity(self):
        people = self.collect(
            "Claude\tnoreply@anthropic.com\t2026-09-30T12:00:00+00:00\t"
            "Claude Sonnet 5 <noreply@anthropic.com>\n"
            "Writer\t123+writer@users.noreply.github.com\t2026-09-29T12:00:00+00:00\t"
            "Claude Opus 5.5 <noreply@anthropic.com>\n"
            "Claude Code\tnoreply@anthropic.com\t2026-09-01T12:00:00+00:00"
        )
        self.assertEqual({person["name"] for person in people}, {"Writer", "Claude Code"})
        assistant = next(person for person in people if person["kind"] == "ai")
        self.assertEqual(assistant["commits"], 3)
        self.assertEqual(assistant["first"], "2026-09-01")
        self.assertEqual(assistant["last"], "2026-09-30")
        self.assertEqual(assistant["models"],
                         ["Claude", "Claude Code", "Claude Opus 5.5", "Claude Sonnet 5"])
        self.assertEqual(len({person["crew_id"] for person in people}), 2)

    def test_multiple_aliases_on_one_commit_count_once(self):
        people = self.collect(
            "Writer\twriter@example.org\t2026-09-30T12:00:00+00:00\t"
            "Claude Opus 5 <noreply@anthropic.com>\x1f"
            "Claude Sonnet 5 <noreply@anthropic.com>\x1f"
            "Writer <writer@example.org>\x1fMalformed credit"
        )
        self.assertEqual({person["name"]: person["commits"] for person in people},
                         {"Writer": 1, "Claude Code": 1})

    def test_human_claude_is_not_merged_into_ai(self):
        people = self.collect(
            "Claude\tclaude@example.org\t2026-09-30T12:00:00+00:00\t"
            "Claude Opus 5 <noreply@anthropic.com>"
        )
        self.assertEqual({person["name"]: person["kind"] for person in people},
                         {"Claude": "human", "Claude Code": "ai"})

    def test_api_metadata_preserves_human_links_and_combined_ai_count(self):
        people = self.collect(
            "Claude Code\t1+claude-bot@users.noreply.github.com\t2026-09-30T12:00:00+00:00\t\n"
            "Writer\t2+writer@users.noreply.github.com\t2026-09-29T12:00:00+00:00\t"
            "Claude Opus 5 <noreply@anthropic.com>",
            [{"login": "claude-bot", "avatar_url": "https://example.org/bot.png", "contributions": 1},
             {"login": "writer", "avatar_url": "https://example.org/writer.png", "contributions": 1}],
        )
        assistant = next(person for person in people if person["kind"] == "ai")
        writer = next(person for person in people if person["kind"] == "human")
        self.assertEqual(assistant["commits"], 2)
        self.assertEqual(writer["url"], "https://github.com/writer")
        self.assertEqual(writer["avatar"], "https://example.org/writer.png")

    def test_offline_build_keeps_ai_credit(self):
        log = "Claude\tnoreply@anthropic.com\t2026-09-30T12:00:00+00:00"
        with patch.object(build, "git", return_value=log), \
                patch.object(build, "crew_countries", return_value={}), \
                patch.object(build, "crew_ai_collaborators", return_value={}), \
                patch.object(build.urllib.request, "urlopen", side_effect=URLError("offline")), \
                patch("builtins.print"):
            people = build.contributors("example/notes")
        self.assertEqual(len(people), 1)
        self.assertEqual(people[0]["name"], "Claude Code")
        self.assertEqual(people[0]["commits"], 1)

    def test_confirmed_ai_credit_does_not_invent_git_metadata(self):
        description = {"zh": "站点与测试", "en": "Site and tests"}
        people = self.collect("", credits={"Codex": description})
        self.assertEqual(len(people), 1)
        self.assertEqual(people[0]["name"], "Codex")
        self.assertEqual(people[0]["kind"], "ai")
        self.assertEqual(people[0]["description"], description)
        self.assertIsNone(people[0]["commits"])
        self.assertIsNone(people[0]["login"])
        self.assertEqual(people[0]["first"], "")
        self.assertEqual(people[0]["last"], "")
        self.assertEqual(people[0]["countries"], ())

    def test_later_codex_git_credit_merges_without_mixing_claude_models(self):
        people = self.collect(
            "OpenAI Codex\tbot@example.org\t2026-09-30T12:00:00+00:00\t"
            "Codex <bot@example.org>\x1fClaude Opus 5 <noreply@anthropic.com>",
            credits={"Codex": {"zh": "测试", "en": "Tests"}},
        )
        self.assertEqual(len(people), 2)
        codex = next(person for person in people if person["name"] == "Codex")
        claude = next(person for person in people if person["name"] == "Claude Code")
        self.assertEqual(codex["commits"], 1)
        self.assertEqual(codex["models"], ["Codex", "OpenAI Codex"])
        self.assertEqual(codex["description"]["en"], "Tests")
        self.assertEqual(claude["models"], ["Claude Opus 5"])

    def test_bilingual_manual_credit_renders_in_sky_credits_and_board(self):
        people = self.collect("", credits={"Codex": {"zh": "测试 <script>", "en": "Tests <script>"}})
        for language, explanation in (("zh", "未单独统计"), ("en", "Not separately tracked")):
            page = SimpleNamespace(lang=language, depth=0, rel=lambda path: path)
            with patch.object(build, "world_dots", return_value={}), \
                    patch.object(build.collaboration, "load_config", return_value={}):
                output = build.contributor_universe_html(people, page, {"repo": "example/notes"})
            self.assertEqual(output.count('data-crew-id="Codex"'), 1)
            self.assertIn('<span class="credit-name"><strong>Codex</strong>', output)
            self.assertIn('<th scope="row"><strong>Codex</strong>', output)
            self.assertIn(explanation, output)
            self.assertIn('&lt;script&gt;', output)
            self.assertNotIn('<script>', output)
            self.assertNotIn('<time datetime="">', output)
            self.assertNotIn('crew-drifter is-recent', output)
            row = re.search(r'<tr class="">.*?</tr>', output, re.S).group()
            self.assertNotIn('<b>0</b>', row)

    def test_registry_requires_bilingual_unique_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(build, "ROOT", root):
                self.assertEqual(build.crew_ai_collaborators(), {})
                (root / "crew.toml").write_text('[[ai]]\nname="Codex"\nzh="测试"\nen="Tests"\n')
                self.assertEqual(build.crew_ai_collaborators(), {"Codex": {"zh": "测试", "en": "Tests"}})
                (root / "crew.toml").write_text('[[ai]]\nname="Codex"\nzh="测试"\n')
                with self.assertRaises(ValueError):
                    build.crew_ai_collaborators()
                (root / "crew.toml").write_text(
                    '[[ai]]\nname="Codex"\nzh="测试"\nen="Tests"\n'
                    '[[ai]]\nname="OpenAI Codex"\nzh="测试"\nen="Tests"\n')
                with self.assertRaises(ValueError):
                    build.crew_ai_collaborators()


if __name__ == "__main__":
    unittest.main()
