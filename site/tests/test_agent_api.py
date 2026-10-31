"""The agent data API (site/agent_api.py) and the interview skill that reads it."""
from pathlib import Path
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "site"))
import agent_api

SCRIPT = ROOT / "skills/cooking-agi-interview/scripts/interview.py"


class AgentApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)
        cls.counts = agent_api.build(cls.out)
        api = cls.out / agent_api.API
        cls.topics = json.loads((api / "topics.json").read_text(encoding="utf-8"))
        cls.bank = json.loads((api / "bank.json").read_text(encoding="utf-8"))
        cls.notes = json.loads((api / "notes.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_script(self, *args, api=None, extra_env=None):
        env = dict(os.environ, COOKING_AGI_API=str(api or self.out / agent_api.API),
                   COOKING_AGI_PROGRESS=str(self.out / "progress.json"), COOKING_AGI_CACHE=str(self.out / "cache"))
        env.update(extra_env or {})
        return subprocess.run([sys.executable, "-I", str(SCRIPT), *args], env=env, capture_output=True, text=True)

    def test_counts_and_cross_references(self):
        self.assertGreater(self.counts["notes"], 200)
        self.assertGreater(self.counts["questions"], 200)
        listed = {n["path"] for t in self.topics for n in t["notes"]}
        self.assertEqual(listed, set(self.notes))
        self.assertTrue({q["note"] for q in self.bank} <= set(self.notes))
        self.assertTrue((self.out / "llms.txt").read_text(encoding="utf-8").count("/api/v1/") >= 3)

    def test_only_published_study_notes(self):
        tops = {path.split("/")[0] for path in self.notes}
        self.assertTrue(tops.isdisjoint({"career", "community", "templates", "papers", "quant"}), tops)
        self.assertTrue(all(path.endswith(".en.md") for path in self.notes))

    def test_removed_company_hiring_content_stays_out(self):
        text = "\n".join(note["text"] for note in self.notes.values())
        for phrase in ("Business Insider", "1point3acres", "一亩三分地"):
            self.assertNotIn(phrase, text)

    def test_drafts_are_flagged(self):
        # a note is a draft while its text carries the review banner; questions inherit the flag
        for path, note in self.notes.items():
            self.assertEqual(note["draft"], "Draft, pending review" in note["text"], path)
        self.assertTrue(all(q["draft"] == self.notes[q["note"]]["draft"] for q in self.bank))

    def test_no_generated_copies_in_the_skill(self):
        skill = SCRIPT.parent.parent
        self.assertFalse((skill / "notes").exists())
        self.assertFalse((skill / "data").exists())

    def test_skill_brand_preserves_install_and_progress_identity(self):
        source = (SCRIPT.parent.parent / 'SKILL.md').read_text()
        guide = (ROOT / 'skills/README.md').read_text()
        self.assertIn('# Cooking AGI interview coach', source)
        self.assertIn('name: cooking-agi-interview\n', source)
        self.assertIn('<a id="cooking-agi-interview-coach"></a>', source)
        self.assertIn('[Cooking AGI](https://github.com/tianyi-zhang-02/cooking-agi)', source)
        self.assertIn('--skill cooking-agi-interview', guide)
        self.assertIn('Cooking AGI', guide)
        for text in [source, guide]:
            self.assertIn('~/.cooking-agi-interview/progress.json', text)

    def test_interview_commands(self):
        topic = next(t for t in self.topics if t["questions"])
        for args in (["topics"], ["notes", topic["id"]], ["ask", "-t", topic["id"], "-n", "1", "--seed", "1"],
                     ["search", "attention"], ["read", topic["notes"][0]["path"]]):
            result = self.run_script(*args)
            self.assertEqual(result.returncode, 0, (args, result.stderr))
        qid = next(q["id"] for q in self.bank if q["topic"] == topic["id"])
        self.assertIn("Q:", self.run_script("reveal", qid).stdout)
        self.assertEqual(self.run_script("record", qid, "3", "--gap", "units").returncode, 0)
        self.assertIn("1 answers", self.run_script("report").stdout)

    def test_read_accepts_links_copied_from_notes(self):
        target = "12-infra/performance.en.md"
        for given in (target, "../../12-infra/performance.en.md#mfu", "12-infra/performance.md",
                      "https://tianyi-zhang-02.github.io/cooking-agi/12-infra/performance.html"):
            result = self.run_script("read", given)
            self.assertEqual(result.returncode, 0, (given, result.stderr))
            self.assertIn(target, result.stdout.splitlines()[0])

    def test_meta_records_build_time(self):
        meta = json.loads((self.out / agent_api.API / "meta.json").read_text(encoding="utf-8"))
        self.assertRegex(meta["built"], r"^\d{4}-\d\d-\d\dT")

    def test_summary_cards_are_concepts_not_questions(self):
        concepts = [q for q in self.bank if q["kind"] == "concept"]
        self.assertTrue(concepts)
        self.assertFalse([q["question"] for q in self.bank if q["kind"] == "interview"
                          and q["question"].lower().startswith(("quick learning", "worked example"))])
        asked = self.run_script("ask", "-n", "1000").stdout
        self.assertFalse([q["id"] for q in concepts if f"[{q['id']}]" in asked])
        self.assertIn(f"[{concepts[0]['id']}]", self.run_script("ask", "-n", "1000", "--kind", "concept").stdout)

    def test_note_links_match_the_site_pages(self):
        import build
        for path, note in self.notes.items():
            page = build.Page(ROOT / path, None, "en")
            self.assertEqual(note["url"], f"https://tianyi-zhang-02.github.io/cooking-agi/{page.url}")
            self.assertEqual(note["zh_url"], note["url"].replace(".en.html", ".html"))
        self.assertTrue(all(q["note_url"] == self.notes[q["note"]]["url"] for q in self.bank))

    def test_section_keeps_its_subsections(self):
        nested = re.compile(r"(?m)^## (.+)\n(?:(?!^## ).*\n)*?^### (.+)$")
        path, match = next((p, m) for p, n in self.notes.items() if (m := nested.search(n["text"])))
        out = self.run_script("read", path, "--section", *match.group(1).split()[:3]).stdout
        self.assertIn("### " + match.group(2), out)

    def test_section_words_in_the_title_do_not_select_the_whole_note(self):
        path = "00-foundations/core/decoding.en.md"  # title: "Decoding strategies: temperature, top-k, top-p"
        out = self.run_script("read", path, "--section", "top-k").stdout
        self.assertNotIn("# Decoding strategies", out.split("\n", 3)[-1])
        self.assertLess(len(out), len(self.notes[path]["text"]) / 2)

    def test_card_fields_do_not_run_together(self):
        text = self.notes["06-systems/inference-frameworks.en.md"]["text"]
        self.assertIn("Generation engine — Choose which requests run together", text)
        self.assertNotIn("engineChoose", text)

    def test_read_one_section(self):
        path, note = next((p, n) for p, n in self.notes.items() if "## Self-check" in n["text"])
        out = self.run_script("read", path, "--section", "self-check").stdout
        self.assertIn("## Self-check", out)
        self.assertLess(len(out), len(note["text"]))
        self.assertIn("headings:", self.run_script("read", path, "--section", "zzz-no-such").stdout)

    def test_first_online_run_caches_everything_for_offline_use(self):
        import functools, http.server, threading
        class Quiet(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *args):
                pass
        handler = functools.partial(Quiet, directory=str(self.out))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_port}/{agent_api.API}"
        with tempfile.TemporaryDirectory() as cache:
            env = {"COOKING_AGI_CACHE": cache}
            self.assertEqual(self.run_script("topics", api=url, extra_env=env).returncode, 0)
            server.shutdown()
            server.server_close()
            for f in Path(cache).iterdir():
                os.utime(f, (0, 0))  # stale: the next run tries the network, fails, falls back
            for args in (["ask", "-n", "1"], ["search", "attention"]):
                result = self.run_script(*args, api=url, extra_env=env)
                self.assertEqual(result.returncode, 0, (args, result.stderr))
                self.assertIn("offline", result.stderr)

    def test_offline_uses_the_cache_or_says_why_not(self):
        offline = "http://127.0.0.1:9/api/v1"
        with tempfile.TemporaryDirectory() as cache:
            empty = self.run_script("topics", api=offline, extra_env={"COOKING_AGI_CACHE": cache})
            self.assertNotEqual(empty.returncode, 0)
            self.assertIn("no cached copy", empty.stderr)
            (Path(cache) / "topics.json").write_text(json.dumps(self.topics), encoding="utf-8")
            os.utime(Path(cache) / "topics.json", (0, 0))  # stale, so a download is attempted first
            cached = self.run_script("topics", api=offline, extra_env={"COOKING_AGI_CACHE": cache})
            self.assertEqual(cached.returncode, 0, cached.stderr)
            self.assertIn("offline", cached.stderr)


if __name__ == "__main__":
    unittest.main()
