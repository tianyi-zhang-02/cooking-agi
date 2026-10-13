from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SITE))
import note_metrics

spec = importlib.util.spec_from_file_location("metrics_preview", SITE / "tools/metrics_preview.py")
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)
NOTE = "note-78153ce088c6824e"
OTHER = "note-0123456789abcdef"
TOKEN = "a" * 32


class NoteMetricsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.now = datetime(2026, 10, 31, 23, 59, tzinfo=timezone.utc)
        self.database = Path(self.directory.name) / "counts.sqlite"
        self.counts = preview.Counts(self.database, [NOTE, OTHER], lambda: self.now)

    def test_empty_counts_are_real_zero_only_after_service_response(self):
        values = self.counts.read(NOTE)
        self.assertEqual(values["reads"], 0)
        self.assertEqual(values["saves"], 0)
        self.assertEqual(values["mode"], "local")
        config = json.loads((SITE / "static/note-metrics-config.json").read_text())
        # The only production endpoint is the reviewed counter service (services/counter).
        self.assertEqual(set(config), {"version", "endpoint"})
        self.assertEqual(config["version"], 1)
        self.assertIn(config["endpoint"], {"", "https://cooking-agi-counter.cooking-agi-counter.workers.dev/api/note-metrics/"})

    def test_daily_deduplication_month_rollover_and_restart(self):
        event = {"kind": "view", "token": TOKEN}
        self.counts.record(NOTE, event)
        self.assertEqual(self.counts.record(NOTE, event)["reads"], 1)
        self.now = datetime(2026, 11, 1, 0, 1, tzinfo=timezone.utc)
        self.assertEqual(self.counts.read(NOTE)["monthReads"], 0)
        result = self.counts.record(NOTE, event)
        self.assertEqual((result["reads"], result["monthReads"]), (2, 1))
        restarted = preview.Counts(self.database, [NOTE, OTHER], lambda: self.now)
        self.assertEqual(restarted.record(NOTE, event)["reads"], 2)
        self.assertEqual(restarted.read(OTHER)["reads"], 0)

    def test_save_retries_unsaves_and_month_semantics(self):
        saved = {"kind": "bookmark", "token": TOKEN, "saved": True}
        for _ in range(4):
            result = self.counts.record(NOTE, saved)
        self.assertEqual((result["saves"], result["monthSaves"]), (1, 1))
        self.now = datetime(2026, 11, 1, tzinfo=timezone.utc)
        result = self.counts.record(NOTE, saved)
        self.assertEqual((result["saves"], result["monthSaves"]), (1, 0))
        for _ in range(3):
            result = self.counts.record(NOTE, {**saved, "saved": False})
        self.assertEqual((result["saves"], result["monthSaves"]), (0, 0))
        result = self.counts.record(NOTE, saved)
        self.assertEqual((result["saves"], result["monthSaves"]), (1, 1))

    def test_parallel_duplicate_reads_increment_once(self):
        with ThreadPoolExecutor(max_workers=8) as workers:
            list(workers.map(lambda _: self.counts.record(NOTE, {"kind": "view", "token": TOKEN}), range(20)))
        self.assertEqual(self.counts.read(NOTE)["reads"], 1)

    def test_unknown_notes_and_untrusted_values_cannot_write(self):
        with self.assertRaises(LookupError):
            self.counts.record("note-ffffffffffffffff", {"kind": "view", "token": TOKEN})
        for event in [None, {}, {"kind": "view", "token": TOKEN, "date": "2000-01-01"},
                      {"kind": "view", "token": "short"}, {"kind": "bookmark", "token": TOKEN, "saved": 1},
                      {"kind": "bookmark", "token": TOKEN, "saved": True, "delta": 9000}]:
            with self.subTest(event=event), self.assertRaises(ValueError):
                self.counts.record(NOTE, event)
        self.assertEqual(self.counts.read(NOTE)["reads"], 0)

    def test_tokens_are_hashed_and_note_scoped(self):
        for note in (NOTE, OTHER):
            self.counts.record(note, {"kind": "bookmark", "token": TOKEN, "saved": True})
        with self.counts.connect() as connection:
            visitors = [row[0] for row in connection.execute("SELECT visitor FROM bookmarks")]
        self.assertEqual(len(set(visitors)), 2)
        self.assertNotIn(TOKEN, visitors)

    def test_panel_uses_shared_ids_and_never_seeds_counts(self):
        for language in ("zh", "en"):
            page = SimpleNamespace(kind="article", src=SITE.parent / ("interview/algorithms/backtracking-and-dp" + (".en" if language == "en" else "") + ".md"), lang=language)
            panel = note_metrics.panel(page, SITE.parent)
            self.assertIn(f'data-note-metrics="{NOTE}"', panel)
            self.assertEqual(panel.count('data-metric="'), 4)
            self.assertEqual(panel.count('>—</dd>'), 4)
            self.assertIn('data-metrics-consent disabled', panel)
            self.assertNotIn('checked', panel)
            for kind in ("home", "index", "project"):
                page.kind = kind
                self.assertEqual(note_metrics.panel(page, SITE.parent), "")

    def test_client_contract_and_retry_behaviors(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js unavailable")
        subprocess.run([node, "--test", str(SITE / "tests/note-metrics.test.cjs"),
                        str(SITE / "tests/note-metrics-dom.test.cjs")], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
