import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class StudyBuddyTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js required")
    def test_buddy_logic(self):
        subprocess.run([shutil.which("node"), "--test", str(ROOT / "site/tests/study-buddy.test.cjs")], check=True, capture_output=True)

    def test_every_page_loads_the_buddy_after_the_reader_hub(self):
        source = (ROOT / "site/build.py").read_text(encoding="utf-8")
        hub = source.index("static/reader-hub.js")
        self.assertLess(hub, source.index("static/study-buddy.js"))
        self.assertIn("static/study-buddy.css", source)


if __name__ == "__main__":
    unittest.main()
