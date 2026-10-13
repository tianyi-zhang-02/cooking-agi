import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class CounterServiceTests(unittest.TestCase):
    """The Cloudflare counter (services/counter) runs its logic on a real SQLite."""

    @unittest.skipUnless(shutil.which("node"), "Node.js required")
    def test_counter_service(self):
        result = subprocess.run([shutil.which("node"), "--test", "test/"], cwd=ROOT / "services/counter",
                                capture_output=True, text=True, env={"PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin", "PYTHON": sys.executable})
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-2000:])
        self.assertRegex(result.stdout, r"# pass [1-9]")
        self.assertIn("# fail 0", result.stdout)

    def test_no_request_logging_and_no_cookies(self):
        source = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "services/counter/src").glob("*.js"))
        self.assertNotIn("console.", source)
        self.assertNotIn("Set-Cookie", source)
        self.assertIn("enabled = false", (ROOT / "services/counter/wrangler.toml").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
