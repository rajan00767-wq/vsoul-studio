from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FrontendJobPollingTests(unittest.TestCase):
    def test_all_pollers_stop_on_missing_jobs(self):
        source = (ROOT / "index.html").read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count("status === 404"), 3)
        self.assertIn("enhanceState.jobId = null", source)
        self.assertIn("uniformState.jobId = null", source)
        self.assertIn("Job expired after server restart. Retry this item.", source)
        self.assertNotIn("if (!resp.ok) return;", source)
        self.assertNotIn("if (!jResp.ok) return;", source)


if __name__ == "__main__":
    unittest.main()
