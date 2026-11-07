from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FrontendJobPollingTests(unittest.TestCase):
    def test_all_pollers_stop_on_missing_jobs(self):
        source = (ROOT / "index.html").read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count("status === 404"), 3)
        self.assertIn("enhanceState.jobId = null", source)
        self.assertIn("uniformState.jobId = null", source)
        uniform_start = source[source.index("async function startUniformSwap()"):source.index("function pollUniformJob")]
        self.assertIn("clearInterval(uniformState.pollInterval)", uniform_start)
        self.assertIn("uniformState.pollInterval = null", uniform_start)
        self.assertIn("Job expired after server restart. Retry this item.", source)
        self.assertNotIn("if (!resp.ok) return;", source)
        self.assertNotIn("if (!jResp.ok) return;", source)

    def test_qwen_uniform_queue_reports_pipeline_progress(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        branch = source[source.index('elif job_type == "qwen_uniform_swap":'):
                        source.index('elif job_type == "uniform_swap":')]
        self.assertIn("def _uniform_cb", branch)
        self.assertIn('stage="uniform_generation"', branch)
        self.assertIn("current_progress", branch)
        self.assertIn("_uniform_cb,", branch)


if __name__ == "__main__":
    unittest.main()
