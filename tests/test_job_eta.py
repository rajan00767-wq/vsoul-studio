import unittest
from unittest.mock import patch

from utils.job_eta import estimate_job_time


class JobEtaTests(unittest.TestCase):
    @patch("utils.job_eta.time.time", return_value=1_000.0)
    def test_uniform_processing_eta_uses_uniform_baseline(self, _now):
        result = estimate_job_time({
            "status": "processing",
            "progress": 50,
            "started_at": "900",
            "metadata": {"type": "uniform_swap"},
        })
        self.assertEqual(result["estimated_total_seconds"], 420)
        self.assertEqual(result["elapsed_seconds"], 100)
        self.assertEqual(result["estimated_remaining_seconds"], 320)
        self.assertIn("remaining", result["estimated_time"])

    @patch("utils.job_eta.time.time", return_value=1_000.0)
    def test_queue_position_includes_waiting_jobs(self, _now):
        result = estimate_job_time({
            "status": "queued",
            "progress": 0,
            "metadata": {"type": "qwen_enhance"},
        }, queue_position=3)
        self.assertEqual(result["estimated_remaining_seconds"], 900)

    def test_terminal_job_has_no_remaining_time(self):
        result = estimate_job_time({"status": "done", "progress": 100})
        self.assertEqual(result["estimated_remaining_seconds"], 0)
        self.assertEqual(result["estimated_time"], "complete")


if __name__ == "__main__":
    unittest.main()
