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

    def test_uniform_queue_has_one_dynamic_vl_worker(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        frontend = (ROOT / "index.html").read_text(encoding="utf-8")
        branch = source[source.index('elif job_type == "uniform_swap":'):]
        self.assertNotIn('job_type == "qwen_uniform_swap"', source)
        self.assertIn("def _cb", branch)
        self.assertIn("qwen_vl_image_edit_uniform_swap", branch)
        self.assertNotIn("prompt_mode=", branch)
        self.assertNotIn("uniform-prompt-mode", frontend)
        self.assertNotIn("Simple Static Prompt Test", frontend)

    def test_enhancement_static_prompt_option_reaches_worker(self):
        frontend = (ROOT / "index.html").read_text(encoding="utf-8")
        routes = (ROOT / "api/routes.py").read_text(encoding="utf-8")
        main = (ROOT / "main.py").read_text(encoding="utf-8")
        enhancer = (ROOT / "gradio_app.py").read_text(encoding="utf-8")

        self.assertIn('id="enhance-prompt-mode"', frontend)
        self.assertIn('value="static_test">Simple Static Prompt (Test)', frontend)
        self.assertIn("fd.append('prompt_mode'", frontend)
        self.assertIn('prompt_mode: str = Form("dynamic")', routes)
        self.assertIn('"prompt_mode": (', routes)
        self.assertIn('prompt_mode=str(metadata.get("prompt_mode") or "dynamic")', main)
        self.assertIn("def build_static_enhancement_prompt", enhancer)
        self.assertIn('normalized_prompt_mode == "static_test"', enhancer)
        self.assertIn('"source": "static_prompt_fast_path"', enhancer)
        self.assertIn("skipped redundant", enhancer)
        self.assertIn("static_generation_edge = 1024", enhancer)
        self.assertIn("max_generation_dimension=generation_edge", enhancer)


if __name__ == "__main__":
    unittest.main()
