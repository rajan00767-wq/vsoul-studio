import tempfile
import unittest
from pathlib import Path

from pipelines.qwen_edit_pipeline import _cleanup_successful_uniform_artifacts


class UniformCleanupTests(unittest.TestCase):
    def test_success_cleanup_keeps_final_and_other_jobs(self):
        scratch = Path(__file__).resolve().parents[1] / "scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            output_dir = Path(folder)
            final = output_dir / "job123_uniform.png"
            final.write_bytes(b"final")
            (output_dir / "job123_qwen_raw.png").write_bytes(b"raw")
            (output_dir / "job123_rough_fit.jpg").write_bytes(b"rough")
            audit = output_dir / "job123_instructions.json"
            audit.write_text("{}", encoding="utf-8")
            other = output_dir / "job999_qwen_raw.png"
            other.write_bytes(b"other")

            removed = _cleanup_successful_uniform_artifacts(
                "job123", final, output_dir=output_dir,
            )

            self.assertEqual(removed, 2)
            self.assertTrue(final.exists())
            self.assertTrue(audit.exists())
            self.assertTrue(other.exists())


if __name__ == "__main__":
    unittest.main()
