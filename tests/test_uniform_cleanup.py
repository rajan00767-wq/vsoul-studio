import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipelines.qwen_edit_pipeline import _cleanup_successful_uniform_artifacts


class UniformCleanupTests(unittest.TestCase):
    def test_cleanup_is_off_by_default_so_artifacts_survive(self):
        """Stage images are the only way to localize a fit/neck/halo defect.

        Deleting them on success meant a reported defect could only be
        investigated by reproducing the whole job.
        """
        with tempfile.TemporaryDirectory() as folder:
            output_dir = Path(folder)
            final = output_dir / "job123_uniform.png"
            final.write_bytes(b"final")
            rough = output_dir / "job123_rough_fit.png"
            rough.write_bytes(b"rough")
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("VSOUL_CLEAN_SUCCESSFUL_UNIFORM_ARTIFACTS", None)
                removed = _cleanup_successful_uniform_artifacts(
                    "job123", final, output_dir=output_dir,
                )
            self.assertEqual(removed, 0)
            self.assertTrue(rough.exists())

    def test_success_cleanup_keeps_final_and_other_jobs(self):
        with tempfile.TemporaryDirectory() as folder:
            output_dir = Path(folder)
            final = output_dir / "job123_uniform.png"
            final.write_bytes(b"final")
            (output_dir / "job123_qwen_raw.png").write_bytes(b"raw")
            (output_dir / "job123_rough_fit.jpg").write_bytes(b"rough")
            audit = output_dir / "job123_instructions.json"
            audit.write_text("{}", encoding="utf-8")
            other = output_dir / "job999_qwen_raw.png"
            other.write_bytes(b"other")

            with mock.patch.dict(
                os.environ,
                {"VSOUL_CLEAN_SUCCESSFUL_UNIFORM_ARTIFACTS": "1"},
            ):
                removed = _cleanup_successful_uniform_artifacts(
                    "job123", final, output_dir=output_dir,
                )

            self.assertEqual(removed, 2)
            self.assertTrue(final.exists())
            self.assertTrue(audit.exists())
            self.assertTrue(other.exists())


if __name__ == "__main__":
    unittest.main()
