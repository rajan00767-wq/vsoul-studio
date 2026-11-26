"""Pre-diffusion uniform-conditioning edge cleanup tests."""
import ast
from pathlib import Path
import unittest

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")


def _load_cleaner():
    tree = ast.parse(PIPELINE)
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "QwenEditPipeline"
    )
    fn = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_decontaminate_uniform_conditioning_edge"
    )
    ns = {"np": np, "cv2": cv2}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "edge", "exec"), ns)
    return ns["_decontaminate_uniform_conditioning_edge"]


class ConditioningEdgeTests(unittest.TestCase):
    def setUp(self):
        self.clean = _load_cleaner()

    @staticmethod
    def _subject():
        rgba = np.zeros((96, 96, 4), dtype=np.uint8)
        cv2.circle(rgba, (48, 48), 31, (24, 20, 18, 255), -1)
        # Simulate a SCHP-hardened white source-backdrop fringe.
        ring = np.zeros((96, 96), dtype=np.uint8)
        cv2.circle(ring, (48, 48), 32, 255, 2)
        rgba[ring > 0, :3] = 246
        rgba[ring > 0, 3] = 250
        return rgba, ring > 0

    def test_hardened_white_fringe_uses_inward_subject_colour(self):
        rgba, ring = self._subject()
        out, count = self.clean(rgba)
        self.assertGreater(count, 0)
        self.assertLess(float(out[ring, :3].mean()), 100.0)
        # The contaminated boundary is softly feathered, but confident
        # interior subject alpha remains exact.
        self.assertLessEqual(int(out[:, :, 3].max()), 255)
        np.testing.assert_array_equal(
            out[32:64, 32:64, 3], rgba[32:64, 32:64, 3],
        )

    def test_real_pale_accessory_is_preserved(self):
        rgba, _ = self._subject()
        # A pale accessory spans edge and inward core, so its nearest inward
        # authority is also pale and must not be darkened.
        rgba[38:58, 14:35, :3] = (238, 230, 220)
        rgba[38:58, 14:35, 3] = 255
        before = rgba.copy()
        out, _ = self.clean(rgba)
        self.assertGreater(float(out[42:54, 17:31, :3].mean()), 205.0)
        np.testing.assert_array_equal(out[42:54, 17:31], before[42:54, 17:31])

    def test_repaired_fringe_is_not_left_as_opaque_staircase(self):
        rgba, ring = self._subject()
        out, _ = self.clean(rgba)
        self.assertTrue(np.any(out[ring, 3] < rgba[ring, 3]))

    def test_cleanup_runs_before_qwen_conditioning_composite(self):
        clean_at = PIPELINE.index("self._decontaminate_uniform_conditioning_edge")
        composite_at = PIPELINE.index("matted_rgb = (")
        self.assertLess(clean_at, composite_at)


if __name__ == "__main__":
    unittest.main()
