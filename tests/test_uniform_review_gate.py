"""Reviewer-gate regression tests.

The gate had two defects, both visible on job 5a1bbeaa:

* ``blocking_failures`` was computed, written to the audit as
  ``delivery_blocked`` and logged as "Bypassed ... to deliver generation
  output" - but nothing ever read it and the render always shipped. A gate
  that fires on every run and blocks nothing trains people to ignore it.
* The semantic reviewer has no neck criterion, so it returned an identical
  verdict for a portrait whose neck was tripled (0.475 face-boxes of exposed
  skin against a 0.162 source) and for a corrected one at 0.280.

The fix is honesty plus a measurement: the list is named for what it is, and
the defect the reviewer structurally could not see is measured deterministically
and surfaced to the user.
"""
import ast
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
MAIN = (ROOT / "main.py").read_text(encoding="utf-8")


def _load_measure():
    """Exec just the measurement helper; no model import."""
    import cv2
    tree = ast.parse(PIPELINE)
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "QwenEditPipeline")
    fn = next(n for n in cls.body
              if isinstance(n, ast.FunctionDef) and n.name == "measure_exposed_neck")
    fn = ast.Module(body=[fn], type_ignores=[])
    ns = {"np": np, "cv2": cv2, "Image": Image}
    exec(compile(fn, "measure", "exec"), ns)
    return ns["measure_exposed_neck"]


class MeasureExposedNeckTests(unittest.TestCase):
    def setUp(self):
        self.measure = _load_measure()

    def _portrait(self, neck_rows):
        """White frame, a face box, and `neck_rows` of skin below the chin."""
        img = Image.new("RGB", (200, 400), (255, 255, 255))
        px = np.array(img)
        px[50:150, 70:130] = (180, 130, 90)          # face, chin at y=150
        if neck_rows:
            px[150:150 + neck_rows, 85:115] = (180, 130, 90)
        return Image.fromarray(px), (70.0, 50.0, 130.0, 150.0)

    def test_more_exposed_skin_measures_higher(self):
        short, box = self._portrait(10)
        long_, _ = self._portrait(45)
        self.assertLess(self.measure(short, box), self.measure(long_, box))

    def test_no_neck_measures_near_zero(self):
        img, box = self._portrait(0)
        self.assertAlmostEqual(self.measure(img, box), 0.0, places=3)

    def test_result_is_scale_invariant(self):
        # Face-box units, so a resized frame of the same portrait must agree.
        img, box = self._portrait(40)
        small = img.resize((100, 200))
        sbox = tuple(v / 2 for v in box)
        # Resampling a hard-edged synthetic mask at 2x costs a little area,
        # so allow a small tolerance; the point is that the units do not
        # scale with pixel count the way a raw pixel count would (4x here).
        self.assertAlmostEqual(
            self.measure(img, box), self.measure(small, sbox), delta=0.03,
        )

    def test_degenerate_face_box_returns_none(self):
        img, _ = self._portrait(20)
        # Chin already at the bottom edge: no region left to measure.
        self.assertIsNone(self.measure(img, (70.0, 50.0, 130.0, 400.0)))

    def test_white_background_is_not_counted_as_skin(self):
        img, box = self._portrait(0)
        self.assertLess(self.measure(img, box), 0.01)


class GateHonestyTests(unittest.TestCase):
    def test_delivery_blocked_is_not_claimed(self):
        # It never blocked anything; it must not say it did.
        self.assertIn('visual_review["delivery_blocked"] = False', PIPELINE)
        self.assertNotIn(
            'visual_review["delivery_blocked"] = bool(blocking_failures)', PIPELINE,
        )

    def test_warnings_are_named_for_what_they_are(self):
        self.assertIn("quality_warnings = [", PIPELINE)
        self.assertIn('visual_review["quality_warnings"] = quality_warnings', PIPELINE)

    def test_bypass_log_no_longer_claims_a_bypass(self):
        self.assertNotIn("Bypassed blocking failure", PIPELINE)
        self.assertIn("Quality warnings on the delivered", PIPELINE)

    def test_neck_is_measured_and_recorded(self):
        self.assertIn("measure_exposed_neck", PIPELINE)
        self.assertIn('visual_review["exposed_neck_growth"]', PIPELINE)
        self.assertIn("neck_stretched_measured", PIPELINE)

    def test_neck_measurement_failure_cannot_fail_the_job(self):
        idx = PIPELINE.index("Could not measure exposed neck")
        window = PIPELINE[idx - 2500: idx]
        self.assertIn("try:", window)
        self.assertIn("except Exception as neck_error:", window)

    def test_user_is_told_about_measured_defects(self):
        # The warning used to reach only the server log.
        self.assertIn("neck_stretched_measured", MAIN)
        self.assertIn("quality_warnings", MAIN)
        self.assertIn("the neck looks longer than it should", MAIN)

    def test_review_read_failure_cannot_fail_the_job(self):
        idx = MAIN.index("Could not read delivered-portrait review")
        self.assertIn("try:", MAIN[idx - 1500: idx])


if __name__ == "__main__":
    unittest.main()
