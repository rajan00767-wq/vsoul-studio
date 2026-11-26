"""Neck-seat measurement range regression tests (source-level, no models).

Job 5a1bbeaa delivered a neck a third too long. Two clamps caused the residual
after the prompt-truncation fix:

* the visible-neck scan stopped at 0.22 face-heights below the chin while its
  own comment promised 0.32, so an open-collar upload whose throat really runs
  to 0.32 could never measure past ~0.23;
* ``neck_cap`` then clipped whatever survived to 0.20, and the consumer in
  qwen_edit_pipeline clipped again to 0.25.

The NECK/COLLAR LOCK prompt reported 0.20 face-heights for a neck measured at
0.32, contradicting the conditioning image Qwen was looking at. These tests pin
the three bounds to one consistent range.
"""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPOSITE = (ROOT / "pipelines/uniform_composite.py").read_text(encoding="utf-8")
PIPELINE = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")

# The anatomical range a measured neck seat may occupy, in face-heights.
NECK_RANGE_MAX = 0.32


def _assign_value(source: str, name: str):
    """Value of the last module-level-ish assignment to ``name``."""
    tree = ast.parse(source)
    found = [n for n in ast.walk(tree)
             if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert found, f"no assignment to {name}"
    return found[-1]


class NeckSeatRangeTests(unittest.TestCase):
    def test_visible_neck_scan_reaches_the_documented_range(self):
        # The scan ceiling and the comment above it must agree.
        self.assertIn("y2 + face_h * .32", COMPOSITE)
        self.assertNotIn("max(source_collar_y, y2 + face_h * .22)", COMPOSITE)

    def test_neck_cap_does_not_clip_inside_the_scan_range(self):
        node = _assign_value(COMPOSITE, "neck_cap")
        self.assertIsInstance(node.value, ast.Call)
        literals = [a.value for a in node.value.args if isinstance(a, ast.Constant)]
        self.assertTrue(literals, "neck_cap must carry an explicit floor")
        self.assertAlmostEqual(
            max(literals), NECK_RANGE_MAX, places=6,
            msg="neck_cap clips measurements the scan can legitimately produce",
        )

    def test_consumer_clamp_matches_the_measurement_range(self):
        node = _assign_value(PIPELINE, "neck_seat_ratio_val")
        bounds = [a.value for a in ast.walk(node.value)
                  if isinstance(a, ast.Constant) and isinstance(a.value, float)]
        self.assertTrue(bounds)
        self.assertAlmostEqual(
            max(bounds), NECK_RANGE_MAX, places=6,
            msg="qwen_edit_pipeline re-clips the composite's measured neck seat",
        )

    def test_all_three_bounds_agree(self):
        # One range, stated the same way in every place that clamps it, so a
        # future edit to one cannot silently re-introduce the mismatch.
        cap = max(a.value for a in _assign_value(COMPOSITE, "neck_cap").value.args
                  if isinstance(a, ast.Constant))
        consumer = max(
            a.value for a in ast.walk(_assign_value(PIPELINE, "neck_seat_ratio_val").value)
            if isinstance(a, ast.Constant) and isinstance(a.value, float)
        )
        self.assertEqual(cap, consumer)
        self.assertIn(f"y2 + face_h * .{str(NECK_RANGE_MAX).split('.')[1]}", COMPOSITE)


class NeckLockPromptTests(unittest.TestCase):
    def test_prompt_reports_the_measured_seat(self):
        # The lock quotes source_neck_gap_ratio, which comes from the
        # composite's measurement rather than a constant.
        self.assertIn("source_neck_gap_ratio = measured_neck_gap_ratio", PIPELINE)
        self.assertIn("{source_neck_gap_ratio:.2f} face heights", PIPELINE)


if __name__ == "__main__":
    unittest.main()
