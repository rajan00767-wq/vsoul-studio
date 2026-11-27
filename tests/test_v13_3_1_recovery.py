"""V13.3.1 tests for recovery rollback and anatomical shoulder reconstruction.

Two properties the V13.3 controller did not have:

* A repair that removes two defects but introduces a third reduced the defect
  COUNT, so it was accepted and shipped a defect the candidate never had.
* `_repair_incomplete_shoulders` pushed every torso row out to fixed canvas
  fractions (0.12 w / 0.88 w), which is the rectangular expansion the spec
  forbids: it squares the silhouette and ignores the subject's own build.
"""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from pipelines.uniform_swap.config import UniformSwapConfig
from pipelines.uniform_swap.recovery_controller import RecoveryController, _defect_key

H, W = 400, 320


def _sloped_torso(clip_rows: int = 0, clip_px: int = 0) -> np.ndarray:
    """Alpha of a torso whose shoulders slope outwards, optionally clipped."""
    alpha = np.zeros((H, W), dtype=np.uint8)
    for y in range(200, H):
        half = int(40 + (y - 200) * 0.9)          # widening shoulder slope
        half = min(half, W // 2 - 10)
        alpha[y, W // 2 - half:W // 2 + half] = 255
    if clip_rows:
        for y in range(H - clip_rows, H):
            xs = np.where(alpha[y] > 30)[0]
            if len(xs):
                alpha[y, xs.min():xs.min() + clip_px] = 0
                alpha[y, max(xs.min(), xs.max() - clip_px):xs.max() + 1] = 0
    return alpha


class ShoulderReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.rc = RecoveryController(max_attempts=2)
        self.id_layer = {"metrics": {"chin_y": 180, "face_height": 110.0,
                                     "face_center_x": W / 2.0}}

    def test_intact_sloped_shoulders_are_left_alone(self):
        alpha = _sloped_torso()
        rgb = np.full((H, W, 3), 180, dtype=np.uint8)
        before = alpha.copy()
        _, out_a, actions = self.rc._repair_incomplete_shoulders(
            rgb, alpha, self.id_layer, W, H)
        self.assertEqual(actions, [], "a complete shoulder line needs no repair")
        self.assertEqual(int(np.abs(out_a.astype(int) - before.astype(int)).sum()), 0)

    def test_flat_cut_shoulder_is_rebuilt_along_the_real_contour(self):
        """A straight vertical cut - what real clipping looks like - is restored.

        Note the detectable class: a cut that runs PARALLEL to the shoulder
        slope leaves a silhouette that is still self-consistently widening, and
        nothing in the alpha distinguishes it from a genuinely narrower child.
        Reconstruction here deliberately targets contour violations, which is
        what a clipped garment, a canvas-edge cut or a matting failure produce.
        """
        truth = _sloped_torso()
        alpha = truth.copy()
        cut_lo, cut_hi = 230, 270
        left_cut = int(np.where(truth[cut_lo] > 30)[0].min())
        right_cut = int(np.where(truth[cut_lo] > 30)[0].max())
        for y in range(cut_lo, cut_hi):
            alpha[y, :left_cut] = 0        # flat vertical boundary, both sides
            alpha[y, right_cut + 1:] = 0
        rgb = np.full((H, W, 3), 180, dtype=np.uint8)
        _, out_a, actions = self.rc._repair_incomplete_shoulders(
            rgb, alpha, self.id_layer, W, H)
        self.assertTrue(any("reconstructed_shoulder_contour" in a for a in actions), actions)

        edges = [int(np.where(out_a[y] > 30)[0].min()) for y in range(cut_lo, cut_hi)]
        true_edges = [int(np.where(truth[y] > 30)[0].min()) for y in range(cut_lo, cut_hi)]
        self.assertGreater(len(set(edges)), 1,
                           "reconstructed edge must follow the slope, not stay a flat vertical cut")
        self.assertLess(max(abs(a - b) for a, b in zip(edges, true_edges)), 6,
                        "reconstructed contour must track the subject's real shoulder line")

    def test_reconstruction_declines_rather_than_distort_when_deficit_is_large(self):
        alpha = _sloped_torso()
        alpha[200:, :int(W * 0.40)] = 0   # most of one side missing
        rgb = np.full((H, W, 3), 180, dtype=np.uint8)
        _, _, actions = self.rc._repair_incomplete_shoulders(
            rgb, alpha, self.id_layer, W, H)
        self.assertTrue(any("declined_shoulder_reconstruction_too_large" in a for a in actions),
                        f"a huge deficit must be declined, not stretched: {actions}")


class RecoveryRollbackTests(unittest.TestCase):
    def test_defect_key_ignores_measured_numbers(self):
        self.assertEqual(_defect_key("missing_crown_bow_1 (559 px < 1524)"), "missing_crown_bow_1")
        self.assertEqual(_defect_key("incomplete_shoulders"), "incomplete_shoulders")

    @staticmethod
    def _halo_candidate() -> Image.Image:
        """Candidate with a genuine white boundary fringe, so the halo route acts."""
        rgb = np.full((H, W, 3), 200, dtype=np.uint8)
        alpha = np.zeros((H, W), dtype=np.uint8)
        alpha[20:H - 20, 20:W - 20] = 255
        rgb[20:26, 20:W - 20] = 235          # white fringe along the top edge
        rgb[H - 26:H - 20, 20:W - 20] = 235
        return Image.fromarray(np.dstack([rgb, alpha]), "RGBA")

    def test_repair_that_introduces_a_new_defect_is_rolled_back(self):
        rc = RecoveryController(max_attempts=1)
        cand = self._halo_candidate()
        src = Image.fromarray(np.full((H, W, 3), 200, dtype=np.uint8))
        id_layer = {
            "metrics": {"chin_y": 180, "face_height": 110.0, "face_center_x": W / 2.0},
            "original_face_mask": np.zeros((H, W), dtype=bool),
            "original_neck_mask": np.zeros((H, W), dtype=bool),
        }
        before = {"defects": ["white_edge_halo", "background_leakage"], "passed": False}

        # Net count drops 2 -> 1, but the one left is a defect that was NOT
        # present before. The old count-only rule would have accepted this.
        def validator(_c, _s, _i):
            return {"defects": ["duplicated_hair"], "passed": False}

        res = rc.attempt_recovery(
            current_candidate_rgba=cand, normalized_source_img=src, identity_layer=id_layer,
            validation_report=before, reference_spec={}, config=UniformSwapConfig(),
            validator_fn=validator, job_id="rollback_test",
        )
        self.assertEqual(res["defects_after"], 2, "introduced defect must trigger a rollback")
        self.assertTrue(any("rolled_back" in a for a in res["actions_taken"]), res["actions_taken"])
        self.assertFalse(res["success"])

    def test_clean_improvement_is_still_accepted(self):
        rc = RecoveryController(max_attempts=1)
        cand = self._halo_candidate()
        src = Image.fromarray(np.full((H, W, 3), 200, dtype=np.uint8))
        id_layer = {
            "metrics": {"chin_y": 180, "face_height": 110.0, "face_center_x": W / 2.0},
            "original_face_mask": np.zeros((H, W), dtype=bool),
            "original_neck_mask": np.zeros((H, W), dtype=bool),
        }
        before = {"defects": ["white_edge_halo"], "passed": False}
        res = rc.attempt_recovery(
            current_candidate_rgba=cand, normalized_source_img=src, identity_layer=id_layer,
            validation_report=before, reference_spec={}, config=UniformSwapConfig(),
            validator_fn=lambda *_: {"defects": [], "passed": True}, job_id="improve_test",
        )
        self.assertEqual(res["defects_after"], 0)
        self.assertTrue(res["success"])


if __name__ == "__main__":
    unittest.main()
