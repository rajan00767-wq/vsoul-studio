"""V13.3.1 regression tests for the three gate defects confirmed in the audit.

Each test encodes a measured fact from a real generated artifact, not a
synthetic ideal:

* REG_05 was gated `incomplete_shoulders` although its shoulders span the whole
  canvas, because the gate measured distance from the BACKGROUND COLOUR and the
  shirt was white on a white backdrop.
* REG_06 was gated `collar_registration_failure` although its collar is
  correctly placed, because the red-pixel centroid was dragged sideways by the
  red SLEEVE CUFFS of the same reference.
* REG_06 was gated `incorrect_button_color` on a count of bright PIXELS inside
  the placket strip, which on a gingham check counts the fabric, not buttons.

Plain unittest with numeric assertions, per the repo convention. No models are
loaded and no image is read from outputs/ - fixtures are synthesized here so the
tests stay runnable after debug artifacts are cleaned up.
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np
from PIL import Image

from pipelines.uniform_swap.quality_validator import (
    _measure_button_fidelity,
    _measure_collar_symmetry,
    _measure_shoulder_extent,
)
from pipelines.uniform_swap.reference_analyzer import measure_button_blobs

H, W = 1024, 800
CHIN_Y, FH, FW = 734, 424.0, 342.0
FCX = 400.0


def _white_shirt_on_white_backdrop() -> Image.Image:
    """Full-width shoulders in a near-white shirt, over a white backdrop.

    This is REG_05's situation: the shoulders are complete, but almost nothing
    in the shoulder band differs from the backdrop by more than a few levels.
    """
    rgb = np.full((H, W, 3), 255, dtype=np.uint8)
    alpha = np.zeros((H, W), dtype=np.uint8)
    body = slice(int(CHIN_Y + 0.2 * FH), H)
    # Real photographed white fabric is a few levels off the backdrop and
    # never exactly equal to it. Measured on REG_05's actual shirt at y=0.85h:
    # 796/800 px differ from white by >4, but only 70 px by >30 - which is why
    # the old colour-distance gate saw 70 px of "shoulder" and failed it.
    rgb[body, :, :] = 248
    rgb[body, ::7, :] = 244  # faint weave so it is not a flat fill
    alpha[body, :] = 255
    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(rgba, "RGBA")


class ShoulderExtentTests(unittest.TestCase):
    def test_white_shirt_on_white_backdrop_is_not_incomplete_shoulders(self):
        img = _white_shirt_on_white_backdrop()
        arr = np.asarray(img)
        dist = np.max(np.abs(arr[:, :, :3].astype(np.float32) - 255.0), axis=2)
        measured, required, basis = _measure_shoulder_extent(
            candidate_image=img, identity_layer=None, cand_dist_bg=dist,
            bg_tolerance=30.0, c_chin_y=CHIN_Y, c_fh=FH, c_fw=FW,
            has_face_box=True, h=H, w=W,
        )
        self.assertGreaterEqual(
            measured, required,
            f"complete shoulders in a white shirt must not be gated (basis={basis})")

    def test_genuinely_clipped_shoulders_still_fail(self):
        arr = np.asarray(_white_shirt_on_white_backdrop()).copy()
        keep = slice(int(W * 0.44), int(W * 0.56))
        arr[:, :keep.start, 3] = 0
        arr[:, keep.stop:, 3] = 0
        arr[:, :keep.start, :3] = 255
        arr[:, keep.stop:, :3] = 255
        img = Image.fromarray(arr, "RGBA")
        dist = np.max(np.abs(arr[:, :, :3].astype(np.float32) - 255.0), axis=2)
        measured, required, _ = _measure_shoulder_extent(
            candidate_image=img, identity_layer=None, cand_dist_bg=dist,
            bg_tolerance=30.0, c_chin_y=CHIN_Y, c_fh=FH, c_fw=FW,
            has_face_box=True, h=H, w=W,
        )
        self.assertLess(measured, required, "clipped shoulders must still be detected")

    def test_shoulders_painted_out_in_rgb_fail_despite_intact_alpha(self):
        """The defect injection erases RGB but leaves the person alpha intact."""
        arr = np.asarray(_white_shirt_on_white_backdrop()).copy()
        arr[int(H * 0.75):, :int(W * 0.40), :3] = 255
        arr[int(H * 0.75):, int(W * 0.60):, :3] = 255
        img = Image.fromarray(arr, "RGBA")
        dist = np.max(np.abs(arr[:, :, :3].astype(np.float32) - 255.0), axis=2)
        measured, required, _ = _measure_shoulder_extent(
            candidate_image=img, identity_layer=None, cand_dist_bg=dist,
            bg_tolerance=30.0, c_chin_y=CHIN_Y, c_fh=FH, c_fw=FW,
            has_face_box=True, h=H, w=W,
        )
        self.assertLess(measured, required,
                        "RGB-erased shoulders must fail even when alpha still claims a full body")

    def test_requirement_scales_with_face_not_canvas(self):
        """A narrow child and a broad adult are judged against their own build."""
        img = _white_shirt_on_white_backdrop()
        dist = np.max(np.abs(np.asarray(img)[:, :, :3].astype(np.float32) - 255.0), axis=2)
        _, req_small, _ = _measure_shoulder_extent(
            candidate_image=img, identity_layer=None, cand_dist_bg=dist, bg_tolerance=30.0,
            c_chin_y=CHIN_Y, c_fh=FH, c_fw=200.0, has_face_box=True, h=H, w=W)
        _, req_large, _ = _measure_shoulder_extent(
            candidate_image=img, identity_layer=None, cand_dist_bg=dist, bg_tolerance=30.0,
            c_chin_y=CHIN_Y, c_fh=FH, c_fw=380.0, has_face_box=True, h=H, w=W)
        self.assertLess(req_small, req_large,
                        "a smaller face must impose a smaller shoulder-width requirement")


def _collar_candidate(red_left: bool = True, red_right: bool = True,
                      with_sleeve_trim: bool = True) -> np.ndarray:
    """Canvas with a red collar either side of the face axis, plus red cuffs.

    The cuffs reproduce REF001's layout, which is what broke the old centroid
    measurement.
    """
    img = np.full((H, W, 3), 245, dtype=np.uint8)
    red = (210, 20, 50)
    y0, y1 = int(CHIN_Y - 0.05 * FH), int(CHIN_Y + 0.18 * FH)
    if red_left:
        img[y0:y1, int(FCX - 150):int(FCX - 20)] = red
    if red_right:
        img[y0:y1, int(FCX + 20):int(FCX + 150)] = red
    if with_sleeve_trim:
        # Sleeve cuffs: far lateral, lower down - same colour, not the collar.
        cy0, cy1 = int(CHIN_Y + 0.30 * FH), int(CHIN_Y + 0.55 * FH)
        img[cy0:cy1, 40:150] = red
        img[cy0:cy1, W - 150:W - 40] = red
    return img


def _ref_cues_red_collar() -> dict:
    lab = cv2.cvtColor(np.array([[[210, 20, 50]]], dtype=np.uint8), cv2.COLOR_RGB2LAB)[0, 0]
    body = cv2.cvtColor(np.array([[[150, 180, 225]]], dtype=np.uint8), cv2.COLOR_RGB2LAB)[0, 0]
    return {
        "collar_lab": [float(v) for v in lab],
        "body_lab": [float(v) for v in body],
        "collar_contrasts_body": True,
    }


class CollarSymmetryTests(unittest.TestCase):
    def test_symmetric_collar_passes_despite_red_sleeve_trim(self):
        res = _measure_collar_symmetry(
            _collar_candidate(), _ref_cues_red_collar(), CHIN_Y, FH, FCX, FW, H, W)
        self.assertEqual(res["status"], "PASS", res["details"])

    def test_collar_present_on_one_side_only_fails(self):
        res = _measure_collar_symmetry(
            _collar_candidate(red_right=False), _ref_cues_red_collar(),
            CHIN_Y, FH, FCX, FW, H, W)
        self.assertEqual(res["status"], "FAIL", res["details"])

    def test_collar_indistinguishable_from_body_is_uncertain_not_pass(self):
        cues = _ref_cues_red_collar()
        cues["collar_contrasts_body"] = False
        res = _measure_collar_symmetry(
            _collar_candidate(), cues, CHIN_Y, FH, FCX, FW, H, W)
        self.assertEqual(res["status"], "UNCERTAIN",
                         "an unmeasurable collar must not be reported as a pass")

    def test_works_for_an_arbitrary_collar_colour(self):
        """No colour is privileged: a green collar is measured the same way."""
        img = np.full((H, W, 3), 245, dtype=np.uint8)
        green = (30, 140, 70)
        y0, y1 = int(CHIN_Y - 0.05 * FH), int(CHIN_Y + 0.18 * FH)
        img[y0:y1, int(FCX - 150):int(FCX - 20)] = green
        img[y0:y1, int(FCX + 20):int(FCX + 150)] = green
        lab = cv2.cvtColor(np.array([[green]], dtype=np.uint8), cv2.COLOR_RGB2LAB)[0, 0]
        cues = {"collar_lab": [float(v) for v in lab],
                "body_lab": [60.0, 128.0, 128.0], "collar_contrasts_body": True}
        self.assertEqual(
            _measure_collar_symmetry(img, cues, CHIN_Y, FH, FCX, FW, H, W)["status"], "PASS")


def _placket_strip(button_rgb, fabric="check", sh: int = 420, sw: int = 80,
                   fabric_rgb=(235, 235, 235)) -> np.ndarray:
    """Placket strip with buttons on either plain or gingham fabric."""
    strip = np.full((sh, sw, 3), fabric_rgb, dtype=np.uint8)
    if fabric == "check":
        strip[::12, :, :] = (90, 130, 200)
        strip[:, ::12, :] = (90, 130, 200)
    for cy in range(40, sh - 20, 85):
        cv2.circle(strip, (sw // 2, cy), 13, tuple(int(c) for c in button_rgb), -1)
    return strip


def _paste_placket(button_rgb, fabric="plain", fabric_rgb=(235, 235, 235)) -> np.ndarray:
    """Candidate canvas with a placket pasted into the real placket window."""
    cand = np.full((H, W, 3), 240, dtype=np.uint8)
    y0 = int(np.clip(CHIN_Y + 0.15 * FH, 0, H - 1))
    y1 = int(np.clip(CHIN_Y + 0.95 * FH, y0 + 1, H))
    x0 = int(FCX - 40)
    strip = _placket_strip(button_rgb, fabric=fabric, sh=y1 - y0, sw=80, fabric_rgb=fabric_rgb)
    cand[y0:y1, x0:x0 + 80] = strip
    return cand


class ButtonFidelityTests(unittest.TestCase):
    def test_buttons_are_measured_not_bright_fabric_pixels(self):
        """Gingham has thousands of bright pixels; the buttons are still black."""
        strip = _placket_strip((20, 20, 24), fabric="check")
        bright = int(np.sum(np.all(strip > 195, axis=2)))
        self.assertGreater(bright, 1500, "fixture must contain the bright fabric that used to trip the gate")
        lab, n = measure_button_blobs(strip, min_area=60, max_area=4000, blur_sigma=6.0)
        self.assertIsNotNone(lab, "black buttons on check fabric must be found")
        rgb = cv2.cvtColor(np.array([[lab]], np.float32).astype(np.uint8), cv2.COLOR_LAB2RGB)[0, 0]
        self.assertLess(int(max(rgb)), 110, f"measured button colour should be dark, got {tuple(rgb)}")

    def test_white_buttons_on_black_button_reference_fail(self):
        ref_lab = cv2.cvtColor(np.array([[[20, 20, 24]]], np.uint8), cv2.COLOR_RGB2LAB)[0, 0]
        cues = {"button_presence": True, "button_lab": [float(v) for v in ref_lab]}
        # White buttons need a mid-tone placket to sit on; white-on-white is
        # not a button the eye or the detector can see.
        cand = _paste_placket((248, 248, 248), fabric_rgb=(120, 125, 140))
        res = _measure_button_fidelity(cand, cues, [FCX - FW / 2, CHIN_Y - FH, FCX + FW / 2, CHIN_Y], H, W)
        self.assertEqual(res["status"], "FAIL", res["details"])

    def test_unmeasurable_buttons_report_uncertain_not_pass(self):
        cues = {"button_presence": True, "button_lab": [20.0, 128.0, 128.0]}
        cand = np.full((H, W, 3), 240, dtype=np.uint8)  # no buttons at all
        res = _measure_button_fidelity(cand, cues, [FCX - FW / 2, CHIN_Y - FH, FCX + FW / 2, CHIN_Y], H, W)
        self.assertEqual(res["status"], "UNCERTAIN", res["details"])

    def test_no_colour_is_hardcoded_gold_buttons_match_gold_reference(self):
        gold = (198, 160, 45)
        ref_lab = cv2.cvtColor(np.array([[gold]], np.uint8), cv2.COLOR_RGB2LAB)[0, 0]
        cues = {"button_presence": True, "button_lab": [float(v) for v in ref_lab]}
        cand = _paste_placket(gold)
        res = _measure_button_fidelity(cand, cues, [FCX - FW / 2, CHIN_Y - FH, FCX + FW / 2, CHIN_Y], H, W)
        self.assertNotEqual(res["status"], "FAIL",
                            f"gold buttons matching a gold reference must not fail: {res['details']}")


if __name__ == "__main__":
    unittest.main()
