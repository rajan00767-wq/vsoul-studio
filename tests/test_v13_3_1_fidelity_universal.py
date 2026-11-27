"""V13.3.1: uniform fidelity must work for arbitrary colours, not just red/navy.

Before V13.3.1, `UniformFidelityValidator` branched on the literal strings
"red" and "navy blue" with hand-tuned RGB/HSV windows. Every other collar
colour fell through to a branch that recorded PASS with score 1 and threshold 1
without measuring anything, so "universal uniform support" was never tested
beyond two colours.
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from pipelines.uniform_swap.fidelity_validator import uniform_fidelity_validator

H, W = 1024, 800
FACE_BBOX = (250, 150, 550, 450)   # x1, y1, x2, y2 -> chin_y = 450


def _lab_of(rgb):
    return [float(v) for v in cv2.cvtColor(np.array([[rgb]], np.uint8), cv2.COLOR_RGB2LAB)[0, 0]]


def _candidate_with_collar(collar_rgb, body_rgb=(200, 205, 215)):
    img = np.full((H, W, 3), 255, dtype=np.uint8)
    mask = np.zeros((H, W), dtype=bool)
    img[440:1024, 150:650] = body_rgb
    mask[440:1024, 150:650] = True
    # Collar band straddling the chin line, inside the validator's collar zone.
    img[445:520, 280:520] = collar_rgb
    return img, mask


def _spec(collar_rgb, **extra):
    s = {
        "ref_type": "generic_uniform",
        "collar_color": "measured",
        "collar_lab": _lab_of(collar_rgb),
        "has_tie": False,
        "has_vest": False,
        "fabric_pattern": "plain",
    }
    s.update(extra)
    return s


class UniversalCollarColourTests(unittest.TestCase):
    def test_arbitrary_green_collar_matching_reference_passes(self):
        img, mask = _candidate_with_collar((30, 140, 70))
        res = uniform_fidelity_validator.validate_fidelity(
            candidate_rgb=img, garment_mask=mask,
            reference_spec=_spec((30, 140, 70)), face_bbox=FACE_BBOX)
        self.assertEqual(res["checks"]["collar_color"]["status"], "PASS",
                         res["checks"]["collar_color"]["details"])

    def test_arbitrary_green_collar_against_maroon_reference_fails(self):
        img, mask = _candidate_with_collar((30, 140, 70))
        res = uniform_fidelity_validator.validate_fidelity(
            candidate_rgb=img, garment_mask=mask,
            reference_spec=_spec((120, 20, 50)), face_bbox=FACE_BBOX)
        self.assertEqual(res["checks"]["collar_color"]["status"], "FAIL")
        self.assertTrue(any("incorrect_collar_color" in d for d in res["defects"]), res["defects"])

    def test_unmeasurable_collar_is_uncertain_not_an_unmeasured_pass(self):
        img, mask = _candidate_with_collar((30, 140, 70))
        spec = _spec((30, 140, 70))
        spec["collar_lab"] = None          # analyzer could not measure it
        res = uniform_fidelity_validator.validate_fidelity(
            candidate_rgb=img, garment_mask=mask, reference_spec=spec, face_bbox=FACE_BBOX)
        self.assertEqual(res["checks"]["collar_color"]["status"], "UNCERTAIN",
                         "an unmeasurable attribute must not be reported as PASS")

    def test_legacy_red_spec_still_works(self):
        """Specs produced before collar_lab existed must keep behaving."""
        img, mask = _candidate_with_collar((210, 30, 40))
        spec = {"ref_type": "red_collar", "collar_color": "red", "has_red_collar": True,
                "has_tie": False, "has_vest": False, "fabric_pattern": "plain"}
        res = uniform_fidelity_validator.validate_fidelity(
            candidate_rgb=img, garment_mask=mask, reference_spec=spec, face_bbox=FACE_BBOX)
        self.assertEqual(res["checks"]["collar_color"]["status"], "PASS",
                         res["checks"]["collar_color"]["details"])

    def test_shading_does_not_break_a_correct_colour_match(self):
        """Fabric shading and white balance must not read as the wrong garment."""
        img, mask = _candidate_with_collar((30, 140, 70))
        shaded = img.astype(np.float32)
        shaded[445:520, 280:400] *= 0.86      # one side of the collar in shadow
        img = np.clip(shaded, 0, 255).astype(np.uint8)
        res = uniform_fidelity_validator.validate_fidelity(
            candidate_rgb=img, garment_mask=mask,
            reference_spec=_spec((30, 140, 70)), face_bbox=FACE_BBOX)
        self.assertEqual(res["checks"]["collar_color"]["status"], "PASS",
                         res["checks"]["collar_color"]["details"])


if __name__ == "__main__":
    unittest.main()
