"""Verification tests for the 12 Studio Quality Gates (V13.2 Section 18).

Tests both:
1. Positive clean baseline conditions (all 12 studio gates PASS)
2. Negative defect injections (each studio gate detects its injected defect and triggers FAIL)
"""

import sys
from pathlib import Path
import unittest
import numpy as np
from PIL import Image, ImageDraw
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.uniform_swap.quality_validator import quality_validator
from scripts.verify_quality_gates import load_baseline


class TestStudioQualityGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src_img, cls.cand_img, cls.id_layer, cls.uniform_ref = load_baseline()

    def test_00_clean_baseline_passes_all_studio_gates(self):
        """Clean baseline portrait must PASS all 12 studio gates with zero defects."""
        res = quality_validator.validate_candidate(
            candidate_image=self.cand_img,
            source_image=self.src_img,
            identity_layer=self.id_layer,
            uniform_reference=self.uniform_ref,
            job_id="test_clean_studio",
        )
        studio_gates = [
            "excessive_head_roll",
            "excessive_head_yaw",
            "excessive_head_pitch",
            "off_camera_gaze",
            "unnatural_eye_alignment",
            "asymmetric_shoulders",
            "pose_correction_identity_drift",
            "facial_geometry_distortion",
            "unnatural_neck_alignment",
            "face_neck_lighting_mismatch",
            "unresolved_harsh_sunlight",
            "original_pose_reintroduced",
        ]
        for gate in studio_gates:
            self.assertIn(gate, res["checks"], f"Missing gate key: {gate}")
            status = res["checks"][gate]["status"]
            self.assertEqual(status, "PASS", f"Gate {gate} failed on clean baseline: {res['checks'][gate]}")

    def test_01_excessive_head_roll(self):
        """Defect injection: Rotate candidate portrait by 18 degrees."""
        w, h = self.cand_img.size
        # Rotate candidate image by 18 degrees around center
        rotated = self.cand_img.rotate(18, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
        res = quality_validator.validate_candidate(
            candidate_image=rotated,
            source_image=self.src_img,
            identity_layer=self.id_layer,
            uniform_reference=self.uniform_ref,
        )
        self.assertEqual(res["checks"]["excessive_head_roll"]["status"], "FAIL")
        self.assertFalse(res["passed"])

    def test_02_excessive_head_yaw(self):
        """Defect injection: Simulated extreme yaw > 20 deg."""
        id_layer = dict(self.id_layer)
        id_layer["pose_analysis"] = {"head_yaw": 24.5, "head_roll": 0.0, "head_pitch": 0.0}
        res = quality_validator.validate_candidate(
            candidate_image=self.cand_img,
            source_image=self.src_img,
            identity_layer=id_layer,
            uniform_reference=self.uniform_ref,
        )
        self.assertEqual(res["checks"]["excessive_head_yaw"]["status"], "FAIL")
        self.assertFalse(res["passed"])

    def test_03_excessive_head_pitch(self):
        """Defect injection: Simulated extreme pitch > 20 deg."""
        id_layer = dict(self.id_layer)
        id_layer["pose_analysis"] = {"head_pitch": 23.5}
        res = quality_validator.validate_candidate(
            candidate_image=self.cand_img,
            source_image=self.src_img,
            identity_layer=id_layer,
            uniform_reference=self.uniform_ref,
        )
        self.assertEqual(res["checks"]["excessive_head_pitch"]["status"], "FAIL")
        self.assertFalse(res["passed"])

    def test_04_off_camera_gaze(self):
        """Defect injection: Extreme gaze deviation > 20 deg."""
        id_layer = dict(self.id_layer)
        id_layer["pose_analysis"] = {"gaze_deviation_deg": 25.0}
        res = quality_validator.validate_candidate(
            candidate_image=self.cand_img,
            source_image=self.src_img,
            identity_layer=id_layer,
            uniform_reference=self.uniform_ref,
        )
        self.assertEqual(res["checks"]["off_camera_gaze"]["status"], "FAIL")
        self.assertFalse(res["passed"])

    def test_05_unnatural_eye_alignment(self):
        """Defect injection: Vertical eye alignment delta > 0.040."""
        id_layer = dict(self.id_layer)
        id_layer["pose_analysis"] = {"eye_alignment_delta": 0.052}
        res = quality_validator.validate_candidate(
            candidate_image=self.cand_img,
            source_image=self.src_img,
            identity_layer=id_layer,
            uniform_reference=self.uniform_ref,
        )
        self.assertEqual(res["checks"]["unnatural_eye_alignment"]["status"], "FAIL")
        self.assertFalse(res["passed"])

    def test_06_asymmetric_shoulders(self):
        """Defect injection: Extreme shoulder tilt > 12 deg."""
        id_layer = dict(self.id_layer)
        id_layer["pose_analysis"] = {"shoulder_tilt_deg": 14.5}
        res = quality_validator.validate_candidate(
            candidate_image=self.cand_img,
            source_image=self.src_img,
            identity_layer=id_layer,
            uniform_reference=self.uniform_ref,
        )
        self.assertEqual(res["checks"]["asymmetric_shoulders"]["status"], "FAIL")
        self.assertFalse(res["passed"])

    def test_07_pose_correction_identity_drift(self):
        """Defect injection: Candidate face replaced with dissimilar face (ArcFace similarity < 0.90)."""
        w, h = self.cand_img.size
        # Create unrelated candidate
        unrelated_arr = np.array(self.cand_img)
        # Heavy blur in face zone to reduce similarity below 0.90
        unrelated_arr[250:550, 250:550] = cv2.GaussianBlur(unrelated_arr[250:550, 250:550], (51, 51), 30)
        res = quality_validator.validate_candidate(
            candidate_image=Image.fromarray(unrelated_arr),
            source_image=self.src_img,
            identity_layer=self.id_layer,
            uniform_reference=self.uniform_ref,
        )
        self.assertEqual(res["checks"]["pose_correction_identity_drift"]["status"], "FAIL")
        self.assertFalse(res["passed"])

    def test_08_facial_geometry_distortion(self):
        """Defect injection: Distort facial geometry using horizontal stretching (geometry RMSE > 0.040)."""
        arr = np.array(self.cand_img).copy()
        # Non-linear affine warp on facial region
        f_roi = arr[250:600, 200:600]
        h_f, w_f = f_roi.shape[:2]
        pts1 = np.float32([[0, 0], [w_f, 0], [0, h_f]])
        pts2 = np.float32([[25, 0], [w_f - 40, 30], [10, h_f]])
        M_warp = cv2.getAffineTransform(pts1, pts2)
        arr[250:600, 200:600] = cv2.warpAffine(f_roi, M_warp, (w_f, h_f), borderMode=cv2.BORDER_REFLECT)
        res = quality_validator.validate_candidate(
            candidate_image=Image.fromarray(arr),
            source_image=self.src_img,
            identity_layer=self.id_layer,
            uniform_reference=self.uniform_ref,
        )
        self.assertEqual(res["checks"]["facial_geometry_distortion"]["status"], "FAIL")
        self.assertFalse(res["passed"])

    def test_09_unnatural_neck_alignment(self):
        """Defect injection: Chin center displaced horizontally from facial vertical axis (> 0.15)."""
        id_layer = dict(self.id_layer)
        id_layer["pose_analysis"] = {"neck_alignment_offset": 0.22}
        res = quality_validator.validate_candidate(
            candidate_image=self.cand_img,
            source_image=self.src_img,
            identity_layer=id_layer,
            uniform_reference=self.uniform_ref,
        )
        self.assertEqual(res["checks"]["unnatural_neck_alignment"]["status"], "FAIL")
        self.assertFalse(res["passed"])

    def test_10_face_neck_lighting_mismatch(self):
        """Defect injection: Neck heavily darkened causing mean L difference > 35."""
        arr = np.array(self.cand_img).copy()
        w, h = self.cand_img.size
        # Darken throat zone from chin y=695 down to y=800
        arr[690:820, int(w*0.35):int(w*0.65)] = (arr[690:820, int(w*0.35):int(w*0.65)] * 0.25).astype(np.uint8)
        res = quality_validator.validate_candidate(
            candidate_image=Image.fromarray(arr),
            source_image=self.src_img,
            identity_layer=self.id_layer,
            uniform_reference=self.uniform_ref,
        )
        self.assertEqual(res["checks"]["face_neck_lighting_mismatch"]["status"], "FAIL")
        self.assertFalse(res["passed"])

    def test_11_unresolved_harsh_sunlight(self):
        """Defect injection: Blown-out specular highlights on face (L > 245 over 400 px)."""
        arr = np.array(self.cand_img).copy()
        # Draw pure white specular blowout circle on forehead / cheek
        cv2.circle(arr, (400, 420), 25, (255, 255, 255), -1)  # pi * 25^2 ~ 1960 px
        res = quality_validator.validate_candidate(
            candidate_image=Image.fromarray(arr),
            source_image=self.src_img,
            identity_layer=self.id_layer,
            uniform_reference=self.uniform_ref,
        )
        self.assertEqual(res["checks"]["unresolved_harsh_sunlight"]["status"], "FAIL")
        self.assertFalse(res["passed"])

    def test_12_original_pose_reintroduced(self):
        """Defect injection: Target roll was 0.0 (or -6 deg corrected), but candidate has roll error > 4.0 deg."""
        id_layer = dict(self.id_layer)
        # Set target roll after correction to 0.0, while setting pose_transform to show original roll was tilted
        id_layer["pose_transform"] = {"roll_after": 0.0}
        id_layer["target_pose_roll"] = 0.0
        # Rotate candidate by 6.5 degrees
        rotated = self.cand_img.rotate(6.5, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
        res = quality_validator.validate_candidate(
            candidate_image=rotated,
            source_image=self.src_img,
            identity_layer=id_layer,
            uniform_reference=self.uniform_ref,
        )
        self.assertEqual(res["checks"]["original_pose_reintroduced"]["status"], "FAIL")
        self.assertFalse(res["passed"])


if __name__ == "__main__":
    unittest.main()
