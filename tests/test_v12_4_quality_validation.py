"""Unit & Regression Tests for V12.4 Negative Validation Cases and Mandatory Check Gating (Section 9).

Verifies:
1. Deliberately removed bindi returns FAIL on bindi_preservation.
2. Missing bow (erasing left bow) returns FAIL on bow_and_ribbon_preservation.
3. Added braids on short hair returns FAIL on added_hair.
4. White buttons instead of black returns FAIL on button_color.
5. Smoothed skin texture (blurred face) returns FAIL on face_texture_preservation.
6. Mandatory check gating: any FAIL or UNCERTAIN sets passed = False.
"""

from unittest.mock import patch
import numpy as np
import pytest
import cv2
from PIL import Image

from pipelines.uniform_swap.quality_validator import QualityValidator


class MockFace:
    def __init__(self, bbox=(250, 200, 550, 500)):
        self.bbox = np.array(bbox, dtype=np.float32)
        emb = np.ones(512, dtype=np.float32)
        self.embedding = emb / np.linalg.norm(emb)
        self.kps = np.array([[300, 300], [500, 300], [400, 380], [330, 440], [470, 440]], dtype=np.float32)


class MockInsightApp:
    def get(self, img):
        return [MockFace()]


def _create_mock_portrait(h=1024, w=800):
    """Generate a synthetic test portrait with facial noise, hair, and torso."""
    img = np.full((h, w, 3), 220, dtype=np.uint8)
    
    # Face region [200:500, 250:550]
    face_m = np.zeros((h, w), dtype=bool)
    face_m[200:500, 250:550] = True
    
    # Add high-frequency facial texture (skin texture)
    np.random.seed(42)
    noise = np.random.randint(-40, 40, size=(300, 300, 3), dtype=np.int16)
    face_tex = np.clip(180 + noise, 0, 255).astype(np.uint8)
    img[200:500, 250:550] = face_tex
    
    # Hair region [100:350, 220:580]
    hair_m = np.zeros((h, w), dtype=bool)
    hair_m[100:350, 220:580] = True
    img[hair_m] = [20, 15, 15]
    
    return img, face_m, hair_m


@patch("pipelines.photo_restoration._get_insight_app", return_value=MockInsightApp())
def test_v12_4_removed_bindi_fails_validation(mock_insight):
    """Negative Test: A portrait with source bindi that is erased in candidate returns FAIL on bindi_preservation."""
    qv = QualityValidator()
    h, w = 1024, 800
    src_img, face_m, hair_m = _create_mock_portrait(h, w)
    cand_img = src_img.copy()

    # In source, draw a distinct red bindi between eyebrows:
    # cand_box: y1=200, fh=300 -> b_y1 = 200 + 300*0.38 = 314, b_y2 = 356. fcx=400, fw=300 -> b_x1=376, b_x2=424
    src_img[325:345, 395:405] = [200, 20, 20]
    # In candidate, bindi is removed (replaced with skin color)
    cand_img[325:345, 395:405] = [180, 150, 140]

    id_layer = {
        "original_face_mask": face_m,
        "original_hair_mask": hair_m,
        "original_accessory_mask": np.zeros((h, w), dtype=bool),
        "has_bindi": True,
    }

    res = qv.validate_candidate(cand_img, src_img, identity_layer=id_layer)
    assert res["checks"]["bindi_preservation"]["status"] == "FAIL"
    assert any("missing_original_bindi" in d for d in res["defects"])
    assert res["passed"] is False


@patch("pipelines.photo_restoration._get_insight_app", return_value=MockInsightApp())
def test_v12_4_missing_bow_fails_validation(mock_insight):
    """Negative Test: Erasing one of two source bows returns FAIL on bow_and_ribbon_preservation."""
    qv = QualityValidator()
    h, w = 1024, 800
    src_img, face_m, hair_m = _create_mock_portrait(h, w)
    cand_img = src_img.copy()

    # Left bow [250:350, 180:230], Right bow [250:350, 570:620]
    l_mask = np.zeros((h, w), dtype=bool)
    l_mask[250:350, 180:230] = True
    r_mask = np.zeros((h, w), dtype=bool)
    r_mask[250:350, 570:620] = True

    src_img[l_mask] = [30, 90, 220]  # vibrant blue
    src_img[r_mask] = [30, 90, 220]

    # Candidate has right bow, but left bow is missing (erased to white/garment)
    cand_img[r_mask] = [30, 90, 220]
    cand_img[l_mask] = [200, 200, 200]

    ind_bows = [
        {"name": "left_bow", "mask": l_mask, "area": int(np.sum(l_mask)), "median_rgb": (30, 90, 220)},
        {"name": "right_bow", "mask": r_mask, "area": int(np.sum(r_mask)), "median_rgb": (30, 90, 220)},
    ]

    id_layer = {
        "original_face_mask": face_m,
        "original_hair_mask": hair_m,
        "original_accessory_mask": (l_mask | r_mask),
        "individual_bows": ind_bows,
    }

    res = qv.validate_candidate(cand_img, src_img, identity_layer=id_layer)
    assert res["checks"]["bow_and_ribbon_preservation"]["status"] == "FAIL"
    assert any("missing_left_bow" in d for d in res["defects"])
    assert res["passed"] is False


@patch("pipelines.photo_restoration._get_insight_app", return_value=MockInsightApp())
def test_v12_4_added_braids_on_short_hair_fails_validation(mock_insight):
    """Negative Test: Adding long hair or braids to a short-haired portrait returns FAIL on added_hair."""
    qv = QualityValidator()
    h, w = 1024, 800
    src_img, face_m, hair_m = _create_mock_portrait(h, w)
    cand_img = src_img.copy()

    # Add dark hair pixels near neck/shoulders [400:600, 150:230]
    cand_img[400:600, 150:230] = [15, 10, 10]

    id_layer = {
        "original_face_mask": face_m,
        "original_hair_mask": hair_m,
        "original_accessory_mask": np.zeros((h, w), dtype=bool),
    }

    res = qv.validate_candidate(cand_img, src_img, identity_layer=id_layer)
    assert res["checks"]["added_hair"]["status"] == "FAIL"
    assert any("unwanted_braids_or_added_hair" in d for d in res["defects"])
    assert res["passed"] is False


@patch("pipelines.photo_restoration._get_insight_app", return_value=MockInsightApp())
def test_v12_4_white_buttons_fails_validation(mock_insight):
    """Negative Test: Rendering white buttons instead of black returns FAIL on button_color."""
    qv = QualityValidator()
    h, w = 1024, 800
    src_img, face_m, hair_m = _create_mock_portrait(h, w)
    cand_img = src_img.copy()

    # Candidate has white buttons down the placket:
    # cand_box: chin_y=500, fh=300 -> placket_y1 = 500 + 300*0.15 = 545, placket_y2 = 500 + 300*0.85 = 755
    # mid_x = 400 -> placket_box is [545:755, 370:430]
    cand_img[550:750, 390:410] = [245, 245, 245]

    id_layer = {
        "original_face_mask": face_m,
        "original_hair_mask": hair_m,
        "original_accessory_mask": np.zeros((h, w), dtype=bool),
    }

    res = qv.validate_candidate(cand_img, src_img, identity_layer=id_layer)
    assert res["checks"]["button_color"]["status"] == "FAIL"
    assert any("incorrect_button_color" in d for d in res["defects"])
    assert res["passed"] is False


@patch("pipelines.photo_restoration._get_insight_app", return_value=MockInsightApp())
def test_v12_4_smoothed_skin_texture_fails_validation(mock_insight):
    """Negative Test: Heavily blurred/smoothed face skin returns FAIL on face_texture_preservation."""
    qv = QualityValidator()
    h, w = 1024, 800
    src_img, face_m, hair_m = _create_mock_portrait(h, w)
    cand_img = src_img.copy()

    # Heavily blur candidate face region (simulating over-smoothed diffusion)
    cand_img[200:500, 250:550] = cv2.GaussianBlur(cand_img[200:500, 250:550], (31, 31), 0)

    id_layer = {
        "original_face_mask": face_m,
        "original_hair_mask": hair_m,
        "original_accessory_mask": np.zeros((h, w), dtype=bool),
    }

    res = qv.validate_candidate(cand_img, src_img, identity_layer=id_layer)
    assert res["checks"]["face_texture_preservation"]["status"] == "FAIL"
    assert any("excessive_skin_smoothing" in d for d in res["defects"])
    assert res["passed"] is False


@patch("pipelines.photo_restoration._get_insight_app", return_value=MockInsightApp())
def test_v12_4_mandatory_check_gating(mock_insight):
    """Verification: Any mandatory check with status FAIL or UNCERTAIN causes passed=False and needs_review=True."""
    qv = QualityValidator()
    h, w = 1024, 800
    src_img, face_m, hair_m = _create_mock_portrait(h, w)
    cand_img = src_img.copy()

    # Candidate has slight hair thinning (missing_ratio ~ 0.30 -> UNCERTAIN)
    # Erase 30% of hair pixels in candidate
    hair_y, hair_x = np.where(hair_m)
    num_to_erase = int(0.32 * len(hair_y))
    cand_img[hair_y[:num_to_erase], hair_x[:num_to_erase]] = [220, 220, 220]

    id_layer = {
        "original_face_mask": face_m,
        "original_hair_mask": hair_m,
        "original_accessory_mask": np.zeros((h, w), dtype=bool),
    }

    res = qv.validate_candidate(cand_img, src_img, identity_layer=id_layer)
    # Missing hair status is UNCERTAIN (0.25 < score <= 0.40)
    assert res["checks"]["missing_hair"]["status"] in ("UNCERTAIN", "FAIL")
    # Gating policy must mark passed=False and needs_review=True
    assert res["passed"] is False
    assert res["needs_review"] is True
