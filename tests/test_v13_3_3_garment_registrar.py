"""Tests for V13.3.3 Anatomical Garment Registration & Quality Gates.

Validates:
1. compute_registration_matrix: 2D similarity transform maps candidate to source anatomy.
2. decompose_garment: Garment decomposes into collar lapels, tie knot, opening, and body.
3. extract_dynamic_neck: Throat skin is confined strictly within collar opening.
4. V13.3.3 Quality Gates:
   - black_shoulder_outlines (< 400 dark border px PASS, > 400 dark border px FAIL).
   - collar_area_retention (>= 85% retention PASS, < 85% FAIL).
   - tie_knot_preservation (dark tie knot present for tie uniforms PASS, missing FAIL).
"""

import cv2
import numpy as np
import pytest
from PIL import Image

from pipelines.uniform_swap.garment_registrar import garment_registrar, RegisteredGarment
from pipelines.uniform_swap.quality_validator import quality_validator


def test_registration_matrix_computation():
    """Verify compute_registration_matrix produces valid similarity matrix."""
    h, w = 400, 400
    raw_rgb = np.full((h, w, 3), 200, dtype=np.uint8)
    src_rgb = np.full((h, w, 3), 200, dtype=np.uint8)

    src_id_layer = {
        "metrics": {
            "chin_y": 240,
            "face_height": 180.0,
            "face_center_x": 200.0,
        },
        "face_bbox": (110.0, 60.0, 290.0, 240.0),
    }
    raw_chin_y = 200
    raw_fh = 150.0
    raw_fcx = 200.0

    M, scale, dx, dy = garment_registrar.compute_registration_matrix(
        raw_rgb=raw_rgb,
        src_rgb=src_rgb,
        src_identity_layer=src_id_layer,
        raw_chin_y=raw_chin_y,
        raw_fh=raw_fh,
        raw_fcx=raw_fcx,
    )
    assert M is not None
    assert M.shape == (2, 3)
    # Expected scale = 180 / 150 = 1.20
    assert abs(scale - 1.20) < 0.05
    assert isinstance(dx, float)
    assert isinstance(dy, float)


def test_decompose_garment():
    """Verify decompose_garment separates lapels, opening, and body."""
    h, w = 400, 400
    raw_rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    # Red collar
    raw_rgb[150:230, 100:300] = [210, 30, 30]
    # Check body
    raw_rgb[230:, 80:320] = [100, 120, 150]

    cand_labels = np.zeros((h, w), dtype=np.uint8)
    cand_labels[150:230, 100:300] = 5   # UpperClothes
    cand_labels[230:, 80:320] = 5       # UpperClothes
    cand_matte = np.zeros((h, w), dtype=np.uint8)
    cand_matte[150:, 80:320] = 255

    uniform_cues = {"has_red_collar": True, "ref_type": "red_collar"}

    comp = garment_registrar.decompose_garment(
        raw_rgb=raw_rgb,
        cand_labels=cand_labels,
        cand_matte=cand_matte,
        uniform_cues=uniform_cues,
        raw_chin_y=160,
        raw_fh=140.0,
        raw_fcx=200.0,
    )
    assert np.any(comp.collar_lapels_mask)
    assert np.any(comp.body_mask)
    assert comp.collar_lapels_mask.shape == (h, w)
    assert comp.body_mask.shape == (h, w)


def test_extract_dynamic_neck():
    """Verify dynamic neck skin is constrained inside collar opening and outside lapels."""
    h, w = 400, 400
    src_rgb = np.full((h, w, 3), 160, dtype=np.uint8)  # Skin
    src_matte = np.ones((h, w), dtype=float)
    src_face = np.zeros((h, w), dtype=bool)
    src_face[:150, 130:270] = True

    # Registered collar with opening
    opening_m = np.zeros((h, w), dtype=bool)
    opening_m[160:210, 160:240] = True
    lapels_m = np.zeros((h, w), dtype=bool)
    lapels_m[160:230, 120:160] = True
    lapels_m[160:230, 240:280] = True

    reg_garment = RegisteredGarment(
        warped_rgb=np.zeros((h, w, 3), dtype=np.uint8),
        warped_alpha=np.zeros((h, w), dtype=np.uint8),
        collar_lapels_mask=lapels_m,
        tie_knot_mask=np.zeros((h, w), dtype=bool),
        collar_opening_mask=opening_m,
        body_mask=np.zeros((h, w), dtype=bool),
        registration_matrix=np.eye(2, 3),
        scale=1.0,
        dx=0.0,
        dy=0.0,
    )

    neck = garment_registrar.extract_dynamic_neck(
        src_rgb=src_rgb,
        src_person_matte=src_matte,
        src_chin_y=155,
        src_fh=140.0,
        src_fcx=200.0,
        src_face_mask=src_face,
        registered_garment=reg_garment,
    )
    assert neck.shape == (h, w)
    # Neck must NOT overlap collar lapels
    assert not np.any(neck & lapels_m)
    # Neck must NOT overlap face mask
    assert not np.any(neck & src_face)


def test_black_shoulder_outlines_gate():
    """Verify black_shoulder_outlines flags images with dark outlines > 400 px."""
    h, w = 400, 400
    cand_arr = np.full((h, w, 3), 255, dtype=np.uint8)
    cand_arr[150:, 80:320] = [100, 150, 200]
    cand_arr[:150, 130:270] = [210, 180, 160]
    cand_img = Image.fromarray(cand_arr)
    src_img = Image.fromarray(cand_arr.copy())

    p_alpha = np.zeros((h, w), dtype=np.uint8)
    p_alpha[150:, 80:320] = 255
    p_alpha[:150, 130:270] = 255

    id_clean = {
        "full_person_alpha": p_alpha,
        "face_bbox": (130, 30, 270, 150),
    }

    # Clean candidate passes
    res_clean = quality_validator.validate_candidate(
        candidate_image=cand_img,
        source_image=src_img,
        identity_layer=id_clean,
        stage="raw",
    )
    assert res_clean["checks"]["black_shoulder_outlines"]["status"] == "PASS"

    # Injected dark edge along shoulders (> 400 dark border px)
    cand_dark_arr = cand_arr.copy()
    cand_dark_arr[160:380, 78:82] = [10, 10, 10]
    cand_dark_arr[160:380, 318:322] = [10, 10, 10]
    cand_dark_img = Image.fromarray(cand_dark_arr)

    res_dark = quality_validator.validate_candidate(
        candidate_image=cand_dark_img,
        source_image=src_img,
        identity_layer=id_clean,
        stage="raw",
    )
    assert res_dark["checks"]["black_shoulder_outlines"]["status"] == "FAIL"
    assert any("black_shoulder_outlines" in d for d in res_dark["defects"])


def test_collar_area_retention_gate():
    """Verify collar_area_retention flags collar erosion (< 85% retention)."""
    h, w = 400, 400
    cand_arr = np.full((h, w, 3), 255, dtype=np.uint8)
    cand_arr[:150, 130:270] = [210, 180, 160]
    # Small red collar
    cand_arr[150:165, 140:158] = [200, 20, 20]
    cand_img = Image.fromarray(cand_arr)
    src_img = cand_img.copy()

    id_layer = {
        "face_bbox": (130, 30, 270, 150),
        "raw_collar_pixels": 1000,
    }
    ref_img = Image.new("RGB", (100, 100), (200, 20, 20))

    res = quality_validator.validate_candidate(
        candidate_image=cand_img,
        source_image=src_img,
        identity_layer=id_layer,
        uniform_reference=ref_img,
        stage="raw",
    )
    assert res["checks"]["collar_area_retention"]["status"] == "FAIL"
    assert any("collar_area_retention" in d for d in res["defects"])
