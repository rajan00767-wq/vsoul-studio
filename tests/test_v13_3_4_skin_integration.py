"""Unit tests for V13.3.4 Continuous Anatomical Skin Integration and Compositing Stabilization.

Tests:
1. integrate_continuous_skin: continuous skin topology, missing skin detection, smooth inpainting.
2. jaw_to_neck_continuity quality gate: detects gaps in chin-to-collar corridor.
3. original_garment_residue quality gate: detects residue of original garment inside collar opening.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pipelines.uniform_swap.garment_registrar import RegisteredGarment, garment_registrar
from pipelines.uniform_swap.quality_validator import (
    _measure_jaw_to_neck_continuity,
    _measure_original_garment_residue,
    quality_validator,
)


def test_integrate_continuous_skin_basic():
    """Test continuous skin generation and hole detection."""
    h, w = 400, 300
    src_rgb = np.full((h, w, 3), 180, dtype=np.uint8)
    src_person_matte = np.full((h, w), 0.95, dtype=np.float32)

    # Face mask in upper center
    face_m = np.zeros((h, w), dtype=bool)
    face_m[50:180, 100:200] = True

    # Neck mask below face
    neck_m = np.zeros((h, w), dtype=bool)
    neck_m[175:240, 120:180] = True

    # Ears mask
    ears_m = np.zeros((h, w), dtype=bool)

    # Original garment mask covering chest and lower neck
    orig_g_m = np.zeros((h, w), dtype=bool)
    orig_g_m[220:400, 50:250] = True

    # Registered target garment with collar opening
    reg_garment = RegisteredGarment(
        warped_rgb=np.zeros((h, w, 3), dtype=np.uint8),
        warped_alpha=np.full((h, w), 255, dtype=np.uint8),
        collar_lapels_mask=np.zeros((h, w), dtype=bool),
        tie_knot_mask=np.zeros((h, w), dtype=bool),
        collar_opening_mask=np.zeros((h, w), dtype=bool),
        body_mask=np.zeros((h, w), dtype=bool),
        registration_matrix=np.eye(2, 3),
        scale=1.0,
        dx=0.0,
        dy=0.0,
    )
    # Target lapels sit outside central neck opening
    reg_garment.collar_lapels_mask[220:300, :110] = True
    reg_garment.collar_lapels_mask[220:300, 190:] = True
    reg_garment.body_mask[280:400, :] = True

    skin_rgb, skin_alpha, opening_m, missing_m, recon_skin = garment_registrar.integrate_continuous_skin(
        src_rgb=src_rgb,
        src_person_matte=src_person_matte,
        face_mask=face_m,
        neck_mask=neck_m,
        ears_mask=ears_m,
        orig_garment_mask=orig_g_m,
        registered_garment=reg_garment,
        raw_registered_rgb=src_rgb,
        cand_labels_warped=None,
        src_chin_y=180,
        src_fh=130.0,
        src_fcx=150.0,
    )

    assert skin_rgb.shape == (h, w, 3)
    assert skin_alpha.shape == (h, w)
    # Face region should have full opacity
    assert np.all(skin_alpha[80:150, 120:180] == 255)
    # Missing throat skin under original garment should be detected
    assert np.any(missing_m)
    # Skin alpha in reconstructed throat area should be positive
    assert np.any(skin_alpha[missing_m] > 0)


def test_jaw_to_neck_continuity_direct():
    """Test direct measurement of throat alpha continuity."""
    h, w = 1000, 800
    c_chin_y = 500
    c_fh = 350.0
    c_fcx = 400.0
    c_fw = 350.0

    # Case 1: Gap in throat corridor (alpha = 0)
    alpha_gap = np.full((h, w), 255, dtype=np.uint8)
    alpha_gap[500:530, 380:420] = 0
    res_gap = _measure_jaw_to_neck_continuity(alpha_gap, c_chin_y, c_fh, c_fcx, c_fw, h, w)
    assert res_gap["status"] == "FAIL"
    assert res_gap["score"] < 220

    # Case 2: Solid throat skin (alpha = 255)
    alpha_solid = np.full((h, w), 255, dtype=np.uint8)
    res_solid = _measure_jaw_to_neck_continuity(alpha_solid, c_chin_y, c_fh, c_fcx, c_fw, h, w)
    assert res_solid["status"] == "PASS"
    assert res_solid["score"] == 255


def test_original_garment_residue_direct():
    """Test direct measurement of original garment color residue."""
    h, w = 1000, 800
    c_chin_y = 500
    c_fh = 350.0
    c_fcx = 400.0
    c_fw = 350.0

    # Source image has a distinct green shirt: RGB(30, 160, 40)
    src_img = np.full((h, w, 3), 200, dtype=np.uint8)
    orig_g_mask = np.zeros((h, w), dtype=bool)
    orig_g_mask[520:900, 150:650] = True
    src_img[orig_g_mask] = [30, 160, 40]

    # Case 1: Residue present inside collar opening (green pixels below chin)
    cand_dirty = np.full((h, w, 3), 240, dtype=np.uint8)
    cand_dirty[510:560, 360:440] = [30, 160, 40]  # ~4000 green pixels in collar opening
    res_dirty = _measure_original_garment_residue(cand_dirty, src_img, orig_g_mask, c_chin_y, c_fh, c_fcx, c_fw, h, w)
    assert res_dirty["status"] == "FAIL"
    assert res_dirty["score"] > 150

    # Case 2: Clean candidate (skin tone inside collar opening)
    cand_clean = np.full((h, w, 3), 240, dtype=np.uint8)
    cand_clean[510:560, 360:440] = [220, 180, 150]
    res_clean = _measure_original_garment_residue(cand_clean, src_img, orig_g_mask, c_chin_y, c_fh, c_fcx, c_fw, h, w)
    assert res_clean["status"] == "PASS"
    assert res_clean["score"] == 0
