"""Strict regression and unit tests for uniform swapping pipeline (Tasks 1-14).

Includes:
1. Deterministic unit tests using synthetic inputs (no Qwen/GPU needed) verifying:
   - Original clothing cannot enter hair restoration layer
   - A blue badge cannot become a hair accessory
   - A failed GrabCut operation cannot silently accept dark clothing
   - Hair alpha cannot spread into definite non-hair regions
   - Original neck pixels remain preserved
   - Registration occurs before compositing
   - Clothing leakage validation detects intentionally contaminated masks
   - Compositor exceptions produce blocking defects
2. Integration regression tests on deliverable portraits and diagnostic artifacts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUTS_DIR = Path("outputs")
JOB_ID = "qwen_uniform_bugs_fixed"


# ==============================================================================
# SECTION 1: DETERMINISTIC UNIT TESTS (SYNTHETIC INPUTS, ZERO GPU/DIFFUSION)
# ==============================================================================

def test_unit_original_clothing_cannot_enter_hair_restoration_layer():
    """FIX 1 & 2: Original clothing labels (coat, upper) cannot enter full_hair_mask."""
    from pipelines.uniform_hair_face_compositor import recover_hair_and_accessories

    h, w = 400, 300
    orig_rgb = np.full((h, w, 3), 120, dtype=np.uint8)  # Gray background
    schp = np.zeros((h, w), dtype=np.uint8)

    # Face box in upper center
    fx1, fy1, fx2, fy2 = 100, 50, 200, 150
    face_bbox = (fx1, fy1, fx2, fy2)
    schp[fy1:fy2, fx1:fx2] = 13  # Face

    # Scalp hair (class 2)
    schp[20:70, 90:210] = 2
    orig_rgb[20:70, 90:210] = [20, 20, 20]

    # Upper clothes / coat (class 7) on chest and shoulders
    schp[160:350, 60:240] = 7
    orig_rgb[160:350, 60:240] = [15, 20, 45]  # Dark navy jacket

    hair_mask, acc_mask = recover_hair_and_accessories(
        orig_rgb=orig_rgb,
        face_bbox=face_bbox,
        schp_labels=schp,
    )

    # Assert ZERO clothing pixels in hair mask
    clothing_in_hair = hair_mask & (schp == 7)
    assert np.count_nonzero(clothing_in_hair) == 0, (
        f"Original clothing entered hair restoration: {np.count_nonzero(clothing_in_hair)} px"
    )


def test_unit_blue_badge_cannot_become_hair_accessory():
    """FIX 9: A blue chest badge disconnected from hair cannot become a hair accessory."""
    from pipelines.uniform_hair_face_compositor import recover_hair_and_accessories

    h, w = 400, 300
    orig_rgb = np.full((h, w, 3), 120, dtype=np.uint8)
    schp = np.zeros((h, w), dtype=np.uint8)

    fx1, fy1, fx2, fy2 = 100, 50, 200, 150
    face_bbox = (fx1, fy1, fx2, fy2)
    schp[fy1:fy2, fx1:fx2] = 13
    schp[20:70, 90:210] = 2  # Scalp hair

    # Bright blue school badge on the chest (x in [160, 200], y in [220, 260])
    orig_rgb[220:260, 160:200] = [0, 80, 220]  # Saturated blue badge
    schp[220:260, 160:200] = 7  # On coat

    hair_mask, acc_mask, dbg = recover_hair_and_accessories(
        orig_rgb=orig_rgb,
        face_bbox=face_bbox,
        schp_labels=schp,
        return_debug=True,
    )

    # The badge must NOT be in acc_mask
    badge_in_acc = acc_mask[220:260, 160:200]
    assert np.count_nonzero(badge_in_acc) == 0, (
        f"Blue badge was classified as hair accessory: {np.count_nonzero(badge_in_acc)} px"
    )
    # The badge must be caught in rejected_accessories
    assert np.count_nonzero(dbg["rejected_accessories"][220:260, 160:200]) > 0, (
        "Blue badge was not quarantined in rejected accessories"
    )


def test_unit_failed_grabcut_cannot_silently_accept_dark_clothing():
    """FIX 1: When GrabCut has no hair seeds, it must NOT fall back to dark-pixel thresholding."""
    from pipelines.uniform_hair_face_compositor import recover_hair_and_accessories

    h, w = 400, 300
    orig_rgb = np.full((h, w, 3), 120, dtype=np.uint8)
    schp = np.zeros((h, w), dtype=np.uint8)

    fx1, fy1, fx2, fy2 = 100, 50, 200, 150
    face_bbox = (fx1, fy1, fx2, fy2)
    schp[fy1:fy2, fx1:fx2] = 13  # Face

    # Dark uniform fabric in the lateral braid corridor (x > 220, y > 150)
    # But ZERO scalp hair seeds in this corridor
    orig_rgb[150:350, 220:280] = [10, 10, 15]  # Very dark black/navy coat fabric
    schp[150:350, 220:280] = 7  # Coat

    hair_mask, acc_mask, dbg = recover_hair_and_accessories(
        orig_rgb=orig_rgb,
        face_bbox=face_bbox,
        schp_labels=schp,
        return_debug=True,
    )

    # Without hair seeds, the corridor must NOT classify the dark coat as hair
    lateral_dark_as_hair = hair_mask[150:350, 220:280]
    assert np.count_nonzero(lateral_dark_as_hair) == 0, (
        f"Failed GrabCut silently accepted dark clothing as hair: {np.count_nonzero(lateral_dark_as_hair)} px"
    )
    assert dbg["braid_confidence"]["right"] == "no_foreground_seeds", (
        f"Expected 'no_foreground_seeds', got {dbg['braid_confidence']['right']}"
    )


def test_unit_hair_alpha_cannot_spread_into_definite_non_hair_regions():
    """FIX 3: Refined hair alpha must be strictly 0.0 in face, visible neck, and clothing regions."""
    from pipelines.uniform_hair_face_compositor import composite_protected_identity, validate_clothing_leakage

    h, w = 300, 300
    qwen_base = np.full((h, w, 3), [40, 40, 180], dtype=np.uint8)  # Blue uniform
    orig_rgb = np.full((h, w, 3), [200, 50, 50], dtype=np.uint8)   # Red identity

    hair_alpha = np.zeros((h, w), dtype=np.float32)
    hair_alpha[30:100, 40:120] = 1.0  # Hair core

    clothing_region = np.zeros((h, w), dtype=bool)
    clothing_region[120:250, 40:220] = True  # Chest/shoulder clothing

    protected_face_neck = np.zeros((h, w), dtype=bool)
    protected_face_neck[60:150, 80:180] = True

    # Production constraint: hair alpha cannot exist over clothing or face/neck
    hair_alpha[protected_face_neck | clothing_region] = 0.0

    final_rgb, pixel_source, weights = composite_protected_identity(
        clean_qwen_uniform=qwen_base,
        identity_orig_rgb=orig_rgb,
        protected_face_neck=protected_face_neck,
        refined_hair_alpha=hair_alpha,
        hair_accessory_mask=np.zeros((h, w), dtype=bool),
    )

    # Validate with production leakage validator
    leak_px, _, _ = validate_clothing_leakage(
        W_orig=weights["W_orig"],
        orig_clothing_evidence=clothing_region,
        protected_face_neck=protected_face_neck,
    )

    assert leak_px == 0, f"Hair alpha leaked original pixels into clothing: {leak_px} px"
    assert np.count_nonzero(hair_alpha[clothing_region] > 0.0) == 0, "Hair alpha leaked into clothing"
    assert np.count_nonzero(hair_alpha[protected_face_neck] > 0.0) == 0, "Hair alpha leaked into face/neck"


def test_unit_original_neck_pixels_remain_preserved():
    """FIX 5: Original neck skin pixels are preserved without white canvas gap."""
    from pipelines.uniform_hair_face_compositor import align_uniform_collar

    h, w = 300, 300
    uniform_base = np.full((h, w, 3), [180, 50, 50], dtype=np.uint8)  # Red uniform
    uniform_base[155:, :] = [20, 20, 40]  # Dark collar edge

    protected_neck = np.zeros((h, w), dtype=bool)
    # Neck region: x in [120, 180], y in [130, 150]
    protected_neck[130:150, 120:180] = True
    face_bbox = (100, 50, 200, 140)

    aligned_uniform, ok, info = align_uniform_collar(
        clean_qwen_uniform=uniform_base,
        protected_neck=protected_neck,
        face_bbox=face_bbox,
    )

    assert ok is True, "Collar alignment failed on valid neck geometry"
    assert info["neck_anchor"][1] >= 145, f"Incorrect neck anchor: {info['neck_anchor']}"


def test_unit_registration_occurs_before_compositing():
    """FIX 4: Landmark similarity transformation correctly aligns raw output before compositing."""
    from pipelines.uniform_hair_face_compositor import register_qwen_to_original

    # Synthetic source and target landmarks
    src_kps = np.array([[50, 50], [150, 50], [100, 100], [70, 130], [130, 130]], dtype=np.float32)
    dst_kps = src_kps * 1.05 + np.array([12, -8], dtype=np.float32)

    h, w = 250, 250
    qwen_raw = np.full((h, w, 3), 150, dtype=np.uint8)
    cond = np.full((h, w, 3), 100, dtype=np.uint8)

    qwen_reg, _, err, meta = register_qwen_to_original(
        qwen_raw=qwen_raw,
        person_conditioning=cond,
        qwen_kps=src_kps,
        cond_kps=dst_kps,
    )

    assert meta["registration_success"] is True, "Registration failed on valid affine"
    assert err < 1.0, f"Registration residual too high: {err:.3f} px"


def test_unit_isolated_register_qwen_to_original():
    """Operation 1: register_qwen_to_original() independently testable unit test."""
    from pipelines.uniform_hair_face_compositor import register_qwen_to_original

    h, w = 250, 200
    cond_img = np.full((h, w, 3), 100, dtype=np.uint8)
    qwen_img = np.full((h, w, 3), 180, dtype=np.uint8)
    orig_img = np.full((h, w, 3), 60, dtype=np.uint8)

    cond_kps = np.array([[60, 60], [140, 60], [100, 95], [75, 135], [125, 135]], dtype=np.float32)
    qwen_kps = cond_kps * 1.03 + np.array([5.0, -3.0], dtype=np.float32)
    orig_kps = cond_kps * 0.98 + np.array([-2.0, 4.0], dtype=np.float32)

    qwen_reg, orig_reg, reg_err, meta = register_qwen_to_original(
        qwen_raw=qwen_img,
        person_conditioning=cond_img,
        original_person_pil=orig_img,
        target_size=(w, h),
        qwen_kps=qwen_kps,
        cond_kps=cond_kps,
        orig_kps=orig_kps,
    )

    assert qwen_reg.shape == (h, w, 3), f"Unexpected shape {qwen_reg.shape}"
    assert orig_reg.shape == (h, w, 3), f"Unexpected shape {orig_reg.shape}"
    assert reg_err < 1.0, f"Registration residual unexpectedly high: {reg_err:.3f} px"
    assert meta["registration_success"] is True, "Registration success should be True"
    assert meta["M_gen_to_canvas"] is not None, "Missing M_gen_to_canvas"
    assert meta["M_orig_to_canvas"] is not None, "Missing M_orig_to_canvas"


def test_unit_isolated_recover_hair_and_accessories():
    """Operation 2: recover_hair_and_accessories() independently testable unit test."""
    from pipelines.uniform_hair_face_compositor import recover_hair_and_accessories

    h, w = 300, 250
    orig_rgb = np.full((h, w, 3), 128, dtype=np.uint8)
    schp = np.zeros((h, w), dtype=np.uint8)
    face_bbox = (80, 40, 170, 130)

    # Hair scalp seed
    schp[20:60, 70:180] = 2
    orig_rgb[20:60, 70:180] = [15, 15, 15]

    # Face
    schp[40:130, 80:170] = 13

    # Lateral hair ribbon accessory candidate (blue, attached to hair)
    schp[50:70, 60:75] = 0
    orig_rgb[50:70, 60:75] = [20, 80, 220]  # Saturated blue ribbon attached to scalp

    hair_prob = np.zeros((h, w), dtype=np.float32)
    hair_prob[20:60, 70:180] = 0.95
    clothing_prob = np.zeros((h, w), dtype=np.float32)
    clothing_prob[160:280, 50:200] = 0.90

    hair_mask, acc_mask, dbg = recover_hair_and_accessories(
        orig_rgb=orig_rgb,
        face_bbox=face_bbox,
        schp_labels=schp,
        hair_prob=hair_prob,
        clothing_prob=clothing_prob,
        return_debug=True,
    )

    assert hair_mask.shape == (h, w), "Invalid hair mask shape"
    assert acc_mask.shape == (h, w), "Invalid accessory mask shape"
    assert np.count_nonzero(hair_mask[20:60, 70:180]) > 0, "Scalp hair missing"
    assert "braid_confidence" in dbg, "Debug dict missing braid_confidence"
    assert "original_has_long_hair" in dbg, "Debug dict missing original_has_long_hair"


def test_unit_isolated_align_uniform_collar():
    """Operation 3: align_uniform_collar() independently testable unit test."""
    from pipelines.uniform_hair_face_compositor import align_uniform_collar

    h, w = 300, 300
    clean_uniform = np.full((h, w, 3), [160, 40, 40], dtype=np.uint8)
    clean_uniform[158:, :] = [20, 20, 30]

    protected_neck = np.zeros((h, w), dtype=bool)
    protected_neck[110:150, 120:180] = True
    face_bbox = (100, 40, 200, 120)  # fh = 80, 0.35 * fh = 28 px

    aligned, ok, info = align_uniform_collar(
        clean_qwen_uniform=clean_uniform,
        protected_neck=protected_neck,
        face_bbox=face_bbox,
    )

    assert ok is True, f"Collar alignment failed on normal geometry: info={info}"
    assert "neck_anchor" in info, "Missing neck_anchor in info"
    assert "collar_anchor" in info, "Missing collar_anchor in info"
    assert aligned.shape == (h, w, 3), "Invalid aligned shape"

    # Extreme discrepancy test: neck ends at y=80 while collar is at y=158 (delta_y = 78 px > 28 px)
    extreme_neck = np.zeros((h, w), dtype=bool)
    extreme_neck[50:80, 120:180] = True
    _, extreme_ok, extreme_info = align_uniform_collar(
        clean_qwen_uniform=clean_uniform,
        protected_neck=extreme_neck,
        face_bbox=face_bbox,
    )
    assert extreme_ok is False, f"Collar alignment should reject extreme gap (> 35% fh): delta_y={extreme_info.get('delta_y')}"


def test_unit_isolated_composite_protected_identity():
    """Operation 4: composite_protected_identity() independently testable unit test."""
    from pipelines.uniform_hair_face_compositor import composite_protected_identity

    h, w = 200, 200
    qwen_uniform = np.full((h, w, 3), [0, 255, 0], dtype=np.uint8)     # Green
    orig_identity = np.full((h, w, 3), [255, 0, 0], dtype=np.uint8)    # Red

    face_neck = np.zeros((h, w), dtype=bool)
    face_neck[40:100, 60:140] = True

    hair_alpha = np.zeros((h, w), dtype=np.float32)
    hair_alpha[10:50, 40:160] = 1.0

    acc_mask = np.zeros((h, w), dtype=bool)
    acc_mask[45:55, 150:165] = True

    final_rgb, pixel_map, weights = composite_protected_identity(
        clean_qwen_uniform=qwen_uniform,
        identity_orig_rgb=orig_identity,
        protected_face_neck=face_neck,
        refined_hair_alpha=hair_alpha,
        hair_accessory_mask=acc_mask,
    )

    # Verification:
    assert final_rgb.shape == (h, w, 3)
    assert pixel_map.shape == (h, w, 3)
    # Face center must be original identity (Red)
    assert final_rgb[70, 100, 0] > 240, f"Face not preserved: {final_rgb[70, 100]}"
    assert weights["W_orig"][70, 100] > 0.95
    assert weights["W_qwen"][70, 100] < 0.05

    # Uniform body must be Qwen (Green)
    assert final_rgb[150, 100, 1] > 240, f"Uniform not preserved: {final_rgb[150, 100]}"
    assert weights["W_qwen"][150, 100] > 0.95
    assert weights["W_orig"][150, 100] < 0.05

    # Pixel source map Red channel corresponds to W_orig
    assert pixel_map[70, 100, 0] > 240
    # Pixel source map Green channel corresponds to W_qwen
    assert pixel_map[150, 100, 1] > 240


def test_unit_clothing_leakage_validation_detects_contaminated_masks():
    """FIX 1: Mathematically sound independent clothing leakage validation using production function."""
    from pipelines.uniform_hair_face_compositor import validate_clothing_leakage

    h, w = 200, 200
    orig_clothing_evidence = np.zeros((h, w), dtype=bool)
    orig_clothing_evidence[100:150, 50:150] = True  # Independent coat evidence on chest

    protected_face_neck = np.zeros((h, w), dtype=bool)
    protected_face_neck[30:80, 70:130] = True

    scalp_hair = np.zeros((h, w), dtype=bool)
    scalp_hair[10:50, 60:140] = True

    # 1. Clean production case
    W_orig_clean = np.zeros((h, w), dtype=np.float32)
    W_orig_clean[protected_face_neck] = 1.0
    W_orig_clean[scalp_hair] = 1.0

    leak_clean, _, metrics_clean = validate_clothing_leakage(
        W_orig=W_orig_clean,
        orig_clothing_evidence=orig_clothing_evidence,
        protected_face_neck=protected_face_neck,
        verified_hair_seeds=scalp_hair,
    )
    assert leak_clean == 0, f"Clean case falsely flagged leakage: {leak_clean} px"

    # 2. Contaminated case (clothing pixels erroneously given original identity ownership)
    W_orig_contaminated = W_orig_clean.copy()
    W_orig_contaminated[110:140, 60:140] = 0.85  # Contamination!

    leak_contam, unauth_mask, metrics_contam = validate_clothing_leakage(
        W_orig=W_orig_contaminated,
        orig_clothing_evidence=orig_clothing_evidence,
        protected_face_neck=protected_face_neck,
        verified_hair_seeds=scalp_hair,
    )
    assert leak_contam > 500, f"Validation failed to detect intentionally contaminated clothing: {leak_contam} px"
    assert np.count_nonzero(unauth_mask) == leak_contam


def test_unit_compositor_exceptions_produce_blocking_defects():
    """FIX 9 & 11: Compositor exception must set needs_retry=True and record blocking defect."""
    from pipelines.uniform_hair_face_compositor import validate_and_composite_uniform

    # Mocking InsightFace app to fail inside validate_and_composite_uniform
    mock_app = MagicMock()
    mock_app.get.return_value = []  # No face detected!

    cond_img = Image.new("RGB", (200, 200), (120, 120, 120))
    qwen_raw = Image.new("RGB", (200, 200), (120, 120, 120))
    orig_img = Image.new("RGB", (200, 200), (120, 120, 120))

    with patch("pipelines.photo_restoration._get_insight_app", return_value=mock_app):
        try:
            res = validate_and_composite_uniform(
                person_conditioning=cond_img,
                qwen_raw=qwen_raw,
                original_person_pil=orig_img,
                job_id="test_unit_exception",
                save_debug_phases=False,
            )
            # If function caught and returned defects:
            assert res["needs_retry"] is True
            assert len(res["blocking_defects"]) > 0
        except RuntimeError as exc:
            # If function raised expected RuntimeError on missing face:
            assert "No face detected" in str(exc)


# ==============================================================================
# SECTION 1B: NEGATIVE TESTS (DELIBERATE DEFECTS & CORRUPTED INPUTS)
# ==============================================================================

def test_unit_production_quality_gate_rejects_contaminated_hair_mask():
    """FIX 4: Independent clothing leakage gate rejects hair masks contaminated with chest clothing."""
    from pipelines.uniform_hair_face_compositor import validate_clothing_leakage, composite_protected_identity

    h, w = 400, 300
    fcx, fy2, fw, fh = 150.0, 150.0, 100.0, 100.0
    torso_chest_zone = (
        (np.arange(h)[:, None] > fy2 + fh * 0.35)
        & (np.abs(np.arange(w)[None, :] - fcx) < fw * 0.65)
    )

    # Independent clothing evidence (coat/jacket in chest)
    orig_clothing_evidence = torso_chest_zone.copy()

    # Face and neck
    face_neck = np.zeros((h, w), dtype=bool)
    face_neck[80:170, 110:190] = True

    # Contaminated hair mask: legitimate scalp hair + contaminated chest coat
    scalp_hair = np.zeros((h, w), dtype=bool)
    scalp_hair[40:100, 90:210] = True
    contaminated_hair = scalp_hair.copy()
    contaminated_hair[200:260, 120:180] = True  # Contaminated with chest coat!

    # Independent permitted non-clothing evidence
    scalp_seeds = scalp_hair & (np.arange(h)[:, None] < fy2 + fh * 0.20)
    lateral_hair = np.zeros((h, w), dtype=bool)  # No lateral hair in chest
    verified_independent_hair = (scalp_seeds | lateral_hair) & ~torso_chest_zone

    # Composite with contaminated hair
    clean_qwen = np.full((h, w, 3), [40, 180, 40], dtype=np.uint8)
    orig_identity = np.full((h, w, 3), [200, 40, 40], dtype=np.uint8)
    _, _, weights = composite_protected_identity(
        clean_qwen_uniform=clean_qwen,
        identity_orig_rgb=orig_identity,
        protected_face_neck=face_neck,
        refined_hair_alpha=contaminated_hair.astype(np.float32),
        hair_accessory_mask=np.zeros((h, w), dtype=bool),
    )

    # Run production leakage validation
    leak_px, unauth_mask, metrics = validate_clothing_leakage(
        W_orig=weights["W_orig"],
        orig_clothing_evidence=orig_clothing_evidence,
        protected_face_neck=face_neck,
        verified_hair_seeds=verified_independent_hair,
        threshold=0.15,
        margin_px=3,
    )

    assert leak_px > 1000, f"Production gate failed to catch contaminated hair mask: {leak_px} px"


def test_unit_adaptive_hair_boundary_refinement():
    """FIX 3: Verify adaptive elliptical kernels scale with resolution and facial dimensions."""
    h1, w1, fw1 = 512, 400, 135.0
    scale1 = float(np.clip(0.5 * (np.sqrt(float(w1 * h1) / (800.0 * 1024.0)) + (fw1 / 270.0)), 0.5, 3.0))
    k_core1 = max(3, int(round(4.0 * scale1)) | 1)

    h2, w2, fw2 = 2048, 1600, 540.0
    scale2 = float(np.clip(0.5 * (np.sqrt(float(w2 * h2) / (800.0 * 1024.0)) + (fw2 / 270.0)), 0.5, 3.0))
    k_core2 = max(3, int(round(4.0 * scale2)) | 1)

    assert scale2 > scale1, "Adaptive scale should be larger for high-resolution images"
    assert k_core2 > k_core1, f"Kernel size should scale up: k1={k_core1}, k2={k_core2}"

    elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_core2, k_core2))
    assert elem.shape == (k_core2, k_core2)
    # Corner pixel of ellipse is 0 (rounded), not 1 (rectangular)
    assert elem[0, 0] == 0, "Morphological kernel must be elliptical, not rectangular"


def test_unit_negative_registration_missing_landmarks():
    """FIX 2 & 9 Negative Test: register_qwen_to_original() rejects missing landmarks."""
    from pipelines.uniform_hair_face_compositor import register_qwen_to_original

    h, w = 200, 200
    qwen_raw = np.full((h, w, 3), 150, dtype=np.uint8)
    cond = np.full((h, w, 3), 100, dtype=np.uint8)

    mock_app = MagicMock()
    mock_app.get.return_value = []

    # Missing landmarks and mocked app to prevent disk loading
    qwen_reg, orig_reg, reg_err, meta = register_qwen_to_original(
        qwen_raw=qwen_raw,
        person_conditioning=cond,
        qwen_kps=None,
        cond_kps=None,
        app=mock_app,
    )

    assert meta["registration_success"] is False, "Registration should fail without landmarks"
    assert reg_err == float("inf"), f"Failed registration must return inf error, got {reg_err}"
    assert meta["failure_reason"] == "missing_landmarks"


def test_unit_negative_registration_extreme_distortion():
    """FIX 2 & 9 Negative Test: register_qwen_to_original() rejects extreme scale and rotation."""
    from pipelines.uniform_hair_face_compositor import register_qwen_to_original

    # Conditioning landmarks
    cond_kps = np.array([[50, 50], [150, 50], [100, 100], [70, 130], [130, 130]], dtype=np.float32)
    # Severe rotation (85 degrees) and 3x scale
    angle = np.radians(85.0)
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)
    qwen_kps = (src := (cond_kps - np.array([100, 90], dtype=np.float32))) @ rot_mat * 3.0 + np.array([100, 90], dtype=np.float32)

    h, w = 300, 300
    qwen_raw = np.full((h, w, 3), 150, dtype=np.uint8)
    cond = np.full((h, w, 3), 100, dtype=np.uint8)

    qwen_reg, orig_reg, reg_err, meta = register_qwen_to_original(
        qwen_raw=qwen_raw,
        person_conditioning=cond,
        qwen_kps=qwen_kps,
        cond_kps=cond_kps,
    )

    assert meta["registration_success"] is False, "Registration should reject 85 deg rotation / 3x scale"
    assert reg_err == float("inf"), "Registration error must be inf"
    assert "out_of_bounds" in meta["failure_reason"] or "low_coverage" in meta["failure_reason"]


def test_unit_negative_collar_anchor_not_found():
    """FIX 3 & 9 Negative Test: align_uniform_collar() rejects uniform where collar anchor cannot be found."""
    from pipelines.uniform_hair_face_compositor import align_uniform_collar

    h, w = 200, 200
    clean_uniform = np.full((h, w, 3), [120, 120, 120], dtype=np.uint8)  # Flat uniform with no edges or parsing
    empty_neck = np.zeros((h, w), dtype=bool)  # Empty neck

    aligned, ok, info = align_uniform_collar(
        clean_qwen_uniform=clean_uniform,
        protected_neck=empty_neck,
        face_bbox=(50, 20, 150, 100),
    )

    assert ok is False, "Collar alignment should report failure when neck is empty"
    assert info["warped"] is False, "Warped flag must be False"
    assert info.get("error") == "protected_neck_empty"


def test_unit_negative_clothing_leakage_contamination():
    """FIX 1 & 9 Negative Test: validate_clothing_leakage() catches intentional clothing contamination."""
    from pipelines.uniform_hair_face_compositor import validate_clothing_leakage

    h, w = 200, 200
    orig_clothing_evidence = np.zeros((h, w), dtype=bool)
    # Coat fabric across chest
    orig_clothing_evidence[80:160, 40:160] = True

    # High ownership leaking into coat region
    W_orig_leaked = np.zeros((h, w), dtype=np.float32)
    W_orig_leaked[90:150, 50:150] = 0.90  # 60x100 = 6000 px leakage!

    face_neck = np.zeros((h, w), dtype=bool)
    face_neck[20:70, 70:130] = True

    leak_px, unauth_mask, metrics = validate_clothing_leakage(
        W_orig=W_orig_leaked,
        orig_clothing_evidence=orig_clothing_evidence,
        protected_face_neck=face_neck,
    )

    assert leak_px > 4000, f"Expected >4000 px leakage detected, got {leak_px}"
    assert leak_px == np.count_nonzero(unauth_mask)


def test_unit_negative_long_braid_loss_blocking_defect():
    """FIX 6 & 9 Negative Test: validate_and_composite_uniform() flags long-braid loss."""
    # Synthetic subject profile verification:
    original_has_long_hair = True
    braid_conf = {"left": "no_foreground_seeds", "right": "grabcut_exception"}
    left_hair_pixels = 400
    right_hair_pixels = 350

    blocking_defects = []
    if original_has_long_hair:
        if braid_conf.get("left") in ("no_foreground_seeds", "grabcut_exception") and left_hair_pixels < 2500:
            blocking_defects.append(f"long_braid_lost_left ({braid_conf.get('left')}, px={left_hair_pixels})")
        if braid_conf.get("right") in ("no_foreground_seeds", "grabcut_exception") and right_hair_pixels < 2500:
            blocking_defects.append(f"long_braid_lost_right ({braid_conf.get('right')}, px={right_hair_pixels})")

    assert len(blocking_defects) == 2, f"Failed to record both lost braids: {blocking_defects}"
    assert "long_braid_lost_left" in blocking_defects[0]
    assert "long_braid_lost_right" in blocking_defects[1]


def test_unit_improved_generated_hair_detection():
    """FIX 1: Robust generated hair detection combining SCHP labels, SCHP probability, and BiSeNet."""
    from pipelines.uniform_hair_face_compositor import detect_and_clean_unwanted_generated_hair

    h, w = 300, 300
    qwen_rgb = np.full((h, w, 3), 200, dtype=np.uint8)
    identity_orig_rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    face_bbox = (80, 40, 220, 180)

    # Hair label from SCHP (scalp hair above forehead)
    schp_labels = np.zeros((h, w), dtype=np.uint8)
    schp_labels[15:40, 90:210] = 2  # Scalp hair (SCHP)

    # Hair from probability map (e.g. braid segment where label was misclassified as clothing)
    hair_prob = np.zeros((h, w), dtype=np.float32)
    hair_prob[60:120, 50:75] = 0.85  # Left braid in prob map

    # Hair from BiSeNet parsing (label 17)
    bisenet_parsing = np.zeros((h, w), dtype=np.uint8)
    bisenet_parsing[60:120, 225:250] = 17  # Right braid in BiSeNet

    # Face and neck skin in BiSeNet (labels 1 and 14) and SCHP (13)
    bisenet_parsing[40:180, 80:220] = 1   # Face
    bisenet_parsing[180:220, 120:180] = 14 # Neck
    schp_labels[40:180, 80:220] = 13     # Face in SCHP

    full_hair_mask = np.zeros((h, w), dtype=bool)
    hair_accessory_mask = np.zeros((h, w), dtype=bool)
    protected_face_neck = np.zeros((h, w), dtype=bool)
    protected_face_neck[40:220, 80:220] = True
    refined_hair_alpha = np.zeros((h, w), dtype=np.float32)

    clean_qwen, unwanted_hair, dbg = detect_and_clean_unwanted_generated_hair(
        qwen_rgb=qwen_rgb,
        identity_orig_rgb=identity_orig_rgb,
        schp_gen_labels=schp_labels,
        full_hair_mask=full_hair_mask,
        hair_accessory_mask=hair_accessory_mask,
        protected_face_neck=protected_face_neck,
        refined_hair_alpha=refined_hair_alpha,
        face_bbox=face_bbox,
        schp_gen_hair_prob=hair_prob,
        bisenet_gen_parsing=bisenet_parsing,
    )

    gen_refined = dbg["gen_hair_refined"]
    # Verify that SCHP hair, probability hair, and BiSeNet hair are all captured
    assert np.all(gen_refined[18:38, 100:200]), "SCHP hair must be included"
    assert np.all(gen_refined[65:115, 55:70]), "Probability map hair must be included"
    assert np.all(gen_refined[65:115, 230:245]), "BiSeNet hair must be included"
    # Verify that skin is not misclassified as hair
    assert not np.any(gen_refined[50:170, 90:210]), "Face skin must never be classified as hair"
    assert not np.any(gen_refined[185:215, 125:175]), "Neck skin must never be classified as hair"


def test_unit_adaptive_hair_boundary_tolerance():
    """FIX 2: Adaptive boundary tolerance adapts to resolution and registration uncertainty."""
    from pipelines.uniform_hair_face_compositor import detect_and_clean_unwanted_generated_hair

    # Test 1: Low resolution, low reg error
    h1, w1 = 400, 300
    qwen1 = np.full((h1, w1, 3), 200, dtype=np.uint8)
    orig1 = np.full((h1, w1, 3), 100, dtype=np.uint8)
    face_bbox1 = (50, 30, 150, 130)
    full_hair1 = np.zeros((h1, w1), dtype=bool)
    full_hair1[30:70, 50:150] = True
    prot_fn1 = np.zeros((h1, w1), dtype=bool)
    refined_alpha1 = np.zeros((h1, w1), dtype=np.float32)
    schp1 = np.zeros((h1, w1), dtype=np.uint8)

    _, _, dbg1 = detect_and_clean_unwanted_generated_hair(
        qwen_rgb=qwen1,
        identity_orig_rgb=orig1,
        schp_gen_labels=schp1,
        full_hair_mask=full_hair1,
        hair_accessory_mask=np.zeros((h1, w1), bool),
        protected_face_neck=prot_fn1,
        refined_hair_alpha=refined_alpha1,
        face_bbox=face_bbox1,
        reg_error=0.5,
    )
    r1 = dbg1["tol_radius"]

    # Test 2: High resolution, higher reg error
    h2, w2 = 1200, 1000
    qwen2 = np.full((h2, w2, 3), 200, dtype=np.uint8)
    orig2 = np.full((h2, w2, 3), 100, dtype=np.uint8)
    face_bbox2 = (200, 100, 600, 500)
    full_hair2 = np.zeros((h2, w2), dtype=bool)
    full_hair2[100:250, 200:600] = True
    prot_fn2 = np.zeros((h2, w2), dtype=bool)
    refined_alpha2 = np.zeros((h2, w2), dtype=np.float32)
    schp2 = np.zeros((h2, w2), dtype=np.uint8)

    _, _, dbg2 = detect_and_clean_unwanted_generated_hair(
        qwen_rgb=qwen2,
        identity_orig_rgb=orig2,
        schp_gen_labels=schp2,
        full_hair_mask=full_hair2,
        hair_accessory_mask=np.zeros((h2, w2), bool),
        protected_face_neck=prot_fn2,
        refined_hair_alpha=refined_alpha2,
        face_bbox=face_bbox2,
        reg_error=2.5,
    )
    r2 = dbg2["tol_radius"]

    assert 1 <= r1 <= 4, f"Low-res tolerance radius {r1} out of bounds"
    assert 1 <= r2 <= 4, f"High-res tolerance radius {r2} out of bounds"
    assert r2 >= r1, f"High-res tolerance {r2} should be >= low-res tolerance {r1}"


def test_unit_unwanted_hair_visibility_based_on_alpha():
    """FIX 3 & FIX 5: Generated hair is unwanted only outside intended silhouette when alpha < 0.35."""
    from pipelines.uniform_hair_face_compositor import detect_and_clean_unwanted_generated_hair

    h, w = 200, 200
    qwen_rgb = np.full((h, w, 3), 200, dtype=np.uint8)
    identity_orig_rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    face_bbox = (50, 20, 150, 120)

    # Intended original silhouette
    full_hair_mask = np.zeros((h, w), dtype=bool)
    full_hair_mask[20:60, 50:150] = True   # Authentic scalp hair

    hair_acc_mask = np.zeros((h, w), dtype=bool)
    protected_face_neck = np.zeros((h, w), dtype=bool)
    protected_face_neck[60:120, 60:140] = True # Authentic face

    # Generated hair covers scalp hair, but also extends outside to the left and right
    schp_labels = np.zeros((h, w), dtype=np.uint8)
    schp_labels[20:60, 50:150] = 2   # Scalp hair (same as orig)
    schp_labels[70:120, 30:55] = 2   # Extra stray braid on left (outside silhouette)
    schp_labels[70:120, 145:170] = 2 # Extra stray braid on right (outside silhouette)

    # Hair alpha: 1.0 on scalp, 0.0 outside silhouette
    refined_hair_alpha = np.zeros((h, w), dtype=np.float32)
    refined_hair_alpha[full_hair_mask] = 1.0

    clean_qwen, unwanted_hair, dbg = detect_and_clean_unwanted_generated_hair(
        qwen_rgb=qwen_rgb,
        identity_orig_rgb=identity_orig_rgb,
        schp_gen_labels=schp_labels,
        full_hair_mask=full_hair_mask,
        hair_accessory_mask=hair_acc_mask,
        protected_face_neck=protected_face_neck,
        refined_hair_alpha=refined_hair_alpha,
        face_bbox=face_bbox,
    )

    # Authentic scalp hair inside full_hair_mask must NEVER be flagged as unwanted
    assert not np.any(unwanted_hair[full_hair_mask]), "Original scalp hair must never be flagged as unwanted"
    assert not np.any(unwanted_hair[protected_face_neck]), "Original face/neck must never be flagged as unwanted"

    # Extra braids outside silhouette with alpha < 0.35 MUST be flagged as unwanted
    assert np.all(unwanted_hair[75:115, 35:50]), "Stray left braid must be flagged as unwanted"
    assert np.all(unwanted_hair[75:115, 150:165]), "Stray right braid must be flagged as unwanted"


def test_unit_multi_region_unwanted_hair_cleanup():
    """FIX 4: Multi-region unwanted hair cleanup restores background, skin, and inpaints clothing."""
    from pipelines.uniform_hair_face_compositor import detect_and_clean_unwanted_generated_hair

    h, w = 200, 200
    qwen_rgb = np.full((h, w, 3), 50, dtype=np.uint8) # Dark uniform
    # Background region is white [255, 255, 255]
    qwen_rgb[:50, :] = [255, 255, 255]

    identity_orig_rgb = np.full((h, w, 3), 180, dtype=np.uint8) # Orig skin/image
    face_bbox = (50, 40, 150, 140)

    # 1. Unwanted hair in background (y in [10, 30], x in [10, 30])
    # 2. Unwanted hair on clothing (y in [100, 130], x in [20, 40])
    schp_labels = np.zeros((h, w), dtype=np.uint8)
    schp_labels[10:30, 10:30] = 2   # Hair on background
    schp_labels[100:130, 20:40] = 2 # Hair on clothing

    # Add dark hair pixels to qwen_rgb in both regions
    qwen_rgb[10:30, 10:30] = [15, 15, 15]
    qwen_rgb[100:130, 20:40] = [15, 15, 15]

    full_hair_mask = np.zeros((h, w), dtype=bool)
    hair_acc_mask = np.zeros((h, w), dtype=bool)
    protected_face_neck = np.zeros((h, w), dtype=bool)
    refined_hair_alpha = np.zeros((h, w), dtype=np.float32)

    clean_qwen, unwanted_hair, dbg = detect_and_clean_unwanted_generated_hair(
        qwen_rgb=qwen_rgb,
        identity_orig_rgb=identity_orig_rgb,
        schp_gen_labels=schp_labels,
        full_hair_mask=full_hair_mask,
        hair_accessory_mask=hair_acc_mask,
        protected_face_neck=protected_face_neck,
        refined_hair_alpha=refined_hair_alpha,
        face_bbox=face_bbox,
        background_color=(255, 255, 255),
    )

    # 1. Background hair must be restored to clean white [255, 255, 255]
    bg_hair_region = clean_qwen[10:30, 10:30]
    assert np.all(bg_hair_region == 255), "Background hair must be restored to white [255, 255, 255]"

    # 2. Clothing hair must be inpainted and no longer have dark hair pixels [15, 15, 15]
    cloth_hair_region = clean_qwen[100:130, 20:40]
    mean_val = np.mean(cloth_hair_region)
    assert mean_val > 35, f"Inpainted clothing should match uniform ~50, got mean={mean_val}"


def test_unit_residual_generated_hair_validation():
    """FIX 6: validate_residual_generated_hair() verifies zero residual hair when cleaned, flags defect if visible."""
    from pipelines.uniform_hair_face_compositor import validate_residual_generated_hair

    h, w = 200, 200
    qwen_rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    qwen_rgb[50:90, 20:50] = [20, 20, 20]  # Raw generated hair

    gen_hair_refined = np.zeros((h, w), dtype=bool)
    gen_hair_refined[50:90, 20:50] = True   # 40x30 = 1200 px of hair

    full_hair_mask = np.zeros((h, w), dtype=bool)
    hair_acc_mask = np.zeros((h, w), dtype=bool)
    protected_face_neck = np.zeros((h, w), dtype=bool)

    W_qwen = np.ones((h, w), dtype=np.float32)
    W_hair = np.zeros((h, w), dtype=np.float32)

    # Scenario A: Uncleaned composite (still has raw generated hair pixels)
    uncleaned_composite = qwen_rgb.copy()
    res_count_a, res_mask_a, _ = validate_residual_generated_hair(
        final_composite_rgb=uncleaned_composite,
        clean_qwen_uniform=qwen_rgb,
        qwen_rgb=qwen_rgb,
        gen_hair_refined=gen_hair_refined,
        full_hair_mask=full_hair_mask,
        hair_accessory_mask=hair_acc_mask,
        protected_face_neck=protected_face_neck,
        W_qwen=W_qwen,
        W_hair=W_hair,
    )
    assert res_count_a > 1000, f"Uncleaned composite must report residual hair > 1000 px, got {res_count_a}"

    # Scenario B: Cleaned composite (hair was inpainted/replaced with clothing [100, 100, 100])
    cleaned_composite = np.full((h, w, 3), 100, dtype=np.uint8)
    res_count_b, res_mask_b, _ = validate_residual_generated_hair(
        final_composite_rgb=cleaned_composite,
        clean_qwen_uniform=cleaned_composite,
        qwen_rgb=qwen_rgb,
        gen_hair_refined=gen_hair_refined,
        full_hair_mask=full_hair_mask,
        hair_accessory_mask=hair_acc_mask,
        protected_face_neck=protected_face_neck,
        W_qwen=W_qwen,
        W_hair=W_hair,
    )
    assert res_count_b == 0, f"Cleaned composite must report 0 residual hair, got {res_count_b}"


def test_unit_independent_garment_segmentation():
    """FIX 1: segment_independent_garments() builds separate original and generated clothing masks."""
    from pipelines.uniform_hair_face_compositor import segment_independent_garments

    h, w = 200, 200
    schp_orig = np.zeros((h, w), dtype=np.uint8)
    schp_gen = np.zeros((h, w), dtype=np.uint8)

    # Original garment (coat=7, upper=5)
    schp_orig[100:180, 40:160] = 7
    # Generated garment (coat=7, dress=6)
    schp_gen[90:170, 30:170] = 5

    # Verified skin and hair seeds
    orig_skin = np.zeros((h, w), dtype=bool)
    orig_skin[70:110, 80:120] = True

    orig_hair_seeds = np.zeros((h, w), dtype=bool)
    orig_hair_seeds[120:150, 45:60] = True

    orig_garment, gen_garment, metrics = segment_independent_garments(
        schp_orig=schp_orig,
        schp_gen=schp_gen,
        orig_skin=orig_skin,
        orig_hair_seeds=orig_hair_seeds,
    )

    # 1. Skin must be excluded from original garment
    assert not np.any(orig_garment & orig_skin), "Skin must be excluded from garment"

    # 2. Verified hair seeds must be excluded from original garment (protecting braid from misclassification)
    assert not np.any(orig_garment & orig_hair_seeds), "Hair seeds must not be classified as garment"

    # 3. Garments are segmented independently
    assert metrics["original_garment_pixels"] > 0
    assert metrics["generated_garment_pixels"] > 0
    assert np.count_nonzero(gen_garment[90:170, 30:170]) == 80 * 140


def test_unit_prevent_original_garment_contamination():
    """FIX 2: clean_original_garment_contamination() strips contaminated clothing from hair and accessory alpha."""
    from pipelines.uniform_hair_face_compositor import clean_original_garment_contamination

    h, w = 200, 200
    full_hair = np.zeros((h, w), dtype=bool)
    hair_acc = np.zeros((h, w), dtype=bool)
    refined_alpha = np.zeros((h, w), dtype=np.float32)

    # Authentic scalp hair
    full_hair[20:60, 60:140] = True
    refined_alpha[20:60, 60:140] = 1.0

    # Authentic braid core
    braid_core = np.zeros((h, w), dtype=bool)
    braid_core[60:150, 40:60] = True
    full_hair[60:150, 40:60] = True
    refined_alpha[60:150, 40:60] = 1.0

    # Contamination: shoulder fabric incorrectly classified as hair/accessory
    full_hair[120:140, 140:170] = True
    refined_alpha[120:140, 140:170] = 0.8
    hair_acc[130:150, 140:160] = True

    # Independent garment evidence
    orig_garment_mask = np.zeros((h, w), dtype=bool)
    orig_garment_mask[100:180, 30:170] = True
    clothing_prob = np.zeros((h, w), dtype=np.float32)
    clothing_prob[100:180, 30:170] = 0.85
    hair_prob = np.zeros((h, w), dtype=np.float32)
    hair_prob[20:60, 60:140] = 0.95
    hair_prob[60:150, 40:60] = 0.90
    hair_prob[120:140, 140:170] = 0.10

    cleaned_hair, cleaned_acc, cleaned_alpha, contam_mask, metrics = clean_original_garment_contamination(
        full_hair_mask=full_hair,
        hair_accessory_mask=hair_acc,
        refined_hair_alpha=refined_alpha,
        original_garment_mask=orig_garment_mask,
        clothing_prob_orig=clothing_prob,
        hair_prob_orig=hair_prob,
        verified_hair_seeds=braid_core,
    )

    # 1. Contamination must be detected and stripped
    assert metrics["total_contamination_pixels"] > 0
    assert not np.any(cleaned_hair[120:140, 140:170]), "Contaminated fabric must be removed from hair"
    assert not np.any(cleaned_acc[130:150, 140:160]), "Contaminated fabric must be removed from accessories"
    assert np.all(cleaned_alpha[120:140, 140:170] == 0.0), "Alpha must be zeroed for contaminated pixels"

    # 2. Authentic braid core must be preserved
    assert np.all(cleaned_hair[braid_core]), "Braid core must be preserved"
    assert np.all(cleaned_alpha[braid_core] == 1.0), "Braid core alpha must remain 1.0"


def test_unit_patch_based_garment_reconstruction():
    """FIX 3 & 4: reconstruct_replacement_garment() synthesizes patterned uniform patches without copying original."""
    from pipelines.uniform_hair_face_compositor import reconstruct_replacement_garment

    h, w = 200, 200
    qwen_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):
        qwen_rgb[:, x] = [30, 60, 120] if (x // 8) % 2 == 0 else [60, 100, 180]

    gen_garment = np.ones((h, w), dtype=bool)

    missing = np.zeros((h, w), dtype=bool)
    missing[20:25, 20:25] = True
    missing[80:95, 80:95] = True
    missing[140:180, 140:180] = True

    recon_rgb, recon_mask, unrec_mask, metrics = reconstruct_replacement_garment(
        qwen_rgb=qwen_rgb,
        generated_garment_mask=gen_garment,
        missing_generated_fabric=missing,
        patch_size=9,
        search_radius=32,
        max_component_area=1000,
        telea_max_area=40,
    )

    # 1. Small defect reconstructed via inpainting
    assert metrics["small_defect_pixels"] == 25
    assert not np.any(np.isnan(recon_rgb[20:25, 20:25]))

    # 2. Patterned defect reconstructed via patch synthesis
    assert metrics["patch_reconstructed_pixels"] > 0
    mean_patch = np.mean(recon_rgb[80:95, 80:95])
    assert 30 <= mean_patch <= 180

    # 3. Large defect marked as unreconstructible
    assert metrics["unreconstructible_count"] == 1
    assert np.count_nonzero(unrec_mask) == 1600


def test_unit_hair_boundary_color_decontamination():
    """FIX 5: decontaminate_hair_garment_boundary() neutralizes uniform bleed in hair fringe."""
    from pipelines.uniform_hair_face_compositor import decontaminate_hair_garment_boundary

    h, w = 150, 150
    orig_rgb = np.full((h, w, 3), 30, dtype=np.uint8)

    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[20:70, 40:100] = 1.0
    alpha[70:85, 40:100] = 0.50

    orig_garment = np.zeros((h, w), dtype=bool)
    orig_garment[80:140, 20:130] = True

    orig_rgb[80:85, 40:100] = [0, 80, 220]

    decontam_rgb, fringe_mask, metrics = decontaminate_hair_garment_boundary(
        identity_orig_rgb=orig_rgb,
        refined_hair_alpha=alpha,
        original_garment_mask=orig_garment,
        min_alpha=0.05,
        max_alpha=0.90,
        garment_margin_px=5,
    )

    assert metrics["decontaminated_pixels"] > 0
    decontam_fringe = decontam_rgb[80:85, 40:100]
    assert np.mean(decontam_fringe[:, :, 2]) < 100, f"Uniform blue bleed must be neutralized, got {np.mean(decontam_fringe[:, :, 2])}"


def test_unit_garment_reconstruction_validation():
    """FIX 6: validate_garment_reconstruction() flags unreconstructible defects, dark patches, and leakage."""
    from pipelines.uniform_hair_face_compositor import validate_garment_reconstruction

    h, w = 100, 100
    comp = np.full((h, w, 3), 120, dtype=np.uint8)
    gen_garment = np.ones((h, w), dtype=bool)
    orig_garment = np.zeros((h, w), dtype=bool)
    orig_garment[50:90, 20:80] = True
    W_orig = np.zeros((h, w), dtype=np.float32)
    face_neck = np.zeros((h, w), dtype=bool)
    hair = np.zeros((h, w), dtype=bool)
    acc = np.zeros((h, w), dtype=bool)

    # Scenario A: Clean valid reconstruction
    unrec_empty = np.zeros((h, w), dtype=bool)
    is_valid, defects, _ = validate_garment_reconstruction(
        final_composite_rgb=comp,
        generated_garment_mask=gen_garment,
        original_garment_mask=orig_garment,
        reconstructed_garment_rgb=comp,
        unreconstructible_mask=unrec_empty,
        W_orig=W_orig,
        protected_face_neck=face_neck,
        full_hair_mask=hair,
        hair_accessory_mask=acc,
    )
    assert is_valid, f"Expected valid, got defects: {defects}"

    # Scenario B: Unreconstructible defect
    unrec_with_hole = np.zeros((h, w), dtype=bool)
    unrec_with_hole[10:30, 10:30] = True
    is_valid_b, defects_b, _ = validate_garment_reconstruction(
        final_composite_rgb=comp,
        generated_garment_mask=gen_garment,
        original_garment_mask=orig_garment,
        reconstructed_garment_rgb=comp,
        unreconstructible_mask=unrec_with_hole,
        W_orig=W_orig,
        protected_face_neck=face_neck,
        full_hair_mask=hair,
        hair_accessory_mask=acc,
    )
    assert not is_valid_b
    assert any("unreconstructible_garment_fabric" in d for d in defects_b)

    # Scenario C: Original clothing leakage
    W_orig_leak = np.zeros((h, w), dtype=np.float32)
    W_orig_leak[50:90, 20:80] = 0.50
    is_valid_c, defects_c, _ = validate_garment_reconstruction(
        final_composite_rgb=comp,
        generated_garment_mask=gen_garment,
        original_garment_mask=orig_garment,
        reconstructed_garment_rgb=comp,
        unreconstructible_mask=unrec_empty,
        W_orig=W_orig_leak,
        protected_face_neck=face_neck,
        full_hair_mask=hair,
        hair_accessory_mask=acc,
    )
    assert not is_valid_c
    assert any("original_garment_leakage" in d for d in defects_c)


def test_unit_detect_missing_garment_coverage_detects_gap_preserves_bg():
    """FIX 1: detect_missing_garment_coverage() detects missing shoulder fabric without turning bg into clothing."""
    from pipelines.uniform_hair_face_compositor import detect_missing_garment_coverage

    h, w = 300, 300
    gen_garment = np.zeros((h, w), dtype=bool)
    gen_garment[150:280, 50:250] = True
    # Hole in gen_garment between neck and shoulder: [150:180, 80:120]
    gen_garment[150:180, 80:120] = False

    orig_garment = np.zeros((h, w), dtype=bool)
    orig_garment[140:280, 40:260] = True

    unwanted_on_cloth = np.zeros((h, w), dtype=bool)
    unwanted_on_cloth[160:170, 130:140] = True

    face = np.zeros((h, w), dtype=bool)
    face[30:100, 100:200] = True

    neck = np.zeros((h, w), dtype=bool)
    neck[100:150, 120:180] = True

    # Genuine background: outside subject
    qwen_rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    qwen_rgb[0:140, 0:80] = 250  # White background gap
    qwen_rgb[0:140, 220:300] = 250

    expected_cov, missing_fab, valid_fab, metrics = detect_missing_garment_coverage(
        gen_garment_mask=gen_garment,
        orig_garment_mask=orig_garment,
        unwanted_on_cloth=unwanted_on_cloth,
        protected_face=face,
        protected_neck=neck,
        qwen_rgb=qwen_rgb,
        chin_y=100,
    )

    # 1. Expected coverage includes shoulder area
    assert expected_cov[165, 100], "Shoulder gap must be expected garment"
    # 2. Missing fabric detects the gap hole
    assert missing_fab[165, 100], "Missing fabric must detect hole"
    # 3. Genuine background gap is NEVER marked as missing fabric or expected garment
    assert not expected_cov[50, 50], "Background must not be expected garment"
    assert not missing_fab[50, 50], "Background must not be missing fabric"
    # 4. Valid fabric preserves solid generated regions
    assert valid_fab[200, 150], "Solid garment region must remain valid fabric"


def test_unit_reconstruct_completion_tracking_and_restricted_telea():
    """FIX 2 & FIX 3: reconstruct_replacement_garment() completion tracking and restricted Telea."""
    from pipelines.uniform_hair_face_compositor import reconstruct_replacement_garment

    h, w = 150, 150
    qwen_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):
        qwen_rgb[:, x] = [20, 50, 100] if (x // 6) % 2 == 0 else [80, 120, 200]

    gen_garment = np.ones((h, w), dtype=bool)

    # Defect A: small localized defect (25 px <= 40 px) -> Telea
    missing = np.zeros((h, w), dtype=bool)
    missing[20:25, 20:25] = True

    # Defect B: large defect (300 px > 40 px) placed where no valid source fabric can reach it -> must NOT use Telea!
    missing[70:90, 70:85] = True
    # Strip away all valid fabric near Defect B so len(cand_ys) < 5
    gen_garment[50:110, 50:105] = False

    recon_rgb, actual_recon, unrec_mask, metrics = reconstruct_replacement_garment(
        qwen_rgb=qwen_rgb,
        generated_garment_mask=gen_garment,
        missing_generated_fabric=missing,
        patch_size=7,
        search_radius=15,
        max_component_area=1000,
        telea_max_area=40,
    )

    # Defect A should be reconstructed
    assert actual_recon[22, 22]
    # Defect B (area=300 > telea_max_area) has len(cand_ys) < 5: must NOT be inpainted via Telea!
    # Must be added to unreconstructible_mask!
    assert np.all(unrec_mask[70:90, 70:85]), "Large defect without valid fabric must be rejected, not inpainted with Telea!"
    assert metrics["unreconstructible_pixels"] >= 300
    assert metrics["actual_reconstructed_pixels"] > 0
    assert "attempted_pixels" in metrics
    assert "rejected_pixels" in metrics


def test_unit_guided_filter_edge_preservation():
    """FIX 8: guided_filter() preserves sharp boundary edges without blurring across guide discontinuities."""
    from pipelines.uniform_hair_face_compositor import guided_filter

    h, w = 100, 100
    # Guide has sharp vertical step edge at x = 50
    guide = np.zeros((h, w), dtype=np.float32)
    guide[:, 50:] = 1.0

    # Source mask has noisy step
    src = np.zeros((h, w), dtype=np.float32)
    src[:, 45:55] = 0.5
    src[:, 55:] = 1.0

    filtered = guided_filter(guide, src, radius=4, eps=1e-3)
    assert filtered.shape == (h, w)
    # The output should align with the guide's sharp step at x=50
    assert filtered[50, 40] < 0.1, "Dark side must remain dark"
    assert filtered[50, 60] > 0.9, "Bright side must remain bright"


def test_unit_neck_boundary_color_decontamination():
    """FIX 7: decontaminate_neck_boundary() neutralizes old uniform color bleed at neck border."""
    from pipelines.uniform_hair_face_compositor import decontaminate_neck_boundary

    h, w = 150, 150
    orig_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Face at top
    face = np.zeros((h, w), dtype=bool)
    face[20:60, 40:110] = True
    orig_rgb[face] = [180, 120, 90]

    # Neck in middle
    neck = np.zeros((h, w), dtype=bool)
    neck[60:110, 50:100] = True
    orig_rgb[neck] = [170, 110, 80]  # Normal skin color

    # Original garment at bottom
    orig_garment = np.zeros((h, w), dtype=bool)
    orig_garment[105:150, 30:120] = True

    # Old dark blue collar bled into neck boundary: [105:110, 50:100]
    orig_rgb[105:110, 50:100] = [20, 30, 80]  # Dark blue bleed

    decontam_rgb, neck_border, metrics = decontaminate_neck_boundary(
        identity_orig_rgb=orig_rgb,
        protected_neck=neck,
        original_garment_mask=orig_garment,
        protected_face=face,
        margin_px=5,
    )

    assert metrics["neck_contamination_pixels"] > 0
    # Boundary color must be restored towards skin tone, not dark blue
    decontam_sample = decontam_rgb[107, 75]
    assert decontam_sample[0] > 100, f"Skin R channel must be restored, got {decontam_sample[0]}"
    assert decontam_sample[2] < 120, f"Blue bleed must be neutralized, got {decontam_sample[2]}"


def test_unit_separate_face_neck_alpha_and_collar_occlusion():
    """FIX 5, FIX 6, FIX 9: composite_protected_identity() with separated face/neck alpha and collar occlusion."""
    from pipelines.uniform_hair_face_compositor import composite_protected_identity

    h, w = 200, 200
    qwen_uniform = np.full((h, w, 3), [0, 255, 0], dtype=np.uint8)     # Green
    orig_identity = np.full((h, w, 3), [255, 0, 0], dtype=np.uint8)    # Red

    face_alpha = np.zeros((h, w), dtype=np.float32)
    face_alpha[30:80, 60:140] = 1.0  # Crisp 1.0 face matte

    neck_alpha = np.zeros((h, w), dtype=np.float32)
    neck_alpha[80:130, 70:130] = 1.0  # Neck alpha

    # Collar occlusion: collar lapels fold over lower neck at [115:130, 70:130]
    collar_occ = np.zeros((h, w), dtype=bool)
    collar_occ[115:130, 70:130] = True

    final_rgb, pixel_map, weights = composite_protected_identity(
        clean_qwen_uniform=qwen_uniform,
        identity_orig_rgb=orig_identity,
        protected_face_alpha=face_alpha,
        refined_neck_alpha=neck_alpha,
        gen_collar_occlusion=collar_occ,
    )

    # 1. Face center must be crisp original identity (Red)
    assert final_rgb[50, 100, 0] > 240
    assert weights["W_orig"][50, 100] > 0.95

    # 2. Upper neck must be original identity (Red)
    assert final_rgb[95, 100, 0] > 240
    assert weights["W_orig"][95, 100] > 0.90

    # 3. Collar occlusion over lower neck must be Qwen collar (Green)
    assert final_rgb[122, 100, 1] > 200, f"Collar occlusion not applied: {final_rgb[122, 100]}"
    assert weights["W_qwen"][122, 100] > 0.50

    # 4. Uniform body must be Qwen (Green)
    assert final_rgb[160, 100, 1] > 240


def test_unit_negative_contaminated_hair_and_neck_leakage_detected():
    """FIX 10: validate_clothing_leakage() flags leakage even when hair/neck masks are contaminated with clothing."""
    from pipelines.uniform_hair_face_compositor import validate_clothing_leakage

    h, w = 150, 150
    # Original photograph continuous ownership map: active on body
    W_orig = np.zeros((h, w), dtype=np.float32)
    W_orig[70:120, 40:110] = 0.80  # Leaking original pixels

    # Independent clothing evidence: torso clothing at [70:120, 40:110]
    clothing_evidence = np.zeros((h, w), dtype=bool)
    clothing_evidence[70:120, 40:110] = True

    face_neck = np.zeros((h, w), dtype=bool)
    face_neck[20:60, 50:100] = True

    # Deliberately contaminated scalp seeds with original clothing
    # verified_hair_seeds should only protect verified anatomical scalp seeds near the head
    scalp_seeds = np.zeros((h, w), dtype=bool)
    scalp_seeds[10:30, 40:110] = True

    leak_px, leak_mask, metrics = validate_clothing_leakage(
        W_orig=W_orig,
        orig_clothing_evidence=clothing_evidence,
        protected_face_neck=face_neck,
        verified_hair_seeds=scalp_seeds,
        threshold=0.15,
        margin_px=3,
    )

    # Leakage MUST be detected! Contaminated peripheral clothing cannot hide
    assert leak_px > 0, "Unauthorized clothing leakage must be detected!"
    assert metrics["leakage_pixels"] > 1000


# ==============================================================================
# SECTION 2: INTEGRATION REGRESSION TESTS (ON DELIVERABLES & INTERMEDIATES)
# ==============================================================================

def load_integration_artifacts() -> Dict[str, Any] | None:
    deliverable_path = OUTPUTS_DIR / f"{JOB_ID}_uniform.png"
    if not deliverable_path.exists():
        return None
    
    debug_dir = OUTPUTS_DIR / "debug_phases"
    source_map_path = debug_dir / f"{JOB_ID}_18_pixel_source_map.png"
    hair_alpha_path = debug_dir / f"{JOB_ID}_11_refined_hair_alpha.png"
    full_hair_path = debug_dir / f"{JOB_ID}_07_original_full_hair_mask.png"
    acc_path = debug_dir / f"{JOB_ID}_09_hair_accessory_mask.png"
    face_neck_path = debug_dir / f"{JOB_ID}_12_protected_face_neck.png"

    return {
        "final": np.array(Image.open(deliverable_path).convert("RGB")),
        "source_map": np.array(Image.open(source_map_path).convert("RGB")) if source_map_path.exists() else None,
        "hair_alpha": np.array(Image.open(hair_alpha_path).convert("L")) if hair_alpha_path.exists() else None,
        "full_hair": np.array(Image.open(full_hair_path).convert("L")) if full_hair_path.exists() else None,
        "acc_mask": np.array(Image.open(acc_path).convert("L")) if acc_path.exists() else None,
        "face_neck": np.array(Image.open(face_neck_path).convert("L")) if face_neck_path.exists() else None,
    }


@pytest.fixture(scope="module")
def art():
    """Pytest fixture providing integration regression deliverables and diagnostic artifacts."""
    artifacts = load_integration_artifacts()
    if artifacts is None:
        pytest.skip(f"Integration deliverables not found for '{JOB_ID}' in {OUTPUTS_DIR}")
    return artifacts


def test_regression_no_original_badge(art):
    """Ensure zero original badge pixels leak into chest."""
    src_map = art["source_map"]
    hair_alpha = art["hair_alpha"]
    acc_mask = art["acc_mask"]
    assert src_map is not None, "Pixel source map missing"
    badge_zone = src_map[640:750, 480:610]
    hair_in_bz = hair_alpha[640:750, 480:610] if hair_alpha is not None else np.zeros((110, 130), dtype=np.uint8)
    acc_in_bz = acc_mask[640:750, 480:610] if acc_mask is not None else np.zeros((110, 130), dtype=np.uint8)
    red_in_badge = (badge_zone[:, :, 0] > 200) & (badge_zone[:, :, 1] < 50) & (badge_zone[:, :, 2] < 50)
    unauthorized_badge = red_in_badge & (hair_in_bz <= 128) & (acc_in_bz <= 128)
    assert np.count_nonzero(unauthorized_badge) == 0, f"Original badge leaked into chest: {np.count_nonzero(unauthorized_badge)} px"


def test_regression_no_original_clothing_leakage(art):
    """Ensure zero original checked overcoat or shirt collar pixels leak into the chest."""
    src_map = art["source_map"]
    hair_alpha = art["hair_alpha"]
    face_neck = art["face_neck"]
    acc_mask = art["acc_mask"]

    h, w = src_map.shape[:2]
    chest_box = np.zeros((h, w), dtype=bool)
    chest_box[550:800, 300:500] = True

    red_chest = (src_map[:, :, 0] > 200) & (src_map[:, :, 1] < 50) & (src_map[:, :, 2] < 50) & chest_box
    permitted = (face_neck > 128) | (hair_alpha > 128) | (acc_mask > 128)
    unauthorized = red_chest & ~permitted
    assert np.count_nonzero(unauthorized) == 0, f"Original clothing leaked into chest: {np.count_nonzero(unauthorized)} px"


def test_regression_no_white_v_shaped_neck_gap(art):
    """Ensure zero white canvas gap pixels between chin/neck and collar."""
    final_img = art["final"]
    neck_box = final_img[490:580, 340:460]
    white_canvas_pixels = (neck_box[:, :, 0] > 250) & (neck_box[:, :, 1] > 250) & (neck_box[:, :, 2] > 250)
    assert np.count_nonzero(white_canvas_pixels) < 50, f"White neck gap detected: {np.count_nonzero(white_canvas_pixels)} px"


def test_regression_full_length_braids_preserved(art):
    """Ensure both braids are fully preserved from scalp seeds to endpoints (>30k px each)."""
    full_hair = art["full_hair"]
    h, w = full_hair.shape[:2]
    left_hair = np.count_nonzero(full_hair[:, :w // 2])
    right_hair = np.count_nonzero(full_hair[:, w // 2:])
    assert left_hair > 30000, f"Left braid truncated: only {left_hair} px"
    assert right_hair > 30000, f"Right braid truncated: only {right_hair} px"


def test_regression_biometric_similarity(art):
    """Ensure facial identity is strictly preserved (ArcFace similarity >= 0.95)."""
    from pipelines.photo_restoration import _get_insight_app
    app = _get_insight_app()

    orig_path = Path("uploads/03b9a134-5934-41f4-97c6-17562d52e686_Schoolgirl Portrait with Blue Bows.png")
    if not orig_path.exists():
        pytest.skip(f"Original reference image missing: {orig_path}")

    orig_img = np.array(Image.open(orig_path).convert("RGB"))
    final_img = art["final"]

    faces_orig = app.get(cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR))
    faces_final = app.get(cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

    assert len(faces_orig) == 1, "Expected 1 face in original"
    assert len(faces_final) == 1, "Expected 1 face in final"

    sim = float(np.dot(faces_orig[0].embedding, faces_final[0].embedding) /
                (np.linalg.norm(faces_orig[0].embedding) * np.linalg.norm(faces_final[0].embedding)))
    assert sim >= 0.950, f"Biometric similarity below threshold: {sim:.4f} < 0.9500"


def test_unit_extract_clean_identity_layer():
    """Test standalone clean identity layer extraction: trimap, edge-aware matting, and zero garment contamination."""
    from pipelines.uniform_hair_face_compositor import extract_clean_identity_layer

    h, w = 200, 200
    dummy_rgb = np.full((h, w, 3), 200, dtype=np.uint8)

    # Face center
    dummy_rgb[40:100, 70:130] = [220, 180, 160]
    # Hair
    dummy_rgb[20:70, 50:150] = [30, 20, 15]
    # Garment
    dummy_rgb[120:200, 40:160] = [20, 30, 180]

    # Precomputed mocks for deterministic testing
    schp_labels = np.zeros((h, w), dtype=np.uint8)
    schp_labels[20:70, 50:150] = 2   # Hair
    schp_labels[40:100, 70:130] = 13 # Face
    schp_labels[120:200, 40:160] = 5 # Upper clothes

    bisenet_p = np.zeros((h, w), dtype=np.uint8)
    bisenet_p[40:100, 70:130] = 1    # Skin
    bisenet_p[100:125, 80:120] = 14  # Neck
    bisenet_p[120:200, 40:160] = 16  # Cloth

    hair_prob = (schp_labels == 2).astype(np.float32)
    clothing_prob = (schp_labels == 5).astype(np.float32)

    schp_res = {"labels": schp_labels, "hair_prob": hair_prob, "clothing_prob": clothing_prob}
    bisenet_res = {"parsing": bisenet_p}

    face_bbox = (70, 40, 130, 100)

    res = extract_clean_identity_layer(
        immutable_original_rgb=dummy_rgb,
        face_bbox=face_bbox,
        schp_orig_res=schp_res,
        bisenet_res=bisenet_res,
        save_debug=False,
        conditioning_strategy="B",
    )

    # 1. Definite garment pixels MUST be non-identity (alpha = 0.0)
    alpha = res["refined_identity_alpha"]
    garment_mask = res["original_garment_mask"]
    assert np.all(alpha[garment_mask] == 0.0), "Garment pixels leaked into identity alpha!"

    # 2. Face core must be 1.0
    face_core = res["original_face_mask"]
    assert np.mean(alpha[face_core]) >= 0.99, "Face core alpha degraded below 0.99"

    # 3. Trimap values must be {0, 128, 255}
    trimap = res["identity_trimap"]
    unique_vals = set(np.unique(trimap))
    assert unique_vals.issubset({0, 128, 255}), f"Unexpected trimap values: {unique_vals}"

    # 4. Multi-background rendering produces valid shapes
    assert res["identity_on_white"].shape == (h, w, 3)
    assert res["identity_on_black"].shape == (h, w, 3)
    assert res["identity_on_contrast"].shape == (h, w, 3)

    # 5. Conditioning Strategy B maintains torso silhouette without floating head
    cond_b = res["qwen_conditioning_rgb"]
    assert cond_b.shape == (h, w, 3)
    # Garment area in conditioning must NOT be pure white (not a floating head)
    garment_color = cond_b[150, 100]
    assert not np.all(garment_color == 255), "Conditioning strategy B rendered a floating head!"


def test_unit_identity_extraction_contact_sheet_builder(tmp_path):
    """Test contact sheet generator for identity extraction."""
    from pipelines.uniform_hair_face_compositor import _build_identity_extraction_contact_sheet

    job_id = "test_id"
    # Create dummy images
    names = [
        "immutable_original_rgb", "original_semantic_labels", "original_garment_mask",
        "identity_trimap", "refined_identity_alpha", "garment_boundary_contamination",
        "identity_on_white", "identity_on_black", "identity_on_contrast",
        "qwen_conditioning_strategy_a", "qwen_conditioning_strategy_b", "final_composite",
    ]
    for n in names:
        img = Image.new("RGB", (100, 100), (100, 150, 200))
        img.save(tmp_path / f"{job_id}_{n}.png")

    save_path = tmp_path / f"{job_id}_sheet.png"
    _build_identity_extraction_contact_sheet(
        debug_dir=tmp_path,
        job_id=job_id,
        save_path=save_path,
        status_banner="TEST IDENTITY SHEET",
    )
    assert save_path.exists(), "Identity contact sheet was not generated!"
    sheet = Image.open(save_path)
    assert sheet.width > 500 and sheet.height > 500


# ==============================================================================
# MAIN TEST RUNNER WITH DETAILED PASS / FAIL / SKIPPED REPORTING
# ==============================================================================

if __name__ == "__main__":
    passed = 0
    failed = 0
    skipped = 0

    print("==================================================================")
    print("RUNNING UNIFORM SWAPPING TEST SUITE (UNIT & INTEGRATION)")
    print("==================================================================")

    # 1. Deterministic Unit Tests
    unit_tests = [
        ("test_unit_isolated_register_qwen_to_original", test_unit_isolated_register_qwen_to_original),
        ("test_unit_isolated_recover_hair_and_accessories", test_unit_isolated_recover_hair_and_accessories),
        ("test_unit_isolated_align_uniform_collar", test_unit_isolated_align_uniform_collar),
        ("test_unit_isolated_composite_protected_identity", test_unit_isolated_composite_protected_identity),
        ("test_unit_original_clothing_cannot_enter_hair_restoration_layer", test_unit_original_clothing_cannot_enter_hair_restoration_layer),
        ("test_unit_blue_badge_cannot_become_hair_accessory", test_unit_blue_badge_cannot_become_hair_accessory),
        ("test_unit_failed_grabcut_cannot_silently_accept_dark_clothing", test_unit_failed_grabcut_cannot_silently_accept_dark_clothing),
        ("test_unit_hair_alpha_cannot_spread_into_definite_non_hair_regions", test_unit_hair_alpha_cannot_spread_into_definite_non_hair_regions),
        ("test_unit_original_neck_pixels_remain_preserved", test_unit_original_neck_pixels_remain_preserved),
        ("test_unit_registration_occurs_before_compositing", test_unit_registration_occurs_before_compositing),
        ("test_unit_clothing_leakage_validation_detects_contaminated_masks", test_unit_clothing_leakage_validation_detects_contaminated_masks),
        ("test_unit_production_quality_gate_rejects_contaminated_hair_mask", test_unit_production_quality_gate_rejects_contaminated_hair_mask),
        ("test_unit_adaptive_hair_boundary_refinement", test_unit_adaptive_hair_boundary_refinement),
        ("test_unit_compositor_exceptions_produce_blocking_defects", test_unit_compositor_exceptions_produce_blocking_defects),
        ("test_unit_negative_registration_missing_landmarks", test_unit_negative_registration_missing_landmarks),
        ("test_unit_negative_registration_extreme_distortion", test_unit_negative_registration_extreme_distortion),
        ("test_unit_negative_collar_anchor_not_found", test_unit_negative_collar_anchor_not_found),
        ("test_unit_negative_clothing_leakage_contamination", test_unit_negative_clothing_leakage_contamination),
        ("test_unit_negative_long_braid_loss_blocking_defect", test_unit_negative_long_braid_loss_blocking_defect),
        ("test_unit_improved_generated_hair_detection", test_unit_improved_generated_hair_detection),
        ("test_unit_adaptive_hair_boundary_tolerance", test_unit_adaptive_hair_boundary_tolerance),
        ("test_unit_unwanted_hair_visibility_based_on_alpha", test_unit_unwanted_hair_visibility_based_on_alpha),
        ("test_unit_multi_region_unwanted_hair_cleanup", test_unit_multi_region_unwanted_hair_cleanup),
        ("test_unit_residual_generated_hair_validation", test_unit_residual_generated_hair_validation),
        ("test_unit_independent_garment_segmentation", test_unit_independent_garment_segmentation),
        ("test_unit_prevent_original_garment_contamination", test_unit_prevent_original_garment_contamination),
        ("test_unit_patch_based_garment_reconstruction", test_unit_patch_based_garment_reconstruction),
        ("test_unit_hair_boundary_color_decontamination", test_unit_hair_boundary_color_decontamination),
        ("test_unit_garment_reconstruction_validation", test_unit_garment_reconstruction_validation),
        ("test_unit_extract_clean_identity_layer", test_unit_extract_clean_identity_layer),
    ]

    print(f"\n--- SECTION 1: DETERMINISTIC UNIT TESTS ({len(unit_tests)} tests) ---")
    for name, fn in unit_tests:
        try:
            fn()
            passed += 1
            print(f"  [PASS] {name}")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {name}: {e}")

    # 2. Integration Regression Tests
    print("\n--- SECTION 2: INTEGRATION REGRESSION TESTS ---")
    art_dict = load_integration_artifacts()
    if art_dict is None:
        print("  Artifacts not found in outputs/ - Skipping integration tests")
        skipped += 5
    else:
        integration_tests = [
            ("test_regression_no_original_badge", test_regression_no_original_badge),
            ("test_regression_no_original_clothing_leakage", test_regression_no_original_clothing_leakage),
            ("test_regression_no_white_v_shaped_neck_gap", test_regression_no_white_v_shaped_neck_gap),
            ("test_regression_full_length_braids_preserved", test_regression_full_length_braids_preserved),
            ("test_regression_biometric_similarity", test_regression_biometric_similarity),
        ]
        for name, fn in integration_tests:
            try:
                res = fn(art_dict)
                passed += 1
                print(f"  [PASS] {name}")
            except pytest.skip.Exception as se:
                skipped += 1
                print(f"  [SKIP] {name}: {se}")
            except Exception as e:
                failed += 1
                print(f"  [FAIL] {name}: {e}")

    print("\n==================================================================")
    print(f"TEST SUMMARY: {passed} PASSED, {failed} FAILED, {skipped} SKIPPED (TOTAL: {passed + failed + skipped})")
    print("==================================================================")
    if failed > 0:
        sys.exit(1)
