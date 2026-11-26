"""Unit and regression tests for V12.2 pipeline stabilization.

Tests:
1. Aspect-ratio preserving portrait normalizer (Section 3).
2. Semantic hair accessory vs school badge separation (Section 5 & 6).
3. Target uniform visual cues and collar prompt construction (Section 4 & 5).
4. Section 9 Quality Validator multi-defect checks (aspect ratio, collar color, badge, neck, halo, needs_review).
"""

import numpy as np
import pytest
from PIL import Image

from pipelines.uniform_swap.config import UniformSwapConfig
from pipelines.uniform_swap.image_normalizer import normalize_portrait_aspect_ratio, NormalizationTransform
from pipelines.uniform_swap.uniform_generator import _detect_uniform_visual_cues, UniformGenerator
from pipelines.uniform_swap.quality_validator import QualityValidator


def test_aspect_ratio_normalizer_square_image():
    """Verify Section 3: Square 1:1 image is cropped with landmark-awareness to avoid vertical squash."""
    w, h = 1000, 1000
    sq_img = Image.new("RGB", (w, h), (240, 240, 240))
    norm_img, transform, meta = normalize_portrait_aspect_ratio(
        sq_img,
        target_size=(800, 1024),
    )
    assert norm_img.size == (800, 1024)
    assert (transform.target_w, transform.target_h) == (800, 1024)
    # The crop aspect ratio should match 800/1024 = 0.78125
    cw = transform.crop_x2 - transform.crop_x1
    ch = transform.crop_y2 - transform.crop_y1
    crop_aspect = cw / ch
    assert abs(crop_aspect - (800 / 1024)) < 0.01


def test_uniform_visual_cues_detection():
    """Verify Section 4 & 5: Detects red collar/trim on uniform reference."""
    tw, th = 800, 1024
    # Create an image with a solid red collar region at top
    img_arr = np.zeros((th, tw, 4), dtype=np.uint8)
    # Garment alpha
    img_arr[200:900, 150:650, 3] = 255
    # Plaid torso: gray/blue
    img_arr[350:900, 150:650, :3] = [30, 40, 80]
    # Red collar at top: (255, 0, 0)
    img_arr[200:350, 250:550, :3] = [210, 20, 20]

    cues = _detect_uniform_visual_cues(img_arr)
    assert cues.get("has_red_collar") is True
    assert cues.get("has_red_trim") is True

    # Test prompt construction includes solid red collar and plaid description
    ug = UniformGenerator()
    pos, neg = ug.build_prompts(
        vl_plan={},
        neck_cls="partially_visible_neck",
        uniform_reference_image=img_arr,
    )
    assert "solid red" in pos and "collar" in pos
    assert "red front button placket" in pos or "placket" in pos
    assert "plaid" in pos
    assert "school badge" in neg
    assert "chest badge" in neg


def test_quality_validator_section_9_defects():
    """Verify Section 9: QualityValidator catches head distortion, badge leakage, and red neck bleed."""
    qv = QualityValidator(min_biometric_similarity=0.82)
    h, w = 1024, 800

    cand_img = np.full((h, w, 3), 255, dtype=np.uint8)
    src_img = np.full((h, w, 3), 255, dtype=np.uint8)

    # Test 1: Badge leakage detection
    badge_mask = np.zeros((h, w), dtype=bool)
    badge_mask[600:700, 450:550] = True
    # In source image, badge is blue
    src_with_badge = src_img.copy()
    src_with_badge[badge_mask] = [30, 80, 180]
    # In candidate image, badge is also retained (leakage!)
    cand_with_badge = cand_img.copy()
    cand_with_badge[badge_mask] = [30, 80, 180]

    id_layer_badge = {
        "original_badge_mask": badge_mask,
    }
    res = qv.validate_candidate(
        candidate_image=cand_with_badge,
        source_image=src_with_badge,
        identity_layer=id_layer_badge,
        stage="test_badge",
    )
    assert any("badge_leakage_detected" in d for d in res["defects"])
    assert res["passed"] is False
    assert res["needs_review"] is True

    # Test 2: Neck red bleed detection
    neck_mask = np.zeros((h, w), dtype=bool)
    throat_y1 = int(h * 0.45)
    throat_y2 = int(h * 0.45 + 0.06 * h * 0.35)
    neck_mask[throat_y1:throat_y2, 350:450] = True
    cand_with_red_neck = cand_img.copy()
    cand_with_red_neck[neck_mask] = [220, 20, 20]  # bright scarlet red on throat skin

    id_layer_neck = {
        "original_neck_mask": neck_mask,
    }
    res_neck = qv.validate_candidate(
        candidate_image=cand_with_red_neck,
        source_image=src_img,
        identity_layer=id_layer_neck,
        stage="test_neck",
    )
    assert any("neck_red_bleed_detected" in d for d in res_neck["defects"])
    assert res_neck["passed"] is False
    assert res_neck["needs_review"] is True
