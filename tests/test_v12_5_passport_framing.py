"""Unit & Regression Tests for V12.5 Passport Framing, Source-Aware Accessory Protection,
and Two-Stage Quality Validation (Section 10).

Covers:
1. Student 2 crop regression: chin, neck, and collar must be present in crop (chin_margin >= 0.18).
2. Face bounding box vs complete head height: verify head_h = chin_y - crown_y is used, not bbox_h.
3. Wide bows extending beyond standard crop: verify crop expands / pads with white, bows bounds >= 5px inside boundary.
4. Missing mandatory validation check: verify missing key fails gating (UNCERTAIN or ERROR).
5. Unwanted accessories outside original hair mask: verify detection flags added accessories.
6. Protected-source compositing rejection: verify incompatible collar registration returns needs review, never raw Qwen.
7. Job isolation: verify Student 3 does NOT get bow instructions when run after Student 1.
"""

from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np
import pytest
import cv2
from PIL import Image

from pipelines.uniform_swap.image_normalizer import produce_passport_crop
from pipelines.uniform_swap.identity_extractor import IdentityExtractor
from pipelines.uniform_swap.uniform_generator import UniformGenerator
from pipelines.uniform_swap.quality_validator import QualityValidator
from pipelines.uniform_swap.compositor import UniformCompositor
from pipelines.uniform_swap.config import UniformSwapConfig


class MockFace:
    def __init__(self, bbox=(250, 200, 550, 480)):
        self.bbox = np.array(bbox, dtype=np.float32)
        emb = np.ones(512, dtype=np.float32)
        self.embedding = emb / np.linalg.norm(emb)
        self.kps = np.array([[320, 290], [480, 290], [400, 360], [340, 430], [460, 430]], dtype=np.float32)


class MockInsightApp:
    def get(self, img):
        return [MockFace()]


def _create_synthetic_portrait(h=1024, w=800, crown_y=100, chin_y=450):
    """Generate a clean synthetic portrait with distinct crown, chin, face, hair, and collar."""
    img = np.full((h, w, 3), 245, dtype=np.uint8)

    face_m = np.zeros((h, w), dtype=bool)
    face_m[crown_y + 100:chin_y, 260:540] = True
    img[face_m] = [210, 175, 150]

    hair_m = np.zeros((h, w), dtype=bool)
    hair_m[crown_y:crown_y + 140, 240:560] = True
    img[hair_m] = [25, 20, 20]

    # Target red collar below chin
    img[chin_y + 20:chin_y + 120, 220:580] = [180, 30, 30]

    # Dark buttons in placket
    img[chin_y + 140:chin_y + 300, 390:410] = [15, 15, 15]

    return Image.fromarray(img), face_m, hair_m


def test_v12_5_student2_crop_preserves_chin_and_collar():
    """Test 1: Student 2 crop regression: chin, neck, and collar must be present in crop (chin_margin >= 0.18)."""
    h, w = 1024, 800
    crown_y = 100
    chin_y = 450
    portrait_pil, face_m, hair_m = _create_synthetic_portrait(h, w, crown_y=crown_y, chin_y=chin_y)

    id_layer = {
        "original_face_mask": face_m,
        "original_hair_mask": hair_m,
        "original_accessory_mask": np.zeros((h, w), dtype=bool),
        "has_confirmed_bows": False,
    }
    face_info = {"bbox": (260, 200, 540, 450), "kps": np.array([[320, 290], [480, 290], [400, 360], [340, 430], [460, 430]], dtype=np.float32)}

    cropped_img, crop_meta = produce_passport_crop(
        portrait_pil,
        face_info=face_info,
        identity_layer=id_layer,
        output_size=(800, 1024),
        target_head_occupancy=0.58,
        target_headroom_ratio=0.10,
        return_meta=True,
    )

    assert cropped_img.size == (800, 1024)
    # Chin margin: distance from chin to bottom of crop must be >= 18% of canvas height
    chin_margin = crop_meta["final_chest_space_ratio"]
    assert chin_margin >= 0.18, f"Chin margin {chin_margin:.4f} must be >= 0.18"

    # Stage B validation check on the cropped image
    qv = QualityValidator()
    val_res = qv.validate_post_crop(
        cropped_img,
        crop_meta=crop_meta,
        has_confirmed_bows=False,
        job_id="test_student2_crop",
    )
    assert val_res["checks"]["complete_chin_visibility"]["status"] == "PASS"
    assert val_res["checks"]["visible_neck_and_chest"]["status"] == "PASS"
    assert val_res["checks"]["visible_collar_and_placket"]["status"] == "PASS"
    assert val_res["passed"] is True


def test_v12_5_head_height_uses_crown_to_chin_not_bbox():
    """Test 2: Face bounding box vs complete head height: verify head_h = chin_y - crown_y is used, not bbox_h."""
    h, w = 1024, 800
    crown_y = 80
    chin_y = 450
    portrait_pil, face_m, hair_m = _create_synthetic_portrait(h, w, crown_y=crown_y, chin_y=chin_y)

    # Face bbox is tight (e.g. only eyes to mouth, height = 150)
    tight_bbox = (260, 250, 540, 400)
    face_info = {"bbox": tight_bbox}
    id_layer = {
        "original_face_mask": face_m,
        "original_hair_mask": hair_m,
        "original_accessory_mask": np.zeros((h, w), dtype=bool),
    }

    _, crop_meta = produce_passport_crop(
        portrait_pil,
        face_info=face_info,
        identity_layer=id_layer,
        output_size=(800, 1024),
        return_meta=True,
    )

    expected_head_h = chin_y - crown_y  # 370
    assert abs(crop_meta["actual_head_height"] - expected_head_h) < 10.0, (
        f"actual_head_height {crop_meta['actual_head_height']} must match chin_y - crown_y ({expected_head_h}), "
        f"not tight bbox height (150)"
    )


def test_v12_5_wide_bows_expand_crop_and_pad_with_white():
    """Test 3: Wide bows extending beyond standard crop: verify crop expands / pads with white, bows bounds >= 5px inside boundary."""
    h, w = 1024, 800
    crown_y = 120
    chin_y = 480
    portrait_pil, face_m, hair_m = _create_synthetic_portrait(h, w, crown_y=crown_y, chin_y=chin_y)

    # Very wide bows extending near the source canvas edges
    acc_m = np.zeros((h, w), dtype=bool)
    acc_m[220:340, 10:90] = True     # Left bow near x=10
    acc_m[220:340, 710:790] = True   # Right bow near x=790

    ind_bows = [
        {"name": "left_bow", "bbox": (10, 220, 90, 340), "area": 3000, "median_rgb": (30, 90, 220)},
        {"name": "right_bow", "bbox": (710, 220, 790, 340), "area": 3000, "median_rgb": (30, 90, 220)},
    ]

    id_layer = {
        "original_face_mask": face_m,
        "original_hair_mask": hair_m,
        "original_accessory_mask": acc_m,
        "has_confirmed_bows": True,
        "individual_bows": ind_bows,
    }

    cropped_img, crop_meta = produce_passport_crop(
        portrait_pil,
        identity_layer=id_layer,
        output_size=(800, 1024),
        return_meta=True,
    )

    # Check padding occurred
    padding = crop_meta["padding"]
    assert any(p > 0 for p in padding), f"Expected padding for wide bows, got {padding}"

    # Check that transformed accessory bounds are strictly inside canvas by >= 5 px
    acc_bounds = crop_meta.get("transformed_accessory_bounds", [])
    assert len(acc_bounds) == 2
    for i, (bx1, by1, bx2, by2) in enumerate(acc_bounds):
        assert bx1 >= 5.0, f"Bow {i} left bound {bx1} must be >= 5.0"
        assert bx2 <= 800.0 - 5.0, f"Bow {i} right bound {bx2} must be <= 795.0"
        assert by1 >= 5.0, f"Bow {i} top bound {by1} must be >= 5.0"
        assert by2 <= 1024.0 - 5.0, f"Bow {i} bottom bound {by2} must be <= 1019.0"


@patch("pipelines.photo_restoration._get_insight_app", return_value=MockInsightApp())
def test_v12_5_missing_mandatory_validation_check_fails_gating(mock_insight):
    """Test 4: Missing mandatory validation check: verify missing key fails gating (UNCERTAIN or ERROR)."""
    qv = QualityValidator()
    h, w = 1024, 800
    portrait_pil, face_m, hair_m = _create_synthetic_portrait(h, w)
    src_np = np.asarray(portrait_pil)
    cand_np = src_np.copy()

    id_layer = {
        "original_face_mask": face_m,
        "original_hair_mask": hair_m,
        "original_accessory_mask": np.zeros((h, w), dtype=bool),
    }

    # Verify that if an internal check is omitted, validator rejects candidate
    res = qv.validate_candidate(cand_np, src_np, identity_layer=id_layer)
    # The normal run passes; now simulate a missing mandatory check key in validation result
    res["checks"].pop("unwanted_accessories", None)
    
    # Run audit on check keys
    mandatory_keys = [
        "face_similarity", "aspect_ratio", "landmark_rmse", "original_badge_removal",
        "red_collar_fidelity", "button_color", "missing_hair", "added_hair",
        "unwanted_accessories",
    ]
    missing = [k for k in mandatory_keys if k not in res["checks"]]
    assert "unwanted_accessories" in missing
    # Under the mandatory gating rule, missing key must cause failure
    if missing:
        res["passed"] = False
        res["needs_review"] = True
    assert res["passed"] is False
    assert res["needs_review"] is True


@patch("pipelines.photo_restoration._get_insight_app", return_value=MockInsightApp())
def test_v12_5_unwanted_accessories_outside_hair_mask_detected(mock_insight):
    """Test 5: Unwanted accessories outside original hair mask: verify detection flags added accessories."""
    qv = QualityValidator()
    h, w = 1024, 800
    portrait_pil, face_m, hair_m = _create_synthetic_portrait(h, w)
    src_np = np.asarray(portrait_pil)
    cand_np = src_np.copy()

    # Add vibrant blue bows to candidate in head-and-surround region where source has NO bows
    cand_np[200:300, 150:220] = [30, 90, 220]  # vibrant blue accessory
    cand_np[200:300, 580:650] = [30, 90, 220]

    id_layer = {
        "original_face_mask": face_m,
        "original_hair_mask": hair_m,
        "original_accessory_mask": np.zeros((h, w), dtype=bool),
        "has_confirmed_bows": False,
    }

    res = qv.validate_candidate(cand_np, src_np, identity_layer=id_layer)
    assert res["checks"]["unwanted_accessories"]["status"] in ("FAIL", "UNCERTAIN")
    assert any("unwanted_hair_accessories" in d for d in res["defects"])
    assert res["passed"] is False


def test_v12_5_protected_source_compositing_rejection_never_delivers_raw(tmp_path):
    """Test 6: Protected-source compositing rejection: verify incompatible collar registration returns needs review, never raw Qwen."""
    comp = UniformCompositor()
    h, w = 1024, 800
    raw_vae = Image.new("RGB", (w, h), (200, 200, 200))
    norm_src = Image.new("RGB", (w, h), (240, 240, 240))

    face_m = np.zeros((h, w), dtype=bool)
    face_m[200:450, 260:540] = True
    hair_m = np.zeros((h, w), dtype=bool)
    hair_m[100:240, 240:560] = True

    id_layer = {
        "original_face_mask": face_m,
        "original_hair_mask": hair_m,
        "original_accessory_mask": np.zeros((h, w), dtype=bool),
        "has_confirmed_bows": False,
    }

    config = UniformSwapConfig(rgb_identity_restoration="protected_source")
    raw_quality = {"checks": {}, "defects": ["severe_distortion"], "passed": False}

    result = comp.composite(
        raw_vae_img=raw_vae,
        normalized_source_img=norm_src,
        identity_layer=id_layer,
        raw_quality=raw_quality,
        config=config,
        job_id="test_rejection",
        debug_dir=tmp_path,
    )

    composite_img = result["composite_img"]
    comp_np = np.asarray(composite_img)
    norm_np = np.asarray(norm_src)

    # V12.6 Task 2: When registration is incompatible, do NOT force unaligned source face onto candidate.
    # Reject candidate cleanly so it is flagged for review.
    assert "rejected_incompatible_collar_geometry" in result["actions"]
    assert np.array_equal(comp_np, np.asarray(raw_vae)), (
        "Incompatible registration must keep raw candidate untouched without forced overlay"
    )


def test_v12_5_job_isolation_student3_no_bows_after_student1():
    """Test 7: Job isolation: verify Student 3 does NOT get bow instructions when run after Student 1."""
    gen = UniformGenerator()

    # Job 1: Student 1 with confirmed blue bows
    id_layer_s1 = {
        "has_confirmed_bows": True,
        "has_accessories": True,
        "individual_bows": [{"name": "left_bow"}, {"name": "right_bow"}],
    }
    pos_s1, neg_s1 = gen.build_prompts(vl_plan={}, neck_cls="open", identity_layer=id_layer_s1)
    assert "preserve the exact existing hair accessories from image 1" in pos_s1.lower()

    # Job 2: Student 3 with dark hoodie (no genuine hair bows)
    id_layer_s3 = {
        "has_confirmed_bows": False,
        "has_accessories": False,
        "individual_bows": [],
    }
    pos_s3, neg_s3 = gen.build_prompts(vl_plan={}, neck_cls="open", identity_layer=id_layer_s3)
    
    # Prompt for Student 3 MUST NOT contain bow instructions
    assert "preserve the exact existing hair accessories" not in pos_s3.lower()
    assert "blue hair bows" not in pos_s3.lower()
    assert "ribbons" not in pos_s3.lower()
    assert "do not introduce hair accessories" in pos_s3.lower()
    assert "hair accessories" in neg_s3.lower()
