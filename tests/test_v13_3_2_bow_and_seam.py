"""Unit tests for instance-level accessory preservation and internal garment seam detection (V13.3.2).

Validates:
- Chest-level braid accessories (large area, contrasting with garment) are extracted as accessories
- Individual bow preservation gate requires each individual accessory to survive
- Internal horizontal rectangular chest seam is caught by rectangular_garment_boundary gate
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pipelines.uniform_swap.identity_extractor import IdentityExtractor
from pipelines.uniform_swap.quality_validator import QualityValidator


def test_chest_level_braid_bow_extraction():
    """Verify that braid bows resting on the chest are detected as hair accessories, not badges."""
    h, w = 600, 500
    rgb = np.full((h, w, 3), (200, 180, 160), dtype=np.uint8)  # Tan garment body

    # Draw hair down sides (braids)
    hair_mask = np.zeros((h, w), dtype=bool)
    hair_mask[100:450, 80:120] = True   # Left braid
    hair_mask[100:450, 380:420] = True  # Right braid
    rgb[hair_mask] = (30, 25, 20)       # Dark hair

    face_bbox = (150, 80, 350, 280)
    face_mask = np.zeros((h, w), dtype=bool)
    face_mask[100:260, 170:330] = True
    neck_mask = np.zeros((h, w), dtype=bool)
    neck_mask[260:320, 200:300] = True

    # Draw two large maroon bows on the chest at the bottom of the braids (area ~3000 px each)
    cv2.ellipse(rgb, (100, 430), (40, 25), 0, 0, 360, (120, 20, 30), -1)  # Left maroon bow
    cv2.ellipse(rgb, (400, 430), (40, 25), 0, 0, 360, (120, 20, 30), -1)  # Right maroon bow

    # SCHP clothing labels (5 = upper clothes on torso)
    schp_labels = np.zeros((h, w), dtype=np.int32)
    schp_labels[300:, 50:450] = 5  # Torso is clothing

    ext = IdentityExtractor()
    acc_mask, acc_objs, badge_mask, badge_objs = ext.extract_semantic_hair_accessories(
        rgb=rgb,
        hair_mask=hair_mask,
        face_mask=face_mask,
        neck_mask=neck_mask,
        schp_labels=schp_labels,
        bisenet_parsing=None,
        birefnet_alpha=None,
        face_bbox=face_bbox,
    )

    # Both maroon bows should be extracted as hair accessories, NOT badges!
    assert len(acc_objs) >= 2
    assert len(badge_objs) == 0
    assert np.sum(acc_mask) > 3000


def test_asymmetric_bow_loss_fails_gate():
    """Verify that erasing one of two original bows triggers a quality gate defect."""
    qv = QualityValidator()
    h, w = 400, 400

    cand_rgb = np.full((h, w, 3), (255, 255, 255), dtype=np.uint8)
    src_rgb = np.full((h, w, 3), (255, 255, 255), dtype=np.uint8)

    # Source has two bows: left and right
    b1_m = np.zeros((h, w), dtype=bool)
    b1_m[200:260, 80:140] = True
    src_rgb[b1_m] = (140, 20, 30)

    b2_m = np.zeros((h, w), dtype=bool)
    b2_m[200:260, 260:320] = True
    src_rgb[b2_m] = (140, 20, 30)

    # Candidate preserves right bow, but left bow is ERASED (replaced by white uniform)
    cand_rgb[b2_m] = (140, 20, 30)  # Right bow survives
    cand_rgb[b1_m] = (255, 255, 255)  # Left bow erased!

    id_layer = {
        "individual_bows": [
            {"name": "left_bow", "mask": b1_m, "area": int(np.sum(b1_m)), "centroid": [110, 230]},
            {"name": "right_bow", "mask": b2_m, "area": int(np.sum(b2_m)), "centroid": [290, 230]},
        ],
        "has_confirmed_bows": True,
    }

    res = qv.validate_candidate(
        candidate_image=cand_rgb,
        source_image=src_rgb,
        identity_layer=id_layer,
        uniform_reference=cand_rgb,
    )

    # Must FAIL because left bow is missing!
    bow_chk = res.get("checks", {}).get("bow_and_ribbon_preservation", {})
    assert bow_chk.get("status") == "FAIL"
    assert any("missing_left_bow" in d for d in res.get("defects", []))
    assert res.get("passed") is False


def test_internal_rectangular_chest_seam_detection():
    """Verify that a sharp horizontal rectangular chest seam is caught by quality gates."""
    qv = QualityValidator()
    h, w = 600, 500

    cand_rgb = np.full((h, w, 3), (240, 240, 240), dtype=np.uint8)
    cand_box = [150, 60, 350, 260]

    # Draw uniform torso
    cand_rgb[280:550, 100:400] = (30, 45, 80)  # Navy uniform

    # Inject sharp horizontal seam across the chest at y = 400:
    # Above y = 400: light peach skin/shirt [220, 160, 120]
    # Below y = 400: deep navy [30, 45, 80]
    cand_rgb[280:400, 160:340] = (220, 160, 120)

    res = qv.validate_candidate(
        candidate_image=cand_rgb,
        source_image=cand_rgb,
        identity_layer={"full_person_alpha": np.full((h, w), 255, dtype=np.uint8), "face_bbox": cand_box},
        uniform_reference=cand_rgb,
    )

    rect_chk = res.get("checks", {}).get("rectangular_garment_boundary", {})
    assert rect_chk.get("status") == "FAIL"
    assert "rectangular_garment_boundary" in res.get("defects", [])
    assert res.get("passed") is False
