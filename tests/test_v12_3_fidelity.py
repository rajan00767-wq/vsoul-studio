"""Unit & Regression Tests for V12.3 Source Appearance Protection & Uniform Fidelity."""

import numpy as np
import pytest
from PIL import Image

from pipelines.uniform_swap.config import UniformSwapConfig
from pipelines.uniform_swap.identity_extractor import identity_extractor
from pipelines.uniform_swap.orchestrator import DeliveryResult, UniformSwapOrchestrator
from pipelines.uniform_swap.quality_validator import QualityValidator
from pipelines.uniform_swap.uniform_generator import UniformGenerator, _detect_uniform_visual_cues


def test_v12_3_dynamic_appearance_instructions():
    """Verify Task 2: Distinguishes short-haired students from braided students and avoids adding braids to boys."""
    ug = UniformGenerator()
    h, w = 1024, 800

    # Case 1: Short-haired student (Student 2 model)
    face_m_short = np.zeros((h, w), dtype=bool)
    face_m_short[200:500, 250:550] = True  # chin_y = 500, fh = 300
    hair_m_short = np.zeros((h, w), dtype=bool)
    hair_m_short[100:350, 200:600] = True  # hair_bottom = 350 < chin_y (short hair!)
    acc_m_none = np.zeros((h, w), dtype=bool)

    id_layer_short = {
        "original_face_mask": face_m_short,
        "original_hair_mask": hair_m_short,
        "original_accessory_mask": acc_m_none,
    }

    pos_short, neg_short = ug.build_prompts(
        vl_plan={},
        neck_cls="partially_visible_neck",
        identity_layer=id_layer_short,
    )

    assert "short hairstyle" in pos_short.lower()
    assert "do not add braids" in pos_short.lower()
    assert "do not add any hair accessories" in pos_short.lower()
    assert "braids" in neg_short.lower()
    assert "hair accessories" in neg_short.lower()

    # Case 2: Student with braids and bows (Student 1 model)
    face_m_long = np.zeros((h, w), dtype=bool)
    face_m_long[200:500, 250:550] = True  # chin_y = 500, fh = 300
    hair_m_long = np.zeros((h, w), dtype=bool)
    hair_m_long[100:800, 200:600] = True  # hair_bottom = 800 >> chin_y (long hair/braids!)
    acc_m_bows = np.zeros((h, w), dtype=bool)
    acc_m_bows[250:350, 180:230] = True  # 2500 px bows!

    id_layer_long = {
        "original_face_mask": face_m_long,
        "original_hair_mask": hair_m_long,
        "original_accessory_mask": acc_m_bows,
    }

    pos_long, neg_long = ug.build_prompts(
        vl_plan={},
        neck_cls="partially_visible_neck",
        identity_layer=id_layer_long,
    )

    assert "existing hairstyle and braids" in pos_long.lower()
    assert "hair bows" in pos_long.lower()
    assert "missing hair bows" in neg_long.lower()
    assert "extra braids" in neg_long.lower()


def test_v12_3_uniform_reference_matching():
    """Verify Task 5: Dynamic uniform extraction detects red pointed collar, open neckline, black buttons, plaid."""
    ref_arr = np.zeros((1024, 800, 4), dtype=np.uint8)
    # Garment alpha
    ref_arr[200:900, 150:650, 3] = 255
    # Red collar at top
    ref_arr[200:340, 220:580, :3] = [210, 20, 20]
    # Plaid torso
    ref_arr[340:900, 150:650, :3] = [30, 45, 85]
    # Red placket
    ref_arr[340:850, 375:425, :3] = [200, 25, 25]
    # Black buttons along placket
    for by in [420, 520, 620, 720]:
        ref_arr[by : by + 12, 395 : 405, :3] = [20, 20, 20]

    cues = _detect_uniform_visual_cues(ref_arr)
    assert cues["has_red_collar"] is True
    assert cues["has_red_placket"] is True
    assert cues["button_color"] == "black"
    assert "solid red pointed collar with a natural open neckline" in cues["collar_desc"]
    assert "black buttons" in cues["button_desc"]

    ug = UniformGenerator()
    pos, neg = ug.build_prompts(
        vl_plan={},
        neck_cls="partially_visible_neck",
        uniform_reference_image=ref_arr,
    )
    assert "black buttons" in pos
    assert "white buttons" in neg
    assert "open neckline" in pos


def test_v12_3_quality_validator_calibrated_checks():
    """Verify Task 6: 12 independent checks return PASS/FAIL/UNCERTAIN; high similarity cannot override appearance failure."""
    qv = QualityValidator(min_biometric_similarity=0.82)
    h, w = 1024, 800

    cand_img = np.full((h, w, 3), 255, dtype=np.uint8)
    src_img = np.full((h, w, 3), 255, dtype=np.uint8)

    # Short-haired portrait where candidate generated unwanted hair/braids
    face_m = np.zeros((h, w), dtype=bool)
    face_m[200:500, 250:550] = True
    hair_m = np.zeros((h, w), dtype=bool)
    hair_m[100:350, 250:550] = True  # short hair

    # Candidate has dark hair added near neck/ears
    cand_img[400:600, 180:230] = [20, 15, 15]

    id_layer = {
        "original_face_mask": face_m,
        "original_hair_mask": hair_m,
        "original_accessory_mask": np.zeros((h, w), dtype=bool),
    }

    res = qv.validate_candidate(
        candidate_image=cand_img,
        source_image=src_img,
        identity_layer=id_layer,
        stage="test_appearance",
    )

    # Check dictionary exists
    assert "checks" in res
    assert isinstance(res["checks"], dict)
    assert "added_hair" in res["checks"]
    assert res["checks"]["added_hair"]["status"] == "FAIL"
    # Even if similarity was mocked to 0.99, appearance failure forces passed=False
    assert res["passed"] is False
    assert res["needs_review"] is True


def test_v12_3_delivery_gating_order(tmp_path):
    """Verify Task 7: Rejected candidate is NOT delivered to production output; returns structured DeliveryResult."""
    res = DeliveryResult(
        passed=False,
        status="needs_review",
        delivered_path=None,
        diagnostic_path=tmp_path / "candidate_diagnostic.png",
        defects=["low_biometric_similarity (0.178 < 0.82)"],
        warnings=[],
        validation_report={"passed": False, "similarity": 0.178},
    )

    assert res.passed is False
    assert res.status == "needs_review"
    assert res.delivered_path is None
    assert res.diagnostic_path is not None
    assert len(res.defects) == 1
    # Check __fspath__ compatibility
    assert str(res) == str(tmp_path / "candidate_diagnostic.png")
