import os
import sys
import numpy as np
from PIL import Image
import pytest

from pipelines.uniform_swap.config import UniformSwapConfig
from pipelines.uniform_swap.uniform_generator import UniformGenerator
from pipelines.uniform_swap.conditioning_builder import ConditioningBuilder

def test_dynamic_prompt_order_independence():
    """Verify that prompt generation is strictly stateless and order-independent."""
    generator = UniformGenerator()
    vl_plan = {"shirt_color_and_pattern": "school uniform", "shirt_collar": "collar"}

    h, w = 1024, 800

    # Synthetic Student 1: long braids (hair extends to row 800, chin at row 500), bows
    f1 = np.zeros((h, w), dtype=bool)
    f1[200:500, 300:500] = True
    h1 = np.zeros((h, w), dtype=bool)
    h1[150:800, 250:550] = True
    a1 = np.zeros((h, w), dtype=bool)
    a1[300:350, 200:250] = True  # > 500 px
    a1[300:350, 550:600] = True

    id1 = {
        "original_face_mask": f1,
        "original_hair_mask": h1,
        "original_accessory_mask": a1,
    }

    # Synthetic Student 2: short hair (hair stops at row 450, chin at row 500), no accessories
    f2 = np.zeros((h, w), dtype=bool)
    f2[200:500, 300:500] = True
    h2 = np.zeros((h, w), dtype=bool)
    h2[150:450, 280:520] = True
    a2 = np.zeros((h, w), dtype=bool)

    id2 = {
        "original_face_mask": f2,
        "original_hair_mask": h2,
        "original_accessory_mask": a2,
    }

    # Sequence A: 1 then 2
    pos1_seq_a, neg1_seq_a = generator.build_prompts(vl_plan, "fully_visible_neck", identity_layer=id1)
    pos2_seq_a, neg2_seq_a = generator.build_prompts(vl_plan, "fully_visible_neck", identity_layer=id2)

    # Sequence B: 2 then 1 (reverse order)
    pos2_seq_b, neg2_seq_b = generator.build_prompts(vl_plan, "fully_visible_neck", identity_layer=id2)
    pos1_seq_b, neg1_seq_b = generator.build_prompts(vl_plan, "fully_visible_neck", identity_layer=id1)

    # Sequence C: 2 independently with fresh instance
    generator_fresh = UniformGenerator()
    pos2_fresh, neg2_fresh = generator_fresh.build_prompts(vl_plan, "fully_visible_neck", identity_layer=id2)

    assert pos1_seq_a == pos1_seq_b, "Student 1 prompt differed under reverse execution order"
    assert neg1_seq_a == neg1_seq_b, "Student 1 negative prompt differed under reverse execution order"
    assert pos2_seq_a == pos2_seq_b, "Student 2 prompt differed under reverse execution order"
    assert neg2_seq_a == neg2_seq_b, "Student 2 negative prompt differed under reverse execution order"
    assert pos2_seq_a == pos2_fresh, "Student 2 prompt differed from fresh instance"

    # Verify Student 2 specifically bans braids and accessories
    assert "short hairstyle" in pos2_seq_a
    assert "braids" in neg2_seq_a
    assert "ribbons" in neg2_seq_a
    assert "bows" in neg2_seq_a

    # Verify Student 1 preserves braids and bows
    assert "braids" in pos1_seq_a
    assert "hair bows" in pos1_seq_a

def test_conditioning_builder_order_independence():
    """Verify that conditioning builder does not retain state between runs."""
    builder = ConditioningBuilder()

    h, w = 1024, 800
    img1 = np.full((h, w, 3), 100, dtype=np.uint8)
    id1 = {
        "original_face_mask": np.zeros((h, w), dtype=bool),
        "original_neck_mask": np.zeros((h, w), dtype=bool),
        "original_garment_mask": np.zeros((h, w), dtype=bool),
        "neck_classification": "fully_visible_neck",
    }
    img2 = np.full((h, w, 3), 200, dtype=np.uint8)
    id2 = {
        "original_face_mask": np.ones((h, w), dtype=bool),
        "original_neck_mask": np.zeros((h, w), dtype=bool),
        "original_garment_mask": np.ones((h, w), dtype=bool),
        "neck_classification": "occluded_neck",
    }

    # Run Sequence A: 1 -> 2
    res1_a = builder.build_conditioning(img1, id1)
    res2_a = builder.build_conditioning(img2, id2)

    # Run Sequence B: 2 -> 1
    builder_b = ConditioningBuilder()
    res2_b = builder_b.build_conditioning(img2, id2)
    res1_b = builder_b.build_conditioning(img1, id1)

    np.testing.assert_array_equal(res1_a["generation_edit_region"], res1_b["generation_edit_region"])
    np.testing.assert_array_equal(res2_a["generation_edit_region"], res2_b["generation_edit_region"])
    np.testing.assert_array_equal(res1_a["conditioning_rgb"], res1_b["conditioning_rgb"])
    np.testing.assert_array_equal(res2_a["conditioning_rgb"], res2_b["conditioning_rgb"])
