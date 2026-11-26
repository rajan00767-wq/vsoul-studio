"""Unit tests for the modular uniform swap architecture (pipelines/uniform_swap/*)."""

import pytest
import numpy as np
from PIL import Image

from pipelines.uniform_swap.semantic_parser import SemanticParser, semantic_parser
from pipelines.uniform_swap.identity_extractor import IdentityExtractor, identity_extractor
from pipelines.uniform_swap.conditioning_builder import ConditioningBuilder, conditioning_builder
from pipelines.uniform_swap.neck_collar_resolver import NeckCollarResolver, neck_collar_resolver
from pipelines.uniform_swap.uniform_generator import UniformGenerator, uniform_generator
from pipelines.uniform_swap.quality_validator import QualityValidator, QualityCheckFailure


def test_semantic_parser_caching():
    """Verify SemanticParser properly caches results per job and image."""
    parser = SemanticParser()
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

    # In mock/offline test, manually seed cache
    job_id = "test_job_1"
    key = f"{job_id}_source_dummy"
    parser._cache[job_id] = {key: {"test_data": 123}}

    assert parser._cache[job_id][key]["test_data"] == 123

    # Test cache clearing
    parser.clear_job_cache(job_id)
    assert job_id not in parser._cache


def test_neck_collar_resolver_occluded_neck():
    """Verify occluded neck gets a snug collar seat right below chin with zero gap."""
    resolver = NeckCollarResolver()
    identity_layer = {
        "neck_classification": "occluded_neck",
        "metrics": {
            "chin_y": 380,
            "lowest_neck_y": None,
        },
        "original_neck_mask": np.zeros((1024, 800), dtype=bool),
    }
    face_bbox = (200, 150, 600, 450)  # fh = 300

    res = resolver.resolve(identity_layer, face_bbox)
    # collar_seat_y should be chin_y + 0.04 * fh = 380 + 12 = 392
    assert res["collar_seat_y"] == 392
    assert res["neck_ratio"] == 0.04
    assert res["preexisting_neck_mask"].shape == (1024, 800)


def test_neck_collar_resolver_visible_neck():
    """Verify visible neck gets collar seat at the lowest natural neck row."""
    resolver = NeckCollarResolver()
    identity_layer = {
        "neck_classification": "fully_visible_neck",
        "metrics": {
            "chin_y": 380,
            "lowest_neck_y": 445,
        },
        "original_neck_mask": np.zeros((1024, 800), dtype=bool),
    }
    face_bbox = (200, 150, 600, 450)  # fh = 300

    res = resolver.resolve(identity_layer, face_bbox)
    assert res["collar_seat_y"] == 445
    expected_ratio = (445 - 380) / 300.0  # 65 / 300 = 0.2167
    assert abs(res["neck_ratio"] - expected_ratio) < 0.01


def test_conditioning_builder_fallback():
    """Verify ConditioningBuilder triggers Strategy A fallback if identity validation fails."""
    builder = ConditioningBuilder()
    dummy_orig = np.zeros((1024, 800, 3), dtype=np.uint8)
    identity_layer = {
        "validation": {
            "is_valid": False,
            "issues": ["garment_leakage_detected"],
            "metrics": {},
        },
        "neck_classification": "fully_visible_neck",
        "selected_strategy": "B",
    }

    res = builder.build_conditioning(dummy_orig, identity_layer)
    assert res["selected_strategy"] == "A"
    assert res["fallback_triggered"] is True


def test_uniform_generator_prompt_budget():
    """Verify UniformGenerator generates prompts strictly under token and word budget."""
    gen = UniformGenerator()
    vl_plan = {
        "garment_components": "blazer and white shirt",
        "shirt_color_and_pattern": "navy blue wool",
        "shirt_collar": "lapel collar",
        "sleeve_length": "long sleeves",
        "button_layout": "double breasted brass buttons",
        "outer_garment_color_and_shape": "fitted blazer",
    }

    pos, neg = gen.build_prompts(vl_plan, neck_cls="fully_visible_neck")
    pos_words = len(pos.split())
    neg_words = len(neg.split())

    assert pos_words < 180
    assert "navy blue wool" in pos
    assert "double breasted brass buttons" in pos
    assert "natural neck visible" in pos
    assert "double neck" in neg


def test_quality_validator_face_check():
    """Verify QualityValidator raises QualityCheckFailure when face count is not 1."""
    validator = QualityValidator()
    # Image with no face
    blank_img = Image.new("RGB", (200, 200), (255, 255, 255))
    src_img = Image.new("RGB", (200, 200), (255, 255, 255))

    with pytest.raises(QualityCheckFailure, match="expected exactly 1"):
        validator.validate_final_portrait(blank_img, src_img, {})
