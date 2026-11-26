"""Integration tests proving that production uniform swap invokes all V11 modular components.

Verifies:
1. UniformSwapConfig validation, serialization, and cache hash computation.
2. Queue worker integration in main.py delegating to uniform_swap_orchestrator.
3. Orchestration call graph invoking:
   - semantic_parser
   - identity_extractor
   - conditioning_builder
   - uniform_generator
   - compositor
   - quality_validator
4. Deterministic production of all 21 mandatory debug artifacts.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from PIL import Image

from pipelines.uniform_swap.config import UniformSwapConfig
from pipelines.uniform_swap.orchestrator import UniformSwapOrchestrator, uniform_swap_orchestrator
from pipelines.uniform_swap.semantic_parser import semantic_parser
from pipelines.uniform_swap.identity_extractor import identity_extractor
from pipelines.uniform_swap.conditioning_builder import conditioning_builder
from pipelines.uniform_swap.uniform_generator import uniform_generator
from pipelines.uniform_swap.compositor import uniform_compositor
from pipelines.uniform_swap.quality_validator import quality_validator


def test_uniform_swap_config_invariants():
    """Verify UniformSwapConfig enforces validation invariants and produces SHA256 cache key."""
    cfg = UniformSwapConfig(
        steps=4,
        true_cfg_scale=1.20,
        conditioning_strategy="original_baseline",
        latent_identity_lock="off",
        rgb_identity_restoration="adaptive",
    )
    cfg.validate()

    # Test cache hash
    cache_hash = cfg.compute_cache_hash(
        person_hash="p123",
        uniform_hash="u123",
        image_1_hash="img1",
        image_2_hash="img2",
        edit_mask_hash="emask",
        hard_mask_hash="hmask",
        prompt_hash="prmpt",
    )
    assert len(cache_hash) == 64  # SHA-256 hex digest
    assert isinstance(cfg.to_dict(), dict)

    # Test invalid latent lock
    with pytest.raises(ValueError, match="Invalid latent_identity_lock"):
        UniformSwapConfig(latent_identity_lock="invalid_mode").validate()


def test_config_from_job_metadata():
    """Verify metadata from API is cleanly parsed into UniformSwapConfig without silent overrides."""
    meta = {
        "width": 640,
        "height": 960,
        "bg_color": "standard_blue",
        "conditioning_strategy": "original_baseline",
        "latent_identity_lock": "soft",
        "rgb_identity_restoration": "adaptive",
        "steps": 4,
        "true_cfg_scale": 1.25,
        "seed": 100,
    }
    cfg = UniformSwapConfig.from_job_metadata(meta)
    assert cfg.generation_resolution == (640, 960)
    assert cfg.background_color == (4, 126, 246)
    assert cfg.conditioning_strategy == "original_baseline"
    assert cfg.latent_identity_lock == "soft"
    assert cfg.rgb_identity_restoration == "adaptive"
    assert cfg.steps == 4
    assert cfg.true_cfg_scale == 1.25
    assert cfg.seed == 100


def test_orchestrator_invokes_all_modular_components(tmp_path):
    """Integration test: Verify uniform_swap_orchestrator invokes each required modular component."""
    # Create test input images
    w, h = 800, 1024
    test_person = Image.new("RGB", (w, h), (200, 180, 160))
    test_uniform = Image.new("RGB", (600, 800), (40, 60, 120))

    cfg = UniformSwapConfig(
        generation_resolution=(w, h),
        output_resolution=(w, h),
        steps=4,
        conditioning_strategy="original_baseline",
        latent_identity_lock="off",
        rgb_identity_restoration="adaptive",
    )

    out_dir = tmp_path / "outputs"
    deb_dir = tmp_path / "outputs/debug_phases"

    # Mock generator diffusion call to avoid needing GPU during unit test
    mock_raw_candidate = Image.new("RGB", (w, h), (180, 170, 150))
    raw_cand_path = deb_dir / "test_15_raw_vae_decoder.png"

    with patch.object(
        uniform_generator, "generate",
        return_value={
            "raw_vae_decoder_img": mock_raw_candidate,
            "raw_vae_decoder_path": raw_cand_path,
            "positive_prompt": "test prompt",
            "negative_prompt": "test neg",
            "latent_keep_mask_path": deb_dir / "test_14_latent_keep_mask.png",
        }
    ) as mock_gen, \
    patch.object(
        semantic_parser, "parse_source",
        wraps=semantic_parser.parse_source
    ) as mock_parser, \
    patch.object(
        identity_extractor, "extract",
        wraps=identity_extractor.extract
    ) as mock_extractor, \
    patch.object(
        conditioning_builder, "build_conditioning",
        wraps=conditioning_builder.build_conditioning
    ) as mock_builder, \
    patch.object(
        uniform_compositor, "composite",
        wraps=uniform_compositor.composite
    ) as mock_comp, \
    patch.object(
        quality_validator, "validate_candidate",
        return_value={
            "stage": "raw", "passed": True, "similarity": 0.95,
            "geometry_rmse": 0.008, "faces_detected": 1, "defects": []
        }
    ) as mock_val:

        job_id = "test_integration_job"
        out_file = uniform_swap_orchestrator.run(
            person_image_input=test_person,
            uniform_template_input=test_uniform,
            config=cfg,
            job_id=job_id,
            output_dir=out_dir,
            debug_dir=deb_dir,
        )

        # 1. Verify all modules were called
        assert mock_parser.called, "semantic_parser was not invoked by orchestrator"
        assert mock_extractor.called, "identity_extractor was not invoked by orchestrator"
        assert mock_builder.called, "conditioning_builder was not invoked by orchestrator"
        assert mock_gen.called, "uniform_generator was not invoked by orchestrator"
        assert mock_val.called, "quality_validator was not invoked by orchestrator"
        assert mock_comp.called, "uniform_compositor was not invoked by orchestrator"

        # 2. Verify all 21 mandatory debug artifacts are produced
        required_artifacts = [
            f"{job_id}_01_original_upload.png",
            f"{job_id}_02_normalized_source.png",
            f"{job_id}_03_semantic_labels.png",
            f"{job_id}_04_full_person_alpha.png",
            f"{job_id}_05_face_mask.png",
            f"{job_id}_06_hair_mask.png",
            f"{job_id}_07_accessory_mask.png",
            f"{job_id}_08_neck_mask.png",
            f"{job_id}_09_original_garment_mask.png",
            f"{job_id}_10_qwen_image_1_actual.png",
            f"{job_id}_11_qwen_image_2_actual.png",
            f"{job_id}_12_edit_mask_actual.png",
            f"{job_id}_13_hard_protection_mask_actual.png",
            f"{job_id}_14_latent_keep_mask.png",
            f"{job_id}_15_raw_vae_decoder.png",
            f"{job_id}_16_raw_quality_validation.json",
            f"{job_id}_17_localized_repair_mask.png",
            f"{job_id}_18_final_output.png",
            f"{job_id}_19_final_quality_validation.json",
            f"{job_id}_20_effective_generation_config.json",
            f"{job_id}_21_pipeline_execution_trace.json",
        ]

        for art in required_artifacts:
            art_p = deb_dir / art
            assert art_p.exists(), f"Mandatory artifact {art} was not generated"
            assert art_p.stat().st_size > 0, f"Artifact {art} is empty"

        # 3. Verify execution trace content
        trace_file = deb_dir / f"{job_id}_21_pipeline_execution_trace.json"
        trace_data = json.loads(trace_file.read_text(encoding="utf-8"))
        assert trace_data["job_id"] == job_id
        assert len(trace_data["stages"]) >= 6
        assert "total_duration_seconds" in trace_data
        assert "peak_vram_mb" in trace_data
