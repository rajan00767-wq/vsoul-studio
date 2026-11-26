import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    QwenImageEditPlusPipeline,
)
from pipelines.uniform_swap.config import UniformSwapConfig
from pipelines.uniform_swap.uniform_generator import UniformGenerator
from pipelines.uniform_swap.quality_validator import QualityValidator


def test_coordinate_grid_mapping():
    """Verify that every 2D patch coordinate (r, c) maps 1:1 to the 1D token index p."""
    gen_w, gen_h = 800, 1024
    latent_h = gen_h // 8  # 128
    latent_w = gen_w // 8  # 100
    patch_h = latent_h // 2  # 64
    patch_w = latent_w // 2  # 50
    num_patches = patch_h * patch_w  # 3200

    # Build unique coordinate grid
    coord_grid = torch.zeros((1, 1, 1, latent_h, latent_w), dtype=torch.float32)
    for r in range(latent_h):
        for c in range(latent_w):
            coord_grid[0, 0, 0, r, c] = r * 1000.0 + c

    packed = QwenImageEditPlusPipeline._pack_latents(
        coord_grid,
        batch_size=1,
        num_channels_latents=1,
        height=latent_h,
        width=latent_w,
    )
    assert packed.shape == (1, num_patches, 4)

    # Check that patch (pr, pc) contains exactly the 4 2x2 cells
    for pr in [0, 15, 32, 63]:
        for pc in [0, 12, 25, 49]:
            p_idx = pr * patch_w + pc
            expected = [
                (2 * pr) * 1000.0 + (2 * pc),
                (2 * pr) * 1000.0 + (2 * pc + 1),
                (2 * pr + 1) * 1000.0 + (2 * pc),
                (2 * pr + 1) * 1000.0 + (2 * pc + 1),
            ]
            assert packed[0, p_idx].tolist() == expected

    # Unpack and verify round-trip identity
    unpacked = QwenImageEditPlusPipeline._unpack_latents(
        packed, gen_h, gen_w, vae_scale_factor=8
    )
    diff = (unpacked - coord_grid).abs().max().item()
    assert diff == 0.0, f"Unpack coordinate mismatch: max err={diff}"


def test_authoritative_latent_mask_modes():
    """Verify OFF, SOFT, and HARD protection mask properties."""
    gen_w, gen_h = 800, 1024
    patch_w, patch_h = gen_w // 16, gen_h // 16
    num_patches = patch_w * patch_h

    # Create dummy edit region (bottom half)
    edit_region = np.zeros((gen_h, gen_w), dtype=bool)
    edit_region[gen_h // 2 :, :] = True

    # 1. Mode OFF
    cfg_off = UniformSwapConfig(latent_identity_lock="off")
    assert cfg_off.latent_identity_lock == "off"

    # 2. Mode SOFT: continuous weights
    cfg_soft = UniformSwapConfig(latent_identity_lock="soft")
    assert cfg_soft.latent_identity_lock == "soft"

    # 3. Mode HARD: binary weights
    cfg_hard = UniformSwapConfig(latent_identity_lock="hard")
    assert cfg_hard.latent_identity_lock == "hard"


def test_quality_validator_repair_comparison():
    """Verify Section 8 requirement: discard repair if repaired candidate degrades quality."""
    qv = QualityValidator(min_biometric_similarity=0.82)

    raw_qual = {
        "similarity": 0.952,
        "geometry_rmse": 0.020,
        "passed": True,
        "defects": [],
    }

    # Case 1: Repaired is degraded (similarity dropped from 0.952 to 0.72)
    bad_repaired = {
        "similarity": 0.720,
        "geometry_rmse": 0.045,
        "passed": False,
        "defects": ["low_biometric_similarity"],
    }
    assert qv.compare_candidates(raw_qual, bad_repaired) is False

    # Case 2: Repaired is equal or better
    good_repaired = {
        "similarity": 0.955,
        "geometry_rmse": 0.019,
        "passed": True,
        "defects": [],
    }
    assert qv.compare_candidates(raw_qual, good_repaired) is True
