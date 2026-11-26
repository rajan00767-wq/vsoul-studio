"""Regression test suite for dynamic background color compositing and quality validation.

Tests:
1. Dynamic background color parsing (Hex, RGB tuples, RGB strings, named colors).
2. End-to-end uniform swap with dynamic background colors (Blue, Red, Green, White, Custom Hex).
3. Reprocessing failed blue-background job d864f4d8 with candidate reuse.
4. Foreground identity, hairstyle, uniform and accessory stability across all background colors.
5. Post-crop corner background cleanliness validation against requested background color.
"""

import time
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from pipelines.uniform_swap.config import UniformSwapConfig
from pipelines.uniform_swap.orchestrator import UniformSwapOrchestrator
from pipelines.uniform_swap.quality_validator import QualityValidator


def test_dynamic_color_parsing():
    """Verify all application color formats normalize into correct (R, G, B) tuples."""
    assert UniformSwapConfig.parse_bg_color("white") == (255, 255, 255)
    assert UniformSwapConfig.parse_bg_color("blue") == (4, 126, 246)
    assert UniformSwapConfig.parse_bg_color("red") == (221, 106, 97)
    assert UniformSwapConfig.parse_bg_color("green") == (119, 191, 145)
    assert UniformSwapConfig.parse_bg_color("grey") == (214, 219, 225)
    
    # Hex with #
    assert UniformSwapConfig.parse_bg_color("#047EF6") == (4, 126, 246)
    assert UniformSwapConfig.parse_bg_color("#DD6A61") == (221, 106, 97)
    assert UniformSwapConfig.parse_bg_color("#77BF91") == (119, 191, 145)
    assert UniformSwapConfig.parse_bg_color("#FFFFFF") == (255, 255, 255)
    assert UniformSwapConfig.parse_bg_color("#8A2BE2") == (138, 43, 226)

    # Hex without #
    assert UniformSwapConfig.parse_bg_color("047EF6") == (4, 126, 246)
    assert UniformSwapConfig.parse_bg_color("dd6a61") == (221, 106, 97)

    # 3-char hex
    assert UniformSwapConfig.parse_bg_color("#FFF") == (255, 255, 255)

    # Comma-separated strings
    assert UniformSwapConfig.parse_bg_color("4,126,246") == (4, 126, 246)
    assert UniformSwapConfig.parse_bg_color("rgb(221, 106, 97)") == (221, 106, 97)

    # Tuple / list
    assert UniformSwapConfig.parse_bg_color((4, 126, 246)) == (4, 126, 246)
    assert UniformSwapConfig.parse_bg_color([119, 191, 145]) == (119, 191, 145)


def test_reprocess_failed_blue_job_d864f4d8(tmp_path):
    """Reprocess the failed blue-background job d864f4d8 reusing existing raw candidate."""
    person_path = Path("uploads/d864f4d8-18aa-4708-aaaa-e99feb16e9fc_Studio Portrait of a Young Boy (1).png")
    uniform_path = Path("uploads/d864f4d8-18aa-4708-aaaa-e99feb16e9fc_Plaid School Uniform Shirt with Emblem.png")
    raw_path = Path("outputs/qwen_uniform_d864f4d8_qwen_raw.png")

    assert person_path.exists(), f"Source image missing: {person_path}"
    assert uniform_path.exists(), f"Uniform template missing: {uniform_path}"
    assert raw_path.exists(), f"Raw candidate missing: {raw_path}"

    config = UniformSwapConfig(
        background_color=(4, 126, 246),  # Blue
        rgb_identity_restoration="adaptive",
        raw_diffusion_cache_policy="reuse_if_available",
    )

    orchestrator = UniformSwapOrchestrator()
    t0 = time.time()
    res = orchestrator.run(
        person_image_input=person_path,
        uniform_template_input=uniform_path,
        config=config,
        job_id="test_reprocess_d864f4d8_blue",
        output_dir=tmp_path / "outputs",
        debug_dir=tmp_path / "debug_phases",
        cached_raw_candidate=raw_path,
    )
    elapsed = time.time() - t0

    assert res.passed is True, f"Reprocessed blue job failed: defects={res.defects}"
    assert res.status in ("completed", "delivered")
    assert res.delivered_path is not None and res.delivered_path.exists()
    stage_names = [s.get("stage") if isinstance(s, dict) else s for s in res.trace.get("stages", [])]
    assert "qwen_diffusion_and_raw_decode" in stage_names
    # Verify execution took only a few seconds since Qwen was not rerun (allowing up to 45s for model cold-start)
    assert elapsed < 45.0, f"Expected fast candidate reuse, took {elapsed:.2f}s"

    # Verify delivered crop corner color is blue [4, 126, 246]
    crop_arr = np.asarray(Image.open(res.delivered_path))
    corner_tl = crop_arr[:30, :30]
    corner_mean = np.mean(corner_tl, axis=(0, 1))
    assert abs(corner_mean[0] - 4) < 10, f"R channel error: {corner_mean[0]}"
    assert abs(corner_mean[1] - 126) < 10, f"G channel error: {corner_mean[1]}"
    assert abs(corner_mean[2] - 246) < 10, f"B channel error: {corner_mean[2]}"

    # Verify no rectangular white canvas padding or white background patch remains
    crop_meta = getattr(res, "crop_metadata", None) or res.validation_report.get("crop_metadata", {})
    assert crop_meta.get("padding") == (0, 0, 0, 0), f"Expected zero padding, got {crop_meta.get('padding')}"
    white_px = (crop_arr[:, :, 0] > 230) & (crop_arr[:, :, 1] > 230) & (crop_arr[:, :, 2] > 230)
    assert np.sum(white_px) < 5000, f"Leaked white background detected: {np.sum(white_px)} px"

    # Verify blue background is uniform across background
    blue_px = (crop_arr[:, :, 0] < 20) & (crop_arr[:, :, 1] > 110) & (crop_arr[:, :, 2] > 230)
    assert np.sum(blue_px) > 250000, f"Expected blue background across canvas, got {np.sum(blue_px)} px"


@pytest.mark.parametrize("color_name,color_val,expected_rgb", [
    ("Blue", "#047EF6", (4, 126, 246)),
    ("Red", "#DD6A61", (221, 106, 97)),
    ("Green", "#77BF91", (119, 191, 145)),
    ("White", "#FFFFFF", (255, 255, 255)),
    ("CustomPurple", "#8A2BE2", (138, 43, 226)),
])
def test_all_background_colors_pass_validation(tmp_path, color_name, color_val, expected_rgb):
    """Test every application background color preset + custom hex."""
    person_path = Path("uploads/d864f4d8-18aa-4708-aaaa-e99feb16e9fc_Studio Portrait of a Young Boy (1).png")
    uniform_path = Path("uploads/d864f4d8-18aa-4708-aaaa-e99feb16e9fc_Plaid School Uniform Shirt with Emblem.png")
    raw_path = Path("outputs/qwen_uniform_d864f4d8_qwen_raw.png")

    config = UniformSwapConfig(
        background_color=color_val,
        rgb_identity_restoration="adaptive",
        raw_diffusion_cache_policy="reuse_if_available",
    )
    assert config.background_color == expected_rgb

    orchestrator = UniformSwapOrchestrator()
    res = orchestrator.run(
        person_image_input=person_path,
        uniform_template_input=uniform_path,
        config=config,
        job_id=f"test_color_{color_name}",
        output_dir=tmp_path / "outputs",
        debug_dir=tmp_path / "debug_phases",
        cached_raw_candidate=raw_path,
    )

    assert res.passed is True, f"Failed for {color_name} ({color_val}): defects={res.defects}"
    assert res.status in ("completed", "delivered")
    assert res.validation_report["pre_crop_passed"] is True
    assert res.validation_report["post_crop_passed"] is True
    assert res.validation_report["similarity"] >= 0.82
    assert res.validation_report["geometry_rmse"] <= 0.040

    # Verify delivered crop corner matches expected background color
    crop_arr = np.asarray(Image.open(res.delivered_path))
    corner_tl = crop_arr[:30, :30]
    corner_mean = np.mean(corner_tl, axis=(0, 1))
    for ch in range(3):
        assert abs(corner_mean[ch] - expected_rgb[ch]) < 10, (
            f"Corner color mismatch for {color_name} channel {ch}: got {corner_mean[ch]}, expected {expected_rgb[ch]}"
        )


def test_post_crop_catches_white_rectangular_leak():
    """Verify that validate_post_crop strictly rejects images with rectangular white background leaks."""
    from pipelines.uniform_swap.quality_validator import QualityValidator
    qv = QualityValidator()

    # Create synthetic delivered image: outer border blue [4, 126, 246], inner rectangle white [255, 255, 255]
    h, w = 1024, 800
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:, :] = [4, 126, 246]
    # Simulate leaked inner white portrait rectangle like in diagnostic_crop.png
    canvas[84:956, 59:740] = [255, 255, 255]

    # Transformed alpha: person in the center
    synth_alpha = np.zeros((h, w), dtype=np.uint8)
    synth_alpha[250:800, 200:600] = 255

    crop_meta = {
        "padding": (69, 98, 69, 78),
        "background_rgb": (4, 126, 246),
        "transformed_person_alpha": synth_alpha,
    }

    report = qv.validate_post_crop(
        cropped_image=canvas,
        crop_meta=crop_meta,
        expected_size=(800, 1024),
        expected_bg_rgb=(4, 126, 246),
    )

    assert report["passed"] is False
    assert any("dirty_background_in_delivered_crop" in d for d in report["defects"])
