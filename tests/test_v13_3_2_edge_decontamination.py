"""Unit tests for background-independent edge contamination detection and decontamination (V13.3.2).

Validates:
- Backdrop color estimation on synthetic studio backgrounds (blue, green, red)
- Boundary trimap construction
- Colored edge halo detection (PASS on clean, FAIL on injected backdrop spill)
- Edge decontamination unmixing and alpha attenuation
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pipelines.uniform_swap.edge_decontamination import (
    build_boundary_trimap,
    decontaminate_edges,
    default_band_px,
    detect_colored_edge_halo,
    estimate_source_background_color,
)


def create_synthetic_subject_with_halo(
    backdrop_rgb: tuple[int, int, int] = (0, 129, 253),
    foreground_rgb: tuple[int, int, int] = (40, 35, 30),
    halo_width: int = 6,
    h: int = 400,
    w: int = 400,
):
    """Create a synthetic dark-haired circular subject with colored backdrop spill along boundary."""
    canvas = np.full((h, w, 3), (255, 255, 255), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.uint8)

    center = (w // 2, h // 2)
    radius = 120

    # Draw solid foreground
    cv2.circle(canvas, center, radius, foreground_rgb, -1)
    cv2.circle(alpha, center, radius, 255, -1)

    # Inject colored backdrop fringe along outer boundary
    outer_band = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(outer_band, center, radius, 255, halo_width)

    # Blend fringe with backdrop color
    b_col = np.array(backdrop_rgb, dtype=np.float32)
    f_col = np.array(foreground_rgb, dtype=np.float32)
    blend_col = (0.7 * b_col + 0.3 * f_col).astype(np.uint8)

    canvas[outer_band > 0] = blend_col
    alpha[outer_band > 0] = 180

    hair_mask = np.zeros((h, w), dtype=bool)
    cv2.circle(hair_mask.astype(np.uint8), center, radius, 1, -1)
    hair_mask = hair_mask.astype(bool)

    return canvas, alpha, hair_mask


def test_estimate_source_background_color_blue():
    """Verify background color estimation on blue studio backdrop."""
    h, w = 300, 300
    src = np.full((h, w, 3), (0, 129, 253), dtype=np.uint8)
    cv2.circle(src, (150, 150), 60, (40, 30, 20), -1)

    est_bg, conf = estimate_source_background_color(src)
    assert est_bg is not None
    assert conf > 0.70
    assert abs(est_bg[0] - 0) < 5
    assert abs(est_bg[1] - 129) < 5
    assert abs(est_bg[2] - 253) < 5


def test_estimate_source_background_color_green():
    """Verify background color estimation on green screen backdrop."""
    h, w = 300, 300
    src = np.full((h, w, 3), (0, 215, 60), dtype=np.uint8)
    cv2.circle(src, (150, 150), 60, (40, 30, 20), -1)

    est_bg, conf = estimate_source_background_color(src)
    assert est_bg is not None
    assert conf > 0.70
    assert abs(est_bg[0] - 0) < 5
    assert abs(est_bg[1] - 215) < 5
    assert abs(est_bg[2] - 60) < 5


def test_build_boundary_trimap():
    """Verify boundary trimap creates disjoint definite foreground, background, and boundary."""
    alpha = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(alpha, (100, 100), 50, 255, -1)

    tri = build_boundary_trimap(alpha, band_px=9)
    assert "definite_fg" in tri
    assert "definite_bg" in tri
    assert "boundary" in tri

    # Disjointness
    assert not np.any(tri["definite_fg"] & tri["boundary"])
    assert not np.any(tri["definite_bg"] & tri["boundary"])
    assert not np.any(tri["definite_fg"] & tri["definite_bg"])
    assert np.any(tri["definite_fg"])
    assert np.any(tri["boundary"])


def test_detect_colored_edge_halo_positive_and_negative():
    """Test halo detector: PASS on clean subject, FAIL on injected blue fringe."""
    # 1. Clean subject (no backdrop spill)
    clean_rgb = np.full((300, 300, 3), (255, 255, 255), dtype=np.uint8)
    clean_alpha = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(clean_rgb, (150, 150), 80, (40, 35, 30), -1)
    cv2.circle(clean_alpha, (150, 150), 80, 255, -1)

    blue_bg = (0.0, 129.0, 253.0)
    det_clean = detect_colored_edge_halo(clean_rgb, clean_alpha, blue_bg)
    assert det_clean["status"] == "PASS"

    # 2. Defective subject with blue spill along edge
    spill_rgb, spill_alpha, hair_m = create_synthetic_subject_with_halo(
        backdrop_rgb=(0, 129, 253), halo_width=8, h=300, w=300
    )
    det_spill = detect_colored_edge_halo(spill_rgb, spill_alpha, blue_bg, hair_mask=hair_m)
    assert det_spill["status"] == "FAIL"
    assert det_spill["score"] >= 150


def test_decontaminate_edges_removes_fringe():
    """Test decontamination repair removes the blue halo and restores clean boundary."""
    blue_bg = (0.0, 129.0, 253.0)
    spill_rgb, spill_alpha, hair_m = create_synthetic_subject_with_halo(
        backdrop_rgb=(0, 129, 253), foreground_rgb=(35, 30, 25), halo_width=8, h=300, w=300
    )

    det_before = detect_colored_edge_halo(spill_rgb, spill_alpha, blue_bg, hair_mask=hair_m)
    assert det_before["status"] == "FAIL"

    repaired_rgb, repaired_alpha, info = decontaminate_edges(
        spill_rgb, spill_alpha, blue_bg, det_before["mask"], hair_mask=hair_m
    )
    assert info["pixels_corrected"] > 0

    # Composite repaired subject over white
    bg = np.array([255, 255, 255], dtype=float)
    a = (repaired_alpha.astype(float) / 255.0)[:, :, None]
    comp = (repaired_rgb.astype(float) * a + bg * (1.0 - a)).clip(0, 255).astype(np.uint8)

    det_after = detect_colored_edge_halo(comp, repaired_alpha, blue_bg, hair_mask=hair_m)
    assert det_after["status"] == "PASS"
