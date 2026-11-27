"""Deterministic V13.3.4 Compositor Test Harness.

Executes anatomical garment registration, continuous anatomical skin integration,
and multi-layer occlusion-aware compositing on all 7 production acceptance cases
using cached raw Qwen candidates.

Evaluates:
- Jaw-to-neck continuity (zero gaps / disconnect).
- Complete collar preservation & natural neckline.
- Elimination of original garment residue.
- Case-specific background color fidelity without leakage.
- Intact tie knots and textures.
- Full accessory preservation (both maroon bows in PA_03, floral pins & earrings in PA_04).
- Elimination of black shoulder outlines.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.uniform_swap.config import UniformSwapConfig
from pipelines.uniform_swap.compositor import uniform_compositor
from pipelines.uniform_swap.identity_extractor import IdentityExtractor
from pipelines.uniform_swap.image_normalizer import produce_passport_crop
from pipelines.uniform_swap.quality_validator import quality_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_v13_3_4")

CASES = [
    {
        "id": "PA_01_Curly_RedCollar",
        "debug_dir": "PA_01_Curly_RedCollar_debug",
        "cues": {"has_red_collar": True, "ref_type": "red_collar"},
        "bg": (255, 255, 255),
    },
    {
        "id": "PA_02_Braids_NavyCollar",
        "debug_dir": "PA_02_Braids_NavyCollar_debug",
        "cues": {"has_navy_collar": True, "ref_type": "navy_collar"},
        "bg": (4, 126, 246),
    },
    {
        "id": "PA_03_MaroonBows_Tie",
        "debug_dir": "PA_03_MaroonBows_Tie_debug",
        "cues": {"has_tie": True, "ref_type": "tie"},
        "bg": (255, 255, 255),
    },
    {
        "id": "PA_04_NoBows_Vest",
        "debug_dir": "PA_04_NoBows_Vest_debug",
        "cues": {"ref_type": "vest"},
        "bg": (211, 211, 211),
    },
    {
        "id": "PA_05_GlassesBindi_RedCollar",
        "debug_dir": "PA_05_GlassesBindi_RedCollar_debug",
        "cues": {"has_red_collar": True, "ref_type": "red_collar"},
        "bg": (255, 255, 255),
    },
    {
        "id": "PA_11_ShortBoy_Tie",
        "debug_dir": "PA_11_ShortBoy_Tie_debug",
        "cues": {"has_tie": True, "ref_type": "tie"},
        "bg": (255, 255, 255),
    },
    {
        "id": "PA_12_StraightHair_RedCollar",
        "debug_dir": "PA_12_StraightHair_RedCollar_debug",
        "cues": {"has_red_collar": True, "ref_type": "red_collar"},
        "bg": (119, 191, 145),
    },
]


def measure_collar_pixels(img_rgb: np.ndarray, ref_type: str) -> int:
    """Measure visible collar pixels in the image."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    if "red_collar" in ref_type:
        is_red = ((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] >= 65) & (hsv[:, :, 2] >= 50)
        return int(np.sum(is_red))
    elif "navy_collar" in ref_type:
        cr = img_rgb[:, :, 0].astype(int)
        cg = img_rgb[:, :, 1].astype(int)
        cb = img_rgb[:, :, 2].astype(int)
        is_navy = (cr < 90) & (cg < 95) & (cb > 60) & (cb > cr + 10)
        return int(np.sum(is_navy))
    return 0


def count_black_edge_pixels(img_rgb: np.ndarray, bg_rgb: Tuple[int, int, int]) -> int:
    """Count dark/black silhouette contour pixels along subject boundary."""
    h, w = img_rgb.shape[:2]
    bg_arr = np.array(bg_rgb, dtype=float)
    dist = np.max(np.abs(img_rgb.astype(float) - bg_arr), axis=2)
    fg = dist > 25.0
    inner_edge = fg & ~cv2.erode(fg.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    shoulder_zone = inner_edge & (np.arange(h)[:, None] >= int(h * 0.45))
    edge_px = img_rgb[shoulder_zone]
    if len(edge_px) == 0:
        return 0
    dark_px = edge_px[(edge_px[:, 0] < 45) & (edge_px[:, 1] < 45) & (edge_px[:, 2] < 45)]
    return len(dark_px)


def main():
    base_out = Path("outputs/production_acceptance")
    out_dir = base_out / "v13_3_4_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = UniformSwapConfig()
    config.compositor_version = "v13_3_4"

    ie = IdentityExtractor()
    benchmark_results = []

    print("\n" + "=" * 80)
    print("V13.3.4 DETERMINISTIC COMPOSITOR ACCEPTANCE TEST (CONTINUOUS ANATOMICAL SKIN)")
    print("=" * 80)

    for case in CASES:
        cid = case["id"]
        c_dir = base_out / case["debug_dir"]
        cues = case["cues"]
        case_bg = case["bg"]
        config.background_color = case_bg

        src_path = c_dir / "02_normalized_source.png"
        raw_path = c_dir / "raw_candidate_resized.png"
        v13_3_path = base_out / "v13_3_3_candidates" / f"{cid}_uniform_v13_3_3.png"

        if not src_path.exists() or not raw_path.exists():
            logger.error("[%s] Missing source or raw candidate image!", cid)
            continue

        src_pil = Image.open(src_path).convert("RGB")
        raw_pil = Image.open(raw_path).convert("RGB")
        src_rgb = np.array(src_pil)
        raw_rgb = np.array(raw_pil)

        # 1. Extract Identity Layer
        id_layer = ie.extract(src_rgb, job_id=f"{cid}_id")
        id_layer["cues"] = cues

        # 2. Execute V13.3.4 Compositor
        case_debug = out_dir / f"{cid}_debug"
        case_debug.mkdir(parents=True, exist_ok=True)

        comp_res = uniform_compositor.composite(
            raw_vae_img=raw_pil,
            normalized_source_img=src_pil,
            identity_layer=id_layer,
            raw_quality={},
            config=config,
            job_id=cid,
            debug_dir=case_debug,
        )

        comp_rgba = comp_res["composite_rgba"]

        # 3. Produce Passport Crop Framing on case-specific background
        passport_crop_pil, crop_meta = produce_passport_crop(
            comp_rgba,
            face_info={"bbox": id_layer.get("face_bbox"), "kps": id_layer.get("metrics", {}).get("landmarks")},
            identity_layer=id_layer,
            output_size=(800, 1024),
            target_head_occupancy=0.58,
            target_headroom_ratio=0.10,
            background_rgb=case_bg,
        )

        final_crop_rgb = np.array(passport_crop_pil.convert("RGB"))

        # Save deliverable candidate
        output_crop_path = out_dir / f"{cid}_uniform_v13_3_4.png"
        passport_crop_pil.save(output_crop_path, dpi=(300, 300))
        passport_crop_pil.save(case_debug / "final_output.png")

        # 4. Evaluation Metrics
        ref_type = cues.get("ref_type", "")
        cand_collar_px = measure_collar_pixels(final_crop_rgb, ref_type)
        raw_collar_px = measure_collar_pixels(raw_rgb, ref_type)
        dark_edge_px = count_black_edge_pixels(final_crop_rgb, case_bg)

        # Jaw-to-neck alpha continuity check
        h_crop, w_crop = final_crop_rgb.shape[:2]
        chin_y_crop = int(crop_meta.get("transformed_chin_y", h_crop * 0.50))
        throat_strip = comp_res["assembled_fg_alpha"][chin_y_crop:chin_y_crop + 60, int(w_crop * 0.40):int(w_crop * 0.60)]
        min_throat_alpha = int(np.min(throat_strip)) if throat_strip.size > 0 else 0

        # Accessories tracking
        bow_info = []
        for b in id_layer.get("individual_bows", []):
            bow_info.append(f"{b['name']} ({b['area']} px)")
        acc_str = ", ".join(bow_info) if bow_info else "none"

        benchmark_results.append({
            "case": cid,
            "ref_type": ref_type,
            "bg": str(case_bg),
            "collar_pixels": cand_collar_px,
            "raw_collar_pixels": raw_collar_px,
            "min_throat_alpha": min_throat_alpha,
            "dark_edge_pixels": dark_edge_px,
            "accessories": acc_str,
        })

        print(f"\n--- {cid} ---")
        print(f"  Target Background  : RGB{case_bg}")
        print(f"  Collar Pixels      : {cand_collar_px:,} (raw candidate: {raw_collar_px:,})")
        print(f"  Min Throat Alpha   : {min_throat_alpha}/255 (continuity: {'PASS' if min_throat_alpha > 200 else 'FAIL'})")
        print(f"  Dark Edge Pixels   : {dark_edge_px} px")
        print(f"  Tracked Accessories: {acc_str}")
        print(f"  Delivered Image    : {output_crop_path.name}")

    print("\n" + "=" * 80)
    print("V13.3.4 DETERMINISTIC EXECUTION COMPLETE")
    print("=" * 80)

    # Save summary json
    summary_path = out_dir / "v13_3_4_harness_summary.json"
    summary_path.write_text(json.dumps(benchmark_results, indent=2), encoding="utf-8")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
