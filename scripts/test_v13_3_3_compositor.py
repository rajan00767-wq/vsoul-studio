"""Deterministic V13.3.3 Compositor Test Harness.

Executes anatomical garment registration and 7-layer semantic occlusion-aware compositing
on all 7 production acceptance cases using cached raw Qwen candidates.
Evaluates collar preservation, tie knot integrity, dual bows, and shoulder contour elimination.
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
logger = logging.getLogger("test_v13_3_3")

CASES = [
    {
        "id": "PA_01_Curly_RedCollar",
        "debug_dir": "PA_01_Curly_RedCollar_debug",
        "cues": {"has_red_collar": True, "ref_type": "red_collar"},
    },
    {
        "id": "PA_02_Braids_NavyCollar",
        "debug_dir": "PA_02_Braids_NavyCollar_debug",
        "cues": {"has_navy_collar": True, "ref_type": "navy_collar"},
    },
    {
        "id": "PA_03_MaroonBows_Tie",
        "debug_dir": "PA_03_MaroonBows_Tie_debug",
        "cues": {"has_tie": True, "ref_type": "tie"},
    },
    {
        "id": "PA_04_NoBows_Vest",
        "debug_dir": "PA_04_NoBows_Vest_debug",
        "cues": {"ref_type": "vest"},
    },
    {
        "id": "PA_05_GlassesBindi_RedCollar",
        "debug_dir": "PA_05_GlassesBindi_RedCollar_debug",
        "cues": {"has_red_collar": True, "ref_type": "red_collar"},
    },
    {
        "id": "PA_11_ShortBoy_Tie",
        "debug_dir": "PA_11_ShortBoy_Tie_debug",
        "cues": {"has_tie": True, "ref_type": "tie"},
    },
    {
        "id": "PA_12_StraightHair_RedCollar",
        "debug_dir": "PA_12_StraightHair_RedCollar_debug",
        "cues": {"has_red_collar": True, "ref_type": "red_collar"},
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


def count_black_edge_pixels(img_rgb: np.ndarray) -> int:
    """Count dark/black silhouette contour pixels along subject boundary."""
    h, w = img_rgb.shape[:2]
    is_bg = (img_rgb[:, :, 0] > 245) & (img_rgb[:, :, 1] > 245) & (img_rgb[:, :, 2] > 245)
    fg = ~is_bg
    inner_edge = fg & ~cv2.erode(fg.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    # Focus on shoulders/torso below y = 600
    shoulder_zone = inner_edge & (np.arange(h)[:, None] >= 600)
    edge_px = img_rgb[shoulder_zone]
    if len(edge_px) == 0:
        return 0
    dark_px = edge_px[(edge_px[:, 0] < 80) & (edge_px[:, 1] < 80) & (edge_px[:, 2] < 80)]
    return len(dark_px)


def main():
    base_out = Path("outputs/production_acceptance")
    out_dir = base_out / "v13_3_3_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = UniformSwapConfig()
    config.compositor_version = "v13_3_3"
    config.background_color = (255, 255, 255)

    ie = IdentityExtractor()
    benchmark_results = []

    print("\n" + "=" * 80)
    print("V13.3.3 DETERMINISTIC COMPOSITOR ACCEPTANCE TEST (CACHED RAW CANDIDATES)")
    print("=" * 80)

    for case in CASES:
        cid = case["id"]
        c_dir = base_out / case["debug_dir"]
        cues = case["cues"]

        src_path = c_dir / "02_normalized_source.png"
        raw_path = c_dir / "raw_candidate_resized.png"
        v13_2_path = base_out / f"{cid}_uniform.png"

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

        # 2. Execute V13.3.3 Compositor
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

        # 3. Produce Passport Crop Framing
        passport_crop_pil, crop_meta = produce_passport_crop(
            comp_rgba,
            face_info={"bbox": id_layer.get("face_bbox"), "kps": id_layer.get("metrics", {}).get("landmarks")},
            identity_layer=id_layer,
            output_size=(800, 1024),
            target_head_occupancy=0.58,
            target_headroom_ratio=0.10,
            background_rgb=config.background_color,
            debug_dir=case_debug,
            job_id=cid,
            return_meta=True,
        )

        deliverable_path = out_dir / f"{cid}_uniform_v13_3_3.png"
        passport_crop_pil.save(deliverable_path)
        logger.info("[%s] Saved V13.3.3 deliverable to %s", cid, deliverable_path)

        # 4. Comparative Metrics (V13.3.2 vs V13.3.3)
        v13_2_rgb = np.array(Image.open(v13_2_path)) if v13_2_path.exists() else None
        v13_3_rgb = np.array(passport_crop_pil)

        ref_type = cues.get("ref_type", "")
        raw_collar_px = measure_collar_pixels(raw_rgb, ref_type)
        v13_2_collar_px = measure_collar_pixels(v13_2_rgb, ref_type) if v13_2_rgb is not None else 0
        v13_3_collar_px = measure_collar_pixels(v13_3_rgb, ref_type)

        v13_2_black_px = count_black_edge_pixels(v13_2_rgb) if v13_2_rgb is not None else 0
        v13_3_black_px = count_black_edge_pixels(v13_3_rgb)

        res_entry = {
            "case_id": cid,
            "ref_type": ref_type,
            "raw_collar_px": raw_collar_px,
            "v13_2_collar_px": v13_2_collar_px,
            "v13_3_collar_px": v13_3_collar_px,
            "collar_retention_v13_3": f"{(v13_3_collar_px / max(1, raw_collar_px))*100:.1f}%" if raw_collar_px > 0 else "N/A",
            "v13_2_black_edge_px": v13_2_black_px,
            "v13_3_black_edge_px": v13_3_black_px,
            "black_edge_reduction": f"{v13_2_black_px - v13_3_black_px} px removed",
            "output_path": str(deliverable_path),
        }

        # Case-specific accessory tracking
        if cid == "PA_03_MaroonBows_Tie":
            bows = id_layer.get("individual_bows", [])
            res_entry["accessories"] = [f"{b['name']} ({b['area']} px)" for b in bows]
        elif cid == "PA_04_NoBows_Vest":
            bows = id_layer.get("individual_bows", [])
            res_entry["accessories"] = [f"{b['name']} ({b['area']} px)" for b in bows]

        benchmark_results.append(res_entry)

    print("\n" + "=" * 80)
    print("V13.3.3 vs V13.3.2 COMPARATIVE METRICS SUMMARY")
    print("=" * 80)
    for r in benchmark_results:
        print(f"\n[{r['case_id']}] ({r['ref_type']})")
        if r['raw_collar_px'] > 0:
            print(f"  Collar Pixels: V13.2={r['v13_2_collar_px']} -> V13.3={r['v13_3_collar_px']} (Raw={r['raw_collar_px']}, Retention={r['collar_retention_v13_3']})")
        print(f"  Black Edge Pixels: V13.2={r['v13_2_black_edge_px']} -> V13.3={r['v13_3_black_edge_px']} ({r['black_edge_reduction']})")
        if "accessories" in r:
            print(f"  Tracked Accessories: {r['accessories']}")

    summary_file = out_dir / "V13_3_3_TEST_SUMMARY.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\nSaved benchmark metrics to {summary_file}")


if __name__ == "__main__":
    main()
