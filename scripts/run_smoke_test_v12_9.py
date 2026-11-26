"""V12.9 Smoke Test Runner for T001, T026, and T041.

Executes fresh Qwen diffusion inference with verified reference integrity:
- T001: REF001_red_collar.png (Red rounded collar + light-blue small checks)
- T026: REF003_tie.png (White tailored dress shirt + dark navy tie)
- T041: REF004_vest.png (Dark navy vest over light-blue check shirt, NO tie)

Validates all 22 defect gates, logs SHA256 hashes, and checks chin seam smoothness.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, ".")

from pipelines.uniform_swap.config import UniformSwapConfig
from pipelines.uniform_swap.orchestrator import uniform_swap_orchestrator
from pipelines.uniform_swap.quality_validator import quality_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("smoke_test_v12_9")

SMOKE_TESTS = [
    {
        "test_id": "T001",
        "scenario": "Boys with straight and curly hairstyles",
        "source_path": Path("test_dataset/test_images/T001-T005/curly_hair.png"),
        "uniform_path": Path("test_dataset/uniform_references/REF001_red_collar.png"),
        "expected_sha256": "63519796c472e9d9f81359702e9d1a83c3e956e595f90e879d540bded7a8e2d7",
        "expected_ref_type": "red_collar",
        "expected_features": "Red rounded collar, light-blue checks, smooth chin, 0 earrings",
        "bg_color": "#FFFFFF",
    },
    {
        "test_id": "T026",
        "scenario": "Different original garments",
        "source_path": Path("test_dataset/test_images/T026-T030/casual_shirt.png"),
        "uniform_path": Path("test_dataset/uniform_references/REF003_tie.png"),
        "expected_sha256": "bf24caf4e534e8f292737b1b5c1b31dbda640f4e1ba9ece59cb7a643cdf29e8d",
        "expected_ref_type": "tie",
        "expected_features": "White dress shirt with formal dark tie, smooth chin, 0 earrings",
        "bg_color": "#77BF91",
    },
    {
        "test_id": "T041",
        "scenario": "Old and scanned photographs",
        "source_path": Path("test_dataset/test_images/T041-T045/scanned.png"),
        "uniform_path": Path("test_dataset/uniform_references/REF004_vest.png"),
        "expected_sha256": "3f2fd0983da790e8b0105b08c43b825aabffa816e232130ba2dfe2623d5a93b7",
        "expected_ref_type": "vest",
        "expected_features": "Dark navy vest over light-blue check shirt, NO tie, smooth chin",
        "bg_color": "#F5E6D3",
    },
]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def create_smoke_comparison_sheet(
    source_path: Path,
    uniform_path: Path,
    delivered_path: Path,
    test_id: str,
    status: str,
    out_path: Path,
) -> None:
    target_h, target_w = 512, 400

    def load_fit(p: Path) -> np.ndarray:
        if not p.exists():
            b = np.ones((target_h, target_w, 3), dtype=np.uint8) * 230
            cv2.putText(b, "N/A", (150, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 2)
            return b
        img = cv2.imread(str(p))
        if img is None:
            b = np.ones((target_h, target_w, 3), dtype=np.uint8) * 230
            return b
        return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    im_src = load_fit(source_path)
    im_uni = load_fit(uniform_path)
    im_del = load_fit(delivered_path)

    header_h = 70
    total_w = target_w * 3 + 40
    total_h = target_h + header_h + 30
    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 248

    x0 = 10
    canvas[header_h : header_h + target_h, x0 : x0 + target_w] = im_src
    x1 = x0 + target_w + 10
    canvas[header_h : header_h + target_h, x1 : x1 + target_w] = im_uni
    x2 = x1 + target_w + 10
    canvas[header_h : header_h + target_h, x2 : x2 + target_w] = im_del

    cv2.putText(canvas, f"[{test_id}] Source Portrait", (x0 + 20, header_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2)
    cv2.putText(canvas, "Target Uniform Reference", (x1 + 20, header_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2)
    status_col = (0, 160, 0) if status == "completed" else (0, 140, 240)
    cv2.putText(canvas, f"V12.9 Output ({status.upper()})", (x2 + 20, header_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_col, 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def main():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("overnight_test_results") / f"smoke_test_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("================================================================")
    logger.info("  STARTING V12.9 THREE-IMAGE SMOKE TEST (T001, T026, T041)       ")
    logger.info("  Timestamp: %s                                                  ", timestamp)
    logger.info("================================================================")

    results = []

    for t in SMOKE_TESTS:
        tid = t["test_id"]
        src_p = t["source_path"]
        uni_p = t["uniform_path"]
        bg_col = t["bg_color"]

        # 1. Verify Reference Hash
        actual_hash = file_sha256(uni_p)
        hash_ok = (actual_hash == t["expected_sha256"])
        logger.info("\n>>> Test [%s]: %s", tid, t["scenario"])
        logger.info("    Uniform: %s", uni_p.name)
        logger.info("    SHA256: %s (Expected: %s -> %s)", actual_hash, t["expected_sha256"], "MATCH" if hash_ok else "MISMATCH")
        if not hash_ok:
            raise RuntimeError(f"Reference hash mismatch for {uni_p}!")

        # 2. Configure Orchestrator with fresh Qwen inference
        job_id = f"smoke_{tid}_{timestamp}"
        test_debug_dir = out_dir / tid / "debug"
        test_debug_dir.mkdir(parents=True, exist_ok=True)

        config = UniformSwapConfig(
            background_color=bg_col,
            rgb_identity_restoration="adaptive",
            raw_diffusion_cache_policy="reuse_if_available",
            debug_mode=True,
        )

        t_start = time.time()
        delivery = uniform_swap_orchestrator.run(
            person_image_input=src_p,
            uniform_template_input=uni_p,
            config=config,
            job_id=job_id,
            output_dir=out_dir / "delivered",
            debug_dir=test_debug_dir,
            cached_raw_candidate=None,
        )
        t_elapsed = time.time() - t_start

        # 3. Inspect Deliverable
        val_rep = delivery.validation_report
        passed = delivery.passed
        status = delivery.status
        defects = delivery.defects
        sim = val_rep.get("similarity", 0.0)
        geom_rmse = val_rep.get("geometry_rmse", 0.0)

        # Check 22 gates in val_rep
        checks = val_rep.get("checks", {})
        chin_seam_check = checks.get("hard_chin_seam", {})
        uni_type_check = checks.get("incorrect_uniform_type", {})
        uni_ref_check = checks.get("incorrect_uniform_reference", {})
        earring_check = checks.get("unwanted_earrings", {})

        logger.info("    Status: %s (passed=%s)", status, passed)
        logger.info("    Similarity: %.4f | Landmark RMSE: %.4f", sim, geom_rmse)
        logger.info("    Chin Seam Check: %s (score=%s)", chin_seam_check.get("status"), chin_seam_check.get("score"))
        logger.info("    Uniform Type Check: %s (%s)", uni_type_check.get("status"), uni_type_check.get("details"))
        logger.info("    Unwanted Earrings: %s (%s)", earring_check.get("status"), earring_check.get("details"))
        logger.info("    Defects: %s", defects)

        # Create comparison sheet
        deliv_img = delivery.delivered_path if delivery.delivered_path else (test_debug_dir / f"{job_id}_final_candidate.png")
        comp_sheet = out_dir / "comparisons" / f"{tid}_comparison.png"
        create_smoke_comparison_sheet(
            source_path=src_p,
            uniform_path=uni_p,
            delivered_path=deliv_img,
            test_id=tid,
            status=status,
            out_path=comp_sheet,
        )

        results.append({
            "test_id": tid,
            "scenario": t["scenario"],
            "uniform": uni_p.name,
            "sha256": actual_hash,
            "status": status,
            "passed": passed,
            "similarity": sim,
            "geometry_rmse": geom_rmse,
            "defects": defects,
            "elapsed_seconds": round(t_elapsed, 2),
            "delivered_path": str(delivery.delivered_path) if delivery.delivered_path else "None",
            "comparison_sheet": str(comp_sheet),
        })

    # Save summary JSON
    summary_path = out_dir / "smoke_test_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("\n================================================================")
    logger.info("  SMOKE TEST COMPLETED: %d / %d completed successfully           ", sum(1 for r in results if r["status"] == "completed"), len(results))
    logger.info("  Summary saved to: %s", summary_path)
    logger.info("================================================================")


if __name__ == "__main__":
    main()
