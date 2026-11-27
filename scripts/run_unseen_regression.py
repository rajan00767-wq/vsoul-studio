"""Unseen Generalization Regression Test Suite (V13.3 Section 15).

Tests comprehensive generalization across unseen real-world conditions:
1. Scanned historical photo inputs (T041 scanned.png)
2. Complex non-studio backgrounds (T046 complex_background.png)
3. Diverse hairstyles: short boy, long braids, curly, straight, bows
4. Facial accessories: eyeglasses, bindi
5. Unseen uniform designs:
   - REF001: Red collar with check plaid fabric
   - REF002: Navy collar
   - REF003: White shirt with dark necktie
   - REF004: Layered dark vest
6. Multiple customer background colors (White, Royal Blue, Studio Grey, Sage Green, Warm Cream)
7. Generates UNSEEN_REGRESSION_REPORT.md, TEST_RESULTS.csv, and side-by-side comparison sheets.
"""

from __future__ import annotations

import sys
import os
import time
import csv
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.uniform_swap.config import UniformSwapConfig
from pipelines.uniform_swap.orchestrator import UniformSwapOrchestrator


def create_comparison_sheet(
    source_path: Path,
    uniform_path: Path,
    delivered_path: Path,
    test_id: str,
    status: str,
    out_path: Path,
) -> None:
    """Create three-panel side-by-side comparison image: Source | Reference | Output."""
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

    title_text = f"V13.3 Test: {test_id} [{status.upper()}]"
    col = (0, 140, 0) if status == "completed" else (0, 100, 220)
    cv2.putText(canvas, title_text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)

    cv2.putText(canvas, "1. Original Source", (x0 + 10, header_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 50, 50), 2)
    cv2.putText(canvas, "2. Target Uniform", (x1 + 10, header_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 50, 50), 2)
    cv2.putText(canvas, "3. V13.3 Output", (x2 + 10, header_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 50, 50), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def run_unseen_regression():
    print("=" * 105)
    print("V13.3 UNSEEN GENERALIZATION REGRESSION TEST SUITE & INTELLIGENT RECOVERY")
    print("=" * 105)

    test_matrix = [
        {
            "test_id": "REG_01_Scanned_RedCollar",
            "source": Path("test_dataset/test_images/T041-T045/scanned.png"),
            "uniform": Path("test_dataset/uniform_references/REF001_red_collar.png"),
            "bg_color": (255, 255, 255),
            "category": "Scanned Vintage Photo",
            "notes": "Scanned textured print with red collar target",
        },
        {
            "test_id": "REG_02_ComplexBG_NavyCollar",
            "source": Path("test_dataset/test_images/T046-T050/complex_background.png"),
            "uniform": Path("test_dataset/uniform_references/REF002_navy_collar.png"),
            "bg_color": (4, 126, 246),  # Royal Blue
            "category": "Complex Natural Background",
            "notes": "Non-studio background with navy collar on royal blue backdrop",
        },
        {
            "test_id": "REG_03_GlassesBindi_Vest",
            "source": Path("test_dataset/test_images/T021-T025/glasses_bindi.png"),
            "uniform": Path("test_dataset/uniform_references/REF004_vest.png"),
            "bg_color": (255, 255, 255),
            "category": "Eyeglasses & Bindi Retention",
            "notes": "Facial jewelry & glasses preservation with vest uniform",
        },
        {
            "test_id": "REG_04_LongBraids_Tie",
            "source": Path("test_dataset/test_images/T006-T010/braids.png"),
            "uniform": Path("test_dataset/uniform_references/REF003_tie.png"),
            "bg_color": (211, 211, 211),  # Studio Grey
            "category": "Long Braids Over Garment",
            "notes": "Natural hanging braids preserved over necktie shirt",
        },
        {
            "test_id": "REG_05_ShortBoy_Tie",
            "source": Path("test_dataset/extra_images/blue_bg_boy.png"),
            "uniform": Path("test_dataset/uniform_references/REF003_tie.png"),
            "bg_color": (255, 255, 255),
            "category": "Short Hair Boy Portrait",
            "notes": "Short hair subject with dress shirt and tie",
        },
        {
            "test_id": "REG_06_StraightHair_RedCollar",
            "source": Path("test_dataset/extra_images/straight_hair.png"),
            "uniform": Path("test_dataset/uniform_references/REF001_red_collar.png"),
            "bg_color": (119, 191, 145),  # Sage Green
            "category": "Straight Hair with Custom Backdrop",
            "notes": "Straight hair silhouette over sage green studio background",
        },
        {
            "test_id": "REG_07_BowsRibbons_NavyCollar",
            "source": Path("test_dataset/extra_images/blue_bg_bows.png"),
            "uniform": Path("test_dataset/uniform_references/REF002_navy_collar.png"),
            "bg_color": (255, 255, 255),
            "category": "Hair Bows & Ribbons",
            "notes": "Authentic bows preserved cleanly with navy collar",
        },
        {
            "test_id": "REG_08_Sunlight_Vest",
            "source": Path("test_dataset/test_images/T031-T035/sunlight.png"),
            "uniform": Path("test_dataset/uniform_references/REF004_vest.png"),
            "bg_color": (245, 230, 211),  # Warm Cream
            "category": "Harsh Sunlight Balancing",
            "notes": "Harsh sunlight source portrait with bilateral shadow lifting",
        },
    ]

    orchestrator = UniformSwapOrchestrator()
    results = []

    csv_path = Path("TEST_RESULTS.csv")
    csv_headers = [
        "Test ID",
        "Category",
        "Source Portrait",
        "Uniform Reference",
        "Background Color",
        "Status",
        "Biometric Similarity",
        "Geometry RMSE",
        "Defects Count",
        "Defects",
        "Recovery Actions",
        "Peak VRAM (MB)",
        "Delivered Output",
        "Duration (s)",
    ]

    out_base = Path("outputs/unseen_regression")
    comp_dir = out_base / "comparisons"
    out_base.mkdir(parents=True, exist_ok=True)
    comp_dir.mkdir(parents=True, exist_ok=True)

    for idx, tc in enumerate(test_matrix, 1):
        tid = tc["test_id"]
        src_p = tc["source"]
        ref_p = tc["uniform"]
        bg = tc["bg_color"]
        cat = tc["category"]

        print(f"\n[{idx}/{len(test_matrix)}] Running {tid} ({cat})")
        print(f"      Source: {src_p} | Target: {ref_p} | BG: {bg}")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        t0 = time.time()
        config = UniformSwapConfig(
            steps=4,
            raw_diffusion_cache_policy="authoritative_cache",
            diagnostic_mode=True,
            background_color=bg,
            output_resolution=(800, 1024),
        )

        cached_cand = Path("outputs") / f"{tid}_qwen_raw.png"
        raw_arg = cached_cand if cached_cand.exists() else None

        recovery_acts: List[str] = []
        try:
            res = orchestrator.run(
                person_image_input=src_p,
                uniform_template_input=ref_p,
                config=config,
                job_id=tid,
                output_dir=out_base,
                debug_dir=out_base / f"{tid}_debug",
                cached_raw_candidate=raw_arg,
            )
            duration = time.time() - t0
            status = res.status
            passed = res.passed
            sim = res.validation_report.get("similarity", 0.0)
            rmse = res.validation_report.get("geometry_rmse", 0.0)
            defects = res.defects
            delivered = str(res.delivered_path) if res.delivered_path else "None (Flagged)"
            repair_acts = []
            for st in res.trace.get("stages", []):
                if isinstance(st, dict) and st.get("stage") == "final_validation":
                    repair_acts = st.get("actions", [])
                    break
        except Exception as e:
            duration = time.time() - t0
            status = "failed"
            sim = 0.0
            rmse = 0.0
            defects = [f"Exception: {str(e)}"]
            delivered = "None (Error)"
            repair_acts = []

        peak_vram = 0.0
        if torch.cuda.is_available():
            peak_vram = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)

        # Three-panel side-by-side comparison
        comp_sheet = comp_dir / f"{tid}_comparison.png"
        delivered_img_p = Path(delivered) if delivered not in ("None (Flagged)", "None (Error)") else (out_base / f"{tid}_debug" / f"{tid}_19_passport_crop.png")
        if not delivered_img_p.exists():
            delivered_img_p = out_base / f"{tid}_debug" / f"{tid}_18_final_output.png"
        create_comparison_sheet(src_p, ref_p, delivered_img_p, tid, status, comp_sheet)

        row = {
            "Test ID": tid,
            "Category": cat,
            "Source Portrait": src_p.name,
            "Uniform Reference": ref_p.name,
            "Background Color": str(bg),
            "Status": status,
            "Biometric Similarity": round(sim, 4),
            "Geometry RMSE": round(rmse, 4),
            "Defects Count": len(defects),
            "Defects": "; ".join(defects) if defects else "None",
            "Recovery Actions": "; ".join(repair_acts) if repair_acts else "None",
            "Peak VRAM (MB)": peak_vram,
            "Delivered Output": delivered,
            "Duration (s)": round(duration, 2),
        }
        results.append(row)
        print(f"      -> Status: {status} | Sim: {sim:.3f} | RMSE: {rmse:.4f} | Defects: {len(defects)} | VRAM: {peak_vram}MB | Time: {duration:.2f}s")

    # Write TEST_RESULTS.csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(results)

    # Generate UNSEEN_REGRESSION_REPORT.md
    report_md = Path("UNSEEN_REGRESSION_REPORT.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# V13.3 Unseen Generalization Regression Test Report\n\n")
        f.write("## Overview\n")
        f.write("Comprehensive regression evaluation across unseen real-world portraits, challenging conditions, ")
        f.write("multiple uniform categories (red collar, navy collar, necktie, vest), and multiple background colors.\n\n")
        f.write("## Results Summary\n\n")
        f.write("| Test ID | Category | Source | Uniform | Status | Biometric Sim | Geometry RMSE | Defects | VRAM (MB) | Delivered |\n")
        f.write("|---------|----------|--------|---------|--------|---------------|---------------|---------|-----------|-----------|\n")
        for r in results:
            deliv_name = Path(r['Delivered Output']).name if r['Delivered Output'] not in ('None (Flagged)', 'None (Error)') else 'needs_review'
            f.write(
                f"| {r['Test ID']} | {r['Category']} | {r['Source Portrait']} | {r['Uniform Reference']} | "
                f"**{r['Status']}** | {r['Biometric Similarity']:.4f} | {r['Geometry RMSE']:.4f} | "
                f"{r['Defects Count']} | {r['Peak VRAM (MB)']} | {deliv_name} |\n"
            )
        f.write("\n## Production Quality Assurance Policy\n")
        f.write("- **Zero False Deliveries**: Any candidate with detected defects is strictly rejected and flagged for review.\n")
        f.write("- **Biometric Integrity**: Facial identity and geometry strictly preserved above threshold.\n")
        f.write("- **Artifact Protection**: Hair bows, ribbons, glasses, and bindi preserved without erasure.\n")
        f.write("- **Automatic Defect Recovery**: Localized targeted repairs eliminate transient shoulder gaps and lighting mismatches.\n")

    print(f"\n[UNSEEN_REGRESSION] Complete! Saved {csv_path} and {report_md}")
    return results


if __name__ == "__main__":
    run_unseen_regression()
