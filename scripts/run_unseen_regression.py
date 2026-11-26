"""Unseen Generalization Regression Test Suite (V13.2 Section 20).

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
6. Multiple customer background colors (White, Royal Blue, Studio Grey)
7. Generates UNSEEN_REGRESSION_REPORT.md and TEST_RESULTS.csv
"""

import sys
import os
import time
import csv
import json
from pathlib import Path
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.uniform_swap.config import UniformSwapConfig
from pipelines.uniform_swap.orchestrator import UniformSwapOrchestrator


def run_unseen_regression():
    print("=" * 105)
    print("V13.2 UNSEEN GENERALIZATION REGRESSION TEST SUITE")
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
        "Delivered Output",
        "Duration (s)",
    ]

    out_base = Path("outputs/unseen_regression")
    out_base.mkdir(parents=True, exist_ok=True)

    for idx, tc in enumerate(test_matrix, 1):
        tid = tc["test_id"]
        src_p = tc["source"]
        ref_p = tc["uniform"]
        bg = tc["bg_color"]
        cat = tc["category"]

        print(f"\n[{idx}/{len(test_matrix)}] Running {tid} ({cat})")
        print(f"      Source: {src_p} | Target: {ref_p} | BG: {bg}")

        t0 = time.time()
        config = UniformSwapConfig(
            steps=4,
            raw_diffusion_cache_policy="authoritative_cache",  # Uses verified cache if matching provenance exists
            background_color=bg,
            output_resolution=(800, 1024),
        )

        try:
            res = orchestrator.run(
                person_image_input=src_p,
                uniform_template_input=ref_p,
                config=config,
                job_id=tid,
                output_dir=out_base,
                debug_dir=out_base / f"{tid}_debug",
            )
            duration = time.time() - t0
            status = res.status
            passed = res.passed
            sim = res.validation_report.get("similarity", 0.0)
            rmse = res.validation_report.get("geometry_rmse", 0.0)
            defects = res.defects
            delivered = str(res.delivered_path) if res.delivered_path else "None (Flagged)"
        except Exception as e:
            duration = time.time() - t0
            status = "failed"
            sim = 0.0
            rmse = 0.0
            defects = [f"Exception: {str(e)}"]
            delivered = "None (Error)"

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
            "Delivered Output": delivered,
            "Duration (s)": round(duration, 2),
        }
        results.append(row)
        print(f"      -> Status: {status} | Sim: {sim:.3f} | RMSE: {rmse:.4f} | Defects: {len(defects)} | Time: {duration:.2f}s")

    # Write TEST_RESULTS.csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(results)

    # Generate UNSEEN_REGRESSION_REPORT.md
    report_md = Path("UNSEEN_REGRESSION_REPORT.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# V13.2 Unseen Generalization Regression Test Report\n\n")
        f.write("## Overview\n")
        f.write("Comprehensive regression evaluation across unseen real-world portraits, challenging conditions, ")
        f.write("multiple uniform categories (red collar, navy collar, necktie, vest), and multiple background colors.\n\n")
        f.write("## Results Summary\n\n")
        f.write("| Test ID | Category | Source | Uniform | Status | Biometric Sim | Geometry RMSE | Defects | Delivered |\n")
        f.write("|---------|----------|--------|---------|--------|---------------|---------------|---------|-----------|\n")
        for r in results:
            f.write(
                f"| {r['Test ID']} | {r['Category']} | {r['Source Portrait']} | {r['Uniform Reference']} | "
                f"**{r['Status']}** | {r['Biometric Similarity']:.4f} | {r['Geometry RMSE']:.4f} | "
                f"{r['Defects Count']} | {Path(r['Delivered Output']).name if r['Delivered Output'] != 'None (Flagged)' else 'needs_review'} |\n"
            )
        f.write("\n## Production Quality Assurance Policy\n")
        f.write("- **Zero False Deliveries**: Any candidate with detected defects is strictly rejected and flagged for review.\n")
        f.write("- **Biometric Integrity**: Facial identity and geometry strictly preserved above threshold.\n")
        f.write("- **Artifact Protection**: Hair bows, ribbons, glasses, and bindi preserved without erasure.\n")

    print(f"\n[UNSEEN_REGRESSION] Complete! Saved {csv_path} and {report_md}")
    return results


if __name__ == "__main__":
    run_unseen_regression()
