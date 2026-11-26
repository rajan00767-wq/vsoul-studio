"""V13.1 Multi-Background Compatibility Verification Test.

Validates that the 5-layer straight RGBA subject and post-crop alpha
compositing cleanly supports any arbitrary background colour (White, Blue,
Green, Light Blue, Grey, Cream, and Custom HEX) without halos, rectangular borders,
or background bleeding.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, ".")

from pipelines.uniform_swap.config import UniformSwapConfig
from pipelines.uniform_swap.orchestrator import uniform_swap_orchestrator
from pipelines.uniform_swap.quality_validator import quality_validator

TEST_SCENARIOS = [
    {
        "test_id": "T001",
        "name": "Boys curly hair",
        "source": Path("test_dataset/test_images/T001-T005/curly_hair.png"),
        "uniform": Path("test_dataset/uniform_references/REF001_red_collar.png"),
        "candidate": Path("outputs/cache/raw_candidates/4fa90dfe80338b06563bdf04.png"),
        "bg_colors": [
            ("#FFFFFF", "White"),
            ("#047EF6", "Royal Blue"),
            ("#77BF91", "Sage Green"),
            ("#ADD8E6", "Light Blue"),
            ("#D3D3D3", "Studio Grey"),
            ("#F5E6D3", "Warm Cream"),
            ("#8B0000", "Custom Dark Red"),
        ],
    },
    {
        "test_id": "T021",
        "name": "Glasses bindi earrings",
        "source": Path("test_dataset/test_images/T021-T025/glasses_bindi.png"),
        "uniform": Path("test_dataset/uniform_references/REF002_navy_collar.png"),
        "candidate": Path("outputs/overnight_T021_20261126_113139_qwen_raw.png"),
        "bg_colors": [
            ("#FFFFFF", "White"),
            ("#047EF6", "Royal Blue"),
            ("#77BF91", "Sage Green"),
            ("#D3D3D3", "Studio Grey"),
        ],
    },
    {
        "test_id": "T046",
        "name": "Complex background & tilted face",
        "source": Path("test_dataset/test_images/T046-T050/complex_background.png"),
        "uniform": Path("test_dataset/uniform_references/REF002_navy_collar.png"),
        "candidate": Path("outputs/overnight_T046_20261126_113139_qwen_raw.png"),
        "bg_colors": [
            ("#047EF6", "Royal Blue"),
            ("#FFFFFF", "White"),
            ("#77BF91", "Sage Green"),
            ("#F5E6D3", "Warm Cream"),
        ],
    },
]


def main():
    out_dir = Path("outputs/v13_multi_bg_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print("V13.1 MULTI-BACKGROUND COLOUR COMPATIBILITY VERIFICATION")
    print("=" * 90)

    total_runs = 0
    passed_runs = 0

    for sc in TEST_SCENARIOS:
        tid = sc["test_id"]
        print(f"\n--- Testing Scenario {tid} ({sc['name']}) ---")
        for hex_code, color_name in sc["bg_colors"]:
            total_runs += 1
            safe_hex = hex_code.replace("#", "")
            job_id = f"v13_bg_{tid}_{safe_hex}"
            run_debug = out_dir / f"{tid}_{safe_hex}"
            run_debug.mkdir(parents=True, exist_ok=True)

            cfg = UniformSwapConfig(
                background_color=hex_code,
                rgb_identity_restoration="adaptive",
                debug_mode=True,
            )

            res = uniform_swap_orchestrator.run(
                person_image_input=sc["source"],
                uniform_template_input=sc["uniform"],
                config=cfg,
                job_id=job_id,
                output_dir=out_dir / "delivered",
                debug_dir=run_debug,
                cached_raw_candidate=sc["candidate"],
            )

            sim = res.validation_report.get("similarity", 0.0)
            rmse = res.validation_report.get("geometry_rmse", 0.0)
            status = res.status
            passed = res.passed
            defects = res.defects

            status_str = "PASS" if passed else f"FAIL ({status})"
            def_str = ", ".join(defects) if defects else "Zero defects"
            print(f"[{tid}] {color_name:16s} ({hex_code}): {status_str:12s} | sim={sim:.4f} | rmse={rmse:.4f} | {def_str}")

            if passed:
                passed_runs += 1

    print("\n" + "=" * 90)
    print(f"MULTI-BACKGROUND TEST SUMMARY: {passed_runs}/{total_runs} PASSED")
    print("=" * 90)

    if passed_runs == total_runs:
        print("ALL MULTI-BACKGROUND TESTS PASSED WITH ZERO DEFECTS!")
        return 0
    else:
        print("SOME MULTI-BACKGROUND TESTS FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
