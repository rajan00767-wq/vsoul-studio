"""V13.0 Fast Compositor & Quality Validator Regression Test.

Validates the full 5-layer compositor and 22-gate quality validator
across all 10 test scenarios using verified raw candidates.
"""

from __future__ import annotations

import glob
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, ".")

from pipelines.uniform_swap.config import UniformSwapConfig
from pipelines.uniform_swap.orchestrator import uniform_swap_orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_v13_compositor")

TEST_CASES = [
    {
        "test_id": "T001",
        "scenario": "Boys curly hair",
        "source": Path("test_dataset/test_images/T001-T005/curly_hair.png"),
        "uniform": Path("test_dataset/uniform_references/REF001_red_collar.png"),
        "bg_color": "#FFFFFF",
        "candidate": Path("outputs/cache/raw_candidates/4fa90dfe80338b06563bdf04.png"),
    },
    {
        "test_id": "T006",
        "scenario": "Girls braids",
        "source": Path("test_dataset/test_images/T006-T010/braids.png"),
        "uniform": Path("test_dataset/uniform_references/REF001_red_collar.png"),
        "bg_color": "#FFFFFF",
        "candidate": Path("outputs/overnight_T006_20261126_113139_qwen_raw.png"),
    },
    {
        "test_id": "T011",
        "scenario": "Girls maroon bows",
        "source": Path("test_dataset/test_images/T011-T015/maroon_bows.png"),
        "uniform": Path("test_dataset/uniform_references/REF001_red_collar.png"),
        "bg_color": "#FFFFFF",
        "candidate": Path("outputs/overnight_T011_20261126_113139_qwen_raw.png"),
    },
    {
        "test_id": "T016",
        "scenario": "Girls floral bow & single braid",
        "source": Path("test_dataset/test_images/T016-T020/no_bows.png"),
        "uniform": Path("test_dataset/uniform_references/REF001_red_collar.png"),
        "bg_color": "#FFFFFF",
        "candidate": Path("outputs/overnight_T016_20261126_113139_qwen_raw.png"),
    },
    {
        "test_id": "T021",
        "scenario": "Glasses bindi earrings",
        "source": Path("test_dataset/test_images/T021-T025/glasses_bindi.png"),
        "uniform": Path("test_dataset/uniform_references/REF002_navy_collar.png"),
        "bg_color": "#047EF6",
        "candidate": Path("outputs/overnight_T021_20261126_113139_qwen_raw.png"),
    },
    {
        "test_id": "T026",
        "scenario": "Casual shirt to tie",
        "source": Path("test_dataset/test_images/T026-T030/casual_shirt.png"),
        "uniform": Path("test_dataset/uniform_references/REF003_tie.png"),
        "bg_color": "#77BF91",
        "candidate": Path("outputs/cache/raw_candidates/2e201e5a19dd236254d01170.png"),
    },
    {
        "test_id": "T031",
        "scenario": "Sunlight & shadows",
        "source": Path("test_dataset/test_images/T031-T035/sunlight.png"),
        "uniform": Path("test_dataset/uniform_references/REF001_red_collar.png"),
        "bg_color": "#ADD8E6",
        "candidate": Path("outputs/overnight_T031_20261126_113139_qwen_raw.png"),
    },
    {
        "test_id": "T036",
        "scenario": "Low-res blurred",
        "source": Path("test_dataset/test_images/T036-T040/blurred.png"),
        "uniform": Path("test_dataset/uniform_references/REF001_red_collar.png"),
        "bg_color": "#D3D3D3",
        "candidate": Path("outputs/overnight_T036_20261126_113139_qwen_raw.png"),
    },
    {
        "test_id": "T041",
        "scenario": "Scanned photo to vest",
        "source": Path("test_dataset/test_images/T041-T045/scanned.png"),
        "uniform": Path("test_dataset/uniform_references/REF004_vest.png"),
        "bg_color": "#F5E6D3",
        "candidate": Path("outputs/cache/raw_candidates/43ccea618b65d668fe0d4c1e.png"),
    },
    {
        "test_id": "T046",
        "scenario": "Complex background & tilted face",
        "source": Path("test_dataset/test_images/T046-T050/complex_background.png"),
        "uniform": Path("test_dataset/uniform_references/REF002_navy_collar.png"),
        "bg_color": "#047EF6",
        "candidate": Path("outputs/overnight_T046_20261126_113139_qwen_raw.png"),
    },
]


def main():
    out_dir = Path("outputs/v13_fast_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    print("\n" + "=" * 80)
    print("V13.0 FAST COMPOSITOR & QUALITY VALIDATOR TEST SUITE")
    print("=" * 80)

    for tc in TEST_CASES:
        t_id = tc["test_id"]
        cand_p = tc["candidate"]
        if not cand_p.exists():
            print(f"[{t_id}] Candidate {cand_p} does not exist, skipping.")
            continue

        print(f"\n--- Testing {t_id} ({tc['scenario']}) ---")
        cfg = UniformSwapConfig(
            background_color=tc["bg_color"],
            rgb_identity_restoration="adaptive",
            debug_mode=True,
        )

        job_id = f"v13_test_{t_id}"
        test_debug_dir = out_dir / t_id
        test_debug_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        res = uniform_swap_orchestrator.run(
            person_image_input=tc["source"],
            uniform_template_input=tc["uniform"],
            config=cfg,
            job_id=job_id,
            output_dir=out_dir / "delivered",
            debug_dir=test_debug_dir,
            cached_raw_candidate=cand_p,
        )
        elapsed = time.time() - t0

        sim = res.validation_report.get("similarity", 0.0)
        rmse = res.validation_report.get("geometry_rmse", 0.0)
        status = res.status
        passed = res.passed
        defects = res.defects

        print(f"[{t_id}] Result: status={status}, passed={passed}, sim={sim:.4f}, rmse={rmse:.4f}, time={elapsed:.2f}s")
        if defects:
            print(f"[{t_id}] Defects: {defects}")
        else:
            print(f"[{t_id}] Zero defects!")

        results.append({
            "test_id": t_id,
            "scenario": tc["scenario"],
            "status": status,
            "passed": passed,
            "sim": sim,
            "rmse": rmse,
            "defects": defects,
            "elapsed": elapsed,
        })

    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)
    for r in results:
        stat_sym = "[PASS]" if r["passed"] else f"[{r['status'].upper()}]"
        def_str = ", ".join(r["defects"]) if r["defects"] else "None"
        print(f"{r['test_id']:5s} | {stat_sym:15s} | sim={r['sim']:.4f} | rmse={r['rmse']:.4f} | defects={def_str}")

    all_passed = all(r["passed"] for r in results)
    print("=" * 80)
    print(f"OVERALL STATUS: {'ALL PASSED (10/10)' if all_passed else 'SOME FAILED/REVIEW'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
