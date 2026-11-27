"""Generate 4-Panel Photographic Quality Comparison Sheets (V13.3.2).

As required by Section 4:
Panels:
1. Original source portrait
2. Target uniform reference
3. Previous V13.3.1 output (from outputs/production_acceptance/v13_3_1_baseline/)
4. New V13.3.2 output (from outputs/production_acceptance/{tid}_debug/20_final_output.png)

Outputs saved to: outputs/production_acceptance/comparisons/{tid}_v13_3_2_comparison.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import cv2
import numpy as np

BASE = Path("outputs/production_acceptance")
BASELINE_DIR = BASE / "v13_3_1_baseline"
MANIFEST = BASE / "provenance_manifest.jsonl"
OUT_DIR = BASE / "comparisons"
PANEL_W, PANEL_H = 400, 500
PAD, HEADER = 14, 70


def load_and_fit_panel(path: Path | None, title: str, subtitle: str = "") -> np.ndarray:
    """Load an image, center-fit it on a fixed canvas with a border and clear label."""
    canvas = np.full((PANEL_H, PANEL_W, 3), 242, dtype=np.uint8)
    img = cv2.imread(str(path)) if path and Path(path).exists() else None
    if img is None:
        cv2.putText(canvas, "NOT AVAILABLE", (PANEL_W // 2 - 80, PANEL_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 2)
    else:
        h, w = img.shape[:2]
        scale = min((PANEL_W - 16) / w, (PANEL_H - 48) / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        y0 = (PANEL_H - 36 - nh) // 2 + 10
        x0 = (PANEL_W - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized

    # Outer border
    cv2.rectangle(canvas, (0, 0), (PANEL_W - 1, PANEL_H - 1), (180, 180, 180), 1)

    # Title banner at bottom
    cv2.rectangle(canvas, (0, PANEL_H - 32), (PANEL_W, PANEL_H), (225, 225, 225), -1)
    cv2.putText(canvas, title, (10, PANEL_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1)
    if subtitle:
        cv2.putText(canvas, subtitle, (PANEL_W - 120, PANEL_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 30, 30), 1)

    return canvas


def build_comparison_sheet(rec: dict) -> Path:
    """Build a side-by-side 4-panel comparison sheet for an executed case."""
    tid = rec["test_id"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    src_p = Path(rec.get("source", ""))
    ref_p = Path(rec.get("reference", ""))
    v131_p = BASELINE_DIR / f"{tid}_v13_3_1.png"
    dbg_dir = BASE / f"{tid}_debug"
    v132_p = dbg_dir / "20_final_output.png"
    if not v132_p.exists():
        v132_p = dbg_dir / "final_candidate.png"

    status = rec.get("status", "unknown").upper()
    sim = rec.get("similarity", 0.0)
    dur = rec.get("duration_s", 0.0)
    defects = rec.get("defects", [])
    vram = rec.get("peak_vram_alloc_mb", 0.0)

    p1 = load_and_fit_panel(src_p, "1. SOURCE PORTRAIT", src_p.name)
    p2 = load_and_fit_panel(ref_p, "2. UNIFORM REFERENCE", ref_p.name)
    p3 = load_and_fit_panel(v131_p, "3. PREVIOUS V13.3.1 OUTPUT", "Prior Baseline")
    p4 = load_and_fit_panel(v132_p, "4. NEW V13.3.2 OUTPUT", f"Status: {status}")

    # Layout 4 panels side by side in a 2x2 grid or 1x4 horizontal strip
    # 2x2 grid: width = 2 * PANEL_W + 3 * PAD, height = HEADER + 2 * PANEL_H + 3 * PAD
    sheet_w = 2 * PANEL_W + 3 * PAD
    sheet_h = HEADER + 2 * PANEL_H + 3 * PAD
    sheet = np.full((sheet_h, sheet_w, 3), 250, dtype=np.uint8)

    # Header
    status_col = (0, 140, 0) if status == "COMPLETED" else (0, 100, 220)
    cv2.putText(sheet, f"V13.3.2 Quality Verification: {tid}", (PAD, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2)
    meta_str = f"Status: {status} | Sim: {sim:.4f} | Time: {dur:.1f}s | VRAM: {vram:.0f}MB | Category: {rec.get('category', '')}"
    cv2.putText(sheet, meta_str, (PAD, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.44, status_col, 1)

    # Place panels
    panels = [p1, p2, p3, p4]
    for idx, panel in enumerate(panels):
        r, c = divmod(idx, 2)
        x = PAD + c * (PANEL_W + PAD)
        y = HEADER + PAD + r * (PANEL_H + PAD)
        sheet[y:y + PANEL_H, x:x + PANEL_W] = panel

    out_file = OUT_DIR / f"{tid}_v13_3_2_comparison.png"
    cv2.imwrite(str(out_file), sheet)
    print(f"Generated comparison sheet: {out_file}")
    return out_file


def main():
    if not MANIFEST.exists():
        print("No provenance manifest found.")
        return 1

    lines = [l for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    records = {}
    for line in lines:
        try:
            r = json.loads(line)
            records[r["test_id"]] = r
        except Exception:
            pass

    print(f"Generating 4-panel comparison sheets for {len(records)} case(s)...")
    for tid, rec in records.items():
        build_comparison_sheet(rec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
