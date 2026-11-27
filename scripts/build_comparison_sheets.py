"""Six-panel comparison sheets for executed production-acceptance cases.

Panels, per the V13.3.1 specification section 12:
  1. Original source portrait
  2. Target uniform reference
  3. Raw Qwen candidate
  4. Corrected candidate
  5. Final delivered output (or the gated candidate, clearly labelled)
  6. Quality metrics and execution status

Driven entirely by outputs/production_acceptance/provenance_manifest.jsonl, so
it never invents a row for a case that was not executed.

    .venv/Scripts/python.exe scripts/build_comparison_sheets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

BASE = Path("outputs/production_acceptance")
MANIFEST = BASE / "provenance_manifest.jsonl"
SHEETS = BASE / "comparisons"
PANEL_W, PANEL_H = 360, 460
PAD, HEADER = 12, 64


def _panel(path: Path | None, label: str) -> np.ndarray:
    canvas = np.full((PANEL_H, PANEL_W, 3), 236, dtype=np.uint8)
    img = cv2.imread(str(path)) if path and Path(path).exists() else None
    if img is None:
        cv2.putText(canvas, "not available", (PANEL_W // 2 - 86, PANEL_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
    else:
        h, w = img.shape[:2]
        s = min((PANEL_W - 8) / w, (PANEL_H - 8) / h)
        rs = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        y0 = (PANEL_H - rs.shape[0]) // 2
        x0 = (PANEL_W - rs.shape[1]) // 2
        canvas[y0:y0 + rs.shape[0], x0:x0 + rs.shape[1]] = rs
    cv2.rectangle(canvas, (0, 0), (PANEL_W - 1, PANEL_H - 1), (190, 190, 190), 1)
    cv2.putText(canvas, label, (8, PANEL_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (60, 60, 60), 1)
    return canvas


def _metrics_panel(rec: dict) -> np.ndarray:
    canvas = np.full((PANEL_H, PANEL_W, 3), 250, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (PANEL_W - 1, PANEL_H - 1), (190, 190, 190), 1)
    st = rec.get("stage_durations_s", {})
    lines = [
        ("6. METRICS & STATUS", (30, 30, 30)),
        ("", None),
        (f"status      : {rec.get('status')}",
         (0, 140, 0) if rec.get("status") == "completed" else (0, 110, 215)),
        (f"provenance  : {str(rec.get('provenance',''))[:30]}", (60, 60, 60)),
        (f"invocations : {rec.get('inference_invocations')}", (60, 60, 60)),
        (f"similarity  : {rec.get('similarity')}", (60, 60, 60)),
        (f"geom RMSE   : {rec.get('geometry_rmse')}", (60, 60, 60)),
        (f"total       : {rec.get('duration_s')} s", (60, 60, 60)),
        (f"diffusion   : {round(st.get('qwen_diffusion_and_raw_decode', 0.0), 1)} s", (60, 60, 60)),
        (f"peak VRAM   : {rec.get('peak_vram_alloc_mb')} MB", (60, 60, 60)),
        (f"synthetic   : {rec.get('synthetic_source')}", (60, 60, 60)),
        ("", None),
        (f"defects ({len(rec.get('defects', []))}):", (30, 30, 30)),
    ]
    y = 26
    for text, col in lines:
        if text:
            cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col or (60, 60, 60), 1)
        y += 19
    for d in rec.get("defects", [])[:6]:
        for chunk in [d[i:i + 42] for i in range(0, min(len(d), 84), 42)]:
            cv2.putText(canvas, f"  {chunk}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0, 0, 190), 1)
            y += 15
    if not rec.get("defects"):
        cv2.putText(canvas, "  none", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 140, 0), 1)
    return canvas


def build() -> int:
    if not MANIFEST.exists():
        print(f"No manifest at {MANIFEST}. Run scripts/run_production_acceptance.py first.")
        return 1
    SHEETS.mkdir(parents=True, exist_ok=True)

    made = 0
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        tid = rec["test_id"]
        dbg = BASE / f"{tid}_debug"
        delivered = rec.get("delivered_path")
        final_label = "5. DELIVERED" if delivered else "5. GATED (not delivered)"
        panels = [
            _panel(Path(rec["source"]) if rec.get("source") else None, "1. SOURCE"),
            _panel(Path(rec["reference"]) if rec.get("reference") else None, "2. UNIFORM REFERENCE"),
            _panel(Path("outputs") / f"{tid}_qwen_raw.png", "3. RAW QWEN CANDIDATE"),
            _panel(dbg / "18_final_output.png", "4. CORRECTED CANDIDATE"),
            _panel(Path(delivered) if delivered else dbg / "20_final_output.png", final_label),
            _metrics_panel(rec),
        ]
        cols = 3
        rows = 2
        sheet_w = cols * PANEL_W + (cols + 1) * PAD
        sheet_h = HEADER + rows * PANEL_H + (rows + 1) * PAD
        sheet = np.full((sheet_h, sheet_w, 3), 248, dtype=np.uint8)
        col = (0, 140, 0) if rec.get("status") == "completed" else (0, 110, 215)
        cv2.putText(sheet, f"{tid}  [{str(rec.get('status','')).upper()}]  {rec.get('category','')}",
                    (PAD, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.74, col, 2)
        for i, p in enumerate(panels):
            r, c = divmod(i, cols)
            y = HEADER + PAD + r * (PANEL_H + PAD)
            x = PAD + c * (PANEL_W + PAD)
            sheet[y:y + PANEL_H, x:x + PANEL_W] = p
        out = SHEETS / f"{tid}_comparison.png"
        cv2.imwrite(str(out), sheet)
        print(f"  wrote {out}")
        made += 1

    print(f"\n{made} comparison sheet(s) in {SHEETS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
