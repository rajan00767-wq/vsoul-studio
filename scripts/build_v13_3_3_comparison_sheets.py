"""Generate 8-panel comparison sheets for V13.3.3 Visual Audit.

Panels:
1. Source portrait
2. Uniform reference
3. Raw Qwen candidate
4. V13.3.2 final output
5. V13.3.3 final output
6. Semantic layer masks
7. Enlarged neck and collar region
8. Enlarged accessory and shoulder regions
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_v13_3_3_sheets")

CASES = [
    {
        "id": "PA_01_Curly_RedCollar",
        "debug_dir": "PA_01_Curly_RedCollar_debug",
        "ref": "test_dataset/uniform_references/REF001_red_collar.png",
        "neck_box": (180, 500, 620, 850),
        "shoulder_box": (30, 650, 420, 1020),
    },
    {
        "id": "PA_02_Braids_NavyCollar",
        "debug_dir": "PA_02_Braids_NavyCollar_debug",
        "ref": "test_dataset/uniform_references/REF002_navy_collar.png",
        "neck_box": (180, 450, 620, 800),
        "shoulder_box": (420, 650, 780, 1020),
    },
    {
        "id": "PA_03_MaroonBows_Tie",
        "debug_dir": "PA_03_MaroonBows_Tie_debug",
        "ref": "test_dataset/uniform_references/REF003_tie.png",
        "neck_box": (200, 450, 600, 800),
        "shoulder_box": (40, 700, 400, 1020),
    },
    {
        "id": "PA_04_NoBows_Vest",
        "debug_dir": "PA_04_NoBows_Vest_debug",
        "ref": "test_dataset/uniform_references/REF004_vest.png",
        "neck_box": (200, 450, 600, 800),
        "shoulder_box": (420, 650, 780, 1020),
    },
    {
        "id": "PA_05_GlassesBindi_RedCollar",
        "debug_dir": "PA_05_GlassesBindi_RedCollar_debug",
        "ref": "test_dataset/uniform_references/REF001_red_collar.png",
        "neck_box": (180, 500, 620, 850),
        "shoulder_box": (40, 650, 420, 1020),
    },
    {
        "id": "PA_11_ShortBoy_Tie",
        "debug_dir": "PA_11_ShortBoy_Tie_debug",
        "ref": "test_dataset/uniform_references/REF003_tie.png",
        "neck_box": (200, 450, 600, 800),
        "shoulder_box": (40, 650, 420, 1020),
    },
    {
        "id": "PA_12_StraightHair_RedCollar",
        "debug_dir": "PA_12_StraightHair_RedCollar_debug",
        "ref": "test_dataset/uniform_references/REF001_red_collar.png",
        "neck_box": (180, 480, 620, 830),
        "shoulder_box": (40, 650, 420, 1020),
    },
]


def add_title(img: Image.Image, text: str, subtext: str = "") -> Image.Image:
    """Add title bar to image."""
    w, h = img.size
    header_h = 75
    canvas = Image.new("RGB", (w, h + header_h), (26, 32, 44))
    canvas.paste(img, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 12), text, fill=(255, 255, 255))
    if subtext:
        draw.text((15, 42), subtext, fill=(203, 213, 225))
    return canvas


def make_semantic_mask_vis(debug_path: Path, size: Tuple[int, int]) -> Image.Image:
    """Build composite visualization of semantic layers."""
    w, h = size
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[:, :] = [240, 240, 240]

    garment_p = debug_path / "04_candidate_garment_alpha.png"
    if garment_p.exists():
        gm = np.array(Image.open(garment_p).resize((w, h))) > 35
        vis[gm] = [100, 150, 240]  # Garment blue

    neck_p = debug_path / "08_neck_mask.png"
    if neck_p.exists():
        nm = np.array(Image.open(neck_p).resize((w, h))) > 35
        vis[nm] = [230, 180, 120]  # Neck tan

    hair_p = debug_path / "06_hair_mask.png"
    if hair_p.exists():
        hm = np.array(Image.open(hair_p).resize((w, h))) > 35
        vis[hm] = [70, 50, 50]  # Hair brown

    face_p = debug_path / "05_face_mask.png"
    if face_p.exists():
        fm = np.array(Image.open(face_p).resize((w, h))) > 35
        vis[fm] = [245, 210, 180]  # Face skin

    acc_p = debug_path / "07_accessory_mask.png"
    if acc_p.exists():
        am = np.array(Image.open(acc_p).resize((w, h))) > 35
        vis[am] = [180, 30, 60]  # Accessory red/maroon

    return Image.fromarray(vis, "RGB")


def make_crop_zoom(img: Image.Image, box: Tuple[int, int, int, int], target_size: Tuple[int, int]) -> Image.Image:
    """Crop region and resize smoothly."""
    x1, y1, x2, y2 = box
    w, h = img.size
    x1 = max(0, min(w - 10, x1))
    y1 = max(0, min(h - 10, y1))
    x2 = max(x1 + 10, min(w, x2))
    y2 = max(y1 + 10, min(h, y2))
    cropped = img.crop((x1, y1, x2, y2))
    return cropped.resize(target_size, Image.Resampling.LANCZOS)


def main():
    base_out = Path("outputs/production_acceptance")
    cand_dir = base_out / "v13_3_3_candidates"
    cmp_dir = base_out / "comparisons"
    cmp_dir.mkdir(parents=True, exist_ok=True)

    panel_w, panel_h = 400, 512

    for case in CASES:
        cid = case["id"]
        c_dir = base_out / case["debug_dir"]
        c_debug = cand_dir / f"{cid}_debug"

        src_p = c_dir / "02_normalized_source.png"
        ref_p = Path(case["ref"])
        raw_p = c_dir / "raw_candidate_resized.png"
        v13_2_p = base_out / f"{cid}_uniform.png"
        v13_3_p = cand_dir / f"{cid}_uniform_v13_3_3.png"

        if not v13_3_p.exists():
            logger.warning("[%s] V13.3.3 output missing, skipping sheet", cid)
            continue

        src_img = Image.open(src_p).convert("RGB").resize((panel_w, panel_h), Image.Resampling.LANCZOS)
        ref_img = Image.open(ref_p).convert("RGB").resize((panel_w, panel_h), Image.Resampling.LANCZOS) if ref_p.exists() else Image.new("RGB", (panel_w, panel_h), (200, 200, 200))
        raw_img = Image.open(raw_p).convert("RGB").resize((panel_w, panel_h), Image.Resampling.LANCZOS)
        v13_2_img = Image.open(v13_2_p).convert("RGB").resize((panel_w, panel_h), Image.Resampling.LANCZOS)
        v13_3_img = Image.open(v13_3_p).convert("RGB").resize((panel_w, panel_h), Image.Resampling.LANCZOS)

        # Panel 6: Semantic Layer Masks
        mask_img = make_semantic_mask_vis(c_debug if c_debug.exists() else c_dir, (panel_w, panel_h))

        # Panel 7: Neck/Collar Zoom on V13.3.3
        neck_zoom = make_crop_zoom(v13_3_img, case["neck_box"], (panel_w, panel_h))

        # Panel 8: Accessory/Shoulder Zoom on V13.3.3
        sh_zoom = make_crop_zoom(v13_3_img, case["shoulder_box"], (panel_w, panel_h))

        # Build 8 panels with headers
        p1 = add_title(src_img, "1. Source Portrait", "Immutable authentic subject")
        p2 = add_title(ref_img, "2. Uniform Reference", case["ref"].split("/")[-1])
        p3 = add_title(raw_img, "3. Raw Qwen Candidate", "Unwarped generative bust")
        p4 = add_title(v13_2_img, "4. V13.3.2 Output", "Defective: erased collar/seams")
        p5 = add_title(v13_3_img, "5. V13.3.3 Output", "Registered & Occlusion-Aware")
        p6 = add_title(mask_img, "6. Semantic Layer Masks", "7-layer decomposition")
        p7 = add_title(neck_zoom, "7. Neck & Collar Zoom", "Natural lapel & throat hollow")
        p8 = add_title(sh_zoom, "8. Accessory & Shoulder Zoom", "Clean border, zero black lines")

        # Layout: 2 rows of 4 columns
        # Each panel is panel_w x (panel_h + 75)
        pw = panel_w
        ph = panel_h + 75
        sheet_w = pw * 4
        sheet_h = ph * 2 + 80  # 80px main banner

        sheet = Image.new("RGB", (sheet_w, sheet_h), (15, 23, 42))
        draw = ImageDraw.Draw(sheet)

        # Banner
        draw.text((30, 22), f"V13.3.3 VISUAL AUDIT & REGRESSION INSPECTION: {cid}", fill=(255, 255, 255))
        draw.text((30, 50), "Anatomical Garment Registration & 7-Layer Semantic Compositing vs V13.3.2", fill=(148, 163, 184))

        # Row 1: Panels 1-4
        sheet.paste(p1, (pw * 0, 80))
        sheet.paste(p2, (pw * 1, 80))
        sheet.paste(p3, (pw * 2, 80))
        sheet.paste(p4, (pw * 3, 80))

        # Row 2: Panels 5-8
        sheet.paste(p5, (pw * 0, 80 + ph))
        sheet.paste(p6, (pw * 1, 80 + ph))
        sheet.paste(p7, (pw * 2, 80 + ph))
        sheet.paste(p8, (pw * 3, 80 + ph))

        sheet_path = cmp_dir / f"{cid}_v13_3_3_comparison.png"
        sheet.save(sheet_path)
        logger.info("[%s] Saved 8-panel comparison sheet to %s", cid, sheet_path)

    print(f"\nAll comparison sheets successfully saved in {cmp_dir}")


if __name__ == "__main__":
    main()
