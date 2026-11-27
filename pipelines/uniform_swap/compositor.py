"""Adaptive Quality-Aware Compositor & Localized Repair Engine (V13.1).

Implements V13.1 Critical RGBA Compositing:
- Candidate Garment-Only RGBA Extraction:
  - Segments candidate raw output using SCHP LIP-20 and BiRefNet matting.
  - Excludes generated background, generated face, generated ears, and generated hair/braids.
  - Inpaints generated hair over garment fabric to eliminate synthetic braid duplicates.
  - Never pastes rectangular RGB regions over the background.
- 5-Layer Pure RGBA Subject Assembly:
  1. Registered garment-only RGBA (straight alpha).
  2. Protected original neck and upper chest (straight alpha).
  3. Protected original face and ears (straight alpha).
  4. Original hair and braids (straight alpha with BiRefNet hair matte).
  5. Original accessories, ribbons, and floral bows (straight alpha).
- Output is a pristine 4-channel RGBA foreground (composite_rgba).
- Background is applied ONLY ONCE after passport cropping.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from pipelines.uniform_swap.config import UniformSwapConfig

logger = logging.getLogger(__name__)


def _blend_straight_rgba(
    base_rgb: np.ndarray,
    base_alpha: np.ndarray,
    top_rgb: np.ndarray,
    top_alpha: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Porter-Duff 'A over B' compositing for straight (non-premultiplied) alpha."""
    ao = top_alpha.astype(np.float32) / 255.0
    ab = base_alpha.astype(np.float32) / 255.0
    a_out = ao + ab * (1.0 - ao)
    pos = a_out > 0.001

    out_rgb = np.zeros_like(base_rgb, dtype=np.float32)
    out_rgb[pos] = (
        top_rgb.astype(np.float32)[pos] * ao[pos, None]
        + base_rgb.astype(np.float32)[pos] * ab[pos, None] * (1.0 - ao[pos, None])
    ) / a_out[pos, None]

    return np.clip(out_rgb, 0, 255).astype(np.uint8), np.clip(a_out * 255.0, 0, 255).astype(np.uint8)


class UniformCompositor:
    """Manages garment-only extraction and 5-layer RGBA subject compositing."""

    def __init__(self) -> None:
        pass

    def composite(
        self,
        raw_vae_img: Image.Image,
        normalized_source_img: Image.Image,
        identity_layer: Dict[str, Any],
        raw_quality: Dict[str, Any],
        config: UniformSwapConfig,
        job_id: str = "qwen_uniform",
        debug_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Execute garment-only extraction and 5-layer RGBA subject compositing."""
        if debug_dir is None:
            debug_dir = Path("outputs/debug_phases")
        debug_dir.mkdir(parents=True, exist_ok=True)

        w, h = config.output_resolution
        raw_rgb = np.asarray(raw_vae_img.resize((w, h), Image.Resampling.LANCZOS).convert("RGB"))
        src_rgb = np.asarray(normalized_source_img.resize((w, h), Image.Resampling.LANCZOS).convert("RGB"))

        mode = config.rgb_identity_restoration
        exec_mode = config.execution_mode

        # ── Diagnostic / Raw-Only Mode ──
        if exec_mode == "raw_only" or mode == "off":
            logger.info("[COMPOSITOR] Mode is '%s'/'%s': bypassing compositing", exec_mode, mode)
            repair_mask = np.zeros((h, w), dtype=np.uint8)
            return {
                "composite_img": raw_vae_img,
                "composite_rgba": raw_vae_img.convert("RGBA"),
                "assembled_fg_rgb": raw_rgb,
                "assembled_fg_alpha": np.full((h, w), 255, dtype=np.uint8),
                "localized_repair_mask": repair_mask,
                "repair_mask_path": debug_dir / f"{job_id}_17_localized_repair_mask.png",
                "actions": ["bypassed_raw_only"],
            }

        repair_actions = []

        # ── Illumination Correction Check (T031) ──
        src_rgb_to_use = identity_layer.get("illumination_corrected_rgb")
        if src_rgb_to_use is None or src_rgb_to_use.shape != src_rgb.shape:
            src_rgb_to_use = src_rgb
        else:
            repair_actions.append("applied_illumination_corrected_source_rgb")

        # ── Geometry and Masks from Identity Layer ──
        face_m = identity_layer.get("original_face_mask", np.zeros((h, w), dtype=bool))
        ears_m = identity_layer.get("original_ears_mask", np.zeros((h, w), dtype=bool))
        earring_m = identity_layer.get("original_earring_mask", np.zeros((h, w), dtype=bool))
        hair_m = identity_layer.get("original_hair_mask", np.zeros((h, w), dtype=bool))
        acc_m = identity_layer.get("original_accessory_mask", np.zeros((h, w), dtype=bool))
        neck_m = identity_layer.get("original_neck_mask", np.zeros((h, w), dtype=bool))
        garment_m = identity_layer.get("original_garment_mask", np.zeros((h, w), dtype=bool))

        biref_src = identity_layer.get("birefnet_alpha")
        if biref_src is None:
            biref_src = identity_layer.get("full_person_alpha")
        if biref_src is not None:
            src_person_matte = (biref_src.astype(np.float32) / 255.0).clip(0.0, 1.0)
        else:
            src_person_matte = np.ones((h, w), dtype=np.float32)

        chin_y = int(identity_layer.get("metrics", {}).get("chin_y", h * 0.50))
        fh = float(identity_layer.get("metrics", {}).get("face_height", h * 0.35))
        fcx = float(identity_layer.get("metrics", {}).get("face_center_x", w * 0.50))
        fw = float(identity_layer.get("metrics", {}).get("face_width", w * 0.35))

        # Face core: face, ears, and earrings bounded strictly by authentic subject matte
        face_core = face_m.copy()
        if ears_m is not None and np.any(ears_m):
            ears_dil = cv2.dilate(ears_m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            # Strictly prevent ear dilation from spilling into source background
            ears_dil &= (src_person_matte > 0.20)
            face_core |= ears_dil
        if earring_m is not None and np.any(earring_m):
            face_core |= earring_m
        # Bound face_core by valid person matte to prevent source background inclusion
        face_core &= (src_person_matte > 0.20)

        # Braids / long hair draping over chest
        braids_m = (hair_m | acc_m) & (np.arange(h)[:, None] >= chin_y) & (src_person_matte > 0.15)
        head_core_m = face_core | (hair_m & ~braids_m) | (acc_m & ~braids_m)

        # ── Step 1: Candidate Garment-Only RGBA Extraction (Section 3) ──
        logger.info("[COMPOSITOR] Extracting garment-only RGBA from candidate...")
        from pipelines.schp_service import parse as parse_schp
        from pipelines.birefnet_service import extract_matte

        cand_schp = parse_schp(raw_rgb)
        cand_labels = cand_schp["labels"]
        cand_matte = extract_matte(Image.fromarray(raw_rgb))  # (h, w) uint8 [0, 255]

        # Valid candidate garment classes in LIP-20: UpperClothes(5), Dress(6), Coat(7), Torso-skin(10)
        cand_garment_classes = (cand_labels == 5) | (cand_labels == 6) | (cand_labels == 7) | (cand_labels == 10)
        cand_is_person = cand_matte > 35
        cand_below_chin = np.arange(h)[:, None] >= (chin_y - int(0.08 * fh))

        # Initial garment zone: valid garment classes below chin within candidate person matte
        cand_garment_zone = (cand_garment_classes | garment_m) & cand_is_person & cand_below_chin

        # Strictly exclude candidate generated face, ears, and head core
        cand_head = (cand_labels == 13) | head_core_m
        cand_head_dil = cv2.dilate(cand_head.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        cand_garment_zone &= ~cand_head_dil

        # Detect and inpaint candidate generated hair/braids on garment (Section 7)
        raw_hsv = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2HSV)
        raw_dark_hair = (raw_hsv[:, :, 2] < 55) & (raw_hsv[:, :, 1] < 80) & (np.arange(h)[:, None] >= chin_y + int(0.10 * fh))
        cand_hair_m = ((cand_labels == 2) | raw_dark_hair) & cand_is_person & (np.arange(h)[:, None] >= chin_y)

        # Clean generated hair that overlaps the garment fabric
        hair_on_garment = cand_hair_m & (cand_matte > 45) & (np.arange(h)[:, None] >= chin_y + int(0.12 * fh))
        if np.any(braids_m):
            # The deep interior of authentic source braids will be layered on top;
            # avoid destructively inpainting huge areas of fabric under solid braids
            deep_braid_interior = cv2.erode(braids_m.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
            hair_on_garment &= ~deep_braid_interior

        cues = identity_layer.get("cues", {})
        if cues.get("has_tie") or cues.get("ref_type") == "tie":
            tie_corridor = (np.abs(np.arange(w)[None, :] - fcx) < fw * 0.22) & (np.arange(h)[:, None] >= chin_y)
            hair_on_garment &= ~tie_corridor

        cleaned_garment_rgb = raw_rgb.copy()
        inpainted_pixels_total = 0
        if np.any(hair_on_garment) and np.sum(hair_on_garment) > 15:
            cleaned_garment_rgb = cv2.inpaint(
                cleaned_garment_rgb, hair_on_garment.astype(np.uint8), inpaintRadius=5, flags=cv2.INPAINT_TELEA
            )
            inpainted_pixels_total += int(np.sum(hair_on_garment))
            repair_actions.append(f"inpainted_candidate_hair_from_garment ({int(np.sum(hair_on_garment))} px)")

        # Respect authentic uniform attributes from reference (V13.2 Section 4):
        # A red collar does not imply that a red placket is incorrect.
        # Only repair/inpaint red placket if the REFERENCE uniform has NO red placket
        # (e.g. REF001 has check placket) AND candidate generated an unwanted red placket!
        ref_has_red_placket = cues.get("has_red_placket", False)
        ref_has_red_collar = cues.get("has_red_collar", False) or cues.get("ref_type") == "red_collar"

        if ref_has_red_collar and not ref_has_red_placket:
            cr = cleaned_garment_rgb[:, :, 0].astype(int)
            cg = cleaned_garment_rgb[:, :, 1].astype(int)
            cb = cleaned_garment_rgb[:, :, 2].astype(int)
            is_red = (cr > 125) & (cg < 85) & (cb < 85) & ((cr - cg) > 40)
            # Collar sits above chin_y + 0.18*fh. Below that, any red in central corridor is an unwanted red placket
            torso_red = is_red & (np.arange(h)[:, None] >= (chin_y + int(0.18 * fh))) & (np.abs(np.arange(w)[None, :] - fcx) < int(w * 0.16))
            if np.sum(torso_red) > 200:
                red_inpaint_m = cv2.dilate(torso_red.astype(np.uint8), np.ones((7, 7), np.uint8))
                cleaned_garment_rgb = cv2.inpaint(cleaned_garment_rgb, red_inpaint_m, inpaintRadius=11, flags=cv2.INPAINT_TELEA)
                inpainted_pixels_total += int(np.sum(torso_red))
                repair_actions.append(f"repaired_solid_red_placket_to_checked_fabric ({int(np.sum(torso_red))} px)")

        # Ensure authentic black/dark tailored buttons ONLY if reference specifically contains visible black buttons!
        ref_has_buttons = cues.get("button_presence", True) if "button_presence" in cues else (cues.get("has_buttons", False) or cues.get("ref_type") == "red_collar")
        ref_btn_color = cues.get("button_color", "")
        if ref_has_buttons and ref_btn_color == "black":
            ifx = int(fcx)
            p_y1 = min(h, int(chin_y + 0.15 * fh))
            p_y2 = min(h, int(chin_y + 0.85 * fh))
            p_box = cleaned_garment_rgb[p_y1:p_y2, max(0, ifx - 30):min(w, ifx + 30)]
            dark_btn_px = int(np.sum((p_box[:, :, 0] < 55) & (p_box[:, :, 1] < 55) & (p_box[:, :, 2] < 55)))
            if dark_btn_px < 30:
                btn_spacing = max(60, int(0.24 * fh))
                first_btn_y = chin_y + int(0.28 * fh)
                btn_radius = max(5, int(0.016 * fh))
                for by_pos in range(first_btn_y, h - 30, btn_spacing):
                    if by_pos < h - 20:
                        cv2.circle(cleaned_garment_rgb, (ifx, by_pos), btn_radius + 1, (45, 48, 52), -1)
                        cv2.circle(cleaned_garment_rgb, (ifx, by_pos), btn_radius, (25, 27, 30), -1)
                        cv2.circle(cleaned_garment_rgb, (ifx, by_pos), max(1, btn_radius // 3), (70, 72, 75), -1)
                repair_actions.append("synthesized_authentic_tailored_black_buttons")

        # Telea inpainting restriction check (V13.2 Section 5: Restrict inpainting to localized defects)
        garment_area_est = max(1, int(np.sum(cand_garment_zone)))
        repair_fraction = float(inpainted_pixels_total) / garment_area_est
        if repair_fraction > 0.12:
            logger.warning("[COMPOSITOR] Inpainted area exceeds 12%% of garment area (%.2f%%). Flagging extensive inpaint.", repair_fraction * 100)
            repair_actions.append(f"extensive_garment_inpaint_flag ({repair_fraction*100:.1f}%)")

        # Save 12_generated_hair_exclusion_mask.png
        gen_hair_mask_img = (hair_on_garment.astype(np.uint8) * 255)
        Image.fromarray(gen_hair_mask_img).save(debug_dir / f"{job_id}_12_generated_hair_exclusion_mask.png")
        Image.fromarray(gen_hair_mask_img).save(debug_dir / "12_generated_hair_exclusion_mask.png")

        # Garment alpha: smooth matting on candidate silhouette
        garment_alpha = np.where(cand_garment_zone, cand_matte, 0).astype(np.uint8)
        # Morphological close to bridge minor texture gaps in weave
        garment_alpha = cv2.morphologyEx(garment_alpha, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        # Ensure garment alpha does not bleed into head core
        garment_alpha[cand_head_dil] = 0

        # Body gap closure: ensure candidate garment covers authentic student shoulders/torso below chin
        body_gap = (src_person_matte > 0.40) & garment_m & ~head_core_m & ~hair_m & (np.arange(h)[:, None] >= chin_y) & (garment_alpha == 0)
        if np.any(body_gap):
            garment_alpha[body_gap] = 255
            # Inpaint garment fabric texture onto missing shoulder areas
            cleaned_garment_rgb = cv2.inpaint(cleaned_garment_rgb, body_gap.astype(np.uint8), inpaintRadius=9, flags=cv2.INPAINT_TELEA)
            repair_actions.append(f"extended_garment_to_authentic_shoulders ({int(np.sum(body_gap))} px)")

        # Fill internal holes inside candidate garment alpha
        cnts_g, hier_g = cv2.findContours(garment_alpha, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier_g is not None and len(cnts_g) > 0:
            for gi in range(len(cnts_g)):
                if hier_g[0, gi, 3] != -1:  # hole inside garment
                    area_g = cv2.contourArea(cnts_g[gi])
                    if area_g < 8000:
                        cv2.drawContours(garment_alpha, cnts_g, gi, 255, -1)

        # V13.1 Anti-Halo: Suppress white background fringe along candidate garment boundary
        outer_boundary = (garment_alpha > 5) & ~cv2.erode((garment_alpha > 5).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        white_fringe = outer_boundary & (cleaned_garment_rgb[:, :, 0] > 200) & (cleaned_garment_rgb[:, :, 1] > 200) & (cleaned_garment_rgb[:, :, 2] > 200)
        if np.any(white_fringe):
            garment_alpha[white_fringe] = 0
            dense_fringe = outer_boundary & (cleaned_garment_rgb[:, :, 0] > 215) & (cleaned_garment_rgb[:, :, 1] > 215) & (cleaned_garment_rgb[:, :, 2] > 215)
            if np.any(dense_fringe):
                cleaned_garment_rgb = cv2.inpaint(cleaned_garment_rgb, dense_fringe.astype(np.uint8), inpaintRadius=5, flags=cv2.INPAINT_TELEA)
            repair_actions.append("suppressed_candidate_white_edge_halo")

        # Save Section 15 debug outputs:
        # Candidate garment RGB, alpha, and registered RGBA
        garment_rgba = np.dstack([cleaned_garment_rgb, garment_alpha])
        Image.fromarray(cleaned_garment_rgb).save(debug_dir / f"{job_id}_04_candidate_garment_rgb.png")
        Image.fromarray(cleaned_garment_rgb).save(debug_dir / "04_candidate_garment_rgb.png")
        Image.fromarray(garment_alpha).save(debug_dir / f"{job_id}_04_candidate_garment_alpha.png")
        Image.fromarray(garment_alpha).save(debug_dir / "04_candidate_garment_alpha.png")
        Image.fromarray(garment_rgba).save(debug_dir / f"{job_id}_04_registered_garment_rgba.png")
        Image.fromarray(garment_rgba).save(debug_dir / "04_registered_garment_rgba.png")

        # ── Step 2: 5-Layer Compositing in Straight RGBA (Section 4) ──
        logger.info("[COMPOSITOR] Assembling 5-layer straight RGBA subject...")

        # Initialize transparent RGBA canvas
        fg_rgb = cleaned_garment_rgb.copy()
        fg_alpha = garment_alpha.copy()

        # Layer 2: Protected Original Neck and Upper Chest
        # Feathers throat skin into open collar neckline
        if np.any(neck_m):
            neck_alpha = cv2.GaussianBlur(neck_m.astype(np.float32) * 255.0, (5, 5), 1.5).clip(0, 255).astype(np.uint8)
            # Neck sits inside open collar opening; where garment collar has fabric, garment takes precedence
            # But feather the transition seamlessly along collar edge
            neck_alpha = np.minimum(neck_alpha, 255)
            fg_rgb, fg_alpha = _blend_straight_rgba(fg_rgb, fg_alpha, src_rgb_to_use, neck_alpha)
            repair_actions.append("layered_protected_original_neck_and_throat")

        # Layer 3: Protected Original Face and Ears
        face_alpha = cv2.GaussianBlur(face_core.astype(np.float32) * 255.0, (3, 3), 1.0).clip(0, 255).astype(np.uint8)
        face_alpha[face_core] = 255  # 100% solid interior
        fg_rgb, fg_alpha = _blend_straight_rgba(fg_rgb, fg_alpha, src_rgb_to_use, face_alpha)
        repair_actions.append("layered_protected_original_face_and_ears")

        # Layer 4: Original Hair and Braids
        if np.any(hair_m):
            if biref_src is not None:
                hair_alpha = np.where(hair_m & (biref_src > 30), biref_src, 0).astype(np.uint8)
            else:
                hair_alpha = (hair_m.astype(np.float32) * 255.0).astype(np.uint8)

            # De-fringe: Suppress white background leakage along hair silhouette
            src_is_white = (src_rgb_to_use[:, :, 0] > 190) & (src_rgb_to_use[:, :, 1] > 190) & (src_rgb_to_use[:, :, 2] > 190)
            hair_white_fringe = hair_m & src_is_white
            if np.any(hair_white_fringe):
                hair_alpha[hair_white_fringe & (hair_alpha < 180)] = 0
                dense_fringe = hair_white_fringe & (hair_alpha >= 180)
                if np.any(dense_fringe):
                    src_rgb_to_use = cv2.inpaint(src_rgb_to_use, dense_fringe.astype(np.uint8), inpaintRadius=5, flags=cv2.INPAINT_TELEA)
                    repair_actions.append("defringed_hair_silhouette_white_boundary")

            # Save 06_hair_alpha.png (Section 15)
            Image.fromarray(hair_alpha).save(debug_dir / f"{job_id}_06_hair_alpha.png")
            Image.fromarray(hair_alpha).save(debug_dir / "06_hair_alpha.png")

            fg_rgb, fg_alpha = _blend_straight_rgba(fg_rgb, fg_alpha, src_rgb_to_use, hair_alpha)
            repair_actions.append("layered_original_hair_and_braids_with_alpha_matte")

        # Layer 5: Original Accessories and Ribbons (placed ON TOP of uniform fabric)
        if np.any(acc_m):
            acc_alpha = cv2.GaussianBlur(acc_m.astype(np.float32) * 255.0, (3, 3), 1.0).clip(0, 255).astype(np.uint8)
            acc_alpha[acc_m] = 255
            fg_rgb, fg_alpha = _blend_straight_rgba(fg_rgb, fg_alpha, src_rgb_to_use, acc_alpha)
            repair_actions.append("layered_authentic_hair_accessories_and_ribbons_on_top")

        # Solid core integrity: inside solid interior of student, alpha must be 255
        solid_interior = face_core | (cand_garment_zone & (cand_matte > 200))
        fg_alpha[solid_interior] = 255

        # Check for any synthetic holes trapped inside the assembled torso below chin
        cnts_fa, hier_fa = cv2.findContours((fg_alpha > 30).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier_fa is not None and len(cnts_fa) > 0:
            for fai in range(len(cnts_fa)):
                if hier_fa[0, fai, 3] != -1:  # child contour (internal hole)
                    area_fa = cv2.contourArea(cnts_fa[fai])
                    if area_fa > 50:
                        bx, by, bw, bh = cv2.boundingRect(cnts_fa[fai])
                        if (by + bh * 0.5) > int(chin_y - 0.35 * fh) and area_fa < 25000:
                            hole_fa_m = np.zeros((h, w), dtype=np.uint8)
                            cv2.drawContours(hole_fa_m, cnts_fa, fai, 1, -1)
                            is_nat_bg = False
                            if src_rgb is not None:
                                s_corner = np.mean(src_rgb[:10, :10], axis=(0, 1))
                                s_dist = np.max(np.abs(src_rgb.astype(float) - s_corner), axis=2)
                                if np.mean(s_dist[hole_fa_m > 0] < 45) > 0.25:
                                    is_nat_bg = True
                            if not is_nat_bg:
                                fg_alpha[hole_fa_m > 0] = 255
                                fg_rgb = cv2.inpaint(fg_rgb, hole_fa_m, inpaintRadius=11, flags=cv2.INPAINT_TELEA)
                                repair_actions.append(f"filled_assembled_torso_alpha_hole ({int(area_fa)} px)")

        # ── Step 3: Save Assembled Foreground RGBA (Section 15) ──
        assembled_rgba = np.dstack([fg_rgb, fg_alpha])
        composite_rgba_pil = Image.fromarray(assembled_rgba, "RGBA")

        Image.fromarray(assembled_rgba).save(debug_dir / f"{job_id}_17_assembled_foreground_rgba.png")
        Image.fromarray(assembled_rgba).save(debug_dir / "17_assembled_foreground_rgba.png")
        Image.fromarray(fg_rgb).save(debug_dir / f"{job_id}_17_assembled_foreground_rgb.png")
        Image.fromarray(fg_rgb).save(debug_dir / "17_assembled_foreground_rgb.png")
        Image.fromarray(fg_alpha).save(debug_dir / f"{job_id}_17_assembled_foreground_alpha.png")
        Image.fromarray(fg_alpha).save(debug_dir / "17_assembled_foreground_alpha.png")

        # Update identity_layer so downstream validators and crop transformer access clean RGBA
        identity_layer["assembled_fg_rgba"] = assembled_rgba
        identity_layer["full_person_alpha"] = fg_alpha
        identity_layer["subject_matte"] = fg_alpha.astype(np.float32) / 255.0

        # Create flat preview RGB composite for pre-crop validation
        target_bg_color = getattr(config, "background_color", (255, 255, 255))
        if not isinstance(target_bg_color, tuple) or len(target_bg_color) != 3:
            target_bg_color = UniformSwapConfig.parse_bg_color(target_bg_color)

        sm_3ch = (fg_alpha.astype(np.float32) / 255.0)[:, :, None]
        bg_arr = np.zeros_like(fg_rgb, dtype=np.float32)
        bg_arr[:, :] = target_bg_color
        precrop_rgb = np.clip(fg_rgb.astype(np.float32) * sm_3ch + bg_arr * (1.0 - sm_3ch), 0, 255).astype(np.uint8)
        precrop_rgb[fg_alpha < 5] = target_bg_color
        final_composite_img = Image.fromarray(precrop_rgb, "RGB")

        # Save 17_protected_source_composite.png & 09_white_background_composite.png
        final_composite_img.save(debug_dir / f"{job_id}_17_protected_source_composite.png")
        final_composite_img.save(debug_dir / "17_protected_source_composite.png")
        final_composite_img.save(debug_dir / f"{job_id}_09_white_background_composite.png")
        final_composite_img.save(debug_dir / "09_white_background_composite.png")

        localized_repair_mask = (garment_alpha > 0).astype(np.uint8) * 255
        repair_mask_path = debug_dir / f"{job_id}_17_localized_repair_mask.png"
        Image.fromarray(localized_repair_mask).save(repair_mask_path)
        Image.fromarray(localized_repair_mask).save(debug_dir / "17_localized_repair_mask.png")

        return {
            "composite_img": final_composite_img,
            "composite_rgba": composite_rgba_pil,
            "assembled_fg_rgb": fg_rgb,
            "assembled_fg_alpha": fg_alpha,
            "localized_repair_mask": localized_repair_mask,
            "repair_mask_path": repair_mask_path,
            "actions": repair_actions,
        }


# Global singleton instance
uniform_compositor = UniformCompositor()
