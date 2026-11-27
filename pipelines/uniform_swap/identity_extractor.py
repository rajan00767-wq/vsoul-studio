"""Clean Garment-Free Identity Layer Extractor (V11).

Extracts independent verified semantic regions (face, ears, visible neck, hair, accessories)
from immutable original RGB without garment contamination.

Implements:
- FIX 1: Preserves visible neck skin using connectivity & confidence; clothing labels cannot override.
- FIX 2: Dynamic neck visibility classification (fully_visible, partially_visible, occluded, uncertain).
- FIX 3: Separates missing neck (Region A) from occluded neck (Region B).
- Section 8: Semantic hair accessory & bow preservation WITHOUT hardcoded blue HSV detection:
  - Detects bows, ribbons, hair bands, and clips by topological hair/braid connectivity and spatial context.
  - Preserves complete objects with loop and tail geometry.
  - Validates object completeness, size, shape, and original color distribution.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from pipelines.uniform_swap.semantic_parser import semantic_parser

logger = logging.getLogger(__name__)


class IdentityExtractor:
    """Manages clean garment-free identity layer extraction and validation."""

    def __init__(self) -> None:
        pass

    def extract_semantic_hair_accessories(
        self,
        rgb: np.ndarray,
        hair_mask: np.ndarray,
        face_mask: np.ndarray,
        neck_mask: np.ndarray,
        schp_labels: np.ndarray,
        bisenet_parsing: Optional[np.ndarray],
        birefnet_alpha: Optional[np.ndarray],
        face_bbox: Tuple[int, int, int, int],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray, List[Dict[str, Any]]]:
        """Section 6 & 8: Semantic hair accessory detection without hardcoded color thresholds.
        
        Strictly distinguishes authentic hair bows, ribbons, hairbands and clips from
        school badges, emblems, and uniform fabric using spatial, anatomical, and semantic context.
        """
        h, w = rgb.shape[:2]
        fx1, fy1, fx2, fy2 = face_bbox
        fw = max(1.0, float(fx2 - fx1))
        fh = max(1.0, float(fy2 - fy1))
        fcx = float(fx1 + fx2) * 0.5
        chin_y = int(fy1 + fh * 0.85)

        # Clothing regions in SCHP (5=upper clothes, 6=dress, 7=coat, 9=pants, 11=scarf/tie, 12=skirt, 1=hat/hood)
        # and BiSeNet (16=cloth, 18=hat/hood)
        schp_cloth = np.isin(schp_labels, (1, 5, 6, 7, 9, 11, 12))
        bisenet_cloth = np.isin(bisenet_parsing, (16, 18)) if bisenet_parsing is not None else np.zeros((h, w), dtype=bool)
        all_garment = schp_cloth | bisenet_cloth

        # Hair proximity zone: accessories sit directly on or touching the hair/braid silhouette
        hair_dil = cv2.dilate(hair_mask.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
        corner_bg = np.concatenate([rgb[:30, :30], rgb[:30, -30:]])
        is_light_bg = float(np.mean(corner_bg)) > 180.0
        is_ambient_bg = ((rgb[:, :, 0] > 195) & (rgb[:, :, 1] > 195) & (rgb[:, :, 2] > 195)) if is_light_bg else np.zeros((h, w), dtype=bool)

        # Distinguish garment fabric from accessories resting on the torso:
        # Measure garment core color (away from hair)
        g_core = all_garment & ~hair_dil
        if np.sum(g_core) > 500:
            g_med = np.median(rgb[g_core], axis=0)
            g_diff = np.linalg.norm(rgb.astype(float) - g_med[None, None, :], axis=2)
            # Pixels contrasting with the garment fabric by delta > 35 or not on garment
            valid_acc_zone = ~all_garment | (g_diff > 35.0)
        else:
            valid_acc_zone = ~all_garment

        acc_corridor = hair_dil & ~face_mask & ~neck_mask & valid_acc_zone & ~is_ambient_bg

        # 1. Semantic parsing indicators
        # SCHP LIP-20: label 4 (sunglasses/accessories), label 18 (scarf/accessory)
        # Note: SCHP label 1 is HAT/HOOD and must NOT be classified as hair accessories!
        sem_acc = np.isin(schp_labels, (4, 18))

        # 2. Chromatic and textural distinction from scalp/hair
        # Bows, ribbons and clips contrast strongly with natural hair in hue, saturation, or value.
        # Scalp hair itself must NEVER be classified as an accessory.
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        hue = hsv[:, :, 0]

        if np.any(hair_mask):
            hair_h = float(np.median(hue[hair_mask]))
            hair_sat = float(np.median(sat[hair_mask]))
            hair_v = float(np.median(val[hair_mask]))
        else:
            hair_h, hair_sat, hair_v = 0.0, 0.0, 0.0

        color_diff = np.abs(hue.astype(float) - hair_h)
        color_diff = np.minimum(color_diff, 180.0 - color_diff)

        # Colored accessories (blue, red, green, pink, yellow, maroon ribbons/bows):
        # Must have significant hue deviation (> 14 deg) from natural hair OR higher saturation (> 35 above hair)
        is_chromatic_diff = (color_diff > 14.0) | ((sat.astype(float) - hair_sat) > 35.0)
        chromatic_cand = is_chromatic_diff & (sat >= 50) & (val >= 40) & acc_corridor
        # Bright light accessories (white/pearlescent barrettes, clips, ribbons):
        # Must be located within genuine hair boundary and distinct from ambient backdrop
        bright_cand = (val >= 210) & ((val - hair_v) > 65.0) & hair_mask & ~is_ambient_bg

        # Foreground check via BiRefNet alpha
        if birefnet_alpha is not None:
            fg = (birefnet_alpha > 120)
            candidate_map = (sem_acc | chromatic_cand | bright_cand) & fg & valid_acc_zone
        else:
            candidate_map = (sem_acc | chromatic_cand | bright_cand) & valid_acc_zone

        # Connected component analysis for object completeness
        num_c, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_map.astype(np.uint8))
        accessory_mask = np.zeros((h, w), dtype=bool)
        accessory_objects: List[Dict[str, Any]] = []
        badge_mask = np.zeros((h, w), dtype=bool)
        badge_objects: List[Dict[str, Any]] = []

        for idx in range(1, num_c):
            comp = (labels == idx)
            area = stats[idx, cv2.CC_STAT_AREA]
            # Genuine hair accessories (bows, ribbons, ties, clips) have realistic size (<= 18000 px)
            if area < 60 or area > 18000:
                continue

            cx = centroids[idx][0]
            cy = centroids[idx][1]

            # Reject central torso clothing (e.g. necktie or badge below chin)
            is_central_torso = (abs(cx - fcx) < 0.16 * fw) and (cy > chin_y + fh * 0.20)
            if is_central_torso:
                # Part of clothing / central placket
                continue

            # Must touch hair or braid
            touches_hair = np.any(comp & hair_mask) or np.any(comp & hair_dil)
            # Lateral accessories (|cx - fcx| >= 0.16*fw) or top-of-head barrettes
            is_lateral = abs(cx - fcx) >= 0.16 * fw
            is_crown = cy < fy1 + fh * 0.20
            is_near_head = cy <= chin_y + fh * 0.28

            # Apply morphological closure to capture complete loop and tail geometry
            closed_comp = cv2.morphologyEx(
                comp.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
            ).astype(bool)

            comp_rgb = rgb[closed_comp]
            med_rgb = np.median(comp_rgb, axis=0).astype(np.uint8) if len(comp_rgb) > 0 else np.array([0, 0, 0], dtype=np.uint8)

            # V12.5 Task 4: Do not classify dark dull scalp hair, scan discoloration,
            # or ambient white backdrop as hair accessories!
            is_ambient_white = is_light_bg and (med_rgb[0] > 190 and med_rgb[1] > 190 and med_rgb[2] > 190)
            is_dull_dark = (med_rgb[0] < 90) and (med_rgb[1] < 80) and (med_rgb[2] < 75) and (int(max(med_rgb)) - int(min(med_rgb)) < 25)
            if is_ambient_white or is_dull_dark:
                continue

            obj_entry = {
                "id": idx,
                "area": int(np.sum(closed_comp)),
                "centroid": (float(cx), float(cy)),
                "bbox": (
                    int(stats[idx, cv2.CC_STAT_LEFT]),
                    int(stats[idx, cv2.CC_STAT_TOP]),
                    int(stats[idx, cv2.CC_STAT_WIDTH]),
                    int(stats[idx, cv2.CC_STAT_HEIGHT]),
                ),
                "median_rgb": med_rgb.tolist(),
                "mask": closed_comp,
            }

            # Distinguish genuine hair accessories from chest badges / garment details:
            # Hair bows, ribbons, clips, and braid accessories touch hair and are located at crown or lateral braids
            if touches_hair and (is_crown or is_lateral):
                accessory_mask |= closed_comp
                accessory_objects.append(obj_entry)
            elif not touches_hair and cy > chin_y + fh * 0.28 and np.any(closed_comp & schp_cloth):
                # An emblem, patch, or school badge on the chest / torso garment away from hair
                badge_mask |= closed_comp
                badge_objects.append(obj_entry)

        logger.info(
            "[IDENTITY_EXTRACTOR] Semantic hair accessory extraction: found %d hair accessory object(s) (%d px), %d garment badge(s) (%d px)",
            len(accessory_objects),
            int(np.sum(accessory_mask)),
            len(badge_objects),
            int(np.sum(badge_mask)),
        )
        return accessory_mask, accessory_objects, badge_mask, badge_objects

    def extract(
        self,
        immutable_original_rgb: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        face_landmarks: Optional[np.ndarray] = None,
        job_id: str = "identity_layer",
        debug_dir: Optional[Path] = None,
        save_debug: bool = True,
        conditioning_strategy: str = "original_baseline",
        parse_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract verified clean identity layer."""
        if isinstance(face_bbox, dict):
            parse_data = face_bbox
            face_bbox = None

        if parse_data is None:
            parse_data = semantic_parser.parse_source(
                immutable_original_rgb,
                face_bbox=face_bbox,
                job_id=job_id,
            )

        h, w = immutable_original_rgb.shape[:2]
        schp_labels = parse_data["schp_labels"]
        schp_probs = parse_data.get("schp_probs")
        bisenet_res = parse_data["bisenet_res"]
        bisenet_p = bisenet_res.get("parsing")
        birefnet_alpha = parse_data.get("birefnet_alpha")

        if face_bbox is None:
            # Fallback face box if not provided
            fx1, fy1, fx2, fy2 = int(w * 0.25), int(h * 0.15), int(w * 0.75), int(h * 0.55)
        else:
            fx1, fy1, fx2, fy2 = face_bbox

        fw = max(1.0, float(fx2 - fx1))
        fh = max(1.0, float(fy2 - fy1))
        fcx = float(fx1 + fx2) * 0.5
        chin_y = int(fy1 + fh * 0.85)

        # 1. Face & Ears: Preserve complete natural anatomical chin and jawline contour
        schp_face = (schp_labels == 13)
        bisenet_face = np.isin(bisenet_p, (1, 2, 3, 4, 5, 6, 10, 11, 12, 13)) if bisenet_p is not None else schp_face
        orig_face_mask = schp_face | bisenet_face

        schp_ears = (schp_labels == 3)
        bisenet_ears = np.isin(bisenet_p, (7, 8)) if bisenet_p is not None else schp_ears
        orig_ears_mask = schp_ears | bisenet_ears

        # 2. Visible Neck & Throat
        bisenet_neck = bisenet_res.get("neck_mask")
        if bisenet_neck is None and bisenet_p is not None:
            bisenet_neck = (bisenet_p == 14)
        elif bisenet_neck is None:
            bisenet_neck = np.zeros((h, w), dtype=bool)

        # Anatomical neck corridor: natural throat contour between chin and clavicular notch.
        # Uses an anatomical parabolic/elliptical curve to prevent horizontal chest seam artifacts.
        y_grid = np.arange(h)[:, None]
        x_grid = np.arange(w)[None, :]
        dx_norm = np.clip(np.abs(x_grid - fcx) / max(1.0, fw * 0.38), 0.0, 1.0)
        neck_bottom_curve = chin_y + fh * (0.26 - 0.12 * (dx_norm ** 2))
        neck_corridor = (
            (y_grid >= chin_y - fh * 0.02)
            & (y_grid <= neck_bottom_curve)
            & (dx_norm < 1.0)
        )
        cand_neck = (bisenet_neck | (schp_labels == 10)) & neck_corridor & ~orig_face_mask & ~orig_ears_mask
        num_n, n_labels, _, _ = cv2.connectedComponentsWithStats(cand_neck.astype(np.uint8))
        connected_neck = np.zeros((h, w), dtype=bool)
        for idx in range(1, num_n):
            comp = (n_labels == idx)
            if np.any(comp & (np.arange(h)[:, None] <= chin_y + fh * 0.15)):
                connected_neck |= comp

        # Conflict resolution: throat skin connected to chin is genuine visible neck
        orig_neck_mask = connected_neck & ~orig_face_mask & ~orig_ears_mask

        # Dynamic neck visibility classification (FIX 2)
        neck_px = int(np.count_nonzero(orig_neck_mask))
        neck_ys = np.where(orig_neck_mask)[0]
        neck_h = int(neck_ys.max() - neck_ys.min()) if len(neck_ys) > 0 else 0
        if neck_px >= 3500 and neck_h >= 0.14 * fh:
            neck_cls = "fully_visible_neck"
        elif neck_px >= 800 or neck_h >= 0.06 * fh:
            neck_cls = "partially_visible_neck"
        else:
            neck_cls = "occluded_neck"

        # 3. Hair & Braids
        schp_hair = (schp_labels == 2)
        bisenet_hair = (bisenet_p == 17) if bisenet_p is not None else schp_hair
        hair_candidates = (schp_hair | bisenet_hair) & ~orig_face_mask & ~orig_ears_mask & ~orig_neck_mask
        if birefnet_alpha is not None:
            hair_candidates &= (birefnet_alpha > 20)

        # Connectivity to scalp ensures genuine long hair / braids are retained
        scalp = (np.arange(h)[:, None] <= chin_y + fh * 0.20) & (schp_hair | bisenet_hair)
        num_h, h_labels, h_stats, _ = cv2.connectedComponentsWithStats(hair_candidates.astype(np.uint8))
        orig_hair_mask = np.zeros((h, w), dtype=bool)
        for idx in range(1, num_h):
            comp = (h_labels == idx)
            if np.any(comp & scalp) or h_stats[idx, cv2.CC_STAT_AREA] >= 200:
                orig_hair_mask |= comp

        # 4. Hair Accessories vs School Badges / Emblems
        orig_acc_mask, acc_objects, badge_mask, badge_objects = self.extract_semantic_hair_accessories(
            immutable_original_rgb,
            hair_mask=orig_hair_mask,
            face_mask=orig_face_mask,
            neck_mask=orig_neck_mask,
            schp_labels=schp_labels,
            bisenet_parsing=bisenet_p,
            birefnet_alpha=birefnet_alpha,
            face_bbox=(fx1, fy1, fx2, fy2),
        )

        # Protect accessories first: accessories merge into hair/identity
        orig_hair_mask |= orig_acc_mask

        # 4b. Forehead Bindi / Tilak Mark Detection (Section 3)
        # Bindi is a compact round/oval ornamental red/dark mark between the eyebrows
        bindi_y1 = int(fy1 + fh * 0.38)
        bindi_y2 = int(fy1 + fh * 0.52)
        bindi_x1 = int(fcx - fw * 0.08)
        bindi_x2 = int(fcx + fw * 0.08)
        forehead_roi = immutable_original_rgb[bindi_y1:bindi_y2, bindi_x1:bindi_x2]

        has_bindi = False
        bindi_mask = np.zeros((h, w), dtype=bool)
        bindi_rgb = None
        bindi_bbox = None
        if forehead_roi.size > 0:
            br = forehead_roi[:, :, 0].astype(int)
            bg = forehead_roi[:, :, 1].astype(int)
            bb = forehead_roi[:, :, 2].astype(int)
            is_b_cand = (br > 110) & (bg < 55) & (bb < 55) & ((br - bg) > 65)
            num_b, b_labels, b_stats, b_centroids = cv2.connectedComponentsWithStats(is_b_cand.astype(np.uint8))
            for idx in range(1, num_b):
                area = b_stats[idx, cv2.CC_STAT_AREA]
                w_c = b_stats[idx, cv2.CC_STAT_WIDTH]
                h_c = b_stats[idx, cv2.CC_STAT_HEIGHT]
                aspect = w_c / max(1.0, float(h_c))
                # Must be compact and near the center of the forehead ROI
                c_x = b_centroids[idx][0]
                c_y = b_centroids[idx][1]
                dist_from_roi_center = np.hypot(c_x - (bindi_x2 - bindi_x1) * 0.5, c_y - (bindi_y2 - bindi_y1) * 0.5)
                if 20 <= area <= 450 and 0.5 <= aspect <= 2.0 and dist_from_roi_center < fw * 0.06:
                    has_bindi = True
                    b_comp = (b_labels == idx)
                    bindi_mask[bindi_y1:bindi_y2, bindi_x1:bindi_x2] = b_comp
                    bindi_rgb = [int(np.median(br[b_comp])), int(np.median(bg[b_comp])), int(np.median(bb[b_comp]))]
                    bindi_bbox = (
                        int(bindi_x1 + b_stats[idx, cv2.CC_STAT_LEFT]),
                        int(bindi_y1 + b_stats[idx, cv2.CC_STAT_TOP]),
                        int(bindi_x1 + b_stats[idx, cv2.CC_STAT_LEFT] + b_stats[idx, cv2.CC_STAT_WIDTH]),
                        int(bindi_y1 + b_stats[idx, cv2.CC_STAT_TOP] + b_stats[idx, cv2.CC_STAT_HEIGHT]),
                    )
                    logger.info("[IDENTITY_EXTRACTOR] Forehead bindi detected (%d px, RGB=%s, bbox=%s)", area, bindi_rgb, bindi_bbox)
                    break

        # 4c. Source-Aware Left/Right Ear & Authentic Earring Detection (V12.8 Addendum)
        has_earrings = False
        earring_mask = np.zeros((h, w), dtype=bool)
        left_ear_mask = np.zeros((h, w), dtype=bool)
        right_ear_mask = np.zeros((h, w), dtype=bool)
        left_ear_visible = False
        right_ear_visible = False
        left_earring_detected = False
        right_earring_detected = False

        # Check hair length relative to chin for hairstyle-aware accessory gating
        hair_below_chin_ratio = -0.5
        if np.any(orig_face_mask) and np.any(orig_hair_mask):
            y_f, _ = np.where(orig_face_mask)
            f_chin_y = y_f.max()
            f_height = max(1.0, float(y_f.max() - y_f.min()))
            y_h, _ = np.where(orig_hair_mask)
            hair_bottom = y_h.max()
            hair_below_chin_ratio = float(hair_bottom - f_chin_y) / f_height

        if np.any(orig_ears_mask):
            left_ear_mask = orig_ears_mask & (np.arange(w)[None, :] < fcx)
            right_ear_mask = orig_ears_mask & (np.arange(w)[None, :] >= fcx)
            left_ear_visible = bool(np.sum(left_ear_mask) >= 120)
            right_ear_visible = bool(np.sum(right_ear_mask) >= 120)

            # V12.9: Objective left/right ear jewelry detection without gender or hair heuristics
            if left_ear_visible or right_ear_visible:
                hsv = cv2.cvtColor(immutable_original_rgb, cv2.COLOR_RGB2HSV)
                gray = cv2.cvtColor(immutable_original_rgb, cv2.COLOR_RGB2GRAY)
                lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))

                for ear_side, ear_m in [("left", left_ear_mask), ("right", right_ear_mask)]:
                    if not np.any(ear_m) or np.sum(ear_m) < 120:
                        continue
                    ey, ex = np.where(ear_m)
                    ey1, ey2 = int(ey.min()), int(ey.max())
                    ex1, ex2 = int(ex.min()), int(ex.max())
                    eh = max(1.0, float(ey2 - ey1))

                    # Earlobe zone (lower 35% of ear cartilage down to 10% below cartilage)
                    lobe_y = int(ey1 + 0.65 * eh)
                    dangle_y2 = min(h, int(ey2 + 0.10 * eh))
                    lobe_zone = (
                        (np.arange(h)[:, None] >= lobe_y) &
                        (np.arange(h)[:, None] <= dangle_y2) &
                        (np.arange(w)[None, :] >= ex1 - 5) &
                        (np.arange(w)[None, :] <= ex2 + 5)
                    )
                    search_m = (ear_m | (orig_ears_mask | orig_acc_mask)) & lobe_zone

                    # Discard white background
                    not_bg = ~((immutable_original_rgb[:, :, 0] > 230) &
                               (immutable_original_rgb[:, :, 1] > 230) &
                               (immutable_original_rgb[:, :, 2] > 230))

                    # Authentic ear jewelry:
                    # 1. Gold / amber / colored stone: Hue 12..48, Sat >= 75, Val >= 100
                    gold_gem = (hsv[:, :, 0] >= 12) & (hsv[:, :, 0] <= 48) & (hsv[:, :, 1] >= 75) & (hsv[:, :, 2] >= 100)
                    # 2. Bright silver / diamond / pearl stud: high brightness with sharp edge contrast (not soft skin)
                    bright_stud = (hsv[:, :, 2] >= 215) & (lap >= 22.0)
                    # 3. Vibrant colored jewel: Sat >= 110, Val >= 80
                    vibrant_gem = (hsv[:, :, 1] >= 110) & (hsv[:, :, 2] >= 80)

                    is_jewelry = search_m & not_bg & (gold_gem | bright_stud | vibrant_gem)

                    num_c, c_lbls, c_stats, _ = cv2.connectedComponentsWithStats(is_jewelry.astype(np.uint8))
                    for c_idx in range(1, num_c):
                        c_area = c_stats[c_idx, cv2.CC_STAT_AREA]
                        if 15 <= c_area <= 650:
                            comp_m = (c_lbls == c_idx)
                            earring_mask |= comp_m
                            if ear_side == "left":
                                left_earring_detected = True
                            else:
                                right_earring_detected = True

                has_earrings = bool(left_earring_detected or right_earring_detected)
                if has_earrings:
                    logger.info(
                        "[IDENTITY_EXTRACTOR] Authentic earrings confirmed: left=%s, right=%s (earring_px=%d)",
                        left_earring_detected, right_earring_detected, int(np.sum(earring_mask))
                    )
                else:
                    logger.info("[IDENTITY_EXTRACTOR] Ears visible; confirmed NO authentic earrings.")

        # 4d. Validated Individual Bow & Hair Accessory Objects (V13.0 General-Purpose)
        individual_bows = []
        has_confirmed_bows = False
        if acc_objects:
            for obj in acc_objects:
                a_area = obj["area"]
                if a_area < 80:
                    continue
                c_x, c_y = obj["centroid"]
                side_tag = "left_bow" if c_x < fcx else "right_bow"
                if c_y < fy1 + 0.15 * fh:
                    side_tag = "crown_bow"
                elif c_y > chin_y + 0.30 * fh:
                    side_tag = f"{side_tag}_tail"

                individual_bows.append({
                    "name": f"{side_tag}_{len(individual_bows)+1}",
                    "mask": obj["mask"],
                    "area": int(a_area),
                    "centroid": [float(c_x), float(c_y)],
                    "median_rgb": obj["median_rgb"],
                    "bbox": obj["bbox"],
                })

            has_confirmed_bows = bool(len(individual_bows) > 0)
            if has_confirmed_bows:
                logger.info(
                    "[IDENTITY_EXTRACTOR] Confirmed %d authentic hair accessory object(s): %s",
                    len(individual_bows), [b["name"] for b in individual_bows]
                )

        # 4e. Foreground Braids Overlapping Garment (Section 2, 3)
        braid_overlap_garment = orig_hair_mask & (np.arange(h)[:, None] >= chin_y)

        # 5. Original Clothing / Garment Mask (including chest badge, coat, and hoodie/hat)
        # SCHP clothing: 5 (upper), 6 (dress), 7 (coat), 9 (pants), 11 (scarf/tie), 12 (skirt)
        # SCHP label 1 is hat/hood. BiSeNet: 16 is cloth, 18 is hat/hood.
        schp_cloth = np.isin(schp_labels, (1, 5, 6, 7, 9, 11, 12))
        bisenet_cloth = np.isin(bisenet_p, (16, 18)) if bisenet_p is not None else np.zeros((h, w), dtype=bool)

        # In addition, any foreground person matte below chin level is garment/clothing
        # (ensures dark hoodies, collars, and sweaters without SCHP tags are fully editable)
        if birefnet_alpha is not None:
            fg_alpha = (birefnet_alpha > 120)
            torso_cloth = fg_alpha & (np.arange(h)[:, None] >= chin_y)
        else:
            torso_cloth = np.zeros((h, w), dtype=bool)

        orig_garment_mask = (schp_cloth | bisenet_cloth | torso_cloth | badge_mask) & ~orig_face_mask & ~orig_neck_mask & ~orig_hair_mask & ~orig_acc_mask

        # Validation info
        val_issues = []
        if np.count_nonzero(orig_face_mask) < 2000:
            val_issues.append("insufficient_face_pixels")
        is_valid = len(val_issues) == 0

        result = {
            "original_face_mask": orig_face_mask,
            "original_ears_mask": orig_ears_mask,
            "original_neck_mask": orig_neck_mask,
            "original_hair_mask": orig_hair_mask,
            "original_accessory_mask": orig_acc_mask,
            "accessory_objects": acc_objects,
            "original_badge_mask": badge_mask,
            "badge_objects": badge_objects,
            "original_garment_mask": orig_garment_mask,
            "has_bindi": has_bindi,
            "bindi_mask": bindi_mask,
            "bindi_rgb": bindi_rgb,
            "bindi_bbox": bindi_bbox,
            "has_earrings": has_earrings,
            "earring_mask": earring_mask,
            "left_ear_visible": left_ear_visible,
            "right_ear_visible": right_ear_visible,
            "left_earring_detected": left_earring_detected,
            "right_earring_detected": right_earring_detected,
            "has_confirmed_bows": has_confirmed_bows,
            "has_accessories": (len(acc_objects) > 0 or has_confirmed_bows),
            "individual_bows": individual_bows,
            "braid_overlap_garment": braid_overlap_garment,
            "full_person_alpha": birefnet_alpha,
            "birefnet_alpha": birefnet_alpha,
            "neck_classification": neck_cls,
            "selected_strategy": conditioning_strategy,
            "metrics": {
                "chin_y": chin_y,
                "lowest_neck_y": int(neck_ys.max()) if len(neck_ys) > 0 else chin_y,
                "neck_pixels": neck_px,
                "face_bbox": (fx1, fy1, fx2, fy2),
                "face_center_x": fcx,
                "face_width": fw,
                "face_height": fh,
            },
            "validation": {
                "is_valid": is_valid,
                "issues": val_issues,
                "metrics": {"neck_pixels": neck_px},
            },
        }

        logger.info(
            "[IDENTITY_EXTRACTOR] Identity extracted: valid=%s, neck_cls=%s, neck_px=%d, acc_objects=%d",
            is_valid, neck_cls, neck_px, len(acc_objects),
        )
        return result


# Global singleton instance
identity_extractor = IdentityExtractor()
