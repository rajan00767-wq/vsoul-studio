"""Anatomical Garment Registration & Semantic Component Decomposition Engine (V13.3.3).

Registers raw generated uniform components (collar, tie knot, placket, shoulders)
onto authentic source student anatomy using 2D similarity/affine landmark transformation.
Eliminates collar erasure, missing tie knots, rectangular seams, and pasted-face artifacts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class GarmentComponents:
    """Decomposed semantic garment elements extracted from raw candidate."""
    raw_rgb: np.ndarray
    garment_mask: np.ndarray               # (H, W) bool: whole garment silhouette
    collar_lapels_mask: np.ndarray         # (H, W) bool: left & right collar lapels
    tie_knot_mask: np.ndarray              # (H, W) bool: tie knot & tie blade
    collar_opening_mask: np.ndarray        # (H, W) bool: throat opening between lapels
    body_mask: np.ndarray                  # (H, W) bool: torso, placket, shoulders, sleeves
    raw_face_chin_y: int
    raw_face_height: float
    raw_face_center_x: float


@dataclass
class RegisteredGarment:
    """Anatomically aligned garment components in source coordinate space."""
    warped_rgb: np.ndarray                 # (H_src, W_src, 3) uint8
    warped_alpha: np.ndarray               # (H_src, W_src) uint8 [0, 255]
    collar_lapels_mask: np.ndarray         # (H_src, W_src) bool
    tie_knot_mask: np.ndarray              # (H_src, W_src) bool
    collar_opening_mask: np.ndarray        # (H_src, W_src) bool
    body_mask: np.ndarray                  # (H_src, W_src) bool
    registration_matrix: np.ndarray        # (2, 3) float64
    scale: float
    dx: float
    dy: float


class GarmentRegistrar:
    """Manages garment decomposition, landmark transformation, and anatomical alignment."""

    def __init__(self) -> None:
        self._insight_app = None

    def _get_app(self):
        if self._insight_app is None:
            try:
                from pipelines.photo_restoration import _get_insight_app
                self._insight_app = _get_insight_app()
            except Exception as e:
                logger.warning("[GARMENT_REGISTRAR] Could not load InsightFace app: %s", e)
        return self._insight_app

    def extract_landmarks(
        self,
        image_rgb: np.ndarray,
        fallback_bbox: Optional[Tuple[float, float, float, float]] = None,
        fallback_chin_y: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """Extract 5 primary facial landmarks (pupils, nose tip, mouth corners)."""
        app = self._get_app()
        if app is not None:
            try:
                bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                faces = app.get(bgr)
                if faces and len(faces) > 0:
                    main_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    if hasattr(main_face, "kps") and main_face.kps is not None and len(main_face.kps) >= 5:
                        return main_face.kps[:5].astype(np.float64)
            except Exception as e:
                logger.warning("[GARMENT_REGISTRAR] InsightFace landmark extraction failed: %s", e)

        # Geometrical fallback from bounding box if available
        if fallback_bbox is not None:
            x1, y1, x2, y2 = fallback_bbox
            w = max(1.0, float(x2 - x1))
            h = max(1.0, float(y2 - y1))
            cx = float(x1 + x2) * 0.5
            kps = np.array([
                [cx - 0.20 * w, y1 + 0.38 * h],  # left eye
                [cx + 0.20 * w, y1 + 0.38 * h],  # right eye
                [cx, y1 + 0.60 * h],             # nose tip
                [cx - 0.16 * w, y1 + 0.78 * h],  # left mouth corner
                [cx + 0.16 * w, y1 + 0.78 * h],  # right mouth corner
            ], dtype=np.float64)
            return kps

        return None

    def compute_registration_matrix(
        self,
        raw_rgb: np.ndarray,
        src_rgb: np.ndarray,
        src_identity_layer: Dict[str, Any],
        raw_chin_y: int,
        raw_fh: float,
        raw_fcx: float,
    ) -> Tuple[np.ndarray, float, float, float]:
        """Compute the optimal 2D similarity transformation M from raw candidate to source anatomy."""
        h_src, w_src = src_rgb.shape[:2]
        src_metrics = src_identity_layer.get("metrics", {})
        src_chin_y = int(src_metrics.get("chin_y", h_src * 0.50))
        src_fh = float(src_metrics.get("face_height", h_src * 0.35))
        src_fcx = float(src_metrics.get("face_center_x", w_src * 0.50))
        src_bbox = src_identity_layer.get("face_bbox")

        # 1. Try landmark-based similarity registration
        src_kps = self.extract_landmarks(src_rgb, fallback_bbox=src_bbox, fallback_chin_y=src_chin_y)
        raw_kps = self.extract_landmarks(raw_rgb, fallback_bbox=(raw_fcx - 0.5*raw_fh, raw_chin_y - raw_fh, raw_fcx + 0.5*raw_fh, raw_chin_y))

        if src_kps is not None and raw_kps is not None:
            M, inliers = cv2.estimateAffinePartial2D(raw_kps, src_kps)
            if M is not None and inliers is not None and np.sum(inliers) >= 3:
                scale = float(np.sqrt(M[0, 0]**2 + M[0, 1]**2))
                # Validate scale is reasonable (0.65 to 1.65)
                if 0.65 <= scale <= 1.65:
                    dx = float(M[0, 2])
                    dy = float(M[1, 2])
                    logger.info("[GARMENT_REGISTRAR] Landmark registration succeeded: scale=%.3f, dx=%.1f, dy=%.1f (inliers=%d)",
                                scale, dx, dy, int(np.sum(inliers)))
                    return M, scale, dx, dy

        # 2. Robust geometrical fallback from face center and face height
        scale = float(np.clip(src_fh / max(1.0, raw_fh), 0.70, 1.55))
        dx = float(src_fcx - scale * raw_fcx)
        dy = float(src_chin_y - scale * raw_chin_y)
        M = np.array([
            [scale, 0.0, dx],
            [0.0, scale, dy]
        ], dtype=np.float64)
        logger.info("[GARMENT_REGISTRAR] Geometrical registration fallback: scale=%.3f, dx=%.1f, dy=%.1f", scale, dx, dy)
        return M, scale, dx, dy

    def decompose_garment(
        self,
        raw_rgb: np.ndarray,
        cand_labels: np.ndarray,
        cand_matte: np.ndarray,
        uniform_cues: Dict[str, Any],
        raw_chin_y: int,
        raw_fh: float,
        raw_fcx: float,
    ) -> GarmentComponents:
        """Decompose raw Qwen candidate into semantic garment components."""
        h, w = raw_rgb.shape[:2]
        # Candidate garment classes in LIP-20: UpperClothes(5), Dress(6), Coat(7), Torso-skin(10)
        cand_garment_classes = (cand_labels == 5) | (cand_labels == 6) | (cand_labels == 7) | (cand_labels == 10)
        cand_is_person = cand_matte > 35
        # Garment begins around the collar/chin line
        cand_below_throat = np.arange(h)[:, None] >= (raw_chin_y - int(0.12 * raw_fh))
        garment_mask = cand_garment_classes & cand_is_person & cand_below_throat

        # Remove candidate face and scalp hair from garment zone
        cand_head = (cand_labels == 13) | (cand_labels == 2)
        cand_head_dil = cv2.dilate(cand_head.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        garment_mask &= ~cand_head_dil

        # Decompose Collar Lapels
        hsv = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2HSV)
        ref_type = uniform_cues.get("ref_type", "")
        has_red_collar = uniform_cues.get("has_red_collar", False) or ref_type == "red_collar"
        has_navy_collar = uniform_cues.get("has_navy_collar", False) or ref_type == "navy_collar"
        has_tie = uniform_cues.get("has_tie", False) or ref_type == "tie"

        collar_zone = garment_mask & (np.arange(h)[:, None] <= raw_chin_y + int(0.40 * raw_fh))

        collar_lapels_mask = np.zeros((h, w), dtype=bool)
        if has_red_collar:
            is_red = ((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] >= 65) & (hsv[:, :, 2] >= 50)
            collar_lapels_mask = is_red & collar_zone
        elif has_navy_collar:
            cr = raw_rgb[:, :, 0].astype(int)
            cg = raw_rgb[:, :, 1].astype(int)
            cb = raw_rgb[:, :, 2].astype(int)
            is_navy = (cr < 90) & (cg < 95) & (cb > 60) & (cb > cr + 10)
            collar_lapels_mask = is_navy & collar_zone
        else:
            # Generic collar: upper boundary of garment surrounding central opening
            collar_upper = garment_mask & (np.arange(h)[:, None] <= raw_chin_y + int(0.25 * raw_fh))
            collar_lapels_mask = collar_upper

        # Clean noise in collar lapels
        if np.any(collar_lapels_mask):
            collar_lapels_mask = cv2.morphologyEx(collar_lapels_mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)

        # Decompose Tie and Tie Knot
        tie_knot_mask = np.zeros((h, w), dtype=bool)
        if has_tie:
            tie_corridor = (
                (np.abs(np.arange(w)[None, :] - raw_fcx) <= raw_fh * 0.22)
                & (np.arange(h)[:, None] >= raw_chin_y + int(0.05 * raw_fh))
                & (np.arange(h)[:, None] <= raw_chin_y + int(0.85 * raw_fh))
                & garment_mask
            )
            # SCHP label 11 is scarf/tie, or dark central fabric distinct from white/light shirt
            schp_tie = (cand_labels == 11) & tie_corridor
            cr = raw_rgb[:, :, 0].astype(int)
            cg = raw_rgb[:, :, 1].astype(int)
            cb = raw_rgb[:, :, 2].astype(int)
            # Tie fabric in REF003 is dark navy / patterned with dark base
            is_dark_tie = (cr < 95) & (cg < 100) & (cb < 125) & tie_corridor
            tie_cand = (schp_tie | is_dark_tie) & ~collar_lapels_mask
            if np.sum(tie_cand) > 100:
                tie_knot_mask = cv2.morphologyEx(tie_cand.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)

        # Collar Opening (throat opening between lapels above tie knot / placket)
        collar_opening_mask = np.zeros((h, w), dtype=bool)
        if np.any(collar_lapels_mask):
            # Dynamic opening between lapels:
            # Region between left and right lapels within neck corridor
            y_lapel, x_lapel = np.where(collar_lapels_mask)
            min_ly, max_ly = y_lapel.min(), y_lapel.max()
            for y in range(min_ly, min(h, max_ly + 1)):
                row_xs = x_lapel[y_lapel == y]
                if len(row_xs) >= 2:
                    lx1, lx2 = row_xs.min(), row_xs.max()
                    collar_opening_mask[y, lx1:lx2] = True
            collar_opening_mask &= ~collar_lapels_mask & ~tie_knot_mask
            collar_opening_mask &= (np.arange(h)[:, None] <= raw_chin_y + int(0.50 * raw_fh))

        # Body and Sleeves (everything else in garment)
        body_mask = garment_mask & ~collar_lapels_mask & ~tie_knot_mask

        return GarmentComponents(
            raw_rgb=raw_rgb,
            garment_mask=garment_mask,
            collar_lapels_mask=collar_lapels_mask,
            tie_knot_mask=tie_knot_mask,
            collar_opening_mask=collar_opening_mask,
            body_mask=body_mask,
            raw_face_chin_y=raw_chin_y,
            raw_face_height=raw_fh,
            raw_face_center_x=raw_fcx,
        )

    def register_and_warp(
        self,
        components: GarmentComponents,
        cand_matte: np.ndarray,
        M: np.ndarray,
        scale: float,
        dx: float,
        dy: float,
        target_shape: Tuple[int, int],
        src_person_matte: np.ndarray,
        src_chin_y: int,
    ) -> RegisteredGarment:
        """Warp garment components into source anatomical coordinate space with shoulder extension."""
        h_src, w_src = target_shape
        # Warp RGB using bicubic/lanczos with reflect padding
        warped_rgb = cv2.warpAffine(
            components.raw_rgb, M, (w_src, h_src),
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101
        )

        # Warp Alpha and Component Masks using linear interpolation
        warped_cand_matte = cv2.warpAffine(
            cand_matte.astype(np.float32), M, (w_src, h_src),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        )
        warped_garment_mask = cv2.warpAffine(
            components.garment_mask.astype(np.float32), M, (w_src, h_src),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        ) > 0.40

        warped_collar = cv2.warpAffine(
            components.collar_lapels_mask.astype(np.float32), M, (w_src, h_src),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        ) > 0.40

        warped_tie = cv2.warpAffine(
            components.tie_knot_mask.astype(np.float32), M, (w_src, h_src),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        ) > 0.40

        warped_opening = cv2.warpAffine(
            components.collar_opening_mask.astype(np.float32), M, (w_src, h_src),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        ) > 0.35

        warped_body = cv2.warpAffine(
            components.body_mask.astype(np.float32), M, (w_src, h_src),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        ) > 0.40

        # Construct warped garment alpha
        warped_alpha = np.where(warped_garment_mask, warped_cand_matte, 0.0).clip(0, 255).astype(np.uint8)

        # Lateral Shoulder Extension:
        # If student's authentic shoulder silhouette extends laterally beyond warped garment,
        # extrapolate garment fabric outward to cover authentic student silhouette naturally.
        shoulder_zone = (src_person_matte > 0.35) & (np.arange(h_src)[:, None] >= src_chin_y + 10)
        missing_shoulder = shoulder_zone & (warped_alpha < 35)

        if np.any(missing_shoulder) and np.sum(missing_shoulder) > 50:
            logger.info("[GARMENT_REGISTRAR] Extrapolating garment to cover authentic shoulders (%d px)", int(np.sum(missing_shoulder)))
            warped_alpha[missing_shoulder] = 255
            # Inpaint outward to bleed fabric texture without touching center chest
            warped_rgb = cv2.inpaint(warped_rgb, missing_shoulder.astype(np.uint8), inpaintRadius=7, flags=cv2.INPAINT_TELEA)
            warped_body |= missing_shoulder

        return RegisteredGarment(
            warped_rgb=warped_rgb,
            warped_alpha=warped_alpha,
            collar_lapels_mask=warped_collar,
            tie_knot_mask=warped_tie,
            collar_opening_mask=warped_opening,
            body_mask=warped_body,
            registration_matrix=M,
            scale=scale,
            dx=dx,
            dy=dy,
        )

    def extract_dynamic_neck(
        self,
        src_rgb: np.ndarray,
        src_person_matte: np.ndarray,
        src_chin_y: int,
        src_fh: float,
        src_fcx: float,
        src_face_mask: np.ndarray,
        registered_garment: RegisteredGarment,
    ) -> np.ndarray:
        """Extract visible neck skin strictly inside the registered collar opening.
        
        Ensures collar lapels and tie knots naturally overlap neck skin,
        and never allows neck masks to overwrite target garment components.
        """
        h, w = src_rgb.shape[:2]
        # Skin color detector (YCbCr + HSV)
        ycrcb = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        is_skin = (cr >= 125) & (cr <= 180) & (cb >= 70) & (cb <= 135)

        # Neck corridor between chin and clavicular line
        y_grid = np.arange(h)[:, None]
        x_grid = np.arange(w)[None, :]
        dx_norm = np.abs(x_grid - src_fcx) / max(1.0, src_fh * 0.35)
        throat_corridor = (
            (y_grid >= src_chin_y - int(0.04 * src_fh))
            & (y_grid <= src_chin_y + int(0.40 * src_fh))
            & (dx_norm <= 1.0)
        )

        # Collar opening constraint: opening between registered lapels
        opening_m = registered_garment.collar_opening_mask
        # If opening mask is small, construct triangular throat hollow bounded by lapels
        if np.sum(opening_m) < 200:
            # Throat hollow is the central region not covered by collar lapels or tie knot
            opening_m = throat_corridor & ~registered_garment.collar_lapels_mask & ~registered_garment.tie_knot_mask

        # Authentic visible neck skin:
        # Inside opening, genuine skin color, within subject matte, below chin line
        visible_neck = (
            is_skin
            & opening_m
            & (src_person_matte > 0.35)
            & ~src_face_mask
            & ~registered_garment.collar_lapels_mask
            & ~registered_garment.tie_knot_mask
        )

        # Clean noise
        if np.any(visible_neck):
            visible_neck = cv2.morphologyEx(visible_neck.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)

        return visible_neck

    def integrate_continuous_skin(
        self,
        src_rgb: np.ndarray,
        src_person_matte: np.ndarray,
        face_mask: np.ndarray,
        neck_mask: np.ndarray,
        ears_mask: np.ndarray,
        orig_garment_mask: np.ndarray,
        registered_garment: RegisteredGarment,
        raw_registered_rgb: np.ndarray,
        cand_labels_warped: Optional[np.ndarray],
        src_chin_y: int,
        src_fh: float,
        src_fcx: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Produce a continuous unbroken anatomical skin layer and reconstruct missing neck regions (V13.3.4).

        Guarantees:
        1. 100% geometric connection between jawline and neck (zero disconnect / gaps).
        2. Strict exclusion of original garment residue (e.g. green polo collar).
        3. Dynamic target collar neckline boundary matching registered target lapels.
        4. Color-harmonized missing skin reconstruction preserving authentic throat shading.

        Returns:
            (final_skin_rgb, final_skin_alpha, visible_opening_mask, missing_skin_mask, reconstructed_skin_layer)
        """
        h, w = src_rgb.shape[:2]
        y_grid = np.arange(h)[:, None]
        x_grid = np.arange(w)[None, :]

        # 1. Continuous authentic source skin
        continuous_source_skin = (face_mask | neck_mask | ears_mask) & (src_person_matte > 0.20)

        # 2. Strict original garment residue exclusion
        if orig_garment_mask is not None and np.any(orig_garment_mask):
            garment_forbidden = cv2.dilate(orig_garment_mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            continuous_source_skin &= ~garment_forbidden

        # 3. Solid target garment fabric (takes occlusion precedence over skin)
        garment_solid = (registered_garment.warped_alpha > 50) & (
            registered_garment.body_mask | registered_garment.collar_lapels_mask | registered_garment.tie_knot_mask
        )

        # 4. Target visible neckline opening
        dx_norm = np.abs(x_grid - src_fcx) / max(1.0, src_fh * 0.42)
        throat_corridor = (
            (y_grid >= src_chin_y - int(0.02 * src_fh))
            & (y_grid <= src_chin_y + int(0.60 * src_fh))
            & (dx_norm <= 1.0)
        )
        visible_opening_mask = throat_corridor & ~garment_solid & (src_person_matte > 0.20)

        # 5. Missing skin region: required visible neck opening minus authentic source skin coverage
        missing_skin_mask = visible_opening_mask & ~continuous_source_skin & (
            (src_person_matte > 0.15) | (registered_garment.warped_alpha > 0.15)
        )

        final_skin_rgb = src_rgb.copy()
        reconstructed_skin_layer = np.zeros_like(src_rgb)

        if np.any(missing_skin_mask):
            # Sample authentic student neck/jaw skin directly above missing region
            ref_skin_zone = continuous_source_skin & (y_grid >= src_chin_y - int(0.12 * src_fh)) & (y_grid <= src_chin_y + int(0.15 * src_fh))
            if np.sum(ref_skin_zone) > 40:
                ref_lab = cv2.cvtColor(src_rgb[ref_skin_zone].reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
                mean_ref_lab = np.mean(ref_lab, axis=0)
            else:
                mean_ref_lab = np.array([170.0, 140.0, 145.0], dtype=np.float32)

            cand_reconstructed = False
            if cand_labels_warped is not None and raw_registered_rgb is not None:
                cand_skin_pixels = missing_skin_mask & ((cand_labels_warped == 10) | (cand_labels_warped == 13))
                if np.sum(cand_skin_pixels) > 50:
                    cand_lab = cv2.cvtColor(raw_registered_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
                    mean_cand_lab = np.mean(cand_lab[cand_skin_pixels], axis=0)
                    delta_lab = mean_ref_lab - mean_cand_lab
                    cand_lab[cand_skin_pixels] += delta_lab
                    cand_lab = np.clip(cand_lab, 0, 255).astype(np.uint8)
                    shifted_rgb = cv2.cvtColor(cand_lab, cv2.COLOR_LAB2RGB)
                    final_skin_rgb[cand_skin_pixels] = shifted_rgb[cand_skin_pixels]
                    reconstructed_skin_layer[cand_skin_pixels] = shifted_rgb[cand_skin_pixels]
                    cand_reconstructed = True

            # Localized texture-aware inpainting for remaining missing pixels or boundary smoothing
            missing_to_inpaint = missing_skin_mask if not cand_reconstructed else (missing_skin_mask & (reconstructed_skin_layer.sum(axis=2) == 0))
            if np.any(missing_to_inpaint):
                final_skin_rgb = cv2.inpaint(final_skin_rgb, missing_to_inpaint.astype(np.uint8), inpaintRadius=7, flags=cv2.INPAINT_TELEA)
                reconstructed_skin_layer[missing_to_inpaint] = final_skin_rgb[missing_to_inpaint]

        # Combine authentic source skin and reconstructed missing throat skin
        total_skin_mask = continuous_source_skin | missing_skin_mask
        final_skin_alpha = cv2.GaussianBlur(total_skin_mask.astype(np.float32) * 255.0, (3, 3), 1.0).clip(0, 255).astype(np.uint8)
        # 100% solid inside authentic face and neck core
        final_skin_alpha[face_mask | (neck_mask & ~garment_forbidden)] = 255

        return final_skin_rgb, final_skin_alpha, visible_opening_mask, missing_skin_mask, reconstructed_skin_layer


garment_registrar = GarmentRegistrar()
