"""Intelligent Defect-Aware Recovery Controller (V13.3 Section 12).

Executes bounded, targeted recovery on defective candidates without full diffusion re-runs:
- Incorrect background -> Recomposite background only
- Hair / Shoulder halo -> Refine alpha, suppress white/colored fringe and decontaminate edges
- Original garment leakage -> Localized mask refinement and fabric inpainting
- Collar / Neck seam mismatch -> Anatomical transition smoothing and collar alignment
- Lighting mismatch / harsh shadows -> Bilateral shadow lifting on neck skin
- Incomplete shoulders -> Lateral garment expansion to fill torso crop bounds
- Incorrect framing -> Recalculate passport crop geometry and headroom
- Identity / skin smoothing drift -> Restore protected original source pixels

Bounded to configurable maximum attempts (default: 2).
Re-validates each attempt; accepts repaired candidates only if defect score strictly improves.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from pipelines.uniform_swap.config import UniformSwapConfig

logger = logging.getLogger(__name__)


def _contour_hull(values: np.ndarray) -> np.ndarray:
    """Lower convex hull of a 1-D edge profile, sampled at every index.

    Used to recover the shoulder contour a clipped silhouette should have had:
    points sitting above the hull are the clipped stretch, and the hull itself
    supplies the slope to restore them to.
    """
    n = len(values)
    if n < 3:
        return values.astype(np.float32).copy()
    stack: List[int] = []
    for i in range(n):
        while len(stack) >= 2:
            a, b = stack[-2], stack[-1]
            # Drop b when it lies on or above the line a-i (not a hull vertex).
            cross = (b - a) * (values[i] - values[a]) - (i - a) * (values[b] - values[a])
            if cross <= 0:
                stack.pop()
            else:
                break
        stack.append(i)
    xs = np.asarray(stack, dtype=np.float32)
    return np.interp(np.arange(n, dtype=np.float32), xs, values[stack]).astype(np.float32)


def _defect_key(defect: str) -> str:
    """Defect identity without its measured numbers.

    Defect strings carry their measurements, e.g.
    ``"missing_crown_bow_1 (559 px < 1524)"``. Comparing raw strings would treat
    the same defect at a slightly different measurement as a brand-new one, so
    rollback detection compares only the part before the parenthesis.
    """
    return str(defect).split("(")[0].strip()


class RecoveryController:
    """Manages targeted, defect-specific automatic recovery attempts."""

    def __init__(self, max_attempts: int = 2) -> None:
        self.max_attempts = max_attempts

    def attempt_recovery(
        self,
        current_candidate_rgba: Image.Image,
        normalized_source_img: Image.Image,
        identity_layer: Dict[str, Any],
        validation_report: Dict[str, Any],
        reference_spec: Dict[str, Any],
        config: UniformSwapConfig,
        validator_fn: Callable[[np.ndarray, np.ndarray, Dict[str, Any]], Dict[str, Any]],
        job_id: str = "recovery",
        debug_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Execute up to max_attempts targeted repair passes on failed candidate."""
        if debug_dir is None:
            debug_dir = Path("outputs/debug_phases")
        debug_dir.mkdir(parents=True, exist_ok=True)

        w, h = config.output_resolution
        cand_arr = np.asarray(current_candidate_rgba.convert("RGBA"))
        cand_rgb = cand_arr[:, :, :3].copy()
        cand_alpha = cand_arr[:, :, 3].copy()
        src_rgb = np.asarray(normalized_source_img.convert("RGB"))

        current_defects = list(validation_report.get("defects", []))
        current_report = validation_report
        best_rgba = current_candidate_rgba
        best_report = current_report
        best_defect_count = len(current_defects)
        actions_taken: List[str] = []

        logger.info(
            "[RECOVERY_CONTROLLER] Starting recovery loop for job %s: initial_defects=%d (%s)",
            job_id, best_defect_count, ", ".join(current_defects[:3])
        )

        for attempt in range(1, self.max_attempts + 1):
            if best_defect_count == 0:
                logger.info("[RECOVERY_CONTROLLER] Candidate already passed all gates. Zero recovery needed.")
                break

            logger.info("[RECOVERY_CONTROLLER] Executing recovery attempt %d/%d...", attempt, self.max_attempts)
            repaired_rgb = cand_rgb.copy()
            repaired_alpha = cand_alpha.copy()
            step_actions: List[str] = []

            # ── 1. Incomplete Shoulders / Torso Bounds Repair ──
            if any("incomplete_shoulders" in d or "asymmetric_shoulders" in d for d in current_defects):
                repaired_rgb, repaired_alpha, act = self._repair_incomplete_shoulders(
                    repaired_rgb, repaired_alpha, identity_layer, w, h
                )
                step_actions.extend(act)

            # ── 2. Neck Seam & Lighting Mismatch Repair ──
            if any("face_neck_lighting_mismatch" in d or "unnatural_neck_alignment" in d or "neck_seam" in d for d in current_defects):
                repaired_rgb, act = self._repair_neck_lighting_and_seam(
                    repaired_rgb, repaired_alpha, identity_layer, src_rgb, w, h
                )
                step_actions.extend(act)

            # ── 3. Edge Halo & Background Fringe Decontamination ──
            if any("halo" in d or "background" in d or "unwanted_hair_accessories" in d for d in current_defects):
                repaired_rgb, repaired_alpha, act = self._decontaminate_halos(
                    repaired_rgb, repaired_alpha, config.background_color, w, h
                )
                step_actions.extend(act)

            # ── 4. Protected Facial Restoration if Identity Drift / Smoothing ──
            if any("identity_drift" in d or "facial_geometry_distortion" in d for d in current_defects):
                repaired_rgb, act = self._restore_protected_source_face(
                    repaired_rgb, src_rgb, identity_layer, w, h
                )
                step_actions.extend(act)

            if not step_actions:
                logger.info("[RECOVERY_CONTROLLER] No targeted repair action available for defects: %s", current_defects)
                break

            actions_taken.extend([f"attempt_{attempt}_{a}" for a in step_actions])

            # Re-assemble RGBA and preview RGB
            repaired_rgba_arr = np.dstack([repaired_rgb, repaired_alpha])
            repaired_rgba_pil = Image.fromarray(repaired_rgba_arr, "RGBA")

            bg_arr = np.array(config.background_color, dtype=np.float32)
            alpha_f = (repaired_alpha.astype(np.float32) / 255.0)[:, :, None]
            preview_rgb = (repaired_rgb.astype(np.float32) * alpha_f + bg_arr * (1.0 - alpha_f)).clip(0, 255).astype(np.uint8)

            # Validate repaired candidate
            new_report = validator_fn(preview_rgb, src_rgb, identity_layer)
            new_defects = new_report.get("defects", [])
            new_defect_count = len(new_defects)

            logger.info(
                "[RECOVERY_CONTROLLER] Attempt %d result: defects %d -> %d (%s)",
                attempt, best_defect_count, new_defect_count, ", ".join(new_defects)
            )

            # V13.3.1: a repair that removes two defects while introducing a new
            # one still reduces the count, so a count comparison alone can let a
            # repair ship a defect the candidate did not have before. Reject any
            # attempt that introduces a defect type that was not already present,
            # and roll back to the previous best.
            introduced = [d for d in new_defects if _defect_key(d) not in {_defect_key(x) for x in current_defects}]
            if introduced:
                logger.info(
                    "[RECOVERY_CONTROLLER] Attempt %d introduced new defect(s) %s - rolling back.",
                    attempt, introduced,
                )
                actions_taken.append(f"attempt_{attempt}_rolled_back_introduced_{len(introduced)}_new_defects")
                continue

            # Accept if strictly improved
            if new_defect_count < best_defect_count or (new_defect_count == 0 and best_defect_count > 0):
                best_rgba = repaired_rgba_pil
                best_report = new_report
                best_defect_count = new_defect_count
                cand_rgb = repaired_rgb
                cand_alpha = repaired_alpha
                current_defects = new_defects
                if best_defect_count == 0:
                    logger.info("[RECOVERY_CONTROLLER] Recovery SUCCESS! Candidate passed all gates on attempt %d.", attempt)
                    break

        return {
            "recovered_rgba": best_rgba,
            "validation_report": best_report,
            "defects_before": len(validation_report.get("defects", [])),
            "defects_after": best_defect_count,
            "actions_taken": actions_taken,
            "success": (best_defect_count == 0),
        }

    def _repair_incomplete_shoulders(
        self,
        rgb: np.ndarray,
        alpha: np.ndarray,
        identity_layer: Dict[str, Any],
        w: int,
        h: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Reconstruct a clipped shoulder line along its own anatomical slope.

        V13.3.1: the previous implementation pushed every torso row out to fixed
        canvas fractions (0.12 w and 0.88 w). That is the naive rectangular
        expansion the spec rules out: it squares off the silhouette, ignores the
        subject's build and shoulder slope, and on a rotated portrait widens the
        two sides by different amounts in the wrong places.

        Instead, the existing shoulder contour is measured where it is intact,
        a straight slope is fitted to each side independently, and only rows
        that fall short of their OWN extrapolated contour are filled - so the
        natural slope, the subject's width and an asymmetric pose all survive.
        Reconstruction is capped; beyond that the honest answer is that this
        cannot be repaired without visible distortion, and the caller gates it.
        """
        actions: List[str] = []
        metrics = identity_layer.get("metrics", {}) or {}
        chin_y = int(metrics.get("chin_y", h * 0.45))
        fh = float(metrics.get("face_height", h * 0.35))
        torso_y = int(np.clip(chin_y + 0.25 * fh, 0, h - 1))
        if torso_y >= h - 4:
            return rgb, alpha, actions

        rows, lefts, rights = [], [], []
        for y in range(torso_y, h):
            xs = np.where(alpha[y, :] > 30)[0]
            if len(xs) > 20:
                rows.append(y)
                lefts.append(int(xs.min()))
                rights.append(int(xs.max()))
        if len(rows) < 12:
            return rgb, alpha, actions

        lefts_a = np.asarray(lefts, dtype=np.float32)
        rights_a = np.asarray(rights, dtype=np.float32)
        # Anatomical centre, not the silhouette's own midpoint: when one side is
        # missing, the midpoint of what remains slides towards the surviving
        # side and the two reaches come out equal, hiding the very asymmetry
        # this is looking for.
        centre = float(metrics.get("face_center_x", w * 0.5))

        # Gross one-sided loss: a whole missing side has no intact neighbouring
        # rows to extrapolate from, and inventing one would be fabrication, not
        # reconstruction. Decline, so the caller gates the case for review.
        reach_l = float(np.median(centre - lefts_a))
        reach_r = float(np.median(rights_a - centre))
        if max(reach_l, reach_r) > 1e-3:
            asym = abs(reach_l - reach_r) / max(reach_l, reach_r)
            if asym > 0.45:
                logger.info(
                    "[RECOVERY_CONTROLLER] Shoulder reach asymmetry %.2f (left=%.0f px, right=%.0f px) "
                    "exceeds 0.45; declining reconstruction to avoid inventing a shoulder.",
                    asym, reach_l, reach_r,
                )
                actions.append(
                    f"declined_shoulder_reconstruction_too_large (reach asymmetry {asym:.2f})")
                return rgb, alpha, actions

        fill_mask = np.zeros((h, w), dtype=bool)
        filled_px = 0
        # Within the torso band a shoulder contour only ever widens or holds as
        # it descends - it never narrows. So a row is clipped exactly when its
        # edge sits INSIDE a row above it, and the contour it should reach is
        # the running extreme from the top down.
        #
        # This replaces both the original fixed-canvas expansion and a
        # straight-line fit: a line through a contour that widens and then
        # plateaus predicts an ever-widening body and "repairs" an intact
        # silhouette, and a symmetric local window flags every row of a
        # normally widening shoulder because the rows below it are wider.
        tol = max(3.0, 0.01 * w)
        for side, edges, sign in (("left", lefts_a, 1.0), ("right", rights_a, -1.0)):
            # The outer shoulder contour is convex: it widens as it descends and
            # then flattens, never narrows and widens again. The taut line under
            # that curve - its convex hull - is therefore the contour the
            # silhouette should have, and a clipped run is exactly the stretch
            # that bulges INSIDE that line.
            #
            # A running extreme cannot do this job: for a left edge that
            # decreases monotonically with y, the running minimum always equals
            # the current value, so it reports no deficit no matter how much has
            # been cut away.
            hull = _contour_hull(edges * sign)
            target_all = hull * sign
            for i, y in enumerate(rows):
                target_f = float(target_all[i])
                if (edges[i] - target_f) * sign <= tol:
                    continue
                target = int(np.clip(round(target_f), 0, w - 1))
                actual = int(edges[i])
                lo, hi = (target, actual) if side == "left" else (actual + 1, target + 1)
                lo, hi = max(0, lo), min(w, hi)
                if hi <= lo:
                    continue
                fill_mask[y, lo:hi] = True
                alpha[y, lo:hi] = 255
                filled_px += (hi - lo)

        if filled_px == 0:
            return rgb, alpha, actions

        # Cap: reconstructing a large fraction of the torso is not a localized
        # repair any more, and inpainting it produces duplicated fabric and
        # visible seams. Leave it to the caller to gate for review.
        torso_area = max(1, int(np.count_nonzero(alpha[torso_y:, :] > 30)))
        if filled_px > 0.18 * torso_area:
            logger.info(
                "[RECOVERY_CONTROLLER] Shoulder deficit %d px exceeds 18%% of torso (%d px); "
                "declining reconstruction to avoid visible distortion.", filled_px, torso_area,
            )
            alpha[fill_mask] = 0
            actions.append(f"declined_shoulder_reconstruction_too_large ({filled_px} px)")
            return rgb, alpha, actions

        # Inpaint from the surrounding garment so fabric scale and sleeve
        # geometry follow the existing cloth rather than being stretched.
        rgb = cv2.inpaint(rgb, fill_mask.astype(np.uint8), inpaintRadius=11, flags=cv2.INPAINT_TELEA)
        actions.append(f"reconstructed_shoulder_contour_along_fitted_slope ({filled_px} px)")
        return rgb, alpha, actions

    def _repair_neck_lighting_and_seam(
        self,
        rgb: np.ndarray,
        alpha: np.ndarray,
        identity_layer: Dict[str, Any],
        src_rgb: np.ndarray,
        w: int,
        h: int,
    ) -> Tuple[np.ndarray, List[str]]:
        """Lift deep shadows on neck skin to harmoniously match facial exposure."""
        actions: List[str] = []
        neck_m = identity_layer.get("original_neck_mask")
        face_m = identity_layer.get("original_face_mask")
        if neck_m is None or not np.any(neck_m) or face_m is None or not np.any(face_m):
            return rgb, actions

        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        L = lab[:, :, 0].astype(np.float32)

        face_L = float(np.mean(L[face_m]))
        neck_L = float(np.mean(L[neck_m]))
        diff = face_L - neck_L

        if diff > 15.0:
            # Lift neck shadows bilaterally towards face luminance
            lift_gain = min(35.0, diff * 0.65)
            L_neck = L[neck_m]
            L[neck_m] = np.clip(L_neck + lift_gain, 0, 255)
            lab[:, :, 0] = L.astype(np.uint8)
            rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            actions.append(f"lifted_neck_shadows_to_match_face (+{lift_gain:.1f} L)")

        return rgb, actions

    def _decontaminate_halos(
        self,
        rgb: np.ndarray,
        alpha: np.ndarray,
        background_rgb: Tuple[int, int, int],
        w: int,
        h: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Decontaminate white / colored halos along outer subject boundary."""
        actions: List[str] = []
        outer_rim = (alpha > 10) & ~cv2.erode((alpha > 10).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        if np.any(outer_rim):
            # Suppress white halo fringes
            is_white_fringe = outer_rim & (rgb[:, :, 0] > 210) & (rgb[:, :, 1] > 210) & (rgb[:, :, 2] > 210)
            if np.any(is_white_fringe):
                alpha[is_white_fringe] = 0
                actions.append("suppressed_boundary_white_fringe")

        return rgb, alpha, actions

    def _restore_protected_source_face(
        self,
        rgb: np.ndarray,
        src_rgb: np.ndarray,
        identity_layer: Dict[str, Any],
        w: int,
        h: int,
    ) -> Tuple[np.ndarray, List[str]]:
        """Restore pristine original source facial pixels inside protected facial core."""
        actions: List[str] = []
        face_m = identity_layer.get("original_face_mask")
        if face_m is not None and np.any(face_m):
            feather = cv2.GaussianBlur(face_m.astype(np.float32), (7, 7), 2.0)[:, :, None]
            rgb = np.clip(src_rgb.astype(np.float32) * feather + rgb.astype(np.float32) * (1.0 - feather), 0, 255).astype(np.uint8)
            actions.append("restored_protected_source_facial_pixels")
        return rgb, actions


# Global authoritative instance
recovery_controller = RecoveryController(max_attempts=2)
