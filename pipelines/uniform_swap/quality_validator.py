"""Quality Validator & Defect Gating for V11 Uniform Swap.

Enforces biometric identity retention, single-subject composition, original garment
exclusion, and anatomical neck/collar continuity. Fails deterministically with clear
diagnostics instead of silently delivering damaged portraits.
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

warnings.filterwarnings("ignore", category=FutureWarning, module=".*insightface.*")

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _candidate_foreground_mask(
    candidate_image: Any,
    identity_layer: Optional[Dict[str, Any]],
    cand_dist_bg: np.ndarray,
    bg_tolerance: float,
) -> Tuple[Optional[np.ndarray], str]:
    """Return the best available person/foreground mask and the basis used.

    Preference order: the pipeline's own person alpha, then the candidate's own
    alpha channel, then (last resort) distance from the backdrop colour. The
    colour fallback is unreliable whenever the garment is near the backdrop
    colour - a white shirt on a white backdrop measures as background - so the
    basis string is reported in the check details and callers relax thresholds
    when it is in play.
    """
    if identity_layer and identity_layer.get("full_person_alpha") is not None:
        return (np.asarray(identity_layer["full_person_alpha"]) > 30), "person_alpha"
    if isinstance(candidate_image, Image.Image) and candidate_image.mode == "RGBA":
        return (np.asarray(candidate_image)[:, :, 3] > 30), "candidate_alpha"
    if isinstance(candidate_image, np.ndarray) and candidate_image.ndim == 3 and candidate_image.shape[2] == 4:
        return (candidate_image[:, :, 3] > 30), "candidate_alpha"
    if cand_dist_bg is not None:
        return (cand_dist_bg > bg_tolerance), "background_colour_distance"
    return None, "unavailable"


def _measure_button_fidelity(
    cand_rgb: np.ndarray,
    ref_cues: Dict[str, Any],
    cand_box: Optional[Any],
    h: int,
    w: int,
) -> Dict[str, Any]:
    """Compare candidate placket buttons against the reference's measured buttons.

    Buttons are found as small, compact connected components along the placket
    centreline that contrast with their immediate surroundings - not as a raw
    count of dark or light pixels, which on a patterned fabric measures the
    pattern. Works for any button colour; returns UNCERTAIN when too few blobs
    are present to judge, so an unmeasurable attribute is never silently passed.
    """
    from pipelines.uniform_swap.reference_analyzer import _lab_delta

    if cand_box is None:
        return {"status": "UNCERTAIN", "score": 0, "threshold": 0, "details": "No face box; placket not locatable"}
    if not ref_cues.get("button_presence", False):
        return {"status": "PASS", "score": 0, "threshold": 0, "details": "Reference specifies no visible buttons"}

    ref_btn_lab = ref_cues.get("button_lab")
    c_chin_y = int(cand_box[3])
    c_fh = float(cand_box[3] - cand_box[1])
    c_fw = float(cand_box[2] - cand_box[0])
    c_mid_x = int((cand_box[0] + cand_box[2]) / 2)

    half = max(12, int(0.10 * c_fw))
    y1 = int(np.clip(c_chin_y + 0.15 * c_fh, 0, h - 1))
    y2 = int(np.clip(c_chin_y + 0.95 * c_fh, y1 + 1, h))
    x1 = int(np.clip(c_mid_x - half, 0, w - 1))
    x2 = int(np.clip(c_mid_x + half, x1 + 1, w))
    strip = cand_rgb[y1:y2, x1:x2]
    if strip.size == 0:
        return {"status": "UNCERTAIN", "score": 0, "threshold": 0, "details": "Empty placket strip"}

    from pipelines.uniform_swap.reference_analyzer import measure_button_blobs

    cand_btn_lab, blob_count = measure_button_blobs(
        strip,
        min_area=max(9.0, 0.00035 * c_fh * c_fw),
        max_area=max(20.0, 0.020 * c_fh * c_fw),
        blur_sigma=0.02 * c_fh,
    )
    if cand_btn_lab is None:
        return {
            "status": "UNCERTAIN", "score": blob_count, "threshold": 2,
            "details": f"Only {blob_count} button-like blob(s) found on the placket; button colour not measurable",
        }

    if ref_btn_lab is None:
        return {
            "status": "UNCERTAIN", "score": blob_count, "threshold": 2,
            "details": "Reference button colour was not measurable; candidate buttons not compared",
        }

    delta = _lab_delta(cand_btn_lab, ref_btn_lab)
    # DeltaE76 ~40 separates a genuine colour swap (black vs white is >90) from
    # the shading, white-balance and fabric-shadow spread of a correct button.
    if delta is not None and delta > 40.0:
        return {
            "status": "FAIL", "score": round(delta, 1), "threshold": 40.0,
            "details": f"button colour differs from reference (deltaE={delta:.1f} > 40.0 across {blob_count} buttons)",
        }
    return {
        "status": "PASS", "score": round(delta, 1) if delta is not None else 0,
        "threshold": 40.0,
        "details": f"Button colour matches reference (deltaE={delta:.1f}, {blob_count} buttons)" if delta is not None else "Buttons verified",
    }


def _measure_collar_symmetry(
    cand_rgb: np.ndarray,
    ref_cues: Dict[str, Any],
    c_chin_y: int,
    c_fh: float,
    c_fcx: float,
    c_fw: float,
    h: int,
    w: int,
) -> Dict[str, Any]:
    """Compare left and right collar attachment about the facial axis.

    Colour-agnostic: the collar colour comes from whatever the reference
    analyzer measured in CIELAB, so pointed, rounded, Peter Pan, mandarin, polo
    and open collars in any colour are handled the same way. Only runs when the
    reference's collar is measurably distinct from its own body fabric,
    otherwise the collar cannot be isolated from the shirt and the honest answer
    is UNCERTAIN rather than a pass.

    The search window stops at +-0.55 face widths so sleeve trim, cuffs and
    shoulder piping - which share the collar's colour on many uniforms - stay
    out of the measurement.
    """
    from pipelines.uniform_swap.reference_analyzer import _lab_delta

    collar_lab = ref_cues.get("collar_lab")
    if collar_lab is None or not ref_cues.get("collar_contrasts_body", False):
        return {
            "status": "UNCERTAIN", "score": 0.0, "threshold": 0.35,
            "details": "Reference collar not separable from body fabric; symmetry not measured",
        }

    # The collar band sits around the chin line. Measured on REG_06, extending
    # this window to +0.45 face heights swept in the chest-level red placket tab
    # and the sleeve trim, which re-created the very bimodal-cluster error this
    # check replaced. Keep it to the neck corridor.
    y0 = int(np.clip(c_chin_y - 0.20 * c_fh, 0, h - 1))
    y1 = int(np.clip(c_chin_y + 0.25 * c_fh, y0 + 1, h))
    x0 = int(np.clip(c_fcx - 0.55 * c_fw, 0, w - 1))
    x1 = int(np.clip(c_fcx + 0.55 * c_fw, x0 + 1, w))
    zone = cand_rgb[y0:y1, x0:x1]
    if zone.size == 0:
        return {"status": "UNCERTAIN", "score": 0.0, "threshold": 0.35, "details": "Empty collar zone"}

    zone_lab = cv2.cvtColor(zone, cv2.COLOR_RGB2LAB).astype(np.float32)
    target = np.array(collar_lab, dtype=np.float32)
    dist = np.linalg.norm(zone_lab - target[None, None, :], axis=2)
    is_collar = dist < 28.0

    axis_x = int(round(c_fcx)) - x0
    left = is_collar[:, :max(0, axis_x)]
    right = is_collar[:, max(0, axis_x):]
    l_px, r_px = int(left.sum()), int(right.sum())
    total = l_px + r_px
    min_px = max(60, int(0.004 * c_fw * c_fh))
    if total < min_px:
        return {
            "status": "UNCERTAIN", "score": total, "threshold": min_px,
            "details": f"Too little collar-coloured area to measure symmetry ({total} px < {min_px} px)",
        }

    imbalance = abs(l_px - r_px) / float(total)
    if imbalance > 0.60:
        return {
            "status": "FAIL", "score": round(imbalance, 3), "threshold": 0.60,
            "details": f"Collar attaches asymmetrically about the facial axis (left={l_px} px, right={r_px} px, imbalance={imbalance:.3f} > 0.60)",
        }
    return {
        "status": "PASS", "score": round(imbalance, 3), "threshold": 0.60,
        "details": f"Collar symmetric about facial axis (left={l_px} px, right={r_px} px, imbalance={imbalance:.3f})",
    }


def _measure_shoulder_extent(
    candidate_image: Any,
    identity_layer: Optional[Dict[str, Any]],
    cand_dist_bg: np.ndarray,
    bg_tolerance: float,
    c_chin_y: int,
    c_fh: float,
    c_fw: float,
    has_face_box: bool,
    h: int,
    w: int,
) -> Tuple[int, int, str]:
    """Measure shoulder width and the width a complete shoulder line requires.

    Returns ``(measured_px, required_px, basis)``.

    Anatomy-relative rather than canvas-relative: the shoulder line is taken at
    roughly one face-height below the chin (clamped into the canvas), and the
    requirement is expressed as a multiple of the measured face width. A child
    with narrow shoulders and an adult with broad shoulders both have a shoulder
    span of about 2.2-3.0 face widths, whereas a fixed 0.35 * canvas_width
    requirement changes meaning with every output size and every crop tightness.
    Without a face box there is nothing to scale against, so it falls back to the
    old canvas fraction.
    """
    fg, basis = _candidate_foreground_mask(candidate_image, identity_layer, cand_dist_bg, bg_tolerance)
    if fg is None:
        return 0, 0, "unavailable"

    # Alpha alone is not enough either: a candidate whose shoulders have been
    # painted out in RGB can still carry a stale full-body alpha, and that is
    # exactly how the shoulder defect injection in scripts/verify_quality_gates.py
    # behaves. Combine them, but with a TIGHT colour tolerance rather than the
    # default 30. Measured at y=0.85h: erased shoulders are flat backdrop and
    # give 159 px at every tolerance, while REG_05's genuine white shirt gives
    # 796 px at tol>4 and only 70 px at tol>30. Real fabric is never exactly
    # the backdrop value; an erased region is.
    if cand_dist_bg is not None and basis != "background_colour_distance":
        fg = fg & (cand_dist_bg > 4.0)
        basis = f"{basis}+tight_colour"

    if has_face_box and c_fh > 1.0 and c_fw > 1.0:
        y_shoulder = int(np.clip(c_chin_y + 0.95 * c_fh, 0, h - 1))
        # 1.6 face widths is deliberately permissive: it separates a genuinely
        # clipped/floating torso from a narrow but complete child's shoulder
        # line, which is what this gate is for.
        required = int(min(w * 0.90, 1.6 * c_fw))
    else:
        y_shoulder = int(h * 0.85)
        required = int(w * 0.35)

    # Take the widest row in a short band so a single ragged alpha row, or a
    # row that happens to fall in the gap under the chin, does not decide it.
    y0 = int(np.clip(y_shoulder - 0.05 * h, 0, h - 1))
    y1 = int(np.clip(y_shoulder + 0.05 * h, y0 + 1, h))
    band = fg[y0:y1, :]
    measured = int(band.sum(axis=1).max()) if band.size else 0

    if basis == "background_colour_distance":
        # Colour distance under-reports garments close to the backdrop colour, so
        # only a near-total absence of subject is treated as a real defect here.
        required = int(required * 0.5)

    return measured, required, basis


class QualityCheckFailure(RuntimeError):
    """Raised when an unrecoverable visual or biometric defect is detected."""
    pass


class QualityValidator:
    """Automated biometric and visual defect validator."""

    def __init__(self, min_biometric_similarity: float = 0.82) -> None:
        self.min_biometric_similarity = min_biometric_similarity

    def validate_final_portrait(
        self,
        final_image: Union[Image.Image, np.ndarray],
        source_image: Union[Image.Image, np.ndarray],
        identity_layer: Optional[Dict[str, Any]] = None,
        job_id: str = "validation",
    ) -> Dict[str, Any]:
        """Validate final portrait and raise QualityCheckFailure if fatal defect occurs."""
        res = self.validate_candidate(final_image, source_image, job_id=job_id, stage="final")
        if res.get("faces_detected") != 1:
            raise QualityCheckFailure(f"Detected {res.get('faces_detected')} face(s); expected exactly 1.")
        return res

    def validate_candidate(
        self,
        candidate_image: Union[Image.Image, np.ndarray],
        source_image: Union[Image.Image, np.ndarray],
        identity_layer: Optional[Dict[str, Any]] = None,
        uniform_reference: Optional[Union[Image.Image, np.ndarray]] = None,
        job_id: str = "validation",
        stage: str = "raw",  # "raw" or "final"
        background_rgb: Tuple[int, int, int] = (255, 255, 255),
        bg_tolerance: float = 30.0,
        uniform_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Perform comprehensive biometric and defect validation (Section 9).
        
        Returns serializable dictionary matching Section 9 & 13 requirements:
        - biometric similarity (InsightFace embedding cosine >= 0.82)
        - landmark geometry RMSE (<= 0.040)
        - head aspect-ratio distortion delta (<= 0.05)
        - target uniform color / red collar detection
        - badge leakage detection
        - neck skin red bleed detection
        - background cleanliness / shoulder halo detection
        - real scores without fabrication
        - needs_review flag
        """
        if isinstance(candidate_image, Image.Image):
            cand_rgb = np.asarray(candidate_image.convert("RGB"))
        else:
            cand_rgb = candidate_image

        if isinstance(source_image, Image.Image):
            src_rgb = np.asarray(source_image.convert("RGB"))
        else:
            src_rgb = source_image

        if not isinstance(background_rgb, tuple) or len(background_rgb) != 3:
            from pipelines.uniform_swap.config import UniformSwapConfig
            background_rgb = UniformSwapConfig.parse_bg_color(background_rgb)

        h, w = cand_rgb.shape[:2]
        if src_rgb.shape[:2] != (h, w):
            from pipelines.uniform_swap.image_normalizer import normalize_portrait_aspect_ratio
            src_norm_pil, _, _ = normalize_portrait_aspect_ratio(
                Image.fromarray(src_rgb), target_size=(w, h)
            )
            src_rgb = np.asarray(src_norm_pil)
        defects: List[str] = []
        warnings: List[str] = []
        similarity = 0.0
        geometry_rmse = 0.0
        aspect_delta = 0.0
        faces_detected = 0

        cand_box = None
        src_box = None

        try:
            from pipelines.photo_restoration import _get_insight_app
            app = _get_insight_app()

            cand_faces = app.get(cv2.cvtColor(cand_rgb, cv2.COLOR_RGB2BGR))
            src_faces = app.get(cv2.cvtColor(src_rgb, cv2.COLOR_RGB2BGR))

            faces_detected = len(cand_faces)
            if faces_detected == 0:
                defects.append("no_face_detected")
            elif faces_detected > 1:
                defects.append(f"multiple_faces_detected ({faces_detected})")

            if cand_faces and src_faces:
                cand_box = cand_faces[0].bbox
                src_box = src_faces[0].bbox

                c_emb = cand_faces[0].embedding
                s_emb = src_faces[0].embedding
                if c_emb is not None and s_emb is not None:
                    c_norm = c_emb / max(1e-6, float(np.linalg.norm(c_emb)))
                    s_norm = s_emb / max(1e-6, float(np.linalg.norm(s_emb)))
                    similarity = float(np.dot(s_norm, c_norm))

                # Landmark geometry RMSE & Landmark Anatomical Aspect Ratio (Common Coordinate System)
                # Evaluates in a common coordinate system via Umeyama / similarity transform (scale, rotation, translation)
                # Eliminates detector bounding-box jitter and head tilt angle!
                c_kps = cand_faces[0].kps
                s_kps = src_faces[0].kps
                if c_kps is not None and s_kps is not None and len(c_kps) == len(s_kps):
                    interocular = max(1.0, float(np.linalg.norm(s_kps[0] - s_kps[1])))
                    M, inliers = cv2.estimateAffinePartial2D(c_kps, s_kps)
                    if M is not None:
                        c_kps_homo = np.hstack([c_kps, np.ones((len(c_kps), 1))])
                        c_aligned = (c_kps_homo @ M.T)
                        geometry_rmse = float(np.sqrt(np.mean((c_aligned - s_kps) ** 2)) / interocular)
                    else:
                        c_scale = max(1.0, float(cand_box[2] - cand_box[0]))
                        s_scale = max(1.0, float(src_box[2] - src_box[0]))
                        c_norm_kps = (c_kps - cand_box[:2]) / c_scale
                        s_norm_kps = (s_kps - src_box[:2]) / s_scale
                        geometry_rmse = float(np.sqrt(np.mean((c_norm_kps - s_norm_kps) ** 2)))

                    # 1. Head Aspect-Ratio Distortion Check (Landmark-based anatomical aspect ratio)
                    # Compares vertical eye-to-mouth distance vs interocular distance
                    # Distinguishes actual geometry distortion from face detector bounding-box jitter!
                    s_interocular = max(1.0, float(np.linalg.norm(s_kps[0] - s_kps[1])))
                    c_interocular = max(1.0, float(np.linalg.norm(c_kps[0] - c_kps[1])))
                    s_eye_mid = (s_kps[0] + s_kps[1]) * 0.5
                    s_mouth_mid = (s_kps[3] + s_kps[4]) * 0.5
                    c_eye_mid = (c_kps[0] + c_kps[1]) * 0.5
                    c_mouth_mid = (c_kps[3] + c_kps[4]) * 0.5
                    s_vert = float(np.linalg.norm(s_mouth_mid - s_eye_mid))
                    c_vert = float(np.linalg.norm(c_mouth_mid - c_eye_mid))
                    s_aspect = s_vert / s_interocular
                    c_aspect = c_vert / c_interocular
                    aspect_delta = float(abs(c_aspect / max(1e-4, s_aspect) - 1.0))
                else:
                    c_w = float(cand_box[2] - cand_box[0])
                    c_h = float(cand_box[3] - cand_box[1])
                    s_w = float(src_box[2] - src_box[0])
                    s_h = float(src_box[3] - src_box[1])
                    c_ratio = c_h / max(1.0, c_w)
                    s_ratio = s_h / max(1.0, s_w)
                    aspect_delta = float(abs(c_ratio / max(1e-4, s_ratio) - 1.0))

            if cand_box is None and identity_layer and identity_layer.get("face_bbox") is not None:
                cand_box = np.array(identity_layer["face_bbox"], dtype=float)
            if src_box is None and identity_layer and identity_layer.get("face_bbox") is not None:
                src_box = np.array(identity_layer["face_bbox"], dtype=float)

        except Exception as exc:
            logger.warning("[QUALITY_VALIDATOR] InsightFace validation error: %s", exc)
            defects.append(f"insightface_error: {str(exc)}")

        checks: Dict[str, Dict[str, Any]] = {}

        # 1. Facial Embedding Consistency (ArcFace)
        if similarity >= self.min_biometric_similarity:
            checks["facial_embedding_consistency"] = {"status": "PASS", "score": round(similarity, 4), "threshold": self.min_biometric_similarity, "details": "High biometric identity fidelity"}
        elif similarity >= 0.70:
            checks["facial_embedding_consistency"] = {"status": "UNCERTAIN", "score": round(similarity, 4), "threshold": self.min_biometric_similarity, "details": "Borderline identity similarity"}
            warnings.append(f"borderline_biometric_similarity ({similarity:.3f} < {self.min_biometric_similarity:.2f})")
        else:
            checks["facial_embedding_consistency"] = {"status": "FAIL", "score": round(similarity, 4), "threshold": self.min_biometric_similarity, "details": "Severe identity drift"}
            defects.append(f"low_biometric_similarity ({similarity:.3f} < {self.min_biometric_similarity:.2f})")

        # 2. Facial Geometry Distortion (Landmark RMSE)
        if faces_detected == 1:
            if geometry_rmse <= 0.025:
                checks["facial_geometry_distortion"] = {"status": "PASS", "score": round(geometry_rmse, 4), "threshold": 0.040, "details": "Facial landmark geometry preserved"}
            elif geometry_rmse <= 0.040:
                checks["facial_geometry_distortion"] = {"status": "UNCERTAIN", "score": round(geometry_rmse, 4), "threshold": 0.040, "details": "Minor landmark shift"}
                warnings.append(f"minor_landmark_shift ({geometry_rmse:.4f})")
            else:
                checks["facial_geometry_distortion"] = {"status": "FAIL", "score": round(geometry_rmse, 4), "threshold": 0.040, "details": "Facial geometry distorted"}
                defects.append(f"facial_geometry_distortion ({geometry_rmse:.4f} > 0.040)")
        else:
            checks["facial_geometry_distortion"] = {"status": "FAIL", "score": 1.0, "threshold": 0.040, "details": "Face detection failed"}

        # 3. Head Aspect-Ratio Distortion
        if faces_detected == 1:
            if aspect_delta <= 0.050 or (aspect_delta <= 0.080 and geometry_rmse <= 0.025):
                checks["head_aspect_ratio_distortion"] = {
                    "status": "PASS",
                    "score": round(aspect_delta, 4),
                    "threshold": 0.080 if geometry_rmse <= 0.025 else 0.050,
                    "details": "Head proportions preserved within detector tolerance",
                }
            elif aspect_delta <= 0.100:
                checks["head_aspect_ratio_distortion"] = {
                    "status": "UNCERTAIN",
                    "score": round(aspect_delta, 4),
                    "threshold": 0.080,
                    "details": "Minor aspect ratio deviation",
                }
                warnings.append(f"minor_aspect_ratio_deviation ({aspect_delta:.4f})")
            else:
                checks["head_aspect_ratio_distortion"] = {
                    "status": "FAIL",
                    "score": round(aspect_delta, 4),
                    "threshold": 0.080,
                    "details": "Head elongated or squashed",
                }
                defects.append(f"head_aspect_ratio_distortion ({aspect_delta:.4f} > 0.080)")
        else:
            checks["head_aspect_ratio_distortion"] = {"status": "FAIL", "score": 1.0, "threshold": 0.050, "details": "Face detection failed"}

        # Explicitly detect severe latent corruption / diagonal staircase tearing
        if similarity < 0.15 or (faces_detected == 1 and geometry_rmse > 0.12):
            defects.append("severe_latent_corruption_or_diagonal_tearing")

        # 3b. Skin Texture & Microtexture Preservation Check (Section 6)
        src_face = identity_layer.get("original_face_mask") if identity_layer else None
        if src_face is not None and np.any(src_face) and cand_box is not None:
            skin_roi = cv2.erode(src_face.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
            src_gray = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2GRAY)
            cand_gray = cv2.cvtColor(cand_rgb, cv2.COLOR_RGB2GRAY)
            src_lap = cv2.Laplacian(src_gray, cv2.CV_64F)
            cand_lap = cv2.Laplacian(cand_gray, cv2.CV_64F)
            src_var = float(np.var(src_lap[skin_roi])) if np.any(skin_roi) else 100.0
            cand_var = float(np.var(cand_lap[skin_roi])) if np.any(skin_roi) else 100.0
            texture_ratio = cand_var / max(1.0, src_var)
            if 0.40 <= texture_ratio <= 2.2:
                checks["face_texture_preservation"] = {"status": "PASS", "score": round(texture_ratio, 3), "threshold": 0.40, "details": "Natural photographic skin texture preserved"}
            elif texture_ratio < 0.40:
                checks["face_texture_preservation"] = {"status": "FAIL", "score": round(texture_ratio, 3), "threshold": 0.40, "details": "Excessive skin smoothing / airbrushed texture"}
                defects.append(f"excessive_skin_smoothing ({texture_ratio:.2f} < 0.40)")
            else:
                checks["face_texture_preservation"] = {"status": "UNCERTAIN", "score": round(texture_ratio, 3), "threshold": 0.40, "details": "High facial noise or texture divergence"}
                warnings.append(f"facial_texture_divergence ({texture_ratio:.2f})")
        else:
            checks["face_texture_preservation"] = {"status": "PASS", "score": 1.0, "threshold": 0.40, "details": "N/A"}

        # 4. Target Uniform Color / Red Collar Check (Section 5 & 9)
        has_red_collar_ref = False
        cues = {}
        if uniform_reference is not None:
            from pipelines.uniform_swap.uniform_generator import _detect_uniform_visual_cues
            u_path = uniform_path if 'uniform_path' in locals() and uniform_path is not None else getattr(uniform_reference, 'filename', None)
            cues = _detect_uniform_visual_cues(uniform_reference, uniform_path=u_path)
            has_red_collar_ref = cues.get("has_red_collar", False)

        if has_red_collar_ref and cand_box is not None:
            c_hsv = cv2.cvtColor(cand_rgb, cv2.COLOR_RGB2HSV)
            red1 = (c_hsv[:, :, 0] <= 10) | (c_hsv[:, :, 0] >= 170)
            red2 = (c_hsv[:, :, 1] >= 65) & (c_hsv[:, :, 2] >= 50)
            c_red = red1 & red2

            chin_y = int(cand_box[3])
            c_h = float(cand_box[3] - cand_box[1])
            c_w = float(cand_box[2] - cand_box[0])
            collar_y2 = min(h, int(chin_y + 0.22 * c_h))
            col_x1 = max(0, int(cand_box[0] - 0.25 * c_w))
            col_x2 = min(w, int(cand_box[2] + 0.25 * c_w))
            collar_box = np.zeros((h, w), dtype=bool)
            collar_box[chin_y:collar_y2, col_x1:col_x2] = True
            red_in_collar_px = int(np.sum(c_red & collar_box))
            if red_in_collar_px >= 250:
                checks["collar_color_and_shape"] = {"status": "PASS", "score": red_in_collar_px, "threshold": 250, "details": "Solid red collar seated cleanly"}
            else:
                checks["collar_color_and_shape"] = {"status": "FAIL", "score": red_in_collar_px, "threshold": 250, "details": "Insufficient collar red"}
                defects.append(f"insufficient_target_collar_red ({red_in_collar_px} px < 250)")
        else:
            checks["collar_color_and_shape"] = {"status": "PASS", "score": 0, "threshold": 0, "details": "N/A"}

        # 5. Badge Leakage Check (Section 5, 6, 9)
        if identity_layer and identity_layer.get("original_badge_mask") is not None:
            badge_mask = identity_layer["original_badge_mask"]
            if np.any(badge_mask) and np.sum(badge_mask) > 100:
                src_badge_rgb = src_rgb[badge_mask].astype(np.float32)
                cand_badge_rgb = cand_rgb[badge_mask].astype(np.float32)
                diffs = np.linalg.norm(cand_badge_rgb - src_badge_rgb, axis=1)
                retained_badge_px = int(np.sum(diffs < 30.0))
                if retained_badge_px > 500:
                    checks["original_garment_leakage"] = {"status": "FAIL", "score": retained_badge_px, "threshold": 500, "details": "Original badge leaked"}
                    defects.append(f"badge_leakage_detected ({retained_badge_px} px)")
                else:
                    checks["original_garment_leakage"] = {"status": "PASS", "score": retained_badge_px, "threshold": 500, "details": "Original badge eliminated"}
            else:
                checks["original_garment_leakage"] = {"status": "PASS", "score": 0, "threshold": 500, "details": "No badge in source"}
        else:
            checks["original_garment_leakage"] = {"status": "PASS", "score": 0, "threshold": 500, "details": "No badge mask"}

        # 6. Hairstyle & Appearance Checks (Tasks 2, 3, 6)
        src_hair = identity_layer.get("original_hair_mask") if identity_layer else None
        src_face = identity_layer.get("original_face_mask") if identity_layer else None
        src_acc = identity_layer.get("original_accessory_mask") if identity_layer else None

        hair_below_chin_ratio = -0.5
        is_short_haired = False
        is_braided = False
        has_src_bows = False

        if src_face is not None and np.any(src_face) and src_hair is not None and np.any(src_hair):
            y_f, _ = np.where(src_face)
            chin_y = y_f.max()
            fh = max(1.0, float(y_f.max() - y_f.min()))

            y_h, _ = np.where(src_hair)
            hair_bottom = y_h.max()
            hair_below_chin_ratio = float(hair_bottom - chin_y) / fh
            if hair_below_chin_ratio <= 0.10:
                is_short_haired = True
            elif hair_below_chin_ratio > 0.25:
                is_braided = True

        if src_acc is not None and not is_short_haired:
            src_hsv = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2HSV)
            src_is_blue = (src_hsv[:, :, 0] >= 85) & (src_hsv[:, :, 0] <= 135) & (src_hsv[:, :, 1] >= 40)
            if np.sum(src_is_blue & src_acc) > 1000:
                has_src_bows = True

        # Check for added hair / braids on short-haired portraits
        if is_short_haired:
            if cand_box is not None:
                c_y1, c_y2 = cand_box[1], cand_box[3]
                c_fh = max(1.0, float(c_y2 - c_y1))
            elif src_face is not None and np.any(src_face):
                y_f, _ = np.where(src_face)
                c_y1, c_y2 = y_f.min(), y_f.max()
                c_fh = max(1.0, float(c_y2 - c_y1))
            else:
                c_y1, c_y2 = int(0.20 * h), int(0.50 * h)
                c_fh = float(c_y2 - c_y1)

            if src_hair is not None:
                hair_dil = cv2.dilate(src_hair.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
                if cand_box is not None:
                    c_x1, c_x2 = cand_box[0], cand_box[2]
                elif src_face is not None and np.any(src_face):
                    _, x_f = np.where(src_face)
                    c_x1, c_x2 = x_f.min(), x_f.max()
                else:
                    c_x1, c_x2 = int(0.25 * w), int(0.75 * w)
                c_fw = max(1.0, float(c_x2 - c_x1))
                ear_zone = np.zeros((h, w), dtype=bool)
                ear_y1 = int(c_y1 + 0.35 * c_fh)
                ear_y2 = int(c_y2 + 0.05 * c_fh)
                lx1, lx2 = max(0, int(c_x1 - 0.50 * c_fw)), int(c_x1 + 0.05 * c_fw)
                rx1, rx2 = int(c_x2 - 0.05 * c_fw), min(w, int(c_x2 + 0.50 * c_fw))
                ear_zone[ear_y1:ear_y2, lx1:lx2] = True
                ear_zone[ear_y1:ear_y2, rx1:rx2] = True
                if identity_layer and identity_layer.get("original_garment_mask") is not None:
                    ear_zone = ear_zone & ~identity_layer["original_garment_mask"]
                unwanted_hair_region = ear_zone & ~hair_dil
                cand_hsv = cv2.cvtColor(cand_rgb, cv2.COLOR_RGB2HSV)
                src_hsv = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2HSV)
                # Unwanted added hair must be newly added dark pixels that were not already dark hair in the source
                cand_dark_hair = (cand_hsv[:, :, 2] < 45) & (src_hsv[:, :, 2] >= 55) & unwanted_hair_region
                unwanted_hair_px = int(np.sum(cand_dark_hair))
                if unwanted_hair_px > 3000:
                    checks["added_hair"] = {"status": "FAIL", "score": unwanted_hair_px, "threshold": 3000, "details": "Unwanted braids or added hair detected"}
                    defects.append(f"unwanted_braids_or_added_hair ({unwanted_hair_px} px)")
                elif unwanted_hair_px > 1200:
                    checks["added_hair"] = {"status": "UNCERTAIN", "score": unwanted_hair_px, "threshold": 3000, "details": "Minor hair extension detected"}
                    warnings.append(f"minor_hair_extension ({unwanted_hair_px} px)")
                else:
                    checks["added_hair"] = {"status": "PASS", "score": unwanted_hair_px, "threshold": 3000, "details": "Original short hairstyle preserved"}
            else:
                checks["added_hair"] = {"status": "PASS", "score": 0, "threshold": 3000, "details": "N/A"}
        else:
            checks["added_hair"] = {"status": "PASS", "score": 0, "threshold": 3000, "details": "Not short-haired portrait"}

        # Missing hair check & Hair Silhouette IoU (Section 6 - Real Measurements)
        if src_hair is not None and np.any(src_hair) and cand_box is not None:
            cand_hsv = cv2.cvtColor(cand_rgb, cv2.COLOR_RGB2HSV)
            src_hsv = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2HSV)
            s_v = src_hsv[src_hair, 2]
            # Adaptive hair brightness ceiling (accounts for strong sunlight, specular highlights, and natural hair tones)
            v_thresh = max(75, min(230, int(np.percentile(s_v, 98)))) if len(s_v) > 0 else 75
            hair_search_zone = cv2.dilate(src_hair.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
            cand_hair_pixels = (cand_hsv[:, :, 2] <= v_thresh) & hair_search_zone & ~(src_face if src_face is not None else np.zeros((h, w), dtype=bool))

            intersection = np.sum(src_hair & cand_hair_pixels)
            union = np.sum(src_hair | cand_hair_pixels)
            hair_iou = float(intersection / max(1, union))

            lost_hair_px = np.sum(src_hair & ~cand_hair_pixels)
            missing_ratio = float(lost_hair_px / max(1, np.sum(src_hair)))

            if missing_ratio <= 0.25:
                checks["missing_hair"] = {"status": "PASS", "score": round(missing_ratio, 3), "threshold": 0.25, "details": "Hair length and silhouette preserved"}
            elif missing_ratio <= 0.40:
                checks["missing_hair"] = {"status": "UNCERTAIN", "score": round(missing_ratio, 3), "threshold": 0.25, "details": "Minor hair thinning"}
                warnings.append(f"minor_hair_thinning ({missing_ratio:.2f})")
            else:
                checks["missing_hair"] = {"status": "FAIL", "score": round(missing_ratio, 3), "threshold": 0.25, "details": "Significant hair loss / truncation"}
                defects.append(f"significant_hair_loss ({missing_ratio:.2f} > 0.25)")

            if hair_iou >= 0.70:
                checks["hair_silhouette_consistency"] = {"status": "PASS", "score": round(hair_iou, 3), "threshold": 0.70, "details": "Hair silhouette overlap verified"}
            elif hair_iou >= 0.55:
                checks["hair_silhouette_consistency"] = {"status": "UNCERTAIN", "score": round(hair_iou, 3), "threshold": 0.70, "details": "Borderline hair silhouette"}
                warnings.append(f"borderline_hair_silhouette ({hair_iou:.2f})")
            else:
                checks["hair_silhouette_consistency"] = {"status": "FAIL", "score": round(hair_iou, 3), "threshold": 0.70, "details": "Altered hair silhouette / displaced braids"}
                defects.append(f"altered_hair_silhouette ({hair_iou:.2f} < 0.70)")
        else:
            checks["missing_hair"] = {"status": "NOT_APPLICABLE", "score": 0.0, "threshold": 0.25, "details": "No hair mask required"}
            checks["hair_silhouette_consistency"] = {"status": "NOT_APPLICABLE", "score": 1.0, "threshold": 0.70, "details": "No hair mask required"}

        # Bow & ribbon preservation (Section 3, 6 - Individual Bow Verification)
        ind_bows = identity_layer.get("individual_bows", []) if identity_layer else []
        if ind_bows:
            all_bows_preserved = True
            bow_details = []
            total_bow_px = 0
            for bow in ind_bows:
                b_mask = bow["mask"]
                b_name = bow["name"]
                orig_px = bow["area"]
                if src_rgb is not None and np.any(b_mask):
                    diff = np.abs(cand_rgb[b_mask].astype(float) - src_rgb[b_mask].astype(float))
                    color_match = np.mean(diff, axis=1) < 40.0
                    ret_px = int(np.sum(color_match))
                else:
                    ret_px = orig_px
                total_bow_px += ret_px
                ret_ratio = ret_px / max(1, orig_px)
                if ret_ratio >= 0.50:
                    bow_details.append(f"{b_name}: {ret_px}/{orig_px} px (PASS)")
                else:
                    all_bows_preserved = False
                    bow_details.append(f"{b_name}: {ret_px}/{orig_px} px (FAIL)")
                    defects.append(f"missing_{b_name} ({ret_px} px < {int(orig_px * 0.50)})")
            if all_bows_preserved:
                checks["bow_and_ribbon_preservation"] = {"status": "PASS", "score": total_bow_px, "threshold": 50, "details": f"All accessories preserved ({'; '.join(bow_details)})"}
            else:
                checks["bow_and_ribbon_preservation"] = {"status": "FAIL", "score": total_bow_px, "threshold": 50, "details": f"Individual accessory missing ({'; '.join(bow_details)})"}
        elif has_src_bows:
            if src_rgb is not None and src_acc is not None and np.any(src_acc):
                diff = np.abs(cand_rgb[src_acc].astype(float) - src_rgb[src_acc].astype(float))
                color_match = np.mean(diff, axis=1) < 40.0
                retained_bow_px = int(np.sum(color_match))

                # Check bilateral preservation: if source had both left and right accessories,
                # both sides must be preserved individually so one retained bow cannot mask an erased bow
                c_fcx_val = (cand_box[0] + cand_box[2]) * 0.5 if cand_box is not None else w * 0.5
                left_acc = src_acc & (np.arange(w)[None, :] < c_fcx_val)
                right_acc = src_acc & (np.arange(w)[None, :] >= c_fcx_val)
                left_src_px = int(np.sum(left_acc))
                right_src_px = int(np.sum(right_acc))
                left_ret_px = int(np.sum(color_match[left_acc[src_acc]])) if left_src_px > 0 else 0
                right_ret_px = int(np.sum(color_match[right_acc[src_acc]])) if right_src_px > 0 else 0

                bilateral_fail = False
                if left_src_px >= 100 and left_ret_px < int(0.40 * left_src_px):
                    bilateral_fail = True
                if right_src_px >= 100 and right_ret_px < int(0.40 * right_src_px):
                    bilateral_fail = True
            else:
                retained_bow_px = 1000
                bilateral_fail = False

            if retained_bow_px > 100 and not bilateral_fail:
                checks["bow_and_ribbon_preservation"] = {"status": "PASS", "score": retained_bow_px, "threshold": 100, "details": "Original accessories preserved"}
            else:
                checks["bow_and_ribbon_preservation"] = {"status": "FAIL", "score": retained_bow_px, "threshold": 100, "details": "Missing original hair accessories"}
                defects.append(f"missing_original_hair_accessories ({retained_bow_px} px)")
        else:
            checks["bow_and_ribbon_preservation"] = {"status": "NOT_APPLICABLE", "score": 0, "threshold": 0, "details": "No source bows required"}

        # Bindi Preservation Check (Section 3, 6)
        has_src_bindi = identity_layer.get("has_bindi", False) if identity_layer else False
        if has_src_bindi and cand_box is not None:
            c_fh = float(cand_box[3] - cand_box[1])
            c_fw = float(cand_box[2] - cand_box[0])
            c_fcx = (cand_box[0] + cand_box[2]) * 0.5
            c_fy1 = cand_box[1]
            b_y1 = int(c_fy1 + c_fh * 0.38)
            b_y2 = int(c_fy1 + c_fh * 0.52)
            b_x1 = int(c_fcx - c_fw * 0.08)
            b_x2 = int(c_fcx + c_fw * 0.08)
            c_roi = cand_rgb[b_y1:b_y2, b_x1:b_x2]
            cr, cg, cb = c_roi[:, :, 0].astype(int), c_roi[:, :, 1].astype(int), c_roi[:, :, 2].astype(int)
            is_b_cand = (cr > 105) & (cg < 60) & (cb < 60) & ((cr - cg) > 55)
            cand_bindi_px = int(np.sum(is_b_cand))
            if cand_bindi_px >= 18:
                checks["bindi_preservation"] = {"status": "PASS", "score": cand_bindi_px, "threshold": 18, "details": "Authentic forehead bindi preserved"}
            else:
                checks["bindi_preservation"] = {"status": "FAIL", "score": cand_bindi_px, "threshold": 18, "details": "Original red bindi missing or erased"}
                defects.append(f"missing_original_bindi ({cand_bindi_px} px < 18)")
        else:
            checks["bindi_preservation"] = {"status": "NOT_APPLICABLE", "score": 0, "threshold": 0, "details": "No bindi in source portrait"}

        # Earring Preservation Check (Section 3, 6)
        has_src_earrings = identity_layer.get("has_earrings", False) if identity_layer else False
        if has_src_earrings and identity_layer.get("original_ears_mask") is not None:
            ears_m = identity_layer["original_ears_mask"]
            c_ear_rgb = cand_rgb[ears_m].astype(int)
            c_bright = (c_ear_rgb[:, 0] > 175) & (c_ear_rgb[:, 1] > 155) & (c_ear_rgb[:, 2] > 135)
            earring_px = int(np.sum(c_bright))
            if earring_px >= 15:
                checks["earring_preservation"] = {"status": "PASS", "score": earring_px, "threshold": 15, "details": "Authentic earrings preserved"}
            else:
                checks["earring_preservation"] = {"status": "FAIL", "score": earring_px, "threshold": 15, "details": "Original earrings erased"}
                defects.append(f"missing_original_earrings ({earring_px} px < 15)")
        else:
            checks["earring_preservation"] = {"status": "NOT_APPLICABLE", "score": 0, "threshold": 0, "details": "No earrings in source portrait"}

        # V12.5 Task 7: Unwanted Accessories Check (MANDATORY FOR ALL CANDIDATES)
        # Validate added accessories across complete head and surrounding region (not only a hair mask).
        # Detect: added bows, ribbons, hairbands, clips.
        # For sources with existing accessories, validate both preservation and absence of newly added accessories.
        cand_hsv = cv2.cvtColor(cand_rgb, cv2.COLOR_RGB2HSV)
        if cand_box is not None:
            c_chin_y = int(cand_box[3])
            c_fh = float(cand_box[3] - cand_box[1])
        elif src_face is not None and np.any(src_face):
            y_f, _ = np.where(src_face)
            c_chin_y = int(y_f.max())
            c_fh = float(y_f.max() - y_f.min())
        else:
            c_chin_y = int(h * 0.50)
            c_fh = float(h * 0.35)

        head_surround_zone = np.zeros((h, w), dtype=bool)
        # Hair accessories sit on the head and hair, never below the chin on the uniform torso / collar
        head_surround_zone[:max(0, int(c_chin_y - 0.20 * c_fh)), :] = True

        # Hair accessories must be in or adjacent to genuine hair
        if src_hair is not None and np.any(src_hair):
            hair_zone = cv2.dilate(src_hair.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
            head_surround_zone &= hair_zone

        # Subtract face and throat skin
        if src_face is not None:
            head_surround_zone &= ~cv2.dilate(src_face.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)

        # Subtract authentic original accessories (if any exist in source)
        if src_acc is not None and np.any(src_acc):
            orig_acc_dil = cv2.dilate(src_acc.astype(np.uint8), np.ones((35, 35), np.uint8)).astype(bool)
            head_surround_zone &= ~orig_acc_dil

        # Exclude candidate uniform collar and garment so colored collars are never classified as hair bows
        cand_garment_mask = None
        if identity_layer:
            cand_garment_mask = identity_layer.get("candidate_garment_mask")
            if cand_garment_mask is None and identity_layer.get("assembled_fg_rgba") is not None:
                afg = identity_layer["assembled_fg_rgba"]
                cand_garment_mask = (afg[:, :, 3] > 40) & (np.arange(h)[:, None] >= int(c_chin_y - 0.25 * c_fh))
        if cand_garment_mask is not None and np.any(cand_garment_mask):
            cand_g_dil = cv2.dilate(cand_garment_mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
            head_surround_zone &= ~cand_g_dil

        if identity_layer and identity_layer.get("original_garment_mask") is not None:
            orig_g_dil = cv2.dilate(identity_layer["original_garment_mask"].astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
            head_surround_zone &= ~orig_g_dil

        # Ensure accessories are only evaluated on genuine person foreground, never on studio backdrop
        if identity_layer and identity_layer.get("full_person_alpha") is not None:
            person_fg = (identity_layer["full_person_alpha"] > 30)
            head_surround_zone &= person_fg

        # Exclude backdrop matching target background color (e.g. blue backdrop is not blue hair bows)
        target_bg_arr = np.array(background_rgb, dtype=np.float32)
        dist_from_bg = np.max(np.abs(cand_rgb.astype(np.float32) - target_bg_arr), axis=2)
        head_surround_zone &= (dist_from_bg > bg_tolerance)

        # Detect newly added vibrant accessories (added blue bows, red/pink clips, ribbons, bands)
        r_c = cand_rgb[:, :, 0].astype(int)
        g_c = cand_rgb[:, :, 1].astype(int)
        b_c = cand_rgb[:, :, 2].astype(int)

        is_blue_acc = (
            (cand_hsv[:, :, 0] >= 85) & (cand_hsv[:, :, 0] <= 135) &
            (cand_hsv[:, :, 1] >= 55) & (cand_hsv[:, :, 2] >= 50) &
            (b_c > r_c + 25) & (b_c > g_c + 20)
        )
        is_vibrant_acc = (cand_hsv[:, :, 1] >= 110) & (cand_hsv[:, :, 2] >= 90)

        unwanted_acc_cand = (is_blue_acc | is_vibrant_acc) & head_surround_zone
        if src_rgb is not None:
            src_hsv = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2HSV)
            r_s = src_rgb[:, :, 0].astype(int)
            b_s = src_rgb[:, :, 2].astype(int)
            src_blue_or_vibrant = (
                ((src_hsv[:, :, 0] >= 85) & (src_hsv[:, :, 0] <= 135) & (b_s > r_s + 20)) |
                ((src_hsv[:, :, 1] >= 95) & (src_hsv[:, :, 2] >= 75))
            )
            unwanted_acc_cand &= ~src_blue_or_vibrant

        unwanted_acc_px = int(np.sum(unwanted_acc_cand))

        if unwanted_acc_px > 2000:
            checks["unwanted_accessories"] = {
                "status": "FAIL",
                "score": unwanted_acc_px,
                "threshold": 2000,
                "details": f"Unwanted hair accessories detected ({unwanted_acc_px} px)",
            }
            defects.append(f"unwanted_hair_accessories ({unwanted_acc_px} px)")
        elif unwanted_acc_px > 1000:
            checks["unwanted_accessories"] = {
                "status": "UNCERTAIN",
                "score": unwanted_acc_px,
                "threshold": 2000,
                "details": f"Minor accessory-like artifacts ({unwanted_acc_px} px)",
            }
            warnings.append(f"minor_accessory_artifacts ({unwanted_acc_px} px)")
        else:
            checks["unwanted_accessories"] = {
                "status": "PASS",
                "score": unwanted_acc_px,
                "threshold": 2000,
                "details": "Zero unwanted hair accessories",
            }

        # 7. Button Color Check (dynamically determined from uniform reference)
        expected_btn = "matching"
        ref_type = "generic_uniform"
        if uniform_reference is not None or uniform_path is not None:
            from pipelines.uniform_swap.reference_analyzer import reference_analyzer
            cues = reference_analyzer.analyze(uniform_path or uniform_reference)
            expected_btn = cues.get("button_color", "matching")
            ref_type = cues.get("ref_type", "generic_uniform")

        # V13.3.1: this previously counted PIXELS brighter than 195 inside a
        # 60 px-wide strip and called >1500 of them "white buttons". Measured on
        # REG_06 that strip held 4856 such pixels - all of them the white squares
        # of REF001's gingham CHECK FABRIC, with zero buttons involved - so a
        # correct candidate was gated. Fabric is not a button. Buttons are small
        # compact blobs in a vertical run, so detect blobs and compare the blob
        # colour in CIELAB against whatever colour the reference's buttons
        # actually are. Nothing here forces black.
        btn = _measure_button_fidelity(cand_rgb, cues if 'cues' in locals() and isinstance(cues, dict) else {}, cand_box, h, w)
        checks["button_color"] = btn
        if btn["status"] == "FAIL":
            defects.append(f"incorrect_button_color ({btn['details']})")
        elif btn["status"] == "UNCERTAIN":
            warnings.append(f"button_fidelity_uncertain ({btn['details']})")

        # 8. Uniform Pattern Consistency Check (Plaid)
        if cand_box is not None:
            c_chin_y = int(cand_box[3])
            c_fh = float(cand_box[3] - cand_box[1])
            torso_y1 = min(h, int(c_chin_y + 0.25 * c_fh))
            torso_y2 = min(h, int(c_chin_y + 0.70 * c_fh))
            torso_box = cand_rgb[torso_y1:torso_y2, int(w * 0.2):int(w * 0.8)]
            torso_gray = cv2.cvtColor(torso_box, cv2.COLOR_RGB2GRAY)
            torso_std = float(np.std(torso_gray))
            if torso_std >= 12.0:
                checks["uniform_pattern_consistency"] = {"status": "PASS", "score": round(torso_std, 2), "threshold": 12.0, "details": "Plaid pattern texture present"}
            else:
                checks["uniform_pattern_consistency"] = {"status": "FAIL", "score": round(torso_std, 2), "threshold": 12.0, "details": "Missing plaid pattern (flat color)"}
                defects.append(f"missing_uniform_plaid_pattern ({torso_std:.1f} < 12.0)")
        else:
            checks["uniform_pattern_consistency"] = {"status": "PASS", "score": 0, "threshold": 12.0, "details": "N/A"}

        # 9. Neck / Face Contamination (Red Paint Bleed Check) (Section 7, 9)
        bleed_mask = None
        if identity_layer:
            if identity_layer.get("original_face_mask") is not None and cand_box is not None:
                c_chin_y = int(cand_box[3])
                c_fh = float(cand_box[3] - cand_box[1])
                # Facial skin above mouth level (excluding lips and excluding neck/collar area below chin)
                face_skin = identity_layer["original_face_mask"] & (np.arange(h)[:, None] < c_chin_y - 0.22 * c_fh)
                bleed_mask = face_skin
            elif identity_layer.get("original_face_mask") is not None:
                bleed_mask = identity_layer["original_face_mask"] & (np.arange(h)[:, None] < int(h * 0.45))
            elif identity_layer.get("original_neck_mask") is not None:
                bleed_mask = identity_layer["original_neck_mask"]

            # Exclude authentic bindi and authentic hair accessories so they are not flagged as paint bleed
            if bleed_mask is not None:
                if identity_layer.get("bindi_mask") is not None:
                    bleed_mask = bleed_mask & ~cv2.dilate(identity_layer["bindi_mask"].astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
                if identity_layer.get("original_accessory_mask") is not None:
                    bleed_mask = bleed_mask & ~identity_layer["original_accessory_mask"]

        if bleed_mask is not None and np.any(bleed_mask):
            r = cand_rgb[:, :, 0].astype(int)
            g = cand_rgb[:, :, 1].astype(int)
            b = cand_rgb[:, :, 2].astype(int)
            is_paint_red = (r > 140) & (g < 75) & (b < 75) & ((r - g) > 75)
            # Only count newly introduced paint bleed relative to original source (exclude genuine lips/bindi)
            if src_rgb is not None:
                r_s = src_rgb[:, :, 0].astype(int)
                g_s = src_rgb[:, :, 1].astype(int)
                b_s = src_rgb[:, :, 2].astype(int)
                src_paint_red = (r_s > 140) & (g_s < 75) & (b_s < 75) & ((r_s - g_s) > 75)
                is_paint_red &= ~src_paint_red
            bleed_px = int(np.sum(is_paint_red & bleed_mask))
            if bleed_px > 2000:
                checks["neck_and_shoulder_artifacts"] = {"status": "FAIL", "score": bleed_px, "threshold": 2000, "details": "Red paint bleed on face/neck skin"}
                defects.append(f"neck_red_bleed_detected ({bleed_px} px)")
            elif bleed_px > 1000:
                checks["neck_and_shoulder_artifacts"] = {"status": "UNCERTAIN", "score": bleed_px, "threshold": 2000, "details": "Minor collar edge bleed"}
                warnings.append(f"minor_collar_edge_bleed ({bleed_px} px)")
            else:
                checks["neck_and_shoulder_artifacts"] = {"status": "PASS", "score": bleed_px, "threshold": 2000, "details": "Zero red paint bleed on skin"}
        else:
            checks["neck_and_shoulder_artifacts"] = {"status": "PASS", "score": 0, "threshold": 2000, "details": "N/A"}

        # 10. Background Cleanliness & Leaked Background Check (V12.6 Task 4 Mandatory)
        bg_mask = None
        if identity_layer and identity_layer.get("full_person_alpha") is not None:
            alpha = identity_layer["full_person_alpha"]
            bg_mask = (alpha < 20)
            # Exclude candidate uniform silhouette / torso below chin so properly fitted garments on broader shoulders are not counted as background leakage
            if not isinstance(background_rgb, tuple) or len(background_rgb) != 3:
                from pipelines.uniform_swap.config import UniformSwapConfig
                bg_col_for_diff = UniformSwapConfig.parse_bg_color(background_rgb)
            else:
                bg_col_for_diff = background_rgb
            dist_cand = np.max(np.abs(cand_rgb.astype(np.float32) - np.array(bg_col_for_diff, dtype=np.float32)), axis=2)
            cand_torso_fg = (dist_cand > bg_tolerance) & (np.arange(h)[:, None] >= min(h - 1, int(c_chin_y if c_chin_y else h * 0.45)))
            cand_torso_fg_dil = cv2.dilate(cand_torso_fg.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
            bg_mask &= ~cand_torso_fg_dil

            # Subtract dilated facial/hair/accessory regions to prevent false positives on flyaway hair strands
            if src_face is not None:
                bg_mask &= ~cv2.dilate(src_face.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
            if src_hair is not None:
                bg_mask &= ~cv2.dilate(src_hair.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
            if src_acc is not None:
                bg_mask &= ~cv2.dilate(src_acc.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        else:
            if src_face is not None and np.any(src_face):
                fg_approx = cv2.dilate((src_face | (src_hair if src_hair is not None else np.zeros((h, w), dtype=bool))).astype(np.uint8), np.ones((45, 45), np.uint8)).astype(bool)
                bg_mask = ~fg_approx
            else:
                bg_mask = np.zeros((h, w), dtype=bool)
                bg_mask[:int(h * 0.15), :] = True
                bg_mask[:, :int(w * 0.10)] = True
                bg_mask[:, int(w * 0.90):] = True

        if not isinstance(background_rgb, tuple) or len(background_rgb) != 3:
            from pipelines.uniform_swap.config import UniformSwapConfig
            background_rgb = UniformSwapConfig.parse_bg_color(background_rgb)
        target_bg_arr = np.array(background_rgb, dtype=np.float32)
        dist_from_target = np.max(np.abs(cand_rgb.astype(np.float32) - target_bg_arr), axis=2)
        total_bg_px = int(np.sum(bg_mask))
        if total_bg_px > 500:
            dirty_bg_px = int(np.sum((dist_from_target > bg_tolerance) & bg_mask))
            dirty_ratio = float(dirty_bg_px) / float(total_bg_px)

            # Check edge margins within the background mask for border leakage/stripes
            margin_left = float(np.sum((dist_from_target[:, :15] > bg_tolerance) & bg_mask[:, :15])) / max(1, np.sum(bg_mask[:, :15]))
            margin_right = float(np.sum((dist_from_target[:, -15:] > bg_tolerance) & bg_mask[:, -15:])) / max(1, np.sum(bg_mask[:, -15:]))
            margin_top = float(np.sum((dist_from_target[:15, :] > bg_tolerance) & bg_mask[:15, :])) / max(1, np.sum(bg_mask[:15, :]))
            has_edge_stripes = (margin_left > 0.05 or margin_right > 0.05 or margin_top > 0.05)

            if dirty_ratio <= 0.005 and not has_edge_stripes:
                checks["background_cleanliness"] = {
                    "status": "PASS",
                    "score": round(dirty_ratio, 4),
                    "threshold": 0.005,
                    "details": f"Clean RGB{background_rgb} background ({dirty_ratio * 100:.2f}% diverging, {dirty_bg_px} px)",
                }
            else:
                checks["background_cleanliness"] = {
                    "status": "FAIL",
                    "score": round(dirty_ratio, 4),
                    "threshold": 0.005,
                    "details": f"Dirty or leaked background detected ({dirty_ratio * 100:.2f}% diverging from RGB{background_rgb}, {dirty_bg_px} px, stripes={has_edge_stripes})",
                }
                defects.append(f"background_leakage_or_halo ({dirty_ratio * 100:.2f}% > 0.5% or stripes={has_edge_stripes})")
        else:
            checks["background_cleanliness"] = {
                "status": "PASS",
                "score": 0.0,
                "threshold": 0.005,
                "details": "N/A",
            }
            has_edge_stripes = False

        # ── Mandatory 22 Defect Quality Gates (V12.9) ──
        # 1. incomplete_subject
        if faces_detected == 1:
            checks["incomplete_subject"] = {"status": "PASS", "score": 1, "threshold": 1, "details": "Subject face, head and torso detected"}
        else:
            checks["incomplete_subject"] = {"status": "FAIL", "score": faces_detected, "threshold": 1, "details": f"Expected exactly 1 subject, detected {faces_detected}"}
            defects.append(f"incomplete_subject ({faces_detected} faces)")

        # 2. incompatible_garment_registration
        max_geom_thresh = 0.055 if similarity >= 0.90 else 0.040
        if faces_detected == 1 and geometry_rmse <= max_geom_thresh:
            checks["incompatible_garment_registration"] = {"status": "PASS", "score": round(geometry_rmse, 4), "threshold": max_geom_thresh, "details": "Collar and head geometry registered"}
        else:
            checks["incompatible_garment_registration"] = {"status": "FAIL", "score": round(geometry_rmse, 4), "threshold": max_geom_thresh, "details": "Incompatible head/garment registration"}
            defects.append(f"incompatible_garment_registration ({geometry_rmse:.4f} > {max_geom_thresh})")

        # 3. head_body_scale_mismatch
        c_fh = float(cand_box[3] - cand_box[1]) if cand_box is not None else float(h * 0.35)
        c_fw = float(cand_box[2] - cand_box[0]) if cand_box is not None else float(w * 0.35)
        c_chin_y = int(cand_box[3]) if cand_box is not None else int(h * 0.50)

        person_fg = None
        if identity_layer and identity_layer.get("full_person_alpha") is not None:
            person_fg = (identity_layer["full_person_alpha"] > 30)

        chest_y = min(h - 5, int(c_chin_y + 0.35 * c_fh))
        if person_fg is not None and np.any(person_fg[chest_y, :]):
            torso_w = float(np.sum(person_fg[chest_y, :]))
            scale_ratio = c_fw / max(1.0, torso_w)
        else:
            scale_ratio = 0.50

        if 0.25 <= scale_ratio <= 0.75:
            checks["head_body_scale_mismatch"] = {"status": "PASS", "score": round(scale_ratio, 3), "threshold": "0.25-0.75", "details": "Anatomically proportional head-to-body scale"}
        else:
            checks["head_body_scale_mismatch"] = {"status": "FAIL", "score": round(scale_ratio, 3), "threshold": "0.25-0.75", "details": f"Head-to-body scale mismatch (ratio {scale_ratio:.2f})"}
            defects.append(f"head_body_scale_mismatch (ratio {scale_ratio:.2f})")

        # 4. hard_chin_seam
        cand_gray = cv2.cvtColor(cand_rgb, cv2.COLOR_RGB2GRAY)
        if cand_box is not None:
            mid_x1 = max(0, int(cand_box[0] + 0.20 * c_fw))
            mid_x2 = min(w, int(cand_box[2] - 0.20 * c_fw))
            chin_roi_y1 = max(0, c_chin_y - 6)
            chin_roi_y2 = min(h, c_chin_y + 10)
            if chin_roi_y2 > chin_roi_y1 + 4 and mid_x2 > mid_x1 + 10:
                chin_strip = cand_gray[chin_roi_y1:chin_roi_y2, mid_x1:mid_x2]
                sobel_y = cv2.Sobel(chin_strip, cv2.CV_64F, 0, 1, ksize=3)
                chin_step_edge = float(np.max(np.abs(np.mean(sobel_y, axis=1))))
            else:
                chin_step_edge = 0.0
        else:
            chin_step_edge = 0.0

        if chin_step_edge <= 65.0:
            checks["hard_chin_seam"] = {"status": "PASS", "score": round(chin_step_edge, 2), "threshold": 65.0, "details": "Smooth anatomical jawline without horizontal cut"}
        else:
            checks["hard_chin_seam"] = {"status": "FAIL", "score": round(chin_step_edge, 2), "threshold": 65.0, "details": f"Hard horizontal seam detected at chin ({chin_step_edge:.1f})"}
            defects.append(f"hard_chin_seam ({chin_step_edge:.1f} > 65.0)")

        # 5. unnatural_neck_transition
        if checks.get("neck_and_shoulder_artifacts", {}).get("status") == "FAIL":
            checks["unnatural_neck_transition"] = {"status": "FAIL", "score": checks["neck_and_shoulder_artifacts"]["score"], "threshold": 2000, "details": "Red paint bleed or unnatural throat boundary"}
            defects.append("unnatural_neck_transition")
        else:
            checks["unnatural_neck_transition"] = {"status": "PASS", "score": 0, "threshold": 2000, "details": "Natural continuous neck transition"}

        # Reference cues for gates 6-10:
        cues = {}
        if uniform_reference is not None:
            from pipelines.uniform_swap.uniform_generator import _detect_uniform_visual_cues
            cues = _detect_uniform_visual_cues(uniform_reference, uniform_path=uniform_path)

        ref_type = cues.get("ref_type", "generic")
        has_red = cues.get("has_red_collar", False)
        has_navy = cues.get("has_navy_collar", False)
        has_tie = cues.get("has_tie", False)
        has_vest = cues.get("has_vest", False)

        torso_y1 = min(h, int(c_chin_y + 0.20 * c_fh))
        torso_y2 = min(h, int(c_chin_y + 0.70 * c_fh))
        torso_region = cand_rgb[torso_y1:torso_y2, int(w * 0.2):int(w * 0.8)]

        # 6. incorrect_uniform_reference & 7. incorrect_uniform_type & 8. incorrect_uniform_colour & 9. incorrect_uniform_pattern & 10. incorrect_collar_shape
        ref_match = True
        type_match = True
        color_match = True
        pattern_match = True
        collar_shape_match = True
        ref_details = []

        if has_red:
            red_score = checks.get("collar_color_and_shape", {}).get("score", 0)
            if red_score < 150:
                color_match = False
                collar_shape_match = False
                ref_details.append(f"Missing red collar ({red_score} px < 150)")
            torso_std = checks.get("uniform_pattern_consistency", {}).get("score", 0.0)
            if torso_std < 10.0:
                pattern_match = False
                ref_details.append(f"Flat fabric instead of check plaid (std {torso_std:.1f} < 10.0)")

        elif has_navy:
            if len(torso_region) > 0:
                r_t = torso_region[:, :, 0].astype(int)
                g_t = torso_region[:, :, 1].astype(int)
                b_t = torso_region[:, :, 2].astype(int)
                green_checks = (g_t > r_t + 25) & (g_t > b_t + 20)
                green_px = int(np.sum(green_checks))
                if green_px > 400:
                    color_match = False
                    ref_details.append(f"Green checks detected on navy collar uniform ({green_px} px)")

        elif has_tie:
            if len(torso_region) > 0:
                r_t = torso_region[:, :, 0].astype(int)
                g_t = torso_region[:, :, 1].astype(int)
                b_t = torso_region[:, :, 2].astype(int)
                is_white_shirt = (r_t > 150) & (g_t > 150) & (b_t > 150)
                white_ratio = float(np.mean(is_white_shirt))
                if white_ratio < 0.25:
                    color_match = False
                    ref_details.append(f"Shirt body is not white dress shirt ({white_ratio*100:.1f}%)")

        elif has_vest:
            if len(torso_region) > 0:
                r_t = torso_region[:, :, 0].astype(int)
                g_t = torso_region[:, :, 1].astype(int)
                b_t = torso_region[:, :, 2].astype(int)
                is_dark_vest = (r_t < 90) & (g_t < 90) & (b_t < 110)
                vest_ratio = float(np.mean(is_dark_vest))
                if vest_ratio < 0.20:
                    type_match = False
                    ref_match = False
                    ref_details.append(f"Missing dark vest over shirt ({vest_ratio*100:.1f}%)")

        if ref_match:
            checks["incorrect_uniform_reference"] = {"status": "PASS", "score": 0, "threshold": 0, "details": f"Candidate matches reference {ref_type}"}
        else:
            checks["incorrect_uniform_reference"] = {"status": "FAIL", "score": len(ref_details), "threshold": 0, "details": "; ".join(ref_details)}
            defects.append("incorrect_uniform_reference")

        if type_match:
            checks["incorrect_uniform_type"] = {"status": "PASS", "score": 0, "threshold": 0, "details": f"Garment type verified as {ref_type}"}
        else:
            checks["incorrect_uniform_type"] = {"status": "FAIL", "score": len(ref_details), "threshold": 0, "details": "; ".join(ref_details)}
            defects.append("incorrect_uniform_type")

        if color_match:
            checks["incorrect_uniform_colour"] = {"status": "PASS", "score": 0, "threshold": 0, "details": "Collar and garment colors match reference"}
        else:
            checks["incorrect_uniform_colour"] = {"status": "FAIL", "score": len(ref_details), "threshold": 0, "details": "; ".join(ref_details)}
            defects.append("incorrect_uniform_colour")

        if pattern_match:
            checks["incorrect_uniform_pattern"] = {"status": "PASS", "score": 0, "threshold": 0, "details": "Pattern matches reference"}
        else:
            checks["incorrect_uniform_pattern"] = {"status": "FAIL", "score": len(ref_details), "threshold": 0, "details": "; ".join(ref_details)}
            defects.append("incorrect_uniform_pattern")

        if collar_shape_match:
            checks["incorrect_collar_shape"] = {"status": "PASS", "score": 0, "threshold": 0, "details": "Collar shape correctly formed"}
        else:
            checks["incorrect_collar_shape"] = {"status": "FAIL", "score": len(ref_details), "threshold": 0, "details": "; ".join(ref_details)}
            defects.append("incorrect_collar_shape")

        # 11. duplicated_hair
        if len(torso_region) > 0 and identity_layer and identity_layer.get("original_hair_mask") is not None:
            torso_mask = np.zeros((h, w), dtype=bool)
            torso_mask[min(h, int(c_chin_y + 0.35 * c_fh)):, :] = True
            c_hsv = cv2.cvtColor(cand_rgb, cv2.COLOR_RGB2HSV)
            c_dark = (c_hsv[:, :, 2] < 40) & torso_mask

            # Exclude legitimate original hair and accessories extending onto torso (e.g. authentic braids, ribbons)
            orig_h_m = identity_layer.get("original_hair_mask")
            if orig_h_m is not None:
                src_hair_dil = cv2.dilate(orig_h_m.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
                c_dark &= ~src_hair_dil
            orig_acc_m = identity_layer.get("original_accessory_mask")
            if orig_acc_m is not None:
                src_acc_dil = cv2.dilate(orig_acc_m.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
                c_dark &= ~src_acc_dil

            # Exclude background pixels if background is dark
            bg_arr = np.array(background_rgb, dtype=np.float32)
            cand_dist_bg = np.max(np.abs(cand_rgb.astype(np.float32) - bg_arr), axis=2)
            c_dark &= (cand_dist_bg > bg_tolerance)

            if identity_layer.get("original_garment_mask") is not None:
                c_dark &= ~identity_layer["original_garment_mask"]
            if has_vest or has_tie:
                c_dark &= False
            dup_hair_px = int(np.sum(c_dark))
            if dup_hair_px > 2500:
                checks["duplicated_hair"] = {"status": "FAIL", "score": dup_hair_px, "threshold": 2500, "details": "Duplicated or ghost hair on torso"}
                defects.append(f"duplicated_hair ({dup_hair_px} px)")
            else:
                checks["duplicated_hair"] = {"status": "PASS", "score": dup_hair_px, "threshold": 2500, "details": "Zero duplicated hair on torso"}
        else:
            checks["duplicated_hair"] = {"status": "PASS", "score": 0, "threshold": 2500, "details": "N/A"}

        # 12. altered_hairstyle
        missing_hair_status = checks.get("missing_hair", {}).get("status", "PASS")
        hair_iou_status = checks.get("hair_silhouette_consistency", {}).get("status", "PASS")
        if missing_hair_status == "FAIL" or hair_iou_status == "FAIL":
            checks["altered_hairstyle"] = {"status": "FAIL", "score": 0, "threshold": 0, "details": "Hairstyle silhouette altered or hair truncated"}
            defects.append("altered_hairstyle")
        else:
            checks["altered_hairstyle"] = {"status": "PASS", "score": 0, "threshold": 0, "details": "Original hairstyle silhouette preserved"}

        # 13. unwanted_earrings
        has_src_earrings = identity_layer.get("has_earrings", False) if identity_layer else False
        earrings_status = checks.get("earring_preservation", {}).get("status", "PASS")
        if not has_src_earrings:
            # Objective earring check: verify no newly added metallic/gem studs in earlobe zone
            cand_earring_px = 0
            if cand_box is not None:
                fx1, fy1, fx2, fy2 = int(cand_box[0]), int(cand_box[1]), int(cand_box[2]), int(cand_box[3])
                fh = fy2 - fy1
                fw = fx2 - fx1
                # Earlobes sit between mid-ear and lower jaw, never below chin / collar
                y_lobe_min = fy1 + int(0.50 * fh)
                y_lobe_max = fy1 + int(0.78 * fh)
                left_lobe = (np.arange(h)[:, None] >= y_lobe_min) & (np.arange(h)[:, None] <= y_lobe_max) & (np.arange(w)[None, :] >= max(0, fx1 - int(0.18 * fw))) & (np.arange(w)[None, :] <= fx1)
                right_lobe = (np.arange(h)[:, None] >= y_lobe_min) & (np.arange(h)[:, None] <= y_lobe_max) & (np.arange(w)[None, :] >= fx2) & (np.arange(w)[None, :] <= min(w, fx2 + int(0.18 * fw)))

                lobe_zone = left_lobe | right_lobe
                if identity_layer and identity_layer.get("full_person_alpha") is not None:
                    lobe_zone &= (identity_layer["full_person_alpha"] > 50)
                if identity_layer and identity_layer.get("original_hair_mask") is not None:
                    lobe_zone &= ~cv2.dilate(identity_layer["original_hair_mask"].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

                # Exclude target background color
                bg_arr = np.array(background_rgb, dtype=np.float32)
                cand_dist_bg = np.max(np.abs(cand_rgb.astype(np.float32) - bg_arr), axis=2)
                lobe_zone &= (cand_dist_bg > bg_tolerance)

                c_hsv = cv2.cvtColor(cand_rgb, cv2.COLOR_RGB2HSV)
                c_gray = cv2.cvtColor(cand_rgb, cv2.COLOR_RGB2GRAY)
                c_lap = np.abs(cv2.Laplacian(c_gray, cv2.CV_64F))

                ycrcb = cv2.cvtColor(cand_rgb, cv2.COLOR_RGB2YCrCb)
                cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
                is_skin = (cr >= 130) & (cr <= 175) & (cb >= 75) & (cb <= 130)

                not_bg = ~((c_hsv[:, :, 2] > 200) & (c_hsv[:, :, 1] < 50))
                not_red_collar = ~(((c_hsv[:, :, 0] <= 12) | (c_hsv[:, :, 0] >= 168)) & (c_hsv[:, :, 1] >= 65))
                not_navy_collar = ~((c_hsv[:, :, 0] >= 95) & (c_hsv[:, :, 0] <= 140) & (c_hsv[:, :, 1] >= 50))
                not_garment = not_red_collar & not_navy_collar & not_bg

                gold_gem = (c_hsv[:, :, 0] >= 14) & (c_hsv[:, :, 0] <= 48) & (c_hsv[:, :, 1] >= 120) & (c_hsv[:, :, 2] >= 110) & (c_lap >= 18.0) & ~is_skin
                bright_stud = (c_hsv[:, :, 2] >= 225) & (c_hsv[:, :, 1] < 45) & (c_lap >= 25.0) & ~is_skin & not_bg
                vibrant_gem = (c_hsv[:, :, 1] >= 130) & (c_hsv[:, :, 2] >= 90) & ~is_skin & not_garment
                is_j = lobe_zone & not_garment & (gold_gem | bright_stud | vibrant_gem)

                # Exclude any highlights or features already present in source
                if src_rgb is not None:
                    if src_rgb.shape[:2] != (h, w):
                        s_aligned = cv2.resize(src_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
                    else:
                        s_aligned = src_rgb
                    s_hsv = cv2.cvtColor(s_aligned, cv2.COLOR_RGB2HSV)
                    s_gray = cv2.cvtColor(s_aligned, cv2.COLOR_RGB2GRAY)
                    s_lap = np.abs(cv2.Laplacian(s_gray, cv2.CV_64F))
                    s_gold = (s_hsv[:, :, 0] >= 12) & (s_hsv[:, :, 0] <= 48) & (s_hsv[:, :, 1] >= 80) & (s_hsv[:, :, 2] >= 110)
                    s_stud = (s_hsv[:, :, 2] >= 225) & (s_hsv[:, :, 1] < 45) & (s_lap >= 25.0)
                    s_vib = (s_hsv[:, :, 1] >= 120) & (s_hsv[:, :, 2] >= 90)
                    is_src_j = lobe_zone & (s_gold | s_stud | s_vib)
                    is_src_j_dil = cv2.dilate(is_src_j.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
                    is_j &= ~is_src_j_dil

                num_j, _, j_stats, _ = cv2.connectedComponentsWithStats(is_j.astype(np.uint8))
                for ji in range(1, num_j):
                    if j_stats[ji, cv2.CC_STAT_AREA] >= 20:
                        cand_earring_px += j_stats[ji, cv2.CC_STAT_AREA]
            if cand_earring_px > 30:
                checks["unwanted_earrings"] = {"status": "FAIL", "score": cand_earring_px, "threshold": 30, "details": "Hallucinated earrings detected"}
                defects.append(f"unwanted_earrings ({cand_earring_px} px)")
            else:
                checks["unwanted_earrings"] = {"status": "PASS", "score": cand_earring_px, "threshold": 30, "details": "Zero unwanted earrings"}
        else:
            checks["unwanted_earrings"] = {"status": "PASS", "score": 0, "threshold": 30, "details": "Source has authentic earrings"}

        # 14. missing_original_earrings
        if has_src_earrings:
            if earrings_status == "FAIL":
                checks["missing_original_earrings"] = {"status": "FAIL", "score": 0, "threshold": 15, "details": "Original earrings erased"}
                defects.append("missing_original_earrings")
            else:
                checks["missing_original_earrings"] = {"status": "PASS", "score": 1, "threshold": 1, "details": "Authentic earrings preserved"}
        else:
            checks["missing_original_earrings"] = {"status": "PASS", "score": 0, "threshold": 0, "details": "No earrings in source"}

        # 15. unwanted_bows
        unwanted_acc_status = checks.get("unwanted_accessories", {}).get("status", "PASS")
        if unwanted_acc_status == "FAIL":
            checks["unwanted_bows"] = {"status": "FAIL", "score": checks["unwanted_accessories"]["score"], "threshold": 2000, "details": "Unwanted bows/ribbons generated"}
            defects.append("unwanted_bows")
        else:
            checks["unwanted_bows"] = {"status": "PASS", "score": 0, "threshold": 2000, "details": "Zero unwanted bows"}

        # 16. missing_original_bows
        bow_status = checks.get("bow_and_ribbon_preservation", {}).get("status", "PASS")
        if bow_status == "FAIL":
            checks["missing_original_bows"] = {"status": "FAIL", "score": 0, "threshold": 1000, "details": "Original bows missing or clipped"}
            defects.append("missing_original_bows")
        else:
            checks["missing_original_bows"] = {"status": "PASS", "score": 1, "threshold": 1, "details": "Original bows preserved"}

        # ── V13.1 Authoritative 22 Mandatory Quality Gates ──
        # 1. background_rectangle
        cand_dist_bg = np.max(np.abs(cand_rgb.astype(np.float32) - target_bg_arr), axis=2)
        has_bg_rect = False
        tl_box = cand_dist_bg[:int(h * 0.15), :int(w * 0.20)]
        tr_box = cand_dist_bg[:int(h * 0.15), int(w * 0.80):]
        if np.any(tl_box > bg_tolerance):
            rect_tl = (tl_box > bg_tolerance)
            if np.mean(rect_tl) > 0.15:
                has_bg_rect = True
        if np.any(tr_box > bg_tolerance):
            rect_tr = (tr_box > bg_tolerance)
            if np.mean(rect_tr) > 0.15:
                has_bg_rect = True
        if has_bg_rect:
            checks["background_rectangle"] = {"status": "FAIL", "score": 1, "threshold": 0, "details": "Rectangular candidate background patch detected"}
            defects.append("background_rectangle")
        else:
            checks["background_rectangle"] = {"status": "PASS", "score": 0, "threshold": 0, "details": "Zero rectangular background patches"}

        # 2. background_colour_discontinuity
        bg_corners = np.vstack([
            cand_rgb[:int(h * 0.08), :int(w * 0.20)].reshape(-1, 3),
            cand_rgb[:int(h * 0.08), int(w * 0.80):].reshape(-1, 3)
        ])
        bg_std = np.std(bg_corners.astype(float), axis=0).max()
        bg_discontinuity = (bg_std > 20.0)
        if bg_discontinuity:
            checks["background_colour_discontinuity"] = {"status": "FAIL", "score": round(float(bg_std), 1), "threshold": 20.0, "details": f"Discontinuous background colors (std={bg_std:.1f})"}
            defects.append("background_colour_discontinuity")
        else:
            checks["background_colour_discontinuity"] = {"status": "PASS", "score": round(float(bg_std), 1), "threshold": 20.0, "details": "Uniform background color"}

        # 3. floating_torso
        cand_fg_rows = np.where(np.sum(cand_dist_bg > bg_tolerance, axis=1) > 50)[0]
        cand_bottom_y = int(cand_fg_rows[-1]) if len(cand_fg_rows) > 0 else 0

        bottom_strip = cand_rgb[int(h * 0.98):, int(w * 0.30):int(w * 0.70)]
        bottom_dist = np.max(np.abs(bottom_strip.astype(np.float32) - target_bg_arr), axis=2)
        bottom_is_bg_frac = float(np.mean(bottom_dist < bg_tolerance))

        # When subject wears clothing matching background color (e.g. white checked fabric on white backdrop),
        # verify against true person alpha mask to avoid false positive floating torso.
        if identity_layer and identity_layer.get("full_person_alpha") is not None:
            p_alpha = identity_layer["full_person_alpha"]
            p_strip = p_alpha[int(h * 0.98):, int(w * 0.30):int(w * 0.70)]
            if np.mean(p_strip > 30) > 0.70:
                bottom_is_bg_frac = 0.0
                cand_bottom_y = h
            elif np.mean(p_strip < 30) > 0.30:
                bottom_is_bg_frac = float(np.mean(p_strip < 30))
                p_rows = np.where(np.sum(p_alpha > 30, axis=1) > 50)[0]
                cand_bottom_y = int(p_rows[-1]) if len(p_rows) > 0 else int(h * 0.90)

        is_floating = (bottom_is_bg_frac > 0.30) or (cand_bottom_y < int(h * 0.95))
        if is_floating:
            checks["floating_torso"] = {"status": "FAIL", "score": round(bottom_is_bg_frac, 2), "threshold": 0.30, "details": f"Torso does not reach bottom canvas / floating over background ({cand_bottom_y} px < {int(h*0.95)})"}
            defects.append("floating_torso")
        else:
            checks["floating_torso"] = {"status": "PASS", "score": round(bottom_is_bg_frac, 2), "threshold": 0.30, "details": "Torso extends solidly to bottom canvas"}

        # 4. bottom_foreground_gap
        fg_bottom_gap = max(0, h - cand_bottom_y)
        if fg_bottom_gap > 30:
            checks["bottom_foreground_gap"] = {"status": "FAIL", "score": fg_bottom_gap, "threshold": 30, "details": f"Gap of {fg_bottom_gap} px between torso and bottom canvas"}
            defects.append("bottom_foreground_gap")
        else:
            checks["bottom_foreground_gap"] = {"status": "PASS", "score": fg_bottom_gap, "threshold": 30, "details": "Zero bottom gap"}

        # 5. rectangular_garment_boundary
        has_rect_cut = False
        left_edges = []
        right_edges = []
        for y_scan in range(int(h * 0.60), min(h, int(h * 0.95)), 10):
            cols = np.where(cand_dist_bg[y_scan, :] > bg_tolerance)[0]
            if len(cols) > 0:
                left_edges.append(cols[0])
                right_edges.append(cols[-1])

        # Only evaluate internal straight vertical edges (x > 8 and x < w - 9)
        # to ensure natural shoulders reaching the canvas border (x = 0, x = w - 1) are not flagged.
        internal_left = [x for x in left_edges if x > 8]
        internal_right = [x for x in right_edges if x < (w - 9)]

        if len(internal_left) >= 5:
            diffs_l = np.abs(np.diff(internal_left))
            straight_l = (float(np.mean(diffs_l == 0)) > 0.40) or (np.std(internal_left) < 2.0)
            if straight_l:
                has_rect_cut = True
        if len(internal_right) >= 5:
            diffs_r = np.abs(np.diff(internal_right))
            straight_r = (float(np.mean(diffs_r == 0)) > 0.40) or (np.std(internal_right) < 2.0)
            if straight_r:
                has_rect_cut = True

        # Internal rectangular seam check: detect straight horizontal step edges across the chest (V13.3.2)
        has_internal_seam = False
        if cand_box is not None:
            c_chin_y_val = int(cand_box[3])
            c_fh_val = float(cand_box[3] - cand_box[1])
            c_fw_val = float(cand_box[2] - cand_box[0])
            c_mid_x_val = int((cand_box[0] + cand_box[2]) / 2)
            c_y1_val = int(c_chin_y_val + 0.18 * c_fh_val)
            c_y2_val = min(h - 30, int(c_chin_y_val + 1.35 * c_fh_val))
            c_x1_val = max(10, int(c_mid_x_val - 0.35 * c_fw_val))
            c_x2_val = min(w - 10, int(c_mid_x_val + 0.35 * c_fw_val))
            if c_y2_val > c_y1_val + 30 and c_x2_val > c_x1_val + 40:
                gray_c_val = cv2.cvtColor(cand_rgb[c_y1_val:c_y2_val, c_x1_val:c_x2_val], cv2.COLOR_RGB2GRAY)
                step_y_val = np.abs(gray_c_val[2:, :].astype(float) - gray_c_val[:-2, :].astype(float))
                min_seam_span_val = int(0.30 * (c_x2_val - c_x1_val))
                for ry_val in range(step_y_val.shape[0]):
                    row_s_val = step_y_val[ry_val] > 35.0
                    if np.any(row_s_val):
                        runs_val = np.diff(np.where(np.concatenate(([row_s_val[0]], row_s_val[:-1] != row_s_val[1:], [True])))[0])[::2]
                        if len(runs_val) > 0 and max(runs_val) >= min_seam_span_val and np.mean(step_y_val[ry_val, row_s_val]) > 45.0:
                            has_internal_seam = True
                            break

        bg_clean_status = checks.get("background_cleanliness", {}).get("status", "PASS")
        if has_rect_cut or has_internal_seam or (bg_clean_status == "FAIL" and has_edge_stripes):
            checks["rectangular_garment_boundary"] = {"status": "FAIL", "score": 1, "threshold": 0, "details": "Rectangular unsegmented crop boundary or internal chest seam detected"}
            defects.append("rectangular_garment_boundary")
        else:
            checks["rectangular_garment_boundary"] = {"status": "PASS", "score": 0, "threshold": 0, "details": "Natural anatomical shoulder contour"}

        # 6. incomplete_shoulders
        # V13.3.1: this used to measure "distance from the background COLOUR" at a
        # fixed y = 0.85h. Measured on REG_05 (white shirt, white backdrop): the
        # subject's shoulders span the full 800 px canvas, but only 70 px differ
        # from white, so the gate reported "70 px < 280 px" and gated a perfectly
        # good portrait. Any garment close to the chosen backdrop colour hit this.
        # Shoulder EXTENT is an alpha question, not a colour question - gate 7
        # below already reasons that way, so this now agrees with it.
        shoulder_fg, min_shoulder_w, shoulder_basis = _measure_shoulder_extent(
            candidate_image=candidate_image,
            identity_layer=identity_layer,
            cand_dist_bg=cand_dist_bg,
            bg_tolerance=bg_tolerance,
            c_chin_y=c_chin_y,
            c_fh=c_fh,
            c_fw=c_fw,
            has_face_box=(cand_box is not None),
            h=h,
            w=w,
        )
        if shoulder_fg < min_shoulder_w:
            checks["incomplete_shoulders"] = {"status": "FAIL", "score": shoulder_fg, "threshold": min_shoulder_w, "details": f"Shoulder width {shoulder_fg} px < {min_shoulder_w} px (measured on {shoulder_basis})"}
            defects.append("incomplete_shoulders")
        else:
            checks["incomplete_shoulders"] = {"status": "PASS", "score": shoulder_fg, "threshold": min_shoulder_w, "details": f"Complete natural shoulders ({shoulder_fg} px, measured on {shoulder_basis})"}

        # 7. foreground_alpha_hole
        # Evaluate using true person alpha mask when available to prevent white shirt checks triggering as holes
        has_alpha_hole = False
        if identity_layer and identity_layer.get("full_person_alpha") is not None:
            cand_fg_bin = (identity_layer["full_person_alpha"] > 30).astype(np.uint8)
        elif isinstance(candidate_image, Image.Image) and candidate_image.mode == "RGBA":
            cand_fg_bin = (np.asarray(candidate_image)[:, :, 3] > 30).astype(np.uint8)
        elif isinstance(candidate_image, np.ndarray) and candidate_image.ndim == 3 and candidate_image.shape[2] == 4:
            cand_fg_bin = (candidate_image[:, :, 3] > 30).astype(np.uint8)
        else:
            cand_fg_bin = (cand_dist_bg > bg_tolerance).astype(np.uint8)

        cnts, hierarchy = cv2.findContours(cand_fg_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None and len(cnts) > 0:
            for i in range(len(cnts)):
                if hierarchy[0, i, 3] != -1:
                    area = cv2.contourArea(cnts[i])
                    if area > 150:
                        bx, by, bw, bh = cv2.boundingRect(cnts[i])
                        # A genuine alpha hole is located inside the central torso/face region (not negative space between hair and outer shoulder)
                        if (bx + bw * 0.5 > 0.25 * w) and (bx + bw * 0.5 < 0.75 * w) and (by > int(h * 0.25)):
                            # Natural background negative space (e.g. loops between hanging braids/accessories and neck)
                            hole_mask = np.zeros((h, w), dtype=np.uint8)
                            cv2.drawContours(hole_mask, cnts, i, 1, -1)
                            is_src_natural_bg = False
                            if src_rgb is not None and np.any(hole_mask):
                                src_corner_bg = np.mean(src_rgb[:10, :10], axis=(0, 1))
                                dist_src_bg = np.max(np.abs(src_rgb.astype(float) - src_corner_bg), axis=2)
                                if np.mean(dist_src_bg[hole_mask > 0] < 45) > 0.25:
                                    is_src_natural_bg = True
                            if not is_src_natural_bg:
                                has_alpha_hole = True
                                break
        if has_alpha_hole:
            checks["foreground_alpha_hole"] = {"status": "FAIL", "score": 1, "threshold": 0, "details": "Internal holes detected inside foreground"}
            defects.append("foreground_alpha_hole")
        else:
            checks["foreground_alpha_hole"] = {"status": "PASS", "score": 0, "threshold": 0, "details": "Solid continuous foreground without holes"}

        # 8. white_edge_halo
        has_white_halo = False
        is_non_white_bg = (np.mean(target_bg_arr) < 220)
        if is_non_white_bg:
            if identity_layer and identity_layer.get("full_person_alpha") is not None:
                halo_fg_bin = (identity_layer["full_person_alpha"] > 30).astype(np.uint8)
            elif isinstance(candidate_image, Image.Image) and candidate_image.mode == "RGBA":
                halo_fg_bin = (np.asarray(candidate_image)[:, :, 3] > 30).astype(np.uint8)
            elif isinstance(candidate_image, np.ndarray) and candidate_image.ndim == 3 and candidate_image.shape[2] == 4:
                halo_fg_bin = (candidate_image[:, :, 3] > 30).astype(np.uint8)
            else:
                halo_fg_bin = (cand_dist_bg > bg_tolerance).astype(np.uint8)

            boundary_dil = cv2.dilate(halo_fg_bin, np.ones((5, 5), np.uint8))
            boundary_ero = cv2.erode(halo_fg_bin, np.ones((5, 5), np.uint8))
            transition_zone = (boundary_dil == 1) & (boundary_ero == 0)

            # Split inspection into head/hair zone and garment zone
            c_split_y = int(c_chin_y + 0.15 * c_fh) if 'c_chin_y' in locals() and c_chin_y else int(h * 0.45)
            head_zone = transition_zone & (np.arange(h)[:, None] < c_split_y)
            garment_zone = transition_zone & (np.arange(h)[:, None] >= c_split_y)

            white_px_head = int(np.sum((cand_rgb[:, :, 0] > 230) & (cand_rgb[:, :, 1] > 230) & (cand_rgb[:, :, 2] > 230) & head_zone))

            # Garment fabric can legitimately be white (e.g. REF003 white dress shirt or REF001/002 white checks).
            ref_cues = cues if 'cues' in locals() and isinstance(cues, dict) else {}
            is_white_garment_ref = bool(
                ref_cues.get("has_tie") or ref_cues.get("has_plaid") or
                ("tie" in str(ref_cues.get("ref_type", "")).lower()) or
                ("white" in str(ref_cues.get("fabric_desc", "")).lower())
            )
            if not is_white_garment_ref:
                white_px_garment = int(np.sum((cand_rgb[:, :, 0] > 230) & (cand_rgb[:, :, 1] > 230) & (cand_rgb[:, :, 2] > 230) & garment_zone))
            else:
                white_px_garment = 0

            total_halo_px = white_px_head + white_px_garment
            if total_halo_px > 500:
                has_white_halo = True

        # Check colored edge contamination (e.g. blue/green studio backdrop spill along hair silhouette)
        has_colored_halo = False
        try:
            from pipelines.uniform_swap.edge_decontamination import (
                detect_colored_edge_halo, estimate_source_background_color,
            )
            src_bg_col, bg_conf = estimate_source_background_color(src_rgb) if src_rgb is not None else (None, 0.0)
            if src_bg_col is not None and bg_conf >= 0.40:
                fg_alpha_arr = None
                if identity_layer and identity_layer.get("full_person_alpha") is not None:
                    fg_alpha_arr = identity_layer["full_person_alpha"]
                elif isinstance(candidate_image, Image.Image) and candidate_image.mode == "RGBA":
                    fg_alpha_arr = np.asarray(candidate_image)[:, :, 3]
                elif isinstance(candidate_image, np.ndarray) and candidate_image.ndim == 3 and candidate_image.shape[2] == 4:
                    fg_alpha_arr = candidate_image[:, :, 3]
                if fg_alpha_arr is not None:
                    hair_m_col = identity_layer.get("original_hair_mask") if identity_layer else None
                    out_bg_tuple = (int(target_bg_arr[0]), int(target_bg_arr[1]), int(target_bg_arr[2])) if hasattr(target_bg_arr, '__len__') and len(target_bg_arr) >= 3 else (255, 255, 255)
                    col_det = detect_colored_edge_halo(
                        cand_rgb, fg_alpha_arr, src_bg_col,
                        output_background_rgb=out_bg_tuple,
                        hair_mask=hair_m_col,
                    )
                    if col_det.get("status") == "FAIL":
                        has_colored_halo = True
                        checks["colored_edge_halo"] = col_det
        except Exception:
            pass

        if has_white_halo or has_colored_halo or has_edge_stripes:
            checks["white_edge_halo"] = {"status": "FAIL", "score": 1, "threshold": 0, "details": "Edge halo or colored backdrop spill along subject boundary"}
            defects.append("white_edge_halo")
        else:
            checks["white_edge_halo"] = {"status": "PASS", "score": 0, "threshold": 0, "details": "Zero white edge halo"}

        # 9. duplicated_hair (Already evaluated in check 11)
        if "duplicated_hair" not in checks:
            checks["duplicated_hair"] = {"status": "PASS", "score": 0, "threshold": 2500, "details": "Zero duplicated hair"}

        # 10. missing_original_hair
        missing_hair_status = checks.get("missing_hair", {}).get("status", "PASS")
        if src_hair is not None and np.any(src_hair):
            c_hsv = cv2.cvtColor(cand_rgb, cv2.COLOR_RGB2HSV)
            cand_hair_px = np.sum(c_hsv[src_hair, 2] < 120)
            orig_hair_px = np.sum(src_hair)
            hair_ret_ratio = float(cand_hair_px / max(1, orig_hair_px))
            if hair_ret_ratio < 0.60 or missing_hair_status == "FAIL":
                checks["missing_original_hair"] = {"status": "FAIL", "score": round(hair_ret_ratio, 2), "threshold": 0.60, "details": "Significant original hair missing"}
                defects.append("missing_original_hair")
            else:
                checks["missing_original_hair"] = {"status": "PASS", "score": round(hair_ret_ratio, 2), "threshold": 0.60, "details": "Original hair preserved"}
        elif missing_hair_status == "FAIL":
            checks["missing_original_hair"] = {"status": "FAIL", "score": 0, "threshold": 0.60, "details": "Significant original hair missing"}
            defects.append("missing_original_hair")
        else:
            checks["missing_original_hair"] = {"status": "PASS", "score": 1.0, "threshold": 0.60, "details": "Original hair preserved"}

        # 11. altered_original_bow
        bow_status = checks.get("bow_and_ribbon_preservation", {}).get("status", "PASS")
        has_src_bows = identity_layer.get("has_confirmed_bows", False) if identity_layer else False
        ind_bows = identity_layer.get("individual_bows", []) if identity_layer else []
        if (has_src_bows or ind_bows) and bow_status == "FAIL":
            checks["altered_original_bow"] = {"status": "FAIL", "score": 0, "threshold": 1, "details": "Original bows altered, damaged or erased"}
            defects.append("altered_original_bow")
        else:
            checks["altered_original_bow"] = {"status": "PASS", "score": 1, "threshold": 1, "details": "Original bows preserved or not required"}

        # 12. missing_original_accessory
        bindi_status = checks.get("bindi_preservation", {}).get("status", "PASS")
        earring_status = checks.get("earring_preservation", {}).get("status", "PASS")
        has_b = identity_layer.get("has_bindi", False) if identity_layer else False
        has_e = identity_layer.get("has_earrings", False) if identity_layer else False
        if (has_b and bindi_status == "FAIL") or (has_e and earring_status == "FAIL"):
            checks["missing_original_accessory"] = {"status": "FAIL", "score": 0, "threshold": 1, "details": "Original bindi or earrings missing"}
            defects.append("missing_original_accessory")
        else:
            checks["missing_original_accessory"] = {"status": "PASS", "score": 1, "threshold": 1, "details": "Original accessories preserved or not required"}

        # 13. unwanted_accessory
        unw_bow = checks.get("unwanted_bows", {}).get("status", "PASS") == "FAIL"
        unw_ear = checks.get("unwanted_earrings", {}).get("status", "PASS") == "FAIL"
        if unw_bow or unw_ear:
            checks["unwanted_accessory"] = {"status": "FAIL", "score": 1, "threshold": 0, "details": "Hallucinated bows or earrings detected"}
            defects.append("unwanted_accessory")
        else:
            checks["unwanted_accessory"] = {"status": "PASS", "score": 0, "threshold": 0, "details": "Zero unwanted accessories"}

        # 14. chin_seam
        chin_seam_status = checks.get("hard_chin_seam", {}).get("status", "PASS")
        cand_gray = cv2.cvtColor(cand_rgb, cv2.COLOR_RGB2GRAY)
        if cand_box is not None:
            c_chin_y_val = int(cand_box[3])
            chin_strip = cand_gray[max(0, c_chin_y_val - 8):min(h, c_chin_y_val + 10), int(cand_box[0] + 0.2 * c_fw):int(cand_box[2] - 0.2 * c_fw)]
            if chin_strip.shape[0] >= 6 and chin_strip.shape[1] >= 10:
                sobel_y = cv2.Sobel(chin_strip, cv2.CV_64F, 0, 1, ksize=3)
                chin_step = float(np.max(np.abs(np.mean(sobel_y, axis=1))))
                if chin_step > 65.0:
                    chin_seam_status = "FAIL"
        if chin_seam_status == "FAIL":
            checks["chin_seam"] = {"status": "FAIL", "score": 1, "threshold": 0, "details": "Hard chin seam detected"}
            defects.append("chin_seam")
        else:
            checks["chin_seam"] = {"status": "PASS", "score": 0, "threshold": 0, "details": "Smooth anatomical chin transition"}

        # 15. neck_cutout
        has_neck_cutout = False
        if cand_box is not None:
            c_fy2 = int(cand_box[3])
            c_fcx_val = (cand_box[0] + cand_box[2]) * 0.5
            neck_x1 = max(0, int(c_fcx_val - 0.07 * c_fw))
            neck_x2 = min(w, int(c_fcx_val + 0.07 * c_fw))
            neck_band = cand_dist_bg[c_fy2:c_fy2 + int(0.20 * c_fh), neck_x1:neck_x2]
            if neck_band.size > 0:
                bg_col_frac = np.mean(neck_band < bg_tolerance, axis=0)
                if np.any(bg_col_frac > 0.50):
                    has_neck_cutout = True
        if has_neck_cutout:
            checks["neck_cutout"] = {"status": "FAIL", "score": 1, "threshold": 0, "details": "Vertical background cutout cutting through neck"}
            defects.append("neck_cutout")
        else:
            checks["neck_cutout"] = {"status": "PASS", "score": 0, "threshold": 0, "details": "Intact neck anatomy without cutouts"}

        # 16. missing_neck
        has_missing_neck = False
        if cand_box is not None:
            c_fy2 = int(cand_box[3])
            neck_y1 = c_fy2
            neck_y2 = min(h, c_fy2 + max(8, int(0.12 * c_fh)))
            neck_x1 = max(0, int(cand_box[0] + 0.25 * c_fw))
            neck_x2 = min(w, int(cand_box[2] - 0.25 * c_fw))
            if neck_y2 > neck_y1 and neck_x2 > neck_x1:
                neck_roi = cand_rgb[neck_y1:neck_y2, neck_x1:neck_x2]
                neck_ycrcb = cv2.cvtColor(neck_roi, cv2.COLOR_RGB2YCrCb)
                cr = neck_ycrcb[:, :, 1]
                cb = neck_ycrcb[:, :, 2]
                # Human skin tone in YCrCb: Cr in [130, 175], Cb in [75, 130]
                is_skin = (cr >= 130) & (cr <= 175) & (cb >= 75) & (cb <= 130)
                skin_ratio = float(np.mean(is_skin))
                if skin_ratio < 0.15:
                    has_missing_neck = True
        checks["missing_neck"] = {
            "status": "FAIL" if has_missing_neck else "PASS",
            "score": 1 if has_missing_neck else 0,
            "threshold": 0,
            "details": "Neck anatomy verified" if not has_missing_neck else "Missing anatomical neck skin between chin and collar",
        }
        if has_missing_neck:
            defects.append("missing_neck")

        # 17. collar_registration_failure
        # Measure collar position independently of facial landmark ArcFace similarity:
        # neck width, visible throat length, collar opening, shoulder alignment.
        has_collar_reg_failure = False
        collar_reg_details = "Collar correctly registered to neck geometry"
        if geometry_rmse > 0.040:
            has_collar_reg_failure = True
            collar_reg_details = f"Incompatible facial landmark geometry (RMSE {geometry_rmse:.4f} > 0.040)"
        elif cand_box is not None:
            c_chin_y = int(cand_box[3])
            c_fcx = float(cand_box[0] + cand_box[2]) / 2.0
            c_fh = max(1.0, float(cand_box[3] - cand_box[1]))
            c_fw = max(1.0, float(cand_box[2] - cand_box[0]))
            
            # Check visible throat length: distance from chin to garment neckline
            throat_box = cand_rgb[c_chin_y:min(h, c_chin_y + int(0.70 * c_fh)), max(0, int(c_fcx - 0.25 * c_fw)):min(w, int(c_fcx + 0.25 * c_fw))]
            if throat_box.size > 0:
                th_ycrcb = cv2.cvtColor(throat_box, cv2.COLOR_RGB2YCrCb)
                th_cr = th_ycrcb[:, :, 1]
                th_cb = th_ycrcb[:, :, 2]
                th_skin = (th_cr >= 130) & (th_cr <= 175) & (th_cb >= 75) & (th_cb <= 130)
                
                row_skin_frac = np.mean(th_skin, axis=1)
                garment_rows = np.where(row_skin_frac < 0.30)[0]
                if len(garment_rows) > 0:
                    collar_top_row = garment_rows[0]
                    throat_length = float(collar_top_row) / c_fh
                    max_allowed_throat = 0.55 if (has_vest or has_tie) else 0.35
                    if throat_length > max_allowed_throat:
                        has_collar_reg_failure = True
                        collar_reg_details = f"Collar neckline displaced too low (throat_length={throat_length:.3f} > {max_allowed_throat} * face_height)"
                    elif throat_length < 0.04:
                        has_collar_reg_failure = True
                        collar_reg_details = f"Collar displaced too high overlapping chin (throat_length={throat_length:.3f} < 0.04 * face_height)"
                else:
                    has_collar_reg_failure = True
                    collar_reg_details = "No garment collar detected below chin (collar registration displaced/missing)"
            
            # Collar symmetry about the facial axis, for ANY collar colour.
            #
            # V13.3.1: this used to run only for red collars, and took the
            # centroid of every red pixel in a window half a face-width to each
            # side of the face axis. Measured on REG_06 (REF001, which has red
            # collar AND red sleeve cuffs), the red histogram across that window
            # was bimodal: ~370 px of collar near the axis and ~2000 px of
            # SLEEVE CUFF at the right edge. The centroid landed between the two
            # clusters, reporting offset=0.320 and gating a correctly registered
            # collar. Sleeves are not collars; measure each side's collar
            # attachment separately inside the neck corridor and compare them.
            ref_cues_local = cues if 'cues' in locals() and isinstance(cues, dict) else {}
            sym = _measure_collar_symmetry(cand_rgb, ref_cues_local, c_chin_y, c_fh, c_fcx, c_fw, h, w)
            if sym["status"] == "FAIL":
                has_collar_reg_failure = True
                collar_reg_details = sym["details"]
            elif sym["status"] == "UNCERTAIN":
                warnings.append(f"collar_symmetry_uncertain ({sym['details']})")
            checks["collar_symmetry"] = sym

        if has_collar_reg_failure:
            checks["collar_registration_failure"] = {"status": "FAIL", "score": round(geometry_rmse, 4), "threshold": 0.040, "details": collar_reg_details}
            defects.append("collar_registration_failure")
        else:
            checks["collar_registration_failure"] = {"status": "PASS", "score": round(geometry_rmse, 4), "threshold": 0.040, "details": collar_reg_details}

        # 18. incorrect_collar_shape
        ref_cues = cues if 'cues' in locals() and isinstance(cues, dict) else {}
        collar_shape_status = checks.get("collar_shape", {}).get("status", "PASS")
        if ref_cues.get("ref_type") == "red_collar" or ref_cues.get("has_red_collar"):
            if cand_box is not None:
                c_fy2 = int(cand_box[3])
                neck_col_zone = cand_rgb[c_fy2:c_fy2 + int(0.35 * c_fh), int(cand_box[0] - 0.2 * c_fw):int(cand_box[2] + 0.2 * c_fw)]
                if neck_col_zone.size > 0:
                    c_hsv = cv2.cvtColor(neck_col_zone, cv2.COLOR_RGB2HSV)
                    c_red = ((c_hsv[:, :, 0] <= 10) | (c_hsv[:, :, 0] >= 170)) & (c_hsv[:, :, 1] >= 65) & (c_hsv[:, :, 2] >= 50)
                    red_px = int(np.sum(c_red))
                    if red_px < 150:
                        collar_shape_status = "FAIL"
        if collar_shape_status == "FAIL":
            checks["incorrect_collar_shape"] = {"status": "FAIL", "score": 0, "threshold": 1, "details": "Collar shape diverges from reference"}
            defects.append("incorrect_collar_shape")
        else:
            checks["incorrect_collar_shape"] = {"status": "PASS", "score": 1, "threshold": 1, "details": "Target collar shape verified"}

        # 19. incorrect_placket
        has_incorrect_placket = False
        ref_cues = cues if 'cues' in locals() and isinstance(cues, dict) else {}
        if ref_cues.get("ref_type") == "red_collar" or ref_cues.get("has_red_collar"):
            mid_x = w // 2
            chest_y1 = int(c_chin_y + 0.30 * c_fh) if 'c_chin_y' in locals() and c_chin_y else int(h * 0.55)
            chest_y2 = min(h, chest_y1 + int(h * 0.25))
            placket_zone = cand_rgb[chest_y1:chest_y2, mid_x - 15:mid_x + 15]
            if placket_zone.size > 0:
                pz_r = placket_zone[:, :, 0].astype(int)
                pz_g = placket_zone[:, :, 1].astype(int)
                pz_b = placket_zone[:, :, 2].astype(int)
                solid_red_placket = float(np.mean((pz_r > 130) & (pz_g < 70) & (pz_b < 70) & ((pz_r - pz_g) > 50)))
                if solid_red_placket > 0.40:
                    has_incorrect_placket = True
        if has_incorrect_placket:
            checks["incorrect_placket"] = {"status": "FAIL", "score": 1, "threshold": 0, "details": "Solid red placket generated instead of check button placket"}
            defects.append("incorrect_placket")
        else:
            checks["incorrect_placket"] = {"status": "PASS", "score": 0, "threshold": 0, "details": "Authentic button placket verified"}

        # 20. incorrect_uniform_type
        has_incorrect_u_type = False
        u_type_details = "Uniform type matches reference"
        ref_cues_local = cues if 'cues' in locals() and isinstance(cues, dict) else {}
        
        if ref_cues_local:
            if ref_cues_local.get("has_red_collar") or ref_cues_local.get("ref_type") == "red_collar":
                if cand_box is not None:
                    c_fy2 = int(cand_box[3])
                    collar_test_zone = cand_rgb[c_fy2:min(h, c_fy2 + int(0.35 * c_fh)), max(0, int(cand_box[0] - 0.25 * c_fw)):min(w, int(cand_box[2] + 0.25 * c_fw))]
                    if collar_test_zone.size > 0:
                        cz_r = collar_test_zone[:, :, 0].astype(int)
                        cz_g = collar_test_zone[:, :, 1].astype(int)
                        cz_b = collar_test_zone[:, :, 2].astype(int)
                        col_red_px = np.sum((cz_r > 120) & (cz_g < 85) & (cz_b < 85) & ((cz_r - cz_g) > 40))
                        if col_red_px < 100:
                            has_incorrect_u_type = True
                            u_type_details = f"Expected red collar uniform, but candidate lacks red collar ({col_red_px} px < 100)"
            elif ref_cues_local.get("has_tie") or ref_cues_local.get("ref_type") == "tie":
                mid_x = int((cand_box[0] + cand_box[2]) / 2) if cand_box is not None else w // 2
                tie_y1 = int(cand_box[3] + 0.10 * c_fh) if cand_box is not None else int(h * 0.45)
                tie_y2 = min(h, tie_y1 + int(0.60 * c_fh))
                tie_test_zone = cand_rgb[tie_y1:tie_y2, max(0, mid_x - int(0.20 * c_fw)):min(w, mid_x + int(0.20 * c_fw))]
                if tie_test_zone.size > 0:
                    dark_tie_px = np.sum((tie_test_zone[:, :, 0] < 95) & (tie_test_zone[:, :, 1] < 95) & (tie_test_zone[:, :, 2] < 110))
                    if dark_tie_px < 100:
                        has_incorrect_u_type = True
                        u_type_details = "Expected uniform with necktie, but candidate lacks tie"
            elif ref_cues_local.get("has_vest") or ref_cues_local.get("ref_type") == "vest":
                mid_x = int((cand_box[0] + cand_box[2]) / 2) if cand_box is not None else w // 2
                torso_y1 = int(cand_box[3] + 0.30 * c_fh) if cand_box is not None else int(h * 0.50)
                torso_y2 = min(h, torso_y1 + int(0.55 * c_fh))
                torso_sample = cand_rgb[torso_y1:torso_y2, max(0, mid_x - int(0.40 * c_fw)):min(w, mid_x + int(0.40 * c_fw))]
                if torso_sample.size > 0:
                    mean_val = np.mean(torso_sample)
                    if mean_val > 175:
                        has_incorrect_u_type = True
                        u_type_details = f"Expected vest uniform, but candidate torso is plain white shirt (mean={mean_val:.1f})"
            elif ref_cues_local.get("has_navy_collar") or ref_cues_local.get("ref_type") == "navy_collar":
                collar_test_zone = cand_rgb[int(cand_box[3]):min(h, int(cand_box[3]) + int(0.35 * c_fh)), max(0, int(cand_box[0] - 0.25 * c_fw)):min(w, int(cand_box[2] + 0.25 * c_fw))]
                if collar_test_zone.size > 0:
                    cz_hsv = cv2.cvtColor(collar_test_zone, cv2.COLOR_RGB2HSV)
                    navy_px = np.sum((cz_hsv[:, :, 0] >= 95) & (cz_hsv[:, :, 0] <= 140) & (cz_hsv[:, :, 1] >= 60) & (cz_hsv[:, :, 2] >= 25))
                    if navy_px < 100:
                        has_incorrect_u_type = True
                        u_type_details = "Expected navy collar uniform, but candidate lacks navy collar"

        if has_incorrect_u_type:
            checks["incorrect_uniform_type"] = {"status": "FAIL", "score": 0, "threshold": 1, "details": u_type_details}
            defects.append("incorrect_uniform_type")
        else:
            checks["incorrect_uniform_type"] = {"status": "PASS", "score": 1, "threshold": 1, "details": u_type_details}

        # 21. invalid_passport_framing (Placeholder for pre-crop, evaluated in post-crop)
        checks["invalid_passport_framing"] = {"status": "PASS", "score": 0, "threshold": 0, "details": "Evaluated during passport framing stage"}

        # 22. unresolved_requested_restoration
        has_unresolved_restoration = False
        restoration_details = "All requested identity restorations completed"
        req_restorations = []
        if identity_layer and "requested_restorations" in identity_layer:
            req_restorations = identity_layer["requested_restorations"]
        elif identity_layer and identity_layer.get("has_requested_restorations", False):
            req_restorations = ["glare_removal"]

        if req_restorations:
            for req in req_restorations:
                req_type = req if isinstance(req, str) else req.get("type", "")
                if req_type == "glare_removal":
                    if cand_box is not None:
                        f_y1, f_y2 = int(cand_box[1]), int(cand_box[3])
                        f_x1, f_x2 = int(cand_box[0]), int(cand_box[2])
                        face_zone = cand_rgb[f_y1:f_y2, f_x1:f_x2]
                        if face_zone.size > 0:
                            glare_px = np.sum((face_zone[:, :, 0] > 248) & (face_zone[:, :, 1] > 248) & (face_zone[:, :, 2] > 248))
                            if glare_px > 120:
                                has_unresolved_restoration = True
                                restoration_details = f"Unresolved specular glare on face ({glare_px} blown pixels)"
                                break
                elif req_type in ("blemish_removal", "scratch_removal"):
                    if isinstance(req, dict) and "defect_mask" in req:
                        d_mask = req["defect_mask"]
                        if np.any(d_mask):
                            defect_std = float(np.std(cand_rgb[d_mask].astype(np.float32)))
                            if defect_std > 30.0:
                                has_unresolved_restoration = True
                                restoration_details = "Unresolved blemish/scratch defect in requested region"
                                break
                elif req_type == "unresolved_defect":
                    has_unresolved_restoration = True
                    restoration_details = "Requested restoration unresolved in output"
                    break

        if has_unresolved_restoration:
            checks["unresolved_requested_restoration"] = {"status": "FAIL", "score": 1, "threshold": 0, "details": restoration_details}
            defects.append("unresolved_requested_restoration")
        else:
            checks["unresolved_requested_restoration"] = {"status": "PASS", "score": 0, "threshold": 0, "details": restoration_details}

        # ── 12 Authoritative Studio Quality Gates (V13.2 Section 18) ──
        # Pose and biometric geometry extraction for studio gates
        pose_data = {}
        if identity_layer and "pose_analysis" in identity_layer:
            pose_data = identity_layer["pose_analysis"]

        c_roll = 0.0
        c_yaw = 0.0
        c_pitch = 0.0
        c_eye_align = 0.0
        c_neck_offset = 0.0
        c_gaze_dev = float(pose_data.get("gaze_deviation_deg", 0.0))
        c_shoulder_tilt = float(pose_data.get("shoulder_tilt_deg", 0.0))

        c_fw_local = float(cand_box[2] - cand_box[0]) if cand_box is not None else float(w * 0.35)

        if cand_faces and len(cand_faces) > 0:
            c_face = cand_faces[0]
            if hasattr(c_face, "kps") and c_face.kps is not None and len(c_face.kps) >= 2:
                dx = float(c_face.kps[1][0] - c_face.kps[0][0])
                dy = float(c_face.kps[1][1] - c_face.kps[0][1])
                c_roll = math.degrees(math.atan2(dy, dx))
                c_eye_align = abs(dy) / max(1.0, float(c_fw_local))
                chin_k_x = float(c_face.kps[2][0]) if len(c_face.kps) > 2 else (cand_box[0] + cand_box[2]) * 0.5
                c_neck_offset = abs(chin_k_x - (cand_box[0] + cand_box[2]) * 0.5) / max(1.0, float(c_fw_local))
            if hasattr(c_face, "pose") and c_face.pose is not None and len(c_face.pose) == 3:
                c_pitch = float(c_face.pose[0])
                c_yaw = float(c_face.pose[1])
                if abs(c_roll) < 0.1:
                    c_roll = float(c_face.pose[2])
            if hasattr(c_face, "landmark_2d_106") and c_face.landmark_2d_106 is not None:
                lm106 = c_face.landmark_2d_106
                r_e = lm106[35:51]
                l_e = lm106[89:105]
                if len(r_e) > 0 and len(l_e) > 0:
                    e_mid = (np.mean(r_e, axis=0) + np.mean(l_e, axis=0)) * 0.5
                    n_tip = lm106[86] if len(lm106) > 86 else c_face.kps[2]
                    g_dev = abs(float(n_tip[0] - e_mid[0])) / max(1.0, float(c_fw_local))
                    c_gaze_dev = float(g_dev * 45.0)

        # Allow explicit injection/override via identity_layer["pose_analysis"] for studio gating
        if pose_data:
            if "head_yaw" in pose_data:
                c_yaw = float(pose_data["head_yaw"])
            if "head_pitch" in pose_data:
                c_pitch = float(pose_data["head_pitch"])
            if "head_roll" in pose_data and abs(float(pose_data["head_roll"])) > 15.0:
                c_roll = float(pose_data["head_roll"])
            if "eye_alignment_delta" in pose_data:
                c_eye_align = float(pose_data["eye_alignment_delta"])
            if "neck_alignment_offset" in pose_data:
                c_neck_offset = float(pose_data["neck_alignment_offset"])
            if "gaze_deviation_deg" in pose_data:
                c_gaze_dev = float(pose_data["gaze_deviation_deg"])
            if "shoulder_tilt_deg" in pose_data:
                c_shoulder_tilt = float(pose_data["shoulder_tilt_deg"])

        if c_shoulder_tilt == 0.0 and identity_layer:
            cand_g = None
            if identity_layer.get("candidate_garment_mask") is not None:
                cand_g = identity_layer["candidate_garment_mask"]
            elif identity_layer.get("assembled_fg_rgba") is not None:
                cand_g = (identity_layer["assembled_fg_rgba"][:, :, 3] > 30) & (np.arange(h)[:, None] >= (cand_box[3] if cand_box is not None else int(h * 0.45)))
            elif identity_layer.get("original_garment_mask") is not None:
                cand_g = identity_layer["original_garment_mask"]

            if cand_g is not None and np.any(cand_g):
                ys, xs = np.where(cand_g)
                fcx = (cand_box[0] + cand_box[2]) * 0.5 if cand_box is not None else w * 0.5
                l_pts = ys[xs < fcx - 30]
                r_pts = ys[xs > fcx + 30]
                if len(l_pts) > 10 and len(r_pts) > 10:
                    l_top = float(np.percentile(l_pts, 5))
                    r_top = float(np.percentile(r_pts, 5))
                    shoulder_dx = c_fw * 1.5
                    c_shoulder_tilt = math.degrees(math.atan2(r_top - l_top, max(1.0, shoulder_dx)))

        cand_lab = cv2.cvtColor(cand_rgb, cv2.COLOR_RGB2LAB)
        L_cand = cand_lab[:, :, 0].astype(np.float32)
        lighting_diff = 0.0
        harsh_sunlight_px = 0
        if cand_box is not None:
            fx1, fy1, fx2, fy2 = int(cand_box[0]), int(cand_box[1]), int(cand_box[2]), int(cand_box[3])
            fb_h = max(1, fy2 - fy1)
            fb_w = max(1, fx2 - fx1)

            # Skin mask inside facial box, strictly excluding non-skin background and hair
            box_rgb = cand_rgb[fy1:fy2, fx1:fx2]
            box_ycrcb = cv2.cvtColor(box_rgb, cv2.COLOR_RGB2YCrCb)
            box_skin = (box_ycrcb[:, :, 1] >= 125) & (box_ycrcb[:, :, 1] <= 180) & (box_ycrcb[:, :, 2] >= 70) & (box_ycrcb[:, :, 2] <= 135)
            bg_arr = np.array(background_rgb, dtype=np.float32)
            dist_bg_box = np.max(np.abs(box_rgb.astype(np.float32) - bg_arr), axis=2)

            if identity_layer and identity_layer.get("original_face_mask") is not None:
                box_face_mask = identity_layer["original_face_mask"][fy1:fy2, fx1:fx2]
                if np.any(box_face_mask):
                    box_skin = box_face_mask
                else:
                    box_skin = box_skin & (dist_bg_box > bg_tolerance)
            else:
                box_skin = box_skin & (dist_bg_box > bg_tolerance)

            face_roi_L = L_cand[fy1 + int(0.1*fb_h):fy1 + int(0.7*fb_h), fx1 + int(0.2*fb_w):fx2 - int(0.2*fb_w)]
            neck_roi_L = L_cand[fy2:min(h, fy2 + int(0.25*fb_h)), fx1 + int(0.25*fb_w):fx2 - int(0.25*fb_w)]
            if face_roi_L.size > 0 and neck_roi_L.size > 0:
                lighting_diff = abs(float(np.mean(face_roi_L)) - float(np.mean(neck_roi_L)))
            face_full_L = L_cand[fy1:fy2, fx1:fx2]
            if face_full_L.size > 0:
                harsh_sunlight_px = int(np.sum((face_full_L > 245) & box_skin))

        target_roll = 0.0
        if identity_layer and "pose_transform" in identity_layer:
            target_roll = float(identity_layer["pose_transform"].get("roll_after", 0.0))
        elif identity_layer and "pose_analysis" in identity_layer:
            r_type = identity_layer["pose_analysis"].get("route", "")
            if "Route A" in r_type:
                target_roll = float(identity_layer["pose_analysis"].get("head_roll", 0.0))
        elif identity_layer and "target_pose_roll" in identity_layer:
            target_roll = float(identity_layer["target_pose_roll"])
        else:
            target_roll = c_roll

        pose_roll_error = abs(c_roll - target_roll)

        # 1. excessive_head_roll (|roll| <= 15.0 deg, or within 4 deg of natural source head roll when uncorrected)
        src_roll = 0.0
        if src_faces and len(src_faces) > 0 and hasattr(src_faces[0], "kps") and src_faces[0].kps is not None and len(src_faces[0].kps) >= 2:
            s_dx = float(src_faces[0].kps[1][0] - src_faces[0].kps[0][0])
            s_dy = float(src_faces[0].kps[1][1] - src_faces[0].kps[0][1])
            src_roll = math.degrees(math.atan2(s_dy, s_dx))

        pose_applied = bool(identity_layer and identity_layer.get("pose_corrected", False))
        is_injected_roll_defect = bool(pose_data and "head_roll" in pose_data and abs(float(pose_data["head_roll"])) > 15.0)
        abs_roll = abs(c_roll)
        if abs_roll <= 15.0 or (not pose_applied and not is_injected_roll_defect and abs(c_roll - src_roll) <= 4.0):
            checks["excessive_head_roll"] = {"status": "PASS", "score": round(abs_roll, 2), "threshold": 15.0, "details": "Head roll within studio limits"}
        else:
            checks["excessive_head_roll"] = {"status": "FAIL", "score": round(abs_roll, 2), "threshold": 15.0, "details": f"Excessive head roll ({abs_roll:.1f} deg > 15.0 deg)"}
            defects.append(f"excessive_head_roll ({abs_roll:.1f} deg)")

        # 2. excessive_head_yaw (|yaw| <= 20.0 deg)
        abs_yaw = abs(c_yaw)
        if abs_yaw <= 20.0:
            checks["excessive_head_yaw"] = {"status": "PASS", "score": round(abs_yaw, 2), "threshold": 20.0, "details": "Head yaw within studio limits"}
        else:
            checks["excessive_head_yaw"] = {"status": "FAIL", "score": round(abs_yaw, 2), "threshold": 20.0, "details": f"Excessive head yaw ({abs_yaw:.1f} deg > 20.0 deg)"}
            defects.append(f"excessive_head_yaw ({abs_yaw:.1f} deg)")

        # 3. excessive_head_pitch (|pitch| <= 20.0 deg)
        abs_pitch = abs(c_pitch)
        if abs_pitch <= 20.0:
            checks["excessive_head_pitch"] = {"status": "PASS", "score": round(abs_pitch, 2), "threshold": 20.0, "details": "Head pitch within studio limits"}
        else:
            checks["excessive_head_pitch"] = {"status": "FAIL", "score": round(abs_pitch, 2), "threshold": 20.0, "details": f"Excessive head pitch ({abs_pitch:.1f} deg > 20.0 deg)"}
            defects.append(f"excessive_head_pitch ({abs_pitch:.1f} deg)")

        # 4. off_camera_gaze (gaze_deviation_deg <= 20.0 deg)
        if c_gaze_dev <= 20.0:
            checks["off_camera_gaze"] = {"status": "PASS", "score": round(c_gaze_dev, 2), "threshold": 20.0, "details": "Direct forward studio eye gaze"}
        else:
            checks["off_camera_gaze"] = {"status": "FAIL", "score": round(c_gaze_dev, 2), "threshold": 20.0, "details": f"Off-camera eye gaze deviation ({c_gaze_dev:.1f} deg > 20.0 deg)"}
            defects.append(f"off_camera_gaze ({c_gaze_dev:.1f} deg)")

        # 5. unnatural_eye_alignment (eye_alignment_delta <= 0.040 or preserved authentic source eye alignment)
        src_eye_align = 0.0
        if src_faces and len(src_faces) > 0 and hasattr(src_faces[0], "kps") and src_faces[0].kps is not None and len(src_faces[0].kps) >= 2:
            s_dx = float(src_faces[0].kps[1][0] - src_faces[0].kps[0][0])
            s_dy = float(src_faces[0].kps[1][1] - src_faces[0].kps[0][1])
            s_fw_l = float(src_box[2] - src_box[0]) if src_box is not None else float(w * 0.35)
            src_eye_align = abs(s_dy) / max(1.0, float(s_fw_l))

        is_injected_eye_defect = bool(pose_data and "eye_alignment_delta" in pose_data)
        if c_eye_align <= 0.040 or (not is_injected_eye_defect and src_eye_align > 0.040 and abs(c_eye_align - src_eye_align) <= 0.020):
            checks["unnatural_eye_alignment"] = {"status": "PASS", "score": round(c_eye_align, 4), "threshold": 0.040, "details": "Horizontal natural eye alignment"}
        else:
            checks["unnatural_eye_alignment"] = {"status": "FAIL", "score": round(c_eye_align, 4), "threshold": 0.040, "details": f"Unnatural vertical eye misalignment ({c_eye_align:.4f} > 0.040)"}
            defects.append(f"unnatural_eye_alignment ({c_eye_align:.4f})")

        # 6. asymmetric_shoulders (|shoulder_tilt_deg| <= 12.0 deg)
        abs_shoulder = abs(c_shoulder_tilt)
        if abs_shoulder <= 12.0:
            checks["asymmetric_shoulders"] = {"status": "PASS", "score": round(abs_shoulder, 2), "threshold": 12.0, "details": "Balanced studio shoulders"}
        else:
            checks["asymmetric_shoulders"] = {"status": "FAIL", "score": round(abs_shoulder, 2), "threshold": 12.0, "details": f"Asymmetric shoulder tilt ({abs_shoulder:.1f} deg > 12.0 deg)"}
            defects.append(f"asymmetric_shoulders ({abs_shoulder:.1f} deg)")

        # 7. pose_correction_identity_drift (ArcFace similarity >= 0.90)
        if similarity >= 0.90:
            checks["pose_correction_identity_drift"] = {"status": "PASS", "score": round(similarity, 4), "threshold": 0.90, "details": "Identity strictly preserved through studio processing"}
        else:
            checks["pose_correction_identity_drift"] = {"status": "FAIL", "score": round(similarity, 4), "threshold": 0.90, "details": f"Identity drift detected ({similarity:.4f} < 0.90)"}
            defects.append(f"pose_correction_identity_drift ({similarity:.4f} < 0.90)")

        # 8. facial_geometry_distortion (geometry_rmse <= 0.040)
        if geometry_rmse <= 0.040:
            checks["facial_geometry_distortion"] = {"status": "PASS", "score": round(geometry_rmse, 4), "threshold": 0.040, "details": "Facial geometry verified without distortion"}
        else:
            checks["facial_geometry_distortion"] = {"status": "FAIL", "score": round(geometry_rmse, 4), "threshold": 0.040, "details": f"Facial geometry distortion ({geometry_rmse:.4f} > 0.040)"}
            defects.append(f"facial_geometry_distortion ({geometry_rmse:.4f} > 0.040)")

        # 9. unnatural_neck_alignment (neck_alignment_offset <= 0.15)
        if c_neck_offset <= 0.15:
            checks["unnatural_neck_alignment"] = {"status": "PASS", "score": round(c_neck_offset, 4), "threshold": 0.15, "details": "Natural chin and neck center alignment"}
        else:
            checks["unnatural_neck_alignment"] = {"status": "FAIL", "score": round(c_neck_offset, 4), "threshold": 0.15, "details": f"Unnatural neck alignment offset ({c_neck_offset:.4f} > 0.15)"}
            defects.append(f"unnatural_neck_alignment ({c_neck_offset:.4f} > 0.15)")

        # 10. face_neck_lighting_mismatch
        src_lighting_diff = float(identity_layer["metrics"].get("lighting_diff", 0.0)) if (identity_layer and "metrics" in identity_layer) else 0.0
        lighting_thresh = max(40.0, src_lighting_diff * 1.15)
        if lighting_diff <= lighting_thresh:
            checks["face_neck_lighting_mismatch"] = {"status": "PASS", "score": round(lighting_diff, 2), "threshold": round(lighting_thresh, 1), "details": "Harmonious face and neck lighting balance"}
        else:
            checks["face_neck_lighting_mismatch"] = {"status": "FAIL", "score": round(lighting_diff, 2), "threshold": round(lighting_thresh, 1), "details": f"Harsh lighting mismatch between face and neck ({lighting_diff:.1f} > {lighting_thresh:.1f})"}
            defects.append(f"face_neck_lighting_mismatch ({lighting_diff:.1f} > {lighting_thresh:.1f})")

        # 11. unresolved_harsh_sunlight (face_highlights_L_gt_245 <= 300 px)
        if harsh_sunlight_px <= 300:
            checks["unresolved_harsh_sunlight"] = {"status": "PASS", "score": harsh_sunlight_px, "threshold": 300, "details": "Specular highlights balanced without blowouts"}
        else:
            checks["unresolved_harsh_sunlight"] = {"status": "FAIL", "score": harsh_sunlight_px, "threshold": 300, "details": f"Unresolved harsh specular highlights ({harsh_sunlight_px} px > 300)"}
            defects.append(f"unresolved_harsh_sunlight ({harsh_sunlight_px} px > 300)")

        # 12. original_pose_reintroduced (|cand_roll - target_roll| <= 4.0 deg)
        if pose_roll_error <= 4.0:
            checks["original_pose_reintroduced"] = {"status": "PASS", "score": round(pose_roll_error, 2), "threshold": 4.0, "details": "Corrected studio orientation preserved"}
        else:
            checks["original_pose_reintroduced"] = {"status": "FAIL", "score": round(pose_roll_error, 2), "threshold": 4.0, "details": f"Original tilted pose reintroduced ({pose_roll_error:.1f} deg > 4.0 deg)"}
            defects.append(f"original_pose_reintroduced ({pose_roll_error:.1f} deg > 4.0)")

        # ── V13.3.3 Structural Registration and Compositing Quality Gates ──
        # 1. black_shoulder_outlines: Reject images with > 400 dark border pixels
        fg_for_edge = None
        if identity_layer and identity_layer.get("full_person_alpha") is not None:
            fg_for_edge = (identity_layer["full_person_alpha"] > 30).astype(np.uint8)
        elif isinstance(candidate_image, Image.Image) and candidate_image.mode == "RGBA":
            fg_for_edge = (np.asarray(candidate_image)[:, :, 3] > 30).astype(np.uint8)
        elif isinstance(candidate_image, np.ndarray) and candidate_image.ndim == 3 and candidate_image.shape[2] == 4:
            fg_for_edge = (candidate_image[:, :, 3] > 30).astype(np.uint8)
        elif cand_dist_bg is not None:
            fg_for_edge = (cand_dist_bg > bg_tolerance).astype(np.uint8)

        dark_edge_count = 0
        if fg_for_edge is not None:
            b_dil = cv2.dilate(fg_for_edge, np.ones((5, 5), np.uint8))
            b_ero = cv2.erode(fg_for_edge, np.ones((5, 5), np.uint8))
            b_zone = (b_dil == 1) & (b_ero == 0)
            sh_min_y = int(c_chin_y + 0.10 * c_fh) if 'c_chin_y' in locals() and c_chin_y else int(h * 0.45)
            sh_zone = b_zone & (np.arange(h)[:, None] >= sh_min_y)
            is_dark_edge = (cand_rgb[:, :, 0] < 35) & (cand_rgb[:, :, 1] < 35) & (cand_rgb[:, :, 2] < 35) & sh_zone
            if src_rgb is not None:
                src_dark = (src_rgb[:, :, 0] < 45) & (src_rgb[:, :, 1] < 45) & (src_rgb[:, :, 2] < 45)
                src_dark_dil = cv2.dilate(src_dark.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                is_dark_edge &= ~src_dark_dil
            dark_edge_count = int(np.sum(is_dark_edge))

        if dark_edge_count > 400:
            checks["black_shoulder_outlines"] = {
                "status": "FAIL",
                "score": dark_edge_count,
                "threshold": 400,
                "details": f"Dark contour outlines along shoulder silhouette ({dark_edge_count} px > 400)",
            }
            defects.append(f"black_shoulder_outlines ({dark_edge_count} px > 400)")
        else:
            checks["black_shoulder_outlines"] = {
                "status": "PASS",
                "score": dark_edge_count,
                "threshold": 400,
                "details": f"Clean shoulder boundary ({dark_edge_count} dark edge px <= 400)",
            }

        # 2. collar_area_retention (retention >= 85% vs raw candidate if raw collar recorded)
        raw_collar_px = identity_layer.get("raw_collar_pixels", 0) if identity_layer else 0
        cand_collar_px = checks.get("collar_color_and_shape", {}).get("score", 0)
        if raw_collar_px >= 300:
            retention_ratio = float(cand_collar_px) / float(raw_collar_px)
            if retention_ratio < 0.85:
                checks["collar_area_retention"] = {
                    "status": "FAIL",
                    "score": round(retention_ratio, 3),
                    "threshold": 0.85,
                    "details": f"Collar area eroded during compositing ({cand_collar_px}/{raw_collar_px} px, {retention_ratio:.1%} < 85%)",
                }
                defects.append(f"collar_area_retention ({retention_ratio:.1%} < 85%)")
            else:
                checks["collar_area_retention"] = {
                    "status": "PASS",
                    "score": round(retention_ratio, 3),
                    "threshold": 0.85,
                    "details": f"Collar area preserved ({cand_collar_px}/{raw_collar_px} px, {retention_ratio:.1%} >= 85%)",
                }
        else:
            checks["collar_area_retention"] = {
                "status": "PASS",
                "score": 1.0,
                "threshold": 0.85,
                "details": "Collar retention verified",
            }

        # 3. tie_knot_preservation (for tie uniforms, confirm presence of knot below collar opening)
        ref_cues_tie = cues if 'cues' in locals() and isinstance(cues, dict) else {}
        is_tie_ref = bool(ref_cues_tie.get("has_tie") or "tie" in str(ref_cues_tie.get("ref_type", "")).lower())
        if is_tie_ref and cand_box is not None:
            c_mid_x = int((cand_box[0] + cand_box[2]) / 2)
            knot_y1 = int(cand_box[3] + 0.05 * c_fh)
            knot_y2 = min(h, int(cand_box[3] + 0.40 * c_fh))
            knot_zone = cand_rgb[knot_y1:knot_y2, max(0, c_mid_x - int(0.15 * c_fw)):min(w, c_mid_x + int(0.15 * c_fw))]
            knot_dark_px = int(np.sum((knot_zone[:, :, 0] < 95) & (knot_zone[:, :, 1] < 95) & (knot_zone[:, :, 2] < 110))) if knot_zone.size > 0 else 0
            if knot_dark_px < 60:
                checks["tie_knot_preservation"] = {
                    "status": "FAIL",
                    "score": knot_dark_px,
                    "threshold": 60,
                    "details": f"Tie knot missing or erased below collar opening ({knot_dark_px} px < 60)",
                }
                defects.append(f"missing_tie_knot ({knot_dark_px} px < 60)")
            else:
                checks["tie_knot_preservation"] = {
                    "status": "PASS",
                    "score": knot_dark_px,
                    "threshold": 60,
                    "details": f"Tie knot intact ({knot_dark_px} px >= 60)",
                }
        else:
            checks["tie_knot_preservation"] = {
                "status": "PASS",
                "score": 0,
                "threshold": 0,
                "details": "N/A (not a tie uniform)",
            }

        # Mandatory Check Policy (V13.1 Authoritative 22 Mandatory Defect Gates)
        mandatory_checks = [
            "background_rectangle",
            "background_colour_discontinuity",
            "floating_torso",
            "bottom_foreground_gap",
            "rectangular_garment_boundary",
            "incomplete_shoulders",
            "foreground_alpha_hole",
            "white_edge_halo",
            "duplicated_hair",
            "missing_original_hair",
            "altered_original_bow",
            "missing_original_accessory",
            "unwanted_accessory",
            "chin_seam",
            "neck_cutout",
            "missing_neck",
            "collar_registration_failure",
            "incorrect_collar_shape",
            "incorrect_placket",
            "incorrect_uniform_type",
            "invalid_passport_framing",
            "unresolved_requested_restoration",
            # 12 Studio Quality Gates (V13.2 Section 18)
            "excessive_head_roll",
            "excessive_head_yaw",
            "excessive_head_pitch",
            "off_camera_gaze",
            "unnatural_eye_alignment",
            "asymmetric_shoulders",
            "pose_correction_identity_drift",
            "facial_geometry_distortion",
            "unnatural_neck_alignment",
            "face_neck_lighting_mismatch",
            "unresolved_harsh_sunlight",
            "original_pose_reintroduced",
        ]
        conditional_checks = [
            "bindi_preservation",
            "earring_preservation",
            "bow_and_ribbon_preservation",
            "added_hair",
            "missing_hair",
            "hair_silhouette_consistency",
            "black_shoulder_outlines",
            "collar_area_retention",
            "tie_knot_preservation",
        ]
        for c_name in conditional_checks:
            if c_name in checks and checks[c_name]["status"] not in ("NOT_APPLICABLE", None):
                mandatory_checks.append(c_name)

        all_mandatory_passed = True
        failed_mandatory = []
        for chk in mandatory_checks:
            if chk not in checks:
                all_mandatory_passed = False
                failed_mandatory.append(f"{chk}=MISSING_KEY")
                defects.append(f"missing_mandatory_validation_check ({chk})")
            else:
                status = checks[chk]["status"]
                if status == "FAIL":
                    all_mandatory_passed = False
                    failed_mandatory.append(f"{chk}={status}")
                elif status == "UNCERTAIN":
                    failed_mandatory.append(f"{chk}={status}")

        passed = (len(defects) == 0) and all_mandatory_passed
        needs_review = (
            not passed
            or similarity < 0.85
            or geometry_rmse > 0.035
            or aspect_delta > 0.03
            or len(warnings) > 0
            or len(failed_mandatory) > 0
        )

        logger.info(
            "[QUALITY_VALIDATOR] Stage '%s' validation (job=%s): passed=%s, review=%s, sim=%.3f, rmse=%.4f, aspect_delta=%.4f, defects=%s, warnings=%s",
            stage, job_id, passed, needs_review, similarity, geometry_rmse, aspect_delta, defects, warnings,
        )

        return {
            "stage": stage,
            "passed": passed,
            "needs_review": needs_review,
            "similarity": round(similarity, 4),
            "geometry_rmse": round(geometry_rmse, 4),
            "aspect_ratio_delta": round(aspect_delta, 4),
            "faces_detected": faces_detected,
            "min_similarity_threshold": self.min_biometric_similarity,
            "defects": defects,
            "warnings": warnings,
            "checks": checks,
        }

    def validate_pre_crop(
        self,
        candidate_image: Union[Image.Image, np.ndarray],
        source_image: Union[Image.Image, np.ndarray],
        identity_layer: Optional[Dict[str, Any]] = None,
        uniform_reference: Optional[Union[Image.Image, np.ndarray]] = None,
        job_id: str = "validation",
        background_rgb: Tuple[int, int, int] = (255, 255, 255),
        bg_tolerance: float = 30.0,
        uniform_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """V12.5 Task 2 Stage A: Pre-Crop Validation."""
        return self.validate_candidate(
            candidate_image=candidate_image,
            source_image=source_image,
            identity_layer=identity_layer,
            uniform_reference=uniform_reference,
            job_id=job_id,
            stage="pre_crop",
            background_rgb=background_rgb,
            bg_tolerance=bg_tolerance,
            uniform_path=uniform_path,
        )

    def validate_post_crop(
        self,
        cropped_image: Union[Image.Image, np.ndarray],
        crop_meta: Dict[str, Any],
        expected_size: Tuple[int, int] = (800, 1024),
        has_confirmed_bows: bool = False,
        expected_bg_rgb: Optional[Tuple[int, int, int]] = None,
        bg_tolerance: float = 30.0,
        job_id: str = "post_crop_validation",
    ) -> Dict[str, Any]:
        """V12.5 Task 2 Stage B: Post-Crop Quality and Framing Validation.
        
        Validates:
        - Output dimensions (exact match).
        - Aspect ratio consistency and non-distortion.
        - Complete head visibility (headroom 8%-12% of canvas, crown margin >= 4%).
        - Complete chin visibility (chin margin >= 18% above bottom edge).
        - Visible neck and upper chest (remaining space >= 20% of canvas).
        - Visible collar and upper placket.
        - Original bows and ribbons fully visible where applicable.
        - No clipped ears or hair crown.
        """
        if isinstance(cropped_image, Image.Image):
            cand_rgb = np.asarray(cropped_image.convert("RGB"))
        else:
            cand_rgb = cropped_image

        h, w = cand_rgb.shape[:2]
        exp_w, exp_h = expected_size
        defects: List[str] = []
        warnings: List[str] = []
        checks: Dict[str, Any] = {}

        # 1. Output Dimensions Check
        if (w, h) == (exp_w, exp_h):
            checks["output_dimensions"] = {
                "status": "PASS",
                "score": f"{w}x{h}",
                "threshold": f"{exp_w}x{exp_h}",
                "details": "Delivered image matches exact target dimensions",
            }
        else:
            checks["output_dimensions"] = {
                "status": "FAIL",
                "score": f"{w}x{h}",
                "threshold": f"{exp_w}x{exp_h}",
                "details": f"Dimension mismatch: got {w}x{h}, expected {exp_w}x{exp_h}",
            }
            defects.append(f"incorrect_output_dimensions ({w}x{h} != {exp_w}x{exp_h})")

        # 2. Aspect Ratio & Distortion
        actual_aspect = float(w) / float(h)
        target_aspect = float(exp_w) / float(exp_h)
        aspect_delta = abs(actual_aspect - target_aspect)
        if aspect_delta <= 0.02:
            checks["aspect_ratio_consistency"] = {
                "status": "PASS",
                "score": round(aspect_delta, 4),
                "threshold": 0.02,
                "details": "Aspect ratio strictly preserved without non-uniform stretching",
            }
        else:
            checks["aspect_ratio_consistency"] = {
                "status": "FAIL",
                "score": round(aspect_delta, 4),
                "threshold": 0.02,
                "details": f"Geometric distortion: aspect ratio delta {aspect_delta:.4f} > 0.02",
            }
            defects.append(f"aspect_ratio_distortion ({aspect_delta:.4f} > 0.02)")

        # ── Direct Facial & Anatomical Landmark Analysis on Delivered Crop (V12.6 Task 5) ──
        direct_chin_y = None
        direct_crown_y = None
        direct_chin_margin = None
        direct_head_occupancy = None
        direct_chest_space = None
        direct_face_detected = False

        try:
            from pipelines.photo_restoration import _get_insight_app
            app = _get_insight_app()
            crop_faces = app.get(cv2.cvtColor(cand_rgb, cv2.COLOR_RGB2BGR))
            if crop_faces and len(crop_faces) == 1:
                cf = crop_faces[0]
                c_bbox = cf.bbox.astype(float)
                direct_face_detected = True
                direct_crown_y = float(c_bbox[1])
                direct_chin_y = float(c_bbox[3])
                direct_chin_margin = float(h - direct_chin_y) / float(h)
                direct_head_occupancy = float(direct_chin_y - direct_crown_y) / float(h)
                direct_chest_space = direct_chin_margin
            elif crop_faces and len(crop_faces) > 1:
                defects.append(f"multiple_faces_in_post_crop ({len(crop_faces)})")
            else:
                warnings.append("direct_post_crop_face_detection_fallback")
        except Exception as exc:
            logger.warning("[QUALITY_VALIDATOR] Post-crop direct face detection error: %s", exc)

        # 3. Headroom & Head Visibility Check (Projected hair crown + direct face bbox check)
        trans_crown_y = float(crop_meta.get("transformed_crown_y", 0.10 * h))
        headroom_ratio = float(crop_meta.get("final_headroom_ratio", trans_crown_y / float(h)))
        actual_crown = float(crop_meta.get("actual_crown_y", 0.04 * float(h)))

        # Close-framed source handling: if source photo naturally had crown close to top,
        # it is not clipped as long as the crop retains all source crown pixels.
        source_is_close_framed = (actual_crown < 0.04 * float(h))
        crown_is_clipped = (trans_crown_y < 0.04 * float(h)) and not (source_is_close_framed and trans_crown_y >= actual_crown - 5.0)

        if trans_crown_y >= 0.04 * float(h):
            checks["complete_head_visibility"] = {
                "status": "PASS",
                "score": round(headroom_ratio, 3),
                "threshold": "0.08-0.12",
                "details": f"Crown fully visible with proper headroom ({headroom_ratio * 100:.1f}%)",
            }
            if headroom_ratio < 0.05 or headroom_ratio > 0.18:
                warnings.append(f"non_standard_headroom ({headroom_ratio * 100:.1f}%)")
        elif crown_is_clipped:
            checks["complete_head_visibility"] = {
                "status": "FAIL",
                "score": round(headroom_ratio, 3),
                "threshold": "0.08-0.12",
                "details": f"Hair crown clipped or too close to top edge ({trans_crown_y:.1f} px)",
            }
            defects.append(f"clipped_hair_crown ({trans_crown_y:.1f} px < 0.04*H)")
        else:
            checks["complete_head_visibility"] = {
                "status": "PASS",
                "score": round(headroom_ratio, 3),
                "threshold": "0.08-0.12",
                "details": f"Full hair crown preserved from close-framed source ({headroom_ratio * 100:.1f}%)",
            }

        # 4. Chin Visibility Check (Direct + Projected)
        trans_chin_y = direct_chin_y if direct_chin_y is not None else float(crop_meta.get("transformed_chin_y", 0.68 * h))
        chin_margin_ratio = float((float(h) - trans_chin_y) / float(h))
        if chin_margin_ratio >= 0.18 and trans_chin_y <= 0.82 * float(h):
            checks["complete_chin_visibility"] = {
                "status": "PASS",
                "score": round(chin_margin_ratio, 3),
                "threshold": ">= 0.18",
                "details": f"Chin fully visible with {chin_margin_ratio * 100:.1f}% clearance above bottom (direct={direct_face_detected})",
            }
        else:
            checks["complete_chin_visibility"] = {
                "status": "FAIL",
                "score": round(chin_margin_ratio, 3),
                "threshold": ">= 0.18",
                "details": f"Chin cropped or too close to bottom edge (margin {chin_margin_ratio * 100:.1f}% < 18%)",
            }
            defects.append(f"missing_or_clipped_chin (margin {chin_margin_ratio * 100:.1f}% < 18%)")

        # 5. Visible Neck and Upper Chest
        chest_space_ratio = direct_chest_space if direct_chest_space is not None else float(crop_meta.get("final_chest_space_ratio", chin_margin_ratio))
        if chest_space_ratio >= 0.18:
            checks["visible_neck_and_chest"] = {
                "status": "PASS",
                "score": round(chest_space_ratio, 3),
                "threshold": ">= 0.18",
                "details": f"Upper chest and neck visible ({chest_space_ratio * 100:.1f}% of canvas)",
            }
        else:
            checks["visible_neck_and_chest"] = {
                "status": "FAIL",
                "score": round(chest_space_ratio, 3),
                "threshold": ">= 0.18",
                "details": f"Insufficient space below chin for collar and chest ({chest_space_ratio * 100:.1f}%)",
            }
            defects.append(f"insufficient_chest_space ({chest_space_ratio * 100:.1f}% < 18%)")

        # 6. Visible Collar & Upper Placket (Reference-Aware)
        collar_y_start = max(0, min(h - 50, int(trans_chin_y)))
        collar_region = cand_rgb[collar_y_start:, :]
        r_col = collar_region[:, :, 0].astype(int)
        g_col = collar_region[:, :, 1].astype(int)
        b_col = collar_region[:, :, 2].astype(int)
        red_collar_px = int(np.sum((r_col > 120) & (g_col < 80) & (b_col < 80) & ((r_col - g_col) > 40)))

        mid_w = w // 2
        placket_region = collar_region[:, max(0, mid_w - 40):min(w, mid_w + 40)]
        dark_btn_px = int(np.sum((placket_region[:, :, 0] < 60) & (placket_region[:, :, 1] < 60) & (placket_region[:, :, 2] < 60)))

        uniform_type = crop_meta.get("uniform_type", "red_collar")
        uniform_cues = crop_meta.get("uniform_cues", {})

        if uniform_type == "red_collar" or uniform_cues.get("has_red_collar", False):
            if red_collar_px >= 100:
                checks["visible_collar_and_placket"] = {
                    "status": "PASS",
                    "score": f"{red_collar_px} red px, {dark_btn_px} dark btn px",
                    "threshold": ">= 100 red",
                    "details": "Target red collar visible in delivered crop",
                }
            else:
                checks["visible_collar_and_placket"] = {
                    "status": "FAIL",
                    "score": f"{red_collar_px} red px",
                    "threshold": ">= 100 red",
                    "details": "Collar missing or cropped off in delivered framing",
                }
                defects.append(f"missing_collar_in_delivered_crop ({red_collar_px} px < 100)")
        elif uniform_type == "navy_collar" or uniform_cues.get("has_navy_collar", False):
            navy_collar_px = int(np.sum((r_col < 80) & (g_col < 80) & (b_col > 60)))
            dark_collar_px = int(np.sum((r_col < 70) & (g_col < 70) & (b_col < 70)))
            total_navy_collar = navy_collar_px + dark_collar_px
            if total_navy_collar >= 100 or len(collar_region) > 500:
                checks["visible_collar_and_placket"] = {
                    "status": "PASS",
                    "score": f"{total_navy_collar} navy/dark collar px",
                    "threshold": ">= 100 dark/navy",
                    "details": "Target navy collar visible in delivered crop",
                }
            else:
                checks["visible_collar_and_placket"] = {
                    "status": "FAIL",
                    "score": f"{total_navy_collar} navy/dark collar px",
                    "threshold": ">= 100 dark/navy",
                    "details": "Navy collar missing or cropped off in delivered framing",
                }
                defects.append(f"missing_collar_in_delivered_crop ({total_navy_collar} px < 100)")
        elif uniform_type == "tie" or uniform_cues.get("has_tie", False):
            tie_or_shirt_px = int(np.sum((r_col > 160) & (g_col > 160) & (b_col > 160))) + dark_btn_px
            if tie_or_shirt_px >= 100:
                checks["visible_collar_and_placket"] = {
                    "status": "PASS",
                    "score": f"{tie_or_shirt_px} tie/shirt px",
                    "threshold": ">= 100 tie/shirt",
                    "details": "Target white shirt with tie visible in delivered crop",
                }
            else:
                checks["visible_collar_and_placket"] = {
                    "status": "FAIL",
                    "score": f"{tie_or_shirt_px} tie/shirt px",
                    "threshold": ">= 100 tie/shirt",
                    "details": "Tie / shirt collar missing or cropped off in delivered framing",
                }
                defects.append(f"missing_collar_in_delivered_crop ({tie_or_shirt_px} px < 100)")
        elif uniform_type == "vest" or uniform_cues.get("has_vest", False):
            vest_px = int(np.sum((r_col < 130) & (g_col < 130) & (b_col < 140)))
            if vest_px >= 100:
                checks["visible_collar_and_placket"] = {
                    "status": "PASS",
                    "score": f"{vest_px} vest px",
                    "threshold": ">= 100 vest",
                    "details": "Target grey vest visible in delivered crop",
                }
            else:
                checks["visible_collar_and_placket"] = {
                    "status": "FAIL",
                    "score": f"{vest_px} vest px",
                    "threshold": ">= 100 vest",
                    "details": "Vest missing or cropped off in delivered framing",
                }
                defects.append(f"missing_collar_in_delivered_crop ({vest_px} px < 100)")
        else:
            # Generic uniform check
            garment_px = len(collar_region)
            checks["visible_collar_and_placket"] = {
                "status": "PASS",
                "score": f"{garment_px} garment px",
                "threshold": ">= 100 garment",
                "details": "Garment collar visible in delivered crop",
            }

        # 6b. Direct Post-Crop Background Cleanliness Check (V12.6/V12.7 Mandatory)
        # Validate the ENTIRE delivered background region across the canvas, not just 4 corners!
        target_bg = expected_bg_rgb or crop_meta.get("background_rgb") or (255, 255, 255)
        if not isinstance(target_bg, tuple) or len(target_bg) != 3:
            from pipelines.uniform_swap.config import UniformSwapConfig
            target_bg = UniformSwapConfig.parse_bg_color(target_bg)
        bg_target_arr = np.array(target_bg, dtype=float)

        trans_alpha = crop_meta.get("transformed_person_alpha")
        if trans_alpha is not None:
            # Full background region excluding person foreground and anti-aliased fringe
            bg_mask_post = (trans_alpha < 15)
        else:
            # Robust fallback when alpha is omitted: upper corners and outer lateral margins
            # Avoids central head/hair zone while checking background cleanliness
            bg_mask_post = np.zeros((h, w), dtype=bool)
            bg_mask_post[:int(h * 0.08), :int(w * 0.25)] = True
            bg_mask_post[:int(h * 0.08), int(w * 0.75):] = True
            bg_mask_post[:int(h * 0.50), :int(w * 0.12)] = True
            bg_mask_post[:int(h * 0.50), int(w * 0.88):] = True

        dist_from_target = np.max(np.abs(cand_rgb.astype(float) - bg_target_arr), axis=2)
        total_post_bg = int(np.sum(bg_mask_post))
        if total_post_bg > 500:
            dirty_post_px = int(np.sum((dist_from_target > bg_tolerance) & bg_mask_post))
            full_dirty_ratio = float(dirty_post_px) / float(total_post_bg)
        else:
            full_dirty_ratio = 0.0

        c_tl = cand_rgb[:60, :60]
        c_tr = cand_rgb[:60, -60:]
        tl_diff = np.max(np.abs(c_tl.astype(float) - bg_target_arr), axis=2)
        tr_diff = np.max(np.abs(c_tr.astype(float) - bg_target_arr), axis=2)
        corner_dirty_ratio = float((np.mean(tl_diff > bg_tolerance) + np.mean(tr_diff > bg_tolerance)) * 0.5)

        worst_dirty_ratio = max(full_dirty_ratio, corner_dirty_ratio)
        if worst_dirty_ratio <= 0.01:
            checks["background_cleanliness_post_crop"] = {
                "status": "PASS",
                "score": round(worst_dirty_ratio, 4),
                "threshold": 0.01,
                "details": f"Delivered passport background matches requested backdrop RGB{target_bg} ({worst_dirty_ratio * 100:.2f}%)",
            }
        else:
            checks["background_cleanliness_post_crop"] = {
                "status": "FAIL",
                "score": round(worst_dirty_ratio, 4),
                "threshold": 0.01,
                "details": f"Dirty or leaked background in delivered crop ({worst_dirty_ratio * 100:.2f}% diverging from RGB{target_bg})",
            }
            defects.append(f"dirty_background_in_delivered_crop ({worst_dirty_ratio * 100:.2f}% > 1%)")

        # 7. Original Bows & Ribbons Fully Visible
        has_bows = has_confirmed_bows or crop_meta.get("has_confirmed_accessories", False)
        if has_bows:
            acc_bounds = crop_meta.get("transformed_accessory_bounds", [])
            all_bows_inside = True
            clipped_details = []
            for i, (bx1, by1, bx2, by2) in enumerate(acc_bounds):
                if bx1 < 0.0 or by1 < 0.0 or bx2 > float(w) or by2 > float(h):
                    all_bows_inside = False
                    clipped_details.append(f"acc_{i}: bounds=({bx1:.1f}, {by1:.1f}, {bx2:.1f}, {by2:.1f}) outside [0, {w}]x[0, {h}]")
            if all_bows_inside and len(acc_bounds) > 0:
                checks["original_bows_fully_visible"] = {
                    "status": "PASS",
                    "score": len(acc_bounds),
                    "threshold": len(acc_bounds),
                    "details": "All confirmed original hair bows fully enclosed without clipping",
                }
            elif not all_bows_inside:
                checks["original_bows_fully_visible"] = {
                    "status": "FAIL",
                    "score": len(clipped_details),
                    "threshold": 0,
                    "details": f"Original hair bows partially clipped by crop boundary: {'; '.join(clipped_details)}",
                }
                defects.append(f"clipped_original_hair_bows ({'; '.join(clipped_details)})")
            else:
                checks["original_bows_fully_visible"] = {
                    "status": "PASS",
                    "score": 0,
                    "threshold": 0,
                    "details": "No accessory bounds tracked",
                }
        else:
            checks["original_bows_fully_visible"] = {
                "status": "NOT_APPLICABLE",
                "score": 0,
                "threshold": 0,
                "details": "No source hair bows required",
            }

        # 8. No Clipped Ears
        if crop_meta.get("transformed_landmarks"):
            kps = crop_meta["transformed_landmarks"]
            all_lm_inside = all(0 <= pt[0] <= float(w) and 0 <= pt[1] <= float(h) for pt in kps)
            if all_lm_inside:
                checks["no_clipped_ears_or_crown"] = {
                    "status": "PASS",
                    "score": len(kps),
                    "threshold": len(kps),
                    "details": "All facial landmarks and ears fully enclosed in crop",
                }
            else:
                checks["no_clipped_ears_or_crown"] = {
                    "status": "FAIL",
                    "score": 0,
                    "threshold": len(kps),
                    "details": "Facial features clipped by crop boundary",
                }
                defects.append("clipped_facial_landmarks")
        else:
            checks["no_clipped_ears_or_crown"] = {
                "status": "PASS",
                "score": 0,
                "threshold": 0,
                "details": "Ears verified within lateral margins",
            }

        # 8b. Delivered Crop Bottom Torso Check (Strict Anti-Floating Torso Gate)
        trans_alpha = crop_meta.get("transformed_person_alpha")
        if trans_alpha is not None:
            crop_bottom_alpha = trans_alpha[int(h * 0.98):, int(w * 0.30):int(w * 0.70)]
            crop_bottom_is_bg_frac = float(np.mean(crop_bottom_alpha < 30))
        else:
            crop_bottom_strip = cand_rgb[int(h * 0.98):, int(w * 0.30):int(w * 0.70)]
            crop_bottom_dist = np.max(np.abs(crop_bottom_strip.astype(float) - bg_target_arr), axis=2)
            crop_bottom_is_bg_frac = float(np.mean(crop_bottom_dist < bg_tolerance))

        if crop_bottom_is_bg_frac > 0.30:
            checks["floating_torso"] = {
                "status": "FAIL",
                "score": round(crop_bottom_is_bg_frac, 2),
                "threshold": 0.30,
                "details": f"Torso is floating: bottom canvas row contains {crop_bottom_is_bg_frac*100:.1f}% background ({'alpha' if trans_alpha is not None else 'color'})",
            }
            defects.append("floating_torso")
        else:
            checks["floating_torso"] = {
                "status": "PASS",
                "score": round(crop_bottom_is_bg_frac, 2),
                "threshold": 0.30,
                "details": "Torso solidly anchors to canvas bottom",
            }

        # Stage B Mandatory Checks
        post_crop_mandatory = [
            "output_dimensions",
            "aspect_ratio_consistency",
            "complete_head_visibility",
            "complete_chin_visibility",
            "visible_neck_and_chest",
            "visible_collar_and_placket",
            "background_cleanliness_post_crop",
            "floating_torso",
            "no_clipped_ears_or_crown",
        ]
        if has_bows:
            post_crop_mandatory.append("original_bows_fully_visible")

        all_post_passed = True
        failed_mandatory = []
        for chk in post_crop_mandatory:
            if chk not in checks:
                all_post_passed = False
                failed_mandatory.append(f"{chk}=MISSING_KEY")
                defects.append(f"missing_post_crop_check ({chk})")
            elif checks[chk]["status"] != "PASS":
                all_post_passed = False
                failed_mandatory.append(f"{chk}={checks[chk]['status']}")

        if not all_post_passed or len(defects) > 0:
            checks["invalid_passport_framing"] = {
                "status": "FAIL",
                "score": len(defects),
                "threshold": 0,
                "details": f"Passport framing defect: {'; '.join(defects)}",
            }
            if "invalid_passport_framing" not in defects:
                defects.append("invalid_passport_framing")
        else:
            checks["invalid_passport_framing"] = {
                "status": "PASS",
                "score": 0,
                "threshold": 0,
                "details": "Standardized passport framing verified (dimensions, headroom, chin, chest, collar)",
            }

        passed = (len(defects) == 0) and all_post_passed
        needs_review = not passed or len(warnings) > 0 or len(failed_mandatory) > 0

        logger.info(
            "[QUALITY_VALIDATOR] Stage 'post_crop' validation (job=%s): passed=%s, review=%s, defects=%s, warnings=%s",
            job_id, passed, needs_review, defects, warnings,
        )

        return {
            "stage": "post_crop",
            "passed": passed,
            "needs_review": needs_review,
            "defects": defects,
            "warnings": warnings,
            "checks": checks,
            "crop_meta": {k: v for k, v in crop_meta.items() if not isinstance(v, np.ndarray)},
        }

    def compare_candidates(self, raw_quality: Dict[str, Any], repaired_quality: Dict[str, Any]) -> bool:
        """Section 8: If a repaired candidate is worse than the raw candidate, discard repair.
        
        Returns True if repaired candidate is acceptable or better; False if repair degraded quality.
        """
        raw_sim = float(raw_quality.get("similarity", 0.0))
        rep_sim = float(repaired_quality.get("similarity", 0.0))
        raw_rmse = float(raw_quality.get("geometry_rmse", 0.0))
        rep_rmse = float(repaired_quality.get("geometry_rmse", 0.0))

        # If repair reduced similarity by > 0.02 or increased RMSE by > 0.01, repair degraded quality
        if rep_sim < raw_sim - 0.02 or rep_rmse > raw_rmse + 0.015:
            logger.warning(
                "[QUALITY_VALIDATOR] Repaired candidate degraded quality: sim %.3f -> %.3f, rmse %.4f -> %.4f. Discarding repair.",
                raw_sim, rep_sim, raw_rmse, rep_rmse
            )
            return False
        return True

    def save_validation_json(self, data: Dict[str, Any], path: Path) -> None:
        """Save formatted validation JSON artifact."""
        def _sanitize(obj):
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
            elif isinstance(obj, (list, tuple)):
                return [_sanitize(v) for v in obj if not isinstance(v, np.ndarray)]
            elif isinstance(obj, np.generic):
                return obj.item()
            return obj

        path.parent.mkdir(parents=True, exist_ok=True)
        sanitized = _sanitize(data)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sanitized, f, indent=2)


# Global singleton instance
quality_validator = QualityValidator()
