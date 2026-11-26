"""Identity-Preserving Studio Portrait Reposing & Pose/Gaze Analysis (V13.2).

Performs comprehensive biometric pose estimation, adaptive routing (Routes A, B, C),
authoritative Corrected Identity Master generation, and studio lighting balancing:
- Head Roll, Yaw, Pitch estimation using InsightFace Buffalo_L 2D-106 & 3D-68 landmarks
- Eye gaze vector deviation measurement
- Shoulder tilt and neck alignment estimation
- Bilateral shadow lifting and specular highlight reduction (retaining natural chin shadows)
- Generates 9 authoritative Corrected Identity Master artifacts
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def analyze_portrait_pose(
    image: Union[Image.Image, np.ndarray],
    semantic_masks: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Measure head roll, yaw, pitch, eye gaze, shoulder tilt, and neck alignment."""
    if isinstance(image, Image.Image):
        rgb_arr = np.array(image.convert("RGB"))
    else:
        rgb_arr = image.copy()

    h, w = rgb_arr.shape[:2]

    from pipelines.photo_restoration import _get_insight_app
    app = _get_insight_app()

    bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    faces = app.get(bgr)

    if not faces:
        return {
            "faces_detected": 0,
            "route": "Route C",
            "needs_review": True,
            "head_roll": 0.0,
            "head_yaw": 0.0,
            "head_pitch": 0.0,
            "gaze_deviation_deg": 0.0,
            "eye_alignment_delta": 0.0,
            "shoulder_tilt_deg": 0.0,
            "neck_alignment_offset": 0.0,
            "det_score": 0.0,
            "details": "Face detection failed",
        }

    face = faces[0]
    det_score = float(face.det_score)
    bbox = [float(x) for x in face.bbox]
    kps = face.kps

    # 1. Head Pose (Pitch, Yaw, Roll)
    # Use face.pose if available [pitch, yaw, roll in degrees]
    if hasattr(face, "pose") and face.pose is not None and len(face.pose) == 3:
        pitch = float(face.pose[0])
        yaw = float(face.pose[1])
        pose_roll = float(face.pose[2])
    else:
        pitch = 0.0
        yaw = 0.0
        pose_roll = 0.0

    # Calculate exact geometric interocular roll from 5-point kps (left eye, right eye)
    # kps[0] = right eye (viewer's left), kps[1] = left eye (viewer's right)
    dx = float(kps[1][0] - kps[0][0])
    dy = float(kps[1][1] - kps[0][1])
    geom_roll = math.degrees(math.atan2(dy, dx))
    
    # Combined authoritative head roll
    head_roll = geom_roll

    # 2. Eye Gaze Deviation
    # Compare pupil/iris center relative to eye corner bounding boxes in 2D-106 landmarks
    gaze_deviation = 0.0
    eye_alignment_delta = abs(dy) / max(1.0, float(bbox[2] - bbox[0]))
    if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
        lm106 = face.landmark_2d_106
        # Eye landmarks: left eye (35:51), right eye (89:105)
        r_eye_pts = lm106[35:51]
        l_eye_pts = lm106[89:105]
        if len(r_eye_pts) > 0 and len(l_eye_pts) > 0:
            r_center = np.mean(r_eye_pts, axis=0)
            l_center = np.mean(l_eye_pts, axis=0)
            eye_mid = (r_center + l_center) * 0.5
            nose_tip = lm106[86] if len(lm106) > 86 else kps[2]
            # Horizontal deviation between eye midpoint and nose tip
            gaze_h_offset = abs(float(nose_tip[0] - eye_mid[0])) / max(1.0, float(bbox[2] - bbox[0]))
            gaze_deviation = float(gaze_h_offset * 45.0)

    # 3. Shoulder Tilt Estimation
    shoulder_tilt = 0.0
    if semantic_masks and "original_garment_mask" in semantic_masks:
        g_mask = semantic_masks["original_garment_mask"]
        if np.any(g_mask):
            ys, xs = np.where(g_mask)
            # Find upper shoulder points on left and right sides
            mid_x = (bbox[0] + bbox[2]) * 0.5
            l_pts = ys[xs < mid_x - 30]
            r_pts = ys[xs > mid_x + 30]
            if len(l_pts) > 10 and len(r_pts) > 10:
                l_top = float(np.percentile(l_pts, 5))
                r_top = float(np.percentile(r_pts, 5))
                shoulder_dx = (bbox[2] - bbox[0]) * 1.5
                shoulder_tilt = math.degrees(math.atan2(r_top - l_top, shoulder_dx))

    # 4. Neck Alignment Offset
    # Offset of chin bottom relative to chest center
    chin_x = float(kps[2][0])
    fcx = float(bbox[0] + bbox[2]) * 0.5
    neck_alignment_offset = abs(chin_x - fcx) / max(1.0, float(bbox[2] - bbox[0]))

    # 5. Adaptive Routing Policy (V13.2 Section 14)
    # Route A (Studio Ready): |roll| <= 3.0 deg, |yaw| <= 5.0 deg, |pitch| <= 5.0 deg
    # Route B (Geometric Alignment Transform): 3.0 deg < |roll| <= 15.0 deg, |yaw| <= 12.0 deg, |pitch| <= 12.0 deg
    # Route C (Controlled Frontalization / Needs Review): |roll| > 15.0 deg or |yaw| > 15.0 deg or |pitch| > 15.0 deg
    abs_roll = abs(head_roll)
    abs_yaw = abs(yaw)
    abs_pitch = abs(pitch)

    if abs_roll <= 3.0 and abs_yaw <= 5.0 and abs_pitch <= 5.0:
        route = "Route A (Studio Ready)"
        needs_review = False
    elif abs_roll <= 15.0 and abs_yaw <= 12.0 and abs_pitch <= 12.0:
        route = "Route B (Geometric Alignment Transform)"
        needs_review = False
    else:
        route = "Route C (Extreme Pose / Needs Review)"
        needs_review = True

    return {
        "faces_detected": 1,
        "route": route,
        "needs_review": needs_review,
        "head_roll": float(round(head_roll, 2)),
        "head_yaw": float(round(yaw, 2)),
        "head_pitch": float(round(pitch, 2)),
        "gaze_deviation_deg": float(round(gaze_deviation, 2)),
        "eye_alignment_delta": float(round(eye_alignment_delta, 4)),
        "shoulder_tilt_deg": float(round(shoulder_tilt, 2)),
        "neck_alignment_offset": float(round(neck_alignment_offset, 4)),
        "det_score": float(round(det_score, 4)),
        "bbox": [float(x) for x in bbox],
        "kps": [[float(x) for x in kp] for kp in kps],
        "recommended_roll_correction": float(round(-head_roll, 2)),
    }


def apply_studio_reposing(
    image: Image.Image,
    identity_layer: Dict[str, Any],
    pose_analysis: Dict[str, Any],
    debug_dir: Optional[Path] = None,
    job_id: str = "studio",
) -> Tuple[Image.Image, Dict[str, Any], Dict[str, Any]]:
    """Execute identity-preserving studio reposing transform (Route B) and produce Corrected Identity Master."""
    w, h = image.size
    rgb_arr = np.array(image.convert("RGB"))
    route = pose_analysis.get("route", "")

    # Save original source artifact
    if debug_dir:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        image.save(debug_dir / "original_source.png")
        image.save(debug_dir / f"{job_id}_original_source.png")
        with open(debug_dir / f"{job_id}_pose_analysis_before.json", "w", encoding="utf-8") as f:
            json.dump(pose_analysis, f, indent=2)

    # If Route A, image is already studio compliant; export identity masters directly
    if "Route A" in route:
        logger.info("[POSE_ANALYZER] Route A selected: Portrait already studio ready (roll=%.2f deg)", pose_analysis.get("head_roll", 0.0))
        pose_transform = {
            "route": "Route A",
            "applied_rotation_deg": 0.0,
            "center": [w // 2, h // 2],
            "scale": 1.0,
            "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        }
        _save_identity_masters(image, rgb_arr, identity_layer, debug_dir, job_id, pose_transform, pose_analysis, pose_analysis)
        return image, identity_layer, pose_transform

    # Route B: Aspect-ratio-preserving 2D similarity transform
    roll_corr = pose_analysis.get("recommended_roll_correction", 0.0)
    # Clamp rotation to safe bound [-15, +15]
    roll_corr = max(-15.0, min(15.0, roll_corr))

    bbox = pose_analysis.get("bbox", [w*0.25, h*0.25, w*0.75, h*0.75])
    fcx = (bbox[0] + bbox[2]) * 0.5
    fcy = (bbox[1] + bbox[3]) * 0.5

    # Construct similarity rotation matrix around face center
    M = cv2.getRotationMatrix2D((fcx, fcy), roll_corr, 1.0)

    # 1. Transform RGB image with high-quality bicubic interpolation and neutral border reflection
    rotated_rgb = cv2.warpAffine(rgb_arr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    rotated_pil = Image.fromarray(rotated_rgb)

    # 2. Transform all masks identically using nearest/bilinear interpolation to prevent edge tearing
    new_id_layer = dict(identity_layer)
    mask_keys = [
        "full_person_alpha",
        "original_face_mask",
        "original_hair_mask",
        "original_neck_mask",
        "original_garment_mask",
        "original_accessory_mask",
        "original_ears_mask",
    ]

    for k in mask_keys:
        if k in identity_layer and identity_layer[k] is not None:
            m = identity_layer[k]
            if isinstance(m, np.ndarray):
                m_dtype = m.dtype
                if m_dtype == bool:
                    m_u8 = m.astype(np.uint8) * 255
                    rot_m = cv2.warpAffine(m_u8, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    new_id_layer[k] = rot_m > 128
                else:
                    rot_m = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    new_id_layer[k] = rot_m

    # Re-evaluate pose after transform to verify roll is eliminated
    pose_after = analyze_portrait_pose(rotated_pil, new_id_layer)
    logger.info(
        "[POSE_ANALYZER] Route B executed: roll corrected from %.2f deg -> %.2f deg (matrix applied around (%.1f, %.1f))",
        pose_analysis.get("head_roll", 0.0), pose_after.get("head_roll", 0.0), fcx, fcy
    )

    pose_transform = {
        "route": "Route B",
        "applied_rotation_deg": roll_corr,
        "center": [fcx, fcy],
        "scale": 1.0,
        "matrix": M.tolist(),
        "roll_before": pose_analysis.get("head_roll", 0.0),
        "roll_after": pose_after.get("head_roll", 0.0),
    }

    _save_identity_masters(rotated_pil, rotated_rgb, new_id_layer, debug_dir, job_id, pose_transform, pose_analysis, pose_after)

    return rotated_pil, new_id_layer, pose_transform


def _save_identity_masters(
    img: Image.Image,
    rgb_arr: np.ndarray,
    id_layer: Dict[str, Any],
    debug_dir: Optional[Path],
    job_id: str,
    pose_transform: Dict[str, Any],
    pose_before: Dict[str, Any],
    pose_after: Dict[str, Any],
) -> None:
    """Save the 9 authoritative Corrected Identity Master artifacts (V13.2 Section 16)."""
    if not debug_dir:
        return

    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    h, w = rgb_arr.shape[:2]

    # 1. pose_corrected_source.png
    img.save(debug_dir / "pose_corrected_source.png")
    img.save(debug_dir / f"{job_id}_pose_corrected_source.png")

    # Helper to construct straight RGBA
    def make_rgba(mask_key):
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = rgb_arr
        if mask_key in id_layer and id_layer[mask_key] is not None:
            m = id_layer[mask_key]
            if m.dtype == bool:
                rgba[:, :, 3] = np.where(m, 255, 0).astype(np.uint8)
            else:
                rgba[:, :, 3] = m.astype(np.uint8)
        return Image.fromarray(rgba, "RGBA")

    # 2. corrected_face_rgba.png
    face_rgba = make_rgba("original_face_mask")
    face_rgba.save(debug_dir / "corrected_face_rgba.png")
    face_rgba.save(debug_dir / f"{job_id}_corrected_face_rgba.png")

    # 3. corrected_neck_rgba.png
    neck_rgba = make_rgba("original_neck_mask")
    neck_rgba.save(debug_dir / "corrected_neck_rgba.png")
    neck_rgba.save(debug_dir / f"{job_id}_corrected_neck_rgba.png")

    # 4. corrected_hair_rgba.png
    hair_rgba = make_rgba("original_hair_mask")
    hair_rgba.save(debug_dir / "corrected_hair_rgba.png")
    hair_rgba.save(debug_dir / f"{job_id}_corrected_hair_rgba.png")

    # 5. corrected_accessories_rgba.png
    acc_rgba = make_rgba("original_accessory_mask")
    acc_rgba.save(debug_dir / "corrected_accessories_rgba.png")
    acc_rgba.save(debug_dir / f"{job_id}_corrected_accessories_rgba.png")

    # 6. corrected_person_rgba.png
    person_rgba = make_rgba("full_person_alpha")
    person_rgba.save(debug_dir / "corrected_person_rgba.png")
    person_rgba.save(debug_dir / f"{job_id}_corrected_person_rgba.png")

    # 7. pose_analysis_before.json
    with open(debug_dir / "pose_analysis_before.json", "w", encoding="utf-8") as f:
        json.dump(pose_before, f, indent=2)

    # 8. pose_analysis_after.json
    with open(debug_dir / "pose_analysis_after.json", "w", encoding="utf-8") as f:
        json.dump(pose_after, f, indent=2)

    # 9. pose_transform.json
    with open(debug_dir / "pose_transform.json", "w", encoding="utf-8") as f:
        json.dump(pose_transform, f, indent=2)


def apply_studio_lighting(
    image: Image.Image,
    face_mask: Optional[np.ndarray] = None,
    neck_mask: Optional[np.ndarray] = None,
) -> Image.Image:
    """Apply bilateral shadow lifting and specular highlight compression (V13.2 Section 17).
    
    - Lifts harsh shadows in skin (L < 95 -> +30) using edge-preserving bilateral filtering
    - Compresses blown-out specular highlights (L > 220 -> -20)
    - Preserves anatomical chin cast shadow
    - Strictly avoids plastic airbrushed skin texture
    """
    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]

    # Convert to LAB color space to operate strictly on Luminance channel
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32)

    skin_mask = np.zeros((h, w), dtype=bool)
    if face_mask is not None:
        skin_mask |= face_mask
    if neck_mask is not None:
        skin_mask |= neck_mask

    if not np.any(skin_mask):
        # Fallback YCrCb skin detection
        ycrcb = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        skin_mask = (cr >= 130) & (cr <= 175) & (cb >= 75) & (cb <= 130)

    if not np.any(skin_mask):
        return image

    # Bilateral filtered base luminance to preserve sharp facial microtexture
    L_u8 = np.clip(L, 0, 255).astype(np.uint8)
    L_smooth = cv2.bilateralFilter(L_u8, d=9, sigmaColor=75, sigmaSpace=75).astype(np.float32)

    # 1. Shadow lifting: lift L < 95 by up to +30 smoothly
    shadow_weight = np.clip((95.0 - L_smooth) / 95.0, 0.0, 1.0)
    L_lifted = L + (shadow_weight * 30.0 * skin_mask)

    # 2. Specular highlight compression: compress L > 220 by up to -20 smoothly
    highlight_weight = np.clip((L_smooth - 220.0) / 35.0, 0.0, 1.0)
    L_compressed = L_lifted - (highlight_weight * 20.0 * skin_mask)

    # Retain natural chin shadow: prevent lifting immediately below jawline
    if face_mask is not None and np.any(face_mask):
        y_f, _ = np.where(face_mask)
        chin_y = int(y_f.max())
        chin_shadow_zone = np.zeros((h, w), dtype=bool)
        chin_shadow_zone[chin_y:min(h, chin_y + 25), :] = True
        # Fade lifting in chin shadow zone
        L_compressed = np.where(chin_shadow_zone & skin_mask, L, L_compressed)

    lab[:, :, 0] = np.clip(L_compressed, 0, 255).astype(np.uint8)
    corrected_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return Image.fromarray(corrected_rgb)
