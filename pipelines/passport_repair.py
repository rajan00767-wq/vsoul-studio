"""Conservative masks and validation for localized passport-photo repair."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image


class FaceValidationError(RuntimeError):
    """Raised when identity-safe processing cannot obtain a reliable face."""


@dataclass(frozen=True)
class RepairSettings:
    min_face_score: float = 0.60
    identity_threshold: float = 0.90
    min_damage_fraction: float = 0.00015
    severe_damage_fraction: float = 0.035
    face_protect_expand: float = 0.08
    minor_strength: float = 0.68
    normal_strength: float = 0.78
    severe_strength: float = 0.86
    inference_steps: int = 18
    guidance_scale: float = 5.5
    controlnet_scale: float = 0.15
    ip_adapter_scale: float = 0.78
    transition_contribution: float = 0.10
    damaged_contribution: float = 0.25
    debug: bool = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None = None) -> "RepairSettings":
        values = values or {}

        def value(name: str, default: Any, cast):
            env_name = f"PASSPORT_REPAIR_{name.upper()}"
            raw = values.get(f"repair_{name}", os.getenv(env_name, default))
            if cast is bool:
                return str(raw).lower() in {"1", "true", "yes", "on"}
            try:
                return cast(raw)
            except (TypeError, ValueError):
                return default

        defaults = cls()
        return cls(**{
            "min_face_score": value("min_face_score", defaults.min_face_score, float),
            "identity_threshold": value("identity_threshold", defaults.identity_threshold, float),
            "min_damage_fraction": value("min_damage_fraction", defaults.min_damage_fraction, float),
            "severe_damage_fraction": value("severe_damage_fraction", defaults.severe_damage_fraction, float),
            "face_protect_expand": value("face_protect_expand", defaults.face_protect_expand, float),
            "minor_strength": value("minor_strength", defaults.minor_strength, float),
            "normal_strength": value("normal_strength", defaults.normal_strength, float),
            "severe_strength": value("severe_strength", defaults.severe_strength, float),
            "inference_steps": value("inference_steps", defaults.inference_steps, int),
            "guidance_scale": value("guidance_scale", defaults.guidance_scale, float),
            "controlnet_scale": value("controlnet_scale", defaults.controlnet_scale, float),
            "ip_adapter_scale": value("ip_adapter_scale", defaults.ip_adapter_scale, float),
            "transition_contribution": value("transition_contribution", defaults.transition_contribution, float),
            "damaged_contribution": value("damaged_contribution", defaults.damaged_contribution, float),
            "debug": value("debug", defaults.debug, bool),
        })


def _largest_face(faces):
    return max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))


def analyze_face(app, image_path: str, min_score: float) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FaceValidationError(f"Cannot read identity image: {image_path}")
    faces = app.get(image)
    if not faces:
        raise FaceValidationError("No face detected; identity-safe generation refused")
    face = _largest_face(faces)
    score = float(getattr(face, "det_score", 0.0))
    if score < min_score:
        raise FaceValidationError(
            f"Face detection confidence {score:.3f} is below required {min_score:.3f}"
        )
    embedding = np.asarray(face.embedding, dtype=np.float32)
    embedding /= max(float(np.linalg.norm(embedding)), 1e-8)
    return {
        "bbox": np.asarray(face.bbox, dtype=np.float32),
        "landmarks": np.asarray(face.kps, dtype=np.float32),
        "det_score": score,
        "embedding": embedding,
        "shape": image.shape[:2],
    }


def create_face_protection_mask(shape: tuple[int, int], bbox, expansion: float = 0.08) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * expansion))
    x2 = min(width, int(math.ceil(x2 + bw * expansion)))
    y1 = max(0, int(y1 - bh * expansion))
    y2 = min(height, int(math.ceil(y2 + bh * expansion)))
    mask = np.zeros((height, width), np.uint8)
    cv2.ellipse(
        mask,
        ((x1 + x2) // 2, (y1 + y2) // 2),
        (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2)),
        0, 0, 360, 255, -1,
    )
    return mask


def create_damage_mask(image_rgb: np.ndarray, alpha: np.ndarray, settings: RepairSettings) -> dict[str, np.ndarray]:
    """Return narrow, evidence-based boundary masks; never a full segmentation mask."""
    alpha = np.asarray(alpha, dtype=np.uint8)
    height, width = alpha.shape
    radius = max(3, int(round(min(height, width) * 0.012)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    high_fg = alpha >= 245
    background = alpha <= 10
    uncertain = (alpha > 10) & (alpha < 245)
    subject = (alpha >= 128).astype(np.uint8) * 255
    boundary = cv2.subtract(cv2.dilate(subject, kernel), cv2.erode(subject, kernel)) > 0
    near_boundary = cv2.dilate(boundary.astype(np.uint8), np.ones((5, 5), np.uint8), 1) > 0

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    red_orange = (((hsv[..., 0] < 18) | (hsv[..., 0] > 172)) & (hsv[..., 1] > 65))
    colored_bg = (hsv[..., 1] > 55) & (alpha < 220)
    white_distance = 255.0 - image_rgb.astype(np.float32).mean(axis=2)
    dark_halo = (white_distance > 22) & (alpha < 100)

    spill = (red_orange | colored_bg) & near_boundary
    halo = dark_halo & boundary
    closed_subject = cv2.morphologyEx(subject, cv2.MORPH_CLOSE, kernel)
    missing_edge = (closed_subject > 0) & (subject == 0) & boundary
    shoulder_band = boundary & (np.indices(alpha.shape)[0] > int(height * 0.58))
    shoulder_damage = shoulder_band & colored_bg

    # Alpha uncertainty alone is not damage: it is expected on healthy hair.
    # Require corroborating colour/halo evidence before allowing generation.
    evidence = uncertain & boundary & (colored_bg | red_orange | dark_halo)
    damage = evidence | spill | halo | missing_edge | shoulder_damage
    # Keep the candidate localized around the silhouette and remove isolated noise.
    damage &= near_boundary
    damage_u8 = damage.astype(np.uint8) * 255
    damage_u8 = cv2.morphologyEx(damage_u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    damage_u8 = cv2.morphologyEx(damage_u8, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return {
        "damage": damage_u8,
        "foreground": high_fg.astype(np.uint8) * 255,
        "uncertain": uncertain.astype(np.uint8) * 255,
        "background": background.astype(np.uint8) * 255,
        "subject": subject,
    }


def create_three_zone_mask(damage: np.ndarray, face_protection: np.ndarray, settings: RepairSettings) -> np.ndarray:
    damage = damage.copy()
    damage[face_protection > 0] = 0
    if not np.any(damage):
        return damage
    core = (damage >= 180).astype(np.uint8)
    transition_radius = max(3, int(round(min(damage.shape) * 0.010)))
    dilated = cv2.dilate(
        core,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (transition_radius * 2 + 1,) * 2),
        iterations=1,
    )
    distance = cv2.distanceTransform(dilated, cv2.DIST_L2, 5)
    transition = np.clip(distance / max(float(distance.max()), 1.0), 0.0, 1.0)
    weights = transition * settings.transition_contribution
    weights[core > 0] = settings.damaged_contribution
    weights = cv2.GaussianBlur(weights.astype(np.float32), (0, 0), 1.2)
    weights[face_protection > 0] = 0.0
    return np.clip(weights * 255.0, 0, 255).astype(np.uint8)


def estimate_damage_severity(damage_fraction: float, settings: RepairSettings) -> tuple[str, float]:
    if damage_fraction < settings.min_damage_fraction:
        return "none", 0.0
    if damage_fraction < 0.008:
        return "minor", settings.minor_strength
    if damage_fraction < settings.severe_damage_fraction:
        return "normal", settings.normal_strength
    return "severe", settings.severe_strength


def protected_composite(
    base_rgb: np.ndarray,
    generated_rgb: np.ndarray,
    blend_mask: np.ndarray,
    face_protection: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    weight = blend_mask.astype(np.float32) / 255.0
    weight[face_protection > 0] = 0.0
    # The repair mask is already evidence-localized around the damaged
    # silhouette. Do not multiply generated pixels by the old alpha matte:
    # that would erase reconstructed hair or clothing extending into pixels
    # previously classified as background.
    repair = generated_rgb.astype(np.float32)
    out = base_rgb.astype(np.float32) * (1.0 - weight[..., None]) + repair * weight[..., None]
    # Keep untouched background white, but allow generated content inside the
    # explicit boundary-repair mask.
    out[(alpha <= 3) & (weight <= 0.01)] = 255
    out[face_protection > 0] = base_rgb[face_protection > 0]
    return np.clip(out, 0, 255).astype(np.uint8)


def validate_identity(original: dict[str, Any], final: dict[str, Any], settings: RepairSettings) -> dict[str, Any]:
    similarity = float(np.dot(original["embedding"], final["embedding"]))
    ob, fb = original["bbox"], final["bbox"]
    okps, fkps = original["landmarks"], final["landmarks"]

    def face_width(box):
        return max(float(box[2] - box[0]), 1.0)

    def eye_metrics(kps, box):
        left, right = kps[0], kps[1]
        dx, dy = float(right[0] - left[0]), float(right[1] - left[1])
        return math.hypot(dx, dy) / face_width(box), math.degrees(math.atan2(dy, dx))

    oed, oang = eye_metrics(okps, ob)
    fed, fang = eye_metrics(fkps, fb)
    final_h, final_w = final["shape"]
    final_center = ((fb[0] + fb[2]) * 0.5 / final_w, (fb[1] + fb[3]) * 0.5 / final_h)
    final_size = (float(fb[2] - fb[0]) / final_w, float(fb[3] - fb[1]) / final_h)
    checks = {
        "identity": bool(similarity >= settings.identity_threshold),
        "detection": bool(final["det_score"] >= settings.min_face_score),
        "eye_distance": bool(abs(fed - oed) <= 0.10),
        "eye_angle": bool(abs(fang - oang) <= 6.0),
        "face_center": bool(abs(final_center[0] - 0.5) <= 0.18 and 0.20 <= final_center[1] <= 0.52),
        "face_size": bool(0.20 <= final_size[0] <= 0.80 and 0.20 <= final_size[1] <= 0.75),
    }
    return {
        "identity_similarity": round(similarity, 4),
        "original_detection_score": round(float(original["det_score"]), 4),
        "final_detection_score": round(float(final["det_score"]), 4),
        "eye_distance_delta": round(abs(fed - oed), 4),
        "eye_angle_delta": round(abs(fang - oang), 3),
        "final_face_center": [round(float(v), 4) for v in final_center],
        "final_face_size": [round(float(v), 4) for v in final_size],
        "checks": checks,
        "passed": all(checks.values()),
    }


def save_debug_image(debug_dir: Path | None, name: str, image) -> None:
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(image, np.ndarray):
        Image.fromarray(image).save(debug_dir / name)
    else:
        image.save(debug_dir / name)
