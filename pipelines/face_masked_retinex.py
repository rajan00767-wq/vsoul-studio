"""Dynamic, identity-preserving Retinex conditioning for harsh-sun portraits.

The returned image is a Qwen conditioning aid only.  The untouched upload
remains the VL, biometric, and delivery-validation authority.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _soft(mask: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(
        mask.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma,
    )


def prepare_face_masked_retinex_conditioning(
    image: Image.Image,
) -> tuple[Image.Image, dict[str, float | int | bool]]:
    """Flatten measured face, hair and exposed-body illumination locally.

    Face geometry comes from InsightFace and hair semantics from SCHP.  No
    fixed skin tone, hair colour, age, gender, or hairstyle is assumed.
    Identity-defining facial features and side-mounted accessories are kept
    outside the correction authority.  Clothing is deliberately excluded so
    Retinex cannot shift the supplied uniform colour or pattern.
    """
    from pipelines.photo_restoration import _get_insight_app
    from pipelines.schp_service import parse as parse_body_parts

    source_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = source_rgb.shape[:2]
    bgr = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR)
    faces = _get_insight_app().get(bgr)
    if len(faces) != 1:
        raise RuntimeError(
            f"Face-masked Retinex requires one face; detected {len(faces)}"
        )

    x1, y1, x2, y2 = [int(round(float(value))) for value in faces[0].bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    face_w, face_h = x2 - x1, y2 - y1
    if face_w < 32 or face_h < 32:
        raise RuntimeError("Face-masked Retinex detected an unusably small face")

    labels = parse_body_parts(source_rgb)["labels"]
    face_u8 = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(
        face_u8,
        ((x1 + x2) // 2, y1 + int(face_h * .48)),
        (int(face_w * .52), int(face_h * .55)),
        0, 0, 360, 255, -1,
    )
    face_core = face_u8 > 0
    if int(face_core.sum()) < 256:
        raise RuntimeError("Face-masked Retinex produced an empty face mask")

    # Use the complete face-relative hair silhouette, not crown pixels alone.
    # On hard-sun portraits the entire crown can be silver, so it cannot supply
    # a dark colour anchor. Side curls/tails provide this same person's shaded
    # hair tone. SCHP label 2 keeps clothing and skin outside the correction.
    hair_window = np.zeros((height, width), dtype=bool)
    hair_window[
        max(0, y1 - int(face_h * .78)):min(height, y2 + int(face_h * .62)),
        max(0, x1 - int(face_w * .78)):min(width, x2 + int(face_w * .78)),
    ] = True
    hair_core = (labels == 2) & hair_window & ~face_core

    protected = np.zeros((height, width), dtype=np.uint8)
    landmarks = np.asarray(getattr(faces[0], "kps", []), dtype=np.float32)
    if landmarks.shape == (5, 2):
        for px, py in landmarks[:2]:
            cv2.circle(
                protected, (int(px), int(py)),
                max(5, int(face_w * .08)), 255, -1,
            )
        cv2.circle(
            protected, (int(landmarks[2, 0]), int(landmarks[2, 1])),
            max(4, int(face_w * .045)), 255, -1,
        )
        mouth = np.mean(landmarks[3:5], axis=0)
        cv2.ellipse(
            protected, (int(mouth[0]), int(mouth[1])),
            (max(7, int(face_w * .12)), max(5, int(face_w * .07))),
            0, 0, 360, 255, -1,
        )
    protected_alpha = _soft(
        protected.astype(np.float32) / 255.0,
        max(2.0, face_w * .018),
    )

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    luminance = lab[:, :, 0]
    sigma = max(11.0, face_w * .11)
    illumination = cv2.GaussianBlur(
        luminance, (0, 0), sigmaX=sigma, sigmaY=sigma,
    )

    face_values = illumination[face_core]
    face_target = float(np.percentile(face_values, 48.0))
    face_delta = np.clip(face_target - illumination, -30.0, 20.0)
    face_delta[np.abs(face_delta) < 5.0] = 0.0
    face_alpha = _soft(
        face_core.astype(np.float32), max(4.0, face_w * .03),
    )
    face_alpha *= 1.0 - protected_alpha * .70
    luminance += face_delta * face_alpha * .78

    stable_face = face_core & (
        illumination <= np.percentile(face_values, 58.0)
    )
    stable_a = float(np.median(lab[:, :, 1][stable_face]))
    stable_b = float(np.percentile(lab[:, :, 2][stable_face], 38.0))
    warm_excess = np.clip(lab[:, :, 2] - (stable_b + 3.0), 0.0, 14.0)
    warm_strength = np.clip(
        (illumination - face_target + 8.0) / 28.0, 0.0, 1.0,
    )
    lab[:, :, 2] -= warm_excess * face_alpha * warm_strength * .62

    hair_alpha = np.zeros((height, width), dtype=np.float32)
    hair_pixels = int(hair_core.sum())
    if hair_pixels >= 128:
        hair_values = illumination[hair_core]
        anchor_cut = float(np.percentile(hair_values, 42.0))
        hair_anchor = hair_core & (illumination <= anchor_cut)
        if int(hair_anchor.sum()) >= 64:
            anchor_l = float(np.median(illumination[hair_anchor]))
            anchor_a = float(np.median(lab[:, :, 1][hair_anchor]))
            anchor_b = float(np.median(lab[:, :, 2][hair_anchor]))
            chroma_distance = np.sqrt(
                (lab[:, :, 1] - anchor_a) ** 2
                + (lab[:, :, 2] - anchor_b) ** 2
            )
            glare = (
                hair_core
                & (illumination > anchor_l + 10.0)
                & (illumination < 225.0)
                & (chroma_distance <= 34.0)
            )
            glare_strength = np.clip(
                (illumination - (anchor_l + 10.0)) / 34.0, 0.0, 1.0,
            )
            hair_alpha = _soft(
                glare.astype(np.float32) * glare_strength,
                max(2.0, face_w * .018),
            )
            hair_delta = np.clip(
                (anchor_l + 8.0) - illumination, -62.0, 0.0,
            )
            luminance += hair_delta * hair_alpha * .88
            lab[:, :, 1] += (
                anchor_a - lab[:, :, 1]
            ) * hair_alpha * .62
            lab[:, :, 2] += (
                anchor_b - lab[:, :, 2]
            ) * hair_alpha * .62

    # Extend the same low-frequency correction to every visible patch of
    # exposed skin.  SCHP supplies high-confidence arms/legs while a dynamic
    # chroma match to this person's own stable face recovers neck/upper-chest
    # pixels that human-parsing models commonly classify as clothing.  This
    # remains image-driven: no complexion, gender or garment colour is fixed.
    semantic_body_skin = np.isin(labels, (14, 15, 16, 17))
    torso_window = np.zeros((height, width), dtype=bool)
    torso_x1 = max(0, x1 - int(face_w * .60))
    torso_x2 = min(width, x2 + int(face_w * .60))
    torso_y1 = max(0, y2 - int(face_h * .14))
    torso_y2 = min(height, y2 + int(face_h * 1.75))
    torso_window[torso_y1:torso_y2, torso_x1:torso_x2] = True
    skin_chroma_distance = np.sqrt(
        (lab[:, :, 1] - stable_a) ** 2
        + (lab[:, :, 2] - stable_b) ** 2
    )
    # The neck is spatially connected to the lower face and shares its
    # chroma.  A conservative threshold avoids admitting shirt, tie or jacket
    # pixels into the illumination authority.
    inferred_neck_skin = (
        torso_window
        & (skin_chroma_distance <= 13.0)
        & (luminance >= 24.0)
        & (labels != 2)
    )
    body_skin_u8 = (
        (semantic_body_skin | inferred_neck_skin)
        & ~face_core
        & ~hair_core
    ).astype(np.uint8) * 255
    open_radius = max(1, int(round(face_w * .012)))
    kernel_size = open_radius * 2 + 1
    body_skin_u8 = cv2.morphologyEx(
        body_skin_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size),
        ),
    )
    body_skin_core = body_skin_u8 > 0
    body_alpha = np.zeros((height, width), dtype=np.float32)
    body_pixels = int(body_skin_core.sum())
    if body_pixels >= 96:
        body_values = illumination[body_skin_core]
        body_target = float(np.median((
            face_target,
            float(np.percentile(body_values, 46.0)),
        )))
        body_delta = np.clip(body_target - illumination, -26.0, 18.0)
        body_delta[np.abs(body_delta) < 4.0] = 0.0
        body_alpha = _soft(
            body_skin_core.astype(np.float32),
            max(2.5, face_w * .022),
        )
        # Prevent Gaussian feathering from acquiring unrelated garment areas.
        body_support = cv2.dilate(
            body_skin_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        ).astype(np.float32) / 255.0
        body_alpha *= body_support
        luminance += body_delta * body_alpha * .66

        # Neutralize only illumination-correlated colour cast.  Stable shaded
        # skin remains untouched, preserving the source person's complexion.
        body_cast = np.clip(
            (illumination - body_target + 6.0) / 30.0, 0.0, 1.0,
        )
        body_cast *= np.clip(
            (skin_chroma_distance - 3.0) / 13.0, 0.0, 1.0,
        )
        lab[:, :, 1] += (
            stable_a - lab[:, :, 1]
        ) * body_alpha * body_cast * .42
        lab[:, :, 2] += (
            stable_b - lab[:, :, 2]
        ) * body_alpha * body_cast * .42

    lab[:, :, 0] = np.clip(luminance, 0.0, 255.0)
    corrected_bgr = cv2.cvtColor(
        np.clip(lab, 0.0, 255.0).astype(np.uint8), cv2.COLOR_LAB2BGR,
    )
    corrected_rgb = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB)
    authority = np.clip(
        face_alpha + hair_alpha + body_alpha, 0.0, 1.0,
    )[:, :, None]
    output = np.rint(
        corrected_rgb.astype(np.float32) * authority
        + source_rgb.astype(np.float32) * (1.0 - authority)
    ).clip(0, 255).astype(np.uint8)

    mean_change = float(np.mean(np.abs(
        output.astype(np.float32) - source_rgb.astype(np.float32)
    )))
    metadata: dict[str, float | int | bool] = {
        "applied": True,
        "face_pixels": int(face_core.sum()),
        "hair_pixels": hair_pixels,
        "hair_glare_pixels": int((hair_alpha > .08).sum()),
        "body_skin_pixels": body_pixels,
        "body_corrected_pixels": int((body_alpha > .08).sum()),
        "mean_absolute_change": mean_change,
    }
    logger.info("[FACE_RETINEX] Prepared dynamic conditioning: %s", metadata)
    return Image.fromarray(output, "RGB"), metadata
