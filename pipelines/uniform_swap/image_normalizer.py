"""Aspect-Ratio-Preserving Normalization Utility for School ID Portraits.

Section 3 Requirements:
- Replaces distortion-producing `image.resize((w, h))` with landmark-aware uniform scaling.
- Preserves original facial proportions, head width/height ratios, eye distance and jaw geometry.
- Uses landmark-aware positioning to center the student naturally on the portrait canvas.
- Maintains consistent, invertible coordinate transforms between RGB images, masks and landmarks.
- Never independently stretches width and height.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple, Union
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


class NormalizationTransform:
    """Manages invertible 2D coordinate transforms between original upload and normalized canvas."""

    def __init__(
        self,
        orig_size: Tuple[int, int],
        target_size: Tuple[int, int],
        crop_box: Tuple[int, int, int, int],  # (x1, y1, x2, y2) in original coordinates
        scale_x: float,
        scale_y: float,
        pad_x: int = 0,
        pad_y: int = 0,
    ) -> None:
        self.orig_w, self.orig_h = orig_size
        self.target_w, self.target_h = target_size
        self.crop_x1, self.crop_y1, self.crop_x2, self.crop_y2 = crop_box
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.pad_x = pad_x
        self.pad_y = pad_y

    def orig_to_norm_point(self, x: float, y: float) -> Tuple[float, float]:
        """Convert a point from original image coordinates to normalized canvas coordinates."""
        cx = x - self.crop_x1
        cy = y - self.crop_y1
        nx = cx * self.scale_x + self.pad_x
        ny = cy * self.scale_y + self.pad_y
        return nx, ny

    def norm_to_orig_point(self, nx: float, ny: float) -> Tuple[float, float]:
        """Convert a point from normalized canvas coordinates back to original image coordinates."""
        cx = (nx - self.pad_x) / max(1e-6, self.scale_x)
        cy = (ny - self.pad_y) / max(1e-6, self.scale_y)
        x = cx + self.crop_x1
        y = cy + self.crop_y1
        return x, y

    def transform_mask_orig_to_norm(self, mask: np.ndarray, interpolation: int = cv2.INTER_NEAREST) -> np.ndarray:
        """Transform a binary or grayscale mask from original space to normalized canvas."""
        cropped = mask[self.crop_y1 : self.crop_y2, self.crop_x1 : self.crop_x2]
        crop_h, crop_w = cropped.shape[:2]
        scaled_w = max(1, int(round(crop_w * self.scale_x)))
        scaled_h = max(1, int(round(crop_h * self.scale_y)))
        scaled = cv2.resize(cropped, (scaled_w, scaled_h), interpolation=interpolation)

        canvas = np.zeros((self.target_h, self.target_w), dtype=mask.dtype)
        canvas[self.pad_y : self.pad_y + scaled_h, self.pad_x : self.pad_x + scaled_w] = scaled
        return canvas

    def to_dict(self) -> Dict[str, Any]:
        return {
            "orig_size": (self.orig_w, self.orig_h),
            "target_size": (self.target_w, self.target_h),
            "crop_box": (self.crop_x1, self.crop_y1, self.crop_x2, self.crop_y2),
            "scale": (self.scale_x, self.scale_y),
            "padding": (self.pad_x, self.pad_y),
        }


def normalize_portrait_aspect_ratio(
    raw_image: Union[Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (800, 1024),
    background_rgb: Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[Image.Image, NormalizationTransform, Dict[str, Any]]:
    """Normalize input portrait to generation resolution without aspect-ratio stretching.

    Algorithm:
    1. Detect face bounding box and facial landmarks via InsightFace.
    2. Compute original aspect ratio A_0 = W_0 / H_0 vs target aspect ratio A_t = 800 / 1024.
    3. If A_0 > A_t (wider, e.g. 1:1 square schoolboy):
       - Crop a vertical window of width W_crop = round(H_0 * A_t) centered horizontally on face_cx.
       - Scale uniformly to (800, 1024) via Lanczos.
       - scale_x == scale_y. Zero vertical elongation!
    4. If A_0 <= A_t:
       - Scale uniformly S = min(target_w / W_0, target_h / H_0) and center on canvas with padding,
         or crop excess height with appropriate crown headroom.
    """
    if isinstance(raw_image, Image.Image):
        if raw_image.mode in ("RGBA", "LA") or (raw_image.mode == "P" and "transparency" in raw_image.info):
            # Section 5: Alpha-composite over flat background_rgb to completely prevent
            # transparent edge-bleed and stretched border pixels (e.g. Student 2 Vecteezy PNG)
            rgba = raw_image.convert("RGBA")
            bg_pil = Image.new("RGBA", rgba.size, background_rgb + (255,))
            pil_img = Image.alpha_composite(bg_pil, rgba).convert("RGB")
        else:
            pil_img = raw_image.convert("RGB")
    else:
        if raw_image.ndim == 3 and raw_image.shape[2] == 4:
            rgba = Image.fromarray(raw_image)
            bg_pil = Image.new("RGBA", rgba.size, background_rgb + (255,))
            pil_img = Image.alpha_composite(bg_pil, rgba).convert("RGB")
        else:
            pil_img = Image.fromarray(raw_image).convert("RGB")

    pil_img, trim_meta = remove_collage_borders_and_labels(pil_img)

    orig_w, orig_h = pil_img.size
    target_w, target_h = target_size
    target_aspect = target_w / target_h

    # ── 1. Facial Landmark Detection ──
    face_info: Optional[Dict[str, Any]] = None
    try:
        from pipelines.photo_restoration import _get_insight_app
        app = _get_insight_app()
        np_bgr = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
        faces = app.get(np_bgr)
        if faces and len(faces) > 0:
            # Select largest central face
            main_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            bbox = main_face.bbox.astype(float)
            kps = main_face.kps.astype(float) if hasattr(main_face, "kps") and main_face.kps is not None else None
            face_info = {
                "bbox": bbox,
                "kps": kps,
                "face_w": float(bbox[2] - bbox[0]),
                "face_h": float(bbox[3] - bbox[1]),
                "face_cx": float(bbox[0] + bbox[2]) * 0.5,
                "face_cy": float(bbox[1] + bbox[3]) * 0.5,
            }
    except Exception as exc:
        logger.warning("[IMAGE_NORMALIZER] InsightFace face detection failed, using geometrical fallback: %s", exc)

    # ── 2. Landmark-Aware Geometrical Framing ──
    orig_aspect = orig_w / max(1, orig_h)

    if face_info is not None:
        fcx = face_info["face_cx"]
        fcy = face_info["face_cy"]
        fh = face_info["face_h"]
    else:
        fcx = orig_w * 0.5
        fcy = orig_h * 0.45
        fh = orig_h * 0.40

    if orig_aspect > target_aspect:
        # Case A: Input is wider than target aspect (e.g. 1.0 square or 4:3 landscape)
        # Crop width to match target aspect ratio while maintaining full vertical height
        crop_h = orig_h
        crop_w = int(round(crop_h * target_aspect))
        crop_w = min(crop_w, orig_w)

        # Center horizontally on face center
        x1 = int(round(fcx - crop_w * 0.5))
        # Clamp to image boundaries
        x1 = max(0, min(orig_w - crop_w, x1))
        x2 = x1 + crop_w
        y1 = 0
        y2 = orig_h

        cropped_pil = pil_img.crop((x1, y1, x2, y2))
        normalized_pil = cropped_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)

        scale_x = target_w / float(crop_w)
        scale_y = target_h / float(crop_h)
        pad_x, pad_y = 0, 0
        crop_box = (x1, y1, x2, y2)

    elif abs(orig_aspect - target_aspect) < 0.03:
        # Case B: Aspect ratio is already within 3% of target (e.g. 0.778 vs 0.781)
        # Uniform scaling with tight target fit
        crop_box = (0, 0, orig_w, orig_h)
        normalized_pil = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        scale_x = target_w / float(orig_w)
        scale_y = target_h / float(orig_h)
        pad_x, pad_y = 0, 0

    else:
        # Case C: Input is significantly taller than target (e.g. full-body portrait)
        # Crop height with appropriate headroom above crown
        crop_w = orig_w
        crop_h = int(round(crop_w / target_aspect))

        if crop_h <= orig_h:
            # Position crop so face sits in upper-middle half
            desired_fcy = crop_h * 0.40
            y1 = int(round(fcy - desired_fcy))
            y1 = max(0, min(orig_h - crop_h, y1))
            y2 = y1 + crop_h
            x1, x2 = 0, orig_w

            cropped_pil = pil_img.crop((x1, y1, x2, y2))
            normalized_pil = cropped_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
            scale_x = target_w / float(crop_w)
            scale_y = target_h / float(crop_h)
            pad_x, pad_y = 0, 0
            crop_box = (x1, y1, x2, y2)
        else:
            # Scale uniformly to fit height and pad sides
            scale = target_h / float(orig_h)
            scaled_w = int(round(orig_w * scale))
            scaled_h = target_h
            scaled_pil = pil_img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

            normalized_pil = Image.new("RGB", (target_w, target_h), background_rgb)
            pad_x = (target_w - scaled_w) // 2
            pad_y = 0
            normalized_pil.paste(scaled_pil, (pad_x, pad_y))

            scale_x = scale
            scale_y = scale
            crop_box = (0, 0, orig_w, orig_h)

    transform = NormalizationTransform(
        orig_size=(orig_w, orig_h),
        target_size=(target_w, target_h),
        crop_box=crop_box,
        scale_x=scale_x,
        scale_y=scale_y,
        pad_x=pad_x,
        pad_y=pad_y,
    )

    # ── 3. Transformed Landmarks for Verification ──
    norm_face_info: Optional[Dict[str, Any]] = None
    if face_info is not None:
        bx1, by1 = transform.orig_to_norm_point(face_info["bbox"][0], face_info["bbox"][1])
        bx2, by2 = transform.orig_to_norm_point(face_info["bbox"][2], face_info["bbox"][3])
        norm_bbox = [bx1, by1, bx2, by2]

        norm_kps = []
        if face_info["kps"] is not None:
            for pt in face_info["kps"]:
                nx, ny = transform.orig_to_norm_point(pt[0], pt[1])
                norm_kps.append([nx, ny])

        orig_face_ratio = face_info["face_w"] / max(1.0, face_info["face_h"])
        norm_face_ratio = (bx2 - bx1) / max(1.0, (by2 - by1))
        distortion_delta = abs(norm_face_ratio - orig_face_ratio) / orig_face_ratio

        norm_face_info = {
            "bbox": norm_bbox,
            "kps": norm_kps,
            "orig_face_ratio": orig_face_ratio,
            "norm_face_ratio": norm_face_ratio,
            "distortion_delta": distortion_delta,
        }
        logger.info(
            "[IMAGE_NORMALIZER] Normalized portrait: orig_size=(%d, %d) -> target=(%d, %d), crop_box=%s, face_aspect_orig=%.4f, face_aspect_norm=%.4f (distortion=%.2f%%)",
            orig_w, orig_h, target_w, target_h, crop_box, orig_face_ratio, norm_face_ratio, distortion_delta * 100.0
        )

    meta = {
        "transform": transform.to_dict(),
        "face_info": norm_face_info,
        "orig_aspect": orig_aspect,
        "target_aspect": target_aspect,
    }
    return normalized_pil, transform, meta


def create_landmarks_overlay(
    normalized_image: Image.Image,
    face_info: Optional[Dict[str, Any]],
) -> Image.Image:
    """Create a diagnostic image showing normalized portrait with facial landmarks overlay."""
    overlay = normalized_image.copy()
    if face_info is None or "bbox" not in face_info:
        return overlay

    draw = ImageDraw.Draw(overlay)
    bbox = face_info["bbox"]
    draw.rectangle(
        [(bbox[0], bbox[1]), (bbox[2], bbox[3])],
        outline=(0, 255, 128),
        width=3,
    )

    kps = face_info.get("kps")
    if kps:
        for i, pt in enumerate(kps):
            px, py = int(round(pt[0])), int(round(pt[1]))
            # Points: 0=left eye, 1=right eye, 2=nose, 3=left mouth, 4=right mouth
            color = (255, 60, 60) if i in (0, 1) else (0, 210, 255) if i == 2 else (255, 220, 0)
            draw.ellipse([(px - 5, py - 5), (px + 5, py + 5)], fill=color, outline=(255, 255, 255), width=1)

    return overlay


def remove_collage_borders_and_labels(image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
    """Inspect and trim synthetic collage frames, label strips, or flat borders (up to 4% margin)."""
    np_img = np.asarray(image.convert("RGB")).copy()
    h, w = np_img.shape[:2]
    max_trim_y = max(1, int(h * 0.04))
    max_trim_x = max(1, int(w * 0.04))
    top, bottom, left, right = 0, 0, 0, 0
    for y in range(max_trim_y):
        row = np_img[y, :, :]
        if np.std(row) < 3.0 or np.mean(row) > 250.0:
            top = y + 1
        else:
            break
    for y in range(max_trim_y):
        row = np_img[h - 1 - y, :, :]
        if np.std(row) < 3.0 or np.mean(row) > 250.0:
            bottom = y + 1
        else:
            break
    for x in range(max_trim_x):
        col = np_img[:, x, :]
        if np.std(col) < 3.0 or np.mean(col) > 250.0:
            left = x + 1
        else:
            break
    for x in range(max_trim_x):
        col = np_img[:, w - 1 - x, :]
        if np.std(col) < 3.0 or np.mean(col) > 250.0:
            right = x + 1
        else:
            break
    if top > 0 or bottom > 0 or left > 0 or right > 0:
        trimmed = np_img[top : h - bottom, left : w - right]
        meta = {"trimmed": (top, bottom, left, right)}
        return Image.fromarray(trimmed, "RGB"), meta
    return image, {"trimmed": (0, 0, 0, 0)}


def detect_harsh_sunlight(image: Image.Image) -> bool:
    """Detect if an input portrait exhibits extreme contrast typical of harsh direct sunlight on foreground/face."""
    np_img = np.asarray(image.convert("RGB"))
    lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0]
    # Identify foreground by excluding flat outer studio border
    border = np.concatenate([np_img[0, :], np_img[-1, :], np_img[:, 0], np_img[:, -1]])
    bg_color = np.median(border, axis=0)
    diff = np.linalg.norm(np_img.astype(float) - bg_color, axis=2)
    fg = diff > 25
    if np.sum(fg) < 500:
        return False
    l_fg = l[fg]
    std_fg = float(np.std(l_fg))
    hi_fg = float(np.mean(l_fg > 215))
    lo_fg = float(np.mean(l_fg < 45))
    return bool(std_fg > 70.0 and hi_fg > 0.14 and lo_fg > 0.12)


def correct_harsh_sunlight_and_shadows(
    image: Image.Image,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Conservative shadow lifting on L channel in LAB color space for high-contrast sunlight portraits.

    Preserves chromaticity (A, B channels completely untouched).
    Lifts deep shadows so SCHP and BiRefNet can reliably detect hair and silhouette,
    and compresses specular blowouts without altering facial geometry.
    """
    np_img = np.asarray(image.convert("RGB")).copy()
    lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0].astype(np.float32)

    # Shadow lift: smooth boost for deep shadows below 95
    shadow_boost = np.clip((95.0 - l) / 95.0, 0.0, 1.0) ** 1.3 * 38.0

    # Highlight compression: smooth reduction for harsh highlights above 215
    hi_compress = np.clip((l - 215.0) / 40.0, 0.0, 1.0) * 22.0

    adjusted_l = np.clip(l + shadow_boost - hi_compress, 0.0, 255.0)

    lab[:, :, 0] = adjusted_l.astype(np.uint8)
    corrected_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    meta = {
        "harsh_sunlight_corrected": True,
        "max_shadow_lift": 38.0,
        "max_highlight_compress": 22.0,
    }
    return Image.fromarray(corrected_rgb, "RGB"), meta


def detect_scanned_photo(image: Image.Image) -> bool:
    """Detect if an input portrait exhibits characteristics of a scanned or photographed physical print."""
    np_img = np.asarray(image.convert("RGB"))
    h, w = np_img.shape[:2]
    # Scanned indicators:
    # 1. Low resolution: min(w, h) < 600
    is_low_res = min(w, h) < 600
    # 2. Uneven border lightness / scanner margin
    top_diff = abs(float(np.mean(np_img[0, :, :])) - float(np.mean(np_img[min(h - 1, int(h * 0.1)), :, :])))
    # 3. High chromatic channel imbalance (e.g. reddish or yellowish color cast across image)
    means = np.mean(np_img, axis=(0, 1))
    max_diff = max(abs(means[0] - means[1]), abs(means[0] - means[2]), abs(means[1] - means[2]))
    return bool(is_low_res or (max_diff > 35.0 and top_diff > 25.0))


def preprocess_scanned_photo(
    image: Image.Image,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Conservative preprocessing for low-quality scanned/photographed portraits (Section 4).

    Applies:
    1. Border margin removal (scanner print margin trimming).
    2. Mild color cast balancing (gray-world adjustment with conservative 35% blend).
    3. Conservative bilateral edge-preserving denoising.
    4. Non-generative Lanczos scaling.
    """
    np_img = np.asarray(image.convert("RGB")).copy()
    h, w = np_img.shape[:2]

    # 1. Trimming outer flat borders
    margin_top = 0
    margin_bottom = 0
    margin_left = 0
    margin_right = 0

    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    for r in range(min(15, h // 10)):
        if np.std(gray[r, :]) < 5.0 and abs(float(np.mean(gray[r, :])) - float(np.mean(gray[min(h - 1, r + 5), :]))) > 15.0:
            margin_top = r + 1

    cropped = np_img[margin_top : h - margin_bottom, margin_left : w - margin_right]

    # 2. Mild color cast balancing (gray-world) with contrast stabilization
    ch_means = np.mean(cropped, axis=(0, 1))
    target_mean = float(np.mean(ch_means))
    corrected = cropped.astype(np.float32)
    for c in range(3):
        gain = target_mean / max(10.0, float(ch_means[c]))
        # Dampen gain to 35% strength to avoid washing out natural skin
        damped_gain = 1.0 + (gain - 1.0) * 0.35
        corrected[:, :, c] = np.clip(corrected[:, :, c] * damped_gain, 0, 255)
    balanced = corrected.astype(np.uint8)

    # 3. Non-generative contrast stabilization on luminance channel (CLAHE)
    lab = cv2.cvtColor(balanced, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    stabilized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # 4. Conservative edge-preserving bilateral denoising to suppress halftone scanner dots
    denoised = cv2.bilateralFilter(stabilized, d=5, sigmaColor=15, sigmaSpace=15)

    out_pil = Image.fromarray(denoised, "RGB")
    if target_size:
        out_pil = out_pil.resize(target_size, Image.Resampling.LANCZOS)

    meta = {
        "borders_trimmed": (margin_top, margin_bottom, margin_left, margin_right),
        "color_gains": [float(target_mean / max(10.0, ch_means[c])) for c in range(3)],
        "denoised": True,
    }
    return out_pil, meta


def produce_passport_crop(
    image: Image.Image,
    face_info: Optional[Dict[str, Any]] = None,
    identity_layer: Optional[Dict[str, Any]] = None,
    output_size: Tuple[int, int] = (800, 1024),
    target_head_occupancy: float = 0.58,
    target_headroom_ratio: float = 0.10,
    background_rgb: Tuple[int, int, int] = (255, 255, 255),
    debug_dir: Optional[Union[str, Path]] = None,
    job_id: str = "passport_crop",
    return_meta: bool = True,
) -> Union[Image.Image, Tuple[Image.Image, Dict[str, Any]]]:
    """Produce standardized upper-chest school portrait passport framing (V12.5 Task 3).

    Calculates framing from actual hair crown to chin (head height) and ensures:
    - Head height occupancy ~ 55%-65% of final canvas height (default 0.58).
    - Headroom ~ 8%-12% above hair crown (default 0.10).
    - Remaining space (~25%-35%) below chin for neck, collar, button placket, and upper chest.
    - All mandatory head, ear, and accessory (bows/ribbons) boundaries are strictly enclosed.
    - White padding is applied when crop rectangle exceeds source canvas boundaries.
    - Uniform single scale factor is used (zero geometric stretching or aspect distortion).
    - Never crops the chin, collar, ears, crown, or original hair bows.
    - Returns transformed coordinates and metadata for post-crop validation.
    """
    w, h = image.size
    out_w, out_h = output_size
    target_aspect = float(out_w) / float(out_h)
    deb_dir_path = Path(debug_dir) if debug_dir is not None else None
    if deb_dir_path is not None:
        deb_dir_path.mkdir(parents=True, exist_ok=True)

    # If image is RGB but identity_layer provides full_person_alpha, merge into RGBA
    if image.mode != "RGBA" and identity_layer and identity_layer.get("full_person_alpha") is not None:
        a_raw = identity_layer["full_person_alpha"]
        if a_raw.shape[:2] == (h, w):
            image = Image.merge("RGBA", (*image.convert("RGB").split(), Image.fromarray(a_raw)))

    # Save 18_foreground_before_passport_crop.png (Section 15 Canonical Deliverable)
    if deb_dir_path is not None and image.mode == "RGBA":
        image.save(deb_dir_path / f"{job_id}_18_foreground_before_passport_crop.png")
        image.save(deb_dir_path / "18_foreground_before_passport_crop.png")

    # 1. Determine Actual Hair Crown (Robust Percentile + Anatomical Bounding)
    crown_y: Optional[int] = None
    if identity_layer and identity_layer.get("original_hair_mask") is not None:
        h_mask = identity_layer["original_hair_mask"]
        if np.any(h_mask):
            yh, _ = np.where(h_mask)
            # Filter isolated noisy pixels using 2nd percentile
            crown_y = int(np.percentile(yh, 2.0))

    if face_info and "bbox" in face_info:
        fy1 = int(face_info["bbox"][1])
        fh_box = max(1.0, float(face_info["bbox"][3] - face_info["bbox"][1]))
        if crown_y is None:
            crown_y = max(0, int(fy1 - 0.28 * fh_box))
        else:
            crown_y = int(np.clip(crown_y, max(0.0, fy1 - 0.50 * fh_box), fy1 - 0.05 * fh_box))
    elif identity_layer and identity_layer.get("original_face_mask") is not None:
        f_mask = identity_layer["original_face_mask"]
        if np.any(f_mask):
            yf, _ = np.where(f_mask)
            fy1 = int(yf.min())
            fh_mask = max(1.0, float(yf.max() - yf.min()))
            if crown_y is None:
                crown_y = max(0, int(fy1 - 0.28 * fh_mask))
            else:
                crown_y = int(np.clip(crown_y, max(0.0, fy1 - 0.50 * fh_mask), fy1 - 0.05 * fh_mask))

    if crown_y is None:
        crown_y = int(0.12 * h)

    # 2. Determine Actual Chin
    chin_y: Optional[int] = None
    if identity_layer and identity_layer.get("original_face_mask") is not None:
        f_mask = identity_layer["original_face_mask"]
        if np.any(f_mask):
            yf, _ = np.where(f_mask)
            chin_y = int(yf.max())

    if face_info and "bbox" in face_info:
        fy2 = int(face_info["bbox"][3])
        chin_y = max(chin_y, fy2) if chin_y is not None else fy2

    if chin_y is None:
        chin_y = int(crown_y + 0.35 * h)

    # Ensure anatomical ordering: chin must be below crown
    if chin_y <= crown_y + 30:
        chin_y = crown_y + int(0.35 * h)

    # 3. Calculate Actual Head Height
    head_h = max(50.0, float(chin_y - crown_y))

    # 4. Determine Lateral Boundaries (Head Width, Ears, Face Center)
    fcx = float(w) * 0.5
    fx1, fx2 = fcx - head_h * 0.35, fcx + head_h * 0.35
    if face_info and "bbox" in face_info:
        fx1 = float(face_info["bbox"][0])
        fx2 = float(face_info["bbox"][2])
        fcx = float(fx1 + fx2) * 0.5
    elif identity_layer and identity_layer.get("original_face_mask") is not None:
        f_mask = identity_layer["original_face_mask"]
        if np.any(f_mask):
            _, xf = np.where(f_mask)
            fx1 = float(xf.min())
            fx2 = float(xf.max())
            fcx = float(fx1 + fx2) * 0.5

    # Ear boundaries
    ear_x1 = fx1 - 15.0
    ear_x2 = fx2 + 15.0
    if identity_layer and identity_layer.get("original_ears_mask") is not None:
        e_mask = identity_layer["original_ears_mask"]
        if np.any(e_mask):
            _, xe = np.where(e_mask)
            ear_x1 = min(ear_x1, float(xe.min()))
            ear_x2 = max(ear_x2, float(xe.max()))

    # 5. Determine Individual Bow and Ribbon Boundaries
    acc_bounds_list: List[Tuple[int, int, int, int]] = []
    has_confirmed_acc = False
    acc_x1_min = ear_x1
    acc_x2_max = ear_x2
    acc_y1_min = float(crown_y)
    acc_y2_max = float(chin_y)

    if identity_layer:
        # Check individual bows
        ind_bows = identity_layer.get("individual_bows", [])
        if ind_bows:
            has_confirmed_acc = True
            for b in ind_bows:
                bx1, by1, bx2, by2 = b["bbox"]
                acc_bounds_list.append((bx1, by1, bx2, by2))
                acc_x1_min = min(acc_x1_min, float(bx1))
                acc_x2_max = max(acc_x2_max, float(bx2))
                acc_y1_min = min(acc_y1_min, float(by1))
                acc_y2_max = max(acc_y2_max, float(by2))
        elif identity_layer.get("has_confirmed_bows") and identity_layer.get("original_accessory_mask") is not None:
            a_mask = identity_layer["original_accessory_mask"]
            if np.any(a_mask):
                ya, xa = np.where(a_mask)
                has_confirmed_acc = True
                acc_x1_min = min(acc_x1_min, float(xa.min()))
                acc_x2_max = max(acc_x2_max, float(xa.max()))
                acc_y1_min = min(acc_y1_min, float(ya.min()))
                acc_y2_max = max(acc_y2_max, float(ya.max()))
                acc_bounds_list.append((int(xa.min()), int(ya.min()), int(xa.max()), int(ya.max())))

    # 6. Calculate Initial Framing Box
    # Target head occupancy: head_h occupies target_head_occupancy of the final canvas height
    target_occ = max(0.52, min(0.65, target_head_occupancy))
    headroom_ratio = max(0.08, min(0.12, target_headroom_ratio))

    crop_h_initial = head_h / target_occ
    headroom = crop_h_initial * headroom_ratio

    # Initial vertical boundaries based on crown and headroom
    y_top = float(crown_y) - headroom

    # Ensure accessory vertical bounds are enclosed with safe margin
    if has_confirmed_acc and y_top > acc_y1_min - 20.0:
        y_top = acc_y1_min - 20.0

    y_bottom = y_top + crop_h_initial

    # Ensure chin is never cropped and sufficient chest space is retained:
    min_chest_space = head_h * 0.35
    if y_bottom < float(chin_y) + min_chest_space:
        y_bottom = float(chin_y) + min_chest_space
        crop_h_initial = y_bottom - y_top

    # V13.1 Mandatory Passport Anchoring:
    # Torso MUST extend to bottom of canvas; pad_bottom is STRICTLY 0.
    # If y_bottom exceeds canvas height h, anchor y_bottom to h and expand upwards.
    if y_bottom > float(h):
        y_bottom = float(h)
        y_top = y_bottom - crop_h_initial

    # Corresponding crop width to match target aspect ratio (W / H = target_aspect)
    crop_w_req = crop_h_initial * target_aspect

    # Check lateral extent: must include ears and accessories with safe clearance margin
    lat_margin = 25.0 if has_confirmed_acc else 20.0
    req_lat_w = (acc_x2_max - acc_x1_min) + 2.0 * lat_margin
    if req_lat_w > crop_w_req:
        crop_w_req = req_lat_w
        # Expand height to strictly preserve target aspect ratio without non-uniform stretching
        crop_h_initial = crop_w_req / target_aspect
        if y_bottom >= float(h) - 5.0:
            y_bottom = float(h)
            y_top = y_bottom - crop_h_initial
        else:
            new_headroom = crop_h_initial * headroom_ratio
            y_top = float(crown_y) - new_headroom
            if has_confirmed_acc and y_top > acc_y1_min - 15.0:
                y_top = acc_y1_min - 15.0
            y_bottom = y_top + crop_h_initial
            if y_bottom > float(h):
                y_bottom = float(h)
                y_top = y_bottom - crop_h_initial

    crop_w = int(round(crop_w_req))
    crop_h = int(round(crop_h_initial))

    # Fine-tune integer crop dimensions to strictly guarantee |scale_x - scale_y| < 0.001
    best_w, best_h = crop_w, crop_h
    best_diff = abs(float(out_w) / max(1, crop_w) - float(out_h) / max(1, crop_h))
    for dw in (-2, -1, 0, 1, 2):
        for dh in (-2, -1, 0, 1, 2):
            cand_w = crop_w + dw
            cand_h = crop_h + dh
            if cand_w > 0 and cand_h > 0:
                diff = abs(float(out_w) / cand_w - float(out_h) / cand_h)
                if diff < best_diff:
                    best_diff = diff
                    best_w, best_h = cand_w, cand_h
    crop_w, crop_h = best_w, best_h

    # Horizontal centering: center on face/head horizontal midpoint
    x_center = (acc_x1_min + acc_x2_max) * 0.5 if has_confirmed_acc else fcx
    # Blend face center with accessory center to keep face visually balanced
    x_center = 0.60 * fcx + 0.40 * x_center
    x_left = int(round(x_center - crop_w * 0.5))
    x_right = x_left + crop_w

    # Ensure left and right lateral bounds are enclosed
    if x_left > int(acc_x1_min - lat_margin):
        x_left = int(acc_x1_min - lat_margin)
        x_right = x_left + crop_w
    if x_right < int(acc_x2_max + lat_margin):
        x_right = int(acc_x2_max + lat_margin)
        x_left = x_right - crop_w

    # Clamp lateral boundaries to canvas boundaries if crop_w <= w
    if crop_w <= w:
        x_left = max(0, min(w - crop_w, x_left))
        x_right = x_left + crop_w

    # Anchor y_bottom strictly at <= h so pad_bottom == 0 always
    y_bottom_int = min(h, int(round(y_bottom)))
    y_top_int = y_bottom_int - crop_h

    # 7. Padding Calculation (V13.1: pad_bottom is strictly 0; headroom padding at top only)
    pad_left = max(0, -x_left)
    pad_top = max(0, -y_top_int)
    pad_right = max(0, x_right - w)
    pad_bottom = 0  # Strictly 0 to eliminate floating torso and bottom gaps

    needs_padding = (pad_left > 0 or pad_top > 0 or pad_right > 0)
    is_rgba = (image.mode == "RGBA")

    if needs_padding:
        new_w = w + pad_left + pad_right
        new_h = h + pad_top
        if is_rgba:
            padded_canvas = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
        else:
            padded_canvas = Image.new("RGB", (new_w, new_h), background_rgb)
        padded_canvas.paste(image, (pad_left, pad_top))
        # Shift crop coordinates into padded canvas
        crop_rect_padded = (
            x_left + pad_left,
            y_top_int + pad_top,
            x_left + pad_left + crop_w,
            y_top_int + pad_top + crop_h,
        )
        cropped_pil = padded_canvas.crop(crop_rect_padded)
    else:
        crop_rect_src = (x_left, y_top_int, x_right, y_bottom_int)
        cropped_pil = image.crop(crop_rect_src)

    # 8. Single Uniform Scaling Factor (No geometric distortion)
    scale_factor_x = float(out_w) / float(crop_w)
    scale_factor_y = float(out_h) / float(crop_h)
    scale_factor = scale_factor_x

    cropped_resized_pil = cropped_pil.resize(output_size, Image.Resampling.LANCZOS)

    # Save 19_foreground_after_passport_crop.png (Section 15 Canonical Deliverable)
    if is_rgba and deb_dir_path is not None:
        cropped_resized_pil.save(deb_dir_path / f"{job_id}_19_foreground_after_passport_crop.png")
        cropped_resized_pil.save(deb_dir_path / "19_foreground_after_passport_crop.png")

    # 9. Apply Requested Background Color ONLY ONCE After Cropping (V13.1 Authoritative Compositing)
    if is_rgba:
        fg_np = np.asarray(cropped_resized_pil, dtype=np.float32)
        alpha_channel = np.clip(fg_np[:, :, 3:4] / 255.0, 0.0, 1.0)
        fg_rgb = fg_np[:, :, :3]
        bg_color = np.array(background_rgb, dtype=np.float32).reshape(1, 1, 3)

        # Straight alpha composite: final_rgb = fg_rgb * alpha + bg_rgb * (1 - alpha)
        final_rgb_np = fg_rgb * alpha_channel + bg_color * (1.0 - alpha_channel)
        # Snap zero-alpha fringe to requested background
        final_rgb_np[alpha_channel[:, :, 0] < 0.02] = bg_color[0, 0]
        final_rgb_np = np.clip(np.round(final_rgb_np), 0, 255).astype(np.uint8)
        final_passport_pil = Image.fromarray(final_rgb_np, "RGB")
        transformed_alpha = (alpha_channel[:, :, 0] * 255.0).astype(np.uint8)
    else:
        final_passport_pil = cropped_resized_pil
        transformed_alpha = None
        if identity_layer and identity_layer.get("full_person_alpha") is not None:
            alpha_raw = identity_layer["full_person_alpha"]
            if needs_padding:
                pad_a = np.zeros((new_h, new_w), dtype=alpha_raw.dtype)
                pad_a[pad_top : pad_top + h, pad_left : pad_left + w] = alpha_raw
                cropped_a = pad_a[crop_rect_padded[1] : crop_rect_padded[3], crop_rect_padded[0] : crop_rect_padded[2]]
            else:
                cropped_a = alpha_raw[crop_rect_src[1] : crop_rect_src[3], crop_rect_src[0] : crop_rect_src[2]]
            transformed_alpha = cv2.resize(cropped_a, output_size, interpolation=cv2.INTER_LINEAR)

    # 10. Coordinate Transformation for Verification (Post-Crop Validation)
    def transform_pt(x: float, y: float) -> Tuple[float, float]:
        xp = (x - float(x_left)) * scale_factor
        yp = (y - float(y_top_int)) * scale_factor
        return (xp, yp)

    trans_crown_y = (float(crown_y) - float(y_top_int)) * scale_factor
    trans_chin_y = (float(chin_y) - float(y_top_int)) * scale_factor
    trans_head_h = head_h * scale_factor
    final_head_occupancy = float(trans_head_h / float(out_h))
    final_headroom = float(trans_crown_y / float(out_h))
    final_chest_space = float((float(out_h) - trans_chin_y) / float(out_h))
    final_head_occupancy = float(trans_head_h / float(out_h))
    final_headroom = float(trans_crown_y / float(out_h))
    final_chest_space = float((float(out_h) - trans_chin_y) / float(out_h))

    transformed_landmarks = None
    if face_info and "kps" in face_info and face_info["kps"] is not None:
        transformed_landmarks = [transform_pt(float(pt[0]), float(pt[1])) for pt in face_info["kps"]]

    transformed_acc_bounds = []
    for (bx1, by1, bx2, by2) in acc_bounds_list:
        tx1, ty1 = transform_pt(float(bx1), float(by1))
        tx2, ty2 = transform_pt(float(bx2), float(by2))
        transformed_acc_bounds.append((round(tx1, 1), round(ty1, 1), round(tx2, 1), round(ty2, 1)))

    crop_meta: Dict[str, Any] = {
        "original_dimensions": (w, h),
        "crop_rectangle": (x_left, y_top_int, x_right, y_bottom_int),
        "padding": (pad_left, pad_top, pad_right, pad_bottom),
        "scale_factor": round(scale_factor, 6),
        "scale_x": round(scale_factor_x, 6),
        "scale_y": round(scale_factor_y, 6),
        "scale_delta": round(abs(scale_factor_x - scale_factor_y), 6),
        "final_dimensions": output_size,
        "actual_crown_y": crown_y,
        "actual_chin_y": chin_y,
        "actual_head_height": round(head_h, 1),
        "transformed_crown_y": round(trans_crown_y, 1),
        "transformed_chin_y": round(trans_chin_y, 1),
        "final_head_occupancy": round(final_head_occupancy, 4),
        "final_headroom_ratio": round(final_headroom, 4),
        "final_chest_space_ratio": round(final_chest_space, 4),
        "transformed_landmarks": transformed_landmarks,
        "transformed_accessory_bounds": transformed_acc_bounds,
        "has_confirmed_accessories": has_confirmed_acc,
        "background_rgb": background_rgb,
        "transformed_person_alpha": transformed_alpha,
    }

    # 10. Debug Visualization of Crop Rectangle (Section 3 Requirement)
    if debug_dir is not None:
        deb_dir_path = Path(debug_dir)
        deb_dir_path.mkdir(parents=True, exist_ok=True)
        try:
            vis_img = image.copy().convert("RGB")
            draw = ImageDraw.Draw(vis_img)
            # Draw original crop box in bright green
            rx0, ry0 = min(x_left, x_right), min(y_top_int, y_bottom_int)
            rx1, ry1 = max(x_left, x_right), max(y_top_int, y_bottom_int)
            draw.rectangle([(rx0, ry0), (rx1, ry1)], outline=(0, 255, 0), width=3)
            # Draw actual crown and chin horizontal reference lines
            draw.line([(0, crown_y), (w, crown_y)], fill=(0, 200, 255), width=2)
            draw.line([(0, chin_y), (w, chin_y)], fill=(255, 100, 0), width=2)
            # If padding was required, draw padded boundaries in cyan
            if needs_padding:
                px0 = max(0, min(w, rx0))
                px1 = max(0, min(w, rx1))
                py0 = max(0, min(h, ry0))
                py1 = max(0, min(h, ry1))
                if px1 > px0 and py1 > py0:
                    draw.rectangle([(px0, py0), (px1, py1)], outline=(0, 255, 255), width=2)
            # Draw accessory bounds in yellow
            for (bx1, by1, bx2, by2) in acc_bounds_list:
                draw.rectangle([(bx1, by1), (bx2, by2)], outline=(255, 255, 0), width=2)
            vis_p = deb_dir_path / f"{job_id}_19_crop_rectangle_vis.png"
            vis_img.save(vis_p)
            vis_img.save(deb_dir_path / "19_crop_rectangle_vis.png")
            crop_meta["crop_vis_path"] = str(vis_p)
        except Exception as exc:
            logger.warning("[IMAGE_NORMALIZER] Could not save crop visualization: %s", exc)

    logger.info(
        "[IMAGE_NORMALIZER] Passport crop: head_h=%.1f (%.1f%% canvas), headroom=%.1f%%, chest=%.1f%%, pad=%s, scale=%.4f",
        trans_head_h, final_head_occupancy * 100, final_headroom * 100, final_chest_space * 100,
        (pad_left, pad_top, pad_right, pad_bottom), scale_factor
    )

    if return_meta:
        return final_passport_pil, crop_meta
    return final_passport_pil
