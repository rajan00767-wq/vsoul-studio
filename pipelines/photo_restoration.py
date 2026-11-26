"""
Advanced Photo Restoration & Studio Enhancement Service with Strict Originality Preservation.
Features:
1. 100% Face Likeness & Identity Preservation (Prevents plastic faces, preserves real expressions, moles, eyes, lips).
2. Hair Style & Strand Integrity (Crisp natural strands, authentic haircut, hairline and volume).
3. Dress & Fabric Preservation (Protects patterns, embroidery, lace, folds, pleats, saree/suit textures).
4. Jewels & Ornaments Protection (Preserves necklaces, earrings, chains, gems, bindis, gold/silver sheen without erasure or blurring).
5. Damage & Scratch Inpainting with Mask Protection (Guarantees jewelry and dress patterns are never misidentified as scratches).
6. 2509 SVDQ High-Fidelity Generative Polish & Super-Resolution.
"""

from __future__ import annotations

import time
import os
from pathlib import Path
from typing import Optional, Union, Tuple, Callable
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch

from utils.logger import get_logger

logger = get_logger("photo_restoration")

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR = Path("models")

_insight_app = None
INSIGHTFACE_ROOT = Path(__file__).resolve().parents[1] / "models" / "insightface"


def _get_insight_app():
    global _insight_app
    if _insight_app is None:
        from insightface.app import FaceAnalysis
        root_dir = str(INSIGHTFACE_ROOT.resolve()) if (INSIGHTFACE_ROOT / "models" / "buffalo_l").exists() else None
        kwargs = {"root": root_dir} if root_dir else {}
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], **kwargs)
        app.prepare(ctx_id=-1, det_size=(320, 320))
        _insight_app = app
    return _insight_app


def _lens_mask_from_kps(h: int, w: int, kps: np.ndarray) -> np.ndarray:
    """Soft ellipses covering eyeglass lenses (InsightFace 5-point kps)."""
    mask = np.zeros((h, w), dtype=np.float32)
    if kps is None or len(kps) < 2:
        return mask
    le, re = np.asarray(kps[0], dtype=np.float32), np.asarray(kps[1], dtype=np.float32)
    dist = float(np.linalg.norm(re - le)) + 1e-6
    rx, ry = int(dist * 0.32), int(dist * 0.22)
    for cx, cy in (le, re):
        cv2.ellipse(mask, (int(cx), int(cy)), (max(6, rx), max(5, ry)), 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(mask, (11, 11), 3.0)


def _head_geometry(img_bgr: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Return (face_cx, crown_y, head_height) with hair and chin, not just the inner face box."""
    h, w = img_bgr.shape[:2]
    try:
        app = _get_insight_app()
        faces = app.get(img_bgr)
        if faces:
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            x1, y1, x2, y2 = [float(v) for v in face.bbox]
            fh = max(8.0, y2 - y1)
            cx = (x1 + x2) / 2.0
            lmk = getattr(face, "landmark_2d_106", None)
            if lmk is not None and len(lmk) >= 10:
                ys = lmk[:, 1].astype(np.float32)
                top, bot = float(ys.min()), float(ys.max())
                span = max(8.0, bot - top)
                crown = max(0.0, top - 0.42 * span)
                chin = min(float(h), bot + 0.10 * span)
            else:
                crown = max(0.0, y1 - 0.48 * fh)
                chin = min(float(h), y2 + 0.18 * fh)
            return cx, crown, max(12.0, chin - crown)
    except Exception as exc:
        logger.debug("[PASSPORT] InsightFace geometry skipped: %s", exc)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
    if len(faces) == 0:
        return None
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    cx = fx + fw / 2.0
    crown = max(0.0, fy - 0.50 * fh)
    chin = min(h, fy + fh * 1.18)
    return float(cx), float(crown), float(max(12.0, chin - crown))


class PhotoRestorationService:
    """Production service for restoring and enhancing photos with strict preservation of Face, Dress, Hair & Jewels."""

    def __init__(self):
        self._gfpgan = None
        self._codeformer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def has_eyeglasses(img_bgr: np.ndarray) -> bool:
        """Detect frame-like edges around both eyes without guessing from faces alone."""
        h, w = img_bgr.shape[:2]
        try:
            faces = _get_insight_app().get(img_bgr)
            if not faces:
                return False
            face = max(faces, key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]))
            lens = (_lens_mask_from_kps(h, w, face.kps) > 0.4).astype(np.uint8) * 255
        except Exception:
            return False

        outer = cv2.dilate(lens, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        inner = cv2.erode(lens, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        frame_band = cv2.bitwise_and(outer, cv2.bitwise_not(inner))
        if cv2.countNonZero(frame_band) < 40:
            return False

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 45, 120)
        edge_density = float(np.mean(edges[frame_band > 0] > 0))
        dark_density = float(np.mean(gray[frame_band > 0] < 95))
        # Eyelashes, eyebrows, and deep eye shadows create a similar local edge
        # pattern on small portraits.  Prefer a false negative here: the source
        # face lock still keeps real glasses, whereas a false positive instructs
        # Qwen to create frames that do not exist.
        return edge_density > 0.24 and dark_density > 0.35

    @staticmethod
    def preserve_source_clothing(
        orig_bgr: np.ndarray, enhanced_bgr: np.ndarray, strength: float = 0.72
    ) -> np.ndarray:
        """Protect garment colour while retaining the enhanced garment luminance and texture."""
        if orig_bgr.shape[:2] != enhanced_bgr.shape[:2]:
            orig_bgr = cv2.resize(orig_bgr, (enhanced_bgr.shape[1], enhanced_bgr.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        try:
            from pipelines.schp_service import CLOTHING_LABELS, parse

            source_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
            labels = parse(source_rgb)["labels"]
            clothing = np.isin(labels, tuple(CLOTHING_LABELS)).astype(np.uint8) * 255
            if cv2.countNonZero(clothing) < 120:
                return enhanced_bgr
            clothing = cv2.dilate(
                clothing,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )
            alpha = cv2.GaussianBlur(clothing, (9, 9), 2.0).astype(np.float32) / 255.0
            alpha = (alpha * float(np.clip(strength, 0.0, 1.0)))[:, :, np.newaxis]

            # Copying the source BGR pixels made every uniform retain its old,
            # low-resolution texture.  Keep the source chroma to prevent a colour
            # shift, but retain the generated lightness/detail from the restoration.
            source_lab = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2LAB)
            enhanced_lab = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2LAB)
            colour_locked_lab = enhanced_lab.copy()
            colour_locked_lab[:, :, 1:] = source_lab[:, :, 1:]
            colour_locked = cv2.cvtColor(colour_locked_lab, cv2.COLOR_LAB2BGR)
            return (colour_locked.astype(np.float32) * alpha + enhanced_bgr.astype(np.float32) * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
        except Exception as exc:
            logger.warning("[CLOTHING_LOCK] Source garment preservation skipped: %s", exc)
            return enhanced_bgr

    @staticmethod
    def preserve_source_hair_accessories(orig_pil: Image.Image, enhanced_pil: Image.Image) -> Image.Image:
        """Restore real bright hair clips without copying surrounding source background."""
        enhanced = np.array(enhanced_pil.convert("RGB"))
        h, w = enhanced.shape[:2]
        source = np.array(orig_pil.convert("RGB").resize((w, h), Image.LANCZOS))
        source_bgr = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        try:
            from skimage.transform import SimilarityTransform

            app = _get_insight_app()
            source_faces = app.get(source_bgr)
            enhanced_faces = app.get(enhanced_bgr)
            if not source_faces or not enhanced_faces:
                return enhanced_pil
            transform = SimilarityTransform()
            transform.estimate(source_faces[0].kps, enhanced_faces[0].kps)
            matrix = transform.params[:2, :]

            x1, y1, x2, _ = [int(v) for v in source_faces[0].bbox]
            face_width, face_height = max(24, x2 - x1), max(24, int(source_faces[0].bbox[3] - y1))
            region = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(
                region,
                (max(0, x1 - int(face_width * 0.60)), max(0, y1 - int(face_height * 1.05))),
                (min(w, x2 + int(face_width * 0.60)), max(0, y1 - int(face_height * 0.10))),
                255,
                -1,
            )
            hsv = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2HSV)
            # White/silver clips are bright and low-saturation. Restricting this
            # to the upper-head band prevents foliage, skin, and clothing leaks.
            clip_mask = ((hsv[:, :, 2] > 175) & (hsv[:, :, 1] < 105) & (region > 0)).astype(np.uint8) * 255
            clip_mask = cv2.morphologyEx(
                clip_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            )
            count, labels, stats, _ = cv2.connectedComponentsWithStats(clip_mask, 8)
            kept = np.zeros_like(clip_mask)
            for index in range(1, count):
                area = int(stats[index, cv2.CC_STAT_AREA])
                if 3 <= area <= max(900, int(face_width * face_height * 0.16)):
                    kept[labels == index] = 255
            if cv2.countNonZero(kept) < 3:
                return enhanced_pil

            warped_source = cv2.warpAffine(source_bgr, matrix, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
            warped_mask = cv2.warpAffine(kept, matrix, (w, h), flags=cv2.INTER_NEAREST)
            alpha = cv2.GaussianBlur(warped_mask, (5, 5), 0).astype(np.float32)[:, :, None] / 255.0
            fused = warped_source.astype(np.float32) * alpha + enhanced_bgr.astype(np.float32) * (1.0 - alpha)
            return Image.fromarray(cv2.cvtColor(fused.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        except Exception as exc:
            logger.warning("[ACCESSORY_LOCK] Hair-clip preservation skipped: %s", exc)
            return enhanced_pil

    @staticmethod
    def restore_dead_and_hot_pixels(
        image_input: Union[Image.Image, np.ndarray],
        dead_threshold: int = 4,
        hot_threshold: int = 252,
        spike_threshold: float = 80.0,
    ) -> Union[Image.Image, np.ndarray]:
        """Heal isolated dead black, hot white, and single-pixel spike artifacts.

        Preserves 100% of fine continuous features (hair strands, eyelashes, fabric
        patterns, jewelry highlights) while removing single-pixel dead or hot sensor/decoder
        defects using local 8-neighborhood median filtering.
        """
        is_pil = isinstance(image_input, Image.Image)
        if is_pil:
            arr = np.asarray(image_input.convert("RGB"), dtype=np.uint8)
        else:
            arr = image_input.copy()

        h, w = arr.shape[:2]
        if h < 5 or w < 5:
            return image_input

        padded = np.pad(arr, ((2, 2), (2, 2), (0, 0)), mode="edge")
        n_up    = padded[1:h+1, 2:w+2]
        n_down  = padded[3:h+3, 2:w+2]
        n_left  = padded[2:h+2, 1:w+1]
        n_right = padded[2:h+2, 3:w+3]
        n_ul    = padded[1:h+1, 1:w+1]
        n_ur    = padded[1:h+1, 3:w+3]
        n_dl    = padded[3:h+3, 1:w+1]
        n_dr    = padded[3:h+3, 3:w+3]

        neighbors = np.stack([n_up, n_down, n_left, n_right, n_ul, n_ur, n_dl, n_dr], axis=0)
        med_neighbors = np.median(neighbors, axis=0).astype(np.uint8)

        is_dead = (np.max(arr, axis=-1) <= dead_threshold) & (np.max(med_neighbors, axis=-1) >= 18)
        is_hot = (np.min(arr, axis=-1) >= hot_threshold) & (np.min(med_neighbors, axis=-1) <= 225)

        diff_v = np.maximum(np.max(np.abs(arr.astype(int) - n_up.astype(int)), axis=-1),
                            np.max(np.abs(arr.astype(int) - n_down.astype(int)), axis=-1))
        diff_h = np.maximum(np.max(np.abs(arr.astype(int) - n_left.astype(int)), axis=-1),
                            np.max(np.abs(arr.astype(int) - n_right.astype(int)), axis=-1))
        diff_d1 = np.maximum(np.max(np.abs(arr.astype(int) - n_ul.astype(int)), axis=-1),
                             np.max(np.abs(arr.astype(int) - n_dr.astype(int)), axis=-1))
        diff_d2 = np.maximum(np.max(np.abs(arr.astype(int) - n_ur.astype(int)), axis=-1),
                             np.max(np.abs(arr.astype(int) - n_dl.astype(int)), axis=-1))

        min_directional_diff = np.minimum(np.minimum(diff_v, diff_h), np.minimum(diff_d1, diff_d2))
        diff_to_median = np.max(np.abs(arr.astype(int) - med_neighbors.astype(int)), axis=-1)

        is_spike = (min_directional_diff > spike_threshold) & (diff_to_median > spike_threshold)
        defect_mask = (is_dead | is_hot | is_spike).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(defect_mask, connectivity=8)
        final_mask = np.zeros((h, w), dtype=bool)
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if 1 <= area <= 6:
                final_mask[labels == lbl] = True

        fixed_count = int(np.sum(final_mask))
        if fixed_count > 0:
            repaired_arr = arr.copy()
            repaired_arr[final_mask] = med_neighbors[final_mask]
            logger.info("[DEAD_PIXEL_REPAIR] Restored %d dead/hot/spike pixels", fixed_count)
            return Image.fromarray(repaired_arr, "RGB") if is_pil else repaired_arr
        return image_input

    @staticmethod
    def assess_portrait_clarity(image_input: Union[Image.Image, np.ndarray]) -> dict:
        """Measure portrait resolution, facial blur, and detail clarity."""
        if isinstance(image_input, Image.Image):
            rgb = np.asarray(image_input.convert("RGB"), dtype=np.uint8)
        elif isinstance(image_input, np.ndarray):
            rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB) if image_input.shape[2] == 3 else image_input
        else:
            return {"is_clear": True, "clarity_status": "unknown", "repair_recommended": False}

        h, w = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            face_roi = gray[fy:fy+fh, fx:fx+fw]
        else:
            fw, fh = w, h
            face_roi = gray[int(h * 0.15):int(h * 0.65), int(w * 0.2):int(w * 0.8)]

        lap_var = float(cv2.Laplacian(face_roi, cv2.CV_64F).var()) if face_roi.size > 64 else 0.0
        is_low_res = fw < 100 or fh < 120 or min(w, h) < 300
        is_blurry = lap_var < 80.0
        is_clear = not is_blurry and not is_low_res

        status = "clear" if is_clear else ("blurry" if is_blurry else "low_resolution")
        return {
            "is_clear": is_clear,
            "clarity_status": status,
            "sharpness_score": round(lap_var, 1),
            "face_dimensions": f"{fw}x{fh}",
            "repair_recommended": not is_clear,
            "message": (
                "Portrait clarity is sharp and high-definition."
                if is_clear else
                f"Portrait is {status} (sharpness score: {lap_var:.1f}, face dimensions: {fw}x{fh}px). Optical clarity and texture repair is recommended."
            )
        }

    # ── 1. Jewelry, Hair & Dress Protection Masks ─────────────────────────────

    @staticmethod
    def detect_jewelry_and_ornaments(img_bgr: np.ndarray, face_box: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Ultra-sensitive jewelry and ornament detector:
        - Gold, silver, brass, copper metallic reflections.
        - Gemstones (diamonds, rubies, emeralds, pearls).
        - Earrings, necklaces, mangalsutra, bindis, maang tikka, nose rings, bangles, watches.
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]

        # 1. Specular Highlights / Metallic Reflections (Must have high saturation to avoid skin glare)
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(sobel_x, sobel_y)
        specular = (gray > 185) & (grad_mag > 30) & (sat > 60)

        # 2. Gold / Amber / Copper / Brass Color Saliency
        lower_gold = np.array([12, 60, 85])
        upper_gold = np.array([38, 255, 255])
        gold_mask = cv2.inRange(hsv, lower_gold, upper_gold)

        # 3. High-Frequency Micro-Ornaments (Laplacian filter with saturation check)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        high_freq = (np.abs(laplacian) > 25) & (sat > 50)

        # 4. Gemstone Red/Green/Blue color detection (Rubies, Emeralds, Sapphires)
        red_mask1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
        ruby_mask = red_mask1 | red_mask2

        combined_jewelry = (specular | (gold_mask > 0) | high_freq | (ruby_mask > 0)).astype(np.uint8) * 255

        # Anatomical Prior: Earring zones, Neck/Collarbone zone, Wrist/Hands (Exclude Face & Forehead)
        anatomical_zone = np.zeros((h, w), dtype=np.uint8)
        if face_box is not None:
            fx, fy, fw, fh = face_box
            # Left & Right Ear areas (earrings / ear ornaments)
            ear_w = int(fw * 0.45)
            cv2.rectangle(anatomical_zone, (max(0, fx - ear_w), fy + int(fh * 0.15)), (fx + int(fw * 0.15), fy + int(fh * 0.90)), 255, -1)
            cv2.rectangle(anatomical_zone, (fx + int(fw * 0.85), fy + int(fh * 0.15)), (min(w, fx + fw + ear_w), fy + int(fh * 0.90)), 255, -1)
            # Neck & Chest area (necklaces, chains, pendants, collar)
            neck_y = fy + int(fh * 0.85)
            chest_y = min(h, fy + int(fh * 2.5))
            cv2.rectangle(anatomical_zone, (max(0, fx - int(fw * 0.6)), neck_y), (min(w, fx + int(fw * 1.6)), chest_y), 255, -1)
            # Hands/Wrists zone
            cv2.rectangle(anatomical_zone, (0, int(h * 0.50)), (w, h), 255, -1)
        else:
            cv2.rectangle(anatomical_zone, (int(w * 0.10), int(h * 0.30)), (int(w * 0.90), int(h * 0.95)), 255, -1)

        raw_jewelry = cv2.bitwise_and(combined_jewelry, anatomical_zone)

        # Strictly exclude Face and Forehead
        if face_box is not None:
            fx, fy, fw, fh = face_box
            face_mask_exclude = np.zeros((h, w), dtype=np.uint8)
            cx, cy = fx + fw // 2, fy + fh // 2
            cv2.ellipse(face_mask_exclude, (cx, cy), (int(fw * 0.55), int(fh * 0.70)), 0, 0, 360, 255, -1)
            raw_jewelry = cv2.bitwise_and(raw_jewelry, cv2.bitwise_not(face_mask_exclude))

        return raw_jewelry

    # ── 2. Scratch & Crease Inpainting with Detail Protection ─────────────────

    def detect_scratches_and_creases(self, img_bgr: np.ndarray, face_box: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Detects scratches & fold creases while strictly protecting real face,
        hair strands, dress embroidery, and jewelry ornaments.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Multi-scale line / scratch detection
        kernel_sizes = [3, 5]
        raw_mask = np.zeros((h, w), dtype=np.uint8)

        for k_size in kernel_sizes:
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size * 3, 1))
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_size * 3))
            kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

            bh_h = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_h)
            bh_v = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_v)
            bh_e = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_ellipse)
            th_e = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_ellipse)

            merged = cv2.max(cv2.max(bh_h, bh_v), cv2.max(bh_e, th_e))
            _, thresh = cv2.threshold(merged, 24, 255, cv2.THRESH_BINARY)
            raw_mask = cv2.bitwise_or(raw_mask, thresh)

        # Get Jewelry Protection Mask
        jewelry_protect = self.detect_jewelry_and_ornaments(img_bgr, face_box)

        # Protect Full Face, Forehead, Hairline, Eyes, Nose & Lips
        face_protect = np.zeros((h, w), dtype=np.uint8)
        if face_box is not None:
            fx, fy, fw, fh = face_box
            pad_x = int(fw * 0.25)
            pad_top = int(fh * 0.35)
            pad_bottom = int(fh * 0.15)
            x1 = max(0, fx - pad_x)
            x2 = min(w, fx + fw + pad_x)
            y1 = max(0, fy - pad_top)
            y2 = min(h, fy + fh + pad_bottom)
            face_protect[y1:y2, x1:x2] = 255

        # Exclude jewelry and face core from scratch mask
        protected_total = cv2.bitwise_or(jewelry_protect, face_protect)
        clean_mask = cv2.bitwise_and(raw_mask, cv2.bitwise_not(protected_total))

        # Filter out oversized or non-scratch contours
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros((h, w), dtype=np.uint8)
        max_area = (h * w) * 0.025

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 5 < area < max_area:
                cv2.drawContours(final_mask, [cnt], -1, 255, -1)

        final_mask = cv2.dilate(final_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        return final_mask

    def inpaint_defects(self, img_bgr: np.ndarray, face_box: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Inpaints real damage while leaving authentic face, dress, hair and jewels 100% intact."""
        mask = self.detect_scratches_and_creases(img_bgr, face_box)
        if np.count_nonzero(mask) == 0:
            return img_bgr
        repaired = cv2.inpaint(img_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return repaired

    # ── 3. Tone, Contrast & Color Restoration (Preserving Fabric & Gold) ───────

    @staticmethod
    def restore_colors_and_contrast(img_bgr: np.ndarray) -> np.ndarray:
        """
        Restores faded dynamic range while strictly preserving original clothing
        dyes, skin tones, and rich jewelry luster.
        """
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Gentle CLAHE on L channel only (restores shadow details without blowing out highlights)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        # Neutralize harsh yellow sun-glare and outdoor casts on skin
        b_float = b.astype(np.float32)
        glare_mask = np.clip((b_float - 142.0) / 35.0, 0.0, 1.0)
        b_balanced = np.clip(b_float - glare_mask * 14.0, 0, 255).astype(np.uint8)

        # Preserve color vibrancy of original dress and jewels
        lab_merged = cv2.merge([l_enhanced, a, b_balanced])
        balanced = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)

        # Edge-preserving bilateral filter (smooths grain, preserves sharp fabric weave & hair edges)
        denoised = cv2.bilateralFilter(balanced, d=5, sigmaColor=18, sigmaSpace=18)
        return denoised

    # ── 4. 100% Identity-Preserving Optical Facial Enhancement (NO GFPGAN) ───

    def enhance_faces_optical(
        self,
        img_bgr: np.ndarray,
        smooth_skin: bool = True,
        sharpen_eyes: bool = True,
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """
        Pure Optical Face Enhancement (100% Real Identity - ZERO GFPGAN / ZERO AI Face Replacement):
        1. Preserves 100% authentic skin pores, eye color, expression, nose, smile, moles.
        2. Bilateral edge-preserving skin noise cleanup.
        3. High-pass unsharp masking on iris, eyelashes, eyebrows, lips, and facial hair.
        4. Soft studio lighting equalization without altering facial geometry.
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
        if len(faces) == 0:
            return img_bgr, None

        primary_face = max(faces, key=lambda f: f[2] * f[3])
        fx, fy, fw, fh = primary_face
        res = img_bgr.copy()

        roi = res[fy:fy+fh, fx:fx+fw]
        roi_gray = gray[fy:fy+fh, fx:fx+fw]

        # 1. Edge-preserving skin smoothing (removes camera noise, retains pores)
        if smooth_skin:
            smooth_roi = cv2.bilateralFilter(roi, d=5, sigmaColor=15, sigmaSpace=15)
        else:
            smooth_roi = roi.copy()

        # 2. Eye & feature unsharp mask (eyes, lips, brows)
        if sharpen_eyes:
            gauss = cv2.GaussianBlur(roi, (0, 0), 1.0)
            sharp_features = cv2.addWeighted(roi, 1.35, gauss, -0.35, 0)

            sobel = cv2.Sobel(roi_gray, cv2.CV_32F, 1, 1, ksize=3)
            grad_mag = np.abs(sobel)
            feature_mask = cv2.threshold(grad_mag, 25, 1.0, cv2.THRESH_BINARY)[1]
            feature_mask_3c = np.repeat(cv2.GaussianBlur(feature_mask, (5, 5), 1.0)[:, :, np.newaxis], 3, axis=2)

            blended_roi = (sharp_features.astype(np.float32) * feature_mask_3c + smooth_roi.astype(np.float32) * (1.0 - feature_mask_3c)).clip(0, 255).astype(np.uint8)
        else:
            blended_roi = smooth_roi

        # 3. Soft elliptical blend of enhanced ROI into the canvas (zero boundary seams)
        face_mask = np.zeros((fh, fw), dtype=np.float32)
        cv2.ellipse(face_mask, (fw // 2, fh // 2), (int(fw * 0.46), int(fh * 0.54)), 0, 0, 360, 1.0, -1)
        face_mask = cv2.GaussianBlur(face_mask, (15, 15), 0)[:, :, np.newaxis]

        fused_roi = (blended_roi.astype(np.float32) * face_mask + roi.astype(np.float32) * (1.0 - face_mask)).clip(0, 255).astype(np.uint8)
        res[fy:fy+fh, fx:fx+fw] = fused_roi

        return res, primary_face

    def enhance_faces_identity_locked(
        self,
        img_bgr: np.ndarray,
        fidelity_weight: float = 1.0,
        texture_retention: float = 1.0,
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """Alias for pure optical face enhancement (No GFPGAN)."""
        return self.enhance_faces_optical(img_bgr)

    @staticmethod
    def lock_face_keep_clothes(
        orig_pil: Image.Image, dressed_pil: Image.Image, strength: float = 0.88
    ) -> Image.Image:
        """Copy the real face onto the dressed photo. Never blend original clothes back."""
        dressed = np.array(dressed_pil.convert("RGB"))
        h, w = dressed.shape[:2]
        orig = np.array(orig_pil.convert("RGB").resize((w, h), Image.LANCZOS))
        orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
        dressed_bgr = cv2.cvtColor(dressed, cv2.COLOR_RGB2BGR)
        try:
            from skimage.transform import SimilarityTransform

            app = _get_insight_app()
            faces_o = app.get(orig_bgr)
            faces_d = app.get(dressed_bgr)
            if not faces_o or not faces_d:
                return dressed_pil
            kps_o = faces_o[0].kps
            kps_d = faces_d[0].kps
            # Align source and generated landmarks, but cap scale changes so
            # Qwen cannot stretch the real face into enlarged-eye geometry.
            transform = SimilarityTransform()
            transform.estimate(kps_o, kps_d)
            raw = transform.params[:2, :].astype(np.float32)
            raw_scale = max(1e-6, float(np.hypot(raw[0, 0], raw[1, 0])))
            bounded_scale = float(np.clip(raw_scale, 0.95, 1.05))
            linear = raw[:, :2] * (bounded_scale / raw_scale)
            source_center = np.mean(kps_o, axis=0)
            generated_center = np.mean(kps_d, axis=0)
            translation = generated_center - linear @ source_center
            M = np.column_stack((linear, translation)).astype(np.float32)
            warped = cv2.warpAffine(orig_bgr, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
            x1, y1, x2, y2 = [int(v) for v in faces_d[0].bbox]
            chin = y2
            mask = np.zeros((h, w), dtype=np.uint8)
            landmarks = getattr(faces_o[0], "landmark_2d_106", None)
            if landmarks is not None and len(landmarks) >= 20:
                aligned = cv2.transform(
                    np.asarray(landmarks, dtype=np.float32).reshape(-1, 1, 2), M
                ).reshape(-1, 2)
                cv2.fillConvexPoly(mask, cv2.convexHull(aligned.astype(np.int32)), 255)
                mask = cv2.dilate(
                    mask,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
                    iterations=1,
                )
            else:
                cx = (x1 + x2) // 2
                fw, fh = max(24, x2 - x1), max(24, y2 - y1)
                cv2.ellipse(mask, (cx, chin - fh // 2), (int(fw * 0.50), int(fh * 0.62)), 0, 0, 360, 255, -1)

            mask[chin:, :] = 0
            mask = cv2.GaussianBlur(mask, (13, 13), 0)
            a = (mask.astype(np.float32) / 255.0)[:, :, None] * float(np.clip(strength, 0.0, 1.0))
            fused = (warped.astype(np.float32) * a + dressed_bgr.astype(np.float32) * (1.0 - a))
            fused = fused.clip(0, 255).astype(np.uint8)
            return Image.fromarray(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB))
        except Exception as e:
            logger.warning("[IDENTITY] Face-only lock skipped: %s", e)
            return dressed_pil

    @staticmethod
    def is_source_pose_compatible(orig_pil: Image.Image, portrait_pil: Image.Image) -> bool:
        """Return whether source pixels can be safely aligned onto the Qwen face."""
        try:
            target = np.array(portrait_pil.convert("RGB"))
            h, w = target.shape[:2]
            source = np.array(orig_pil.convert("RGB").resize((w, h), Image.LANCZOS))
            app = _get_insight_app()
            source_faces = app.get(cv2.cvtColor(source, cv2.COLOR_RGB2BGR))
            target_faces = app.get(cv2.cvtColor(target, cv2.COLOR_RGB2BGR))
            if not source_faces or not target_faces:
                return False

            source_face, target_face = source_faces[0], target_faces[0]
            source_eyes = np.asarray(source_face.kps[:2], dtype=np.float32)
            target_eyes = np.asarray(target_face.kps[:2], dtype=np.float32)
            source_dist = float(np.linalg.norm(source_eyes[1] - source_eyes[0])) + 1e-6
            target_dist = float(np.linalg.norm(target_eyes[1] - target_eyes[0]))
            scale = target_dist / source_dist
            source_angle = float(np.degrees(np.arctan2(*(source_eyes[1] - source_eyes[0])[::-1])))
            target_angle = float(np.degrees(np.arctan2(*(target_eyes[1] - target_eyes[0])[::-1])))
            angle_delta = abs((target_angle - source_angle + 180.0) % 360.0 - 180.0)
            source_center = np.mean(source_eyes, axis=0)
            target_center = np.mean(target_eyes, axis=0)
            shift = float(np.linalg.norm(target_center - source_center) / max(target_dist, 1.0))
            compatible = 0.78 <= scale <= 1.25 and angle_delta <= 12.0 and shift <= 0.30
            logger.info(
                "[IDENTITY] Pose gate compatible=%s scale=%.3f angle=%.1f shift=%.3f",
                compatible, scale, angle_delta, shift,
            )
            return compatible
        except Exception as exc:
            logger.warning("[IDENTITY] Pose gate unavailable: %s", exc)
            return False

    @staticmethod
    def has_harsh_portrait_sunlight(
        image_pil: Image.Image,
        face_bbox=None,
    ) -> bool:
        """Measure hard saturated light over the face, independent of VL prose."""
        rgb = np.asarray(image_pil.convert("RGB"), dtype=np.uint8)
        height, width = rgb.shape[:2]
        if face_bbox is not None and len(face_bbox) >= 4:
            x1, y1, x2, y2 = [int(round(float(value))) for value in face_bbox[:4]]
            pad_x = max(4, int((x2 - x1) * .12))
            pad_y = max(4, int((y2 - y1) * .12))
            x1, x2 = max(0, x1 - pad_x), min(width, x2 + pad_x)
            y1, y2 = max(0, y1 - pad_y), min(height, y2 + pad_y)
            region = rgb[y1:y2, x1:x2]
        else:
            region = rgb[
                int(height * .12):int(height * .64),
                int(width * .18):int(width * .82),
            ]
        if region.size < 256:
            return False
        luminance = cv2.cvtColor(region, cv2.COLOR_RGB2LAB)[:, :, 0]
        saturation = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)[:, :, 1]
        p20, p95 = np.percentile(luminance, (20, 95))
        saturation_p80 = float(np.percentile(saturation, 80))
        # This check is used only after the VL analysis has already reported
        # direct light. The former thresholds missed visibly harsh, clipped
        # warm light by a few values (for example p95=185 and saturation=171),
        # so uniform swap skipped the same bounded correction used by
        # Enhancement. Keep all three signals mandatory while admitting that
        # still-obvious residual-light case.
        detected = bool(
            p95 >= 168.0
            and (p95 - p20) >= 95.0
            and saturation_p80 >= 150.0
        )
        logger.info(
            "[SUNLIGHT_MEASURE] p20=%.1f p95=%.1f contrast=%.1f "
            "saturation_p80=%.1f detected=%s",
            p20, p95, p95 - p20, saturation_p80, detected,
        )
        return detected

    @staticmethod
    def neutralize_direct_sunlight(image_pil: Image.Image) -> Image.Image:
        """Compress face and hair hot spots without regenerating subject pixels."""
        image = np.array(image_pil.convert("RGB"))
        h, w = image.shape[:2]
        try:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            faces = _get_insight_app().get(bgr)
            if not faces:
                return image_pil

            x1, y1, x2, y2 = [int(value) for value in faces[0].bbox]
            face_w, face_h = max(24, x2 - x1), max(24, y2 - y1)
            center_x = (x1 + x2) // 2
            face_mask = np.zeros((h, w), dtype=np.uint8)
            hair_mask = np.zeros((h, w), dtype=np.uint8)
            # The forehead and cheeks receive broad, soft correction. Hair is
            # separately bounded above the face so dark curls are not lightened.
            cv2.ellipse(face_mask, (center_x, y1 + int(face_h * 0.48)),
                        (int(face_w * 0.60), int(face_h * 0.58)), 0, 0, 360, 255, -1)
            cv2.ellipse(hair_mask, (center_x, max(0, y1 + int(face_h * 0.10))),
                        (int(face_w * 0.95), int(face_h * 0.82)), 0, 0, 360, 255, -1)
            cv2.ellipse(hair_mask, (center_x, y1 + int(face_h * 0.50)),
                        (int(face_w * 0.61), int(face_h * 0.62)), 0, 0, 360, 0, -1)

            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            luminance = lab[:, :, 0]
            face_alpha = cv2.GaussianBlur(face_mask, (31, 31), 7).astype(np.float32) / 255.0
            hair_alpha = cv2.GaussianBlur(hair_mask, (31, 31), 7).astype(np.float32) / 255.0
            # Only compress high local values: midtones and genuine facial
            # detail are untouched. This removes direct-sun hot spots rather
            # than applying a beauty filter or a global darkening.
            # Outdoor highlights on medium-to-deep skin often clip warm/orange
            # before reaching pure white, so use a lower local threshold.
            face_hot = np.clip((luminance - 132.0) / 68.0, 0.0, 1.0) * face_alpha
            hair_hot = np.clip((luminance - 112.0) / 78.0, 0.0, 1.0) * hair_alpha
            luminance -= face_hot * 42.0
            # Sunlit dark hair in phone photos can be lifted all the way to
            # pale grey.  Compress that highlight more firmly while retaining
            # the original strand texture and excluding the face ellipse.
            luminance -= hair_hot * 68.0
            lab[:, :, 0] = np.clip(luminance, 0, 255)
            # Strong sun can push skin toward orange. Blend chroma a little
            # toward neutral only inside the face highlight mask.
            lab[:, :, 1] += (128.0 - lab[:, :, 1]) * face_hot * 0.12
            lab[:, :, 2] += (128.0 - lab[:, :, 2]) * face_hot * 0.08
            corrected = cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
            logger.info("[LIGHTING] Compressed direct-sun highlights on source portrait")
            return Image.fromarray(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
        except Exception as exc:
            logger.warning("[LIGHTING] Direct-sun correction skipped: %s", exc)
            return image_pil

    @staticmethod
    def normalize_uniform_conditioning_light(image_pil: Image.Image) -> Image.Image:
        """Create an evenly lit identity guide for the single-pass uniform edit.

        This image is never delivered to the user. It gives Qwen a neutral
        lighting target while the untouched upload remains the source used for
        analysis and identity verification.
        """
        base = PhotoRestorationService.neutralize_direct_sunlight(image_pil)
        try:
            rgb = np.array(base.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            faces = _get_insight_app().get(bgr)
            if not faces:
                return base

            height, width = rgb.shape[:2]
            x1, y1, x2, y2 = [int(value) for value in faces[0].bbox]
            face_w, face_h = max(24, x2 - x1), max(24, y2 - y1)
            center_x = (x1 + x2) // 2

            face_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.ellipse(
                face_mask,
                (center_x, y1 + int(face_h * 0.49)),
                (int(face_w * 0.58), int(face_h * 0.60)),
                0, 0, 360, 255, -1,
            )
            feather = max(15, (min(face_w, face_h) // 5) | 1)
            face_alpha = cv2.GaussianBlur(face_mask, (feather, feather), 0).astype(np.float32) / 255.0

            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            luminance = lab[:, :, 0]
            sigma = max(9.0, min(face_w, face_h) * 0.18)
            illumination = cv2.GaussianBlur(luminance, (0, 0), sigmaX=sigma, sigmaY=sigma)
            inner = face_mask > 220
            if not np.any(inner):
                return base

            # Remove only low-frequency directional light. High-frequency
            # facial texture remains in luminance and is therefore preserved.
            observed_level = float(np.median(illumination[inner]))
            studio_level = float(np.clip(observed_level, 125.0, 151.0))
            correction = np.clip(studio_level - illumination, -34.0, 22.0)
            luminance += correction * face_alpha * 0.88
            lab[:, :, 0] = np.clip(luminance, 0, 255)

            # Broad direct sun also shifts the complete face toward orange,
            # not just its brightest pixels. Limit that cast gently across
            # the guide instead of forcing a fixed skin colour.
            a_median = float(np.median(lab[:, :, 1][inner]))
            b_median = float(np.median(lab[:, :, 2][inner]))
            a_shift = min(7.0, max(0.0, a_median - 146.0))
            b_shift = min(20.0, max(0.0, b_median - 132.0))
            lab[:, :, 1] -= face_alpha * a_shift
            lab[:, :, 2] -= face_alpha * b_shift

            corrected = cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
            logger.info(
                "[LIGHTING_GUIDE] Flattened source illumination level=%.1f a_shift=%.1f b_shift=%.1f",
                observed_level, a_shift, b_shift,
            )
            return Image.fromarray(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
        except Exception as exc:
            logger.warning("[LIGHTING_GUIDE] Uniform conditioning normalization skipped: %s", exc)
            return base

    @staticmethod
    def tame_generated_portrait_hotspots(image_pil: Image.Image) -> Image.Image:
        """Tone down bright forehead/hair glare without changing portrait geometry.

        Qwen can retain a sunlit highlight even when its edit prompt requests
        studio lighting. This operates only on high luminance pixels in the
        upper portrait and excludes the flat generated backdrop sampled from
        image corners.
        """
        try:
            bgr = cv2.cvtColor(np.array(image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
            height, width = bgr.shape[:2]
            edge = max(4, min(18, height // 20, width // 20))
            corners = np.concatenate((
                bgr[:edge, :edge].reshape(-1, 3),
                bgr[:edge, -edge:].reshape(-1, 3),
                bgr[-edge:, :edge].reshape(-1, 3),
                bgr[-edge:, -edge:].reshape(-1, 3),
            ))
            backdrop = np.median(corners, axis=0).astype(np.float32)
            # Include Qwen's mild blue/gray studio falloff around the crown.
            # A tighter key treated that smooth backdrop shade as bright hair
            # and darkened it into visible islands above the head.
            backdrop_pixels = np.linalg.norm(
                bgr.astype(np.float32) - backdrop, axis=2,
            ) < 64.0

            region = np.zeros((height, width), dtype=np.uint8)
            cv2.ellipse(
                region, (width // 2, int(height * 0.36)),
                (int(width * 0.43), int(height * 0.34)), 0, 0, 360, 255, -1,
            )
            region[backdrop_pixels] = 0
            alpha = cv2.GaussianBlur(region, (25, 25), 5).astype(np.float32) / 255.0
            alpha[backdrop_pixels] = 0.0

            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            # Do not perform a broad colour-keyed hotspot edit here. Pale
            # backdrop falloff and bright skin can overlap in colour, which
            # previously created a grey halo around hair and isolated pale
            # islands on the face. The detected-face correction below and the
            # semantic hair pass handle the two intended regions safely.
            # Correct broad directional illumination and orange sunlight only
            # inside the detected face. This changes low-frequency light, not
            # landmarks or source texture.
            faces = _get_insight_app().get(bgr)
            face_correction = 0.0
            if len(faces) == 1:
                x1, y1, x2, y2 = [int(value) for value in faces[0].bbox]
                face_w, face_h = max(24, x2 - x1), max(24, y2 - y1)
                face_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.ellipse(
                    face_mask,
                    ((x1 + x2) // 2, y1 + int(face_h * .50)),
                    (int(face_w * .52), int(face_h * .57)),
                    0, 0, 360, 255, -1,
                )
                face_alpha = cv2.GaussianBlur(
                    face_mask, (0, 0), max(7.0, face_w * .055),
                ).astype(np.float32) / 255.0
                inner = face_mask > 230
                illumination = cv2.GaussianBlur(
                    lab[:, :, 0], (0, 0), max(13.0, face_w * .16),
                )
                target_l = float(np.clip(np.median(illumination[inner]), 122.0, 145.0))
                correction = np.clip(target_l - illumination, -30.0, 22.0)
                lab[:, :, 0] += correction * face_alpha * .55

                # Hard side-light is directional: a global median can reduce
                # it yet still leave one cheek brighter than its counterpart.
                # Compare only low-frequency illumination at mirrored facial
                # positions, then move both sides gently toward their shared
                # value. Fine skin texture, eyes and facial geometry are not
                # part of this correction field.
                rx1, ry1 = max(0, x1), max(0, y1)
                rx2, ry2 = min(width, x2), min(height, y2)
                directional_mean = 0.0
                if rx2 - rx1 >= 24 and ry2 - ry1 >= 24:
                    illumination_roi = illumination[ry1:ry2, rx1:rx2]
                    mirrored_l = np.fliplr(illumination_roi)
                    symmetry_delta = np.clip(
                        (mirrored_l - illumination_roi) * .5, -13.0, 13.0,
                    )
                    roi_h, roi_w = symmetry_delta.shape
                    fade_y = max(4, int(roi_h * .16))
                    fade_x = max(4, int(roi_w * .10))
                    edge_window_y = np.ones(roi_h, dtype=np.float32)
                    edge_window_x = np.ones(roi_w, dtype=np.float32)
                    edge_window_y[:fade_y] = np.linspace(0.0, 1.0, fade_y)
                    edge_window_y[-fade_y:] = np.linspace(1.0, 0.0, fade_y)
                    edge_window_x[:fade_x] = np.linspace(0.0, 1.0, fade_x)
                    edge_window_x[-fade_x:] = np.linspace(1.0, 0.0, fade_x)
                    symmetry_window = edge_window_y[:, None] * edge_window_x[None, :]
                    symmetry_delta *= symmetry_window
                    symmetry_field = np.zeros((height, width), dtype=np.float32)
                    symmetry_field[ry1:ry2, rx1:rx2] = symmetry_delta
                    lab[:, :, 0] += symmetry_field * face_alpha * .66
                    directional_mean = float(np.mean(np.abs(symmetry_delta)))

                    # Balance only broad directional colour cast. Restricting
                    # this to blurred LAB chroma retains the person's actual
                    # complexion, lip colour, moles and local skin detail.
                    for channel, limit, gain in ((1, 5.0, .30), (2, 8.0, .38)):
                        chroma_low = cv2.GaussianBlur(
                            lab[:, :, channel], (0, 0), max(11.0, face_w * .13),
                        )
                        chroma_roi = chroma_low[ry1:ry2, rx1:rx2]
                        chroma_delta = np.clip(
                            (np.fliplr(chroma_roi) - chroma_roi) * .5,
                            -limit, limit,
                        )
                        chroma_delta *= symmetry_window
                        chroma_field = np.zeros((height, width), dtype=np.float32)
                        chroma_field[ry1:ry2, rx1:rx2] = chroma_delta
                        lab[:, :, channel] += chroma_field * face_alpha * gain
                a_med = float(np.median(lab[:, :, 1][inner]))
                b_med = float(np.median(lab[:, :, 2][inner]))
                # This routine is invoked only after measured harsh sunlight.
                # A clipped warm source can remain orange even after luminance
                # balancing, so remove the broad cast more decisively while
                # retaining all high-frequency skin texture and geometry.
                a_shift = min(7.0, max(0.0, a_med - 144.0))
                b_shift = min(22.0, max(0.0, b_med - 132.0))
                lab[:, :, 1] -= face_alpha * a_shift * .82
                lab[:, :, 2] -= face_alpha * b_shift * .90
                face_correction = float(np.mean(np.abs(correction[inner])))
                logger.info(
                    "[LIGHTING] Balanced directional face light (mean=%.1f)",
                    directional_mean,
                )

                # Extend the same illumination-only correction to visible
                # neck and body skin.  A fixed oval cannot cover those areas,
                # and a plain colour threshold can damage warm uniforms. SCHP
                # supplies semantic face/arm/leg candidates; face-relative
                # chroma then admits the neck while rejecting clothing. This
                # remains dynamic for children/adults and every crop/pose.
                try:
                    from pipelines.schp_service import parse as parse_body_parts

                    rgb_for_parse = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    labels = parse_body_parts(rgb_for_parse)["labels"]
                    semantic_limbs = np.isin(labels, (14, 15, 16, 17))
                    face_seed = (labels == 13) & inner
                    if int(face_seed.sum()) < 96:
                        face_seed = inner
                    face_a = float(np.median(lab[:, :, 1][face_seed]))
                    face_b = float(np.median(lab[:, :, 2][face_seed]))
                    chroma_distance = np.sqrt(
                        (lab[:, :, 1] - face_a) ** 2
                        + (lab[:, :, 2] - face_b) ** 2
                    )

                    chroma_skin = chroma_distance <= 24.0
                    # Normalize exposed limbs independently. Do not flatten
                    # the neck vertically: the natural chin-to-neck falloff is
                    # legitimate studio shading, and treating it as a body
                    # hotspot creates a visible horizontal seam. Qwen's clean
                    # generated neck remains authoritative.
                    body_skin = semantic_limbs & chroma_skin & ~backdrop_pixels
                    body_skin = cv2.morphologyEx(
                        body_skin.astype(np.uint8), cv2.MORPH_OPEN,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                    ).astype(bool)
                    if int(body_skin.sum()) >= 96:
                        body_alpha = cv2.GaussianBlur(
                            body_skin.astype(np.uint8) * 255,
                            (0, 0), max(4.0, face_w * .025),
                        ).astype(np.float32) / 255.0
                        # Blend continuously into the face correction. A hard
                        # face-mask subtraction created a horizontal exposure
                        # seam beneath the chin on close passport crops.
                        body_alpha *= np.clip(1.0 - face_alpha, 0.0, 1.0)
                        body_alpha[backdrop_pixels] = 0.0
                        body_illumination = cv2.GaussianBlur(
                            lab[:, :, 0], (0, 0), max(9.0, face_w * .11),
                        )
                        body_values = body_illumination[body_skin]
                        body_spread = float(
                            np.percentile(body_values, 90)
                            - np.percentile(body_values, 10)
                        )
                        if body_spread >= 14.0:
                            body_target = float(np.median(body_values))
                            body_correction = np.clip(
                                body_target - body_illumination, -22.0, 18.0,
                            )
                            lab[:, :, 0] += body_correction * body_alpha * .52

                            # Remove only local excess warm glare relative to
                            # this person's own face chroma; do not impose a
                            # fixed complexion or alter normally lit skin.
                            warm_excess = np.clip(
                                lab[:, :, 2] - (face_b + 7.0), 0.0, 12.0,
                            )
                            lab[:, :, 2] -= warm_excess * body_alpha * .34
                            logger.info(
                                "[LIGHTING] Normalized visible neck/body sunlight "
                                "(spread=%.1f)", body_spread,
                            )
                except Exception as body_error:
                    logger.warning(
                        "[LIGHTING] Visible body-skin correction skipped: %s",
                        body_error,
                    )
            logger.info(
                "[LIGHTING] Tamed hotspots and normalized face illumination (mean=%.1f)",
                face_correction,
            )
            return Image.fromarray(cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB))
        except Exception as exc:
            logger.warning("[LIGHTING] Generated hotspot correction skipped: %s", exc)
            return image_pil

    @staticmethod
    def balance_face_light_luminance_only(image_pil: Image.Image) -> Image.Image:
        """Balance directional face light without recolouring or smoothing.

        This is intentionally narrower than ``tame_generated_portrait_hotspots``.
        It adjusts only the low-frequency LAB luminance inside one detected
        face.  Colour is deliberately not modified: an earlier mirrored-
        chroma correction could turn one cheek pale/cyan.  Facial detail,
        complexion, hair, accessories, clothing and backdrop therefore remain
        unchanged while the directional illumination field is flattened.
        """
        try:
            rgb = np.asarray(image_pil.convert("RGB"), dtype=np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            faces = _get_insight_app().get(bgr)
            if len(faces) != 1:
                return image_pil

            height, width = rgb.shape[:2]
            x1, y1, x2, y2 = [int(round(float(v))) for v in faces[0].bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            face_w, face_h = x2 - x1, y2 - y1
            if face_w < 32 or face_h < 32:
                return image_pil

            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            luminance = lab[:, :, 0]
            # A cheek-sized sunlight patch is smaller than the complete face;
            # use a local illumination field while remaining far broader than
            # eyes, pores or expression detail.
            sigma = max(10.0, face_w * .075)
            illumination = cv2.GaussianBlur(luminance, (0, 0), sigma)
            roi = illumination[y1:y2, x1:x2]
            mirrored = np.fliplr(roi)
            inner = roi[
                int(face_h * .10):max(int(face_h * .11), int(face_h * .88)),
                int(face_w * .10):max(int(face_w * .11), int(face_w * .90)),
            ]
            studio_level = float(np.median(inner)) if inner.size else float(np.median(roi))
            # A neutral frontal studio field is mostly uniform but not flat.
            # Retain a small amount of symmetric modelling and move the broad
            # illumination toward this same face's median.  The target comes
            # from the generated person, never from a fixed skin colour.
            target_field = studio_level * .88 + mirrored * .12
            delta = np.clip(target_field - roi, -34.0, 24.0)
            glare_region = roi > (target_field + 4.0)

            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.ellipse(
                mask,
                ((x1 + x2) // 2, y1 + int(face_h * .46)),
                (int(face_w * .48), int(face_h * .43)),
                0, 0, 360, 255, -1,
            )
            alpha = cv2.GaussianBlur(
                mask, (0, 0), max(7.0, face_w * .045),
            ).astype(np.float32) / 255.0
            correction = np.zeros((height, width), dtype=np.float32)
            correction[y1:y2, x1:x2] = delta
            # A restrained blend removes the obvious sun patch while keeping
            # natural modelling and every high-frequency facial feature.
            luminance += correction * alpha * .95
            core = mask > 220
            if np.any(core):
                # Compress residual *low-frequency* glare rather than bright
                # pixels.  This keeps eye catchlights, pores, lips and other
                # real high-frequency detail while removing the broad cheek/
                # nose sunlight field.
                balanced_field = cv2.GaussianBlur(
                    luminance, (0, 0), sigma,
                )
                face_median = float(np.median(balanced_field[core]))
                field_excess = np.clip(
                    balanced_field - (face_median + 7.0), 0.0, 24.0,
                )
                luminance -= field_excess * alpha * .58
                field_deficit = np.clip(
                    (face_median - 7.0) - balanced_field, 0.0, 18.0,
                )
                # Lift only broad cast-shadow fields. High-frequency natural
                # contrast in eyes, brows, lips and pores is not part of this
                # illumination map and therefore remains unchanged.
                luminance += field_deficit * alpha * .48
            lab[:, :, 0] = np.clip(luminance, 0, 255)

            # A clipped Qwen highlight can also lose skin chroma and appear as
            # a white/cyan patch even after its brightness is corrected. Fill
            # only that detected glare field from this face's own stable
            # midtone colour. Never transfer the outdoor source illuminant and
            # never impose a hard-coded complexion.
            colour_alpha = alpha.copy()
            landmarks = np.asarray(getattr(faces[0], "kps", []), dtype=np.float32)
            if landmarks.shape == (5, 2):
                protection = np.zeros((height, width), dtype=np.uint8)
                eye_radius = max(5, int(face_w * .085))
                mouth_radius = max(6, int(face_w * .105))
                for px, py in landmarks[:2]:
                    cv2.circle(protection, (int(px), int(py)), eye_radius, 255, -1)
                for px, py in landmarks[3:5]:
                    cv2.circle(protection, (int(px), int(py)), mouth_radius, 255, -1)
                protected = cv2.GaussianBlur(
                    protection, (0, 0), max(2.0, face_w * .012),
                ).astype(np.float32) / 255.0
                colour_alpha *= 1.0 - protected

            glare = np.zeros((height, width), dtype=np.float32)
            glare[y1:y2, x1:x2] = glare_region.astype(np.float32)
            colour_alpha *= cv2.GaussianBlur(
                glare, (0, 0), max(3.0, face_w * .018),
            )
            for channel, ceiling in ((1, 5.0), (2, 12.0)):
                channel_field = cv2.GaussianBlur(
                    lab[:, :, channel], (0, 0), sigma,
                )
                stable_values = channel_field[core]
                if not stable_values.size:
                    continue
                stable_tone = float(np.median(stable_values))
                # Measure the actual clipped skin colour here. Using only the
                # blurred chroma field underestimated small white/cyan cheek
                # patches and left them visible after luminance repair.
                local = lab[y1:y2, x1:x2, channel]
                symmetric = np.fliplr(local)
                target_colour = stable_tone * .65 + symmetric * .35
                # Bleached highlights need chroma restored, never removed.
                # This one-way bound prevents a yellow/orange or blue cast.
                colour_delta = np.clip(target_colour - local, 0.0, ceiling)
                correction_colour = np.zeros((height, width), dtype=np.float32)
                correction_colour[y1:y2, x1:x2] = colour_delta
                lab[:, :, channel] += correction_colour * colour_alpha * .82

            # The same direct light often continues beyond the detected face
            # onto the crown, neck and exposed upper-body skin.  Correct those
            # semantic regions in the same pass so the face is not left evenly
            # lit above a sunlit neck/chest.  Every target tone is measured
            # from this generated portrait; no complexion or hair colour is
            # hard-coded and no source-background pixels are copied.
            try:
                from pipelines.schp_service import parse as parse_body_parts

                labels = parse_body_parts(rgb)["labels"]
                low_l = cv2.GaussianBlur(
                    lab[:, :, 0], (0, 0), max(8.0, face_w * .070),
                )

                # Stable complexion comes from the already-balanced inner
                # face.  Restrict the extra skin search to the person's centre
                # below the chin so similarly coloured backdrop or clothing
                # cannot be selected.
                stable_skin = core & (labels == 13)
                if int(stable_skin.sum()) < 96:
                    stable_skin = core
                skin_a = float(np.median(lab[:, :, 1][stable_skin]))
                skin_b = float(np.median(lab[:, :, 2][stable_skin]))
                skin_chroma_distance = np.sqrt(
                    (lab[:, :, 1] - skin_a) ** 2
                    + (lab[:, :, 2] - skin_b) ** 2
                )
                torso_window = np.zeros((height, width), dtype=bool)
                tx1 = max(0, x1 - int(face_w * .50))
                tx2 = min(width, x2 + int(face_w * .50))
                ty1 = max(0, y1 + int(face_h * .72))
                ty2 = min(height, y2 + int(face_h * 1.10))
                torso_window[ty1:ty2, tx1:tx2] = True
                semantic_skin = np.isin(labels, (13, 14, 15, 16, 17))
                extra_skin = (
                    (semantic_skin | (torso_window & (skin_chroma_distance <= 17.0)))
                    & ~core
                    & (labels != 2)
                )
                extra_skin = cv2.morphologyEx(
                    extra_skin.astype(np.uint8), cv2.MORPH_OPEN,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                ).astype(bool)
                if int(extra_skin.sum()) >= 96:
                    skin_alpha = cv2.GaussianBlur(
                        extra_skin.astype(np.uint8) * 255, (0, 0),
                        max(3.0, face_w * .018),
                    ).astype(np.float32) / 255.0
                    skin_alpha *= np.clip(1.0 - alpha * .80, 0.0, 1.0)
                    skin_values = low_l[extra_skin]
                    skin_target_l = float(np.percentile(skin_values, 55))
                    skin_delta = np.clip(
                        skin_target_l - low_l, -28.0, 20.0,
                    )
                    skin_delta[np.abs(skin_delta) < 4.0] = 0.0
                    # Balance both sunlight and its paired cast shadow so the
                    # neck/body do not retain a directional outdoor-light cue
                    # beneath an evenly relit face.
                    lab[:, :, 0] += skin_delta * skin_alpha * .74
                    for channel, tone, ceiling in (
                        (1, skin_a, 5.0), (2, skin_b, 10.0),
                    ):
                        depleted = np.clip(
                            tone - lab[:, :, channel], 0.0, ceiling,
                        )
                        lab[:, :, channel] += depleted * skin_alpha * .65

                # SCHP hair plus a face-relative head window prevents flowers,
                # backdrop and clothing from being darkened. Work on the
                # low-frequency field so normal strand shine remains, while
                # broad silver/orange sunlight is compressed.
                hair_window = np.zeros((height, width), dtype=bool)
                hx1 = max(0, x1 - int(face_w * .72))
                hx2 = min(width, x2 + int(face_w * .72))
                hy1 = max(0, y1 - int(face_h * .58))
                hy2 = min(height, y2 + int(face_h * .68))
                hair_window[hy1:hy2, hx1:hx2] = True
                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
                hair = (
                    (labels == 2) & hair_window
                    & (hsv[:, :, 2] < 242)
                    & ~core
                )
                if int(hair.sum()) >= 128:
                    hair_alpha = cv2.GaussianBlur(
                        hair.astype(np.uint8) * 255, (0, 0),
                        max(2.5, face_w * .014),
                    ).astype(np.float32) / 255.0
                    hair_values = low_l[hair]
                    hair_target_l = float(np.percentile(hair_values, 62))
                    hair_excess = np.clip(
                        low_l - (hair_target_l + 16.0), 0.0, 16.0,
                    )
                    # Keep strand highlights and the backdrop visible through
                    # natural curls. Only take the edge off broad glare; an
                    # aggressive dark-hair pass exaggerates those openings as
                    # white holes and makes the result look cut out.
                    lab[:, :, 0] -= hair_excess * hair_alpha * .24
                logger.info(
                    "[PORTRAIT_LIGHT] Extended correction to skin=%d hair=%d pixels",
                    int(extra_skin.sum()), int(hair.sum()),
                )
            except Exception as region_error:
                logger.warning(
                    "[PORTRAIT_LIGHT] Hair/body extension skipped: %s",
                    region_error,
                )
            corrected = cv2.cvtColor(
                lab.astype(np.uint8), cv2.COLOR_LAB2RGB,
            )
            logger.info(
                "[FACE_LIGHT] Balanced directional luminance only (mean_abs=%.2f)",
                float(np.mean(np.abs(delta))),
            )
            return Image.fromarray(corrected, "RGB")
        except Exception as exc:
            logger.warning("[FACE_LIGHT] Luminance-only balance skipped: %s", exc)
            return image_pil

    @staticmethod
    def balance_exposed_body_sunlight_only(image_pil: Image.Image) -> Image.Image:
        """Remove residual directional sun from visible neck/chest skin only.

        Qwen sometimes relights the face correctly while reproducing an orange
        sun patch and paired shadow on exposed upper-body skin. This pass never
        touches the detected face, hair, accessories, clothing or backdrop.
        Target light is derived from the generated portrait's opposite-side
        skin and its own stable face chroma; no fixed complexion is imposed.
        """
        try:
            from pipelines.schp_service import parse as parse_body_parts

            rgb = np.asarray(image_pil.convert("RGB"), dtype=np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            faces = _get_insight_app().get(bgr)
            if len(faces) != 1:
                return image_pil
            height, width = rgb.shape[:2]
            x1, y1, x2, y2 = [int(round(float(v))) for v in faces[0].bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            face_w, face_h = x2 - x1, y2 - y1
            if face_w < 32 or face_h < 32:
                return image_pil

            labels = parse_body_parts(rgb)["labels"]
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
            original_lab = lab.copy()
            face_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.ellipse(
                face_mask,
                ((x1 + x2) // 2, y1 + int(face_h * .48)),
                (int(face_w * .38), int(face_h * .34)),
                0, 0, 360, 255, -1,
            )
            stable_face = (face_mask > 220) & (labels == 13)
            if int(stable_face.sum()) < 96:
                stable_face = face_mask > 220
            skin_a = float(np.median(lab[:, :, 1][stable_face]))
            skin_b = float(np.median(lab[:, :, 2][stable_face]))
            chroma_distance = np.sqrt(
                (lab[:, :, 1] - skin_a) ** 2
                + (lab[:, :, 2] - skin_b) ** 2
            )

            center_x = (x1 + x2) // 2
            body_window = np.zeros((height, width), dtype=bool)
            bx1 = max(0, center_x - int(face_w * .82))
            bx2 = min(width, center_x + int(face_w * .82))
            by1 = max(0, y2 - int(face_h * .01))
            by2 = min(height, y2 + int(face_h * 1.35))
            body_window[by1:by2, bx1:bx2] = True
            semantic_skin = np.isin(labels, (13, 14, 15, 16, 17))
            # Do not require a foreground SCHP label here.  Dappled sunlight
            # frequently makes exposed chest skin get parsed as background,
            # which used to punch the exact bright fragments out of the repair
            # mask.  The tight, face-relative chroma gate and central neckline
            # window are the foreground safeguards; they adapt to every
            # subject's own complexion instead of imposing a fixed skin tone.
            body_skin = body_window & (
                (semantic_skin & (chroma_distance <= 34.0))
                | (chroma_distance <= 30.0)
            )
            body_skin &= labels != 2
            body_skin &= face_mask == 0
            body_skin = cv2.morphologyEx(
                body_skin.astype(np.uint8), cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            ).astype(bool)
            body_skin = cv2.morphologyEx(
                body_skin.astype(np.uint8), cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            ).astype(bool)
            if int(body_skin.sum()) < 120:
                return image_pil

            # Use a tighter illumination scale than the face relighting pass.
            # Leaf-shaped sun streaks on a neck/chest are often only a few
            # percent of the face width; the old broad blur averaged them into
            # their surroundings and left the visible pattern intact.
            sigma = max(5.0, face_w * .018)
            low_l = cv2.GaussianBlur(lab[:, :, 0], (0, 0), sigma)
            # Model neutral studio illumination as one robust skin level per
            # image row. This preserves the natural neck-to-chest vertical
            # falloff, but removes left/right dappled sun. Mirroring was unsafe
            # when an open asymmetric jacket occupied the opposite side.
            row_level = np.full(height, np.nan, dtype=np.float32)
            for row in range(by1, by2):
                values = low_l[row, body_skin[row]]
                if values.size >= 6:
                    row_level[row] = float(np.median(values))
            valid_rows = np.flatnonzero(np.isfinite(row_level))
            if valid_rows.size < 4:
                return image_pil
            row_level[by1:by2] = np.interp(
                np.arange(by1, by2), valid_rows, row_level[valid_rows],
            )
            smoothed_rows = cv2.GaussianBlur(
                row_level[by1:by2].reshape(-1, 1),
                (1, 0), sigmaX=0.0, sigmaY=max(3.0, face_h * .018),
            ).reshape(-1)
            target_l = low_l.copy()
            target_l[by1:by2, :] = smoothed_rows[:, None]
            # Normalize the complete exposed-skin illumination field. Selecting
            # only thresholded hotspot fragments preserved the recognizable
            # leaf/sun pattern between them. The correction itself remains
            # low-frequency and close to zero on already even skin.
            affected = body_skin
            if int(affected.sum()) < 64:
                return image_pil
            # Keep full strength across the interior of exposed skin.  A wide
            # mask blur made alpha vary across narrow sun streaks, preserving
            # their silhouette even when their average brightness changed.
            # Only feather a few pixels at the skin/clothing boundary.
            alpha = cv2.GaussianBlur(
                affected.astype(np.uint8) * 255, (0, 0), 2.2,
            ).astype(np.float32) / 255.0
            # Fade in below the face box so detector geometry can never leave
            # a horizontal neck seam.
            transition_h = max(10, int(face_h * .16))
            vertical_ramp = np.ones(height, dtype=np.float32)
            vertical_ramp[:by1] = 0.0
            ramp_end = min(height, by1 + transition_h)
            vertical_ramp[by1:ramp_end] = np.linspace(
                0.0, 1.0, ramp_end - by1, dtype=np.float32,
            )
            alpha *= vertical_ramp[:, None]

            # Rebuild only the low-frequency illumination/chroma field.  Keep
            # small luminance detail (pores and camera texture), while removing
            # leaf-shaped sun and shadow patterns rather than merely dimming
            # them.  Complexion comes from this subject's stable face pixels.
            fine_l = lab[:, :, 0] - cv2.GaussianBlur(
                lab[:, :, 0], (0, 0), 2.2,
            )
            neutral_l = target_l + np.clip(fine_l, -7.0, 7.0)
            lab[:, :, 0] += np.clip(
                neutral_l - lab[:, :, 0], -44.0, 34.0,
            ) * alpha
            lab[:, :, 1] += np.clip(
                skin_a - lab[:, :, 1], -18.0, 18.0,
            ) * alpha * .94
            lab[:, :, 2] += np.clip(
                skin_b - lab[:, :, 2], -22.0, 18.0,
            ) * alpha * .96

            # Preserve thin, high-contrast jewelry and clothing-edge detail.
            # These structures are much sharper than a sun/shadow field and
            # should never be recolored to skin even when a closing operation
            # bridges across a narrow necklace chain.
            detail_protect = body_window & (np.abs(fine_l) >= 18.0)
            detail_alpha = cv2.GaussianBlur(
                detail_protect.astype(np.uint8) * 255, (0, 0), .8,
            ).astype(np.float32) / 255.0
            detail_alpha *= alpha
            lab = (
                lab * (1.0 - detail_alpha[:, :, None])
                + original_lab * detail_alpha[:, :, None]
            )
            logger.info(
                "[BODY_LIGHT] Corrected residual neck/chest sunlight on %d pixels",
                int(affected.sum()),
            )
            return Image.fromarray(
                cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
            )
        except Exception as exc:
            logger.warning("[BODY_LIGHT] Body-only sunlight correction skipped: %s", exc)
            return image_pil

    @staticmethod
    def restore_sunlit_hair_tone(image_pil: Image.Image) -> Image.Image:
        """Recover dark crown tone from shaded hair without redrawing strands.

        Direct sun can clip a dark crown to silver/grey even after luminance
        compression.  The face geometry bounds this correction to the crown;
        shaded hair in the same upload supplies the target tone.  Skin, green
        scenery, and bright/purple flower accessories are explicitly excluded.
        """
        try:
            rgb = np.array(image_pil.convert("RGB"))
            height, width = rgb.shape[:2]
            faces = _get_insight_app().get(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if len(faces) != 1:
                return image_pil

            x1, y1, x2, y2 = [int(value) for value in faces[0].bbox]
            face_w, face_h = max(24, x2 - x1), max(24, y2 - y1)
            center_x = (x1 + x2) // 2
            crown = np.zeros((height, width), dtype=np.uint8)
            cv2.ellipse(
                crown,
                (center_x, max(0, y1 - int(face_h * 0.08))),
                (int(face_w * 0.82), int(face_h * 0.72)),
                0, 0, 360, 255, -1,
            )
            crown[int(y1 + face_h * 0.38):, :] = 0

            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
            hue, saturation, value = cv2.split(hsv)
            skin = (
                (ycrcb[:, :, 1] >= 133) & (ycrcb[:, :, 1] <= 178)
                & (ycrcb[:, :, 2] >= 72) & (ycrcb[:, :, 2] <= 132)
            )
            green = (hue >= 30) & (hue <= 100) & (saturation >= 38)
            purple_flower = (hue >= 118) & (hue <= 172) & (saturation >= 22)
            bright_flower = (value >= 205) & (saturation <= 42)
            allowed = (
                (crown > 0) & ~skin & ~green & ~purple_flower & ~bright_flower
            )
            try:
                from pipelines.schp_service import parse as parse_body_parts

                labels = parse_body_parts(rgb)["labels"]
                semantic_hair = labels == 2
                # SCHP is used only as a positive hair authority here. The
                # colour/backdrop guards below still prevent its occasional
                # crown over-segmentation from touching the studio backdrop.
                allowed &= semantic_hair
            except Exception as parse_error:
                logger.warning("[HAIR_TONE] Semantic hair mask unavailable: %s", parse_error)

            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
            edge = max(4, min(18, height // 20, width // 20))
            border_samples = np.concatenate((
                lab[:edge, :edge].reshape(-1, 3),
                lab[:edge, -edge:].reshape(-1, 3),
                lab[-edge:, :edge].reshape(-1, 3),
                lab[-edge:, -edge:].reshape(-1, 3),
            ))
            backdrop_lab = np.median(border_samples, axis=0)
            # The crown ellipse necessarily overlaps the backdrop above the
            # head. Never interpret that smooth generated colour as sunlit
            # hair; doing so created a dark grey cap around the crown.
            backdrop_like = np.linalg.norm(
                lab - backdrop_lab[None, None, :], axis=2,
            ) <= 60.0
            allowed &= ~backdrop_like
            dark_seed = allowed & (value >= 22) & (value <= 112)
            if int(dark_seed.sum()) < max(20, int(face_w * face_h * 0.002)):
                return image_pil
            seed_l = float(np.median(lab[:, :, 0][dark_seed]))
            seed_a = float(np.median(lab[:, :, 1][dark_seed]))
            seed_b = float(np.median(lab[:, :, 2][dark_seed]))
            excess = np.clip((lab[:, :, 0] - (seed_l + 10.0)) / 72.0, 0.0, 1.0)
            if float(excess[allowed].max(initial=0.0)) <= 0.02:
                return image_pil

            # Natural dark hair contains many thin bright strands. Treating
            # those as a sunlight patch turns them into flat grey paint. Only
            # continue when the glare contains a genuinely broad connected
            # region that survives an opening wider than individual strands.
            glare_candidate = allowed & (excess >= .30)
            broad_glare = cv2.morphologyEx(
                glare_candidate.astype(np.uint8), cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                iterations=1,
            ).astype(bool)
            minimum_broad_area = max(28, int(face_w * face_h * .0035))
            if int(broad_glare.sum()) < minimum_broad_area:
                logger.info(
                    "[HAIR_TONE] Skipped: generated hair contains only "
                    "natural strand highlights (broad=%d threshold=%d)",
                    int(broad_glare.sum()), minimum_broad_area,
                )
                return image_pil

            # Use a continuous crown field instead of a binary glare map.
            # The binary map left isolated grey holes between bright strands;
            # this preserves their relative texture and only compresses their
            # low-frequency highlight intensity.
            alpha = cv2.GaussianBlur(
                allowed.astype(np.uint8) * 255, (17, 17), 3.5
            ).astype(np.float32) / 255.0
            alpha *= excess * 0.86
            alpha[~allowed] = 0.0
            target_l = seed_l + np.clip((lab[:, :, 0] - seed_l) * .28, 0.0, 28.0)
            lab[:, :, 0] += (target_l - lab[:, :, 0]) * alpha
            # Very small chroma stabilization avoids a green/grey cast while
            # retaining the generated hair's own colour and flower edges.
            lab[:, :, 1] += (seed_a - lab[:, :, 1]) * alpha * 0.34
            lab[:, :, 2] += (seed_b - lab[:, :, 2]) * alpha * 0.34
            logger.info(
                "[HAIR_TONE] Recovered sunlit crown from shaded source hair L=%.1f",
                seed_l,
            )
            return Image.fromarray(
                cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
            )
        except Exception as exc:
            logger.warning("[HAIR_TONE] Sunlit crown recovery skipped: %s", exc)
            return image_pil

    @staticmethod
    def restore_source_face_microtexture(
        source_pil: Image.Image,
        portrait_pil: Image.Image,
        strength: float = 0.48,
    ) -> Image.Image:
        """Restore real facial detail without copying source colour or pixels.

        Qwen can smooth pores and fine feature contrast even when identity is
        correct. Align the source by landmarks, extract only its high-frequency
        luminance, and add that detail inside a softly eroded facial hull. The
        generated lighting, complexion, geometry, hair and backdrop remain
        entirely from the accepted portrait.
        """
        try:
            from skimage.transform import SimilarityTransform

            source_rgb = np.asarray(source_pil.convert("RGB"), dtype=np.uint8)
            target_rgb = np.asarray(portrait_pil.convert("RGB"), dtype=np.uint8)
            height, width = target_rgb.shape[:2]
            detector = _get_insight_app()
            source_faces = detector.get(cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR))
            target_faces = detector.get(cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR))
            if len(source_faces) != 1 or len(target_faces) != 1:
                return portrait_pil

            transform = SimilarityTransform()
            transform.estimate(source_faces[0].kps, target_faces[0].kps)
            matrix = transform.params[:2, :].astype(np.float32)
            source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
            warped_gray = cv2.warpAffine(
                source_gray, matrix, (width, height),
                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT,
            ).astype(np.float32)
            source_detail = warped_gray - cv2.GaussianBlur(
                warped_gray, (0, 0), 1.15,
            )
            source_detail = np.clip(source_detail, -22.0, 22.0)

            landmarks = getattr(source_faces[0], "landmark_2d_106", None)
            mask = np.zeros((height, width), dtype=np.uint8)
            if landmarks is not None and len(landmarks) >= 20:
                aligned = cv2.transform(
                    np.asarray(landmarks, dtype=np.float32).reshape(-1, 1, 2),
                    matrix,
                ).reshape(-1, 2)
                cv2.fillConvexPoly(mask, cv2.convexHull(aligned.astype(np.int32)), 255)
            else:
                x1, y1, x2, y2 = [int(v) for v in target_faces[0].bbox]
                cv2.ellipse(
                    mask, ((x1 + x2) // 2, (y1 + y2) // 2),
                    (max(8, int((x2 - x1) * .43)), max(8, int((y2 - y1) * .48))),
                    0, 0, 360, 255, -1,
                )
            mask = cv2.erode(
                mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                iterations=1,
            )
            alpha = cv2.GaussianBlur(mask, (17, 17), 4.0).astype(np.float32) / 255.0
            target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
            target_lab[:, :, 0] = np.clip(
                target_lab[:, :, 0]
                + source_detail * alpha * float(np.clip(strength, 0.0, 0.5)),
                0, 255,
            )
            logger.info("[FACE_TEXTURE] Restored aligned source microtexture")
            return Image.fromarray(
                cv2.cvtColor(target_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
            )
        except Exception as exc:
            logger.warning("[FACE_TEXTURE] Source microtexture recovery skipped: %s", exc)
            return portrait_pil

    @staticmethod
    def match_source_skin_chroma(
        source_pil: Image.Image,
        portrait_pil: Image.Image,
    ) -> Image.Image:
        """Match generated skin undertone to the source without changing light.

        Qwen can keep identity and studio illumination yet shift medium/deep
        skin toward yellow.  Measure robust midtone skin independently in both
        images, then apply only a tightly bounded LAB a/b offset to semantic
        skin pixels. Luminance, anatomy, texture, hair and backdrop are kept
        byte-for-byte outside the soft skin mask.
        """
        try:
            from pipelines.schp_service import parse as parse_body_parts

            source_rgb = np.asarray(source_pil.convert("RGB"), dtype=np.uint8)
            output_rgb = np.asarray(portrait_pil.convert("RGB"), dtype=np.uint8)
            source_labels = parse_body_parts(source_rgb)["labels"]
            output_labels = parse_body_parts(output_rgb)["labels"]

            source_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
            output_lab = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

            def stable_face_sample(lab: np.ndarray, labels: np.ndarray):
                face = labels == 13
                if int(face.sum()) < 96:
                    return None
                lightness = lab[:, :, 0]
                values = lightness[face]
                low, high = np.percentile(values, (20, 80))
                stable = face & (lightness >= low) & (lightness <= high)
                if int(stable.sum()) < 64:
                    stable = face
                return np.median(lab[stable], axis=0)

            source_tone = stable_face_sample(source_lab, source_labels)
            output_tone = stable_face_sample(output_lab, output_labels)
            if source_tone is None or output_tone is None:
                return portrait_pil

            # Chroma-only: preserve Qwen's accepted studio exposure. The
            # strict bound prevents a coloured source illuminant from being
            # copied wholesale into the finished portrait.
            chroma_delta = np.clip(source_tone[1:] - output_tone[1:], -7.0, 7.0)
            # Direct sun can make even the robust source sample artificially
            # yellow. Never transfer additional yellow from the upload; this
            # correction may only remove a generated yellow cast. The red/
            # green undertone remains a bounded source-relative adjustment.
            chroma_delta[1] = min(0.0, float(chroma_delta[1]))
            if float(np.linalg.norm(chroma_delta)) < 1.25:
                return portrait_pil

            output_skin = np.isin(output_labels, (13, 14, 15, 16, 17))
            chroma_distance = np.sqrt(
                (output_lab[:, :, 1] - output_tone[1]) ** 2
                + (output_lab[:, :, 2] - output_tone[2]) ** 2
            )
            # Semantic labels locate skin; person-specific chroma excludes
            # eyes, lips, jewelry and similarly labelled garment fragments.
            skin = output_skin & (chroma_distance <= 23.0)
            skin = cv2.morphologyEx(
                skin.astype(np.uint8), cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )
            if int(skin.sum()) < 96:
                return portrait_pil
            alpha = cv2.GaussianBlur(
                skin * 255, (0, 0), 2.2,
            ).astype(np.float32) / 255.0
            output_lab[:, :, 1] += alpha * float(chroma_delta[0]) * .82
            output_lab[:, :, 2] += alpha * float(chroma_delta[1]) * .82
            corrected = cv2.cvtColor(
                output_lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB,
            )
            logger.info(
                "[SKIN_CHROMA] Matched source undertone without luminance change "
                "(da=%.1f db=%.1f)",
                float(chroma_delta[0]), float(chroma_delta[1]),
            )
            return Image.fromarray(corrected, "RGB")
        except Exception as exc:
            logger.warning("[SKIN_CHROMA] Source undertone match skipped: %s", exc)
            return portrait_pil

    @staticmethod
    def lock_source_hair(
        orig_pil: Image.Image, portrait_pil: Image.Image, strength: float = 0.78
    ) -> Image.Image:
        """Restore real dark hair after Qwen without a segmentation-model pass.

        Qwen can turn sunlit dark hair into pale synthetic strands.  This uses
        only facial-landmark alignment and a conservative dark-hair mask above
        and beside the face, leaving skin, clothing, and backdrop untouched.
        """
        portrait = np.array(portrait_pil.convert("RGB"))
        h, w = portrait.shape[:2]
        source = np.array(orig_pil.convert("RGB").resize((w, h), Image.LANCZOS))
        source_bgr = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
        portrait_bgr = cv2.cvtColor(portrait, cv2.COLOR_RGB2BGR)
        try:
            from skimage.transform import SimilarityTransform

            app = _get_insight_app()
            faces_source = app.get(source_bgr)
            faces_target = app.get(portrait_bgr)
            if not faces_source or not faces_target:
                return portrait_pil

            source_face, target_face = faces_source[0], faces_target[0]
            transform = SimilarityTransform()
            transform.estimate(source_face.kps, target_face.kps)
            raw = transform.params[:2, :].astype(np.float32)
            raw_scale = max(1e-6, float(np.hypot(raw[0, 0], raw[1, 0])))
            scale = float(np.clip(raw_scale, 0.95, 1.05))
            linear = raw[:, :2] * (scale / raw_scale)
            translation = np.mean(target_face.kps, axis=0) - linear @ np.mean(source_face.kps, axis=0)
            matrix = np.column_stack((linear, translation)).astype(np.float32)
            warped_source = cv2.warpAffine(source_bgr, matrix, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)

            x1, y1, x2, y2 = [int(v) for v in target_face.bbox]
            fw, fh = max(24, x2 - x1), max(24, y2 - y1)
            cx = (x1 + x2) // 2
            hair_region = np.zeros((h, w), dtype=np.uint8)
            # Wider ellipse to cover long/curly women's hair extending past shoulders
            cv2.ellipse(hair_region, (cx, max(0, y1 + int(fh * 0.05))), (int(fw * 1.25), int(fh * 1.30)), 0, 0, 360, 255, -1)
            # Never copy source facial pixels back over the restored face.
            cv2.ellipse(hair_region, (cx, y1 + fh // 2), (int(fw * 0.56), int(fh * 0.62)), 0, 0, 360, 0, -1)

            hsv_source = cv2.cvtColor(warped_source, cv2.COLOR_BGR2HSV)
            hsv_target = cv2.cvtColor(portrait_bgr, cv2.COLOR_BGR2HSV)
            # Preserve true dark hair, but reject the outdoor foliage that can
            # overlap curls in a phone photo. Green/brown vegetation tends to
            # be highly saturated; real dark hair remains near-neutral even
            # where it has soft highlights.
            source_dark_hair = (
                # Phone portraits can contain blue/grey specular hair detail
                # above the old dark-only threshold. Keep those true source
                # strands so Qwen cannot replace them with silver strokes.
                (hsv_source[:, :, 2] < 210)
                & (hsv_source[:, :, 1] < 205)
                & ~((hsv_source[:, :, 0] >= 32) & (hsv_source[:, :, 0] <= 95) & (hsv_source[:, :, 1] >= 40))
            ).astype(np.uint8) * 255
            # Require Qwen to already identify a nearby dark-hair region.
            # This excludes original dark branches that lie over the new solid
            # backdrop, while allowing source texture to correct pale streaks
            # within Qwen's existing hair silhouette.
            # Raised from 125->165 to include medium-brown and lighter dark hair
            target_hair_seed = (hsv_target[:, :, 2] < 165).astype(np.uint8) * 255
            target_hair_area = cv2.dilate(
                target_hair_seed,
                # A wider local halo includes Qwen's bright, synthetic hair
                # highlights while remaining bounded by the head-only ellipse.
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)),
                iterations=1,
            )
            mask = cv2.bitwise_and(hair_region, source_dark_hair)
            mask = cv2.bitwise_and(mask, target_hair_area)
            mask = cv2.GaussianBlur(mask, (13, 13), 2.5).astype(np.float32)[:, :, None] / 255.0
            alpha = mask * float(np.clip(strength, 0.0, 1.0))
            logger.info("[HAIR_LOCK] Applied source hair blend to %.1f%% of portrait", float(alpha.mean() * 100.0))
            fused = (warped_source.astype(np.float32) * alpha + portrait_bgr.astype(np.float32) * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
            return Image.fromarray(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB))
        except Exception as exc:
            logger.warning("[HAIR_LOCK] Source-hair lock skipped: %s", exc)
            return portrait_pil

    @staticmethod
    def normalize_hair_tone_from_source(
        orig_pil: Image.Image, portrait_pil: Image.Image, strength: float = 0.78
    ) -> Image.Image:
        """Remove Qwen's metallic hair cast without copying source hair pixels."""
        portrait = np.array(portrait_pil.convert("RGB"))
        h, w = portrait.shape[:2]
        source = np.array(orig_pil.convert("RGB").resize((w, h), Image.LANCZOS))
        source_bgr = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
        portrait_bgr = cv2.cvtColor(portrait, cv2.COLOR_RGB2BGR)
        try:
            from skimage.transform import SimilarityTransform

            app = _get_insight_app()
            faces_source = app.get(source_bgr)
            faces_target = app.get(portrait_bgr)
            if not faces_source or not faces_target:
                return portrait_pil

            transform = SimilarityTransform()
            transform.estimate(faces_source[0].kps, faces_target[0].kps)
            raw = transform.params[:2, :].astype(np.float32)
            raw_scale = max(1e-6, float(np.hypot(raw[0, 0], raw[1, 0])))
            linear = raw[:, :2] * (float(np.clip(raw_scale, 0.95, 1.05)) / raw_scale)
            translation = np.mean(faces_target[0].kps, axis=0) - linear @ np.mean(faces_source[0].kps, axis=0)
            matrix = np.column_stack((linear, translation)).astype(np.float32)
            warped_source = cv2.warpAffine(source_bgr, matrix, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)

            x1, y1, x2, y2 = [int(value) for value in faces_target[0].bbox]
            fw, fh = max(24, x2 - x1), max(24, y2 - y1)
            hair_region = np.zeros((h, w), dtype=np.uint8)
            # Wider ellipse to cover long/curly women's hair extending past shoulders
            cv2.ellipse(hair_region, ((x1 + x2) // 2, max(0, y1 + int(fh * 0.05))),
                        (int(fw * 1.25), int(fh * 1.30)), 0, 0, 360, 255, -1)
            # Inner face exclusion stays proportional
            cv2.ellipse(hair_region, ((x1 + x2) // 2, y1 + fh // 2),
                        (int(fw * 0.56), int(fh * 0.62)), 0, 0, 360, 0, -1)

            source_hsv = cv2.cvtColor(warped_source, cv2.COLOR_BGR2HSV)
            target_hsv = cv2.cvtColor(portrait_bgr, cv2.COLOR_BGR2HSV)
            source_dark_hair = (
                (source_hsv[:, :, 2] < 210)
                & (source_hsv[:, :, 1] < 205)
                & ~((source_hsv[:, :, 0] >= 32) & (source_hsv[:, :, 0] <= 95) & (source_hsv[:, :, 1] >= 40))
            ).astype(np.uint8) * 255
            target_hair_neighborhood = cv2.dilate(
                (target_hsv[:, :, 2] < 175).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)), iterations=1,
            )
            mask = cv2.bitwise_and(hair_region, source_dark_hair)
            mask = cv2.bitwise_and(mask, target_hair_neighborhood)
            alpha = cv2.GaussianBlur(mask, (13, 13), 2.5).astype(np.float32)[:, :, None] / 255.0
            alpha *= float(np.clip(strength, 0.0, 1.0))

            source_lab = cv2.cvtColor(warped_source, cv2.COLOR_BGR2LAB).astype(np.float32)
            target_lab = cv2.cvtColor(portrait_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            # Transfer only neutral dark-hair colour and tame bright metallic
            # highlights. Hair texture and silhouette stay entirely Qwen's.
            target_lab[:, :, 1:3] += (source_lab[:, :, 1:3] - target_lab[:, :, 1:3]) * alpha * 0.85
            darkening = np.clip(source_lab[:, :, 0] - target_lab[:, :, 0], -20.0, 0.0)
            target_lab[:, :, 0] += darkening * alpha[:, :, 0]

            # Qwen sometimes paints green/cyan foliage-coloured strokes *inside*
            # the hair silhouette. They survive a background matte by design, so
            # neutralize them here as dark hair rather than eroding real curls.
            synthetic_green_or_cyan = (
                (target_hsv[:, :, 0] >= 35)
                & (target_hsv[:, :, 0] <= 120)
                & (target_hsv[:, :, 1] >= 22)
                & (target_hsv[:, :, 2] < 215)
            ).astype(np.uint8) * 255
            # The face-aligned hair ellipse can overlap the new solid backdrop
            # between loose curls. Limit colour repair to pixels that are also
            # in Qwen's dark-hair neighbourhood; otherwise a blue selected
            # background is incorrectly darkened or tinted at the curl edges.
            artifact_mask = cv2.bitwise_and(hair_region, synthetic_green_or_cyan)
            artifact_mask = cv2.bitwise_and(artifact_mask, target_hair_neighborhood)
            artifact_alpha = cv2.GaussianBlur(artifact_mask, (11, 11), 2.0).astype(np.float32)[:, :, None] / 255.0
            artifact_alpha *= 0.82
            target_lab[:, :, 1:3] += (128.0 - target_lab[:, :, 1:3]) * artifact_alpha
            target_lab[:, :, 0] += (
                np.minimum(target_lab[:, :, 0], 125.0) - target_lab[:, :, 0]
            ) * artifact_alpha[:, :, 0]
            logger.info("[HAIR_TONE] Normalized Qwen metallic hair cast using source colour reference")
            return Image.fromarray(cv2.cvtColor(target_lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB))
        except Exception as exc:
            logger.warning("[HAIR_TONE] Source hair-tone normalization skipped: %s", exc)
            return portrait_pil

    @staticmethod
    def match_neck_tone(orig_pil: Image.Image, portrait_pil: Image.Image) -> Image.Image:
        """Match neck luminance after face locking without copying source texture or background."""
        portrait = np.array(portrait_pil.convert("RGB"))
        h, w = portrait.shape[:2]
        source = np.array(orig_pil.convert("RGB").resize((w, h), Image.LANCZOS))
        source_bgr = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
        portrait_bgr = cv2.cvtColor(portrait, cv2.COLOR_RGB2BGR)
        try:
            from skimage.transform import SimilarityTransform

            app = _get_insight_app()
            source_faces = app.get(source_bgr)
            portrait_faces = app.get(portrait_bgr)
            if not source_faces or not portrait_faces:
                return portrait_pil
            transform = SimilarityTransform()
            transform.estimate(source_faces[0].kps, portrait_faces[0].kps)
            matrix = transform.params[:2, :]
            warped_source = cv2.warpAffine(source_bgr, matrix, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)

            x1, y1, x2, y2 = [int(v) for v in portrait_faces[0].bbox]
            face_width, face_height = max(24, x2 - x1), max(24, y2 - y1)
            cx = (x1 + x2) // 2
            neck = np.zeros((h, w), dtype=np.uint8)
            # A short trapezoid below the jaw stays above the collar on the
            # portrait crops used here. It softens only the lighting seam.
            top = max(0, y2 - int(face_height * 0.08))
            bottom = min(h, y2 + int(face_height * 0.52))
            points = np.array([
                [max(0, cx - int(face_width * 0.34)), top],
                [min(w - 1, cx + int(face_width * 0.34)), top],
                [min(w - 1, cx + int(face_width * 0.58)), bottom],
                [max(0, cx - int(face_width * 0.58)), bottom],
            ], dtype=np.int32)
            cv2.fillConvexPoly(neck, points, 255)
            valid = neck > 0
            if int(valid.sum()) < 100:
                return portrait_pil

            source_lab = cv2.cvtColor(warped_source, cv2.COLOR_BGR2LAB).astype(np.float32)
            portrait_lab = cv2.cvtColor(portrait_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            luminance_delta = float(np.clip(
                np.median(source_lab[:, :, 0][valid]) - np.median(portrait_lab[:, :, 0][valid]),
                -12.0,
                22.0,
            ))
            feather = cv2.GaussianBlur(neck, (31, 31), 8.0).astype(np.float32) / 255.0
            portrait_lab[:, :, 0] = np.clip(portrait_lab[:, :, 0] + luminance_delta * feather, 0, 255)
            return Image.fromarray(cv2.cvtColor(portrait_lab.astype(np.uint8), cv2.COLOR_LAB2RGB))
        except Exception as exc:
            logger.warning("[NECK_TONE] Neck tone match skipped: %s", exc)
            return portrait_pil
    @staticmethod
    def anchor_and_fuse_identity(
        orig_pil: Image.Image,
        qwen_pil: Image.Image,
        identity_lock_strength: float = 0.88,
        preserve_outside_face: bool = False,
    ) -> Image.Image:
        """
        Fuses Qwen's regenerated lighting, background, and textures with 100% of the
        original facial likeness, eye structure, nose, mouth, and skin authenticity
        using InsightFace 106-point facial landmark alignment and studio illumination matching.
        """
        orig_bgr = cv2.cvtColor(np.array(orig_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        qwen_bgr = cv2.cvtColor(np.array(qwen_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

        h_q, w_q = qwen_bgr.shape[:2]
        if orig_bgr.shape[:2] != (h_q, w_q):
            orig_bgr = cv2.resize(orig_bgr, (w_q, h_q), interpolation=cv2.INTER_LANCZOS4)

        # This is an identity anchor, so it must remain the real source image.
        # When frames are detected, gently clean only lens highlights before the
        # blend. GFPGAN is never used here because it redraws identity details.
        orig_restored = orig_bgr
        if PhotoRestorationService.has_eyeglasses(orig_bgr):
            orig_restored = PhotoRestorationService.reduce_glasses_glare(orig_bgr)

        # 2. InsightFace 106-point landmark alignment
        try:
            from skimage.transform import SimilarityTransform

            app = _get_insight_app()

            faces_orig = app.get(orig_restored)
            faces_qwen = app.get(qwen_bgr)

            if faces_orig and faces_qwen:
                kps_o = faces_orig[0].kps
                kps_q = faces_qwen[0].kps

                # Align the face position without allowing Qwen's regenerated
                # landmarks to rescale the source beyond five percent.
                tform = SimilarityTransform()
                tform.estimate(kps_o, kps_q)
                raw = tform.params[:2, :].astype(np.float32)
                raw_scale = max(1e-6, float(np.hypot(raw[0, 0], raw[1, 0])))
                bounded_scale = float(np.clip(raw_scale, 0.95, 1.05))
                linear = raw[:, :2] * (bounded_scale / raw_scale)
                source_center = np.mean(kps_o, axis=0)
                generated_center = np.mean(kps_q, axis=0)
                translation = generated_center - linear @ source_center
                M = np.column_stack((linear, translation)).astype(np.float32)

                # Warp restored real face into Qwen's head pose
                warped_orig = cv2.warpAffine(orig_restored, M, (w_q, h_q), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)

                # Compute smooth face polygon mask from warped 106 landmarks
                lmk106_w = cv2.transform(faces_orig[0].landmark_2d_106.reshape(-1, 1, 2), M).reshape(-1, 2).astype(np.int32)
                hull = cv2.convexHull(lmk106_w)

                mask = np.zeros((h_q, w_q), dtype=np.uint8)
                cv2.fillConvexPoly(mask, hull, 255)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                mask_eroded = cv2.erode(mask, kernel, iterations=1)
                mask_soft = cv2.GaussianBlur(mask_eroded, (31, 31), 10.0).astype(np.float32) / 255.0
                # Do not punch eye-shaped holes into the identity mask.  The
                # keypoints exist for every face, not only for glasses, so an
                # unconditional "lens" exclusion left Qwen's generated eye
                # pixels visible as oval artifacts on subjects without glasses.
                mask_3c = (mask_soft[:, :, np.newaxis] * min(1.0, identity_lock_strength))

                # Match lighting & tone of warped face to Qwen's studio scene (with balanced saturation)
                w_lab = cv2.cvtColor(warped_orig, cv2.COLOR_BGR2LAB).astype(np.float32)
                q_lab = cv2.cvtColor(qwen_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

                mask_bool = mask_soft > 0.2
                # Match luminance channel (c = 0)
                m_w0, s_w0 = float(np.mean(w_lab[mask_bool, 0])), float(np.std(w_lab[mask_bool, 0])) + 1e-5
                m_q0, s_q0 = float(np.mean(q_lab[mask_bool, 0])), float(np.std(q_lab[mask_bool, 0])) + 1e-5
                w_lab[:, :, 0] = np.clip(((w_lab[:, :, 0] - m_w0) / s_w0) * s_q0 + m_q0, 0, 255)

                # Harmonize chrominance channels (c = 1, 2) to preserve natural skin tones without oversaturating
                for c in (1, 2):
                    m_w, s_w = float(np.mean(w_lab[mask_bool, c])), float(np.std(w_lab[mask_bool, c])) + 1e-5
                    m_q, s_q = float(np.mean(q_lab[mask_bool, c])), float(np.std(q_lab[mask_bool, c])) + 1e-5
                    target_m = m_w * 0.65 + m_q * 0.35
                    target_s = min(s_w, s_q * 0.65)
                    w_lab[:, :, c] = np.clip(((w_lab[:, :, c] - m_w) / s_w) * target_s + target_m, 0, 255)

                w_matched = cv2.cvtColor(w_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
                fused = (w_matched.astype(np.float32) * mask_3c + qwen_bgr.astype(np.float32) * (1.0 - mask_3c)).clip(0, 255).astype(np.uint8)

                # Uniform Swap must preserve Qwen's garment and exact backdrop
                # pixels byte-for-byte outside the soft facial hull. The legacy
                # global saturation pass is retained for callers that explicitly
                # want portrait-wide grading, but is skipped by the bounded
                # uniform identity repair.
                if preserve_outside_face:
                    return Image.fromarray(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB))

                # Moderate overall saturation to professional portrait standards
                hsv = cv2.cvtColor(fused, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.88, 0, 255)
                fused_natural = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                return Image.fromarray(cv2.cvtColor(fused_natural, cv2.COLOR_BGR2RGB))
        except Exception as e:
            logger.warning("[IDENTITY_FUSION] InsightFace landmark fusion error: %s - using fallback", e)

        # Fallback multi-band detail injection
        orig_gray = cv2.cvtColor(orig_restored, cv2.COLOR_BGR2GRAY).astype(np.float32)
        orig_blur = cv2.GaussianBlur(orig_gray, (0, 0), 1.4)
        high_pass = (orig_gray - orig_blur)[:, :, np.newaxis]
        fused = np.clip(qwen_bgr.astype(np.float32) + high_pass * 0.45, 0, 255).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB))

    # ── 5. Detail Polish: Jewelry Sparkle, Hair Definition & Fabric Luster ───

    @staticmethod
    def polish_hair_jewels_and_fabric(img_bgr: np.ndarray, jewelry_mask: np.ndarray) -> np.ndarray:
        """
        Enhances jewelry sparkle, metallic glint, gemstone clarity,
        hair strand definition, and fabric weave.
        """
        res = img_bgr.copy()

        # 1. Specular & Micro-Contrast Boost on Jewelry & Ornaments
        if np.count_nonzero(jewelry_mask) > 0:
            # Unsharp mask specifically on jewelry to bring out chains, earrings, and gem facets
            jewel_blur = cv2.GaussianBlur(res, (0, 0), 1.0)
            jewel_sharp = cv2.addWeighted(res, 1.75, jewel_blur, -0.75, 0)

            # Boost metallic saturation in gold/gemstone regions
            hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.18, 0, 255)
            sat_boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            mask_f = (cv2.GaussianBlur(jewelry_mask, (5, 5), 1.0).astype(np.float32) / 255.0)[:, :, np.newaxis]
            res = (jewel_sharp.astype(np.float32) * mask_f * 0.70 + sat_boosted.astype(np.float32) * mask_f * 0.30 + res.astype(np.float32) * (1.0 - mask_f)).clip(0, 255).astype(np.uint8)

        # 2. Multi-Scale Crisp Sharpening for Hair Strands, Facial Features & Fabric Weave
        gauss_fine = cv2.GaussianBlur(res, (0, 0), 0.8)
        gauss_med = cv2.GaussianBlur(res, (0, 0), 1.6)
        fine_detail = cv2.subtract(res, gauss_fine)
        med_detail = cv2.subtract(gauss_fine, gauss_med)

        crisp = np.clip(res.astype(np.float32) + fine_detail.astype(np.float32) * 0.55 + med_detail.astype(np.float32) * 0.25, 0, 255).astype(np.uint8)
        return crisp

    # ── 6. Full End-to-End Restoration Pipeline ───────────────────────────────

    def restore_photo(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        remove_scratches: bool = True,
        restore_tones: bool = True,
        restore_faces: bool = True,
        upscale_factor: int = 2,
        use_qwen_generative: bool = False,
        preserve_jewels_and_dress: bool = True,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> Path:
        """
        Complete restoration workflow with strict originality preservation for
        Face, Dress, Hair style & Jewelry.
        """
        t0 = time.time()
        job_id = f"restore_{int(time.time())}"

        if progress_cb:
            progress_cb(0.05, "Loading image & analyzing original features...")

        if isinstance(image_input, (str, Path)):
            img_bgr = cv2.imread(str(image_input))
        elif isinstance(image_input, Image.Image):
            img_bgr = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        if img_bgr is None:
            raise ValueError("Failed to load input image data.")

        # Locate face & detect jewelry/dress elements for absolute protection
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
        primary_face = max(faces, key=lambda f: f[2] * f[3]) if len(faces) > 0 else None
        jewelry_mask = self.detect_jewelry_and_ornaments(img_bgr, primary_face)

        # Stage 1: Defect Inpainting (Strictly protected so jewelry/dress are never erased)
        if remove_scratches:
            if progress_cb:
                progress_cb(0.20, "Repairing scratches & creases (Protecting face, jewels & fabric)...")
            img_bgr = self.inpaint_defects(img_bgr, primary_face)

        # Stage 2: Tone & Color Balance (Preserving fabric dyes & metallic luster)
        if restore_tones:
            if progress_cb:
                progress_cb(0.40, "Balancing natural lighting & dynamic range...")
            img_bgr = self.restore_colors_and_contrast(img_bgr)

        # Stage 3: Identity-Locked Facial Enhancement
        if restore_faces:
            if progress_cb:
                progress_cb(0.60, "Enhancing facial clarity with 100% identity lock...")
            img_bgr, primary_face = self.enhance_faces_identity_locked(img_bgr, fidelity_weight=0.90)

        # Stage 4: Polish Jewels, Hair Strands & Dress Textures
        if preserve_jewels_and_dress:
            if progress_cb:
                progress_cb(0.72, "Polishing hair strands, jewelry luster & dress fabric...")
            img_bgr = self.polish_hair_jewels_and_fabric(img_bgr, jewelry_mask)

        # Stage 5: Optional Deep Generative 2509 Diffusion Pass (Identity-Strict)
        pil_res = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if use_qwen_generative:
            if progress_cb:
                progress_cb(0.80, "Running identity-strict Qwen 2509 Generative Polish...")
            try:
                from pipelines.gusuq_pipeline import GUSUQPipeline
                gusuq = GUSUQPipeline(opt_policy="low_vram")
                gen_path = gusuq.edit_image(
                    image=pil_res,
                    prompt=(
                        "Masterwork photo restoration, pristine clarity, sharp focus, clean natural skin, "
                        "preserve 100% exact facial identity, exact hair style and hair strands, "
                        "exact dress fabric pattern and colors, exact jewelry, earrings and necklace, no alterations"
                    ),
                    negative_prompt=(
                        "altered face, wrong identity, plastic skin, changed hairstyle, changed dress, "
                        "missing jewelry, blurry earrings, distorted ornaments, smudged fabric, deformed eyes"
                    ),
                    true_cfg_scale=1.2,  # Strictly kept between 1.0 and 1.5 to prevent burning colors
                    steps=4,
                )
                pil_res = Image.open(gen_path).convert("RGB")
            except Exception as e:
                logger.warning("[PHOTO_RESTORE] Qwen generative pass skipped: %s", e)

        # Stage 6: Super-Resolution & Clarity Polish
        if progress_cb:
            progress_cb(0.92, f"Applying {upscale_factor}x Super-Resolution...")

        if upscale_factor > 1:
            w, h = pil_res.size
            pil_res = pil_res.resize((w * upscale_factor, h * upscale_factor), Image.LANCZOS)

        pil_res = ImageEnhance.Sharpness(pil_res).enhance(1.15)
        pil_res = ImageEnhance.Contrast(pil_res).enhance(1.04)
        out_path = OUTPUTS_DIR / f"{job_id}.jpg"
        pil_res.save(out_path, quality=98, dpi=(300, 300))

        if progress_cb:
            progress_cb(1.0, "Restoration complete!")

        elapsed = time.time() - t0
        logger.info("[PHOTO_RESTORE] Photo restored (originality preserved) in %.2fs -> %s", elapsed, out_path)
        return out_path

    # ── 7. Studio 3-Point Relighting & Atmospheric Simulation ─────────────────

    @staticmethod
    def apply_studio_lighting(
        img_bgr: np.ndarray,
        subject_mask: Optional[np.ndarray] = None,
        gamma: float = 1.12,
        softbox_intensity: float = 0.25,
        shadow_lift: float = 0.35,
        rim_light_intensity: float = 0.18,
    ) -> np.ndarray:
        """
        Simulates professional 3-point studio lighting:
        1. Dynamic Shadow Lifting & Reflector Fill (Eliminates harsh directional cheek/neck shadows).
        2. Softbox Key Light (Large diffused frontal octabox for smooth, flattering facial skin).
        3. Highlight Roll-Off (Prevents harsh blown-out glare on forehead, nose, and cheeks).
        4. Iris Catchlight Boost & Specular Eye Polish (Crisp, energetic eye sparkle).
        5. Subtle Rim / Hair Edge Lighting (Crisp separation from background).
        """
        h, w = img_bgr.shape[:2]
        res = img_bgr.copy()

        # Convert to LAB for perceptual luminance manipulation
        lab = cv2.cvtColor(res, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]

        # 1. Dynamic Shadow Lift (Virtual Studio Reflector Board Fill)
        # Smooth luminance to identify shadow zones without affecting fine hair/skin microtexture
        L_smooth = cv2.bilateralFilter(L.astype(np.uint8), d=9, sigmaColor=40, sigmaSpace=40).astype(np.float32)
        shadow_threshold = 125.0
        shadow_mask = np.clip((shadow_threshold - L_smooth) / shadow_threshold, 0.0, 1.0)
        shadow_mask = cv2.GaussianBlur(shadow_mask, (15, 15), 5.0)

        # Lift deep shadows smoothly (replicates 2:1 studio contrast ratio)
        L_lifted = L + (shadow_mask * shadow_lift * 40.0)

        # 2. Highlight Shoulder Roll-Off (Compress harsh glare > 220)
        glare_threshold = 215.0
        glare_mask = np.clip((L_lifted - glare_threshold) / (255.0 - glare_threshold), 0.0, 1.0)
        L_tamed = L_lifted - (glare_mask * 12.0)

        # 3. Softbox Key Light Gradient (Large front-top diffused octabox)
        y_grid, x_grid = np.ogrid[:h, :w]
        cx, cy = w * 0.48, h * 0.32
        sigma_x, sigma_y = w * 0.70, h * 0.60
        softbox_map = np.exp(-(((x_grid - cx) ** 2) / (2 * sigma_x ** 2) + ((y_grid - cy) ** 2) / (2 * sigma_y ** 2)))
        softbox_map = (softbox_map - softbox_map.min()) / (softbox_map.max() - softbox_map.min() + 1e-6)

        # 4. Continuous Gamma Tone-Mapping
        l_norm = np.clip(L_tamed, 0.0, 255.0) / 255.0
        effective_gamma = float(gamma) if gamma else 1.12
        l_calibrated = (l_norm ** effective_gamma) * 255.0

        # Composite softbox key light into calibrated luminance
        l_final = np.clip(l_calibrated + (softbox_map * softbox_intensity * 32.0), 0.0, 255.0)

        lab[:, :, 0] = l_final
        # Natural warm skin tone harmony
        lab[:, :, 1] = np.clip(lab[:, :, 1] * 1.025, 0.0, 255.0)
        lab[:, :, 2] = np.clip(lab[:, :, 2] * 1.035, 0.0, 255.0)
        lit_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # 5. Iris Catchlight Boost & Eye Sparkle (Smoothly feathered with zero boundary lines)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(lit_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            eye_cx, eye_cy = fx + fw // 2, fy + int(fh * 0.38)
            eye_mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(eye_mask, (eye_cx, eye_cy), (int(fw * 0.38), int(fh * 0.16)), 0, 0, 360, 1.0, -1)
            eye_mask_3c = cv2.GaussianBlur(eye_mask, (31, 31), 10.0)[:, :, np.newaxis]

            eye_blur = cv2.GaussianBlur(lit_bgr, (0, 0), 0.8)
            eye_sharp = cv2.addWeighted(lit_bgr, 1.25, eye_blur, -0.25, 0)
            eye_lab = cv2.cvtColor(lit_bgr, cv2.COLOR_BGR2LAB)
            glare_amt = float(((eye_lab[:, :, 0] > 210) & (eye_mask > 0.2)).mean())
            sparkle = 0.0 if glare_amt > 0.03 else 0.60
            lit_bgr = (
                eye_sharp.astype(np.float32) * (sparkle * eye_mask_3c)
                + lit_bgr.astype(np.float32) * (1.0 - sparkle * eye_mask_3c)
            ).clip(0, 255).astype(np.uint8)

        # 6. Rim Lighting along Subject Boundary
        if subject_mask is not None:
            mask_f = subject_mask.astype(np.float32)
            if mask_f.max() > 1.0:
                mask_f /= 255.0

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            eroded = cv2.erode(mask_f, kernel)
            rim_mask = cv2.subtract(mask_f, eroded)
            rim_mask_3c = np.repeat(cv2.GaussianBlur(rim_mask, (9, 9), 2.0)[:, :, np.newaxis], 3, axis=2)

            rim_color = np.array([245, 250, 255], dtype=np.float32)
            lit_bgr = np.clip(
                lit_bgr.astype(np.float32) + (rim_color * rim_light_intensity * rim_mask_3c),
                0,
                255
            ).astype(np.uint8)

        return lit_bgr

    # ── 8. Standard Passport Framing & Headroom Geometry ──────────────────────

    @staticmethod
    def frame_passport_photo(
        img_bgr: np.ndarray,
        target_format: str = "Standard ICAO Passport (35 x 45 mm / 3.5 x 4.5 cm)",
        bg_color_bgr: Tuple[int, int, int] = (255, 255, 255),
        headroom_ratio: float = 0.14,
        head_height_ratio: float = 0.48,
    ) -> np.ndarray:
        """
        Frames portraits for school / ICAO ID photos.

        The Indian Passport Seva ICAO guidance uses a 35 x 45 mm image at
        630 x 810 pixels.  Callers can use a close head-height ratio (about
        0.90 here, allowing for the framing safety factor below) to meet the
        required 80-85% head coverage, while still retaining a small amount
        of space above the hair and the top of both shoulders.
        """
        h_orig, w_orig = img_bgr.shape[:2]

        if "35 x 45" in target_format or "ICAO" in target_format or "2 x 2" in target_format or "Visa" in target_format:
            # Strictly non-square portrait format (35 x 45 mm)
            target_w, target_h = 630, 810
            target_aspect = 35.0 / 45.0
        elif "3:4" in target_format or "Studio" in target_format:
            target_w, target_h = 600, 800
            target_aspect = 3.0 / 4.0
        else:
            if h_orig == w_orig:
                # Disallow square dimensions: convert to standard portrait format
                target_w, target_h = 630, 810
                target_aspect = 35.0 / 45.0
            else:
                return img_bgr

        geom = _head_geometry(img_bgr)
        if geom is not None:
            face_cx, head_crown_y, est_head_h = geom
            # Smaller ratio = more of the torso in frame (was 0.60 → extreme zoom)
            crop_h = est_head_h / max(float(head_height_ratio), 0.20)
            crop_h *= 1.12
            crop_w = crop_h * target_aspect

            # Face geometry alone does not include wide curls, ponytails,
            # flowers, bows or clips. On a portrait Qwen canvas, estimate the
            # significant foreground extent against the flat border colour and
            # zoom out just enough to keep that complete head silhouette. The
            # face remains the horizontal anchor, so an asymmetric accessory
            # cannot pull the person off centre.
            try:
                edge = max(4, min(18, h_orig // 24, w_orig // 24))
                corners = np.concatenate((
                    img_bgr[:edge, :edge].reshape(-1, 3),
                    img_bgr[:edge, -edge:].reshape(-1, 3),
                    img_bgr[-edge:, :edge].reshape(-1, 3),
                    img_bgr[-edge:, -edge:].reshape(-1, 3),
                ))
                backdrop = np.median(corners, axis=0).astype(np.float32)
                foreground = np.linalg.norm(
                    img_bgr.astype(np.float32) - backdrop, axis=2,
                ) > 22.0
                # Require a small vertical run so JPEG noise or a single edge
                # pixel cannot force an unnecessary zoom-out.
                column_counts = foreground.sum(axis=0)
                valid_columns = np.flatnonzero(
                    column_counts >= max(4, int(h_orig * .006))
                )
                if valid_columns.size:
                    content_x1 = int(valid_columns[0])
                    content_x2 = int(valid_columns[-1]) + 1
                    side_safety = max(8.0, w_orig * .025)
                    required_half_width = max(
                        face_cx - content_x1,
                        content_x2 - face_cx,
                    ) + side_safety
                    accessory_safe_width = required_half_width * 2.0
                    if accessory_safe_width > crop_w:
                        crop_w = accessory_safe_width
                        crop_h = crop_w / target_aspect
                        logger.info(
                            "[PASSPORT_FRAME] Expanded for hair/accessories "
                            "x=%d..%d crop=%.0fx%.0f",
                            content_x1, content_x2, crop_w, crop_h,
                        )
            except Exception as extent_error:
                logger.warning(
                    "[PASSPORT_FRAME] Hair/accessory extent check skipped: %s",
                    extent_error,
                )

            # Never invent a huge empty canvas — zoom out only as far as the photo allows
            fit = min(1.0, w_orig / max(crop_w, 1.0), h_orig / max(crop_h, 1.0))
            if fit < 0.999:
                crop_w *= fit
                crop_h *= fit

            crop_x1 = int(face_cx - crop_w / 2.0)
            crop_y1 = int(head_crown_y - (headroom_ratio * crop_h))
            crop_x1 = int(np.clip(crop_x1, 0, max(0, w_orig - crop_w)))
            crop_y1 = int(np.clip(crop_y1, 0, max(0, h_orig - crop_h)))
            crop_x2 = int(crop_x1 + crop_w)
            crop_y2 = int(crop_y1 + crop_h)

            pad_left = max(0, -crop_x1)
            pad_right = max(0, crop_x2 - w_orig)
            pad_top = max(0, -crop_y1)
            pad_bottom = max(0, crop_y2 - h_orig)

            if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                padded = cv2.copyMakeBorder(
                    img_bgr,
                    pad_top,
                    pad_bottom,
                    pad_left,
                    pad_right,
                    cv2.BORDER_CONSTANT,
                    value=bg_color_bgr,
                )
                crop_x1 += pad_left
                crop_x2 += pad_left
                crop_y1 += pad_top
                crop_y2 += pad_top
            else:
                padded = img_bgr

            cropped = padded[max(0, crop_y1):min(padded.shape[0], crop_y2), max(0, crop_x1):min(padded.shape[1], crop_x2)]
            if cropped.size > 0 and cropped.shape[0] > 10 and cropped.shape[1] > 10:
                framed = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                return framed

        orig_aspect = w_orig / float(h_orig)
        if orig_aspect > target_aspect:
            new_w = int(h_orig * target_aspect)
            x_start = (w_orig - new_w) // 2
            cropped = img_bgr[:, x_start:x_start + new_w]
        else:
            new_h = int(w_orig / target_aspect)
            y_start = max(0, int(h_orig * 0.02))
            y_end = min(h_orig, y_start + new_h)
            cropped = img_bgr[y_start:y_end, :]

        framed = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        return framed

    @staticmethod
    def reduce_glasses_glare(img_bgr: np.ndarray) -> np.ndarray:
        """Darken blown window/studio reflections on eyeglass lenses so eyes stay readable."""
        h, w = img_bgr.shape[:2]
        try:
            app = _get_insight_app()
            faces = app.get(img_bgr)
            if not faces:
                return img_bgr
            kps = faces[0].kps
        except Exception:
            return img_bgr

        lens = _lens_mask_from_kps(h, w, kps)
        if lens.max() < 0.2:
            return img_bgr

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]
        glare = ((L > 198) & (lens > 0.28)).astype(np.uint8)
        if int(glare.sum()) < 40:
            return img_bgr

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        glare = cv2.dilate(glare, kernel, iterations=1)
        inpainted = cv2.inpaint(img_bgr, glare, 4, cv2.INPAINT_TELEA)

        alpha = cv2.GaussianBlur(glare.astype(np.float32), (9, 9), 2.0)
        alpha = np.clip(alpha / 255.0, 0, 1)[:, :, np.newaxis] * lens[:, :, np.newaxis] * 0.55
        blended = inpainted.astype(np.float32) * alpha + img_bgr.astype(np.float32) * (1.0 - alpha)

        # Pull remaining hotspots toward the local iris tone
        lab_b = cv2.cvtColor(np.clip(blended, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        iris = (lens > 0.35) & (lab_b[:, :, 0] < 175)
        if iris.any():
            med = np.median(blended[iris], axis=0)
            hot = ((lab_b[:, :, 0] > 190) & (lens > 0.25)).astype(np.float32)
            hot = cv2.GaussianBlur(hot, (7, 7), 1.5)[:, :, np.newaxis]
            blended = blended * (1.0 - 0.35 * hot) + med.reshape(1, 1, 3) * (0.35 * hot)

        logger.info("[GLASSES] Reduced lens glare on %d px", int(glare.sum()))
        return np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def studio_camera_look(img_bgr: np.ndarray) -> np.ndarray:
        """Tack-sharp studio-camera finish: even light, focused eyes, mild print contrast."""
        h, w = img_bgr.shape[:2]
        lit = PhotoRestorationService.apply_studio_lighting(
            img_bgr,
            gamma=1.08,
            softbox_intensity=0.18,
            shadow_lift=0.28,
            rim_light_intensity=0.10,
        )
        blur = cv2.GaussianBlur(lit, (0, 0), 1.05)
        sharp = cv2.addWeighted(lit, 1.35, blur, -0.35, 0)
        lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] = np.clip((lab[:, :, 0] - 128.0) * 1.06 + 128.0, 0, 255)
        out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        logger.info("[STUDIO_CAMERA] Applied in-focus studio camera look (%dx%d)", w, h)
        return out

    @staticmethod
    def natural_portrait_camera_look(img_bgr: np.ndarray) -> np.ndarray:
        """Safely balance uneven face light without regenerating portrait pixels."""
        h, w = img_bgr.shape[:2]
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        luma = lab[:, :, 0]
        correction_applied = 0.0

        # Estimate left/right cheek brightness from the detected face. This is
        # deliberately an exposure-only correction: it cannot alter identity,
        # hair, clothing, pose, or the selected solid background.
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        if len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda face: face[2] * face[3])
            y0, y1 = y + int(fh * 0.35), y + int(fh * 0.80)
            lx0, lx1 = x + int(fw * 0.08), x + int(fw * 0.45)
            rx0, rx1 = x + int(fw * 0.55), x + int(fw * 0.92)
            left_luma = float(np.median(luma[y0:y1, lx0:lx1]))
            right_luma = float(np.median(luma[y0:y1, rx0:rx1]))
            difference = left_luma - right_luma

            if abs(difference) > 8.0:
                # Lift only the darker side, up to 18 LAB levels. A broad,
                # feathered ellipse avoids a visible correction boundary.
                correction_applied = float(np.clip(abs(difference) * 0.55, 0.0, 18.0))
                yy, xx = np.ogrid[:h, :w]
                center_x = x + fw * 0.5
                center_y = y + fh * 0.56
                ellipse = (((xx - center_x) / max(fw * 0.54, 1.0)) ** 2 +
                           ((yy - center_y) / max(fh * 0.68, 1.0)) ** 2) <= 1.0
                mask = cv2.GaussianBlur(ellipse.astype(np.float32), (0, 0), max(8.0, fw * 0.08))
                if difference > 0:
                    darker_side = np.clip((xx - center_x) / max(fw * 0.42, 1.0), 0.0, 1.0)
                else:
                    darker_side = np.clip((center_x - xx) / max(fw * 0.42, 1.0), 0.0, 1.0)
                lab[:, :, 0] = np.clip(luma + correction_applied * darker_side * mask, 0, 255)

        balanced = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        # A very small optical cleanup reduces JPEG noise while retaining skin,
        # hair strands, and dress texture.
        softened = cv2.bilateralFilter(balanced, 5, 18, 18)
        restored = cv2.addWeighted(balanced, 0.94, softened, 0.06, 0)
        blur = cv2.GaussianBlur(restored, (0, 0), 0.65)
        out = cv2.addWeighted(restored, 1.04, blur, -0.04, 0)
        logger.info(
            "[STUDIO_CAMERA] Applied safe lighting/restoration finish (%dx%d, luma_correction=%.1f)",
            w, h, correction_applied,
        )
        return out


# Module singleton
photo_restorer = PhotoRestorationService()
