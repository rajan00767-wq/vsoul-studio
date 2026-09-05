"""
Studio Uniform Pipeline
-----------------------
A dedicated 4-stage engine for studio-grade uniform swaps:
1. Pose & Roll Normalization (InsightFace 3D Roll Angle correction).
2. Tailored Garment Inpainting (SDXL Inpainting with exact garment styling).
3. 100% Biometric Identity & Hair Anchoring (PhotoRestorationService).
4. BiRefNet Studio Backdrop Matting.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

from pipelines.birefnet_service import background_removal
from pipelines.photo_restoration import PhotoRestorationService
from utils.logger import get_logger

logger = get_logger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT_DIR / "outputs"
MODELS_DIR = ROOT_DIR / "models"
INPAINT_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"


def fix_neck_shadow_artifact(composite_cv2_img: np.ndarray, neck_mask_cv2: np.ndarray) -> np.ndarray:
    """
    Samples the natural skin color from the lower face/jawline
    and automatically paints it over the gray artifact gap.
    """
    # 1. Convert mask to binary if it isn't already
    _, binary_mask = cv2.threshold(neck_mask_cv2, 127, 255, cv2.THRESH_BINARY)

    # 2. Find the topmost coordinates of your neck mask (where the gap is)
    y_indices, x_indices = np.where(binary_mask == 255)
    if len(y_indices) == 0:
        return composite_cv2_img  # Return original if no mask found

    top_y = np.min(y_indices)

    # 3. Sample a clean patch of skin color 15 pixels ABOVE the mask (her actual jaw skin)
    sample_y = max(0, top_y - 15)
    sample_x_start = int(np.mean(x_indices) - 20)
    sample_x_end = int(np.mean(x_indices) + 20)

    skin_sample_patch = composite_cv2_img[sample_y:max(sample_y + 1, top_y - 5), sample_x_start:sample_x_end]
    if skin_sample_patch.size == 0:
        return composite_cv2_img
    mean_skin_color = cv2.mean(skin_sample_patch)[:3]  # Grab BGR values

    # 4. Dilate the targeted fix area slightly to fully overlap the gray artifact
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    target_area = cv2.dilate(binary_mask, kernel, iterations=1)

    # Only target the top 20 pixels of the neck area where the artifact sits
    artifact_zone = np.zeros_like(target_area)
    artifact_zone[top_y:top_y + 20, :] = target_area[top_y:top_y + 20, :]

    # 5. Flood the artifact zone with her natural sampled skin color
    fixed_base = composite_cv2_img.copy()
    fixed_base[artifact_zone == 255] = mean_skin_color

    # 6. Smoothly blend the edges so there are no harsh lines
    blur_radius = 15
    blurred_img = cv2.GaussianBlur(fixed_base, (blur_radius, blur_radius), 0)

    # Combine original and blurred using the artifact mask as a weight map
    mask_normalized = artifact_zone.astype(float) / 255.0
    mask_normalized = cv2.GaussianBlur(mask_normalized, (5, 5), 0)  # Soften mask edges

    for c in range(3):  # Apply blending channel by channel
        composite_cv2_img[:, :, c] = (
            mask_normalized * blurred_img[:, :, c] +
            (1.0 - mask_normalized) * composite_cv2_img[:, :, c]
        ).astype(np.uint8)

    return composite_cv2_img


def _detect_face_metrics(rgb: np.ndarray, app) -> dict:
    """Chin, jaw width, and face box from InsightFace (106-pt jaw when available)."""
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    faces = app.get(bgr) if app is not None else []
    if not faces:
        return {
            "chin_x": w * 0.5,
            "chin_y": h * 0.55,
            "jaw_w": w * 0.28,
            "face_h": h * 0.32,
            "face_cx": w * 0.5,
            "face_cy": h * 0.38,
        }
    face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    x1, y1, x2, y2 = [float(v) for v in face.bbox]
    face_h = max(1.0, y2 - y1)
    face_cx = (x1 + x2) * 0.5
    face_cy = (y1 + y2) * 0.5
    chin_x, chin_y = face_cx, y2 + face_h * 0.04
    jaw_w = (x2 - x1) * 0.92
    lm = getattr(face, "landmark_2d_106", None)
    if lm is not None and len(lm) >= 33:
        jaw = np.asarray(lm[:33], dtype=np.float32)
        chin_idx = int(np.argmax(jaw[:, 1]))
        chin_x, chin_y = float(jaw[chin_idx, 0]), float(jaw[chin_idx, 1])
        jaw_w = float(jaw[:, 0].max() - jaw[:, 0].min())
    elif getattr(face, "kps", None) is not None and len(face.kps) >= 5:
        mouth_y = float((face.kps[3][1] + face.kps[4][1]) * 0.5)
        chin_y = mouth_y + face_h * 0.28
        chin_x = float((face.kps[3][0] + face.kps[4][0]) * 0.5)
    return {
        "chin_x": chin_x,
        "chin_y": chin_y,
        "jaw_w": max(8.0, jaw_w),
        "face_h": face_h,
        "face_cx": face_cx,
        "face_cy": face_cy,
    }


# LIP-20: clothing vs background. Dress/cloth is never treated as background.
# 0=BG, 1=Hat, 2=Hair, 5=UpperClothes, 6=Dress, 7=Coat, 10=Jumpsuit,
# 11=Scarf/Tie, 12=Skirt, 13=Face
CLOTH_LABELS = (5, 6, 7, 10, 11, 12)
HAT_LABELS = (1,)
HEAD_LABELS = (2, 4, 13)
TEMPLATE_IDENTITY = (2, 13)


def _checker_pixels(rgb: np.ndarray) -> np.ndarray:
    """True on e-commerce transparency checkerboard (low-chroma light gray/white)."""
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    chroma = np.maximum(np.abs(r - g), np.maximum(np.abs(g - b), np.abs(r - b)))
    mean = rgb.mean(axis=2)
    return (chroma < 12) & (mean > 158)


def _dilate_u8(mask: np.ndarray, k: int, iters: int = 1) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel, iterations=iters)


def _border_checker_mask(rgb: np.ndarray) -> np.ndarray:
    """Erase checkerboard that touches the frame. Keep interior white dress fabric."""
    h, w = rgb.shape[:2]
    chk = _checker_pixels(rgb).astype(np.uint8)
    if int(chk.sum()) < 50:
        return np.ones((h, w), dtype=np.uint8) * 255
    _n, labels = cv2.connectedComponents(chk, connectivity=8)
    border = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    border.discard(0)
    keep = np.ones((h, w), dtype=np.uint8) * 255
    for i in border:
        keep[labels == i] = 0
    return keep


def _fill_tiny_holes_keep_collar(mask_u8: np.ndarray) -> np.ndarray:
    """Close fabric pinholes. Leave the top collar / neck opening open."""
    h, w = mask_u8.shape[:2]
    bin_m = (mask_u8 > 40).astype(np.uint8)
    contours, _ = cv2.findContours(bin_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask_u8
    filled = np.zeros_like(bin_m)
    cv2.drawContours(filled, contours, -1, 1, thickness=-1)
    holes = ((filled == 1) & (bin_m == 0)).astype(np.uint8)
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(holes, 8)
    out = mask_u8.copy()
    collar_band = int(h * 0.20)
    max_hole = max(40, int(h * w * 0.008))
    for i in range(1, nlab):
        if int(stats[i, cv2.CC_STAT_TOP]) < collar_band:
            continue
        if int(stats[i, cv2.CC_STAT_AREA]) <= max_hole:
            out[labels == i] = 255
    return out


def _extract_clothing_mask(
    uniform_rgb: np.ndarray,
    u_labels: Optional[np.ndarray],
    biref_alpha: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Keep the full dress. Drop only background and the template model's head."""
    h, w = uniform_rgb.shape[:2]
    keep_bg = _border_checker_mask(uniform_rgb)
    cloth = np.zeros((h, w), dtype=np.uint8)
    if u_labels is not None:
        cloth = np.isin(u_labels, CLOTH_LABELS).astype(np.uint8) * 255
        cloth[np.isin(u_labels, TEMPLATE_IDENTITY)] = 0
    if biref_alpha is not None:
        ba = biref_alpha if biref_alpha.ndim == 2 else biref_alpha[:, :, 3]
        if ba.shape[:2] != (h, w):
            ba = cv2.resize(ba, (w, h), interpolation=cv2.INTER_LINEAR)
        if ba.max() <= 1.0:
            ba = (np.clip(ba, 0, 1) * 255).astype(np.uint8)
        elif ba.dtype != np.uint8:
            ba = np.clip(ba, 0, 255).astype(np.uint8)
        cloth = np.maximum(cloth, ba)
        if u_labels is not None:
            cloth[np.isin(u_labels, TEMPLATE_IDENTITY)] = 0
    cloth = np.minimum(cloth, keep_bg)
    if u_labels is not None:
        cloth[np.isin(u_labels, CLOTH_LABELS)] = 255
        cloth[np.isin(u_labels, TEMPLATE_IDENTITY)] = 0
        cloth = np.minimum(cloth, keep_bg)
    cloth = _dilate_u8(cloth, 3, 1)
    return _fill_tiny_holes_keep_collar(cloth)


def _collar_opening(alpha: np.ndarray) -> Tuple[int, int, int]:
    """Neck hole at the top of the dress collar (not the vest V)."""
    h, w = alpha.shape[:2]
    binary = (alpha > 40).astype(np.uint8)
    top_h = max(8, int(h * 0.22))
    min_gap = max(6, int(w * 0.06))
    max_gap = max(min_gap + 4, int(w * 0.28))
    cx = w * 0.5
    best = None
    best_score = -1e9
    for y in range(top_h):
        xs = np.where(binary[y] > 0)[0]
        if len(xs) < 10:
            continue
        span = np.zeros(w, dtype=np.uint8)
        span[int(xs[0]) : int(xs[-1]) + 1] = 1
        hx = np.where((span == 1) & (binary[y] == 0))[0]
        if len(hx) < 6:
            continue
        splits = np.where(np.diff(hx) > 1)[0]
        starts = np.concatenate([[0], splits + 1])
        ends = np.concatenate([splits, [len(hx) - 1]])
        for i in range(len(starts)):
            gl, gr = int(hx[starts[i]]), int(hx[ends[i]])
            gw = gr - gl
            if gw < min_gap or gw > max_gap:
                continue
            mid = (gl + gr) * 0.5
            score = (1.0 - y / max(1, top_h)) * 3.0 - abs(mid - cx) / max(1.0, w * 0.5)
            if score > best_score:
                best_score = score
                best = (y, gl, gr)
    if best is not None:
        return best
    ys = np.where(binary[:top_h].max(axis=1) > 0)[0]
    y0 = int(ys.min()) if len(ys) else 0
    return y0, int(w * 0.44), int(w * 0.56)


def _front_collar_wings(alpha: np.ndarray, inner_y: int, gap_l: int, gap_r: int) -> np.ndarray:
    """Collar points that wrap in front of the neck."""
    h, w = alpha.shape[:2]
    cx = (gap_l + gap_r) * 0.5
    neck_half = float(np.clip((gap_r - gap_l) * 0.5, w * 0.035, w * 0.10))
    y_idx = np.arange(h)[:, None]
    x_idx = np.arange(w)[None, :]
    top = max(0, inner_y - int(h * 0.04))
    bot = min(h, inner_y + int(h * 0.16))
    lateral = (x_idx < cx - neck_half * 0.25) | (x_idx > cx + neck_half * 0.25)
    near = np.abs(x_idx - cx) < w * 0.20
    wings = (alpha > 40) & (y_idx >= top) & (y_idx < bot) & lateral & near
    return cv2.GaussianBlur(np.where(wings, 255, 0).astype(np.uint8), (5, 5), 1.2)


def _erase_original_outfit(
    p_arr: np.ndarray,
    labels: Optional[np.ndarray],
    metrics: dict,
) -> np.ndarray:
    """Remove the original dress/shirt (including red collars) but keep face, hair, and skin."""
    rgb = p_arr[:, :, :3].copy()
    alpha = p_arr[:, :, 3].copy()
    h, w = alpha.shape[:2]
    clothes = np.zeros((h, w), dtype=np.uint8)
    if labels is not None:
        clothes[np.isin(labels, (5, 6, 7, 10, 11, 12))] = 255
        clothes[np.isin(labels, HEAD_LABELS)] = 0
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    # Catch leftover polo/collar colours SCHP may miss
    reddish = (r > 110) & ((r - g) > 28) & ((r - b) > 20)
    clothes[reddish] = 255
    lock_y = int(np.clip(metrics["chin_y"] - metrics["face_h"] * 0.18, 0, h - 1))
    clothes[:lock_y] = 0
    if labels is not None:
        clothes[labels == 2] = 0
        clothes[labels == 13] = 0
    clothes = _dilate_u8(clothes, 7, 1)
    clothes[alpha < 20] = 0
    if clothes.sum() > 50:
        rgb = cv2.inpaint(rgb, clothes, 5, cv2.INPAINT_TELEA)
    # Explicit neck cylinder in sampled jaw skin so the collar has a throat to wrap
    jaw_y = int(np.clip(metrics["chin_y"] - metrics["face_h"] * 0.08, 0, h - 1))
    jx0 = int(np.clip(metrics["chin_x"] - metrics["jaw_w"] * 0.22, 0, w - 1))
    jx1 = int(np.clip(metrics["chin_x"] + metrics["jaw_w"] * 0.22, 0, w - 1))
    patch = rgb[max(0, jaw_y - 6) : jaw_y + 2, jx0:jx1]
    if patch.size:
        skin = np.median(patch.reshape(-1, 3), axis=0).astype(np.uint8)
        neck = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            neck,
            (int(metrics["chin_x"]), int(metrics["chin_y"] + metrics["face_h"] * 0.12)),
            (max(6, int(metrics["jaw_w"] * 0.28)), max(8, int(metrics["face_h"] * 0.22))),
            0, 0, 360, 255, -1,
        )
        neck[: int(np.clip(metrics["chin_y"] - 2, 0, h))] = 0
        neck[alpha < 20] = 0
        wgt = (cv2.GaussianBlur(neck, (11, 11), 3.0).astype(np.float32) / 255.0)[:, :, None]
        rgb = (skin.astype(np.float32) * wgt + rgb.astype(np.float32) * (1.0 - wgt)).astype(np.uint8)
    return np.dstack([rgb, alpha])


def _match_garment_light(garment_rgb: np.ndarray, ref_rgb: np.ndarray, ref_mask: np.ndarray) -> np.ndarray:
    """Match garment brightness to the face so the stitch is less obvious."""
    if ref_mask.sum() < 50:
        return garment_rgb
    g = cv2.cvtColor(garment_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    r = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    gm, gs = float(g[:, :, 0].mean()), float(g[:, :, 0].std()) + 1e-5
    rm, rs = float(r[:, :, 0][ref_mask > 40].mean()), float(r[:, :, 0][ref_mask > 40].std()) + 1e-5
    g[:, :, 0] = np.clip((g[:, :, 0] - gm) / gs * (rs * 0.85) + rm * 0.92 + gm * 0.08, 0, 255)
    return cv2.cvtColor(g.astype(np.uint8), cv2.COLOR_LAB2RGB)


def _crop_rgba(rgb: np.ndarray, mask_u8: np.ndarray, pad: int = 6) -> Optional[np.ndarray]:
    ys, xs = np.where(mask_u8 > 40)
    if len(ys) == 0:
        return None
    h, w = mask_u8.shape[:2]
    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad + 1)
    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad + 1)
    return np.dstack([rgb[y0:y1, x0:x1], mask_u8[y0:y1, x0:x1]])
    ys, xs = np.where(mask_u8 > 40)
    if len(ys) == 0:
        return None
    h, w = mask_u8.shape[:2]
    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad + 1)
    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad + 1)
    return np.dstack([rgb[y0:y1, x0:x1], mask_u8[y0:y1, x0:x1]])


def _paste_rgba(canvas: np.ndarray, layer: np.ndarray, x: int, y: int) -> None:
    """Alpha-blend an RGBA layer onto an RGB canvas in place."""
    lh, lw = layer.shape[:2]
    ch, cw = canvas.shape[:2]
    x0, y0 = x, y
    x1, y1 = x + lw, y + lh
    sx0 = 0 if x0 >= 0 else -x0
    sy0 = 0 if y0 >= 0 else -y0
    dx0 = max(0, x0)
    dy0 = max(0, y0)
    dx1 = min(cw, x1)
    dy1 = min(ch, y1)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    sx1 = sx0 + (dx1 - dx0)
    sy1 = sy0 + (dy1 - dy0)
    src = layer[sy0:sy1, sx0:sx1].astype(np.float32)
    a = (src[:, :, 3:4] / 255.0)
    dst = canvas[dy0:dy1, dx0:dx1].astype(np.float32)
    canvas[dy0:dy1, dx0:dx1] = np.clip(src[:, :, :3] * a + dst * (1.0 - a), 0, 255).astype(np.uint8)


def _head_protect_mask(p_alpha: np.ndarray, labels: Optional[np.ndarray], metrics: dict) -> np.ndarray:
    """Soft mask of face + hair + upper neck that the uniform must not cover."""
    h, w = p_alpha.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    if labels is not None:
        mask[np.isin(labels, HEAD_LABELS)] = 1.0
    chin_x, chin_y = metrics["chin_x"], metrics["chin_y"]
    jaw_w, face_h = metrics["jaw_w"], metrics["face_h"]
    face_cx, face_cy = metrics["face_cx"], metrics["face_cy"]
    cv2.ellipse(
        mask,
        (int(face_cx), int(face_cy)),
        (max(8, int(jaw_w * 0.72)), max(8, int(face_h * 0.85))),
        0, 0, 360, 1.0, -1,
    )
    # Keep a short throat so the head is not a sticker, but fade into the collar
    neck_top = int(np.clip(chin_y - face_h * 0.02, 0, h - 1))
    neck_bot = int(np.clip(chin_y + face_h * 0.36, 0, h - 1))
    x0 = int(np.clip(chin_x - jaw_w * 0.42, 0, w - 1))
    x1 = int(np.clip(chin_x + jaw_w * 0.42, 0, w - 1))
    if neck_bot > neck_top and x1 > x0:
        col = np.linspace(1.0, 0.0, neck_bot - neck_top, dtype=np.float32)[:, None]
        existing = mask[neck_top:neck_bot, x0:x1]
        person = (p_alpha[neck_top:neck_bot, x0:x1].astype(np.float32) / 255.0) * col
        mask[neck_top:neck_bot, x0:x1] = np.maximum(existing, person)
    # Everything above the mouth stays locked
    lock_y = int(np.clip(chin_y - face_h * 0.12, 0, h - 1))
    head_band = p_alpha[:lock_y] > 24
    mask[:lock_y][head_band] = 1.0
    mask_u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    mask_u8 = cv2.GaussianBlur(mask_u8, (21, 21), 6.0)
    return mask_u8


def _shoulder_width(p_alpha: np.ndarray, labels: Optional[np.ndarray], chin_y: float) -> float:
    h, w = p_alpha.shape[:2]
    y0 = int(np.clip(chin_y + 6, 0, h - 2))
    y1 = int(np.clip(chin_y + max(24, (h - chin_y) * 0.32), y0 + 8, h))
    band = p_alpha[y0:y1] > 40
    if labels is not None:
        band = band | np.isin(labels[y0:y1], (5, 6, 7, 14, 15))
    xs = np.where(band.any(axis=0))[0]
    if len(xs) < 8:
        xs = np.where(p_alpha > 40)[1]
    if len(xs) < 4:
        return w * 0.58
    return float(xs.max() - xs.min())


def _frame_studio_portrait(
    rgb: np.ndarray,
    bg_color: Tuple[int, int, int],
    target_size: Tuple[int, int] = (768, 1024),
) -> Image.Image:
    tw, th = target_size
    bg = np.array(bg_color, dtype=np.int16)
    diff = np.abs(rgb.astype(np.int16) - bg).max(axis=2)
    ys, xs = np.where(diff > 16)
    if len(ys) == 0:
        img = Image.fromarray(rgb)
        return img.resize(target_size, Image.LANCZOS)
    y0, y1 = max(0, ys.min() - 8), min(rgb.shape[0], ys.max() + 16)
    x0, x1 = max(0, xs.min() - 8), min(rgb.shape[1], xs.max() + 8)
    cropped = Image.fromarray(rgb[y0:y1, x0:x1])
    cw, ch = cropped.size
    scale = min(tw * 0.92 / max(cw, 1), th * 0.90 / max(ch, 1))
    nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
    resized = cropped.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (tw, th), bg_color)
    canvas.paste(resized, ((tw - nw) // 2, max(0, int(th * 0.06))))
    return canvas


class StudioUniformPipeline:
    """Production studio uniform synthesizer with pose correction and identity lock."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StudioUniformPipeline, cls).__new__(cls)
            cls._instance._pipe = None
            cls._app = None
        return cls._instance

    def _get_insightface(self):
        if self._app is None:
            try:
                from insightface.app import FaceAnalysis
                app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                app.prepare(ctx_id=-1, det_size=(320, 320))
                self._app = app
            except Exception as e:
                logger.warning("[StudioUniform] InsightFace init warning: %s", e)
        return self._app

    def _get_inpaint_pipe(self):
        if self._pipe is None:
            from diffusers import AutoPipelineForInpainting
            model_path = MODELS_DIR / "sdxl" / "inpaint"
            src = str(model_path) if (model_path / "model_index.json").exists() else INPAINT_ID
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            logger.info("[StudioUniform] Loading SDXL Inpainting from local path: %s on %s...", src, device)
            pipe = AutoPipelineForInpainting.from_pretrained(
                src,
                torch_dtype=dtype,
                variant="fp16",
                use_safetensors=True,
                local_files_only=(model_path / "model_index.json").exists(),
            )
            if device == "cuda":
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cpu")
            self._pipe = pipe
        return self._pipe

    def finish_uniform_seams(
        self,
        person_image: Image.Image,
        fitted_uniform: Image.Image,
        tryon_mask: Image.Image,
        steps: int = 14,
    ) -> Image.Image:
        """Use SDXL only on a narrow garment seam ring, never on identity pixels."""
        person = np.array(person_image.convert("RGB"))
        fitted = np.array(fitted_uniform.convert("RGB"))
        fallback = fitted_uniform.convert("RGB").copy()
        h, w = fitted.shape[:2]
        if person.shape[:2] != (h, w):
            person = cv2.resize(person, (w, h), interpolation=cv2.INTER_LANCZOS4)

        delta = cv2.cvtColor(cv2.absdiff(fitted, person), cv2.COLOR_RGB2GRAY)
        garment = ((delta > 22).astype(np.uint8) * 255)
        torso = np.array(tryon_mask.convert("L").resize((w, h), Image.NEAREST))
        garment = cv2.bitwise_and(garment, torso)
        garment = cv2.morphologyEx(
            garment, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        )
        seam = cv2.subtract(
            cv2.dilate(garment, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))),
            cv2.erode(garment, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))),
        )

        app = self._get_insightface()
        try:
            faces = app.get(cv2.cvtColor(person, cv2.COLOR_RGB2BGR)) if app is not None else []
            if faces:
                face = max(faces, key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]))
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                cv2.ellipse(
                    seam, ((x1 + x2) // 2, (y1 + y2) // 2),
                    (max(8, int((x2 - x1) * 0.9)), max(8, int((y2 - y1) * 1.12))),
                    0, 0, 360, 0, -1,
                )
                seam[: min(h, int(y2 + 0.08 * (y2 - y1))), :] = 0
        except Exception as e:
            logger.warning("[StudioUniform] Face exclusion for seam finish skipped: %s", e)
            seam[: int(h * 0.42), :] = 0

        seam = cv2.dilate(seam, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        if cv2.countNonZero(seam) < 80:
            return fitted_uniform

        try:
            pipe = self._get_inpaint_pipe()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(42)
            with torch.inference_mode():
                result = pipe(
                    prompt=(
                        "photorealistic studio portrait seam repair only, natural collar-to-neck and shoulder-to-uniform transition, "
                        "preserve the exact navy vest, light blue checked shirt, fabric pattern, buttons, and lighting"
                    ),
                    negative_prompt=(
                        "changed face, altered identity, glasses, changed uniform color, changed pattern, changed buttons, "
                        "extra collar, extra shoulder, blurry, warped fabric"
                    ),
                    image=fitted_uniform.convert("RGB"),
                    mask_image=Image.fromarray(seam, mode="L"),
                    strength=0.38,
                    num_inference_steps=max(10, min(steps, 16)),
                    guidance_scale=4.0,
                    generator=generator,
                ).images[0].convert("RGB")
            generated = np.array(result)
            if generated.shape[:2] != (h, w):
                generated = cv2.resize(generated, (w, h), interpolation=cv2.INTER_LANCZOS4)
            alpha = cv2.GaussianBlur(seam, (5, 5), 0).astype(np.float32)[:, :, None] / 255.0
            # Copy only the masked seam output; all identity and garment detail
            # outside the ring remains exactly from the fitted image.
            finished = (generated.astype(np.float32) * alpha + fitted.astype(np.float32) * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
            logger.info("[StudioUniform] Applied SDXL natural seam finish to %d pixels", cv2.countNonZero(seam))
            return Image.fromarray(finished, mode="RGB")
        except Exception as e:
            logger.warning("[StudioUniform] SDXL seam finish skipped: %s", e)
            return fitted_uniform

    def finish_uniform_seams_qwen(
        self,
        person_image: Image.Image,
        fitted_uniform: Image.Image,
        tryon_mask: Image.Image,
    ) -> Image.Image:
        """Repair only the garment boundary with one bounded Qwen edit pass.

        Qwen Image Edit has no native pixel-mask input.  The generated result is
        therefore never used as a full image: Qwen sees a tight crop around the
        detected clothing seam and only the original seam-ring pixels are blended
        back.  Face, hair, neck, and the body of the uploaded garment stay intact.
        """
        person = np.array(person_image.convert("RGB"))
        fitted = np.array(fitted_uniform.convert("RGB"))
        fallback = fitted_uniform.convert("RGB").copy()
        h, w = fitted.shape[:2]
        if person.shape[:2] != (h, w):
            person = cv2.resize(person, (w, h), interpolation=cv2.INTER_LANCZOS4)

        delta = cv2.cvtColor(cv2.absdiff(fitted, person), cv2.COLOR_RGB2GRAY)
        garment = ((delta > 22).astype(np.uint8) * 255)
        torso = np.array(tryon_mask.convert("L").resize((w, h), Image.NEAREST))
        garment = cv2.bitwise_and(garment, torso)
        garment = cv2.morphologyEx(
            garment,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        )
        seam = cv2.subtract(
            cv2.dilate(garment, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))),
            cv2.erode(garment, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))),
        )

        app = self._get_insightface()
        try:
            faces = app.get(cv2.cvtColor(person, cv2.COLOR_RGB2BGR)) if app is not None else []
            if faces:
                face = max(faces, key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]))
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                cv2.ellipse(
                    seam,
                    ((x1 + x2) // 2, (y1 + y2) // 2),
                    (max(8, int((x2 - x1) * 0.9)), max(8, int((y2 - y1) * 1.12))),
                    0,
                    0,
                    360,
                    0,
                    -1,
                )
                seam[: min(h, int(y2 + 0.08 * (y2 - y1))), :] = 0
            else:
                seam[: int(h * 0.42), :] = 0
        except Exception as e:
            logger.warning("[StudioUniform] Face exclusion for Qwen seam finish skipped: %s", e)
            seam[: int(h * 0.42), :] = 0

        seam = cv2.dilate(seam, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        if cv2.countNonZero(seam) < 80:
            return fitted_uniform

        points = cv2.findNonZero(seam)
        if points is None:
            return fitted_uniform
        x, y, crop_w, crop_h = cv2.boundingRect(points)
        # Context gives Qwen enough fabric information without allowing a full
        # portrait rewrite.  The final alpha still limits writes to the seam.
        padding = max(24, min(72, int(max(crop_w, crop_h) * 0.08)))
        left, top = max(0, x - padding), max(0, y - padding)
        right, bottom = min(w, x + crop_w + padding), min(h, y + crop_h + padding)
        crop = Image.fromarray(fitted[top:bottom, left:right], mode="RGB")
        seam_crop = seam[top:bottom, left:right]

        try:
            from pipelines.qwen_edit_pipeline import qwen_service

            output_path = qwen_service.qwen_edit_enhancer(
                image_input=crop,
                prompt=(
                    "Repair only the neck-to-shirt collar seam. Replace the detached circular or raised collar "
                    "with one natural, flat collar that begins directly below the neck and follows its contour. "
                    "Preserve the exact navy vest, pale blue checked shirt, buttons, check pattern, fabric color, "
                    "fit, lighting, person, hair, neck, and background."
                ),
                negative_prompt=(
                    "floating collar, circular collar, double collar, exposed gap below neck, changed identity, glasses, "
                    "altered neck, extra clothing, changed uniform color, changed check pattern, missing buttons, warped fabric"
                ),
                background_color="keep",
                job_id=f"uniform_seam_{int(time.time())}",
                width=crop.width,
                height=crop.height,
                steps=4,
                max_sequence_length=128,
                timeout_seconds=None,
            )
            with Image.open(str(output_path)) as generated_file:
                generated = np.array(generated_file.convert("RGB"))
            expected_size = (right - left, bottom - top)
            if generated.shape[:2] != (expected_size[1], expected_size[0]):
                generated = cv2.resize(generated, expected_size, interpolation=cv2.INTER_LANCZOS4)
            if generated.ndim != 3 or generated.shape[2] != 3 or not np.isfinite(generated).all():
                raise ValueError("Qwen seam result is not a valid RGB image")
            # Do not accept blank or partly decoded images.  Nothing has been
            # written to the portrait at this point, so this remains atomic.
            if float(generated.std()) < 4.0:
                raise ValueError("Qwen seam result has insufficient visual detail")

            alpha = cv2.GaussianBlur(seam_crop, (5, 5), 0).astype(np.float32)[:, :, None] / 255.0
            finished = fitted.copy()
            region = finished[top:bottom, left:right]
            finished[top:bottom, left:right] = (
                generated.astype(np.float32) * alpha + region.astype(np.float32) * (1.0 - alpha)
            ).clip(0, 255).astype(np.uint8)
            logger.info(
                "[StudioUniform] Applied bounded Qwen seam finish to %d pixels (%dx%d crop)",
                cv2.countNonZero(seam),
                expected_size[0],
                expected_size[1],
            )
            return Image.fromarray(finished, mode="RGB")
        except Exception as e:
            logger.warning("[StudioUniform] Qwen seam finish skipped: %s", e)
            return fallback

    def _reference_neural_tryon(
        self,
        person_image: Image.Image,
        uniform_reference: Image.Image,
        uniform_preset: str,
        bg_color: Tuple[int, int, int],
        steps: int,
        guidance_scale: float,
        job_id: str,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> Path:
        """True try-on: CatVTON if local weights exist, else SDXL inpaint on the torso only."""
        from pipelines.photo_restoration import PhotoRestorationService

        tw, th = 768, 1024
        person = person_image.convert("RGB").resize((tw, th), Image.LANCZOS)

        # 1) CatVTON diffusion drape
        try:
            from pipelines.catvton_pipeline import CatVTONPipeline
            cat = CatVTONPipeline()
            if cat.is_available():
                if progress_cb:
                    progress_cb(0.25, "CatVTON: draping fabric onto the body (face locked)...")
                raw_path = cat.tryon(
                    person_image=person,
                    uniform_image=uniform_reference,
                    steps=max(12, min(steps, 25)),
                    guidance_scale=2.5,
                    apply_compositing=False,
                    bg_color=bg_color,
                    job_id=f"{job_id}_catvton",
                )
                tryon_pil = Image.open(str(raw_path)).convert("RGB").resize((tw, th), Image.LANCZOS)
                logger.info("[StudioUniform] CatVTON try-on succeeded")
            else:
                raise FileNotFoundError("CatVTON weights not complete")
        except Exception as e:
            logger.warning("[StudioUniform] CatVTON skipped (%s) — SDXL torso inpaint", e)
            if progress_cb:
                progress_cb(0.28, "SDXL: inpainting uniform onto the torso only...")
            tryon_pil = self._sdxl_torso_inpaint(
                person, uniform_reference, uniform_preset, steps, guidance_scale, progress_cb
            )

        if progress_cb:
            progress_cb(0.82, "Locking original face, neck and hair...")
        fused = PhotoRestorationService.anchor_and_fuse_identity(
            orig_pil=person,
            qwen_pil=tryon_pil,
            identity_lock_strength=0.96,
        )

        if progress_cb:
            progress_cb(0.93, "Studio background...")
        try:
            with io.BytesIO() as bio:
                fused.save(bio, format="PNG")
                fg_bytes = background_removal.remove_background(bio.getvalue(), job_id=job_id)
            fg_rgba = Image.open(io.BytesIO(fg_bytes)).convert("RGBA")
            studio = Image.new("RGBA", fg_rgba.size, (*bg_color, 255))
            studio.paste(fg_rgba, (0, 0), fg_rgba)
            final = studio.convert("RGB")
        except Exception as e:
            logger.warning("[StudioUniform] Studio matte fallback: %s", e)
            final = fused

        out_path = OUTPUTS_DIR / f"{job_id}_official_passport_master.png"
        final.resize((tw, th), Image.LANCZOS).save(out_path, format="PNG", dpi=(300, 300))
        if progress_cb:
            progress_cb(1.0, "✅ Neural try-on complete (torso inpainted, face locked)")
        return out_path

    def _sdxl_torso_inpaint(
        self,
        person: Image.Image,
        uniform_reference: Image.Image,
        uniform_preset: str,
        steps: int,
        guidance_scale: float,
        progress_cb: Optional[Callable[[float, str], None]],
    ) -> Image.Image:
        """Inpaint only clothing below the chin; never touch the face."""
        tw, th = person.size
        p_arr = np.array(person.convert("RGB"))
        u_arr = np.array(uniform_reference.convert("RGB"))
        labels = u_labels = None
        try:
            from pipelines.schp_service import parse as schp_parse
            labels = schp_parse(p_arr)["labels"]
            u_labels = schp_parse(u_arr)["labels"]
        except Exception as e:
            logger.warning("[StudioUniform] SCHP for try-on mask skipped: %s", e)

        metrics = _detect_face_metrics(p_arr, self._get_insightface())
        mask = np.zeros((th, tw), dtype=np.uint8)
        if labels is not None:
            mask[np.isin(labels, (5, 6, 7, 10, 11, 12, 14, 15))] = 255
            mask[np.isin(labels, HEAD_LABELS)] = 0
        chin = int(np.clip(metrics["chin_y"] + metrics["face_h"] * 0.10, 0, th - 1))
        mask[:chin] = 0
        cv2.ellipse(
            mask,
            (int(metrics["face_cx"]), int(metrics["face_cy"])),
            (max(8, int(metrics["jaw_w"] * 0.75)), max(8, int(metrics["face_h"] * 0.95))),
            0, 0, 360, 0, -1,
        )
        if mask.sum() < 800:
            mask[chin:, :] = 255
            cv2.ellipse(
                mask,
                (int(metrics["face_cx"]), int(metrics["face_cy"])),
                (max(8, int(metrics["jaw_w"] * 0.75)), max(8, int(metrics["face_h"] * 0.95))),
                0, 0, 360, 0, -1,
            )
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), iterations=1)
        mask_soft = cv2.GaussianBlur(mask, (21, 21), 6.0)

        hint = p_arr.copy()
        cloth = _extract_clothing_mask(u_arr, u_labels, None)
        garment = _crop_rgba(u_arr, cloth, pad=4)
        ys, xs = np.where(mask_soft > 80)
        if garment is not None and len(xs) > 20:
            gw, gh = garment.shape[1], garment.shape[0]
            twid = int(xs.max() - xs.min())
            sc = float(np.clip(twid / max(gw, 1), 0.2, 2.5))
            nw, nh = max(1, int(gw * sc)), max(1, int(gh * sc))
            gs = cv2.resize(garment, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
            gx = int((xs.min() + xs.max()) / 2 - nw / 2)
            gy = chin - int(nh * 0.03)
            sx0 = 0 if gx >= 0 else -gx
            sy0 = 0 if gy >= 0 else -gy
            dx0, dy0 = max(0, gx), max(0, gy)
            dx1, dy1 = min(tw, gx + nw), min(th, gy + nh)
            if dx1 > dx0 and dy1 > dy0:
                sl = gs[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
                a = (sl[:, :, 3].astype(np.float32) / 255.0)
                a *= mask_soft[dy0:dy1, dx0:dx1].astype(np.float32) / 255.0
                a = np.clip(a * 0.72, 0, 1)[:, :, None]
                hint[dy0:dy1, dx0:dx1] = (
                    sl[:, :, :3].astype(np.float32) * a + hint[dy0:dy1, dx0:dx1].astype(np.float32) * (1.0 - a)
                ).astype(np.uint8)

        pipe = self._get_inpaint_pipe()
        prompt = (
            f"studio passport photo of the same child wearing a {uniform_preset}, "
            "collar wrapping the neck, realistic fabric, matching lighting, photorealistic"
        )
        negative = (
            "changed face, different identity, extra person, checkerboard, "
            "deformed collar, floating clothes, red polo, casual t-shirt, blurry"
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(42)
        with torch.inference_mode():
            res = pipe(
                prompt=prompt,
                negative_prompt=negative,
                image=Image.fromarray(hint),
                mask_image=Image.fromarray(mask_soft, mode="L"),
                strength=0.70,
                num_inference_steps=max(12, min(steps, 28)),
                guidance_scale=guidance_scale,
                generator=generator,
            )
        return res.images[0].convert("RGB")

    def execute_swap(
        self,
        person_image: Image.Image,
        uniform_reference: Optional[Image.Image] = None,
        uniform_preset: str = "navy blue formal school uniform with tie and white collar",
        bg_color: Tuple[int, int, int] = (205, 230, 248),
        steps: int = 25,
        guidance_scale: float = 7.5,
        job_id: Optional[str] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> Path:
        """Executes the full 4-stage studio uniform transformation."""
        if job_id is None:
            job_id = f"studio_uniform_{int(time.time())}"

        # ── Branch A: Neural try-on (CatVTON / SDXL torso inpaint, face locked) ──
        if uniform_reference is not None:
            return self._reference_neural_tryon(
                person_image=person_image,
                uniform_reference=uniform_reference,
                uniform_preset=uniform_preset,
                bg_color=bg_color,
                steps=steps,
                guidance_scale=guidance_scale,
                job_id=job_id,
                progress_cb=progress_cb,
            )

        # ── Branch B: Neural Diffusion Preset Inpainting ──────────────────────
        # ── Stage 1: Pose & Roll Angle Correction ─────────────────────────────
        if progress_cb:
            progress_cb(0.12, "📐 Analyzing 3D facial posture & correcting head tilt...")

        person_bgr = np.array(person_image.convert("RGB"))[:, :, ::-1]
        h, w = person_bgr.shape[:2]
        app = self._get_insightface()
        aligned_bgr = person_bgr.copy()
        collar_cut = int(h * 0.45)

        if app is not None:
            try:
                faces = app.get(person_bgr)
                if len(faces) > 0:
                    pf = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                    eye_l, eye_r = pf.kps[0], pf.kps[1]
                    dx = eye_r[0] - eye_l[0]
                    dy = eye_r[1] - eye_l[1]
                    roll_deg = float(np.degrees(np.arctan2(dy, dx)))
                    logger.info("[StudioUniform] Detected head tilt: %.2f degrees", roll_deg)

                    face_center = ((pf.bbox[0] + pf.bbox[2]) / 2.0, (pf.bbox[1] + pf.bbox[3]) / 2.0)
                    rot_mat = cv2.getRotationMatrix2D(face_center, roll_deg * 0.80, 1.0)
                    aligned_bgr = cv2.warpAffine(
                        person_bgr, rot_mat, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT
                    )

                    faces_aligned = app.get(aligned_bgr)
                    if len(faces_aligned) > 0:
                        pf_a = max(faces_aligned, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                        fh = int(pf_a.bbox[3] - pf_a.bbox[1])
                        mouth_y = int((pf_a.kps[3][1] + pf_a.kps[4][1]) * 0.5)
                        chin_y = mouth_y + int(fh * 0.22)
                        collar_cut = chin_y + int(fh * 0.05)
            except Exception as e:
                logger.warning("[StudioUniform] Pose alignment fallback: %s", e)

        aligned_pil = Image.fromarray(aligned_bgr[:, :, ::-1])

        # ── Stage 2: Create Inpainting Torso Mask ─────────────────────────────
        if progress_cb:
            progress_cb(0.30, "⚡ Creating anatomical torso inpainting mask...")

        target_w, target_h = 768, 1024
        aligned_res = aligned_pil.resize((target_w, target_h), Image.LANCZOS)
        scale_y = target_h / float(h)
        collar_cut_s = int(collar_cut * scale_y)

        inpaint_mask = np.zeros((target_h, target_w), dtype=np.uint8)
        inpaint_mask[collar_cut_s:, :] = 255
        inpaint_mask_soft = cv2.GaussianBlur(inpaint_mask, (21, 21), 6.0)
        mask_pil = Image.fromarray(inpaint_mask_soft, mode="L")

        # ── Stage 3: Neural Uniform Inpainting ────────────────────────────────
        if progress_cb:
            progress_cb(0.45, "🎨 Synthesizing tailored uniform with studio softbox lighting...")

        pipe = self._get_inpaint_pipe()

        prompt = (
            f"commercial studio passport portrait of the child wearing a tailored {uniform_preset}, "
            "buttoned crisp white collar shirt, sharp necktie, flattering studio softbox lighting, centered camera focus, "
            "realistic fabric weave, photorealistic 8k detail"
        )
        negative_prompt = (
            "casual clothes, open chest, denim, deformed collar, distorted face, blurry, bad anatomy"
        )

        # Pre-process neck boundary with sampled skin flood
        aligned_cv2 = cv2.cvtColor(np.array(aligned_res), cv2.COLOR_RGB2BGR)
        clean_composite = fix_neck_shadow_artifact(aligned_cv2, inpaint_mask)
        padded_img_pil = Image.fromarray(cv2.cvtColor(clean_composite, cv2.COLOR_BGR2RGB))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(42)
        with torch.inference_mode():
            res = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=padded_img_pil,
                mask_image=mask_pil,
                strength=0.98,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        inpainted_img = res.images[0].convert("RGB")

        # ── Stage 4: 100% Authentic Biometric Identity & Hair Clip Lock ──────
        if progress_cb:
            progress_cb(0.85, "✨ Anchoring authentic biometric face, smile, eyes & hair clips...")

        fused_img = PhotoRestorationService.anchor_and_fuse_identity(
            orig_pil=aligned_pil,
            qwen_pil=inpainted_img,
            identity_lock_strength=0.95,
        )

        # ── Stage 5: Clean Studio Backdrop Matting with BiRefNet ───────────────
        if progress_cb:
            progress_cb(0.95, "Compositing onto clean studio backdrop...")

        with io.BytesIO() as bio:
            fused_img.save(bio, format="PNG")
            fg_bytes = background_removal.remove_background(bio.getvalue(), job_id=job_id)
        fg_rgba = Image.open(io.BytesIO(fg_bytes)).convert("RGBA")

        studio_bg = Image.new("RGBA", fg_rgba.size, (bg_color[0], bg_color[1], bg_color[2], 255))
        studio_bg.paste(fg_rgba, (0, 0), fg_rgba)
        final_portrait = studio_bg.convert("RGB")

        out_path = OUTPUTS_DIR / f"{job_id}_studio_portrait.png"
        final_portrait.save(out_path, format="PNG", dpi=(300, 300))

        if progress_cb:
            progress_cb(1.0, "✅ Studio Uniform Swap completed successfully!")

        return out_path


studio_uniform_pipeline = StudioUniformPipeline()
