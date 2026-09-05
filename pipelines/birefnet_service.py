"""
BackgroundRemovalService — portrait-quality background matting.

Uses the strongest available portrait matting BiRefNet checkpoint first, then
falls back safely through lighter models if a checkpoint cannot be loaded.

Post-processing pipeline:
  1. INTER_LINEAR resize to original resolution.
  2. Edge-aware refinement on the probability matte.
  3. No binary thresholding or erosion on the production alpha.
  4. Local-background foreground decontamination to remove halos/spill.

Debug: saves raw_mask.png, alpha_matte.png, foreground_rgba.png per job.
"""

import cv2
import io
import os
from pathlib import Path
import threading

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from utils.logger import get_logger as _get_logger
from pipelines.schp_service import available as schp_available, fuse_with_birefnet, parse

logger = _get_logger(__name__)

_IMAGE_SIZE = (1024, 1024)

_TRANSFORM = transforms.Compose([
    transforms.Resize(_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Prefer matting models over lite segmentation for hair, accessories, and
# semi-transparent portrait edges. Keep lite as the last-resort fallback.
_MODEL_CHAIN = [
    "ZhengPeng7/BiRefNet_HR-matting",
    "ZhengPeng7/BiRefNet_dynamic-matting",
    "ZhengPeng7/BiRefNet-matting",
    "ZhengPeng7/BiRefNet-portrait",
    "ZhengPeng7/BiRefNet_lite",
]


def _guided_filter_alpha(
    img_rgb: np.ndarray,
    mask:    np.ndarray,
    radius:  int   = 10,
    eps:     float = 1e-4,
) -> np.ndarray:
    """
    Guided image filter for alpha matting (He et al. 2013).

    r=10 (21x21 window): wide enough to bridge BiRefNet's 1024→original
    resolution boundary offset while staying edge-adaptive (eps=1e-4).
    """
    guide = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    p     = mask.astype(np.float64)
    k     = 2 * radius + 1

    mean_I  = cv2.blur(guide,           (k, k))
    mean_p  = cv2.blur(p,               (k, k))
    mean_Ip = cv2.blur(guide * p,       (k, k))
    mean_II = cv2.blur(guide * guide,   (k, k))

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I  = mean_II - mean_I * mean_I
    a      = cov_Ip / (var_I + eps)
    b      = mean_p - a * mean_I

    mean_a = cv2.blur(a, (k, k))
    mean_b = cv2.blur(b, (k, k))
    return np.clip(mean_a * guide + mean_b, 0.0, 1.0).astype(np.float32)


def _refine_mask(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Refine the raw probability matte without converting it to a hard mask.

    The raw BiRefNet probability map already contains wispy hair and accessory
    edges. Earlier cleanup compressed that soft range too aggressively, which
    caused strand/flower clipping and jagged halos. Here the raw matte remains
    the source of truth; guided/bilateral smoothing is blended in only around
    uncertain edge pixels.
    """
    mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
    guided = _guided_filter_alpha(img_rgb, mask, radius=18, eps=5e-5)
    bilateral = cv2.bilateralFilter(mask, 9, 0.08, 9)

    uncertain = np.clip(1.0 - np.abs(mask - 0.5) * 2.0, 0.0, 1.0)
    refined = (
        mask * (1.0 - 0.45 * uncertain)
        + guided * (0.30 * uncertain)
        + bilateral * (0.15 * uncertain)
    )
    return np.clip(refined, 0.0, 1.0).astype(np.float32)


def _clean_alpha(alpha_u8: np.ndarray) -> np.ndarray:
    """Keep the full 0-255 soft matte; remove only isolated salt/pepper noise."""
    median = cv2.medianBlur(alpha_u8, 3)
    noise = ((alpha_u8 < 8) & (median > 48)) | ((alpha_u8 > 247) & (median < 207))
    cleaned = alpha_u8.copy()
    cleaned[noise] = median[noise]
    return cleaned


def _local_background(img_rgb: np.ndarray, alpha_u8: np.ndarray) -> np.ndarray:
    """Estimate nearby background color behind each foreground/edge pixel."""
    unknown = (alpha_u8 > 18).astype(np.uint8) * 255
    local_bg = cv2.inpaint(img_rgb.copy(), unknown, 5, cv2.INPAINT_TELEA).astype(np.float32)
    return cv2.GaussianBlur(local_bg, (0, 0), 1.2)


def _largest_foreground_band(alpha_u8: np.ndarray) -> np.ndarray:
    """
    Keep edge alpha only near confident foreground components.

    This removes floating background haze from leaves/walls while avoiding
    erosion. The band is intentionally generous so hair curls and accessories
    connected to the subject remain inside it.
    """
    h, w = alpha_u8.shape
    core = (alpha_u8 >= 150).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(core, 8)
    if n <= 1:
        return np.ones_like(alpha_u8, dtype=bool)

    img_area = h * w
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    keep = labels == largest
    min_area = max(12, int(img_area * 0.00008))
    for idx in range(1, n):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area >= min_area:
            x = stats[idx, cv2.CC_STAT_LEFT]
            y = stats[idx, cv2.CC_STAT_TOP]
            cw = stats[idx, cv2.CC_STAT_WIDTH]
            ch = stats[idx, cv2.CC_STAT_HEIGHT]
            # Keep small confident accessories near the main subject box.
            lx = stats[largest, cv2.CC_STAT_LEFT]
            ly = stats[largest, cv2.CC_STAT_TOP]
            lw = stats[largest, cv2.CC_STAT_WIDTH]
            lh = stats[largest, cv2.CC_STAT_HEIGHT]
            near_x = x + cw >= lx - 80 and x <= lx + lw + 80
            near_y = y + ch >= ly - 80 and y <= ly + lh + 80
            if near_x and near_y:
                keep |= labels == idx

    radius = max(18, min(h, w) // 18)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius | 1, radius | 1))
    return cv2.dilate(keep.astype(np.uint8), k, iterations=1).astype(bool)


def _trim_background_alpha(
    img_rgb: np.ndarray,
    alpha_u8: np.ndarray,
    local_bg: np.ndarray,
) -> np.ndarray:
    """
    Reduce alpha where edge pixels look like the original background.

    The model's soft mask is preserved for true hair/flowers because those
    pixels are chromatically different from the local background. Green leaves,
    white wall haze, and old background bleed are close to local_bg, so they are
    faded before compositing.
    """
    alpha = alpha_u8.astype(np.float32) / 255.0
    rgb = img_rgb.astype(np.float32)

    dist = np.linalg.norm(rgb - local_bg, axis=2)
    bg_like = np.clip(1.0 - dist / 72.0, 0.0, 1.0)
    semi = np.clip((0.82 - alpha) / 0.82, 0.0, 1.0)

    trimmed = alpha.copy()
    trim_strength = bg_like * semi
    trimmed *= (1.0 - 0.88 * trim_strength)

    green_bg = (local_bg[:, :, 1] > local_bg[:, :, 0] + 8) & (local_bg[:, :, 1] > local_bg[:, :, 2] + 8)
    green_px = (rgb[:, :, 1] > rgb[:, :, 0] + 8) & (rgb[:, :, 1] > rgb[:, :, 2] + 8)
    green_spill = green_bg & green_px & (alpha < 0.90)
    trimmed[green_spill] *= 0.18 + 0.70 * alpha[green_spill]

    bright_bg = (local_bg.mean(axis=2) > 205) & (rgb.mean(axis=2) > 205) & (alpha < 0.72)
    trimmed[bright_bg] *= 0.28 + 0.72 * alpha[bright_bg]

    band = _largest_foreground_band(alpha_u8)
    far_haze = (~band) & (alpha < 0.78)
    trimmed[far_haze] *= 0.08

    return np.clip(trimmed * 255.0, 0, 255).astype(np.uint8)


def _remove_schp_background_intrusions(
    img_rgb: np.ndarray,
    alpha_u8: np.ndarray,
    schp_labels: np.ndarray | None,
) -> np.ndarray:
    """Safely cleans background noise while strictly preserving all human skin, face, hair, and clothing."""
    cleaned = alpha_u8.copy()

    # If SCHP is available, strictly protect all core human body parts (Face, Neck, Clothes, Dress, Limbs)
    if schp_labels is not None:
        rejected_apparel = np.zeros(cleaned.shape, dtype=bool)
        # A leaf touching the frame can occasionally be classified as upper
        # clothing.  Remove only detached, green, frame-touching apparel
        # components before applying the core-human alpha protection.  This is
        # deliberately narrower than color-keying: a connected green uniform,
        # badge, face, hair, and all non-border clothing remain untouched.
        apparel = np.isin(schp_labels, [5, 6, 7, 10, 11, 12]).astype(np.uint8)
        count, components, stats, _ = cv2.connectedComponentsWithStats(apparel, 8)
        if count > 2:
            main_component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            rgb = img_rgb.astype(np.int16)
            green = (
                (rgb[:, :, 1] > rgb[:, :, 0] + 12)
                & (rgb[:, :, 1] > rgb[:, :, 2] + 12)
            )
            height, width = cleaned.shape
            frame_margin = max(4, min(height, width) // 64)
            for component_id in range(1, count):
                if component_id == main_component:
                    continue
                x, y, w, h, area = stats[component_id]
                touches_frame = (
                    x <= frame_margin or y <= frame_margin
                    or x + w >= width - frame_margin or y + h >= height - frame_margin
                )
                component = components == component_id
                is_small = area < max(128, int(cleaned.size * 0.04))
                if touches_frame and is_small and np.mean(green[component]) > 0.25:
                    rejected_apparel |= component

        # LIP-20 core labels, including hair (2), protect the subject before
        # suppressing disconnected outdoor-background fragments.
        core_human = np.isin(schp_labels, [1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15])
        cleaned[core_human] = np.maximum(cleaned[core_human], 250)
        # Include the soft BiRefNet boundary around the rejected component,
        # where the parser may emit generic background rather than apparel.
        rejected_apparel = cv2.dilate(
            rejected_apparel.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
        ).astype(bool)
        cleaned[rejected_apparel] = 0

    # Clear detached background pieces that are too small to be a person but
    # large enough to remain visible after a studio-blue composite.
    fg = (cleaned > 24).astype(np.uint8)
    fg_count, fg_components, fg_stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if fg_count > 1:
        main_id = 1 + int(np.argmax(fg_stats[1:, cv2.CC_STAT_AREA]))
        near_main = cv2.dilate(
            (fg_components == main_id).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        ).astype(bool)
        for idx in range(1, fg_count):
            if idx == main_id:
                continue
            component = fg_components == idx
            component_area = int(fg_stats[idx, cv2.CC_STAT_AREA])
            if component_area < max(96, int(cleaned.size * 0.002)) and not np.any(component & near_main):
                cleaned[component] = 0
    return cleaned
def _estimate_background_color(img_rgb: np.ndarray, alpha_u8: np.ndarray) -> np.ndarray:
    """Robust color estimate from definite background around image borders."""
    border = np.zeros(alpha_u8.shape, dtype=bool)
    h, w = alpha_u8.shape
    pad = max(3, min(h, w) // 32)
    border[:pad, :] = True
    border[-pad:, :] = True
    border[:, :pad] = True
    border[:, -pad:] = True
    samples = img_rgb[(alpha_u8 < 24) & border]
    if samples.size == 0:
        samples = img_rgb[alpha_u8 < 12]
    if samples.size == 0:
        return np.array([255.0, 255.0, 255.0], dtype=np.float32)
    return np.median(samples.reshape(-1, 3), axis=0).astype(np.float32)


def _decontaminate_foreground(img_rgb: np.ndarray, alpha_u8: np.ndarray) -> np.ndarray:
    """
    Pull edge pixels away from the estimated source background color.

    This targets white/black/color halos without shrinking the alpha matte, so
    hair strands, flowers, earrings, and soft accessories remain present.
    """
    rgb = img_rgb.astype(np.float32)
    alpha = alpha_u8.astype(np.float32) / 255.0

    # Estimate local background colors by inpainting foreground pixels from
    # nearby definite-background pixels. This handles green gardens, white
    # walls, and mixed school-photo backgrounds better than one global color.
    local_bg = _local_background(img_rgb, alpha_u8)

    edge = (alpha > 0.01) & (alpha < 0.98)
    uncertain = np.clip(1.0 - np.abs(alpha - 0.5) * 2.0, 0.0, 1.0)
    strength = np.clip((1.0 - alpha) * (0.35 + 0.55 * uncertain), 0.0, 0.88)

    corrected_full = (rgb - local_bg * strength[:, :, None]) / np.maximum(
        1.0 - strength[:, :, None],
        0.28,
    )
    corrected = rgb.copy()
    corrected[edge] = corrected_full[edge]


def _estimate_background_color(img_rgb: np.ndarray, alpha_u8: np.ndarray) -> np.ndarray:
    """Robust color estimate from definite background around image borders."""
    border = np.zeros(alpha_u8.shape, dtype=bool)
    h, w = alpha_u8.shape
    pad = max(3, min(h, w) // 32)
    border[:pad, :] = True
    border[-pad:, :] = True
    border[:, :pad] = True
    border[:, -pad:] = True
    samples = img_rgb[(alpha_u8 < 24) & border]
    if samples.size == 0:
        samples = img_rgb[alpha_u8 < 12]
    if samples.size == 0:
        return np.array([255.0, 255.0, 255.0], dtype=np.float32)
    return np.median(samples.reshape(-1, 3), axis=0).astype(np.float32)
def _decontaminate_foreground(img_rgb: np.ndarray, alpha_u8: np.ndarray) -> np.ndarray:
    """
    Pull edge pixels away from the estimated source background color.

    This targets white/black/color halos without shrinking the alpha matte, so
    hair strands, flowers, earrings, and soft accessories remain present.
    """
    rgb = img_rgb.astype(np.float32)
    alpha = alpha_u8.astype(np.float32) / 255.0

    local_bg = _local_background(img_rgb, alpha_u8)

    edge = (alpha > 0.01) & (alpha < 0.98)
    uncertain = np.clip(1.0 - np.abs(alpha - 0.5) * 2.0, 0.0, 1.0)
    strength = np.clip((1.0 - alpha) * (0.35 + 0.55 * uncertain), 0.0, 0.88)

    corrected_full = (rgb - local_bg * strength[:, :, None]) / np.maximum(
        1.0 - strength[:, :, None],
        0.28,
    )
    corrected = rgb.copy()
    corrected[edge] = corrected_full[edge]

    # Extra green-spill suppression on semi-transparent edges only. This does
    # not touch opaque clothing/accessories and avoids desaturating real colors.
    green_excess = corrected[:, :, 1] - np.maximum(corrected[:, :, 0], corrected[:, :, 2])
    green_spill = edge & (green_excess > 8) & (local_bg[:, :, 1] > local_bg[:, :, 0] + 8)
    corrected[green_spill, 1] -= np.minimum(green_excess[green_spill] * 0.90, 64)

    # White/bright halos: borrow chroma from nearby corrected pixels while
    # preserving luminance gently, so hair strands remain visible.
    luma = corrected.mean(axis=2)
    white_halo = edge & (luma > 210) & (alpha < 0.86)
    if white_halo.any():
        local_fg = cv2.GaussianBlur(corrected.astype(np.uint8), (0, 0), 1.6).astype(np.float32)
        corrected[white_halo] = 0.25 * corrected[white_halo] + 0.75 * local_fg[white_halo]

    return np.clip(corrected, 0, 255).astype(np.uint8)


class BackgroundRemovalService:
    """Thread-safe singleton — loads the best available BiRefNet once."""

    def __init__(self):
        self._model    = None
        self._device   = None
        self._model_id = None
        self._lock     = threading.Lock()

    def _load_model(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return

            from transformers import AutoModelForImageSegmentation

            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

            for model_id in _MODEL_CHAIN:
                try:
                    logger.info("Loading %s from local cache…", model_id)
                    try:
                        model = AutoModelForImageSegmentation.from_pretrained(
                            model_id,
                            trust_remote_code=True,
                            local_files_only=True,
                        )
                    except Exception:
                        logger.info("Local cache check failed, querying remote %s…", model_id)
                        model = AutoModelForImageSegmentation.from_pretrained(
                            model_id,
                            trust_remote_code=True,
                        )
                    model = model.float().eval().to(device)
                    self._model    = model
                    self._device   = device
                    self._model_id = model_id
                    logger.info("Loaded %s on %s (local offline cache)", model_id, device)
                    return
                except Exception as e:
                    logger.warning("Failed to load %s: %s — trying next", model_id, e)

            raise RuntimeError(f"All models in chain failed to load: {_MODEL_CHAIN}")

    def remove_background(
        self,
        image_bytes: bytes,
        job_id: str = None,
        debug: bool = False,
        source_image: Image.Image | None = None,
        use_schp: bool = True,
    ) -> bytes:
        """
        Remove background. Returns RGBA PNG.

        Alpha = true soft matte from BiRefNet + guided filter refinement.
        RGB   = decontaminated foreground pixels to prevent edge halos.

        Saves job-specific raw_mask, alpha_matte, and foreground_rgba files.
        final_output is saved by orchestrator after compositing.
        """
        self._load_model()

        img     = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_arr = np.array(img)
        orig_w, orig_h = img.size

        _ddir = Path("outputs")
        _ddir.mkdir(exist_ok=True)
        if debug:
            img.save(str(_ddir / "debug_original.png"))

        # ── Inference ────────────────────────────────────────────────────────
        inp = _TRANSFORM(img).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            preds = self._model(inp)
        raw    = preds[-1] if isinstance(preds, (list, tuple)) else preds
        mask_f = raw.sigmoid().cpu().squeeze().numpy().astype(np.float32)

        # ── Resize to original resolution ────────────────────────────────────
        mask_r = cv2.resize(mask_f, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        raw_mask_img = Image.fromarray(
            (mask_r * 255).clip(0, 255).astype(np.uint8), 'L'
        )
        if debug:
            raw_mask_img.save(str(_ddir / "debug_mask_raw.png"))
            Image.fromarray(
                ((mask_r > 0.5) * 255).astype(np.uint8), 'L'
            ).save(str(_ddir / "debug_mask_threshold.png"))

        # ── Guided filter + soft clamps (v3 approach) ────────────────────────
        alpha = _refine_mask(img_arr, mask_r)
        schp_labels = None
        source_labels = None
        source_arr = None
        if use_schp and schp_available():
            try:
                alpha, schp_labels = fuse_with_birefnet(img_arr, alpha)
                if source_image is not None:
                    source_arr = np.array(
                        source_image.convert("RGB").resize((orig_w, orig_h), Image.LANCZOS)
                    )
                    source_labels = parse(source_arr)["labels"]
                if debug:
                    Image.fromarray(schp_labels, "L").save(
                        str(_ddir / "debug_schp_labels.png")
                    )
                if job_id:
                    Image.fromarray(schp_labels, "L").save(
                        str(_ddir / f"{job_id}_schp_labels.png")
                    )
                logger.info("[MASK_FUSION] BiRefNet + SCHP LIP-20 job=%s", job_id or "?")
            except Exception as exc:
                logger.warning("SCHP fusion unavailable; retaining BiRefNet matte: %s", exc)

        alpha_u8 = (alpha * 255).clip(0, 255).astype(np.uint8)
        alpha_u8 = _clean_alpha(alpha_u8)
        local_bg = _local_background(img_arr, alpha_u8)
        alpha_u8 = _trim_background_alpha(img_arr, alpha_u8, local_bg)
        alpha_u8 = _remove_schp_background_intrusions(img_arr, alpha_u8, schp_labels)
        if source_labels is not None:
            # Qwen can repaint background foliage as a foreground-colored edge.
            # Remove that green only when the corresponding original pixel was
            # not apparel, so real green uniforms keep their source color.
            rgb = img_arr.astype(np.int16)
            generated_green = (
                (rgb[:, :, 1] > rgb[:, :, 0] + 4)
                & (rgb[:, :, 1] > rgb[:, :, 2] + 4)
            )
            source_apparel = np.isin(source_labels, [5, 6, 7, 10, 11, 12])
            alpha_u8[generated_green & ~source_apparel] = 0
            source_rgb = source_arr.astype(np.int16)
            source_green = (
                (source_rgb[:, :, 1] > source_rgb[:, :, 0] + 4)
                & (source_rgb[:, :, 1] > source_rgb[:, :, 2] + 4)
            )
            # Faint green contamination can be labelled as clothing by the
            # generated-image parser. Remove it only in the soft outside edge
            # when the matching source pixel was not green fabric.
            # When the source is not green at this point, any green repaint is
            # a background leak, even if the generated matte marked it opaque.
            alpha_u8[generated_green & ~source_green] = 0
        alpha_u8 = _clean_alpha(alpha_u8)

        alpha_img = Image.fromarray(alpha_u8, 'L')
        if debug:
            alpha_img.save(str(_ddir / "debug_mask_refined.png"))

        # ── Diagnostic log ───────────────────────────────────────────────────
        alpha_f = alpha_u8.astype(np.float32) / 255.0
        logger.info(
            "[BIREFNET] model=%s job=%s uncertain=%.1f%% boundary=%dpx",
            self._model_id.split("/")[-1], job_id or "?",
            float(((mask_r > 0.10) & (mask_r < 0.90)).mean() * 100),
            int(((alpha_f > 0.05) & (alpha_f < 0.95)).sum()),
        )

        # ── Compose RGBA with decontaminated foreground RGB ──────────────────
        fg_rgb = _decontaminate_foreground(img_arr, alpha_u8)
        rgba = Image.fromarray(fg_rgb, "RGB").convert("RGBA")
        rgba.putalpha(Image.fromarray(alpha_u8, 'L'))

        if debug:
            alpha_img.save(str(_ddir / "debug_alpha_matte.png"))
            rgba.save(str(_ddir / "debug_foreground_rgba.png"))

        if job_id:
            raw_mask_img.save(str(_ddir / "raw_mask.png"))
            raw_mask_img.save(str(_ddir / f"{job_id}_raw_mask.png"))
            alpha_img.save(str(_ddir / "alpha_matte.png"))
            alpha_img.save(str(_ddir / f"{job_id}_alpha_matte.png"))
            rgba.save(str(_ddir / "foreground_rgba.png"))
            rgba.save(str(_ddir / f"{job_id}_foreground_rgba.png"))

        out = io.BytesIO()
        rgba.save(out, format="PNG")
        return out.getvalue()

    def warm_up(self):
        """Pre-load model weights. Called at startup."""
        self._load_model()

    def unload(self):
        """Release the resident matting model before a full SDXL workflow."""
        with self._lock:
            model = self._model
            self._model = None
            self._device = None
            self._model_id = None
        if model is not None:
            del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Module-level singleton — imported by orchestrator.py
background_removal = BackgroundRemovalService()
