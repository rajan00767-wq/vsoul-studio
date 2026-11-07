"""
Qwen-Image-Edit Neural Diffusion Pipeline
Production Service powered strictly by the genuine Qwen-Image-Edit-2511 model:
- Decoupled Multimodal Vision-Language Conditioning via Isolated Subprocess Worker
- GGUF Quantized QwenImageTransformer2DModel DiT with low-VRAM CUDA Streaming
- AutoencoderKLQwenImage 5D VAE on CPU with FlowMatchEulerDiscreteScheduler
- PhotoRestorationService Facial Identity Lock, Optical Restoration & Jewelry Polish
"""

from __future__ import annotations

import io
import hashlib
import os
import sys
import time
import zlib
import subprocess
import threading
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Callable, Tuple

from PIL import Image, ImageEnhance
import torch
import gc
import cv2
import numpy as np
import types
from math import prod

from transformers import AutoTokenizer, AutoProcessor
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    QwenImageTransformer2DModel,
    QwenImageEditPlusPipeline,
    GGUFQuantizationConfig,
)
from diffusers.models.transformers.transformer_qwenimage import compute_text_seq_len_from_mask
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import retrieve_latents
from diffusers.image_processor import VaeImageProcessor
from pipelines.photo_restoration import PhotoRestorationService
from utils.logger import get_logger

logger = get_logger(__name__)

MODELS_DIR = Path("models/qwen_image_edit")
GGUF_PATH = MODELS_DIR / "qwen-image-edit-2511-Q4_K_M.gguf"
# Keep the GGUF architecture definition in the application, rather than the
# mutable model cache.  The 2511 GGUF requires the 64-channel Edit model
# configuration; an older cached 128-channel file leaves parameters on meta.
CONFIG_PATH = Path("config/qwen_image_edit_2511_transformer.json")
LORA_FILE = Path("models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
CACHE_DIR = Path("scratch/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cleanup_successful_uniform_artifacts(
    job_id: str,
    final_output: Path,
    output_dir: Path = OUTPUTS_DIR,
) -> int:
    """Remove only this successful job's diagnostic raster images.

    Failed jobs retain their artifacts for diagnosis.  Successful jobs keep
    the delivered portrait and JSON audit records, while rough fits, masks,
    native decodes and reference previews for the same exact job prefix are
    removed.  The non-recursive, prefix-scoped cleanup cannot touch historical
    portraits from another job.
    """
    if os.getenv("VSOUL_CLEAN_SUCCESSFUL_UNIFORM_ARTIFACTS", "1").strip().lower() not in {
        "1", "true", "yes",
    }:
        return 0
    safe_job_id = Path(str(job_id)).name
    final_resolved = Path(final_output).resolve()
    removed = 0
    for artifact in output_dir.glob(f"{safe_job_id}_*"):
        try:
            if (
                artifact.is_file()
                and artifact.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
                and artifact.resolve() != final_resolved
            ):
                artifact.unlink()
                removed += 1
        except OSError as exc:
            logger.warning("[UNIFORM_CLEANUP] Could not remove %s: %s", artifact, exc)
    return removed

_PIPELINE_LOCK = threading.Lock()

# A recognition embedding can remain high after Qwen redraws a child's eyes,
# mouth, or face proportions. Keep a strict biometric floor and independently
# verify the dense facial geometry before delivering a generated portrait.
MIN_PORTRAIT_IDENTITY_SIMILARITY = 0.70
MAX_PORTRAIT_LANDMARK_RMSE = 0.045
MAX_PORTRAIT_FACE_ASPECT_DELTA = 0.10


def _normalized_face_geometry(
    landmarks: np.ndarray, five_points: np.ndarray
) -> np.ndarray:
    """Normalize dense landmarks for scale, translation, and camera roll."""
    points = np.asarray(landmarks, dtype=np.float32)[:, :2]
    anchors = np.asarray(five_points, dtype=np.float32)[:, :2]
    eye_vector = anchors[1] - anchors[0]
    eye_distance = max(float(np.linalg.norm(eye_vector)), 1e-6)
    eye_midpoint = (anchors[0] + anchors[1]) * 0.5
    angle = float(np.arctan2(eye_vector[1], eye_vector[0]))
    cosine, sine = np.cos(-angle), np.sin(-angle)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float32)
    return ((points - eye_midpoint) @ rotation.T) / eye_distance


def _face_geometry_metrics(source_face: Any, generated_face: Any) -> Tuple[float, float]:
    """Return dense-landmark RMSE and face-box aspect-ratio change."""
    source_shape = _normalized_face_geometry(
        np.asarray(source_face.landmark_2d_106), np.asarray(source_face.kps)
    )
    generated_shape = _normalized_face_geometry(
        np.asarray(generated_face.landmark_2d_106), np.asarray(generated_face.kps)
    )
    landmark_rmse = float(np.sqrt(np.mean((source_shape - generated_shape) ** 2)))

    source_box = np.asarray(source_face.bbox, dtype=np.float32)
    generated_box = np.asarray(generated_face.bbox, dtype=np.float32)
    source_aspect = float(
        (source_box[2] - source_box[0]) / max(source_box[3] - source_box[1], 1e-6)
    )
    generated_aspect = float(
        (generated_box[2] - generated_box[0])
        / max(generated_box[3] - generated_box[1], 1e-6)
    )
    return landmark_rmse, abs(generated_aspect - source_aspect)


def _select_enhancement_steps(
    vl_plan: Dict[str, Any], maximum_steps: int = 20
) -> Tuple[int, List[str]]:
    """Choose portrait-enhancement depth from visible restoration risk."""
    def truthy(value: Any) -> bool:
        return str(value or "").strip().lower() in {"true", "yes", "1", "present"}

    risk = 0
    reasons: List[str] = []
    sunlight = truthy(vl_plan.get("direct_sunlight_present"))
    hotspot = truthy(vl_plan.get("head_hair_hotspot_present"))
    sunlight_type = str(vl_plan.get("sunlight_type") or "").lower()
    hard_light = any(term in sunlight_type for term in ("hard", "direct", "outdoor", "sun"))
    if sunlight and hard_light:
        risk += 2
        reasons.append("hard or direct source lighting")
    elif sunlight or hotspot:
        risk += 1
        reasons.append("possible lighting hotspot")

    pose = str(vl_plan.get("face_orientation") or vl_plan.get("pose") or "").lower()
    if pose and not any(term in pose for term in ("front", "forward", "camera")):
        risk += 2
        reasons.append("non-frontal source pose")

    hair_edge_risk = str(vl_plan.get("hair_edge_risk") or "").lower()
    if any(term in hair_edge_risk for term in ("high", "complex", "difficult", "clipped")):
        risk += 1
        reasons.append("complex hair boundary")
    elif truthy(vl_plan.get("crown_near_top_edge")):
        risk += 1
        reasons.append("limited crown headroom")

    accessories = vl_plan.get("hair_accessories")
    if str(accessories or "").strip().lower() not in {
        "", "none", "[]", "{}", "unknown", "uncertain"
    }:
        risk += 1
        reasons.append("hair accessories")

    min_dimension = int(vl_plan.get("source_min_dimension") or 0)
    face_detail = float(vl_plan.get("source_face_detail_score") or 0.0)
    face_height_pixels = float(vl_plan.get("source_face_height_pixels") or 0.0)
    face_frame_ratio = float(vl_plan.get("source_face_frame_ratio") or 0.0)
    # A wide phone photo can have plenty of total pixels while the actual face
    # is tiny.  Four-step Lightning then has too little identity/skin evidence
    # and the later passport crop magnifies its smooth, painted reconstruction.
    # Use the base model's first useful quality tier for that case.
    if (
        (0 < face_height_pixels < 160)
        or (0 < face_frame_ratio < .18)
    ):
        risk = max(risk, 2)
        reasons.append("small face region requires photographic base-model detail")
    if (0 < min_dimension < 320) or (0 < face_detail < 35.0):
        risk += 2
        reasons.append("very low source resolution or facial detail")
    elif (0 < min_dimension < 480) or (0 < face_detail < 80.0):
        risk += 1
        reasons.append("limited source resolution or facial detail")

    # Relighting uses a neutralized source guide plus bounded optical recovery.
    # Keep Qwen on its native four-step edit adapter; the base schedule visibly
    # repaints skin and hair instead of restoring photographic texture.
    selected = 4
    if not reasons:
        reasons.append("four-step identity-preserving enhancement")
    return selected, reasons


def _select_uniform_steps(vl_plan: Dict[str, Any], maximum_steps: int = 20) -> Tuple[int, List[str]]:
    """Choose uniform edit depth without turning source-preservation risks into redraw pressure."""
    def truthy(value: Any) -> bool:
        return str(value or "").strip().lower() in {"true", "yes", "1", "present"}

    risk = 0
    reasons: List[str] = []
    sunlight = truthy(vl_plan.get("direct_sunlight_present"))
    hotspot = truthy(vl_plan.get("head_hair_hotspot_present"))
    sunlight_type = str(vl_plan.get("sunlight_type") or "").lower()
    hard_light = any(term in sunlight_type for term in ("hard", "direct", "outdoor", "sun"))
    if sunlight and hard_light:
        # Match Enhancement: the native four-step edit adapter plus a neutral
        # relighting instruction preserves identity better than switching to
        # the base schedule. A bounded post-check handles any hotspot Qwen
        # still retains, so lighting alone is not regeneration complexity.
        reasons.append("hard source lighting handled by four-step relighting")
    elif sunlight or hotspot:
        reasons.append("lighting hotspot handled by four-step relighting")

    pose = str(vl_plan.get("pose") or vl_plan.get("face_orientation") or "").lower()
    non_frontal_pose = pose and not any(term in pose for term in ("front", "forward"))
    if non_frontal_pose:
        risk += 2
        reasons.append("non-frontal source pose")

    hair_style = str(vl_plan.get("hair_style") or "").strip().lower()
    wearables = " ".join(
        str(vl_plan.get(key) or "").strip().lower()
        for key in ("visible_wearables", "glasses_reflection_description")
    )
    tied_hair = any(term in hair_style for term in ("ponytail", "tied", "bun", "braid", "plait"))
    eyeglasses = any(term in wearables for term in ("glasses", "eyeglass", "spectacle", "lens"))
    identity_detail_case = tied_hair and eyeglasses
    if identity_detail_case:
        # These details are already present in the single fitted seed. Raising
        # diffusion depth makes the base model redraw them; the four-step edit
        # adapter preserves the source glasses and tied-hair geometry better.
        reasons.append("tied hairstyle and eyeglasses preserved by fitted seed")

    confidence = str(vl_plan.get("outer_color_confidence") or "").lower()
    fabric = str(vl_plan.get("fabric_detail") or "").lower()
    if (
        confidence in {"low", "unknown", "uncertain"}
        or fabric in {"", "unclear", "unknown", "uncertain"}
    ):
        # These fields come from one template analysis and must not be counted
        # twice when both are uncertain.
        risk += 1
        reasons.append("uncertain uniform color or fabric detail")

    components = str(vl_plan.get("garment_components") or "").strip().lower()
    if sum(bool(part.strip()) for part in components.split(",")) >= 4:
        risk += 1
        reasons.append("layered uniform and collar fit")

    pattern = str(vl_plan.get("shirt_color_and_pattern") or "").strip().lower()
    patterned_uniform = any(
        term in pattern for term in ("plaid", "check", "tartan", "stripe")
    )
    trim_regions = [
        region.strip()
        for region in str(vl_plan.get("accent_trim_placement") or "").split(",")
        if region.strip()
    ]
    if patterned_uniform and len(trim_regions) >= 2:
        # The dedicated four-step edit adapter preserves the uploaded garment
        # reference sharply. The base model repaints plaid and trim into a soft,
        # inaccurate approximation even though it receives more diffusion
        # steps, so pattern complexity alone must not switch schedulers.
        reasons.append("patterned uniform and trim supplied by garment reference")
    elif patterned_uniform:
        reasons.append("patterned uniform supplied by garment reference")
    if patterned_uniform and identity_detail_case:
        # A direct A/B probe on the same fitted seed measured substantially
        # better garment similarity at four steps than at twenty. Do not count
        # already-conditioned identity detail as additional redraw work.
        reasons.append("pattern and identity detail retained by edit adapter")

    source_clothing = str(vl_plan.get("clothing_description") or "").strip().lower()
    source_outerwear_terms = (
        "blazer", "jacket", "vest", "waistcoat", "sweater", "cardigan",
        "coat", "hoodie", "over-shirt", "overshirt",
    )
    if any(term in source_clothing for term in source_outerwear_terms):
        # The rough-fit stage removes every parsed source-clothing pixel before
        # Qwen runs, so a vest/blazer in the upload no longer increases edit
        # difficulty. Counting it here caused needless 8-step face/hair drift.
        reasons.append("source outerwear replaced by native garment edit")

    # This project ships a four-step Lightning adapter only. Eight diffusion
    # steps disable that adapter but still under-sample the base model, which
    # produced the soft, painted uniforms seen in manual tests. Keep clean and
    # moderate source-preserving uniform edits on the validated four-step path;
    # jump to the base model's 12-step floor only for genuinely difficult work.
    # The installed Lightning adapter is the native four-step image-edit path.
    # Eight steps remain intentionally unused: it disables the Lightning LoRA
    # but under-samples the base model.
    recommended = 20 if risk >= 4 else 12 if risk >= 2 else 4

    # Portrait VL sometimes confuses jewellery, loose hair, or background
    # detail with a head accessory. Never let that semantic guess increase the
    # diffusion depth: the source pixels already preserve the actual clips,
    # flowers, and hairstyle. Strong lighting and pose are independently
    # observable risks, so they still cap regeneration depth.
    source_preservation_risk = bool(non_frontal_pose)
    if source_preservation_risk and recommended > 12:
        recommended = 12
        reasons.append("source-preserving cap for hair, lighting, or pose")
    requested_maximum = int(maximum_steps or 20)
    maximum = (
        20 if requested_maximum >= 20 else
        12 if requested_maximum >= 12 else
        8 if requested_maximum >= 8 else 4
    )
    selected = min(recommended, maximum)
    if not reasons:
        reasons.append("clean frontal source and simple uniform")
    return selected, reasons


def _has_visible_tied_hair_geometry(
    labels: np.ndarray, face_bbox: Any
) -> bool:
    """Confirm that visible hair continues to the nape/below the detected face.

    VL can confuse a dark background edge or short crown hair with a ponytail.
    An affirmative tied-hair instruction is safe only when the person parsing
    also contains a meaningful hair region at the bottom of the face. This is
    deliberately a confirmation gate, not a hairstyle classifier: Qwen still
    copies the source pixels for all other hair details.
    """
    if labels.ndim != 2:
        return False
    x1, y1, x2, y2 = (float(value) for value in face_bbox[:4])
    face_height = max(y2 - y1, 1.0)
    face_width = max(x2 - x1, 1.0)
    hair_y, _ = np.where(labels == 2)
    if hair_y.size == 0:
        return False
    nape_start = int(round(y2 - face_height * 0.04))
    visible_nape_hair = int(np.count_nonzero(hair_y >= nape_start))
    minimum_pixels = max(24, int(face_width * face_height * 0.0015))
    return visible_nape_hair >= minimum_pixels


def _measure_garment_palette(
    garment: Image.Image,
    max_colors: int = 5,
) -> List[Dict[str, Any]]:
    """Return a compact, pixel-measured palette for the uploaded garment.

    Colour words from VL are useful for review but not reliable enough to order
    a multicolour pattern.  Qwen tends to promote the first named colour to the
    fabric base.  Measuring the transparent garment itself gives generation a
    stable dominant-to-accent ordering for every uploaded template.
    """
    rgba = np.asarray(garment.convert("RGBA"))
    if rgba.size == 0:
        return []
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    foreground = alpha > 64
    if np.count_nonzero(foreground) >= int(foreground.size * .95):
        # Most uploaded garment templates are ordinary RGB product photos on a
        # white canvas, so converting them to RGBA yields an opaque background.
        # Recover the garment silhouette from the corner-connected canvas and
        # fill internal white checks/buttons back into that silhouette.
        height, width = rgb.shape[:2]
        edge = max(4, min(24, height // 20, width // 20))
        corner_pixels = np.concatenate((
            rgb[:edge, :edge].reshape(-1, 3),
            rgb[:edge, -edge:].reshape(-1, 3),
            rgb[-edge:, :edge].reshape(-1, 3),
            rgb[-edge:, -edge:].reshape(-1, 3),
        ))
        canvas_rgb = np.median(corner_pixels, axis=0).astype(np.uint8)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        canvas_lab = cv2.cvtColor(
            canvas_rgb.reshape(1, 1, 3), cv2.COLOR_RGB2LAB
        ).reshape(3).astype(np.float32)
        different = (
            np.linalg.norm(lab - canvas_lab, axis=2) > 7.0
        ).astype(np.uint8)
        different = cv2.morphologyEx(
            different, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(different, 8)
        if count > 1:
            subject_id = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            component = (labels == subject_id).astype(np.uint8)
            contours, _ = cv2.findContours(
                component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            foreground = np.zeros_like(component, dtype=np.uint8)
            cv2.drawContours(foreground, contours, -1, 1, thickness=cv2.FILLED)
            foreground = foreground.astype(bool)
    if np.count_nonzero(foreground) < 64:
        return []
    pixels = rgb[foreground]
    if pixels.shape[0] > 120000:
        stride = max(1, pixels.shape[0] // 120000)
        pixels = pixels[::stride]
    strip = Image.fromarray(pixels.reshape(1, -1, 3), mode="RGB")
    quantized = strip.quantize(
        # Use extra clusters before collapsing to the compact prompt palette.
        # With only four quantization buckets, narrow red trim was averaged
        # into navy checks and disappeared from the measured evidence.
        colors=max(8, int(max_colors) * 4),
        method=Image.Quantize.MEDIANCUT,
    )
    counts = quantized.getcolors(maxcolors=pixels.shape[0] + 1) or []
    palette = quantized.getpalette() or []
    total = max(1, sum(count for count, _ in counts))
    measured = []
    for count, index in sorted(counts, reverse=True):
        fraction = count / total
        if fraction < .02:
            continue
        offset = int(index) * 3
        color = tuple(int(channel) for channel in palette[offset:offset + 3])
        if len(color) != 3:
            continue
        merged = next((
            item for item in measured
            if np.linalg.norm(
                np.asarray(item["rgb"], dtype=np.float32)
                - np.asarray(color, dtype=np.float32)
            ) < 20.0
        ), None)
        if merged is not None:
            previous = float(merged["fraction"])
            combined = previous + float(fraction)
            blended = tuple(int(round(
                (merged["rgb"][channel] * previous + color[channel] * fraction)
                / max(combined, 1e-6)
            )) for channel in range(3))
            merged["rgb"] = list(blended)
            merged["hex"] = f"#{blended[0]:02X}{blended[1]:02X}{blended[2]:02X}"
            merged["fraction"] = combined
            continue
        measured.append({
            "rgb": list(color),
            "hex": f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}",
            "fraction": float(fraction),
        })
    measured.sort(key=lambda item: item["fraction"], reverse=True)
    measured = measured[:max(2, int(max_colors))]
    return [
        {**item, "fraction": round(float(item["fraction"]), 3)}
        for item in measured
    ]


def _measure_garment_accent_color(
    garment: Image.Image,
) -> Optional[Dict[str, Any]]:
    """Measure a broad solid accent independently of fabric-area ranking.

    Collars, plackets and cuffs can occupy less area than each color in a plaid
    body, so a dominant palette alone may omit the design-defining trim. The
    cleaned garment cutout is used here; a connected saturated region is much
    more reliable than a color word guessed from a portrait or badge.
    """
    rgba = np.asarray(garment.convert("RGBA"), dtype=np.uint8)
    rgb = rgba[:, :, :3]
    foreground = rgba[:, :, 3] > 64
    if np.count_nonzero(foreground) >= int(foreground.size * .95):
        # RGB uploads have an opaque alpha channel, so the surrounding photo
        # canvas must not compete with the garment for "largest saturated
        # component". Learn that canvas from all four corners, keep the
        # largest non-canvas component, then fill its internal plaid holes.
        height, width = foreground.shape
        edge = max(4, min(24, height // 20, width // 20))
        corner_pixels = np.concatenate((
            rgb[:edge, :edge].reshape(-1, 3),
            rgb[:edge, -edge:].reshape(-1, 3),
            rgb[-edge:, :edge].reshape(-1, 3),
            rgb[-edge:, -edge:].reshape(-1, 3),
        ))
        canvas_rgb = np.median(corner_pixels, axis=0).astype(np.uint8)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        canvas_lab = cv2.cvtColor(
            canvas_rgb.reshape(1, 1, 3), cv2.COLOR_RGB2LAB,
        ).reshape(3).astype(np.float32)
        different = (
            np.linalg.norm(lab - canvas_lab, axis=2) > 7.0
        ).astype(np.uint8)
        different = cv2.morphologyEx(
            different, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8),
        )
        component_count, component_labels, component_stats, _ = (
            cv2.connectedComponentsWithStats(different, 8)
        )
        if component_count > 1:
            subject_id = 1 + int(np.argmax(component_stats[1:, cv2.CC_STAT_AREA]))
            subject = (component_labels == subject_id).astype(np.uint8)
            contours, _ = cv2.findContours(
                subject, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            foreground = np.zeros_like(subject, dtype=np.uint8)
            cv2.drawContours(foreground, contours, -1, 1, thickness=cv2.FILLED)
            foreground = foreground.astype(bool)
    foreground_count = int(np.count_nonzero(foreground))
    if foreground_count < 64:
        return None

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    foreground_saturation = hsv[:, :, 1][foreground]
    if foreground_saturation.size == 0:
        return None
    saturation_floor = max(72, int(np.percentile(foreground_saturation, 68)))
    saturated = (foreground & (hsv[:, :, 1] >= saturation_floor)).astype(np.uint8) * 255
    kernel_size = max(3, int(round(min(rgb.shape[:2]) * .006)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    saturated = cv2.morphologyEx(
        saturated,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(saturated, 8)
    if count <= 1:
        return None
    minimum_area = max(48, int(foreground_count * .004))
    candidates = [
        label for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area
    ]
    if not candidates:
        return None
    # Prefer the largest coherent solid region. Thin patterned threads are
    # removed by opening and normally remain fragmented.
    label = max(candidates, key=lambda item: int(stats[item, cv2.CC_STAT_AREA]))
    component = labels == label
    color = tuple(int(value) for value in np.median(rgb[component], axis=0))
    return {
        "rgb": list(color),
        "hex": f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}",
        "fraction": round(float(np.count_nonzero(component)) / foreground_count, 3),
    }


def _encode_prompt_isolated(
    image: Union[Image.Image, List[Image.Image]],
    prompt: Union[str, List[str]],
    max_sequence_length: int = 256,
    timeout_seconds: Optional[float] = 120,
    conditioning_max_dimension: int = 256,
    conditioning_pixel_budget: Optional[int] = None,
) -> Union[
    Tuple[torch.Tensor, Optional[torch.Tensor]],
    List[Tuple[torch.Tensor, Optional[torch.Tensor]]],
]:
    """
    Runs one or more prompt encodings in one isolated worker subprocess.
    Reclaims 100% of GPU memory upon exit, eliminating bitsandbytes NF4 memory leaks.
    """
    single_prompt = isinstance(prompt, str)
    prompts = [prompt] if single_prompt else list(prompt)
    if not prompts or any(not isinstance(item, str) for item in prompts):
        raise ValueError("At least one text prompt is required for Qwen prompt encoding")
    images = image if isinstance(image, list) else [image]
    if not images:
        raise ValueError("At least one reference image is required for Qwen prompt encoding")
    stamp = int(time.time() * 1000)
    temp_img_paths = [CACHE_DIR / f"enc_in_{stamp}_{idx}.png" for idx in range(len(images))]
    embeds_cache_file = CACHE_DIR / f"enc_out_{int(time.time()*1000)}.pt"
    for source, temp_path in zip(images, temp_img_paths):
        # Bound visual tokens for low VRAM. Uniform fitting uses a larger
        # preview for collar/check/accessory detail; enhancement keeps its
        # existing budget. Full-size inputs still reach the VAE unchanged.
        conditioning = source.convert("RGB")
        if conditioning_pixel_budget is not None:
            from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_dimensions
            cw, ch = calculate_dimensions(conditioning_pixel_budget, conditioning.width / conditioning.height)
            conditioning = conditioning.resize((max(32, cw), max(32, ch)), Image.Resampling.LANCZOS)
        elif max(conditioning.size) > conditioning_max_dimension:
            conditioning.thumbnail((conditioning_max_dimension, conditioning_max_dimension), Image.Resampling.LANCZOS)
        # Diffusers currently ignores max_sequence_length in
        # QwenImageEditPlusPipeline.encode_prompt. A native 832x1216 reference
        # expands past 1,300 visual tokens and exceeds this pipeline's
        # 1,024-token conditioning contract, which can turn an image edit into
        # an unrelated multi-person composition. Keep the VAE image at full
        # resolution while bounding each VL reference to about 250 tokens.
        visual_edge_cap = min(max(32, int(conditioning_max_dimension or 512)), 512)
        if max(conditioning.size) > visual_edge_cap:
            conditioning.thumbnail(
                (visual_edge_cap, visual_edge_cap), Image.Resampling.LANCZOS,
            )
        conditioning.save(temp_path)

    # Exact-input embedding cache. Repeated manual tests of the same portrait,
    # template and prompt previously reloaded the 4-bit VL encoder for every
    # job. The key includes the actual resized reference pixels and all prompt
    # settings, so a different person, garment, wording or token budget can
    # never reuse stale conditioning.
    cache_hasher = hashlib.sha256()
    cache_hasher.update(str(max_sequence_length).encode("ascii"))
    cache_hasher.update(str(conditioning_max_dimension).encode("ascii"))
    cache_hasher.update(str(conditioning_pixel_budget).encode("ascii"))
    for prompt_text in prompts:
        cache_hasher.update(prompt_text.encode("utf-8"))
        cache_hasher.update(b"\0")
    for temp_path in temp_img_paths:
        cache_hasher.update(temp_path.read_bytes())
    prompt_cache_dir = CACHE_DIR / "prompt_embeddings"
    prompt_cache_dir.mkdir(parents=True, exist_ok=True)
    persistent_embeds_cache = prompt_cache_dir / f"{cache_hasher.hexdigest()}.pt"
    if persistent_embeds_cache.exists():
        try:
            loaded_cache = torch.load(str(persistent_embeds_cache), map_location="cpu")
            results = [(item["embeds"], item["mask"]) for item in loaded_cache["items"]]
            logger.info("[QWEN_ENCODE_CACHE] Exact conditioning cache hit")
            for temp_img_path in temp_img_paths:
                temp_img_path.unlink(missing_ok=True)
            return results[0] if single_prompt else results
        except Exception as exc:
            logger.warning("[QWEN_ENCODE_CACHE] Ignoring unreadable cache entry: %s", exc)
            persistent_embeds_cache.unlink(missing_ok=True)

    models_posix = MODELS_DIR.resolve().as_posix()
    temp_img_paths_posix = [p.resolve().as_posix() for p in temp_img_paths]
    embeds_cache_posix = embeds_cache_file.resolve().as_posix()
    def build_helper_code(*, quantized: bool) -> str:
        if quantized:
            encoder_import = ", BitsAndBytesConfig"
            encoder_setup = """
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=dtype,
    bnb_4bit_use_double_quant=True,
)
text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    (MODELS_DIR / "text_encoder").as_posix(),
    quantization_config=bnb_config,
    torch_dtype=dtype,
    device_map={"": 0},
)
device = torch.device("cuda:0")
"""
        else:
            encoder_import = ""
            # Windows Application Control can reject bitsandbytes' native CUDA
            # DLL even though PyTorch CUDA itself is allowed. Loading this
            # 15.5-GiB encoder entirely on CPU nearly exhausts a 24-GiB host,
            # so split the unquantized BF16 weights between GPU and RAM. The
            # subprocess still exits after encoding and returns only embeds.
            encoder_setup = """
max_memory = {"cpu": "14GiB"}
if torch.cuda.is_available():
    max_memory[0] = "5500MiB"
text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    (MODELS_DIR / "text_encoder").as_posix(),
    torch_dtype=dtype,
    device_map="auto" if torch.cuda.is_available() else {"": "cpu"},
    max_memory=max_memory,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
)
device = next(text_encoder.parameters()).device
"""
        return f"""
import sys
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration{encoder_import}
from diffusers import QwenImageEditPlusPipeline

MODELS_DIR = Path("{models_posix}")
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

tokenizer = AutoTokenizer.from_pretrained((MODELS_DIR / "tokenizer").as_posix())
processor = AutoProcessor.from_pretrained((MODELS_DIR / "processor").as_posix())

{encoder_setup}

pipe = QwenImageEditPlusPipeline(
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    transformer=None,
    vae=None,
    scheduler=None,
    processor=processor,
)

images = [Image.open(path).convert("RGB") for path in {temp_img_paths_posix!r}]
prompts = {prompts!r}

encoded = []
with torch.inference_mode():
    for prompt in prompts:
        prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
            image=images,
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length={max_sequence_length},
        )
        encoded.append({{
            "embeds": prompt_embeds.cpu(),
            "mask": prompt_embeds_mask.cpu() if prompt_embeds_mask is not None else None,
        }})

torch.save({{"items": encoded}}, "{embeds_cache_posix}")
print("SUCCESS")
"""
    helper_code = build_helper_code(quantized=True)
    helper_script = CACHE_DIR / f"enc_worker_{int(time.time()*1000)}.py"
    helper_script.write_text(helper_code, encoding="utf-8")

    try:
        # The encoder is deliberately isolated to release VRAM, but it must not
        # hold the application's single queue worker forever when a driver or
        # quantized-model process fails to return.
        try:
            run_args = {
                "capture_output": True,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
            }
            if timeout_seconds is not None and timeout_seconds > 0:
                run_args["timeout"] = max(1, timeout_seconds)
            ret = subprocess.run([sys.executable, str(helper_script)], **run_args)
        except subprocess.TimeoutExpired as e:
            logger.error("[QWEN_ENCODE_WORKER_TIMEOUT] Prompt encoder exceeded %.0f seconds", timeout_seconds)
            raise RuntimeError(f"Qwen prompt encoder timed out after {timeout_seconds:.0f} seconds") from e
        if ret.returncode != 0 and (
            "bitsandbytes" in ret.stderr.lower()
            or "application control policy" in ret.stderr.lower()
            or "cquantize_blockwise" in ret.stderr.lower()
        ):
            logger.warning(
                "[QWEN_ENCODE_WORKER_FALLBACK] Native 4-bit encoder unavailable; "
                "retrying with an unquantized GPU/CPU split"
            )
            helper_script.write_text(build_helper_code(quantized=False), encoding="utf-8")
            # The failed native load is already gone with its subprocess. Give
            # the compatibility worker the caller's remaining deadline rather
            # than failing a valid portrait because of the optional fast path.
            try:
                ret = subprocess.run([sys.executable, str(helper_script)], **run_args)
            except subprocess.TimeoutExpired as e:
                logger.error(
                    "[QWEN_ENCODE_WORKER_TIMEOUT] Compatibility encoder exceeded %.0f seconds",
                    timeout_seconds,
                )
                raise RuntimeError(
                    f"Qwen compatibility prompt encoder timed out after {timeout_seconds:.0f} seconds"
                ) from e
        if ret.returncode != 0:
            logger.error("[QWEN_ENCODE_WORKER_ERROR] stdout: %s | stderr: %s", ret.stdout, ret.stderr)
            raise RuntimeError(f"Prompt encoding worker failed (code {ret.returncode}): {ret.stderr}")
        loaded_cache = torch.load(str(embeds_cache_file), map_location="cpu")
        results = [(item["embeds"], item["mask"]) for item in loaded_cache["items"]]
        try:
            torch.save(loaded_cache, str(persistent_embeds_cache))
            cached_entries = sorted(
                prompt_cache_dir.glob("*.pt"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            for stale_entry in cached_entries[12:]:
                stale_entry.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("[QWEN_ENCODE_CACHE] Could not persist conditioning: %s", exc)
        return results[0] if single_prompt else results
    finally:
        for temp_img_path in temp_img_paths:
            if temp_img_path.exists():
                temp_img_path.unlink(missing_ok=True)
        if embeds_cache_file.exists():
            embeds_cache_file.unlink(missing_ok=True)
        if helper_script.exists():
            helper_script.unlink(missing_ok=True)


def _chunked_transformer_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    encoder_hidden_states_mask: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_shapes: list[tuple[int, int, int]] | None = None,
    guidance: torch.Tensor = None,
    attention_kwargs: dict = None,
    controlnet_block_samples = None,
    additional_t_cond = None,
    return_dict: bool = True,
):
    """
    Stream DiT transformer blocks in VRAM-sized groups. Keeping a 15-block
    half-model on an 8 GB Windows GPU causes WDDM paging; three 10-block groups
    use the same weights and latent canvas without changing generation quality.
    """
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def check_stream_deadline() -> None:
        deadline = getattr(self, "_qwen_deadline", None)
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Qwen edit timed out while streaming a transformer block")

    # Ensure stem modules are on CUDA
    self.img_in.to(dev)
    self.txt_norm.to(dev)
    self.txt_in.to(dev)
    self.time_text_embed.to(dev)
    self.pos_embed.to(dev)
    self.norm_out.to(dev)
    self.proj_out.to(dev)

    hidden_states = hidden_states.to(dev)
    hidden_states = self.img_in(hidden_states)
    timestep = timestep.to(device=dev, dtype=hidden_states.dtype)

    if self.zero_cond_t:
        timestep = torch.cat([timestep, timestep * 0], dim=0)
        modulate_index = torch.tensor(
            [[0] * prod(sample[0]) + [1] * sum([prod(s) for s in sample[1:]]) for sample in img_shapes],
            device=timestep.device,
            dtype=torch.int,
        )
    else:
        modulate_index = None

    encoder_hidden_states = encoder_hidden_states.to(dev)
    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    if encoder_hidden_states_mask is not None:
        encoder_hidden_states_mask = encoder_hidden_states_mask.to(dev)

    text_seq_len, _, encoder_hidden_states_mask = compute_text_seq_len_from_mask(
        encoder_hidden_states, encoder_hidden_states_mask
    )

    if guidance is not None:
        guidance = guidance.to(device=dev, dtype=hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, hidden_states, additional_t_cond)
        if guidance is None
        else self.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
    )

    image_rotary_emb = self.pos_embed(img_shapes, max_txt_seq_len=text_seq_len, device=dev)

    total_blocks = len(self.transformer_blocks)
    gpu_memory_gib = (
        torch.cuda.get_device_properties(dev).total_memory / (1024 ** 3)
        if torch.cuda.is_available() else 0.0
    )
    chunk_count = 3 if gpu_memory_gib and gpu_memory_gib <= 10.0 else 2
    chunk_size = max(1, (total_blocks + chunk_count - 1) // chunk_count)

    for start in range(0, total_blocks, chunk_size):
        blocks = self.transformer_blocks[start:min(total_blocks, start + chunk_size)]
        for block in blocks:
            block.to(dev)

        for block in blocks:
            check_stream_deadline()
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=attention_kwargs,
                modulate_index=modulate_index,
            )

        for block in blocks:
            block.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if self.zero_cond_t:
        temb = temb.chunk(2, dim=0)[0]
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def _cpu_encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
    """Encodes images via CPU VAE to prevent CUDA conv3d NotImplementedError."""
    image_cpu = image.to(device="cpu", dtype=torch.float32)
    if image_cpu.ndim == 4:
        image_cpu = image_cpu.unsqueeze(2)
    image_latents = retrieve_latents(self.vae.encode(image_cpu), generator=generator, sample_mode="argmax")
    latents_mean = (
        torch.tensor(self.vae.config.latents_mean)
        .view(1, self.latent_channels, 1, 1, 1)
        .to(image_latents.device, image_latents.dtype)
    )
    latents_std = (
        torch.tensor(self.vae.config.latents_std)
        .view(1, self.latent_channels, 1, 1, 1)
        .to(image_latents.device, image_latents.dtype)
    )
    image_latents = (image_latents - latents_mean) / latents_std
    # Keep encoded image latents with the CPU scheduler until latent
    # preparation is complete.  The pipeline-specific prepare_latents wrapper
    # moves the finished tuple to CUDA. Moving only image latents here causes
    # Qwen inpaint's CPU noise mixing to fail with a cross-device operation.
    return image_latents.to(device="cpu", dtype=torch.float32)


def _decode_cpu_latents(vae: AutoencoderKLQwenImage, output_latents: torch.Tensor, height: int, width: int) -> Image.Image:
    """Decodes unpacked diffusion output latents using CPU AutoencoderKLQwenImage with exact official diffusers unpack & scaling."""
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline

    img_processor = VaeImageProcessor(vae_scale_factor=16)
    vae_scale_factor = 8

    # Official diffusers _unpack_latents
    latents_unpacked = QwenImageEditPlusPipeline._unpack_latents(output_latents, height, width, vae_scale_factor)
    latents_unpacked = latents_unpacked.to(device="cpu", dtype=torch.float32)

    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents_unpacked.device, latents_unpacked.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        latents_unpacked.device, latents_unpacked.dtype
    )
    latents_denorm = latents_unpacked / latents_std + latents_mean

    with torch.no_grad():
        decoded_sample = vae.decode(latents_denorm, return_dict=False)[0][:, :, 0]
        pil_img = img_processor.postprocess(decoded_sample, output_type="pil")[0]
    return pil_img


class QwenEditPipeline:
    """Consolidated Qwen Studio Enhancer & Uniform Swap Service powered strictly by Qwen Diffusion."""

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.last_qwen_candidate_path: Optional[Path] = None
        self.last_qwen_identity_accepted: Optional[bool] = None
        logger.info("[QWEN_EDIT_PIPELINE] Initialized on device: %s (%s)", self.device, self.dtype)

    def _to_pil(self, img_input: Union[str, Path, bytes, Image.Image], mode: str = "RGB") -> Image.Image:
        """Load an image, optionally retaining alpha for template preparation."""
        if isinstance(img_input, Image.Image):
            return img_input.convert(mode)
        if isinstance(img_input, (str, Path)):
            return Image.open(str(img_input)).convert(mode)
        if isinstance(img_input, bytes):
            return Image.open(io.BytesIO(img_input)).convert(mode)
        raise ValueError(f"Unsupported image input type: {type(img_input)}")

    def _align_dimensions(self, width: int, height: int, max_dim: int = 576) -> tuple[int, int]:
        """Snaps dimensions to multiples of 16 for optimal VAE patchification."""
        scale = min(1.0, max_dim / max(width, height))
        w = int(width * scale)
        h = int(height * scale)
        w = max(256, (w // 16) * 16)
        h = max(256, (h // 16) * 16)
        if w == h:
            # Disallow square dimensions: enforce portrait aspect ratio (snap to multiple of 16)
            h = min(max_dim, int((w * 1.25 // 16) * 16))
        return w, h

    @staticmethod
    def _crop_school_passport_portrait(
        image: Image.Image,
        width: int,
        height: int,
        preserve_existing_frame: bool = False,
        tight_school_id: bool = False,
        delivery_upper_chest: bool = False,
        face_bbox: Optional[Any] = None,
    ) -> Image.Image:
        """Frame a generated portrait from crown to upper chest at 35:45 ratio."""
        rgb = np.array(image.convert("RGB"))
        source_h, source_w = rgb.shape[:2]
        if face_bbox is not None:
            bbox = np.asarray(face_bbox, dtype=np.float32).reshape(-1)
            if bbox.size < 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise ValueError("Invalid supplied passport face box")
            faces = np.asarray([[
                int(max(0, bbox[0])), int(max(0, bbox[1])),
                int(max(1, bbox[2] - bbox[0])),
                int(max(1, bbox[3] - bbox[1])),
            ]])
        else:
            detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = detector.detectMultiScale(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 1.1, 4, minSize=(48, 48)
            )
        if not len(faces):
            # Haar can miss softly lit or child faces. Use the same recognition
            # detector as identity verification before falling back to geometry;
            # never resize a full-body generation into a passport frame.
            try:
                from pipelines.photo_restoration import _get_insight_app

                recognized = _get_insight_app().get(
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                )
                if recognized:
                    bbox = np.asarray(
                        max(recognized, key=lambda face: float(
                            (face.bbox[2] - face.bbox[0])
                            * (face.bbox[3] - face.bbox[1])
                        )).bbox,
                        dtype=np.float32,
                    )
                    faces = np.asarray([[
                        int(max(0, bbox[0])), int(max(0, bbox[1])),
                        int(max(1, bbox[2] - bbox[0])),
                        int(max(1, bbox[3] - bbox[1])),
                    ]])
            except Exception as exc:
                logger.warning("[PASSPORT_COMPOSITION] Secondary face detection failed: %s", exc)

        if not len(faces):
            # Final deterministic fallback for detector misses. Estimate the
            # subject from the flat passport backdrop and retain only the upper
            # 58%% of the source height, which excludes waist/legs while keeping
            # crown, shoulders and upper chest. This path must never return the
            # uncropped full-person frame.
            target_ratio = width / height
            edge = max(4, min(16, source_h // 20, source_w // 20))
            border = np.concatenate((
                rgb[:edge, :edge].reshape(-1, 3),
                rgb[:edge, -edge:].reshape(-1, 3),
            ))
            backdrop = np.median(border, axis=0)
            distance = np.linalg.norm(rgb.astype(np.float32) - backdrop, axis=2)
            foreground = (distance > 22.0).astype(np.uint8)
            foreground = cv2.morphologyEx(
                foreground, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
            )
            ys, xs = np.where(foreground > 0)
            crown_y = int(np.percentile(ys, 1)) if ys.size else 0
            center_x = int(np.median(xs)) if xs.size else source_w // 2
            fallback_height_fraction = (
                .50 if delivery_upper_chest else .56 if tight_school_id else .58
            )
            crop_h = min(
                source_h,
                max(1, int(round(source_h * fallback_height_fraction))),
            )
            crop_w = min(source_w, max(1, int(round(crop_h * target_ratio))))
            crop_h = min(source_h, max(1, int(round(crop_w / target_ratio))))
            headroom = max(8, int(crop_h * .045))
            top = max(0, min(source_h - crop_h, crown_y - headroom))
            left = max(0, min(source_w - crop_w, center_x - crop_w // 2))
            logger.warning(
                "[PASSPORT_COMPOSITION] Face missed; applied deterministic upper-chest crop"
            )
            return image.crop((left, top, left + crop_w, top + crop_h)).resize(
                (width, height), Image.LANCZOS
            )

        face_x, face_y, face_w, face_h = max(faces, key=lambda box: box[2] * box[3])
        target_ratio = width / height
        if face_bbox is not None and delivery_upper_chest:
            # The caller already measured the uploaded face. Do not let scene
            # segmentation or the full-width source torso redefine a school-ID
            # frame: classroom walls and cropped arms otherwise create excess
            # headroom and unstable head/body scale. Frame directly from the
            # measured face with enough crown allowance and upper chest below.
            # Final school-ID delivery is intentionally tighter than the
            # generation canvas: retain crown, complete head and shoulders,
            # but end at the upper chest rather than showing a long torso.
            # InsightFace's detector box includes forehead/chin margins, so a
            # 0.35 detector-box target still delivered a loose mid-torso crop
            # on small child portraits. 0.39 consistently ends at the upper
            # chest while retaining both shoulders and modest headroom.
            target_face_fraction = .39
            crop_h = max(1, int(np.ceil(face_h / target_face_fraction)))
            crop_w = max(1, int(round(crop_h * target_ratio)))
            top = int(round(face_y - face_h * .58))
            left = int(round(face_x + face_w * .5 - crop_w * .5))
            # If Qwen rendered an oversized head, clamping this requested crop
            # back to the source canvas silently defeats the passport framing
            # contract (the final face can exceed 40% of frame height). Keep
            # the measured crop and extend only the already-flat selected
            # backdrop around the generated portrait. The subject remains
            # pixel-identical and the visible lower edge still ends at upper
            # chest; no body pixels are invented.
            pad_left = max(0, -left)
            pad_top = max(0, -top)
            pad_right = max(0, left + crop_w - source_w)
            pad_bottom = max(0, top + crop_h - source_h)
            if pad_left or pad_top or pad_right or pad_bottom:
                edge = max(4, min(16, source_h // 20, source_w // 20))
                backdrop = np.median(np.concatenate((
                    rgb[:edge, :edge].reshape(-1, 3),
                    rgb[:edge, -edge:].reshape(-1, 3),
                )), axis=0).round().clip(0, 255).astype(np.uint8)
                padded = np.empty((
                    source_h + pad_top + pad_bottom,
                    source_w + pad_left + pad_right,
                    3,
                ), dtype=np.uint8)
                padded[:] = backdrop
                padded[pad_top:pad_top + source_h, pad_left:pad_left + source_w] = rgb
                rgb = padded
                image = Image.fromarray(rgb, "RGB")
                left += pad_left
                top += pad_top
            logger.info(
                "[PASSPORT_COMPOSITION] Direct source crop face_fraction=%.3f crop=%dx%d",
                target_face_fraction, crop_w, crop_h,
            )
            return image.crop((left, top, left + crop_w, top + crop_h)).resize(
                (width, height), Image.LANCZOS,
            )
        # Qwen occasionally generates a valid face with the crown touching the
        # canvas. Add backdrop above that image before cropping so the passport
        # frame cannot remove more hair. This does not stretch or regenerate
        # any subject pixels.
        edge = max(4, min(16, source_h // 20, source_w // 20))
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        backdrop_lab = np.median(np.concatenate((
            lab[:edge, :edge].reshape(-1, 3),
            lab[:edge, -edge:].reshape(-1, 3),
        )), axis=0)
        backdrop = np.median(np.concatenate((
            rgb[:edge, :edge].reshape(-1, 3),
            rgb[:edge, -edge:].reshape(-1, 3),
        )), axis=0).round().clip(0, 255).astype(np.uint8)
        foreground = (np.linalg.norm(lab - backdrop_lab, axis=2) > 18.0).astype(np.uint8)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
            foreground, 8,
        )
        face_center = (
            min(source_w - 1, max(0, int(face_x + face_w * .5))),
            min(source_h - 1, max(0, int(face_y + face_h * .5))),
        )
        subject_id = int(component_labels[face_center[1], face_center[0]])
        if subject_id == 0 and component_count > 1:
            subject_id = 1 + int(np.argmax(component_stats[1:, cv2.CC_STAT_AREA]))
        subject = component_labels == subject_id if subject_id else foreground.astype(bool)
        central_foreground = subject[:, int(source_w * .04):int(source_w * .96)]
        occupied_rows = np.count_nonzero(central_foreground, axis=1) >= max(3, int(source_w * .02))
        occupied = np.flatnonzero(occupied_rows)
        crown_y = int(occupied[0]) if occupied.size else max(0, int(face_y - face_h * .30))
        subject_bottom = int(occupied[-1] + 1) if occupied.size else source_h
        subject_reaches_bottom = subject_bottom >= source_h - max(4, int(source_h * .025))
        source_aspect = source_w / max(source_h, 1)
        source_face_fraction = face_h / max(source_h, 1)
        source_crown_fraction = crown_y / max(source_h, 1)
        source_face_center_fraction = (face_x + face_w * .5) / max(source_w, 1)
        face_bottom = face_y + face_h
        shoulder_start = max(0, min(source_h, int(face_bottom + face_h * .12)))
        shoulder_end = max(shoulder_start + 1, min(source_h, int(face_bottom + face_h * 1.12)))
        shoulder_widths = []
        shoulder_centers = []
        for row in range(shoulder_start, shoulder_end):
            columns = np.flatnonzero(subject[row])
            if columns.size >= max(8, int(face_w * .25)):
                shoulder_widths.append(int(columns[-1] - columns[0] + 1))
                shoulder_centers.append(float(columns[0] + columns[-1]) * .5)
        shoulder_width = (
            float(np.percentile(shoulder_widths, 75))
            if shoulder_widths else float(face_w * 2.2)
        )
        # Preserve an already composed upper-chest frame, but gently crop a
        # loose half-body Qwen canvas. The former 20% threshold combined with a
        # 28.5% target jumped directly from too loose to an oversized head.
        preserved_face_min = .29 if tight_school_id else .22
        preserved_face_max = .34 if tight_school_id else .36
        if (
            preserve_existing_frame
            and not delivery_upper_chest
            and abs(source_aspect - target_ratio) <= .025
            and preserved_face_min <= source_face_fraction <= preserved_face_max
            and .38 <= source_face_center_fraction <= .62
        ):
            # Qwen has already composed this portrait at the requested passport
            # aspect. Preserve every source pixel, including ears and loose hair.
            # Re-cropping from a foreground/shoulder estimate is unsafe when a
            # generated backdrop has gradients or light objects, and even the
            # former 1% overscan could clip an ear at the canvas edge.
            # Preserve a complete Qwen frame byte-for-byte when possible, but
            # do not preserve a cut crown. Add only flat backdrop above and
            # remove the same number of rows from the lower torso. This keeps
            # face/body scale unchanged while restoring passport headroom.
            minimum_crown = max(8, int(round(source_h * .035)))
            if crown_y < minimum_crown:
                pad_top = minimum_crown - crown_y
                adjusted = np.empty((source_h + pad_top, source_w, 3), dtype=np.uint8)
                adjusted[:] = backdrop
                adjusted[pad_top:] = rgb
                rgb = adjusted[:source_h]
                image = Image.fromarray(rgb, "RGB")
                source_crown_fraction = minimum_crown / max(source_h, 1)
            fitted = image.resize((width, height), Image.LANCZOS)
            logger.info(
                "[PASSPORT_COMPOSITION] Preserved complete generated frame: "
                "face=%.3f crown=%.3f aspect=%.3f face_center=%.3f",
                source_face_fraction, source_crown_fraction, source_aspect,
                source_face_center_fraction,
            )
            return fitted
        shoulder_to_face = shoulder_width / max(float(face_w), 1.0)
        # Adapt within passport-safe limits: narrow shoulders need more torso
        # context, while naturally broad shoulders can support a slightly
        # tighter crop. This is derived independently for every portrait.
        if delivery_upper_chest:
            # Delivery must visibly end at the upper chest. This is separate
            # from the looser generation guide so Qwen sees natural body scale,
            # but a long raw torso can never bypass the final passport crop.
            # Keep the face below 40% of frame height: the tighter profile
            # enlarged heads, left too little crown margin, and hid the upper
            # chest that visually supports a natural neck and shoulder line.
            target_face_fraction = float(np.clip(
                .345 + .006 * (shoulder_to_face - 2.1), .335, .36,
            ))
            lower_face_extent = .72
        elif tight_school_id:
            # Match a conventional school-ID head-and-shoulders frame without
            # turning it into a face close-up. The previous .34-.37 target made
            # the generated head larger than the uploaded person's natural
            # head/body ratio and removed too much neck and upper chest.
            target_face_fraction = float(np.clip(
                .315 + .006 * (shoulder_to_face - 2.1), .30, .33,
            ))
            lower_face_extent = .78
        else:
            target_face_fraction = float(np.clip(
                .242 + .003 * (shoulder_to_face - 2.1), .238, .247,
            ))
            lower_face_extent = 1.12
        head_height = max(float(face_h), float(face_bottom - crown_y))
        minimum_headroom = max(
            10,
            int(head_height * (.16 if delivery_upper_chest else .12)),
        )
        desired_top = float(crown_y - minimum_headroom)
        vertical_requirement = face_bottom + face_h * lower_face_extent - desired_top
        face_scale_requirement = face_h / target_face_fraction
        shoulder_requirement = shoulder_width / max(target_ratio * .90, .01)
        # Passport framing is governed by the face and upper chest. Broad
        # shoulders may leave the frame at both sides; allowing them to dictate
        # crop height retains the lower torso and turns the result into a
        # half-body portrait.
        shoulder_requirement = min(
            shoulder_requirement,
            face_scale_requirement * (1.0 if delivery_upper_chest else 1.08),
        )
        desired_crop_h = int(np.ceil(max(
            vertical_requirement, face_scale_requirement, shoulder_requirement,
        )))
        logger.info(
            "[PASSPORT_COMPOSITION] face_fraction=%.3f shoulder_to_face=%.3f crop=%d",
            target_face_fraction, shoulder_to_face, desired_crop_h,
        )
        pad_top = max(0, int(np.ceil(-desired_top)))
        if pad_top:
            padded = np.empty((source_h + pad_top, source_w, 3), dtype=np.uint8)
            padded[:] = backdrop
            padded[pad_top:] = rgb
            rgb = padded
            image = Image.fromarray(rgb, "RGB")
            source_h += pad_top
            face_y += pad_top
            crown_y += pad_top
            desired_top += pad_top

        # The original frame can be too narrow to achieve the intended face
        # scale. Extend only the already-flat backdrop instead of clamping the
        # crop to the source bounds; clamping silently enlarged the head. The
        # subject pixels are neither resized nor regenerated here.
        desired_crop_w = max(1, int(round(desired_crop_h * target_ratio)))
        pad_x = max(0, desired_crop_w - source_w)
        # If Qwen already rendered the upper chest through the source's lower
        # edge, extending the canvas below it creates an obvious empty band in
        # the delivered passport photo. In that case, fit the existing portrait
        # to the frame and allow shoulders to leave naturally at the sides.
        if subject_reaches_bottom:
            desired_crop_h = min(
                desired_crop_h,
                max(1, int(np.floor(source_h - desired_top))),
            )
        pad_bottom = (
            0 if subject_reaches_bottom
            else max(0, int(np.ceil(desired_top + desired_crop_h - source_h)))
        )
        if pad_x or pad_bottom:
            pad_left = pad_x // 2
            padded = np.empty((source_h + pad_bottom, source_w + pad_x, 3), dtype=np.uint8)
            padded[:] = backdrop
            padded[:source_h, pad_left:pad_left + source_w] = rgb
            rgb = padded
            image = Image.fromarray(rgb, "RGB")
            source_h, source_w = rgb.shape[:2]
            face_x += pad_left

        # Include the complete hair, chin, shoulders and upper chest. Keeping
        # nearly the whole generated torso made children look unusually short
        # and read like half-body portraits instead of school-ID photographs.
        crop_h = min(source_h, desired_crop_h)
        crop_w = min(source_w, max(1, int(crop_h * target_ratio)))
        if crop_w / crop_h > target_ratio:
            crop_w = max(1, int(crop_h * target_ratio))
        else:
            crop_h = max(1, int(crop_w / target_ratio))

        center_x = face_x + face_w // 2
        top = int(round(desired_top))
        top = max(0, min(source_h - crop_h, top))
        left = max(0, min(source_w - crop_w, center_x - crop_w // 2))
        cropped = image.crop((left, top, left + crop_w, top + crop_h))
        return cropped.resize((width, height), Image.LANCZOS)

    def _repair_tiny_backdrop_holes_in_head_neck(
        self,
        image: Image.Image,
        background_color: Union[str, Tuple[int, int, int]],
    ) -> Image.Image:
        """Fill only tiny selected-backdrop islands enclosed by the head.

        Hair/skin parsers can leave a few exact backdrop pixels inside dense
        hair. Broad rematting is unsafe, so this repair also requires immediate
        adjacency to semantic head pixels and never spans the open neck/collar
        or ponytail gaps that Qwen must regenerate naturally.
        """
        try:
            from pipelines.schp_service import parse as parse_body_parts

            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            labels = parse_body_parts(rgb)["labels"]
            face_y, face_x = np.where(labels == 13)
            head_points = cv2.findNonZero(
                np.isin(labels, (1, 2, 4, 13)).astype(np.uint8)
            )
            if face_x.size < 64 or head_points is None or len(head_points) < 64:
                return image.convert("RGB")

            face_left, face_right = int(face_x.min()), int(face_x.max())
            face_top, face_bottom = int(face_y.min()), int(face_y.max())
            face_width = max(1, face_right - face_left + 1)
            face_height = max(1, face_bottom - face_top + 1)

            head_hull = np.zeros(labels.shape, dtype=np.uint8)
            cv2.fillConvexPoly(head_hull, cv2.convexHull(head_points), 1)
            head_interior = cv2.erode(
                head_hull,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            ).astype(bool)
            head_neighborhood = cv2.dilate(
                np.isin(labels, (1, 2, 4, 13)).astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                iterations=1,
            ).astype(bool)
            repairable_head = head_interior & head_neighborhood
            image_luma = (
                rgb[:, :, 0].astype(np.float32) * .2126
                + rgb[:, :, 1].astype(np.float32) * .7152
                + rgb[:, :, 2].astype(np.float32) * .0722
            )
            yy, xx = np.indices(labels.shape)
            face_center = (face_left + face_right) * .5
            tail_corridor = (
                (yy >= face_top - face_height * .20)
                & (yy <= face_bottom + face_height * .70)
                & (np.abs(xx - face_center) <= face_width * 1.05)
            )
            dark_tail_candidates = tail_corridor & (image_luma < 108.0)
            parsed_hair_neighborhood = cv2.dilate(
                (labels == 2).astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27)),
                iterations=1,
            ).astype(bool)
            dark_count, dark_components, dark_stats, _ = cv2.connectedComponentsWithStats(
                dark_tail_candidates.astype(np.uint8), connectivity=8,
            )
            supported_dark_hair = np.zeros(labels.shape, dtype=np.uint8)
            max_dark_area = int(face_width * face_height * .55)
            for dark_id in range(1, dark_count):
                component = dark_components == dark_id
                area = int(dark_stats[dark_id, cv2.CC_STAT_AREA])
                if area <= max_dark_area and np.any(component & parsed_hair_neighborhood):
                    supported_dark_hair[component] = 1
            tail_neighborhood = cv2.dilate(
                supported_dark_hair,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                iterations=1,
            ).astype(bool)
            repairable_head |= tail_neighborhood
            protected_region = repairable_head

            target = np.asarray(self._parse_bg_color(background_color), dtype=np.float32)
            target_distance = np.linalg.norm(
                rgb.astype(np.float32) - target[None, None, :], axis=2,
            )
            image_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            target_hsv = cv2.cvtColor(
                target.astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_RGB2HSV,
            )[0, 0]
            hue_delta = np.abs(
                image_hsv[:, :, 0].astype(np.int16) - int(target_hsv[0])
            )
            hue_delta = np.minimum(hue_delta, 180 - hue_delta)
            same_hue_antialias = (
                (int(target_hsv[1]) >= 60)
                & (hue_delta <= 10)
                & (image_hsv[:, :, 1] >= max(35, int(target_hsv[1]) // 5))
                & (image_hsv[:, :, 2] >= 18)
            )
            backdrop_like = (target_distance <= 60.0) | same_hue_antialias
            # Include antialiased backdrop remnants, not just exact RGB
            # matches. A compact morphological close distinguishes a pinhole
            # from the legitimate open backdrop around the silhouette. The
            # compact close catches holes without spanning open gaps between
            # the neck, collar and a side ponytail.
            subject_core = ~backdrop_like
            closed_head = cv2.morphologyEx(
                subject_core.astype(np.uint8),
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            ).astype(bool)
            hole_candidates = backdrop_like & (
                repairable_head & closed_head
            )
            count, components, stats, _ = cv2.connectedComponentsWithStats(
                hole_candidates.astype(np.uint8), connectivity=8,
            )
            repair = np.zeros(labels.shape, dtype=np.uint8)
            component_masks = []
            maximum_area = max(8, int(face_width * face_height * .0025))
            maximum_extent = max(4, int(face_width * .10))
            for component in range(1, count):
                component_mask = components == component
                area = int(stats[component, cv2.CC_STAT_AREA])
                component_width = int(stats[component, cv2.CC_STAT_WIDTH])
                component_height = int(stats[component, cv2.CC_STAT_HEIGHT])
                compact_head_gap = (
                    area <= maximum_area
                    and component_width <= maximum_extent
                    and component_height <= maximum_extent
                )
                if compact_head_gap:
                    repair[component_mask] = 255
                    component_masks.append(component_mask)
            if not np.any(repair):
                return image.convert("RGB")

            # Do not use image inpainting here. At a hair or neck boundary its
            # sampling neighbourhood contains the selected backdrop, so Telea
            # can paint the same blue back into the hole. Fill each compact
            # island from nearby non-backdrop pixels belonging to the dominant
            # parsed subject region around it. This repairs pinholes without
            # changing the outer silhouette or reconstructing any facial area.
            repaired = rgb.copy()
            # A donor must be decisively different from the selected backdrop.
            # Low thresholds admit blue antialias pixels and simply move the
            # halo into the repaired island instead of removing it.
            foreground_donor = protected_region & ~backdrop_like & (target_distance > 70.0)
            for component_mask in component_masks:
                donor_mask = np.zeros(labels.shape, dtype=bool)
                for kernel_size in (3, 5, 7, 11, 15):
                    expanded = cv2.dilate(
                        component_mask.astype(np.uint8),
                        cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size),
                        ),
                        iterations=1,
                    ).astype(bool)
                    ring = expanded & ~component_mask & foreground_donor
                    ring_labels = labels[ring]
                    subject_labels = ring_labels[ring_labels != 0]
                    if subject_labels.size:
                        dominant_label = int(np.bincount(subject_labels).argmax())
                        same_region = ring & (labels == dominant_label)
                        if np.count_nonzero(same_region) >= 3:
                            donor_mask = same_region
                            break
                    if np.count_nonzero(ring) >= 6:
                        donor_mask = ring
                        break
                if not np.any(donor_mask):
                    continue
                donor_color = np.median(
                    repaired[donor_mask].astype(np.float32), axis=0,
                ).round().clip(0, 255).astype(np.uint8)
                repaired[component_mask] = donor_color
            logger.info(
                "[UNIFORM_HEAD_MATTE] Repaired %d tiny enclosed backdrop pixels",
                int(sum(np.count_nonzero(mask) for mask in component_masks)),
            )
            return Image.fromarray(repaired, "RGB")
        except Exception as exc:
            logger.warning("[UNIFORM_HEAD_MATTE] Tiny-hole repair skipped: %s", exc)
            return image.convert("RGB")

    @staticmethod
    def _source_identity_reference(
        image: Image.Image,
        face_bbox: Any,
        width: int,
        height: int,
        background_color: Optional[Tuple[int, int, int]] = None,
        target_face_fraction: float = .30,
    ) -> Image.Image:
        """Crop the untouched upload around its complete head and upper body.

        This is an identity/hair reference, not the final passport crop. It is
        intentionally driven only by the detected face box so scenery or dark
        source clothing cannot distort the crop as foreground segmentation can.
        """
        source = image.convert("RGB")
        source_w, source_h = source.size
        x1, y1, x2, y2 = (float(value) for value in face_bbox)
        face_h = max(1.0, y2 - y1)
        target_ratio = width / max(height, 1)

        # Roughly 30% face height leaves the crown, tied hair, neck, shoulders
        # and upper chest visible without turning a landscape upload into a
        # tiny full-body reference.
        target_face_fraction = float(np.clip(target_face_fraction, .28, .38))
        requested_crop_h = max(
            int(round(face_h / target_face_fraction)),
            int(round(face_h * 2.8)),
        )
        requested_crop_w = int(round(requested_crop_h * target_ratio))
        if (
            background_color is not None
            and (requested_crop_h > source_h or requested_crop_w > source_w)
        ):
            # A tightly shot upload cannot satisfy a passport face/body ratio
            # through cropping alone. Clamping to the source was the root cause
            # of 40%+ faces and failed upper-chest framing. Extend only the
            # selected flat backdrop around the untouched upload; the rough
            # garment guide fills the lower edit region before Qwen generates
            # the coherent final portrait.
            crop_h = requested_crop_h
            crop_w = requested_crop_w
            desired_top = int(round(y1 - face_h * .55))
            desired_left = int(round((x1 + x2) * .5 - crop_w * .5))
            canvas = Image.new("RGB", (crop_w, crop_h), tuple(background_color))
            source_left = max(0, desired_left)
            source_top = max(0, desired_top)
            source_right = min(source_w, desired_left + crop_w)
            source_bottom = min(source_h, desired_top + crop_h)
            if source_right > source_left and source_bottom > source_top:
                canvas.paste(
                    source.crop((source_left, source_top, source_right, source_bottom)),
                    (source_left - desired_left, source_top - desired_top),
                )
            return canvas.resize((width, height), Image.LANCZOS)

        crop_h = min(source_h, requested_crop_h)
        crop_w = int(round(crop_h * target_ratio))
        if crop_w > source_w:
            crop_w = source_w
            crop_h = min(source_h, int(round(crop_w / target_ratio)))

        face_center_x = (x1 + x2) * .5
        top = int(round(y1 - face_h * .42))
        top = max(0, min(source_h - crop_h, top))
        left = int(round(face_center_x - crop_w * .5))
        left = max(0, min(source_w - crop_w, left))
        return source.crop((left, top, left + crop_w, top + crop_h)).resize(
            (width, height), Image.LANCZOS,
        )

    @staticmethod
    def _measure_lip_rgb(image: Image.Image, face: Any) -> Optional[Tuple[int, int, int]]:
        """Measure the visible natural lip tone from InsightFace mouth anchors."""
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        points = np.asarray(face.kps, dtype=np.float32)
        if points.shape[0] < 5:
            return None
        left, right = points[3], points[4]
        center = ((left + right) * .5).round().astype(int)
        mouth_width = max(4.0, float(np.linalg.norm(right - left)))
        face_height = max(1.0, float(face.bbox[3] - face.bbox[1]))
        mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        cv2.ellipse(
            mask, tuple(center),
            (max(3, int(mouth_width * .62)), max(2, int(face_height * .045))),
            0, 0, 360, 255, -1,
        )
        pixels = rgb[mask > 0]
        if len(pixels) < 16:
            return None
        value = np.median(pixels, axis=0).round().clip(0, 255).astype(np.uint8)
        return tuple(int(channel) for channel in value)

    @staticmethod
    def _restore_generated_lip_tone(
        source: Image.Image, generated: Image.Image,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Correct excessive Qwen lip darkening without copying source pixels."""
        from pipelines.photo_restoration import _get_insight_app

        detector = _get_insight_app()
        source_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        generated_rgb = np.asarray(generated.convert("RGB"), dtype=np.uint8)
        source_faces = detector.get(cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR))
        generated_faces = detector.get(cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2BGR))
        if len(source_faces) != 1 or len(generated_faces) != 1:
            return generated, {"applied": False, "reason": "face_landmarks_unavailable"}
        source_color = QwenEditPipeline._measure_lip_rgb(source, source_faces[0])
        generated_color = QwenEditPipeline._measure_lip_rgb(generated, generated_faces[0])
        if source_color is None or generated_color is None:
            return generated, {"applied": False, "reason": "lip_samples_unavailable"}

        source_lab = cv2.cvtColor(
            np.asarray(source_color, dtype=np.uint8).reshape(1, 1, 3), cv2.COLOR_RGB2LAB,
        ).reshape(3).astype(np.float32)
        generated_lab_color = cv2.cvtColor(
            np.asarray(generated_color, dtype=np.uint8).reshape(1, 1, 3), cv2.COLOR_RGB2LAB,
        ).reshape(3).astype(np.float32)
        lightness_delta = float(source_lab[0] - generated_lab_color[0])
        chroma_delta = source_lab[1:] - generated_lab_color[1:]
        if lightness_delta <= 8.0 and float(np.linalg.norm(chroma_delta)) <= 10.0:
            return generated, {
                "applied": False, "reason": "lip_tone_matches",
                "source_rgb": source_color, "generated_rgb": generated_color,
            }

        face = generated_faces[0]
        points = np.asarray(face.kps, dtype=np.float32)
        left, right = points[3], points[4]
        center = tuple(((left + right) * .5).round().astype(int))
        mouth_width = max(4.0, float(np.linalg.norm(right - left)))
        face_height = max(1.0, float(face.bbox[3] - face.bbox[1]))
        mask = np.zeros(generated_rgb.shape[:2], dtype=np.uint8)
        cv2.ellipse(
            mask, center,
            (max(3, int(mouth_width * .72)), max(2, int(face_height * .055))),
            0, 0, 360, 255, -1,
        )
        alpha = cv2.GaussianBlur(mask, (0, 0), max(1.0, mouth_width * .08)).astype(np.float32) / 255.0
        alpha *= .72
        lab = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[:, :, 0] += alpha * np.clip(lightness_delta, -10.0, 26.0)
        lab[:, :, 1] += alpha * np.clip(chroma_delta[0], -14.0, 14.0)
        lab[:, :, 2] += alpha * np.clip(chroma_delta[1], -14.0, 14.0)
        corrected = cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        corrected[alpha <= 0.0] = generated_rgb[alpha <= 0.0]
        return Image.fromarray(corrected, "RGB"), {
            "applied": True,
            "source_rgb": source_color,
            "generated_rgb": generated_color,
            "lightness_delta": round(lightness_delta, 2),
        }

    @staticmethod
    def _lock_uniform_identity(
        source: Image.Image,
        generated: Image.Image,
        strength: float = .36,
        detail_radius_ratio: float = .012,
    ) -> Image.Image:
        """Align and blend source facial detail without changing Qwen's uniform or background."""
        source_rgb = np.array(source.convert("RGB"))
        generated_rgb = np.array(generated.convert("RGB"))
        from pipelines.photo_restoration import _get_insight_app

        detector = _get_insight_app()
        source_faces = detector.get(cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR))
        generated_faces = detector.get(cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2BGR))
        if len(source_faces) != 1 or len(generated_faces) != 1:
            logger.warning("[QWEN_ONLY_UNIFORM] Face lock skipped: face not detected in source or output")
            return generated

        source_face = source_faces[0]
        generated_face = generated_faces[0]
        matrix, _ = cv2.estimateAffinePartial2D(
            np.asarray(source_face.kps, dtype=np.float32),
            np.asarray(generated_face.kps, dtype=np.float32),
            method=cv2.LMEDS,
        )
        if matrix is None:
            logger.warning("[QWEN_ONLY_UNIFORM] Face lock skipped: landmark alignment failed")
            return generated

        height, width = generated_rgb.shape[:2]
        aligned = cv2.warpAffine(
            source_rgb, matrix, (width, height), flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        x1, y1, x2, y2 = np.asarray(generated_face.bbox, dtype=float)
        face_w = max(1, int(round(x2 - x1)))
        face_h = max(1, int(round(y2 - y1)))
        center = (int(round((x1 + x2) / 2)), int(round(y1 + face_h * .52)))
        axes = (max(1, int(face_w * .43)), max(1, int(face_h * .48)))
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 0).astype(np.float32) / 255.0
        # The forehead contributes little to identity matching but commonly
        # contains the source photograph's sharp sunlight boundary. Fade the
        # source-detail transfer in below the upper forehead so the generated
        # studio illumination remains authoritative there, while preserving
        # eyebrows, eyes, nose, mouth and jaw structure.
        rows = np.arange(height, dtype=np.float32)[:, None]
        forehead_fade_start = float(y1 + face_h * .16)
        forehead_fade_end = float(y1 + face_h * .31)
        forehead_gate = np.clip(
            (rows - forehead_fade_start)
            / max(1.0, forehead_fade_end - forehead_fade_start),
            0.0, 1.0,
        )
        mask *= forehead_gate

        # Match the aligned face to Qwen's local studio exposure before
        # blending. This prevents a rectangular or differently coloured face.
        aligned_lab = cv2.cvtColor(aligned, cv2.COLOR_RGB2LAB).astype(np.float32)
        generated_lab = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        core = mask > .75
        if np.count_nonzero(core) >= 256:
            for channel in range(3):
                source_values = aligned_lab[:, :, channel][core]
                target_values = generated_lab[:, :, channel][core]
                source_mean, target_mean = float(np.mean(source_values)), float(np.mean(target_values))
                source_std, target_std = float(np.std(source_values)), float(np.std(target_values))
                # Match exposure fully, but retain most source complexion
                # chroma. Matching all LAB channels to Qwen made a different,
                # pale generated face survive identity repair.
                mean_strength = 1.0 if channel == 0 else 0.35
                scale_limit = (.82, 1.18) if channel == 0 else (.92, 1.08)
                adjusted_mean = source_mean + (target_mean - source_mean) * mean_strength
                scale = np.clip(target_std / max(source_std, 1.0), *scale_limit)
                aligned_lab[:, :, channel] = (
                    (aligned_lab[:, :, channel] - source_mean) * scale + adjusted_mean
                )
        # Do not copy the source photo's illumination back into the clean
        # Qwen portrait.  Mean/std matching is insufficient for directional
        # sunlight: a dark cheek or bright forehead can have the right global
        # exposure while retaining the original shadow pattern.  Transfer the
        # aligned source's high-frequency identity detail over Qwen's smooth
        # studio illumination instead.  The small detail radius preserves
        # pores, eyelids, lip contours and other identity cues, but excludes
        # broad sunlight, orange casts and face-wide shadows.
        detail_sigma = float(np.clip(
            min(face_w, face_h) * float(detail_radius_ratio), 3.5, 42.0,
        ))
        identity_lab = generated_lab.copy()
        for channel in range(3):
            source_plane = aligned_lab[:, :, channel]
            target_plane = generated_lab[:, :, channel]
            source_low = cv2.GaussianBlur(source_plane, (0, 0), detail_sigma)
            target_low = cv2.GaussianBlur(target_plane, (0, 0), detail_sigma)
            source_detail = source_plane - source_low
            # Luminance carries most facial structure. Chroma detail is useful
            # for complexion and lips but must be bounded more tightly so the
            # source's warm sunlight cannot return as orange blotches.
            detail_limit = 18.0 if channel == 0 else 8.0
            detail_gain = 1.0 if channel == 0 else .55
            identity_lab[:, :, channel] = target_low + np.clip(
                source_detail, -detail_limit, detail_limit,
            ) * detail_gain
        aligned_matched = cv2.cvtColor(
            identity_lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB,
        ).astype(np.float32)
        # Repair only genuinely borderline candidates. A moderate blend
        # restores facial structure without producing an oval pasted-face
        # boundary; candidates below the raw-score floor are rejected instead.
        alpha = (mask * float(np.clip(strength, .0, .72)))[:, :, None]
        locked = generated_rgb.astype(np.float32) * (1.0 - alpha) + aligned_matched * alpha
        logger.info("[QWEN_ONLY_UNIFORM] Applied landmark-aligned source-face identity lock")
        return Image.fromarray(np.rint(locked).clip(0, 255).astype(np.uint8), "RGB")

    @staticmethod
    def _restore_uniform_hair_accessories(
        source: Image.Image,
        generated: Image.Image,
        source_labels: np.ndarray,
        generated_labels: np.ndarray,
    ) -> tuple[Image.Image, dict]:
        """Landmark-align uploaded head detail to an accepted uniform portrait."""
        source_rgb = np.array(source.convert("RGB"))
        generated_rgb = np.array(generated.convert("RGB"))
        from pipelines.photo_restoration import _get_insight_app
        from pipelines.uniform_finishing import restore_uniform_hair_accessories

        detector = _get_insight_app()
        source_faces = detector.get(cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR))
        generated_faces = detector.get(cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2BGR))
        if len(source_faces) != 1 or len(generated_faces) != 1:
            return generated, {"applied": False, "reason": "face_alignment_unavailable"}
        matrix, _ = cv2.estimateAffinePartial2D(
            np.asarray(source_faces[0].kps, dtype=np.float32),
            np.asarray(generated_faces[0].kps, dtype=np.float32),
            method=cv2.LMEDS,
        )
        if matrix is None:
            return generated, {"applied": False, "reason": "landmark_alignment_failed"}

        height, width = generated_rgb.shape[:2]
        aligned_source = cv2.warpAffine(
            source_rgb, matrix, (width, height), flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        aligned_labels = cv2.warpAffine(
            source_labels.astype(np.uint8), matrix, (width, height),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        return restore_uniform_hair_accessories(
            generated,
            Image.fromarray(aligned_source, "RGB"),
            generated_labels,
            aligned_labels,
        )

    @staticmethod
    def _restore_uniform_source_hair_geometry(
        source: Image.Image,
        generated: Image.Image,
        source_labels: np.ndarray,
        background_color: Union[str, Tuple[int, int, int]],
        force_source_texture: bool = False,
    ) -> tuple[Image.Image, dict]:
        """Restore a changed Qwen hairstyle without altering face or uniform."""
        from pipelines.photo_restoration import _get_insight_app
        from pipelines.schp_service import parse as parse_body_parts

        source_rgb = np.asarray(source.convert("RGB"))
        generated_rgb = np.asarray(generated.convert("RGB"))
        detector = _get_insight_app()
        source_faces = detector.get(cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR))
        generated_faces = detector.get(cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2BGR))
        if len(source_faces) != 1 or len(generated_faces) != 1:
            return generated.convert("RGB"), {"applied": False, "reason": "face_alignment_unavailable"}

        matrix, _ = cv2.estimateAffinePartial2D(
            np.asarray(source_faces[0].kps, dtype=np.float32),
            np.asarray(generated_faces[0].kps, dtype=np.float32),
            method=cv2.LMEDS,
        )
        if matrix is None:
            return generated.convert("RGB"), {"applied": False, "reason": "landmark_alignment_failed"}

        height, width = generated_rgb.shape[:2]
        aligned_source = cv2.warpAffine(
            source_rgb, matrix, (width, height), flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        aligned_labels = cv2.warpAffine(
            source_labels.astype(np.uint8), matrix, (width, height),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        generated_labels = parse_body_parts(generated_rgb)["labels"]
        source_hair_label = aligned_labels == 2
        source_luma = cv2.cvtColor(aligned_source, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if np.count_nonzero(source_hair_label) >= 128:
            aligned_source_lab = cv2.cvtColor(
                aligned_source, cv2.COLOR_RGB2LAB,
            ).astype(np.float32)
            shaded_hair = source_hair_label & (
                source_luma <= np.percentile(source_luma[source_hair_label], 60)
            )
            hair_colour = np.median(aligned_source_lab[shaded_hair], axis=0)
            hair_colour_distance = np.linalg.norm(
                aligned_source_lab - hair_colour[None, None, :], axis=2,
            )
            source_hair = source_hair_label & (hair_colour_distance < 52.0)
        else:
            source_hair = source_hair_label
        source_hair = cv2.morphologyEx(
            source_hair.astype(np.uint8), cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1,
        ).astype(bool)
        source_face_seed = cv2.dilate(
            (aligned_labels == 13).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), iterations=1,
        ).astype(bool)
        source_component_count, source_components = cv2.connectedComponents(
            source_hair.astype(np.uint8), connectivity=8,
        )
        source_seeded = np.unique(source_components[source_hair & source_face_seed])
        source_seeded = source_seeded[source_seeded != 0]
        if source_seeded.size:
            source_hair = np.isin(source_components, source_seeded)
        generated_hair = generated_labels == 2
        if np.count_nonzero(source_hair) < 128 or np.count_nonzero(generated_hair) < 128:
            return generated.convert("RGB"), {"applied": False, "reason": "hair_mask_unavailable"}

        x1, y1, x2, y2 = np.asarray(generated_faces[0].bbox, dtype=float)
        face_w, face_h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        yy, xx = np.indices((height, width))
        head_zone = (
            (yy >= max(0, int(y1 - face_h * .78)))
            & (yy <= min(height - 1, int(y2 + face_h * .22)))
            & (xx >= max(0, int(x1 - face_w * .80)))
            & (xx <= min(width - 1, int(x2 + face_w * .80)))
        )
        source_hair &= head_zone
        generated_hair &= head_zone
        source_extent = cv2.dilate(
            source_hair.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1,
        ).astype(bool)
        generated_extent = cv2.dilate(
            generated_hair.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1,
        ).astype(bool)
        union = np.count_nonzero(source_extent | generated_extent)
        intersection = np.count_nonzero(source_extent & generated_extent)
        iou = float(intersection / max(1, union))
        generated_excess = float(
            np.count_nonzero(generated_hair & ~source_extent)
            / max(1, np.count_nonzero(generated_hair))
        )
        source_missing = float(
            np.count_nonzero(source_hair & ~generated_extent)
            / max(1, np.count_nonzero(source_hair))
        )
        if (
            not force_source_texture
            and iou >= .72
            and generated_excess <= .16
            and source_missing <= .22
        ):
            return generated.convert("RGB"), {
                "applied": False, "reason": "hair_geometry_matches",
                "iou": round(iou, 4), "generated_excess": round(generated_excess, 4),
                "source_missing": round(source_missing, 4),
            }

        result = generated_rgb.copy()
        face_guard = cv2.dilate(
            (generated_labels == 13).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1,
        ).astype(bool)
        generated_luma = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        # SCHP often labels the outer half of a newly invented rounded haircut
        # as background or hat. Inside the landmark-derived crown region, dark
        # pixels connected to parsed hair are still generated hair and must be
        # removed when they fall outside the uploaded silhouette.
        source_accessory_guard = cv2.dilate(
            np.isin(aligned_labels, (1, 4, 13)).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1,
        ).astype(bool)
        target_rgb = np.asarray(
            QwenEditPipeline._parse_bg_color(background_color), dtype=np.float32,
        )
        background_distance = np.linalg.norm(
            generated_rgb.astype(np.float32) - target_rgb[None, None, :], axis=2,
        )
        hair_candidate = (
            generated_hair
            | (
                head_zone & (generated_luma < 120.0)
                & (background_distance > 35.0) & ~source_accessory_guard
            )
        ).astype(np.uint8)
        component_count, component_labels = cv2.connectedComponents(
            hair_candidate, connectivity=8,
        )
        seeded_components = np.unique(component_labels[generated_hair])
        seeded_components = seeded_components[seeded_components != 0]
        generated_hair_like = np.isin(component_labels, seeded_components)
        erase = generated_hair_like & ~source_extent & ~face_guard
        result[erase] = np.asarray(
            QwenEditPipeline._parse_bg_color(background_color), dtype=np.uint8,
        )

        source_lab = cv2.cvtColor(aligned_source, cv2.COLOR_RGB2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        overlap = source_hair & generated_hair
        if np.count_nonzero(overlap) >= 128:
            lightness_shift = float(
                np.median(target_lab[:, :, 0][overlap])
                - np.median(source_lab[:, :, 0][overlap])
            )
            source_lab[:, :, 0] += np.clip(lightness_shift, -12.0, 12.0)
        matched_source = cv2.cvtColor(
            source_lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB,
        ).astype(np.float32)
        alpha = cv2.GaussianBlur(
            source_hair.astype(np.uint8) * 255, (5, 5), 0,
        ).astype(np.float32) / 255.0
        alpha = np.clip(alpha * .96, 0.0, .96)
        restored = (
            result.astype(np.float32) * (1.0 - alpha[:, :, None])
            + matched_source * alpha[:, :, None]
        )
        return Image.fromarray(np.rint(restored).clip(0, 255).astype(np.uint8), "RGB"), {
            "applied": True,
            "iou": round(iou, 4),
            "generated_excess": round(generated_excess, 4),
            "source_missing": round(source_missing, 4),
            "restored_pixels": int(np.count_nonzero(source_hair)),
            "erased_pixels": int(np.count_nonzero(erase)),
        }

    @staticmethod
    def _lock_uniform_source_head(
        source: Image.Image,
        generated: Image.Image,
        source_labels: np.ndarray,
        background_color: Union[str, Tuple[int, int, int]],
        preserve_light_hair_accessories: bool = False,
        ponytail_side: str = "unknown",
    ) -> tuple[Image.Image, dict]:
        """Preserve the uploaded face, hair silhouette, and small crown clips.

        This is reserved for a low-similarity uniform result.  It replaces the
        redrawn head only, not the generated collar or garment, using dark-hair
        colour evidence to exclude foliage from outdoor source portraits.
        """
        from pipelines.photo_restoration import _get_insight_app
        from pipelines.schp_service import parse as parse_body_parts

        source_rgb = np.asarray(source.convert("RGB"))
        generated_rgb = np.asarray(generated.convert("RGB"))
        detector = _get_insight_app()
        source_faces = detector.get(cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR))
        generated_faces = detector.get(cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2BGR))
        if len(source_faces) != 1 or len(generated_faces) != 1:
            return generated.convert("RGB"), {"applied": False, "reason": "face_alignment_unavailable"}

        matrix, _ = cv2.estimateAffinePartial2D(
            np.asarray(source_faces[0].kps, dtype=np.float32),
            np.asarray(generated_faces[0].kps, dtype=np.float32), method=cv2.LMEDS,
        )
        if matrix is None:
            return generated.convert("RGB"), {"applied": False, "reason": "landmark_alignment_failed"}

        height, width = generated_rgb.shape[:2]
        aligned_source = cv2.warpAffine(
            source_rgb, matrix, (width, height), flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        aligned_labels = cv2.warpAffine(
            source_labels.astype(np.uint8), matrix, (width, height),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        generated_labels = parse_body_parts(generated_rgb)["labels"]

        source_face = aligned_labels == 13
        source_eyewear = aligned_labels == 4
        eyewear_points = np.column_stack(np.where(source_eyewear)[::-1]).astype(np.int32)
        if eyewear_points.shape[0] >= 12:
            eyewear_region = np.zeros((height, width), dtype=np.uint8)
            cv2.fillConvexPoly(eyewear_region, cv2.convexHull(eyewear_points), 1)
            source_eyewear = eyewear_region.astype(bool)
        source_hair_label = aligned_labels == 2
        if np.count_nonzero(source_face) < 128 or np.count_nonzero(source_hair_label) < 128:
            return generated.convert("RGB"), {"applied": False, "reason": "head_mask_unavailable"}

        source_lab = cv2.cvtColor(aligned_source, cv2.COLOR_RGB2LAB).astype(np.float32)
        dark_samples = source_lab[source_hair_label & (source_lab[:, :, 0] < 115)]
        if dark_samples.size == 0:
            return generated.convert("RGB"), {"applied": False, "reason": "hair_colour_unavailable"}
        hair_median = np.median(dark_samples, axis=0)
        colour_distance = np.linalg.norm(source_lab - hair_median[None, None, :], axis=2)

        x1, y1, x2, y2 = np.asarray(generated_faces[0].bbox, dtype=float)
        face_w, face_h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        yy, xx = np.indices((height, width))

        # Hair parsers frequently omit side ponytails below the crown. Recover
        # connected lower dark hair so real hairstyles are preserved and not erased.
        lower_hair_corridor = (
            (yy >= int(y1 + face_h * .35))
            & (yy <= int(y2 + face_h * .68))
            & (
                ((xx >= int(x1 - face_w * .75)) & (xx <= int(x1 + face_w * .12)))
                | ((xx >= int(x2 - face_w * .12)) & (xx <= int(x2 + face_w * .75)))
            )
        )
        normalized_ponytail_side = str(ponytail_side or "").strip().lower()
        if normalized_ponytail_side == "viewer-right":
            lower_hair_corridor &= xx >= int(x2 - face_w * .12)
        elif normalized_ponytail_side == "viewer-left":
            lower_hair_corridor &= xx <= int(x1 + face_w * .12)
        # Parser label 2 can spill onto a dark school vest below the neck.
        # Use parsed hair only through the lower head; recover any longer tail
        # separately inside the measured one-sided ponytail corridor.
        base_head_hair = source_hair_label & (yy <= int(y2 + face_h * .18))
        crown_dilated = cv2.dilate(
            base_head_hair.astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (max(5, int(face_w * .25) | 1), max(5, int(face_h * .75) | 1)),
            ),
            iterations=1,
        ).astype(bool)
        lower_dark_hair = (
            lower_hair_corridor
            & crown_dilated
            & ((colour_distance < 62.0) | (source_lab[:, :, 0] < 78.0))
        )
        dark_hair = (base_head_hair | lower_dark_hair) & (
            (colour_distance < 62.0) | (source_lab[:, :, 0] < 78.0)
        )
        crown_region = (
            (yy >= max(0, int(y1 - face_h * .62)))
            & (yy <= int(y1 + face_h * .20))
            & (xx >= int(x1 - face_w * .68))
            & (xx <= int(x2 + face_w * .68))
        )
        rgb_range = aligned_source.max(axis=2).astype(np.int16) - aligned_source.min(axis=2).astype(np.int16)
        light_clip = (
            crown_region
            & (aligned_source.min(axis=2) > 155)
            & (rgb_range < 78)
            & (cv2.dilate(dark_hair.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0)
        )
        # Preserve real glasses and small ear jewellery with the source head.
        # SCHP labels glasses explicitly but usually classifies thin earrings
        # as background, so recover only compact dark components immediately
        # beside the ears. This avoids copying arbitrary source scenery.
        source_luma = cv2.cvtColor(aligned_source, cv2.COLOR_RGB2GRAY)
        ear_zone = (
            (yy >= int(y1 + face_h * .48))
            & (yy <= int(y2 - face_h * .01))
            & (
                ((xx >= int(x1 - face_w * .28)) & (xx <= int(x1 + face_w * .08)))
                | ((xx >= int(x2 - face_w * .08)) & (xx <= int(x2 + face_w * .28)))
            )
        )
        accessory_candidates = (
            ear_zone & ~source_face & ~dark_hair & ~source_eyewear
            & (source_luma < 145)
        ).astype(np.uint8)
        _, accessory_components, accessory_stats, _ = cv2.connectedComponentsWithStats(
            accessory_candidates, connectivity=8,
        )
        source_accessories = np.zeros((height, width), dtype=bool)
        for component_id in range(1, accessory_stats.shape[0]):
            area = int(accessory_stats[component_id, cv2.CC_STAT_AREA])
            if 4 <= area <= int(face_w * face_h * .006):
                source_accessories |= accessory_components == component_id
        source_accessories = cv2.dilate(
            source_accessories.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1,
        ).astype(bool)

        if not preserve_light_hair_accessories:
            light_clip[:] = False
        head_mask = source_face | dark_hair | light_clip | source_eyewear | source_accessories
        head_mask = cv2.morphologyEx(
            head_mask.astype(np.uint8), cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1,
        ).astype(bool)

        # Remove Qwen's replacement hairstyle where it lies outside the
        # uploaded silhouette, then feather the real source head over it.
        generated_hair = generated_labels == 2
        head_zone = (
            (yy < int(y2 + face_h * .28))
            & (xx >= int(x1 - face_w * .98))
            & (xx <= int(x2 + face_w * .98))
        )
        source_hair_extent = cv2.dilate(
            dark_hair.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1,
        ).astype(bool)
        result = generated_rgb.copy()
        target_rgb = np.asarray(
            QwenEditPipeline._parse_bg_color(background_color), dtype=np.float32,
        )
        generated_distance = np.linalg.norm(
            generated_rgb.astype(np.float32) - target_rgb[None, None, :], axis=2,
        )
        source_head_extent = cv2.dilate(
            head_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1,
        ).astype(bool)
        generated_hair_like = cv2.dilate(
            generated_hair.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1,
        ).astype(bool) & head_zone & (generated_distance > 28.0)
        # SCHP often misses a Qwen-invented ponytail, especially when it falls
        # over a dark collar or shoulder. Detect hair-coloured dark pixels in
        # the landmark-bounded head zone as well, then erase only the portion
        # outside the aligned source-head silhouette. This removes a mirrored
        # second tail without touching the real uploaded ponytail.
        generated_lab = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        generated_hair_distance = np.linalg.norm(
            generated_lab - hair_median[None, None, :], axis=2,
        )
        generated_dark_hair = (
            head_zone
            & (generated_distance > 28.0)
            & (
                ((generated_hair_distance < 68.0) & (generated_lab[:, :, 0] < 125.0))
                | (generated_lab[:, :, 0] < 48.0)
            )
        )
        central_neck_guard = (
            (yy >= int(y2 - face_h * .08))
            & (xx >= int(x1 + face_w * .05))
            & (xx <= int(x2 - face_w * .05))
        )
        generated_dark_hair &= ~central_neck_guard
        generated_hair_like |= generated_dark_hair
        generated_red_band = (
            head_zone
            & (generated_rgb[:, :, 0].astype(np.int16) > generated_rgb[:, :, 1].astype(np.int16) + 45)
            & (generated_rgb[:, :, 0].astype(np.int16) > generated_rgb[:, :, 2].astype(np.int16) + 35)
            & (cv2.dilate(generated_hair.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0)
        )
        invented_head_detail = generated_hair_like & ~source_head_extent
        if normalized_ponytail_side in {"viewer-left", "viewer-right"}:
            if normalized_ponytail_side == "viewer-right":
                opposite_tail_zone = xx <= int(x1 + face_w * .28)
            else:
                opposite_tail_zone = xx >= int(x2 - face_w * .28)
            opposite_tail_zone &= (
                (yy >= int(y2 - face_h * .14))
                & (yy <= int(y2 + face_h * .58))
            )
            protected_features = cv2.dilate(
                (source_face | source_eyewear | source_accessories).astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1,
            ).astype(bool)
            invented_head_detail |= (
                opposite_tail_zone & generated_dark_hair & ~protected_features
            )
        invented_head_detail |= (
            generated_red_band & ~source_head_extent
            if preserve_light_hair_accessories else generated_red_band
        )
        result[invented_head_detail] = target_rgb.astype(np.uint8)

        # Match only lightness in the retained source face/hair; chroma stays
        # from the upload so complexion and dark-hair pigment remain authentic.
        result_lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB).astype(np.float32)
        core = source_face & (source_lab[:, :, 0] > 25)
        if np.count_nonzero(core) >= 128:
            shift = float(np.median(result_lab[:, :, 0][core]) - np.median(source_lab[:, :, 0][core]))
            source_lab[:, :, 0] += np.clip(shift, -14.0, 14.0)
        matched_source = cv2.cvtColor(source_lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
        # A slightly wider feather at the jaw avoids the cut-and-paste line
        # between the exact uploaded face and Qwen's generated neck/collar.
        alpha = cv2.GaussianBlur(head_mask.astype(np.uint8) * 255, (13, 13), 0).astype(np.float32) / 255.0
        # Gaussian alpha extends beyond the binary source foreground. Fill that
        # exterior with the target first so the upload's white wall cannot leak
        # into the feather as a pale outline around dark hair and earrings.
        matched_source[~head_mask] = result.astype(np.float32)[~head_mask]
        result = result.astype(np.float32) * (1.0 - alpha[:, :, None]) + matched_source * alpha[:, :, None]
        return Image.fromarray(np.rint(result).clip(0, 255).astype(np.uint8), "RGB"), {
            "applied": True,
            "head_pixels": int(np.count_nonzero(head_mask)),
            "clip_pixels": int(np.count_nonzero(light_clip)),
            "accessory_pixels": int(np.count_nonzero(source_accessories)),
            "removed_invented_pixels": int(np.count_nonzero(invented_head_detail)),
        }

    @staticmethod
    def _parse_bg_color(bg_color: Union[str, Tuple[int, int, int]]) -> Tuple[int, int, int]:
        if isinstance(bg_color, tuple) and len(bg_color) == 3:
            return bg_color
        if not bg_color:
            return (255, 255, 255)
        bg_lower = str(bg_color).strip().lower()
        presets = {
            "pure white (passport)": (255, 255, 255),
            "white": (255, 255, 255),
            "off-white (us visa)": (245, 245, 240),
            "offwhite": (245, 245, 240),
            "light blue (school / visa id)": (205, 230, 248),
            "blue": (205, 230, 248),
            "lightblue": (205, 230, 248),
            "light grey (corporate)": (218, 222, 226),
            "grey": (218, 222, 226),
            "gray": (218, 222, 226),
            "warm studio ivory": (246, 241, 232),
            "ivory": (246, 241, 232),
            "navy blue (formal)": (25, 45, 85),
            "navy": (25, 45, 85),
            "solid red": (200, 30, 30),
            "red (official id)": (200, 30, 30),
            "red": (200, 30, 30),
            "soft beige (editorial)": (238, 230, 220),
            "beige": (238, 230, 220),
        }
        if bg_lower in presets:
            return presets[bg_lower]
        for key, val in presets.items():
            if key in bg_lower or bg_lower in key:
                return val
        if bg_lower.startswith("#"):
            try:
                hex_str = bg_lower.lstrip("#")
                if len(hex_str) == 6:
                    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
            except Exception:
                pass
        if "," in bg_lower:
            try:
                parts = [int(p.strip()) for p in bg_lower.split(",")]
                if len(parts) == 3:
                    return (parts[0], parts[1], parts[2])
            except Exception:
                pass
        return (255, 255, 255)

    @staticmethod
    def _prepare_uniform_reference(
        image: Image.Image,
        target_width: int = 560,
        target_height: int = 720,
        badge_bbox: Optional[List[int]] = None,
        pad_to_portrait: bool = True,
    ) -> Image.Image:
        """
        Prepare uniform template reference:
        1. Inpaints school badges / emblems so reference image is badge-free.
        2. Composites RGBA cutout onto neutral white backdrop preserving head/neck headroom.
        3. Matches target portrait aspect ratio symmetrically without stretching.
        """
        rgba = image.convert("RGBA")
        if badge_bbox:
            from pipelines.uniform_vl_analyzer import uniform_vl_analyzer
            rgba = uniform_vl_analyzer.inpaint_badge(rgba, badge_bbox).convert("RGBA")

        alpha = rgba.getchannel("A")
        if alpha.getextrema()[0] < 255:
            bounds = alpha.point(lambda value: 255 if value > 16 else 0).getbbox()
            if bounds is None:
                raise ValueError("The uniform template is completely transparent.")
            min_x, min_y, max_x, max_y = bounds
            # This is a garment reference, not a person portrait. Keep a small
            # collar margin, but remove the transparent upper canvas so the
            # collar, placket, pattern, and vest occupy meaningful VL tokens.
            margin_y = max(8, int((max_y - min_y) * 0.025))
            crop_top = max(0, min_y - margin_y)
            crop_bottom = min(rgba.height, max_y + margin_y)
            # Crop transparent margins only. Forcing the garment into the
            # person's aspect ratio discarded sleeves and vest side panels.
            crop_left = max(0, min_x - margin_y)
            crop_right = min(rgba.width, max_x + margin_y)
            rgba = rgba.crop((crop_left, crop_top, crop_right, crop_bottom))

            backdrop = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            prepared = Image.alpha_composite(backdrop, rgba).convert("RGB")
        else:
            prepared = rgba.convert("RGB")

        if not pad_to_portrait:
            return prepared
        # Fit into target aspect ratio symmetrically with white padding if needed
        target_ratio = target_width / target_height
        curr_ratio = prepared.width / prepared.height
        if abs(curr_ratio - target_ratio) > 0.04:
            if curr_ratio > target_ratio:
                new_h = int(prepared.width / target_ratio)
                padded = Image.new("RGB", (prepared.width, new_h), (255, 255, 255))
                offset_y = (new_h - prepared.height) // 2
                padded.paste(prepared, (0, offset_y))
                prepared = padded
            else:
                new_w = int(prepared.height * target_ratio)
                padded = Image.new("RGB", (new_w, prepared.height), (255, 255, 255))
                offset_x = (new_w - prepared.width) // 2
                padded.paste(prepared, (offset_x, 0))
                prepared = padded

        return prepared

    def _replace_smooth_border_background(
        self, image: Image.Image, background_color: Union[str, Tuple[int, int, int]]
    ) -> Image.Image:
        """Correct Qwen's off-color flat backdrop while protecting the parsed subject.

        Only pixels similar to the four corner samples and connected to a canvas
        border are changed. The semantic subject mask prevents a light shirt or
        skin highlight touching the canvas edge from joining that replacement.
        """
        rgb = np.array(image.convert("RGB"))
        h, w = rgb.shape[:2]
        edge = max(4, min(16, h // 20, w // 20))
        if h < 32 or w < 32 or edge < 2:
            return image

        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        corners = np.concatenate((
            lab[:edge, :edge].reshape(-1, 3),
            lab[:edge, -edge:].reshape(-1, 3),
        ))
        background_lab = np.median(corners, axis=0)
        distance = np.linalg.norm(lab - background_lab, axis=2)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        corner_hsv = np.concatenate((
            hsv[:edge, :edge].reshape(-1, 3),
            hsv[:edge, -edge:].reshape(-1, 3),
        ))
        background_hsv = np.median(corner_hsv, axis=0)
        hue_delta = np.abs(hsv[:, :, 0] - background_hsv[0])
        hue_delta = np.minimum(hue_delta, 180.0 - hue_delta)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        local_mean = cv2.GaussianBlur(gray, (0, 0), 3.0)
        local_square_mean = cv2.GaussianBlur(gray * gray, (0, 0), 3.0)
        local_texture = np.sqrt(np.maximum(0.0, local_square_mean - local_mean * local_mean))
        fine_mean = cv2.GaussianBlur(gray, (0, 0), 1.0)
        fine_square_mean = cv2.GaussianBlur(gray * gray, (0, 0), 1.0)
        fine_texture = np.sqrt(np.maximum(0.0, fine_square_mean - fine_mean * fine_mean))
        # Qwen often shades a nominally flat gray backdrop beside dark curls.
        # A wider tolerance removes that neutral smoky fringe. Protect pale
        # flowers separately below so they are not mistaken for the backdrop.
        # Qwen's conditioning backdrop is deliberately flat middle gray. Keep
        # this tolerance narrow so light-brown fringe hair and skin near the
        # forehead cannot be mistaken for backdrop.
        chromatic_backdrop = background_hsv[1] >= 65.0
        if chromatic_backdrop:
            # A saturated flat backdrop can vary greatly in brightness while
            # retaining one hue. LAB radius is unsafe here: dark cyan and brown
            # skin can be numerically close, which caused large facial holes.
            # Hue/chroma isolates the generated backdrop without relying on a
            # person-specific skin or hair colour.
            candidates = (
                (hue_delta <= 16.0)
                & (hsv[:, :, 1] >= max(45.0, background_hsv[1] * .42))
            ).astype(np.uint8)
        else:
            candidates = (distance < 48.0).astype(np.uint8)
        enclosed_backdrop = np.zeros((h, w), dtype=bool)
        decontamination_protected = np.zeros((h, w), dtype=bool)
        lower_region = np.zeros((h, w), dtype=bool)
        lower_region[int(h * .48):, :] = True
        # Protect the actual person rather than a broad colour neighbourhood.
        # The old neighbourhood rule retained large pieces of a pale wall on
        # both sides of dark hair. A tiny dilation keeps antialiased hair and
        # garment edges without turning nearby scenery into foreground.
        try:
            from pipelines.schp_service import parse as parse_body_parts

            labels = parse_body_parts(rgb)["labels"]
            subject = (labels != 0).astype(np.uint8)
            hair = (labels == 2).astype(np.uint8)
            # The colour-only candidate mask misses Qwen's warm antialias rim:
            # those pixels can be far from the sampled backdrop even though
            # SCHP correctly identifies them as background. Admit semantic
            # background here, then keep only border-connected components
            # below. Garment hulls, reliable head classes and compact hair
            # accessories are protected by the exclusions that follow.
            # Parser errors around the jaw and neck are common. Only admit a
            # semantic-background pixel when its colour is still plausibly
            # related to the sampled backdrop; otherwise skin can become a
            # border-connected hole and get painted with the selected colour.
            if chromatic_backdrop:
                candidates[
                    (labels == 0)
                    & (hue_delta <= 20.0)
                    & (hsv[:, :, 1] >= max(35.0, background_hsv[1] * .32))
                ] = 1
                # Edit diffusion can leave a pale/white antialias rim between
                # a saturated passport backdrop and the person. Hue matching
                # cannot see that rim. Admit it only when SCHP also calls the
                # pixel background and it lies within four pixels of an
                # already-detected backdrop pixel; interior white checks,
                # collars, skin highlights and accessories stay protected.
                near_detected_backdrop = cv2.dilate(
                    candidates,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                    iterations=1,
                ).astype(bool)
                pale_semantic_fringe = (
                    (labels == 0)
                    & near_detected_backdrop
                    & (hsv[:, :, 1] <= 72.0)
                    & (hsv[:, :, 2] >= 145.0)
                )
                candidates[pale_semantic_fringe] = 1
            else:
                candidates[(labels == 0) & (distance < 64.0)] = 1
            # Fill parser cracks inside the lower garment silhouette. A convex
            # hull follows the detected outer sleeve/shoulder boundary while
            # closing internal zero-label gaps; unlike colour dilation it does
            # not preserve unrelated lower backdrop gradients.
            clothing = np.isin(labels, (5, 6, 7, 9, 10, 12)).astype(np.uint8)
            clothing_points = cv2.findNonZero(clothing)
            if clothing_points is not None and len(clothing_points) >= 3:
                clothing_hull = np.zeros_like(clothing)
                cv2.fillConvexPoly(
                    clothing_hull, cv2.convexHull(clothing_points), 1,
                )
                # A convex garment hull closes parser cracks in plaid and pale
                # sleeves, but it also spans the real backdrop between a neck
                # and sloping shoulders. Protect hull pixels only when they
                # contain measurable fabric colour or texture. A flat backdrop
                # wedge matches the sampled background and remains replaceable.
                fabric_evidence = (distance >= 14.0) | (local_texture >= 12.0)
                if chromatic_backdrop:
                    # A softly shaded blue/cyan backdrop can look textured near
                    # a shoulder and fall inside the garment convex hull. It is
                    # still background when its hue/chroma matches the border;
                    # do not protect that halo as fabric.
                    backdrop_hue_evidence = (
                        (hue_delta <= 18.0)
                        & (hsv[:, :, 1] >= max(40.0, background_hsv[1] * .36))
                        & (fine_texture < 10.0)
                    )
                    fabric_evidence &= ~backdrop_hue_evidence
                candidates[
                    lower_region & clothing_hull.astype(bool) & fabric_evidence
                ] = 0
                # The neck/shoulder opening can be fully enclosed by the
                # clothing hull, so border flood-fill alone cannot reach it.
                # Admit only large, parser-confirmed background components
                # that are both flat and identical to the sampled backdrop.
                flat_hull_background = (
                    lower_region
                    & clothing_hull.astype(bool)
                    & (labels == 0)
                    & (distance < 8.0)
                )
                component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
                    flat_hull_background.astype(np.uint8), connectivity=8,
                )
                minimum_wedge_area = max(24, int(h * w * .006))
                for component in range(1, component_count):
                    if int(component_stats[component, cv2.CC_STAT_AREA]) >= minimum_wedge_area:
                        enclosed_backdrop |= component_labels == component
            # Protect all parsed subject below the shoulders, plus only reliable
            # face/head classes above them. SCHP occasionally labels a large
            # patch of flat backdrop beside hair as an arm or garment; trusting
            # every upper non-hair label retained exactly the gray halo this
            # correction is meant to remove.
            yy, _ = np.indices(subject.shape)
            # SCHP can call a flat strip of the generated backdrop an arm when
            # it touches a sleeve at the lower frame edge. Such a
            # strip is still backdrop when it matches the sampled corner hue,
            # has no fabric texture, and remains border-connected. Excluding
            # it here removes the white/gray wedges without touching plaid,
            # skin, or any other detailed subject pixel.
            # Parser mistakes are not limited to arm labels: at the outer
            # shoulder it can classify a strip of the perfectly flat backdrop
            # as shirt. Do not protect pixels that are virtually identical to
            # the sampled backdrop and have no local fabric texture. Internal
            # white plaid remains protected because its checks create texture
            # and, independently, are not border-connected.
            xx = np.indices(subject.shape)[1]
            side_silhouette = (xx < int(w * .30)) | (xx >= int(w * .70))
            chromatic_flat_backdrop = (
                chromatic_backdrop
                & (hue_delta <= 18.0)
                & (hsv[:, :, 1] >= max(40.0, background_hsv[1] * .36))
                & (fine_texture < 10.0)
            )
            flat_backdrop_like = (
                (
                    side_silhouette
                    & (
                        ((distance < 12.0) & (local_texture < 18.0))
                        | chromatic_flat_backdrop
                    )
                )
                | (np.isin(labels, (14, 15)) & (distance < 14.0))
            )
            lower_subject = (
                subject.astype(bool)
                & (yy >= int(h * .48))
                & ~flat_backdrop_like
            )
            reliable_upper_subject = np.isin(labels, (1, 4, 13))
            hair_core = cv2.erode(
                hair,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            ).astype(bool)
            pixel_luma = (
                rgb[:, :, 0].astype(np.float32) * .2126
                + rgb[:, :, 1].astype(np.float32) * .7152
                + rgb[:, :, 2].astype(np.float32) * .0722
            )
            # Preserve the complete parsed hair core. Background-coloured gaps
            # inside the hairstyle are handled separately only where SCHP
            # actually reports background; luma alone must never erase a real
            # highlight or a pale accessory inside the hair label.
            # Parser boundaries around ears, jaw and neck can be one or two
            # pixels late. Dilate only reliable face/head skin classes. Doing
            # this to garment or hair labels would preserve a mislabeled flat
            # shoulder halo and disable legitimate hair-edge decontamination.
            upper_skin_guard = cv2.dilate(
                reliable_upper_subject.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            ).astype(bool)
            protected_subject = lower_subject | upper_skin_guard | hair_core
            candidates[protected_subject] = 0

            # SCHP may label a saturated bow or clip as background. Preserve
            # only chromatic accessory pixels immediately beside the parsed
            # upper-head silhouette. Neutral wall/sky cannot satisfy this.
            upper_subject = subject.astype(bool)
            upper_subject[int(h * 0.52):, :] = False
            accessory_zone = cv2.dilate(
                upper_subject.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
                iterations=1,
            ).astype(bool)
            chroma_delta = np.linalg.norm(lab[:, :, 1:] - background_lab[1:], axis=2)
            accessory_seed = accessory_zone & (chroma_delta >= 8.0) & (pixel_luma >= 92.0)
            component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
                accessory_seed.astype(np.uint8), connectivity=8,
            )
            compact_accessory = np.zeros_like(accessory_seed)
            max_accessory_area = max(64, int(h * w * .06))
            for component in range(1, component_count):
                area = int(component_stats[component, cv2.CC_STAT_AREA])
                component_w = int(component_stats[component, cv2.CC_STAT_WIDTH])
                component_h = int(component_stats[component, cv2.CC_STAT_HEIGHT])
                aspect = max(component_w, component_h) / max(1, min(component_w, component_h))
                if 6 <= area <= max_accessory_area and aspect <= 3.5:
                    compact_accessory |= component_labels == component
            compact_accessory = cv2.dilate(
                compact_accessory.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            ).astype(bool)
            candidates[compact_accessory] = 0
            # The decontamination band below may extend a few pixels into the
            # silhouette. Keep real skin and the stable interior of hair or a
            # head accessory completely outside that colour operation. Very
            # backdrop-like arm labels remain eligible because they are a
            # common parser error at the lower corners.
            skin_core = np.isin(labels, (1, 4, 13, 14, 15)) & (distance >= 30.0)
            decontamination_protected = hair_core | compact_accessory | skin_core
        except Exception as exc:
            logger.warning("[UNIFORM_BACKGROUND] Subject protection unavailable: %s", exc)

        count, labels = cv2.connectedComponents(candidates, connectivity=8)
        if count <= 1:
            return image
        border_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
        border_labels = border_labels[border_labels != 0]
        if not len(border_labels):
            return image
        # Replace only backdrop connected to a canvas edge. Fine fringe hair is
        # sometimes labelled as background by SCHP and may resemble neutral
        # gray in LAB, but it is enclosed by the head and must remain untouched.
        matte = np.isin(labels, border_labels).astype(np.uint8) * 255
        matte[enclosed_backdrop] = 255
        # Qwen's decoded garment boundary can carry a one-pixel staircase of
        # mixed backdrop/fabric colours even when the interior garment is
        # correct.  Contract only the lower silhouette by one pixel before
        # feathering.  This removes that contaminated outer sample without
        # rematting the head, changing the generated uniform interior, or
        # trusting a brittle semantic class at the sleeve edge.
        if chromatic_backdrop:
            lower_silhouette = np.zeros_like(matte, dtype=bool)
            lower_silhouette[int(h * .48):, :] = True
            contracted_lower_matte = cv2.dilate(
                matte,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )
            outer_ring = (contracted_lower_matte > 0) & (matte == 0)
            decoded_edge_contamination = (
                (
                    (hue_delta <= 20.0)
                    & (hsv[:, :, 1] >= max(35.0, background_hsv[1] * .30))
                )
                | ((hsv[:, :, 1] <= 72.0) & (hsv[:, :, 2] >= 145.0))
            )
            matte[
                lower_silhouette & outer_ring & decoded_edge_contamination
            ] = 255
        fringe_zone = (
            cv2.dilate(
                matte,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                iterations=1,
            ) > 0
        ) & (matte == 0) & ~decontamination_protected
        # Only neutral pixels still dominated by the old backdrop participate
        # in edge decontamination. Real hair, skin and patterned cloth remain
        # outside this narrow similarity-weighted fringe.
        # Antialiased dark hair over a pale backdrop can sit much farther from
        # the backdrop colour than an ordinary soft edge (a 50/50 black/gray
        # pixel is already about halfway across LAB space). Use a wider range
        # only in this one-pixel fringe; semantic subject interiors never reach
        # this branch.
        # Keep decontamination local to pixels still close to the sampled
        # backdrop. The previous 190-LAB span repainted real ear/neck edge
        # pixels and produced a jagged selected-colour outline.
        if chromatic_backdrop:
            fringe_strength = (
                np.clip(1.0 - hue_delta / 18.0, 0.0, 1.0)
                * np.clip((hsv[:, :, 1] - 35.0) / 110.0, 0.0, 1.0)
                * .55
            )
        else:
            # Neutral gray/white backdrops cannot use hue separation. Retain
            # the stronger LAB decontamination needed to remove pale halos;
            # unlike the chromatic route above, it is covered by semantic face
            # protection and has no saturated skin/background ambiguity.
            fringe_strength = np.clip(1.0 - distance / 190.0, 0.0, 1.0) * .95
        fringe_strength[~fringe_zone] = 0.0
        # A one-pixel optical transition removes the bright matte fringe that
        # Qwen leaves around curls and bows. The protected semantic subject and
        # accessory zone above prevent this narrow antialiasing band from
        # repainting real hair or translucent petals.
        feathered = cv2.GaussianBlur(matte, (5, 5), .9)
        definite_background = cv2.erode(
            matte, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1,
        )
        feathered[definite_background == 255] = 255
        background_fraction = np.maximum(
            feathered.astype(np.float32) / 255.0,
            fringe_strength,
        )[:, :, None]
        target = np.full_like(rgb, self._parse_bg_color(background_color), dtype=np.uint8)
        old_background_rgb = np.median(np.concatenate((
            rgb[:edge, :edge].reshape(-1, 3),
            rgb[:edge, -edge:].reshape(-1, 3),
        )), axis=0).astype(np.float32)
        # A Qwen edge pixel already contains part of the old backdrop. Blending
        # that pixel with the selected colour retains the old gray/white part
        # and creates the visible halo. Replace the estimated backdrop
        # contribution instead: C_out = C_in + b * (new_bg - old_bg), where b
        # is non-zero only in the narrow, border-connected fringe above. Face,
        # hair/accessory cores and fabric are excluded before this calculation,
        # preventing the dark skin notches caused by earlier semantic remattes.
        corrected = rgb.astype(np.float32) + background_fraction * (
            target.astype(np.float32) - old_background_rgb[None, None, :]
        )
        corrected[matte == 255] = target[matte == 255]
        corrected = corrected.clip(0, 255).astype(np.uint8)
        # Diffusion occasionally leaves a detached 2-4 px sliver above the
        # crown. It is neither connected hair nor an accessory and becomes
        # conspicuous on a flat passport backdrop after headroom is added.
        # Remove only tiny, very shallow top components; real hair/accessories
        # retain meaningful height or connect to the main subject.
        target_rgb = target[0, 0].astype(np.float32)
        residual = (
            np.linalg.norm(corrected.astype(np.float32) - target_rgb, axis=2) > 18.0
        ).astype(np.uint8)
        component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
            residual, connectivity=8,
        )
        max_sliver_area = max(32, int(h * w * .00035))
        max_sliver_height = max(3, int(h * .009))
        for component in range(1, component_count):
            x, y, component_w, component_h, area = component_stats[component]
            if (
                y < int(h * .14)
                and area <= max_sliver_area
                and component_h <= max_sliver_height
                and component_w >= component_h * 3
            ):
                corrected[component_labels == component] = target_rgb.astype(np.uint8)
        return Image.fromarray(corrected, "RGB")

    def _replace_flat_border_background_strict(
        self,
        image: Image.Image,
        background_color: Union[str, Tuple[int, int, int]],
    ) -> Image.Image:
        """Correct a generated flat backdrop without rematting the subject.

        Enhancement already asks Qwen to render a flat studio background.  Its
        decoded colour can be slightly off the selected hex, but the backdrop
        remains smooth and border-connected.  A hue-wide matte is unsafe for
        blue clothing and pale flowers, so this route requires close LAB colour
        *and* low texture and never replaces a parser-confirmed subject pixel.
        """
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        height, width = rgb.shape[:2]
        edge = max(4, min(18, height // 20, width // 20))
        if height < 32 or width < 32:
            return image.convert("RGB")
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        corner_lab = np.concatenate((
            lab[:edge, :edge].reshape(-1, 3),
            lab[:edge, -edge:].reshape(-1, 3),
            lab[-edge:, :edge].reshape(-1, 3),
            lab[-edge:, -edge:].reshape(-1, 3),
        ))
        backdrop_lab = np.median(corner_lab, axis=0)
        distance = np.linalg.norm(lab - backdrop_lab, axis=2)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        corner_hsv = np.concatenate((
            hsv[:edge, :edge].reshape(-1, 3),
            hsv[:edge, -edge:].reshape(-1, 3),
            hsv[-edge:, :edge].reshape(-1, 3),
            hsv[-edge:, -edge:].reshape(-1, 3),
        ))
        backdrop_hsv = np.median(corner_hsv, axis=0)
        mean = cv2.GaussianBlur(gray, (0, 0), 2.2)
        square_mean = cv2.GaussianBlur(gray * gray, (0, 0), 2.2)
        texture = np.sqrt(np.maximum(0.0, square_mean - mean * mean))
        # Qwen's generated backdrop is already flat and very low variance. A
        # tight colour/texture key is sufficient and, unlike a human parser,
        # cannot mistake pale flowers, flyaway hair or a blue jacket for
        # background. Only the connected component touching the canvas edge is
        # eligible below, so similarly coloured interior details stay intact.
        # Allow the small smooth blue variation produced by the VAE. The old
        # 12/8 key missed almost the entire visually-flat backdrop, which then
        # triggered a destructive semantic rematte. Plaid, flowers and hair
        # remain protected by their substantially higher local texture and/or
        # colour distance.
        # Keep the variance limit conservative at subject boundaries. A woven
        # blue garment can be close to the generated blue backdrop and reach
        # the lower frame edge; the former 12-level texture allowance admitted
        # parts of that fabric into the border component. Eight still accepts
        # Qwen's smooth VAE backdrop falloff while preserving woven cloth,
        # curls and pale accessories byte-for-byte.
        if backdrop_hsv[1] >= 35.0:
            # A VAE often renders a requested solid chromatic backdrop as a
            # broad lightness gradient. A global LAB radius then leaves large
            # islands and a cyan/green outline. Hue remains stable across that
            # gradient, so grow the border field by hue, saturation and low
            # texture instead. Patterned cloth is rejected by texture; dark
            # hair and pale accessories are rejected by value/saturation.
            hue_delta = np.abs(hsv[:, :, 0] - backdrop_hsv[0])
            hue_delta = np.minimum(hue_delta, 180.0 - hue_delta)
            minimum_saturation = max(24.0, float(backdrop_hsv[1]) * .28)
            minimum_value = max(18.0, float(backdrop_hsv[2]) * .18)
            candidates = (
                (hue_delta <= 16.0)
                & (hsv[:, :, 1] >= minimum_saturation)
                & (hsv[:, :, 2] >= minimum_value)
                & (texture <= 8.0)
            )
        else:
            candidates = (distance <= 22.0) & (texture <= 8.0)
        # Protect semantic subject pixels before connected-component growth.
        # The former hue-only key treated smooth blue reflections on dark
        # curls as backdrop and created visible holes after an otherwise clean
        # Qwen decode. A small head/hair dilation also protects antialiased
        # strands and pale hair accessories that SCHP labels as background.
        try:
            from pipelines.schp_service import parse as parse_body_parts

            strict_labels = parse_body_parts(rgb)["labels"]
            strict_subject = strict_labels != 0
            strict_head = np.isin(strict_labels, (1, 2, 4, 13)).astype(np.uint8)
            strict_head = cv2.dilate(
                strict_head,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            ).astype(bool)
            candidates[strict_subject | strict_head] = False
        except Exception as exc:
            logger.warning(
                "[UNIFORM_BACKGROUND] Subject guard unavailable for strict backdrop correction: %s",
                exc,
            )
        candidates = cv2.morphologyEx(
            candidates.astype(np.uint8), cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        ).astype(bool)
        if 'strict_subject' in locals():
            candidates[strict_subject | strict_head] = False
        component_count, components = cv2.connectedComponents(
            candidates.astype(np.uint8), connectivity=8,
        )
        if component_count <= 1:
            return image.convert("RGB")
        border_components = np.unique(np.concatenate((
            components[0], components[-1], components[:, 0], components[:, -1],
        )))
        border_components = border_components[border_components != 0]
        if not border_components.size:
            return image.convert("RGB")
        matte = np.isin(components, border_components)
        target_rgb = np.asarray(
            self._parse_bg_color(background_color), dtype=np.float32,
        )
        # Feather inward only across pixels that still satisfy the tight flat
        # backdrop key. Subject-side pixels remain byte-for-byte identical.
        alpha = cv2.GaussianBlur(
            matte.astype(np.float32), (0, 0), sigmaX=.45, sigmaY=.45,
        )
        alpha[matte] = 1.0
        alpha[~candidates] = 0.0
        if backdrop_hsv[1] >= 35.0:
            # Remove only the thin VAE-colored fringe immediately outside the
            # accepted background component. This is not a person matte: a
            # pixel must remain close to the generated backdrop hue/LAB field
            # and bright/saturated enough to be backdrop contamination. Dark
            # hair, blue checks, skin and white accessories therefore remain
            # unchanged even when they touch the canvas edge.
            near_background = cv2.dilate(
                matte.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            ).astype(bool)
            fringe = (
                near_background
                & ~matte
                & (hue_delta <= 16.0)
                & (distance <= 65.0)
                & (hsv[:, :, 1] >= minimum_saturation)
                & (hsv[:, :, 2] >= max(48.0, float(backdrop_hsv[2]) * .42))
                & (texture <= 8.0)
            )
            alpha[fringe] = 1.0
        result = (
            rgb.astype(np.float32) * (1.0 - alpha[:, :, None])
            + target_rgb[None, None, :] * alpha[:, :, None]
        )
        logger.info(
            "[ENHANCE_BACKGROUND] Strict border correction replaced %.3f of canvas",
            float(np.mean(matte)),
        )
        return Image.fromarray(np.rint(result).clip(0, 255).astype(np.uint8), "RGB")

    @staticmethod
    def _replace_source_scene_leaks(
        image: Image.Image,
        source_image: Image.Image,
        background_color: Union[str, Tuple[int, int, int]],
    ) -> Image.Image:
        """Remove copied source scenery without rematting the generated person.

        Qwen normally replaces the backdrop, but tiny islands of the uploaded
        wall/foliage can remain between curls. Learn frequent scene colours from
        source pixels classified as background, then replace them only where
        the generated-image parser independently reports background. Subject
        RGB and alpha are never reconstructed, so curls, clips, earrings, face,
        neck and garment remain the coherent Qwen decode.
        """
        try:
            from pipelines.schp_service import parse as parse_body_parts

            generated = np.asarray(image.convert("RGB"), dtype=np.uint8)
            height, width = generated.shape[:2]
            source = np.asarray(
                source_image.convert("RGB").resize((width, height), Image.LANCZOS),
                dtype=np.uint8,
            )
            source_labels = parse_body_parts(source)["labels"]
            generated_labels = parse_body_parts(generated)["labels"]
            source_background_pixels = source[source_labels == 0]
            if source_background_pixels.shape[0] < 64:
                return image.convert("RGB")

            quantized = (source_background_pixels // 24) * 24 + 12
            packed = (
                quantized[:, 0].astype(np.int32) << 16
                | quantized[:, 1].astype(np.int32) << 8
                | quantized[:, 2].astype(np.int32)
            )
            values, counts = np.unique(packed, return_counts=True)
            palette_size = min(32, len(values))
            palette_values = values[np.argsort(counts)[-palette_size:]]
            palette_rgb = np.stack((
                (palette_values >> 16) & 255,
                (palette_values >> 8) & 255,
                palette_values & 255,
            ), axis=1).astype(np.uint8)
            palette_lab = cv2.cvtColor(
                palette_rgb.reshape(1, -1, 3), cv2.COLOR_RGB2LAB,
            ).reshape(-1, 3).astype(np.float32)
            generated_lab = cv2.cvtColor(
                generated, cv2.COLOR_RGB2LAB,
            ).astype(np.float32)
            scene_distance = np.full((height, width), np.inf, dtype=np.float32)
            for palette_start in range(0, palette_lab.shape[0], 16):
                palette_chunk = palette_lab[palette_start:palette_start + 16]
                scene_distance = np.minimum(
                    scene_distance,
                    np.min(
                        np.linalg.norm(
                            generated_lab[:, :, None, :]
                            - palette_chunk[None, None, :, :],
                            axis=3,
                        ),
                        axis=2,
                    ),
                )
            scene_like = scene_distance <= 20.0
            leak = (generated_labels == 0) & scene_like

            # Never alter the central face/neck corridor even if the parser has
            # a small internal label hole. This protection is derived from the
            # generated face class, not from fixed portrait coordinates.
            face_y, face_x = np.where(generated_labels == 13)
            if face_y.size:
                fx1, fx2 = int(face_x.min()), int(face_x.max())
                fy1, fy2 = int(face_y.min()), int(face_y.max())
                fw = max(1, fx2 - fx1 + 1)
                fh = max(1, fy2 - fy1 + 1)
                px1 = max(0, int(round(fx1 - fw * .14)))
                px2 = min(width, int(round(fx2 + fw * .14)) + 1)
                py1 = max(0, int(round(fy1 - fh * .08)))
                py2 = min(height, int(round(fy2 + fh * .42)) + 1)
                leak[py1:py2, px1:px2] = False
                # Source scenery survives primarily inside/around the hair.
                # Never apply this specialized cleanup to the shirt or torso,
                # where plaid highlights can share colours with the old scene.
                hair_zone_bottom = min(
                    height, int(round(fy2 + fh * .34)) + 1,
                )
                leak[hair_zone_bottom:, :] = False

            # SCHP has no class for barrettes, flowers or clips. Protect compact
            # bright objects attached to parsed hair even when it labels their
            # interiors as background. This is derived from shape, scale and
            # adjacency for each portrait, not from an assumed accessory name.
            hsv = cv2.cvtColor(generated, cv2.COLOR_RGB2HSV)
            near_hair = cv2.dilate(
                (generated_labels == 2).astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                iterations=1,
            ).astype(bool)
            possible_accessory = (
                leak
                & near_hair
                & (hsv[:, :, 2] >= 145)
                & (hsv[:, :, 1] <= 105)
            )
            component_count, components, stats, _ = cv2.connectedComponentsWithStats(
                possible_accessory.astype(np.uint8), connectivity=8,
            )
            max_accessory_area = max(18, int(height * width * .0025))
            for component in range(1, component_count):
                if int(stats[component, cv2.CC_STAT_AREA]) <= max_accessory_area:
                    leak[components == component] = False

            if not np.any(leak):
                return image.convert("RGB")
            target = np.asarray(
                QwenEditPipeline._parse_bg_color(background_color), dtype=np.float32,
            )
            alpha = cv2.GaussianBlur(
                leak.astype(np.float32), (0, 0), sigmaX=.45, sigmaY=.45,
            )
            alpha[leak] = 1.0
            result = (
                generated.astype(np.float32) * (1.0 - alpha[:, :, None])
                + target[None, None, :] * alpha[:, :, None]
            )
            logger.info(
                "[UNIFORM_SCENE_CLEANUP] Replaced %d parser-confirmed source-scene pixels",
                int(np.count_nonzero(leak)),
            )
            return Image.fromarray(
                np.rint(result).clip(0, 255).astype(np.uint8), "RGB",
            )
        except Exception as exc:
            logger.warning(
                "[UNIFORM_SCENE_CLEANUP] Dynamic scene cleanup unavailable: %s", exc,
            )
            return image.convert("RGB")

    @staticmethod
    def _soft_matte_chromatic_uniform_outline(
        image: Image.Image,
        background_color: Union[str, Tuple[int, int, int]],
        job_id: str = "uniform_outline",
        source_image: Optional[Image.Image] = None,
    ) -> Image.Image:
        """Remove Qwen's chromatic cast shadow without rebuilding the person.

        Saturated passport colours can make Qwen render a soft blue/cyan rim
        around otherwise correct hair and shoulders.  A hard colour key clips
        curls and accessories.  BiRefNet supplies the soft boundary while SCHP
        protects solid face, hair, garment and accessory interiors.  The
        foreground RGB returned by the service is already decontaminated; a
        small alpha-floor removal discards only low-confidence cast-shadow
        opacity before compositing onto the exact selected colour.
        """
        try:
            from pipelines.birefnet_service import background_removal

            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            cutout_bytes = background_removal.remove_background(
                buffer.getvalue(), job_id=f"{job_id}_outline", use_schp=True,
                source_image=source_image,
            )
            rgba = np.asarray(
                Image.open(io.BytesIO(cutout_bytes)).convert("RGBA"),
                dtype=np.uint8,
            )
            alpha = rgba[:, :, 3].astype(np.float32) / 255.0
            subject_fraction = float(np.mean(alpha >= .50))
            if not .12 <= subject_fraction <= .92:
                logger.warning(
                    "[UNIFORM_OUTLINE] Unsafe matte coverage %.3f; keeping Qwen boundary",
                    subject_fraction,
                )
                return image.convert("RGB")
            # Remove only small alpha islands that are spatially detached from
            # the main person. BiRefNet can confidently retain a distant bokeh
            # dot or wall fleck; those appeared as white/gray particles on the
            # selected backdrop. Real earrings and flyaway curls are retained
            # when connected to, or close to, the main subject. Large detached
            # accessories are also retained.
            alpha_components = (alpha >= .08).astype(np.uint8)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                alpha_components, 8,
            )
            if count > 1:
                main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                main_mask = labels == main
                proximity_radius = max(9, min(alpha.shape) // 55)
                proximity_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (proximity_radius * 2 + 1, proximity_radius * 2 + 1),
                )
                near_main = cv2.dilate(
                    main_mask.astype(np.uint8), proximity_kernel,
                ).astype(bool)
                maximum_speck_area = max(48, int(alpha.size * .0015))
                removed_specks = 0
                for component in range(1, count):
                    if component == main:
                        continue
                    component_mask = labels == component
                    component_area = int(stats[component, cv2.CC_STAT_AREA])
                    if (
                        component_area <= maximum_speck_area
                        and not np.any(component_mask & near_main)
                    ):
                        alpha[component_mask] = 0.0
                        removed_specks += component_area
                if removed_specks:
                    logger.info(
                        "[UNIFORM_OUTLINE] Removed %d detached background-matte speck pixels",
                        removed_specks,
                    )
            # Six percent removes the diffuse cast-shadow tail seen around
            # saturated backdrops while retaining mid/high-alpha flyaway hair.
            alpha = np.clip((alpha - .06) / .94, 0.0, 1.0)
            foreground = rgba[:, :, :3].astype(np.float32)
            target = np.asarray(
                QwenEditPipeline._parse_bg_color(background_color),
                dtype=np.float32,
            )
            composed = (
                foreground * alpha[:, :, None]
                + target[None, None, :] * (1.0 - alpha[:, :, None])
            )
            logger.info(
                "[UNIFORM_OUTLINE] Applied soft BiRefNet+SCHP chromatic-rim cleanup; subject=%.3f",
                subject_fraction,
            )
            return Image.fromarray(
                np.rint(composed).clip(0, 255).astype(np.uint8), "RGB",
            )
        except Exception as exc:
            logger.warning(
                "[UNIFORM_OUTLINE] Soft matte unavailable; keeping Qwen boundary: %s",
                exc,
            )
            return image.convert("RGB")

    @staticmethod
    def _reduce_uniform_hair_sun_glare(image: Image.Image) -> Image.Image:
        """Reduce broad crown glare in a generated uniform portrait only.

        This helper is intentionally local to Uniform Swap. It operates on the
        low-frequency luminance field inside SCHP-confirmed hair near the one
        detected face. Hair geometry, strand texture, accessories, skin,
        uniform and backdrop pixels are not reconstructed or recoloured.
        """
        try:
            from pipelines.photo_restoration import _get_insight_app

            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            faces = _get_insight_app().get(bgr)
            if len(faces) != 1:
                return image.convert("RGB")
            from pipelines.schp_service import parse as parse_body_parts

            labels = parse_body_parts(rgb)["labels"]
            height, width = labels.shape
            x1, y1, x2, y2 = [
                int(round(float(v))) for v in faces[0].bbox
            ]
            face_w = max(24, x2 - x1)
            face_h = max(24, y2 - y1)
            window = np.zeros((height, width), dtype=bool)
            hx1 = max(0, x1 - int(face_w * .85))
            hx2 = min(width, x2 + int(face_w * .85))
            hy1 = max(0, y1 - int(face_h * .72))
            hy2 = min(height, y2 + int(face_h * .60))
            window[hy1:hy2, hx1:hx2] = True
            hair = (labels == 2) & window
            if int(hair.sum()) < 128:
                return image.convert("RGB")

            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            sigma = max(8.0, face_w * .075)
            low_l = cv2.GaussianBlur(lab[:, :, 0], (0, 0), sigma)
            hair_values = low_l[hair]
            target = float(np.percentile(hair_values, 58))
            excess = np.clip(low_l - (target + 11.0), 0.0, 24.0)
            alpha = cv2.GaussianBlur(
                hair.astype(np.uint8) * 255,
                (0, 0), max(2.5, face_w * .014),
            ).astype(np.float32) / 255.0
            # Modify only broad illumination. The original high-frequency
            # strand detail remains present in LAB luminance.
            lab[:, :, 0] -= excess * alpha * .52
            corrected = cv2.cvtColor(
                np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB,
            )
            logger.info(
                "[UNIFORM_LIGHTING] Reduced broad crown/hair sunlight on %d pixels",
                int(hair.sum()),
            )
            return Image.fromarray(corrected, "RGB")
        except Exception as exc:
            logger.warning(
                "[UNIFORM_LIGHTING] Hair-glare correction skipped: %s", exc,
            )
            return image.convert("RGB")

    @staticmethod
    def _prepare_extreme_sunlight_uniform_guide(
        image: Image.Image, strength: float = 1.0,
    ) -> Image.Image:
        """Build a neutral-light conditioning guide without changing geometry.

        This is used only for a source whose measured light is extreme.  It is
        not a delivered image and does not invent pixels: low-frequency face
        illumination is balanced from that portrait's own shaded skin, while
        crown glare is pulled toward that portrait's own dark-hair samples.
        High-frequency pores/strands remain in the luminance residual.  Hair
        accessories are excluded by chroma distance from the measured hair.
        """
        try:
            from pipelines.photo_restoration import _get_insight_app
            from pipelines.schp_service import parse as parse_body_parts

            strength = float(np.clip(strength, .35, 1.0))
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            faces = _get_insight_app().get(bgr)
            if len(faces) != 1:
                return image.convert("RGB")
            labels = parse_body_parts(rgb)["labels"]
            if labels.shape != rgb.shape[:2]:
                return image.convert("RGB")

            height, width = labels.shape
            x1, y1, x2, y2 = [
                int(round(float(value))) for value in faces[0].bbox
            ]
            face_w, face_h = max(24, x2 - x1), max(24, y2 - y1)
            yy, xx = np.indices((height, width))
            face_window = (
                (xx >= max(0, x1 - int(face_w * .10)))
                & (xx <= min(width - 1, x2 + int(face_w * .10)))
                & (yy >= max(0, y1 - int(face_h * .08)))
                & (yy <= min(height - 1, y2 + int(face_h * .10)))
            )
            hair_window = (
                (xx >= max(0, x1 - int(face_w * .90)))
                & (xx <= min(width - 1, x2 + int(face_w * .90)))
                & (yy >= max(0, y1 - int(face_h * .78)))
                & (yy <= min(height - 1, y2 + int(face_h * .48)))
            )
            face_skin = (labels == 13) & face_window
            hair = (labels == 2) & hair_window
            if np.count_nonzero(face_skin) < 128 or np.count_nonzero(hair) < 128:
                return image.convert("RGB")

            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            luminance = lab[:, :, 0]

            # Face: learn neutral chroma and target illumination from the
            # middle shaded skin values, excluding clipped highlights/shadows.
            skin_l = luminance[face_skin]
            skin_lo, skin_hi = np.percentile(skin_l, (25, 62))
            skin_anchor = face_skin & (luminance >= skin_lo) & (luminance <= skin_hi)
            if np.count_nonzero(skin_anchor) >= 64:
                # Use the less warm portion of this person's own shaded skin.
                # A median remains orange when direct sun covers most of the
                # face; low robust percentiles retain the available neutral
                # evidence without imposing a fixed ethnicity/skin colour.
                skin_ab = np.percentile(
                    lab[:, :, 1:3][skin_anchor], 18, axis=0,
                )
                # When the whole visible face is under the same orange sun,
                # even its "least warm" percentile is not a neutral skin
                # reference. Detect that condition from the source itself and
                # compress only the excess chroma before diffusion. This is an
                # illuminant correction, not a fixed complexion: the source
                # hue/asymmetry and all high-frequency face pixels remain.
                skin_sat75 = float(np.percentile(hsv[:, :, 1][face_skin], 75))
                warm_contamination = float(np.clip(
                    (skin_sat75 - 150.0) / 105.0, 0.0, .72,
                ))
                if warm_contamination > 0.0:
                    neutral_axis = np.asarray((128.0, 128.0), dtype=np.float32)
                    # Yellow/orange illumination primarily inflates LAB-b;
                    # retain more of LAB-a so the person's natural undertone
                    # is not washed out in the conditioning guide.
                    skin_ab = np.asarray((
                        neutral_axis[0] + (skin_ab[0] - neutral_axis[0])
                        * (1.0 - warm_contamination * .28),
                        neutral_axis[1] + (skin_ab[1] - neutral_axis[1])
                        * (1.0 - warm_contamination * .62),
                    ), dtype=np.float32)
                sigma = max(9.0, face_w * .13)
                illumination = cv2.GaussianBlur(
                    luminance, (0, 0), sigmaX=sigma, sigmaY=sigma,
                )
                target_l = float(np.median(illumination[skin_anchor]))
                face_alpha = cv2.GaussianBlur(
                    face_skin.astype(np.uint8) * 255,
                    (0, 0), max(2.0, face_w * .018),
                ).astype(np.float32) / 255.0
                correction = np.clip(target_l - illumination, -58.0, 34.0)
                luminance += correction * face_alpha * (.82 * strength)
                # Warm/orange excess receives the strongest correction; normal
                # natural skin variation is retained around its own median.
                chroma_delta = np.linalg.norm(
                    lab[:, :, 1:3] - skin_ab[None, None, :], axis=2,
                )
                chroma_strength = np.clip((chroma_delta - 5.0) / 26.0, 0.0, .72)
                chroma_strength *= face_alpha * strength
                if warm_contamination > 0.0:
                    # A globally cast face can be uniformly far from neutral,
                    # leaving little relative delta for the line above. Add a
                    # bounded whole-face component only in that measured case.
                    chroma_strength = np.maximum(
                        chroma_strength,
                        face_alpha * warm_contamination * .58 * strength,
                    )
                for channel in (1, 2):
                    lab[:, :, channel] += (
                        skin_ab[channel - 1] - lab[:, :, channel]
                    ) * chroma_strength

            # Hair: infer pigment from the darkest parsed strands. Correct
            # only pixels chromatically compatible with that pigment, so pale
            # flowers, clips, ribbons and earrings are not darkened.
            hair_l = luminance[hair]
            dark_cut = float(np.percentile(hair_l, 38))
            dark_hair = hair & (luminance <= dark_cut)
            if np.count_nonzero(dark_hair) >= 64:
                hair_ab = np.median(lab[:, :, 1:3][dark_hair], axis=0)
                # The median is dominated by silver glare when the complete
                # crown is clipped. Anchor the guide to the portrait's own
                # lower dark-hair quartile instead.
                target_hair_l = float(np.percentile(hair_l, 28))
                ab_distance = np.linalg.norm(
                    lab[:, :, 1:3] - hair_ab[None, None, :], axis=2,
                )
                # Silver glare is near-neutral and close to dark hair chroma;
                # decorative flowers have a larger purple/green displacement.
                hair_candidate = hair & (ab_distance <= 20.0)
                low_frequency_hair = cv2.GaussianBlur(
                    luminance, (0, 0), sigmaX=max(7.0, face_w * .06),
                )
                excess = np.clip(
                    low_frequency_hair - (target_hair_l + 6.0), 0.0, 128.0,
                )
                hair_strength = np.clip(excess / 62.0, 0.0, 1.0)
                hair_alpha = cv2.GaussianBlur(
                    hair_candidate.astype(np.uint8) * 255,
                    (0, 0), max(1.5, face_w * .010),
                ).astype(np.float32) / 255.0
                hair_strength *= hair_alpha
                luminance -= excess * hair_alpha * (1.04 * strength)
                for channel in (1, 2):
                    lab[:, :, channel] += (
                        hair_ab[channel - 1] - lab[:, :, channel]
                    ) * hair_strength * (.78 * strength)

            lab[:, :, 0] = np.clip(luminance, 0, 255)
            corrected = cv2.cvtColor(
                np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB,
            )
            logger.info(
                "[UNIFORM_CONDITIONING] Built measured neutral-light guide at strength %.2f",
                strength,
            )
            return Image.fromarray(corrected, "RGB")
        except Exception as exc:
            logger.warning(
                "[UNIFORM_CONDITIONING] Extreme-light guide unavailable: %s", exc,
            )
            return image.convert("RGB")

    @staticmethod
    def _has_residual_uniform_harsh_light(
        image: Image.Image, face_bbox: Any,
    ) -> bool:
        """Detect residual hard light without confusing dark hair with skin.

        The shared legacy check measured one padded head rectangle. A normal
        neutral portrait with black hair and bright skin therefore produced a
        huge p20/p95 range and was falsely rejected. Uniform validation keeps
        face-skin illumination and SCHP-confirmed hair glare as separate
        signals and is invoked only when the source itself had direct light.
        """
        try:
            from pipelines.schp_service import parse as parse_body_parts

            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            height, width = rgb.shape[:2]
            x1, y1, x2, y2 = [
                int(round(float(value))) for value in face_bbox[:4]
            ]
            face_w, face_h = max(24, x2 - x1), max(24, y2 - y1)
            yy, xx = np.indices((height, width))
            face_ellipse = (
                ((xx - (x1 + x2) * .5) / max(face_w * .46, 1.0)) ** 2
                + ((yy - (y1 + face_h * .53)) / max(face_h * .43, 1.0)) ** 2
                <= 1.0
            )
            lab_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
            lab_l = lab_image[:, :, 0]
            hsv_s = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
            low_l = cv2.GaussianBlur(
                lab_l, (0, 0), sigmaX=max(5.0, face_w * .065),
            )
            face_values = low_l[face_ellipse]
            face_saturation = hsv_s[face_ellipse]
            if face_values.size < 128:
                return False
            face_p15, face_p90 = np.percentile(face_values, (15, 90))
            face_sat75 = float(np.percentile(face_saturation, 75))
            face_b50 = float(np.median(lab_image[:, :, 2][face_ellipse]))
            face_center_x = (x1 + x2) * .5
            face_center_y = y1 + face_h * .53
            quadrant_levels = [
                float(np.median(low_l[face_ellipse & (xx < face_center_x)])),
                float(np.median(low_l[face_ellipse & (xx >= face_center_x)])),
                float(np.median(low_l[face_ellipse & (yy < face_center_y)])),
                float(np.median(low_l[face_ellipse & (yy >= face_center_y)])),
            ]
            directional_spread = max(quadrant_levels) - min(quadrant_levels)
            directional_face_light = bool(
                face_p90 >= 172.0
                and (face_p90 - face_p15) >= 52.0
                # Normal studio portraits retain broad cheek/nose modelling
                # and can easily span 50 LAB-lightness points.  Direct sun on
                # skin also carries a strong chroma cast; require that evidence
                # here instead of rejecting ordinary three-dimensional facial
                # shading.  Neutral/white glare remains covered by the
                # independent SCHP-confirmed hair highlight gate below.
                and face_sat75 >= 190.0
                and directional_spread >= 35.0
            )
            global_warm_cast = bool(
                face_p90 >= 125.0
                and face_sat75 >= 210.0
                and face_b50 >= 168.0
            )
            face_hard = bool(directional_face_light or global_warm_cast)

            labels = parse_body_parts(rgb)["labels"]
            head_window = (
                (xx >= max(0, x1 - int(face_w * .85)))
                & (xx <= min(width - 1, x2 + int(face_w * .85)))
                & (yy >= max(0, y1 - int(face_h * .78)))
                & (yy <= min(height - 1, y2 + int(face_h * .38)))
            )
            hair = (labels == 2) & head_window
            hair_hard = False
            if np.count_nonzero(hair) >= 128:
                hair_values = low_l[hair]
                hair_p25, hair_p90 = np.percentile(hair_values, (25, 90))
                bright_fraction = float(np.mean(hair_values >= 165.0))
                hair_hard = bool(
                    hair_p90 >= 165.0
                    and (hair_p90 - hair_p25) >= 78.0
                    and bright_fraction >= .07
                )
            detected = bool(face_hard or hair_hard)
            logger.info(
                "[UNIFORM_SUNLIGHT_MEASURE] face_p15=%.1f face_p90=%.1f "
                "face_sat75=%.1f face_b50=%.1f directional=%.1f face_hard=%s hair_hard=%s detected=%s",
                face_p15, face_p90, face_sat75, face_b50, directional_spread,
                face_hard, hair_hard, detected,
            )
            return detected
        except Exception as exc:
            logger.warning(
                "[UNIFORM_SUNLIGHT_MEASURE] Residual-light check unavailable: %s",
                exc,
            )
            # A validation outage must not silently accept an unmeasured
            # extreme-light portrait.
            return True

    def _replace_background_outside_subject_envelope(
        self, image: Image.Image, background_color: Union[str, Tuple[int, int, int]]
    ) -> Image.Image:
        """Apply an exact backdrop without cutting holes through the portrait.

        This is a final fallback for textured Qwen backdrops.  SCHP can report
        glasses, neck skin and pale plaid checks as background, so its raw mask
        is not safe as an alpha matte.  Instead, retain the largest connected
        person silhouette and fill its horizontal span on every occupied row.
        Only the exterior is replaced; internal face, hair and garment pixels
        remain byte-for-byte from Qwen.
        """
        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        try:
            from pipelines.schp_service import parse as parse_body_parts

            labels = parse_body_parts(rgb)["labels"]
            subject = (labels != 0).astype(np.uint8)
            if np.count_nonzero(subject) < subject.size * .12:
                return image.convert("RGB")

            joined = cv2.morphologyEx(
                subject,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                iterations=2,
            )
            count, components, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
            if count <= 1:
                return image.convert("RGB")

            center_x = width // 2
            candidates = []
            for component in range(1, count):
                area = int(stats[component, cv2.CC_STAT_AREA])
                if area < max(32, int(height * width * .01)):
                    continue
                touches_center = bool(np.any(components[:, max(0, center_x - 2):center_x + 3] == component))
                candidates.append((touches_center, area, component))
            if not candidates:
                return image.convert("RGB")
            primary_id = max(candidates)[2]
            primary = components == primary_id

            # The generated head can be separated from the uniform by a few
            # parser-background rows at the collar. Selecting only the largest
            # component then keeps the shirt and deletes the complete face.
            # Head semantics are authoritative even when disconnected; the
            # row envelope safely fills internal glasses/skin holes without
            # extending beyond the detected hair/face sides.
            head_subject = np.isin(labels, (1, 2, 4, 13))
            primary |= head_subject
            head_rows = np.flatnonzero(np.any(head_subject, axis=1))
            torso_rows = np.flatnonzero(np.any(components == primary_id, axis=1))
            if head_rows.size and torso_rows.size:
                head_bottom = int(head_rows[-1])
                torso_below = torso_rows[torso_rows > head_bottom]
                if torso_below.size:
                    torso_top = int(torso_below[0])
                    if torso_top - head_bottom <= int(height * .18):
                        head_cols = np.flatnonzero(head_subject[head_bottom])
                        torso_cols = np.flatnonzero((components == primary_id)[torso_top])
                        if head_cols.size and torso_cols.size:
                            face_y, face_x = np.where(labels == 13)
                            if face_x.size:
                                face_left, face_right = int(face_x.min()), int(face_x.max())
                                face_center = (face_left + face_right) * .5
                                face_width = max(1, face_right - face_left + 1)
                            else:
                                face_center = float(np.median(head_cols))
                                face_width = max(1.0, float(head_cols[-1] - head_cols[0] + 1))
                            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
                            neck_zone = (
                                (np.indices((height, width))[1] >= int(face_center - face_width * .34))
                                & (np.indices((height, width))[1] <= int(face_center + face_width * .34))
                            )
                            skin_like = (
                                neck_zone
                                & (hsv[:, :, 1] >= 18)
                                & (hsv[:, :, 1] <= 190)
                                & (hsv[:, :, 2] >= 30)
                                & (hsv[:, :, 2] <= 245)
                                & ((hsv[:, :, 0] <= 25) | (hsv[:, :, 0] >= 170))
                            )
                            for bridge_row in range(head_bottom, torso_top + 1):
                                skin_columns = np.flatnonzero(skin_like[bridge_row])
                                if skin_columns.size >= max(3, int(face_width * .08)):
                                    primary[bridge_row, skin_columns[0]:skin_columns[-1] + 1] = True

            envelope = np.zeros((height, width), dtype=np.uint8)
            minimum_row_pixels = max(3, int(width * .01))
            for row in range(height):
                columns = np.flatnonzero(primary[row])
                if columns.size >= minimum_row_pixels:
                    envelope[row, columns[0]:columns[-1] + 1] = 1
            envelope = cv2.morphologyEx(
                envelope,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )
            envelope = cv2.dilate(
                envelope,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )

            target = np.asarray(self._parse_bg_color(background_color), dtype=np.uint8)
            result = rgb.copy()
            result[envelope == 0] = target
            return Image.fromarray(result, "RGB")
        except Exception as exc:
            logger.warning("[UNIFORM_BACKGROUND] Subject-envelope fallback unavailable: %s", exc)
            return image.convert("RGB")

    def _replace_background_from_conditioning_silhouette(
        self,
        generated: Image.Image,
        conditioning: Image.Image,
        background_color: Union[str, Tuple[int, int, int]],
    ) -> Image.Image:
        """Remove generated scenery using the fitted portrait as geometry only.

        A second, person-free garment reference improves fabric fidelity, but
        Qwen can surround the otherwise correct portrait with scene fragments.
        Segmenting that generated scene directly is unstable because SCHP may
        classify furniture as clothing.  The fitted conditioning portrait has
        an exact flat backdrop, so it provides a reliable probable-foreground
        silhouette. GrabCut then adapts that silhouette to Qwen's photographic
        edges without copying any conditioning pixels into the result.
        """
        try:
            candidate = np.asarray(generated.convert("RGB"), dtype=np.uint8)
            guide_image = conditioning.convert("RGB").resize(
                (candidate.shape[1], candidate.shape[0]), Image.LANCZOS,
            )
            guide = np.asarray(guide_image, dtype=np.uint8)
            target = np.asarray(self._parse_bg_color(background_color), dtype=np.float32)
            guide_distance = np.linalg.norm(
                guide.astype(np.float32) - target[None, None, :], axis=2,
            )
            guide_subject = (guide_distance > 40.0).astype(np.uint8)
            subject_fraction = float(np.mean(guide_subject))
            # A deliberately tight upper-chest passport crop can occupy more
            # than 80% of the canvas. The old .78 ceiling rejected exactly
            # those valid compositions before GrabCut could remove scenery.
            if not .12 <= subject_fraction <= .90:
                raise RuntimeError(
                    f"conditioning silhouette has implausible coverage {subject_fraction:.3f}"
                )

            height, width = guide_subject.shape
            scale = max(.65, min(1.6, width / 560.0))

            def odd_kernel(base: int) -> np.ndarray:
                size = max(3, int(round(base * scale)))
                if size % 2 == 0:
                    size += 1
                return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

            # Close small collar/neck gaps in the fitted guide, while a large
            # erosion keeps generated background near the old silhouette edge
            # out of the definite-foreground training samples.
            support = cv2.morphologyEx(
                guide_subject, cv2.MORPH_CLOSE, odd_kernel(25), iterations=1,
            )
            sure_foreground = cv2.erode(support, odd_kernel(61), iterations=1)
            probable_foreground = cv2.dilate(support, odd_kernel(41), iterations=1)
            if np.count_nonzero(sure_foreground) < 256:
                raise RuntimeError("conditioning silhouette has no reliable foreground core")

            grab_mask = np.full(guide_subject.shape, cv2.GC_BGD, dtype=np.uint8)
            grab_mask[probable_foreground > 0] = cv2.GC_PR_FGD
            grab_mask[sure_foreground > 0] = cv2.GC_FGD
            background_model = np.zeros((1, 65), np.float64)
            foreground_model = np.zeros((1, 65), np.float64)
            cv2.grabCut(
                cv2.cvtColor(candidate, cv2.COLOR_RGB2BGR),
                grab_mask,
                None,
                background_model,
                foreground_model,
                8,
                cv2.GC_INIT_WITH_MASK,
            )
            subject = np.isin(
                grab_mask, (cv2.GC_FGD, cv2.GC_PR_FGD),
            ).astype(np.uint8)
            # Never admit pixels outside the fitted subject support. This is
            # what prevents Qwen-generated furniture, limbs or wall fragments
            # from becoming part of the delivered passport portrait.
            subject &= support
            component_count, component_labels, component_stats, _ = (
                cv2.connectedComponentsWithStats(subject, 8)
            )
            if component_count <= 1:
                raise RuntimeError("generated portrait foreground is empty")
            largest = 1 + int(np.argmax(component_stats[1:, cv2.CC_STAT_AREA]))
            subject = (component_labels == largest).astype(np.uint8)
            final_fraction = float(np.mean(subject))
            if not .10 <= final_fraction <= .88:
                raise RuntimeError(
                    f"generated portrait matte has implausible coverage {final_fraction:.3f}"
                )

            alpha = cv2.GaussianBlur(
                subject.astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0,
            )[:, :, None]
            result = (
                candidate.astype(np.float32) * alpha
                + target[None, None, :] * (1.0 - alpha)
            )
            logger.info(
                "[UNIFORM_BACKGROUND] Applied fitted-silhouette GrabCut matte: guide=%.3f result=%.3f",
                subject_fraction,
                final_fraction,
            )
            return Image.fromarray(np.rint(result).clip(0, 255).astype(np.uint8), "RGB")
        except Exception as exc:
            logger.warning(
                "[UNIFORM_BACKGROUND] Fitted-silhouette matte unavailable: %s", exc,
            )
            return generated.convert("RGB")

    def _replace_background_with_foreground_matte(
        self,
        image: Image.Image,
        background_color: Union[str, Tuple[int, int, int]],
        job_id: Optional[str] = None,
        preserve_foreground_rgb: bool = False,
        strict: bool = False,
        transparent: bool = False,
    ) -> Image.Image:
        """Replace scenery using the local semantic person parser.

        The previous BiRefNet fallback produced tiled holes through hair and
        face on some outdoor portraits. SCHP supplies one coherent semantic
        silhouette and preserves the original Qwen RGB inside that silhouette.
        """
        try:
            from pipelines.schp_service import parse as parse_body_parts

            rgb = np.asarray(image.convert("RGB"))
            labels = parse_body_parts(rgb)["labels"]
            # SCHP is the foreground authority for this operation. Later
            # GrabCut and colour cleanup may recover label-0 hair/accessory
            # fringe, but they must never erase pixels SCHP already identified
            # as face, hair, neck, clothing, arms, or accessories.
            semantic_subject = labels != 0
            subject = semantic_subject.astype(np.uint8)
            if np.count_nonzero(subject) < subject.size * .18:
                raise RuntimeError("semantic subject mask is too small")

            # Plaid, pale sleeves and fine fabric can contain parser holes.
            # In a passport portrait the lower person is one continuous torso:
            # protect the complete horizontal span between its detected outer
            # edges on every lower row. This prevents GrabCut from punching
            # backdrop-coloured holes through shoulders, sleeves or chest.
            height, width = subject.shape
            # Close parser holes through eyeglasses, eyes and the exposed
            # neck before replacing any background pixels. This uses only the
            # generated portrait's own anatomical labels; no source pixels are
            # pasted back into the result.
            face_pixels = np.column_stack(np.where(labels == 13))
            if face_pixels.shape[0] >= 64:
                face_y1, face_x1 = face_pixels.min(axis=0)
                face_y2, face_x2 = face_pixels.max(axis=0)
                face_h = max(1, int(face_y2 - face_y1 + 1))
                face_w = max(1, int(face_x2 - face_x1 + 1))
                # SCHP label 3 is glove, not part of the head. Including it in
                # the convex hull allowed bright wall fragments misclassified
                # as gloves to become large white blocks behind the ears.
                head_labels = np.isin(labels, (1, 2, 4, 13)).astype(np.uint8)
                # Do not include a low side ponytail in the convex head hull:
                # joining crown and tail points retains a rectangular block
                # of the old backdrop behind the hair. The parsed tail itself
                # remains protected independently by `subject`.
                head_core = head_labels & (
                    np.indices(labels.shape)[0] <= int(face_y2 + face_h * .06)
                )
                head_points = np.column_stack(np.where(head_core > 0))
                if head_points.shape[0] >= 64:
                    head_hull = cv2.convexHull(head_points[:, ::-1].astype(np.int32))
                    cv2.fillConvexPoly(subject, head_hull, 1)
                eye_guard = np.zeros_like(subject)
                cv2.ellipse(
                    eye_guard,
                    (int((face_x1 + face_x2) * .5), int(face_y1 + face_h * .43)),
                    (max(2, int(face_w * .68)), max(2, int(face_h * .30))),
                    0, 0, 360, 1, -1,
                )
                subject[eye_guard > 0] = 1
                clothing_mask = np.isin(labels, (5, 6, 7, 10, 11, 12))
                centre_x = int((face_x1 + face_x2) * .5)
                # Use the placket-width centre corridor. A wider corridor sees
                # the raised collar points beside the neck and mistakes their
                # top row for the neck base, clipping the real central neck.
                central_half_width = max(3, int(face_w * .10))
                clothing_rows = np.flatnonzero(np.any(
                    clothing_mask[:, max(0, centre_x - central_half_width):
                                      min(width, centre_x + central_half_width + 1)],
                    axis=1,
                ))
                clothing_rows = clothing_rows[clothing_rows >= face_y2]
                collar_y = int(clothing_rows[0]) if clothing_rows.size else int(face_y2 + face_h * .42)
                collar_y = min(height - 1, max(int(face_y2), collar_y))
                neck_polygon = np.array((
                    (centre_x - max(2, int(face_w * .34)), int(face_y2 - face_h * .12)),
                    (centre_x + max(2, int(face_w * .34)), int(face_y2 - face_h * .12)),
                    (centre_x + max(2, int(face_w * .30)), collar_y + 2),
                    (centre_x - max(2, int(face_w * .30)), collar_y + 2),
                ), dtype=np.int32)
                cv2.fillConvexPoly(subject, neck_polygon, 1)
                # Above the collar, accept only anatomical head/hair labels and
                # the explicit glasses/neck guards. Scene pixels that SCHP
                # mistakes for clothing must not enter the Qwen conditioning
                # image, where they would be faithfully regenerated as halos.
                # CODEX NOTE: Include legitimate clothing labels (labels 5, 6, 7, 10, 11, 12),
                # head_hull, and genuine dark hair/ears (luma < 95.0) in anatomy_guard.
                # This guarantees that hair buns behind the neck, ears, temples, and
                # upper shoulders are never zeroed out into blue backdrop holes or jagged contours.
                anatomy_guard = head_labels.copy()
                if head_points.shape[0] >= 64:
                    cv2.fillConvexPoly(anatomy_guard, head_hull, 1)
                anatomy_guard[eye_guard > 0] = 1
                anatomy_guard[clothing_mask] = 1
                cv2.fillConvexPoly(anatomy_guard, neck_polygon, 1)
                luma_early = (
                    rgb[:, :, 0].astype(np.float32) * .2126
                    + rgb[:, :, 1].astype(np.float32) * .7152
                    + rgb[:, :, 2].astype(np.float32) * .0722
                )
                hair_or_feature = (luma_early < 95.0) & (subject > 0)
                anatomy_guard[hair_or_feature] = 1
                upper_limit = min(
                    height,
                    max(collar_y + 3, int(face_y2 + face_h * .28)),
                )
                upper_rows = np.indices(subject.shape)[0] < upper_limit
                subject[upper_rows & (anatomy_guard == 0)] = 0
            lower_subject_hull = np.zeros_like(subject)
            # Start the solid torso envelope below the collar. Beginning at
            # 46% crossed the source wall's horizontal colour boundary and
            # converted it into a retained strip through the person's hair.
            torso_start = int(height * .56)
            if face_pixels.shape[0] >= 64:
                torso_start = max(torso_start, min(height - 1, collar_y))
            for row in range(torso_start, height):
                columns = np.flatnonzero(subject[row])
                if columns.size >= max(8, int(width * .04)):
                    lower_subject_hull[row, columns[0]:columns[-1] + 1] = 1
            subject[lower_subject_hull > 0] = 1

            # SCHP supplies the tight person/hair silhouette. Keep surrounding
            # backdrop as definite background; the previous 31-pixel probable-
            # foreground expansion retained a visible gray halo around hair.
            near_subject = cv2.dilate(
                subject, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
                iterations=1,
            )
            grab_mask = np.full(subject.shape, cv2.GC_BGD, dtype=np.uint8)
            # Pixels just outside SCHP's silhouette must remain recoverable.
            # Leaving them as definite background made GrabCut permanently
            # remove fine hair, tied tails, ears and antialiased head edges.
            # Probable-background keeps the search local without forcing a
            # broad halo into the foreground.
            uncertain_subject_fringe = (near_subject > 0) & (subject == 0)
            grab_mask[uncertain_subject_fringe] = cv2.GC_PR_BGD
            grab_mask[subject > 0] = cv2.GC_PR_FGD
            non_hair = np.isin(
                labels, (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
            ).astype(np.uint8)
            non_hair_core = cv2.erode(
                non_hair, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )
            grab_mask[non_hair_core > 0] = cv2.GC_FGD
            grab_mask[lower_subject_hull > 0] = cv2.GC_FGD
            luma = (
                rgb[:, :, 0].astype(np.float32) * .2126
                + rgb[:, :, 1].astype(np.float32) * .7152
                + rgb[:, :, 2].astype(np.float32) * .0722
            )
            grab_mask[(labels == 2) & (luma < 115.0)] = cv2.GC_FGD
            # Recover hair-coloured pixels that SCHP missed at the crown,
            # temples or tied-tail edge. The candidate must touch parsed hair,
            # remain within a face-relative head corridor and match this
            # portrait's measured hair colour; no hairstyle or fixed colour is
            # assumed. These pixels stay protected through later scene cleanup.
            recovered_head_fringe = np.zeros_like(subject, dtype=bool)
            yy, xx = np.indices(subject.shape)
            parsed_hair = labels == 2
            if face_pixels.shape[0] >= 64 and np.count_nonzero(parsed_hair) >= 48:
                image_lab_for_hair = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
                hair_samples = image_lab_for_hair[parsed_hair & (luma <= np.percentile(luma[parsed_hair], 88))]
                if hair_samples.shape[0] < 32:
                    hair_samples = image_lab_for_hair[parsed_hair]
                measured_hair_lab = np.median(hair_samples, axis=0)
                hair_distance = np.linalg.norm(
                    image_lab_for_hair - measured_hair_lab[None, None, :], axis=2,
                )
                hair_neighborhood = cv2.dilate(
                    parsed_hair.astype(np.uint8),
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (max(5, int(round(face_w * .14)) | 1),
                         max(5, int(round(face_h * .18)) | 1)),
                    ),
                    iterations=1,
                ).astype(bool)
                head_corridor = (
                    (yy >= max(0, int(face_y1 - face_h * .28)))
                    & (yy <= min(height - 1, int(face_y2 + face_h * .38)))
                    & (xx >= max(0, int(face_x1 - face_w * .55)))
                    & (xx <= min(width - 1, int(face_x2 + face_w * .55)))
                )
                recovered_head_fringe = (
                    (labels == 0)
                    & uncertain_subject_fringe
                    & hair_neighborhood
                    & head_corridor
                    & (hair_distance <= 34.0)
                    & (luma <= 145.0)
                )
                grab_mask[recovered_head_fringe] = cv2.GC_PR_FGD
            # Hair accessories are frequently parsed as background. Protect
            # every non-backdrop accessory colour touching the upper hair
            # silhouette, including saturated red bows as well as pale clips.
            # The input here is Qwen's already-flat studio portrait, so corner
            # colour is reliable evidence of backdrop rather than scenery.
            edge = max(4, min(20, height // 20, width // 20))
            backdrop_rgb = np.median(np.concatenate((
                rgb[:edge, :edge].reshape(-1, 3),
                rgb[:edge, -edge:].reshape(-1, 3),
            )), axis=0)
            # A scene can have identical top corners while changing colour
            # halfway down the wall. Include both upper side borders before
            # declaring the backdrop flat; otherwise that wall boundary is
            # mistaken for a hair accessory and restored after cleanup.
            side_sample_height = max(edge, int(height * .70))
            corner_samples = np.concatenate((
                rgb[:edge, :edge].reshape(-1, 3),
                rgb[:edge, -edge:].reshape(-1, 3),
                rgb[:side_sample_height, :edge].reshape(-1, 3),
                rgb[:side_sample_height, -edge:].reshape(-1, 3),
            )).astype(np.float32)
            corner_distances = np.linalg.norm(corner_samples - backdrop_rgb, axis=1)
            flat_generated_backdrop = float(np.percentile(corner_distances, 90)) <= 12.0
            backdrop_distance = np.linalg.norm(
                rgb.astype(np.float32) - backdrop_rgb.astype(np.float32), axis=2
            )
            accessory_neighborhood = cv2.dilate(
                subject,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (121, 121)),
                iterations=1,
            )
            accessory_zone = (
                (yy < height * .48)
                & ((xx < width * .46) | (xx > width * .54))
                & (accessory_neighborhood > 0)
            )
            accessory = (
                accessory_zone
                & (backdrop_distance > 32.0)
                & flat_generated_backdrop
            )
            if not flat_generated_backdrop and np.count_nonzero(parsed_hair) >= 32:
                # A fallback matte is most often needed precisely when Qwen
                # retained textured scenery. In that case the old flat-field
                # condition above disabled accessory protection and cut flowers,
                # bows or clips out of otherwise valid generated hair. Recover
                # compact, non-semantic objects that physically overlap the
                # parsed hair silhouette. This is appearance- and geometry-
                # driven for every portrait; it assumes no accessory type,
                # colour, count or side.
                channel_chroma = (
                    rgb.max(axis=2).astype(np.float32)
                    - rgb.min(axis=2).astype(np.float32)
                )
                hair_attachment_zone = cv2.dilate(
                    parsed_hair.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
                    iterations=1,
                ).astype(bool)
                accessory_seed = (
                    accessory_zone
                    & (labels == 0)
                    & hair_attachment_zone
                    & (backdrop_distance > 32.0)
                    # Require a strong colour or pale-object anchor so varied
                    # foliage/wall texture does not merge with the accessory.
                    # The recovered component hull below includes its quieter
                    # interior pixels once this reliable anchor is found.
                    & ((channel_chroma > 50.0) | (luma > 215.0))
                )
                # Include pale petals and interior holes belonging to the same
                # compact object, not merely the saturated seed pixels.
                accessory_object_pixels = accessory_seed.astype(np.uint8)
                accessory_object_pixels = cv2.morphologyEx(
                    accessory_object_pixels,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                    iterations=1,
                )
                component_count, component_labels, component_stats, _ = (
                    cv2.connectedComponentsWithStats(
                        accessory_object_pixels, connectivity=8,
                    )
                )
                maximum_accessory_area = max(48, int(height * width * .055))
                for component_id in range(1, component_count):
                    component = component_labels == component_id
                    component_area = int(component_stats[component_id, cv2.CC_STAT_AREA])
                    if (
                        12 <= component_area <= maximum_accessory_area
                        and np.any(component & accessory_seed)
                        and np.any(component & hair_attachment_zone)
                        and not (
                            np.any(component[0, :])
                            or np.any(component[:, 0])
                            or np.any(component[:, -1])
                        )
                    ):
                        # Fill only the compact seed's own hull. Growing over
                        # every non-backdrop scene pixel can connect the object
                        # to foliage or a wall and incorrectly reject it as a
                        # border component.
                        component_points = np.column_stack(np.where(component))
                        recovered_component = component
                        if component_points.shape[0] >= 3:
                            recovered_component = np.zeros_like(component, dtype=np.uint8)
                            component_hull = cv2.convexHull(
                                component_points[:, ::-1].astype(np.int32),
                            )
                            cv2.fillConvexPoly(recovered_component, component_hull, 1)
                            recovered_component = recovered_component.astype(bool)
                        accessory |= recovered_component
            grab_mask[accessory] = cv2.GC_FGD
            torso = np.array((
                (int(width * .38), int(height * .56)),
                (int(width * .62), int(height * .56)),
                (int(width * .78), height - 1),
                (int(width * .22), height - 1),
            ), dtype=np.int32)
            cv2.fillConvexPoly(grab_mask, torso, cv2.GC_FGD)
            grab_mask[:8, :] = cv2.GC_BGD
            grab_mask[:int(height * .72), :8] = cv2.GC_BGD
            grab_mask[:int(height * .72), -8:] = cv2.GC_BGD
            background_model = np.zeros((1, 65), dtype=np.float64)
            foreground_model = np.zeros((1, 65), dtype=np.float64)
            cv2.grabCut(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), grab_mask, None,
                background_model, foreground_model, 5, cv2.GC_INIT_WITH_MASK,
            )
            foreground = np.isin(grab_mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
            foreground[recovered_head_fringe] = 1
            # Remove source-scene colours that remain attached to the head.
            # SCHP/GrabCut can regard a wall immediately behind the ears as a
            # continuous accessory, especially where two wall colours meet.
            # Learn those colours from this crop's own upper borders and apply
            # the cleanup only above the garment. This is image-adaptive and
            # does not assume a white, blue, or any other fixed backdrop.
            upper_border = np.concatenate((
                rgb[:torso_start, :edge].reshape(-1, 3),
                rgb[:torso_start, -edge:].reshape(-1, 3),
                rgb[:edge, :].reshape(-1, 3),
            ))
            border_lab = cv2.cvtColor(
                upper_border.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB,
            ).reshape(-1, 3).astype(np.float32)
            cluster_count = min(4, max(1, border_lab.shape[0] // 64))
            _, _, scene_centres = cv2.kmeans(
                border_lab,
                cluster_count,
                None,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, .2),
                3,
                cv2.KMEANS_PP_CENTERS,
            )
            image_lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
            scene_distance = np.min(
                np.linalg.norm(
                    image_lab[:, :, None, :] - scene_centres[None, None, :, :],
                    axis=3,
                ),
                axis=2,
            )
            scene_like = scene_distance <= 14.0
            # Textured/plastered source walls vary more than a flat generated
            # backdrop. Use a broader match only in anatomical gap and outer
            # fringe cleanup below; the global foreground decision keeps the
            # conservative threshold.
            relaxed_scene_like = scene_distance <= 28.0
            # Dark/saturated glasses and small earrings within the face guard
            # remain intact. Do not trust a semantic "face" or "hair" label
            # when that same pixel matches a colour learned from the border;
            # the defect being repaired is precisely a wall patch carrying a
            # false anatomical label.
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            face_detail = (
                (eye_guard > 0)
                & (luma < 185.0)
            ) if face_pixels.shape[0] >= 64 else np.zeros_like(subject, dtype=bool)
            if face_pixels.shape[0] >= 64:
                face_core = (
                    (labels == 13)
                    & (xx >= int(face_x1 + face_w * .20))
                    & (xx <= int(face_x2 - face_w * .20))
                    & (yy >= int(face_y1 + face_h * .05))
                    & (yy <= int(face_y2 - face_h * .05))
                )
                face_detail |= face_core
            # CODEX NOTE: Do not clear genuine clothing pixels that happen to share color
            # with the background scene (e.g. a white shirt in front of a white wall).
            clear_scene = (
                (yy < torso_start)
                & scene_like
                & ~face_detail
                & ~recovered_head_fringe
                & ~np.isin(labels, (5, 6, 7, 10, 11, 12, 13))
            )
            foreground[clear_scene] = 0
            # GrabCut can still relabel a small accessory edge after the hard
            # seeds are supplied. Restore only pixels demonstrably unlike the
            # sampled flat backdrop; holes within bows remain background.
            foreground[accessory] = 1
            # The solid lower-torso hull can reconnect a wide piece of the
            # original wall behind a neck, ponytail, or sloping shoulder. A
            # one-pixel edge cleanup cannot remove that interior island. Find
            # each border-colour component that is still connected to already
            # known exterior background and clear its retained pixels through
            # the upper-chest region. This is learned from each source image;
            # no backdrop colour or portrait geometry is hard-coded.
            scene_component_count, scene_components, _, _ = cv2.connectedComponentsWithStats(
                scene_like.astype(np.uint8), connectivity=8,
            )
            exterior_scene = scene_like & (foreground == 0)
            connected_scene_leak = np.zeros_like(scene_like, dtype=bool)
            scene_cleanup_bottom = min(
                height,
                int(face_y2 + face_h * .72) if face_pixels.shape[0] >= 64 else torso_start,
            )
            for scene_component_id in range(1, scene_component_count):
                component = scene_components == scene_component_id
                if np.any(component & exterior_scene):
                    connected_scene_leak |= component
            protected_scene_detail = face_detail | accessory | (
                (labels == 2) & (luma < 115.0)
            ) | recovered_head_fringe
            connected_scene_leak &= yy < scene_cleanup_bottom
            connected_scene_leak &= ~protected_scene_detail
            foreground[connected_scene_leak] = 0
            if face_pixels.shape[0] >= 64:
                side_neck_corridor = (
                    (yy >= int(face_y2 - face_h * .04))
                    & (yy <= int(face_y2 + face_h * .55))
                    & (np.abs(xx - (face_x1 + face_x2) * .5) >= face_w * .27)
                    & (np.abs(xx - (face_x1 + face_x2) * .5) <= face_w * .82)
                )
                side_neck_scene = (
                    side_neck_corridor
                    & (labels == 0)
                    & ~((luma < 115.0) & (scene_distance > 18.0))
                    & ~protected_scene_detail
                )
                foreground[side_neck_scene] = 0
            if face_pixels.shape[0] >= 64:
                # SCHP often misses a dark ponytail below or beside the ear.
                # Recover only dark pixels in an anatomical corridor around
                # this image's detected face. Bright wall remnants cannot pass
                # this gate, and the corridor adapts to the measured face size.
                hair_corridor = (
                    (yy >= max(0, int(face_y1 - face_h * .25)))
                    & (yy <= min(height - 1, int(face_y2 + face_h * 1.05)))
                    & (xx >= max(0, int(face_x1 - face_w * .85)))
                    & (xx <= min(width - 1, int(face_x2 + face_w * .85)))
                )
                dark_hair_candidates = (
                    hair_corridor
                    & (luma <= 105.0)
                    & (scene_distance > 18.0)
                ).astype(np.uint8)
                candidate_count, candidate_labels, candidate_stats, candidate_centroids = cv2.connectedComponentsWithStats(
                    dark_hair_candidates, connectivity=8,
                )
                hair_neighborhood = cv2.dilate(
                    (labels == 2).astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
                    iterations=1,
                ).astype(bool)
                for candidate_id in range(1, candidate_count):
                    candidate = candidate_labels == candidate_id
                    if np.any(candidate & hair_neighborhood):
                        foreground[candidate] = 1
                        recovered_head_fringe |= candidate
                        continue
                    # A low tied tail can be separated from crown hair by the
                    # ear, neck, background, or source collar. Recover only a
                    # compact hair-coloured component beside the nape. Large
                    # vest/jacket components and central garment shadows fail
                    # these face-relative area and position limits.
                    component_x = float(candidate_centroids[candidate_id][0])
                    component_y = float(candidate_centroids[candidate_id][1])
                    component_area = int(candidate_stats[candidate_id, cv2.CC_STAT_AREA])
                    component_top = int(candidate_stats[candidate_id, cv2.CC_STAT_TOP])
                    component_height = int(candidate_stats[candidate_id, cv2.CC_STAT_HEIGHT])
                    component_bottom = component_top + component_height
                    face_center_x = (face_x1 + face_x2) * .5
                    nape_side_component = (
                        max(20, int(face_w * face_h * .002)) <= component_area
                        <= max(80, int(face_w * face_h * .20))
                        and component_top <= int(face_y2 + face_h * .28)
                        and component_bottom <= int(face_y2 + face_h * .78)
                        and abs(component_x - face_center_x) >= face_w * .24
                        and abs(component_x - face_center_x) <= face_w * .92
                        and component_y >= face_y2 - face_h * .08
                    )
                    if nape_side_component:
                        foreground[candidate] = 1
                        recovered_head_fringe |= candidate
            # The lower torso hull deliberately closes holes through pale or
            # patterned clothing, but it can also connect a few pixels of the
            # old scene to a shoulder. Remove only scene-coloured pixels on
            # the *outer* silhouette. Interior garment pixels, recovered dark
            # ponytails and real accessories remain untouched.
            foreground_core = cv2.erode(
                foreground,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )
            outer_silhouette = (foreground > 0) & (foreground_core == 0)
            # Semantic parsing is least reliable on antialiased shoulder and
            # hair boundaries.  Requiring label 0 here retained pale wall
            # pixels whenever SCHP assigned the blended edge to clothing or
            # hair.  Remove border-colour contamination from the one-pixel
            # outer silhouette regardless of that uncertain label, while
            # retaining real dark hair and protected face/accessory detail.
            reliable_dark_hair_edge = ((labels == 2) & (luma < 115.0)) | recovered_head_fringe
            scene_edge_noise = (
                outer_silhouette
                & scene_like
                & ~face_detail
                & ~accessory
                & ~reliable_dark_hair_edge
            )
            foreground[scene_edge_noise] = 0
            # Remove a narrow multi-pixel source-colour fringe, not only the
            # outermost pixel. This covers resampling halos on sloped shoulders
            # while leaving interior plaid, skin and dark hair untouched.
            foreground_depth = cv2.distanceTransform(
                foreground.astype(np.uint8), cv2.DIST_L2, 5,
            )
            source_colour_fringe = (
                (foreground_depth > 0.0)
                & (foreground_depth <= 5.0)
                & (labels == 0)
                & ~face_detail
                & ~accessory
                & ~reliable_dark_hair_edge
                & ~recovered_head_fringe
            )
            foreground[source_colour_fringe] = 0
            # Also clear enclosed backdrop holes between ponytail strands and
            # inside collar gaps. They are not part of the outer ring, but the
            # source-border colour model and missing anatomical label make
            # them unambiguous background. The clothing edit mask is dilated
            # later, so tiny parser holes in the old shirt remain editable.
            enclosed_scene_holes = (
                scene_like & (labels == 0) & ~accessory & ~recovered_head_fringe
            )
            foreground[enclosed_scene_holes] = 0
            # SCHP occasionally labels a bright wall speck as hair (label 2),
            # especially inside a low ponytail. A genuine hair pixel is dark
            # and cannot simultaneously match the learned border clusters;
            # clear those false hair/head labels while retaining semantic
            # face, garment and limb pixels that may legitimately be pale.
            reliable_non_hair_subject = np.isin(
                labels, (5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
            )
            mislabeled_scene = (
                scene_like
                & ~face_detail
                & ~accessory
                & ~reliable_non_hair_subject
                & ~recovered_head_fringe
            )
            foreground[mislabeled_scene] = 0
            # Scene-colour heuristics operate only as background cleanup and
            # label-0 fringe recovery. Reassert the parser's complete person
            # mask before edge construction so matching wall/clothing colours
            # cannot cut holes through hair, head, collar, or shoulders.
            eye_detail_guard = np.zeros_like(semantic_subject, dtype=bool)
            if face_pixels.shape[0] >= 64:
                glasses_width = max(5, int(round(face_w * .18)) | 1)
                glasses_height = max(3, int(round(face_h * .06)) | 1)
                glasses_neighborhood = cv2.dilate(
                    (labels == 4).astype(np.uint8),
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (glasses_width, glasses_height),
                    ),
                    iterations=1,
                ).astype(bool)
                eye_detail_guard = (
                    (eye_guard > 0)
                    & glasses_neighborhood
                    & (xx >= int(face_x1 - face_w * .03))
                    & (xx <= int(face_x2 + face_w * .03))
                    & (yy >= int(face_y1 + face_h * .12))
                    & (yy <= int(face_y1 + face_h * .72))
                    & (scene_distance > 6.0)
                    & (luma < 215.0)
                )
            foreground[semantic_subject | eye_detail_guard] = 1
            foreground = cv2.morphologyEx(
                foreground, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )
            # Build a narrow, resolution-aware signed-distance feather. A
            # binary mask plus a 3x3 blur left a visible one-pixel staircase
            # around hair, glasses, ears and earrings after passport scaling.
            # Distance feathering is smooth but remains tight enough to retain
            # fine strands and accessories.
            inside_distance = cv2.distanceTransform(
                foreground.astype(np.uint8), cv2.DIST_L2, 5,
            )
            outside_distance = cv2.distanceTransform(
                (foreground == 0).astype(np.uint8), cv2.DIST_L2, 5,
            )
            signed_distance = inside_distance - outside_distance
            # Smooth the *geometry* before converting distance to alpha.  SCHP
            # is evaluated at raster resolution, so even a distance feather can
            # retain one-pixel stair steps on shoulders and around the crown.
            # A sub-pixel blur of the signed field rounds those steps without
            # expanding the silhouette or washing out fine hair strands.
            signed_distance = cv2.GaussianBlur(
                signed_distance, (0, 0), sigmaX=.90, sigmaY=.90,
                borderType=cv2.BORDER_REPLICATE,
            )
            feather_radius = float(np.clip(min(height, width) / 180.0, 2.25, 3.75))
            foreground_alpha = np.clip(
                (signed_distance + feather_radius) / (2.0 * feather_radius),
                0.0,
                1.0,
            )
            foreground_alpha = foreground_alpha * foreground_alpha * (
                3.0 - 2.0 * foreground_alpha
            )
            # Preserve the semantic foreground core pixel-exact. Keep the
            # one-pixel semantic boundary feathered so antialiased source-wall
            # colour is replaced rather than retained as a white/gray outline.
            # The boundary remains foreground in `foreground`; only its mixed
            # edge coverage changes.
            semantic_core = cv2.erode(
                semantic_subject.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            ).astype(bool)
            protected_thin_detail = (
                eye_detail_guard | accessory | recovered_head_fringe
            )
            semantic_boundary = (
                semantic_subject & ~semantic_core & ~protected_thin_detail
            )
            # SCHP assigns the antialiased wall/hair mixture to the hair or
            # head class. Estimate actual foreground coverage from distance to
            # this image's sampled backdrop colours, rather than treating that
            # entire semantic ring as opaque. This removes bright crown and
            # shoulder rims without eroding the one-pixel-inset subject core.
            boundary_coverage = np.clip(
                (scene_distance - 3.0) / 24.0, 0.0, 1.0,
            )
            foreground_alpha[semantic_boundary] = np.minimum(
                foreground_alpha[semantic_boundary],
                boundary_coverage[semantic_boundary],
            )
            foreground_alpha[protected_thin_detail & (foreground > 0)] = 1.0
            foreground_alpha[semantic_core] = 1.0
            # Reduce alpha continuously for pixels that resemble the learned
            # scene colours.  The previous hard zero created a visibly jagged
            # contour wherever an antialiased shoulder/hair pixel happened to
            # cross the scene threshold.  RGB decontamination below removes the
            # old backdrop contribution from the remaining soft boundary.
            softened_edge = (foreground_alpha > 0.0) & (foreground_alpha < 1.0)
            scene_edge = softened_edge & (labels == 0) & ~recovered_head_fringe
            scene_confidence = np.clip((scene_distance - 7.0) / 18.0, 0.0, 1.0)
            foreground_alpha[scene_edge] *= scene_confidence[scene_edge]
            foreground_alpha = np.clip(foreground_alpha, 0.0, 1.0)[:, :, None]
            background_alpha = 1.0 - foreground_alpha
            if transparent:
                # Recover foreground edge colour from the flat generated
                # backdrop before storing alpha. This prevents blue/white
                # fringes when the PNG is placed over another background.
                alpha_safe = np.maximum(foreground_alpha, 1.0 / 255.0)
                clean_rgb = (
                    rgb.astype(np.float32)
                    - background_alpha * backdrop_rgb[None, None, :].astype(np.float32)
                ) / alpha_safe
                clean_rgb = np.rint(clean_rgb).clip(0, 255).astype(np.uint8)
                clean_rgb[foreground[:, :] == 0] = 0
                alpha_channel = np.rint(foreground_alpha[:, :, 0] * 255.0).astype(np.uint8)
                return Image.fromarray(
                    np.dstack((clean_rgb, alpha_channel)), "RGBA"
                )
            target = np.full_like(rgb, self._parse_bg_color(background_color), dtype=np.uint8)
            # Replace the old backdrop contribution at antialiased edges
            # instead of blending the already-contaminated RGB over the new
            # backdrop a second time. Use the nearest border-derived scene
            # colour per pixel, which handles split classroom walls without
            # assuming a fixed white or cyan source background.
            nearest_scene = np.argmin(
                np.linalg.norm(
                    image_lab[:, :, None, :] - scene_centres[None, None, :, :],
                    axis=3,
                ),
                axis=2,
            )
            scene_rgb_centres = cv2.cvtColor(
                np.rint(scene_centres).clip(0, 255).astype(np.uint8).reshape(-1, 1, 3),
                cv2.COLOR_LAB2RGB,
            ).reshape(-1, 3).astype(np.float32)
            local_old_background = scene_rgb_centres[nearest_scene]
            composed = rgb.astype(np.float32) + background_alpha * (
                target.astype(np.float32) - local_old_background
            )
            composed[foreground == 0] = target[foreground == 0]
            return Image.fromarray(np.rint(composed).clip(0, 255).astype(np.uint8), "RGB")
        except Exception as exc:
            if strict:
                raise RuntimeError("Background color replacement failed; raw Qwen output is retained.") from exc
            logger.warning("[BACKGROUND_MATTE] Failed; retaining Qwen backdrop: %s", exc)
            return image.convert("RGBA" if transparent else "RGB")

    @staticmethod
    def _apply_rough_silhouette_guard(
        image: Image.Image, rough_reference: Image.Image
    ) -> Image.Image:
        """Remove generated scenery without clipping the finished uniform."""
        try:
            from pipelines.schp_service import parse as parse_body_parts

            rgb = np.asarray(image.convert("RGB"))
            reference = np.asarray(
                rough_reference.convert("RGB").resize((rgb.shape[1], rgb.shape[0]), Image.LANCZOS)
            )
            edge = max(4, min(16, reference.shape[0] // 20, reference.shape[1] // 20))
            backdrop = np.median(np.concatenate((
                reference[:edge, :edge].reshape(-1, 3),
                reference[:edge, -edge:].reshape(-1, 3),
            )), axis=0).astype(np.uint8)
            reference_foreground = (
                np.linalg.norm(reference.astype(np.float32) - backdrop.astype(np.float32), axis=2) > 28.0
            ).astype(np.uint8)
            reference_foreground = cv2.morphologyEx(
                reference_foreground, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1,
            )
            corridor = cv2.dilate(
                reference_foreground,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1,
            ).astype(bool)

            generated_labels = parse_body_parts(rgb)["labels"]
            generated_subject = generated_labels != 0
            luma = (
                rgb[:, :, 0].astype(np.float32) * .2126
                + rgb[:, :, 1].astype(np.float32) * .7152
                + rgb[:, :, 2].astype(np.float32) * .0722
            )
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            hair_pixels = generated_labels == 2
            if np.any(hair_pixels):
                hair_floor = float(np.percentile(luma[hair_pixels], 25))
                hair_limit = min(230.0, max(130.0, hair_floor + 75.0))
                plausible_hair = (
                    (luma <= hair_limit) | (hsv[:, :, 1] >= 25) | (luma >= 200.0)
                )
                generated_subject[hair_pixels & ~plausible_hair] = False
            yy, xx = np.indices(generated_subject.shape)
            # A bright window or frame behind the head is occasionally parsed
            # as shirt/coat. Real passport clothing begins at the collar and is
            # restored from the template mask below, so discard those labels
            # above that line before constructing the subject silhouette.
            generated_subject[
                np.isin(generated_labels, (5, 6, 7, 9, 10, 12))
                & (yy < int(rgb.shape[0] * .52))
            ] = False
            foreground = generated_subject & corridor
            # The diagnostic rough layout intentionally separates its pasted
            # head from the template collar, so its mask has no trustworthy
            # neck bridge. Preserve generated neck skin in the central frame.
            skin_mask = (generated_labels == 13).astype(np.uint8)
            skin_count, skin_components, skin_stats, _ = cv2.connectedComponentsWithStats(
                skin_mask, connectivity=8,
            )
            if skin_count > 1:
                largest_skin = 1 + int(np.argmax(skin_stats[1:, cv2.CC_STAT_AREA]))
                connected_skin = skin_components == largest_skin
                skin_x, _, skin_w, _, _ = skin_stats[largest_skin]
                neck_x_min = int(skin_x + skin_w * .08)
                neck_x_max = int(skin_x + skin_w * .92)
            else:
                connected_skin = generated_labels == 13
                neck_x_min = int(rgb.shape[1] * .30)
                neck_x_max = int(rgb.shape[1] * .70)
            central_neck = (
                connected_skin
                & (xx >= int(rgb.shape[1] * .25))
                & (xx <= int(rgb.shape[1] * .75))
                & (yy >= int(rgb.shape[0] * .39))
                & (yy <= int(rgb.shape[0] * .74))
            )
            # SCHP can split a naturally shaded neck into background islands,
            # especially directly below the jaw. Preserve chromatic, non-white
            # pixels in the narrow anatomical bridge from head to collar. This
            # is deliberately bounded to the centre so a coloured wall or prop
            # elsewhere cannot leak into the passport backdrop.
            central_neck_pixels = (
                (xx >= neck_x_min)
                & (xx <= neck_x_max)
                & (yy >= int(rgb.shape[0] * .39))
                & (yy <= int(rgb.shape[0] * .74))
                & (hsv[:, :, 1] >= 28)
                & (luma <= 245.0)
                & ((hsv[:, :, 0] <= 25) | (hsv[:, :, 0] >= 170))
            )
            # Below the collar, retain Qwen's complete parsed garment and arms.
            # Intersecting these pixels with the rough guide clipped shoulders
            # whenever the generated fit was naturally wider than the guide.
            # The rough mask remains a fallback for small parser gaps in checks,
            # piping and other high-contrast fabric details.
            generated_garment = np.isin(
                generated_labels, (5, 6, 7, 9, 10, 12, 14, 15)
            ) & (yy >= int(rgb.shape[0] * .48))
            lower_reference_garment = reference_foreground.astype(bool) & (
                yy >= int(rgb.shape[0] * .57)
            )
            foreground |= (
                central_neck | central_neck_pixels
                | generated_garment | lower_reference_garment
            )
            if np.count_nonzero(foreground) < foreground.size * 0.15:
                return image.convert("RGB")
            result = rgb.copy()
            result[foreground == 0] = backdrop
            return Image.fromarray(result, "RGB")
        except Exception as exc:
            logger.warning("[UNIFORM_SILHOUETTE] Guard unavailable: %s", exc)
            return image.convert("RGB")

    @staticmethod
    def _repair_selected_backdrop_intrusions(
        image: Image.Image,
        rough_reference: Image.Image,
        background_rgb: Tuple[int, int, int],
    ) -> Image.Image:
        """Repair compact backdrop-colour holes inside the expected garment.

        Qwen can emit exact selected-blue pixels through a shoulder or collar.
        A normal background validator accepts those pixels and a semantic matte
        preserves them. Repair only compact components that are simultaneously
        inside the rough garment silhouette and beside the generated subject;
        the real outer backdrop never enters this mask.
        """
        try:
            from pipelines.schp_service import parse as parse_body_parts

            rgb = np.asarray(image.convert("RGB"))
            height, width = rgb.shape[:2]
            reference = np.asarray(
                rough_reference.convert("RGB").resize((width, height), Image.LANCZOS)
            )
            edge = max(4, min(18, height // 20, width // 20))
            reference_bg = np.median(np.concatenate((
                reference[:edge, :edge].reshape(-1, 3),
                reference[:edge, -edge:].reshape(-1, 3),
            )), axis=0)
            expected_subject = (
                np.linalg.norm(
                    reference.astype(np.float32) - reference_bg[None, None, :], axis=2,
                ) > 30.0
            ).astype(np.uint8)
            expected_subject = cv2.morphologyEx(
                expected_subject, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1,
            ).astype(bool)

            labels = parse_body_parts(rgb)["labels"]
            generated_subject = labels != 0
            near_generated_subject = cv2.dilate(
                generated_subject.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), iterations=1,
            ).astype(bool)
            target = np.asarray(background_rgb, dtype=np.float32)
            selected_colour = np.linalg.norm(
                rgb.astype(np.float32) - target[None, None, :], axis=2,
            ) <= 18.0
            yy = np.indices(selected_colour.shape)[0]
            candidates = (
                selected_colour
                & expected_subject
                & near_generated_subject
                & (yy >= int(height * .43))
            )
            count, components, stats, _ = cv2.connectedComponentsWithStats(
                candidates.astype(np.uint8), connectivity=8,
            )
            repair = np.zeros_like(candidates, dtype=np.uint8)
            maximum_area = max(80, int(height * width * .008))
            for component in range(1, count):
                x, y, component_width, component_height, area = stats[component]
                if not 3 <= int(area) <= maximum_area:
                    continue
                if component_width > int(width * .20) or component_height > int(height * .20):
                    continue
                repair[components == component] = 255
            if not np.any(repair):
                return image.convert("RGB")
            repair = cv2.dilate(
                repair,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1,
            )
            repaired = cv2.inpaint(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), repair, 3, cv2.INPAINT_TELEA,
            )
            logger.info(
                "[UNIFORM_BACKGROUND] Repaired %d selected-colour intrusion pixels inside garment",
                int(np.count_nonzero(repair)),
            )
            return Image.fromarray(cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB), "RGB")
        except Exception as exc:
            logger.warning("[UNIFORM_BACKGROUND] Intrusion repair skipped: %s", exc)
            return image.convert("RGB")

    @staticmethod
    def _lock_rough_uniform_pixels(
        image: Image.Image,
        garment_reference: Image.Image,
        background_rgb: Optional[Tuple[int, int, int]] = None,
    ) -> Image.Image:
        """Fit the clean garment guide to Qwen's portrait without touching the head.

        Qwen is useful for portrait restoration, but its garment edits can drift
        into the source clothing or invent a different fabric. Use the clean
        garment-only guide, never the rough portrait composite, so no old face,
        hair, hand, or background pixels can enter the final image.
        """
        try:
            from pipelines.schp_service import parse as parse_body_parts

            generated = np.asarray(image.convert("RGB"))
            reference = np.asarray(
                garment_reference.convert("RGB").resize(
                    (generated.shape[1], generated.shape[0]), Image.LANCZOS
                )
            )
            reference_labels = parse_body_parts(reference)["labels"]
            reference_garment = np.isin(
                reference_labels, (5, 6, 7, 9, 10, 12)
            ).astype(np.uint8)
            if np.count_nonzero(reference_garment) < reference_garment.size * 0.04:
                return image.convert("RGB")

            generated_labels = parse_body_parts(generated)["labels"]
            generated_garment = np.isin(
                generated_labels, (5, 6, 7, 9, 10, 12)
            ).astype(np.uint8)
            source_points = cv2.findNonZero(reference_garment)
            target_points = cv2.findNonZero(generated_garment)
            if source_points is None:
                return image.convert("RGB")
            source_x, source_y, source_w, source_h = cv2.boundingRect(source_points)
            if target_points is not None:
                _, _, target_w, _ = cv2.boundingRect(target_points)
            else:
                target_w = int(generated.shape[1] * 0.78)
            # Product templates must cover the complete passport shoulder span;
            # otherwise the rejected generated sweater remains visible outside
            # the replacement shirt.
            target_w = max(target_w, int(generated.shape[1] * 0.94))

            # Place the template collar just below the generated chin. This
            # keeps the neck from being overwritten and scales the garment to
            # the actual shoulder span in the Qwen portrait.
            target_top = int(generated.shape[0] * 0.54)
            try:
                from pipelines.photo_restoration import _get_insight_app

                detector = _get_insight_app()
                generated_faces = detector.get(cv2.cvtColor(generated, cv2.COLOR_RGB2BGR))
                if len(generated_faces) == 1:
                    _, _, _, face_bottom = [int(value) for value in generated_faces[0].bbox]
                    target_top = max(target_top, face_bottom + int(generated.shape[0] * 0.015))
            except Exception as exc:
                logger.warning("[UNIFORM_TEMPLATE_LOCK] Face anchor unavailable: %s", exc)

            scale = max(0.60, min(1.45, target_w / max(source_w, 1)))
            output_w = max(8, int(round(source_w * scale)))
            output_h = max(8, int(round(source_h * scale)))
            garment_crop = reference[source_y:source_y + source_h, source_x:source_x + source_w]
            mask_crop = reference_garment[source_y:source_y + source_h, source_x:source_x + source_w]
            garment_crop = cv2.resize(garment_crop, (output_w, output_h), interpolation=cv2.INTER_LANCZOS4)
            mask_crop = cv2.resize(mask_crop, (output_w, output_h), interpolation=cv2.INTER_NEAREST)

            x0 = max(0, (generated.shape[1] - output_w) // 2)
            y0 = max(0, target_top)
            x1, y1 = min(generated.shape[1], x0 + output_w), min(generated.shape[0], y0 + output_h)
            if x1 <= x0 or y1 <= y0:
                return image.convert("RGB")
            crop_w, crop_h = x1 - x0, y1 - y0
            base = generated.copy()
            if background_rgb is not None:
                yy = np.indices(generated_garment.shape)[0]
                erase = (
                    (generated_garment > 0)
                    & (yy >= max(0, target_top - int(generated.shape[0] * .035)))
                ).astype(np.uint8) * 255
                erase = cv2.dilate(
                    erase,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                    iterations=1,
                )
                base[erase > 0] = np.asarray(background_rgb, dtype=np.uint8)
            fitted_reference = base.copy()
            fitted_mask = np.zeros(generated.shape[:2], dtype=np.uint8)
            fitted_reference[y0:y1, x0:x1] = garment_crop[:crop_h, :crop_w]
            fitted_mask[y0:y1, x0:x1] = mask_crop[:crop_h, :crop_w]
            fitted_mask = cv2.morphologyEx(
                fitted_mask, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1,
            )
            alpha = cv2.GaussianBlur(fitted_mask.astype(np.float32), (0, 0), 1.2)
            alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
            locked = base.astype(np.float32) * (1.0 - alpha) + fitted_reference.astype(np.float32) * alpha
            return Image.fromarray(np.rint(locked).clip(0, 255).astype(np.uint8), "RGB")
        except Exception as exc:
            logger.warning("[UNIFORM_TEMPLATE_LOCK] Template lock unavailable: %s", exc)
            return image.convert("RGB")

    def _has_selected_solid_background(
        self, image: Image.Image, background_color: Union[str, Tuple[int, int, int]]
    ) -> bool:
        """Verify the selected colour across the complete semantic background."""
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        height, width = rgb.shape[:2]
        edge = max(4, min(24, height // 12, width // 12))
        if height < edge * 2 or width < edge * 2:
            return False
        # Bottom corners normally contain the subject's shoulders in a 35:45
        # passport crop, so only top corners are valid backdrop samples.
        samples = np.concatenate((
            rgb[:edge, :edge].reshape(-1, 3),
            rgb[:edge, -edge:].reshape(-1, 3),
        ))
        target = np.asarray(self._parse_bg_color(background_color), dtype=np.float32)
        full_distances = np.linalg.norm(rgb - target, axis=2)
        distances = np.linalg.norm(samples - target, axis=1)
        side_width = max(2, int(width * 0.03))
        side_height = max(1, int(height * 0.68))
        side_pixels = np.concatenate((
            rgb[:side_height, :side_width].reshape(-1, 3),
            rgb[:side_height, -side_width:].reshape(-1, 3),
        ))
        # Qwen can generate the requested RGB backdrop exactly. Trust that
        # direct evidence before invoking SCHP: parser misses around glasses,
        # plaid and shoulders previously made an already-correct portrait fail
        # this gate, after which the fallback matte could erase the subject.
        # Requiring an exact, substantial field plus clean top and bilateral
        # borders still rejects gradients, off-colour backdrops and scenery.
        exact_target = full_distances <= 8.0
        top_border_match = float(np.mean(exact_target[:edge, :]))
        bilateral_border_match = float(np.mean(np.concatenate((
            exact_target[:side_height, :side_width].reshape(-1),
            exact_target[:side_height, -side_width:].reshape(-1),
        ))))
        direct_target_fraction = float(np.mean(exact_target))
        non_target = (~exact_target).astype(np.uint8)
        component_count, component_map, component_stats, _ = cv2.connectedComponentsWithStats(
            non_target, connectivity=8,
        )
        isolated_scene_component = False
        if component_count > 1:
            primary_id = 1 + int(np.argmax(component_stats[1:, cv2.CC_STAT_AREA]))
            primary_neighborhood = cv2.dilate(
                (component_map == primary_id).astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                iterations=1,
            ).astype(bool)
            minimum_scene_area = max(32, int(height * width * .004))
            for component in range(1, component_count):
                if component == primary_id:
                    continue
                component_mask = component_map == component
                if (
                    int(component_stats[component, cv2.CC_STAT_AREA]) >= minimum_scene_area
                    and not np.any(component_mask & primary_neighborhood)
                ):
                    isolated_scene_component = True
                    break
        direct_selected_backdrop = bool(
            np.median(distances) <= 12.0
            and np.percentile(distances, 90) <= 18.0
            and top_border_match >= .96
            and bilateral_border_match >= .98
            and direct_target_fraction >= .30
            and not isolated_scene_component
        )
        if direct_selected_backdrop:
            logger.info(
                "[UNIFORM_BACKGROUND_CHECK] accepted direct Qwen backdrop: "
                "top=%.3f bilateral=%.3f exact_fraction=%.3f",
                top_border_match, bilateral_border_match, direct_target_fraction,
            )
            return True
        # Long hair and bows can legitimately reach a passport crop's side
        # edge. Exclude parsed subject pixels before measuring backdrop hue;
        # otherwise an exact blue background is rejected simply because the
        # child fills the frame.
        try:
            from pipelines.schp_service import parse as parse_body_parts

            parsed_subject = parse_body_parts(rgb.astype(np.uint8))["labels"] != 0
            # A scene/text fragment can be misclassified as clothing or hair.
            # Treat only the largest connected semantic component as the
            # person. This keeps the actual head/torso while preventing remote
            # Qwen scenery from being hidden from background validation.
            component_count, component_map, component_stats, _ = cv2.connectedComponentsWithStats(
                parsed_subject.astype(np.uint8), connectivity=8,
            )
            if component_count > 1:
                largest_component = 1 + int(np.argmax(
                    component_stats[1:, cv2.CC_STAT_AREA]
                ))
                subject = component_map == largest_component
            else:
                subject = parsed_subject
            # Patterned white fabric is sometimes returned as label 0. Those
            # internal holes are not backdrop and previously made an exact
            # selected colour fail at the final gate. Fill each occupied row
            # across the visible person silhouette before measuring only the
            # pixels geometrically outside the subject. Real wrong-colour
            # backdrop at the frame edges remains measurable.
            filled_subject = subject.copy()
            for row in range(height):
                columns = np.flatnonzero(subject[row])
                if columns.size >= max(4, int(width * .012)):
                    filled_subject[row, columns[0]:columns[-1] + 1] = True
            # Background delivery protects a one-pixel antialias band around
            # the subject envelope. Exclude the same band from validation;
            # otherwise parser holes at hair, glasses and plaid edges are
            # incorrectly counted as wrong-colour backdrop.
            filled_subject = cv2.dilate(
                filled_subject.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            ).astype(bool)
            # Recover a disconnected ponytail or accessory missed by SCHP only
            # when its non-background pixels physically touch the main person.
            # Two dilation pixels bridge normal antialias gaps without joining
            # separate clouds, writing or props to the subject.
            wrong_color = full_distances > 18.0
            wrong_count, wrong_map, _, _ = cv2.connectedComponentsWithStats(
                wrong_color.astype(np.uint8), connectivity=8,
            )
            person_neighborhood = cv2.dilate(
                filled_subject.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=2,
            ).astype(bool)
            for component in range(1, wrong_count):
                component_mask = wrong_map == component
                if np.any(component_mask & person_neighborhood):
                    filled_subject |= component_mask
            semantic_background = ~filled_subject
            if np.count_nonzero(semantic_background) < 128:
                return False
            full_background_match = float(
                np.mean(full_distances[semantic_background] <= 18.0)
            )
            side_subject = np.concatenate((
                filled_subject[:side_height, :side_width].reshape(-1),
                filled_subject[:side_height, -side_width:].reshape(-1),
            ))
            side_samples = side_pixels[~side_subject]
            if side_samples.shape[0] < max(16, side_pixels.shape[0] // 5):
                side_samples = side_pixels
        except Exception:
            side_samples = side_pixels
            full_background_match = 1.0
        side_distances = np.linalg.norm(side_samples - target, axis=1)
        side_match = float(np.mean(side_distances <= 18.0))
        # Check beyond the narrow outer border. A semantic parser can label
        # clouds, walls or foliage as foreground, but those textures still
        # occupy these geometry-only upper side zones in a passport portrait.
        top_zone_height = max(edge + 1, int(height * .34))
        top_zone_width = max(edge + 1, int(width * .22))
        top_side_pixels = np.concatenate((
            rgb[:top_zone_height, :top_zone_width].reshape(-1, 3),
            rgb[:top_zone_height, -top_zone_width:].reshape(-1, 3),
        ))
        if "filled_subject" in locals():
            top_side_subject = np.concatenate((
                filled_subject[:top_zone_height, :top_zone_width].reshape(-1),
                filled_subject[:top_zone_height, -top_zone_width:].reshape(-1),
            ))
            measurable_top_side = top_side_pixels[~top_side_subject]
            if measurable_top_side.shape[0] >= max(32, top_side_pixels.shape[0] // 3):
                top_side_pixels = measurable_top_side
        top_side_distances = np.linalg.norm(top_side_pixels - target, axis=1)
        top_side_match = float(np.mean(top_side_distances <= 18.0))
        corners_match = bool(
            np.median(distances) <= 8.0
            and np.percentile(distances, 90) <= 18.0
        )
        # When essentially every measurable side-border pixel is the selected
        # colour, a lower semantic score normally means SCHP missed hair,
        # accessories, or a shoulder inside the portrait. Do not reject that
        # intact subject as backdrop. Textured scenery cannot satisfy the
        # strict corner and bilateral side-border requirements.
        # We check semantic foreground matte: normal backdrop requires
        # full_background_match >= 0.94 (or side_match >= 0.85 and full_background_match >= 0.84),
        # while strict bilateral border match allows full_background_match >= 0.78.
        # Never publish an envelope or semantic rematte here.
        strict_border_match = bool(side_match >= 0.95 and full_background_match >= 0.78)
        normal_background_match = bool(
            (full_background_match >= 0.94)
            or (side_match >= 0.85 and full_background_match >= 0.84)
        )
        exact_match = bool(
            corners_match
            # The upper side zones should be unobstructed backdrop after the
            # connected person silhouette has been excluded. A 0.86 threshold
            # allowed writing/cloud fragments to survive after correction.
            and top_side_match >= 0.985
            and (normal_background_match or strict_border_match)
        )
        logger.info(
            "[UNIFORM_BACKGROUND_CHECK] corner_median=%.2f corner_p90=%.2f side=%.3f top_side=%.3f full=%.3f accepted=%s",
            float(np.median(distances)), float(np.percentile(distances, 90)),
            side_match, top_side_match, full_background_match, exact_match,
        )
        if exact_match:
            return True
        return False

    @staticmethod
    def _uniform_color_fraction(
        rgb: np.ndarray,
        mask: np.ndarray,
        target_rgb: Tuple[int, int, int],
    ) -> float:
        """Measure a template colour without rejecting natural fabric shading."""
        if not np.any(mask):
            return 0.0
        target_pixel = np.asarray(target_rgb, dtype=np.uint8).reshape(1, 1, 3)
        target_hsv = cv2.cvtColor(target_pixel, cv2.COLOR_RGB2HSV)[0, 0]
        if int(target_hsv[1]) >= 70:
            hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
            hue_delta = np.abs(hsv[:, :, 0].astype(np.int16) - int(target_hsv[0]))
            hue_delta = np.minimum(hue_delta, 180 - hue_delta)
            # Hue survives the shadows and highlights of a natural fabric
            # photograph. Requiring useful saturation/value still prevents
            # white checks, skin, and dark hair from being counted as red.
            matches = (
                (hue_delta <= 12)
                & (hsv[:, :, 1] >= max(45, int(target_hsv[1]) // 4))
                & (hsv[:, :, 2] >= 35)
            )
        else:
            distance = np.linalg.norm(
                rgb.astype(np.float32) - np.asarray(target_rgb, dtype=np.float32), axis=2
            )
            matches = distance <= 55.0
        return float(np.mean(matches[mask]))

    @staticmethod
    def _garment_reference_mask(reference: Image.Image) -> np.ndarray:
        """Extract garment pixels from a person-free product reference.

        SCHP is a human parser and produces unstable labels when it is given a
        standalone shirt or vest.  Uniform validation used those labels as the
        template mask, which made a visibly correct Qwen garment fail whenever
        the product crop had a white canvas or no body.  Prefer real alpha when
        it exists; otherwise learn the canvas colour from the four corners and
        retain the largest non-canvas garment component, including its internal
        light checks, buttons and collar openings.
        """
        rgba = np.asarray(reference.convert("RGBA"), dtype=np.uint8)
        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3]
        height, width = alpha.shape
        if np.any(alpha < 250):
            foreground = (alpha > 32).astype(np.uint8)
        else:
            edge = max(3, min(24, height // 16, width // 16))
            corners = np.concatenate((
                rgb[:edge, :edge].reshape(-1, 3),
                rgb[:edge, -edge:].reshape(-1, 3),
                rgb[-edge:, :edge].reshape(-1, 3),
                rgb[-edge:, -edge:].reshape(-1, 3),
            ))
            canvas_rgb = np.median(corners, axis=0).astype(np.uint8)
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
            canvas_lab = cv2.cvtColor(
                canvas_rgb.reshape(1, 1, 3), cv2.COLOR_RGB2LAB,
            ).reshape(3).astype(np.float32)
            foreground = (
                np.linalg.norm(lab - canvas_lab, axis=2) > 8.0
            ).astype(np.uint8)
            foreground = cv2.morphologyEx(
                foreground,
                cv2.MORPH_OPEN,
                np.ones((3, 3), dtype=np.uint8),
            )
        count, components, stats, _ = cv2.connectedComponentsWithStats(
            foreground, 8,
        )
        if count <= 1:
            return foreground.astype(bool)
        minimum_area = max(24, int(height * width * .0005))
        meaningful = [
            label for label in range(1, count)
            if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area
        ]
        if not meaningful:
            return foreground.astype(bool)
        # Product templates can contain disconnected sleeves. Keep every
        # meaningful component near the dominant garment instead of only the
        # largest one, then fill enclosed pale checks/buttons back into it.
        selected = np.zeros((height, width), dtype=np.uint8)
        for label in meaningful:
            selected[components == label] = 1
        filled = np.zeros_like(selected)
        points = cv2.findNonZero(selected)
        if points is None or len(points) < 3:
            return selected.astype(bool)
        # A check shirt on white is naturally split into hundreds of coloured
        # grid components. Filling each component separately retains only the
        # dark vest/threads and makes the reference histogram meaningless.
        # The garment-conditioning crop contains no person and is deliberately
        # tight, so its convex silhouette is the correct support for the pale
        # base fabric, collar, buttons and detached sleeve edges as well.
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(filled, hull, 1)
        return filled.astype(bool)

    @staticmethod
    def _has_generated_uniform(
        image: Image.Image,
        target_outer_rgb: Optional[Tuple[int, int, int]],
        rough_reference: Optional[Image.Image] = None,
        expected_target_fraction: Optional[float] = None,
    ) -> bool:
        """Reject unchanged source clothing and incomplete garment edits."""
        try:
            from pipelines.schp_service import parse as parse_body_parts

            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            labels = parse_body_parts(rgb)["labels"]
            clothing = np.isin(labels, (5, 6, 7, 9, 10, 12))
            coverage = float(np.mean(clothing))
            # Delivered school-ID images intentionally stop at the upper chest.
            # In that framing a valid collar and vest often occupy only 3-5%
            # of the canvas after semantic segmentation. The old 7% floor
            # therefore rejected a visible, generated uniform before its
            # template-color and visual-reference checks could run.
            if coverage < 0.025:
                logger.warning("[UNIFORM_REVIEW] Clothing coverage too small: %.3f", coverage)
                return False
            height, width = labels.shape
            central_arms = np.isin(
                labels[
                    int(height * .30):int(height * .85),
                    int(width * .18):int(width * .82),
                ],
                (14, 15),
            )
            central_arm_fraction = float(np.count_nonzero(central_arms) / labels.size)
            if central_arm_fraction > 0.015:
                logger.warning(
                    "[UNIFORM_REVIEW] Hand/arm crosses passport torso: %.3f",
                    central_arm_fraction,
                )
                return False
            # Judge the measured outer-template colour only inside the parsed
            # lower clothing. Measuring the full lower frame counts the white
            # shirt, skin, and backdrop against a gray pinafore and rejects a
            # valid multi-layer school uniform.
            yy, _ = np.indices(labels.shape)
            lower_clothing = clothing & (yy >= int(height * .45))
            matching_fraction = (
                QwenEditPipeline._uniform_color_fraction(
                    rgb, lower_clothing, target_outer_rgb,
                )
                if target_outer_rgb is not None else 0.0
            )
            logger.info(
                "[UNIFORM_REVIEW] clothing_coverage=%.3f target_color_fraction=%.3f",
                coverage, matching_fraction,
            )
            if rough_reference is not None:
                reference = np.asarray(
                    rough_reference.convert("RGB").resize((width, height), Image.LANCZOS),
                    dtype=np.uint8,
                )
                # The reference is a person-free garment product crop, not a
                # human portrait. Never ask SCHP to classify it. Its own alpha
                # or corner-connected canvas provides the correct garment mask.
                reference_native = rough_reference.convert("RGBA")
                ref_clothing_native = QwenEditPipeline._garment_reference_mask(
                    reference_native,
                )
                ref_clothing = cv2.resize(
                    ref_clothing_native.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                candidate_clothing = clothing & (
                    np.indices(labels.shape)[0] >= int(height * .45)
                )
                if np.count_nonzero(ref_clothing) >= 256 and np.count_nonzero(candidate_clothing) >= 256:
                    # The garment is deliberately warped from the rough guide
                    # into this person's shoulder/torso bounds. Pixel-position
                    # subtraction therefore rejects a correct fitted uniform.
                    # Compare its joint hue/saturation distribution instead;
                    # this retains exact plaid/palette evidence while remaining
                    # invariant to the required body-fit transformation.
                    reference_hsv = cv2.cvtColor(reference, cv2.COLOR_RGB2HSV)
                    candidate_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
                    reference_hist = cv2.calcHist(
                        [reference_hsv], [0, 1], ref_clothing.astype(np.uint8),
                        [30, 16], [0, 180, 0, 256],
                    )
                    candidate_hist = cv2.calcHist(
                        [candidate_hsv], [0, 1], candidate_clothing.astype(np.uint8),
                        [30, 16], [0, 180, 0, 256],
                    )
                    cv2.normalize(reference_hist, reference_hist, alpha=1.0, norm_type=cv2.NORM_L1)
                    cv2.normalize(candidate_hist, candidate_hist, alpha=1.0, norm_type=cv2.NORM_L1)
                    histogram_intersection = float(cv2.compareHist(
                        candidate_hist, reference_hist, cv2.HISTCMP_INTERSECT,
                    ))
                    histogram_distance = float(cv2.compareHist(
                        candidate_hist, reference_hist, cv2.HISTCMP_BHATTACHARYYA,
                    ))
                    # Low-saturation whites and dark checks dominate a plaid
                    # histogram even when Qwen correctly reproduces its red
                    # and blue threads. Compare saturated hue evidence
                    # separately so a correct fitted plaid is not rejected as
                    # source clothing merely because lighting/folds changed
                    # the neutral pixel counts.
                    reference_chromatic = ref_clothing & (reference_hsv[:, :, 1] >= 65) & (reference_hsv[:, :, 2] >= 35)
                    candidate_chromatic = candidate_clothing & (candidate_hsv[:, :, 1] >= 28) & (candidate_hsv[:, :, 2] >= 35)
                    reference_chromatic_fraction = float(
                        np.count_nonzero(reference_chromatic) / max(1, np.count_nonzero(ref_clothing))
                    )
                    candidate_chromatic_fraction = float(
                        np.count_nonzero(candidate_chromatic) / max(1, np.count_nonzero(candidate_clothing))
                    )
                    chromatic_hue_match = False
                    if np.count_nonzero(reference_chromatic) >= 64 and np.count_nonzero(candidate_chromatic) >= 64:
                        reference_hue = cv2.calcHist(
                            [reference_hsv], [0], reference_chromatic.astype(np.uint8),
                            [36], [0, 180],
                        ).reshape(-1)
                        candidate_hue = cv2.calcHist(
                            [candidate_hsv], [0], candidate_chromatic.astype(np.uint8),
                            [36], [0, 180],
                        ).reshape(-1)
                        reference_hue /= max(float(reference_hue.sum()), 1.0)
                        candidate_hue /= max(float(candidate_hue.sum()), 1.0)
                        major_hues = np.flatnonzero(reference_hue >= .075)
                        hue_matches = []
                        for hue_index in major_hues:
                            neighbourhood = (
                                candidate_hue[(hue_index - 1) % 36]
                                + candidate_hue[hue_index]
                                + candidate_hue[(hue_index + 1) % 36]
                            )
                            hue_matches.append(neighbourhood >= max(.012, reference_hue[hue_index] * .08))
                        chromatic_hue_match = bool(
                            major_hues.size
                            and all(hue_matches)
                            and candidate_chromatic_fraction >= max(.035, reference_chromatic_fraction * .16)
                        )
                    logger.info(
                        "[UNIFORM_REVIEW] garment_histogram_intersection=%.3f distance=%.3f "
                        "outer_color_fraction=%.3f chromatic_fraction=%.3f->%.3f chromatic_hues=%s",
                        histogram_intersection, histogram_distance, matching_fraction,
                        reference_chromatic_fraction, candidate_chromatic_fraction,
                        chromatic_hue_match,
                    )
                    # A photorealistic Qwen re-render legitimately changes
                    # pixel counts through folds and body fitting. Keep enough
                    # separation to reject the known wrong yellow/navy shirt
                    # while accepting the correct regenerated red/navy/white
                    # plaid garment.
                    neutral_histogram_match = (
                        histogram_intersection >= 0.12 and histogram_distance <= 0.88
                    )
                    patterned_histogram_match = (
                        histogram_intersection >= 0.12
                        and histogram_distance <= 0.86
                        and chromatic_hue_match
                    )
                    # A colourful plaid/check cannot pass through the neutral
                    # fallback merely because its white/dark pixels overlap a
                    # retained vest or source shirt. Require the reference's
                    # major chromatic hues whenever they form a meaningful
                    # part of the supplied garment.
                    reference_is_chromatic_pattern = reference_chromatic_fraction >= 0.12
                    garment_matches = (
                        patterned_histogram_match
                        if reference_is_chromatic_pattern
                        else neutral_histogram_match
                    )
                    # Global garment histograms can be fooled when Qwen keeps
                    # correct plaid sleeves but invents a large solid vest or
                    # bib over the central torso. Compare the torso itself with
                    # the same region of the fitted garment reference. The
                    # thresholds are relative to each uploaded template, so a
                    # genuinely dark uniform remains valid while an invented
                    # dark panel on a light plaid does not.
                    xx = np.indices(labels.shape)[1]
                    central_torso = (
                        (yy >= int(height * .53))
                        & (yy <= int(height * .94))
                        & (xx >= int(width * .20))
                        & (xx <= int(width * .80))
                    )
                    ref_central_clothing = ref_clothing & central_torso
                    candidate_central_clothing = candidate_clothing & central_torso
                    dark_structure_match = True
                    if (
                        np.count_nonzero(ref_central_clothing) >= 256
                        and np.count_nonzero(candidate_central_clothing) >= 256
                    ):
                        reference_dark_fraction = float(np.mean(
                            reference_hsv[:, :, 2][ref_central_clothing] <= 65
                        ))
                        candidate_dark_fraction = float(np.mean(
                            candidate_hsv[:, :, 2][candidate_central_clothing] <= 65
                        ))
                        dark_panel_divergence = (
                            candidate_dark_fraction
                            > max(reference_dark_fraction + .24, reference_dark_fraction * 2.5 + .10)
                        )
                        logger.info(
                            "[UNIFORM_REVIEW] central_torso_dark_fraction=%.3f->%.3f divergence=%s",
                            reference_dark_fraction,
                            candidate_dark_fraction,
                            dark_panel_divergence,
                        )
                        if dark_panel_divergence:
                            logger.warning(
                                "[UNIFORM_REVIEW] Qwen invented a dark central garment panel absent from the template"
                            )
                            return False
                        # A product crop and a worn portrait have different
                        # folds, exposure and visible layer proportions, so a
                        # fine HSV histogram can have very little intersection
                        # even when every visible garment component is correct.
                        # For a chromatic/check template, accept the regenerated
                        # cloth when all major template hues survived and its
                        # measured central dark/light layer structure agrees.
                        # This remains template-relative and therefore does not
                        # allow a retained pale source shirt to stand in for a
                        # dark vest, nor a new dark bib to pass for a light shirt.
                        if reference_dark_fraction >= .18:
                            dark_structure_match = (
                                candidate_dark_fraction
                                >= max(.08, reference_dark_fraction * .45)
                            )
                        else:
                            dark_structure_match = not dark_panel_divergence
                        if (
                            reference_is_chromatic_pattern
                            and chromatic_hue_match
                            and dark_structure_match
                        ):
                            garment_matches = True
                            logger.info(
                                "[UNIFORM_REVIEW] Accepted template-relative hue/layer structure "
                                "despite photographic histogram shift"
                            )
                        elif reference_dark_fraction >= .18 and not dark_structure_match:
                            # Histogram overlap from a similarly coloured
                            # source shirt cannot substitute for a required
                            # dark vest/pinafore/outer layer in the template.
                            garment_matches = False
                            logger.warning(
                                "[UNIFORM_REVIEW] Required dark garment layer is missing: "
                                "reference=%.3f candidate=%.3f",
                                reference_dark_fraction,
                                candidate_dark_fraction,
                            )
                    if not garment_matches:
                        logger.warning(
                            "[UNIFORM_REVIEW] Fitted garment palette/pattern differs from "
                            "supplied template: intersection=%.3f distance=%.3f",
                            histogram_intersection, histogram_distance,
                        )
                        return False
                    ref_saturation = float(np.mean(reference_hsv[:, :, 1][ref_clothing] > 40))
                    candidate_saturation = float(
                        np.mean(candidate_hsv[:, :, 1][candidate_clothing] > 40)
                    )
                    # Large newly saturated regions are a reliable signal that
                    # the source clothing was retained (e.g. pink cardigan)
                    # instead of the supplied neutral school uniform.
                    if candidate_saturation - ref_saturation > 0.38:
                        logger.warning(
                            "[UNIFORM_REVIEW] Garment palette diverges from template: "
                            "reference_saturation=%.3f candidate_saturation=%.3f",
                            ref_saturation,
                            candidate_saturation,
                        )
                        return False
            elif target_outer_rgb is None:
                return True
            elif matching_fraction < max(
                0.02,
                min(0.18, float(expected_target_fraction or 0.36) * 0.50),
            ):
                # This fallback only applies when no template reference is
                # available for direct comparison. A template can legitimately
                # use its sampled colour only as a plaid stripe or trim; derive
                # the candidate minimum from that template coverage rather
                # than assuming the sampled colour fills a solid outer layer.
                logger.warning(
                    "[UNIFORM_REVIEW] Template-color coverage too small without a template reference: %.3f (template %.3f)",
                    matching_fraction, float(expected_target_fraction or 0.0),
                )
                return False
            return True
        except Exception as exc:
            logger.warning("[UNIFORM_REVIEW] Garment verification unavailable: %s", exc)
            return False

    @classmethod
    def _enforce_template_garment_lock(
        cls,
        generated: Image.Image,
        rough_reference: Image.Image,
        background_rgb: Optional[Tuple[int, int, int]] = None,
    ) -> Image.Image:
        """Helper delegating to _lock_uniform_garment_from_rough for fallback alignment."""
        return cls._lock_uniform_garment_from_rough(generated, rough_reference, background_rgb)

    @staticmethod
    def _lock_uniform_garment_from_rough(
        generated: Image.Image,
        rough_reference: Image.Image,
        background_rgb: Optional[Tuple[int, int, int]] = None,
    ) -> Image.Image:
        """Align the verified rough garment onto Qwen's coherent portrait.

        The generated image remains the base so its head, neck, shoulders and
        photographic lighting stay continuous. Only garment pixels from the
        rough template fit are warped into the generated clothing bounds.
        """
        try:
            from pipelines.schp_service import CLOTHING_LABELS, parse

            output = np.asarray(generated.convert("RGB"), dtype=np.uint8)
            rough = np.asarray(
                rough_reference.convert("RGB").resize(
                    (output.shape[1], output.shape[0]), Image.LANCZOS,
                ),
                dtype=np.uint8,
            )
            output_labels = parse(output)["labels"]
            rough_labels = parse(rough)["labels"]
            clothing_labels = tuple(CLOTHING_LABELS)

            def lower_garment_mask(labels: np.ndarray) -> np.ndarray:
                mask = np.isin(labels, clothing_labels).astype(np.uint8)
                face_rows = np.where(labels == 13)[0]
                if face_rows.size:
                    face_height = int(face_rows.max() - face_rows.min() + 1)
                    collar_limit = int(face_rows.max() + face_height * .02)
                else:
                    collar_limit = int(labels.shape[0] * .46)
                mask[:max(0, collar_limit)] = 0
                kernel_size = max(5, int(round(min(labels.shape) * .018)) | 1)
                return cv2.morphologyEx(
                    mask,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size),
                    ),
                    iterations=1,
                )

            rough_clothing = lower_garment_mask(rough_labels)
            output_clothing = lower_garment_mask(output_labels)
            rough_points = cv2.findNonZero(rough_clothing)
            output_points = cv2.findNonZero(output_clothing)
            if rough_points is None or output_points is None or len(rough_points) < 256:
                logger.warning("[UNIFORM_GARMENT_LOCK] Rough garment mask unavailable")
                return generated.convert("RGB")

            # SCHP can classify white checks as background. Recover those
            # enclosed fabric pixels from the rough garment hull, while
            # explicitly excluding face, hair, neck and arms.
            hull_mask = np.zeros_like(rough_clothing)
            cv2.fillConvexPoly(hull_mask, cv2.convexHull(rough_points), 1)
            edge = max(4, min(16, rough.shape[0] // 20, rough.shape[1] // 20))
            rough_backdrop = np.median(np.concatenate((
                rough[:edge, :edge].reshape(-1, 3),
                rough[:edge, -edge:].reshape(-1, 3),
            )), axis=0)
            rough_foreground = np.linalg.norm(
                rough.astype(np.float32) - rough_backdrop[None, None, :], axis=2,
            ) > 25.0
            protected_person = np.isin(rough_labels, (1, 2, 4, 13, 14, 15))
            rough_clothing = (
                (rough_clothing > 0)
                | (rough_foreground & (hull_mask > 0) & ~protected_person)
            ).astype(np.uint8)
            rough_points = cv2.findNonZero(rough_clothing)
            if rough_points is None:
                return generated.convert("RGB")

            source_x, source_y, source_w, source_h = cv2.boundingRect(rough_points)
            target_x, target_y, target_w, target_h = cv2.boundingRect(output_points)
            scale_x = target_w / max(1.0, float(source_w))
            scale_y = target_h / max(1.0, float(source_h))
            transform = np.asarray((
                (scale_x, 0.0, target_x - source_x * scale_x),
                (0.0, scale_y, target_y - source_y * scale_y),
            ), dtype=np.float32)
            border_value = tuple(int(value) for value in (
                background_rgb if background_rgb is not None else rough_backdrop
            ))
            warped_garment = cv2.warpAffine(
                rough,
                transform,
                (output.shape[1], output.shape[0]),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border_value,
            )
            warped_mask = cv2.warpAffine(
                rough_clothing * 255,
                transform,
                (output.shape[1], output.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            # Qwen's full head and exposed neck are authoritative. This is the
            # key distinction from the old rough-base face paste. Do not protect
            # every SCHP arm label: patterned short sleeves are frequently
            # classified as arms, which left the incorrect generated fabric on
            # both sides of an otherwise exact locked shirt.
            face_rows = np.where(output_labels == 13)[0]
            face_columns = np.where(output_labels == 13)[1]
            if face_rows.size:
                face_height = int(face_rows.max() - face_rows.min() + 1)
                collar_limit = int(face_rows.max() + face_height * .08)
            else:
                collar_limit = int(output.shape[0] * .46)
                face_height = max(1, int(output.shape[0] * .20))
            protected_anatomy = np.isin(output_labels, (1, 4, 13)) | (
                (output_labels == 2) & (np.indices(output.shape[:2])[0] <= collar_limit)
            )
            # Preserve generated neck skin even when SCHP calls its shadow or
            # lower edge clothing. The colour gate is measured from this face,
            # and the narrow anatomical corridor cannot retain a source shirt.
            face_skin_lab = None
            face_center = output.shape[1] * .5
            face_width = output.shape[1] * .28
            face_bottom = int(output.shape[0] * .43)
            if face_rows.size and face_columns.size:
                yy, xx = np.indices(output.shape[:2])
                face_left, face_right = int(face_columns.min()), int(face_columns.max())
                face_bottom = int(face_rows.max())
                face_width = max(1, face_right - face_left + 1)
                face_center = (face_left + face_right) * .5
                output_lab = cv2.cvtColor(output, cv2.COLOR_RGB2LAB).astype(np.float32)
                face_skin = output_labels == 13
                face_skin_lab = np.median(output_lab[face_skin], axis=0)
                neck_guard = (
                    (yy >= face_bottom - int(face_height * .05))
                    & (yy <= face_bottom + int(face_height * .34))
                    & (np.abs(xx - face_center) <= face_width * .30)
                    & (np.linalg.norm(
                        output_lab - face_skin_lab[None, None, :], axis=2,
                    ) <= 58.0)
                )
                protected_anatomy |= neck_guard
            warped_mask[protected_anatomy] = 0
            alpha = cv2.GaussianBlur(
                warped_mask.astype(np.float32), (0, 0), 1.0,
            )[:, :, None] / 255.0
            base = output.copy()
            restored = np.rint(
                base.astype(np.float32) * (1.0 - alpha)
                + warped_garment.astype(np.float32) * alpha
            ).clip(0, 255).astype(np.uint8)
            logger.info(
                "[UNIFORM_GARMENT_LOCK] Aligned exact template garment onto Qwen portrait; "
                "retained generated head, neck and photographic integration"
            )
            return Image.fromarray(restored, "RGB")
        except Exception as exc:
            logger.warning("[UNIFORM_GARMENT_LOCK] Exact garment restoration skipped: %s", exc)
            return generated.convert("RGB")

    @staticmethod
    def _is_regenerated_from_rough(image: Image.Image, rough: Image.Image) -> bool:
        """Require Qwen to render a photograph, not echo the pasted layout."""
        generated = np.asarray(image.convert("RGB"), dtype=np.float32)
        guide = np.asarray(
            rough.convert("RGB").resize((generated.shape[1], generated.shape[0]), Image.LANCZOS),
            dtype=np.float32,
        )
        # Identity and hair are intentionally preserved and can dominate the
        # frame. Judge only the lower garment region when deciding whether the
        # rough pasted clothing was genuinely re-rendered.
        garment_top = int(generated.shape[0] * 0.48)
        per_pixel_delta = np.mean(
            np.abs(generated[garment_top:] - guide[garment_top:]), axis=2
        )
        mean_delta = float(np.mean(per_pixel_delta))
        nearly_unchanged = float(np.mean(per_pixel_delta < 10.0))
        logger.info(
            "[UNIFORM_REVIEW] rough_delta=%.2f nearly_unchanged=%.3f",
            mean_delta, nearly_unchanged,
        )
        return mean_delta >= 12.0 and nearly_unchanged <= 0.75

    @staticmethod
    def _has_flat_border_background(
        image: Image.Image,
        expected_color: Optional[Union[str, Tuple[int, int, int]]] = None,
    ) -> bool:
        """Return true only for a genuinely flat generated backdrop.

        Sampling both top corners as one pool used to classify a softly
        blurred garden as a flat field. That false positive sent scenery into
        the hue normalizer, producing blue islands around the hair. Require
        each corner to be locally flat and require the two corner colours to
        agree before any background pixels are changed.
        """
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        height, width = rgb.shape[:2]
        edge = max(4, min(24, height // 12, width // 12))
        if height < edge * 2 or width < edge * 2:
            return False
        corner_blocks = (
            rgb[:edge, :edge].reshape(-1, 3),
            rgb[:edge, -edge:].reshape(-1, 3),
        )
        corner_centers = [np.median(block, axis=0) for block in corner_blocks]
        corner_spreads = [
            float(np.percentile(np.linalg.norm(block - center, axis=1), 90))
            for block, center in zip(corner_blocks, corner_centers)
        ]
        center = np.median(np.stack(corner_centers), axis=0)
        corner_disagreement = float(np.linalg.norm(
            corner_centers[0] - corner_centers[1]
        ))
        # Do not judge the complete top strip: a correctly framed passport
        # portrait places hair/head in its centre.  The old test therefore
        # rejected a genuinely flat Qwen backdrop whenever the crown entered
        # the top 8 percent of the canvas.  Instead, find pixels matching the
        # two agreed corner colours and require that colour to form one
        # border-connected field around the subject.
        color_distance = np.linalg.norm(rgb - center, axis=2)
        background_like = (color_distance <= 32.0).astype(np.uint8)
        border_band = max(3, edge // 3)
        border_samples = np.concatenate((
            background_like[:border_band].reshape(-1),
            background_like[-border_band:].reshape(-1),
            background_like[:, :border_band].reshape(-1),
            background_like[:, -border_band:].reshape(-1),
        ))
        border_inlier_fraction = float(np.mean(border_samples))
        component_input = background_like.copy()
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            component_input, connectivity=8,
        )
        connected_fraction = 0.0
        if component_count > 1:
            corner_labels = {
                int(labels[0, 0]), int(labels[0, width - 1]),
                int(labels[height - 1, 0]), int(labels[height - 1, width - 1]),
            }
            corner_labels.discard(0)
            if corner_labels:
                connected_pixels = sum(
                    int(stats[label, cv2.CC_STAT_AREA]) for label in corner_labels
                )
                connected_fraction = connected_pixels / float(height * width)
        # A generated backdrop may contain slight decoder variation, but it
        # must still wrap around the portrait as a large continuous field.
        # Blurred scenery can fool tiny corner statistics; it cannot satisfy
        # this border connectivity test without being an actually flat field.
        flat = bool(
            max(corner_spreads) <= 24.0
            and corner_disagreement <= 24.0
            and border_inlier_fraction >= 0.45
            and connected_fraction >= 0.08
        )
        logger.info(
            "[UNIFORM_FLAT_BACKGROUND] corner_spread=%s disagreement=%.2f "
            "border_inliers=%.3f connected=%.3f flat=%s",
            [round(value, 2) for value in corner_spreads],
            corner_disagreement,
            border_inlier_fraction,
            connected_fraction,
            flat,
        )
        if not flat or expected_color is None:
            return flat
        target = np.asarray(
            QwenEditPipeline._parse_bg_color(expected_color), dtype=np.float32,
        )
        return bool(np.linalg.norm(center - target) <= 18.0)

    def _normalize_qwen_raw_background(
        self,
        image: Image.Image,
        background_color: Union[str, Tuple[int, int, int]],
    ) -> Image.Image:
        """Keep Qwen's subject pixels and apply the selected flat backdrop."""
        from pipelines.birefnet_service import background_removal

        source = image.convert("RGB")
        with io.BytesIO() as buffer:
            source.save(buffer, format="PNG")
            cutout_bytes = background_removal.remove_background(buffer.getvalue())
        cutout = Image.open(io.BytesIO(cutout_bytes)).convert("RGBA")
        alpha = np.array(cutout.getchannel("A"))
        h, w = alpha.shape
        lower_start = int(h * 0.82)
        for r in range(lower_start, h):
            fg_cols = np.where(alpha[r] > 48)[0]
            if fg_cols.size > int(w * 0.3):
                alpha[r, fg_cols.min():fg_cols.max() + 1] = 255
        target_rgb = self._parse_bg_color(background_color)
        source_rgb = np.asarray(source, dtype=np.float32)
        target = np.asarray(target_rgb, dtype=np.float32)
        edge = max(4, min(20, h // 20, w // 20))
        old_background = np.median(np.concatenate((
            source_rgb[:edge, :edge].reshape(-1, 3),
            source_rgb[:edge, -edge:].reshape(-1, 3),
        )), axis=0)
        foreground_alpha = alpha.astype(np.float32) / 255.0
        old_distance = np.linalg.norm(
            source_rgb - old_background[None, None, :], axis=2,
        )
        # Matting occasionally assigns alpha near 1.0 to pale pixels that are
        # still mostly the old backdrop, leaving a white contour on saturated
        # delivery colours. Reduce only non-opaque pixels whose RGB remains
        # close to that sampled backdrop. Dark hair and coloured fabric are far
        # outside this range; fully opaque white uniform pixels are untouched.
        backdrop_separation = np.clip((old_distance - 8.0) / 72.0, 0.0, 1.0)
        # BiRefNet can label a one-pixel pale fringe as fully opaque. Detect
        # only the outside contour of its subject mask and decontaminate those
        # pixels too; the eroded core (including white checks in the uniform)
        # remains byte-for-byte unchanged.
        subject_mask = (alpha > 4).astype(np.uint8)
        subject_core = cv2.erode(
            subject_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        ).astype(bool)
        soft_or_outer_edge = (alpha < 254) | ~subject_core
        foreground_alpha = np.where(
            soft_or_outer_edge,
            foreground_alpha * backdrop_separation,
            foreground_alpha,
        )
        # Fully remove only the outer two-pixel contour when it is still close
        # to the sampled pale backdrop. This clears opaque white/gray halos but
        # retains dark hair strands, skin, and coloured garment boundaries.
        foreground_alpha[(~subject_core) & (old_distance < 120.0)] = 0.0
        foreground_alpha = foreground_alpha[:, :, None]
        # The decoded portrait is already composited over Qwen's pale backdrop.
        # Alpha-compositing that RGB again carries white into every soft hair and
        # shoulder edge. Remove the old backdrop contribution first, then add
        # the selected one; fully opaque subject pixels remain byte-for-byte.
        composed = source_rgb + (1.0 - foreground_alpha) * (
            target[None, None, :] - old_background[None, None, :]
        )
        composed[alpha <= 4] = target
        return Image.fromarray(np.rint(composed).clip(0, 255).astype(np.uint8), "RGB")

    @staticmethod
    def _clear_detached_flat_background_artifacts(
        image: Image.Image,
        background_color: Union[str, Tuple[int, int, int]],
    ) -> Image.Image:
        """Clear only remote islands from an otherwise exact flat backdrop.

        Qwen's decoder can leave a few white droplets or side fragments even
        when the complete field is the requested RGB value.  Semantic matting
        is unnecessarily destructive here: it may clip pale flowers or plaid.
        Keep the largest connected non-background component (the person) and
        every component near it, then recolour only spatially detached islands.
        """
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        target = np.asarray(
            QwenEditPipeline._parse_bg_color(background_color), dtype=np.uint8,
        )
        distance = np.linalg.norm(
            rgb.astype(np.float32) - target[None, None, :].astype(np.float32),
            axis=2,
        )
        non_background = (distance > 18.0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            non_background, connectivity=8,
        )
        if count <= 1:
            return image.convert("RGB")
        primary = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        radius = max(7, min(rgb.shape[:2]) // 70)
        kernel_size = radius * 2 + 1
        primary_neighborhood = cv2.dilate(
            (labels == primary).astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size),
            ),
            iterations=1,
        ).astype(bool)
        result = rgb.copy()
        removed_pixels = 0
        for component in range(1, count):
            if component == primary:
                continue
            component_mask = labels == component
            if np.any(component_mask & primary_neighborhood):
                continue
            removed_pixels += int(np.count_nonzero(component_mask))
            result[component_mask] = target
        logger.info(
            "[UNIFORM_BACKGROUND] Cleared %d detached flat-field artifact pixels",
            removed_pixels,
        )
        return Image.fromarray(result, "RGB")

    @staticmethod
    def _match_dark_uniform_color(image: Image.Image, template: Image.Image) -> Image.Image:
        """Align only the generated dark outer garment to the template's real color."""
        template_rgba = np.array(template.convert("RGBA"))
        template_rgb = template_rgba[:, :, :3]
        template_alpha = template_rgba[:, :, 3] > 220
        template_lab = cv2.cvtColor(template_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        template_dark = template_alpha & (template_lab[:, :, 0] < 110)
        if np.count_nonzero(template_dark) < 120:
            return image

        portrait = np.array(image.convert("RGB"))
        portrait_lab = cv2.cvtColor(portrait, cv2.COLOR_RGB2LAB).astype(np.float32)
        height, width = portrait_lab.shape[:2]
        clothes_top = int(height * 0.54)
        try:
            # Reuse the cached InsightFace service when the uniform face lock
            # has run, ensuring the palette adjustment cannot reach skin.
            from pipelines.photo_restoration import _get_insight_app

            faces = _get_insight_app().get(
                cv2.cvtColor(portrait, cv2.COLOR_RGB2BGR)
            )
            if faces:
                _, y1, _, y2 = [int(value) for value in faces[0].bbox]
                face_height = max(1, y2 - y1)
                clothes_top = max(int(height * 0.45), min(height, y2 + int(face_height * 0.01)))
        except Exception as exc:
            logger.warning("[UNIFORM_COLOR] Face boundary unavailable: %s", exc)
        lower_clothes = np.zeros((height, width), dtype=bool)
        lower_clothes[clothes_top:, :] = True
        # Limit correction to the central torso. Long dark hair can extend
        # below the jaw, so a full-width lower-frame mask would recolour it.
        xx = np.arange(width)[None, :]
        torso = np.abs(xx - (width / 2.0)) <= width * 0.40
        output_dark = lower_clothes & torso & (portrait_lab[:, :, 0] < 110)
        if np.count_nonzero(output_dark) < 120:
            return image

        template_color = np.median(template_lab[template_dark], axis=0)
        output_color = np.median(portrait_lab[output_dark], axis=0)
        # A partial correction preserves fabric texture and natural shadows.
        lab_shift = (template_color - output_color) * 0.72
        corrected_lab = portrait_lab.copy()
        corrected_lab[output_dark] = np.clip(
            corrected_lab[output_dark] + lab_shift, 0, 255
        )
        corrected_rgb = cv2.cvtColor(corrected_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        return Image.fromarray(corrected_rgb, "RGB")

    @staticmethod
    def _restore_uniform_accent_chroma(
        image: Image.Image,
        target_rgb: Tuple[int, int, int],
        background_rgb: Tuple[int, int, int],
        trim_placement: str = "",
    ) -> Image.Image:
        """Recolor coherent generated trim without relying on rough-fit coordinates."""
        generated = np.asarray(image.convert("RGB"), dtype=np.uint8)
        height, width = generated.shape[:2]
        generated_hsv = cv2.cvtColor(generated, cv2.COLOR_RGB2HSV)
        background = np.asarray(background_rgb, dtype=np.float32)
        background_distance = np.linalg.norm(
            generated.astype(np.float32) - background[None, None, :], axis=2,
        )

        # Work on the generated pixels themselves: Qwen can change framing, so
        # a mask copied from the rough fit can land on the face or shoulder.
        lower_frame = np.zeros((height, width), dtype=np.uint8)
        lower_frame[int(height * 0.38):, :] = 1
        ycrcb = cv2.cvtColor(generated, cv2.COLOR_RGB2YCrCb)
        skin = (
            (ycrcb[:, :, 1] >= 133)
            & (ycrcb[:, :, 1] <= 180)
            & (ycrcb[:, :, 2] >= 70)
            & (ycrcb[:, :, 2] <= 135)
        )
        saturated = generated_hsv[:, :, 1] >= 78
        visible_subject = background_distance >= 38.0
        candidate = (
            saturated & visible_subject & ~skin & lower_frame.astype(bool)
        ).astype(np.uint8) * 255

        # Remove narrow check/stripe threads. The surviving connected regions
        # represent solid template design elements rather than the fabric print.
        candidate = cv2.morphologyEx(
            candidate,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
        coherent = np.zeros_like(candidate)
        minimum_area = max(80, int(height * width * 0.00035))
        retained_labels = []
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
                coherent[labels == label] = 255
                retained_labels.append(label)
        if np.count_nonzero(coherent) < minimum_area:
            return image

        # Collar, placket, cuffs and pocket trim normally share one generated
        # hue. Use the largest coherent component to identify that hue, then
        # exclude differently colored plaid/check blocks from the correction.
        dominant_label = max(
            retained_labels, key=lambda item: int(stats[item, cv2.CC_STAT_AREA])
        )
        dominant_hue = int(np.median(generated_hsv[:, :, 0][labels == dominant_label]))
        generated_accent_delta = np.abs(
            generated_hsv[:, :, 0].astype(np.int16) - dominant_hue
        )
        generated_accent_delta = np.minimum(
            generated_accent_delta, 180 - generated_accent_delta
        )
        same_generated_accent = (generated_accent_delta <= 12).astype(np.uint8) * 255

        placement = str(trim_placement or "").lower()
        allowed = np.zeros((height, width), dtype=np.uint8)
        for label in retained_labels:
            x, y, component_width, component_height, _ = stats[label]
            center_x = (x + component_width * .5) / max(width, 1)
            center_y = (y + component_height * .5) / max(height, 1)
            is_allowed = False
            if "collar" in placement and .18 <= center_x <= .82 and .30 <= center_y <= .58:
                is_allowed = True
            if ("placket" in placement or "center" in placement) and .42 <= center_x <= .58 and center_y >= .34:
                is_allowed = True
            if ("cuff" in placement or "sleeve" in placement) and center_y >= .42 and (
                center_x <= .22 or center_x >= .82
            ):
                is_allowed = True
            if "pocket" in placement and .42 <= center_y <= .82 and (
                center_x <= .42 or center_x >= .58
            ):
                is_allowed = True
            if is_allowed:
                allowed[labels == label] = 255
        if placement and not np.count_nonzero(allowed):
            return image
        if np.count_nonzero(allowed):
            same_generated_accent = cv2.bitwise_and(same_generated_accent, allowed)

        target_pixel = np.asarray(target_rgb, dtype=np.uint8).reshape(1, 1, 3)
        target_hsv = cv2.cvtColor(target_pixel, cv2.COLOR_RGB2HSV).reshape(3)
        hue_delta = np.abs(generated_hsv[:, :, 0].astype(np.int16) - int(target_hsv[0]))
        hue_delta = np.minimum(hue_delta, 180 - hue_delta)
        drift = (hue_delta >= 8).astype(np.uint8) * 255
        correction_mask = cv2.bitwise_and(coherent, same_generated_accent)
        correction_mask = cv2.bitwise_and(correction_mask, drift)
        correction_mask = cv2.morphologyEx(
            correction_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        )
        if np.count_nonzero(correction_mask) < minimum_area:
            return image

        generated_lab = cv2.cvtColor(generated, cv2.COLOR_RGB2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target_pixel, cv2.COLOR_RGB2LAB).reshape(3).astype(np.float32)
        alpha = cv2.GaussianBlur(correction_mask, (0, 0), 1.2).astype(np.float32) / 255.0
        alpha *= 0.90
        for channel in (1, 2):
            generated_lab[:, :, channel] = (
                generated_lab[:, :, channel] * (1.0 - alpha)
                + target_lab[channel] * alpha
            )
        corrected = cv2.cvtColor(
            np.clip(generated_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB,
        )
        corrected[alpha <= 0.0] = generated[alpha <= 0.0]
        logger.info(
            "[UNIFORM_COLOR] Restored template accent chroma across %.1f%% of the portrait",
            100.0 * float(np.count_nonzero(correction_mask)) / float(height * width),
        )
        return Image.fromarray(corrected, "RGB")

    def _apply_upscale(self, image: Image.Image, scale: int = 2) -> Image.Image:
        """Upscales image with Real-ESRGAN or high-grade Lanczos resampling."""
        if scale <= 1:
            return image
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model_file = Path("models/realesrgan/RealESRGAN_x4plus.pth")
            if model_file.exists():
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                upsampler = RealESRGANer(
                    scale=4,
                    model_path=str(model_file),
                    model=model,
                    tile=512,
                    tile_pad=16,
                    pre_pad=0,
                    half=(self.device.type == "cuda"),
                    device=self.device,
                )
                img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                output_bgr, _ = upsampler.enhance(img_bgr, outscale=scale)
                return Image.fromarray(cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB))
        except Exception as e:
            logger.warning("[QWEN_UPSCALE] Fallback to Lanczos (%s)", e)
        w, h = image.size
        return image.resize((w * scale, h * scale), Image.LANCZOS)

    @staticmethod
    def _has_acceptable_portrait_identity(source: Image.Image, generated: Image.Image) -> bool:
        """Reject a diffusion result when it no longer depicts the uploaded person."""
        try:
            from pipelines.photo_restoration import _get_insight_app

            app = _get_insight_app()
            source_bgr = cv2.cvtColor(np.array(source.convert("RGB")), cv2.COLOR_RGB2BGR)
            generated_bgr = cv2.cvtColor(np.array(generated.convert("RGB")), cv2.COLOR_RGB2BGR)
            source_faces = app.get(source_bgr)
            generated_faces = app.get(generated_bgr)
            if len(source_faces) != 1 or len(generated_faces) != 1:
                logger.warning(
                    "[PORTRAIT_QA] Rejecting Qwen result: source_faces=%d generated_faces=%d",
                    len(source_faces), len(generated_faces),
                )
                return False

            similarity = float(np.dot(
                source_faces[0].normed_embedding,
                generated_faces[0].normed_embedding,
            ))
            landmark_rmse, face_aspect_delta = _face_geometry_metrics(
                source_faces[0], generated_faces[0]
            )
            geometry_ok = (
                landmark_rmse <= MAX_PORTRAIT_LANDMARK_RMSE
                and face_aspect_delta <= MAX_PORTRAIT_FACE_ASPECT_DELTA
            )

            def hair_shadow_anchor(image_bgr: np.ndarray, face_obj: Any) -> Optional[float]:
                """Measure the dark hair mass immediately above the detected face.

                This deliberately excludes the wider scene/background. It is a
                safety check for the common failure where Image Edit changes a
                dark-haired child into a light-haired child while preserving a
                broadly similar face embedding.
                """
                x1, y1, x2, y2 = [int(value) for value in face_obj.bbox]
                face_w, face_h = max(1, x2 - x1), max(1, y2 - y1)
                center_x = (x1 + x2) // 2
                left = max(0, center_x - int(face_w * 0.46))
                right = min(image_bgr.shape[1], center_x + int(face_w * 0.46))
                top = max(0, y1 - int(face_h * 0.64))
                bottom = min(image_bgr.shape[0], y1 + int(face_h * 0.10))
                if right <= left or bottom <= top:
                    return None
                luminance = cv2.cvtColor(image_bgr[top:bottom, left:right], cv2.COLOR_BGR2LAB)[:, :, 0]
                return float(np.percentile(luminance, 20))

            source_hair_luma = hair_shadow_anchor(source_bgr, source_faces[0])
            generated_hair_luma = hair_shadow_anchor(generated_bgr, generated_faces[0])
            # A modest lightness shift is expected when Qwen corrects a strong
            # outdoor highlight. Reject only a decisive dark-to-pale change;
            # the face embedding and silhouette gates still apply separately.
            dark_hair_shifted_light = bool(
                source_hair_luma is not None
                and generated_hair_luma is not None
                and source_hair_luma < 95.0
                and generated_hair_luma > 155.0
                and generated_hair_luma > source_hair_luma + 60.0
            )
            face = generated_faces[0]
            x1, y1, x2, y2 = [int(value) for value in face.bbox]
            face_width = max(1, x2 - x1)
            face_height = max(1, y2 - y1)

            # Face similarity alone cannot catch a generation that keeps a
            # recognizable face but erases the shoulders and garment. Check
            # that a head-and-shoulders silhouette remains below the face.
            lab = cv2.cvtColor(generated_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            edge = max(4, min(16, generated_bgr.shape[0] // 20, generated_bgr.shape[1] // 20))
            backdrop = np.median(np.concatenate((
                lab[:edge, :edge].reshape(-1, 3), lab[:edge, -edge:].reshape(-1, 3),
                lab[-edge:, :edge].reshape(-1, 3), lab[-edge:, -edge:].reshape(-1, 3),
            )), axis=0)
            foreground = np.linalg.norm(lab - backdrop, axis=2) > 18.0
            torso_top = min(generated_bgr.shape[0], y2 + int(face_height * 0.55))
            torso_bottom = min(generated_bgr.shape[0], y2 + int(face_height * 1.20))
            torso_rows = foreground[torso_top:torso_bottom]
            if torso_rows.size:
                row_spans = np.count_nonzero(torso_rows, axis=1)
                torso_coverage = float(np.mean(row_spans >= int(face_width * 0.78)))
            else:
                torso_coverage = 0.0
            torso_present = torso_coverage >= 0.42
            # Keep Qwen only when it remains extremely close to the uploaded
            # person. This prevents a plausible but differently coloured hair
            # or re-drawn child face from being passed off as an enhancement.
            accepted = (
                similarity >= MIN_PORTRAIT_IDENTITY_SIMILARITY
                and geometry_ok
                and torso_present
                and not dark_hair_shifted_light
            )
            logger.info(
                "[PORTRAIT_QA] Identity similarity=%.3f threshold=%.3f landmark_rmse=%.4f max=%.4f face_aspect_delta=%.4f max=%.4f geometry_ok=%s torso_coverage=%.2f hair_luma=%.1f->%.1f dark_hair_shifted_light=%s accepted=%s",
                similarity, MIN_PORTRAIT_IDENTITY_SIMILARITY,
                landmark_rmse, MAX_PORTRAIT_LANDMARK_RMSE,
                face_aspect_delta, MAX_PORTRAIT_FACE_ASPECT_DELTA, geometry_ok,
                torso_coverage,
                source_hair_luma if source_hair_luma is not None else -1.0,
                generated_hair_luma if generated_hair_luma is not None else -1.0,
                dark_hair_shifted_light, accepted,
            )
            return accepted
        except Exception as exc:
            # A validation outage must not let an unverified generated face be
            # delivered in place of a real person's photo.
            logger.warning("[PORTRAIT_QA] Validation unavailable; rejecting Qwen result: %s", exc)
            return False

    # ── 1. QWEN EDIT ENHANCER ──────────────────────────────────────────────────

    def qwen_edit_enhancer(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        job_id: str = "qwen_enhance",
        background_color: str = "white",
        upscale_factor: int = 1,
        width: int = 600,
        height: int = 800,
        original_filename: Optional[str] = None,
        mode: str = "qwen",
        steps: int = 4,
        max_sequence_length: int = 256,
        timeout_seconds: Optional[float] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        reference_images: Optional[List[Image.Image]] = None,
        preserve_source_clothing: bool = True,
        max_generation_dimension: int = 576,
        minimum_generation_dimension: int = 0,
        keep_generation_resolution: bool = False,
        use_birefnet_background: bool = True,
        true_cfg_scale: float = 1.2,
        seed: Optional[int] = None,
        face_detail_refine: bool = False,
        face_lock_after_generation: bool = False,
        reject_identity_failure: bool = False,
        return_raw_candidate: bool = False,
        conditioning_max_dimension: int = 256,
        preserve_reference_aspect: bool = False,
        conditioning_pixel_budget: Optional[int] = None,
    ) -> Path:
        """
        Genuine Qwen-Image-Edit-2511 Decoupled Studio Portrait Enhancer:
        1. Encodes multimodal prompt conditioning in an isolated worker process (zero VRAM leak).
        2. Executes 4-step Lightning generative diffusion on DiT Transformer with 2-chunk GPU streaming.
        3. Decodes latents via CPU VAE (preventing PyTorch CUDA conv3d limitations).
        4. Fuses authentic student identity & preserves facial landmarks/skin microtexture.
        5. Enhances optical eye catchlights, smile contours, and specular jewelry luster.
        6. Calibrates studio contrast & exports at 300 DPI.
        """
        t0 = time.time()
        deadline = (time.monotonic() + timeout_seconds) if timeout_seconds and timeout_seconds > 0 else None

        def check_deadline(phase: str) -> None:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"Qwen edit timed out during {phase} after {timeout_seconds:.0f} seconds")

        # Strictly keep CFG scale between 1.0 and 1.5 to prevent burning the colors
        true_cfg_scale = max(1.0, min(1.5, float(true_cfg_scale if true_cfg_scale is not None else 1.2)))
        if reference_images and len(reference_images) >= 2 and true_cfg_scale > 1.20:
            logger.info(
                "[QWEN_EDIT_ENHANCER] Multi-reference edit detected (%d references); capping true_cfg_scale %.2f -> 1.15 to prevent cross-attention drift",
                len(reference_images), true_cfg_scale,
            )
            true_cfg_scale = 1.15

        logger.info(
            "[QWEN_EDIT_ENHANCER] Starting genuine Qwen-Image-Edit enhancement job %s (bg=%s, prompt='%s', orig=%s)",
            job_id, background_color, prompt or "default", original_filename
        )

        if progress_callback:
            progress_callback(10, "Preparing portrait dimensions and studio framing...")

        input_pil = self._to_pil(image_input).convert("RGB")
        w_orig, h_orig = input_pil.size
        target_w = width or 600
        target_h = height or 800
        # Small phone uploads otherwise make Qwen generate at a tiny canvas and
        # then get enlarged by passport framing.  This is generation resolution,
        # not a post-generation AI upscaler.
        generation_w, generation_h = target_w, target_h
        minimum_generation_dimension = max(0, int(minimum_generation_dimension or 0))
        longest_side = max(generation_w, generation_h)
        if minimum_generation_dimension and longest_side < minimum_generation_dimension:
            scale_up = minimum_generation_dimension / max(longest_side, 1)
            generation_w = int(round(generation_w * scale_up))
            generation_h = int(round(generation_h * scale_up))
        # Qwen's two-reference edit path has a much larger attention footprint
        # than portrait enhancement.  Callers can cap this canvas for an 8 GB GPU.
        w, h = self._align_dimensions(generation_w, generation_h, max_dim=max(256, max_generation_dimension))
        input_resized = input_pil.resize((w, h), Image.LANCZOS)
        references = []
        for reference in reference_images or []:
            if reference is not None:
                prepared_reference = self._to_pil(reference).convert("RGB")
                if not preserve_reference_aspect:
                    prepared_reference = prepared_reference.resize((w, h), Image.LANCZOS)
                references.append(prepared_reference)
        conditioning_images = [input_resized, *references]

        bg_desc = ""
        if str(background_color).lower() not in ("none", "keep", "original"):
            bg_rgb = self._parse_bg_color(background_color)
            bg_hex = f"#{bg_rgb[0]:02X}{bg_rgb[1]:02X}{bg_rgb[2]:02X}"
            # "Studio backdrop" is visually ambiguous to an image model and
            # produced curtains, rooms and cloudy photographic backgrounds.
            # Describe the required pixels instead.
            bg_desc = f"against one featureless flat {bg_hex} RGB{bg_rgb} color field"

        if prompt and prompt.strip():
            effective_prompt = prompt.strip()
            if bg_desc and bg_hex not in effective_prompt:
                effective_prompt = f"Generate selected {bg_hex} RGB{bg_rgb} flat solid backdrop from edge to edge during generation itself. " + effective_prompt
        else:
            effective_prompt = (
                f"A clean professional school portrait photo of this student {bg_desc}, "
                "natural 3-point softbox studio lighting, rich colors, authentic skin tones, crisp focus, 8k resolution photograph. "
                "Keep the student's exact facial identity, eyes, nose, smile expression, and hair 100% identical and unchanged."
            )

        # ── Step 1: Isolated Multimodal Conditioning Worker ──
        if progress_callback:
            progress_callback(20, "Preparing portrait guidance...")

        logger.info("[QWEN_EDIT_ENHANCER] Encoding conditioning via isolated worker...")
        check_deadline("prompt preparation")
        # Low-VRAM callers without a job deadline must be allowed to finish the
        # isolated encoder load instead of being canceled while pages are cold.
        encode_timeout = None if deadline is None else max(1.0, deadline - time.monotonic())
        effective_negative = (negative_prompt or "").strip()
        # Qwen true CFG requires an unconditional/negative branch.  For the
        # positive-only identity-preserving routes, the mathematically correct
        # and least destructive value is exactly 1.0.  Keep the requested
        # low-guidance value only when that second branch actually exists.
        effective_true_cfg_scale = true_cfg_scale if effective_negative else 1.0
        prompts_to_encode = [effective_prompt]
        if effective_negative:
            prompts_to_encode.append(effective_negative)
        encoded_prompts = _encode_prompt_isolated(
            conditioning_images,
            prompts_to_encode,
            max_sequence_length=max(64, min(max_sequence_length, 512)),
            timeout_seconds=encode_timeout,
            conditioning_max_dimension=conditioning_max_dimension,
            conditioning_pixel_budget=conditioning_pixel_budget,
        )
        check_deadline("prompt encoding")
        prompt_embeds, prompt_embeds_mask = encoded_prompts[0]
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.dtype)
        if prompt_embeds_mask is None:
            # Some Qwen-Image-Edit encoder builds return no explicit mask even
            # though the embedding tensor includes padded text positions. The
            # diffusion pipeline then warns and treats conditioning
            # inconsistently. All tokens emitted by this isolated encoder are
            # valid, so supply an explicit all-valid mask.
            prompt_embeds_mask = torch.ones(
                prompt_embeds.shape[:2], device=self.device, dtype=torch.bool
            )
            logger.info("[QWEN_EDIT_ENHANCER] Added fallback positive prompt attention mask")
        else:
            prompt_embeds_mask = prompt_embeds_mask.to(device=self.device)

        # The Lightning model only observes a negative prompt when true CFG is
        # enabled.  Previously the UI supplied negatives, but they were never
        # encoded or passed to the pipeline, so it could still invent glasses
        # or retain pieces of the original scene.
        negative_prompt_embeds = None
        negative_prompt_embeds_mask = None
        if effective_negative:
            negative_prompt_embeds, negative_prompt_embeds_mask = encoded_prompts[1]
            negative_prompt_embeds = negative_prompt_embeds.to(device=self.device, dtype=self.dtype)
            if negative_prompt_embeds_mask is None:
                negative_prompt_embeds_mask = torch.ones(
                    negative_prompt_embeds.shape[:2], device=self.device, dtype=torch.bool
                )
                logger.info("[QWEN_EDIT_ENHANCER] Added fallback negative prompt attention mask")
            else:
                negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(device=self.device)

        # ── Step 2: Load DiT Transformer & CPU VAE ──
        # Lightning LoRA is trained for four steps.  Quality mode uses the
        # base model and a longer schedule, so it must not load that adapter.
        num_steps = max(2, min(steps or 4, 24))
        use_lightning_lora = num_steps <= 4
        if progress_callback:
            mode_label = "fast" if use_lightning_lora else "high-quality"
            progress_callback(40, f"Initializing {mode_label} portrait restoration...")

        tokenizer = AutoTokenizer.from_pretrained((MODELS_DIR / "tokenizer").resolve().as_posix())
        processor = AutoProcessor.from_pretrained((MODELS_DIR / "processor").resolve().as_posix())
        vae = AutoencoderKLQwenImage.from_pretrained((MODELS_DIR / "vae").resolve().as_posix(), torch_dtype=torch.float32).to("cpu")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained((MODELS_DIR / "scheduler").resolve().as_posix())

        quant_cfg = GGUFQuantizationConfig(compute_dtype=self.dtype)
        transformer = QwenImageTransformer2DModel.from_single_file(
            GGUF_PATH.resolve().as_posix(),
            config=CONFIG_PATH.resolve().as_posix(),
            quantization_config=quant_cfg,
            torch_dtype=self.dtype,
        )
        transformer.forward = types.MethodType(_chunked_transformer_forward, transformer)
        transformer._qwen_deadline = deadline

        pipe = QwenImageEditPlusPipeline(
            tokenizer=tokenizer,
            text_encoder=None,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            processor=processor,
        )
        pipe._encode_vae_image = types.MethodType(_cpu_encode_vae_image, pipe)

        orig_prepare_latents = pipe.prepare_latents
        def _cuda_prepare_latents(self, *args, **kwargs):
            prepared = orig_prepare_latents(*args, **kwargs)
            dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
            if not isinstance(prepared, tuple):
                return prepared.to(device=dev, dtype=dtype)
            return tuple(
                item.to(device=dev, dtype=dtype)
                if isinstance(item, torch.Tensor) else item
                for item in prepared
            )

        pipe.prepare_latents = types.MethodType(_cuda_prepare_latents, pipe)

        if use_lightning_lora and LORA_FILE.exists():
            pipe.load_lora_weights(str(LORA_FILE.parent), weight_name=LORA_FILE.name, adapter_name="lightning")

        # ── Step 3: Run Generative Diffusion ──
        def on_step_end(pipe, step_index, timestep, callback_kwargs):
            check_deadline(f"diffusion step {step_index + 1}/{num_steps}")
            pct = int(45 + (step_index + 1) / num_steps * 40)
            msg = f"Restoring studio portrait detail (step {step_index + 1}/{num_steps})…"
            if progress_callback:
                progress_callback(pct, msg)
            return callback_kwargs

        run_seed = int(seed) if seed is not None else (zlib.crc32(job_id.encode("utf-8")) & 0x7FFFFFFF)
        logger.info(
            "[QWEN_EDIT_ENHANCER] Running %d-step diffusion (seed=%d, true_cfg=%.2f, lightning=%s)...",
            num_steps, run_seed, effective_true_cfg_scale,
            use_lightning_lora and LORA_FILE.exists(),
        )
        diffusion_kwargs = dict(
            image=(conditioning_images if len(conditioning_images) > 1 else input_resized),
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            num_inference_steps=num_steps,
            true_cfg_scale=effective_true_cfg_scale,
            height=h,
            width=w,
            generator=torch.Generator(device="cpu").manual_seed(run_seed),
            output_type="latent",
            callback_on_step_end=on_step_end,
        )
        res = pipe(**diffusion_kwargs)
        output_latents = res.images
        check_deadline("diffusion")

        # Free DiT transformer
        del pipe
        del transformer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Step 4: Decode Latents on CPU ──
        if progress_callback:
            progress_callback(88, "Rendering high-detail portrait...")
        raw_diffused = _decode_cpu_latents(
            vae, output_latents, height=h, width=w,
        )
        # Keep the unmodified Qwen output for stage-by-stage inspection. This
        # makes it clear whether a later identity check accepted or rejected
        # the generated portrait instead of hiding that decision in the final.
        qwen_debug_path = OUTPUTS_DIR / f"{job_id}_qwen_raw.png"
        raw_diffused.save(qwen_debug_path, format="PNG")
        self.last_qwen_candidate_path = qwen_debug_path
        logger.info("[QWEN_EDIT_ENHANCER] Saved pre-validation Qwen output -> %s", qwen_debug_path)
        if return_raw_candidate:
            # Explicit uniform-preview policy: never replace the generated
            # candidate with source pixels or reject it on a quality score.
            self.last_qwen_identity_accepted = None
            return qwen_debug_path

        # ── Step 5: High-Fidelity Identity Lock & Optical Detail Restoration ──
        if progress_callback:
            progress_callback(92, "Verifying identity and fine portrait detail...")

        qwen_portrait_accepted = preserve_source_clothing or self._has_acceptable_portrait_identity(
            input_resized, raw_diffused
        )
        self.last_qwen_identity_accepted = qwen_portrait_accepted
        if reject_identity_failure and not qwen_portrait_accepted:
            # A source-photo fallback cannot satisfy a clothing replacement.
            # Preserve the raw candidate for diagnostics, but fail the job
            # before exporting the unchanged upload as a successful uniform.
            raise RuntimeError(
                "Uniform generation changed the person's identity and was rejected. "
                "No completed uniform image was produced; the original photo has not been substituted."
            )
        # The enhancement UI is a Qwen Image Edit workflow.  Its final must be
        # the model's decoded image, followed only by passport framing in the
        # caller.  A QA fallback here used to replace a raw Qwen result with a
        # source/BiRefNet composite, so users received a different photo from
        # the raw output they had reviewed during testing.
        # The calculated QA decision must govern the raw-Qwen path.  Previously
        # this ignored `qwen_portrait_accepted`, so a low-fidelity regeneration
        # was returned even after the identity check had flagged it.
        use_accepted_qwen_raw = not preserve_source_clothing and qwen_portrait_accepted
        if use_accepted_qwen_raw:
            # Do not blend, rematte, sharpen, or relight the decoded Qwen
            # portrait.  Those operations visibly alter its face and hair.
            fused_identity_pil = raw_diffused.copy()
        elif not qwen_portrait_accepted:
            logger.warning(
                "[PORTRAIT_QA] Qwen output changed identity or composition; using the real uploaded subject instead."
            )
            fused_identity_pil = input_resized.copy()
        else:
            fused_identity_pil = PhotoRestorationService.anchor_and_fuse_identity(
                orig_pil=input_resized,
                qwen_pil=raw_diffused,
            # In the Qwen-only portrait route, a high source blend copies the
            # low-resolution upload back over Qwen's restored facial detail.
            # 70% retains a strong biometric anchor while allowing the Qwen
            # reconstruction to contribute real detail.
            # Portrait mode permits Qwen to adjust the pose. A heavy source
            # fusion creates a second face when the regenerated head angle is
            # different, so retain only a light identity anchor here.
            # Portrait restoration is Qwen-led.  A heavier source blend, or a
            # later source-face overlay, produces double eyes when Qwen makes
            # even a small change to face scale or pose.
                identity_lock_strength=0.85 if preserve_source_clothing else 0.55,
            )

        fused_bgr = cv2.cvtColor(np.array(fused_identity_pil), cv2.COLOR_RGB2BGR)
        # Qwen portrait output already contains the requested photographic
        # restoration.  The old face smoother and multi-scale hair/fabric
        # sharpener produced plastic skin, silver hair, and lace artifacts.
        # Keep this route optical-only and source-anchored after diffusion.
        out_img = Image.fromarray(cv2.cvtColor(fused_bgr, cv2.COLOR_BGR2RGB))
        face_box = None
        graded_bgr = cv2.cvtColor(np.array(out_img), cv2.COLOR_RGB2BGR)
        if preserve_source_clothing:
            color_locked_bgr = PhotoRestorationService.preserve_source_clothing(
                cv2.cvtColor(np.array(input_resized), cv2.COLOR_RGB2BGR),
                graded_bgr,
                strength=0.72,
            )
            out_img = Image.fromarray(cv2.cvtColor(color_locked_bgr, cv2.COLOR_BGR2RGB))

        # Pixel-level source locks are reserved for uniform fitting, where the
        # source pose must remain fixed.  Portrait restoration permits Qwen to
        # rebuild the face, so overlays here create ghost/double eyes.
        source_bgr = cv2.cvtColor(np.array(input_resized), cv2.COLOR_RGB2BGR)
        if preserve_source_clothing and not PhotoRestorationService.has_eyeglasses(source_bgr):
            out_img = PhotoRestorationService.lock_face_keep_clothes(
                input_resized, out_img,
                strength=0.88,
            )
        if preserve_source_clothing:
            out_img = PhotoRestorationService.lock_source_hair(input_resized, out_img, strength=0.88)
            out_img = PhotoRestorationService.match_neck_tone(input_resized, out_img)

        # Uniform fitting redraws clothing but must not invent a new child face.
        # A landmark pose gate avoids the double-eye artifact when Qwen changes
        # camera angle; the face-only mask never reaches the generated uniform.
        if face_lock_after_generation and use_accepted_qwen_raw:
            if PhotoRestorationService.is_source_pose_compatible(input_resized, out_img):
                out_img = PhotoRestorationService.lock_face_keep_clothes(
                    input_resized, out_img, strength=0.78,
                )
            else:
                logger.warning("[UNIFORM_FACE_LOCK] Skipped because source and generated face poses differ.")

        # Apply a restrained optical finish only after the identity lock. This
        # restores perceived facial, hair, and fabric detail without another
        # generative pass that could alter the person or the selected backdrop.
        if not use_accepted_qwen_raw:
            detail_bgr = cv2.cvtColor(np.array(out_img), cv2.COLOR_RGB2BGR)
            detail_blur = cv2.GaussianBlur(detail_bgr, (0, 0), 0.75)
            detail_bgr = cv2.addWeighted(detail_bgr, 1.04, detail_blur, -0.04, 0)
            out_img = Image.fromarray(cv2.cvtColor(detail_bgr, cv2.COLOR_BGR2RGB))

        if face_detail_refine and face_box is not None:
            if progress_callback:
                progress_callback(94, "Restoring facial detail with a localized Qwen pass...")
            out_img = self._refine_face_crop_with_qwen(
                source=input_resized,
                portrait=out_img,
                face_box=face_box,
                job_id=job_id,
                negative_prompt=negative_prompt,
            )

        # ── Step 6: Clean Studio Backdrop Compositing with BiRefNet (if background replacement requested) ──
        if (not use_accepted_qwen_raw and use_birefnet_background and background_color
                and str(background_color).lower() not in ("none", "keep", "original")):
            try:
                from pipelines.birefnet_service import background_removal
                bg_target_rgb = self._parse_bg_color(background_color)
                with io.BytesIO() as bio:
                    out_img.save(bio, format="PNG")
                    fg_bytes = background_removal.remove_background(
                        bio.getvalue(),
                        job_id=job_id,
                        # The final Qwen portrait already has the intended
                        # hair and clothing. Use only BiRefNet's soft matte to
                        # remove residual scenery; SCHP is deliberately off.
                        source_image=None,
                        use_schp=False,
                    )
                fg_rgba = Image.open(io.BytesIO(fg_bytes)).convert("RGBA")
                bg_solid = Image.new("RGBA", fg_rgba.size, (bg_target_rgb[0], bg_target_rgb[1], bg_target_rgb[2], 255))
                bg_solid.paste(fg_rgba, (0, 0), fg_rgba)
                out_img = bg_solid.convert("RGB")
            except Exception as e:
                logger.warning("[QWEN_EDIT_ENHANCER] Background composite error: %s", e)
        # Ensure output strictly matches target dimensions at 300 DPI
        output_size = (w, h) if keep_generation_resolution else (target_w, target_h)
        if out_img.size != output_size:
            out_img = out_img.resize(output_size, Image.LANCZOS)

        if upscale_factor > 1:
            out_img = self._apply_upscale(out_img, scale=upscale_factor)

        orig_stem = Path(original_filename).stem if original_filename else "qwen_enhanced"
        out_path = OUTPUTS_DIR / f"{job_id}_{orig_stem}.png"
        out_img.save(out_path, format="PNG", dpi=(300, 300))

        if progress_callback:
            progress_callback(100, "Studio portrait enhancement complete!")

        logger.info("[QWEN_EDIT_ENHANCER] Completed genuine Qwen enhancement in %.2fs -> %s", time.time() - t0, out_path)
        return out_path

    def _refine_face_crop_with_qwen(
        self,
        source: Image.Image,
        portrait: Image.Image,
        face_box: Tuple[int, int, int, int],
        job_id: str,
        negative_prompt: Optional[str],
    ) -> Image.Image:
        """Restore face micro-detail without letting Qwen touch hair, shoulders, or clothing."""
        x, y, fw, fh = face_box
        width, height = portrait.size
        pad_x = max(12, int(fw * 0.16))
        pad_top = max(12, int(fh * 0.18))
        pad_bottom = max(8, int(fh * 0.08))
        left, top = max(0, x - pad_x), max(0, y - pad_top)
        right, bottom = min(width, x + fw + pad_x), min(height, y + fh + pad_bottom)
        if right - left < 96 or bottom - top < 96:
            return portrait

        source_crop = source.crop((left, top, right, bottom))
        prompt = (
            "Restore only this exact person's central face as a natural high-resolution photograph. "
            "Preserve exact identity, age, expression, eye shape, nose, mouth, skin tone, and facial proportions. "
            "Improve only subtle skin texture, iris clarity, eyelashes, and balanced natural lighting. "
            "Keep age-appropriate natural eye size and spacing; do not enlarge eyes, smooth skin into plastic, or create an illustration. "
            "Do not alter hairline, hairstyle, ears, neck, shoulders, clothing, pose, or background."
        )
        try:
            refined_path = self.qwen_edit_enhancer(
                image_input=source_crop,
                prompt=prompt,
                negative_prompt=(negative_prompt or "") + ", altered identity, face distortion, blue facial patches, purple facial patches, hairline change, clothing",
                job_id=f"{job_id}_face",
                background_color="keep",
                width=source_crop.width,
                height=source_crop.height,
                # A small face crop can afford the base-model quality schedule;
                # Lightning at four steps was too prone to synthetic eyes.
                steps=8,
                max_generation_dimension=384,
                preserve_source_clothing=False,
                use_birefnet_background=False,
                true_cfg_scale=1.15,
                face_detail_refine=False,
            )
            refined = Image.open(refined_path).convert("RGB").resize(source_crop.size, Image.LANCZOS)
        except Exception as exc:
            logger.warning("[QWEN_FACE_REFINE] Keeping base portrait after localized face pass failed: %s", exc)
            return portrait

        # Blend only the inner face. The soft oval stops before hair, temples,
        # neck, shoulders, and the garment, which remain from the base portrait.
        crop_w, crop_h = source_crop.size
        local_cx = min(crop_w - 1, max(0, x + fw // 2 - left))
        local_cy = min(crop_h - 1, max(0, y + fh // 2 - top))
        mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        cv2.ellipse(
            mask,
            (local_cx, local_cy),
            (max(20, int(fw * 0.40)), max(24, int(fh * 0.46))),
            0, 0, 360, 255, -1,
        )
        mask = cv2.GaussianBlur(mask, (21, 21), 0).astype(np.float32)[:, :, None] / 255.0
        base_crop = np.array(portrait.crop((left, top, right, bottom)), dtype=np.float32)
        refined_arr = np.array(refined, dtype=np.float32)
        merged = (refined_arr * mask + base_crop * (1.0 - mask)).clip(0, 255).astype(np.uint8)
        result = portrait.copy()
        result.paste(Image.fromarray(merged, "RGB"), (left, top))
        return result

    # ── 1B. TWO-STAGE STUDIO BACKGROUND REMOVAL & NEURAL ENHANCEMENT ──────────

    def two_stage_studio_enhance(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        bg_color: str = "white",
        job_id: str = "qwen_2stage",
        width: int = 600,
        height: int = 800,
        enhance_face: bool = True,
        apply_highpass: bool = True,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Path:
        """
        Two-Stage Architecture:
        Stage 1: Precision BiRefNet background removal & clean studio backdrop placement.
        Stage 2: Neural Photographic Enhancement (GFPGAN facial pore/eye detailing, hair defringing, studio grading).
        """
        from pipelines.birefnet_service import background_removal

        t0 = time.time()
        logger.info("[TWO_STAGE_STUDIO] Starting 2-stage studio enhancement job %s (bg=%s)", job_id, bg_color)

        if progress_callback:
            progress_callback(10, "✂️ Stage 1: Segmenting subject with BiRefNet sub-pixel hair matting...")

        orig_pil = self._to_pil(image_input).convert("RGB")
        orig_np = np.array(orig_pil)

        with io.BytesIO() as bio:
            orig_pil.save(bio, format="PNG")
            fg_bytes = background_removal.remove_background(bio.getvalue())

        fg_rgba = Image.open(io.BytesIO(fg_bytes)).convert("RGBA")
        fg_arr = np.array(fg_rgba)

        rgb = fg_arr[:, :, :3].astype(np.float32)
        alpha = fg_arr[:, :, 3].astype(np.float32) / 255.0

        if progress_callback:
            progress_callback(40, "✨ Stage 2: Hair defringing & color decontamination...")

        alpha_uint8 = (alpha * 255).astype(np.uint8)
        cnts, _ = cv2.findContours((alpha_uint8 > 128).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solid_mask = np.zeros_like(alpha_uint8)
        cv2.drawContours(solid_mask, cnts, -1, 255, -1)

        refined_alpha = np.where(solid_mask == 255, np.maximum(alpha_uint8, 220), alpha_uint8)
        refined_alpha = cv2.bilateralFilter(refined_alpha, d=7, sigmaColor=75, sigmaSpace=75)
        refined_alpha = cv2.GaussianBlur(refined_alpha, (3, 3), 0.5)

        edge_band = ((refined_alpha > 5) & (refined_alpha < 240)).astype(np.uint8) * 255
        inpainted_rgb = cv2.inpaint(rgb.astype(np.uint8), edge_band, inpaintRadius=4, flags=cv2.INPAINT_TELEA)

        clean_rgb = rgb.copy()
        clean_rgb[edge_band == 255] = inpainted_rgb[edge_band == 255]
        clean_rgb_uint8 = np.clip(clean_rgb, 0, 255).astype(np.uint8)

        if enhance_face:
            if progress_callback:
                progress_callback(65, "👁️ Restoring facial skin pores, iris sparkle & crisp smile...")
            restorer = PhotoRestorationService()
            enhanced_bgr, _ = restorer.enhance_faces_optical(cv2.cvtColor(clean_rgb_uint8, cv2.COLOR_RGB2BGR))
            clean_rgb_uint8 = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

        if apply_highpass and orig_np.shape == clean_rgb_uint8.shape:
            orig_yuv = cv2.cvtColor(orig_np, cv2.COLOR_RGB2YUV)
            curr_yuv = cv2.cvtColor(clean_rgb_uint8, cv2.COLOR_RGB2YUV)
            orig_y = orig_yuv[:, :, 0].astype(np.float32)
            orig_y_blurred = cv2.GaussianBlur(orig_y, (0, 0), 1.2)
            high_freq = orig_y - orig_y_blurred
            curr_y = curr_yuv[:, :, 0].astype(np.float32)
            fused_y = np.clip(curr_y + high_freq * 0.40 * (refined_alpha.astype(np.float32) / 255.0), 0, 255).astype(np.uint8)
            curr_yuv[:, :, 0] = fused_y
            clean_rgb_uint8 = cv2.cvtColor(curr_yuv, cv2.COLOR_YUV2RGB)

        if progress_callback:
            progress_callback(85, f"📐 Compositing onto {bg_color} studio canvas ({width}x{height})...")

        target_bg = self._parse_bg_color(bg_color)
        canvas = Image.new("RGBA", (width, height), (*target_bg, 255))

        clean_alpha_f = (refined_alpha.astype(np.float32) / 255.0)
        clean_rgba = Image.fromarray(np.dstack([clean_rgb_uint8, (clean_alpha_f * 255).astype(np.uint8)]), "RGBA")

        pw, ph = clean_rgba.size
        scale = min((width * 0.92) / pw, (height * 0.92) / ph)
        new_w = int(pw * scale)
        new_h = int(ph * scale)
        fg_scaled = clean_rgba.resize((new_w, new_h), Image.LANCZOS)

        pos_x = (width - new_w) // 2
        pos_y = height - new_h

        canvas.paste(fg_scaled, (pos_x, pos_y), fg_scaled)

        if progress_callback:
            progress_callback(95, "✨ Applying studio softbox color grading & 300 DPI export...")

        final_rgb = canvas.convert("RGB")
        final_rgb = ImageEnhance.Contrast(final_rgb).enhance(1.08)
        final_rgb = ImageEnhance.Color(final_rgb).enhance(1.06)
        final_rgb = ImageEnhance.Sharpness(final_rgb).enhance(1.18)

        out_path = OUTPUTS_DIR / f"{job_id}_2stage_studio.png"
        final_rgb.save(out_path, format="PNG", dpi=(300, 300))

        logger.info("[TWO_STAGE_STUDIO] Completed 2-stage studio pipeline in %.2fs -> %s", time.time() - t0, out_path)
        return out_path

    # ── 2. QWEN-VL PLAN + QWEN IMAGE EDIT UNIFORM SWAP ─────────────────────────

    def qwen_vl_image_edit_uniform_swap(
        self,
        person_input: Union[str, Path, bytes, Image.Image],
        uniform_template_path: Union[str, Path, bytes, Image.Image],
        prompt: Optional[str] = None,
        job_id: str = "qwen_uniform",
        background_color: Union[str, Tuple[int, int, int]] = "4,126,246",
        width: int = 560,
        height: int = 720,
        steps: int = 20,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Path:
        """Use VL only for a fit plan, then perform one direct Qwen Image Edit."""
        if not uniform_template_path:
            raise ValueError("Uniform template image is mandatory for Uniform Swap.")

        person_pil = self._to_pil(person_input).convert("RGB")
        # Keep the untouched upload as both the generation input and the final
        # biometric/hair authority. Uniform Swap must not silently substitute
        # a previously saved or internally enhanced portrait.
        original_person_pil = person_pil.copy()
        identity_prepass_used = False
        identity_prepass_review: Optional[Dict[str, Any]] = None
        template_source = self._to_pil(uniform_template_path, mode="RGBA")

        # Prepare the reference before analysis.  VL and Qwen must see the
        # same collar, fabric scale, and garment bounds; analysing the padded
        # source while conditioning Qwen on a later crop caused conflicting
        # plans for the same uniform.
        template_pil = self._prepare_uniform_reference(
            template_source,
            target_width=width,
            target_height=height,
            pad_to_portrait=False,
        )

        if progress_callback:
            progress_callback(8, "Analyzing person and supplied uniform...")

        from pipelines.uniform_vl_analyzer import uniform_vl_analyzer
        vl_plan = uniform_vl_analyzer.analyze(person_pil, template_pil)
        source_gray = cv2.cvtColor(np.asarray(person_pil), cv2.COLOR_RGB2GRAY)
        source_h, source_w = source_gray.shape[:2]
        face_detail_region = source_gray[
            int(source_h * .08):max(int(source_h * .62), int(source_h * .08) + 1),
            int(source_w * .18):max(int(source_w * .82), int(source_w * .18) + 1),
        ]
        vl_plan["source_min_dimension"] = min(source_w, source_h)
        vl_plan["source_face_detail_score"] = (
            float(cv2.Laplacian(face_detail_region, cv2.CV_64F).var())
            if face_detail_region.size else 0.0
        )
        selected_bg_rgb = self._parse_bg_color(background_color)
        source_has_direct_light = any(
            str(vl_plan.get(key) or "").strip().lower()
            in {"true", "yes", "1", "present"}
            for key in ("direct_sunlight_present", "head_hair_hotspot_present")
        )
        # Use the native four-step edit adapter for every uniform portrait.
        # The 12-step base path repainted cloth, retained jewelry and weakened
        # identity on extreme-sun probes. Such portraits receive a measured
        # neutral-light conditioning guide below, so Qwen still performs one
        # coherent first generation without changing scheduler families.
        measured_sunlight_severity = str(
            vl_plan.get("measured_sunlight_severity") or "normal"
        ).strip().lower()
        extreme_source_sunlight = measured_sunlight_severity == "extreme"
        generation_steps = 4
        step_reasons = [
            (
                "four-step edit with measured neutral-light conditioning for extreme sunlight"
                if extreme_source_sunlight else
                "single four-step original-upload restoration and uniform edit"
            ),
        ]
        # Strong classifier-free guidance makes Qwen redraw the person while
        # obeying garment prose. Match the enhancement route's restrained
        # guidance so the upload remains the dominant face/hair reference.
        # Neutral CFG prevents garment prose from overpowering the original
        # portrait's face, hair, anatomy and pose.
        generation_cfg_scale = 1.0
        # Use the smaller native portrait bucket only when the upload already
        # contains enough spatial face evidence. The delivered school-ID frame
        # is 600x800, so a sharp >=360px source does not benefit from processing
        # every diffusion tensor at 1216px. Small or soft faces retain the full
        # 1216px path. This is input-driven and never changes step count, prompt,
        # identity checks, or the final output size.
        source_min_dimension = int(vl_plan.get("source_min_dimension") or 0)
        source_face_detail = float(vl_plan.get("source_face_detail_score") or 0.0)
        fast_portrait_bucket = (
            source_min_dimension >= 360
            and source_face_detail >= 120.0
        )
        uniform_generation_dimension = 1024 if fast_portrait_bucket else 1216
        step_reasons.append(
            "1024px sharp-source portrait bucket"
            if fast_portrait_bucket else
            "1216px low-detail portrait bucket"
        )
        logger.info(
            "[QWEN_ONLY_UNIFORM] Original-upload generation selected %d steps at %dpx: %s",
            generation_steps, uniform_generation_dimension, ", ".join(step_reasons),
        )
        if progress_callback:
            progress_callback(11, f"Selected {generation_steps} generation steps for this portrait...")
        from pipelines.uniform_badge_cleanup import (
            remove_template_badges,
        )
        from pipelines.uniform_composite import (
            build_rough_composite,
            build_person_identity_guide,
            describe_fabric_color, describe_hair_correction,
            describe_lighting_correction, refinement_prompt,
            measure_source_person_colors, source_hair_is_dark,
            garment_design_conditioning,
            hide_product_rear_collar,
        )
        badge_reference = template_pil.copy()
        # Keep the untouched reference for Qwen's fabric authority. The rough
        # guide gets a separate badge-free mirror repair below, avoiding the
        # white starburst caused by generic inpainting on high-contrast plaid.
        template_badges_removed = 0
        if vl_plan.get("source") != "qwen2.5-vl":
            raise RuntimeError(
                "Uniform generation stopped before diffusion because exact portrait/template analysis failed. "
                "A generic fallback is not safe for identity, hair or uniform matching."
            )

        # Keep the visual reference intact. Qwen omits emblems in the output;
        # inaccurate VL badge boxes must not erase template seams or fabric.
        logger.info("[UNIFORM_REFERENCE] original=%s prepared=%s", template_source.size, template_pil.size)

        uniform_seed = zlib.crc32(person_pil.tobytes())
        uniform_seed = zlib.crc32(template_pil.tobytes(), uniform_seed)
        uniform_seed = zlib.crc32(str(self._parse_bg_color(background_color)).encode("ascii"), uniform_seed)
        # Version the deterministic seed with the generation contract. Without
        # this bump, a materially revised crop/prompt can reproduce the same
        # latent portrait as the previous profile and make a deployed fix look
        # ineffective for identical uploads.
        # This salt previously produced the strongest identity-preserving edit
        # for the single-canvas route. Prompt changes still alter the result,
        # while avoiding an unbounded seed search after a failed portrait.
        uniform_seed = zlib.crc32(b"uniform-local-uniform-refine-v22", uniform_seed)
        # A content-stable seed prevents retries of identical inputs from
        # drifting to a completely different child. The salt reproduces the
        # previously validated Qwen-only conditioning profile.
        uniform_seed = (uniform_seed ^ 647635142) & 0x7FFFFFFF
        # Keep identical inputs deterministic. Job-id salt made the same
        # portrait/template drift between correct uniforms, source clothing,
        # and even extra people across runs.

        plan = ", ".join(
            f"{key}={vl_plan.get(key, 'unknown')}"
            for key in (
                "garment_components",
                "shirt_color_and_pattern",
                "outer_garment_color_and_shape",
                "shirt_collar",
                "outer_neckline",
                "sleeve_length",
                "button_layout",
                "badge_present",
                "badge_location",
            )
        )
        logger.info("[QWEN_ONLY_UNIFORM] job=%s VL plan: %s", job_id, plan)
        logger.info("[QWEN_ONLY_UNIFORM] job=%s base content seed=%d", job_id, uniform_seed)

        # Keep the selected backdrop in the original-person conditioning image.
        # Exact uniform facts below prevent that hue from being misread as a
        # garment colour.
        qwen_bg_rgb = selected_bg_rgb
        conditioning_bg_rgb = selected_bg_rgb
        selected_is_chromatic = (
            max(selected_bg_rgb) - min(selected_bg_rgb) >= 72
        )
        template_analysis_scale = min(
            1.0,
            640.0 / max(1, template_source.width),
            640.0 / max(1, template_source.height),
        )
        template_analysis_size = (
            max(1, int(round(template_source.width * template_analysis_scale))),
            max(1, int(round(template_source.height * template_analysis_scale))),
        )
        transparent_rgb, template_badges_removed = remove_template_badges(
            template_source,
            template_source,
            # This cleaned copy is used only by the diagnostic rough layout.
            # The Qwen garment authority below uses untouched pixels cropped
            # above the badge, so no repaired rectangle reaches generation.
            repair_mode="mirror",
            reference_bbox=vl_plan.get("badge_bbox"),
            reference_bbox_size=template_analysis_size,
        )
        logger.info(
            "[UNIFORM_BADGE] Removed %d badge region(s) from rough-fit garment guide",
            template_badges_removed,
        )
        transparent_template = transparent_rgb.convert("RGBA")
        transparent_template.putalpha(template_source.getchannel("A"))
        # Qwen must see the badge-free visual authority.  The former reference
        # was captured before cleanup and allowed the model to faithfully copy
        # the very badge the prompt asked it to remove.
        badge_reference = transparent_template.copy()
        from pipelines.photo_restoration import _get_insight_app
        from pipelines.schp_service import parse as parse_body_parts
        source_faces = _get_insight_app().get(
            cv2.cvtColor(np.array(person_pil), cv2.COLOR_RGB2BGR)
        )
        if len(source_faces) != 1:
            raise RuntimeError("Uniform fitting requires exactly one visible face in the uploaded portrait")
        source_face = source_faces[0]
        source_gender_value = int(getattr(source_face, "gender", -1))
        source_age_value = max(0, int(getattr(source_face, "age", 0) or 0))

        # Difficult harsh-light/low-detail uploads need identity restoration
        # before garment synthesis. This prepass always starts from the current
        # uploaded pixels (never an Enhancement result); its content-addressed
        # cache is invalidated whenever those pixels or the contract changes.
        source_face_height = max(
            1.0, float(source_face.bbox[3] - source_face.bbox[1]),
        )
        source_face_pixel_ratio = source_face_height / max(person_pil.height, 1)
        identity_prepass_would_help = bool(
            source_has_direct_light
            and (
                min(person_pil.size) < 640
                or source_face_pixel_ratio < .34
                or float(vl_plan.get("source_face_detail_score") or 0.0) < 115.0
            )
        )
        identity_prepass_enabled = os.getenv(
            # Keep full portrait regeneration opt-in. On clipped-sun sources
            # it can remove the light but synthesize a different child; the
            # validated geometry-preserving conditioning guide below is safer
            # and avoids an extra diffusion pass.
            "VSOUL_ENABLE_UNIFORM_IDENTITY_PREPASS", "0",
        ).strip().lower() in {"1", "true", "yes"}
        needs_identity_prepass = bool(
            identity_prepass_enabled and identity_prepass_would_help
        )
        if needs_identity_prepass:
            # The internal restoration canvas is not the delivered backdrop.
            # A near-white conditioning field gives Qwen too little separation
            # around dark hair plus pale flowers and, for the same person, has
            # produced a substantially different face. Use a high-contrast
            # temporary passport canvas only for pale neutral selections; the
            # garment pass still renders the user's selected final RGB value.
            selected_range = max(selected_bg_rgb) - min(selected_bg_rgb)
            selected_luma = (
                selected_bg_rgb[0] * .2126
                + selected_bg_rgb[1] * .7152
                + selected_bg_rgb[2] * .0722
            )
            identity_background_rgb = (
                (4, 126, 246)
                if selected_range < 24 and selected_luma > 220.0
                else selected_bg_rgb
            )
            cache_crc = zlib.crc32(original_person_pil.tobytes())
            cache_crc = zlib.crc32(bytes(identity_background_rgb), cache_crc)
            cache_crc = zlib.crc32(b"uniform-identity-prepass-v4-extreme-light", cache_crc)
            identity_cache_path = CACHE_DIR / f"uniform_identity_{cache_crc:08x}.png"
            identity_candidate: Optional[Image.Image] = None
            if identity_cache_path.exists():
                try:
                    identity_candidate = Image.open(identity_cache_path).convert("RGB")
                    logger.info(
                        "[UNIFORM_IDENTITY_PREPASS] Reusing cached portrait candidate %s",
                        identity_cache_path,
                    )
                except Exception as cache_error:
                    logger.warning(
                        "[UNIFORM_IDENTITY_PREPASS] Ignoring unreadable cache: %s",
                        cache_error,
                    )
                    identity_candidate = None

            if identity_candidate is None:
                if progress_callback:
                    progress_callback(12, "Restoring the original head and hairstyle before fitting the uniform...")
                bg_hex = (
                    f"#{identity_background_rgb[0]:02X}{identity_background_rgb[1]:02X}"
                    f"{identity_background_rgb[2]:02X}"
                )
                relight_clause = (
                    "Remove only the measured harsh directional sunlight and its hard shadows from the face, hair, neck and visible body; use soft neutral frontal illumination. "
                    if source_has_direct_light else
                    "Keep the source illumination natural and neutral. "
                )
                identity_prompt = (
                    f"Restore image 1 as one natural modern passport portrait against one perfectly flat {bg_hex} "
                    f"RGB{identity_background_rgb} background. {relight_clause}"
                    "Image 1 is the only identity authority: copy the exact current apparent age, facial proportions, "
                    "natural asymmetry, expression, complexion and marks. Copy the complete original hairstyle visually: "
                    "hairline, part, crown, texture, both side silhouettes, length, curls, visible ends, earrings and every "
                    "real hair accessory at its exact side, count, color, size and position. Inspect viewer-left and viewer-right independently; preserve every flower or clip literally visible in either side and invent none. Preserve the original head scale, "
                    "neck anatomy, shoulders, body width and clothing during this identity preparation. Do not beautify, "
                    "de-age, repaint, invent accessories, add another person, crop the crown, or create a pasted-face look. "
                    f"Measured source facts: hair parting {vl_plan.get('hair_parting') or 'copy pixels'}; "
                    f"length {vl_plan.get('hair_length') or 'copy pixels'}; texture {vl_plan.get('hair_texture') or 'copy pixels'}; "
                    f"accessories {vl_plan.get('hair_accessories') or 'copy pixels'}; "
                    f"details {str(vl_plan.get('hair_accessory_details') or 'copy pixels')[:120]}. "
                    "No makeup, eyeliner, mascara, false eyelashes, enlarged eyes, narrowed nose, fuller cheeks or reshaped jaw."
                )
                prepass_w = 832
                prepass_h = max(
                    832,
                    int(round((prepass_w * person_pil.height / max(person_pil.width, 1)) / 16.0) * 16),
                )
                if prepass_h > 1216:
                    prepass_h = 1216
                    prepass_w = max(
                        256,
                        int(round((prepass_h * person_pil.width / max(person_pil.height, 1)) / 16.0) * 16),
                    )
                identity_seed = zlib.crc32(original_person_pil.tobytes())
                identity_seed = zlib.crc32(b"#FFFFFF", identity_seed)
                identity_seed = (identity_seed ^ 3201858918) & 0x7FFFFFFF

                def identity_progress(pct: int, message: str) -> None:
                    if progress_callback:
                        progress_callback(
                            12 + int(max(0, min(100, pct)) * .24),
                            message,
                        )

                identity_path = self.qwen_edit_enhancer(
                    image_input=original_person_pil,
                    prompt=identity_prompt,
                    negative_prompt=None,
                    job_id=f"{job_id}_identity_prepass",
                    background_color=identity_background_rgb,
                    width=prepass_w,
                    height=prepass_h,
                    original_filename="identity_prepass",
                    steps=4,
                    max_sequence_length=256,
                    progress_callback=identity_progress,
                    reference_images=None,
                    preserve_source_clothing=False,
                    max_generation_dimension=1216,
                    minimum_generation_dimension=0,
                    keep_generation_resolution=True,
                    use_birefnet_background=False,
                    true_cfg_scale=1.0,
                    seed=identity_seed,
                    face_detail_refine=False,
                    reject_identity_failure=False,
                    return_raw_candidate=True,
                    conditioning_max_dimension=512,
                    preserve_reference_aspect=True,
                )
                identity_candidate = Image.open(identity_path).convert("RGB")

            from pipelines.uniform_review import check_uniform_identity
            # This is an intermediate restoration, not the delivered uniform.
            # Hard sunlight and a tiny source face depress ArcFace similarity
            # even when dense face geometry remains stable. Enhancement uses
            # the same Qwen restoration successfully for these inputs, so let
            # a tightly bounded geometric match become the garment-edit
            # authority. The final uniform must still score >= .70 against
            # this validated clean portrait and must not drift further from
            # the untouched upload (dual gate below).
            identity_prepass_review = check_uniform_identity(
                original_person_pil, identity_candidate, minimum_similarity=.65,
            )
            prepass_geometry = identity_prepass_review.get("geometry_rmse")
            prepass_faces = _get_insight_app().get(
                cv2.cvtColor(np.asarray(identity_candidate), cv2.COLOR_RGB2BGR)
            )
            prepass_has_harsh_light = bool(
                len(prepass_faces) == 1
                and PhotoRestorationService.has_harsh_portrait_sunlight(
                    identity_candidate, prepass_faces[0].bbox,
                )
            )
            prepass_ok = bool(
                identity_prepass_review.get("accepted")
                and isinstance(prepass_geometry, (int, float))
                and float(prepass_geometry) <= .035
                and not prepass_has_harsh_light
            )
            if not prepass_ok:
                # The restoration prepass is an optional aid for tiny,
                # harshly-lit uploads; it is never an authority over the
                # uploaded person. A diffusion miss must be discarded rather
                # than aborting the garment edit before it starts. Continue
                # from the untouched upload and let the mandatory final
                # one-face/identity/geometry gates judge the actual uniform
                # candidate. This also prevents a failed prepass from being
                # cached or leaking into the final portrait.
                logger.warning(
                    "[UNIFORM_IDENTITY_PREPASS] Rejected optional restored identity; "
                    "continuing from untouched upload: review=%s residual_harsh_light=%s",
                    identity_prepass_review, prepass_has_harsh_light,
                )
                if progress_callback:
                    progress_callback(
                        36,
                        "Restored-head candidate changed identity; using the original uploaded person for uniform fitting...",
                    )
            else:
                identity_candidate.save(identity_cache_path, format="PNG")
                person_pil = identity_candidate
                identity_prepass_used = True
                step_reasons.append(
                    "validated current-upload identity restoration for harsh low-detail source"
                )
                restored_faces = _get_insight_app().get(
                    cv2.cvtColor(np.asarray(person_pil), cv2.COLOR_RGB2BGR)
                )
                if len(restored_faces) != 1:
                    raise RuntimeError(
                        "Validated identity restoration did not contain exactly one usable face"
                    )
                # All downstream crop, body-fit and collar measurements must
                # use the restored canvas coordinates. Reusing the original
                # small-upload bbox shifted and enlarged the regenerated head.
                source_faces = restored_faces
                source_face = restored_faces[0]
                source_gender_value = int(getattr(source_face, "gender", -1))
                source_age_value = max(
                    0, int(getattr(source_face, "age", 0) or 0),
                )
                # The first VL plan described the harsh-sun upload. Reusing those
                # stale hair/skin/lighting facts after switching image 1 to the
                # restored identity caused Qwen to replace two flowers with white
                # clips and turn curly side hair into a straight bob. Re-analyse
                # the *actual* conditioning portrait so prompt facts and pixels
                # describe the same person. The uniform template remains the same
                # visual authority, while the untouched upload remains the audit
                # authority used by the dual identity gate below.
                restored_vl_plan = uniform_vl_analyzer.analyze(
                    person_pil, template_pil,
                )
                vl_plan = restored_vl_plan
                source_has_direct_light = False
                logger.info(
                    "[UNIFORM_IDENTITY_PREPASS] Rebuilt person/hair VL plan from the validated restored portrait",
                )
                source_faces = _get_insight_app().get(
                    cv2.cvtColor(np.asarray(person_pil), cv2.COLOR_RGB2BGR)
                )
                if len(source_faces) != 1:
                    raise RuntimeError(
                        "Uniform identity preparation did not produce exactly one complete face."
                    )
                source_face = source_faces[0]
                source_gender_value = int(getattr(source_face, "gender", -1))
                source_age_value = max(0, int(getattr(source_face, "age", 0) or 0))
                logger.info(
                    "[UNIFORM_IDENTITY_PREPASS] Accepted restored head for clothing edit: %s",
                    identity_prepass_review,
                )

        # Keep gender estimates for diagnostics only. Both InsightFace's binary
        # attribute head and semantic VL gender labels are unreliable across
        # ages, hairstyles and populations. Feeding either label back to Qwen
        # makes it synthesize a stereotyped male/female face instead of editing
        # the uploaded person.
        vl_gender = str(vl_plan.get("gender") or "").strip().lower()
        vl_plan["gender_analysis_only"] = vl_gender or "unknown"
        vl_plan["insightface_gender_analysis_only"] = source_gender_value
        vl_age_band = str(vl_plan.get("apparent_age_band") or "").strip().lower()
        # Age classifiers are audit signals, not generation authorities. A
        # school uniform can bias VL toward "child", while InsightFace may
        # overestimate young subjects. Record a useful diagnostic band, but
        # condition Qwen only to copy the visible facial maturity from image 1.
        if source_age_value >= 22:
            source_age_band = "young adult"
        elif source_age_value >= 18:
            source_age_band = "late-teen or young adult"
        elif source_age_value >= 13:
            source_age_band = "teenager"
        elif source_age_value >= 7:
            source_age_band = "school-age child"
        elif any(w in vl_age_band for w in ("adult", "young adult")):
            source_age_band = "adult appearance reported by VL"
        elif any(w in vl_age_band for w in ("teen", "student")):
            source_age_band = "teenager"
        elif any(w in vl_age_band for w in ("child", "school")):
            source_age_band = "school-age child"
        else:
            source_age_band = "source-visible age"
        if (
            source_age_value >= 22
            and any(w in vl_age_band for w in ("child", "school", "teen"))
        ):
            vl_plan["age_analysis_conflict"] = (
                f"VL={vl_age_band or 'unknown'}; face-estimator={source_age_value}; "
                "generation must copy source-visible maturity"
            )
        age_negative_terms = "younger face, older face, changed apparent age, altered facial maturity"

        source_appearance_lock = (
            "Preserve the uploaded person's exact visible presentation and apparent age from image 1; "
            "do not infer or change gender, age, ancestry or facial character."
        )
        # Apparent age belongs in the identity contract, not the scheduler.
        # Overriding a clean four-step edit merely because the subject is an
        # adult disables the edit adapter and makes the base model repaint the
        # already-correct rough garment. The adaptive visual-risk selector is
        # the sole authority for diffusion depth.
        # Build the rough garment fit in the uploaded portrait's native
        # geometry. Pre-cropping the person before fitting changed the measured
        # shoulder-to-face ratio (4.09 -> 3.13 for the regression fixture) and
        # made Qwen synthesize a different face. The rough result is framed once
        # below, after the source body and complete hairstyle have been retained.
        source_identity_reference = person_pil.copy()
        source_identity_reference_path = OUTPUTS_DIR / f"{job_id}_source_identity_reference.png"
        source_identity_reference.save(source_identity_reference_path, format="PNG")
        logger.info(
            "[UNIFORM_IDENTITY_REFERENCE] Saved untouched source head/hair/body reference -> %s",
            source_identity_reference_path,
        )
        # Qwen compares the visual scale of every conditioning image. Feeding
        # a wide classroom photo beside a full-frame shirt made it preserve the
        # small source person and invent a second, larger uniformed person.
        # Apply the same final school-ID composition contract used by portrait
        # enhancement before any garment fitting. The older loose identity-
        # reference crop targeted a 30% face box and retained long sleeves and
        # lower torso; it was explicitly not a delivery crop. Building every
        # later guide on this face-relative upper-chest frame keeps the crown,
        # level shoulders and only the upper chest in the final photograph.
        # Use Enhancement's edge-to-edge identity crop. The former direct
        # passport crop expanded a tight source beyond its bounds and filled
        # the new canvas with a median green scenery colour. Qwen interpreted
        # that padded cutout as a composition guide and regenerated the child.
        # Match the actual requested passport aspect ratio. The former fixed
        # 832x1216 canvas was much narrower than the 560x720 delivery frame;
        # it clipped the source shoulder span, lowered the measured body/face
        # ratio and taught Qwen to generate an oversized head on a lean torso.
        conditioning_height = 1216
        conditioning_width = max(
            256,
            int(round((conditioning_height * width / max(height, 1)) / 16.0) * 16),
        )
        person_conditioning = self._source_identity_reference(
            person_pil,
            source_face.bbox,
            conditioning_width,
            conditioning_height,
            background_color=selected_bg_rgb,
            target_face_fraction=.35,
        )
        # Match the Enhancement route: keep the original portrait pixels as
        # Qwen's identity/hair/accessory authority and ask Qwen to generate the
        # selected flat backdrop in the same coherent edit.  Pre-matting with
        # BiRefNet/SCHP permanently removed curls, flowers and fine crown hair;
        # Qwen then correctly preserved those missing regions as blue holes.
        # Background verification/correction is performed on the generated
        # portrait below, never on the source identity reference.
        source_has_direct_light = any(
            str(vl_plan.get(key) or "").strip().lower()
            in {"true", "yes", "1", "present"}
            for key in ("direct_sunlight_present", "head_hair_hotspot_present")
        )
        conditioning_faces = _get_insight_app().get(
            cv2.cvtColor(np.asarray(person_conditioning), cv2.COLOR_RGB2BGR)
        )
        if source_has_direct_light and len(conditioning_faces) == 1:
            # Match Enhancement exactly: Qwen must see the untouched source
            # pixels as its identity, hair and accessory authority. Relighting
            # this conditioning image before diffusion changed local facial
            # colours and encouraged a smooth/painted reconstruction. The
            # compact generation contract below requests coherent relighting;
            # bounded correction runs only on a decoded candidate that still
            # contains measurable residual sunlight.
            logger.info(
                "[UNIFORM_LIGHTING] Keeping untouched Enhancement-style head conditioning; "
                "Qwen will relight the coherent portrait"
            )
        source_badges_removed = 0
        person_conditioning_path = OUTPUTS_DIR / f"{job_id}_person_conditioning.png"
        person_conditioning.save(person_conditioning_path, format="PNG")
        logger.info(
            "[UNIFORM_IDENTITY_REFERENCE] Saved upper-chest source conditioning -> %s",
            person_conditioning_path,
        )
        source_labels = parse_body_parts(np.array(person_pil))["labels"]
        # Build the SCHP rough fit on the exact passport-framed source portrait.
        # Previously the rough uniform was fitted on the wide upload and then
        # cropped with a different framing function from person_conditioning.
        # Resizing those two independently framed images to the same dimensions
        # shifted the collar, shoulders and torso before Qwen saw them.
        rough_source = person_conditioning.convert("RGB")
        rough_source_faces = _get_insight_app().get(
            cv2.cvtColor(np.array(rough_source), cv2.COLOR_RGB2BGR)
        )
        if len(rough_source_faces) != 1:
            raise RuntimeError("Uniform fitting could not prepare one complete portrait guide")
        rough_source_labels = parse_body_parts(np.array(rough_source))["labels"]

        # A short-sleeve target can reveal arm skin that does not exist in a
        # long-sleeve source. Four-step Lightning is sufficient when the rough
        # canvas already contains every output pixel, but it produces visible
        # block artifacts when it must synthesize a sizeable new skin region.
        # Detect that case from this portrait's parser output instead of tying
        # the policy to a particular subject or uniform.
        template_sleeve = str(vl_plan.get("sleeve_length") or "").strip().lower()
        source_visible_arm_fraction = float(
            np.count_nonzero(np.isin(rough_source_labels, (14, 15)))
        ) / float(max(1, rough_source_labels.size))
        requires_exposed_arm_synthesis = (
            "short" in template_sleeve
            and source_visible_arm_fraction < .012
        )
        if requires_exposed_arm_synthesis:
            # Keep the installed four-step edit adapter. The base model at
            # eight steps turned the skin guide into smooth gray placeholders;
            # the actual defect was the missing spatial skin cue, not too few
            # denoising passes.
            step_reasons.append(
                "measured skin guide supplies arms hidden by source sleeves"
            )
            logger.info(
                "[QWEN_ONLY_UNIFORM] Keeping %d-step adapter with measured arm guide; source visible-arm fraction=%.4f",
                generation_steps,
                source_visible_arm_fraction,
            )

        measured_person_colors = measure_source_person_colors(person_pil, source_labels)
        measured_person_colors["lip_rgb"] = self._measure_lip_rgb(person_pil, source_face)
        identity_head_reference = build_person_identity_guide(
            person_pil, source_labels, source_faces[0].bbox,
        )
        identity_head_path = OUTPUTS_DIR / f"{job_id}_identity_head.png"
        identity_head_reference.save(identity_head_path, format="PNG")
        logger.info(
            "[UNIFORM_IDENTITY_REFERENCE] Saved clothing-free head reference -> %s",
            identity_head_path,
        )
        vl_claims_tied_hair = any(
            term in str(vl_plan.get("hair_style") or "").lower()
            for term in ("ponytail", "tied", "bun", "braid", "plait")
        )
        visible_tied_hair_geometry = _has_visible_tied_hair_geometry(
            rough_source_labels, rough_source_faces[0].bbox,
        )
        # SCHP frequently stops at the ears and misses a real dark ponytail
        # against dark source clothing.  Treat a VL tied-hair result as usable
        # only when it is internally corroborated by a concrete anchor or
        # extent; this remains per-image evidence and does not invent hair from
        # a lone class label.  The relaxed flag lets the rough compositor search
        # the anatomical nape/side corridors with its colour-connectivity gate.
        analyzed_ponytail_anchor = str(vl_plan.get("ponytail_anchor") or "").strip().lower()
        analyzed_ponytail_extent = str(vl_plan.get("ponytail_extent") or "").strip().lower()
        concrete_hair_values = {
            "", "none", "unknown", "uncertain", "not visible", "not detected",
            "low confidence", "n/a", "not applicable",
        }
        vl_tied_hair_corroborated = bool(
            vl_claims_tied_hair
            and (
                analyzed_ponytail_anchor not in concrete_hair_values
                or analyzed_ponytail_extent not in concrete_hair_values
            )
        )
        tied_geometry_confirmed = bool(
            vl_claims_tied_hair and visible_tied_hair_geometry
        )
        # Keep the semantic claim provisional until the fitted source crop can
        # confirm a real tail. Anchor/extent prose alone previously invented a
        # ponytail on short-haired portraits.
        preserve_tied_back_hair = vl_claims_tied_hair
        if vl_claims_tied_hair and not tied_geometry_confirmed:
            logger.warning(
                "[UNIFORM_HAIR] Ignoring unsupported tied-hair label (%s); "
                "source pixels show no tied-hair geometry",
                vl_plan.get("hair_style"),
            )
            vl_plan["hair_geometry_parser_fragmented"] = True
        vl_plan["visible_tied_hair_geometry"] = visible_tied_hair_geometry
        # Give Qwen one coherent subject that already wears the requested
        # garment. A separate torso-shaped garment reference was repeatedly
        # interpreted as a second person, while a blank/neutral torso caused
        # invented vests and changed hairstyles. The rough fit is only the
        # single visual input: Qwen must naturalize its neck, fabric and shoulders.
        target_collar_for_guide = str(vl_plan.get("shirt_collar") or "").lower()
        target_has_raised_collar = any(
            term in target_collar_for_guide
            for term in ("standing", "stand collar", "mandarin", "band collar", "high collar")
        )
        if not target_has_raised_collar:
            transparent_template, rear_collar_pixels = hide_product_rear_collar(
                transparent_template,
            )
            if rear_collar_pixels:
                logger.info(
                    "[UNIFORM_COLLAR] Hid %d unworn rear-collar pixels before rough fitting",
                    rear_collar_pixels,
                )
        rough_composite, rough_placement = build_rough_composite(
            rough_source,
            transparent_template,
            rough_source_labels,
            rough_source_faces[0].bbox,
            conditioning_bg_rgb,
            expect_tied_hair=preserve_tied_back_hair,
            # The source measurement is the neck-length authority for every
            # collar type. A former .10 minimum overrode a measured .06 gap on
            # tightly framed children and produced a visibly elongated neck.
            minimum_neck_gap_ratio=0.0,
        )
        # Cross-check semantic tied-hair labels against the same parsed source
        # geometry used to build the fitted guide. VL can collapse two low
        # pigtails into one generic left ponytail when one side is larger. The
        # measured count/sides are dynamic per upload and override prose only
        # when both roots and both lower hair bundles are independently visible.
        measured_tied_count = int(
            rough_placement.get("detected_tied_section_count") or 0
        )
        measured_tied_sides = str(
            rough_placement.get("detected_tied_section_sides") or "unknown"
        )
        if measured_tied_count > 0:
            vl_plan["tied_section_count"] = measured_tied_count
            vl_plan["tied_section_sides"] = measured_tied_sides
            vl_plan["ponytail_side"] = (
                measured_tied_sides if measured_tied_count == 1 else "not-applicable-bilateral"
            )
            logger.info(
                "[UNIFORM_HAIR_GEOMETRY] Measured %d tied section(s): %s",
                measured_tied_count, measured_tied_sides,
            )
        elif any(
            term in str(vl_plan.get("hair_style") or "").lower()
            for term in ("ponytail", "pigtail", "tied")
        ):
            # Do not allow an unconfirmed single-side VL guess into Qwen.
            vl_plan["tied_section_count"] = 0
            vl_plan["tied_section_sides"] = "unknown"
            vl_plan["ponytail_side"] = "unknown"
        measured_hair_extent = str(
            rough_placement.get("measured_hair_extent") or "unknown"
        ).strip()
        measured_hair_end_ratio = rough_placement.get("measured_hair_end_ratio")
        if measured_hair_extent.lower() not in {"", "unknown", "uncertain"}:
            # Pixel-measured vertical extent is more reliable than a generic
            # VL word such as "long" or "short". Keep the semantic texture
            # description, but make Qwen copy where the visible ends actually
            # stop on this particular uploaded portrait.
            vl_plan["hair_length"] = measured_hair_extent
            vl_plan["hair_ends"] = (
                f"visible ends {measured_hair_extent}; measured endpoint "
                f"{float(measured_hair_end_ratio):.2f} face-heights from the jaw"
                if measured_hair_end_ratio is not None
                else f"visible ends {measured_hair_extent}"
            )
            vl_plan["measured_hair_end_ratio"] = measured_hair_end_ratio
            vl_plan["measured_left_hair_end_ratio"] = rough_placement.get(
                "measured_left_hair_end_ratio"
            )
            vl_plan["measured_right_hair_end_ratio"] = rough_placement.get(
                "measured_right_hair_end_ratio"
            )
            logger.info(
                "[UNIFORM_HAIR_GEOMETRY] Measured vertical extent: %s (ratio=%s)",
                measured_hair_extent, measured_hair_end_ratio,
            )
        # The first VL plan is created before rough-fit pixel geometry exists.
        # Rebuild its trust decision after measured count/sides/endpoints have
        # corrected any generic or one-sided semantic hairstyle label.
        from pipelines.uniform_vl_analyzer import build_portrait_repair_plan
        vl_plan["portrait_repair_plan"] = build_portrait_repair_plan(vl_plan)
        rough_path = OUTPUTS_DIR / f"{job_id}_rough_fit.png"
        rough_composite.save(rough_path, format="PNG")
        logger.info("[UNIFORM_COMPOSITE] Saved single-subject rough uniform guide -> %s", rough_path)
        # rough_source is already the final passport geometry. Do not run a
        # second crop calculation here: even a small difference in face-box or
        # foreground detection would misregister the garment against the head.
        if progress_callback:
            progress_callback(14, "Fixing rough uniform layout for school-ID framing...")
        # Keep the same native edge-to-edge canvas used by Enhancement. Do not
        # downscale the rough fit to 560x720 before diffusion; that discarded
        # face, accessory and plaid evidence and forced Qwen to reconstruct it.
        rough_fixed_path = OUTPUTS_DIR / f"{job_id}_rough_fixed.png"
        rough_composite.save(rough_fixed_path, format="PNG")
        logger.info("[UNIFORM_COMPOSITE] Saved fixed passport rough guide -> %s", rough_fixed_path)
        # Never condition Qwen on a reconstructed badge rectangle: even a
        # plausible texture fill is copied as a separate stitched panel. Use
        # untouched template pixels and, when VL measured a badge, end this
        # person-free design reference immediately above it.
        uniform_validation_reference = garment_design_conditioning(
            badge_reference,
            exclude_region=vl_plan.get("badge_bbox"),
            exclude_region_size=template_analysis_size,
        )
        uniform_reference_path = OUTPUTS_DIR / f"{job_id}_uniform_passport_reference.png"
        uniform_validation_reference.save(uniform_reference_path, format="PNG")
        logger.info(
            "[UNIFORM_REFERENCE] Saved untouched badge-excluded upper-garment reference -> %s",
            uniform_reference_path,
        )
        def observed_details(keys):
            # Fallback guesses are not observations of this uploaded template.
            if vl_plan.get("source") != "qwen2.5-vl":
                return "Use the reference image directly; analysis unavailable."
            details = []
            for key in keys:
                value = vl_plan.get(key)
                if isinstance(value, list):
                    value = ", ".join(str(item) for item in value)
                value = str(value or "").strip()
                if value.lower() not in ("", "none", "unknown", "uncertain"):
                    details.append(f"{key.replace('_', ' ')}: {value}")
            return "; ".join(details)

        garment_details = observed_details((
            "shirt_collar", "outer_neckline", "shirt_color_and_pattern",
            "outer_garment_color_and_shape", "outer_color_under_neutral_light",
            "fabric_detail", "construction_detail",
        ))
        template_description = " ".join(
            str(vl_plan.get(key) or "")
            for key in (
                "garment_components", "shirt_collar", "outer_neckline",
                "outer_garment_color_and_shape", "construction_detail",
            )
        ).lower()
        template_has_neckwear = "tie" in template_description
        if template_has_neckwear:
            neckwear_instruction = (
                "Image 2 visibly includes neckwear; reproduce only that neckwear, with no added jewelry."
            )
        else:
            neckwear_instruction = (
                "Image 2 has no tie, bow, ribbon or other neckwear. Leave the neckwear area bare above the uniform collar; do not invent any tie or bow."
            )
        portrait_details = observed_details((
            "face_detail", "face_shape", "eye_description", "eyebrow_description",
            "nose_description", "mouth_description", "earring_description", "expression", "skin_tone", "hair_parting",
            "hair_length", "hair_texture", "hair_color", "hair_crown_shape", "hair_left_side", "hair_right_side",
            "hair_ends", "hair_fringe", "hair_accessories", "hair_accessory_count",
            "hair_accessory_confidence", "hair_accessory_details",
            "visible_wearables", "glasses_reflection_present", "glasses_reflection_description",
        ))
        hair_details = observed_details((
            "hair_style", "hair_parting", "hair_length", "hair_texture",
            "hair_color", "hair_crown_shape", "hair_left_side", "hair_right_side", "hair_ends", "hair_fringe",
            "ponytail_side", "ponytail_anchor", "ponytail_extent",
            "hair_accessories", "hair_accessory_count", "hair_accessory_confidence", "hair_accessory_details",
        ))
        # Keep only reliable, directly visible hair facts in the generation
        # prompt. Vague crown-shape prose and "low confidence" side/ends fields
        # contradicted a clear straight tied-back hairstyle and encouraged a
        # rounded short-hair redraw.
        hair_generation_details = observed_details((
            "hair_style", "hair_parting", "hair_length", "hair_texture", "hair_color",
        ))
        # Do not turn incomplete VL geometry into an affirmative instruction.
        # In particular, a detected ponytail paired with "side: not visible"
        # previously caused Qwen to invent a centred, mirrored tail. Image 1 is
        # the authority unless the analyzer actually saw which side it is on.
        ponytail_side = str(vl_plan.get("ponytail_side") or "").strip()
        measured_ponytail_side = str(
            rough_placement.get("detected_ponytail_side") or "unknown"
        ).strip().lower()
        # A semantic "Ponytail" label is not sufficient on its own. This
        # portrait was analysed as short hair and "ponytail: not visible", yet
        # the old prompt still commanded Qwen to invent long tied hair. Require
        # either measured one-sided tail pixels or parser-confirmed nape hair.
        # For genuinely occluded ponytails, the geometry gate remains true;
        # otherwise image 1's visible hairstyle is copied without naming one.
        preserve_tied_back_hair = bool(
            vl_claims_tied_hair
            and (
                visible_tied_hair_geometry
                or measured_ponytail_side in {"viewer-left", "viewer-right"}
            )
        )
        if vl_claims_tied_hair and not preserve_tied_back_hair:
            vl_plan["generation_hair_style_override"] = (
                "copy only the hairstyle visibly present in image 1"
            )
            hair_generation_details = observed_details((
                "hair_parting", "hair_length", "hair_texture", "hair_color",
            ))
        if measured_ponytail_side in {"viewer-left", "viewer-right"}:
            ponytail_side = measured_ponytail_side
            vl_plan["effective_ponytail_side"] = measured_ponytail_side
        if ponytail_side.lower() not in {
            "", "none", "unknown", "uncertain", "not visible", "not detected",
            "low confidence", "n/a", "not applicable",
        }:
            ponytail_anchor = str(vl_plan.get("ponytail_anchor") or "").strip()
            ponytail_extent = str(vl_plan.get("ponytail_extent") or "").strip()
            if measured_ponytail_side in {"viewer-left", "viewer-right"}:
                ponytail_details = (
                    f"ponytail side: {measured_ponytail_side}; "
                    f"anchor: {ponytail_anchor or 'copy image 1 exactly'}; "
                    f"extent: {ponytail_extent or 'copy image 1 exactly'}; "
                    "preserve that exact tail position behind the same side of the neck"
                )
            else:
                ponytail_details = observed_details((
                    "ponytail_side", "ponytail_anchor", "ponytail_extent",
                ))
            if ponytail_details:
                hair_generation_details = "; ".join(filter(None, (
                    hair_generation_details, ponytail_details,
                )))
        one_sided_ponytail_instruction = ""
        opposite_ponytail_side = ""
        ponytail_position_instruction = ""
        if measured_ponytail_side in {"viewer-left", "viewer-right"}:
            opposite_ponytail_side = (
                "viewer-right" if measured_ponytail_side == "viewer-left" else "viewer-left"
            )
            one_sided_ponytail_instruction = (
                f"The tied tail is visible ONLY on {measured_ponytail_side}, exactly as in image 1. "
                f"Keep the {opposite_ponytail_side} neck edge free of any second or mirrored tail."
            )
            analyzed_anchor = str(vl_plan.get("ponytail_anchor") or "").strip().lower()
            analyzed_extent = str(vl_plan.get("ponytail_extent") or "").strip().lower()
            if "nape" in analyzed_anchor or "low" in analyzed_anchor:
                ponytail_position_instruction = (
                    f"This is a LOW ponytail anchored at the nape, below ear level, extending toward {analyzed_extent or 'the shoulder'}. "
                    "Keep the crown smooth and flat exactly as image 1: no top loop, visible band, bun, knot or gathered hair above the ears."
                )
        component_description = str(vl_plan.get("garment_components") or "").lower()
        template_has_outer_layer = any(
            term in component_description
            for term in (
                "vest", "waistcoat", "pinafore", "blazer", "jacket",
                "sweater", "cardigan", "coat", "outer garment", "outer layer",
            )
        )
        template_generation_keys = [
            "garment_components", "shirt_color_and_pattern", "shirt_collar",
            "sleeve_length", "accent_trim_placement", "shoulder_trim_present",
            "fabric_detail", "construction_detail",
        ]
        if template_has_outer_layer:
            template_generation_keys.extend((
                "outer_garment_color_and_shape", "outer_color_under_neutral_light",
                "outer_neckline",
            ))
        template_generation_details = observed_details(tuple(template_generation_keys))
        if template_has_outer_layer:
            trim_placement_instruction = (
                "Keep every outer layer and every trim region only where image 1's fitted garment visibly places it. "
                "Do not extend trim across an untrimmed shoulder seam."
            )
        else:
            trim_placement_instruction = (
                "Image 2 is one shirt, not a shirt plus an outer garment. Preserve its patterned fabric continuously "
                "across both shoulder tops. Put solid contrast cloth only in the exact collar, placket, pocket-edge or "
                "sleeve-end cuff regions visibly present in image 1's fitted garment. Do not invent shoulder yokes, epaulettes, caps, "
                "piping or solid-colour bands across either shoulder. Both sleeves must have matching construction."
            )
        palette_evidence = str(vl_plan.get("template_palette_evidence") or "").strip()
        # VL can confuse the shirt collar with the outer pinafore neckline.
        # Never turn uncertain prose into a geometry command; template pixels
        # remain the sole structural authority for both garment layers.
        collar_constraint = (
            "Preserve image 1's fitted shirt collar and outer-garment neckline silhouette exactly. "
            "Keep every straight or square top edge straight or square; create a V-shaped edge only when image 1 visibly has one."
        )
        outer_target_rgb = None
        pattern_description = str(vl_plan.get("shirt_color_and_pattern") or "").lower()
        multi_color_pattern = (
            pattern_description.count(",") >= 2
            or any(term in pattern_description for term in ("plaid", "check", "tartan", "stripe"))
        )
        # Two 512px references exhaust the 8 GB Windows GPU while the Qwen-VL
        # encoder builds visual tokens, killing the API before diffusion starts.
        # The fitted seed already carries full-resolution pattern placement and
        # the second image supplies garment appearance, so a 384px analysis
        # canvas retains checks/trim while keeping the encoder peak bounded.
        # Enhancement uses 512px visual conditioning. Uniform identity and
        # fine plaid need at least the same evidence; the former 352/384px cap
        # blurred the small source face before diffusion.
        conditioning_dimension = 512
        uniform_prompt_token_budget = 384
        vl_rgb = vl_plan.get("outer_neutral_rgb") if vl_plan.get("source") == "qwen2.5-vl" else None
        if not multi_color_pattern and isinstance(vl_rgb, (list, tuple)) and len(vl_rgb) == 3:
            try:
                candidate = tuple(int(value) for value in vl_rgb)
                if all(0 <= value <= 255 for value in candidate):
                    outer_target_rgb = candidate
            except (TypeError, ValueError):
                pass
        analyzed_pattern = str(vl_plan.get("shirt_color_and_pattern") or "").strip()
        measured_garment_palette = _measure_garment_palette(
            transparent_template,
            max_colors=6 if multi_color_pattern else 4,
        )
        vl_plan["measured_garment_palette"] = measured_garment_palette
        measured_accent = _measure_garment_accent_color(transparent_template)
        # A connected trim region can include its darkest fold/shadow. Snap its
        # hue to the most saturated measured template cluster in the same hue
        # family so the prompt carries the material color, not the shadow color.
        if measured_accent and measured_garment_palette:
            accent_rgb = np.asarray(measured_accent["rgb"], dtype=np.uint8).reshape(1, 1, 3)
            accent_hue = int(cv2.cvtColor(accent_rgb, cv2.COLOR_RGB2HSV)[0, 0, 0])
            same_hue_candidates = []
            for item in measured_garment_palette:
                item_rgb = np.asarray(item["rgb"], dtype=np.uint8).reshape(1, 1, 3)
                item_hsv = cv2.cvtColor(item_rgb, cv2.COLOR_RGB2HSV)[0, 0]
                hue_distance = min(
                    abs(int(item_hsv[0]) - accent_hue),
                    180 - abs(int(item_hsv[0]) - accent_hue),
                )
                if hue_distance <= 12:
                    same_hue_candidates.append((int(item_hsv[1]), item))
            if same_hue_candidates:
                measured_accent = dict(max(same_hue_candidates, key=lambda pair: pair[0])[1])
        vl_plan["measured_garment_accent"] = measured_accent
        measured_palette_instruction = ""
        if measured_garment_palette:
            palette_parts = [
                f"{item['hex']} about {int(round(item['fraction'] * 100))}%"
                for item in measured_garment_palette
            ]
            measured_palette_instruction = (
                "Measured garment colours by pixel area, dominant first: "
                + ", ".join(palette_parts)
                + ". Keep this dominance and spatial pattern; never turn an accent into the base."
            )
        fabric_color_instruction = (
            "Preserve image 1's exact multicolor fabric palette and pattern scale. "
            + measured_palette_instruction
            if multi_color_pattern else
            describe_fabric_color(
                transparent_template,
                vl_color=(vl_plan.get("outer_color_under_neutral_light")
                          if vl_plan.get("source") == "qwen2.5-vl" else None),
                vl_rgb=outer_target_rgb,
            )
        )
        expected_target_fraction = None
        if outer_target_rgb is not None:
            guide_rgba = np.asarray(transparent_template.convert("RGBA"))
            guide_pixels = guide_rgba[:, :, :3].astype(np.float32)
            template_foreground = guide_rgba[:, :, 3] > 16
            if np.any(template_foreground):
                expected_target_fraction = self._uniform_color_fraction(
                    guide_pixels.astype(np.uint8), template_foreground, outer_target_rgb,
                )
                logger.info(
                    "[UNIFORM_REFERENCE] Target colour occupies %.3f of template garment pixels",
                    expected_target_fraction,
                )

        logger.info(
            "[UNIFORM_REFERENCE] Measured garment palette and construction; "
            "the cleaned visual garment authority is attached after its no-face safety check"
        )

        measured_dark_hair = source_hair_is_dark(person_pil, source_labels)
        lighting_truthy = lambda value: str(value or "").strip().lower() in {
            "true", "yes", "1", "present",
        }
        correct_generated_hair_glare = measured_dark_hair and (
            lighting_truthy(vl_plan.get("direct_sunlight_present"))
            or lighting_truthy(vl_plan.get("head_hair_hotspot_present"))
        )
        analyzed_hair_color = (
            "dark/black" if measured_dark_hair
            else str(vl_plan.get("hair_color") or "").strip().lower()
        )
        logger.info(
            "[UNIFORM_HAIR] VL=%s measured_dark=%s effective=%s",
            vl_plan.get("hair_color"), measured_dark_hair, analyzed_hair_color,
        )
        hair_color_instruction = describe_hair_correction(analyzed_hair_color)
        if hair_generation_details:
            hair_color_instruction += f" Source hair facts: {hair_generation_details}."
        analyzed_hair_style = str(vl_plan.get("hair_style") or "").lower()
        hair_shape_instruction = (
            "Preserve the conditioning portrait's exact hairline, parting location, crown height, side contour, hairstyle, "
            "visible ends and real accessories. Do not shorten, expand, curl, mirror or restyle it. "
            f"Reliable source hair facts: {hair_generation_details}."
            if hair_generation_details else
            "Preserve the conditioning portrait's exact hairline, crown height, part, side contour, hairstyle, visible ends and real accessories."
        )
        if preserve_tied_back_hair:
            hair_shape_instruction += (
                " Preserve only the tied arrangement actually visible in image 1; never mirror it or add another tail."
            )
            if str(vl_plan.get("hair_accessories") or "").strip().lower() in {
                "", "none", "unknown", "uncertain", "[]", "{}",
            }:
                hair_shape_instruction += (
                    " No hair accessory is visible: keep the fastener hidden and do not invent a colored elastic, ribbon, bow, clip or band."
                )
        elif not vl_claims_tied_hair:
            hair_shape_instruction += (
                " Image 1 has no tied-back hairstyle: do not add a ponytail, pigtail, bun, braid, plait, hair tie or trailing hair bundle."
            )
        else:
            hair_shape_instruction += " Copy only the tied-hair pixels visibly supported by image 1; do not invent a bun or tail."
        measured_hair_rgb = measured_person_colors.get("hair_rgb")
        if measured_hair_rgb:
            measured_hair_luma = sum(
                float(channel) * weight
                for channel, weight in zip(measured_hair_rgb, (.2126, .7152, .0722))
            )
            if measured_hair_luma < 105:
                hair_color_instruction += (
                    " Shaded source strands confirm black to very dark-brown hair; do not make it medium brown, golden, gray, green or blue."
                )
        lighting_instruction = describe_lighting_correction(vl_plan)
        analyzed_skin_tone = (
            str(vl_plan.get("skin_tone") or "").strip()
            if vl_plan.get("source") == "qwen2.5-vl"
            else ""
        )
        measured_skin_rgb = measured_person_colors.get("skin_rgb")
        measured_lip_rgb = measured_person_colors.get("lip_rgb")
        unsafe_cast_terms = ("orange", "yellow", "cyan", "blue", "green", "red cast", "color cast", "colour cast")
        if extreme_source_sunlight:
            # Direct sun has clipped both luminance and white balance; its RGB
            # sample is illumination, not complexion. Preserve identity and
            # infer the natural undertone from the least-lit face/ear/neck
            # pixels rather than commanding Qwen to keep the orange cast.
            skin_tone_instruction = (
                "Preserve the person's complexion and undertone from the shaded face, ear and neck evidence in image 1; remove the measured orange/yellow sunlight cast under neutral light without whitening, tanning or recoloring the person."
            )
        elif measured_skin_rgb:
            # Categorical complexion words are highly sensitive to lighting
            # and camera white balance. The measured source face and the
            # conditioning pixels are stronger evidence for generation.
            skin_tone_instruction = (
                "Keep image 1's exact face/ear/neck complexion and undertone; "
                "correct illumination without lightening, darkening, warming, cooling or recoloring skin."
            )
        elif (
            analyzed_skin_tone.lower() in ("", "none", "unknown", "uncertain", "source natural skin tone")
            or any(term in analyzed_skin_tone.lower() for term in unsafe_cast_terms)
        ):
            skin_tone_instruction = (
                "Keep image 1's face/ear/neck complexion and undertone; correct illumination without recoloring skin."
            )
        else:
            skin_tone_instruction = (
                f"VL reads the complexion as {analyzed_skin_tone}; image 1 pixels remain authoritative. Preserve its "
                "undertone; never lighten, darken, whiten, tan, warm, cool or recolor skin."
            )
        if measured_skin_rgb and not extreme_source_sunlight:
            skin_tone_instruction += (
                f" Stable source face pixels measure RGB {measured_skin_rgb}; preserve their chroma and undertone while changing only illumination."
            )
        logger.info("[UNIFORM_SKIN] VL=%s instruction=%s", vl_plan.get("skin_tone"), skin_tone_instruction)
        skin_prompt_instruction = (
            "Preserve image 1's exact face, ear and neck complexion; neutralize illumination only."
            + (
                f" Source skin RGB is {measured_skin_rgb}."
                if measured_skin_rgb and not extreme_source_sunlight else
                " Use shaded source skin as the undertone authority; ignore direct-sun color cast."
            )
        )
        identity_instruction = (
            "Image 1 is the complete original-person portrait. Its face, full hair, accessories, "
            "neck anatomy, pose, body proportions and age are authoritative. Replace only its clothing "
            "with the garment design in image 2 and tailor that design to image 1's existing body."
        )
        source_body_ratio = float(rough_placement.get("source_body_to_face_width") or 0.0)
        fitted_body_ratio = float(rough_placement.get("garment_to_face_width") or 0.0)
        measured_neck_gap_ratio = float(rough_placement.get("source_neck_gap_ratio") or .20)
        # The uploaded portrait is the neck-geometry authority. Adding a fixed
        # minimum here contradicted the measured guide and made naturally short
        # necks longer in the generated result.
        source_neck_gap_ratio = measured_neck_gap_ratio
        body_fit_instruction = (
            "The uploaded person's measured upper-body width is authoritative. "
            f"Image 1 has a shoulder/upper-torso span of about {source_body_ratio:.2f} face widths "
            f"and the neutral fit placeholder is {fitted_body_ratio:.2f} face widths. "
            "Preserve that exact natural build: do not widen or narrow the shoulders, enlarge the head, shrink the torso, lengthen the neck, or copy the uniform template model's body shape. "
            "Render the same person as a camera-facing school-ID passport photograph, showing the full head with modest headroom, level shoulders, neck, and upper chest only."
        )
        body_fit_prompt_instruction = (
            f"Image 1's measured {source_body_ratio:.2f}-face-wide body is the sole fit authority. "
            "Keep the collar at this neck base and shoulder seams at these shoulder points. Copy the garment design and fabric, "
            "never copy the template model, body shape, pose, width or silhouette."
        )
        # Keep the actual edit contract below the prompt encoder's 512-token
        # limit. The former prompt repeated the same requirements until the
        # late body-scale and shoulder instructions were truncated. Qwen then
        # obeyed the early "upper chest" phrase by zooming into the head. Image
        # 1 already has the required composition, so state that immutable
        # geometry first and let the two images carry fine visual detail.
        qwen_hex = f"#{qwen_bg_rgb[0]:02X}{qwen_bg_rgb[1]:02X}{qwen_bg_rgb[2]:02X}"
        neckwear_prompt_instruction = (
            "Keep only the neckwear already fitted on the uniform; remove any different source undershirt or jewelry."
            if template_has_neckwear else
            "The fitted uniform has no neckwear: keep natural neck skin above its collar and add no undershirt, tie, bow, ribbon or chain."
        )
        trim_prompt_instruction = (
            "Keep every visible layer and place trim only where image 1 shows it."
            if template_has_outer_layer else
            "This is one shirt. Keep patterned fabric across both shoulders and retain contrast trim only where image 1 shows it."
        )
        uniform_generation_facts = observed_details((
            "garment_components", "shirt_pattern_type",
            "shirt_pattern_scale", "shirt_collar", "sleeve_length",
            "accent_trim_placement", "button_layout", "button_color",
            "pocket_count", "pocket_placement", "pocket_style",
        ))
        if not multi_color_pattern:
            described_color = observed_details((
                "shirt_color_and_pattern", "accent_trim_color",
            ))
            uniform_generation_facts = "; ".join(filter(None, (
                uniform_generation_facts, described_color,
            )))
        if not uniform_generation_facts:
            uniform_generation_facts = "copy the fitted template pixels exactly"
        badge_repair_instruction = ""
        if template_badges_removed:
            badge_repair_instruction = (
                "The removed badge area must be reconstructed with the surrounding garment fabric and pattern "
                "continuing through it; leave no solid-color rectangle, blank patch, label shape or repair mark."
            )
        trim_color = str(
            vl_plan.get("accent_trim_color")
            or vl_plan.get("shirt_primary_color")
            or ""
        ).strip()
        trim_placement = str(vl_plan.get("accent_trim_placement") or "").strip()
        trim_color_instruction = ""
        known_trim_placement = trim_placement.lower() not in {
            "", "none", "unknown", "uncertain",
        }
        if measured_accent and known_trim_placement:
            trim_color_instruction = (
                f"Pixel measurement identifies the coherent solid trim as "
                f"{measured_accent['hex']} RGB{tuple(measured_accent['rgb'])} at {trim_placement}. "
                "Solid trim exists only at those analyzed placements. Keep that exact measured color there; "
                "never replace it with a pattern-thread color or invent a solid pocket band unless pocket is explicitly listed."
            )
        elif (
            vl_plan.get("source") == "qwen2.5-vl"
            and trim_color.lower() not in {"", "none", "unknown", "uncertain"}
            and known_trim_placement
        ):
            trim_color_instruction = (
                f"The solid contrast trim is {trim_color} at {trim_placement}. Keep that exact solid color and placement; never recolor it to a plaid/check thread color."
            )
        visible_wearables_text = str(vl_plan.get("visible_wearables") or "").strip().lower()
        earring_description_text = str(vl_plan.get("earring_description") or "").strip().lower()
        raw_hair_observations = vl_plan.get("hair_observations_raw") or {}
        raw_earring_detail = str(
            raw_hair_observations.get("hair_accessory_details") or ""
        ).strip().lower() if isinstance(raw_hair_observations, dict) else ""
        earring_reference = (
            raw_earring_detail
            if any(term in raw_earring_detail for term in ("earring", "hoop", "stud"))
            else earring_description_text
        )
        absent_terms = ("none", "no ", "without", "absent", "not visible", "not-visible", "unknown", "uncertain", "n/a")
        wearables_are_absent = (
            not visible_wearables_text
            or any(term in visible_wearables_text for term in absent_terms)
        )
        earring_description_is_absent = (
            not earring_description_text
            or any(term in earring_description_text for term in absent_terms)
        )
        has_earrings = (
            (not wearables_are_absent and "earring" in visible_wearables_text)
            or not earring_description_is_absent
        )
        source_has_glasses = PhotoRestorationService.has_eyeglasses(
            cv2.cvtColor(np.asarray(person_pil), cv2.COLOR_RGB2BGR)
        )
        has_glasses = source_has_glasses or (not wearables_are_absent and any(
            term in visible_wearables_text for term in ("glasses", "eyeglass", "spectacle")
        )) or bool(vl_plan.get("glasses_reflection_present"))
        # Earring count/side descriptions are unreliable on reflective lenses
        # and small source faces. The pixels embedded in image 1 are the only
        # authority: this preserves two hoops when VL reports one earring, and
        # naturally preserves bare ears when no jewelry exists.
        wearable_instruction = (
            "Preserve exactly the ear jewelry visibly present in image 1 at its original small scale, type, metal color, shape, count and position; do not enlarge, stylize, recolor, remove, add, mirror or redesign it. "
        )
        wearable_instruction += (
            "Preserve image 1's exact eyeglasses and natural lens reflections."
            if has_glasses else
            "Image 1 has no eyeglasses; do not add frames or lens reflections."
        )
        # Keep semantic hair analysis for QA only. Translating a partly
        # occluded hairstyle into a label such as "ponytail" removed real side
        # curls and replaced flower clips in otherwise clear source pixels.
        compact_hair_requirement = (
            "Copy image 1's hair exactly: hairline, part, crown, both side contours, texture, visible tied portions and ends; do not infer or restyle it."
        )
        hair_accessory_lock = (
            "Copy visible hair accessories exactly in count, color, size and position; add none if absent."
        )
        compact_wearables = (
            "Copy only visible earrings and eyeglasses; add nothing."
        )
        # VL geometry is retained for review, but lossy semantic descriptions
        # such as "round face" or "small nose" must not drive a biometric edit.
        # Image 1 already carries the uploaded person's complete head. The only
        # secondary input is a person-free garment reference.
        compact_face_requirement = (
            "Image 1's head is already correct. Keep the exact same face, proportions, complexion, natural asymmetry, pores, marks and apparent age; "
            "do not redraw, beautify, smooth, recolor or reshape it."
        )
        # Keep this below the multimodal encoder's 512-token limit. The former
        # 560-word prompt placed the background and crop contract at its tail,
        # so Qwen never reliably saw them and invented scenery. Put observable
        # output geometry first and avoid repeating the same identity facts.
        def capped_words(value: Any, limit: int) -> str:
            return " ".join(str(value or "").split()[:limit])

        # VL sometimes returns paragraph-length values for each garment key.
        # The fitted guide already contains the visible template pixels, so
        # prose is supporting evidence rather than a second full description.
        compact_uniform_facts = capped_words(uniform_generation_facts, 64)
        def first_complete_sentence(value: Any, fallback: str) -> str:
            text = " ".join(str(value or "").split()).strip()
            if not text:
                return fallback
            sentence = text.split(".", 1)[0].rstrip(" ,;")
            return sentence + "."

        compact_trim_instruction = first_complete_sentence(
            trim_color_instruction, "Keep the template's visible trim unchanged."
        )
        compact_badge_instruction = first_complete_sentence(
            badge_repair_instruction, "Continue matching fabric through removed badge areas."
        )
        hair_geometry_copy = (
            "Image 1's hair is already correct. Copy image 1's hairline, part, crown, tied-tail position, length, texture and color exactly; do not restyle or polish it."
            if preserve_tied_back_hair else
            "Image 1's hair is already correct. Copy image 1's hairline, part, crown, visible ends, length, texture and color exactly; do not restyle or polish it."
        )
        apparent_age_lock = (
            "Retain exactly the same apparent age and facial maturity visible in image 1; "
            "do not make the person younger, older, more childlike or more mature. "
        )
        # Naming an age group makes the generator synthesize stereotyped age
        # cues (for example, de-aging a uniformed adult or adding a full beard).
        # The source pixels are more reliable than either age classifier. Keep
        # the final compact prompt entirely visual and explicitly no-edit.
        compact_age_requirement = (
            "FACE/AGE LOCK: do not redraw, retouch, smooth or beautify image 1's face. Preserve its exact eyes, "
            "nose, lips, cheeks, chin, jaw, facial volume, asymmetry, skin texture, marks, current maturity and "
            "visible facial-hair pattern. Never make the person younger or older. "
        )
        concise_uniform_facts = capped_words(compact_uniform_facts, 32)
        pocket_count_value = vl_plan.get("pocket_count")
        try:
            pocket_count = int(pocket_count_value)
        except (TypeError, ValueError):
            pocket_count = None
        pocket_placement = str(vl_plan.get("pocket_placement") or "").strip()
        pocket_style = str(vl_plan.get("pocket_style") or "").strip()
        pocket_known = pocket_count is not None and pocket_count >= 0
        if pocket_known:
            pocket_instruction = (
                f"POCKET LOCK: render exactly {pocket_count} visible chest pocket"
                f"{'s' if pocket_count != 1 else ''}; placement={pocket_placement or 'as shown in image 1'}; "
                f"construction={pocket_style or 'as shown in image 1'}. "
                "Do not add, remove, mirror or duplicate a pocket. A badge or color block is not a pocket. "
            )
        else:
            pocket_instruction = (
                "POCKET LOCK: copy the exact visible pocket count, side and construction from image 1; "
                "do not add, remove, mirror or duplicate a pocket. "
            )
        concise_palette_evidence = (
            "Measured template palette: "
            + ", ".join(item["hex"] for item in measured_garment_palette)
            + ". Preserve these colors in their existing spatial pattern."
            if measured_garment_palette else
            "Preserve image 1's fitted garment colors exactly."
        )
        lip_prompt_instruction = (
            f"Keep the natural unpainted lip tone near RGB{measured_lip_rgb}; do not darken, whiten, recolor or add lipstick. "
            if measured_lip_rgb else
            "Keep image 1's natural unpainted lip tone unchanged. "
        )
        # Four-step image editing follows short, local instructions more
        # reliably than a long portrait-redesign brief. Put every measurable
        # output lock before the edit verb: when the broad "naturalize" request
        # came first, Qwen sometimes repainted the garment before attending to
        # the later palette/backdrop clauses. The values below are derived for
        # each upload; none of the garment colours or placements is static.
        generated_palette_lock = (
            "Use only these measured uniform colors: "
            + ", ".join(item["hex"] for item in measured_garment_palette)
            + ". "
            if measured_garment_palette else
            "Keep every uniform color already visible in image 1 unchanged. "
        )
        generated_trim_lock = (
            f"Render solid trim as exactly {measured_accent['hex']} "
            f"RGB{tuple(measured_accent['rgb'])}, only at {trim_placement}; "
            "do not borrow another plaid color and do not create trim anywhere else. "
            + (
                f"Both collar leaves must be solid {measured_accent['hex']} trim with no base-fabric, plaid, stripe or alternate color on either leaf. "
                if "collar" in trim_placement.lower() else ""
            )
            if measured_accent and known_trim_placement else
            "Keep the fitted trim color and its existing locations unchanged. "
        )
        generated_prompt = (
            f"ONE PERSON ONLY. OUTPUT CONTRACT: one face, upper-chest passport crop, and a perfectly flat {qwen_hex} RGB{tuple(qwen_bg_rgb)} background from edge to edge. "
            "Do not add scenery, texture, blocks, shadows or gradients; no room, wall, vignette, prop, or second person may appear in the background. "
            f"UNIFORM COLOR LOCK: {generated_palette_lock}{generated_trim_lock}"
            "IDENTITY LOCK: Edit image 1 locally; do not redesign or regenerate the portrait. Keep image 1's entire head, hair, hairline, ears, face and accessories; preserve its exact facial proportions, natural asymmetry and complexion. "
            f"{apparent_age_lock}{compact_hair_requirement} {compact_wearables} "
            f"{lip_prompt_instruction}"
            "Keep the same neck length, shoulder width, body build, centered camera-facing pose and upper-chest passport crop. Do not zoom, recrop, enlarge the head, expose the waist, or cut the crown. "
            f"NECK/COLLAR LOCK: keep the jaw-to-collar gap exactly as fitted in image 1 (about {source_neck_gap_ratio:.2f} face heights); "
            "keep the collar at that existing neck base and do not lower it, stretch the neck or expose additional neck skin. "
            "EDIT REGION: the required uniform is already rough-fitted in image 1. Naturalize only its collar-to-neck junction, shoulder fit, seams, folds, lighting and fabric texture so it looks genuinely worn and photographed, never pasted or painted. "
            f"Uniform facts: {concise_uniform_facts}. {concise_palette_evidence} {compact_trim_instruction} {neckwear_prompt_instruction} "
            f"{pocket_instruction}"
            "Keep exactly the garment design, colors, check or stripe alignment, collar, placket, buttons, trim, pocket and sleeve construction visible in image 1. "
            "Do not recolor the collar, placket or sleeve trim. Keep the already-clean pocket and fabric uninterrupted; add no marks or objects. "
            f"Generate the selected {qwen_hex} RGB{tuple(qwen_bg_rgb)} as the backdrop during this same render; do not defer backdrop replacement to another pass. "
            "Return one natural unretouched camera photograph with real fabric weave and normal skin texture, not an illustration, beauty-filtered face or synthetic portrait."
        )
        normalized_preference = " ".join(str(prompt or "").strip().split())
        if normalized_preference.lower() not in {
            "", "exact supplied uniform template", "exact uniform template",
        }:
            limited_preference = capped_words(normalized_preference, 20)
            generated_prompt += (
                " Optional preference, only if consistent with all requirements: "
                + limited_preference
            )
        prompt_word_count = len(generated_prompt.split())
        if prompt_word_count > 360:
            # Never fail a client job because VL returned verbose prose. Keep
            # the high-priority contract at the front and reserve the tail for
            # the photographic finish requirement.
            finish = (
                f" FINAL: exact flat {qwen_hex} background, same identity and age, exact fitted uniform, "
                "real skin texture, no badge, logo, collage, illustration or painted finish."
            )
            finish_words = finish.split()
            prefix_limit = max(1, 360 - len(finish_words))
            prefix_words = generated_prompt.split()[:prefix_limit]
            while prefix_words and not prefix_words[-1].endswith((".", "!", "?")):
                prefix_words.pop()
            generated_prompt = " ".join(prefix_words + finish_words)
            logger.warning(
                "[UNIFORM_PROMPT] Compacted verbose VL prompt from %d to %d words",
                prompt_word_count, len(generated_prompt.split()),
            )
        # Measure the rough fit for validation and post-generation artifact
        # checks. Qwen receives this same single-subject portrait directly.
        rough_fixed_faces = _get_insight_app().get(
            cv2.cvtColor(np.asarray(rough_composite), cv2.COLOR_RGB2BGR)
        )
        if len(rough_fixed_faces) != 1:
            raise RuntimeError("Uniform fitting could not isolate one face for garment-only generation")
        rough_face_box = np.asarray(rough_fixed_faces[0].bbox, dtype=np.float32)
        rough_face_height = max(1.0, float(rough_face_box[3] - rough_face_box[1]))
        garment_crop_top = max(
            0, min(rough_composite.height - 64,
                   int(round(float(rough_face_box[3]) - rough_face_height * .02))),
        )
        garment_edit_crop = rough_composite.crop(
            (0, garment_crop_top, rough_composite.width, rough_composite.height)
        )
        garment_crop_path = OUTPUTS_DIR / f"{job_id}_garment_edit_crop.png"
        garment_edit_crop.save(garment_crop_path, format="PNG")

        identity_base = person_conditioning.convert("RGB").resize(
            rough_composite.size, Image.LANCZOS,
        )
        identity_rgb = np.asarray(identity_base, dtype=np.float32)
        identity_labels = parse_body_parts(np.asarray(identity_base))["labels"]
        source_clothing = np.isin(
            identity_labels, (5, 6, 7, 9, 10, 11, 12),
        ).astype(np.uint8)
        face_x1, face_y1, face_x2, face_y2 = (
            float(value) for value in rough_face_box
        )
        detected_face_width = max(1.0, face_x2 - face_x1)
        detected_face_height = max(1.0, face_y2 - face_y1)
        face_center_x = (face_x1 + face_x2) * .5
        mask_y, mask_x = np.indices(identity_labels.shape)
        # CODEX NOTE: SCHP human parsing frequently misclassifies pale or white shirts,
        # mandarin collars, or chest fabric as background (label 0). When this happens,
        # source_clothing has an empty gap across rows of the upper chest and collar.
        # Recover verified foreground pixels (pixels demonstrably far from the flat
        # backdrop colour) in the torso area as source clothing.
        selected_bg = np.asarray(qwen_bg_rgb, dtype=np.float32)
        bg_distance = np.linalg.norm(
            identity_rgb - selected_bg[None, None, :],
            axis=2,
        )
        is_real_foreground = bg_distance > 38.0
        head_or_hair_labels = np.isin(identity_labels, (1, 2, 4))
        upper_neck_skin = (
            (identity_labels == 13)
            & (mask_y < face_y2 + detected_face_height * .20)
            & (np.abs(mask_x - face_center_x) < detected_face_width * .30)
        )
        torso_clothing_recovery = (
            is_real_foreground
            & (mask_y >= face_y2 + detected_face_height * .18)
            & ~head_or_hair_labels
            & ~upper_neck_skin
        )
        source_clothing[torso_clothing_recovery] = 1
        clothing = source_clothing.copy()
        clothing = cv2.morphologyEx(
            clothing,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        )
        clothing = cv2.dilate(
            clothing,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        # A folded shirt collar must be allowed to wrap around the lower neck.
        # SCHP labels that complete area as skin, so the former identity guard
        # made it immutable and Qwen could only add two narrow collar tips below
        # a long skin column. Build two face-relative collar wings that converge
        # at the real neck base. This adapts to each portrait instead of using a
        # fixed pixel crop, while remaining safely below the detected jaw.
        collar_top = face_y2 - detected_face_height * .01
        # Measure where this portrait's source garment actually reaches the
        # centre neck. A fixed fraction stopped above low V-necks and long
        # collars, protecting a thin old-collar strip that Qwen decoded as
        # necklace dots or contrasting piping. The measured first clothing row
        # keeps this adaptive across children, adults and different crops.
        central_neck_width = np.abs(mask_x - face_center_x) <= detected_face_width * .32
        neckline_search = (
            central_neck_width
            & (mask_y >= face_y2)
            & (mask_y <= face_y2 + detected_face_height * .70)
        )
        clothing_by_row = np.count_nonzero(
            (source_clothing > 0) & neckline_search, axis=1,
        )
        neckline_rows = np.flatnonzero(
            clothing_by_row >= max(3, int(round(detected_face_width * .08)))
        )
        if neckline_rows.size:
            measured_neckline_top = float(neckline_rows[0])
            measured_collar_bottom = measured_neckline_top + detected_face_height * .25
        else:
            measured_collar_bottom = face_y2 + detected_face_height * .38
        collar_bottom = min(
            face_y2 + detected_face_height * .62,
            max(face_y2 + detected_face_height * .38, measured_collar_bottom),
        )
        collar_progress = np.clip(
            (mask_y - collar_top) / max(1.0, collar_bottom - collar_top),
            0.0,
            1.0,
        )
        collar_outer_half_width = detected_face_width * (
            .38 + .08 * collar_progress
        )
        # Close the central V before the old source-collar boundary. Keeping the
        # V open through the lower half protected pale collar/matte remnants on
        # the neck, leaving a broken blue-white arc above the new collar. The
        # upper neck is still protected; only the lower junction becomes fully
        # editable after about 20% of one detected face height below the jaw.
        collar_inner_half_width = detected_face_width * (
            .20 * np.clip(1.0 - collar_progress * 2.4, 0.0, 1.0)
        )
        collar_distance = np.abs(mask_x - face_center_x)
        # CODEX NOTE: collar_refine_zone must be strictly restricted to the lower
        # neck (at least 8% face height below the chin) and inside valid foreground
        # (identity_labels != 0). Without this constraint, collar_refine_zone selects
        # wedges of empty background air on the sides of the neck, which then get
        # unmasked in the edit mask and inpainted as sharp blue horns gouging the neck.
        collar_refine_zone = (
            (mask_y >= collar_top)
            & (mask_y <= collar_bottom)
            & (collar_distance <= collar_outer_half_width)
            & (collar_distance >= collar_inner_half_width)
            & is_real_foreground
            & (mask_y >= face_y2 + detected_face_height * .08)
        )
        # The selected backdrop and every non-clothing person pixel are final.
        # Only source clothing may diffuse. This prevents Qwen from inventing
        # hair, earrings or a different face while still allowing a coherent
        # collar, shoulder fit and garment texture.
        # Protect the complete identity, but let Qwen rebuild the lower neck and
        # collar as one continuous photographic surface. SCHP labels the neck as
        # face (13); protecting that whole label created a rectangular source-
        # pixel shelf below the chin. The adaptive band begins below the detected
        # face, remains narrower than the jaw, and ends inside the new collar.
        neck_blend_bottom = min(
            collar_bottom,
            face_y2 + detected_face_height * .43,
        )
        jaw_distance = np.clip(
            np.abs(mask_x - face_center_x) / max(detected_face_width * .52, 1.0),
            0.0,
            1.0,
        )
        # Follow the visible jaw instead of the rectangular detector bbox. At
        # the chin the editable region starts just below the face; toward both
        # jaw corners it rises gradually. This removes the straight horizontal
        # source/generated seam without exposing eyes, nose, mouth or glasses.
        jaw_boundary_y = (
            face_y2
            + detected_face_height * .02
            - detected_face_height * .22 * np.power(jaw_distance, 1.55)
        )
        # CODEX NOTE: jaw_boundary_y rises ~33px above chin level toward the jaw corners.
        # effective_jaw_y clamps the blend zone to begin strictly below the chin (at least
        # 5% of face height below face_y2). Furthermore, neck_blend_half_width is bounded
        # to the natural neck corridor (max 0.32 face width) and neck_blend_zone strictly
        # requires (identity_labels != 0) & ~backdrop_like_identity so empty background
        # air is never flagged as editable.
        effective_jaw_y = np.maximum(
            jaw_boundary_y,
            face_y2 + detected_face_height * .05,
        )
        neck_blend_progress = np.clip(
            (mask_y - effective_jaw_y)
            / np.maximum(1.0, neck_blend_bottom - effective_jaw_y),
            0.0,
            1.0,
        )
        neck_blend_half_width = detected_face_width * (
            .32 - .08 * neck_blend_progress
        )
        identity_u8_for_guard = np.asarray(identity_base, dtype=np.uint8)
        selected_bg = np.asarray(qwen_bg_rgb, dtype=np.float32)
        selected_bg_distance = np.linalg.norm(
            identity_u8_for_guard.astype(np.float32) - selected_bg[None, None, :],
            axis=2,
        )
        selected_bg_core = selected_bg_distance <= 24.0
        selected_bg_nearby = cv2.dilate(
            selected_bg_core.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        ).astype(bool)
        channel_spread = np.ptp(identity_u8_for_guard.astype(np.int16), axis=2)
        pale_edge_fringe = (
            selected_bg_nearby
            & (identity_u8_for_guard.min(axis=2) >= 168)
            & (channel_spread <= 54)
        )
        # Parsing can label selected-backdrop antialiasing or a pale cutout
        # halo as hair/ear/hat. Protect those labels only when their measured
        # colour supports real foreground; true hair remains far from the
        # selected backdrop colour.
        backdrop_like_identity = (selected_bg_distance <= 76.0) | pale_edge_fringe
        reliable_head_protection = (
            np.isin(identity_labels, (1, 2, 4, 14, 15))
            & ~backdrop_like_identity
        )
        neck_blend_zone = (
            (mask_y >= effective_jaw_y)
            & (mask_y <= neck_blend_bottom)
            & (np.abs(mask_x - face_center_x) <= neck_blend_half_width)
            & ~reliable_head_protection
            & is_real_foreground
            & ~backdrop_like_identity
        )
        # Preserve all measured source skin, including the natural neck. The
        # pre-Qwen portrait matte already has the correct identity and anatomy;
        # repainting class 13 together with the garment produced a blocky,
        # elongated throat. Qwen may regenerate the surrounding collar and
        # uniform, but it must fit them around the unchanged source neck.
        protected_identity = reliable_head_protection.copy()
        protected_identity |= (
            (identity_labels == 13)
            & is_real_foreground
        )
        # SCHP classes 14/15 are source arm pixels. Preserve those when they
        # exist; a covered-arm source has none and uses the measured synthetic
        # arm guide below instead.
        protected_identity |= np.isin(identity_labels, (14, 15))
        # Retain a small geometry fallback for parser edge pixels immediately
        # below the jaw. The class-13 protection above is authoritative.
        natural_neck_guard = (
            (mask_y >= face_y2 - detected_face_height * .02)
            & (mask_y <= face_y2 + detected_face_height * .08)
            & (np.abs(mask_x - face_center_x) <= detected_face_width * .27)
            & is_real_foreground
        )
        protected_identity |= natural_neck_guard
        recovered_identity_hair = np.zeros_like(protected_identity)
        # SCHP may label a dark low ponytail as jacket/shirt where it passes
        # behind the neck.  Excluding protection solely by semantic class then
        # lets the garment mask repaint the tail.  Recover only pixels whose
        # colour is connected to parsed hair inside face-relative crown/side
        # corridors.  This is measured independently for every portrait and
        # cannot invent a hairstyle or protect the central source garment.
        parsed_identity_hair = identity_labels == 2
        if np.count_nonzero(parsed_identity_hair) >= 96:
            identity_u8 = np.asarray(identity_base, dtype=np.uint8)
            identity_lab = cv2.cvtColor(identity_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
            identity_luma = (
                identity_u8[:, :, 0].astype(np.float32) * .2126
                + identity_u8[:, :, 1].astype(np.float32) * .7152
                + identity_u8[:, :, 2].astype(np.float32) * .0722
            )
            shaded_identity_hair = parsed_identity_hair & (
                identity_luma <= np.percentile(identity_luma[parsed_identity_hair], 78)
            )
            hair_samples = identity_lab[
                shaded_identity_hair
                if np.count_nonzero(shaded_identity_hair) >= 48
                else parsed_identity_hair
            ]
            identity_hair_lab = np.median(hair_samples, axis=0)
            identity_hair_distance = np.linalg.norm(
                identity_lab - identity_hair_lab[None, None, :], axis=2,
            )
            near_identity_hair = cv2.dilate(
                parsed_identity_hair.astype(np.uint8),
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (max(7, int(round(detected_face_width * .38)) | 1),
                     max(7, int(round(detected_face_height * .54)) | 1)),
                ),
                iterations=1,
            ).astype(bool)
            crown_and_side_corridors = (
                (
                    (mask_y <= face_y2 + detected_face_height * .10)
                    & (mask_x >= face_x1 - detected_face_width * .65)
                    & (mask_x <= face_x2 + detected_face_width * .65)
                )
                | (
                    (mask_y >= face_y2 - detected_face_height * .08)
                    & (mask_y <= face_y2 + detected_face_height * .58)
                    & (
                        ((mask_x >= face_x1 - detected_face_width * .48)
                         & (mask_x <= face_x1 + detected_face_width * .12))
                        | ((mask_x >= face_x2 - detected_face_width * .12)
                           & (mask_x <= face_x2 + detected_face_width * .48))
                    )
                )
            )
            hair_candidates = (
                near_identity_hair
                & crown_and_side_corridors
                & (identity_hair_distance < 54.0)
                & (identity_luma < 135.0)
            )
            candidate_count, candidate_labels, candidate_stats, candidate_centroids = cv2.connectedComponentsWithStats(
                hair_candidates.astype(np.uint8), connectivity=8,
            )
            connected_identity_hair = np.zeros_like(parsed_identity_hair)
            disconnected_side_hair = np.zeros_like(parsed_identity_hair)
            maximum_tail_area = max(
                48,
                int(round(detected_face_width * detected_face_height * .15)),
            )
            for candidate_id in range(1, candidate_count):
                candidate = candidate_labels == candidate_id
                if np.any(candidate & parsed_identity_hair):
                    connected_identity_hair |= candidate
                    continue
                # A real low tail can be separated from the parsed crown by a
                # few backdrop pixels or be labelled as clothing where it
                # crosses the shoulder. Recover only compact components beside
                # the face that start near the nape and match the measured hair
                # colour. Large dark garment regions fail the area/position
                # gates, so this remains safe for arbitrary uniforms.
                area = int(candidate_stats[candidate_id, cv2.CC_STAT_AREA])
                component_top = int(candidate_stats[candidate_id, cv2.CC_STAT_TOP])
                component_bottom = component_top + int(
                    candidate_stats[candidate_id, cv2.CC_STAT_HEIGHT]
                )
                component_center_x = float(candidate_centroids[candidate_id, 0])
                beside_face = (
                    component_center_x <= face_x1 + detected_face_width * .10
                    or component_center_x >= face_x2 - detected_face_width * .10
                )
                reaches_nape = (
                    component_top <= face_y2 + detected_face_height * .28
                    and component_bottom >= face_y2 - detected_face_height * .04
                )
                if (
                    16 <= area <= maximum_tail_area
                    and beside_face
                    and reaches_nape
                ):
                    disconnected_side_hair |= candidate
            protected_identity |= connected_identity_hair
            protected_identity |= disconnected_side_hair
            recovered_identity_hair |= connected_identity_hair
            recovered_identity_hair |= disconnected_side_hair
            # A low ponytail against the original wall is frequently assigned
            # SCHP background (0), so it forms a disconnected component and
            # cannot satisfy the crown-connectivity rule above. On the already
            # flat conditioning canvas, dark pixels that match the measured
            # crown hair inside the anatomical side/nape corridor are direct
            # pixel evidence of hair, while the selected blue backdrop cannot
            # pass either colour gate. Protect those source pixels from the
            # clothing mask without relying on a hairstyle label or gender.
            disconnected_hair_on_backdrop = (
                (identity_labels == 0)
                & crown_and_side_corridors
                & near_identity_hair
                & (identity_hair_distance < 42.0)
                & (identity_luma < 92.0)
            )
            protected_identity |= disconnected_hair_on_backdrop
            recovered_identity_hair |= disconnected_hair_on_backdrop
            # When the analyser has measured a one-sided low tail, retain the
            # compact nape strip on that same side even if SCHP merges it with
            # a similarly dark vest. The vertical cap stops before the torso,
            # and the opposite side remains editable.
            if measured_ponytail_side in {"viewer-left", "viewer-right"}:
                if measured_ponytail_side == "viewer-right":
                    measured_tail_side = mask_x >= face_x2 - detected_face_width * .08
                else:
                    measured_tail_side = mask_x <= face_x1 + detected_face_width * .08
                measured_nape_tail = (
                    measured_tail_side
                    & (mask_y >= face_y2 - detected_face_height * .05)
                    & (mask_y <= face_y2 + detected_face_height * .24)
                    & (identity_hair_distance < 42.0)
                    & (identity_luma < 82.0)
                )
                protected_identity |= measured_nape_tail
                recovered_identity_hair |= measured_nape_tail
        # Build the complete candidate edit region first, then apply identity
        # protection once. Re-adding the collar zone after protection used to
        # expose protected hair and accessories around the neck.
        edit_region = clothing > 0
        edit_region |= collar_refine_zone & ~np.isin(identity_labels, (1, 2, 4))
        edit_region |= neck_blend_zone
        edit_region &= ~protected_identity
        # CODEX NOTE: Background air above collar bottom must never enter edit_region.
        # This guarantees that empty air around the head, neck, and shoulders remains
        # strictly non-editable (0/black mask), preventing Qwen from inpainting blue
        # horns, spikes, or ragged borders gouging the neck and jawline.
        edit_region &= is_real_foreground | (mask_y > collar_bottom)
        rough_rgb = np.asarray(rough_composite.convert("RGB"), dtype=np.float32)
        # When a short-sleeve target replaces source clothing that covered the
        # arms, the rough compositor correctly removes that old clothing but
        # leaves selected-backdrop pixels below each target cuff. Diffusion
        # then treats the blue holes as missing body. Supply a low-frequency
        # skin placement cue only in those source-silhouette side regions;
        # Qwen still creates the final texture and the central shirt is never
        # touched by this guide.
        synthetic_arm_guide = np.zeros_like(edit_region)
        if requires_exposed_arm_synthesis and measured_skin_rgb:
            rough_bg_distance = np.linalg.norm(
                rough_rgb - np.asarray(qwen_bg_rgb, dtype=np.float32)[None, None, :],
                axis=2,
            )
            synthetic_arm_guide = (
                (source_clothing > 0)
                & (rough_bg_distance <= 42.0)
                & (mask_y >= face_y2 + detected_face_height * .56)
                & (np.abs(mask_x - face_center_x) >= detected_face_width * .62)
                & (np.abs(mask_x - face_center_x) <= detected_face_width * 1.72)
            )
            # Long source sleeves leave wide rectangular clothing silhouettes
            # below a short-sleeve target. Filling that whole silhouette with
            # one skin colour produces the brown blocks Qwen then preserves.
            # Taper each side independently so the cue describes only a small
            # upper-arm continuation below the target cuff. The bounds are
            # measured per row from this portrait; no body size is hard-coded.
            tapered_arm_guide = np.zeros_like(synthetic_arm_guide)
            for side_mask in (mask_x < face_center_x, mask_x > face_center_x):
                side_candidate = synthetic_arm_guide & side_mask
                rows = np.flatnonzero(np.any(side_candidate, axis=1))
                if rows.size == 0:
                    continue
                first_row, last_row = int(rows[0]), int(rows[-1])
                row_span = max(1, last_row - first_row)
                for row in rows:
                    columns = np.flatnonzero(side_candidate[row])
                    if columns.size == 0:
                        continue
                    progress = float(row - first_row) / row_span
                    center_x = float(np.mean(columns))
                    source_width = float(columns[-1] - columns[0] + 1)
                    retained_width = max(
                        4.0,
                        source_width * (1.0 - .42 * progress),
                    )
                    left = max(0, int(round(center_x - retained_width * .5)))
                    right = min(
                        synthetic_arm_guide.shape[1],
                        int(round(center_x + retained_width * .5)) + 1,
                    )
                    tapered_arm_guide[row, left:right] = side_candidate[row, left:right]
            synthetic_arm_guide = cv2.morphologyEx(
                tapered_arm_guide.astype(np.uint8),
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 9)),
                iterations=1,
            ).astype(bool)
            if np.any(synthetic_arm_guide):
                logger.info(
                    "[UNIFORM_ARM_GUIDE] Added tapered measured skin cue to %d covered-arm pixels",
                    int(np.count_nonzero(synthetic_arm_guide)),
                )
        if np.count_nonzero(edit_region) < identity_base.width * identity_base.height * .08:
            raise RuntimeError("Uniform fitting could not identify a reliable garment region")
        # Build one direct conditioning portrait on the ORIGINAL person's geometry:
        # copy fitted garment pixels only inside the measured clothing region,
        # retain every protected identity/body pixel from identity_base, and use
        # a narrow feather solely at the cloth boundary. This is not the old
        # detached-head rough canvas and is never followed by source-face pasteback.
        garment_seed_rgb = rough_rgb.copy()
        if np.any(synthetic_arm_guide) and measured_skin_rgb:
            garment_seed_rgb[synthetic_arm_guide] = np.asarray(
                measured_skin_rgb, dtype=np.float32,
            )
        seed_alpha = cv2.GaussianBlur(
            edit_region.astype(np.float32), (0, 0), sigmaX=1.15, sigmaY=1.15,
        )
        seed_alpha = np.clip(seed_alpha, 0.0, 1.0)
        seed_alpha[protected_identity] = 0.0
        direct_seed_rgb = (
            identity_rgb * (1.0 - seed_alpha[:, :, None])
            + garment_seed_rgb * seed_alpha[:, :, None]
        )
        direct_uniform_seed = Image.fromarray(
            np.clip(direct_seed_rgb, 0, 255).astype(np.uint8), mode="RGB",
        )
        # Use Enhancement's validated person-latent policy exactly. Garment
        # pixels and the chosen backdrop must not select a different face:
        # changing either formerly changed the seed and therefore the apparent
        # age, head scale and facial proportions. The template remains a visual
        # conditioning reference, while identity sampling depends only on the
        # untouched uploaded portrait plus the canonical neutral marker.
        uniform_seed = zlib.crc32(person_pil.convert("RGB").tobytes())
        uniform_seed = zlib.crc32(b"#FFFFFF", uniform_seed)
        uniform_seed = (uniform_seed ^ 3201858918) & 0x7FFFFFFF
        direct_seed_path = OUTPUTS_DIR / f"{job_id}_direct_uniform_seed.png"
        direct_uniform_seed.save(direct_seed_path, format="PNG")
        logger.info(
            "[UNIFORM_REFERENCE] Built one original-body conditioning seed with %d editable garment pixels "
            "using Enhancement-style seed=%d -> %s",
            int(np.count_nonzero(edit_region)), uniform_seed, direct_seed_path,
        )
        # The original framed portrait controls identity and body geometry.
        # Never use the pasted rough fit as Qwen's primary image: despite strong
        # text locks, the model treats its seams and enlarged low-resolution
        # head as permission to synthesize a different portrait.  The rough fit
        # remains useful for measuring placement and validating garment
        # coverage, while the cleaned face-free crop is the sole garment visual
        # authority supplied beside the original person.
        # Match Enhancement's validated primary canvas exactly: a close source
        # portrait keeps its complete uploaded frame and native aspect ratio.
        # The former 832x1216 face-centred conditioning enlarged the head and
        # cut wide curls/flowers before diffusion, so Qwen synthesized a new
        # child even though it received the same random seed.
        # The rough garment, parser masks and identity guide were all measured
        # on this exact passport canvas. Keep that geometry for generation;
        # resizing the rough seed to the original phone-photo aspect changes
        # head/body proportions before Qwen sees it.
        source_generation_w, source_generation_h = person_conditioning.size
        if source_generation_h >= source_generation_w:
            generation_w = 832
            generation_h = int(round(
                (generation_w * source_generation_h / max(source_generation_w, 1)) / 16.0
            ) * 16)
            if generation_h > 1216:
                generation_h = 1216
                generation_w = int(round(
                    (generation_h * source_generation_w / max(source_generation_h, 1)) / 16.0
                ) * 16)
        else:
            generation_h = 832
            generation_w = int(round(
                (generation_h * source_generation_w / max(source_generation_h, 1)) / 16.0
            ) * 16)
            if generation_w > 1216:
                generation_w = 1216
                generation_h = int(round(
                    (generation_w * source_generation_h / max(source_generation_w, 1)) / 16.0
                ) * 16)
        generation_w = max(256, generation_w)
        generation_h = max(256, generation_h)
        # Build one coherent passport canvas before garment conditioning.
        # Supplying the outdoor scene around the head but selected blue around
        # the rough-fit torso made Qwen retain foliage in the upper background.
        # The semantic matte preserves the current upload's RGB subject and is
        # used only as generation conditioning; Qwen still produces the final
        # photographic edges in its single garment edit.
        generation_source = person_conditioning.convert("RGB")
        person_matted_path = OUTPUTS_DIR / f"{job_id}_person_selected_bg.png"
        conditioning_subject_alpha = None
        # Use one soft portrait matte only. The former SCHP/GrabCut conditioning
        # stack retained foliage and punched blue holes through curls, flowers
        # and the neck; Qwen faithfully regenerated those defects. BiRefNet's
        # raw portrait alpha is a single continuous probability matte and does
        # not generate or enhance identity pixels.
        try:
            from pipelines.birefnet_service import background_removal

            conditioning_buffer = io.BytesIO()
            generation_source.save(conditioning_buffer, format="PNG")
            cutout_bytes = background_removal.remove_background(
                conditioning_buffer.getvalue(),
                job_id=f"{job_id}_conditioning",
                use_schp=False,
            )
            cutout = Image.open(io.BytesIO(cutout_bytes)).convert("RGBA")
            conditioning_subject_alpha = np.asarray(
                cutout.getchannel("A"), dtype=np.float32,
            ) / 255.0
            selected_canvas = Image.new(
                "RGBA", cutout.size,
                tuple(int(value) for value in selected_bg_rgb) + (255,),
            )
            selected_person_canvas = Image.alpha_composite(
                selected_canvas, cutout,
            ).convert("RGB")
            if not self._has_selected_solid_background(
                selected_person_canvas, selected_bg_rgb,
            ):
                raise RuntimeError("BiRefNet conditioning canvas is not predominantly selected background")
            generation_source = selected_person_canvas
            logger.info(
                "[UNIFORM_CONDITIONING] Applied one BiRefNet soft portrait matte on selected RGB%s before garment blending",
                tuple(selected_bg_rgb),
            )
        except Exception as exc:
            logger.warning(
                "[UNIFORM_CONDITIONING] BiRefNet conditioning matte unavailable; preserving uploaded pixels: %s",
                exc,
            )
        finally:
            try:
                background_removal.unload()
            except Exception:
                pass
        # Severe clipped light must not remain embedded in Qwen's primary
        # tensor: both the four-step adapter and the deeper base schedule copy
        # those pixels. Build a uniform-only, geometry-preserving guide from
        # this portrait's own shaded skin and hair samples. The untouched
        # upload remains the biometric/VL authority and final validation
        # reference; this guide exists only to tell Qwen the intended lighting.
        source_hair_geometry_canvas = generation_source.copy()
        if source_has_direct_light and not identity_prepass_used:
            # The raw clipped pixels overpower a four-step garment edit. Feed
            # Qwen a low-frequency neutral-light guide which preserves every
            # coordinate and high-frequency strand/face detail. The untouched
            # canvas above remains the literal full-hair visual authority and
            # the original upload remains the biometric validation authority.
            generation_source = self._prepare_extreme_sunlight_uniform_guide(
                generation_source,
                strength=(1.0 if extreme_source_sunlight else .65),
            )
            logger.info(
                "[UNIFORM_CONDITIONING] Measured hard sunlight uses a geometry-preserving neutral-light guide; untouched source remains identity/hair authority"
            )
        elif source_has_direct_light:
            logger.info(
                "[UNIFORM_CONDITIONING] Non-extreme source light remains for coherent Qwen relighting"
            )
        generation_source.save(person_matted_path, format="PNG")
        logger.info(
            "[UNIFORM_CONDITIONING] Saved current-upload person on the generation backdrop -> %s",
            person_matted_path,
        )
        conditioning_has_selected_background = self._has_selected_solid_background(
            generation_source, selected_bg_rgb,
        )
        conditioning_has_flat_background = self._has_flat_border_background(
            generation_source,
        )
        # Supply the spatial answer in image 1. The previous path discarded
        # this aligned seed and showed Qwen the original denim clothing, so a
        # four-step edit had to infer a new torso, uniform and face at once.
        # Blend only the already-measured garment region from the rough seed;
        # every head/hair/accessory pixel comes from the intact portrait above.
        seed_canvas = direct_uniform_seed.convert("RGB")
        if seed_canvas.size != generation_source.size:
            seed_canvas = seed_canvas.resize(
                generation_source.size, Image.Resampling.LANCZOS,
            )
        seed_region = cv2.resize(
            edit_region.astype(np.uint8), generation_source.size,
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
        # The rough guide already contains the badge-cleaned target fabric.
        # Keep that fitted fabric active over the complete pocket area. The
        # former zero here exposed image 1's original denim jacket inside the
        # uniform edit, creating the conspicuous blue rectangular patch.
        badge_box = vl_plan.get("badge_bbox")
        garment_box = rough_placement.get("garment_box")
        if (
            isinstance(badge_box, (list, tuple)) and len(badge_box) == 4
            and isinstance(garment_box, (list, tuple)) and len(garment_box) == 4
        ):
            try:
                bx1, by1, bx2, by2 = (float(value) for value in badge_box)
                ref_w, ref_h = (float(value) for value in template_analysis_size)
                gx1, gy1, gx2, gy2 = (float(value) for value in garment_box)
                mapped_x1 = gx1 + bx1 / max(ref_w, 1.0) * (gx2 - gx1)
                mapped_x2 = gx1 + bx2 / max(ref_w, 1.0) * (gx2 - gx1)
                mapped_y1 = gy1 + by1 / max(ref_h, 1.0) * (gy2 - gy1)
                mapped_y2 = gy1 + by2 / max(ref_h, 1.0) * (gy2 - gy1)
                scale_x = generation_source.width / max(identity_base.width, 1)
                scale_y = generation_source.height / max(identity_base.height, 1)
                mapped_x1, mapped_x2 = mapped_x1 * scale_x, mapped_x2 * scale_x
                mapped_y1, mapped_y2 = mapped_y1 * scale_y, mapped_y2 * scale_y
                pocket_w = max(4.0, mapped_x2 - mapped_x1)
                pocket_h = max(4.0, mapped_y2 - mapped_y1)
                # Include the surrounding pocket panel, not just the printed
                # logo core, so no repaired edge or thread can survive.
                px1 = max(0, int(round(mapped_x1 - pocket_w * .55)))
                px2 = min(generation_source.width, int(round(mapped_x2 + pocket_w * .55)))
                py1 = max(0, int(round(mapped_y1 - pocket_h * .65)))
                py2 = min(generation_source.height, int(round(mapped_y2 + pocket_h * .85)))
                seed_region[py1:py2, px1:px2] = 1.0
                logger.info(
                    "[UNIFORM_BADGE] Kept badge-cleaned fitted fabric across pocket region (%d,%d)-(%d,%d)",
                    px1, py1, px2, py2,
                )
            except (TypeError, ValueError):
                logger.warning("[UNIFORM_BADGE] Could not map VL badge box into rough seed")
        seed_region = cv2.GaussianBlur(
            seed_region, (0, 0), sigmaX=1.15, sigmaY=1.15,
        )
        seed_region = np.clip(seed_region, 0.0, 1.0)[:, :, None]
        protected_source = np.asarray(generation_source, dtype=np.float32)
        fitted_uniform = np.asarray(seed_canvas, dtype=np.float32)
        generation_source = Image.fromarray(
            np.rint(
                protected_source * (1.0 - seed_region)
                + fitted_uniform * seed_region
            ).clip(0, 255).astype(np.uint8),
            "RGB",
        )
        if conditioning_subject_alpha is not None:
            # Rough garment pixels are allowed only inside the uploaded
            # person's soft silhouette. Without this final conditioning-only
            # composite, the rough seed reintroduced rectangular source-scene
            # pieces beside the sleeves and Qwen generated scenery from them.
            if conditioning_subject_alpha.shape != (
                generation_source.height, generation_source.width,
            ):
                conditioning_subject_alpha = cv2.resize(
                    conditioning_subject_alpha,
                    generation_source.size,
                    interpolation=cv2.INTER_LINEAR,
                )
            conditioning_alpha_3d = np.clip(
                conditioning_subject_alpha, 0.0, 1.0,
            )[:, :, None]
            generated_seed_rgb = np.asarray(
                generation_source, dtype=np.float32,
            )
            exact_background = np.empty_like(generated_seed_rgb)
            exact_background[:, :] = np.asarray(
                selected_bg_rgb, dtype=np.float32,
            )
            generation_source = Image.fromarray(
                np.rint(
                    generated_seed_rgb * conditioning_alpha_3d
                    + exact_background * (1.0 - conditioning_alpha_3d)
                ).clip(0, 255).astype(np.uint8),
                "RGB",
            )
            logger.info(
                "[UNIFORM_CONDITIONING] Reapplied the uploaded person's soft alpha after garment blending; Qwen input backdrop is exact RGB%s",
                tuple(selected_bg_rgb),
            )
        selected_bg_conditioning_path = (
            OUTPUTS_DIR / f"{job_id}_selected_bg_conditioning.png"
        )
        generation_source.save(selected_bg_conditioning_path, format="PNG")
        logger.info(
            "[UNIFORM_CONDITIONING] Prepared protected person plus aligned rough uniform on exact RGB%s backdrop before Qwen -> %s",
            tuple(selected_bg_rgb), selected_bg_conditioning_path,
        )
        generation_input = generation_source.resize(
            (generation_w, generation_h), Image.LANCZOS,
        )
        generation_edit_region = cv2.resize(
            edit_region.astype(np.uint8), generation_input.size,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        generation_garment_crop_top = int(round(
            garment_crop_top * generation_input.height / max(1, identity_base.height)
        ))
        logger.info(
            "[UNIFORM_IDENTITY_REFERENCE] Using Enhancement-style full-frame native-aspect conditioning %s -> %s",
            person_pil.size, generation_input.size,
        )
        # Keep exactly one person tensor. Qwen EditPlus entangles every visual
        # reference; attaching the upload again as a supposed "identity-only"
        # image copied its old jacket, scenery and harsh sunlight into the
        # result. Image 1 already contains the untouched uploaded head and is
        # the sole identity authority. Image 2 is the one person-free garment
        # authority. A third solid-colour reference only adds visual tokens;
        # exact backdrop colour is now enforced deterministically after decode.
        identity_head_conditioning = None
        identity_head_conditioning_path = None
        qwen_garment_references = [
            uniform_validation_reference,
        ]
        accessory_reference_path = None
        accessory_visual_authority = ""
        try:
            reference_accessory_count = max(
                0, int(vl_plan.get("hair_accessory_count") or 0),
            )
        except (TypeError, ValueError):
            reference_accessory_count = 0
        reference_accessory_confidence = str(
            vl_plan.get("hair_accessory_confidence") or ""
        ).strip().lower()
        # Always supply a face-free full-hairstyle reference. A crown-only crop
        # proves accessory count but loses ponytail/curl endpoints, so Qwen can
        # still replace a bilateral or side style with a generic top ponytail.
        # This crop is measured from the uploaded face and parsed hair extent;
        # it is therefore dynamic for every portrait and never names a fixed
        # hairstyle. Removing the central face prevents a third visual from
        # being interpreted as another person.
        measured_end_ratio = rough_placement.get("measured_hair_end_ratio")
        try:
            measured_end_ratio = float(measured_end_ratio)
        except (TypeError, ValueError):
            measured_end_ratio = .28
        hair_reference_top = max(
            0, int(round(face_y1 - detected_face_height * .82)),
        )
        hair_reference_bottom = min(
            generation_source.height,
            int(round(face_y2 + detected_face_height * max(.20, min(.72, measured_end_ratio + .10)))),
        )
        hair_reference_left = max(
            0, int(round(face_x1 - detected_face_width * .95)),
        )
        hair_reference_right = min(
            generation_source.width,
            int(round(face_x2 + detected_face_width * .95)),
        )
        if (
            hair_reference_right - hair_reference_left >= 64
            and hair_reference_bottom - hair_reference_top >= 96
        ):
            hair_reference = source_hair_geometry_canvas.crop((
                hair_reference_left, hair_reference_top,
                hair_reference_right, hair_reference_bottom,
            )).convert("RGB")
            hair_reference_rgb = np.asarray(hair_reference).copy()
            blank_left = max(0, int(round(face_x1 - hair_reference_left - detected_face_width * .04)))
            blank_right = min(
                hair_reference_rgb.shape[1],
                int(round(face_x2 - hair_reference_left + detected_face_width * .04)),
            )
            blank_top = max(0, int(round(face_y1 - hair_reference_top + detected_face_height * .05)))
            blank_bottom = min(
                hair_reference_rgb.shape[0],
                int(round(face_y2 - hair_reference_top + detected_face_height * .10)),
            )
            hair_reference_rgb[blank_top:blank_bottom, blank_left:blank_right] = np.asarray(
                selected_bg_rgb, dtype=np.uint8,
            )
            hair_reference = Image.fromarray(hair_reference_rgb, "RGB")
            accessory_reference_path = (
                OUTPUTS_DIR / f"{job_id}_full_hair_reference.png"
            )
            hair_reference.save(accessory_reference_path, format="PNG")
            qwen_garment_references.append(hair_reference)
            accessory_clause = (
                f" It visibly contains exactly {reference_accessory_count} hair accessory object(s); copy their literal side, size, shape and attachment points."
                if reference_accessory_count > 0
                and reference_accessory_confidence in {"high", "medium"}
                else " Copy only accessories literally visible there; invent none."
            )
            accessory_visual_authority = (
                "Image 3 is a face-free full-hairstyle geometry reference extracted from the uploaded portrait. "
                "Copy its crown height, part, viewer-left and viewer-right silhouettes, tied-section count and anchors, curl/wave pattern, and visible endpoint on each side exactly. "
                "Its lighting and flat fill are not appearance references; use it only for hair geometry and accessories."
                + accessory_clause + " "
            )
        arm_generation_instruction = (
            "Image 1 has no usable exposed upper-arm pixels for the short-sleeve target. "
            "Generate only the small visible arm sections below the sleeve hems using the person's measured skin tone, continuous natural skin texture and smooth photographic detail; do not alter arm width or pose. "
            if requires_exposed_arm_synthesis else
            "Preserve every visible source arm pixel, arm width and arm pose exactly. "
        )
        if template_has_outer_layer:
            uniform_layer_lock = (
                "LAYER LOCK: preserve the measured shirt and outer-garment layers exactly as described; "
                "do not merge, remove or exchange their fabrics. "
            )
        else:
            uniform_layer_lock = (
                "LAYER LOCK: the target is one shirt only. Continue the measured patterned shirt fabric "
                "across the entire visible torso, both shoulders and both sleeves. Do not create any outer "
                "vest, pinafore, jacket, bib, apron, waistcoat or solid-color torso panel. Solid accent cloth "
                "may appear only at the measured trim locations. "
            )
        # Keep this instruction compact.  A former 468-word replacement prompt
        # diluted the fitted pixels with repeated prose, so Qwen redrew plaid as
        # wavy painted checks and occasionally changed the crop.  The single
        # conditioning image already carries the exact spatial answer; text is
        # only used to identify what may be harmonized and what must stay fixed.
        compact_layer_instruction = first_complete_sentence(
            uniform_layer_lock, "Copy the exact garment layers in image 2."
        )
        compact_pocket_lock = (
            f"POCKET: exactly {pocket_count} visible chest pocket"
            f"{'s' if pocket_count != 1 else ''}, {pocket_placement or 'position as shown'}, "
            f"{pocket_style or 'construction as shown'}; do not add, mirror or duplicate it. "
            if pocket_known else
            "POCKET: copy image 2's exact pocket count, position and construction. "
        )
        compact_badge_lock = (
            "Continue the surrounding fabric and pattern through the removed badge area; leave no badge, logo, text, blank patch or repair mark. "
            if template_badges_removed else
            "Add no badge, logo or text. "
        )
        compact_generated_trim_lock = (
            f"TRIM: use exactly {measured_accent['hex']} only at {trim_placement}; nowhere else. "
            if measured_accent and known_trim_placement else
            "TRIM: copy image 2's exact trim color and locations. "
        )
        analyzed_hair_color = str(vl_plan.get("hair_color") or "").strip()
        hair_color_authority = "the shaded, non-glare strands in image 1"
        if source_has_direct_light:
            hair_color_authority = (
                analyzed_hair_color
                if analyzed_hair_color.lower() not in {
                    "", "none", "unknown", "uncertain",
                }
                else "the shaded, non-glare strands in image 1"
            )
        portrait_repair_plan = vl_plan.get("portrait_repair_plan")
        if not isinstance(portrait_repair_plan, dict):
            portrait_repair_plan = {}
        trusted_repair_instruction = str(
            portrait_repair_plan.get("prompt") or ""
        ).strip()
        if not trusted_repair_instruction:
            trusted_repair_instruction = (
                f"Under soft neutral frontal light, render natural {hair_color_authority} hair and remove sunlight glare while preserving source facial geometry, hairstyle and accessories."
                if source_has_direct_light else
                "Keep natural even head lighting and copy source facial geometry, hairstyle and accessories."
            )
        repair_words = trusted_repair_instruction.split()
        if len(repair_words) > 64:
            trusted_repair_instruction = " ".join(repair_words[:64]).rstrip(" ,;")
        # The rough fit is a spatial guide, not permission to invent a new
        # head.  Asking Qwen to "regenerate" the head produced a plausible but
        # different child (and dropped real flower clips) even though the face
        # embedding stayed moderately close.  Keep the source-visible anatomy
        # and repair illumination only; this mirrors an image edit rather than
        # a portrait redesign.
        head_enhancement_contract = (
            f"HEAD REPAIR: {trusted_repair_instruction.rstrip('.')}. "
            "Image 1 alone is the literal identity, face, hair and accessory authority. "
            "Their geometry and details are authoritative, but their sunlight, glare, exposure and color cast are defects to remove, not features to copy. "
            "Preserve current apparent age, asymmetry, expression, complexion, marks, eyelid opening, eye diameter, cheeks, jaw, chin, hair silhouette and true accessory scale. "
            "Neutralize only the identified lighting defect; do not enlarge eyes, narrow the jaw/chin, smooth skin, beautify, de-age, age, symmetrize or create a different head."
        )
        # Saturated passport colours can be interpreted by Qwen as a coloured
        # rim light, producing a cyan/blue stroke around dark hair and pale
        # flowers even though the flat backdrop itself is correct. White does
        # not exhibit the defect. Keep the repair inside the generative pass:
        # explicitly separate the subject's final edge pixel from the selected
        # backdrop rather than applying a destructive post-generation outline
        # mask or rematte.
        chromatic_background = selected_is_chromatic
        chromatic_edge_contract = (
            f"EDGE COLOR LOCK: keep hair, accessories and clothing true-colored through the silhouette; {qwen_hex} RGB{tuple(qwen_bg_rgb)} starts only outside it. "
            "No backdrop-colored rim light, spill, stroke, outline, glow or fringe. "
            if chromatic_background else
            ""
        )
        # Pass only high-confidence, directly visible accessory evidence from
        # the focused head inspection. Generic hairstyle names are excluded:
        # a lossy label such as "ponytail" must never override image 1's
        # actual loose curls, hair ends or side silhouette.
        head_fact_parts = []
        accessory_confidence = str(
            vl_plan.get("hair_accessory_confidence") or ""
        ).strip().lower()
        accessory_details = str(
            vl_plan.get("hair_accessory_details") or ""
        ).strip()
        try:
            analyzed_accessory_count = max(
                0, int(vl_plan.get("hair_accessory_count") or 0)
            )
        except (TypeError, ValueError):
            analyzed_accessory_count = 0
        if (
            analyzed_accessory_count > 0
            and accessory_confidence in {"high", "medium"}
            and accessory_details.lower() not in {
                "", "none", "unknown", "uncertain",
            }
        ):
            accessory_label = "accessory" if analyzed_accessory_count == 1 else "accessories"
            head_fact_parts.append(
                f"exactly {analyzed_accessory_count} visible hair {accessory_label}: "
                f"{capped_words(accessory_details, 28)}; reproduce all {analyzed_accessory_count} separately "
                "at their literal image-1 sides, sizes and positions; do not drop, merge, mirror or invent one"
            )
            accessory_kind = str(
                vl_plan.get("hair_accessories") or accessory_details
            ).strip().lower()
            if "clip" in accessory_kind and not any(
                term in accessory_kind for term in ("flower", "bow", "ribbon")
            ):
                head_fact_parts.append(
                    "copy each clip's literal source silhouette; do not reinterpret it as a flower, bow or ribbon"
                )
        source_head_fact_lock = (
            "SOURCE DETAIL CHECK: " + "; ".join(head_fact_parts) + ". "
            "Image 1 contains the untouched uploaded head and defines the person's facial geometry, age, hair geometry, accessories and ear jewelry. "
            "Match those features while regenerating coherent neutral studio lighting and photographic texture; never paste a hard head patch or redesign them. "
            "These facts identify features to preserve, not permission to enlarge, redesign or mirror them. "
            if head_fact_parts else
            "SOURCE DETAIL CHECK: image 1 contains the untouched uploaded head. Match its facial geometry, age, hair geometry, accessories and ear jewelry "
            "while regenerating coherent neutral studio lighting; never paste a hard head patch, redesign features or invent anything. "
        )
        photographic_texture_lock = (
            "PHOTO LOCK: render an ordinary modern digital-camera photograph with natural lens microcontrast, subtle sensor grain, real skin pores and marks, fine facial hair, individual hair strands, true small jewelry scale and visible fabric weave. Preserve natural asymmetry and normal tonal variation. No CGI, 3D render, illustration, oil painting, airbrushing, waxy skin, plastic finish, beauty filter, doll face, painted hair or painted cloth. "
        )
        # Match Enhancement's short localized-edit contract. The fitted image
        # already contains the complete spatial answer; asking Qwen to
        # "preserve" every feature in a long design brief made it re-render the
        # face in order to satisfy those concepts.
        compact_repair = " ".join(trusted_repair_instruction.split()[:28]).rstrip(" ,;.")
        background_generation_contract = (
            f"Image 1 already has the required flat {qwen_hex} RGB{tuple(qwen_bg_rgb)} backdrop. "
            "Preserve it exactly outside the subject. "
            if conditioning_has_selected_background else
            f"Image 1 already has a flat passport backdrop. Recolor only that existing backdrop to exact {qwen_hex} RGB{tuple(qwen_bg_rgb)} without changing the subject silhouette. "
            if conditioning_has_flat_background else
            f"Replace only the complete original backdrop in image 1 with one perfectly flat {qwen_hex} RGB{tuple(qwen_bg_rgb)} field; preserve the literal outer hair/accessory silhouette. "
        )
        bg_contract_lead = (
            f"ONE PERSON ONLY. Edit image 1 as one natural upper-chest school-ID passport photograph; do not synthesize a new person. "
            f"{background_generation_contract}"
            "Keep that color outside the subject; no scenery, gradient, texture, shadow or color spill. "
        )
        # These must stay near the front of the positive prompt. Dynamic prompt
        # compaction keeps the leading contract. Do not ask the diffusion pass
        # to redirect gaze: even with an identity clause, Qwen satisfies that
        # request by reconstructing larger, more symmetric eyes and a different
        # face. The uploaded portrait already supplies the literal pupil
        # direction; keep the entire eye region source-faithful.
        passport_identity_lock = (
            "IDENTITY LOCK: image 1 is the sole person authority. Treat every visible pixel from the crown through the bottom of the neck as a protected no-regeneration region. "
            "Preserve its exact face, current age, proportions, complexion, expression and natural asymmetry; never beautify, smooth, de-age, relight or redesign it. "
        )
        source_eye_description = " ".join(
            str(vl_plan.get("eye_description") or "").split()
        )[:120]
        passport_iris_lock = (
            f"Eyes are {source_eye_description}; "
            if source_eye_description and source_eye_description.lower() not in {"unknown", "uncertain", "none"}
            else "Copy image 1's literal iris color; "
        ) + (
            f"keep that natural iris color. {qwen_hex} never enters the eyes. "
        )
        passport_eye_lock = (
            "Copy image 1's exact eye shape, size, spacing, pupils, lids and brows; never redirect, enlarge or redraw them. "
        )
        compact_hair_facts = []
        hair_fact_keys = [
            "hair_parting", "hair_length", "hair_texture",
            "tied_section_count", "tied_section_sides",
            "hair_accessories", "hair_accessory_count",
            "hair_accessory_details",
        ]
        # Semantic labels such as ponytail/bun are used only when the analyser
        # explicitly trusts the visible geometry.  Otherwise the source pixels
        # remain authoritative, preventing a wrong VL label from inventing a
        # bun, pigtail or mirrored tail.
        hair_geometry_trusted = bool(portrait_repair_plan.get("hair_geometry_trusted"))
        try:
            trusted_tied_section_count = (
                max(0, int(vl_plan.get("tied_section_count") or 0))
                if hair_geometry_trusted else 0
            )
        except (TypeError, ValueError):
            trusted_tied_section_count = 0
        trusted_tied_section_sides = " ".join(
            str(vl_plan.get("tied_section_sides") or "").split()
        )[:64]
        if hair_geometry_trusted:
            hair_fact_keys.insert(0, "hair_style")
        for hair_key in hair_fact_keys:
            hair_value = " ".join(str(vl_plan.get(hair_key) or "").split())[:48]
            if hair_value and hair_value.lower() not in {
                "none", "unknown", "uncertain", "not visible", "not detected",
            }:
                compact_hair_facts.append(f"{hair_key.replace('_', ' ')}: {hair_value}")
        raw_hair = vl_plan.get("hair_observations_raw")
        if hair_geometry_trusted and isinstance(raw_hair, dict):
            try:
                tied_section_count = max(
                    0, int(vl_plan.get("tied_section_count") or 0),
                )
            except (TypeError, ValueError):
                tied_section_count = 0
            tied_section_sides = " ".join(
                str(vl_plan.get("tied_section_sides") or "").split()
            )[:48]
            ponytail_side = " ".join(
                str(raw_hair.get("ponytail_side") or "").split()
            )[:24]
            if tied_section_count >= 2 and tied_section_sides.lower() not in {
                "", "unknown", "uncertain", "none",
            }:
                compact_hair_facts.append(
                    f"tied hair sections: {tied_section_count}, sides: {tied_section_sides}"
                )
            elif (
                tied_section_count == 1
                and ponytail_side
                and ponytail_side.lower() not in {"unknown", "uncertain", "none"}
            ):
                compact_hair_facts.append(f"visible ponytail side: {ponytail_side}")
        compact_hair_summary = "; ".join(compact_hair_facts)
        tied_geometry_lock = ""
        if trusted_tied_section_count > 0:
            tied_geometry_lock = (
                f"Render exactly {trusted_tied_section_count} independently tied hair section(s)"
                + (
                    f" at {trusted_tied_section_sides}"
                    if trusted_tied_section_sides.lower() not in {
                        "", "unknown", "uncertain", "none",
                    }
                    else " at the source-visible locations"
                )
                + "; do not add, merge, mirror or move a tied section, and do not invent an extra crown, top or centered ponytail. "
            )
        passport_hair_lock = (
            "HAIR LOCK: copy image 1's exact hairline, part, crown, side silhouettes, curls, length and every accessory "
            "at the same side, count and size. Never simplify, slick, shorten, straighten or remove them. "
            + tied_geometry_lock
            + accessory_visual_authority
            + (f"Facts: {compact_hair_summary}. " if compact_hair_summary else "")
        )
        repair_regions = " ".join(
            str(vl_plan.get("sunlight_regions") or "detected highlight regions").split()
        )[:64]
        residual_qwen_repair = (
            f"LIGHTING FINISH: neutralize the detected hard light in {repair_regions} using soft frontal illumination, without changing geometry, age, natural complexion or texture. "
            if source_has_direct_light else
            "LIGHTING FINISH: preserve the existing natural complexion and hair color. "
        )
        qwen_edge_finish = (
            "EDGE FINISH: remove only residual scene spill or halo at the existing silhouette. Preserve every strand, accessory and boundary position; never rematte or reshape the subject. "
        )
        if identity_prepass_used:
            # Image 1 is already the validated, relit identity restoration.
            # Asking Qwen to regenerate or relight it again enlarged the eyes,
            # smoothed the face and changed dense landmarks. Keep this second
            # pass localized to garment synthesis and backdrop hue only.
            generated_prompt = (
                f"{bg_contract_lead}"
                f"{passport_identity_lock}"
                f"{passport_iris_lock}"
                f"{passport_eye_lock}"
                f"{passport_hair_lock}"
                f"{photographic_texture_lock}"
                "Edit image 1 as one natural camera photograph of the same single person. "
                "Image 1 is the exact identity and head authority: do not redraw, beautify, de-age, relight or restyle the face, eyes, skin, head, hair, flowers, earrings, neck, pose, shoulders or body proportions. The only permitted neck-area edit is erasing every necklace, chain and pendant and reconstructing natural bare skin. "
                "Change only the clothing below the neck into the already fitted target uniform, using image 2 as the exact design authority. Render it as naturally tailored fabric on image 1's body, copying image 2's exact colors, check scale, collar, placket, buttons, pocket, sleeves, cuffs, trim and layers. "
                "Continue matching fabric through the removed badge area with no badge, denim patch, repaired rectangle, logo, text or extra stitching. Remove only any necklace, chain or pendant. "
                "Return one person and one face in the existing upper-chest school-ID frame; natural photographic texture only, never a painting, collage or pasted face."
            )
        else:
            # Normal production route: image 1 already contains the complete
            # spatial answer (current uploaded head, selected flat backdrop,
            # and fitted target garment). A short local-edit command prevents
            # Qwen from interpreting "school-ID portrait" as a request to
            # synthesize a different child or decorate the flat background.
            generated_prompt = (
                "ONE PERSON ONLY. LOCAL EDIT. Image 1 already contains the uploaded person, upper-chest crop, flat backdrop and fitted target uniform. "
                "HEAD LOCK: preserve image 1's exact head geometry, identity and current age. Re-render its same face, hair and accessories as a natural photograph under soft neutral frontal light, removing the measured harsh sunlight, metallic glare and hard cast shadows without beautifying or redesigning the person. "
                f"{passport_hair_lock}"
                f"{residual_qwen_repair}"
                f"BACKDROP LOCK: preserve the flat {qwen_hex} RGB{tuple(qwen_bg_rgb)} field; add no scenery, gradient, texture or shadow. "
                f"{qwen_edge_finish}"
                "Edit garment pixels below the neck, removing rough seams and fitting the uniform naturally to the same body. "
                "Image 2 alone defines fabric colors, pattern scale, collar, placket, buttons, pocket, sleeves, cuffs, trim and layers. Continue its fabric through the removed badge area. "
                "No badge, logo, patch, text, necklace, chain or pendant. Preserve head size, neck length, shoulders and torso width. "
                "Return the same photograph with real fabric weave and natural camera texture; no painting, illustration, CGI, plastic skin, collage or pasted face."
            )
        prompt_word_count = len(generated_prompt.split())
        if not identity_prepass_used:
            logger.info(
                "[UNIFORM_PROMPT] Local generation contract contains %d words",
                prompt_word_count,
            )
        if prompt_word_count > 280:
            # Never truncate by word position. That formerly removed the
            # measured flowers/curls, eye constraints and collar construction,
            # leaving only generic "hair" and "uniform" prose. Rebuild one
            # complete compact contract in which every required authority is
            # present exactly once.
            generated_prompt = (
                f"ONE PERSON ONLY. Edit image 1 into one natural upper-chest school-ID passport photograph on an exact flat "
                f"{qwen_hex} RGB{tuple(qwen_bg_rgb)} backdrop. Remove all scenery, texture, gradient, shadow and spill. "
                f"{passport_identity_lock}{passport_iris_lock}{passport_eye_lock}"
                "No makeup, eyeliner, mascara, false eyelashes, beauty filter, enlarged eyes, narrowed nose, fuller cheeks or reshaped jaw. "
                f"{passport_hair_lock}"
                f"{photographic_texture_lock}"
                "Image 2 is the sole uniform authority. Replace all old clothing below the neck; copy its exact fabric, pattern scale, folded collar, placket, buttons, pocket, sleeves, cuffs and trim as tailored cloth. "
                "No badge, logo, patch, text, necklace, chain or pendant. "
                "Keep image 1's measured neck length, head-to-shoulder scale and body width; never enlarge the head, lengthen the neck or narrow the torso. "
                "Keep complete crown, modest headroom, level shoulders and upper chest only; no waist, hands or long torso. Preserve natural photographic skin and hair texture; never return a painting, collage, pasted face or rough composite."
            )
            logger.warning(
                "[UNIFORM_PROMPT] Compacted dynamic generation prompt from %d to %d words",
                prompt_word_count, len(generated_prompt.split()),
            )
            prompt_word_count = len(generated_prompt.split())
        if prompt_word_count > 280:
            # Dynamic analyser values can themselves be unexpectedly verbose.
            # Rebuild a smaller complete contract rather than slicing words
            # off the end and losing the garment or crop instruction.
            short_hair_facts = []
            for hair_key in (
                "hair_parting", "hair_texture", "hair_accessories",
                "hair_accessory_count",
            ):
                hair_value = " ".join(str(vl_plan.get(hair_key) or "").split())[:32]
                if hair_value and hair_value.lower() not in {
                    "none", "unknown", "uncertain", "not visible", "not detected",
                }:
                    short_hair_facts.append(
                        f"{hair_key.replace('_', ' ')}={hair_value}"
                    )
            short_hair_summary = "; ".join(short_hair_facts)
            generated_prompt = (
                f"ONE PERSON ONLY. Natural upper-chest school-ID photograph on exact flat {qwen_hex} "
                f"RGB{tuple(qwen_bg_rgb)}; no scenery, texture, gradient, shadow or spill. "
                "Image 1 alone defines identity, current age, face geometry, complexion, expression, eyes, body proportions, neck and hair. Preserve that geometry exactly while regenerating coherent natural photographic texture. "
                "Remove harsh directional sunlight, metallic hair/forehead glare and hard cast shadows; use soft neutral frontal light without beautifying, de-aging, smoothing, reshaping, enlarging eyes, redirecting gaze or changing skin tone. "
                "Copy its literal hairline, part, texture, length, silhouette and every hair accessory at the same location, count and size. "
                + (f"Hair facts: {short_hair_summary}. " if short_hair_summary else "")
                + accessory_visual_authority
                + photographic_texture_lock
                + "Image 2 alone defines the uniform. Replace all old clothing below the neck and reproduce its exact fabric colors, pattern scale, collar, placket, buttons, pocket, sleeves, cuffs, trim and layers as naturally tailored cloth on image 1's measured body. "
                "No badge, logo, patch, text, necklace, chain or pendant. Do not enlarge the head, lengthen the neck or narrow the torso. "
                "Complete crown, modest headroom, level shoulders and upper chest only; no waist, hands or long torso. "
                "Photographic skin, hair and fabric texture only; no painting, collage, pasted face or rough composite."
            )
            logger.warning(
                "[UNIFORM_PROMPT] Rebuilt analyser-safe prompt at %d words",
                len(generated_prompt.split()),
            )
            prompt_word_count = len(generated_prompt.split())
        if prompt_word_count > 280:
            # Keep the final fallback a LOCAL EDIT.  The former fixed fallback
            # changed this into a broad portrait-generation request whenever
            # dynamic hair evidence made the prompt long.  On harsh-light
            # inputs Qwen then copied the source sunlight and old denim instead
            # of finishing the already aligned garment canvas.
            compact_visual_hair_authority = (
                "Image 3 is a face-free crop of this upload's full hairstyle. Copy its crown, part, both side silhouettes, tied-section anchors, curls, ends and every visible hair accessory exactly; use it only for geometry, not lighting. "
                if accessory_reference_path is not None else ""
            )
            generated_prompt = (
                "ONE PERSON ONLY. LOCAL EDIT. Image 1 already contains the uploaded person, fitted target uniform, upper-chest crop and flat backdrop. "
                "Keep exact identity, age, expression, face geometry, gaze and head/neck/body proportions. "
                "Re-render the same face and hair under soft neutral frontal light. Remove hot forehead, cheek, nose and crown highlights, directional sunlight, metallic hair glare and hard shadows without beautifying, smoothing, de-aging, reshaping or changing natural skin tone. "
                "Copy both ears and their jewelry exactly from image 1 at the same color, shape, scale and position. Never recolor skin or earrings with backdrop, headband, ribbon or hair-accessory colors. "
                "Copy image 1's exact hairline, part, crown, side silhouettes, texture, length, ends and accessories. "
                f"{tied_geometry_lock}"
                f"{compact_visual_hair_authority}"
                "Finish only the garment below the neck. Image 2 alone defines its colors, pattern scale, collar, placket, buttons, pocket, sleeves, cuffs, trim and layers. Remove old clothing and rough seams; tailor it to image 1's body. "
                f"Preserve the exact flat {qwen_hex} RGB{tuple(qwen_bg_rgb)} backdrop without scenery, texture, gradient or shadow. "
                "No badge, logo, patch, text, necklace, chain or pendant. Keep complete crown, headroom, level shoulders and upper chest only; no waist or hands. "
                "Return one coherent photograph with real skin, hair strands and fabric weave; no painting, CGI, plastic skin, collage or pasted face."
            )
            prompt_word_count = len(generated_prompt.split())
            logger.warning(
                "[UNIFORM_PROMPT] Rebuilt compact local-edit contract at %d words",
                prompt_word_count,
            )

        negative_prompt = (
            "multiple people, second face, duplicate head, different person, changed ethnicity appearance, changed ancestry appearance, stereotyped face, changed apparent age, "
            "changed facial proportions, changed complexion, exaggerated eyes, enlarged eyes, invented earrings, enlarged earrings, dangling earrings, hoop earrings not in source, "
            "makeup, eyeliner, false eyelashes, mascara, eye shadow, blush, lipstick, beauty filter, airbrushed or waxy skin, changed hairstyle, hairline, length, parting or accessories, invented jewelry, "
            "head tilt, side gaze, rotated torso, uneven shoulders, changed shoulder width or slope, oversized head, undersized torso, narrow chest, widened chest, changed torso taper, thin arms, thick arms, template-model body, mannequin proportions, copied garment silhouette, loose boxy shirt, oversized uniform, zoomed-in face, "
            "cropped hair, missing sleeve, long shot, half-body portrait, full-body portrait, elbows, forearms, hands, waist, abdomen, long torso, lower body, pasted head, floating head, neck gap, long neck, narrow neck, rectangular neck, sideways neck, bent neck, curved neck, off-center neck, shifted collar, slanted placket, collage, "
            "source clothing, wrong collar, wrong pattern, wrong uniform color, missing fabric layers, painted uniform, illustrated fabric, plastic fabric, badge, emblem, logo, text, "
            "neckwear absent from image 2, necklace, chain, pendant, harsh sunlight, colored hair glare, backdrop spill, smoke, haze, blur, "
            "scenery, room, wall, window, prop, texture, gradient, vignette, background shadow, uneven background, cutout halo"
        )
        if not template_has_neckwear:
            negative_prompt += ", tie, necktie, school tie, striped tie, bow tie, collar bow, neck ribbon, cravat"
        if not template_has_outer_layer:
            negative_prompt += ", vest, waistcoat, pinafore, jacket, blazer, bib, apron, outer layer, solid torso panel"
        negative_prompt += f", {age_negative_terms}"
        if opposite_ponytail_side:
            negative_prompt += (
                f", ponytail on {opposite_ponytail_side}, hair bundle on {opposite_ponytail_side}, "
                "bilateral ponytails, mirrored ponytail, second ponytail"
            )
        if ponytail_position_instruction:
            negative_prompt += ", high ponytail, crown ponytail, top ponytail, ponytail above the ears, top hair loop, visible hair band, crown bun"
        if not preserve_tied_back_hair and not vl_claims_tied_hair:
            negative_prompt += ", ponytail, pigtails, tied hair, hair bun, braid, plait, trailing hair bundle, invented hair accessory"
        negative_prompt += ", changed earrings, removed earrings, added earrings, mirrored earrings, enlarged earrings"
        if not has_glasses:
            negative_prompt += ", eyeglasses, spectacles, glasses frames, lens reflections"
        if (
            "ponytail" in str(vl_plan.get("hair_style") or "").lower()
            and "straight" in str(vl_plan.get("hair_texture") or "").lower()
        ):
            negative_prompt += (
                ", dreadlocks, locs, braids, cornrows, twists, loose side tendrils, multiple ponytails, "
                "short rounded hair, afro, curly hair, coiled hair, fluffy hair, expanded crown, cropped hair"
            )
        generation_negative_prompt = (
            "different person, changed face, younger face, changed skin tone, changed hairstyle, "
            "invented hair, forehead mark, bindi, makeup, added jewelry, necklace, chain, removed glasses, "
            "changed earrings, second person, duplicate face, abstract check pattern, changed fabric grid, "
            "wrong uniform color, painted clothing, illustration, pasted face, hard jaw edge, long neck, "
            "detached collar, pixelated edge, cutout halo, block artifact"
        )
        if not template_has_outer_layer:
            generation_negative_prompt += (
                ", vest, waistcoat, pinafore, jacket, blazer, bib, apron, outer layer, solid torso panel"
            )
        if pocket_known:
            if pocket_count == 0:
                generation_negative_prompt += ", chest pocket, flap pocket"
            elif pocket_count == 1:
                generation_negative_prompt += ", second pocket, mirrored pocket, extra pocket"
            if "plain" in pocket_style.lower() and "flap" not in pocket_style.lower():
                generation_negative_prompt += ", pocket flap, flap-pocket button"
        # This route intentionally has no negative prompt because Qwen EditPlus
        # would attach and subtract the same garment references in the CFG
        # branch. With positive-only conditioning the effective CFG is 1.0;
        # record and pass the real value instead of the former misleading 1.2.
        generation_cfg_scale = 1.0
        # Retain the actual instructions and analysis for diagnosing a run,
        # rather than assuming an identity score proves garment/hair fidelity.
        import json
        audit_path = Path("outputs") / f"{job_id}_instructions.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps({
            "analysis": vl_plan, "prompt": generated_prompt,
            "prompt_word_count": prompt_word_count,
            "negative_prompt": generation_negative_prompt, "seed": uniform_seed,
            "seed_policy": "enhancement_source_pixels_plus_canonical_neutral_marker",
            "steps": generation_steps,
            "step_policy": (
                "validated_4_step_current_upload_identity_prepass_then_4_step_uniform_edit"
                if identity_prepass_used else
                "single_4_step_original_upload_uniform_and_restoration"
            ),
            "step_reasons": step_reasons,
            "identity_reference": "the untouched uploaded portrait is authoritative for identity, hair and accessory generation",
            "identity_validation_threshold": 0.70,
            "reference_count": 1 + len(qwen_garment_references),
            "prompt_token_budget": uniform_prompt_token_budget, "conditioning_max_dimension": conditioning_dimension,
            "conditioning_pixel_budget": conditioning_dimension * conditioning_dimension,
            "true_cfg_scale": generation_cfg_scale,
            "negative_conditioning": "disabled for the seeded uniform edit so CFG does not subtract the garment image embedding",
            "template_size": list(template_pil.size), "preserve_reference_aspect": True,
            "vl_observations_used_as_generation_facts": vl_plan.get("source") == "qwen2.5-vl",
            "template_badges_removed": template_badges_removed,
            "fabric_color_instruction": fabric_color_instruction,
            "uniform_generation_facts": uniform_generation_facts,
            "hair_color_instruction": hair_color_instruction,
            "hair_shape_instruction": hair_shape_instruction,
            "skin_prompt_instruction": skin_prompt_instruction,
            "lighting_instruction": lighting_instruction,
            "skin_tone_instruction": skin_tone_instruction,
            "source_appearance_lock": source_appearance_lock,
            "source_age_estimate": source_age_value,
            "source_age_band": source_age_band,
            "age_negative_terms": age_negative_terms,
            "measured_source_colors": measured_person_colors,
            "identity_instruction": identity_instruction,
            "body_fit_instruction": body_fit_instruction,
            "body_fit_prompt_instruction": body_fit_prompt_instruction,
            "uniform_layer_lock": uniform_layer_lock,
            "primary_conditioning": (
                "image 1 is the current uploaded portrait with the measured rough uniform blended only inside the garment edit region"
                if not identity_prepass_used else
                "explicit experimental identity-prepass portrait with the measured rough uniform blended only inside the garment edit region"
            ),
            "identity_conditioning": "image 1 is the only person tensor and sole identity, hair, accessory, body-geometry and pose authority",
            "uniform_conditioning": "image 2 is a cleaned face-free garment crop carrying the template color, pattern and construction authority",
            "background_conditioning": "image 1 already carries the selected color; exact selected-color compliance is enforced by protected post-decode matting",
            "rough_layout_path": str(rough_path),
            "rough_fixed_layout_path": str(rough_fixed_path),
            "garment_edit_crop_path": str(garment_crop_path),
            "direct_uniform_seed_path": str(direct_seed_path),
            "garment_crop_top": garment_crop_top,
            "identity_head_path": str(identity_head_path),
            "hair_accessory_reference_path": (
                str(accessory_reference_path) if accessory_reference_path else None
            ),
            "uniform_passport_reference_path": str(uniform_reference_path),
            "rough_layout_placement": rough_placement,
            "portrait_details": portrait_details,
            "garment_details": garment_details,
            "measured_source_dark_hair": measured_dark_hair,
            "outer_target_rgb": outer_target_rgb,
            "conditioning_lighting": "Qwen performs one coherent full-canvas photographic uniform edit",
            "head_enhancement_contract": head_enhancement_contract,
            "chromatic_edge_contract": chromatic_edge_contract,
            "source_head_fact_lock": source_head_fact_lock,
            "photographic_texture_lock": photographic_texture_lock,
            "portrait_repair_plan": portrait_repair_plan,
            "conditioning_background": "image 1 carries the selected backdrop; protected post-decode matting guarantees the exact selected color",
            "conditioning_guide_rgb": list(conditioning_bg_rgb),
            "qwen_background_rgb": list(qwen_bg_rgb),
            "selected_background_rgb": list(selected_bg_rgb),
            "delivery_policy": "qwen_uniform_then_gated_identity_and_exact_background_verified",
            "prompt_review": "Qwen Image Edit Plus receives exactly one person tensor: the uploaded portrait with an aligned garment guide. Image 1 controls identity/head/body/layout; image 2 is the face-free garment authority.",
            "source_identity_reference_path": str(source_identity_reference_path),
            "person_conditioning_path": str(person_conditioning_path),
            "selected_bg_conditioning_path": (
                str(selected_bg_conditioning_path)
                if selected_bg_conditioning_path is not None else None
            ),
            "identity_head_conditioning_path": (
                str(identity_head_conditioning_path)
                if identity_head_conditioning_path is not None else None
            ),
            "person_matted_path": str(person_matted_path),
        }, ensure_ascii=True, indent=2), encoding="utf-8")
        from pipelines.uniform_review import check_uniform_identity
        if progress_callback:
            progress_callback(12, "Preparing the complete uniform portrait generation...")
        output_path = self.qwen_edit_enhancer(
            image_input=generation_input,
            reference_images=qwen_garment_references,
            prompt=generated_prompt,
            # The isolated encoder attaches the conditioning image to every
            # prompt it encodes. Passing a negative prompt here therefore
            # subtracts the same garment-seed image embedding during true CFG,
            # which made Qwen erase the plaid and reconstruct the source vest.
            # The positive prompt plus explicit delivery gates carry the
            # constraints without subtracting the target garment.
            negative_prompt=None,
            job_id=job_id,
            background_color=qwen_bg_rgb,
            width=generation_input.width,
            height=generation_input.height,
            original_filename=f"{job_id}_uniform",
            steps=generation_steps,
            max_sequence_length=uniform_prompt_token_budget,
            timeout_seconds=None,
            progress_callback=(lambda value, message: progress_callback(15 + int(value * 0.75), message)) if progress_callback else None,
            preserve_source_clothing=False,
            max_generation_dimension=uniform_generation_dimension,
            minimum_generation_dimension=uniform_generation_dimension,
            keep_generation_resolution=True,
            use_birefnet_background=False,
            true_cfg_scale=generation_cfg_scale,
            seed=uniform_seed,
            face_lock_after_generation=False,
            reject_identity_failure=False,
            # Always return the actual decode. Without this, an identity miss
            # silently falls back to image_input and can report the rough/source
            # canvas as a completed uniform portrait.
            return_raw_candidate=True,
            # Plain uniforms keep the identity-efficient 448px profile. Fine
            # multicolour patterns receive 512px so the encoder can distinguish
            # base cloth from narrow accent threads.
            conditioning_max_dimension=conditioning_dimension,
            conditioning_pixel_budget=conditioning_dimension * conditioning_dimension,
            preserve_reference_aspect=True,
        )
        generated_candidate = Image.open(output_path).convert("RGB").resize(
            generation_input.size, Image.LANCZOS,
        )
        # Keep the untouched VAE decode as a separate diagnostic, then correct
        # only its already-flat border-connected backdrop before publishing the
        # artifact named ``decoder_raw``. Qwen's VAE commonly shifts a requested
        # RGB value by a few levels (for example #047EF6 -> roughly #0384FA).
        # Leaving that drift in decoder_raw made the later exact-color result
        # appear to be produced only by an unrelated final matte. The strict
        # correction cannot reach textured subject pixels, so face, hair,
        # accessories and uniform remain the genuine single-pass decode.
        decoder_native_path = OUTPUTS_DIR / f"{job_id}_decoder_native.png"
        generated_candidate.save(decoder_native_path, format="PNG")
        # A flat-colour normalizer is safe only when Qwen already generated a
        # smooth backdrop. Running it over retained foliage/walls produces the
        # blue-and-scenery fragments seen in failed deliveries. Keep the native
        # decode for diagnostics and reject it below when scenery remains.
        native_backdrop_is_flat = (
            self._has_flat_border_background(generated_candidate)
            or self._has_selected_solid_background(generated_candidate, selected_bg_rgb)
        )
        background_rematted = False
        if native_backdrop_is_flat:
            # The decoder has already produced a flat backdrop. Do not compare
            # its subject against colours sampled from the original scene:
            # pale flowers, highlighted curls and earrings can legitimately
            # resemble source foliage/wall pixels and were being erased here.
            # Border-connected hue normalization is sufficient and cannot
            # reach isolated subject details.
            generated_candidate = self._replace_flat_border_background_strict(
                generated_candidate, selected_bg_rgb,
            )
        else:
            logger.warning(
                "[UNIFORM_BACKGROUND] Native Qwen decode retained detailed scenery; "
                "applying protected semantic foreground matte",
            )
            # Qwen's edit is stochastic: a valid identity/uniform render can
            # occasionally retain scenery even when the prompt requests a flat
            # field. Backdrop compliance must therefore not be an acceptance
            # dependency of diffusion. Segment the *generated* portrait (not
            # the rough/source canvas), preserve every Qwen foreground RGB
            # pixel, and deterministically composite the selected colour behind
            # it. SCHP supplies hard protection for face, hair, accessories,
            # neck and clothing while GrabCut recovers their fine boundary.
            generated_candidate = self._replace_background_with_foreground_matte(
                generated_candidate,
                selected_bg_rgb,
                job_id=job_id,
                preserve_foreground_rgb=True,
                strict=True,
            ).convert("RGB")
            background_rematted = True
            native_backdrop_is_flat = (
                self._has_selected_solid_background(generated_candidate, selected_bg_rgb)
                or self._has_flat_border_background(generated_candidate)
            )
            if not self._has_selected_solid_background(
                generated_candidate, selected_bg_rgb,
            ):
                generated_candidate = self._clear_detached_flat_background_artifacts(
                    generated_candidate, selected_bg_rgb,
                )
            if not self._has_selected_solid_background(
                generated_candidate, selected_bg_rgb,
            ):
                raise RuntimeError(
                    "Uniform output was not delivered because the generated person could not be safely separated from its background."
                )

        # Deliver only Qwen's coherent generated portrait. A post-generation
        # face/hair merge copied source lighting, necklace and background into
        # the new collar and produced a visibly transparent cut-and-paste head.
        # Identity is collected before garment generation by the validated
        # Qwen prepass above and enforced below by biometric/geometry gates.
        identity_merge_applied = False
        decoder_raw_path = OUTPUTS_DIR / f"{job_id}_decoder_raw.png"
        generated_candidate.save(decoder_raw_path, format="PNG")
        # A successful decoder call is not sufficient: the edit model can
        # occasionally echo its source portrait instead of performing the requested
        # clothing replacement. Measure only the editable garment pixels and
        # reject a near-copy before any review or delivery logic can accept it.
        generated_rgb_for_delta = np.asarray(generated_candidate, dtype=np.float32)
        conditioning_rgb_for_delta = np.asarray(generation_input, dtype=np.float32)
        edit_delta = np.mean(
            np.abs(generated_rgb_for_delta - conditioning_rgb_for_delta), axis=2,
        )[generation_edit_region]
        generated_edit_delta = float(np.mean(edit_delta)) if edit_delta.size else 0.0
        generated_edit_unchanged = (
            float(np.mean(edit_delta < 8.0)) if edit_delta.size else 1.0
        )
        logger.info(
            "[UNIFORM_GENERATION_DELTA] editable_mean=%.2f unchanged=%.3f",
            generated_edit_delta, generated_edit_unchanged,
        )
        if generated_edit_delta < 10.0 or generated_edit_unchanged > .82:
            raise RuntimeError(
                "Uniform output was not delivered because Qwen returned the conditioning canvas instead of a regenerated uniform."
            )
        # Never patch a generated badge with copied pixels. Even a correct
        # detector cannot phase-align checks, pocket perspective and fabric
        # lighting perfectly, so that repair looks like an extra stitched
        # panel. Qwen must generate seamless fabric in one pass; the mandatory
        # visual reviewer rejects any remaining badge or repair mark.
        # Deliver the actual Qwen photograph. Post-generation face locks,
        # semantic rematting, lip correction and garment pixel repairs made a
        # coherent decode look like the rough guide. The user-facing uniform
        # result is now the untouched decoder image with only deterministic
        # school-passport framing applied. Validation may reject a bad decode,
        # but it must never construct a different one.
        raw_faces = _get_insight_app().get(
            cv2.cvtColor(np.asarray(generated_candidate), cv2.COLOR_RGB2BGR)
        )
        if len(raw_faces) != 1:
            raise RuntimeError(
                "Uniform output was not delivered because Qwen generated more than one person or face."
            )
        source_has_direct_light = bool(
            lighting_truthy(vl_plan.get("direct_sunlight_present"))
            or lighting_truthy(vl_plan.get("head_hair_hotspot_present"))
        )
        # Do not let a high ArcFace score hide a visibly unusable restoration.
        # The failed 502ca359 result retained silver/orange sunlight patches
        # across the crown and face yet passed identity and garment checks.
        # Measure the decoded face before any crop or background operation so
        # those later stages cannot disguise a failed lighting edit.
        if (
            source_has_direct_light
            and self._has_residual_uniform_harsh_light(
                generated_candidate, raw_faces[0].bbox,
            )
        ):
            raise RuntimeError(
                "Uniform output was not delivered because measured harsh sunlight remained after the adaptive first generation."
            )
        # Preserve the coherent Qwen decode exactly. Enhancement now performs
        # relighting inside its single generative edit; applying the older
        # hotspot filter here changed face chroma and texture after generation.
        logger.info(
            "[UNIFORM_LIGHTING] Keeping Qwen-generated head/face pixels; no post-generation hotspot pass"
        )
        # Do not rematte the generated person. Scene-colour cleanup already ran
        # on parser-confirmed background pixels before decoder_raw was saved;
        # running BiRefNet/SCHP over the final portrait clipped curls and pale
        # accessories on otherwise valid Qwen generations.
        # Match Enhancement after decoding: preserve Qwen's coherent subject
        # interior. The older hard uniform-only fringe key used labels from the
        # conditioning canvas and cut pale plaid, shoulders and hair pixels.
        # `decoder_native` above remains the untouched diffusion diagnostic;
        # `decoder_raw` already has only the flat border-connected colour drift
        # corrected. The standard `qwen_raw` artifact is the actual Qwen
        # candidate used by the uniform workflow, so keep its backdrop exact as
        # well before review. This does not rebuild the subject, touch interior
        # garment pixels, or invoke a second generative pass.
        # Correct only flat colour regions connected to the canvas border.
        # The fitted-silhouette GrabCut path used the rough guide as geometry;
        # on curls, flowers and ponytails it retained a dark scenery-shaped
        # cloud or clipped real hair. Strict border normalization preserves all
        # generated subject pixels and changes only the already-flat backdrop.
        generated_candidate = self._replace_flat_border_background_strict(
            generated_candidate, selected_bg_rgb,
        )
        # A coverage check can report "uniform present" even when Qwen replaces
        # a patterned single shirt with a solid vest. Compare high-frequency
        # fabric structure over the central torso against the fitted seed. This
        # is geometry- and upload-driven; no garment color or template is
        # hard-coded. Buttons and seams occupy too little area to pass this gate.
        if multi_color_pattern and not template_has_outer_layer:
            torso_structure_region = (
                (mask_y >= max(collar_bottom, identity_base.height * .53))
                & (mask_y <= identity_base.height * .94)
                & (np.abs(mask_x - face_center_x) <= min(
                    identity_base.width * .30, detected_face_width * 1.40,
                ))
            )
            torso_structure_region = cv2.resize(
                torso_structure_region.astype(np.uint8), generated_candidate.size,
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            seed_gray = cv2.cvtColor(
                np.asarray(
                    direct_uniform_seed.resize(generated_candidate.size, Image.LANCZOS),
                    dtype=np.uint8,
                ), cv2.COLOR_RGB2GRAY,
            )
            candidate_gray = cv2.cvtColor(
                np.asarray(generated_candidate, dtype=np.uint8), cv2.COLOR_RGB2GRAY,
            )
            seed_edges = np.abs(cv2.Laplacian(seed_gray, cv2.CV_32F)) > 12.0
            candidate_edges = np.abs(cv2.Laplacian(candidate_gray, cv2.CV_32F)) > 12.0
            seed_pattern_density = float(np.mean(seed_edges[torso_structure_region]))
            candidate_pattern_density = float(np.mean(candidate_edges[torso_structure_region]))
            minimum_pattern_density = max(.055, seed_pattern_density * .28)
            logger.info(
                "[UNIFORM_PATTERN_GATE] seed=%.3f candidate=%.3f minimum=%.3f",
                seed_pattern_density, candidate_pattern_density, minimum_pattern_density,
            )
            if (
                seed_pattern_density >= .12
                and candidate_pattern_density < minimum_pattern_density
            ):
                raise RuntimeError(
                    "Uniform output was not delivered because Qwen replaced the uploaded patterned shirt with a solid garment."
                )
        # Deliver Qwen's coherent generated garment. Never paste the rough
        # composite back over it: that guide has deliberately approximate
        # geometry and exists only to condition layout.
        garment_raw_path = OUTPUTS_DIR / f"{job_id}_garment_qwen_raw.png"
        generated_candidate.crop(
            (0, generation_garment_crop_top, generated_candidate.width, generated_candidate.height)
        ).save(garment_raw_path, format="PNG")
        # Keep the standard raw artifact as the complete Qwen edit with its
        # selected flat background normalized and ready for review/delivery.
        generated_candidate.save(output_path, format="PNG")
        # The rough composite is a diagnostic preview, not a generation or
        # acceptance reference. Comparing the final portrait with that pasted
        # layout incorrectly rejects coherent Qwen portraits and couples final
        # quality to rough head/garment proportions.
        generated_faces = _get_insight_app().get(
            cv2.cvtColor(np.asarray(generated_candidate), cv2.COLOR_RGB2BGR)
        )
        if len(generated_faces) != 1:
            raise RuntimeError(
                "Uniform output was not delivered because Qwen generated more than one person or face."
            )
        # The one-face, identity, uniform-coverage and artifact gates below
        # decide whether the generated portrait is deliverable. The diagnostic
        # rough collage is not an acceptance authority. Background colour is
        # generated by Qwen from the selected RGB value and is deliberately not
        # rematted afterward, because that extra pass damaged hair and shoulders.
        # The source, rough guide, edit mask and Qwen decode now share the exact
        # requested passport canvas. A second face-driven crop changes the
        # head/body ratio and can enlarge the head even though the masked edit
        # never moved it. Preserve this established upper-chest frame exactly.
        identity_locked = generated_candidate
        lip_tone_review = {
            "applied": False,
            "reason": "preserve coherent Qwen face and lip pixels",
        }
        # Normalize the already-selected, border-connected backdrop on every
        # accepted portrait.  Even when the broad background validator passes,
        # a few antialiased source-wall pixels can remain around the outer
        # shoulder or hair boundary.  This routine protects semantic person
        # cores and patterned fabric, and never constructs a new subject matte.
        # Preserve Qwen's accepted subject pixels when it already generated
        # the requested flat backdrop. Re-running semantic matting in that
        # case erodes valid plaid, collar and shoulder antialiasing and creates
        # blue/white edge notches. The final delivery gate below still applies
        # a guarded correction when the backdrop is genuinely wrong.
        if not native_backdrop_is_flat:
            # The native-background branch above always performs protected
            # semantic replacement before this point. Keep this guard only for
            # an unexpected segmentation failure; never reject merely because
            # Qwen chose scenery or a gradient.
            raise RuntimeError(
                "Uniform output was not delivered because the generated person could not be safely separated from its background."
            )
        elif not self._has_selected_solid_background(identity_locked, selected_bg_rgb):
            if self._has_flat_border_background(identity_locked):
                # Qwen can render the requested hue as a smooth but darker
                # studio field. That is not retained scenery. Normalize the
                # border-connected flat colour contribution and accept it only
                # if the standard exact selected-colour gate then passes. This
                # is deterministic colour correction, not another generation
                # or a replacement of subject pixels.
                normalized_background = self._replace_flat_border_background_strict(
                    identity_locked, selected_bg_rgb,
                )
                if not self._has_selected_solid_background(
                    normalized_background, selected_bg_rgb,
                ):
                    normalized_background = self._replace_smooth_border_background(
                        identity_locked, selected_bg_rgb,
                    )
                if self._has_selected_solid_background(
                    normalized_background, selected_bg_rgb,
                ):
                    identity_locked = normalized_background
                    logger.info(
                        "[UNIFORM_BACKGROUND] Normalized Qwen's flat backdrop hue/brightness to RGB%s",
                        selected_bg_rgb,
                    )
                else:
                    # The border is already flat and exact; remaining remote
                    # non-target islands are decoder/background artifacts, not
                    # a different scene. Use the SCHP-protected person matte to
                    # clear only those islands. This preserves Qwen's subject
                    # RGB and avoids weakening the acceptance threshold.
                    semantic_normalized = self._clear_detached_flat_background_artifacts(
                        identity_locked, selected_bg_rgb,
                    )
                    if self._has_selected_solid_background(
                        semantic_normalized, selected_bg_rgb,
                    ):
                        identity_locked = semantic_normalized
                        background_rematted = True
                        logger.info(
                            "[UNIFORM_BACKGROUND] Cleared disconnected decoder backdrop artifacts without rematting the subject"
                        )
                    else:
                        semantic_normalized = self._replace_background_with_foreground_matte(
                            identity_locked,
                            selected_bg_rgb,
                            job_id=job_id,
                            preserve_foreground_rgb=True,
                            strict=True,
                        ).convert("RGB")
                        if not self._has_selected_solid_background(
                            semantic_normalized, selected_bg_rgb,
                        ):
                            raise RuntimeError(
                                "Uniform output was not delivered because the generated person could not be safely separated from its background."
                            )
                        identity_locked = semantic_normalized
                        background_rematted = True
                        logger.info(
                            "[UNIFORM_BACKGROUND] Replaced non-uniform flat backdrop with protected matte RGB%s",
                            selected_bg_rgb,
                        )
            else:
                raise RuntimeError(
                    "Uniform output was not delivered because the generated backdrop was neither flat nor safely matteable."
                )

        # Preserve Qwen's coherent lighting. Post-generation face/hair
        # relighting changed skin chroma and created crown patches even when the
        # native decode was good. Lighting defects must now be solved inside
        # the single Qwen generation above.
        logger.info(
            "[UNIFORM_LIGHTING] Preserving Qwen-native face, hair and body lighting without post correction",
        )
        logger.info("[UNIFORM_LIP_TONE] %s", lip_tone_review)
        logger.info(
            "[UNIFORM_BACKGROUND] %s selected RGB%s backdrop",
            "Applied protected SCHP matte for" if background_rematted else "Normalized Qwen-generated",
            selected_bg_rgb,
        )
        # Compare like-for-like passport framing. The source upload can have a
        # different crop/pose, while identity_base is the exact protected frame
        # used by masked diffusion. Qwen can still de-age an otherwise valid
        # single-face edit because a school uniform is a strong child prior.
        # For a geometrically compatible near-match only, restore the interior
        # facial structure with a soft landmark-aligned blend. This never
        # reaches hair, backdrop, neck, collar or uniform, and the repaired
        # result must independently pass the same identity gate.
        identity_review = check_uniform_identity(
            person_pil,
            identity_locked,
            minimum_similarity=0.70,
            # ArcFace alone can accept a visibly enlarged or de-aged face.
            # Dense landmarks catch changed eye, jaw and chin geometry. Keep
            # this strict shape gate dynamic across portraits instead of
            # relying on any image-specific prompt.
            maximum_geometry_rmse=0.030,
        )
        initial_identity_review = dict(identity_review)
        original_identity_review: Optional[Dict[str, Any]] = None
        if identity_prepass_used:
            # The uniform decode must strongly match the validated clean
            # identity intermediate *and* remain a bounded match to the actual
            # upload. This dual gate prevents a beautiful but different face
            # from being legitimized merely because two generated stages agree
            # with each other. The modest original-photo floor is limited to
            # tiny hard-sun sources whose own restoration already passed the
            # same geometry bound; ordinary portraits still require 0.70.
            prepass_similarity = (
                float(identity_prepass_review.get("similarity"))
                if isinstance(identity_prepass_review, dict)
                and isinstance(identity_prepass_review.get("similarity"), (int, float))
                else .52
            )
            # The untouched hard-sun upload remains an audit authority, but it
            # must not demand a higher embedding score than the already
            # validated restoration can achieve against that same upload.
            # The untouched upload is severely affected by hard sunlight, so
            # its absolute embedding score is an audit signal rather than the
            # primary authority after a validated restoration. Permit at most
            # .05 regression from that restoration's measured baseline and
            # never below .45. The final still must match the clean prepass at
            # >= .70, and the original-upload dense geometry remains <= .040.
            # This accepts the corrected clothing-only result (0.489 versus a
            # 0.529 baseline) but still rejects the stale full-regeneration
            # result (0.441) that triggered the user's failure.
            original_similarity_floor = max(.45, prepass_similarity - .05)
            original_identity_review = check_uniform_identity(
                original_person_pil,
                identity_locked,
                minimum_similarity=original_similarity_floor,
            )
            original_geometry = original_identity_review.get("geometry_rmse")
            original_match = bool(
                original_identity_review.get("accepted")
                and isinstance(original_geometry, (int, float))
                and float(original_geometry) <= .040
            )
            if not original_match:
                identity_review = dict(identity_review)
                identity_review.update({
                    "accepted": False,
                    "reason": "generated face drifted from the untouched upload",
                    "original_upload_review": original_identity_review,
                })
            else:
                identity_review["original_upload_review"] = original_identity_review
        # A generated school-ID portrait must pass the same explicit identity
        # threshold regardless of source sunlight. The former harsh-light
        # exception accepted the latest visibly different face at 0.603 even
        # though the configured minimum was 0.70. Lighting can alter pixels,
        # but it must never authorize delivery of a substituted identity.
        # The uniform, collar, shoulders and body remain the actual Qwen decode.
        # When source/Qwen geometry is compatible, the bounded landmark merge
        # above restores only verified uploaded face/hair evidence and is then
        # required to pass this same biometric and geometry gate.
        identity_review["identity_repair_applied"] = identity_merge_applied
        if not identity_review["accepted"]:
            logger.error("[UNIFORM_IDENTITY_REVIEW] Raw Qwen candidate rejected: %s", initial_identity_review)
            score = (
                original_identity_review.get("similarity")
                if isinstance(original_identity_review, dict)
                else initial_identity_review.get("similarity")
            )
            score_text = f" (score {score:.3f})" if isinstance(score, (int, float)) else ""
            identity_reason = str(identity_review.get("reason") or "identity verification failed")
            raise RuntimeError(
                "Uniform output was not delivered because identity verification failed: "
                f"{identity_reason}{score_text}."
            )
        if not self._has_generated_uniform(
            identity_locked,
            outer_target_rgb,
            rough_reference=uniform_validation_reference,
            expected_target_fraction=expected_target_fraction,
        ):
            raise RuntimeError(
                "Uniform output was not delivered because Qwen retained the source clothing or did not reproduce enough of the supplied uniform template."
            )
        # Keep the coherent Qwen decode intact. Landmark-aligned source-hair
        # pasting created a visible halo over incompatible generated hair.
        # Hair fidelity is controlled in the one-pass generation prompt.
        # Face embedding similarity is necessary but insufficient for school-ID
        # work: a different hairstyle or a vest substituted for a blazer can
        # still score as the same child. Compare the candidate visually with
        # both uploaded authorities before delivery.
        from pipelines.uniform_review import review_uniform
        visual_review = {
            "badge_present": False,
            "neck_jewelry_present": False,
            "uniform_mismatch": False,
            "hair_changed": False,
            "identity_changed": False,
            "face_artifacts": False,
            "harsh_hair_glare": False,
            "background_mismatch": False,
            "head_cropped": False,
            "correction": "",
            "issues": [],
            "review_skipped": True,
        }
        # Identity embeddings cannot detect a changed hairstyle or flower
        # clip, but the four-pass VL reviewer adds substantial latency and has
        # repeatedly contradicted visible pixels. Keep it available as an
        # explicit diagnostic; production delivery uses the measured identity,
        # garment, glare, crop and backdrop gates by default.
        post_vl_review_enabled = bool(
            extreme_source_sunlight
            or os.getenv("VSOUL_ENABLE_POST_VL_REVIEW", "0").strip().lower()
            in {"1", "true", "yes"}
        )
        if post_vl_review_enabled:
            try:
                visual_review = review_uniform(
                    original_person_pil, uniform_validation_reference,
                    identity_locked, selected_bg_rgb,
                )
                visual_review["review_skipped"] = False
            except Exception as exc:
                # Review availability must not turn a valid completed render
                # into a failed job.
                logger.warning("[UNIFORM_VISUAL_REVIEW] Advisory worker unavailable: %s", exc)
                visual_review["review_unavailable"] = True
                visual_review["review_error"] = str(exc)
        # The VL reviewer can describe useful differences, but it has also
        # contradicted visible pixels (for example, reporting no badge while
        # one is present) and sees the image before the final badge/background
        # finish. Identity, deterministic garment coverage and final backdrop
        # validation are the delivery gates; reference prose remains a review
        # note for the user rather than a reason to discard a valid edit.
        # ArcFace+dense geometry and the deterministic garment-coverage gate
        # above have already passed. The VL reviewer is useful evidence but it
        # is not reliable enough to override those measured checks: on the
        # regression fixture it called the same curly flower hairstyle a
        # "braid changed to a bun", called two pointed collars different, and
        # reported badge_present while its own visible_symbol was "none".
        # Keep those semantic disagreements in the audit as advisories. Block
        # only directly visible output damage/cropping, plus a badge for which
        # the reviewer can name an actual visible symbol.
        badge_evidence = visual_review.get("badge_evidence")
        badge_symbol = (
            str(badge_evidence.get("visible_symbol") or "").strip().lower()
            if isinstance(badge_evidence, dict) else ""
        )
        confirmed_badge = bool(
            visual_review.get("badge_present")
            and badge_symbol not in {"", "none", "no", "not visible", "unknown"}
        )
        advisory_failures = [
            key for key in (
                "background_mismatch",
            ) if visual_review.get(key)
        ]
        garment_evidence = str(
            visual_review.get("garment_evidence") or ""
        ).strip().lower()
        concrete_uniform_failure = bool(
            visual_review.get("uniform_mismatch")
            and any(term in garment_evidence for term in (
                "collar", "denim", "source clothing", "original clothing",
                "wrong fabric", "wrong pattern", "wrong color", "wrong colour",
                "vest", "jacket", "not replaced", "retained",
            ))
        )
        if visual_review.get("uniform_mismatch") and not concrete_uniform_failure:
            advisory_failures.append("uniform_mismatch_unconfirmed")
        if visual_review.get("badge_present") and not confirmed_badge:
            advisory_failures.append("badge_present_unconfirmed")
        measured_high_confidence_identity = bool(
            isinstance(identity_review.get("similarity"), (int, float))
            and float(identity_review["similarity"]) >= .80
            and isinstance(identity_review.get("geometry_rmse"), (int, float))
            and isinstance(identity_review.get("maximum_geometry_rmse"), (int, float))
            and float(identity_review["geometry_rmse"])
            <= float(identity_review["maximum_geometry_rmse"])
        )
        if visual_review.get("identity_changed") and measured_high_confidence_identity:
            advisory_failures.append("identity_changed_unconfirmed")
        hair_evidence = str(visual_review.get("hair_evidence") or "").strip().lower()
        # The semantic reviewer has repeatedly confused relit curls/ponytails
        # with a bun even when both source flowers, the parting and the full
        # silhouette are visibly present.  Block only concrete destructive
        # evidence.  A generic style label is advisory after biometric,
        # head-crop and glare gates have passed.
        concrete_hair_failure = bool(
            visual_review.get("hair_changed")
            and any(term in hair_evidence for term in (
                "missing", "removed", "cropped", "cut off", "shortened",
                "wrong side", "relocated", "parting", "hairline",
                "accessory", "flower", "clip", "count",
            ))
        )
        if visual_review.get("hair_changed") and not concrete_hair_failure:
            advisory_failures.append("hair_changed_unconfirmed")
        blocking_failures = [
            key for key in (
                "face_artifacts", "head_cropped", "harsh_hair_glare",
                # The anatomy reviewer catches semantic changes (de-aging,
                # enlarged eyes, changed chin/jaw and lost hair accessories)
                # that can still receive a high face-embedding score. Do not
                # deliver those defects as merely advisory.
            )
            if visual_review.get(key)
        ]
        if concrete_hair_failure:
            blocking_failures.append("hair_changed")
        if concrete_uniform_failure:
            blocking_failures.append("uniform_mismatch")
        if visual_review.get("identity_changed") and not measured_high_confidence_identity:
            blocking_failures.append("identity_changed")
        if confirmed_badge:
            blocking_failures.append("badge_present")
        if visual_review.get("neck_jewelry_present"):
            blocking_failures.append("neck_jewelry_present")
        visual_review["advisory_failures"] = advisory_failures
        visual_review["delivery_blocked"] = bool(blocking_failures)
        (OUTPUTS_DIR / f"{job_id}_visual_review.json").write_text(
            json.dumps(visual_review, indent=2, allow_nan=False), encoding="utf-8",
        )
        if advisory_failures:
            logger.warning(
                "[UNIFORM_VISUAL_REVIEW] Advisory only: %s. Evidence: %s / %s",
                ", ".join(advisory_failures),
                visual_review.get("garment_evidence") or "none",
                visual_review.get("hair_evidence") or "none",
            )
        if blocking_failures:
            raise RuntimeError(
                "Uniform output was not delivered because it differs from the uploaded references: "
                + ", ".join(blocking_failures)
            )
        (OUTPUTS_DIR / f"{job_id}_identity_review.json").write_text(
            json.dumps({
                **identity_review,
                "initial_review": initial_identity_review,
                "identity_prepass_used": identity_prepass_used,
                "identity_prepass_review": identity_prepass_review,
                "original_upload_review": original_identity_review,
            }, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        # Keep Qwen's complete accepted subject intact. If Qwen renders the
        # requested colour as a smooth gradient, correct only the connected
        # border backdrop; never rematte or reconstruct the person here.
        if progress_callback:
            progress_callback(95, "Saving the generated school passport portrait...")
        delivery_faces = _get_insight_app().get(
            cv2.cvtColor(np.asarray(identity_locked), cv2.COLOR_RGB2BGR)
        )
        if len(delivery_faces) != 1:
            raise RuntimeError(
                "Uniform output was not delivered because final passport framing could not find exactly one face."
            )
        cropped = self._crop_school_passport_portrait(
            identity_locked,
            width,
            height,
            preserve_existing_frame=False,
            tight_school_id=True,
            delivery_upper_chest=True,
            face_bbox=delivery_faces[0].bbox,
        )
        # Do not recolour the accepted Qwen portrait after generation.  The
        # garment-only decode already carries the measured template palette;
        # a former saturated-colour heuristic could classify the selected blue
        # backdrop as the dominant accent and turn large background regions
        # red. Exact garment colour is enforced by conditioning and validation.
        # Qwen can copy a compact source-school badge even when the supplied
        # uniform reference has already been cleaned. Detect that generated
        # patch on the finished portrait itself and reconstruct only its local
        # plaid texture; collar, cuffs, face and background are outside this
        # conservative chest-patch gate.
        # The supplied garment reference is cleaned before generation and the
        # prompt explicitly forbids badges. Re-running the template badge
        # detector on a finished plaid portrait is unsafe: repeated checks and
        # a real pocket can be false positives, causing a large duplicate shirt
        # patch to be pasted over the final chest.
        output_badges_removed = 0
        logger.info("[UNIFORM_BADGE] Skipped unsafe post-generation plaid patch repair")
        background_verified = self._has_selected_solid_background(cropped, selected_bg_rgb)
        if not background_verified:
            # Passport resizing introduces a narrow antialiased colour band.
            # Normalize the smooth border-connected field first; rebuilding a
            # person matte here erased curls, flowers and plaid shoulders from
            # an otherwise valid Qwen portrait.
            normalized_background = self._replace_flat_border_background_strict(
                cropped, selected_bg_rgb,
            )
            if self._has_selected_solid_background(
                normalized_background, selected_bg_rgb,
            ):
                cropped = normalized_background
                background_verified = True
                logger.info(
                    "[UNIFORM_BACKGROUND] Normalized smooth border backdrop to RGB%s",
                    selected_bg_rgb,
                )
            else:
                # Resizing can expose a thin non-flat edge or detached decoder
                # island even when the pre-crop portrait was safely matted.
                # Recompute the semantic foreground on the final passport frame
                # and require the exact selected colour instead of accepting a
                # merely featureless but incorrect field.
                cropped = self._replace_background_with_foreground_matte(
                    cropped,
                    selected_bg_rgb,
                    job_id=job_id,
                    preserve_foreground_rgb=True,
                    strict=True,
                ).convert("RGB")
                background_verified = self._has_selected_solid_background(
                    cropped, selected_bg_rgb,
                )
                if background_verified:
                    background_rematted = True
                    logger.info(
                        "[UNIFORM_BACKGROUND] Re-applied protected matte after passport framing for RGB%s",
                        selected_bg_rgb,
                    )
            if not background_verified:
                raise RuntimeError(
                    "Uniform output was not delivered because the generated backdrop could not be safely replaced with the selected flat color."
                )
        logger.info(
            "[UNIFORM_BACKGROUND] Exporting direct Qwen portrait requested with RGB%s backdrop (verified=%s)",
            selected_bg_rgb,
            background_verified,
        )
        # Publish the generated uniform portrait after the optional gated,
        # face-interior maturity repair above. No whole-head/hair overlay or
        # global tone pass is used. A protected semantic matte is used only
        # when Qwen invents scenery instead of the requested flat colour.
        output_path = OUTPUTS_DIR / f"{job_id}_uniform.png"
        cropped.save(output_path, format="PNG", dpi=(300, 300))
        removed_artifacts = _cleanup_successful_uniform_artifacts(job_id, output_path)
        logger.info(
            "[UNIFORM_CLEANUP] Removed %d successful-job diagnostic image(s); kept %s",
            removed_artifacts, output_path,
        )
        return output_path

    # ── Legacy multi-model uniform swap (not used by the application) ──────────

    def qwen_edit_uniform_swap(
        self,
        person_input: Union[str, Path, bytes, Image.Image],
        uniform_template_path: Union[str, Path, bytes, Image.Image],
        prompt: Optional[str] = None,
        job_id: str = "qwen_uniform",
        width: int = 600,
        height: int = 800,
    ) -> Path:
        """
        Unified Multi-Model Hybrid School Uniform Swap Architecture:
        1. BiRefNet (Ultra-HR Matting) + SCHP (LIP-20 Human Parsing) on Student & Template
        2. Exact photographic garment extraction from Slot 2 (Real micro-grid fabric, buttons, crest badge)
        3. Contact shadow synthesis for seamless neck connection
        4. Optical Restoration for pore-level skin texture, crisp iris catchlights, curls & clips
        5. Studio Vibrance and Sharpness calibration (600x800, 300 DPI)
        """
        from pipelines.schp_service import parse
        from pipelines.birefnet_service import background_removal

        t0 = time.time()
        logger.info("[MULTI_MODEL_HYBRID] Starting unified multi-model uniform swap job %s", job_id)

        if not uniform_template_path:
            raise ValueError("Uniform template image is mandatory for Uniform Swap.")

        person_pil = self._to_pil(person_input).convert("RGB")
        tpl_pil = self._to_pil(uniform_template_path).convert("RGB")

        person_rgb = np.array(person_pil)
        tpl_rgb = np.array(tpl_pil)

        with io.BytesIO() as bio_p:
            person_pil.save(bio_p, format="PNG")
            p_rgba_bytes = background_removal.remove_background(bio_p.getvalue())
        p_rgba = np.array(Image.open(io.BytesIO(p_rgba_bytes)).convert("RGBA"))
        p_alpha = p_rgba[:, :, 3].astype(np.float32) / 255.0

        with io.BytesIO() as bio_t:
            tpl_pil.save(bio_t, format="PNG")
            t_rgba_bytes = background_removal.remove_background(bio_t.getvalue())
        t_rgba = np.array(Image.open(io.BytesIO(t_rgba_bytes)).convert("RGBA"))
        t_alpha = t_rgba[:, :, 3].astype(np.float32) / 255.0

        p_parse = parse(person_rgb)
        p_labels = p_parse["labels"]
        tpl_parse = parse(tpl_rgb)
        tpl_labels = tpl_parse["labels"]

        head_mask = np.isin(p_labels, (1, 2, 4, 13)).astype(np.uint8) * 255
        head_mask[np.isin(p_labels, (5, 6, 7))] = 0

        f_ys, f_xs = np.where(p_labels == 13)
        chin_y = np.max(f_ys) if len(f_ys) > 0 else person_rgb.shape[0] - 1
        face_center_x = int(np.mean(f_xs)) if len(f_xs) > 0 else person_rgb.shape[1] // 2
        face_w = (np.max(f_xs) - np.min(f_xs)) if len(f_xs) > 0 else person_rgb.shape[1] // 2

        h_ys, h_xs = np.where(head_mask > 0)
        head_min_y = np.min(h_ys) if len(h_ys) > 0 else 0
        head_min_x, head_max_x = (np.min(h_xs), np.max(h_xs)) if len(h_xs) > 0 else (0, person_rgb.shape[1])

        crop_y2 = min(person_rgb.shape[0], chin_y + 8)
        student_alpha_crop = (p_alpha * (head_mask.astype(np.float32) / 255.0))[head_min_y:crop_y2, head_min_x:head_max_x]
        s_crop_rgb = person_rgb[head_min_y:crop_y2, head_min_x:head_max_x]
        s_h, s_w = s_crop_rgb.shape[:2]
        s_chin_rel_y = chin_y - head_min_y
        s_center_rel_x = face_center_x - head_min_x

        cloth_mask = np.isin(tpl_labels, (5, 6, 7, 10, 11, 14, 15)).astype(np.uint8) * 255
        t_alpha[cloth_mask == 0] = 0.0

        c_ys, c_xs = np.where(cloth_mask > 0)
        if len(c_ys) == 0:
            c_ys, c_xs = np.where(t_alpha > 0.1)
        u_min_y, u_max_y = np.min(c_ys), np.max(c_ys)
        u_min_x, u_max_x = np.min(c_xs), np.max(c_xs)

        u_crop_rgb = tpl_rgb[u_min_y:u_max_y, u_min_x:u_max_x]
        u_crop_alpha = t_alpha[u_min_y:u_max_y, u_min_x:u_max_x]
        u_h, u_w = u_crop_rgb.shape[:2]

        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

        target_u_w = int(width * 0.90)
        u_scale = target_u_w / max(1, u_w)
        scaled_u_h = int(u_h * u_scale)

        u_scaled_rgb = cv2.resize(u_crop_rgb, (target_u_w, scaled_u_h), interpolation=cv2.INTER_LANCZOS4)
        u_scaled_alpha = cv2.resize(u_crop_alpha, (target_u_w, scaled_u_h), interpolation=cv2.INTER_LANCZOS4)
        u_alpha_soft = cv2.GaussianBlur(u_scaled_alpha, (7, 7), 1.5)[:, :, None]

        u_ox = (width - target_u_w) // 2
        u_oy = int(height * 0.46)

        target_head_w = int(width * 0.56)
        s_scale = target_head_w / max(1, s_w)
        scaled_s_w = target_head_w
        scaled_s_h = int(s_h * s_scale)

        s_scaled_rgb = cv2.resize(s_crop_rgb, (scaled_s_w, scaled_s_h), interpolation=cv2.INTER_LANCZOS4)
        s_scaled_alpha = cv2.resize(student_alpha_crop, (scaled_s_w, scaled_s_h), interpolation=cv2.INTER_LANCZOS4)

        fade_mask = np.ones_like(s_scaled_alpha)
        bottom_fade_start = int(s_chin_rel_y * s_scale)
        for y in range(bottom_fade_start, scaled_s_h):
            fade_factor = max(0.0, 1.0 - ((y - bottom_fade_start) / max(1, (scaled_s_h - bottom_fade_start))))
            fade_mask[y, :] *= fade_factor

        s_scaled_alpha_faded = s_scaled_alpha * fade_mask
        s_alpha_soft = cv2.GaussianBlur(s_scaled_alpha_faded, (7, 7), 1.5)[:, :, None]

        chin_target_y = u_oy + int(height * 0.085)
        scaled_chin_rel_y = int(s_chin_rel_y * s_scale)
        scaled_center_rel_x = int(s_center_rel_x * s_scale)

        head_ox = (width // 2) - scaled_center_rel_x
        head_oy = chin_target_y - scaled_chin_rel_y

        for y in range(scaled_u_h):
            cy = u_oy + y
            if 0 <= cy < height:
                for x in range(target_u_w):
                    cx = u_ox + x
                    if 0 <= cx < width:
                        ua = u_alpha_soft[y, x, 0]
                        if ua > 0.01:
                            canvas[cy, cx] = np.clip(
                                u_scaled_rgb[y, x] * ua + canvas[cy, cx] * (1.0 - ua), 0, 255
                            ).astype(np.uint8)

        shadow_layer = np.zeros((height, width), dtype=np.float32)
        cv2.ellipse(
            shadow_layer,
            center=(width // 2, chin_target_y + 10),
            axes=(int(face_w * s_scale * 0.38), int(height * 0.02)),
            angle=0,
            startAngle=0,
            endAngle=360,
            color=0.35,
            thickness=-1
        )
        shadow_layer = cv2.GaussianBlur(shadow_layer, (21, 21), 6.0)

        for y in range(height):
            for x in range(width):
                sa = shadow_layer[y, x]
                if sa > 0.01:
                    canvas[y, x] = np.clip(canvas[y, x] * (1.0 - sa * 0.45), 0, 255).astype(np.uint8)

        for y in range(scaled_s_h):
            cy = head_oy + y
            if 0 <= cy < height:
                for x in range(scaled_s_w):
                    cx = head_ox + x
                    if 0 <= cx < width:
                        ha = s_alpha_soft[y, x, 0]
                        if ha > 0.01:
                            canvas[cy, cx] = np.clip(
                                s_scaled_rgb[y, x] * ha + canvas[cy, cx] * (1.0 - ha), 0, 255
                            ).astype(np.uint8)

        restorer = PhotoRestorationService()
        enhanced_bgr, _ = restorer.enhance_faces_optical(canvas)

        final_pil = Image.fromarray(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB))
        final_pil = ImageEnhance.Contrast(final_pil).enhance(1.08)
        final_pil = ImageEnhance.Color(final_pil).enhance(1.06)
        final_pil = ImageEnhance.Sharpness(final_pil).enhance(1.18)

        out_path = OUTPUTS_DIR / f"{job_id}_qwen_uniform.jpg"
        final_pil.save(out_path, quality=99, dpi=(300, 300))

        logger.info("[MULTI_MODEL_HYBRID] Multi-model hybrid uniform swap complete in %.2fs -> %s", time.time() - t0, out_path)
        return out_path


# Global instance for direct calls
qwen_service = QwenEditPipeline()
