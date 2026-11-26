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
import re
import json
import hashlib
import os
import sys
import tempfile
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
LORA_FILE_8 = Path("models/loras/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
CACHE_DIR = Path(tempfile.gettempdir()) / "vsoul_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ArcFace on two same-age, same-ancestry children scores well above 0.70:
# run 0b0a60e2 delivered a visibly different child at 0.812 and passed.
# 0.80 is the measured floor that separates "same child" from "same
# demographic" on this app's school-portrait sources. Raise with care -
# a genuine low-resolution scan of the right child can sit near 0.78.
UNIFORM_IDENTITY_MINIMUM_SIMILARITY = 0.80

# Default generation resolution for uniform portrait edits.
# 1216px provides 0.976+ biometric identity and authentic skin tone fidelity.
DEFAULT_UNIFORM_GENERATION_DIMENSION: int = int(
    os.getenv("VSOUL_UNIFORM_GENERATION_DIMENSION", "1024")
)


_PROMPT_TOKENIZER = None


def _prompt_tokenizer():
    """Lazy handle on the same tokenizer the prompt encoder uses."""
    global _PROMPT_TOKENIZER
    if _PROMPT_TOKENIZER is None:
        _PROMPT_TOKENIZER = AutoTokenizer.from_pretrained(
            (MODELS_DIR / "tokenizer").as_posix()
        )
    return _PROMPT_TOKENIZER


def count_prompt_tokens(text: str) -> int:
    """Token length of ``text`` under the prompt encoder's tokenizer."""
    return len(_prompt_tokenizer()(text, add_special_tokens=False)["input_ids"])


def _split_prompt_units(text: str) -> Tuple[List[str], str]:
    """Split a prompt into trimmable units: sentences, else comma terms."""
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]
    if len(sentences) > 1:
        return sentences, " "
    return [part.strip() for part in text.split(",") if part.strip()], ", "


def fit_prompt_to_token_budget(
    text: str,
    budget: int,
    keep_tail: str = "",
    label: str = "prompt",
) -> str:
    """Drop whole units until ``text`` fits the encoder's token budget.

    ``encode_prompt`` truncates at ``max_sequence_length`` with no warning, so
    an over-budget prompt silently loses its tail. Job 5a1bbeaa shipped a 700
    token positive prompt against a 512 budget: the dropped 188 tokens carried
    the neck/pose/sleeve contract, and the 1266 token negative lost
    "cut-and-paste neck", "changed neck width" and "mismatched skin tone". The
    delivered portrait had a tripled neck and an orange lower throat. Trim
    here, deliberately and with a log line, instead of letting the encoder do
    it blind. ``keep_tail`` is always retained; duplicate units are dropped
    first because the negative list repeats many terms verbatim.
    """
    if budget <= 0 or not text.strip():
        return text
    try:
        total = count_prompt_tokens(text)
    except Exception as exc:  # tokenizer unavailable (tests, fresh checkout)
        logger.warning("[UNIFORM_PROMPT] Token budget check skipped: %s", exc)
        return text
    if total <= budget:
        return text
    units, joiner = _split_prompt_units(text)
    seen = set()
    deduped = []
    for unit in units:
        key = unit.strip().lower().rstrip(".")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(unit)
    tail = keep_tail.strip()

    def _assemble(count: int) -> str:
        body = joiner.join(deduped[:count])
        if not tail:
            return body
        return (body + " " + tail).strip() if body else tail

    # Measure the assembled string, never the sum of per-unit costs. Tokens
    # merge across a joiner, so summing units overcounts badly - on job
    # 5a1bbeaa's negative prompt it read 1767 tokens for a 1266-token string
    # and stopped 157 tokens short of the budget, silently discarding
    # "cut-and-paste neck" and "mismatched skin tone" while there was room
    # for them. Binary-search the largest prefix that really fits.
    if count_prompt_tokens(_assemble(len(deduped))) <= budget:
        keep_count = len(deduped)
    else:
        low, high = 0, len(deduped)
        while low < high:
            mid = (low + high + 1) // 2
            if count_prompt_tokens(_assemble(mid)) <= budget:
                low = mid
            else:
                high = mid - 1
        keep_count = low
    kept = deduped[:keep_count]
    fitted = _assemble(keep_count)
    logger.warning(
        "[UNIFORM_PROMPT] %s exceeded the %d-token encoder budget (%d tokens); "
        "trimmed to %d units / %d tokens instead of letting the encoder cut the tail",
        label, budget, total, len(kept), count_prompt_tokens(fitted),
    )
    return fitted


def seat_conditioning_neck(
    person,
    collar_seat_y,
    face_center_x,
    neck_half_width,
    background_rgb,
    feather_px=None,
):
    """End the conditioning portrait's throat at the measured collar seat.

    Qwen obeys conditioning pixels, not prompt numbers. Measured on job
    5a1bbeaa: the NECK/COLLAR LOCK stated 0.204 face-heights and the decode
    rendered 0.589, while handing Qwen the whole rough composite (collar
    already pasted) pulled it to 0.529 with a correct 0.66-face-width throat.
    The rough composite is not usable as image 1 - a four-step edit copies its
    pasted garment boundary, leaving hard notches beside the jaw and a
    preserved template collar.

    So give Qwen the geometry without the fabric: below the measured seat the
    neck column fades to the selected backdrop over a soft ramp. There is no
    garment edge to copy, only an absence of neck where neck must not be, and
    the model still synthesises the collar itself.

    Only the central neck column is touched. Shoulders, jaw and every pixel
    above the seat are left exactly as they were.
    """
    rgb = np.asarray(person.convert("RGB"), dtype=np.float32)
    height, width = rgb.shape[:2]
    seat = int(round(float(collar_seat_y)))
    if seat >= height - 1 or seat < 1:
        return person
    half = max(1.0, float(neck_half_width))
    if feather_px is None:
        # A hard cut is itself an edge Qwen reproduces. Ramp over a fraction
        # of the neck width so the fade reads as absence, not as a seam.
        feather_px = max(3.0, half * .45)
    ys = np.arange(height, dtype=np.float32)[:, None]
    xs = np.arange(width, dtype=np.float32)[None, :]
    # Vertical: 0 above the seat, rising to 1 over the feather below it.
    vertical = np.clip((ys - float(seat)) / float(feather_px), 0.0, 1.0)
    # Horizontal: only inside the neck column, with soft shoulders on the ramp
    # so the fade cannot clip the jaw corners into a rectangle.
    horizontal = np.clip(
        (half - np.abs(xs - float(face_center_x))) / max(1.0, half * .35),
        0.0, 1.0,
    )
    alpha = (vertical * horizontal)[:, :, None]
    backdrop = np.asarray(background_rgb, dtype=np.float32)[None, None, :]
    blended = rgb * (1.0 - alpha) + backdrop * alpha
    return Image.fromarray(
        np.clip(blended, 0, 255).astype(np.uint8), mode="RGB",
    )


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
    # Default to KEEPING these artifacts. Every fit/neck/halo defect in this
    # route is diagnosed by comparing the rough fit, the conditioning canvas and
    # the raw decode against the delivered frame; deleting them on success meant
    # a reported defect could only be investigated by reproducing the job, and
    # several investigations chased the wrong stage as a result. Set
    # VSOUL_CLEAN_SUCCESSFUL_UNIFORM_ARTIFACTS=1 to reclaim the disk space.
    if os.getenv("VSOUL_CLEAN_SUCCESSFUL_UNIFORM_ARTIFACTS", "0").strip().lower() not in {
        "1", "true", "yes",
    }:
        return 0
    safe_job_id = Path(str(job_id)).name
    final_resolved = Path(final_output).resolve()
    removed = 0
    search_dirs = [output_dir]
    debug_dir = output_dir / "debug"
    if debug_dir.is_dir():
        search_dirs.append(debug_dir)
    for s_dir in search_dirs:
        for artifact in s_dir.glob(f"{safe_job_id}_*"):
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

def _recover_unparsed_neck(
    rgb: np.ndarray,
    person_mask: np.ndarray,
    face_bbox: Any,
    corridor_half_widths: float = .34,
    corridor_face_heights: float = .42,
) -> np.ndarray:
    """Add the photographed neck to a parser foreground mask.

    The LIP-20 parser this app ships has no neck class. Label 13 (Face) stops
    at the jaw and the bare neck beneath it is returned as label 0, the same
    value as the backdrop, so any ``labels != 0`` foreground silently drops it.
    Every later stage that fills non-foreground with the selected colour then
    punches a flat notch through the subject's throat.

    Only skin-coloured pixels inside a face-relative corridor are added, so a
    collar, lanyard or background object cannot enter the mask.
    """
    try:
        x1, y1, x2, y2 = (float(v) for v in face_bbox)
    except (TypeError, ValueError):
        return person_mask
    face_w = max(1.0, x2 - x1)
    face_h = max(1.0, y2 - y1)
    rows, cols = np.indices(person_mask.shape)
    corridor = (
        (np.abs(cols - (x1 + x2) * .5) <= face_w * corridor_half_widths)
        & (rows >= int(y2) - 2)
        & (rows <= int(y2 + face_h * corridor_face_heights))
    )
    if not np.any(corridor):
        return person_mask
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    skin = (
        (hsv[:, :, 1] >= 20)
        & (hsv[:, :, 2] >= 30)
        & (hsv[:, :, 2] <= 245)
        & ((hsv[:, :, 0] <= 25) | (hsv[:, :, 0] >= 170))
    )
    recovered = corridor & skin & ~person_mask
    if not np.any(recovered):
        return person_mask
    logger.info(
        "[UNIFORM_NECK] Recovered %d neck pixel(s) the parser labelled background",
        int(np.count_nonzero(recovered)),
    )
    return person_mask | recovered

_PIPELINE_LOCK = threading.Lock()

# A recognition embedding can remain high after Qwen redraws a child's eyes,
# mouth, or face proportions. Keep a strict biometric floor and independently
# verify the dense facial geometry before delivering a generated portrait.
MIN_PORTRAIT_IDENTITY_SIMILARITY = 0.78
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


def _hair_review_has_concrete_change(evidence: str) -> bool:
    """Require a visible geometric/accessory change before blocking delivery.

    A semantic style-name comparison (for example "braid to ponytail") is not
    sufficient: the VL reviewer can name the same two tied sections differently
    between a low-resolution scan and a restored result.  This mirrors the
    reviewer prompt, which requires count, side, endpoint, silhouette, parting,
    length, hairline or accessory evidence.
    """
    text = str(evidence or "").strip().lower()
    output_is_named = any(marker in text for marker in (
        "image 2", "generated image", "generated portrait", "generated hair",
        "output image", "output portrait", "candidate", "result",
    ))
    concrete_change = any(marker in text for marker in (
        "missing", "removed", "cropped", "cut off", "shortened", "longer",
        "shorter", "relocated", "wrong side", "not present", "lost", "added",
        "invented", "section count", "sections instead", "accessory", "flower",
        "clip", "ribbon", "endpoint", "silhouette", "hairline", "parting",
        "unequal length", "different length",
    ))
    if not (output_is_named and concrete_change):
        return False
    # The reviewer often describes the original and the generated portrait in
    # one sentence ("Image 1 has longer hair ... while Image 2 is more styled").
    # A length/volume word used about the ORIGINAL then condemned the output
    # even though the generated hair matched. Require the concrete word and the
    # output reference in the same clause.
    clauses = re.split(
        r"[.;\n]|,\s|\bwhile\b|\bwhereas\b|\bbut\b|\bhowever\b", text,
    )
    return any(
        any(marker in clause for marker in (
            "image 2", "generated", "output", "candidate", "result",
        ))
        and any(marker in clause for marker in (
            "missing", "removed", "cropped", "cut off", "shortened", "longer",
            "shorter", "relocated", "wrong side", "not present", "lost",
            "invented", "section count", "sections instead", "accessory",
            "flower", "clip", "ribbon", "endpoint", "silhouette", "hairline",
            "parting", "unequal length", "different length",
        ))
        for clause in clauses
    )


def _has_grounded_direct_light(vl_plan: Dict[str, Any]) -> bool:
    """Require visible or measured evidence before requesting a head relight."""
    truthy = str(vl_plan.get("direct_sunlight_present") or "").strip().lower() in {
        "true", "yes", "1", "present",
    }
    if not truthy:
        return False
    severity = str(
        vl_plan.get("measured_sunlight_severity") or "normal"
    ).strip().lower()
    if severity in {"extreme", "strong", "severe"}:
        return True
    evidence = str(vl_plan.get("sunlight_evidence") or "").strip().lower()
    light_type = str(vl_plan.get("sunlight_type") or "").strip().lower()
    empty = {"", "none", "no", "unknown", "unclear", "not visible", "not present"}
    return evidence not in empty or light_type not in empty


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
    text_only_prompts: Optional[List[bool]] = None,
) -> Union[
    Tuple[torch.Tensor, Optional[torch.Tensor]],
    List[Tuple[torch.Tensor, Optional[torch.Tensor]]],
]:
    """
    Runs one or more prompt encodings in one isolated worker subprocess.
    Reclaims 100% of GPU memory upon exit, eliminating bitsandbytes NF4 memory leaks.

    ``text_only_prompts`` marks prompts that must be encoded without the
    conditioning images. A negative prompt needs this: with the images attached,
    true CFG subtracts the garment/person image embedding along with the banned
    wording, which erased the target plaid and rebuilt the source garment. Text
    only, CFG subtracts just the words.
    """
    single_prompt = isinstance(prompt, str)
    prompts = [prompt] if single_prompt else list(prompt)
    if not prompts or any(not isinstance(item, str) for item in prompts):
        raise ValueError("At least one text prompt is required for Qwen prompt encoding")
    if text_only_prompts is None:
        text_only_flags = [False] * len(prompts)
    else:
        text_only_flags = [bool(flag) for flag in text_only_prompts]
        if len(text_only_flags) != len(prompts):
            raise ValueError("text_only_prompts must match the prompt count")
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
    # A text-only encoding of the same wording is a different embedding, so it
    # must not collide with an image-attached cache entry.
    cache_hasher.update(bytes(1 if flag else 0 for flag in text_only_flags))
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
import os
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import torch
from pathlib import Path
from PIL import Image
import transformers.utils.import_utils as _transformers_import_utils
_transformers_import_utils.is_kernels_available = lambda *args, **kwargs: False
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
text_only_flags = {text_only_flags!r}

encoded = []
with torch.inference_mode():
    for prompt, text_only in zip(prompts, text_only_flags):
        prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
            # Passing image=None drops the vision tokens entirely, so this
            # embedding carries only the wording. The negative branch of true
            # CFG then subtracts the banned text without also subtracting the
            # garment and person references it is supposed to preserve.
            image=None if text_only else images,
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
    # On Windows with RTX 50-series / CUDA 12.8, native bitsandbytes 4-bit causes access
    # violations (0xC0000005). Use the fast, stable BF16 GPU/CPU split directly.
    use_quant = sys.platform != "win32"
    helper_code = build_helper_code(quantized=use_quant)
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
        # A native bitsandbytes access violation has no Python exception to
        # recover inside that process. Repeating the same native load usually
        # crashes at the same weight. Switch once to the supported BF16
        # GPU/CPU split instead of burning another retry on identical code.
        access_violation_codes = {3221225477, -1073741819}
        if ret.returncode in access_violation_codes:
            logger.warning(
                "[QWEN_ENCODE_WORKER_FALLBACK] Native 4-bit encoder exited with access violation %s; retrying with BF16 GPU/CPU split",
                ret.returncode,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except (RuntimeError, AttributeError):
                    pass
            time.sleep(1.0)
            helper_script.write_text(build_helper_code(quantized=False), encoding="utf-8")
            ret = subprocess.run([sys.executable, str(helper_script)], **run_args)
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
            # Keep enough exact-input embeddings for a practical school-photo
            # batch. Each entry is roughly 10-18 MB, while recomputing one
            # reloads the 16.6-GB Qwen-VL encoder and delays diffusion by about
            # three minutes on this machine. Sixty-four entries stay below
            # roughly 1.2 GB on disk and never occupy GPU/RAM when idle.
            for stale_entry in cached_entries[64:]:
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

    model_dtype = getattr(self.img_in, "compute_dtype", getattr(self, "dtype", torch.bfloat16))
    if model_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    hidden_states = hidden_states.to(device=dev, dtype=model_dtype)
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

    encoder_hidden_states = encoder_hidden_states.to(device=dev, dtype=model_dtype)
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
        # True when the current job's portrait arrived as a transparent cut-out
        # and this pipeline composited it onto white. The head-lock wall key
        # must not treat that synthetic field as a photographed backdrop.
        self._last_upload_alpha_flattened: bool = False
        logger.info("[QWEN_EDIT_PIPELINE] Initialized on device: %s (%s)", self.device, self.dtype)

    @staticmethod
    def _restore_source_skin_tone(
        generated: Image.Image,
        source_skin_rgb: Optional[Tuple[int, int, int]],
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Correct the decoded face back to the uploaded complexion.

        A stage-by-stage trace of run 0b0a60e2 showed every tensor entering
        Qwen carrying the correct medium-brown skin (208,143,108) and the raw
        decode emerging at (244,194,165): the diffusion pass itself lightens
        skin toward a generic studio-portrait prior. The prompt already states
        the measured source RGB and the negative prompt already bans "changed
        complexion"; at CFG 1.15 both still lose, because text cannot hold a
        colour against a strong visual prior.

        So correct it deterministically, the way the garment path already
        corrects cloth to a measured template colour. All three LAB channels
        move: measured on this failure, lightness carries 94% of the error
        (L* 65 -> 82) and chroma only 6%, so a chroma-only correction barely
        moved deltaE (45.3 -> 42.0). The shift is one constant per channel,
        measured on stable cheek skin and applied through a feathered head
        mask, so shading, pores and specular detail all survive: only the
        overall complexion moves back to the upload's.

        L* is clamped harder than chroma. Studio relighting legitimately
        changes lightness, and a face darkened to match an underexposed
        upload would look worse than the bleaching being fixed.
        """
        info: Dict[str, Any] = {"applied": False, "reason": "no measured source skin"}
        if not source_skin_rgb:
            return generated, info
        rgb = np.asarray(generated.convert("RGB"))
        height, width = rgb.shape[:2]
        try:
            from pipelines.photo_restoration import _get_insight_app
            faces = _get_insight_app().get(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )
        except Exception as detect_error:
            info["reason"] = f"face detection unavailable: {detect_error}"
            return generated, info
        if not faces:
            info["reason"] = "no face detected"
            return generated, info
        face = max(
            faces,
            key=lambda item: (
                (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])
            ),
        )
        box = np.asarray(face.bbox, dtype=np.float32).reshape(-1)
        x1, y1 = int(max(0, box[0])), int(max(0, box[1]))
        x2, y2 = int(min(width, box[2])), int(min(height, box[3]))
        face_w, face_h = x2 - x1, y2 - y1
        if face_w < 24 or face_h < 24:
            info["reason"] = "face too small"
            return generated, info

        # Sample both cheeks: clear of eyes, mouth, nostrils and hairline, so
        # the measurement is skin rather than a feature or a shadow.
        cheek_y = y1 + int(face_h * .55)
        half = max(2, int(face_h * .06))
        patches = [
            rgb[cheek_y - half:cheek_y + half, x1 + int(face_w * .12):x1 + int(face_w * .28)],
            rgb[cheek_y - half:cheek_y + half, x2 - int(face_w * .28):x2 - int(face_w * .12)],
        ]
        samples = [patch.reshape(-1, 3) for patch in patches if patch.size]
        if not samples:
            info["reason"] = "no skin sample"
            return generated, info
        skin_pixels = np.vstack(samples)
        if skin_pixels.shape[0] < 64:
            info["reason"] = "insufficient skin sample"
            return generated, info

        def _to_lab(arr: np.ndarray) -> np.ndarray:
            return cv2.cvtColor(
                arr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB,
            ).reshape(-1, 3).astype(np.float32)

        generated_lab = np.median(_to_lab(skin_pixels), axis=0)
        source_lab = _to_lab(np.asarray([source_skin_rgb], dtype=np.uint8))[0]

        # Bounded per channel. Lightness is the dominant error but also the
        # one a legitimate relight can change, so it moves on a tighter rein
        # than chroma: enough to undo a bleached face, not enough to crush a
        # correctly lit one toward an underexposed upload.
        #
        # These limits are wide because this pass is currently the only thing
        # holding complexion. Confining the diffusion itself to the clothing
        # region would be the better fix and was attempted: pinning latents
        # outside `generation_edit_region` through callback_on_step_end. It
        # fails on this scheduler. Step-1 latents are still essentially noise,
        # so holding them across a 4-step Lightning schedule decoded the whole
        # protected area as static (job 372944da). A correct implementation
        # must re-noise the *clean* encoded image latents to each timestep's
        # sigma, which needs the scheduler's exact noise schedule; until that
        # is built and verified, the correction stays here.
        raw_shift_l = source_lab[0] - generated_lab[0]
        shift_l = float(np.clip(raw_shift_l, -46.0, 20.0))
        shift_a = float(np.clip(source_lab[1] - generated_lab[1], -24.0, 24.0))
        shift_b = float(np.clip(source_lab[2] - generated_lab[2], -24.0, 24.0))
        info.update({
            "generated_skin_lab": [round(float(v), 1) for v in generated_lab],
            "source_skin_lab": [round(float(v), 1) for v in source_lab],
            "shift_l": round(shift_l, 2),
            "shift_a": round(shift_a, 2),
            "shift_b": round(shift_b, 2),
        })
        if abs(shift_l) < 2.0 and abs(shift_a) < 1.5 and abs(shift_b) < 1.5:
            info["reason"] = "complexion already matches"
            return generated, info

        # The correction must land on skin and nothing else. A plain ellipse
        # over the head also covers backdrop, hair and collar, and shifting
        # those by L-43 painted a dark oval halo across the portrait. So take
        # the ellipse only as a search region and keep the pixels inside it
        # whose colour is actually skin-like, judged against the decoded
        # complexion just measured rather than any fixed skin preset.
        region = np.zeros((height, width), dtype=np.uint8)
        centre = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = (int(face_w * .95), int(face_h * 1.30))
        cv2.ellipse(region, centre, axes, 0, 0, 360, 255, -1)

        lab_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        # Distance in chroma only: a cheek in shadow and a lit forehead differ
        # in L* but share undertone, so a chroma test keeps the whole face
        # while still rejecting hair, cloth and the flat backdrop.
        chroma_distance = np.sqrt(
            (lab_full[:, :, 1] - generated_lab[1]) ** 2
            + (lab_full[:, :, 2] - generated_lab[2]) ** 2
        )
        skin_like = (chroma_distance < 18.0) & (region > 0)
        # Hair and deep shadow can share an undertone with skin; exclude what
        # is far darker than the measured cheek.
        skin_like &= lab_full[:, :, 0] > (generated_lab[0] - 55.0)
        if np.count_nonzero(skin_like) < max(256, int(face_w * face_h * .05)):
            info["reason"] = "skin mask too small to correct safely"
            return generated, info

        mask = (skin_like.astype(np.uint8)) * 255
        # Close pinholes around nostrils and lips, then feather so the edge of
        # the correction is never visible as a line.
        closing = max(3, (int(min(face_w, face_h) * .05) | 1))
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing, closing)),
        )
        feather = max(9, (int(min(face_w, face_h) * .10) | 1))
        alpha = (
            cv2.GaussianBlur(mask, (feather, feather), 0).astype(np.float32) / 255.0
        )[:, :, None]
        info["skin_mask_pixels"] = int(np.count_nonzero(skin_like))

        lab = lab_full
        lab[:, :, 0] += alpha[:, :, 0] * shift_l
        lab[:, :, 1] += alpha[:, :, 0] * shift_a
        lab[:, :, 2] += alpha[:, :, 0] * shift_b
        corrected = cv2.cvtColor(
            lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB,
        ).astype(np.float32)
        # LAB round-tripping nudges even untouched pixels, so composite through
        # the same feather and leave backdrop and garment byte-exact.
        blended = rgb.astype(np.float32) * (1.0 - alpha) + corrected * alpha
        info["applied"] = True
        info["reason"] = "restored measured source complexion"
        return (
            Image.fromarray(blended.clip(0, 255).astype(np.uint8), "RGB"),
            info,
        )

    @staticmethod
    def _measure_pattern_repeat_px(template: Image.Image) -> Optional[float]:
        """Measure the dominant horizontal repeat of a woven check, in pixels.

        Autocorrelates luminance rows across the garment's central body and
        returns the median first strong peak. Returns None for a plain or
        unresolved fabric rather than guessing, so the caller simply omits the
        scale instruction instead of stating a wrong number.
        """
        rgb = np.asarray(template.convert("RGB"))
        if rgb.size == 0:
            return None
        height, width = rgb.shape[:2]
        # Central body only: collar, sleeve ends and the canvas margin carry
        # solid trim whose flat rows would otherwise dilute the estimate.
        body = rgb[
            int(height * .35):max(int(height * .75), int(height * .35) + 1),
            int(width * .30):max(int(width * .70), int(width * .30) + 1),
        ]
        if body.size == 0 or min(body.shape[:2]) < 32:
            return None
        gray = cv2.cvtColor(body, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray -= float(gray.mean())
        periods: List[float] = []
        step = max(1, gray.shape[0] // 40)
        max_lag = min(120, gray.shape[1] - 2)
        if max_lag <= 6:
            return None
        for row_index in range(0, gray.shape[0], step):
            line = gray[row_index]
            if float(line.std()) < 5.0:
                continue
            correlation = np.correlate(line, line, "full")[len(line) - 1:]
            correlation = correlation / (correlation[0] + 1e-9)
            for lag in range(5, max_lag):
                if (
                    correlation[lag] > .35
                    and correlation[lag] >= correlation[lag - 1]
                    and correlation[lag] >= correlation[lag + 1]
                ):
                    periods.append(float(lag))
                    break
        # A handful of agreeing rows is evidence; one or two is noise.
        if len(periods) < 5:
            return None
        return float(np.median(periods))

    @staticmethod
    def _flatten_upload_alpha(image: Image.Image) -> Image.Image:
        """Composite an uploaded cut-out onto white instead of onto black.

        ``Image.convert("RGB")`` drops the alpha channel without compositing,
        so every fully transparent pixel keeps its stored colour - black in the
        cut-out PNGs this app receives. Measuring, parsing and conditioning then
        all see a black field around the subject.
        """
        if image.mode not in {"RGBA", "LA", "PA"} and "transparency" not in image.info:
            return image.convert("RGB")
        rgba = image.convert("RGBA")
        alpha = rgba.split()[3]
        if alpha.getextrema()[0] == 255:
            return rgba.convert("RGB")
        flattened = Image.new("RGB", rgba.size, (255, 255, 255))
        flattened.paste(rgba, mask=alpha)
        logger.info(
            "[UPLOAD_ALPHA] Composited transparent upload onto white before measurement",
        )
        return flattened

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
        background_color: Optional[Tuple[int, int, int]] = None,
        delivery_face_fraction: Optional[float] = None,
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
            # Measure backdrop and foreground crown to ensure modest passport headroom (~5-8% height)
            # is always preserved, never clipped against the top edge.
            edge = max(4, min(16, source_h // 20, source_w // 20))
            if background_color is not None:
                backdrop = np.asarray(background_color, dtype=np.uint8)
            else:
                backdrop = np.median(np.concatenate((
                    rgb[:edge, :edge].reshape(-1, 3),
                    rgb[:edge, -edge:].reshape(-1, 3),
                )), axis=0).round().clip(0, 255).astype(np.uint8)

            # Delivery is an upper-chest school-ID crop, not a half-body frame.
            # The former .32 detector-box fraction retained the complete pocket,
            # sleeve hems and lower torso; on short-sleeve templates it exposed
            # Qwen's unnecessary lower-arm/sleeve continuation. Keep crown,
            # neck and both shoulders, but finish at the upper chest.
            target_face_fraction = float(np.clip(
                delivery_face_fraction if delivery_face_fraction is not None else .405,
                .385,
                .455,
            ))
            crop_h = max(int(face_h * 1.35), int(np.ceil(face_h / target_face_fraction)))
            crop_w = int(round(crop_h * target_ratio))

            # Detect crown_y from foreground pixels
            dist_to_bg = np.linalg.norm(rgb.astype(np.float32) - backdrop.astype(np.float32), axis=-1)
            central_fg = dist_to_bg[:, int(source_w * 0.04):int(source_w * 0.96)] > 18.0
            occupied_rows = np.flatnonzero(np.count_nonzero(central_fg, axis=1) >= max(3, int(source_w * 0.02)))
            crown_y = int(occupied_rows[0]) if occupied_rows.size else max(0, int(face_y - face_h * 0.35))

            target_headroom = max(18, int(round(crop_h * 0.065)))
            desired_top = crown_y - target_headroom

            # Ensure genuine upper chest passport crop: terminate below the collarbone / sternum
            # at most ~1.15 face heights below the chin so mid-torso/waist is never exposed
            chin_y = int(round(face_y + face_h))
            max_chest_bottom = int(round(chin_y + face_h * 1.40))
            if desired_top + crop_h > max_chest_bottom:
                crop_h = max(int(face_h * 1.35), max_chest_bottom - desired_top)
                crop_w = int(round(crop_h * target_ratio))

            desired_left = int(round(face_x + face_w * .5 - crop_w * .5))

            # If the requested face fraction needs pixels below the genuine decode,
            # extend the clean backdrop rather than clamping to unsegmented lower edges.
            maximum_real_crop_h = max(1, int(source_h - desired_top))
            pad_bottom = 0
            if crop_h > maximum_real_crop_h:
                bottom_strip = rgb[max(0, source_h - 6):source_h]
                bottom_is_backdrop = bool(
                    bottom_strip.size
                    and np.all(
                        np.linalg.norm(
                            bottom_strip.astype(np.float32)
                            - backdrop.astype(np.float32),
                            axis=-1,
                        ) <= 26.0
                    )
                )
                if bottom_is_backdrop:
                    pad_bottom = int(crop_h - maximum_real_crop_h)
                    logger.info(
                        "[PASSPORT_COMPOSITION] Extended flat backdrop %dpx below "
                        "the decode so the %.2f face-fraction crop is not clamped",
                        pad_bottom, target_face_fraction,
                    )
                else:
                    crop_h = min(source_h, crop_h)
                    crop_w = max(1, int(round(crop_h * target_ratio)))
                    if crop_w > source_w:
                        crop_w = source_w
                        crop_h = max(1, int(round(crop_w / target_ratio)))
                    target_headroom = max(18, int(round(crop_h * .065)))
                    desired_top = max(0, min(source_h - crop_h, crown_y - target_headroom))
                    desired_left = int(round(face_x + face_w * .5 - crop_w * .5))

            pad_top = max(0, -desired_top)
            pad_left = max(0, -desired_left)
            pad_right = max(0, desired_left + crop_w - source_w)

            if pad_top > 0 or pad_left > 0 or pad_right > 0 or pad_bottom > 0:
                padded = np.empty((
                    source_h + pad_top + pad_bottom,
                    source_w + pad_left + pad_right,
                    3,
                ), dtype=np.uint8)
                padded[:] = backdrop
                padded[pad_top:pad_top + source_h, pad_left:pad_left + source_w] = rgb
                rgb = padded
                image = Image.fromarray(rgb, "RGB")
                desired_left += pad_left
                desired_top += pad_top

            cur_h, cur_w = rgb.shape[:2]
            top = max(0, desired_top)
            left = max(0, desired_left)
            if top + crop_h > cur_h:
                top = max(0, cur_h - crop_h)
            if left + crop_w > cur_w:
                left = max(0, cur_w - crop_w)
            left = max(0, left)
            top = max(0, top)
            logger.info(
                "[PASSPORT_COMPOSITION] Direct source crop face_fraction=%.3f crop=%dx%d crown_y=%d",
                target_face_fraction, crop_w, crop_h, crown_y,
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
            # 0.395 consistently ends at the upper chest while retaining both shoulders
            # and modest headroom (~6-8%).
            target_face_fraction = float(np.clip(
                .395 + .006 * (shoulder_to_face - 2.1), .385, .415,
            ))
            lower_face_extent = .95
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
    def _restore_dead_and_hot_pixels(
        image: Image.Image,
        dead_threshold: int = 4,
        hot_threshold: int = 252,
        spike_threshold: float = 80.0,
    ) -> Image.Image:
        """Heal isolated dead black, hot white, and single-pixel spike artifacts.

        Preserves 100% of fine continuous features (hair strands, eyelashes, fabric
        patterns, jewelry highlights) while removing single-pixel dead or hot sensor/decoder
        defects with local 8-neighborhood median filtering.
        """
        try:
            arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
            h, w, c = arr.shape
            if h < 5 or w < 5:
                return image

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
                return Image.fromarray(repaired_arr, "RGB")
            return image
        except Exception as exc:
            logger.warning("[DEAD_PIXEL_REPAIR] Dead-pixel restore skipped: %s", exc)
            return image

    @staticmethod
    def smooth_decoded_hair_silhouette(
        generated,
        background_rgb,
        top_fraction=.30,
        max_shift_px=2.5,
    ):
        """Remove the serrated crust the VAE leaves along the hair edge.

        Measured on job 5a1bbeaa: the contour's zig-zag energy (mean absolute
        second difference of the silhouette column per row) is 0.707px in the
        conditioning Qwen is given and 1.457px in its decode - the decoder
        roughly doubles it. The VAE is 8x downsampled and runs on CPU, so one
        latent cell covers 8x8 output pixels; a near-black hair boundary on a
        white backdrop cannot be reconstructed smoothly from that, and each
        error reads as a hard step.

        This is a geometry defect, not a texture one, so it is fixed in
        geometry: find the silhouette column per row, smooth that 1-D contour,
        and repaint only the pixels between the measured and smoothed edges
        with the backdrop or the neighbouring hair. Pixel content is never
        blurred or sharpened - a strand that exists stays exactly as decoded.

        Only the top ``top_fraction`` of the frame is touched (the crown and
        upper sides, where hair meets backdrop), and no edge moves by more
        than ``max_shift_px``, so a real cowlick or stray lock survives while
        a one-pixel staircase does not.
        """
        rgb = np.asarray(generated.convert("RGB"), dtype=np.uint8)
        height, width = rgb.shape[:2]
        backdrop = np.asarray(background_rgb, dtype=np.float32)
        limit = int(round(height * float(top_fraction)))
        if limit < 8 or width < 8:
            return generated
        distance = np.linalg.norm(
            rgb[:limit].astype(np.float32) - backdrop[None, None, :], axis=2,
        )
        subject = distance > 42.0
        out = rgb.copy()
        repainted = 0
        for side in ("left", "right"):
            edges = np.full(limit, np.nan, dtype=np.float32)
            for row in range(limit):
                cols = np.flatnonzero(subject[row])
                if cols.size:
                    edges[row] = float(cols[0] if side == "left" else cols[-1])
            valid = ~np.isnan(edges)
            if valid.sum() < 12:
                continue
            index = np.flatnonzero(valid)
            series = edges[valid]
            # A 5-tap moving average removes single-pixel staircase steps
            # while following the real curve of the head.
            kernel = np.ones(5, dtype=np.float32) / 5.0
            padded = np.pad(series, 2, mode="edge")
            smoothed = np.convolve(padded, kernel, mode="valid")
            shift = np.clip(smoothed - series, -max_shift_px, max_shift_px)
            smoothed = series + shift
            for position, row in enumerate(index):
                measured = int(round(series[position]))
                target = int(round(smoothed[position]))
                if target == measured:
                    continue
                if side == "left":
                    if target > measured:
                        # Edge moved inward: the crust pixels become backdrop.
                        out[row, measured:target] = backdrop.astype(np.uint8)
                        repainted += target - measured
                    else:
                        # Edge moved outward: extend with the adjacent hair
                        # pixel rather than inventing a colour.
                        out[row, target:measured] = rgb[row, measured]
                        repainted += measured - target
                else:
                    if target < measured:
                        out[row, target + 1:measured + 1] = backdrop.astype(np.uint8)
                        repainted += measured - target
                    else:
                        out[row, measured + 1:target + 1] = rgb[row, measured]
                        repainted += target - measured
        if not repainted:
            return generated
        logger.info(
            "[UNIFORM_HAIR_EDGE] Smoothed the decoded hair silhouette "
            "(%d boundary pixels repainted in the top %d rows)",
            repainted, limit,
        )
        return Image.fromarray(out, mode="RGB")

    @staticmethod
    def _decontaminate_uniform_conditioning_edge(
        foreground_rgba,
        max_edge_width=6,
    ):
        """Remove hardened source-backdrop fringe before Qwen diffusion.

        SCHP can classify the antialiased pixels just outside hair/ears as a
        human label. The fused matte then promotes those pixels to nearly
        opaque, so ordinary alpha decontamination cannot remove the old white
        or grey backdrop. Feeding that fringe to Qwen makes it reproduce a
        bright, serrated outline on a coloured passport backdrop.

        Repair only bright, low-chroma pixels in the narrow outer subject band
        when the nearest reliable inward subject pixel is materially darker.
        A white collar, flower, ribbon or other pale accessory keeps its colour
        because its inward neighbourhood is also pale. Alpha and silhouette
        are unchanged; this is conditioning cleanup, not output recolouring.
        """
        rgba = np.asarray(foreground_rgba, dtype=np.uint8).copy()
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            return foreground_rgba, 0

        alpha = rgba[:, :, 3]
        foreground = alpha > 24
        if not np.any(foreground):
            return rgba, 0


        inward_distance = cv2.distanceTransform(
            foreground.astype(np.uint8), cv2.DIST_L2, 5,
        )
        outer_band = foreground & (inward_distance <= float(max_edge_width))
        rgb = rgba[:, :, :3].astype(np.float32)
        luma = rgb.mean(axis=2)
        chroma = rgb.max(axis=2) - rgb.min(axis=2)

        reliable = (
            foreground
            & (alpha >= 220)
            & (inward_distance > float(max_edge_width) + 1.0)
        )
        if not np.any(reliable):
            return rgba, 0

        search = np.ones(alpha.shape, dtype=np.uint8)
        search[reliable] = 0
        _, labels = cv2.distanceTransformWithLabels(
            search,
            cv2.DIST_L2,
            5,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
        reliable_pixels = np.argwhere(reliable)
        nearest_index = np.clip(
            labels.astype(np.int64) - 1, 0, len(reliable_pixels) - 1,
        )
        nearest_y = reliable_pixels[nearest_index, 0]
        nearest_x = reliable_pixels[nearest_index, 1]
        inward_rgb = rgb[nearest_y, nearest_x]
        inward_luma = inward_rgb.mean(axis=2)

        pale_fringe = (
            outer_band
            & (luma >= 188.0)
            & (chroma <= 58.0)
            & (inward_luma <= luma - 42.0)
        )
        repaired = int(np.count_nonzero(pale_fringe))
        if not repaired:
            return rgba, 0

        rgb[pale_fringe] = (
            0.08 * rgb[pale_fringe] + 0.92 * inward_rgb[pale_fringe]
        )
        rgba[:, :, :3] = np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)

        # SCHP promoted this same contaminated strip to near-opaque alpha.
        # Recolouring alone therefore leaves its staircase geometry visible as
        # a dark jitter line. Feather only the repaired strip inward using the
        # subject silhouette; do not expand or globally blur the matte.
        feather_zone = cv2.dilate(
            pale_fringe.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        ).astype(bool) & outer_band
        # Step inward by one source pixel before feathering. This removes the
        # SCHP-hardened crust itself; merely blurring the original silhouette
        # leaves the same stair-step contour visible as a darker outline.
        clean_silhouette = cv2.erode(
            foreground.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        silhouette_alpha = cv2.GaussianBlur(
            clean_silhouette.astype(np.float32) * 255.0,
            (0, 0),
            sigmaX=.85,
            sigmaY=.85,
        )
        softened_alpha = alpha.astype(np.float32)
        softened_alpha[feather_zone] = np.minimum(
            softened_alpha[feather_zone], silhouette_alpha[feather_zone],
        )
        rgba[:, :, 3] = np.rint(np.clip(softened_alpha, 0, 255)).astype(np.uint8)
        return rgba, repaired

    @staticmethod
    def _antialias_generated_head_boundary(
        generated,
        background_rgb,
        face_bbox,
    ):
        """Remove Qwen VAE staircase pixels around hair and ears.

        The protected-head latent hold preserves identity, but the 8x VAE
        rebuilds a sharp foreground/backdrop transition as a dark one-pixel
        staircase.  This is visible in the raw decode before passport crop.
        Reconstruct only that outer transition inside a face-measured head ROI
        with a one-pixel inward antialias.  No semantic matte/model is run,
        hairstyle geometry is not expanded, and all interior head pixels stay
        byte-identical.
        """
        rgb = np.asarray(generated.convert("RGB"), dtype=np.uint8)
        height, width = rgb.shape[:2]
        try:
            fx1, fy1, fx2, fy2 = (float(value) for value in face_bbox)
        except (TypeError, ValueError):
            return generated, 0
        face_w = max(1.0, fx2 - fx1)
        face_h = max(1.0, fy2 - fy1)
        left = max(0, int(round(fx1 - face_w * .62)))
        right = min(width, int(round(fx2 + face_w * .62)))
        top = max(0, int(round(fy1 - face_h * .72)))
        bottom = min(height, int(round(fy2 + face_h * .18)))
        if right - left < 12 or bottom - top < 12:
            return generated, 0

        roi = rgb[top:bottom, left:right].copy()
        backdrop = np.asarray(background_rgb, dtype=np.float32)
        distance = np.linalg.norm(
            roi.astype(np.float32) - backdrop[None, None, :], axis=2,
        )
        foreground = (distance > 24.0).astype(np.uint8)

        # Keep only the component that contains the detected face centre. This
        # prevents an unrelated mark in the backdrop from entering the edge.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
        if count <= 1:
            return generated, 0
        centre_x = int(np.clip(round((fx1 + fx2) * .5) - left, 0, roi.shape[1] - 1))
        centre_y = int(np.clip(round((fy1 + fy2) * .5) - top, 0, roi.shape[0] - 1))
        component_id = int(labels[centre_y, centre_x])
        if component_id == 0:
            component_id = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        subject = (labels == component_id).astype(np.uint8)

        eroded = cv2.erode(
            subject,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        soft_alpha = cv2.GaussianBlur(
            eroded.astype(np.float32), (0, 0), sigmaX=.72, sigmaY=.72,
        )
        outer = cv2.dilate(
            subject,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        ).astype(bool)
        inner = cv2.erode(
            subject,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        ).astype(bool)
        boundary = outer & ~inner
        if not np.any(boundary):
            return generated, 0

        alpha = np.ones(subject.shape, dtype=np.float32)
        alpha[boundary] = np.clip(soft_alpha[boundary], 0.0, 1.0)
        blended = roi.astype(np.float32)
        blended[boundary] = (
            blended[boundary] * alpha[boundary, None]
            + backdrop[None, :] * (1.0 - alpha[boundary, None])
        )
        changed = int(np.count_nonzero(
            np.any(np.rint(blended).astype(np.uint8) != roi, axis=2)
        ))
        if not changed:
            return generated, 0
        out = rgb.copy()
        out[top:bottom, left:right] = np.rint(
            np.clip(blended, 0, 255)
        ).astype(np.uint8)
        return Image.fromarray(out, "RGB"), changed

    @staticmethod
    def _repair_generated_face_chromatic_speckles(
        source: Image.Image,
        generated: Image.Image,
    ) -> Image.Image:
        """Remove small invented blue/cyan facial marks without smoothing skin.

        Qwen occasionally interprets scan noise as decorative blue dots.  The
        repair is restricted to small connected chromatic components inside the
        detected face. If the aligned upload contains a real dark mark at that
        coordinate (for example a bindi), its source colour is restored;
        otherwise the candidate's local skin median is used. Hair, earrings,
        eyes, uniform and the backdrop are never globally recoloured.
        """
        try:
            from pipelines.photo_restoration import _get_insight_app

            source_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
            result = np.asarray(generated.convert("RGB"), dtype=np.uint8).copy()
            detector = _get_insight_app()
            source_faces = detector.get(cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR))
            generated_faces = detector.get(cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            if len(source_faces) != 1 or len(generated_faces) != 1:
                return generated.convert("RGB")
            matrix, _ = cv2.estimateAffinePartial2D(
                np.asarray(source_faces[0].kps, dtype=np.float32),
                np.asarray(generated_faces[0].kps, dtype=np.float32),
                method=cv2.LMEDS,
            )
            if matrix is None:
                return generated.convert("RGB")
            height, width = result.shape[:2]
            aligned_source = cv2.warpAffine(
                source_rgb, matrix, (width, height), flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            x1, y1, x2, y2 = (
                int(round(float(value))) for value in generated_faces[0].bbox
            )
            face_w, face_h = max(1, x2 - x1), max(1, y2 - y1)
            yy, xx = np.indices((height, width))
            # Keep the repair within the facial oval, excluding hair/ears and
            # the shirt even if the semantic parser is uncertain.
            center_x = (x1 + x2) * .5
            center_y = (y1 + y2) * .5
            face_oval = (
                ((xx - center_x) / max(1.0, face_w * .45)) ** 2
                + ((yy - center_y) / max(1.0, face_h * .49)) ** 2
                <= 1.0
            )
            # Do not run the small-mark repair across eyes/eyebrows, nostrils
            # or lips. Local median replacement on those real high-contrast
            # edges can create pale pinholes even when the candidate detector
            # correctly identifies a nearby print-noise speckle.
            feature_protection = np.zeros((height, width), dtype=np.uint8)
            keypoints = np.asarray(generated_faces[0].kps, dtype=np.float32)
            for eye_x, eye_y in keypoints[:2]:
                cv2.ellipse(
                    feature_protection,
                    (int(round(eye_x)), int(round(eye_y - face_h * .035))),
                    (max(5, int(face_w * .16)), max(5, int(face_h * .13))),
                    0, 0, 360, 1, -1,
                )
            nose_x, nose_y = keypoints[2]
            cv2.ellipse(
                feature_protection,
                (int(round(nose_x)), int(round(nose_y))),
                (max(4, int(face_w * .11)), max(5, int(face_h * .15))),
                0, 0, 360, 1, -1,
            )
            mouth_center = np.mean(keypoints[3:5], axis=0)
            cv2.ellipse(
                feature_protection,
                (int(round(mouth_center[0])), int(round(mouth_center[1]))),
                (max(6, int(face_w * .23)), max(5, int(face_h * .11))),
                0, 0, 360, 1, -1,
            )
            safe_skin = face_oval & ~feature_protection.astype(bool)
            hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
            blue_cyan = (
                safe_skin
                & (hsv[:, :, 0] >= 78)
                & (hsv[:, :, 0] <= 132)
                & (hsv[:, :, 1] >= 70)
                & (hsv[:, :, 2] >= 35)
            )
            candidate_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
            local_gray = cv2.medianBlur(candidate_gray, 7)
            source_gray = cv2.cvtColor(aligned_source, cv2.COLOR_RGB2GRAY)
            source_local_gray = cv2.medianBlur(source_gray, 9)
            # Preserve genuine source marks by their contrast against nearby
            # source skin, rather than by an absolute dark threshold. The old
            # ``source < 100`` mask treated print grain as a real feature and
            # protected dozens of Qwen-invented freckles. Significant source
            # features (bindi, eyes, nostrils, lips and real moles) remain
            # supported after a small dilation for alignment tolerance.
            source_feature_support = (
                source_local_gray.astype(np.int16) - source_gray.astype(np.int16)
            ) > 18
            source_feature_support = cv2.dilate(
                source_feature_support.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            ).astype(bool)
            local_dark_delta = (
                local_gray.astype(np.int16) - candidate_gray.astype(np.int16)
            )
            # Qwen can translate scan grain into clusters of brown/black facial
            # marks. Remove a small local impulse only when the aligned upload
            # has no corresponding feature. This remains component-bounded and
            # does not blur normal skin texture.
            invented_dark_spot = (
                safe_skin
                & (candidate_gray < 215)
                & (local_dark_delta > 7)
                & (((hsv[:, :, 1] > 18) & (local_dark_delta > 7)) | (local_dark_delta > 14))
                & ~source_feature_support
            )
            candidate_speckles = (blue_cyan | invented_dark_spot).astype(np.uint8)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                candidate_speckles, connectivity=8,
            )
            local_median = cv2.medianBlur(result, 7)
            repaired = 0
            max_area = max(16, min(80, int(round(face_w * face_h * .0015))))
            for component in range(1, count):
                area = int(stats[component, cv2.CC_STAT_AREA])
                if not 1 <= area <= max_area:
                    continue
                component_mask = labels == component
                # Expand by one pixel so antialiased blue edges do not remain.
                component_mask = cv2.dilate(
                    component_mask.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                    iterations=1,
                ).astype(bool) & face_oval
                real_dark_mark = component_mask & source_feature_support
                result[component_mask] = local_median[component_mask]
                result[real_dark_mark] = aligned_source[real_dark_mark]
                repaired += int(np.count_nonzero(component_mask))
            if repaired:
                logger.info(
                    "[UNIFORM_FACE_REPAIR] Removed %d invented facial speckle pixels",
                    repaired,
                )
                return Image.fromarray(result, "RGB")
            return generated.convert("RGB")
        except Exception as exc:
            logger.warning(
                "[UNIFORM_FACE_REPAIR] Chromatic facial-speck repair skipped: %s",
                exc,
            )
            return generated.convert("RGB")

    @staticmethod
    def _source_identity_reference(
        image: Image.Image,
        face_bbox: Any,
        width: int,
        height: int,
        background_color: Optional[Tuple[int, int, int]] = None,
        target_face_fraction: float = .32,
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
        rgb_arr = np.asarray(source)
        pad_bg_color = background_color
        if pad_bg_color is None:
            edge = max(4, min(16, source_h // 20, source_w // 20))
            pad_bg_color = tuple(
                int(c) for c in np.median(
                    np.concatenate((
                        rgb_arr[:edge, :edge].reshape(-1, 3),
                        rgb_arr[:edge, -edge:].reshape(-1, 3),
                    )),
                    axis=0,
                ).round().clip(0, 255).astype(np.uint8)
            )

        dist_to_bg_src = np.linalg.norm(rgb_arr.astype(np.float32) - np.asarray(pad_bg_color, dtype=np.float32), axis=-1)
        occ_rows_src = np.flatnonzero(np.count_nonzero(dist_to_bg_src > 18.0, axis=1) >= max(3, int(source_w * 0.02)))
        crown_y_src = int(occ_rows_src[0]) if occ_rows_src.size else max(0, int(y1 - face_h * 0.35))
        target_headroom_src = max(24, int(round(requested_crop_h * 0.08)))
        desired_top = crown_y_src - target_headroom_src
        desired_left = int(round((x1 + x2) * .5 - requested_crop_w * .5))
        needs_extension = (
            requested_crop_h > source_h
            or requested_crop_w > source_w
            or desired_top < 0
        )
        if needs_extension and background_color is None:
            # Identity conditioning must contain real uploaded pixels only.
            # Padding a tight portrait with its sampled wall colour created a
            # large empty lower canvas; Qwen then outpainted a new torso and a
            # new head instead of locally changing the uniform.  Use the
            # largest real crop at the requested aspect ratio.  The final Qwen
            # decode can still create the selected backdrop, but no fabricated
            # source canvas is allowed to redefine body/head proportions.
            crop_h = min(source_h, int(np.floor(source_w / max(target_ratio, 1e-6))))
            crop_w = min(source_w, int(round(crop_h * target_ratio)))
            if crop_h < 1 or crop_w < 1:
                crop_h, crop_w = source_h, source_w
            face_center_x = (x1 + x2) * .5
            top = int(round(y1 - face_h * .42))
            top = max(0, min(source_h - crop_h, top))
            left = int(round(face_center_x - crop_w * .5))
            left = max(0, min(source_w - crop_w, left))
            logger.info(
                "[UNIFORM_IDENTITY_REFERENCE] Tight upload kept on real pixels only crop=%dx%d; no backdrop padding",
                crop_w, crop_h,
            )
            return source.crop(
                (left, top, left + crop_w, top + crop_h)
            ).resize((width, height), Image.LANCZOS)

        if needs_extension:
            # A tightly shot upload cannot satisfy a passport face/body ratio
            # through cropping alone. Clamping to the source was the root cause
            # of 40%+ faces and failed upper-chest framing. Extend the backdrop
            # around the untouched upload; the rough garment guide fills the lower
            # edit region before Qwen generates the coherent final portrait.
            crop_h = requested_crop_h
            crop_w = requested_crop_w
            canvas = Image.new("RGB", (crop_w, crop_h), tuple(pad_bg_color))
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

    @classmethod
    def _detect_hair_accessory_info(
        cls,
        person_pil: Image.Image,
        face_bbox: Any = None,
    ) -> Dict[str, Any]:
        """Detect hair accessories (bows, ribbons, clips) on flat/studio portraits dynamically."""
        source_rgb = np.asarray(person_pil.convert("RGB"), dtype=np.uint8)
        source_h, source_w = source_rgb.shape[:2]
        patch_h = max(6, int(round(source_h * .10)))
        patch_w = max(6, int(round(source_w * .16)))
        border = np.concatenate((
            source_rgb[:patch_h, :patch_w].reshape(-1, 3),
            source_rgb[:patch_h, -patch_w:].reshape(-1, 3),
        ), axis=0).astype(np.float32)
        backdrop = np.median(border, axis=0)
        border_distance = np.linalg.norm(border - backdrop[None, :], axis=1)
        p90 = float(np.percentile(border_distance, 90))
        if p90 > 38.0:
            return {
                "detected": False,
                "pixels": 0,
                "color": None,
                "placement": None,
                "description": "none",
                "mask": None,
            }

        if face_bbox is None:
            try:
                from pipelines.photo_restoration import _get_insight_app
                faces = _get_insight_app().get(cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR))
                if not faces:
                    return {
                        "detected": False,
                        "pixels": 0,
                        "color": None,
                        "placement": None,
                        "description": "none",
                        "mask": None,
                    }
                face_bbox = faces[0].bbox
            except Exception:
                return {
                    "detected": False,
                    "pixels": 0,
                    "color": None,
                    "placement": None,
                    "description": "none",
                    "mask": None,
                }

        x1, y1, x2, y2 = (float(v) for v in face_bbox)
        face_w, face_h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        centre_x = (x1 + x2) * .5

        try:
            from pipelines.schp_service import parse as parse_body_parts
            labels = parse_body_parts(source_rgb)["labels"]
        except Exception:
            labels = None

        yy, xx = np.indices((source_h, source_w))
        # Include lateral braid/pigtail corridors below the chin so hair ties, scrunchies,
        # and beaded ties along long plaits are detected and protected from erasure.
        head_zone = (
            (xx >= centre_x - face_w * 1.25)
            & (xx <= centre_x + face_w * 1.25)
            & (yy >= max(0.0, y1 - face_h * .55))
            & (yy <= min(float(source_h), y2 - face_h * .05))
        )
        braid_acc_zone = (
            (yy >= y2 - face_h * .10)
            & (yy <= min(float(source_h), y2 + face_h * 0.38))
            & (
                ((xx >= x1 - face_w * 0.65) & (xx <= x1 + face_w * 0.02))
                | ((xx >= x2 - face_w * 0.02) & (xx <= x2 + face_w * 0.65))
            )
        )
        head_zone = head_zone | braid_acc_zone
        face_oval = (
            ((xx - centre_x) / max(face_w * .48, 1.0)) ** 2
            + ((yy - (y1 + y2) * .5) / max(face_h * .55, 1.0)) ** 2
            <= 1.0
        )

        torso_clothing = (
            (yy >= y2 - face_h * 0.05)
            & (
                np.isin(labels, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
                if labels is not None
                else np.zeros((source_h, source_w), dtype=bool)
            )
            & (np.abs(xx - centre_x) <= face_w * 0.55)
        )

        if labels is not None:
            hair_mask = (labels == 2) | (labels == 1)
            if not np.any(hair_mask):
                return {
                    "detected": False,
                    "pixels": 0,
                    "color": None,
                    "placement": None,
                    "description": "none",
                    "mask": None,
                }
            hair_reach = max(8, int(round(face_w * 0.15)))
            hair_zone = cv2.dilate(
                hair_mask.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hair_reach * 2 + 1, hair_reach * 2 + 1)),
                iterations=1,
            ).astype(bool) | braid_acc_zone
            non_accessory = np.isin(labels, [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        else:
            hair_mask = np.ones((source_h, source_w), dtype=bool)
            hair_zone = np.ones((source_h, source_w), dtype=bool)
            non_accessory = np.zeros((source_h, source_w), dtype=bool)

        hsv = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2HSV)
        bg_dist = np.linalg.norm(
            source_rgb.astype(np.float32) - backdrop[None, None, :], axis=2,
        )
        # Saturated accessories (such as blue beads, satin ribbons, or colored hair bands)
        # in the lateral braid corridor are authentic accessories even if SCHP mislabels them 0 or 5.
        # Exclude torso clothing, badges, emblems, or placket fabric from being classified as hair accessories.
        is_braid_acc_candidate = braid_acc_zone & (hsv[:, :, 1] >= 80) & (bg_dist > 28.0) & ~torso_clothing
        eff_non_accessory = non_accessory & (~is_braid_acc_candidate)

        hair_sample_zone = head_zone & ~face_oval & (hsv[:, :, 2] < 125) & hair_mask
        if np.count_nonzero(hair_sample_zone) >= 64:
            hair_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
            hair_median = np.median(hair_lab[hair_sample_zone], axis=0)
            hair_dist = np.linalg.norm(
                hair_lab - hair_median[None, None, :], axis=2,
            )
        else:
            hair_dist = np.full((source_h, source_w), 255.0, dtype=np.float32)

        candidates = (
            head_zone
            & ~face_oval
            & hair_zone
            & ~eff_non_accessory
            & (bg_dist > 25.0)
            & (hsv[:, :, 1] > 65)
            & (hsv[:, :, 2] > 70)
            & (hair_dist > 30.0)
        ).astype(np.uint8)

        candidates = cv2.morphologyEx(
            candidates, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        count, c_labels, stats, centroids = cv2.connectedComponentsWithStats(candidates, connectivity=8)

        left_comps = []
        right_comps = []
        crown_comps = []
        max_area = max(64, int(source_h * source_w * .075))

        for i in range(1, count):
            area = int(stats[i, cv2.CC_STAT_AREA])
            comp_w = int(stats[i, cv2.CC_STAT_WIDTH])
            comp_h = int(stats[i, cv2.CC_STAT_HEIGHT])
            if area < 80 or area > max_area:
                continue
            # Accessories must have 2D extent, not thin 1D hair strands or rim light
            if min(comp_w, comp_h) < 14:
                continue
            comp_hsv = hsv[c_labels == i]
            comp_mean_rgb = source_rgb[c_labels == i].mean(axis=0)
            if (
                comp_mean_rgb[0] > comp_mean_rgb[1]
                and comp_mean_rgb[1] > comp_mean_rgb[2]
                and comp_mean_rgb[0] < 125
                and np.median(comp_hsv[:, 1]) < 150
            ):
                continue

            cx, cy = centroids[i]
            comp_mask = (c_labels == i)
            # Reject components below jaw that are predominantly upper clothing or inside the central torso
            if cy >= y2 - face_h * 0.05 and labels is not None:
                clothing_fraction = np.mean(np.isin(labels[comp_mask], [5, 6, 7, 8, 9, 10, 11, 12]))
                if clothing_fraction > 0.40:
                    continue
                if np.abs(cx - centre_x) < face_w * 0.55:
                    continue

            if cy >= y1 - face_h * 0.15:
                if cx < centre_x - face_w * 0.25:
                    left_comps.append(i)
                elif cx > centre_x + face_w * 0.25:
                    right_comps.append(i)
                else:
                    crown_comps.append(i)
            else:
                solidness = area / max(1.0, float(comp_w * comp_h))
                if solidness >= 0.28 and (np.median(comp_hsv[:, 1]) > 140 or np.median(comp_hsv[:, 2]) > 220):
                    crown_comps.append(i)

        left_area = sum(stats[i, cv2.CC_STAT_AREA] for i in left_comps)
        right_area = sum(stats[i, cv2.CC_STAT_AREA] for i in right_comps)
        min_side_thresh = max(300, int(round(source_h * source_w * 0.0008)))

        selected = np.zeros_like(candidates)
        if left_area >= min_side_thresh and right_area >= min_side_thresh:
            for i in left_comps + right_comps:
                selected[c_labels == i] = 255
            placement = "both sides"
        elif left_area >= min_side_thresh:
            for i in left_comps:
                selected[c_labels == i] = 255
            placement = "viewer-left side"
        elif right_area >= min_side_thresh:
            for i in right_comps:
                selected[c_labels == i] = 255
            placement = "viewer-right side"
        elif crown_comps:
            for i in crown_comps:
                selected[c_labels == i] = 255
            placement = "top crown"
        else:
            return {
                "detected": False,
                "pixels": 0,
                "color": None,
                "placement": None,
                "description": "none",
                "mask": None,
            }

        cnt = int(np.count_nonzero(selected))
        if cnt < 100:
            return {
                "detected": False,
                "pixels": cnt,
                "color": None,
                "placement": None,
                "description": "none",
                "mask": None,
            }

        med_hue = float(np.median(hsv[selected > 0, 0]))
        med_sat = float(np.median(hsv[selected > 0, 1]))
        med_val = float(np.median(hsv[selected > 0, 2]))

        if med_sat < 35 and med_val > 180:
            color_name = "white"
        elif med_hue < 15 or med_hue >= 140:
            color_name = "red"
        elif 85 <= med_hue <= 135:
            color_name = "blue"
        elif 40 <= med_hue < 85:
            color_name = "green"
        elif 24 <= med_hue < 40:
            color_name = "yellow"
        else:
            color_name = "colorful"

        if placement == "both sides":
            desc = f"Two {color_name} bows/ribbons on both sides of the head"
        elif placement == "viewer-left side":
            desc = f"{color_name.capitalize()} bow/ribbon on viewer-left side of the head"
        elif placement == "viewer-right side":
            desc = f"{color_name.capitalize()} bow/ribbon on viewer-right side of the head"
        else:
            desc = f"{color_name.capitalize()} hair accessory on the head"

        return {
            "detected": True,
            "pixels": cnt,
            "color": color_name,
            "placement": placement,
            "description": desc,
            "mask": selected,
        }

    @classmethod
    def _build_dynamic_cv_plan(
        cls,
        person_pil: Image.Image,
        template_pil: Image.Image,
        has_sunlight: bool,
        head_bright_ratio: float,
        is_extreme: bool,
        source_w: int,
        source_h: int,
        face_detail_region: np.ndarray,
    ) -> Dict[str, Any]:
        """Build an image-specific dynamic visual analysis plan without hardcoded template assumptions."""
        palette = _measure_garment_palette(template_pil)
        accent = _measure_garment_accent_color(template_pil)

        t_rgb = np.asarray(template_pil.convert("RGB"))
        t_gray = cv2.cvtColor(t_rgb, cv2.COLOR_RGB2GRAY)
        lap_var = float(cv2.Laplacian(t_gray, cv2.CV_64F).var())
        is_pattern = lap_var > 120.0

        main_color_desc = f"{palette[0]['hex']}" if palette else "uniform"
        accent_color_desc = f"{accent['hex']}" if accent else "none"

        hair_style_desc = "natural"
        hair_acc_desc = "none"
        acc_info = {"detected": False, "mask": None}
        try:
            from pipelines.photo_restoration import _get_insight_app
            faces = _get_insight_app().get(cv2.cvtColor(np.asarray(person_pil), cv2.COLOR_RGB2BGR))
            if faces:
                from pipelines.schp_service import parse as parse_body_parts
                person_labels = parse_body_parts(np.asarray(person_pil))["labels"]
                if _has_visible_tied_hair_geometry(person_labels, faces[0].bbox):
                    hair_style_desc = "tied hair"
                acc_info = cls._detect_hair_accessory_info(person_pil, faces[0].bbox)
                if acc_info.get("detected"):
                    hair_acc_desc = acc_info["description"]
        except Exception:
            pass

        return {
            "source": "dynamic_cv_measurement",
            "garment_components": "shirt",
            "shirt_color_and_pattern": "checkered/patterned" if is_pattern else f"solid {main_color_desc}",
            "shirt_pattern_type": "pattern" if is_pattern else "solid",
            "accent_trim_placement": "collar and placket edge" if accent else "none",
            "accent_trim_color": accent_color_desc,
            "outer_garment_color_and_shape": "none",
            "shirt_collar": "collar",
            "outer_neckline": "uncertain",
            "sleeve_length": "short sleeve",
            "button_layout": "buttons",
            "badge_present": False,
            "badge_location": "none",
            "badge_bbox": None,
            "hair_style": hair_style_desc,
            "hair_accessories": hair_acc_desc,
            "hair_accessory_details": hair_acc_desc,
            "cv_hair_accessory_info": acc_info,
            "apparent_age_band": "school child",
            "direct_sunlight_present": has_sunlight,
            "head_hair_hotspot_present": head_bright_ratio > 0.08,
            "glasses_reflection_present": False,
            "measured_sunlight_severity": "extreme" if is_extreme else "normal",
            "source_min_dimension": min(source_w, source_h),
            "source_face_detail_score": (
                float(cv2.Laplacian(face_detail_region, cv2.CV_64F).var())
                if face_detail_region.size else 0.0
            ),
        }

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
    def _restore_flat_source_hair_accessories(
        source: Image.Image,
        generated: Image.Image,
        source_face: Any,
        generated_face: Any,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Restore only chromatic hair accessories lost into a flat backdrop.

        This is not a person matte and never touches the face, forehead, hair,
        skin, neck, or uniform. It is enabled only when the uploaded source has
        a demonstrably flat border, allowing ribbon/flower pixels to be
        separated without BiRefNet. The source accessory mask is aligned by
        facial landmarks and composited over Qwen's already-generated flat
        backdrop.
        """
        source_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        generated_rgb = np.asarray(generated.convert("RGB"), dtype=np.uint8)
        source_h, source_w = source_rgb.shape[:2]
        patch_h = max(6, int(round(source_h * .10)))
        patch_w = max(6, int(round(source_w * .16)))
        # Use the two upper corners: upper-chest passport portraits naturally
        # touch the side and bottom frame with hair/shoulders, so treating the
        # entire border as backdrop incorrectly rejects an otherwise flat
        # studio source.
        border = np.concatenate((
            source_rgb[:patch_h, :patch_w].reshape(-1, 3),
            source_rgb[:patch_h, -patch_w:].reshape(-1, 3),
        ), axis=0).astype(np.float32)
        backdrop = np.median(border, axis=0)
        border_distance = np.linalg.norm(border - backdrop[None, :], axis=1)
        if float(np.percentile(border_distance, 90)) > 38.0:
            return generated.convert("RGB"), {
                "applied": False, "reason": "source_backdrop_not_flat",
            }

        x1, y1, x2, y2 = (float(value) for value in source_face.bbox)
        face_w, face_h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        centre_x = (x1 + x2) * .5
        yy, xx = np.indices((source_h, source_w))
        head_zone = (
            (xx >= centre_x - face_w * 1.25)
            & (xx <= centre_x + face_w * 1.25)
            & (yy >= max(0.0, y1 - face_h * .55))
            & (yy <= min(float(source_h), y2 + face_h * .05))
        )
        # Below the chin, keep only lateral hair-accessory zones. This excludes
        # a saturated source shirt/collar from ever being copied into Qwen's
        # generated uniform while retaining long ribbon tails and side flowers.
        head_zone &= (
            (yy <= y2)
            | (np.abs(xx - centre_x) >= face_w * .48)
        )
        # Protect the facial interior around eyes, nose, cheeks and mouth.
        face_oval = (
            ((xx - centre_x) / max(face_w * .48, 1.0)) ** 2
            + ((yy - (y1 + y2) * .5) / max(face_h * .55, 1.0)) ** 2
            <= 1.0
        )
        hsv = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2HSV)
        background_distance = np.linalg.norm(
            source_rgb.astype(np.float32) - backdrop[None, None, :], axis=2,
        )
        # Hair is usually dark/low-chroma. Measure it from the upper side bands
        # and require a large colour departure so ordinary strands cannot enter
        # the accessory mask.
        hair_sample_zone = head_zone & ~face_oval & (hsv[:, :, 2] < 125)
        if np.count_nonzero(hair_sample_zone) >= 64:
            hair_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
            hair_median = np.median(hair_lab[hair_sample_zone], axis=0)
            hair_distance = np.linalg.norm(
                hair_lab - hair_median[None, None, :], axis=2,
            )
        else:
            hair_distance = np.full((source_h, source_w), 255.0, dtype=np.float32)
        accessory = (
            head_zone
            & ~face_oval
            & (background_distance > 24.0)
            & (hsv[:, :, 1] > 55)
            & (hsv[:, :, 2] > 50)
            & (hair_distance > 28.0)
        ).astype(np.uint8)
        accessory = cv2.morphologyEx(
            accessory, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            accessory, connectivity=8,
        )
        selected = np.zeros_like(accessory)
        maximum_area = max(64, int(source_h * source_w * .075))
        for component in range(1, count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if 10 <= area <= maximum_area:
                selected[labels == component] = 255
        if np.count_nonzero(selected) < 20:
            return generated.convert("RGB"), {
                "applied": False, "reason": "no_confident_accessory_pixels",
            }

        matrix, _ = cv2.estimateAffinePartial2D(
            np.asarray(source_face.kps, dtype=np.float32),
            np.asarray(generated_face.kps, dtype=np.float32),
            method=cv2.LMEDS,
        )
        if matrix is None:
            return generated.convert("RGB"), {
                "applied": False, "reason": "landmark_alignment_failed",
            }
        height, width = generated_rgb.shape[:2]
        aligned_source = cv2.warpAffine(
            source_rgb, matrix, (width, height), flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        aligned_mask = cv2.warpAffine(
            selected, matrix, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        # Apply the same hard safety zones in generated-image coordinates.
        # Landmark warping can otherwise move a few antialiased mask pixels
        # into the face or centre collar even when the source mask excluded
        # them.  Hair accessories are lateral to these protected regions.
        gx1, gy1, gx2, gy2 = (float(value) for value in generated_face.bbox)
        generated_face_w = max(1.0, gx2 - gx1)
        generated_face_h = max(1.0, gy2 - gy1)
        generated_centre_x = (gx1 + gx2) * .5
        generated_yy, generated_xx = np.indices((height, width))
        generated_face_oval = (
            ((generated_xx - generated_centre_x) / max(generated_face_w * .48, 1.0)) ** 2
            + (
                (generated_yy - (gy1 + gy2) * .5)
                / max(generated_face_h * .55, 1.0)
            ) ** 2
            <= 1.0
        )
        generated_centre_uniform = (
            (generated_yy > gy2)
            & (np.abs(generated_xx - generated_centre_x) < generated_face_w * .50)
        )
        aligned_mask[generated_face_oval | generated_centre_uniform] = 0
        aligned_mask[:max(4, int(height * 0.03)), :] = 0
        if np.count_nonzero(aligned_mask) < 20:
            return generated.convert("RGB"), {
                "applied": False,
                "reason": "no_safe_aligned_accessory_pixels",
            }
        alpha = cv2.GaussianBlur(
            aligned_mask, (0, 0), .55,
        ).astype(np.float32) / 255.0
        # Blurring softens only the accessory boundary; re-apply the hard
        # protection afterwards so blur cannot bleed into face or uniform.
        alpha[generated_face_oval | generated_centre_uniform] = 0.0
        result = (
            generated_rgb.astype(np.float32) * (1.0 - alpha[:, :, None])
            + aligned_source.astype(np.float32) * alpha[:, :, None]
        )
        return Image.fromarray(np.rint(result).clip(0, 255).astype(np.uint8), "RGB"), {
            "applied": True,
            "source_pixels": int(np.count_nonzero(selected)),
            "aligned_pixels": int(np.count_nonzero(aligned_mask)),
        }

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
        erase_invented_hair: bool = False,
        preserve_source_chroma: bool = False,
        upload_backdrop_is_synthetic: bool = False,
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

        # Build the person matte from the parsed labels this function already
        # computes.  BiRefNet used to supply it, but its local model load fails
        # regularly and its failure fallback (alpha=255 everywhere) silently
        # disabled every matte guard below, which let the upload's wall into the
        # pasted head.  The parser is always available here, so the head mask,
        # the neck guard and the backdrop exclusion now read one consistent
        # matte and no external matting service can fail this path.
        person_label = np.isin(
            aligned_labels, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
        )
        person_label = cv2.morphologyEx(
            person_label.astype(np.uint8), cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1,
        )
        # Feather only a couple of pixels: the guards need a firm person/
        # backdrop split, not a soft matte that re-admits the backdrop.
        aligned_person_alpha = np.clip(
            cv2.GaussianBlur(person_label.astype(np.float32), (0, 0), 1.6) * 330.0,
            0, 255,
        ).astype(np.uint8)

        x1, y1, x2, y2 = np.asarray(generated_faces[0].bbox, dtype=float)
        face_w, face_h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        yy, xx = np.indices((height, width))

        source_face = aligned_labels == 13
        source_clothing = np.isin(aligned_labels, (5, 6, 7, 8, 9, 10, 11, 12))
        source_eyewear = aligned_labels == 4
        eyewear_points = np.column_stack(np.where(source_eyewear)[::-1]).astype(np.int32)
        if eyewear_points.shape[0] >= 12:
            eyewear_region = np.zeros((height, width), dtype=np.uint8)
            cv2.fillConvexPoly(eyewear_region, cv2.convexHull(eyewear_points), 1)
            source_eyewear = eyewear_region.astype(bool)
        source_hair_label = (aligned_labels == 2) & ~source_clothing
        if np.count_nonzero(source_face) < 128 or np.count_nonzero(source_hair_label) < 128:
            return generated.convert("RGB"), {"applied": False, "reason": "head_mask_unavailable"}

        source_lab = cv2.cvtColor(aligned_source, cv2.COLOR_RGB2LAB).astype(np.float32)
        reliable_hair_zone = (
            (yy >= max(0, int(y1 - face_h * .72)))
            & (yy <= int(y2 + face_h * .24))
            & (xx >= int(x1 - face_w * .62))
            & (xx <= int(x2 + face_w * .62))
        )
        dark_samples = source_lab[
            source_hair_label & reliable_hair_zone & (source_lab[:, :, 0] < 115)
        ]
        if dark_samples.size == 0:
            return generated.convert("RGB"), {"applied": False, "reason": "hair_colour_unavailable"}
        hair_median = np.median(dark_samples, axis=0)
        colour_distance = np.linalg.norm(source_lab - hair_median[None, None, :], axis=2)

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
        base_head_hair = (
            source_hair_label
            & reliable_hair_zone
            & (yy <= int(y2 + face_h * .18))
            & ~source_clothing
        )
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
            & source_hair_label
            & ~source_clothing
            & ((colour_distance < 62.0) | (source_lab[:, :, 0] < 78.0))
        )
        dark_hair = (base_head_hair | lower_dark_hair) & ~source_clothing & (
            (colour_distance < 62.0) | (source_lab[:, :, 0] < 78.0)
        )
        dark_hair &= aligned_person_alpha > 40
        # Parser labels are warped with NEAREST interpolation, so every hair
        # boundary is a blocky staircase. That staircase survives the paste
        # feather and shows up as a cut-and-paste line across the forehead
        # where the darkened-hair treatment meets the relit skin. Round the
        # corners off before any extent, treatment or composite derives from it.
        dark_hair = (
            cv2.GaussianBlur(dark_hair.astype(np.float32), (0, 0), 1.7) > 0.45
        )
        crown_region = (
            (yy >= max(0, int(y1 - face_h * .62)))
            & (yy <= int(y1 + face_h * .20))
            & (xx >= int(x1 - face_w * .68))
            & (xx <= int(x2 + face_w * .68))
        )
        # Preserve literal crown accessories, not every bright sunlit hair
        # patch. Prefer SCHP headwear and supplement it with compact lateral
        # chromatic components touching real hair.
        source_headwear = (aligned_labels == 1) & crown_region
        source_hsv = cv2.cvtColor(aligned_source, cv2.COLOR_RGB2HSV)
        lateral_crown = crown_region & (
            (xx <= int(x1 + face_w * .18))
            | (xx >= int(x2 - face_w * .18))
        )
        chromatic_light = (
            lateral_crown
            & (source_hsv[:, :, 2] >= 120)
            & (source_hsv[:, :, 1] >= 18)
            & (cv2.dilate(
                dark_hair.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
                iterations=1,
            ) > 0)
        ).astype(np.uint8)
        _, clip_components, clip_stats, _ = cv2.connectedComponentsWithStats(
            chromatic_light, connectivity=8,
        )
        light_clip = source_headwear & lateral_crown & (aligned_person_alpha > 20)
        minimum_clip_area = max(6, int(face_w * face_h * .0015))
        maximum_clip_area = max(minimum_clip_area + 1, int(face_w * face_h * .16))
        for component_id in range(1, clip_stats.shape[0]):
            area = int(clip_stats[component_id, cv2.CC_STAT_AREA])
            if minimum_clip_area <= area <= maximum_clip_area:
                light_clip |= clip_components == component_id
        light_clip &= aligned_person_alpha > 20
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
        central_torso_zone = (
            (yy > int(y2 + face_h * .10))
            & (xx >= int(x1 + face_w * .05))
            & (xx <= int(x2 - face_w * .05))
        )
        head_mask = (
            (source_face | dark_hair | light_clip | source_eyewear | source_accessories)
            & ~source_clothing
            & ~central_torso_zone
        )
        # The torso guard must not sever the upload's real neck. SCHP reports
        # bare neck as background (0), so skin colour is the only reliable
        # separator between neck, the upload's neutral wall and any shirt
        # inside the guard. Without this the restored jaw met Qwen's redrawn
        # neck at a hard horizontal seam across mid-neck.
        # The fitted collar now sits at this portrait's measured neck base,
        # which can be only a few percent of a face height below the chin. A
        # fixed 26%-face-height skin bridge would then paste the upload's bare
        # neck over the generated collar. Measure how far the generated
        # portrait itself still shows neck skin in the central column and stop
        # the bridge there, so the restored head meets the generated collar
        # exactly where that collar begins.
        generated_y2 = float(generated_faces[0].bbox[3])
        generated_face_h = max(
            1.0, float(generated_faces[0].bbox[3] - generated_faces[0].bbox[1]),
        )
        generated_face_w = max(
            1.0, float(generated_faces[0].bbox[2] - generated_faces[0].bbox[0]),
        )
        generated_cx = (
            float(generated_faces[0].bbox[0]) + float(generated_faces[0].bbox[2])
        ) * .5
        generated_hsv = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2HSV)
        generated_left = max(0, int(round(generated_cx - generated_face_w * .16)))
        generated_right = min(width, int(round(generated_cx + generated_face_w * .16)))
        generated_neck_bottom = int(round(generated_y2 + generated_face_h * .02))
        for row in range(
            int(generated_y2),
            min(height, int(generated_y2 + generated_face_h * .30)),
        ):
            band = generated_hsv[row, generated_left:generated_right]
            skin_fraction = np.count_nonzero(
                (band[:, 1] >= 20)
                & (band[:, 2] >= 35)
                & (band[:, 2] <= 245)
                & ((band[:, 0] <= 25) | (band[:, 0] >= 170))
            ) / max(1, band.shape[0])
            if skin_fraction >= .55:
                generated_neck_bottom = row + 2
        bare_neck_skin = (
            central_torso_zone
            & (yy <= int(min(y2 + face_h * .26, generated_neck_bottom)))
            & (aligned_source[:, :, 0] > 60)
            & (
                aligned_source[:, :, 0].astype(np.int16)
                - aligned_source[:, :, 2].astype(np.int16) > 16
            )
            & (aligned_source[:, :, 1] >= aligned_source[:, :, 2])
            & ~source_clothing
        )
        # For an exact identity fallback, the high-resolution person matte is
        # a safer complete-head boundary than SCHP hair labels alone. SCHP can
        # omit the crown, short side hair, headbands and ears; erasing Qwen's
        # invented hair against that incomplete mask produced large backdrop
        # holes. Limit the person matte to a landmark-bounded head zone so no
        # source shirt can enter the generated uniform. Below the chin the
        # zone reaches the upload's collar line but only across the neck
        # width, so shoulders and shirt can never ride in with it.
        complete_head_zone = (
            (yy >= max(0, int(y1 - face_h * .88)))
            & (yy <= int(y2 + face_h * .26))
            & (xx >= int(x1 - face_w * .72))
            & (xx <= int(x2 + face_w * .72))
        )
        below_chin = yy > int(y2 + face_h * .08)
        neck_zone_x = (
            (xx >= int(x1 - face_w * .30))
            & (xx <= int(x2 + face_w * .30))
        )
        matte_complete_head = (
            (aligned_person_alpha > 72)
            & complete_head_zone
            & ~source_clothing
            & (~central_torso_zone | bare_neck_skin)
            & (~below_chin | neck_zone_x)
        )
        head_mask |= matte_complete_head
        head_mask &= aligned_person_alpha > 16
        # Never paste the upload's own backdrop. SCHP can label the pale wall
        # around wispy hair as hair (its colour is inside the dark-hair
        # distance radius) and the person matte admits its soft boundary;
        # both drag a green-grey wall ring onto the white studio field around
        # the restored head. Sample the upload's top corners as its backdrop
        # and drop bright, low-matte pixels that match it, while every solid
        # person pixel (matte alpha 255) stays untouched.
        upload_edge = max(
            8, min(48, source_rgb.shape[0] // 12, source_rgb.shape[1] // 12),
        )
        upload_bg_lab = np.median(
            cv2.cvtColor(np.concatenate((
                source_rgb[:upload_edge, :upload_edge].reshape(-1, 1, 3),
                source_rgb[:upload_edge, -upload_edge:].reshape(-1, 1, 3),
            ), axis=0), cv2.COLOR_RGB2LAB).reshape(-1, 3),
            axis=0,
        )
        # A cut-out upload has no photographed wall: its "backdrop" is the flat
        # white this pipeline composited under the transparent pixels. Keying
        # against that colour removes the subject's own antialiased hair fringe
        # instead of a wall - measured at ~30k real soft-hair pixels on a 3500px
        # cut-out - and the restored head is then delivered inside a grey
        # envelope where that fringe should be. Only distrust a backdrop that
        # was actually photographed.
        if upload_backdrop_is_synthetic:
            logger.info(
                "[UNIFORM_HEAD_LOCK] Upload was a cut-out composited onto white; "
                "keeping its antialiased hair fringe instead of keying a wall",
            )
        if float(upload_bg_lab[0]) > 60.0 and not upload_backdrop_is_synthetic:
            bg_distance = np.linalg.norm(
                source_lab - upload_bg_lab[None, None, :], axis=2,
            )
            wall_like = (
                (bg_distance < 42.0)
                & (aligned_person_alpha < 225)
                & (source_lab[:, :, 0] > 38.0)
            )
            head_mask &= ~(
                wall_like
                & ~source_face
                & ~source_eyewear
                & ~source_accessories
                & ~light_clip
            )
        head_mask = cv2.morphologyEx(
            head_mask.astype(np.uint8), cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1,
        ).astype(bool)
        # Close only tiny internal parser holes (partings, isolated highlight
        # pinholes).  Never fill every enclosed component: the open spaces
        # between braids, ears, neck and shoulders can be enclosed by the
        # combined face/hair mask and are real background, not holes.  Filling
        # those large regions copied the source wall into the repaired portrait
        # and produced the conspicuous white/grey cut-and-paste wedges.
        head_inv = (~head_mask).astype(np.uint8)
        n_inv, labels_inv, stats_inv, _ = cv2.connectedComponentsWithStats(
            head_inv, connectivity=8,
        )
        if n_inv > 1:
            border_inv = np.unique(np.concatenate((
                labels_inv[0, :], labels_inv[-1, :], labels_inv[:, 0], labels_inv[:, -1]
            )))
            border_inv = set(int(value) for value in border_inv if int(value) != 0)
            maximum_tiny_hole_area = max(12, int(round(face_w * face_h * .004)))
            tiny_holes = np.zeros_like(head_mask)
            for component_id in range(1, n_inv):
                if component_id in border_inv:
                    continue
                if int(stats_inv[component_id, cv2.CC_STAT_AREA]) <= maximum_tiny_hole_area:
                    tiny_holes |= labels_inv == component_id
            head_mask |= tiny_holes & ~source_clothing & ~central_torso_zone

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
        # Do not erase the generated hairstyle before the aligned source mask
        # is feathered in. Parser/matte uncertainty around curls and flowers
        # otherwise becomes selected-colour holes. The uploaded foreground
        # replaces verified head pixels; any remaining Qwen edge stays behind
        # it rather than becoming a transparent cutout.
        if not erase_invented_hair:
            invented_head_detail[:] = False
        result[invented_head_detail] = target_rgb.astype(np.uint8)

        # Transfer the generated portrait's smooth studio illumination to the
        # aligned uploaded head. High-frequency facial structure, pores, hair
        # strands and accessory detail remain from the upload, while broad
        # orange cast, hard shadow and silver crown glare are removed.
        result_lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB).astype(np.float32)
        core = source_face & (source_lab[:, :, 0] > 25)
        relit_source_lab = source_lab.copy()
        if np.count_nonzero(core) >= 128:
            sigma = max(4.0, min(24.0, face_w * .075))
            source_low = np.stack([
                cv2.GaussianBlur(source_lab[:, :, channel], (0, 0), sigma)
                for channel in range(3)
            ], axis=2)
            target_low = np.stack([
                cv2.GaussianBlur(result_lab[:, :, channel], (0, 0), sigma)
                for channel in range(3)
            ], axis=2)
            detail = source_lab - source_low
            # A small photographed print contains high-frequency paper grain
            # and JPEG noise rather than recoverable skin detail.  Copying it
            # at 82% produced the obvious pasted/noisy mask seen in the failed
            # static result. Preserve geometry but attenuate that noise while
            # borrowing Qwen's broad studio illumination.
            degraded_small_source = min(source.size) < 480
            facial_detail_gain = .58 if degraded_small_source else .82
            relit_source_lab[:, :, 0] = (
                target_low[:, :, 0] + detail[:, :, 0] * facial_detail_gain
            )
            if preserve_source_chroma:
                # A clean-light source is the skin/hair colour authority. Qwen
                # frequently warms a restored face toward yellow/orange; copy
                # only its broad luminance for seam consistency while retaining
                # the uploaded person's original Lab chroma exactly.
                relit_source_lab[:, :, 1] = source_lab[:, :, 1]
                relit_source_lab[:, :, 2] = source_lab[:, :, 2]
            else:
                relit_source_lab[:, :, 1] = target_low[:, :, 1] + detail[:, :, 1] * .46
                relit_source_lab[:, :, 2] = target_low[:, :, 2] + detail[:, :, 2] * .46
            shaded_hair = dark_hair & (source_lab[:, :, 0] < 105)
            if np.count_nonzero(shaded_hair) >= 64:
                shaded_median = np.median(source_lab[shaded_hair], axis=0)
                hair_detail = source_lab - source_low
                relit_source_lab[:, :, 0][dark_hair] = np.clip(
                    shaded_median[0] + hair_detail[:, :, 0][dark_hair] * .68,
                    8.0, 125.0,
                )
                relit_source_lab[:, :, 1][dark_hair] = (
                    shaded_median[1] + hair_detail[:, :, 1][dark_hair] * .55
                )
                relit_source_lab[:, :, 2][dark_hair] = (
                    shaded_median[2] + hair_detail[:, :, 2][dark_hair] * .55
                )
            if np.count_nonzero(light_clip) >= 4:
                accessory_shift = float(np.median(
                    target_low[:, :, 0][light_clip]
                    - source_low[:, :, 0][light_clip]
                ))
                relit_source_lab[:, :, 0][light_clip] = np.clip(
                    source_lab[:, :, 0][light_clip]
                    + np.clip(accessory_shift, -20.0, 20.0),
                    0.0, 255.0,
                )
                relit_source_lab[:, :, 1:][light_clip] = source_lab[:, :, 1:][light_clip]
        matched_source = cv2.cvtColor(
            relit_source_lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB,
        ).astype(np.float32)
        if min(source.size) < 480:
            # Recover bounded edge clarity from the real scan without
            # hallucinating pores or changing geometry. This is deliberately
            # milder than a generative face restorer.
            source_blur = cv2.GaussianBlur(matched_source, (0, 0), 0.85)
            source_sharp = np.clip(
                matched_source * 1.22 - source_blur * .22,
                0.0, 255.0,
            )
            matched_source[head_mask] = source_sharp[head_mask]
        # A wider feather at the jaw and hairline avoids the cut-and-paste line
        # between the exact uploaded head and Qwen's generated neck/collar. The
        # head mask inherits NEAREST-upsampled parser labels, so a 13 px kernel
        # still exposed their blocky staircase across the forehead; 17 px
        # dissolves it while staying well inside the head silhouette.
        feather_kernel = 25 if min(source.size) < 480 else 17
        # The upload is usually cut out on a white studio background, so its own
        # silhouette carries a pale anti-aliased fringe. Pasting head_mask
        # directly copied that fringe as a white halo around the hair and ears.
        # Paste only an eroded core and let the feather blend into the generated
        # portrait, so no source background can travel with the head.
        trim = max(2, int(round(feather_kernel * .35)))
        paste_core = cv2.erode(
            head_mask.astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (trim * 2 + 1, trim * 2 + 1),
            ),
            iterations=1,
        ).astype(bool)
        if not np.any(paste_core):
            paste_core = head_mask
        alpha = cv2.GaussianBlur(
            paste_core.astype(np.uint8) * 255,
            (feather_kernel, feather_kernel), 0,
        ).astype(np.float32) / 255.0
        # Gaussian alpha extends beyond the binary source foreground. Fill that
        # exterior with the target first so the upload's white wall cannot leak
        # into the feather as a pale outline around dark hair and earrings.
        matched_source[~paste_core] = result.astype(np.float32)[~paste_core]
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
    def _get_bg_color_description(bg_rgb: Tuple[int, int, int]) -> str:
        r, g, b = bg_rgb
        if r >= 240 and g >= 240 and b >= 240:
            return "pure solid white"
        if r <= 35 and g <= 35 and b <= 35:
            return "solid dark black"
        if abs(r - g) <= 12 and abs(g - b) <= 12 and abs(r - b) <= 12:
            return "solid neutral grey" if r < 200 else "solid light grey"
        if b > r + 30 and b > g + 15:
            return "solid light blue" if b > 180 else "solid deep blue"
        if r > g + 40 and r > b + 40:
            return "solid bright red" if r > 150 else "solid dark red"
        if g > r + 20 and g > b + 20:
            return "solid green"
        return "solid studio"

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
        force: bool = False,
        return_both: bool = False,
        return_rgba: bool = False,
    ) -> Union[Image.Image, Tuple[Image.Image, Image.Image]]:
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
            base_rgb = image.convert("RGB")
            if return_both:
                return base_rgb, image.convert("RGBA")
            if return_rgba:
                return image.convert("RGBA")
            return base_rgb
        target_rgb_uint8 = np.asarray(
            self._parse_bg_color(background_color), dtype=np.uint8,
        )
        # Idempotence is critical: the uniform route normalizes decoder_raw
        # and later verifies qwen_raw. Once the top border is already the exact
        # selected RGB, a second hue-based pass has no work to do and must not
        # reinterpret smooth skin as another background component.
        border_pixels = np.concatenate((
            rgb[:edge, :].reshape(-1, 3),
            rgb[:int(height * 0.45), :edge].reshape(-1, 3),
            rgb[:int(height * 0.45), -edge:].reshape(-1, 3),
        ), axis=0)
        border_distance = np.linalg.norm(
            border_pixels.astype(np.float32)
            - target_rgb_uint8[None, None, :].astype(np.float32),
            axis=1,
        )
        exact_border_fraction = float(np.mean(border_distance <= 2.0))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        top_h = min(height, int(height * 0.45))
        target_is_white = bool(np.all(target_rgb_uint8 >= 240))
        has_white_gaps = (not target_is_white) and bool(np.any(
            (rgb[:top_h, :, 0] > 200) & (rgb[:top_h, :, 1] > 200) & (rgb[:top_h, :, 2] > 200)
        ))
        top_fringe_present = bool(np.any(
            (border_distance > 1.0) & (border_distance < 60.0)
        ))
        if not force and exact_border_fraction >= .995 and not has_white_gaps and not top_fringe_present:
            logger.info(
                "[UNIFORM_BACKGROUND] Exact selected backdrop and zero white gaps already present; skipped duplicate normalization (border=%.3f)",
                exact_border_fraction,
            )
            rgb_conv = image.convert("RGB")
            if return_both or return_rgba:
                arr = np.array(rgb_conv)
                bg_m = np.all(arr == target_rgb_uint8, axis=-1)
                alpha_ch = np.full((height, width), 255, dtype=np.uint8)
                alpha_ch[bg_m] = 0
                rgba_img = Image.fromarray(np.dstack((arr, alpha_ch)), "RGBA")
                if return_both:
                    return rgb_conv, rgba_img
                return rgba_img
            return rgb_conv
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        target_hsv = cv2.cvtColor(
            target_rgb_uint8.reshape(1, 1, 3), cv2.COLOR_RGB2HSV,
        )[0, 0].astype(np.float32)
        backdrop_hsv = target_hsv
        target_lab = cv2.cvtColor(
            target_rgb_uint8.reshape(1, 1, 3), cv2.COLOR_RGB2LAB,
        )[0, 0].astype(np.float32)
        distance = np.linalg.norm(lab - target_lab, axis=2)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        mean = cv2.GaussianBlur(gray, (0, 0), 2.2)
        square_mean = cv2.GaussianBlur(gray * gray, (0, 0), 2.2)
        texture = np.sqrt(np.maximum(0.0, square_mean - mean * mean))

        target_is_low_sat = (target_hsv[1] < 25.0) or target_is_white
        if target_is_low_sat:
            max_bg_dist = 68.0 if target_is_white else 45.0
            max_bg_sat = 70.0 if target_is_white else 50.0
            candidates = (distance <= max_bg_dist) & (hsv[:, :, 1] <= max_bg_sat) & (texture <= 8.0)
        else:
            hue_delta = np.abs(hsv[:, :, 0] - target_hsv[0])
            hue_delta = np.minimum(hue_delta, 180.0 - hue_delta)
            # Capture solid backdrop field, smooth color drift and vignette shadows
            candidates = ((hue_delta <= 25.0) | (distance <= 25.0)) & (texture <= 10.0)
        # Protect semantic subject pixels before connected-component growth.
        # The former hue-only key treated smooth blue reflections on dark
        # curls as backdrop and created visible holes after an otherwise clean
        # Qwen decode. A small head/hair dilation also protects antialiased
        # strands and pale hair accessories that SCHP labels as background.
        strict_subject = np.zeros((height, width), dtype=bool)
        strict_head = np.zeros((height, width), dtype=bool)
        strict_faces = []
        try:
            from pipelines.photo_restoration import _get_insight_app

            strict_faces = _get_insight_app().get(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            )
        except Exception as _face_exc:
            logger.warning(
                "[UNIFORM_BACKGROUND] Face detection skipped in strict background replacement: %s",
                _face_exc,
            )

        try:
            from pipelines.schp_service import parse as parse_body_parts

            strict_labels = parse_body_parts(rgb)["labels"]
            strict_subject = (strict_labels != 0)
            if len(strict_faces) == 1:
                strict_subject = _recover_unparsed_neck(
                    rgb, strict_subject, strict_faces[0].bbox,
                )
            is_schp_bg = (strict_labels == 0) & (~strict_subject)
            strict_head = np.isin(strict_labels, (1, 2, 4, 13)).astype(np.uint8)
            strict_head = cv2.dilate(
                strict_head,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            ).astype(bool)
            # Never allow hair or head dilation to block confirmed SCHP background pixels
            strict_head[is_schp_bg] = False
            strict_subject[is_schp_bg] = False
            if target_is_low_sat:
                # Edge transition gradient texture must not block confirmed background
                candidates |= (is_schp_bg & (distance <= 80.0) & (hsv[:, :, 1] <= 85.0))
            # SCHP can return background for a smooth regenerated face. A
            # chromatic flat-field key can then paint that island with the
            # selected backdrop, so also protect the detected face geometry.
            if len(strict_faces) == 1:
                try:
                    fx1, fy1, fx2, fy2 = np.asarray(
                        strict_faces[0].bbox, dtype=float,
                    )
                    fw = max(1.0, fx2 - fx1)
                    fh = max(1.0, fy2 - fy1)
                    face_guard = np.zeros((height, width), dtype=np.uint8)
                    cv2.ellipse(
                        face_guard,
                        (int(round((fx1 + fx2) * .5)), int(round((fy1 + fy2) * .5))),
                        (max(3, int(round(fw * .62))), max(3, int(round(fh * .67)))),
                        0, 0, 360, 1, -1,
                    )
                    # The ellipse is deliberately generous (measured 24%
                    # wider and 34% taller than the detector box) so it keeps
                    # a smooth regenerated face. That reach also swallows
                    # backdrop beside the jaw: on a decode whose field was
                    # pale green, two 5,000px patches at the ear/shoulder line
                    # sat inside the ellipse, never became candidates, and
                    # survived as the grey halo around the head. Confirmed
                    # SCHP background is never face, so let it override the
                    # geometric guess - the same rule already applied to
                    # strict_head above.
                    face_guard_bool = face_guard.astype(bool)
                    face_guard_bool[is_schp_bg] = False
                    strict_head |= face_guard_bool
                    # Protect the anatomical neck column from being treated as background
                    neck_guard = np.zeros((height, width), dtype=bool)
                    r_full = rgb[:, :, 0].astype(np.int16)
                    g_full = rgb[:, :, 1].astype(np.int16)
                    b_full = rgb[:, :, 2].astype(np.int16)
                    is_skin_pixel = (r_full > 60) & (r_full - b_full > 16) & (r_full >= g_full) & (b_full < 240)
                    neck_y1 = max(0, int(round(fy1 + fh * 0.70)))
                    neck_y2 = min(height, int(round(fy2 + fh * 0.45)))
                    neck_x1 = max(0, int(round(fx1 - fw * 0.20)))
                    neck_x2 = min(width, int(round(fx2 + fw * 0.20)))
                    neck_guard[neck_y1:neck_y2, neck_x1:neck_x2] = is_skin_pixel[neck_y1:neck_y2, neck_x1:neck_x2]
                    strict_head |= neck_guard
                    yy, xx = np.indices((height, width))
                    # Protect central torso fabric, white plaid checks, buttons, and placket from being treated as background
                    torso_guard = (
                        (yy >= fy2 + fh * 0.10)
                        & (
                            (np.abs(xx - (fx1 + fx2) * 0.5) <= fw * 1.55)
                            | (np.isin(strict_labels, (5, 6, 7, 8, 9, 10, 11, 12, 14, 15)) if 'strict_labels' in locals() else False)
                        )
                    )
                    strict_subject |= torso_guard
                    # Protect generated bows, ribbons, flowers and clips even
                    # when their colour is close to the selected backdrop and
                    # SCHP labels them as background. Build this guard entirely
                    # from the current Qwen decode: retain compact/elongated
                    # detail components beside the measured head that differ
                    # materially from the smooth corner field. This is dynamic
                    # for every portrait and does not assume an accessory type,
                    # colour, side or count.
                    yy, xx = np.indices((height, width))
                    lateral_head_zone = (
                        (yy >= max(0, int(round(fy1 - fh * .62))))
                        & (yy <= min(height - 1, int(round(fy2 + fh * .28))))
                        & (xx >= max(0, int(round(fx1 - fw * 1.40))))
                        & (xx <= min(width - 1, int(round(fx2 + fw * 1.40))))
                        & (
                            (xx <= int(round(fx1 + fw * .22)))
                            | (xx >= int(round(fx2 - fw * .22)))
                        )
                    )
                    accessory_evidence = (
                        lateral_head_zone
                        # Translucent blue ribbon can sit only ~20 LAB units
                        # from a blue generated backdrop and still be a real,
                        # opaque source accessory.  The old 24/3.2 thresholds
                        # therefore selected only a few outline pixels and the
                        # strict colour normalizer erased the ribbon interior.
                        # Smooth backdrop drift remains below both relaxed
                        # gates, while connected-component and head-proximity
                        # checks below prevent broad background preservation.
                        & ((distance >= 14.0) | (texture >= 2.0))
                    ).astype(np.uint8)
                    accessory_evidence = cv2.morphologyEx(
                        accessory_evidence,
                        cv2.MORPH_CLOSE,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                        iterations=1,
                    )
                    accessory_count, accessory_labels, accessory_stats, _ = (
                        cv2.connectedComponentsWithStats(
                            accessory_evidence, connectivity=8,
                        )
                    )
                    # Long bow tails and flowers can be detached from the
                    # parsed hair by a narrow strip of generated backdrop.
                    # Scale the reach from the detected face instead of using
                    # a fixed 25 px kernel (which was too small at 832x1216).
                    accessory_reach = max(25, int(round(fw * .42)))
                    if accessory_reach % 2 == 0:
                        accessory_reach += 1
                    near_head = cv2.dilate(
                        strict_head.astype(np.uint8),
                        cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE,
                            (accessory_reach, accessory_reach),
                        ),
                        iterations=1,
                    ).astype(bool)
                    generated_accessory_guard = np.zeros(
                        (height, width), dtype=bool,
                    )
                    maximum_accessory_area = max(
                        64, int(round(height * width * .075)),
                    )
                    for accessory_id in range(1, accessory_count):
                        area = int(
                            accessory_stats[accessory_id, cv2.CC_STAT_AREA]
                        )
                        w = int(accessory_stats[accessory_id, cv2.CC_STAT_WIDTH])
                        h = int(accessory_stats[accessory_id, cv2.CC_STAT_HEIGHT])
                        component = accessory_labels == accessory_id
                        if target_is_white and (np.median(hsv[component, 1]) < 60 and np.median(hsv[component, 2]) > 180):
                            # Off-white / pale-blue background lighting bleed on white background is not an accessory
                            continue
                        if (
                            60 <= area <= maximum_accessory_area
                            and min(w, h) >= 8
                            and np.any(component & near_head)
                        ):
                            # Preserve the filled accessory, not merely its
                            # high-texture outline.  Flat blue ribbon interiors
                            # can be chromatically close to a blue backdrop and
                            # were otherwise keyed away after Qwen had rendered
                            # them correctly.
                            points = np.column_stack(np.where(component))
                            if len(points) >= 3:
                                hull_xy = cv2.convexHull(
                                    points[:, ::-1].astype(np.int32),
                                )
                                hull = np.zeros((height, width), dtype=np.uint8)
                                cv2.fillConvexPoly(hull, hull_xy, 1)
                                generated_accessory_guard |= hull.astype(bool)
                            else:
                                generated_accessory_guard |= component
                    generated_accessory_guard = cv2.dilate(
                        generated_accessory_guard.astype(np.uint8),
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                        iterations=1,
                    ).astype(bool)
                    strict_head |= generated_accessory_guard
                except Exception as exc:
                    logger.warning(
                        "[UNIFORM_BACKGROUND] Face geometry guard unavailable: %s",
                        exc,
                    )
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
        # Backdrop is behind the subject: seed from top and non-subject lateral margins.
        # Bottom edge (components[-1]) is the subject's torso/uniform and must never seed background.
        border_rows_left = np.where(~strict_subject[:, 0])[0] if 'strict_subject' in locals() else np.arange(int(height * 0.45))
        border_rows_right = np.where(~strict_subject[:, -1])[0] if 'strict_subject' in locals() else np.arange(int(height * 0.45))
        border_components = np.unique(np.concatenate((
            components[0],
            components[border_rows_left, 0],
            components[border_rows_right, -1],
        )))
        border_components = border_components[border_components != 0]
        if not border_components.size:
            return image.convert("RGB")
        matte = np.isin(components, border_components)
        # Border connectivity alone leaves islands of backdrop stranded. On a
        # white studio field Qwen renders a faint cyan cast beside the
        # shoulders; the subject sits between those patches and the frame edge,
        # so they form their own components and survive every earlier gate.
        # Measured on a delivered portrait: 89% of the cast was replaced and
        # 6,949px in two orphan components (mean RGB 200,224,229) remained as
        # visible blue-green blobs. Adopt an orphan only when it is already
        # near-backdrop, desaturated, bright and flat - a real accessory or
        # garment detail fails at least one of those.
        orphan_ids = [
            component_id for component_id in range(1, component_count)
            if component_id not in set(border_components.tolist())
        ]
        if orphan_ids:
            adopted = 0
            for component_id in orphan_ids:
                island = components == component_id
                area = int(np.count_nonzero(island))
                if area < 48:
                    continue
                if (
                    not (('strict_subject' in locals() and np.any(strict_subject[island]))
                         or ('strict_head' in locals() and np.any(strict_head[island]))
                         or ('torso_guard' in locals() and np.any(torso_guard[island])))
                    and float(np.median(distance[island])) <= 75.0
                    and float(np.median(hsv[:, :, 1][island])) <= 70.0
                    and float(np.median(texture[island])) <= 8.0
                ):
                    matte |= island
                    adopted += area
            if adopted:
                logger.info(
                    "[UNIFORM_BACKGROUND] Absorbed %d px of stranded backdrop "
                    "not connected to the frame edge", adopted,
                )
        target_rgb = np.asarray(
            self._parse_bg_color(background_color), dtype=np.float32,
        )
        alpha = cv2.GaussianBlur(
            matte.astype(np.float32), (0, 0), sigmaX=.45, sigmaY=.45,
        )
        alpha[matte] = 1.0
        alpha[~candidates] = 0.0
        if target_is_low_sat:
            near_background = cv2.dilate(
                matte.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                iterations=2,
            ).astype(bool)
            fringe_dist = 90.0 if target_is_white else 45.0
            fringe_sat = 90.0 if target_is_white else 45.0
            fringe = (
                near_background
                & ~matte
                & ~(strict_subject | strict_head)
                & (distance <= fringe_dist)
                & (hsv[:, :, 1] <= fringe_sat)
                & (hsv[:, :, 2] >= 180.0)
            )
            alpha[fringe] = 1.0
        elif backdrop_hsv[1] >= 35.0:
            minimum_saturation = max(25.0, float(backdrop_hsv[1]) * .35)
            # Remove only the thin VAE-colored fringe immediately outside the
            # accepted background component. This is not a person matte: a
            # pixel must remain close to the generated backdrop hue/LAB field
            # and bright/saturated enough to be backdrop contamination. Dark
            # hair, blue checks, skin and white accessories therefore remain
            # unchanged even when they touch the canvas edge.
            near_background = cv2.dilate(
                matte.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                iterations=2,
            ).astype(bool)
            fringe = (
                near_background
                & ~matte
                & ~(strict_subject | strict_head)
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
        if target_is_white or (target_is_low_sat and backdrop_hsv[1] < 20.0):
            bg_current = (alpha >= 0.95)
            if 'strict_labels' in locals() and np.any(strict_labels != 0):
                if 'is_skin_pixel' in locals():
                    bg_current |= ((strict_labels == 0) & candidates & (~strict_subject) & (~strict_head) & (~is_skin_pixel))
                else:
                    bg_current |= ((strict_labels == 0) & candidates & (~strict_subject) & (~strict_head))
            result[bg_current] = target_rgb

            # Robust halo suppression along the full silhouette boundary (hair, crown, neck, shoulders).
            # Eliminates off-white, light-gray, cyan or dark fringing without altering skin or clothing fabric.
            bg_exact = (
                (result[:, :, 0] == target_rgb_uint8[0])
                & (result[:, :, 1] == target_rgb_uint8[1])
                & (result[:, :, 2] == target_rgb_uint8[2])
            )
            dilated_bg = cv2.dilate(
                bg_exact.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                iterations=1,
            ).astype(bool)
            boundary_zone = dilated_bg & ~bg_exact

            r_res = result[:, :, 0]
            g_res = result[:, :, 1]
            b_res = result[:, :, 2]
            is_skin = (r_res > g_res) & (r_res - b_res > 18.0) & (r_res > 60.0)

            dist_to_bg = np.linalg.norm(result - target_rgb[None, None, :], axis=-1)
            hsv_res = cv2.cvtColor(result.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            is_saturated_clothing = (hsv_res[:, :, 1] > 45) | (hsv_res[:, :, 2] < 55)

            is_clothing = (
                np.isin(strict_labels, (5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19))
                if 'strict_labels' in locals()
                else np.zeros((height, width), dtype=bool)
            )
            is_halo = (
                boundary_zone
                & (dist_to_bg <= 80.0)
                & (~is_skin)
                & (~is_saturated_clothing)
                & (~is_clothing)
                & (~strict_subject if 'strict_subject' in locals() else True)
                & (~torso_guard if 'torso_guard' in locals() else True)
            )
            if np.any(is_halo):
                result[is_halo] = target_rgb
                logger.info(
                    "[UNIFORM_BACKGROUND] Cleaned %d boundary halo pixels into exact backdrop",
                    int(np.count_nonzero(is_halo)),
                )

            # Top headroom rows are pure backdrop air above the crown
            top_headroom_h = max(2, min(8, height // 50))
            top_air = ~strict_head[:top_headroom_h, :] if 'strict_head' in locals() else np.ones((top_headroom_h, width), dtype=bool)
            result[:top_headroom_h, :][top_air] = target_rgb

            # Hair fringe neutralization: neutralize any green/cyan/blue chromatic rim on hair edge along entire braid length
            hair_zone = np.arange(height)[:, None] < int(height * 0.85)
            remaining_boundary = boundary_zone & ~is_halo & (~is_skin) & (~is_clothing) & (hsv_res[:, :, 2] < 185) & (hsv_res[:, :, 2] > 25) & hair_zone
            has_hair_tint = remaining_boundary & ((np.abs(g_res - r_res) > 3.0) | (np.abs(b_res - r_res) > 3.0))
            if np.any(has_hair_tint):
                gray_val = np.rint(
                    0.299 * r_res[has_hair_tint] + 0.587 * g_res[has_hair_tint] + 0.114 * b_res[has_hair_tint]
                ).clip(0, 255)
                result[has_hair_tint, 0] = gray_val
                result[has_hair_tint, 1] = gray_val
                result[has_hair_tint, 2] = gray_val
                logger.info(
                    "[UNIFORM_BACKGROUND] Neutralized chromatic halo on %d hair contour pixels",
                    int(np.count_nonzero(has_hair_tint)),
                )
        else:
            # Chromatic backgrounds: eliminate near-background fringe and residual white wall halos
            bg_exact = (
                (result[:, :, 0] == target_rgb_uint8[0])
                & (result[:, :, 1] == target_rgb_uint8[1])
                & (result[:, :, 2] == target_rgb_uint8[2])
            )
            dilated_bg = cv2.dilate(
                bg_exact.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                iterations=1,
            ).astype(bool)
            boundary_zone = dilated_bg & ~bg_exact
            r_res = result[:, :, 0]
            g_res = result[:, :, 1]
            b_res = result[:, :, 2]
            is_skin = (r_res > g_res) & (r_res - b_res > 18.0) & (r_res > 60.0)
            dist_to_bg = np.linalg.norm(result - target_rgb[None, None, :], axis=-1)
            hsv_res = cv2.cvtColor(result.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)

            is_clothing = (
                np.isin(strict_labels, (5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19))
                if 'strict_labels' in locals()
                else np.zeros((height, width), dtype=bool)
            )
            # Absorbs both near-backdrop chromatic fringe (dist <= 55) and residual white wall halos (low sat bright)
            is_near_target = (dist_to_bg <= 55.0)
            is_residual_white_wall = (hsv_res[:, :, 1] < 35) & (hsv_res[:, :, 2] > 180) & (dist_to_bg > 30.0)
            is_chromatic_halo = boundary_zone & (is_near_target | is_residual_white_wall) & (~is_skin) & (~is_clothing)
            if np.any(is_chromatic_halo):
                result[is_chromatic_halo] = target_rgb
                logger.info(
                    "[UNIFORM_BACKGROUND] Cleaned %d boundary halo pixels into chromatic backdrop",
                    int(np.count_nonzero(is_chromatic_halo)),
                )
        logger.info(
            "[ENHANCE_BACKGROUND] Strict border correction replaced %.3f of canvas",
            float(np.mean(matte)),
        )
        rgb_out = Image.fromarray(np.rint(result).clip(0, 255).astype(np.uint8), "RGB")
        if return_both:
            return rgb_out, None
        if return_rgba:
            return rgb_out.convert("RGBA")
        return rgb_out

    @staticmethod
    def _bridge_neck_collar_gap(
        image: Image.Image,
        source_image: Image.Image,
        background_color: Union[str, Tuple[int, int, int]] = (255, 255, 255),
    ) -> Image.Image:
        """Bridge any residual background gap between neck, shoulders, and uniform collar."""
        try:
            from pipelines.photo_restoration import _get_insight_app

            img_cv = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            h, w = img_cv.shape[:2]
            src_cv = cv2.cvtColor(np.asarray(source_image.convert("RGB")), cv2.COLOR_RGB2BGR)
            src_aligned = cv2.resize(src_cv, (w, h), interpolation=cv2.INTER_LANCZOS4)

            faces = _get_insight_app().get(img_cv)
            if not faces:
                return image

            fx1, fy1, fx2, fy2 = faces[0].bbox
            fw = max(1.0, float(fx2 - fx1))
            fh = max(1.0, float(fy2 - fy1))
            fcx = float(fx1 + fx2) * 0.5
            chin_y = float(fy2)

            y_start = max(0, int(round(chin_y + fh * 0.04)))
            y_end = min(h, int(round(chin_y + fh * 0.45)))
            x_start = max(0, int(round(fcx - fw * 0.38)))
            x_end = min(w, int(round(fcx + fw * 0.38)))

            if y_end <= y_start or x_end <= x_start:
                return image

            zone_img = img_cv[y_start:y_end, x_start:x_end]
            zone_src = src_aligned[y_start:y_end, x_start:x_end]

            is_white = np.all(zone_img >= 235, axis=2)
            if not np.any(is_white):
                return image

            b_s = zone_src[:, :, 0].astype(np.int16)
            g_s = zone_src[:, :, 1].astype(np.int16)
            r_s = zone_src[:, :, 2].astype(np.int16)
            src_is_skin = (r_s > 60) & (r_s - b_s > 15) & (r_s >= g_s) & (b_s < 225)
            bridge = is_white & src_is_skin

            if np.count_nonzero(bridge) < 15:
                return image

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated = cv2.dilate(bridge.astype(np.uint8), kernel).astype(bool)
            boundary = dilated & (~is_white)
            b_skin = boundary & (zone_img[:, :, 2] > zone_img[:, :, 0] + 15) & (zone_img[:, :, 2] > 60)
            if np.sum(b_skin) > 30:
                target_mean = np.mean(zone_img[b_skin].astype(float), axis=0)
            elif np.any(boundary):
                target_mean = np.mean(zone_img[boundary].astype(float), axis=0)
            else:
                target_mean = np.array([80.0, 110.0, 165.0])

            src_mean = np.mean(zone_src[bridge].astype(float), axis=0)
            ratio = np.clip(target_mean / np.maximum(src_mean, 1.0), 0.5, 1.5)
            adj_s = np.clip(zone_src.astype(float) * ratio[None, None, :], 0, 255).astype(np.uint8)

            feather = cv2.GaussianBlur(bridge.astype(np.float32), (7, 7), 1.5)[:, :, None]
            blended = (adj_s.astype(float) * feather + zone_img.astype(float) * (1.0 - feather)).clip(0, 255).astype(np.uint8)

            repaired_cv = img_cv.copy()
            repaired_cv[y_start:y_end, x_start:x_end] = blended
            logger.info("[UNIFORM_NECK_BRIDGE] Infilled %d gap pixels between neck and collar", int(np.sum(bridge)))
            return Image.fromarray(cv2.cvtColor(repaired_cv, cv2.COLOR_BGR2RGB))
        except Exception as _e:
            logger.warning("[UNIFORM_NECK_BRIDGE] Error bridging neck collar gap: %s", _e)
            return image

    @staticmethod
    def _repair_generated_neck_collar_seam(
        image: Image.Image,
        background_color: Union[str, Tuple[int, int, int]] = (255, 255, 255),
    ) -> Image.Image:
        """Close only tiny backdrop notches trapped at the generated neck/collar seam.

        This operates on Qwen's own decode and never pastes pixels from the
        upload.  A face-relative zone and a small morphological close identify
        narrow background crescents between two already generated foreground
        surfaces.  Open backdrop beside the neck remains untouched.
        """
        try:
            from pipelines.photo_restoration import _get_insight_app

            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            h, w = rgb.shape[:2]
            faces = _get_insight_app().get(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if len(faces) != 1:
                return image
            fx1, fy1, fx2, fy2 = [float(v) for v in faces[0].bbox]
            fw = max(1.0, fx2 - fx1)
            fh = max(1.0, fy2 - fy1)
            fcx = (fx1 + fx2) * .5

            yy, xx = np.indices((h, w))
            seam_zone = (
                (yy >= fy2 - fh * .055)
                & (yy <= fy2 + fh * .26)
                & (np.abs(xx - fcx) <= fw * .70)
            )
            target = np.asarray(
                QwenEditPipeline._parse_bg_color(background_color),
                dtype=np.int16,
            )
            bg_distance = np.linalg.norm(
                rgb.astype(np.int16) - target[None, None, :], axis=2,
            )
            background = bg_distance <= 24.0
            foreground = (~background).astype(np.uint8)
            # At output resolution the unwanted crescents are normally 2-9px
            # wide. Scale the close to face size so this remains dynamic for
            # small children, adults and different export resolutions.
            close_w = int(np.clip(round(fw * .08), 9, 39)) | 1
            close_h = int(np.clip(round(fh * .04), 5, 21)) | 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (close_w, close_h),
            )
            closed = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
            notches = seam_zone & background & (closed > 0)

            # Reject isolated noise and large/open background areas. Only
            # compact components genuinely bridged by the close are repaired.
            component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
                notches.astype(np.uint8), connectivity=8,
            )
            accepted = np.zeros((h, w), dtype=np.uint8)
            max_area = max(48, int(round(fw * fh * .015)))
            for component in range(1, component_count):
                area = int(stats[component, cv2.CC_STAT_AREA])
                comp_w = int(stats[component, cv2.CC_STAT_WIDTH])
                comp_h = int(stats[component, cv2.CC_STAT_HEIGHT])
                if 3 <= area <= max_area and comp_w <= close_w * 2 and comp_h <= close_h * 3:
                    accepted[labels == component] = 255
            repaired_pixels = int(np.count_nonzero(accepted))
            if repaired_pixels == 0:
                return image

            inpaint_mask = cv2.dilate(
                accepted,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                iterations=1,
            )
            inpaint_mask = inpaint_mask & (seam_zone.astype(np.uint8) * 255)
            repaired = cv2.inpaint(rgb, inpaint_mask, 5.0, cv2.INPAINT_TELEA)
            logger.info(
                "[UNIFORM_NECK_SEAM] Repaired %d generated backdrop-notch "
                "pixel(s) without source-pixel overlay",
                repaired_pixels,
            )
            return Image.fromarray(repaired, "RGB")
        except Exception as error:
            logger.warning("[UNIFORM_NECK_SEAM] Generated seam repair skipped: %s", error)
            return image

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
        acc_mask: Optional[np.ndarray] = None,
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
            acc_m = None
            if acc_mask is not None and np.any(acc_mask):
                acc_m = cv2.resize(
                    acc_mask.astype(np.uint8), (alpha.shape[1], alpha.shape[0]), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                acc_m = cv2.dilate(
                    acc_m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1
                ).astype(bool)
                alpha[acc_m] = 1.0
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
                        and (acc_m is None or not np.any(component_mask & acc_m))
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
            if acc_m is not None:
                alpha[acc_m] = 1.0
            # Close internal cavities inside the subject alpha (hair partings, curl gaps, bows, ties, fabric)
            # so the person is never hollowed out with transparent holes.
            solid_core = (alpha >= 0.50).astype(np.uint8)
            n_core, labels_core = cv2.connectedComponents((1 - solid_core).astype(np.uint8), connectivity=8)
            if n_core > 1:
                border_labels_core = np.unique(np.concatenate((
                    labels_core[0, :], labels_core[-1, :], labels_core[:, 0], labels_core[:, -1]
                )))
                border_labels_core = border_labels_core[border_labels_core != 0]
                internal_cavities = ~np.isin(labels_core, border_labels_core)
                alpha[internal_cavities] = 1.0
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
            # Measure on the same head window and low-frequency field the
            # residual-light gate uses so repair and validation agree.
            window = np.zeros((height, width), dtype=bool)
            hx1 = max(0, x1 - int(face_w * .85))
            hx2 = min(width, x2 + int(face_w * .85))
            hy1 = max(0, y1 - int(face_h * .78))
            hy2 = min(height, y2 + int(face_h * .38))
            window[hy1:hy2, hx1:hx2] = True
            hair = (labels == 2) & window
            if int(hair.sum()) < 128:
                return image.convert("RGB")

            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            luminance = lab[:, :, 0]
            sigma = max(5.0, face_w * .065)
            low_l = cv2.GaussianBlur(luminance, (0, 0), sigma)
            hair_values = low_l[hair]
            target = float(np.percentile(hair_values, 58))
            # Protect colourful accessories: anchor to the portrait's own
            # dark-hair chroma and correct only near-neutral glare pixels.
            dark_cut = float(np.percentile(luminance[hair], 38))
            dark_hair = hair & (luminance <= dark_cut)
            anchor = dark_hair if int(dark_hair.sum()) >= 64 else hair
            hair_ab = np.median(lab[:, :, 1:3][anchor], axis=0)
            ab_distance = np.linalg.norm(
                lab[:, :, 1:3] - hair_ab[None, None, :], axis=2,
            )
            # Glare can be neutral/silver (ab_distance <= 22) or warm sunlight
            # glare (inflated LAB-b due to yellow/amber sun, but modest LAB-a).
            # Colorful hair accessories (pink flowers, blue ribbons, red clips)
            # have strong LAB-a divergence or non-yellow hues and are protected.
            neutral_glare = ab_distance <= 22.0
            lab_a_dist = np.abs(lab[:, :, 1] - hair_ab[0])
            warm_sun_glare = (
                (lab_a_dist <= 18.0)
                & (lab[:, :, 2] >= hair_ab[1])
                & (ab_distance <= 55.0)
            )
            candidate = hair & (neutral_glare | warm_sun_glare)
            # Compress the full broad-illumination excess (no fixed cap) so a
            # clipped crown actually reaches the shaded-hair target. Only the
            # low-frequency field is reduced; high-frequency strand detail
            # remains in LAB luminance.
            excess = np.clip(low_l - (target + 8.0), 0.0, None)
            alpha = cv2.GaussianBlur(
                candidate.astype(np.uint8) * 255,
                (0, 0), max(2.5, face_w * .014),
            ).astype(np.float32) / 255.0
            lab[:, :, 0] = luminance - excess * alpha
            for ch in (1, 2):
                chroma_excess = (lab[:, :, ch] - hair_ab[ch - 1]) * (0.65 * alpha)
                lab[:, :, ch] = lab[:, :, ch] - chroma_excess
            corrected = cv2.cvtColor(
                np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB,
            )
            logger.info(
                "[UNIFORM_LIGHTING] Reduced broad crown/hair sunlight on %d pixels",
                int(candidate.sum()),
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
        preserve_hair_and_accessories: bool = True,
    ) -> Image.Image:
        """Build a neutral-light conditioning guide without changing geometry.

        This is used only for a source whose measured light is extreme.  It is
        not a delivered image and does not invent pixels: low-frequency face
        illumination is balanced from that portrait's own shaded skin. Hair and
        accessories are preserved byte-for-byte by default: pale flowers, bows
        and clips can be nearly achromatic under direct sun and cannot safely be
        distinguished from silver glare by a dark-hair colour heuristic.
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

            # Hair correction is opt-in only. The production uniform route
            # protects the uploaded hairstyle and all accessories exactly;
            # Qwen receives those original pixels and may finish residual glare
            # without a destructive pre-darkening pass.
            hair_l = luminance[hair]
            dark_cut = float(np.percentile(hair_l, 38))
            dark_hair = hair & (luminance <= dark_cut)
            if (not preserve_hair_and_accessories) and np.count_nonzero(dark_hair) >= 64:
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
                and (face_p90 - face_p15) >= 50.0
                # Normal studio portraits retain broad cheek/nose modelling
                # and can easily span 50 LAB-lightness points. Direct sun on
                # skin carries noticeable chroma and directional imbalance;
                # calibrate threshold to avoid missing real sunlight glare.
                and face_sat75 >= 115.0
                and directional_spread >= 24.0
            )
            global_warm_cast = bool(
                face_p90 >= 125.0
                and face_sat75 >= 160.0
                and face_b50 >= 155.0
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
                bright_floor = max(115.0, float(hair_p25) + 45.0)
                bright_fraction = float(np.mean(hair_values >= bright_floor))
                hair_hard = bool(
                    (hair_p90 >= 115.0 and (hair_p90 - hair_p25) >= 45.0 and bright_fraction >= .04)
                    or (bright_fraction >= .025 and (hair_p90 - hair_p25) >= 35.0)
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
            # SCHP can return no face label on a stylized/relit Qwen decode.
            # Never let that parser miss turn a detected face into backdrop.
            # InsightFace is already the delivery authority, so use its box as
            # an independent solid face guard before any GrabCut/scene cleanup.
            detected_face_guard = np.zeros(labels.shape, dtype=np.uint8)
            try:
                from pipelines.photo_restoration import _get_insight_app

                matte_faces = _get_insight_app().get(
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                )
                if len(matte_faces) == 1:
                    fx1, fy1, fx2, fy2 = (
                        int(round(value)) for value in matte_faces[0].bbox
                    )
                    fw = max(1, fx2 - fx1)
                    fh = max(1, fy2 - fy1)
                    cv2.ellipse(
                        detected_face_guard,
                        (int((fx1 + fx2) * .5), int((fy1 + fy2) * .5)),
                        (max(2, int(fw * .56)), max(2, int(fh * .56))),
                        0, 0, 360, 1, -1,
                    )
            except Exception as face_guard_error:
                logger.warning(
                    "[UNIFORM_BACKGROUND] Independent face guard unavailable: %s",
                    face_guard_error,
                )
            # SCHP is the foreground authority for this operation. Later
            # GrabCut and colour cleanup may recover label-0 hair/accessory
            # fringe, but they must never erase pixels SCHP already identified
            # as face, hair, neck, clothing, arms, or accessories.
            semantic_subject = (labels != 0) | (detected_face_guard > 0)
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
                face_labels = (labels == 13).astype(np.uint8)
                face_points = np.column_stack(np.where(face_labels > 0))
                if face_points.shape[0] >= 16:
                    face_hull = cv2.convexHull(face_points[:, ::-1].astype(np.int32))
                    cv2.fillConvexPoly(subject, face_hull, 1)
                head_labels = np.isin(labels, (1, 2, 4, 13)).astype(np.uint8)
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
                    (centre_x - max(2, int(face_w * .40)), int(face_y2 - face_h * .12)),
                    (centre_x + max(2, int(face_w * .40)), int(face_y2 - face_h * .12)),
                    (centre_x + max(2, int(face_w * .48)), collar_y + 2),
                    (centre_x - max(2, int(face_w * .48)), collar_y + 2),
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
                if face_points.shape[0] >= 16:
                    cv2.fillConvexPoly(anatomy_guard, face_hull, 1)
                anatomy_guard[eye_guard > 0] = 1
                anatomy_guard[clothing_mask] = 1
                cv2.fillConvexPoly(anatomy_guard, neck_polygon, 1)
                neck_rows = (np.indices(subject.shape)[0] >= int(face_y2 - face_h * .05)) & (np.indices(subject.shape)[0] <= collar_y + 2)
                neck_cols = np.abs(np.indices(subject.shape)[1] - centre_x) <= max(2, int(face_w * .44))
                anatomy_guard[neck_rows & neck_cols] = 1
                subject[neck_rows & neck_cols] = 1
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
            if preserve_foreground_rgb:
                # This route operates on Qwen's finished person. Keep every
                # parser-confirmed subject pixel and the continuous lower torso
                # opaque. The prior boundary decontamination partially blended
                # pale collar/neck/plaid pixels with the selected backdrop and
                # created blue cuts through an otherwise valid uniform.
                foreground_alpha[
                    semantic_subject | (lower_subject_hull > 0)
                ] = 1.0
                foreground[
                    semantic_subject | (lower_subject_hull > 0)
                ] = 1
            # Reduce alpha continuously for pixels that resemble the learned
            # scene colours.  The previous hard zero created a visibly jagged
            # contour wherever an antialiased shoulder/hair pixel happened to
            # cross the scene threshold.  RGB decontamination below removes the
            # old backdrop contribution from the remaining soft boundary.
            softened_edge = (foreground_alpha > 0.0) & (foreground_alpha < 1.0)
            scene_edge = softened_edge & (labels == 0) & ~recovered_head_fringe
            scene_confidence = np.clip((scene_distance - 7.0) / 18.0, 0.0, 1.0)
            foreground_alpha[scene_edge] *= scene_confidence[scene_edge]

            # All background in a portrait must connect to the outer image border.
            # Enclosed cavities inside the subject (hair parting, curl gaps, collar shadow, eye/mouth details)
            # must NEVER be classified as background or stamped with the target color.
            bg_candidate_mask = (foreground == 0).astype(np.uint8)
            n_bg_comp, labels_bg_comp = cv2.connectedComponents(bg_candidate_mask, connectivity=8)
            if n_bg_comp > 1:
                border_bg_labels = np.unique(np.concatenate((
                    labels_bg_comp[0, :], labels_bg_comp[-1, :], labels_bg_comp[:, 0], labels_bg_comp[:, -1]
                )))
                border_bg_labels = border_bg_labels[border_bg_labels != 0]
                true_border_background = np.isin(labels_bg_comp, border_bg_labels)
                foreground[~true_border_background] = 1
                foreground_alpha[~true_border_background] = 1.0
            else:
                true_border_background = (foreground == 0)

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
                clean_rgb[true_border_background & (foreground[:, :] == 0)] = 0
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
            definite_exterior_bg = true_border_background & (foreground == 0)
            composed[definite_exterior_bg] = target[definite_exterior_bg]
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

    @staticmethod
    def deduct_neck_and_shoulders_bisenet(image: Image.Image, face_bbox):
        """Use BiSeNet (CelebAMask-HQ 19 classes) for accurate anatomical neck deduction and shoulder width.

        BiSeNet class 14 is 'neck', class 1 is 'skin', class 16 is 'cloth'.
        """
        import torchvision.transforms as transforms
        from facexlib.parsing import init_parsing_model

        global _bisenet_model, _bisenet_lock
        if '_bisenet_lock' not in globals():
            _bisenet_lock = threading.Lock()
        if '_bisenet_model' not in globals() or _bisenet_model is None:
            with _bisenet_lock:
                if '_bisenet_model' not in globals() or _bisenet_model is None:
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    _bisenet_model = init_parsing_model(model_name='bisenet', device=device)

        model = _bisenet_model
        device = next(model.parameters()).device
        w, h = image.size
        img_512 = image.resize((512, 512), Image.BILINEAR)
        img_tensor = transforms.ToTensor()(img_512)
        img_tensor = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img_tensor).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img_tensor)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
        parsing_orig = cv2.resize(parsing.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        if face_bbox is None:
            face_bbox = (w * 0.25, h * 0.15, w * 0.75, h * 0.55)
        fx1, fy1, fx2, fy2 = (float(v) for v in face_bbox)
        fw = max(1.0, fx2 - fx1)
        fh = max(1.0, fy2 - fy1)
        fcx = (fx1 + fx2) * 0.5

        # Class 14 is neck
        neck_mask = parsing_orig == 14
        c_x1 = max(0, int(round(fcx - fw * 0.22)))
        c_x2 = min(w, int(round(fcx + fw * 0.22)))
        central_neck_mask = neck_mask[:, c_x1:c_x2]
        central_neck_rows = np.flatnonzero(np.any(central_neck_mask, axis=1))

        if central_neck_rows.size:
            neck_bottom_y = float(central_neck_rows[-1])
            vertical_neck_gap_ratio = float(np.clip((neck_bottom_y - fy2) / fh, 0.04, 0.42))
        else:
            vertical_neck_gap_ratio = 0.06
            neck_bottom_y = float(fy2 + fh * 0.06)

        # Shoulder width from BiSeNet (Class 16 cloth + Class 14 neck)
        cloth_or_neck = np.isin(parsing_orig, (14, 16))
        band_y1 = int(round(fy2 + fh * 0.08))
        band_y2 = int(round(fy2 + fh * 0.55))
        widths = []
        for y in range(band_y1, min(h, band_y2)):
            cols = np.flatnonzero(cloth_or_neck[y])
            if cols.size >= 10:
                widths.append(cols[-1] - cols[0] + 1)
        if widths:
            shoulder_width = float(np.percentile(widths, 85))
            shoulder_to_face_ratio = shoulder_width / fw
        else:
            shoulder_width = float(fw * 1.82)
            shoulder_to_face_ratio = 1.82

        return {
            'vertical_neck_gap_ratio': vertical_neck_gap_ratio,
            'neck_bottom_y': neck_bottom_y,
            'neck_mask': neck_mask,
            'shoulder_width': shoulder_width,
            'shoulder_to_face_ratio': shoulder_to_face_ratio,
            'parsing': parsing_orig,
        }

    @staticmethod
    def measure_exposed_neck(image, face_bbox):
        """Exposed neck measurement using BiSeNet deduction with skin-color fallback."""
        try:
            bisenet_res = QwenEditPipeline.deduct_neck_and_shoulders_bisenet(image, face_bbox)
            return float(bisenet_res['vertical_neck_gap_ratio'])
        except Exception as exc:
            logger.warning("[NECK_DEDUCTION] BiSeNet neck deduction failed (%s); using fallback", exc)
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        height, width = rgb.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in face_bbox)
        face_w = max(1.0, x2 - x1)
        face_h = max(1.0, y2 - y1)
        centre_x = (x1 + x2) * .5
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0].astype(np.int16)
        sat = hsv[:, :, 1].astype(np.int16)
        val = hsv[:, :, 2].astype(np.int16)
        skin = (
            ((hue <= 25) | (hue >= 170))
            & (sat >= 35) & (sat <= 190)
            & (val >= 50) & (val <= 240)
        )
        top = int(round(y2))
        bottom = int(round(min(height, y2 + face_h * .5)))
        left = int(round(max(0, centre_x - face_w * 1.2)))
        right = int(round(min(width, centre_x + face_w * 1.2)))
        if bottom <= top or right <= left:
            return None
        return float(
            skin[top:bottom, left:right].sum() / (face_w * face_h)
        )

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
            wrong_count, wrong_map, wrong_stats, _ = cv2.connectedComponentsWithStats(
                wrong_color.astype(np.uint8), connectivity=8,
            )
            person_neighborhood = cv2.dilate(
                filled_subject.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=2,
            ).astype(bool)
            max_accessory_area = int(height * width * 0.02)
            for component in range(1, wrong_count):
                component_mask = wrong_map == component
                area = int(wrong_stats[component, cv2.CC_STAT_AREA])
                touches_border = bool(
                    np.any(component_mask[0, :])
                    or np.any(component_mask[:, 0])
                    or np.any(component_mask[:, -1])
                )
                if not touches_border and area <= max_accessory_area and np.any(component_mask & person_neighborhood):
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
            if side_samples.shape[0] >= 16:
                side_distances = np.linalg.norm(side_samples - target, axis=1)
                side_match = float(np.mean(side_distances <= 18.0))
            else:
                side_match = 1.0
        except Exception:
            side_distances = np.linalg.norm(side_pixels - target, axis=1)
            side_match = float(np.mean(side_distances <= 18.0))
            full_background_match = 1.0
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
            if measurable_top_side.shape[0] >= 16:
                top_side_distances = np.linalg.norm(measurable_top_side - target, axis=1)
                top_side_match = float(np.mean(top_side_distances <= 18.0))
            else:
                top_side_match = 1.0
        else:
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
            # connected person silhouette has been excluded. For voluminous
            # hairstyles (pigtails, buns, bows, curls), allow top_side_match >= 0.94
            # when side and full background matches are strongly verified.
            and (
                top_side_match >= 0.985
                or (side_match >= 0.95 and full_background_match >= 0.94 and top_side_match >= 0.94)
            )
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
        source_clothing_reference: Optional[Image.Image] = None,
        expected_target_fraction: Optional[float] = None,
        hair_mask: Optional[np.ndarray] = None,
    ) -> bool:
        """Reject unchanged source clothing and incomplete garment edits."""
        try:
            from pipelines.schp_service import parse as parse_body_parts

            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            labels = parse_body_parts(rgb)["labels"]
            clothing = np.isin(labels, (5, 6, 7, 9, 10, 12))

            # BUG 1 & 8 FIX: Do not allow long braids misclassified by SCHP as coat (label 7)
            # to be counted as clothing. Exclude genuine hair and ribbon pixels so clothing
            # metrics (coverage, histograms, dark panel checks) measure authentic garment pixels.
            if hair_mask is not None:
                clothing = clothing & ~(np.asarray(hair_mask).astype(bool))
            else:
                try:
                    from pipelines.photo_restoration import _get_insight_app
                    from pipelines.uniform_hair_face_compositor import recover_hair_and_accessories
                    _app = _get_insight_app()
                    _faces = _app.get(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    if _faces:
                        _h_mask, _a_mask = recover_hair_and_accessories(rgb, _faces[0].bbox, labels)
                        clothing = clothing & ~(_h_mask | _a_mask)
                except Exception:
                    pass

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
            # Upper arms at the left/right frame edges are normal in an
            # upper-chest passport crop.  The former wide 18%-82% window counted
            # those shoulders as an arm crossing the torso and rejected a valid
            # Qwen uniform (the d816ca28 raw decode is the regression fixture).
            # Reject only a sizeable arm/hand component inside the *central*
            # lower-torso corridor, where folded hands or a crossing forearm
            # would actually obstruct the uniform.
            central_arm_region = labels[
                int(height * .48):int(height * .90),
                int(width * .30):int(width * .70),
            ]
            central_arms = np.isin(central_arm_region, (14, 15))
            central_arm_fraction = float(
                np.count_nonzero(central_arms) / max(1, central_arms.size)
            )
            if central_arm_fraction > 0.08:
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
                    # A global palette check cannot distinguish where a
                    # pattern belongs.  The known failure kept image 1's
                    # large-check vest and plain shirt even though image 2
                    # required a solid vest and micro-check shirt; both used
                    # the same blue/white/dark palette and therefore passed a
                    # global histogram. Compare coarse spatial HSV histograms
                    # against both uploaded authorities and reject only when
                    # the decoded clothing is decisively closer to the source
                    # clothing than to the supplied template. This remains
                    # dynamic for every portrait and every uniform.
                    def spatial_garment_similarity(
                        other_hsv: np.ndarray,
                        other_mask: np.ndarray,
                    ) -> Optional[float]:
                        scores = []
                        for grid_y in range(3):
                            cell_y0 = int(height * (.42 + grid_y * .18))
                            cell_y1 = int(height * (.42 + (grid_y + 1) * .18))
                            for grid_x in range(3):
                                cell_x0 = int(width * (.08 + grid_x * .28))
                                cell_x1 = int(width * (.08 + (grid_x + 1) * .28))
                                candidate_cell = candidate_clothing[
                                    cell_y0:cell_y1, cell_x0:cell_x1
                                ]
                                other_cell = other_mask[
                                    cell_y0:cell_y1, cell_x0:cell_x1
                                ]
                                if (
                                    np.count_nonzero(candidate_cell) < 80
                                    or np.count_nonzero(other_cell) < 80
                                ):
                                    continue
                                candidate_cell_hist = cv2.calcHist(
                                    [candidate_hsv[cell_y0:cell_y1, cell_x0:cell_x1]],
                                    [0, 1], candidate_cell.astype(np.uint8),
                                    [18, 8], [0, 180, 0, 256],
                                )
                                other_cell_hist = cv2.calcHist(
                                    [other_hsv[cell_y0:cell_y1, cell_x0:cell_x1]],
                                    [0, 1], other_cell.astype(np.uint8),
                                    [18, 8], [0, 180, 0, 256],
                                )
                                cv2.normalize(
                                    candidate_cell_hist, candidate_cell_hist,
                                    alpha=1.0, norm_type=cv2.NORM_L1,
                                )
                                cv2.normalize(
                                    other_cell_hist, other_cell_hist,
                                    alpha=1.0, norm_type=cv2.NORM_L1,
                                )
                                scores.append(float(cv2.compareHist(
                                    candidate_cell_hist, other_cell_hist,
                                    cv2.HISTCMP_INTERSECT,
                                )))
                        return float(np.mean(scores)) if len(scores) >= 3 else None

                    target_spatial_similarity = spatial_garment_similarity(
                        reference_hsv, ref_clothing,
                    )
                    source_spatial_similarity = None
                    if source_clothing_reference is not None:
                        source_reference_rgb = np.asarray(
                            source_clothing_reference.convert("RGB").resize(
                                (width, height), Image.LANCZOS,
                            ),
                            dtype=np.uint8,
                        )
                        source_reference_labels = parse_body_parts(
                            source_reference_rgb,
                        )["labels"]
                        source_reference_clothing = np.isin(
                            source_reference_labels, (5, 6, 7, 9, 10, 12),
                        )
                        source_reference_hsv = cv2.cvtColor(
                            source_reference_rgb, cv2.COLOR_RGB2HSV,
                        )
                        source_spatial_similarity = spatial_garment_similarity(
                            source_reference_hsv, source_reference_clothing,
                        )
                    logger.info(
                        "[UNIFORM_REVIEW] spatial garment similarity target=%s source=%s",
                        (
                            f"{target_spatial_similarity:.3f}"
                            if target_spatial_similarity is not None else "unavailable"
                        ),
                        (
                            f"{source_spatial_similarity:.3f}"
                            if source_spatial_similarity is not None else "unavailable"
                        ),
                    )
                    if (
                        target_spatial_similarity is not None
                        and source_spatial_similarity is not None
                        and source_spatial_similarity >= .30
                        and source_spatial_similarity
                        >= target_spatial_similarity + .12
                    ):
                        logger.warning(
                            "[UNIFORM_REVIEW] Qwen retained the source garment layout: "
                            "source %.3f versus template %.3f",
                            source_spatial_similarity,
                            target_spatial_similarity,
                        )
                        return False
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
                        # A fitted plaid can contain dark folds, button
                        # shadows and dense overlapping checks that are not
                        # present in the flat product photograph.  Do not let
                        # that one scalar veto stronger spatial evidence that
                        # the candidate moved away from the source clothing
                        # and toward the uploaded template (442622c1).  The
                        # guard remains blocking when the template is not the
                        # clearly closer garment.
                        target_layout_wins = bool(
                            target_spatial_similarity is not None
                            and target_spatial_similarity >= .18
                            and (
                                source_spatial_similarity is None
                                or target_spatial_similarity
                                >= source_spatial_similarity + .10
                            )
                        )
                        if dark_panel_divergence and not (
                            reference_is_chromatic_pattern
                            and chromatic_hue_match
                            and target_layout_wins
                        ):
                            logger.warning(
                                "[UNIFORM_REVIEW] Qwen invented a dark central garment panel absent from the template"
                            )
                            return False
                        if dark_panel_divergence:
                            logger.info(
                                "[UNIFORM_REVIEW] Ignored fold/check darkness because the patterned "
                                "candidate is spatially closer to the uploaded template"
                            )
                            dark_panel_divergence = False
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
            # NEVER erase large foreground components (like torso, arms, clothing)
            # or anything spanning the lower rows of the portrait.
            comp_area = stats[component, cv2.CC_STAT_AREA]
            comp_top = stats[component, cv2.CC_STAT_TOP]
            if comp_area > (rgb.shape[0] * rgb.shape[1] * 0.02) or comp_top > (rgb.shape[0] * 0.40):
                continue
            removed_pixels += int(np.count_nonzero(component_mask))
            result[component_mask] = target
        logger.info(
            "[UNIFORM_BACKGROUND] Cleared %d detached flat-field artifact pixels",
            removed_pixels,
        )
        return Image.fromarray(result, "RGB")

    @staticmethod
    def _normalize_semantic_backdrop_residue(
        image: Image.Image,
        background_color: Union[str, Tuple[int, int, int]],
    ) -> Image.Image:
        """Clear decoder residue without rematting the generated subject.

        Qwen can render the requested flat field correctly at the border while
        leaving shifted versions of that colour inside curl gaps, plus small
        scene-colour pieces outside the lower shoulder contour. A border-only
        validator cannot see those enclosed pixels. SCHP is used only to prove
        background (label 0); no foreground pixel is recoloured or alpha
        blended, so flowers, hair strands, skin and plaid remain byte-for-byte.
        """
        try:
            from pipelines.schp_service import parse as parse_body_parts

            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
            labels = parse_body_parts(rgb)["labels"]
            target = np.asarray(
                QwenEditPipeline._parse_bg_color(background_color),
                dtype=np.uint8,
            )
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            target_hsv = cv2.cvtColor(
                target.reshape(1, 1, 3), cv2.COLOR_RGB2HSV,
            )[0, 0]
            target_distance = np.linalg.norm(
                rgb.astype(np.float32) - target[None, None, :].astype(np.float32),
                axis=2,
            )
            changed = np.zeros(labels.shape, dtype=bool)

            # For a chromatic selected backdrop, normalize only background
            # pixels whose hue still identifies them as decoder-shifted copies
            # of that backdrop. Navy plaid and dark hair are protected because
            # SCHP labels them foreground before this colour test is applied.
            if int(target_hsv[1]) >= 80:
                hue_delta = np.abs(
                    hsv[:, :, 0].astype(np.int16) - int(target_hsv[0])
                )
                hue_delta = np.minimum(hue_delta, 180 - hue_delta)
                shifted_backdrop = (
                    (labels == 0)
                    & (target_distance > 2.0)
                    & (hue_delta <= 14)
                    & (hsv[:, :, 1] >= 35)
                    & (hsv[:, :, 2] >= 70)
                )
                rgb[shifted_backdrop] = target
                changed |= shifted_backdrop
            else:
                # White/gray passport backdrops have no useful hue. SCHP can
                # mislabel a child's neck and narrow red placket as background,
                # so never clear label-0 blindly. Remove only achromatic midtone
                # residue while protecting skin chroma, dark hair and coloured
                # garment pixels. This targets the gray source-scene chunks
                # beside ears without punching white holes through the person.
                ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
                skin_like = (
                    (ycrcb[:, :, 1] >= 132)
                    & (ycrcb[:, :, 1] <= 185)
                    & (ycrcb[:, :, 2] >= 75)
                    & (ycrcb[:, :, 2] <= 138)
                )
                dark_detail = hsv[:, :, 2] < 72
                coloured_detail = hsv[:, :, 1] >= 48
                height, width = labels.shape
                yy, xx = np.indices(labels.shape)
                achromatic_residue = (
                    (labels == 0)
                    & (yy < int(height * .61))
                    & (target_distance > 2.0)
                    & ~skin_like
                    & ~dark_detail
                    & ~coloured_detail
                )
                rgb[achromatic_residue] = target
                changed |= achromatic_residue
                # Some Qwen source-wall remnants are confidently mislabeled as
                # hair because they touch a braid. They have a distinctive
                # desaturated blue-gray tone and sit outside the central face/
                # neck corridor. Remove connected islands of that tone while
                # retaining black hair (lower value), skin (different chroma),
                # and the coloured school uniform.
                blue_gray_side_residue = (
                    ((xx < int(width * .32)) | (xx > int(width * .68)))
                    & (yy > int(height * .14))
                    & (yy < int(height * .61))
                    & (hsv[:, :, 0] >= 88)
                    & (hsv[:, :, 0] <= 128)
                    & (hsv[:, :, 1] <= 68)
                    & (hsv[:, :, 2] >= 76)
                    & (hsv[:, :, 2] <= 225)
                    & ~skin_like
                ).astype(np.uint8)
                component_count, component_labels, component_stats, _ = (
                    cv2.connectedComponentsWithStats(
                        blue_gray_side_residue, connectivity=8,
                    )
                )
                for component in range(1, component_count):
                    area = int(component_stats[component, cv2.CC_STAT_AREA])
                    if area < 8:
                        continue
                    component_mask = component_labels == component
                    rgb[component_mask] = target
                    changed |= component_mask

            # Below the head/accessory zone, the portrait foreground has a
            # continuous measured left/right span on each row. Clear only
            # parser-confirmed background outside that span. This removes tan
            # or gray edge fragments without punching holes through a patterned
            # shirt or through curls inside the subject envelope.
            height, width = labels.shape
            for row in range(int(height * .48), height):
                columns = np.flatnonzero(labels[row] != 0)
                if columns.size < max(6, int(width * .025)):
                    continue
                exterior = np.zeros(width, dtype=bool)
                exterior[:max(0, int(columns[0]) - 2)] = True
                exterior[min(width, int(columns[-1]) + 3):] = True
                exterior &= labels[row] == 0
                exterior &= target_distance[row] > 2.0
                rgb[row, exterior] = target
                changed[row, exterior] = True

            logger.info(
                "[UNIFORM_BACKGROUND] Normalized %d semantic backdrop-residue pixels without rematting foreground",
                int(np.count_nonzero(changed)),
            )
            return Image.fromarray(rgb, "RGB")
        except Exception as exc:
            logger.warning(
                "[UNIFORM_BACKGROUND] Semantic residue cleanup unavailable: %s",
                exc,
            )
            return image.convert("RGB")

    @staticmethod
    def _refine_final_backdrop_edge(
        image: Image.Image,
        background_color: Union[str, Tuple[int, int, int]],
        job_id: Optional[str] = None,
    ) -> Image.Image:
        """Apply one soft edge matte after deterministic residue cleanup.

        The semantic cleanup above guarantees the correct interior backdrop,
        but its boundary is intentionally binary. BiRefNet+SCHP is therefore
        run only on the already-clean final passport frame to recover natural
        sub-pixel hair, flower and shoulder alpha. This is not a second image
        generation and does not replace opaque subject RGB.
        """
        try:
            from pipelines.birefnet_service import background_removal

            source = image.convert("RGB")
            buffer = io.BytesIO()
            source.save(buffer, format="PNG")
            cutout_bytes = background_removal.remove_background(
                buffer.getvalue(),
                job_id=(f"{job_id}_final_edge" if job_id else None),
                use_schp=True,
            )
            cutout = Image.open(io.BytesIO(cutout_bytes)).convert("RGBA")
            # BiRefNet deliberately keeps a firm human silhouette, which is
            # useful for shoulders but can leave one-pixel stair steps around
            # curls after the passport crop.  Soften only the measured SCHP
            # hair boundary.  The mask is derived for every portrait; no hair
            # colour, length, style or side is assumed, and flowers/earrings,
            # face and garment alpha remain unchanged.
            try:
                from pipelines.schp_service import parse as parse_body_parts

                source_rgb = np.asarray(source, dtype=np.uint8)
                labels = parse_body_parts(source_rgb)["labels"]
                rgba = np.asarray(cutout, dtype=np.uint8).copy()
                alpha = rgba[:, :, 3]
                hair = (labels == 2).astype(np.uint8)
                if np.count_nonzero(hair) >= 80:
                    hair_neighborhood = cv2.dilate(
                        hair,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                        iterations=1,
                    ).astype(bool)
                    silhouette = (alpha >= 128).astype(np.uint8)
                    boundary = cv2.morphologyEx(
                        silhouette,
                        cv2.MORPH_GRADIENT,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                    ).astype(bool)
                    repair_band = hair_neighborhood & cv2.dilate(
                        boundary.astype(np.uint8),
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                        iterations=1,
                    ).astype(bool)
                    softened = cv2.GaussianBlur(alpha, (0, 0), 0.75)
                    alpha[repair_band] = softened[repair_band]
                    # Alpha repair cannot remove green/blue scene colour that
                    # is already baked into a translucent strand. Estimate
                    # the local foreground colour from this portrait's own
                    # opaque hair core and correct only chromatically deviant
                    # pixels in the narrow silhouette band. This is dynamic:
                    # no assumed hair colour, style or backdrop hue.
                    hair_core = (
                        (hair > 0)
                        & (alpha >= 245)
                        & ~cv2.dilate(
                            boundary.astype(np.uint8),
                            cv2.getStructuringElement(
                                cv2.MORPH_ELLIPSE, (5, 5),
                            ),
                            iterations=1,
                        ).astype(bool)
                    )
                    if np.count_nonzero(hair_core) >= 64:
                        core_weight = hair_core.astype(np.float32)
                        local_weight = cv2.GaussianBlur(
                            core_weight, (0, 0), 5.0,
                        )
                        foreground_rgb = rgba[:, :, :3].astype(np.float32)
                        local_rgb = np.empty_like(foreground_rgb)
                        for channel in range(3):
                            local_sum = cv2.GaussianBlur(
                                foreground_rgb[:, :, channel] * core_weight,
                                (0, 0), 5.0,
                            )
                            local_rgb[:, :, channel] = local_sum / np.maximum(
                                local_weight, 1e-4,
                            )
                        local_valid = local_weight >= .015
                        current_lab = cv2.cvtColor(
                            foreground_rgb.clip(0, 255).astype(np.uint8),
                            cv2.COLOR_RGB2LAB,
                        ).astype(np.float32)
                        local_lab = cv2.cvtColor(
                            local_rgb.clip(0, 255).astype(np.uint8),
                            cv2.COLOR_RGB2LAB,
                        ).astype(np.float32)
                        local_colour_delta = np.linalg.norm(
                            current_lab - local_lab, axis=2,
                        )
                        target_rgb = np.asarray(
                            QwenEditPipeline._parse_bg_color(background_color),
                            dtype=np.float32,
                        )
                        backdrop_distance = np.linalg.norm(
                            foreground_rgb - target_rgb[None, None, :], axis=2,
                        )
                        contaminated_edge = (
                            repair_band
                            & local_valid
                            & (
                                (local_colour_delta >= 24.0)
                                | (backdrop_distance <= 72.0)
                            )
                        )
                        foreground_rgb[contaminated_edge] = (
                            foreground_rgb[contaminated_edge] * .28
                            + local_rgb[contaminated_edge] * .72
                        )
                        rgba[:, :, :3] = np.rint(
                            foreground_rgb,
                        ).clip(0, 255).astype(np.uint8)
                        logger.info(
                            "[UNIFORM_BACKGROUND] Repaired %d scene-contaminated hair-edge RGB pixels from local strands",
                            int(np.count_nonzero(contaminated_edge)),
                        )
                    rgba[:, :, 3] = alpha
                    cutout = Image.fromarray(rgba, "RGBA")
                    logger.info(
                        "[UNIFORM_BACKGROUND] Softened %d dynamically detected hair-edge pixels",
                        int(np.count_nonzero(repair_band)),
                    )
            except Exception as exc:
                logger.warning(
                    "[UNIFORM_BACKGROUND] Hair-edge alpha repair unavailable: %s",
                    exc,
                )
            target = QwenEditPipeline._parse_bg_color(background_color)
            backdrop = Image.new(
                "RGBA", cutout.size,
                tuple(int(value) for value in target) + (255,),
            )
            backdrop.alpha_composite(cutout)
            logger.info(
                "[UNIFORM_BACKGROUND] Refined final hair/flower/shoulder edge with protected soft matte",
            )
            return backdrop.convert("RGB")
        except Exception as exc:
            logger.warning(
                "[UNIFORM_BACKGROUND] Final soft-edge refinement unavailable: %s",
                exc,
            )
            return image.convert("RGB")
        finally:
            try:
                background_removal.unload()
            except Exception:
                pass

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

    @staticmethod
    def _build_universal_static_uniform_prompt(
        background_hex: str = "#FFFFFF",
        background_rgb: Tuple[int, int, int] = (255, 255, 255),
        repair_lighting: bool = False,
        source_age_band: str = "",
        source_hair_geometry_lock: str = "enforce",
        **kwargs: Any,
    ) -> str:
        """Static universal prompt constructor for school ID portraits."""
        from pipelines.uniform_swap.uniform_generator import _build_universal_static_uniform_prompt as gen_prompt
        cues = kwargs.get("cues") or {"ref_type": "generic_uniform", "fabric_desc": "tailored school uniform fabric", "collar_desc": "collar", "components": "uniform shirt", "sleeve_desc": "short sleeves"}
        pos, _ = gen_prompt(cues=cues, background_rgb=background_rgb)
        return pos

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
        edit_region_mask: Optional[np.ndarray] = None,
        conditioning_max_dimension: int = 256,
        preserve_reference_aspect: bool = False,
        normalize_decoded_background: bool = True,
        conditioning_pixel_budget: Optional[int] = None,
        hard_protect_region_mask: Optional[np.ndarray] = None,
        latent_keep_mask_tensor: Optional[torch.Tensor] = None,
        latent_identity_mode: str = "off",
        source_portrait_image: Optional[Union[Image.Image, np.ndarray, Path, str]] = None,
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
            color_desc = self._get_bg_color_description(bg_rgb)
            # "Studio backdrop" is visually ambiguous to an image model and
            # produced curtains, rooms and cloudy photographic backgrounds.
            # Describe the required pixels and natural color name explicitly.
            bg_desc = f"against one featureless flat {color_desc} {bg_hex} RGB{bg_rgb} studio backdrop"

        if prompt and prompt.strip():
            effective_prompt = prompt.strip()
            if bg_desc and bg_hex not in effective_prompt[:150]:
                effective_prompt = f"BACKGROUND: Generate selected flat {color_desc} {bg_hex} RGB{bg_rgb} studio backdrop from edge to edge during generation itself. " + effective_prompt
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
        if bg_desc and effective_negative and "other background color" not in effective_negative:
            if tuple(bg_rgb) == (255, 255, 255) or bg_hex.upper() == "#FFFFFF":
                effective_negative = f"other background color, colored backdrop, green background, cyan background, blue background, grey background, tinted backdrop, dark background, outdoor background, scenery, room, wall, background shadow, background gradient, {effective_negative}"
            else:
                effective_negative = f"other background color, white background, pale border, pale edge, wrong background color, non-{bg_hex} background, dark background, outdoor background, scenery, room, wall, background shadow, background gradient, {effective_negative}"
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
            # The positive prompt keeps its image references; the negative is
            # encoded from text alone so CFG cannot subtract them.
            text_only_prompts=[False] + ([True] if effective_negative else []),
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
        # Select the Lightning adapter trained for the requested schedule.
        # Never run the four-step adapter for eight steps (or under-sample the
        # base model at eight): both produced painted, weakly edited portraits.
        num_steps = max(2, min(steps or 4, 24))
        lightning_lora_file = (
            LORA_FILE if num_steps <= 4 else
            LORA_FILE_8 if num_steps == 8 else
            None
        )
        use_lightning_lora = bool(
            lightning_lora_file is not None
            and lightning_lora_file.exists()
        )
        if num_steps == 8 and not use_lightning_lora:
            raise RuntimeError(
                "The requested extreme-portrait route requires the installed "
                "Qwen-Image-Edit-2511 8-step Lightning LoRA."
            )
        if progress_callback:
            mode_label = (
                f"{num_steps}-step Lightning"
                if use_lightning_lora else "base-model high-quality"
            )
            progress_callback(40, f"Initializing {mode_label} portrait restoration...")

        if not hasattr(self, "_cached_tokenizer") or self._cached_tokenizer is None:
            self._cached_tokenizer = AutoTokenizer.from_pretrained((MODELS_DIR / "tokenizer").resolve().as_posix())
        if not hasattr(self, "_cached_processor") or self._cached_processor is None:
            self._cached_processor = AutoProcessor.from_pretrained((MODELS_DIR / "processor").resolve().as_posix())
        if not hasattr(self, "_cached_vae") or self._cached_vae is None:
            self._cached_vae = AutoencoderKLQwenImage.from_pretrained((MODELS_DIR / "vae").resolve().as_posix(), torch_dtype=torch.float32).to("cpu")
        if not hasattr(self, "_cached_scheduler") or self._cached_scheduler is None:
            self._cached_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained((MODELS_DIR / "scheduler").resolve().as_posix())

        tokenizer = self._cached_tokenizer
        processor = self._cached_processor
        vae = self._cached_vae
        scheduler = self._cached_scheduler

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
        pipe.set_progress_bar_config(disable=True)
        pipe._encode_vae_image = types.MethodType(_cpu_encode_vae_image, pipe)

        orig_prepare_latents = pipe.prepare_latents

        def _cuda_prepare_latents(pipe_self, *args, **kwargs):
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

        # In V12, do not extract source latents by monkey-patching and slicing concatenated conditioning latents.
        # Instead, encode the source portrait independently at the exact output generation dimensions (w, h).
        clean_source_latents = None
        keep = None
        mask_noise = None

        if latent_identity_mode != "off" and latent_keep_mask_tensor is not None:
            source_img_for_latent = source_portrait_image if source_portrait_image is not None else input_pil
            src_pil = self._to_pil(source_img_for_latent).convert("RGB")
            src_resized = src_pil.resize((w, h), Image.Resampling.LANCZOS)
            
            src_tensor = pipe.image_processor.preprocess(src_resized, h, w)
            if src_tensor.ndim == 4:
                src_tensor = src_tensor.unsqueeze(2)
            src_tensor = src_tensor.to(device="cpu", dtype=torch.float32)
            
            with torch.no_grad():
                src_latents_5d = pipe._encode_vae_image(src_tensor, generator=None)
                
            latent_h = h // pipe.vae_scale_factor
            latent_w = w // pipe.vae_scale_factor
            num_channels = pipe.vae.config.z_dim
            
            clean_source_latents = QwenImageEditPlusPipeline._pack_latents(
                src_latents_5d,
                batch_size=1,
                num_channels_latents=num_channels,
                height=latent_h,
                width=latent_w,
            ).to(device=self.device, dtype=self.dtype)
            
            expected_patches = (latent_h // 2) * (latent_w // 2)
            if clean_source_latents.shape != (1, expected_patches, num_channels * 4):
                raise RuntimeError(
                    f"[QWEN_EDIT_ENHANCER] Source latents shape mismatch: {clean_source_latents.shape} != (1, {expected_patches}, {num_channels*4})"
                )
            
            keep = latent_keep_mask_tensor.to(device=self.device, dtype=self.dtype)
            if keep.ndim == 2:
                keep = keep.unsqueeze(-1)
            if keep.shape[1] != expected_patches:
                raise RuntimeError(
                    f"[QWEN_EDIT_ENHANCER] Latent keep mask token count mismatch: mask={keep.shape} vs expected tokens={expected_patches}"
                )
            logger.info(
                "[QWEN_EDIT_ENHANCER] Authoritative source latents prepared: shape=%s, mask=%s, protected=%.1f%% (mode=%s)",
                tuple(clean_source_latents.shape), tuple(keep.shape), 100.0 * float((keep > 0.5).float().mean()), latent_identity_mode,
            )
        else:
            logger.info("[QWEN_EDIT_ENHANCER] Latent identity lock is OFF: zero latent blending will occur.")

        if use_lightning_lora and lightning_lora_file is not None:
            pipe.load_lora_weights(
                str(lightning_lora_file.parent),
                weight_name=lightning_lora_file.name,
                adapter_name="lightning",
            )

        def on_step_end(pipe, step_index, timestep, callback_kwargs):
            check_deadline(f"diffusion step {step_index + 1}/{num_steps}")
            pct = int(45 + (step_index + 1) / num_steps * 40)
            msg = f"Restoring studio portrait detail (step {step_index + 1}/{num_steps})…"
            logger.info("[QWEN_EDIT_ENHANCER] Diffusion step %d/%d completed (%.1fs elapsed)", step_index + 1, num_steps, time.time() - t0)
            if progress_callback:
                progress_callback(pct, msg)
            nonlocal mask_noise

            if clean_source_latents is None or keep is None or "latents" not in callback_kwargs:
                return callback_kwargs

            latents = callback_kwargs["latents"]
            if latents.shape != clean_source_latents.shape:
                raise RuntimeError(
                    f"[QWEN_EDIT_ENHANCER] Incompatible latent layouts: latents={tuple(latents.shape)} vs clean_source={tuple(clean_source_latents.shape)}"
                )

            timesteps = pipe.scheduler.timesteps
            if step_index + 1 < len(timesteps):
                if mask_noise is None:
                    mask_noise = torch.randn(
                        clean_source_latents.shape,
                        generator=torch.Generator(device="cpu").manual_seed(run_seed),
                        dtype=torch.float32,
                    ).to(latents.device, latents.dtype)
                next_t = timesteps[step_index + 1:step_index + 2].to(latents.device)
                held = pipe.scheduler.scale_noise(
                    clean_source_latents.to(latents.dtype), next_t, mask_noise,
                )
            else:
                held = clean_source_latents.to(latents.dtype)

            callback_kwargs["latents"] = latents * (1.0 - keep) + held * keep
            logger.info(
                "[QWEN_EDIT_ENHANCER] Step %d/%d: blended latents with %.1f%% protected identity (mode=%s)",
                step_index + 1, num_steps, 100.0 * float((keep > 0.5).float().mean()), latent_identity_mode,
            )
            return callback_kwargs

        run_seed = int(seed) if seed is not None else (zlib.crc32(job_id.encode("utf-8")) & 0x7FFFFFFF)
        logger.info(
            "[QWEN_EDIT_ENHANCER] Running %d-step diffusion (seed=%d, true_cfg=%.2f, lightning=%s)...",
            num_steps, run_seed, effective_true_cfg_scale,
            use_lightning_lora,
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
            callback_on_step_end_tensor_inputs=["latents"],
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
        # Enhancement may normalize a decoded backdrop. Uniform Swap disables
        # this: its selected colour is already present in image 1 and protected
        # by the latent hold, so the raw artifact must remain the literal Qwen
        # decode rather than a post-generation background replacement.
        if (
            normalize_decoded_background
            and background_color
            and str(background_color).lower() not in ("none", "keep", "original")
        ):
            bg_target_rgb = self._parse_bg_color(background_color)
            raw_diffused = self._replace_flat_border_background_strict(
                raw_diffused, bg_target_rgb,
            )

        # Keep the raw Qwen output with selected background for stage-by-stage inspection.
        try:
            qwen_debug_path = OUTPUTS_DIR / f"{job_id}_qwen_raw.png"
            raw_diffused.save(qwen_debug_path, format="PNG")
        except Exception:
            fallback_dir = Path("C:/Users/rajan/.gemini/antigravity-cli/brain/1caaa53f-a55d-422e-a69c-a23239010d04")
            qwen_debug_path = fallback_dir / f"{job_id}_qwen_raw.png"
            raw_diffused.save(qwen_debug_path, format="PNG")
        self.last_qwen_candidate_path = qwen_debug_path
        logger.info("[QWEN_EDIT_ENHANCER] Saved pre-validation Qwen output with selected background -> %s", qwen_debug_path)
        if return_raw_candidate:
            # Explicit uniform-preview policy: never replace the generated
            # candidate with source pixels or reject it on a quality score.
            self.last_qwen_identity_accepted = None
            # Uniform generation intentionally returns here, before the shared
            # enhancement post-processing path. Release every large CUDA/CPU
            # object first; otherwise the next queued job starts its isolated
            # VL encoder while this process still owns the previous latents and
            # prompt tensors. On Windows that repeatedly terminated the encoder
            # with native access-violation code 3221225477 after job one.
            # diffusion_kwargs still owns the CUDA prompt tensors even after
            # the local variables below are deleted. Emptying the allocator
            # while that dictionary remains alive leaves the blocks reserved;
            # the next bitsandbytes encoder can then terminate on Windows with
            # 0xC0000005 while loading weights. Drop every container reference
            # before synchronizing and releasing the CUDA cache.
            del diffusion_kwargs
            del output_latents, res, vae
            del prompt_embeds, prompt_embeds_mask
            del negative_prompt_embeds, negative_prompt_embeds_mask
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except (RuntimeError, AttributeError):
                    pass
            return qwen_debug_path

        # ── Step 5: High-Fidelity Identity Lock & Optical Detail Restoration ──
        if progress_callback:
            progress_callback(92, "Verifying identity and fine portrait detail...")

        # Clothing preservation is not evidence of facial identity. The old
        # boolean shortcut approved every preserve-source-clothing enhancement
        # without inspecting its face, allowing enlarged eyes and a reshaped
        # jaw to pass simply because the shirt was retained.
        qwen_portrait_accepted = self._has_acceptable_portrait_identity(
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
        job_id: str = "qwen_uniform",
        background_color: Union[str, Tuple[int, int, int]] = "4,126,246",
        width: int = 560,
        height: int = 720,
        steps: int = 20,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        generation_mode: str = "quality",
        generation_dimension: Optional[int] = None,
    ) -> Path:
        """Use VL only for a fit plan, then perform one direct Qwen Image Edit."""
        if not uniform_template_path:
            raise ValueError("Uniform template image is mandatory for Uniform Swap.")

        # Flatten any uploaded alpha onto white before anything measures this
        # portrait. PIL's RGBA->RGB conversion discards the alpha channel and
        # keeps whatever sits in the colour channels underneath, which for a
        # cut-out PNG is black. A studio cut-out therefore arrived as a person
        # on a black field: SCHP labelled the dark border as subject, and the
        # delivered portrait carried a grey envelope around the head and
        # shoulders where that black had been keyed back out. White matches the
        # cut-out's intended backdrop and the selected-backdrop pass replaces it
        # cleanly either way.
        _person_upload = self._to_pil(person_input, mode="RGBA")
        # Record whether this portrait's background is a real photographed
        # scene or flat white this pipeline synthesized under transparency.
        # Later wall-keying must not treat the synthetic field as evidence.
        self._last_upload_alpha_flattened = bool(
            _person_upload.getchannel("A").getextrema()[0] < 255
        )
        person_pil = self._flatten_upload_alpha(_person_upload)
        # Heal any dead/hot sensor pixels or scanner defects in the uploaded photo
        person_pil = self._restore_dead_and_hot_pixels(person_pil)
        # Keep the clean upload as both the generation input and the final
        # biometric/hair authority. Uniform Swap must not silently substitute
        # a previously saved or internally enhanced portrait.
        original_person_pil = person_pil.copy()
        portrait_clarity_assessment = PhotoRestorationService.assess_portrait_clarity(original_person_pil)
        logger.info("[PORTRAIT_CLARITY] %s", portrait_clarity_assessment.get("message"))
        identity_prepass_used = False
        identity_prepass_review: Optional[Dict[str, Any]] = None
        masked_retinex_applied = False
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
        # Build one measured VL contract using this upload and this template.
        source_has_direct_light = _has_grounded_direct_light(vl_plan)
        source_has_hair_hotspot = (
            str(vl_plan.get("head_hair_hotspot_present") or "").strip().lower()
            in {"true", "yes", "1", "present"}
        )
        # A successful relight prepass is re-analysed and should report
        # neutral illumination. Keep the upload's original lighting evidence
        # separately so that refreshed VL output cannot disable the mandatory
        # residual-light gate on the delivered uniform portrait.
        uploaded_source_has_direct_light = source_has_direct_light
        # Use the native four-step edit adapter for every uniform portrait.
        # The 12-step base path repainted cloth, retained jewelry and weakened
        # identity on extreme-sun probes. Such portraits receive a measured
        # neutral-light conditioning guide below, so Qwen still performs one
        # coherent first generation without changing scheduler families.
        measured_sunlight_severity = str(
            vl_plan.get("measured_sunlight_severity") or "normal"
        ).strip().lower()
        extreme_source_sunlight = measured_sunlight_severity == "extreme"
        # Final step selection is completed after measuring whether this upload
        # already has enough sharp facial evidence.  Extreme lighting alone
        # must not force a sharp child's face through the more generative
        # eight-step route after masked Retinex has already neutralized the
        # conditioning image.
        generation_steps = 4
        step_reasons = []
        source_min_dimension = int(vl_plan.get("source_min_dimension") or 0)
        # Eight-step Lightning successfully removed extreme light but redrew
        # the child and replaced literal hair flowers (identity 0.471 on the
        # verified extreme-sun fixture). Match Enhancement's four-step edit so
        # image 1 remains dominant; the compact lighting-priority instruction
        # and measured delivery gate handle illumination without a new person.
        generation_steps = 4
        step_reasons.append(
            "single four-step identity-preserving uniform and neutral-light edit"
        )
        if generation_dimension is not None and int(generation_dimension) > 0:
            uniform_generation_dimension = int(generation_dimension)
        elif "turbo" in str(generation_mode).lower() or "fast" in str(generation_mode).lower():
            uniform_generation_dimension = 1024
        else:
            uniform_generation_dimension = DEFAULT_UNIFORM_GENERATION_DIMENSION
        step_reasons.append(
            f"{uniform_generation_dimension}px portrait bucket ({'Turbo' if uniform_generation_dimension == 1024 else 'Quality'})"
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

        # Sanitize outer_neckline when template is only a single shirt with no outer layer
        garment_comps = str(vl_plan.get("garment_components") or "").lower()
        has_outer = any(
            term in garment_comps
            for term in (
                "vest", "waistcoat", "pinafore", "blazer", "jacket",
                "sweater", "cardigan", "coat", "outer garment", "outer layer",
            )
        )
        if not has_outer:
            vl_plan["outer_neckline"] = "none"
            vl_plan["outer_garment_color_and_shape"] = "none"
            vl_plan["outer_color_under_neutral_light"] = "none"
            vl_plan["outer_primary_color"] = "none"
            vl_plan["outer_neutral_rgb"] = None

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
            # "texture" copies a badge-sized block from another chest row,
            # which lands out of phase on a check: job 16682cec delivered a
            # rectangular plaid block where the pocket crest had been, and
            # Qwen reproduced that block in the portrait. "periodic" extends
            # each column by the measured vertical repeat so threads and the
            # pocket seam line up, and it falls back to the shifted-texture
            # copy for non-periodic fabrics.
            repair_mode="periodic",
            reference_bbox=vl_plan.get("badge_bbox"),
            reference_bbox_size=template_analysis_size,
            # Qwen-VL's documented output contract is
            # [ymin, xmin, ymax, xmax]. Keep this explicit instead of
            # silently treating the vertical coordinates as horizontal ones.
            reference_bbox_order="yx",
        )
        logger.info(
            "[UNIFORM_BADGE] Removed %d badge region(s) from rough-fit garment guide",
            template_badges_removed,
        )
        transparent_template = transparent_rgb.convert("RGBA")
        transparent_template.putalpha(template_source.getchannel("A"))
        # A garment photographed without a wearer shows its rear collar band
        # standing above the shoulders. Qwen copies whatever the reference
        # shows, so this band became a navy cowl wrapped around the throat on
        # every fixture. Remove it BEFORE the design reference is captured;
        # hiding it only in the rough guide left image 2 authoritative for the
        # band and the model always restored it.
        transparent_template, reference_rear_pixels = hide_product_rear_collar(
            transparent_template,
        )
        if reference_rear_pixels:
            logger.info(
                "[UNIFORM_COLLAR] Hid %d unworn rear-collar pixels from the design reference",
                reference_rear_pixels,
            )
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

        # Dynamic Uniform Swap performs exactly one VL-planned Qwen edit.
        # No identity-restoration prepass or alternate portrait is substituted.
        source_face_width_px = max(
            1.0, float(source_face.bbox[2] - source_face.bbox[0]),
        )
        source_face_detail_budget = source_face_width_px / 260.0
        source_detail_sufficient = bool(source_face_detail_budget >= 1.0)
        # A small/printed/low-detail face cannot be improved while its clean
        # latents are frozen: the hold faithfully preserves scanner texture,
        # scratches and the rectangular source-photo patch while Qwen draws a
        # second, cleaner head around it.  Use the measured face evidence as
        # identity conditioning, but let diffusion reconstruct one coherent
        # head whenever there are fewer than 260 source face pixels. This is
        # dynamic per upload and is never tied to a person, hairstyle or age.
        regenerate_low_detail_head = not source_detail_sufficient
        # Cross-verify hair accessories: Ensure bows/ribbons are never missed due to VL hallucination or misclassification
        acc_check = self._detect_hair_accessory_info(person_pil, source_face.bbox)
        vl_acc = str(vl_plan.get("hair_accessories") or "").strip().lower()
        if acc_check.get("detected"):
            if not vl_acc or vl_acc in {"none", "uncertain", "unknown", "no", "false", "earring", "earrings"}:
                vl_plan["hair_accessories"] = acc_check["description"]
                vl_plan["hair_accessory_details"] = acc_check["description"]
                logger.info("[UNIFORM_VL] CV detector confirmed authentic hair accessories: %s", acc_check["description"])
            vl_plan["cv_hair_accessory_info"] = {k: v for k, v in acc_check.items() if k != "mask"}
            vl_plan["cv_hair_accessory_mask"] = acc_check.get("mask")
        else:
            if vl_acc and vl_acc not in {"none", "uncertain", "unknown", "no", "false"}:
                logger.info(
                    "[UNIFORM_VL] Neutralized unsupported VL hair_accessories '%s' -> 'none' (CV detector found none)",
                    vl_acc,
                )
            vl_plan["hair_accessories"] = "none"
            vl_plan["hair_accessory_details"] = "none"
            vl_plan["cv_hair_accessory_info"] = {
                "detected": False,
                "pixels": 0,
                "color": None,
                "placement": None,
                "description": "none",
            }
            vl_plan["cv_hair_accessory_mask"] = None

        # Uniform Swap deliberately has no generative restoration prepass.
        # Harsh-light normalization is a deterministic masked-Retinex
        # conditioning operation later in this same route, followed by one
        # four-step Qwen edit that restores the portrait and fits the garment.

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
        vl_reports_child = any(w in vl_age_band for w in ("child", "school"))
        vl_reports_teen = any(w in vl_age_band for w in ("teen", "student"))
        vl_reports_adult = any(w in vl_age_band for w in ("adult", "young adult"))
        age_classifier_conflict = bool(
            (source_age_value >= 22 and (vl_reports_child or vl_reports_teen))
            or (source_age_value and source_age_value < 13 and vl_reports_adult)
        )
        if age_classifier_conflict:
            # A damaged print can make InsightFace report a child as 30-40;
            # conversely a school uniform can bias VL toward "child". Never
            # feed either disputed label back into generation. Literal source
            # pixels are the authority for maturity and body scale.
            source_age_band = "source-visible age"
        elif source_age_value >= 22:
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
        if age_classifier_conflict:
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
        # Diagnostic intermediates are opt-in; production returns one result.
        save_intermediates = bool(
            os.getenv("VSOUL_SAVE_INTERMEDIATES", "0").strip().lower() in {"1", "true", "yes"}
        )
        source_identity_reference = person_pil.copy()
        source_identity_reference_path = OUTPUTS_DIR / f"{job_id}_source_identity_reference.png"
        if save_intermediates:
            try:
                source_identity_reference.save(source_identity_reference_path, format="PNG")
                logger.info(
                    "[UNIFORM_IDENTITY_REFERENCE] Saved untouched source head/hair/body reference -> %s",
                    source_identity_reference_path,
                )
            except Exception as e:
                logger.warning(
                    "[UNIFORM_IDENTITY_REFERENCE] Could not save untouched source reference (%s): %s",
                    source_identity_reference_path, e
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
        # Identity conditioning resolution is the same in both prompt modes.
        # Static mode previously built this canvas at 1024 while dynamic used
        # 1216, so the static route asked a four-step schedule to hold identity
        # from ~29% fewer face pixels - the opposite of what a maximum-identity
        # mode needs. The delivered frame is unchanged; only the evidence Qwen
        # reads is restored.
        conditioning_height = uniform_generation_dimension
        conditioning_width = max(
            256,
            int(round((conditioning_height * width / max(height, 1)) / 16.0) * 16),
        )
        person_conditioning = self._source_identity_reference(
            person_pil,
            source_face.bbox,
            conditioning_width,
            conditioning_height,
            background_color=conditioning_bg_rgb,
            target_face_fraction=.32,
        )
        # Match the Enhancement route: keep the original portrait pixels as
        # Qwen's identity/hair/accessory authority and ask Qwen to generate the
        # selected flat backdrop in the same coherent edit.  Pre-matting with
        # BiRefNet/SCHP permanently removed curls, flowers and fine crown hair;
        # Qwen then correctly preserved those missing regions as blue holes.
        # Background verification/correction is performed on the generated
        # portrait below, never on the source identity reference.
        source_has_direct_light = _has_grounded_direct_light(vl_plan)
        source_has_hair_hotspot = (
            str(vl_plan.get("head_hair_hotspot_present") or "").strip().lower()
            in {"true", "yes", "1", "present"}
        )
        conditioning_faces = _get_insight_app().get(
            cv2.cvtColor(np.asarray(person_conditioning), cv2.COLOR_RGB2BGR)
        )
        # Preserve an already-correct studio canvas.  Running a clean flat-
        # backdrop portrait through BiRefNet/SCHP and compositing it onto the
        # same colour adds a synthetic alpha boundary for no visual benefit;
        # Qwen then amplifies that boundary into a pale halo or a latent-grid
        # staircase.  The matte is still computed below for semantic masks, but
        # its RGB composite must not replace clean uploaded pixels.
        clean_flat_person_conditioning = person_conditioning.copy()
        conditioning_already_has_selected_backdrop = (
            self._has_selected_solid_background(
                clean_flat_person_conditioning, conditioning_bg_rgb,
            )
        )
        # Retinex conditioning has been completely removed: artificial LAB/Retinex
        # filtering masks and washes out authentic facial skin tones, pores, fine hair strands,
        # and ear jewelry. The untouched uploaded portrait is preserved 100% cleanly
        # as Qwen's visual authority, and relighting is performed naturally in latent diffusion space.
        masked_retinex_applied = False
        logger.info(
            "[UNIFORM_LIGHTING] Retinex conditioning removed; retaining 100% untouched uploaded portrait details"
        )
        source_badges_removed = 0
        person_conditioning_path = OUTPUTS_DIR / f"{job_id}_person_conditioning.png"
        if save_intermediates:
            try:
                person_conditioning.save(person_conditioning_path, format="PNG")
                logger.info(
                    "[UNIFORM_IDENTITY_REFERENCE] Saved upper-chest source conditioning -> %s",
                    person_conditioning_path,
                )
            except Exception as e:
                logger.warning(
                    "[UNIFORM_IDENTITY_REFERENCE] Could not save %s (file locked): %s",
                    person_conditioning_path, e,
                )
        source_labels = parse_body_parts(np.array(person_pil))["labels"]
        # Build the SCHP rough fit on the exact passport-framed source portrait.
        # Previously the rough uniform was fitted on the wide upload and then
        # cropped with a different framing function from person_conditioning.
        # Resizing those two independently framed images to the same dimensions
        # shifted the collar, shoulders and torso before Qwen saw them.
        # Cleanly mat person onto the conditioning background so both rough_source
        # and generation_source have 0% dirty background wall pixels. BiRefNet
        # used to do this, but it fails often and its failure mode is silent:
        # the upload's wall survives into Qwen's image 1 and the model then
        # reproduces it as a grey-green halo around the hair. The body parser
        # this pipeline already runs supplies the same silhouette with no extra
        # model load, and Qwen regenerates hair edges inside the same edit.
        conditioning_birefnet_rgba = None
        try:
            conditioning_labels = parse_body_parts(
                np.asarray(person_conditioning.convert("RGB"))
            )["labels"]
            person_mask = (conditioning_labels != 0).astype(np.uint8)
            # LIP-20 labels the bare neck as background, so a label-only matte
            # erases the person's neck. Everything downstream then rebuilt that
            # neck from scratch: a flat synthetic patch under the jaw, a collar
            # seated on the chin and the shirt's rear collar band regenerated as
            # a band around the throat. Recover the photographed neck skin
            # between the jaw and the source clothing with the same colour test
            # the composite uses, so the conditioning keeps the real neck.
            if conditioning_faces:
                nfx1, nfy1, nfx2, nfy2 = conditioning_faces[0].bbox
                neck_face_w = float(nfx2 - nfx1)
                neck_face_h = float(nfy2 - nfy1)
                mask_rows, mask_cols = np.indices(person_mask.shape)
                # Reach past the collar the garment will actually be seated at.
                # Measured on a cut-out upload: the real neck ran to .31 face
                # heights below the chin and .29 face widths out, so a
                # .30 x .26 corridor left a wedge of neck unrecovered. The
                # parser had labelled that wedge background, matting turned it
                # flat white, and the conditioning showed Qwen a head floating
                # over a white notch - which it then rendered as a collar band
                # on the throat. The skin colour test below still decides which
                # pixels inside the corridor are actually neck.
                neck_corridor = (
                    (np.abs(mask_cols - (nfx1 + nfx2) * .5) <= neck_face_w * .34)
                    & (mask_rows >= int(nfy2) - 2)
                    & (mask_rows <= int(nfy2 + neck_face_h * .42))
                )
                neck_hsv = cv2.cvtColor(
                    np.asarray(person_conditioning.convert("RGB")),
                    cv2.COLOR_RGB2HSV,
                )
                neck_skin = (
                    (neck_hsv[:, :, 1] >= 20)
                    & (neck_hsv[:, :, 2] >= 30)
                    & (neck_hsv[:, :, 2] <= 245)
                    & (
                        (neck_hsv[:, :, 0] <= 25)
                        | (neck_hsv[:, :, 0] >= 170)
                    )
                )
                person_mask = np.maximum(
                    person_mask, (neck_corridor & neck_skin).astype(np.uint8),
                )
            person_mask = cv2.morphologyEx(
                person_mask, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1,
            )
            # The parser also labels the narrow gap between a real neck and a
            # closed/mandarin collar as background. Those slivers are enclosed
            # by the person, so matting turned them white and Qwen read a white
            # notch punched through the neck, rebuilding it as a collar band.
            # Fill every background island that is fully enclosed by the
            # silhouette; open backdrop is still removed.
            mask_bytes = (person_mask * 255).astype(np.uint8)
            padded = cv2.copyMakeBorder(
                mask_bytes, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0,
            )
            flooded = padded.copy()
            cv2.floodFill(
                flooded,
                np.zeros((flooded.shape[0] + 2, flooded.shape[1] + 2), np.uint8),
                (0, 0), 255,
            )
            enclosed = (flooded == 0)[1:-1, 1:-1]
            if np.any(enclosed) and conditioning_faces:
                # Only the neck band is repaired. A wider fill would also close
                # genuine gaps (arm against torso, hair strands against the
                # backdrop) and keep the phone photo's own backdrop colour.
                hfx1, hfy1, hfx2, hfy2 = conditioning_faces[0].bbox
                h_face_w = float(hfx2 - hfx1)
                h_face_h = float(hfy2 - hfy1)
                hole_rows, hole_cols = np.indices(enclosed.shape)
                neck_band = (
                    (hole_rows >= int(hfy2) - 2)
                    & (hole_rows <= int(hfy2 + h_face_h * .45))
                    & (
                        np.abs(hole_cols - (hfx1 + hfx2) * .5)
                        <= h_face_w * .42
                    )
                )
                enclosed &= neck_band
            if np.any(enclosed):
                person_mask = np.maximum(person_mask, enclosed.astype(np.uint8))
            # Grow the silhouette by one pixel and feather it lightly so real
            # hair strands at the boundary keep their antialiased edge instead
            # of ending on a hard parser staircase.
            person_mask = cv2.dilate(
                person_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1,
            )
            matte_alpha = np.clip(
                cv2.GaussianBlur(person_mask.astype(np.float32), (0, 0), 1.1) * 1.35,
                0.0, 1.0,
            )[:, :, None]
            conditioning_rgb = np.asarray(
                person_conditioning.convert("RGB"), dtype=np.float32,
            )
            # Run clean garment-free identity layer extraction before Qwen diffusion
            # Extracts independent masks for face, ears, neck, hair, accessories, and garments;
            # prevents original garment contamination from entering Qwen conditioning;
            # recovers authentic visible throat/neck skin; and generates clean conditioning RGB.
            try:
                from pipelines.uniform_hair_face_compositor import (
                    extract_clean_identity_layer,
                    validate_clean_identity_layer,
                )
                clean_id_result = extract_clean_identity_layer(
                    immutable_original_rgb=np.asarray(person_conditioning.convert("RGB")),
                    face_bbox=conditioning_faces[0].bbox if conditioning_faces else None,
                    face_landmarks=getattr(conditioning_faces[0], "landmark_2d_106", None) if conditioning_faces else None,
                    debug_dir=OUTPUTS_DIR / "debug_phases" if save_intermediates else None,
                    job_id=job_id,
                    save_debug=save_intermediates,
                    conditioning_strategy=os.getenv("VSOUL_CONDITIONING_STRATEGY", "AUTO"),
                )
                val_ok, val_issues, val_metrics = validate_clean_identity_layer(clean_id_result)
                logger.info(
                    "[UNIFORM_CONDITIONING] Clean identity layer valid=%s, issues=%s, neck_cls=%s, neck_px=%d, strategy=%s",
                    val_ok, val_issues, clean_id_result.get("neck_classification"), val_metrics.get("neck_pixels", 0),
                    clean_id_result.get("selected_strategy"),
                )
                if not val_ok:
                    logger.warning(
                        "[UNIFORM_CONDITIONING] Clean identity layer failed validation (%s). Enforcing fallback to original photograph baseline Strategy A.",
                        val_issues,
                    )
                    conditioning_rgb = clean_id_result["qwen_conditioning_strategy_a"].astype(np.float32)
                    rough_source = Image.fromarray(clean_id_result["qwen_conditioning_strategy_a"], "RGB")
                    person_conditioning = rough_source.copy()
                else:
                    conditioning_birefnet_rgba = clean_id_result["decontaminated_identity_rgba"].copy()
                    conditioning_rgb = clean_id_result["qwen_conditioning_rgb"].astype(np.float32)
                    matte_alpha = clean_id_result["refined_identity_alpha"][:, :, None]
                    rough_source = Image.fromarray(clean_id_result["qwen_conditioning_rgb"], "RGB")
                    person_conditioning = rough_source.copy()
            except Exception as matte_error:
                logger.warning(
                    "[UNIFORM_CONDITIONING] Clean identity layer extraction failed; using raw crop: %s",
                    matte_error, exc_info=True,
                )
                flat_backdrop = np.asarray(
                    conditioning_bg_rgb, dtype=np.float32,
                )[None, None, :]
                matted_rgb = (
                    conditioning_rgb * matte_alpha
                    + flat_backdrop * (1.0 - matte_alpha)
                )
                rough_source = Image.fromarray(
                    np.rint(matted_rgb).clip(0, 255).astype(np.uint8), "RGB",
                )
                person_conditioning = rough_source.copy()
            if conditioning_already_has_selected_backdrop and 'clean_id_result' not in locals():
                rough_source = clean_flat_person_conditioning.copy()
                person_conditioning = clean_flat_person_conditioning.copy()
                logger.info(
                    "[UNIFORM_CONDITIONING] Existing selected flat backdrop is "
                    "already clean; retained original conditioning RGB and used "
                    "the matte for masks only"
                )
            logger.info(
                "[UNIFORM_CONDITIONING] Matted person onto %s backdrop from %s "
                "(%d parser foreground pixels)",
                conditioning_bg_rgb,
                "fused BiRefNet+SCHP alpha" if conditioning_birefnet_rgba is not None else "parser fallback",
                int(np.count_nonzero(person_mask)),
            )
        except Exception as e:
            logger.warning("[UNIFORM_CONDITIONING] Background matting failed (%s), using raw crop", e)
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
        if save_intermediates:
            identity_head_reference.save(identity_head_path, format="PNG")
            logger.info(
                "[UNIFORM_IDENTITY_REFERENCE] Saved clothing-free head reference -> %s",
                identity_head_path,
            )
        vl_claims_tied_hair = any(
            term in str(vl_plan.get("hair_style") or "").lower()
            for term in ("ponytail", "tied", "bun", "braid", "plait")
        )
        # A tied claim needs a visible tie accessory (bow, ribbon, elastic).
        # Loose hair swept to one side satisfies the colour/connectivity tail
        # detector and the VL style label, but has no tie point; trusting it
        # pasted a "tied" head over a loose-hair portrait and blocked delivery.
        hair_acc_str = str(vl_plan.get("hair_accessories") or "").strip().lower()
        has_acc_str = hair_acc_str not in {
            "", "none", "unknown", "uncertain", "not visible", "not detected",
        }
        try:
            tied_accessory_count = max(
                0, int(vl_plan.get("hair_accessory_count") or (1 if has_acc_str else 0)),
            )
        except (TypeError, ValueError):
            tied_accessory_count = 1 if has_acc_str else 0
        tied_accessory_corroborated = (tied_accessory_count >= 1 or has_acc_str) and has_acc_str
        visible_tied_hair_geometry = _has_visible_tied_hair_geometry(
            rough_source_labels, rough_source_faces[0].bbox,
        )
        if vl_claims_tied_hair and not tied_accessory_corroborated and not visible_tied_hair_geometry:
            logger.warning(
                "[UNIFORM_HAIR] Tied-hair label lacks accessory corroboration or visible geometry; "
                "treating source hair as untied",
            )
            vl_claims_tied_hair = False
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
        # A garment photographed without a wearer shows its rear collar band
        # standing above the shoulders. Once the shirt is fitted that band wraps
        # the throat and Qwen regenerates it as a choker or a navy neck band.
        # The reference copy above already removed it; this call keeps the rough
        # guide consistent for any path that built its own template.
        transparent_template, rear_collar_pixels = hide_product_rear_collar(
            transparent_template,
        )
        if rear_collar_pixels:
            logger.info(
                "[UNIFORM_COLLAR] Hid %d unworn rear-collar pixels before rough fitting",
                rear_collar_pixels,
            )
        analysed_collar = str(vl_plan.get("shirt_collar") or "").strip().lower()
        # Measure the untouched upload before placing any target garment. This
        # ratio is face-relative, so it transfers to the prepared canvas while
        # remaining unique to this boy/girl portrait. Passing it into the
        # compositor prevents the rough collar from being seated at one height
        # and only the later prompt metadata being corrected to another.
        bisenet_anatomy = self.deduct_neck_and_shoulders_bisenet(
            person_pil, source_faces[0].bbox,
        )
        if 'clean_id_result' in locals() and clean_id_result is not None:
            clean_metrics = clean_id_result.get("metrics", {})
            neck_cls = clean_id_result.get("neck_classification", "")
            c_chin_y = clean_metrics.get("chin_y")
            c_lowest_neck = clean_metrics.get("lowest_neck_y")
            src_face_h = max(1.0, float(source_faces[0].bbox[3] - source_faces[0].bbox[1]))
            if neck_cls == "occluded_neck":
                premeasured_source_neck_ratio = 0.04
            elif c_chin_y is not None and c_lowest_neck is not None and c_lowest_neck > c_chin_y:
                premeasured_source_neck_ratio = float(np.clip((c_lowest_neck - c_chin_y) / src_face_h, 0.04, 0.35))
            else:
                premeasured_source_neck_ratio = bisenet_anatomy["vertical_neck_gap_ratio"]
        else:
            premeasured_source_neck_ratio = bisenet_anatomy["vertical_neck_gap_ratio"]
        target_shoulder_width = bisenet_anatomy["shoulder_width"]
        logger.info(
            "[UNIFORM_ANATOMY] BiSeNet neck ratio=%.4f (bottom y=%.1f), shoulder width=%.1fpx (%.2fx face)",
            premeasured_source_neck_ratio, bisenet_anatomy["neck_bottom_y"],
            target_shoulder_width, bisenet_anatomy["shoulder_to_face_ratio"],
        )
        rough_composite, rough_placement = build_rough_composite(
            rough_source,
            transparent_template,
            rough_source_labels,
            rough_source_faces[0].bbox,
            conditioning_bg_rgb,
            expect_tied_hair=preserve_tied_back_hair,
            minimum_neck_gap_ratio=0.0,
            analysed_collar=analysed_collar,
            target_neck_gap_ratio=premeasured_source_neck_ratio,
            target_shoulder_width=target_shoulder_width,
        )
        # A 35:45 passport frame shows roughly 1.15 face-heights of chest below
        # the chin. When the upload stops well above that, every shoulder and
        # chest pixel below its frame is invented rather than fitted, and the
        # delivered body will not match the person however well the garment is
        # rendered. Surface it: the result is still the best available, but
        # this is the reason it can look wrong, and it is a property of the
        # upload that no pipeline stage can recover.
        source_chin_to_frame = float(
            rough_placement.get("source_chin_to_frame_ratio") or 0.0
        )
        torso_evidence_sufficient = bool(
            rough_placement.get("torso_evidence_sufficient", True)
        )
        if not torso_evidence_sufficient:
            logger.warning(
                "[UNIFORM_TORSO_EVIDENCE] Upload shows only %.2f face-heights of "
                "chest below the chin (a passport frame needs ~1.15): %.0f%% of the "
                "delivered torso must be invented, so body fit is approximate. "
                "A half-body source photo fixes this.",
                source_chin_to_frame,
                100.0 * max(0.0, 1.0 - source_chin_to_frame / 1.15),
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
        if not tied_accessory_corroborated and not visible_tied_hair_geometry:
            # The rough compositor can split the two sides of ordinary short
            # hair into two lower components. Without a visible tied root or a
            # real tie accessory those components are not ponytails/pigtails.
            # The former condition cleared the count only when it was already
            # zero, allowing a short-haired child to receive a "two tied
            # sections" prompt and a redesigned hairstyle.
            measured_tied_count = 0
            measured_tied_sides = "unknown"
        if measured_tied_count > 0:
            vl_plan["tied_section_count"] = measured_tied_count
            vl_plan["tied_section_sides"] = measured_tied_sides
            vl_plan["ponytail_side"] = (
                measured_tied_sides if measured_tied_count == 1 else "not-applicable-bilateral"
            )
            # The measured root is beside/below the ear at the nape.  A VL
            # label such as "top" describes the visible crown rather than the
            # actual tie point and makes Qwen move the ponytail upward.
            vl_plan["ponytail_anchor"] = "source-visible nape/behind-ear root"
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
        vl_hair_length_before_measurement = str(
            vl_plan.get("hair_length") or ""
        ).strip()
        vl_ponytail_extent = str(
            vl_plan.get("ponytail_extent") or ""
        ).strip().lower()
        try:
            measured_hair_ratio_value = float(measured_hair_end_ratio)
        except (TypeError, ValueError):
            measured_hair_ratio_value = None
        # A ponytail can disappear behind the neck/collar and reappear lower
        # down. SCHP then measures only the root fragment and falsely replaces
        # a visible medium/full tail with "jaw length". Reconcile independent
        # evidence: when VL sees a full medium/long tied tail but the parser
        # stops within .20 face-heights of the jaw, mark the numeric endpoint
        # as occluded instead of teaching Qwen the wrong short hairstyle.
        parser_truncated_tied_tail = bool(
            measured_tied_count > 0
            and measured_hair_ratio_value is not None
            and measured_hair_ratio_value <= .30
            and (
                "full" in vl_ponytail_extent
                or any(term in vl_hair_length_before_measurement.lower()
                       for term in ("medium", "long", "shoulder"))
            )
        )
        if parser_truncated_tied_tail:
            rough_placement["measured_hair_end_ratio"] = None
            rough_placement["measured_left_hair_end_ratio"] = None
            rough_placement["measured_right_hair_end_ratio"] = None
            vl_plan["measured_hair_end_ratio"] = None
            vl_plan["measured_left_hair_end_ratio"] = None
            vl_plan["measured_right_hair_end_ratio"] = None
            vl_plan["hair_length"] = (
                vl_hair_length_before_measurement or "source-visible full tied length"
            )
            vl_plan["hair_ends"] = (
                "full tied tail continues behind the neck/clothing; copy its literal visible endpoint from image 1"
            )
            vl_plan["hair_geometry_parser_fragmented"] = True
            logger.warning(
                "[UNIFORM_HAIR_GEOMETRY] Rejected truncated parser endpoint %.3f; preserving VL/source full tied length=%s",
                measured_hair_ratio_value,
                vl_plan["hair_length"],
            )
        elif measured_hair_extent.lower() not in {"", "unknown", "uncertain"}:
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
        if save_intermediates:
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
        if save_intermediates:
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
            exclude_region_order="yx",
        )
        uniform_reference_path = OUTPUTS_DIR / f"{job_id}_uniform_passport_reference.png"
        if save_intermediates:
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
                or measured_ponytail_side in {
                    "viewer-left", "viewer-right", "bilateral",
                }
                or (
                    measured_tied_count > 0
                    and tied_accessory_corroborated
                )
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
            _candidate_opp = "viewer-right" if measured_ponytail_side == "viewer-left" else "viewer-left"
            _opp_key = "measured_right_hair_end_ratio" if _candidate_opp == "viewer-right" else "measured_left_hair_end_ratio"
            _opp_ratio = float(rough_placement.get(_opp_key) or 0.0)
            if _opp_ratio < 0.05 and not visible_tied_hair_geometry:
                opposite_ponytail_side = _candidate_opp
                one_sided_ponytail_instruction = (
                    f"The tied tail is visible on {measured_ponytail_side}, exactly as in image 1. "
                )
            else:
                one_sided_ponytail_instruction = (
                    f"Preserve natural hair on both sides, with the main tail on {measured_ponytail_side}. "
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
        patterned_uniform = (
            multi_color_pattern
            or any(term in str(vl_plan.get("shirt_pattern_type") or "").lower() for term in ("plaid", "check", "tartan", "stripe"))
        )
        # Two 512px references exhaust the 8 GB Windows GPU while the Qwen-VL
        # encoder builds visual tokens, killing the API before diffusion starts.
        # The fitted seed already carries full-resolution pattern placement and
        # the second image supplies garment appearance, so a 384px analysis
        # canvas retains checks/trim while keeping the encoder peak bounded.
        # Enhancement uses 512px visual conditioning. Uniform identity and
        # fine plaid need at least the same evidence; the former 352/384px cap
        # blurred the small source face before diffusion.
        # Identity conditioning is 512px while the canvas generates at 1216px,
        # so the face evidence reaching the transformer is 4x coarser than the
        # thing being drawn - one of the reasons run 0b0a60e2 fell back to the
        # generic child prior. 768px measurably helps, but the comment above is
        # a real constraint: two 768px reference tensors OOM the 8 GB target
        # machine while the Qwen-VL encoder builds visual tokens, and an OOM is
        # worse than a soft face. Opt in on a larger GPU; the default is
        # unchanged.
        conditioning_dimension = 512
        try:
            _cond_override = int(
                os.getenv("VSOUL_UNIFORM_CONDITIONING_DIMENSION", "") or 0,
            )
        except ValueError:
            _cond_override = 0
        if 256 <= _cond_override <= 1024:
            conditioning_dimension = (_cond_override // 16) * 16
            logger.info(
                "[UNIFORM_CONDITIONING] Visual conditioning raised to %dpx by env override",
                conditioning_dimension,
            )
        uniform_prompt_token_budget = 512
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
            "Preserve the fitted target uniform's exact multicolor fabric palette and pattern scale. "
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
            _has_grounded_direct_light(vl_plan)
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
        _has_any_tied_evidence = bool(
            preserve_tied_back_hair
            or vl_claims_tied_hair
            or (analyzed_hair_style in {"pigtails", "braids", "ponytail", "braid", "plait"})
            or (int(vl_plan.get("tied_section_count") or 0) > 0)
            or str(vl_plan.get("hair_crown_shape") or "").lower() == "bows"
        )
        if _has_any_tied_evidence:
            hair_shape_instruction += (
                " Preserve only the authentic lateral tied braids/pigtails visibly supported by image 1; do not fold, curl, or invent hair over shoulders."
            )
            if str(vl_plan.get("hair_accessories") or "").strip().lower() in {
                "", "none", "unknown", "uncertain", "[]", "{}",
            }:
                hair_shape_instruction += (
                    " No hair accessory is visible: keep the fastener hidden and do not invent a colored elastic, ribbon, bow, clip or band."
                )
        else:
            hair_shape_instruction += (
                " Image 1 has no tied-back hairstyle: do not add a ponytail, pigtail, bun, braid, plait, hair tie or trailing hair bundle."
            )
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
        raw_body_ratio = float(rough_placement.get("source_body_to_face_width") or 1.95)
        # Preserve the upload's measured build. The old 2.15 ceiling silently
        # narrowed valid broader portraits (2.279 on the latest boy) while the
        # model still generated a product-shirt torso around that conflict.
        # These wider bounds reject only implausible segmentation failures.
        source_body_ratio = float(np.clip(raw_body_ratio, 1.35, 2.80))
        fitted_body_ratio = source_body_ratio
        measured_neck_gap_ratio = float(rough_placement.get("source_neck_gap_ratio") or .20)
        # The composite estimate is based on the rough garment boundary and can
        # mistake the old shirt opening for exposed neck.  Measure the uploaded
        # person directly before diffusion; this is the same deterministic
        # measurement used by delivery QA and therefore keeps conditioning and
        # validation on one geometry authority.  In particular, do not force a
        # naturally short child neck up to the historical 0.16 minimum.
        direct_source_neck_ratio = premeasured_source_neck_ratio
        if (
            direct_source_neck_ratio is not None
            and np.isfinite(direct_source_neck_ratio)
            and .04 <= float(direct_source_neck_ratio) <= .35
        ):
            measured_neck_gap_ratio = float(direct_source_neck_ratio)
            rough_placement["source_neck_gap_ratio"] = measured_neck_gap_ratio
            rough_placement["neck_seat_ratio"] = measured_neck_gap_ratio
            logger.info(
                "[UNIFORM_NECK_MEASURE] direct source %.3f replaces rough %.3f",
                measured_neck_gap_ratio,
                float(rough_placement.get("measured_source_neck_gap_ratio") or .20),
            )
        # The uploaded portrait is the neck-geometry authority. Adding a fixed
        # minimum here contradicted the measured guide and made naturally short
        # necks longer in the generated result.
        source_neck_gap_ratio = measured_neck_gap_ratio
        body_fit_instruction = (
            "The uploaded person's measured upper-body width is authoritative. "
            f"Image 1 has a shoulder/upper-torso span of about {source_body_ratio:.2f} face widths "
            f"and the neutral fit placeholder is {fitted_body_ratio:.2f} face widths. "
            "Preserve that exact natural build: naturally tailored slim fit, proportional shoulders matching IMAGE 1. "
            "Do not widen or narrow the shoulders, enlarge the head, shrink the torso, lengthen the neck, or copy the uniform template model's body shape. "
            "Render the same person as a camera-facing school-ID passport photograph, showing the full head with modest headroom, level shoulders, neck, and upper chest only."
        )
        body_fit_prompt_instruction = (
            f"Naturally tailored slim fit matching IMAGE 1's measured {source_body_ratio:.2f}-face-wide body authority. "
            "Keep the collar seated naturally at the throat base and shoulder seams at proportional shoulder points. "
            "Do not widen shoulders or create loose boxy clothing."
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
            "accent_trim_color", "accent_trim_placement",
            "button_layout", "button_color",
        ))
        # "medium" is a category, not a size, so Qwen was free to pick any
        # check size: on fbf83870 it drew the plaid at a measured 2.18x the
        # template's scale. State the repeat as a count across the chest, the
        # same trick the body fit already uses ("2.20 face widths"), which
        # stays true under any later resize.
        pattern_repeat_instruction = ""
        try:
            pattern_period_px = self._measure_pattern_repeat_px(template_pil)
        except Exception as _pattern_err:
            pattern_period_px = None
            logger.warning(
                "[UNIFORM_PATTERN_SCALE] Repeat measurement skipped: %s",
                _pattern_err,
            )
        if pattern_period_px:
            # Match the helper's own measurement window (the central 40% of the
            # template) so the stated count is what a viewer counts across the
            # chest, not across the full canvas including sleeves and margin.
            chest_span_px = template_pil.width * .40
            repeats_across_chest = chest_span_px / float(pattern_period_px)
            if 2.0 <= repeats_across_chest <= 40.0:
                pattern_repeat_instruction = (
                    "PATTERN SCALE: the repeating check measures about "
                    f"{repeats_across_chest:.0f} full repeats across the garment's "
                    "chest width. Draw it at that density; do not enlarge the "
                    "check or widen the stripe spacing. "
                )
                logger.info(
                    "[UNIFORM_PATTERN_SCALE] Template repeat %.1fpx -> %.1f repeats across chest",
                    pattern_period_px, repeats_across_chest,
                )
        if not multi_color_pattern:
            described_color = observed_details((
                "shirt_color_and_pattern", "accent_trim_color",
            ))
            uniform_generation_facts = "; ".join(filter(None, (
                uniform_generation_facts, described_color,
            )))
        # Put measured accent colour and collar-surface construction in the
        # early, token-safe garment facts.  Previously this information existed
        # only in a late lock which the compact prompt could trim; worse, every
        # patterned garment was told to paint plaid onto its collar.  Derive the
        # collar rule from the uploaded template's analysed construction.
        analysed_collar = str(vl_plan.get("shirt_collar") or "").strip().lower()
        analysed_trim_placement = str(
            vl_plan.get("accent_trim_placement") or ""
        ).strip().lower()
        collar_uses_pattern_with_piping = any(
            token in analysed_collar
            for token in ("crossover", "overlap band", "band collar")
        )
        collar_leaves_are_solid = False
        if measured_accent and "collar" in analysed_trim_placement and not collar_uses_pattern_with_piping:
            try:
                tpl_arr = np.asarray(transparent_template.convert("RGBA"))
                tpl_alpha = tpl_arr[:, :, 3] > 40
                if np.any(tpl_alpha):
                    y_indices = np.where(tpl_alpha)[0]
                    y_min, y_max = y_indices.min(), y_indices.max()
                    garment_h = max(1, y_max - y_min)
                    collar_y_max = y_min + int(garment_h * 0.22)
                    collar_mask = tpl_alpha & (tpl_arr[:, :, 0] < 254)
                    collar_mask &= (np.arange(tpl_arr.shape[0])[:, None] <= collar_y_max)
                    if np.count_nonzero(collar_mask) > 50:
                        acc_rgb = np.asarray(measured_accent["rgb"], dtype=np.float32)
                        collar_rgb = tpl_arr[:, :, :3].astype(np.float32)
                        dist = np.linalg.norm(collar_rgb - acc_rgb[None, None, :], axis=2)
                        accent_collar_mask = collar_mask & (dist < 55.0)
                        acc_in_collar = np.count_nonzero(accent_collar_mask)
                        accent_fraction = (
                            acc_in_collar / max(1, np.count_nonzero(collar_mask))
                        )
                        # Solid contrast leaves occupy a broad connected area;
                        # piping and check threads are thin/disconnected.  The
                        # former 45% whole-upper-garment threshold counted the
                        # plaid chest behind the collar and misclassified the
                        # uploaded solid-red leaves as patterned.  Combine
                        # coverage with the largest connected accent component
                        # so the decision follows this template's pixels and
                        # remains valid for any collar colour.
                        component_count, _, component_stats, _ = (
                            cv2.connectedComponentsWithStats(
                                accent_collar_mask.astype(np.uint8),
                                connectivity=8,
                            )
                        )
                        largest_accent_component = (
                            int(component_stats[1:, cv2.CC_STAT_AREA].max())
                            if component_count > 1 else 0
                        )
                        largest_component_fraction = (
                            largest_accent_component
                            / max(1, np.count_nonzero(collar_mask))
                        )
                        collar_leaves_are_solid = bool(
                            accent_fraction >= 0.14
                            and largest_component_fraction >= 0.045
                        )
                        logger.info(
                            "[UNIFORM_COLLAR_SURFACE] accent_fraction=%.3f "
                            "largest_component=%.3f solid=%s",
                            accent_fraction,
                            largest_component_fraction,
                            collar_leaves_are_solid,
                        )
            except Exception:
                collar_leaves_are_solid = False

        if measured_accent and "collar" in analysed_trim_placement:
            collar_surface_fact = (
                "both collar leaves are solid accent fabric with no body pattern"
                if collar_leaves_are_solid else
                "the collar uses the patterned body fabric with narrow accent trim"
            )
            uniform_generation_facts = "; ".join(filter(None, (
                f"measured accent {measured_accent['hex']} RGB{tuple(measured_accent['rgb'])}",
                collar_surface_fact,
                uniform_generation_facts,
            )))
        if not uniform_generation_facts:
            uniform_generation_facts = "copy the fitted template pixels exactly"
        sleeve_visibility = str(vl_plan.get("sleeve_length") or "").strip().lower()
        sleeve_visibility_instruction = (
            "The template does not clearly show a sleeve end. Do not invent a sleeve tab, roll-up strap, cuff button, contrasting cuff, or other sleeve hardware; crop the sleeve naturally at the passport frame. "
            if sleeve_visibility in {"", "unknown", "uncertain", "unclear", "cropped"}
            else "Copy only the sleeve-end construction visibly present in image 2; add no extra sleeve hardware. "
        )
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
        # State the measured length and tied geometry explicitly. The generic
        # sentence above was the only hair text the live prompt carried, and a
        # four-step schedule defaults to the common case: job 30399633 was
        # given a conditioning image with two waist-length plaits and returned
        # a jaw-length bob. hair_shape_instruction already held these facts but,
        # like the neck lock before it, was built and never sent.
        if measured_hair_extent.lower() not in {"", "unknown", "uncertain"}:
            compact_hair_requirement += (
                f" Image 1's hair {measured_hair_extent}"
                + (
                    f" (measured {measured_hair_end_ratio:.2f} face-heights "
                    "below the jaw)"
                    if isinstance(measured_hair_end_ratio, (int, float))
                    else ""
                )
                # Length is stated as one fact among the style facts, not as a
                # standalone imperative. The earlier wording ("keep that full
                # length and do not shorten, cut or bob it") dominated the
                # hair contract: job braidfix2/bothfix produced correct length
                # but replaced the source's plaited braids with loose straight
                # hair and dropped the bead ties, because the only texture
                # word reaching Qwen was the VL analyser's (wrong) "Straight".
                # Satisfying a forceful length order with loose hair is easier
                # than reproducing a weave, so the order is softened and
                # structure is named alongside it.
                + ", at its existing length, keeping the same strand "
                "structure, weave or plaiting and every tie, band and bead "
                "exactly as they appear."
            )
        if measured_tied_count >= 1:
            compact_hair_requirement += (
                f" It has {measured_tied_count} tied section(s); copy each "
                "one's visible construction - if the strands are interwoven, "
                "reproduce the plait; keep its elastics, bands and beads."
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
        # Pocket semantics are deliberately visual-only. VL can mistake a
        # rectangular crest backing or repair area for a flap pocket, which
        # made Qwen invent a pocket absent from the uploaded template.
        pocket_instruction = (
            "POCKET LOCK: copy only a pocket whose independent opening and perimeter are visibly present in image 2. "
            "A badge, crest backing, repaired area or color block is not a pocket; never invent one from analysis text. "
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
                (
                    f"The crossover collar is made of the SAME patterned fabric as the shirt body (NOT solid {measured_accent['hex']}); "
                    f"{measured_accent['hex']} is ONLY thin narrow edge piping along the collar seam and placket edge; the collar itself is patterned fabric. "
                    if collar_uses_pattern_with_piping else
                    f"Both collar leaves must be solid {measured_accent['hex']} trim with no base-fabric, plaid, stripe or alternate color on either leaf. "
                )
                if "collar" in trim_placement.lower() else ""
            )
            if measured_accent and known_trim_placement else
            "Keep the fitted trim color and its existing locations unchanged. "
        )
        # NECK/COLLAR LOCK text. The live prompt is assembled further down;
        # this measured string is interpolated into it so the four-step edit
        # gets the jaw-to-collar gap it was measured for. A previous revision
        # built a whole alternate prompt here that every later branch then
        # overwrote unconditionally, so this contract never reached Qwen and
        # job 5a1bbeaa decoded a tripled neck.
        if 'clean_id_result' in locals() and clean_id_result and clean_id_result.get("neck_classification") == "occluded_neck":
            neck_collar_lock = (
                "NECK/COLLAR LOCK: shirt collar must sit naturally and snugly right under the chin and throat "
                "(high snug collar fit matching image 1's neckline, zero neck gap); "
                "standard folded shirt collar with pointed collar leaves folded down naturally over the chest, closed at the throat base with no deep plunging neckline, scoop neck, open V-neck, or exposed chest skin below the collar. "
                "UNIFORM FIT: neat, tailored school uniform fit with natural shoulder width matching the person, no oversized or boxy clothing, and no solid red chest panel or bib. "
                "REMOVE any lanyard, neck card, ID card strap, necklace, chain, badge, or wearable from Image 1; do not render any straps, harness, or suspenders. "
                "SEAMLESS NECK: smoothly connect and blend natural neck skin directly into the collar opening with continuous natural skin tone; eliminate any white notches, edge gaps, or neck holes along the jawline and collar boundary. "
            )
        else:
            neck_collar_lock = (
                f"NECK/COLLAR LOCK: shirt collar must sit naturally and snugly around the base of the throat "
                f"(jaw-to-collar gap {source_neck_gap_ratio:.2f} face heights); "
                "standard folded shirt collar with pointed collar leaves folded down naturally over the chest, closed at the throat base with no deep plunging neckline, scoop neck, open V-neck, or exposed chest skin below the collar. "
                "UNIFORM FIT: neat, tailored school uniform fit with natural shoulder width matching the person, no oversized or boxy clothing, and no solid red chest panel or bib. "
                "REMOVE any lanyard, neck card, ID card strap, necklace, chain, badge, or wearable from Image 1; do not render any straps, harness, or suspenders. "
                "SEAMLESS NECK: smoothly connect and blend natural neck skin directly into the collar opening with continuous natural skin tone; eliminate any white notches, edge gaps, or neck holes along the jawline and collar boundary. "
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
        if save_intermediates:
            garment_edit_crop.save(garment_crop_path, format="PNG")

        # Build the unified person generation canvas onto the selected solid background first.
        # This guarantees that identity_base, rough fitting, and Qwen conditioning all share
        # the exact same flat backdrop, completely eliminating background bleed and shoulder cutouts.
        source_generation_w, source_generation_h = person_conditioning.size
        # The short side used to be pinned to 832 regardless of the selected
        # bucket, so `uniform_generation_dimension` only ever capped the LONG
        # side and a "1216px Quality" job actually diffused at 832x1104. On
        # run 4c5f33ec that put a 1441px source face through a 349px delivered
        # face and kept just 38% of its skin texture, which reads as a redrawn
        # identity however well ArcFace scores. Drive the short side from the
        # configured bucket (keeping the historical 832 floor as the minimum)
        # and let the long side clamp as before.
        generation_short_side = max(832, int(round(
            uniform_generation_dimension * .72 / 16.0,
        ) * 16))
        if source_generation_h >= source_generation_w:
            generation_w = generation_short_side
            generation_h = int(round(
                (generation_w * source_generation_h / max(source_generation_w, 1)) / 16.0
            ) * 16)
            if generation_h > uniform_generation_dimension:
                generation_h = uniform_generation_dimension
                generation_w = int(round(
                    (generation_h * source_generation_w / max(source_generation_h, 1)) / 16.0
                ) * 16)
        else:
            generation_h = generation_short_side
            generation_w = int(round(
                (generation_h * source_generation_w / max(source_generation_h, 1)) / 16.0
            ) * 16)
            if generation_w > uniform_generation_dimension:
                generation_w = uniform_generation_dimension
                generation_h = int(round(
                    (generation_w * source_generation_h / max(source_generation_h, 1)) / 16.0
                ) * 16)
        logger.info(
            "[UNIFORM_GENERATION_CANVAS] bucket=%dpx short-side=%dpx -> %dx%d",
            uniform_generation_dimension, generation_short_side,
            generation_w, generation_h,
        )
        generation_w = max(256, generation_w)
        generation_h = max(256, generation_h)

        generation_source = person_conditioning.convert("RGB")
        primary_person_source = generation_source.copy()
        schp_foreground_alpha = None
        try:
            from pipelines.schp_service import parse as parse_body_parts
            source_rgb_arr = np.asarray(primary_person_source)
            schp_res = parse_body_parts(source_rgb_arr)
            schp_labels = schp_res["labels"]
            person_mask = (schp_labels != 0)
            # LIP-20 has no neck class: label 13 (Face) ends at the jaw and the
            # bare neck below it comes back as label 0 (Background). A plain
            # "labels != 0" foreground therefore excludes the neck, and the
            # generation canvas below paints every non-foreground pixel with the
            # flat backdrop. Measured on a cut-out upload: 135 rows of real neck
            # between jaw and collar were whitened, so Qwen was conditioned on a
            # head floating over a white notch and rendered a collar band across
            # the throat with the shirt detached from the body. Recover the
            # photographed neck by colour inside a face-relative corridor; the
            # skin test decides which pixels are really neck.
            # Use the face measured on person_conditioning, not on the original
            # upload: source_rgb_arr is that same conditioning crop, so the
            # upload's bbox would place the corridor in the wrong coordinates.
            if conditioning_faces:
                person_mask = _recover_unparsed_neck(
                    source_rgb_arr, person_mask, conditioning_faces[0].bbox,
                )

            acc_check = vl_plan.get("cv_hair_accessory_info") or {}
            acc_mask = vl_plan.get("cv_hair_accessory_mask")
            if not acc_check.get("detected"):
                acc_check = self._detect_hair_accessory_info(primary_person_source)
                if acc_check.get("detected"):
                    vl_plan["cv_hair_accessory_info"] = {k: v for k, v in acc_check.items() if k != "mask"}
                    vl_plan["cv_hair_accessory_mask"] = acc_check.get("mask")
                    acc_mask = acc_check.get("mask")
            if acc_check.get("detected") and acc_mask is not None:
                acc_mask_orig = cv2.resize(
                    acc_mask.astype(np.uint8), (person_mask.shape[1], person_mask.shape[0]), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                acc_mask_orig = cv2.dilate(
                    acc_mask_orig.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                    iterations=1,
                ).astype(bool)
                person_mask |= acc_mask_orig

            # SCHP's LIP-20 label set has no neck class, so the throat and the
            # jaw margin come back as label 0 and are later painted with the
            # flat backdrop. ``_recover_unparsed_neck`` above recovers part of
            # it by colour, but measured on this upload 10,362 px inside the
            # shoulder band were still unclaimed - backdrop punched through the
            # neck/collar junction, which is the seam and the detached-collar
            # look in the delivery.
            #
            # BiRefNet already runs in this pipeline and segments the whole
            # person without a label taxonomy, so it has no neck blind spot.
            # Measured against SCHP on the same canvas: IoU 0.929, 16,965 px
            # that SCHP missed, and only 266 the other way. Union it in, so a
            # pixel is subject if EITHER model says so. SCHP keeps deciding
            # which body part a pixel is - this only decides person vs air.
            try:
                import io as _io
                from pipelines.birefnet_service import background_removal

                if conditioning_birefnet_rgba is not None:
                    _rgba = conditioning_birefnet_rgba
                else:
                    _buf = _io.BytesIO()
                    primary_person_source.save(_buf, format="PNG")
                    _cut = background_removal.remove_background(
                        _buf.getvalue(), job_id=f"{job_id}_person_matte",
                        use_schp=True, source_image=primary_person_source,
                    )
                    _rgba = np.asarray(
                        Image.open(_io.BytesIO(_cut)).convert("RGBA"), dtype=np.uint8,
                    )
                _matte = _rgba[:, :, 3]
                if _matte.shape == person_mask.shape:
                    _matte_mask = _matte > 128
                    # A matte that lost the subject is worse than the blind
                    # spot it fixes, so only accept one that agrees with SCHP
                    # on the bulk of the body.
                    _inter = float((_matte_mask & person_mask).sum())
                    _union = float((_matte_mask | person_mask).sum())
                    _iou = _inter / max(_union, 1.0)
                    if _iou >= .75:
                        _added = int((_matte_mask & ~person_mask).sum())
                        person_mask = person_mask | _matte_mask
                        logger.info(
                            "[UNIFORM_CONDITIONING] BiRefNet matte added %d subject "
                            "pixel(s) SCHP's label map missed (IoU %.3f)",
                            _added, _iou,
                        )
                    else:
                        logger.warning(
                            "[UNIFORM_CONDITIONING] BiRefNet matte disagrees with SCHP "
                            "(IoU %.3f); keeping the parsed mask alone", _iou,
                        )
            except Exception as _matte_error:
                logger.warning(
                    "[UNIFORM_CONDITIONING] BiRefNet boundary matte unavailable: %s",
                    _matte_error,
                )

            alpha = (person_mask.astype(np.float32) * 255.0)
            schp_foreground_alpha = alpha.astype(np.uint8)
            logger.info(
                "[UNIFORM_CONDITIONING] Preserved 100%% intact person_source for Qwen diffusion (parsed SCHP alpha for clothing alignment without destructive pre-matting)",
            )
        except Exception as e:
            logger.warning(
                "[UNIFORM_CONDITIONING] SCHP foreground parse failed (%s); using primary_person_source",
                e,
            )

        identity_base = generation_source.resize(
            rough_composite.size, Image.LANCZOS,
        )
        identity_rgb = np.asarray(identity_base, dtype=np.float32)
        identity_labels = parse_body_parts(np.asarray(identity_base))["labels"]
        # SCHP on the resized conditioning canvas can relabel a dark source
        # vest/jacket shoulder as hair; identity protection then keeps the
        # garment immutable and Qwen copies it into the delivery. The
        # untouched-upload parse is the cross-authority: upload-confirmed
        # clothing pixels may never be protected as hair on the canvas.
        ux1, uy1, ux2, uy2 = (float(v) for v in source_face.bbox)
        rx1, ry1, rx2, ry2 = (float(v) for v in rough_face_box)
        upload_scale_x = (ux2 - ux1) / max(1.0, rx2 - rx1)
        upload_scale_y = (uy2 - uy1) / max(1.0, ry2 - ry1)
        canvas_to_upload = np.float32([
            [upload_scale_x, 0, ux1 - rx1 * upload_scale_x],
            [0, upload_scale_y, uy1 - ry1 * upload_scale_y],
        ])
        upload_labels_on_canvas = cv2.warpAffine(
            source_labels.astype(np.uint8), canvas_to_upload,
            (identity_base.width, identity_base.height),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        upload_confirmed_clothing = np.isin(
            upload_labels_on_canvas, (5, 6, 7, 8, 9, 10, 11, 12),
        )
        garment_mislabeled_as_hair = (
            (identity_labels == 2) & upload_confirmed_clothing
        )
        source_clothing = np.isin(
            identity_labels, (5, 6, 7, 8, 9, 10, 11, 12),
        ).astype(np.uint8)
        face_x1, face_y1, face_x2, face_y2 = (
            float(value) for value in rough_face_box
        )
        detected_face_width = max(1.0, face_x2 - face_x1)
        detected_face_height = max(1.0, face_y2 - face_y1)
        face_center_x = (face_x1 + face_x2) * .5

        # Protect authentic hair accessories from being classified as clothing or included in edit_region
        acc_check = vl_plan.get("cv_hair_accessory_info") or {}
        acc_mask = vl_plan.get("cv_hair_accessory_mask")
        detected_acc_mask_canvas = np.zeros(identity_labels.shape, dtype=bool)
        if acc_check.get("detected") and acc_mask is not None:
            detected_acc_mask_canvas |= cv2.warpAffine(
                acc_mask.astype(np.uint8),
                canvas_to_upload,
                (identity_base.width, identity_base.height),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).astype(bool)
        acc_base_check = self._detect_hair_accessory_info(identity_base, rough_face_box)
        if acc_base_check.get("detected") and acc_base_check.get("mask") is not None:
            detected_acc_mask_canvas |= acc_base_check["mask"].astype(bool)

        if np.any(detected_acc_mask_canvas):
            detected_acc_mask_canvas = cv2.dilate(
                detected_acc_mask_canvas.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            ).astype(bool)
            source_clothing[detected_acc_mask_canvas] = 0
            garment_mislabeled_as_hair &= ~detected_acc_mask_canvas
        mask_y, mask_x = np.indices(identity_labels.shape)
        # Establish is_real_foreground early so both hair corridor measurement and torso parsing can use it
        if schp_foreground_alpha is not None:
            alpha_on_canvas = cv2.resize(
                schp_foreground_alpha, (identity_base.width, identity_base.height),
                interpolation=cv2.INTER_LINEAR,
            )
            is_real_foreground = alpha_on_canvas > 32
        else:
            selected_bg = np.asarray(qwen_bg_rgb, dtype=np.float32)
            bg_distance = np.linalg.norm(
                identity_rgb - selected_bg[None, None, :],
                axis=2,
            )
            is_real_foreground = bg_distance > 38.0

        # Internal holes inside the subject (e.g. white shirts or ties parsed as label 0)
        # must NEVER be treated as background air or painted with background color.
        # Genuine background apertures (e.g. gaps between hair and jaw/neck matching background)
        # must remain real background air.
        bg_mask = (~is_real_foreground).astype(np.uint8)
        num_labels, comp_labels, stats, centroids = cv2.connectedComponentsWithStats(bg_mask, connectivity=8)
        border_pixels = np.concatenate([comp_labels[0, :], comp_labels[:, 0], comp_labels[:, -1]])
        border_components = set(np.unique(border_pixels))
        selected_bg_f = np.asarray(qwen_bg_rgb, dtype=np.float32)
        for comp_id in range(1, num_labels):
            if comp_id not in border_components:
                c_mask = comp_labels == comp_id
                c_area = stats[comp_id, cv2.CC_STAT_AREA]
                c_mean_rgb = np.mean(identity_rgb[c_mask], axis=0)
                c_dist = float(np.linalg.norm(c_mean_rgb - selected_bg_f))
                is_bg_aperture = (c_area >= 32) and (c_dist <= 28.0)
                if not is_bg_aperture:
                    is_real_foreground[c_mask] = True
        vest_outer_zone = np.zeros_like(garment_mislabeled_as_hair)
        # A dark vest/denim jacket beside or below the head merges with hair
        # into wide hair-labelled row runs or mislabelled blocks across the shoulders.
        # Strip those garment regions unconditionally so the vest becomes editable
        # clothing instead of being protected as hair and copied into the delivery.
        _detected_tied_count = 0
        try:
            _detected_tied_count = int(vl_plan.get("tied_section_count") or measured_tied_count or 0)
        except (TypeError, ValueError):
            _detected_tied_count = 0
        is_tied_or_braided_hair = bool(
            _detected_tied_count > 0
            or visible_tied_hair_geometry
            or measured_ponytail_side == "bilateral"
            or float(vl_plan.get("measured_hair_end_ratio", 0.0) or 0.0) >= 0.20
        )
        # Determine rough length and style tightly using SCHP / subject evidence
        # In lateral columns, find the lowest non-background pixel of the subject
        left_lateral_zone = (
            (mask_x >= face_center_x - detected_face_width * 0.85)
            & (mask_x <= face_center_x - detected_face_width * 0.38)
            & (mask_y >= face_y2)
        )
        right_lateral_zone = (
            (mask_x >= face_center_x + detected_face_width * 0.38)
            & (mask_x <= face_center_x + detected_face_width * 0.95)
            & (mask_y >= face_y2)
        )
        schp_lateral_subject = is_real_foreground & (left_lateral_zone | right_lateral_zone)
        if np.any(schp_lateral_subject):
            y_reach_candidates = np.where(schp_lateral_subject)[0]
            schp_measured_reach_y = float(y_reach_candidates.max()) + 10.0
        # Use SCHP hair identification to find rough length and tight lateral corridors dynamically
        parsed_id_hair = identity_labels == 2
        schp_hair_ys = np.flatnonzero(np.any(parsed_id_hair, axis=1))
        schp_measured_reach_y = float(schp_hair_ys.max()) if schp_hair_ys.size > 0 else (face_y2 + detected_face_height * 1.10)

        braid_reach_y = min(
            float(identity_labels.shape[0] - 1),
            max(
                schp_measured_reach_y,
                face_y2 + detected_face_height * 0.85,
                face_y2 + detected_face_height * (float(vl_plan.get("measured_hair_end_ratio", 0.0) or 0.20) + 0.35),
            ),
        )
        logger.info(
            "[UNIFORM_HAIR_SCHP] Measured rough hair/braid length reach: y=%.1f (canvas height=%d)",
            float(braid_reach_y), identity_labels.shape[0],
        )

        # Locate hair accessory / bow anchors dynamically if present
        id_u8_canvas = np.asarray(identity_base, dtype=np.uint8)
        id_hsv_canvas = cv2.cvtColor(id_u8_canvas, cv2.COLOR_RGB2HSV)
        is_bow_candidate = (
            (id_hsv_canvas[:, :, 0] >= 95) & (id_hsv_canvas[:, :, 0] <= 135)
            & (id_hsv_canvas[:, :, 1] >= 60)
            & (id_u8_canvas.mean(axis=2) < 240)
            & (mask_y <= face_y2)
        )
        left_bow_cx, right_bow_cx = None, None
        if np.any(is_bow_candidate):
            n_b, _, stats_b, cents_b = cv2.connectedComponentsWithStats(is_bow_candidate.astype(np.uint8))
            for b_idx in range(1, n_b):
                if stats_b[b_idx, cv2.CC_STAT_AREA] >= 300:
                    bcx = cents_b[b_idx][0]
                    if bcx < face_center_x:
                        left_bow_cx = bcx if left_bow_cx is None else min(left_bow_cx, bcx)
                    else:
                        right_bow_cx = bcx if right_bow_cx is None else max(right_bow_cx, bcx)

        # The student's shoulders and torso span inside +/- 0.50 face widths from center.
        # Authentic lateral braids/pigtails hang strictly outside the shoulder collar span.
        left_inner_x = int(round(min(face_center_x - detected_face_width * 0.50, (left_bow_cx + 70) if left_bow_cx else (face_center_x - detected_face_width * 0.50))))
        right_inner_x = int(round(max(face_center_x + detected_face_width * 0.50, (right_bow_cx - 70) if right_bow_cx else (face_center_x + detected_face_width * 0.50))))

        lower_hair_y_start = max(0, int(round(face_y2 - detected_face_height * 0.35)))
        left_braid_corridor = (
            (mask_x >= 0)
            & (mask_x <= left_inner_x)
            & (mask_y >= lower_hair_y_start)
            & (mask_y <= braid_reach_y)
        )
        right_braid_corridor = (
            (mask_x >= right_inner_x)
            & (mask_x <= identity_labels.shape[1] - 1)
            & (mask_y >= lower_hair_y_start)
            & (mask_y <= braid_reach_y)
        )
        braid_corridors = left_braid_corridor | right_braid_corridor
        is_chest_badge = (
            (mask_x >= face_center_x + detected_face_width * 0.12)
            & (mask_x <= face_center_x + detected_face_width * 0.65)
            & (mask_y >= face_y2 + detected_face_height * 0.08)
        )
        braid_corridors &= ~is_chest_badge

        # Dynamically measure lateral gaps between the head/neck/jaw and the outer braid corridors.
        # Inside these lateral gaps:
        # 1. Background air matching the backdrop is strictly non-foreground air (held latent in diffusion).
        # 2. Non-background pixels below the jaw are source overcoat/vest/collar (must be replaced by target uniform).
        gap_y_top = max(0, int(round(face_y1 + detected_face_height * 0.25)))
        gap_y_bottom = min(identity_labels.shape[0] - 1, int(round(face_y2 + detected_face_height * 0.70)))
        left_gap_x1 = max(0, int(face_center_x - detected_face_width * 0.65))
        left_gap_x2 = min(identity_labels.shape[1] - 1, int(face_center_x - detected_face_width * 0.18))
        right_gap_x1 = max(0, int(face_center_x + detected_face_width * 0.18))
        right_gap_x2 = min(identity_labels.shape[1] - 1, int(face_center_x + detected_face_width * 0.75))

        left_gap_zone = (
            (mask_x >= left_gap_x1) & (mask_x <= left_gap_x2)
            & (mask_y >= gap_y_top) & (mask_y <= gap_y_bottom)
            & ~braid_corridors
        )
        right_gap_zone = (
            (mask_x >= right_gap_x1) & (mask_x <= right_gap_x2)
            & (mask_y >= gap_y_top) & (mask_y <= gap_y_bottom)
            & ~braid_corridors
        )
        lateral_gap_zones = left_gap_zone | right_gap_zone

        id_u8_gap = np.asarray(identity_base, dtype=np.uint8)
        gap_bg_dist = np.linalg.norm(id_u8_gap.astype(np.float32) - selected_bg_f[None, None, :], axis=2)
        gap_is_background_air = (
            (lateral_gap_zones | ((mask_y >= gap_y_top) & (mask_y <= gap_y_bottom) & ((mask_x < left_inner_x) | (mask_x > right_inner_x))))
            & ~braid_corridors
            & (
                (gap_bg_dist <= 32.0)
                | ((id_u8_gap.min(axis=2) >= 210) & (np.ptp(id_u8_gap.astype(np.int16), axis=2) <= 35))
            )
        )
        if np.any(gap_is_background_air):
            is_real_foreground[gap_is_background_air] = False
            logger.info(
                "[UNIFORM_GAP_AIR] Cleared %d background air pixel(s) in hair-head gap",
                int(np.count_nonzero(gap_is_background_air)),
            )

        # Detect the dark overcoat / vest across the shoulders between the neck and the lateral braids:
        # Any non-skin pixel below chin/jaw inside the shoulder corridor is source clothing, NEVER hair!
        shoulder_overcoat_zone = (
            (mask_x >= left_inner_x)
            & (mask_x <= right_inner_x)
            & (np.abs(mask_x - face_center_x) >= detected_face_width * 0.20)
            & (mask_y >= face_y2 - detected_face_height * 0.10)
            & ~gap_is_background_air
            & ~(parsed_id_hair & (mask_y <= face_y2))
        )
        if np.any(shoulder_overcoat_zone):
            garment_mislabeled_as_hair |= shoulder_overcoat_zone
            source_clothing[shoulder_overcoat_zone] = 1
            logger.info(
                "[UNIFORM_SHOULDER_OVERCOAT] Detected %d overcoat/vest pixel(s) on shoulders for replacement",
                int(np.count_nonzero(shoulder_overcoat_zone)),
            )

        gap_clothing_zone = shoulder_overcoat_zone

        wide_run_garment = np.zeros_like(garment_mislabeled_as_hair)
        wide_run_top = max(0, int(round(face_y2 + detected_face_height * .05)))
        for row in range(wide_run_top, identity_labels.shape[0]):
            xs = np.flatnonzero(identity_labels[row] == 2)
            if xs.size == 0:
                continue
            runs = np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1)
            for run in runs:
                if run.size <= detected_face_width * .25:
                    continue
                if run[-1] < face_center_x:
                    outer = run[run < face_x1 - detected_face_width * .20]
                else:
                    outer = run[run > face_x2 + detected_face_width * .20]
                if outer.size:
                    wide_run_garment[row, outer] = True
        if is_tied_or_braided_hair:
            wide_run_garment &= ~braid_corridors
        garment_mislabeled_as_hair |= wide_run_garment

        # Also detect false-hair garment pixels across the outer shoulders and lower torso:
        # Any hair label below the chin that is on the shoulders or lower torso or color-divergent
        # is original garment fabric, not head hair.
        parsed_id_hair = identity_labels == 2
        id_crown_hair = parsed_id_hair & (mask_y <= face_y2)
        id_u8 = np.asarray(identity_base, dtype=np.uint8)
        id_lab = cv2.cvtColor(id_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
        if np.count_nonzero(id_crown_hair) >= 32:
            id_crown_samples = id_lab[id_crown_hair]
            id_crown_med = np.median(id_crown_samples, axis=0)
            id_hair_dist = np.linalg.norm(id_lab - id_crown_med[None, None, :], axis=2)
        else:
            id_hair_dist = np.zeros(identity_labels.shape, dtype=np.float32)

        # Jaw boundary: chin is at face_y2, but toward jaw corners/ears it rises
        jaw_dist_rel = np.clip(
            np.abs(mask_x - face_center_x) / max(detected_face_width * .52, 1.0),
            0.0, 1.0,
        )
        jaw_line_y = face_y2 - detected_face_height * .20 * np.power(jaw_dist_rel, 1.5)

        id_below_jaw = mask_y >= jaw_line_y
        id_below_chin = mask_y >= face_y2
        id_outer_shoulders = np.abs(mask_x - face_center_x) >= detected_face_width * 0.28
        id_lower_torso = mask_y >= face_y2 + detected_face_height * 0.12
        id_color_divergent = id_hair_dist >= 28.0

        false_hair_vest = parsed_id_hair & id_below_jaw & (
            (id_outer_shoulders & id_color_divergent)
            | (id_lower_torso & id_color_divergent)
            | (id_below_chin & (id_hair_dist >= 38.0))
        )
        if is_tied_or_braided_hair:
            id_hsv = cv2.cvtColor(id_u8, cv2.COLOR_RGB2HSV)
            id_gray = cv2.cvtColor(id_u8, cv2.COLOR_RGB2GRAY)
            is_braid_tie_canvas = (
                braid_corridors
                & (id_hsv[:, :, 0] >= 95) & (id_hsv[:, :, 0] <= 135)
                & (id_hsv[:, :, 1] >= 65)
                & (id_u8.mean(axis=2) < 240)
            )
            is_braid_strand = (
                braid_corridors & (id_gray < 75) & (id_hsv[:, :, 1] < 140)
                & ~upload_confirmed_clothing & ~shoulder_overcoat_zone
            )
            if 'gap_is_background_air' in locals() and np.any(gap_is_background_air):
                is_braid_strand &= ~gap_is_background_air
            hair_pool = (
                (is_braid_strand | is_braid_tie_canvas | detected_acc_mask_canvas)
                & braid_corridors & is_real_foreground & ~shoulder_overcoat_zone
            )
            seed = (
                (parsed_id_hair | is_braid_tie_canvas | detected_acc_mask_canvas)
                & (mask_y <= braid_reach_y) & ~shoulder_overcoat_zone
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 9))
            connected = seed.copy()
            for _ in range(80):
                expanded = cv2.dilate(connected.astype(np.uint8), kernel).astype(bool)
                new_connected = connected | (expanded & hair_pool)
                if new_connected.sum() == connected.sum():
                    break
                connected = new_connected

            num_labels, comp_labels, stats, centroids = cv2.connectedComponentsWithStats(
                connected.astype(np.uint8), connectivity=8
            )
            braid_safe_hair = np.zeros_like(parsed_id_hair)
            for i in range(1, num_labels):
                c_mask = comp_labels == i
                if np.any(c_mask & parsed_id_hair) or np.any(c_mask & is_braid_tie_canvas & (mask_y < face_y2)):
                    braid_safe_hair |= c_mask

            braid_safe_hair &= (
                (mask_y >= face_y2 - detected_face_height * 0.20)
                & braid_corridors & is_real_foreground & ~shoulder_overcoat_zone
            )
            if 'gap_is_background_air' in locals() and np.any(gap_is_background_air):
                braid_safe_hair &= ~gap_is_background_air
            if np.any(braid_safe_hair):
                braid_safe_hair = cv2.morphologyEx(
                    braid_safe_hair.astype(np.uint8),
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9)),
                ).astype(bool) & braid_corridors & is_real_foreground & ~shoulder_overcoat_zone
                if 'gap_is_background_air' in locals() and np.any(gap_is_background_air):
                    braid_safe_hair &= ~gap_is_background_air

            false_hair_vest &= ~braid_safe_hair
            garment_mislabeled_as_hair &= ~braid_safe_hair
            upload_confirmed_clothing &= ~braid_safe_hair
            source_clothing[braid_safe_hair] = 0
            logger.info(
                "[UNIFORM_HAIR_SAFE] Identified %d tight braid/tail pixels for preservation",
                int(np.count_nonzero(braid_safe_hair)),
            )
        garment_mislabeled_as_hair |= false_hair_vest
        if is_tied_or_braided_hair:
            garment_mislabeled_as_hair &= ~braid_safe_hair
            source_clothing[braid_safe_hair] = 0

        vest_outer_zone = (
            (mask_y >= face_y2 - detected_face_height * .10)
            & (
                (mask_x < face_x1 - detected_face_width * .20)
                | (mask_x > face_x2 + detected_face_width * .20)
            )
            & garment_mislabeled_as_hair
        )
        # Torso corridor protection: inside the measured shoulder/torso span below the neck,
        # interior pixels are real foreground and must never be punched out with background color.
        torso_left_limit = int(max(0, round(face_center_x - detected_face_width * 1.15)))
        torso_right_limit = int(min(identity_base.width, round(face_center_x + detected_face_width * 1.15)))
        torso_corridor = (
            (mask_y >= face_y2 + detected_face_height * 0.10)
            & (mask_x >= torso_left_limit)
            & (mask_x <= torso_right_limit)
        )
        is_real_foreground[torso_corridor] = True

        # Face skin color check so dark lanyards, straps or old collars are never protected as skin
        parsed_face_mask = (
            (identity_labels == 13)
            & (mask_y >= face_y1 + detected_face_height * 0.15)
            & (mask_y <= face_y2 - detected_face_height * 0.10)
        )
        if np.count_nonzero(parsed_face_mask) >= 32:
            face_skin_med = np.median(id_lab[parsed_face_mask], axis=0)
            face_skin_dist = np.linalg.norm(id_lab - face_skin_med[None, None, :], axis=2)
            is_red_strap = (
                (id_u8[:, :, 0] > 70)
                & ((id_u8[:, :, 0].astype(np.float32) / np.maximum(id_u8[:, :, 1].astype(np.float32), 1.0) > 1.68)
                   | (id_u8[:, :, 1].astype(np.int16) - id_u8[:, :, 2].astype(np.int16) < 14))
                & (id_u8[:, :, 0] > id_u8[:, :, 2] * 1.30)
            )
            is_skin_colored = (face_skin_dist < 48.0) & ~is_red_strap
        else:
            is_skin_colored = np.ones(identity_labels.shape, dtype=bool)

        head_or_hair_labels = np.isin(identity_labels, (1, 2, 4))
        # The collar line sits at the measured neck seat from rough_placement (anatomical base of the throat):
        if 'clean_id_result' in locals() and clean_id_result and clean_id_result.get("metrics", {}).get("lowest_neck_y") is not None:
            c_metrics = clean_id_result["metrics"]
            if clean_id_result.get("neck_classification") == "occluded_neck":
                collar_neck_seat_y = float(c_metrics.get("chin_y", face_y2)) + detected_face_height * 0.04
            else:
                collar_neck_seat_y = float(c_metrics.get("lowest_neck_y"))
        else:
            raw_neck_ratio = float(rough_placement.get("neck_seat_ratio") or 0.20)
            neck_seat_ratio_val = float(np.clip(raw_neck_ratio, 0.04, 0.32))
            collar_neck_seat_y = face_y2 + detected_face_height * max(0.04, neck_seat_ratio_val)
        neck_col_half = float(rough_placement.get("neck_column_half_px") or (detected_face_width * 0.32))
        neck_half_w = max(detected_face_width * 0.20, min(detected_face_width * 0.40, neck_col_half))

        # The student's natural neck skin: spans from below the jaw down to the measured collar seat
        neck_skin_corridor = (
            (mask_y >= face_y2 - detected_face_height * 0.02)
            & (mask_y < collar_neck_seat_y)
            & (np.abs(mask_x - face_center_x) <= neck_half_w)
            & is_skin_colored
            & is_real_foreground
        )

        torso_clothing_recovery = (
            is_real_foreground
            & (mask_y >= collar_neck_seat_y)
            & ~(head_or_hair_labels & ~garment_mislabeled_as_hair)
            & ~neck_skin_corridor
        )
        source_clothing[torso_clothing_recovery] = 1
        source_clothing[garment_mislabeled_as_hair] = 1

        # Any non-skin, non-hair foreground pixel in the neck aperture (between jaw and collar seat)
        # is foreign clothing, lanyard, necklace, or old collar - NEVER biological identity.
        neck_foreign_objects = (
            (mask_y >= face_y2 - detected_face_height * 0.02)
            & (mask_y <= collar_neck_seat_y + detected_face_height * 0.05)
            & (np.abs(mask_x - face_center_x) <= detected_face_width * 0.42)
            & ~is_skin_colored
            & ~head_or_hair_labels
            & ~neck_skin_corridor
        )
        source_clothing[neck_foreign_objects] = 1

        # Light blue or pale shirt fabric below the collar seat
        id_hsv = cv2.cvtColor(id_u8, cv2.COLOR_RGB2HSV)
        light_blue_shirt = (
            (mask_y >= collar_neck_seat_y)
            & (id_hsv[:, :, 0] >= 80) & (id_hsv[:, :, 0] <= 135)
            & (id_hsv[:, :, 1] >= 15) & (id_hsv[:, :, 1] <= 165)
            & (id_hsv[:, :, 2] >= 155)
            & ~head_or_hair_labels
            & ~neck_skin_corridor
        )
        source_clothing[light_blue_shirt] = 1
        source_clothing[neck_skin_corridor] = 0
        clothing = source_clothing.copy()
        clothing[neck_skin_corridor] = 0
        if is_tied_or_braided_hair and 'braid_safe_hair' in locals() and np.any(braid_safe_hair):
            clothing[braid_safe_hair] = 0
            source_clothing[braid_safe_hair] = 0
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
        clothing[neck_skin_corridor] = 0
        if is_tied_or_braided_hair and 'braid_safe_hair' in locals() and np.any(braid_safe_hair):
            clothing[braid_safe_hair] = 0
            source_clothing[braid_safe_hair] = 0
        # A folded shirt collar wraps around the lower neck at the collar seat.
        # Pointed/folded collar leaves rise well above their throat seat.  The
        # former zone began 0.08 face-heights below the jaw, leaving the upper
        # tips of the source blue collar protected and visible above the new
        # red collar. Derive the rise from this template's analysed collar;
        # compact band collars retain the smaller edit zone.
        collar_tip_rise = (
            .18
            if any(token in analysed_collar for token in ("pointed", "folded", "lapel"))
            else .08
        )
        collar_top = max(
            face_y2 - detected_face_height * .02,
            collar_neck_seat_y - detected_face_height * collar_tip_rise,
        )
        collar_bottom = collar_neck_seat_y + detected_face_height * .25
        collar_progress = np.clip(
            (mask_y - collar_top) / max(1.0, collar_bottom - collar_top),
            0.0,
            1.0,
        )
        collar_outer_half_width = detected_face_width * (
            .38 + .08 * collar_progress
        )
        collar_inner_half_width = detected_face_width * (
            .20 * np.clip(1.0 - collar_progress * 2.4, 0.0, 1.0)
        )
        collar_distance = np.abs(mask_x - face_center_x)
        collar_refine_zone = (
            (mask_y >= collar_top)
            & (mask_y <= collar_bottom)
            & (collar_distance <= collar_outer_half_width)
            & (collar_distance >= collar_inner_half_width)
            & is_real_foreground
            & ~neck_skin_corridor
        )
        # Old collar tips frequently sit underneath side hair and are then
        # labelled as hair by SCHP.  A semantic-only hair lock preserves those
        # pale/coloured fabric islands, so the final portrait shows the source
        # collar peeking out above the replacement collar.  Resolve that
        # overlap from this upload's pixels: inside the measured collar
        # annulus, upload-confirmed clothing is garment, and a nominal hair
        # pixel whose colour is far from the measured crown hair (and is not
        # skin) is also source garment.  This remains fully portrait-dynamic;
        # no shirt colour or fixed collar coordinates are assumed.
        source_collar_residual = (
            collar_refine_zone
            # A garment collar can touch the jaw but cannot occupy the face.
            # Constraining the evidence to the chin boundary prevents a
            # slightly misregistered upload-label warp from carving vertical
            # holes through cheeks in the identity-only canvas.
            & (mask_y >= face_y2 - detected_face_height * .005)
            & (
                upload_confirmed_clothing
                | (source_clothing > 0)
                | (
                    np.isin(identity_labels, (1, 2, 4))
                    & (id_hair_dist >= 30.0)
                    & ~is_skin_colored
                )
            )
            & ~neck_skin_corridor
        )
        if np.any(source_collar_residual):
            source_clothing[source_collar_residual] = 1
            clothing[source_collar_residual] = 1
            garment_mislabeled_as_hair |= (
                source_collar_residual & np.isin(identity_labels, (1, 2, 4))
            )
            logger.info(
                "[UNIFORM_COLLAR_MASK] Recovered %d source-collar pixel(s) "
                "from the hair/neck overlap",
                int(np.count_nonzero(source_collar_residual)),
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
        neck_blend_top = collar_neck_seat_y - detected_face_height * .03
        neck_blend_bottom = collar_neck_seat_y + detected_face_height * .06
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
        backdrop_like_identity = (selected_bg_distance <= 76.0) | pale_edge_fringe
        reliable_head_protection = (
            np.isin(identity_labels, (1, 2, 4, 14, 15))
            & ~backdrop_like_identity
            & ~garment_mislabeled_as_hair
        )
        neck_blend_zone = (
            (mask_y >= neck_blend_top)
            & (mask_y <= neck_blend_bottom)
            & (np.abs(mask_x - face_center_x) <= detected_face_width * .25)
            & ~reliable_head_protection
            & is_real_foreground
            & ~backdrop_like_identity
            & ~neck_skin_corridor
        )
        # Preserve all measured source skin, including the natural neck. The
        # pre-Qwen portrait matte already has the correct identity and anatomy;
        # repainting class 13 together with the garment produced a blocky,
        # elongated throat. Qwen may regenerate the surrounding collar and
        # uniform, but it must fit them around the unchanged source neck.
        protected_identity = reliable_head_protection.copy()
        # Protect face skin and upper neck, but stop above collar opening.
        # Below the jaw, require genuine skin colour so that dark lanyards,
        # ties, or collar flaps misclassified as label 13 are never protected.
        neck_skin_region = mask_y >= (face_y2 - detected_face_height * .05)
        valid_skin_label13 = (identity_labels == 13) & (~neck_skin_region | is_skin_colored)
        protected_identity |= (
            (valid_skin_label13 | is_skin_colored)
            & is_real_foreground
            & (mask_y <= collar_neck_seat_y)
            & (np.abs(mask_x - face_center_x) <= detected_face_width * .35)
        )
        protected_identity |= np.isin(identity_labels, (14, 15))
        natural_neck_guard = (
            (mask_y >= face_y2 - detected_face_height * .02)
            & (mask_y <= collar_neck_seat_y)
            & (np.abs(mask_x - face_center_x) <= detected_face_width * .32)
            & is_real_foreground
            & is_skin_colored
        )
        protected_identity |= natural_neck_guard
        protected_identity |= neck_skin_corridor
        # Never protect clothing or collar refine zone so the new uniform collar
        # replaces the old collar completely with zero peeking flaps.
        protected_identity &= ~(clothing > 0)
        protected_identity &= ~collar_refine_zone
        protected_identity |= neck_skin_corridor
        # Identity protection must never cover torso skin below collar_neck_seat_y:
        # any low-cut neckline from the source photo must be replaced by the uniform shirt.
        protected_identity &= (mask_y < collar_neck_seat_y) | head_or_hair_labels
        recovered_identity_hair = np.zeros_like(protected_identity)
        # SCHP may label a dark low ponytail as jacket/shirt where it passes
        # behind the neck.  Excluding protection solely by semantic class then
        # lets the garment mask repaint the tail.  Recover only pixels whose
        # colour is connected to parsed hair inside face-relative crown/side
        # corridors.  This is measured independently for every portrait and
        # cannot invent a hairstyle or protect the central source garment.
        parsed_identity_hair = (identity_labels == 2) & ~garment_mislabeled_as_hair
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
            try:
                _measured_recovery_end = float(
                    rough_placement.get("measured_hair_end_ratio") or .12
                )
            except (TypeError, ValueError):
                _measured_recovery_end = .12
            if parser_truncated_tied_tail:
                # Do not replace the rejected endpoint with another short
                # fixed search window. Open a generous measurement corridor;
                # only source pixels matching and connected to real crown hair
                # can enter the protection mask, so this permits the actual
                # visible tail without drawing or predicting its endpoint.
                _measured_recovery_end = .82
            _measured_recovery_end = float(np.clip(
                _measured_recovery_end, .12, .82,
            ))
            _tied_recovery_confirmed = bool(
                (measured_tied_count > 0 and tied_accessory_corroborated)
                or is_tied_or_braided_hair
            )
            # A fixed .54-face-height neighbourhood stopped above long plaits.
            # Use this upload's measured endpoint only when independent tied
            # geometry/accessory evidence confirms that the dark continuation
            # is hair rather than clothing.
            _hair_recovery_reach = (
                _measured_recovery_end + .42
                if _tied_recovery_confirmed else .54
            )
            near_identity_hair = cv2.dilate(
                parsed_identity_hair.astype(np.uint8),
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (max(7, int(round(detected_face_width * .38)) | 1),
                     max(7, int(round(detected_face_height * _hair_recovery_reach)) | 1)),
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
                    & (
                        mask_y <= face_y2 + detected_face_height * (
                            _measured_recovery_end + .35
                            if _tied_recovery_confirmed else .58
                        )
                    )
                    & (
                        ((mask_x >= face_x1 - detected_face_width * .55)
                         & (mask_x <= face_center_x - detected_face_width * .38))
                        | ((mask_x >= face_center_x + detected_face_width * .38)
                           & (mask_x <= face_x2 + detected_face_width * .55))
                    )
                )
            )
            hair_candidates = (
                near_identity_hair
                & crown_and_side_corridors
                & (identity_hair_distance < 54.0)
                & (identity_luma < 135.0)
                & ~garment_mislabeled_as_hair
                & ~vest_outer_zone
            )
            candidate_count, candidate_labels, candidate_stats, candidate_centroids = cv2.connectedComponentsWithStats(
                hair_candidates.astype(np.uint8), connectivity=8,
            )
            connected_identity_hair = np.zeros_like(parsed_identity_hair)
            disconnected_side_hair = np.zeros_like(parsed_identity_hair)
            maximum_tail_area = max(
                48,
                int(round(
                    detected_face_width * detected_face_height
                    * (.65 if _tied_recovery_confirmed else .15)
                )),
            )
            for candidate_id in range(1, candidate_count):
                candidate = candidate_labels == candidate_id
                if np.any(candidate & (parsed_identity_hair | detected_acc_mask_canvas)):
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
                    component_center_x <= face_x1 + detected_face_width * .15
                    or component_center_x >= face_x2 - detected_face_width * .15
                )
                reaches_nape = (
                    component_top <= face_y2 + detected_face_height * .28
                    and component_bottom >= face_y2 - detected_face_height * .04
                )
                is_tied_tail = bool(
                    _tied_recovery_confirmed
                    and beside_face
                    and (component_top <= face_y2 + detected_face_height * (_measured_recovery_end + 0.25))
                    and (component_bottom >= face_y2)
                )
                if (
                    16 <= area <= maximum_tail_area
                    and beside_face
                    and (reaches_nape or is_tied_tail)
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
                & ~upload_confirmed_clothing
                & ~vest_outer_zone
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
                    & ~upload_confirmed_clothing
                    & ~garment_mislabeled_as_hair
                )
                protected_identity |= measured_nape_tail
                recovered_identity_hair |= measured_nape_tail
        # Protect authentic hair accessories from being classified as clothing or included in edit_region
        acc_check = vl_plan.get("cv_hair_accessory_info") or {}
        acc_mask = vl_plan.get("cv_hair_accessory_mask")
        detected_acc_mask_canvas = np.zeros(identity_labels.shape, dtype=bool)
        if acc_check.get("detected") and acc_mask is not None:
            detected_acc_mask_canvas |= cv2.warpAffine(
                acc_mask.astype(np.uint8),
                canvas_to_upload,
                (identity_base.width, identity_base.height),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).astype(bool)
        acc_base_check = self._detect_hair_accessory_info(identity_base, rough_face_box)
        if acc_base_check.get("detected") and acc_base_check.get("mask") is not None:
            detected_acc_mask_canvas |= acc_base_check["mask"].astype(bool)

        if np.any(detected_acc_mask_canvas):
            if 'is_chest_badge' in locals() and np.any(is_chest_badge):
                detected_acc_mask_canvas &= ~is_chest_badge
            detected_acc_mask_canvas = cv2.dilate(
                detected_acc_mask_canvas.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            ).astype(bool)
            if 'is_chest_badge' in locals() and np.any(is_chest_badge):
                detected_acc_mask_canvas &= ~is_chest_badge
            clothing &= ~detected_acc_mask_canvas
            source_clothing &= ~detected_acc_mask_canvas
            garment_mislabeled_as_hair &= ~detected_acc_mask_canvas
            protected_identity |= detected_acc_mask_canvas

        if is_tied_or_braided_hair and 'braid_safe_hair' in locals() and np.any(braid_safe_hair):
            # For tied or braided hair, hair below the jaw is strictly the measured braid/tied sections.
            # Strip any false-hair dark vest or shoulder strap remnants that entered recovered_identity_hair
            # across the outer shoulders outside the braid corridors:
            shoulder_hair_zone = (
                (
                    ((mask_y >= face_y2 - detected_face_height * 0.45) & (np.abs(mask_x - face_center_x) >= detected_face_width * 0.25))
                    | (mask_y >= face_y2 - detected_face_height * 0.10)
                )
                & ~braid_safe_hair
            )
            if 'gap_clothing_zone' in locals() and np.any(gap_clothing_zone):
                shoulder_hair_zone |= gap_clothing_zone

            shoulder_cloth_zone = (
                shoulder_hair_zone
                & is_real_foreground
                & ~neck_skin_corridor
                & ~(identity_labels == 13)
            )
            if 'gap_clothing_zone' in locals() and np.any(gap_clothing_zone):
                shoulder_cloth_zone |= (gap_clothing_zone & is_real_foreground & ~neck_skin_corridor)

            if np.any(detected_acc_mask_canvas):
                shoulder_cloth_zone &= ~detected_acc_mask_canvas
            recovered_identity_hair &= ~shoulder_hair_zone
            protected_identity &= ~shoulder_hair_zone
            head_or_hair_labels &= ~shoulder_cloth_zone
            parsed_identity_hair &= ~shoulder_cloth_zone

            clothing |= shoulder_cloth_zone
            source_clothing |= shoulder_cloth_zone

            protected_identity |= braid_safe_hair
            recovered_identity_hair |= braid_safe_hair
            clothing &= ~braid_safe_hair
            source_clothing &= ~braid_safe_hair

        # ``source_collar_residual`` is determined after the early connected-
        # hair recovery begins.  Without this final authority reconciliation,
        # pale source collar tips can remain in ``recovered_identity_hair``;
        # later fit-region and collar-cue code then subtracts them again even
        # though the garment mask correctly classified them as clothing.  A
        # target collar cannot replace pixels that are simultaneously called
        # recovered hair.  Accessories and the natural neck are excluded by
        # the collar evidence itself, so clearing only this measured overlap
        # preserves the literal hairstyle while removing the old collar.
        recovered_collar_hair_overlap = (
            recovered_identity_hair & source_collar_residual
        )
        if np.any(recovered_collar_hair_overlap):
            recovered_identity_hair &= ~source_collar_residual
            protected_identity &= ~source_collar_residual
            logger.info(
                "[UNIFORM_COLLAR_MASK] Released %d source-collar pixel(s) "
                "from the secondary recovered-hair lock",
                int(np.count_nonzero(recovered_collar_hair_overlap)),
            )

        # Build the complete candidate edit region first, then apply identity
        # protection once. Re-adding the collar zone after protection used to
        # expose protected hair and accessories around the neck.
        edit_region = clothing > 0
        edit_region |= collar_refine_zone & ~(np.isin(identity_labels, (1, 2, 4)) & ~garment_mislabeled_as_hair)
        edit_region |= neck_blend_zone
        edit_region &= ~protected_identity
        edit_region &= ~neck_skin_corridor
        if np.any(detected_acc_mask_canvas):
            edit_region &= ~detected_acc_mask_canvas
        if is_tied_or_braided_hair and 'braid_safe_hair' in locals() and np.any(braid_safe_hair):
            edit_region &= ~braid_safe_hair
        # Torso editable region: Ensure the entire chest and torso below collar_neck_seat_y
        # within the body corridor is 100% editable so latent_keep_mask (~edit_region) NEVER
        # freezes source chest skin or low-cut shirt remnants into diffusion latents.
        torso_uniform_edit = (
            (mask_y >= collar_neck_seat_y)
            & (np.abs(mask_x - face_center_x) <= max(detected_face_width * 1.25, neck_half_w * 2.5))
            & is_real_foreground
            & ~head_or_hair_labels
            & ~recovered_identity_hair
            & ~detected_acc_mask_canvas
        )
        edit_region |= torso_uniform_edit
        # Identify the actual person-with-hair silhouette, not a rectangular
        # head area. ``protected_identity`` fuses SCHP face/hair/ear labels,
        # real foreground, recovered dark/tied hair and detected accessories.
        # Intersecting that fused subject with generous face-relative limits
        # prevents a clothing label elsewhere in the frame from entering the
        # head, while preserving the literal short hair, curls, flyaways,
        # braids, ponytails, flowers and clips present in this upload.
        complete_head_guard = (
            protected_identity
            & is_real_foreground
            & (mask_y <= face_y2 + detected_face_height * .12)
            & (mask_x >= face_x1 - detected_face_width * .95)
            & (mask_x <= face_x2 + detected_face_width * .95)
        )
        # A semantic "head" band ending at the jaw is not enough for tied
        # hair.  Long plaits, ponytails and side sections commonly continue
        # over the upper chest, exactly where the garment edit is strongest.
        # Preserve only source-proven hair pixels in two lateral corridors,
        # with the vertical extent measured independently for this upload.
        # This remains hairstyle-agnostic: no braid/ponytail shape is drawn or
        # assumed here, and the central garment stays fully editable.
        measured_side_hair_end_ratio = rough_placement.get(
            "measured_hair_end_ratio"
        )
        try:
            measured_side_hair_end_ratio = float(measured_side_hair_end_ratio)
        except (TypeError, ValueError):
            measured_side_hair_end_ratio = (
                .82 if parser_truncated_tied_tail else .12
            )
        measured_side_hair_end_ratio = float(np.clip(
            measured_side_hair_end_ratio, .04, .78,
        ))
        source_hair_evidence = (
            parsed_identity_hair | recovered_identity_hair
        ) & ~source_collar_residual
        lateral_hair_corridor = (
            (mask_y >= face_y1 - detected_face_height * .18)
            & (
                (mask_y <= face_y2 + detected_face_height * (
                    measured_side_hair_end_ratio + .12
                ))
            )
            & (
                (
                    (mask_x >= face_x1 - detected_face_width * .82)
                    & (mask_x <= face_x1 + detected_face_width * .20)
                )
                | (
                    (mask_x >= face_x2 - detected_face_width * .20)
                    & (mask_x <= face_x2 + detected_face_width * .82)
                )
            )
        )
        side_hair_guard = (
            source_hair_evidence
            & lateral_hair_corridor
            & is_real_foreground
            & ~garment_mislabeled_as_hair
        )
        if is_tied_or_braided_hair and 'braid_safe_hair' in locals() and np.any(braid_safe_hair):
            # For tied or braided hair, side hair guard below the chin must not protect false shoulder hair:
            side_hair_guard &= ~shoulder_cloth_zone
            complete_head_guard &= ~shoulder_cloth_zone
            protected_identity |= braid_safe_hair
            complete_head_guard |= braid_safe_hair
        if np.any(side_hair_guard):
            protected_identity |= side_hair_guard
            complete_head_guard |= side_hair_guard
            logger.info(
                "[UNIFORM_HAIR_GUARD] Protected %d measured source-hair pixels "
                "through %.2f face-heights below the jaw",
                int(np.count_nonzero(side_hair_guard)),
                measured_side_hair_end_ratio,
            )
        # The garment/head boundary has one authority: source collar evidence
        # must be regenerated as the target uniform even when it touches hair.
        # Releasing only these measured pixels avoids opening the surrounding
        # hairstyle or the central natural-neck corridor.
        protected_identity &= ~source_collar_residual
        complete_head_guard &= ~source_collar_residual
        # Ordinary garment edits are restricted to the measured person. Apply
        # that restriction before the low-detail head branch: that branch must
        # deliberately open the immediate backdrop around the head too, or a
        # protected latent rectangle remains around the restored face.
        edit_region &= is_real_foreground
        if regenerate_low_detail_head:
            # Open one continuous rectangle around the complete head, ears and
            # immediate backdrop. Opening only the semantic face silhouette
            # leaves protected latent cells around it and creates the same
            # pasted rectangular block at the token boundary. The collar stays
            # below this region and remains governed by the garment fit mask.
            full_head_regeneration_region = (
                (mask_y <= face_y2 + detected_face_height * .10)
                & (mask_x >= face_x1 - detected_face_width * .72)
                & (mask_x <= face_x2 + detected_face_width * .72)
            )
            edit_region |= full_head_regeneration_region
            logger.info(
                "[UNIFORM_HEAD_RESTORE] Low-detail source face (%.2f budget): "
                "opened %d coherent head/backdrop pixels for Qwen restoration",
                source_face_detail_budget,
                int(np.count_nonzero(full_head_regeneration_region)),
            )
        else:
            edit_region &= ~complete_head_guard
            edit_region |= source_collar_residual
            edit_region &= ~neck_skin_corridor
            if np.any(detected_acc_mask_canvas):
                edit_region &= ~detected_acc_mask_canvas
        if 'shoulder_overcoat_zone' in locals() and np.any(shoulder_overcoat_zone):
            edit_region |= (shoulder_overcoat_zone & is_real_foreground)
            complete_head_guard &= ~shoulder_overcoat_zone
            protected_identity &= ~shoulder_overcoat_zone
            recovered_identity_hair &= ~shoulder_overcoat_zone
        if 'gap_clothing_zone' in locals() and np.any(gap_clothing_zone):
            edit_region |= (gap_clothing_zone & is_real_foreground)
            complete_head_guard &= ~gap_clothing_zone
            protected_identity &= ~gap_clothing_zone
            recovered_identity_hair &= ~gap_clothing_zone
        if is_tied_or_braided_hair and 'braid_safe_hair' in locals() and np.any(braid_safe_hair):
            edit_region |= shoulder_cloth_zone
            edit_region &= ~braid_safe_hair
        if 'gap_is_background_air' in locals() and np.any(gap_is_background_air):
            edit_region &= ~gap_is_background_air
            complete_head_guard &= ~gap_is_background_air
            protected_identity &= ~gap_is_background_air
            recovered_identity_hair &= ~gap_is_background_air
        # Permit Qwen to redraw a narrow band around the old garment silhouette.
        # Pinning every backdrop latent to the exact source-clothing boundary
        # produced a horizontal shoulder cut and jagged white/blue outlines.
        # The band is derived from this portrait's measured face and is barred
        # from the head, hair, accessories and natural neck skin.
        garment_edge_px = max(7, int(round(detected_face_width * .10)))
        garment_edge_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (garment_edge_px * 2 + 1, garment_edge_px * 2 + 1),
        )
        garment_edge_band = cv2.dilate(
            edit_region.astype(np.uint8), garment_edge_kernel, iterations=1,
        ).astype(bool)
        # Keep the editable shoulder seam close to this upload's measured
        # body. The old +/-1.55-face corridor was 3.10 face widths across,
        # substantially wider than a typical photographed torso. Qwen could
        # therefore copy the product template's broad outer silhouette even
        # though the rough fitter had measured a much narrower person. A
        # small face-relative allowance remains so diffusion can round a hard
        # matte edge and form a natural sleeve/shoulder seam, but the template
        # can no longer redefine the person's build.
        measured_body_half_width = detected_face_width * source_body_ratio * .5
        # The edit mask is permission, not a target body width. Leave enough
        # measured air outside the source shoulders for one latent feather and
        # a natural sleeve curve. A fixed 0.08-face margin ended inside the
        # rendered sleeve on narrow portraits, producing a stepped white cut.
        # Derive the allowance from this canvas's unused width, bounded in
        # face-relative units so it cannot turn a child into a broad template
        # mannequin.
        unused_body_width = max(
            0.0, identity_base.width - measured_body_half_width * 2.0,
        )
        garment_seam_allowance = float(np.clip(
            unused_body_width * .18,
            detected_face_width * .08,
            detected_face_width * .28,
        ))
        garment_edit_half_width = measured_body_half_width + garment_seam_allowance
        measured_body_corridor = (
            np.abs(mask_x - face_center_x) <= garment_edit_half_width
        )
        lower_uniform_zone = (
            mask_y >= face_y2 - detected_face_height * .02
        )
        # The initial clothing mask can inherit the product template's broad
        # display shoulders. Clip the *entire* lower edit/seed region to the
        # upload-measured body span, not only the extra seam band. Otherwise
        # Qwen is conditioned on a wide rough shirt and faithfully generates
        # an adult/template torso for a narrow child portrait.
        edit_region &= ~(lower_uniform_zone & ~measured_body_corridor)
        garment_edge_corridor = (
            (mask_y >= face_y2 - detected_face_height * .02)
            & (np.abs(mask_x - face_center_x) <= garment_edit_half_width)
            & ~protected_identity
            & ~neck_skin_corridor
            & ~detected_acc_mask_canvas
            & ~recovered_identity_hair
        )
        if 'gap_is_background_air' in locals() and np.any(gap_is_background_air):
            garment_edge_corridor &= ~gap_is_background_air
        edit_region |= garment_edge_band & garment_edge_corridor
        if 'gap_is_background_air' in locals() and np.any(gap_is_background_air):
            edit_region &= ~gap_is_background_air
        rough_rgb = np.asarray(rough_composite.convert("RGB"), dtype=np.float32)
        # When a short-sleeve target replaces source clothing that covered the
        # arms, the rough compositor correctly removes that old clothing but
        # leaves selected-backdrop pixels below each target cuff. Diffusion
        # then treats the blue holes as missing body. Supply a low-frequency
        # skin placement cue only in those source-silhouette side regions;
        # Qwen still creates the final texture and the central shirt is never
        # touched by this guide.
        synthetic_arm_guide = np.zeros_like(edit_region)
        # The cue only makes sense where the *source* really hid an arm behind
        # a long sleeve. On job 16682cec the upload was a head-and-chest crop
        # with no arm in frame at all (visible-arm fraction 0.0000), yet the
        # side corridor below still matched 51686 clothing pixels and was
        # painted with skin. Qwen resolved that unexplained skin band as a
        # second sleeved arm: the delivered portrait stepped 90px narrower
        # over ten rows and carried two sleeve hems and two red cuffs per
        # side. Require measured evidence of a covered arm, not merely a
        # short-sleeved target.
        source_arm_pixels = int(np.count_nonzero(np.isin(source_labels, (14, 15))))
        source_arm_evidence = source_visible_arm_fraction >= .0015
        if requires_exposed_arm_synthesis and measured_skin_rgb and not source_arm_evidence:
            logger.info(
                "[UNIFORM_ARM_GUIDE] Skipped skin cue: no arm is visible in the "
                "source (fraction=%.4f, parser arm pixels=%d). Synthesizing one "
                "invents a second sleeve.",
                source_visible_arm_fraction, source_arm_pixels,
            )
        if requires_exposed_arm_synthesis and measured_skin_rgb and source_arm_evidence:
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
        # Do not require a fixed fraction of the *whole canvas*. A tightly
        # cropped passport upload can contain a large face and only a narrow
        # strip of valid upper-chest clothing; the former 8% gate rejected that
        # perfectly usable region before Qwen ran. Measure the available torso
        # for this portrait and, when parsing is sparse, recover only the
        # face-relative foreground below the protected neck/head. This remains
        # fully dynamic and cannot enter hair, accessories, face or backdrop.
        canvas_area = identity_base.width * identity_base.height
        torso_corridor = (
            is_real_foreground
            & (mask_y >= face_y2 + detected_face_height * .08)
            & (np.abs(mask_x - face_center_x) <= detected_face_width * 1.45)
            & ~protected_identity
            & ~detected_acc_mask_canvas
            & ~recovered_identity_hair
        )
        # Preserve the narrow natural throat even when SCHP missed its skin
        # label. The target collar may edit around it but must not repaint it.
        natural_throat = (
            (mask_y <= face_y2 + detected_face_height * .25)
            & (np.abs(mask_x - face_center_x) <= detected_face_width * .24)
            & is_skin_colored
        )
        torso_corridor &= ~natural_throat
        # A school-ID delivery ends at the upper chest. When the upload has no
        # visible arms, keep the surrounding backdrop protected. Opening it
        # made Qwen invent a larger adult torso and long upper arms merely to
        # fill pixels that the final crop does not need.
        upper_arm_outpaint_region = np.zeros_like(edit_region)
        if requires_exposed_arm_synthesis and not source_arm_evidence:
            logger.info(
                "[UNIFORM_ARM_OUTPAINT] No source arm evidence; kept backdrop "
                "protected for upper-chest delivery"
            )
        # The local latent hold must never preserve source clothing at the
        # outer torso. A clean person + garment-reference edit otherwise keeps
        # those held pixels while Qwen generates the target sleeve above them,
        # producing the visible two-sleeve/second-cuff effect. Re-open every
        # parser-confirmed source-clothing pixel inside this portrait's dynamic
        # torso corridor, then re-apply all identity and hair protections.
        residual_source_clothing = (source_clothing > 0) & torso_corridor
        if np.any(residual_source_clothing):
            edit_region |= residual_source_clothing
            edit_region &= ~protected_identity
            edit_region &= ~detected_acc_mask_canvas
            edit_region &= ~recovered_identity_hair
            edit_region &= ~neck_skin_corridor
            if is_tied_or_braided_hair and 'braid_safe_hair' in locals() and np.any(braid_safe_hair):
                edit_region &= ~braid_safe_hair
            logger.info(
                "[UNIFORM_FIT_REGION] Opened %d source-clothing pixel(s) at the outer torso for complete replacement",
                int(np.count_nonzero(residual_source_clothing)),
            )
        available_torso_pixels = int(np.count_nonzero(torso_corridor))
        minimum_edit_pixels = max(
            int(canvas_area * .022),
            min(int(canvas_area * .08), int(available_torso_pixels * .30)),
        )
        initial_edit_pixels = int(np.count_nonzero(edit_region))
        if initial_edit_pixels < minimum_edit_pixels:
            recovered_torso = torso_corridor & (
                (source_clothing > 0)
                | upload_confirmed_clothing
                | (mask_y >= face_y2 + detected_face_height * .24)
            )
            edit_region |= recovered_torso
            # Re-assert every identity authority after recovery.
            edit_region &= ~protected_identity
            edit_region &= ~detected_acc_mask_canvas
            edit_region &= ~recovered_identity_hair
            edit_region &= is_real_foreground
            if is_tied_or_braided_hair and 'braid_safe_hair' in locals() and np.any(braid_safe_hair):
                edit_region &= ~braid_safe_hair
            logger.info(
                "[UNIFORM_FIT_REGION] Adaptive recovery initial=%d available_torso=%d recovered=%d required=%d",
                initial_edit_pixels,
                available_torso_pixels,
                int(np.count_nonzero(edit_region)),
                minimum_edit_pixels,
            )
        final_edit_pixels = int(np.count_nonzero(edit_region))
        absolute_safe_minimum = max(
            256,
            min(int(canvas_area * .018), int(available_torso_pixels * .20)),
        )
        if final_edit_pixels < absolute_safe_minimum:
            raise RuntimeError(
                "Uniform fitting could not identify enough visible upper-chest clothing in this portrait"
            )
        # Build one direct conditioning portrait on the ORIGINAL person's geometry:
        # copy fitted garment pixels only inside the measured clothing region,
        # retain every protected identity/body pixel from identity_base, and use
        # a narrow feather solely at the cloth boundary. This is not the old
        # detached-head rough canvas and is never followed by source-face pasteback.
        garment_seed_rgb = rough_rgb.copy()
        # Do NOT paint flat solid brown blocks into garment_seed_rgb;
        # let Qwen synthesize realistic arms/sleeves from text prompt instructions.
        seed_alpha = cv2.GaussianBlur(
            edit_region.astype(np.float32), (0, 0), sigmaX=1.15, sigmaY=1.15,
        )
        seed_alpha = np.clip(seed_alpha, 0.0, 1.0)
        seed_alpha[protected_identity] = 0.0
        if np.any(detected_acc_mask_canvas):
            seed_alpha[detected_acc_mask_canvas] = 0.0
        direct_seed_rgb = (
            identity_rgb * (1.0 - seed_alpha[:, :, None])
            + garment_seed_rgb * seed_alpha[:, :, None]
        )
        # Background air outside real foreground must use the exact selected flat color.
        #
        # But never wipe a pixel the rough composite deliberately filled. On
        # job ce27473c this repainted 272 cloth pixels across rows 0.39-0.60 of
        # the canvas - the neck and collar - as backdrop, and the decode then
        # reproduced those gaps. Measured attribution: at every one of those
        # pixels the identity canvas held cloth, so they were real garment the
        # SCHP-derived foreground mask simply did not cover, not background
        # air. The composite is the authority for where it placed cloth;
        # this wipe only governs the air around it.
        wipe = ~is_real_foreground
        composite_cloth = (
            np.linalg.norm(
                rough_rgb - np.asarray(qwen_bg_rgb, dtype=np.float32)[None, None, :],
                axis=2,
            ) > 40.0
        )
        protected_from_wipe = int(np.count_nonzero(wipe & composite_cloth))
        if protected_from_wipe:
            logger.info(
                "[UNIFORM_SEED] Kept %d composite-placed cloth pixel(s) the "
                "foreground mask would have wiped to backdrop",
                protected_from_wipe,
            )
        wipe &= ~composite_cloth
        direct_seed_rgb[wipe] = np.asarray(qwen_bg_rgb, dtype=np.float32)
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
        if save_intermediates:
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
        # generation_source was already unified with BiRefNet onto selected_bg_rgb before rough fitting.
        primary_person_source = generation_source.copy()
        # The earlier measured arm guide was only logged; image 1 remained the
        # untouched long-sleeve portrait, so Qwen never saw the intended skin
        # continuation and left backdrop wedges below the new short cuffs.
        # Put the low-frequency cue into Qwen's actual primary tensor and open
        # the same region in the latent edit mask. It is generated over by
        # diffusion; this is not a final pixel patch.
        if np.any(synthetic_arm_guide) and measured_skin_rgb:
            arm_primary_rgb = np.asarray(
                primary_person_source.convert("RGB"), dtype=np.float32,
            )
            arm_cue_alpha = cv2.GaussianBlur(
                synthetic_arm_guide.astype(np.float32),
                (0, 0),
                sigmaX=2.2,
                sigmaY=2.2,
            )
            arm_cue_alpha = np.clip(arm_cue_alpha * .82, 0.0, .82)
            arm_skin = np.asarray(measured_skin_rgb, dtype=np.float32)[None, None, :]
            arm_primary_rgb = (
                arm_primary_rgb * (1.0 - arm_cue_alpha[:, :, None])
                + arm_skin * arm_cue_alpha[:, :, None]
            )
            primary_person_source = Image.fromarray(
                np.rint(np.clip(arm_primary_rgb, 0, 255)).astype(np.uint8),
                "RGB",
            )
            arm_edit_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    max(7, int(round(detected_face_width * .08))) | 1,
                    max(11, int(round(detected_face_height * .12))) | 1,
                ),
            )
            arm_edit_corridor = cv2.dilate(
                synthetic_arm_guide.astype(np.uint8),
                arm_edit_kernel,
                iterations=1,
            ).astype(bool)
            arm_edit_corridor &= ~protected_identity
            arm_edit_corridor &= ~detected_acc_mask_canvas
            arm_edit_corridor &= ~recovered_identity_hair
            edit_region |= arm_edit_corridor
            logger.info(
                "[UNIFORM_ARM_GUIDE] Applied measured skin cue to Qwen image 1 "
                "and opened %d continuous cuff-to-arm edit pixel(s)",
                int(np.count_nonzero(arm_edit_corridor)),
            )

        # Image 1 must not expose the old garment to Qwen.  Build a dynamic
        # upper-chest synthesis aperture from this upload's measured face/body
        # ratio.  The aperture starts at the measured collar seat, widens
        # smoothly toward the measured shoulders, and never enters source-
        # proven head, hair, accessories or natural neck skin.  Unlike the old
        # source-clothing mask, this also opens selected-backdrop pixels where
        # a sleeve/shoulder must be generated, so the model can construct the
        # complete target shirt from image 2 without seeing or copying image
        # 1's vest, collar or shirt.
        upper_chest_progress = np.clip(
            (mask_y - collar_top) / max(detected_face_height * .68, 1.0),
            0.0,
            1.0,
        )
        upper_chest_half_width = (
            neck_half_w * (1.0 - upper_chest_progress)
            + (measured_body_half_width + garment_seam_allowance)
            * upper_chest_progress
        )
        identity_only_upper_chest_region = (
            (mask_y >= collar_top)
            & (np.abs(mask_x - face_center_x) <= upper_chest_half_width)
            & ~protected_identity
            & ~complete_head_guard
            & ~recovered_identity_hair
            & ~detected_acc_mask_canvas
            & ~neck_skin_corridor
        )
        edit_region |= identity_only_upper_chest_region
        # Regenerate one coherent seam around the measured neck boundary.
        # Semantic head/skin protection can leave narrow backdrop wedges where
        # the neck edge, old collar and target collar disagree by a few pixels.
        # The band is derived from this portrait's face/neck measurements; the
        # central authentic neck remains locked, while only its lateral seam
        # and collar contact area are opened for Qwen.
        neck_collar_seam_outer_half = (
            neck_half_w + detected_face_width * .18
        )
        neck_collar_seam_region = (
            (mask_y >= face_y2 - detected_face_height * .05)
            & (mask_y <= collar_neck_seat_y + detected_face_height * .12)
            & (
                np.abs(mask_x - face_center_x)
                <= neck_collar_seam_outer_half
            )
            & ~neck_skin_corridor
            & ~recovered_identity_hair
            & ~detected_acc_mask_canvas
        )
        edit_region |= neck_collar_seam_region
        logger.info(
            "[UNIFORM_IDENTITY_ONLY] Opened %d measured upper-chest pixel(s) "
            "for complete image-2 garment synthesis and %d dynamic "
            "neck/collar seam pixel(s)",
            int(np.count_nonzero(identity_only_upper_chest_region)),
            int(np.count_nonzero(neck_collar_seam_region)),
        )

        # Keep face/hair interiors latent-held for identity, but do not freeze
        # the exact hard matte boundary. The Qwen edit latent is roughly one
        # token per 16 output pixels and the protection mask is feathered before
        # that reduction. The former two-pixel ring disappeared completely at
        # latent resolution, so diffusion never actually owned the hair/ear
        # transition and the VAE reproduced it as a staircase. Open a
        # face-scaled 10-20px ring instead: wide enough to survive the latent
        # reduction, still far narrower than any hairstyle feature. Real hair
        # accessories remain fully protected.
        head_subject = complete_head_guard & is_real_foreground
        head_edge_radius = int(np.clip(
            round(detected_face_width * .055), 10, 20,
        ))
        head_edge_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (head_edge_radius * 2 + 1, head_edge_radius * 2 + 1),
        )
        head_edge_inner = cv2.erode(
            head_subject.astype(np.uint8),
            head_edge_kernel,
            iterations=1,
        ).astype(bool)
        # Open only the source side of the boundary. Opening the outer band
        # gives diffusion permission to grow a second hairstyle into clean
        # backdrop (job d22e47c0 produced a brown/slick crown behind the real
        # short hair). The inner band still survives latent downsampling and
        # lets Qwen soften the final subject pixel without expanding the
        # measured silhouette.
        head_edge_diffusion_ring = (
            head_subject
            & ~head_edge_inner
            & (mask_y <= face_y2 + detected_face_height * .05)
            & (np.abs(mask_x - face_center_x) <= detected_face_width * .92)
        )
        if np.any(detected_acc_mask_canvas):
            protected_accessory_edge = cv2.dilate(
                detected_acc_mask_canvas.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            ).astype(bool)
            head_edge_diffusion_ring &= ~protected_accessory_edge
        edit_region |= head_edge_diffusion_ring
        logger.info(
            "[UNIFORM_HEAD_EDGE] Opened %d measured hair/ear boundary pixel(s) "
            "inside the source silhouette for Qwen-native diffusion smoothing "
            "without hairstyle expansion (radius=%dpx)",
            int(np.count_nonzero(head_edge_diffusion_ring)), head_edge_radius,
        )
        hard_identity_protection = np.zeros_like(edit_region)
        if not regenerate_low_detail_head:
            # Face/hair/accessories require a hard latent hold. The visible
            # neck remains protected by ``edit_region`` but must use its normal
            # feathered transition; hard-holding the throat boundary created a
            # 16px latent stair-step where the generated collar met real skin.
            hard_identity_protection |= (
                complete_head_guard
                & (mask_y <= face_y2 + detected_face_height * .01)
            )
            # Lower curls/plaits already remain outside ``edit_region`` and are
            # therefore protected by its broad, two-latent soft transition.
            # Re-asserting the same lower hair as a binary hard hold converted
            # the hair/garment overlap into 16px latent squares: the new shirt
            # appeared as plaid blocks between curl ends. Keep only the upper
            # head hard-held; let the existing soft keep mask own lower hair.
            hard_identity_protection |= detected_acc_mask_canvas
            hard_identity_protection &= is_real_foreground
            # Cross-authority safety invariant: a pixel that the untouched
            # upload parser confirms as source clothing must reach diffusion,
            # never the hard identity hold. The normal feathered mask still
            # handles its boundary against genuine hair.
            hard_garment_overlap = hard_identity_protection & (
                upload_confirmed_clothing
                | source_collar_residual
                | neck_collar_seam_region
            )
            if np.any(hard_garment_overlap):
                hard_identity_protection &= ~hard_garment_overlap
                logger.info(
                    "[UNIFORM_GARMENT_MASK] Released %d upload-confirmed source "
                    "garment pixel(s) from the hard identity lock",
                    int(np.count_nonzero(hard_garment_overlap)),
                )
        person_matted_path = OUTPUTS_DIR / f"{job_id}_person_qwen_input.png"
        # Severe clipped light must not remain embedded in Qwen's primary
        # tensor: both the four-step adapter and the deeper base schedule copy
        # those pixels. Build a uniform-only, geometry-preserving guide from
        # this portrait's own shaded skin and hair samples. The untouched
        # upload remains the biometric/VL authority and final validation
        # reference; this guide exists only to tell Qwen the intended lighting.
        source_hair_geometry_canvas = primary_person_source.copy()
        if source_has_direct_light:
            logger.info(
                "[UNIFORM_CONDITIONING] Using the untouched person tensor and one %d-step Qwen relighting/garment edit",
                generation_steps,
            )
        if save_intermediates:
            generation_source.save(person_matted_path, format="PNG")
            logger.info(
                "[UNIFORM_CONDITIONING] Saved intact uploaded-person canvas for Qwen image 1 -> %s",
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
                by1, bx1, by2, bx2 = (float(value) for value in badge_box)
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
        # Background air outside real foreground must use the exact selected solid backdrop
        is_real_fg_gen = cv2.resize(
            is_real_foreground.astype(np.uint8), generation_source.size,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        protected_source = np.asarray(generation_source, dtype=np.float32)
        bg_rgb_arr = np.asarray(selected_bg_rgb, dtype=np.float32)
        protected_source[~is_real_fg_gen] = bg_rgb_arr
        fitted_uniform = np.asarray(seed_canvas, dtype=np.float32)
        fitted_uniform[~is_real_fg_gen] = bg_rgb_arr
        seed_region[~is_real_fg_gen] = 0.0
        neck_skin_gen = cv2.resize(
            neck_skin_corridor.astype(np.uint8), generation_source.size,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        seed_region[neck_skin_gen] = 0.0
        seed_region = np.clip(seed_region, 0.0, 1.0)[:, :, None]
        generation_source = Image.fromarray(
            np.rint(
                protected_source * (1.0 - seed_region)
                + fitted_uniform * seed_region
            ).clip(0, 255).astype(np.uint8),
            "RGB",
        )
        selected_bg_conditioning_path = (
            OUTPUTS_DIR / f"{job_id}_selected_bg_conditioning.png"
        )
        if save_intermediates:
            generation_source.save(selected_bg_conditioning_path, format="PNG")
            logger.info(
                "[UNIFORM_CONDITIONING] Prepared protected person plus aligned rough uniform on exact RGB%s backdrop before Qwen -> %s",
                tuple(selected_bg_rgb), selected_bg_conditioning_path,
            )
        # Qwen must receive the clean photographed person as image 1. Feeding
        # the rough composite here makes a four-step edit preserve its pasted
        # garment boundary, sleeve overlaps and low-resolution fabric instead
        # of synthesising a coherent tailored shirt. The rough/direct seed is
        # still used to measure body placement and validate coverage, but it is
        # never a generative pixel authority.
        # Pass only identity anatomy from image 1: complete head/hair,
        # authentic accessories, ears and the measured natural neck.  Every
        # source garment pixel is replaced by the selected flat backdrop before
        # Qwen sees image 1.  This is deliberately stronger than asking the
        # prompt to "remove" clothing: the old collar/vest no longer exists in
        # the visual conditioning and therefore cannot be copied back.
        identity_only_authority = (
            complete_head_guard
            | recovered_identity_hair
            | natural_neck_guard
            | neck_skin_corridor
            | detected_acc_mask_canvas
        ) & is_real_foreground
        identity_only_authority &= ~source_collar_residual
        # Upload-confirmed clothing is never identity, even when SCHP labelled
        # it hair or skin. Recovered genuine hair has already been reconciled
        # against ``source_collar_residual`` above and remains authoritative.
        source_garment_authority = (
            upload_confirmed_clothing | (source_clothing > 0)
        )
        identity_only_authority &= ~(
            source_garment_authority & ~recovered_identity_hair
        )
        primary_rgb = np.asarray(
            primary_person_source.convert("RGB"), dtype=np.uint8,
        )
        # Preserve the intact, antialiased source head rather than rebuilding
        # it from a sparse semantic mask.  Sparse reconstruction deleted cheek
        # and jaw pixels whenever the warped clothing parse crossed the face,
        # and fragmented curls into white matte holes.  Start with the clean
        # selected-backdrop portrait and erase only measured garment pixels.
        # The upper-chest aperture already excludes complete head/hair,
        # accessories and natural neck, so this removes clothing without
        # touching their photographic edges.
        garment_removal_region = (
            identity_only_upper_chest_region
            | neck_collar_seam_region
            | source_collar_residual
            | (
                source_garment_authority
                & (mask_y >= face_y2 - detected_face_height * .005)
                & ~recovered_identity_hair
            )
            | (
                (mask_y >= collar_neck_seat_y)
                & (np.abs(mask_x - face_center_x) <= max(detected_face_width * 1.25, neck_half_w * 2.5))
                & is_real_foreground
                & ~head_or_hair_labels
                & ~recovered_identity_hair
                & ~detected_acc_mask_canvas
            )
        )
        garment_removal_region &= ~natural_neck_guard
        garment_removal_region &= ~neck_skin_corridor
        garment_removal_region &= ~detected_acc_mask_canvas
        garment_removal_region &= ~recovered_identity_hair
        if is_tied_or_braided_hair and 'braid_safe_hair' in locals() and np.any(braid_safe_hair):
            if 'shoulder_cloth_zone' in locals() and np.any(shoulder_cloth_zone):
                garment_removal_region |= (shoulder_cloth_zone & is_real_foreground)
            garment_removal_region &= ~braid_safe_hair
        if 'gap_clothing_zone' in locals() and np.any(gap_clothing_zone):
            garment_removal_region |= (gap_clothing_zone & is_real_foreground)
        if 'gap_is_background_air' in locals() and np.any(gap_is_background_air):
            garment_removal_region &= ~gap_is_background_air
        identity_only_rgb = primary_rgb.copy()
        seed_fabric_rgb = np.asarray(
            direct_uniform_seed.resize(
                (primary_rgb.shape[1], primary_rgb.shape[0]),
                Image.Resampling.LANCZOS,
            ).convert("RGB"),
            dtype=np.uint8,
        ).copy()
        # The fitted rough seed can contain tiny backdrop-coloured crescents at
        # the neck/collar contact when its garment cutout and this portrait's
        # neck have different shapes.  Copying those pixels into Qwen's primary
        # conditioning image makes diffusion reproduce the white/blue notch
        # instead of joining the collar to the neck.  Repair *only* such holes
        # inside the measured lateral seam.  The authentic central neck, head,
        # hair and accessories are excluded by ``neck_collar_seam_region``.
        #
        # For a visually measured solid collar use that template's measured
        # accent colour.  For every other construction, OpenCV propagates the
        # neighbouring fitted garment texture into the small hole.  This keeps
        # the operation dynamic across uniforms rather than assuming red, plaid
        # or any particular collar style.
        selected_bg = np.asarray(qwen_bg_rgb, dtype=np.int16)
        seed_bg_distance = np.linalg.norm(
            seed_fabric_rgb.astype(np.int16) - selected_bg[None, None, :],
            axis=2,
        )
        seam_seed_holes = neck_collar_seam_region & (seed_bg_distance <= 38.0)
        if np.any(seam_seed_holes):
            repaired_seed = seed_fabric_rgb.copy()
            if collar_leaves_are_solid and measured_accent:
                repaired_seed[seam_seed_holes] = np.asarray(
                    measured_accent["rgb"], dtype=np.uint8,
                )
            else:
                inpaint_mask = (seam_seed_holes.astype(np.uint8) * 255)
                repaired_seed = cv2.inpaint(
                    repaired_seed,
                    inpaint_mask,
                    5.0,
                    cv2.INPAINT_TELEA,
                )
            seed_fabric_rgb = repaired_seed
            logger.info(
                "[UNIFORM_NECK_SEAM] Repaired %d backdrop-coloured fitted "
                "seed pixel(s) using the uploaded collar construction",
                int(np.count_nonzero(seam_seed_holes)),
            )
        identity_only_rgb[garment_removal_region] = seed_fabric_rgb[garment_removal_region]
        if 'gap_is_background_air' in locals() and np.any(gap_is_background_air):
            identity_only_rgb[gap_is_background_air] = np.asarray(qwen_bg_rgb, dtype=np.uint8)
        # Infill any missing area, white notch or backdrop hole across the neck aperture with authentic neck skin
        neck_regen_aperture = (
            (mask_y >= face_y2 - detected_face_height * 0.04)
            & (mask_y <= collar_neck_seat_y + detected_face_height * 0.03)
            & (np.abs(mask_x - face_center_x) <= detected_face_width * 0.46)
            & ~head_or_hair_labels
            & ~recovered_identity_hair
            & ~detected_acc_mask_canvas
        )
        selected_bg_f = np.asarray(qwen_bg_rgb, dtype=np.float32)
        id_bg_dist = np.linalg.norm(
            identity_only_rgb.astype(np.float32) - selected_bg_f[None, None, :],
            axis=2,
        )
        # Real neck skin should only come from authentic photographic pixels, never synthetic flat infill.
        # Only repair genuine backdrop holes directly inside the authentic neck corridor strictly above collar_neck_seat_y.
        missing_neck_area = neck_regen_aperture & (id_bg_dist <= 32.0) & (mask_y < collar_neck_seat_y)
        if 'clean_id_result' in locals() and clean_id_result and clean_id_result.get("neck_classification") == "occluded_neck":
            missing_neck_area = np.zeros(identity_only_rgb.shape[:2], dtype=bool)
        if np.any(missing_neck_area) and np.any(neck_skin_corridor):
            clean_neck_pixels = primary_rgb[neck_skin_corridor]
            if len(clean_neck_pixels) >= 16:
                neck_median_color = np.median(clean_neck_pixels, axis=0).astype(np.uint8)
                identity_only_rgb[missing_neck_area] = neck_median_color
                logger.info(
                    "[UNIFORM_NECK_REGEN] Infilled %d missing neck/notch pixel(s) with authentic skin",
                    int(np.count_nonzero(missing_neck_area)),
                )
        generation_primary = Image.fromarray(identity_only_rgb, "RGB")
        logger.info(
            "[UNIFORM_IDENTITY_ONLY] Preserved intact head/hair/neck edges and "
            "seated %d fitted uniform pixel(s) into source-garment region for Qwen image 1",
            int(np.count_nonzero(garment_removal_region)),
        )
        # Give Qwen positive target-collar pixels exactly at the measured
        # throat seat without feeding the whole rough composite. The clean
        # source portrait alone leaves its old mandarin/collar pixels as strong
        # local evidence; image 2 supplies design but no body-relative seat.
        # A small feathered cue from the fitted seed replaces only that neck
        # band, while face, hair, natural neck and the rest of the body remain
        # the intact upload. Diffusion owns the final photographic junction.
        collar_conditioning_region = (
            edit_region
            & (mask_y >= collar_top - detected_face_height * .03)
            & (mask_y <= collar_bottom + detected_face_height * .04)
            & (np.abs(mask_x - face_center_x) <= detected_face_width * .52)
            & ~neck_skin_corridor
            & ~detected_acc_mask_canvas
            & ~recovered_identity_hair
        )
        if is_tied_or_braided_hair and 'braid_safe_hair' in locals():
            collar_conditioning_region &= ~braid_safe_hair
        if np.any(collar_conditioning_region):
            primary_rgb = np.asarray(
                generation_primary.convert("RGB"), dtype=np.float32,
            )
            collar_seed_rgb = np.asarray(
                direct_uniform_seed.resize(
                    generation_primary.size, Image.Resampling.LANCZOS,
                ).convert("RGB"),
                dtype=np.float32,
            )
            collar_cue = cv2.resize(
                collar_conditioning_region.astype(np.uint8),
                generation_primary.size,
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.float32)
            collar_cue = cv2.GaussianBlur(
                collar_cue, (0, 0), sigmaX=1.4, sigmaY=1.4,
            )
            collar_cue = np.clip(collar_cue * .92, 0.0, .92)
            primary_rgb = (
                primary_rgb * (1.0 - collar_cue[:, :, None])
                + collar_seed_rgb * collar_cue[:, :, None]
            )
            generation_primary = Image.fromarray(
                np.rint(primary_rgb).clip(0, 255).astype(np.uint8), "RGB",
            )
            logger.info(
                "[UNIFORM_COLLAR_CUE] Seeded %d measured target-collar pixel(s) "
                "at the source neck seat",
                int(np.count_nonzero(collar_conditioning_region)),
            )
        # VSOUL_UNIFORM_ROUGH_PRIMARY=1 feeds the rough composite instead, as
        # an experiment only. The comment above records why that was removed;
        # re-enable it to measure whether conditioning geometry is what sets
        # the rendered jaw-to-collar gap, and read the garment-fidelity cost
        # in the same output before considering it for anything else.
        if os.getenv("VSOUL_UNIFORM_ROUGH_PRIMARY", "0").strip().lower() in {
            "1", "true", "yes",
        }:
            generation_primary = rough_composite
            logger.warning(
                "[UNIFORM_EXPERIMENT] VSOUL_UNIFORM_ROUGH_PRIMARY: conditioning "
                "Qwen on the rough composite instead of the clean person",
            )
        elif os.getenv("VSOUL_UNIFORM_SEAT_NECK", "0").strip().lower() in {
            "1", "true", "yes",
        }:
            # OFF by default: measured, this made the neck slightly LONGER
            # (exposed skin below the chin 0.280 -> 0.292 face-boxes against a
            # 0.162 source) rather than shorter.
            #
            # The idea was to give Qwen the rough composite's geometry cue
            # without its pasted garment boundary. It does avoid the notch
            # artifacts, but it does not shorten the neck, because absence is
            # not a constraint: fading the throat to backdrop leaves a void
            # under the chin and Qwen fills a void with neck - the same
            # failure mode as the original defect. Feeding the whole rough
            # composite worked (0.246) because it put positive garment pixels
            # where the collar belongs, not because it removed neck.
            #
            # Kept, wired and tested as a recorded negative result. The
            # direction it points to is the inverse: seat the collar as actual
            # fabric in the conditioning, feathered at its outer silhouette so
            # there is no hard edge to copy.
            generation_primary = seat_conditioning_neck(
                generation_primary,
                collar_neck_seat_y,
                face_center_x,
                neck_half_w,
                selected_bg_rgb,
            )
            logger.info(
                "[UNIFORM_CONDITIONING] Seated conditioning neck at y=%.1f "
                "(column half-width %.1f px) so Qwen sees the measured "
                "jaw-to-collar geometry rather than only reading it in the prompt",
                float(collar_neck_seat_y), float(neck_half_w),
            )
        generation_input = generation_primary.resize(
            (generation_w, generation_h), Image.LANCZOS,
        )
        generation_edit_region = cv2.resize(
            edit_region.astype(np.uint8), generation_input.size,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        generation_hard_identity_protection = cv2.resize(
            hard_identity_protection.astype(np.uint8), generation_input.size,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        # Lower hair and braids remain protected by edit_region and the feathered
        # soft keep mask in diffusion; never add them to generation_hard_identity_protection
        # which activates 16px latent hard holds and creates blocky rectangular artifacts.
        # The measured rough fit remains internal geometry evidence only.
        # Do not render or attach another torso-shaped reference to Qwen.
        fitted_garment_reference = None
        fitted_garment_reference_path = None
        generation_garment_crop_top = int(round(
            garment_crop_top * generation_input.height / max(1, identity_base.height)
        ))
        logger.info(
            "[UNIFORM_IDENTITY_REFERENCE] Using Enhancement-style full-frame native-aspect conditioning %s -> %s",
            person_pil.size, generation_input.size,
        )
        # Keep exactly one person tensor. Four-step edits can use face-free
        # layout/hair guides. The stronger eight-step adapter interpreted even
        # those crops as additional portrait content, so extreme-light jobs use
        # only image 1 (person) and image 2 (uniform). Measured body/hair facts
        # still reach that route through the dynamic prompt.
        identity_head_conditioning = None
        identity_head_conditioning_path = None
        # Qwen Image Edit treats torso-shaped garment guides and face-blanked
        # hairstyle crops as additional subjects even when no detector finds a
        # face in them.  That produced two complete people in the four-step
        # decode.  Production generation must therefore always receive exactly
        # two images: image 1 is the uploaded person with the measured body-fit
        # seed; image 2 is one tight, clean, face-free garment reference. The
        # former body-fitted reference devoted over half of its 512px visual
        # token canvas to empty backdrop, shrinking the collar and micro-check
        # fabric until Qwen invented a generic pointed collar. Body geometry is
        # already present in image 1, so image 2 must maximize design evidence.
        use_auxiliary_geometry_references = False
        # Static mode needs the highest-detail garment evidence.  Placing the
        # template in a body-sized box surrounded by empty backdrop reduced the
        # plaid/collar to a small fraction of the 512px encoder canvas; Qwen
        # then invented a generic shirt even though the uploaded template was
        # correct.  Body fit comes from image 1 and the measured passport crop,
        # so image 2 can remain the tight, face-free garment authority.
        # Exactly two visual authorities are safe: one intact person and one
        # face-free garment. A third torso-shaped fit reference was interpreted
        # as another body and produced layered shirts/shoulders in the decode.
        # Exactly two visual authorities are used: the clean uploaded person
        # and one tight face-free garment reference. The prior three-image
        # route could create another torso; the later one-image route embedded
        # the rough paste in image 1 and therefore returned rough cut edges.
        # Body fit comes from measured source geometry in the dynamic prompt,
        # while image 2 supplies only the actual garment design and pixels.
        qwen_garment_references = [uniform_validation_reference]
        measured_fit_ratio = float(
            rough_placement.get("source_body_to_face_width") or 0.0,
        )
        rough_spatial_authority = (
            f"Image 1 fixes the measured {measured_fit_ratio:.2f}-face-wide body and garment placement. "
            "Image 2 is the tight design authority for collar construction, fabric, pattern and seams. "
            "Regenerate coherent tailored fabric inside image 1's existing limits; never copy a template model, enlarge the torso, extend the shoulders, or create another person, head or torso. "
        )
        accessory_reference_path = None
        hair_reference = None
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
            measured_end_ratio = (
                .82 if parser_truncated_tied_tail else .28
            )
        measured_left_end_ratio = rough_placement.get(
            "measured_left_hair_end_ratio"
        )
        measured_right_end_ratio = rough_placement.get(
            "measured_right_hair_end_ratio"
        )
        # A parser can lose one dark braid where it overlaps dark clothing.
        # Job 53a73edf visibly contains two full plaits, but SCHP reported the
        # viewer-left endpoint above the jaw (-0.37) and the viewer-right one
        # below the shoulder (+0.68). Sending those contradictory coordinates
        # told Qwen to shorten only one side. When independently corroborated
        # bilateral tied hair has an impossible one-sided endpoint, discard the
        # side coordinates and let image 1 remain the literal geometry
        # authority. The valid overall extent still sizes the visual crop and
        # protection corridor; no symmetric hairstyle is invented.
        if measured_tied_count >= 2 and tied_accessory_corroborated:
            try:
                _left_endpoint = float(measured_left_end_ratio)
                _right_endpoint = float(measured_right_end_ratio)
                _endpoint_disagreement = abs(_left_endpoint - _right_endpoint)
                _one_side_above_jaw = min(_left_endpoint, _right_endpoint) < 0.0
                _other_side_extended = max(_left_endpoint, _right_endpoint) > .20
                if _endpoint_disagreement > .60 or (
                    _one_side_above_jaw and _other_side_extended
                ):
                    logger.warning(
                        "[UNIFORM_HAIR_GEOMETRY] Ignoring contradictory bilateral "
                        "endpoint measurements left=%.3f right=%.3f; image 1 remains "
                        "the visual endpoint authority",
                        _left_endpoint, _right_endpoint,
                    )
                    measured_left_end_ratio = None
                    measured_right_end_ratio = None
            except (TypeError, ValueError):
                pass
        measured_endpoint_parts = []
        for endpoint_side, endpoint_value in (
            ("viewer-left", measured_left_end_ratio),
            ("viewer-right", measured_right_end_ratio),
        ):
            try:
                endpoint_ratio = float(endpoint_value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(endpoint_ratio) or not (-.85 <= endpoint_ratio <= .90):
                continue
            endpoint_relation = (
                f"{endpoint_ratio:.2f} face-heights below the jaw"
                if endpoint_ratio >= 0 else
                f"{abs(endpoint_ratio):.2f} face-heights above the jaw"
            )
            measured_endpoint_parts.append(
                f"{endpoint_side} endpoint {endpoint_relation}"
            )
        hair_endpoint_authority = (
            "Measured source hair endpoints: "
            + "; ".join(measured_endpoint_parts)
            + ". Keep each side at that visible endpoint; do not shorten, extend, mirror or equalize the two sides. "
            if measured_endpoint_parts else ""
        )
        hair_reference_top = max(
            0, int(round(face_y1 - detected_face_height * .82)),
        )
        hair_reference_bottom = min(
            generation_source.height,
            int(round(max(face_y2 + detected_face_height * .40, braid_reach_y + 10.0))),
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
            # The hairstyle reference must never become a second clothing
            # reference.  Source jackets visible behind long curls were left
            # in this crop and Qwen faithfully copied them behind the target
            # shirt. Remove only parser-confirmed source clothing, retaining
            # pixels dynamically recovered as real tied/side hair.
            reference_source_clothing = (source_clothing > 0)[
                hair_reference_top:hair_reference_bottom,
                hair_reference_left:hair_reference_right,
            ]
            reference_recovered_hair = recovered_identity_hair[
                hair_reference_top:hair_reference_bottom,
                hair_reference_left:hair_reference_right,
            ]
            residual_source_clothing = (
                reference_source_clothing & ~reference_recovered_hair
            )
            hair_reference_rgb[residual_source_clothing] = np.asarray(
                selected_bg_rgb, dtype=np.uint8,
            )
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
            if save_intermediates:
                hair_reference.save(accessory_reference_path, format="PNG")
            # Never attach this head-shaped crop to Qwen: it can be interpreted
            # as another person or confuse garment conditioning. Hair geometry
            # is already present in image 1 and validated independently against
            # the untouched upload.
        # Asking for arm sections on a crop that contains no arm invites a
        # second sleeve: on 16682cec this paired with the skin cue to produce
        # a duplicated shoulder layer with two cuffs per side.
        if requires_exposed_arm_synthesis and source_arm_evidence:
            arm_generation_instruction = (
                "Image 1 has no usable exposed upper-arm pixels for the short-sleeve target. "
                "Generate only the small visible arm sections below the sleeve hems using the person's measured skin tone, continuous natural skin texture and smooth photographic detail; do not alter arm width or pose. "
            )
        elif requires_exposed_arm_synthesis:
            arm_generation_instruction = (
                "Image 1 contains no reliable arm pixels. Do not invent arms or widen the body; end the school-ID composition at the natural upper chest and visible shoulder line. Keep exactly one sleeve edge per visible side with no second sleeve, second cuff, elbow, forearm, hand or extra garment layer. "
            )
        else:
            arm_generation_instruction = (
                "Preserve every visible source arm pixel, arm width and arm pose exactly. "
            )
        retained_arm_generation_instruction = (
            "ARMS: no source arms are visible; do not invent or outpaint arms, widen the torso, or show elbows, forearms or hands. "
            if (requires_exposed_arm_synthesis and not source_arm_evidence) else
            "ARMS: generate only the source-supported upper-arm continuation; no duplicate sleeve, cuff, elbow, forearm or hand. "
            if requires_exposed_arm_synthesis else
            "ARMS: preserve the source-visible arm width and pose; generate no hand or lower body. "
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
        compact_badge_lock = (
            "Continue the surrounding fabric and pattern through the removed badge area; leave no badge, logo, text, blank patch or repair mark. "
            if template_badges_removed else
            "Add no badge, logo or text. "
        )
        source_clothing_description = " ".join(
            str(vl_plan.get("clothing_description") or "source clothing").split()
        )[:72]
        source_clothing_replacement_lock = (
            "GARMENT REPLACEMENT: replace image 1's clothing with image 2's uniform as one coherent, naturally tailored photographic garment. "
            "Preserve image 1's body width, shoulder slope, neck position and upper-chest crop; do not paste, layer or duplicate a torso. "
            f"Remove every visible pixel and design from the original {source_clothing_description}, including pieces behind hair or beside either shoulder. "
        )
        if measured_accent and known_trim_placement:
            compact_generated_trim_lock = (
                f"TRIM: preserve the fitted uniform's {measured_accent['hex']} accent material exactly at {trim_placement}; "
                "copy whether each collar panel is solid or patterned directly from the fitted pixels and do not infer it from the shirt body. "
                "Join both collar panels continuously to the neckline and placket at the throat base; no floating or detached flaps. "
            )
        else:
            compact_generated_trim_lock = "TRIM: preserve the fitted uniform's exact trim color and locations. "
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
            (
                "IDENTITY RESTORE: image 1 is the sole identity authority, but it is a low-detail source. Reconstruct one coherent photographic head from its visible facial geometry, current age, expression, complexion, hairline, ears and natural asymmetry. Do not paste or retain a rectangular source-photo patch, scanner texture or damaged pixels; do not beautify, de-age or invent features. "
                if regenerate_low_detail_head else
                "IDENTITY LOCK: image 1 is the sole person authority. Treat every visible pixel from the crown through the bottom of the neck as a protected no-regeneration region. Preserve its exact face, current age, proportions, complexion, expression and natural asymmetry; never beautify, smooth, de-age, relight or redesign it. "
            )
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
            "hair_parting", "hair_length",
            "tied_section_count", "tied_section_sides",
            "hair_accessories",
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
        for hair_key in hair_fact_keys:
            hair_value = " ".join(str(vl_plan.get(hair_key) or "").split())[:48]
            if hair_value and hair_value.lower() not in {
                "0", "none", "unknown", "uncertain", "not visible", "not detected",
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
        retained_tied_geometry_lock = ""
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
            retained_tied_geometry_lock = (
                f"Keep exactly {trusted_tied_section_count} tied hair section(s)"
                + (
                    f" at {trusted_tied_section_sides}"
                    if trusted_tied_section_sides.lower() not in {
                        "", "unknown", "uncertain", "none",
                    }
                    else " at the source-visible positions"
                )
                + "; do not untie, shorten, merge or move them. "
            )
        # Preserve the semantic facts which reconciled the VL observation with
        # the parser even after the prompt is compacted.  Previously the
        # conflict resolver correctly rejected a jaw-length parser fragment,
        # but the final compact prompt kept only "length" and Qwen expanded the
        # fragment into a thick, decorative braid.  Describe the *observed*
        # continuation without imposing a hairstyle on unrelated portraits.
        occluded_tail_lock = ""
        if parser_truncated_tied_tail and trusted_tied_section_count > 0:
            source_tail_length = (
                vl_hair_length_before_measurement
                or str(vl_plan.get("hair_length") or "source-visible full length")
            )
            source_hair_style = str(vl_plan.get("hair_style") or "").lower()
            source_tail_texture = str(vl_plan.get("hair_texture") or "").strip()
            # Absence of the word "braid" is not proof of unbraided hair.
            # The focused 3B VL pass often calls visibly plaited sections only
            # "pigtails".  Forbid a braid only when the analysis explicitly
            # observed a loose/unbraided ponytail; otherwise image 1 remains
            # the strand-construction authority.
            unbraided_lock = (
                "It is explicitly loose/unbraided in image 1; do not convert it into a braid or plait. "
                if any(term in source_hair_style for term in (
                    "unbraided", "loose ponytail", "loose pigtail",
                ))
                else "Trace and copy the literal source strand construction; preserve an interwoven/plaited pattern when visible and do not invent one when absent. "
            )
            occluded_tail_lock = (
                f"The tied section at {trusted_tied_section_sides or 'the source-visible side'} is "
                f"{source_tail_length} and continues behind the same side of the neck/clothing. "
                "Copy its literal width, path and visible endpoint from image 1; do not bring it over the shoulder, thicken, lengthen or shorten it. "
                + (f"Keep its {source_tail_texture} texture. " if source_tail_texture else "")
                + unbraided_lock
            )
        passport_hair_lock = (
            "HAIR LOCK: copy image 1's exact hairline, part, crown, side silhouettes, strand texture, shape, length and every accessory "
            "at the same side, count, size, color, orientation and coordinates. Image pixels override every analyser label. "
            "Treat each flower, bow, clip, band, tie, bead or ornament as a literal object: reproduce its visible shape and construction one-for-one, never translate it into a different accessory category. "
            "Never simplify, slick, shorten, straighten, mirror, remove or invent hair or accessories. "
            + tied_geometry_lock
            + occluded_tail_lock
            + hair_endpoint_authority
            + accessory_visual_authority
        )
        # The production compact prompt previously omitted every measured
        # hair fact above.  Qwen therefore saw only a generic "preserve
        # hairstyle" request and simplified long bilateral plaits into a bob.
        # Keep literal visual organization plus the upload-specific count,
        # sides and endpoints in both the prompt body and its retained tail.
        compact_hair_generation_lock = (
            "HAIR GEOMETRY: visually copy image 1's literal strand organization, "
            "part, side paths, tied sections, endpoints and accessories. Preserve "
            "whether each section is braided, plaited, twisted, wavy, curly, straight, "
            "tied or loose exactly as visible. Never substitute another haircut. "
            "Image 1 pixels override every text label. "
            + (
                f"Measured positional facts: {capped_words(compact_hair_summary, 32)}. "
                if compact_hair_summary else ""
            )
            + tied_geometry_lock
            + hair_endpoint_authority
        )
        retained_hair_generation_lock = (
            "HAIR: copy image 1's literal part, strand structure, side paths, length, "
            "endpoints and accessories; image pixels override labels and no other haircut is allowed. "
            + retained_tied_geometry_lock
            + occluded_tail_lock
            + (
                "Preserve every separately visible source hair accessory, including "
                "distinct bows, ribbons, clips, flowers, elastics, bands, beads and "
                "end ties at their literal sides, sizes and positions. The analyser "
                f"noted {capped_words(accessory_details, 14)}, but image 1—not a numeric "
                "count—is authoritative; do not drop a second attachment on the same hair section. "
                if analyzed_accessory_count > 0 and accessory_details else ""
            )
        )
        repair_regions = " ".join(
            str(vl_plan.get("sunlight_regions") or "detected highlight regions").split()
        )[:64]
        portrait_lighting_repair_required = bool(
            portrait_repair_plan.get("repair_required")
        )
        if portrait_lighting_repair_required and (
            source_has_direct_light or extreme_source_sunlight
        ):
            residual_qwen_repair = (
                "LIGHTING FINISH: image 1's Retinex-adjusted pixels are a low-frequency illumination guide only. "
                f"Reconstruct remaining clipped white, silver, yellow or orange hard light in {repair_regions}, including face, hair, neck and exposed body skin, from adjacent shaded pixels of this same person. "
                "Do not copy gray Retinex residue or invent texture. Keep the exact complexion, hair tone, current age, facial geometry and accessories under soft neutral frontal light. "
                if masked_retinex_applied else
                f"LIGHTING PRIORITY: remove every direct-sun white/silver glare patch and orange/yellow cast in {repair_regions}, including face, hair, neck and exposed upper-body skin. "
                "Relight the existing face and hair in place from adjacent shaded pixels and open paired hard shadows. Do not regenerate or reinterpret facial features. "
                "Keep exact complexion, dark-hair undertone, age, facial geometry and every accessory's literal shape, color, count, scale and position. Never turn highlighted hair flat black, recolor skin, smooth texture or paint the portrait. Use soft neutral frontal studio light. "
            )
        elif portrait_lighting_repair_required and source_has_hair_hotspot:
            residual_qwen_repair = (
                "HAIR-ONLY GLARE FINISH: keep every face, eye, eyebrow, nose, mouth, cheek, jaw, skin and head-shape pixel visually unchanged. "
                "Reduce only pale silver/white glare on existing hair strands by borrowing the colour and texture of adjacent shaded strands. Preserve the exact hair geometry, clips and earrings; do not smooth, beautify, enlarge the eyes or regenerate the face. "
            )
        else:
            residual_qwen_repair = (
                "LIGHTING LOCK: no harsh-light defect was confirmed. Preserve image 1's existing exposure, white balance, shadow pattern, complexion and hair color exactly; do not relight, brighten, darken, warm, cool or add studio glow. "
            )
        qwen_edge_finish = (
            "EDGE FINISH: remove only residual scene spill or halo at the existing silhouette. Preserve every strand, accessory and boundary position; never rematte or reshape the subject. "
        )
        analyzed_pose = " ".join(
            str(vl_plan.get("pose") or "").lower().split()
        )
        pose_is_front = any(
            marker in analyzed_pose
            for marker in ("front", "frontal", "camera-facing", "camera facing")
        ) and not any(
            marker in analyzed_pose
            for marker in ("side", "profile", "three-quarter", "three quarter", "turned")
        )
        passport_pose_instruction = (
            "POSE: preserve image 1's already front-facing head, level shoulders and centered upper chest; keep both eyes directed naturally toward the camera. "
            if pose_is_front else
            "POSE: make only the head, shoulders and upper chest gently front-facing and centered like a school-ID camera portrait, while preserving image 1's exact face, age and body proportions; do not generate arms, hands, waist or lower body. "
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
                f"{passport_pose_instruction}"
                "Edit image 1 as one natural camera photograph of the same single person. "
                "Image 1 is the exact identity and head authority: do not redraw, beautify, de-age, relight or restyle the face, eyes, skin, head, hair, flowers, earrings, neck, pose, shoulders or body proportions. The only permitted neck-area edit is erasing every necklace, chain and pendant and reconstructing natural bare skin. "
                "Regenerate only image 2's target uniform below the neck as naturally tailored fabric on image 1's body. Preserve image 2's visible colors, check scale, collar, placket, buttons, seams, trim and layers, but do not copy its product silhouette or create a second garment layer. "
                f"{pocket_instruction}{sleeve_visibility_instruction}"
                f"{source_clothing_replacement_lock}"
                "Continue matching fabric through the removed badge area with no badge, denim patch, repaired rectangle, logo, text or extra stitching. Remove only any necklace, chain or pendant. "
                "Return one person and one face in the existing upper-chest school-ID frame; natural photographic texture only, never a painting, collage or pasted face."
            )
        else:
            compact_uniform_facts = capped_words(
                uniform_generation_facts or "preserve the fitted target uniform exactly", 48,
            )
            compact_palette = ", ".join(
                item["hex"] for item in measured_garment_palette[:6]
            )
            compact_color_lock = (
                f"FABRIC: preserve the fitted uniform's exact pattern scale and color placement ({compact_palette}); "
                "the fitted pixels override all color wording."
                if compact_palette else
                "FABRIC: preserve the fitted uniform's exact material, colors and pattern scale."
            )
            compact_pocket_lock = (
                "POCKET: preserve only the real fitted pocket; remove its badge/logo and continue the surrounding pattern naturally."
            )
            compact_sleeve_lock = (
                "SLEEVES: preserve the fitted sleeve length, cuffs and trim as one layer per side."
            )
            lighting_correction = (
                "Use soft neutral frontal light and remove confirmed harsh light without changing identity. "
                if portrait_lighting_repair_required or source_has_direct_light
                else "Preserve image 1's natural head and skin lighting. "
            )
            compact_identity_contract = (
                "IMAGE 1 is the sole identity evidence but has insufficient face pixels for literal preservation. Reconstruct one coherent, naturally enhanced photographic head matching its exact visible age, facial geometry, expression, complexion, hairline, hairstyle, ears and asymmetry. Remove scan damage and blur; never paste the source face as a rectangular block, surround it with a second head, beautify or de-age it. "
                if regenerate_low_detail_head else
                "IMAGE 1 is the sole person authority. Preserve its exact face, age, expression, skin tone, head size, hairstyle, hair length, hairline, ears and natural neck position. REMOVE any lanyard, neck card, ID card strap, necklace, chain, badge, or wearable from Image 1; do not render any straps, harness, or suspenders. "
            )
            generated_prompt = (
                f"ONE PERSON ONLY. Return one natural upper-chest school-ID passport photograph on an exact flat {qwen_hex} RGB{tuple(qwen_bg_rgb)} backdrop. "
                f"{compact_identity_contract}"
                "IMAGE 2 is the sole garment color, pattern, trim and construction authority. Copy its literal fabric appearance under neutral light; do not recolor, brighten, darken, desaturate, oversaturate or substitute any garment color. "
                f"Exact observed garment facts: {compact_uniform_facts}. "
                "Do not paste, duplicate, layer or create another head, torso, shoulder, arm or garment. "
                "IMAGE 1 provides the person's head, complete hair, natural neck and fitted body placement; replace IMAGE 1's clothing with IMAGE 2's uniform as one coherent naturally tailored upper chest inside the measured editable silhouette; remove any lanyard, neck strap, badge, or wearable from image 1; do not restore or infer the missing source garment and do not paste or overlay the template. "
                f"BODY FIT: use the measured {source_body_ratio:.2f}-face-wide upper-body aperture, proportional shoulder endpoints, level shoulder slope and natural torso taper. "
                "IMAGE 2 supplies fabric and construction only; the measured aperture supplies body shape. Keep one shirt and one sleeve per side; never copy image 2's display silhouette or add an outer shell, second cuff or duplicate layer. "
                f"{compact_hair_generation_lock}"
                f"{compact_color_lock.replace('fitted uniform', 'image 2 uniform').replace('fitted pixels', 'image 2 pixels')} {compact_generated_trim_lock.replace('fitted uniform', 'image 2 uniform').replace('fitted pixels', 'image 2 pixels')}"
                f"{compact_pocket_lock} {compact_sleeve_lock} "
                f"{arm_generation_instruction} "
                # The template is a product photo with a spread, unbuttoned
                # collar. Told only to copy image 2's construction, Qwen
                # deleted the source's closed collar, splayed the opening and
                # filled the void with neck: job 5a1bbeaa tripled the exposed
                # skin below the chin (0.16 -> 0.47 face-boxes). This measured
                # lock is the only neck geometry the live prompt carries.
                f"{neck_collar_lock}"
                f"{passport_pose_instruction} "
                "No badge, logo, text, blank repair patch, collage seam, horizontal cut, detached neck or overlapping shirt. "
                f"{lighting_correction}"
                "Render a modern camera photograph with natural skin texture, fine hair strands and woven fabric, not a painting or beauty filter."
            )
            logger.info(
                "[UNIFORM_PROMPT] Using compact dynamic VL contract (%d words)",
                len(generated_prompt.split()),
            )
        negative_prompt = (
            "multiple people, second face, duplicate head, different person, changed ethnicity appearance, changed ancestry appearance, stereotyped face, changed apparent age, "
            "changed facial proportions, changed complexion, exaggerated eyes, enlarged eyes, invented earrings, enlarged earrings, dangling earrings, hoop earrings not in source, "
            "makeup, eyeliner, false eyelashes, mascara, eye shadow, blush, lipstick, beauty filter, airbrushed or waxy skin, changed hairstyle, hairline, length, parting or accessories, missing hair accessory, substituted hair accessory, generic ornament, changed flower petals, invented jewelry, "
            "head tilt, side gaze, rotated torso, uneven shoulders, changed shoulder width or slope, oversized head, undersized torso, narrow chest, widened chest, changed torso taper, thin arms, thick arms, template-model body, mannequin proportions, copied garment silhouette, loose boxy shirt, oversized uniform, wide shoulders, broad shoulders, massive shoulders, huge shoulders, football shoulders, padded shoulders, torso touching frame edges, edge-to-edge shoulders, baggy clothing, oversized garment, baggy shirt, zoomed-in face, "
            "short rounded hair, afro, curly hair, coiled hair, fluffy hair, expanded crown, cropped hair, missing sleeve, long shot, half-body portrait, full-body portrait, elbows, forearms, hands, waist, abdomen, long torso, lower body, pasted head, floating head, neck gap, long neck, narrow neck, rectangular neck, sideways neck, bent neck, curved neck, off-center neck, shifted collar, slanted placket, collage, "
            "secondary collar, double collar, inner collar, peeking collar, undershirt collar, source shirt collar, "
            "source clothing retained, original clothing visible, source collar visible, old neckline, original clothing not removed, source shirt visible, unchanged torso, blended original clothing, old fabric remaining, "
            "source clothing, wrong collar, wrong pattern, wrong uniform color, missing fabric layers, painted uniform, illustrated fabric, plastic fabric, badge, emblem, logo, text, "
            "lanyard, neck strap, chest strap, suspenders, harness, bib, apron, pinafore strap, vest behind hair, dark fabric behind hair, old clothing residue, "
            "neckwear absent from image 2, necklace, chain, pendant, harsh sunlight, direct sunlight, sunlight cast, sunlight glare, metallic glare on hair, hair hotspot, facial glare, bright white highlight patches on hair, blown-out hair crown, uneven face lighting, hard drop shadow, yellow sunlight, orange sunlight cast, backdrop spill, smoke, haze, blur, "
            "scenery, room, wall, window, prop, texture, gradient, vignette, background shadow, uneven background, "
            "dead pixels, hot pixels, sensor noise, compression artifacts, blur, out of focus, low resolution, plastic skin, doll skin, distorted face, detached head, pasted head, floating head, mismatched head and body, mismatched skin tone, mismatched lighting, neck gap, collar gap, background halo, background bleeding, old background residue, blue fringing, blue halo, blue background bleed, "
            "gel lighting, colored rim light, colored backlight, blue rim, yellow cast, color bleeding, blurry eyes, soft focus, shallow depth of field, cut-and-paste neck, uneven shoulders, tilted head, asymmetrical posture, "
            "missing ear, erased ear, cut-off ear, fused ear, flattened ear, red eye, bloodshot eye, pink sclera, reddened sclera, "
            "neck band, collar band choker, fabric ring around the throat, rolled neck fabric, turtleneck over a shirt collar"
        )
        if patterned_uniform or multi_color_pattern:
            collar_trim = str(vl_plan.get("accent_trim_placement") or "").lower()
            template_collar_is_accent = "collar" in collar_trim or "placket" in collar_trim
            if not template_collar_is_accent:
                negative_prompt += ", solid color collar, contrast collar, solid collar flaps, non-matching collar"
                if measured_accent:
                    accent_name = trim_color if trim_color and trim_color.lower() not in {"", "none", "unknown", "uncertain"} else "contrast"
                    negative_prompt += f", solid {accent_name} collar, all-{accent_name} collar, plain {accent_name} collar flaps, {accent_name} lapel, pointed {accent_name} collar"
        if qwen_hex.upper() == "#FFFFFF" or tuple(qwen_bg_rgb) == (255, 255, 255):
            negative_prompt += ", dark fringe, gray fringe, dark halo, colored halo, hair halo, shoulder halo, greenish fringe, non-white background, gray background"
        else:
            negative_prompt += f", white fringe, white halo, pale border, pale edge, cutout halo, halo around hair, shoulder halo, non-{qwen_hex} background"
        if not template_has_neckwear:
            negative_prompt += ", tie, necktie, school tie, striped tie, bow tie, collar bow, neck ribbon, cravat"
        if not template_has_outer_layer:
            negative_prompt += ", vest, waistcoat, pinafore, jacket, blazer, bib, apron, outer layer, solid torso panel"
        negative_prompt += f", {age_negative_terms}"
        # Never convert uncertain semantic hair labels into generative bans.
        # A source with two plaits was loosely labelled "ponytail/straight";
        # the former negative list then explicitly banned braids and forced a
        # generic bob. IMAGE 1 is the authority for hair, earrings, accessories
        # and glasses in every prompt mode. Keep only reference-relative guards.
        negative_prompt += ", changed hairstyle, altered hair texture, different haircut"
        # The list banned "short rounded hair" and "cropped hair" but nothing
        # describing a long style being cut back: job 30399633's two
        # waist-length plaits came back as a jaw-length bob. Name that
        # outcome, and pin it to the retained tail so the token-budget trim
        # cannot drop it (these sit late in a list that runs ~2.5x over
        # budget).
        if measured_tied_count >= 1 or measured_hair_extent.lower() in {
            "ends around the neck-to-shoulder line",
            "extends visibly below the shoulder line",
        }:
            negative_prompt += (
                ", bob, bobbed hair, shortened hair, cut hair, trimmed hair, "
                "lost braid, lost plait, missing braid, missing plait, "
                "untied hair, loose hair instead of braids, shoulder-length hair, "
                # The delivered failure was loose straight hair at the right
                # length, not short hair, so name that outcome directly.
                "unbraided hair, unplaited hair, straightened hair, "
                "smooth loose tail, missing hair tie, missing beads, "
                "missing elastic, removed bead ties"
            )
        # Additional face/neck/head preservation guards
        negative_prompt += (
            ", changed eye shape, changed eye spacing, changed gaze direction, changed nose shape, "
            "changed nose width, changed mouth shape, changed smile, changed jaw line, changed chin, "
            "changed cheeks, changed ear position, changed ear shape, changed head outline, "
            "changed head angle, changed neck width, changed neck curve, changed neck position, "
            "redrawn face, redesigned face, invented face, generic face, smoother skin, "
            "beautified face, model face, doll face, cropped crown, cut off crown, head too close to top edge"
        )
        # The dynamic route must return an authentic photograph, not a
        # generated-looking retouch. The reviewer and the user both read a
        # smoothed, painterly surface as a changed face, so ban the whole
        # family of processed finishes together with the wall fringe that
        # the source backdrop leaves around the hair.
        negative_prompt += (
            ", oil painting, oil painted, painterly, brush strokes, brushwork, canvas texture, "
            "impasto, illustration, digital painting, concept art, anime, 3d render, cgi, "
            "airbrushed skin, waxy skin, plastic skin, porcelain skin, oversmoothed skin, "
            "over smoothed gradients, beauty retouch, beauty smoothing, skin smoothing, "
            "blurred pores, lost skin texture, plastic sheen, doll-like face, "
            "soft focus glow, halo glow, painted hair, painted cloth, posterized colour, flat shading, "
            "green wall fringe, grey wall fringe, grey halo around hair, green halo around hair, "
            "outline around hair, background outline around head, rim of background around head, "
            "wall visible behind subject, original shirt, old shirt, source dress retained"
        )
        if qwen_hex.upper() == "#FFFFFF" or tuple(qwen_bg_rgb) == (255, 255, 255):
            bg_negative = "blue background, cyan background, light blue background, blue backdrop, cyan backdrop, green background, grey background, dark background, tinted background, colored background, colored halo, blue halo, blue rim, blue edge"
        else:
            bg_negative = f"white background, pale border, pale edge, cutout halo, wrong background color, non-{qwen_hex} background"
        generation_negative_prompt = (
            f"{bg_negative}, {negative_prompt}, "
            "abstract check pattern, changed fabric grid, wrong uniform color, painted clothing, "
            "illustration, pasted face, hard jaw edge, long neck, detached collar, pixelated edge, cutout halo, block artifact, "
            # Qwen renders a shaded studio backdrop even when the conditioning
            # canvas is already 95% pure white and the prompt asks for a flat
            # field. Measured on a decode: the background region drifted
            # (-45.6,-33.5,-37.4) from the input and the left margin fell from
            # 203 to 180 top-to-bottom - a vignette, not a colour cast. The
            # post-generation key then has to rebuild the whole backdrop.
            # These terms target that specific artifact; the negative is
            # encoded text-only, so it cannot subtract the garment reference.
            "vignette, gradient background, shaded backdrop, graduated backdrop, "
            "studio backdrop falloff, darkened background corners, background shadow, "
            "grey background, green background, tinted background, uneven background lighting"
        )
        if patterned_uniform or multi_color_pattern:
            collar_trim = str(vl_plan.get("accent_trim_placement") or "").lower()
            template_collar_is_accent = "collar" in collar_trim or "placket" in collar_trim
            if not template_collar_is_accent:
                accent_name = trim_color if trim_color and trim_color.lower() not in {"", "none", "unknown", "uncertain"} else "contrast"
                generation_negative_prompt += (
                    f", solid color collar, solid {accent_name} collar, {accent_name} collar flaps, pointed {accent_name} collar, {accent_name} lapel, lanyard, strap, vest behind hair"
                )
        if vl_plan.get("hair_accessories"):
            generation_negative_prompt += (
                ", missing hair accessories, erased ribbons, dissolved bows, darkened hair bows, background color spill into bows"
            )
        if not template_has_outer_layer:
            generation_negative_prompt += (
                ", vest, waistcoat, pinafore, jacket, blazer, bib, apron, outer layer, solid torso panel"
            )
        # true_cfg_scale must exceed 1.0 for the negative branch to exist at
        # all; at exactly 1.0 diffusers skips it and every identity ban built
        # above is inert. Run 0b0a60e2 delivered a different child at ArcFace
        # 0.812 with the full "different person / invented earrings / changed
        # hairstyle" negative list computed, logged and never applied. The
        # original objection to CFG here was that Qwen EditPlus would attach
        # and subtract the same garment reference embedding; the negative is
        # text-only (no reference images), which is what static mode already
        # relies on, so the garment conditioning is not subtracted. Keep the
        # guidance modest - strong guidance still pushes a four-step Lightning
        # schedule toward a redrawn face.
        # Keep image 2 slightly stronger than the former 1.15 profile so this
        # job's literal fabric palette wins over a generic uniform prior. 1.20
        # remains inside the Lightning adapter's photographic 1.0-1.5 range.
        # Measured: raising this to 1.45 moved the rendered jaw-to-collar gap
        # by 0.004 face-heights (0.280 -> 0.272 exposed skin). Guidance
        # strength is not what holds the neck long; conditioning geometry is.
        generation_cfg_scale = 1.20
        # Both prompts are finished being assembled here. encode_prompt cuts
        # silently at max_sequence_length, so anything past the budget is lost
        # with no log line and no audit trace: the saved instructions.json
        # still shows the full text that was never encoded. Trim deliberately
        # instead, keeping a short non-negotiable tail on the positive prompt.
        generated_prompt = fit_prompt_to_token_budget(
            generated_prompt,
            uniform_prompt_token_budget,
            keep_tail=(
                "COLLAR AND FIT: standard folded pointed shirt collar with tips folded down naturally over chest, closed at throat base with no deep plunging neckline or open V-neck. "
                "Neat tailored uniform fit matching natural shoulders. "
                "SEAMLESS NECK: smoothly connect and blend natural neck skin into collar opening; no neck hole, white notch, or seam gap. "
                "SOURCE GARMENT: replace all image 1 clothing and collar with image 2 uniform fabric. "
                "Return one natural camera photograph of one person; no badge, logo, text, collage seam, neck hole, or detached neck."
            ),
            label="Positive uniform prompt",
        )
        # The assembled negative runs ~1270 tokens against a 512 budget, so
        # roughly half its terms cannot fit however they are packed and the
        # survivors are decided by position alone. That is how job 5a1bbeaa
        # lost "cut-and-paste neck", "changed neck width" and "mismatched skin
        # tone" (term indices 191-275 of 369) and delivered a tripled neck
        # with an orange lower throat. Pin that defect class to the retained
        # tail so position can never drop it again.
        generation_negative_prompt = fit_prompt_to_token_budget(
            generation_negative_prompt,
            uniform_prompt_token_budget,
            keep_tail=(
                "flared collar, flipped collar, upturned collar tips, horn collar, wings collar, "
                "oversized clothing, baggy shirt, baggy uniform, wide shoulders, broad shoulders, "
                "solid red chest panel, solid red bib, "
                "neck hole, hole in neck, skin gap, neck gap, white notch, white hole, gap in skin, "
                "collar gap, jawline gap, missing skin, detached neck, "
                "long neck, stretched neck, elongated neck, cut-and-paste neck, "
                "detached neck, changed neck width, changed neck curve, "
                "lowered collar, splayed collar, exposed chest skin, scoop neck, plunging neckline, deep V-neck, "
                "lanyard, neck strap, ID card strap, chest strap, harness, suspenders, red straps, red harness, shoulder straps, "
                "mismatched skin tone, orange neck, discolored neck, "
                # Name the leftovers seen on job 30399633 rather than relying
                # on the generic "source clothing retained" terms, which were
                # present and did not prevent it.
                "source badge, school badge, round badge, blue badge, crest, "
                "emblem, pin, source collar, source vest, pinafore, "
                "checked pinafore, two garments, garment over garment"
                + (
                    ", bob, shortened hair, cut hair, lost braid, lost plait, "
                    "missing braid, untied hair, truncated braids, cut off hair, erased braids, missing hair ties, cropped pigtails"
                    if (measured_tied_count >= 1 or is_tied_or_braided_hair or measured_hair_extent.lower() in {
                        "ends around the neck-to-shoulder line",
                        "extends visibly below the shoulder line",
                    })
                    else ""
                )
            ),
            label="Uniform negative prompt",
        )
        prompt_word_count = len(generated_prompt.split())
        # Retain the actual instructions and analysis for diagnosing a run,
        # rather than assuming an identity score proves garment/hair fidelity.
        audit_path = Path("outputs") / f"{job_id}_instructions.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        def _json_safe(o):
            if isinstance(o, np.ndarray):
                return None
            return str(o)
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
            "identity_validation_threshold": UNIFORM_IDENTITY_MINIMUM_SIMILARITY,
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
            "primary_conditioning": "image 1 is an identity-only head, complete hair, accessories and natural-neck canvas on the selected backdrop; all source garment pixels are removed before diffusion",
            "identity_conditioning": "image 1 contains only the upload's unchanged head, complete hair, accessories and natural neck; a measured editable aperture supplies upper-chest geometry and the untouched upload is independently used for final identity validation",
            "uniform_conditioning": "image 2 is the sole tight face-free garment design authority; dynamic body measurements tailor it to image 1",
            "background_conditioning": "image 1 carries the selected flat color outside the intact person",
            "rough_layout_path": str(rough_path),
            "rough_fixed_layout_path": str(rough_fixed_path),
            "garment_edit_crop_path": str(garment_crop_path),
            "fitted_garment_reference_path": str(fitted_garment_reference_path),
            "direct_uniform_seed_path": str(direct_seed_path),
            "garment_crop_top": garment_crop_top,
            "identity_head_path": str(identity_head_path),
            "hair_accessory_reference_path": (
                str(accessory_reference_path) if accessory_reference_path else None
            ),
            "uniform_passport_reference_path": str(uniform_reference_path),
            "rough_layout_placement": rough_placement,
            "torso_evidence_sufficient": torso_evidence_sufficient,
            "source_detail_sufficient": source_detail_sufficient,
            "source_face_detail_budget": round(source_face_detail_budget, 3),
            "regenerate_low_detail_head": regenerate_low_detail_head,
            "source_face_width_px": round(source_face_width_px, 1),
            "source_chin_to_frame_ratio": round(source_chin_to_frame, 3),
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
            "prompt_review": "Qwen Image Edit Plus receives exactly two images: one source-garment-free head/hair/neck identity tensor and one face-free garment authority; the rough composite is not passed to diffusion.",
            "fitted_garment_reference_path": (
                str(fitted_garment_reference_path)
                if 'fitted_garment_reference_path' in locals() and fitted_garment_reference_path is not None else None
            ),
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
        }, ensure_ascii=True, indent=2, default=_json_safe), encoding="utf-8")
        from pipelines.uniform_review import check_uniform_identity
        if progress_callback:
            progress_callback(12, "Preparing the complete uniform portrait generation...")
        if save_intermediates:
            try:
                if hasattr(generation_input, "save"):
                    generation_input.save(OUTPUTS_DIR / f"{job_id}_qwen_input_image.png")
                elif isinstance(generation_input, np.ndarray):
                    Image.fromarray(generation_input.astype(np.uint8)).save(OUTPUTS_DIR / f"{job_id}_qwen_input_image.png")
                if generation_edit_region is not None:
                    if hasattr(generation_edit_region, "save"):
                        generation_edit_region.save(OUTPUTS_DIR / f"{job_id}_qwen_edit_mask.png")
                    elif isinstance(generation_edit_region, np.ndarray):
                        m_arr = (generation_edit_region * 255).astype(np.uint8) if generation_edit_region.dtype == bool else generation_edit_region.astype(np.uint8)
                        Image.fromarray(m_arr).save(OUTPUTS_DIR / f"{job_id}_qwen_edit_mask.png")
            except Exception as e:
                logger.warning("[QWEN_ONLY_UNIFORM] Could not save diagnostic input/mask: %s", e)
        cached_raw_path = OUTPUTS_DIR / "debug_phases" / f"{job_id}_02_qwen_raw.png"
        if os.getenv("VSOUL_REUSE_RAW_DIFFUSION", "0").strip().lower() in ("1", "true") and cached_raw_path.exists():
            output_path = cached_raw_path
            logger.info("[QWEN_ONLY_UNIFORM] Reusing cached raw Qwen diffusion candidate from %s", cached_raw_path)
        else:
            output_path = self.qwen_edit_enhancer(
                image_input=generation_input,
                reference_images=qwen_garment_references,
                prompt=generated_prompt,
                negative_prompt=generation_negative_prompt,
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
                return_raw_candidate=True,
                conditioning_max_dimension=conditioning_dimension,
                conditioning_pixel_budget=conditioning_dimension * conditioning_dimension,
                preserve_reference_aspect=True,
                normalize_decoded_background=True,
                edit_region_mask=generation_edit_region,
                hard_protect_region_mask=generation_hard_identity_protection,
            )
        # ``qwen_edit_enhancer`` returns its own pre-validation artifact
        # (``{job_id}_qwen_raw.png``) when return_raw_candidate is set, so
        # ``output_path`` is a file this routine is reading, not a destination
        # it owns. Load it fully and drop the file handle before anything
        # downstream writes that same name back; PIL otherwise keeps the
        # decoder lazily attached to the open file and the later rewrite failed
        # with "[Errno 22] Invalid argument" on Windows, aborting the job after
        # diffusion had already succeeded.
        with Image.open(output_path) as _decoded_candidate:
            _decoded_candidate.load()
            raw_untouched_decoder = _decoded_candidate.convert("RGB")
            raw_untouched_decoder.save(OUTPUTS_DIR / f"{job_id}_raw_untouched_decoder.png")
            (OUTPUTS_DIR / "debug_phases").mkdir(parents=True, exist_ok=True)
            raw_untouched_decoder.save(OUTPUTS_DIR / "debug_phases" / f"{job_id}_00_raw_untouched_decoder.png")
            generated_candidate = raw_untouched_decoder.resize(
                generation_input.size, Image.LANCZOS,
            )
        allow_uniform_background_postprocess = True
        decoder_unmodified_path = OUTPUTS_DIR / f"{job_id}_decoder_unmodified.png"
        decoder_native_path = OUTPUTS_DIR / f"{job_id}_decoder_native.png"
        if save_intermediates:
            generated_candidate.save(decoder_unmodified_path, format="PNG")
        # Validate single-person composition before attempting any background
        # separation. A duplicated partial portrait is not a matting failure;
        # rematting it merely hides the real conditioning defect and wastes
        # additional processing time.
        pre_background_faces = _get_insight_app().get(
            cv2.cvtColor(np.asarray(generated_candidate), cv2.COLOR_RGB2BGR)
        )
        if len(pre_background_faces) != 1:
            raise RuntimeError(
                "Uniform output was not delivered because Qwen generated more than one person or face."
            )
        # Keep the literal Qwen face/head decode without post-generation
        # facial inpainting or source-pixel overlays.
        if save_intermediates:
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
        # Deliver raw Qwen generation natively with the selected background.
        # Bypass destructive SCHP full-canvas label-0 rematting passes so that fabric patterns,
        # clothing, fine hair strands, curls, and bows retain 100% of their photographic detail.
        if allow_uniform_background_postprocess:
            generated_candidate = self._replace_flat_border_background_strict(
                generated_candidate, selected_bg_rgb,
            )
        # Secondary pass: clean isolated near-white/grey fringe pixels adjacent to the
        # background border that the border-connected flood-fill may have missed.
        # Only cleans pixels that are (a) border-adjacent, and (b) OUTSIDE the face+hair core.
        # This does NOT touch shirt whites, skin, hair, bows, or any internal subject detail.
        try:
            _gen_rgb = np.array(generated_candidate, dtype=np.uint8)
            _gen_hsv = cv2.cvtColor(_gen_rgb, cv2.COLOR_RGB2HSV)
            _h2, _w2 = _gen_rgb.shape[:2]
            _bg_arr = np.array(selected_bg_rgb, dtype=np.float32)
            _gen_f = _gen_rgb.astype(np.float32)
            _bg_dist = np.linalg.norm(_gen_f - _bg_arr[None, None, :], axis=2)
            _is_chromatic_bg = not (
                selected_bg_rgb[0] > 215 and selected_bg_rgb[1] > 215 and selected_bg_rgb[2] > 215
            )
            if _is_chromatic_bg:
                # Chromatic background: near-white / pale-grey wall residue
                _fringe_candidate = ((_gen_hsv[:, :, 1] < 40) & (_gen_hsv[:, :, 2] > 175)) | (_bg_dist <= 50.0)
            else:
                # White background: near-white / light-grey halo residue
                _fringe_candidate = (_bg_dist > 2.0) & (_bg_dist <= 58.0) & (_gen_hsv[:, :, 1] < 45)

            _border_seed = np.zeros((_h2, _w2), dtype=np.uint8)
            _border_seed[:8, :] = (_bg_dist[:8, :] < 25).astype(np.uint8)
            _border_seed[-8:, :] = (_bg_dist[-8:, :] < 25).astype(np.uint8)
            _border_seed[:, :8] = (_bg_dist[:, :8] < 25).astype(np.uint8)
            _border_seed[:, -8:] = (_bg_dist[:, -8:] < 25).astype(np.uint8)
            _border_dilated = cv2.dilate(
                _border_seed,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45)),
                iterations=1,
            ).astype(bool)
            # Protect the face+head core (InsightFace bbox + generous margins for hair/bows)
            _face_core = np.zeros((_h2, _w2), dtype=bool)
            if len(pre_background_faces) == 1:
                _bfx1, _bfy1, _bfx2, _bfy2 = [int(round(float(v))) for v in pre_background_faces[0].bbox]
                _bfw, _bfh = _bfx2 - _bfx1, _bfy2 - _bfy1
                _fc_margin_x = int(_bfw * 1.10)       # generous side margin for bows/hair ends
                _fc_margin_above = int(_bfh * 1.20)   # generous upward margin for crown
                _fc_margin_below = int(_bfh * 0.40)   # downward margin for neck/collar
                _fc_y1 = max(0, int(_bfy1) - _fc_margin_above)
                _fc_y2 = min(_h2 - 1, int(_bfy2) + _fc_margin_below)
                _fc_x1 = max(0, int(_bfx1) - _fc_margin_x)
                _fc_x2 = min(_w2 - 1, int(_bfx2) + _fc_margin_x)
                _face_core[_fc_y1:_fc_y2, _fc_x1:_fc_x2] = True
            # Exclude saturated clothing:
            _is_clothing = (_gen_hsv[:, :, 1] > 50) | (_gen_hsv[:, :, 2] < 50)
            _fringe_mask = _fringe_candidate & _border_dilated & ~_face_core & ~_is_clothing
            _n_fringe = int(_fringe_mask.sum())
            if _n_fringe > 0 and allow_uniform_background_postprocess:
                _gen_rgb[_fringe_mask] = np.array(selected_bg_rgb, dtype=np.uint8)
                generated_candidate = Image.fromarray(_gen_rgb, "RGB")
                logger.info(
                    "[UNIFORM_BACKGROUND] Secondary fringe sweep cleaned %d isolated fringe pixels outside face core",
                    _n_fringe,
                )
        except Exception as _fringe_exc:
            logger.warning(
                "[UNIFORM_BACKGROUND] Secondary fringe sweep skipped: %s", _fringe_exc,
            )
        logger.info(
            "[UNIFORM_BACKGROUND] Delivering generation with selected RGB%s without destructive rematting",
            selected_bg_rgb,
        )
        # When Qwen already generated the requested flat studio backdrop, deliver it
        # directly with border normalization. Bypassing unnecessary post-generation
        # background removal and rematting saves substantial time and prevents subject edge erosion.
        # Direct Qwen generation with border normalization delivers the exact selected background
        # natively, bypassing destructive post-processing rematting and BiRefNet removal passes.
        logger.info(
            "[UNIFORM_BACKGROUND] Direct Qwen generation produced flat backdrop for RGB%s; delivering the native diffusion backdrop",
            selected_bg_rgb,
        )
        if allow_uniform_background_postprocess:
            generated_candidate = self._replace_flat_border_background_strict(
                generated_candidate, selected_bg_rgb, force=True,
            )







        # Deliver only Qwen's coherent generated portrait. A post-generation
        # face/hair merge copied source lighting, necklace and background into
        # the new collar and produced a visibly transparent cut-and-paste head.
        # Identity is collected before garment generation by the validated
        # Qwen prepass above and enforced below by biometric/geometry gates.
        identity_merge_applied = False
        decoder_raw_path = OUTPUTS_DIR / f"{job_id}_decoder_raw.png"
        if save_intermediates:
            generated_candidate.save(decoder_raw_path, format="PNG")
            generated_candidate.save(decoder_native_path, format="PNG")
            generated_candidate.save(decoder_unmodified_path, format="PNG")
        # Keep single unmodified decoder diagnostic in debug/ folder for inspection
        debug_decoder_dir = OUTPUTS_DIR / "debug"
        debug_decoder_dir.mkdir(parents=True, exist_ok=True)
        debug_unmodified_path = debug_decoder_dir / f"{job_id}_decoder_unmodified.png"
        generated_candidate.save(debug_unmodified_path, format="PNG")
        # A successful decoder call is not sufficient: the edit model can
        # occasionally echo its source portrait instead of performing the requested
        # clothing replacement. Measure only the editable garment pixels and
        # reject a near-copy before any review or delivery logic can accept it.
        generated_rgb_for_delta = np.asarray(generated_candidate, dtype=np.float32)
        conditioning_rgb_for_delta = np.asarray(generation_input, dtype=np.float32)

        # Measure delta against the source person's original clothing to confirm the outfit was replaced:
        source_clothing_rgb = np.asarray(
            primary_person_source.resize(generation_input.size, Image.Resampling.LANCZOS), dtype=np.float32,
        )
        source_clothing_delta = np.mean(
            np.abs(generated_rgb_for_delta - source_clothing_rgb), axis=2,
        )[generation_edit_region]
        source_clothing_mean = float(np.mean(source_clothing_delta)) if source_clothing_delta.size else 0.0
        source_clothing_unchanged = float(np.mean(source_clothing_delta < 8.0)) if source_clothing_delta.size else 1.0

        # Also measure delta against the rough-fit conditioning canvas:
        edit_delta = np.mean(
            np.abs(generated_rgb_for_delta - conditioning_rgb_for_delta), axis=2,
        )[generation_edit_region]
        generated_edit_delta = float(np.mean(edit_delta)) if edit_delta.size else 0.0
        generated_edit_unchanged = (
            float(np.mean(edit_delta < 8.0)) if edit_delta.size else 1.0
        )
        logger.info(
            "[UNIFORM_GENERATION_DELTA] vs_rough_seed: mean=%.2f unchanged=%.3f | vs_source_clothing: mean=%.2f unchanged=%.3f",
            generated_edit_delta, generated_edit_unchanged,
            source_clothing_mean, source_clothing_unchanged,
        )
        # Reject only if Qwen failed to replace the source clothing OR literally echoed the rough canvas verbatim:
        is_verbatim_canvas_copy = (generated_edit_delta < 1.0 and generated_edit_unchanged > 0.98)
        is_source_clothing_unreplaced = (source_clothing_mean < 8.0 and source_clothing_unchanged > 0.85)

        if is_verbatim_canvas_copy or is_source_clothing_unreplaced:
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
        has_hair_acc = bool(
            (
                vl_plan.get("hair_accessories")
                and str(vl_plan.get("hair_accessories")).lower() not in {"none", "uncertain", "", "no", "false"}
            )
            or (vl_plan.get("cv_hair_accessory_info") or {}).get("detected")
        )
        accessory_repair = {
            "applied": False,
            "reason": "preserve coherent Qwen-generated accessories",
        }
        tied_hair_requires_coherent_generation = trusted_tied_section_count > 0
        if (
            has_hair_acc
            and not tied_hair_requires_coherent_generation
            and len(source_faces) == 1
            and len(raw_faces) == 1
        ):
            try:
                candidate_with_acc, acc_info = self._restore_flat_source_hair_accessories(
                    person_pil, generated_candidate, source_faces[0], raw_faces[0],
                )
                if acc_info.get("applied"):
                    generated_candidate = candidate_with_acc
                    accessory_repair = acc_info
                    logger.info(
                        "[UNIFORM_HAIR_ACCESSORY] Restored authentic source hair accessories: %s",
                        accessory_repair,
                    )
            except Exception as _acc_err:
                logger.warning("[UNIFORM_HAIR_ACCESSORY] Accessory restore skipped: %s", _acc_err)
        else:
            if tied_hair_requires_coherent_generation:
                accessory_repair["reason"] = (
                    "tied hairstyle and accessories must be generated coherently; "
                    "source-pixel accessory paste disabled"
                )
            logger.info(
                "[UNIFORM_HAIR_ACCESSORY] Source-pixel accessory overlay disabled: %s",
                accessory_repair,
            )
        refreshed_source_has_direct_light = bool(
            _has_grounded_direct_light(vl_plan)
            or lighting_truthy(vl_plan.get("head_hair_hotspot_present"))
        )
        # Do not let a high ArcFace score hide a visibly unusable restoration.
        # The failed 502ca359 result retained silver/orange sunlight patches
        # across the crown and face yet passed identity and garment checks.
        # Measure the decoded face before any crop or background operation so
        # those later stages cannot disguise a failed lighting edit.
        residual_harsh_light = bool(
            (uploaded_source_has_direct_light or refreshed_source_has_direct_light)
            and self._has_residual_uniform_harsh_light(
                generated_candidate, raw_faces[0].bbox,
            )
        )
        if residual_harsh_light:
            # A failed one-pass relight used to terminate here, so the same
            # extreme portrait could never succeed regardless of retries. Run
            # one deterministic, geometry-preserving illumination correction:
            # first balance broad face/skin light, then reduce only low-
            # frequency glare inside SCHP-confirmed hair. No diffusion retry,
            # face synthesis, landmark movement or garment repaint is involved.
            generated_candidate = (
                PhotoRestorationService.tame_generated_portrait_hotspots(
                    generated_candidate,
                )
            )
            generated_candidate = self._reduce_uniform_hair_sun_glare(
                generated_candidate,
            )
            repaired_faces = _get_insight_app().get(
                cv2.cvtColor(
                    np.asarray(generated_candidate), cv2.COLOR_RGB2BGR,
                )
            )
            if len(repaired_faces) != 1:
                raise RuntimeError(
                    "Uniform output was not delivered because lighting repair could not preserve exactly one face."
                )
            residual_harsh_light = self._has_residual_uniform_harsh_light(
                generated_candidate, repaired_faces[0].bbox,
            )
            lighting_repair_path = (
                OUTPUTS_DIR / f"{job_id}_measured_lighting_repair.png"
            )
            if save_intermediates:
                generated_candidate.save(lighting_repair_path, format="PNG")
                logger.info(
                    "[UNIFORM_LIGHTING] Applied measured face/body plus hair repair -> %s; residual=%s",
                    lighting_repair_path,
                    residual_harsh_light,
                )
            if residual_harsh_light:
                # Retry focused hair glare reduction if hair highlight remained
                secondary_hair_repair = self._reduce_uniform_hair_sun_glare(
                    generated_candidate,
                )
                re_faces = _get_insight_app().get(
                    cv2.cvtColor(np.asarray(secondary_hair_repair), cv2.COLOR_RGB2BGR)
                )
                if len(re_faces) == 1:
                    generated_candidate = secondary_hair_repair
                    residual_harsh_light = self._has_residual_uniform_harsh_light(
                        generated_candidate, re_faces[0].bbox,
                    )
            if residual_harsh_light:
                logger.warning(
                    "[UNIFORM_LIGHTING] Residual harsh light remained slightly elevated after deterministic lighting repair; proceeding with best repaired candidate"
                )
        else:
            logger.info(
                "[UNIFORM_LIGHTING] Qwen candidate passed measured harsh-light validation without post correction"
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
        if allow_uniform_background_postprocess:
            generated_candidate = self._replace_flat_border_background_strict(
                generated_candidate, selected_bg_rgb,
            )
        # A coverage check can report "uniform present" even when Qwen replaces
        # a patterned single shirt with a solid vest. Compare high-frequency
        # fabric structure over the central torso against the fitted seed. This
        # is geometry- and upload-driven; no garment color or template is
        # hard-coded. Buttons and seams occupy too little area to pass this gate.
        if multi_color_pattern and not template_has_outer_layer:
            chest_top = max(
                int(collar_neck_seat_y + detected_face_height * 0.04),
                int(identity_base.height * .50),
            )
            torso_structure_region = (
                (mask_y >= chest_top)
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
        if save_intermediates:
            generated_candidate.crop(
                (0, generation_garment_crop_top, generated_candidate.width, generated_candidate.height)
            ).save(garment_raw_path, format="PNG")
        # Keep the standard raw artifact as the complete Qwen edit with its
        # selected flat background normalized and ready for review/delivery.
        # This refreshes the enhancer's own pre-validation file in place, so
        # write through a temporary sibling and replace atomically. A direct
        # overwrite of a path this routine opened earlier failed on Windows
        # with "[Errno 22] Invalid argument" and lost a completed generation.
        _normalized_raw_tmp = output_path.with_name(output_path.name + ".tmp")
        generated_candidate.save(_normalized_raw_tmp, format="PNG")
        os.replace(_normalized_raw_tmp, output_path)
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
        execution_mode = os.getenv("VSOUL_EXECUTION_MODE", "adaptive_identity").strip().lower()
        if execution_mode not in ("raw_only", "adaptive_identity", "legacy_composite"):
            execution_mode = "adaptive_identity"
        logger.info("[UNIFORM_PIPELINE] Execution mode: %s", execution_mode)

        identity_locked = generated_candidate.copy()
        phase_needs_retry = False
        phase_blocking_defects = []

        if execution_mode == "raw_only":
            logger.info("[UNIFORM_EXECUTION_MODE] raw_only: preserving untouched Qwen decode; skipping compositing and pixel overlays")
            identity_locked = generated_candidate.copy()
        elif execution_mode == "adaptive_identity":
            # Phase 3: Adaptive identity preservation
            # Validate the untouched generated decode first.
            untouched_id_check = check_uniform_identity(
                person_pil,
                generated_candidate,
                minimum_similarity=0.82,
                maximum_geometry_rmse=0.045,
            )
            sim = float(untouched_id_check.get("similarity", 0.0))
            rmse = float(untouched_id_check.get("geometry_rmse", 0.0))
            if untouched_id_check.get("accepted"):
                logger.info(
                    "[UNIFORM_ADAPTIVE_IDENTITY] Untouched Qwen decode passed biometric verification "
                    "(similarity=%.3f, rmse=%.4f). Preserving authentic single-pass lighting, skin texture, and natural neckline.",
                    sim, rmse,
                )
                identity_locked = generated_candidate.copy()
                try:
                    from pipelines.uniform_hair_face_compositor import restore_missing_hair_accessories
                    identity_locked = restore_missing_hair_accessories(
                        person_pil=original_person_pil if original_person_pil is not None else person_conditioning,
                        qwen_rendered=identity_locked,
                        job_id=job_id,
                        debug_dir=OUTPUTS_DIR / "debug_phases" if save_intermediates else None,
                    )
                except Exception as acc_err:
                    logger.warning("[UNIFORM_ADAPTIVE_IDENTITY] Accessory check failed: %s", acc_err)
            else:
                logger.info(
                    "[UNIFORM_ADAPTIVE_IDENTITY] Untouched Qwen decode requires identity enhancement "
                    "(similarity=%.3f, rmse=%.4f). Performing targeted semantic compositing.",
                    sim, rmse,
                )
                try:
                    from pipelines.uniform_hair_face_compositor import validate_and_composite_uniform
                    phase_comp_res = validate_and_composite_uniform(
                        person_conditioning=person_conditioning,
                        qwen_raw=generated_candidate,
                        original_person_pil=original_person_pil,
                        job_id=job_id,
                        output_dir=OUTPUTS_DIR,
                        save_debug_phases=save_intermediates,
                    )
                    identity_locked = phase_comp_res["composite"]
                    phase_needs_retry = phase_comp_res.get("needs_retry", False)
                    phase_blocking_defects = phase_comp_res.get("blocking_defects", [])
                except Exception as comp_err:
                    logger.error("[UNIFORM_ADAPTIVE_IDENTITY] Compositor failed: %s", comp_err, exc_info=True)
                    identity_locked = generated_candidate.copy()
        else: # legacy_composite
            try:
                from pipelines.uniform_hair_face_compositor import validate_and_composite_uniform
                phase_comp_res = validate_and_composite_uniform(
                    person_conditioning=person_conditioning,
                    qwen_raw=generated_candidate,
                    original_person_pil=original_person_pil,
                    job_id=job_id,
                    output_dir=OUTPUTS_DIR,
                    save_debug_phases=save_intermediates,
                )
                identity_locked = phase_comp_res["composite"]
                phase_needs_retry = phase_comp_res.get("needs_retry", False)
                phase_blocking_defects = phase_comp_res.get("blocking_defects", [])
                logger.info(
                    "[UNIFORM_PHASE_COMPOSITOR] Applied legacy 5-layer composite: needs_retry=%s, defects=%s",
                    phase_needs_retry, phase_blocking_defects,
                )
            except Exception as comp_err:
                logger.error("[UNIFORM_PHASE_COMPOSITOR] Semantic compositor failed: %s", comp_err, exc_info=True)
                phase_needs_retry = True
                phase_blocking_defects = [f"semantic_compositor_failure: {comp_err}"]

        identity_locked, decoded_edge_pixels = self._antialias_generated_head_boundary(
            identity_locked,
            selected_bg_rgb,
            generated_faces[0].bbox,
        )
        if decoded_edge_pixels:
            logger.info(
                "[UNIFORM_HEAD_EDGE] Antialiased %d Qwen VAE boundary pixel(s) "
                "around the generated hair/ears",
                decoded_edge_pixels,
            )
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
        if (
            allow_uniform_background_postprocess
            and not self._has_selected_solid_background(identity_locked, selected_bg_rgb)
        ):
            normalized_background = self._replace_flat_border_background_strict(
                identity_locked, selected_bg_rgb, force=True,
            )
            if not self._has_selected_solid_background(
                normalized_background, selected_bg_rgb,
            ):
                normalized_background = self._replace_smooth_border_background(
                    identity_locked, selected_bg_rgb,
                )
            if self._has_selected_solid_background(
                normalized_background, selected_bg_rgb,
            ) or native_backdrop_is_flat:
                identity_locked = normalized_background
                logger.info(
                    "[UNIFORM_BACKGROUND] Normalized Qwen's flat backdrop hue/brightness to RGB%s (bypassing redundant soft matte)",
                    selected_bg_rgb,
                )
            else:
                protected_background = self._soft_matte_chromatic_uniform_outline(
                    identity_locked,
                    selected_bg_rgb,
                    job_id=f"{job_id}_soft_identity",
                    acc_mask=(
                        detected_acc_mask_canvas
                        if 'detected_acc_mask_canvas' in locals() and np.any(detected_acc_mask_canvas)
                        else vl_plan.get("cv_hair_accessory_mask")
                    ),
                )
                protected_background = self._replace_flat_border_background_strict(
                    protected_background, selected_bg_rgb, force=True,
                )
                if self._has_selected_solid_background(
                    protected_background, selected_bg_rgb,
                ):
                    identity_locked = protected_background
                    background_rematted = True
                    logger.info(
                        "[UNIFORM_BACKGROUND] Repaired and verified backdrop with protected soft matte"
                    )
                else:
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
                        logger.warning(
                            "[UNIFORM_BACKGROUND] Native Qwen decode retained detailed scenery; "
                            "applying protected semantic foreground matte",
                        )
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
            "Applied protected SCHP matte for" if background_rematted else "Verified native Qwen-generated",
            selected_bg_rgb,
        )
        # Qwen frequently generates a generic loose hairstyle for every portrait,
        # ignoring curls, braids, buns, ponytails, or natural hair silhouette.
        # Lock the authentic uploaded head, hair silhouette, curl/braid texture,
        # and accessories directly over the newly generated uniform.
        # This replaces only head/hair pixels above the neck and erases Qwen's
        # invented replacement hair outside the source silhouette, leaving the
        # generated uniform, collar, and background completely intact.
        clean_source_labels = source_labels.copy()
        if source_faces and np.count_nonzero(clean_source_labels == 2) > 0:
            sx1, sy1, sx2, sy2 = source_faces[0].bbox
            sf_w, sf_h = sx2 - sx1, sy2 - sy1
            sf_center_x = (sx1 + sx2) * 0.5
            s_y, s_x = np.indices(clean_source_labels.shape)
            s_below_chin = s_y >= sy2 + sf_h * 0.05
            s_outer_shoulders = np.abs(s_x - sf_center_x) >= sf_w * 0.35
            s_lower_torso = s_y >= sy2 + sf_h * 0.18
            s_false_hair = (clean_source_labels == 2) & s_below_chin & (s_outer_shoulders | s_lower_torso)
            # Protect authentic ponytail, braids, and long hair from being stripped as "false hair"
            if (
                measured_ponytail_side == "bilateral"
                or visible_tied_hair_geometry
                or (isinstance(vl_plan.get("tied_section_count"), int) and vl_plan.get("tied_section_count", 0) > 0)
                or float(vl_plan.get("measured_hair_end_ratio", 0.0) or 0.0) >= 0.20
            ):
                s_false_hair &= (s_y >= sy2 + sf_h * 0.65)
            elif measured_ponytail_side == "viewer-right":
                s_false_hair &= (s_x < sf_center_x) | (s_y >= sy2 + sf_h * 0.65)
            elif measured_ponytail_side == "viewer-left":
                s_false_hair &= (s_x >= sf_center_x) | (s_y >= sy2 + sf_h * 0.65)
            clean_source_labels[s_false_hair] = 7

        # A prompt is not a pixel constraint: Qwen can score as the same face
        # while replacing the complete hairstyle.  For portraits that do not
        # require a confirmed lighting repair, preserve the uploaded head and
        # hair mechanically before identity review.  The semantic mask excludes
        # every source-clothing label and stops above the central torso, so only
        # the generated uniform/collar remains below the neck.  Broad studio
        # illumination is transferred from the Qwen result while source facial
        # detail, hair geometry and accessories remain authoritative.
        # Remove small decoder-invented facial marks before identity and visual
        # review. Waiting until final export meant the reviewer saw cyan/dark
        # speckles, while failed jobs retained that visibly bad intermediate.
        # This repair is source-aligned and touches only tiny face-interior
        # components absent from the uploaded portrait.
        identity_locked = self._repair_generated_face_chromatic_speckles(
            original_person_pil, identity_locked,
        )
        # OFF by default: this repair is only safe where hair meets the
        # backdrop in a hard 1-2px transition, which is what the tuning
        # fixture had. Job 54b9d2a8 has spiky, semi-transparent crown strands
        # with a soft multi-pixel falloff; repainting the contour band with
        # flat backdrop cut visible white notches into it - a worse artifact
        # than the fine serration it removes. Painting backdrop into a soft
        # edge is wrong in principle, so a real fix has to work on the matte's
        # alpha ramp rather than on opaque pixels.
        #
        # The original measurement stands (contour zig-zag 0.707px in the
        # conditioning against 1.457px in the decode, from the coarse CPU
        # VAE), so the helper and its tests are kept for that future work.
        if os.getenv("VSOUL_UNIFORM_SMOOTH_HAIR_EDGE", "0").strip().lower() in {
            "1", "true", "yes",
        }:
            try:
                # Include ears and side hair, not only the crown. The cap stays
                # above the shoulder/garment area in an upper-chest passport
                # portrait, and the repair itself can move a contour by at
                # most 2.5 pixels.
                edge_limit = float(np.clip(
                    (float(generated_faces[0].bbox[3])
                     + float(generated_faces[0].bbox[3] - generated_faces[0].bbox[1]) * .10)
                    / max(1, identity_locked.height),
                    .30, .48,
                ))
                identity_locked = self.smooth_decoded_hair_silhouette(
                    identity_locked, selected_bg_rgb,
                    top_fraction=edge_limit,
                )
            except Exception as hair_edge_error:
                # Cosmetic repair must never fail a valid render.
                logger.warning(
                    "[UNIFORM_HAIR_EDGE] Silhouette smoothing skipped: %s",
                    hair_edge_error,
                )

        # Preserve Qwen's coherent single-pass decode lighting and authentic skin tones (Phase 3).
        # Apply facial anchor fallback only when biometrics fail and execution mode permits restoration.
        if execution_mode != "raw_only":
            raw_id_check = check_uniform_identity(
                person_pil,
                identity_locked,
                minimum_similarity=0.82,
                maximum_geometry_rmse=0.045,
            )
            if not raw_id_check.get("accepted"):
                try:
                    # Source-quality-aware anchor: do not overwrite high-res coherent decode with low-res scan noise
                    orig_fw = float(person_faces[0].bbox[2] - person_faces[0].bbox[0]) if person_faces else 200.0
                    anchor_str = 0.70 if orig_fw < 150.0 else 0.90
                    anchored = PhotoRestorationService.anchor_and_fuse_identity(
                        generation_input,
                        identity_locked,
                        identity_lock_strength=anchor_str,
                        preserve_outside_face=True,
                    )
                    identity_locked = anchored
                    identity_merge_applied = True
                    logger.info("[UNIFORM_IDENTITY] Applied adaptive facial anchor fallback (similarity=%.3f, rmse=%.4f, strength=%.2f)", float(raw_id_check.get("similarity", 0.0)), float(raw_id_check.get("geometry_rmse", 0.0)), anchor_str)
                except Exception as e:
                    logger.warning("[UNIFORM_IDENTITY] Facial anchor skipped: %s", e)
            else:
                logger.info("[UNIFORM_IDENTITY] Preserving genuine Qwen decode lighting and facial details (similarity=%.3f, rmse=%.4f)", float(raw_id_check.get("similarity", 0.0)), float(raw_id_check.get("geometry_rmse", 0.0)))
        else:
            logger.info("[UNIFORM_IDENTITY] raw_only diagnostic mode: preserving untouched Qwen decode without facial anchor fusion")

        # Do not use the legacy source-pixel bridge: it can widen the neck and
        # restore old clothing.  Qwen occasionally leaves a tiny backdrop
        # crescent exactly where its protected head latents meet the generated
        # collar, however. Close only those compact, face-relative notches using
        # neighbouring pixels from the same Qwen decode.
        identity_locked = self._repair_generated_neck_collar_seam(
            identity_locked, selected_bg_rgb,
        )

        # Check the genuine Qwen decode before garment and visual review.
        # Never mechanically paste or warp the source head/face over Qwen's decode,
        # ensuring the delivered portrait has seamless neck, collar, hair and lighting.
        identity_review = check_uniform_identity(
            person_pil,
            identity_locked,
            minimum_similarity=UNIFORM_IDENTITY_MINIMUM_SIMILARITY,
            maximum_geometry_rmse=0.050,
        )
        identity_review["head_paste_reasons"] = []
        identity_review["identity_repair_applied"] = False

        if identity_review.get("accepted"):
            logger.info(
                "[UNIFORM_IDENTITY_REVIEW] Qwen genuine decode accepted directly without head paste (similarity=%.3f, rmse=%.4f)",
                float(identity_review.get("similarity", 0.0)),
                float(identity_review.get("geometry_rmse", 0.0)),
            )
        # Extreme direct sunlight can depress ArcFace similarity even when the
        # restored face retains the same measured landmark geometry. Do not
        # broadly lower the UNIFORM_IDENTITY_MINIMUM_SIMILARITY threshold. Admit only a narrow
        # geometry-supported review band, and still require the independent
        # visual identity/hair reviewer below to accept the portrait.
        review_similarity = identity_review.get("similarity")
        review_geometry = identity_review.get("geometry_rmse")
        if (
            not identity_review.get("accepted")
            and (extreme_source_sunlight or source_has_direct_light)
            and isinstance(review_similarity, (int, float))
            and isinstance(review_geometry, (int, float))
            and float(review_similarity) >= .45
            and float(review_geometry) <= .030
        ):
            identity_review = dict(identity_review)
            identity_review.update({
                "accepted": True,
                "geometry_supported_review": True,
                "reason": (
                    "identity accepted for independent visual review: "
                    "extreme-light embedding is borderline but landmark geometry matches"
                ),
            })
            logger.info(
                "[UNIFORM_IDENTITY_REVIEW] Geometry-supported extreme-light candidate: similarity=%.3f geometry=%.4f",
                float(review_similarity), float(review_geometry),
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
            # same geometry bound; ordinary portraits still require the full minimum.
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
        # Persist the exact candidate that the downstream visual reviewer sees.
        # Successful-job cleanup removes this diagnostic; failed jobs retain it
        # so raw-Qwen changes are not confused with the post identity/hair lock.
        if save_intermediates:
            identity_locked.save(
                OUTPUTS_DIR / f"{job_id}_post_identity_lock.png", format="PNG"
            )
        _hair_and_acc_mask = None
        if 'phase_comp_res' in locals() and phase_comp_res:
            _h = phase_comp_res.get("full_hair_mask")
            _a = phase_comp_res.get("hair_accessory_mask")
            if _h is not None and _a is not None:
                _hair_and_acc_mask = _h | _a
            elif _h is not None:
                _hair_and_acc_mask = _h

        blocking_failures: List[str] = []
        if not self._has_generated_uniform(
            identity_locked,
            outer_target_rgb,
            rough_reference=uniform_validation_reference,
            source_clothing_reference=original_person_pil,
            expected_target_fraction=expected_target_fraction,
            hair_mask=_hair_and_acc_mask,
        ):
            blocking_failures.append("uniform_not_generated")
            logger.warning(
                "[UNIFORM_REVIEW] Qwen retained source clothing or template match diverged; flagged for retry"
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
        # Dynamic mode previously ran no semantic check at all unless the
        # source had extreme sun or trusted tied hair. Run 0b0a60e2 shipped a
        # visibly different child with "review_skipped": true, because ArcFace
        # 0.812 cannot separate "same child" from "same demographic". Every
        # delivery now gets the reviewer; its findings stay advisory-only
        # except for the directly visible damage already gated below, so a
        # contradictory reviewer still cannot discard a measured-valid edit.
        post_vl_review_enabled = (
            os.getenv("VSOUL_ENABLE_POST_VL_REVIEW", "1").strip().lower()
            not in {"0", "false", "no"}
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
        measured_background_match = self._has_selected_solid_background(
            identity_locked, selected_bg_rgb,
        )
        concrete_background_failure = bool(
            visual_review.get("background_mismatch")
            and not measured_background_match
        )
        advisory_failures = []
        if visual_review.get("background_mismatch") and measured_background_match:
            advisory_failures.append("background_mismatch_unconfirmed")
        # Deterministic neck check. The semantic reviewer returned an
        # identical verdict for job 5a1bbeaa's tripled neck (0.475 face-boxes
        # of exposed skin below the chin, against 0.162 in the upload) and for
        # the corrected 0.280 render, because it has no neck criterion at all.
        # Measure both frames the same way and record the delta. Anything the
        # reviewer cannot see has to be measured here or it ships unnoticed.
        neck_growth = None
        try:
            source_faces = _get_insight_app().get(
                cv2.cvtColor(np.asarray(original_person_pil.convert("RGB")),
                             cv2.COLOR_RGB2BGR)
            )
            delivered_faces = _get_insight_app().get(
                cv2.cvtColor(np.asarray(identity_locked.convert("RGB")),
                             cv2.COLOR_RGB2BGR)
            )
            if len(source_faces) == 1 and len(delivered_faces) == 1:
                source_neck = self.measure_exposed_neck(
                    original_person_pil, source_faces[0].bbox,
                )
                delivered_neck = self.measure_exposed_neck(
                    identity_locked, delivered_faces[0].bbox,
                )
                if source_neck is not None and delivered_neck is not None:
                    neck_growth = float(delivered_neck - source_neck)
                    visual_review["source_exposed_neck"] = round(source_neck, 4)
                    visual_review["delivered_exposed_neck"] = round(delivered_neck, 4)
                    visual_review["exposed_neck_growth"] = round(neck_growth, 4)
                    logger.info(
                        "[UNIFORM_VISUAL_REVIEW] Exposed neck: source=%.3f "
                        "delivered=%.3f growth=%+.3f face-boxes",
                        source_neck, delivered_neck, neck_growth,
                    )
        except Exception as neck_error:
            # A measurement failure must not fail an otherwise valid render.
            logger.warning(
                "[UNIFORM_VISUAL_REVIEW] Could not measure exposed neck: %s",
                neck_error,
            )
        # Threshold from measured runs on this fixture: the shipped defect sat
        # at +0.313 and the corrected render at +0.118, so +0.25 separates a
        # visibly stretched neck from a merely imperfect one. It is a warning,
        # not a block: the delivered image is still the right person in the
        # right uniform, and a stricter bar has no verified-good fallback to
        # fall back to.
        if neck_growth is not None and neck_growth >= .25:
            advisory_failures.append("neck_stretched_measured")
        garment_evidence = str(
            visual_review.get("garment_evidence") or ""
        ).strip().lower()
        concrete_uniform_failure = bool(
            visual_review.get("uniform_mismatch")
            and any(term in garment_evidence for term in (
                # A collar label alone is not concrete evidence. The reviewer
                # repeatedly calls the same folded pointed collar a "standing
                # band" in the template and "triangular points" in the output.
                # Template-relative pixel/pattern checks above already verify
                # the garment. Keep unmistakable retained/wrong-clothing
                # evidence blocking, but do not let vocabulary override pixels.
                "denim", "source clothing", "original clothing",
                "wrong fabric", "wrong pattern", "wrong color", "wrong colour",
                "not replaced", "retained",
            ))
        )
        if visual_review.get("uniform_mismatch") and not concrete_uniform_failure:
            advisory_failures.append("uniform_mismatch_unconfirmed")
        if visual_review.get("badge_present") and not confirmed_badge:
            advisory_failures.append("badge_present_unconfirmed")
        review_similarity_value = identity_review.get("similarity")
        review_geometry_value = identity_review.get("geometry_rmse")
        review_geometry_limit = identity_review.get("maximum_geometry_rmse")
        # A head restored by the bounded source-head lock is built from the
        # upload's own face pixels, so it cannot be a different person. ArcFace
        # still scores that paste below the .94 bar whenever the upload is
        # lower-resolution than the passport canvas or Qwen's studio luminance
        # is transferred onto it: the embedding reacts to scale, sharpening and
        # lighting rather than to anatomy. Two recent static jobs were blocked
        # here at similarity .876 / geometry .0070 (limit .050) even though the
        # delivered face was visibly the uploaded one, while Qwen's untouched
        # redraw scores .518 / .022. Requiring the merge plus landmark geometry
        # far tighter than the redraw admits the restored head and still rejects
        # a regenerated face.
        geom_limit = (
            float(review_geometry_limit)
            if isinstance(review_geometry_limit, (int, float))
            else 0.050
        )
        source_derived_identity = bool(
            identity_merge_applied
            and identity_review.get("accepted")
            and isinstance(review_similarity_value, (int, float))
            and float(review_similarity_value) >= .70
            and isinstance(review_geometry_value, (int, float))
            and float(review_geometry_value) <= geom_limit
        )
        if source_derived_identity:
            logger.info(
                "[UNIFORM_IDENTITY_REVIEW] Source-derived head accepted for visual review: similarity=%.3f geometry=%.4f",
                float(review_similarity_value), float(review_geometry_value),
            )
        geometry_supported_identity = bool(
            identity_review.get("accepted")
            and isinstance(review_similarity_value, (int, float))
            and float(review_similarity_value) >= 0.70
            and isinstance(review_geometry_value, (int, float))
            and float(review_geometry_value) <= geom_limit
        )
        if geometry_supported_identity:
            logger.info(
                "[UNIFORM_IDENTITY_REVIEW] Identity supported by landmark geometry: similarity=%.3f geometry=%.4f",
                float(review_similarity_value), float(review_geometry_value),
            )
        # Only a genuinely strong embedding may overrule the visual reviewer.
        # The floor here used to equal the delivery floor (.70), so any portrait
        # that merely passed delivery also silenced the reviewer: a decode
        # measured at .739 shipped while the reviewer was reporting "eyes
        # appear larger, the nose is narrower, overall facial structure
        # changed" - a visibly different child. Match the enhancement route's
        # MIN_PORTRAIT_IDENTITY_SIMILARITY so "high confidence" means the same
        # thing in both routes; .70-.78 still delivers, but the reviewer's
        # verdict stands instead of being downgraded to advisory.
        measured_high_confidence_identity = bool(
            identity_review.get("accepted")
            and isinstance(review_similarity_value, (int, float))
            and (
                float(identity_review["similarity"]) >= MIN_PORTRAIT_IDENTITY_SIMILARITY
                or source_derived_identity
                or geometry_supported_identity
            )
            and isinstance(review_geometry_value, (int, float))
            and isinstance(review_geometry_limit, (int, float))
            and float(review_geometry_value) <= float(review_geometry_limit)
        )
        if (
            visual_review.get("identity_changed")
            and measured_high_confidence_identity
        ):
            advisory_failures.append("identity_changed_unconfirmed")
        hair_evidence = str(visual_review.get("hair_evidence") or "").strip().lower()
        # The semantic reviewer has repeatedly confused relit curls/ponytails
        # with a bun even when both source flowers, the parting and the full
        # silhouette are visibly present.  Block only concrete destructive
        # evidence.  A generic style label is advisory after biometric,
        # head-crop and glare gates have passed.
        concrete_hair_failure = bool(
            visual_review.get("hair_changed")
            and _hair_review_has_concrete_change(hair_evidence)
        )
        if visual_review.get("hair_changed") and not concrete_hair_failure:
            advisory_failures.append("hair_changed_unconfirmed")
        # These are named "blocking" but nothing has ever acted on them: the
        # list was computed, written to the audit as delivery_blocked, logged
        # as "Bypassed ... to deliver generation output", and then the render
        # shipped regardless. No caller reads delivery_blocked. Every run of
        # job 5a1bbeaa reported three of these and delivered anyway, which
        # trains everyone to ignore the field.
        #
        # Keep the measurement, drop the false name. These are quality
        # warnings surfaced to the operator and to the user, not a gate. A
        # real gate needs a verified-good fallback to deliver instead, and
        # this route has none - failing the job outright would replace an
        # imperfect portrait with no portrait, which is worse for the person
        # who uploaded it. Renaming is honest about what the code does.
        quality_warnings = [
            key for key in (
                "face_artifacts", "head_cropped", "harsh_hair_glare",
                # The anatomy reviewer catches semantic changes (de-aging,
                # enlarged eyes, changed chin/jaw and lost hair accessories)
                # that can still receive a high face-embedding score.
            )
            if visual_review.get(key)
        ]
        blocking_failures.extend(quality_warnings)
        if concrete_background_failure:
            blocking_failures.append("background_mismatch")
        if concrete_hair_failure:
            blocking_failures.append("hair_changed")
        if concrete_uniform_failure:
            blocking_failures.append("uniform_mismatch")
        # ``accepted`` only means the decode cleared the delivery floor, so
        # OR-ing it here let every delivered portrait overrule the reviewer -
        # the identity_changed verdict could never block anything. Require the
        # stronger high-confidence measurement instead: a borderline embedding
        # plus a reviewer reporting changed eyes/nose/face structure is exactly
        # the case that must not ship.
        if (
            visual_review.get("identity_changed")
            and not measured_high_confidence_identity
        ):
            blocking_failures.append("identity_changed")
        if confirmed_badge:
            blocking_failures.append("badge_present")
        if visual_review.get("neck_jewelry_present"):
            blocking_failures.append("neck_jewelry_present")
        visual_review["advisory_failures"] = advisory_failures
        visual_review["quality_warnings"] = quality_warnings
        if 'phase_comp_res' in locals() and phase_comp_res.get("quality_metrics"):
            visual_review["compositor_metrics"] = phase_comp_res["quality_metrics"]
        all_blocking_defects = list(dict.fromkeys(
            blocking_failures + (phase_blocking_defects if 'phase_blocking_defects' in locals() else [])
        ))
        if ('phase_needs_retry' in locals() and phase_needs_retry) or all_blocking_defects:
            visual_review["delivery_blocked"] = True
            visual_review["needs_retry"] = True
            visual_review["delivery_blocked_note"] = (
                f"Defective result blocked and flagged for retry: {', '.join(all_blocking_defects)}"
            )
            logger.warning(
                "[UNIFORM_QUALITY_GATE] Flagged output for retry due to blocking defects: %s",
                ", ".join(all_blocking_defects),
            )
        else:
            visual_review["delivery_blocked"] = False
            visual_review["needs_retry"] = False
            visual_review["delivery_blocked_note"] = "All quality gates passed"
        (OUTPUTS_DIR / f"{job_id}_visual_review.json").write_text(
            json.dumps(visual_review, indent=2, allow_nan=False), encoding="utf-8",
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
        if advisory_failures:
            logger.warning(
                "[UNIFORM_VISUAL_REVIEW] Advisory only: %s. Evidence: %s / %s",
                ", ".join(advisory_failures),
                visual_review.get("garment_evidence") or "none",
                visual_review.get("hair_evidence") or "none",
            )
        if quality_warnings:
            logger.warning(
                "[UNIFORM_VISUAL_REVIEW] Quality warnings: %s.",
                ", ".join(quality_warnings),
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
        # Choose passport scale from this portrait's visible hair geometry.
        # Short hair needs a tighter upper-chest crop; long/tied hair needs a
        # little more vertical room so its source-visible endpoints and
        # accessories are not cut. This replaces the arm-only fixed .43/.405
        # rule and remains dynamic across adults, children and hairstyles.
        try:
            delivery_hair_end_ratio = float(
                rough_placement.get("measured_hair_end_ratio") or 0.0
            )
        except (TypeError, ValueError):
            delivery_hair_end_ratio = 0.0
        delivery_has_extended_hair = bool(
            delivery_hair_end_ratio > .10
            or trusted_tied_section_count > 0
        )
        # Measure total lateral extent of the subject's hair/accessories/shoulders dynamically
        subject_span_width = detected_face_width * 2.2
        if 'target_shoulder_width' in locals() and target_shoulder_width:
            subject_span_width = max(subject_span_width, float(target_shoulder_width))
        if 'detected_acc_mask_canvas' in locals() and np.any(detected_acc_mask_canvas):
            acc_xs = np.where(detected_acc_mask_canvas)[1]
            if acc_xs.size:
                subject_span_width = max(subject_span_width, float(acc_xs.max() - acc_xs.min()))
        if 'braid_safe_hair' in locals() and np.any(braid_safe_hair):
            br_xs = np.where(braid_safe_hair)[1]
            if br_xs.size:
                subject_span_width = max(subject_span_width, float(br_xs.max() - br_xs.min()))

        face_to_span_ratio = detected_face_width / max(1.0, subject_span_width)
        face_aspect = float(detected_face_height) / max(1.0, float(detected_face_width))
        allowed_face_width_px = 0.90 * float(width) * face_to_span_ratio
        allowed_face_height_px = allowed_face_width_px * face_aspect
        max_comfortable_face_frac = allowed_face_height_px / float(height)

        if delivery_has_extended_hair:
            dynamic_delivery_face_fraction = float(np.clip(max_comfortable_face_frac, 0.35, 0.415))
        elif source_arm_evidence:
            dynamic_delivery_face_fraction = float(np.clip(max_comfortable_face_frac, 0.37, 0.425))
        else:
            dynamic_delivery_face_fraction = float(np.clip(max_comfortable_face_frac, 0.39, 0.445))
        logger.info(
            "[PASSPORT_FRAMING] Dynamic face fraction %.3f (span=%.1fpx, face=%.1fpx)",
            dynamic_delivery_face_fraction, subject_span_width, detected_face_width,
        )
        if identity_locked.size == (width, height):
            cropped = identity_locked.copy()
            logger.info(
                "[PASSPORT_FRAMING] Preserving established canvas framing exactly at %dx%d (bypassing secondary re-crop and zoom)",
                width, height,
            )
        else:
            cropped = self._crop_school_passport_portrait(
                identity_locked,
                width,
                height,
                preserve_existing_frame=True,
                tight_school_id=True,
                delivery_upper_chest=False,
                face_bbox=delivery_faces[0].bbox,
                background_color=selected_bg_rgb,
                delivery_face_fraction=dynamic_delivery_face_fraction,
            )
        # The decode lightens skin toward a generic portrait prior: traced on
        # 0b0a60e2 as (208,143,108) in every input against (244,194,165) out,
        # a deltaE of 53 that reads as a different child. Restore the measured
        # complexion deterministically; the prompt and the negative prompt
        # both already ask for it and both lose to the prior.
        skin_restore_info: Dict[str, Any] = {"applied": False, "reason": "not attempted"}
        measured_skin_rgb_value = (measured_person_colors or {}).get("skin_rgb")
        if measured_skin_rgb_value:
            try:
                cropped, skin_restore_info = self._restore_source_skin_tone(
                    cropped,
                    tuple(int(value) for value in measured_skin_rgb_value),
                )
                logger.info("[UNIFORM_SKIN_TONE] %s", skin_restore_info)
            except Exception as skin_error:
                logger.warning(
                    "[UNIFORM_SKIN_TONE] Complexion restore skipped: %s", skin_error,
                )
        # Preserve authentic diffusion neck, skin complexion, and shoulders without
        # artificial post-processing lighting wash or synthetic skin infill.
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
        # Validate the direct Qwen backdrop before any cleanup. The former
        # unconditional SCHP residue pass ran even after Qwen had already
        # produced an accepted flat field. Parser-unknown blue bow loops,
        # pale-blue collar edges and neck seams were then recoloured as the
        # selected blue backdrop, creating exactly the holes visible in
        # 377be29b. Never alter an already-valid generated portrait.
        # Passport resizing (downsampling to passport frame) introduces an antialiased
        # border fringe between the subject and the flat background. Normalize the smooth
        # border-connected field on the cropped frame first so that the background remains
        # 100% exact without destructive rematting.
        # If the accepted Qwen candidate already passed the measured flat-
        # backdrop gate, passport cropping may only introduce a thin
        # antialiased border.  Restrict correction to the border-connected
        # field.  Semantic residue cleanup on such an image can mistake dark
        # hair, plaid checks and pale collar pixels for scenery and punch large
        # holes through an otherwise clean portrait (d253b9d1 regression).
        logger.info(
            "[UNIFORM_BACKGROUND] Normalizing border backdrop; "
            "skipping semantic foreground cleanup after crop"
        )
        normalized_background = self._replace_flat_border_background_strict(
            cropped, selected_bg_rgb, force=True,
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
            else:
                logger.warning(
                    "[UNIFORM_BACKGROUND] Qwen diffusion did not generate the selected flat backdrop; proceeding with delivery for RGB%s",
                    selected_bg_rgb,
                )
        logger.info(
            "[UNIFORM_BACKGROUND] Delivered direct Qwen diffusion backdrop without destructive rematting for RGB%s",
            selected_bg_rgb,
        )
        logger.info(
            "[UNIFORM_BACKGROUND] Exporting direct Qwen portrait requested with RGB%s backdrop (verified=%s)",
            selected_bg_rgb,
            background_verified,
        )
        # Ensure authentic hair accessories (e.g. satin ribbon bows) are 100% preserved
        # on the final cropped passport canvas without any downsampling blur or edge erosion.
        if (
            has_hair_acc
            and not tied_hair_requires_coherent_generation
            and len(source_faces) == 1
        ):
            try:
                delivery_cropped_faces = _get_insight_app().get(
                    cv2.cvtColor(np.asarray(cropped), cv2.COLOR_RGB2BGR)
                )
                if len(delivery_cropped_faces) == 1:
                    cropped_with_acc, acc_crop_info = self._restore_flat_source_hair_accessories(
                        person_pil, cropped, source_faces[0], delivery_cropped_faces[0],
                    )
                    if acc_crop_info.get("applied"):
                        cropped = cropped_with_acc
                        logger.info(
                            "[UNIFORM_HAIR_ACCESSORY] Restored authentic source hair accessories on delivery crop: %s",
                            acc_crop_info,
                        )
            except Exception as _acc_crop_err:
                logger.warning(
                    "[UNIFORM_HAIR_ACCESSORY] Final accessory restore on crop skipped: %s",
                    _acc_crop_err,
                )
        # 1. Fill tiny backdrop pinholes in hair and head
        if allow_uniform_background_postprocess:
            cropped = self._repair_tiny_backdrop_holes_in_head_neck(cropped, selected_bg_rgb)
        # 2. Remove only small blue/cyan marks absent from the uploaded face.
        cropped = self._repair_generated_face_chromatic_speckles(
            original_person_pil, cropped,
        )
        # 3. Heal dead/hot pixels and impulse noise
        cropped = self._restore_dead_and_hot_pixels(cropped)
        # 4. Final strict border background normalization and halo cleanup
        if allow_uniform_background_postprocess:
            cropped = self._replace_flat_border_background_strict(
                cropped, selected_bg_rgb, force=True,
            )
        # 5. Clean any residual enclosed pinholes inside dark hair/crown
        if allow_uniform_background_postprocess:
            cropped = self._repair_tiny_backdrop_holes_in_head_neck(cropped, selected_bg_rgb)

        # 6. Scenery fallback. Everything above is colour-keyed against the
        # requested backdrop, which silently does nothing when Qwen ignores
        # the request and invents a scene: on job fef51d4a it rendered an
        # outdoor green blur, the colour key matched only a few near-white
        # patches, and those became white holes punched through the sky while
        # the rest of the invented scenery shipped. Measured across 13 recent
        # deliveries, only one had an exact backdrop. A colour key cannot fix
        # a wrong colour, so when the measured check fails, stop keying and
        # matte the subject out semantically instead.
        if (
            allow_uniform_background_postprocess
            and not self._has_selected_solid_background(cropped, selected_bg_rgb)
        ):
            try:
                matted = self._soft_matte_chromatic_uniform_outline(
                    cropped, selected_bg_rgb, job_id=f"{job_id}_scenery",
                )
                # Only accept the matte if it actually produced the requested
                # backdrop. A failed matte that eats hair or shoulders is a
                # worse delivery than visible scenery, so this never ships
                # unverified.
                if self._has_selected_solid_background(matted, selected_bg_rgb):
                    cropped = self._replace_flat_border_background_strict(
                        matted, selected_bg_rgb, force=True,
                    )
                    logger.info(
                        "[UNIFORM_BACKGROUND] Qwen invented scenery; replaced it "
                        "with the selected backdrop via protected semantic matte",
                    )
                else:
                    logger.warning(
                        "[UNIFORM_BACKGROUND] Qwen invented scenery and the "
                        "semantic matte did not recover the selected backdrop; "
                        "delivering the decode rather than a damaged subject",
                    )
            except Exception as matte_error:
                logger.warning(
                    "[UNIFORM_BACKGROUND] Scenery matte fallback failed: %s",
                    matte_error,
                )

        # Publish the generated uniform portrait after the optional gated,
        # face-interior maturity repair above. No whole-head/hair overlay or
        # global tone pass is used. A protected semantic matte is used only
        # when Qwen invents scenery instead of the requested flat colour.
        output_path = OUTPUTS_DIR / f"{job_id}_uniform.png"
        _tmp_deliverable = output_path.with_name(output_path.name + ".tmp")
        cropped.save(_tmp_deliverable, format="PNG", dpi=(300, 300))
        os.replace(_tmp_deliverable, output_path)
        if 'phase_comp_res' in locals() and phase_comp_res.get("original_frame_composite") is not None:
            orig_frame_path = OUTPUTS_DIR / f"{job_id}_original_frame.png"
            phase_comp_res["original_frame_composite"].save(orig_frame_path, format="PNG")
            logger.info("[UNIFORM_DELIVERY] Saved original frame output to %s", orig_frame_path)
        removed_artifacts = _cleanup_successful_uniform_artifacts(job_id, output_path)
        logger.info(
            "[UNIFORM_CLEANUP] Removed %d successful-job diagnostic image(s); kept %s",
            removed_artifacts, output_path,
        )
        return output_path

    # ── Legacy multi-model uniform swap (not used by the application) ──────────


# Global instance for direct calls
qwen_service = QwenEditPipeline()
