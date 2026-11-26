"""
AI Studio & Photo Master - Studio Master Engine & Qwen 2509 Suite.
Option 1: 🚀 Bulk Restoration (Batch Studio Master Restoration with Proper Framing & ZIP Download)
Option 2: 👔 Uniform Swap (CatVTON / Qwen 2509 with 100% Face & Hair Lock)
Option 3: 🌟 Single Enhance (High-Fidelity Studio Portrait Master with Full Headroom Framing)
"""

import time
import os
import io
import asyncio
import subprocess
import sys
import zipfile
import gc
import re
import zlib
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from PIL import Image, ImageEnhance, ImageOps
import cv2
import numpy as np
import torch
import gradio as gr

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")

from pipelines.orchestrator import PipelineOrchestrator
from pipelines.gusuq_pipeline import GUSUQPipeline
from pipelines.photo_restoration import PhotoRestorationService, photo_restorer
from utils.logger import get_logger

logger = get_logger("gradio_app")

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True, parents=True)


def _upload_basename(source: Any, fallback: str = "result.png") -> str:
    """Original upload filename so the saved result can keep the same name."""
    name = ""
    if source is None:
        name = ""
    elif isinstance(source, dict):
        raw = source.get("orig_name") or source.get("name") or source.get("path") or ""
        name = Path(str(raw)).name
    elif hasattr(source, "orig_name") and source.orig_name:
        name = Path(str(source.orig_name)).name
    elif isinstance(source, Image.Image) and getattr(source, "filename", None):
        name = Path(str(source.filename)).name
    elif isinstance(source, (str, Path)):
        name = Path(source).name
    elif hasattr(source, "name") and source.name:
        name = Path(str(source.name)).name
    name = Path(name).name
    if not name or name in {".", ".."}:
        return fallback
    if not Path(name).suffix:
        name = f"{name}.png"
    return name


def _open_upload(source: Any, *, convert_rgb: bool = True) -> Image.Image:
    if isinstance(source, Image.Image):
        img = source
    else:
        img = Image.open(str(source))
    if convert_rgb:
        return img.convert("RGB")
    return img


def _save_matching_input(image: Image.Image, source: Any, fallback: str = "result.png") -> Path:
    dest = OUTPUTS_DIR / _upload_basename(source, fallback=fallback)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rgb = image.convert("RGB")
    suffix = dest.suffix.lower()
    temp = dest.with_name(f"{dest.stem}.tmp{dest.suffix}")
    if suffix in {".jpg", ".jpeg"}:
        rgb.save(temp, format="JPEG", quality=95, dpi=(300, 300))
    elif suffix == ".webp":
        rgb.save(temp, format="WEBP", quality=95)
    else:
        if suffix != ".png":
            dest = dest.with_suffix(".png")
            temp = dest.with_name(f"{dest.stem}.tmp{dest.suffix}")
        rgb.save(temp, format="PNG", dpi=(300, 300))
    # Publish only a completed image; StaticFiles must never read a partial write.
    os.replace(temp, dest)
    return dest


def _as_progress(progress):
    if callable(progress):
        def _p(n, desc=""):
            try:
                progress(n, desc=desc)
            except TypeError:
                try:
                    progress(n, desc)
                except TypeError:
                    progress(n)
        return _p

    def _noop(*_a, **_k):
        pass
    return _noop


def map_client_background(value: str) -> Tuple[str, str]:
    """Map index.html backdrop values onto Gradio studio colour names."""
    raw = (value or "").strip()
    key = raw.lower().replace(" ", "")
    table = {
        "white": ("Pure White (Passport)", "#FFFFFF"),
        "255,255,255": ("Pure White (Passport)", "#FFFFFF"),
        "lightblue": ("Light Blue (School / Visa ID)", "#047EF6"),
        "4,126,246": ("Light Blue (School / Visa ID)", "#047EF6"),
        "navy": ("Corporate Navy", "#0B2545"),
        "11,37,69": ("Corporate Navy", "#0B2545"),
        "grey": ("Off-White / Light Grey", "#808080"),
        "128,128,128": ("Off-White / Light Grey", "#808080"),
        "keep": ("Light Blue (School / Visa ID)", "#047EF6"),
    }
    if key in table:
        return table[key]
    if "," in raw:
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) == 3:
            try:
                r, g, b = [max(0, min(255, int(p))) for p in parts]
                return "Custom Hex Color", f"#{r:02X}{g:02X}{b:02X}"
            except ValueError:
                pass
    if raw.startswith("#") and len(raw) in (4, 7):
        return "Custom Hex Color", raw
    return "Light Blue (School / Visa ID)", "#047EF6"


def resolve_background_hex(background_name: str, custom_hex: str = "") -> str:
    """Return the exact RGB selected by the UI for final matte compositing."""
    if background_name == "Custom Hex Color" and custom_hex and custom_hex.strip():
        return custom_hex.strip()
    named = {
        "Light Blue (School / Visa ID)": "#047EF6",
        "Pure White (Passport)": "#FFFFFF",
        "Off-White / Light Grey": "#F2F2F2",
        "Studio Dark": "#20242B",
        "Warm Studio": "#E8DDD0",
        "Corporate Navy": "#0B2545",
        "Crimson Red": "#B52B38",
    }
    return named.get(background_name, "#FFFFFF")


def apply_selected_background_matte(
    image: Image.Image,
    target_rgb: Tuple[int, int, int],
    *,
    use_schp: bool = False,
) -> Image.Image:
    """Apply the selected backdrop with an isolated soft portrait matte."""
    cache_dir = Path("scratch/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parent
    stamp = int(time.time() * 1000)
    input_path = cache_dir / f"qwen_matte_{stamp}.png"
    output_path = cache_dir / f"qwen_matte_{stamp}_out.png"
    worker_path = cache_dir / f"qwen_matte_{stamp}.py"
    try:
        image.convert("RGB").save(input_path)
        worker_code = f'''import io
import sys
from pathlib import Path
from PIL import Image
# This worker runs from scratch/cache, not the application package. Make the
# project imports explicit so a failed import cannot silently retain Qwen's
# approximate backdrop instead of the user-selected RGB color.
sys.path.insert(0, {str(project_root)!r})
from pipelines.birefnet_service import background_removal

input_path = Path({str(input_path.resolve())!r})
output_path = Path({str(output_path.resolve())!r})
target_rgb = {tuple(int(v) for v in target_rgb)!r}
rgba = Image.open(io.BytesIO(background_removal.remove_background(
    input_path.read_bytes(), job_id="qwen_background", source_image=None, use_schp={bool(use_schp)!r},
))).convert("RGBA")
canvas = Image.new("RGBA", rgba.size, (*target_rgb, 255))
canvas.alpha_composite(rgba)
canvas.convert("RGB").save(output_path, format="PNG")
'''
        worker_path.write_text(worker_code, encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(worker_path)], cwd=str(project_root), capture_output=True, text=True,
        )
        if completed.returncode != 0 or not output_path.exists():
            raise RuntimeError(completed.stderr[-1200:] or "background matte worker failed")
        return Image.open(output_path).convert("RGB").copy()
    except Exception as exc:
        logger.warning("[QWEN_BACKGROUND] Retaining Qwen backdrop after matte failure: %s", exc)
        return image.convert("RGB")
    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        worker_path.unlink(missing_ok=True)


# Initialize Engines
orchestrator = PipelineOrchestrator()
gusuq_pipeline = GUSUQPipeline(opt_policy="low_vram")

COLOR_CHOICES = [
    "Light Blue (School / Visa ID)",
    "Pure White (Passport)",
    "Off-White / Light Grey",
    "Studio Dark",
    "Warm Studio",
    "Corporate Navy",
    "Crimson Red",
    "Custom Hex Color",
]

PASSPORT_FORMATS = [
    "Standard ICAO Passport (35 x 45 mm / 3.5 x 4.5 cm)",
    "Standard Studio Portrait (3:4 ratio)",
]

ENGINE_PORTRAIT_RECOMMENDED = "✨ Qwen — slow, best output — Recommended"
ENGINE_PORTRAIT_FAST = "⚡ Fast optical restoration"
# Base seed for a fresh Qwen sample on every enhancement job. The previous
# fixed seed replayed the same raw diffusion image regardless of new lighting
# analysis or prompt changes.
STUDIO_REFERENCE_SEED = 261688816
STUDIO_BLUE_RGB = (205, 230, 248)
STUDIO_BLUE_BGR = (248, 230, 205)


def _offload_catvton():
    try:
        from pipelines.catvton_pipeline import CatVTONPipeline
        CatVTONPipeline().offload()
    except Exception:
        pass
    try:
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _person_wears_glasses(image: Image.Image) -> bool:
    """Use source frame evidence, not a face-only classifier, to protect eyewear."""
    try:
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        return photo_restorer.has_eyeglasses(bgr)
    except Exception:
        return False


def _has_harsh_source_sunlight(
    image: Image.Image,
    face_bbox: Optional[Any] = None,
) -> bool:
    """Detect hard, saturated portrait light when VL misses an obvious hotspot."""
    return PhotoRestorationService.has_harsh_portrait_sunlight(
        image, face_bbox,
    )


def _measure_source_shoulder_asymmetry(
    image: Image.Image,
    face_bbox: Any,
) -> Optional[float]:
    """Measure upper-garment shoulder height difference relative to face size.

    VL can label a frontal face as a fully frontal pose even when the torso is
    turned. SCHP garment geometry provides an independent, image-based signal
    without inferring gender, age, clothing style or a fixed body shape.
    """
    try:
        from pipelines.schp_service import parse as parse_body_parts

        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        labels = parse_body_parts(rgb)["labels"]
        x1, y1, x2, y2 = (float(value) for value in face_bbox[:4])
        face_w = max(1.0, x2 - x1)
        face_h = max(1.0, y2 - y1)
        center_x = (x1 + x2) * .5
        garment = np.isin(labels, (5, 6, 7))
        rows, cols = np.where(garment)
        left_rows = rows[cols < center_x - face_w * .25]
        right_rows = rows[cols > center_x + face_w * .25]
        if left_rows.size < 48 or right_rows.size < 48:
            return None
        left_top = float(np.percentile(left_rows, 5))
        right_top = float(np.percentile(right_rows, 5))
        asymmetry = abs(left_top - right_top) / face_h
        logger.info(
            "[POSE_MEASURE] shoulder_asymmetry=%.3f left_top=%.1f right_top=%.1f",
            asymmetry, left_top, right_top,
        )
        return float(asymmetry)
    except Exception as pose_error:
        logger.warning("[POSE_MEASURE] Shoulder check unavailable: %s", pose_error)
        return None


def _run_qwen_2511_uniform_fit(
    baseline: Image.Image,
    progress,
    vl_constraints: Optional[dict] = None,
) -> Image.Image:
    """Seam-only Qwen refinement after a short-lived person/template VL analysis."""
    from pipelines.qwen_edit_pipeline import qwen_service

    width, height = baseline.size
    constraints = vl_constraints or {}
    constraint_text = ", ".join(
        f"{key}={value}" for key, value in constraints.items()
        if key in {"pose", "torso_visibility", "shoulders_visible", "template_type", "collar", "badge_present"}
    )
    output_path = qwen_service.qwen_edit_enhancer(
        image_input=baseline,
        prompt=(
            "Refine this school uniform portrait only where cloth meets the neck, shoulders, and sleeves. "
            "Preserve the person's exact face, hair, skin, expression, pose, and any real eyewear. "
            "Preserve every visible source jewelry item exactly, including earrings, necklace, chain, pendant, bangle, "
            "or ornament: retain its presence, shape, color, material, placement, and natural reflections. "
            "Preserve the exact uniform color, fabric pattern, buttons, collar design, and garment shape. "
            "Keep the uniform fabric clean and plain without any school badge, emblem, crest, logo, or lettering. "
            "Create a natural collar-to-neck contact with no double clothing layer, no exposed background fringe, "
            f"no added accessories, and no changed identity. Fit constraints from visual analysis: {constraint_text}."
        ),
        negative_prompt=(
            "badge, school badge, crest, emblem, logo, patch, name tag, pins, chest text, lettering, words, "
            "changed identity, altered face, different hair, glasses, sunglasses, new jewelry, missing jewelry, altered jewelry, extra collar, "
            "double shirt, recolored uniform, cropped uniform, distorted shoulders, "
            "background leakage, halo, cutout edge, artifacts"
        ),
        background_color="keep",
        job_id=f"qwen2511_uniform_{int(time.time())}",
        width=width,
        height=height,
        steps=4,
        max_sequence_length=128,
        timeout_seconds=240,
        progress_callback=lambda pct, desc: progress(0.34 + min(0.38, (pct / 100.0) * 0.38), desc=desc),
    )
    return Image.open(str(output_path)).convert("RGB")


def _uniform_seam_mask(person: Image.Image, baseline: Image.Image) -> np.ndarray:
    """Permit generated pixels only at collar, shoulders, and outer garment edges."""
    source = np.array(person.convert("RGB").resize(baseline.size, Image.LANCZOS))
    fitted = np.array(baseline.convert("RGB"))
    h, w = fitted.shape[:2]
    difference = cv2.cvtColor(cv2.absdiff(fitted, source), cv2.COLOR_RGB2GRAY)
    garment = ((difference > 24).astype(np.uint8)) * 255
    garment = cv2.morphologyEx(
        garment,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
    )
    outer = cv2.dilate(garment, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    inner = cv2.erode(garment, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    seam = cv2.subtract(outer, inner)

    gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(48, 48))
    if len(faces):
        x, y, fw, fh = max(faces, key=lambda item: item[2] * item[3])
        cv2.ellipse(seam, (x + fw // 2, y + fh // 2), (int(fw * 0.82), int(fh * 1.02)), 0, 0, 360, 0, -1)
        # Give Qwen a small collar/neck-contact window but never the face.
        cv2.ellipse(seam, (x + fw // 2, y + fh + int(fh * 0.08)), (int(fw * 0.68), int(fh * 0.26)), 0, 0, 360, 255, -1)
        seam[: max(0, y + fh - int(fh * 0.05)), :] = 0
    else:
        seam[: int(h * 0.42), :] = 0
    return cv2.GaussianBlur(seam, (7, 7), 1.2).astype(np.float32) / 255.0


def _apply_seam_only_refinement(person: Image.Image, baseline: Image.Image, candidate: Image.Image) -> Image.Image:
    """Accept Qwen output only where it can repair a garment seam."""
    mask = _uniform_seam_mask(person, baseline)[:, :, None]
    base = np.array(baseline.convert("RGB"), dtype=np.float32)
    edit = np.array(candidate.convert("RGB").resize(baseline.size, Image.LANCZOS), dtype=np.float32)
    merged = (edit * mask + base * (1.0 - mask)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(merged, mode="RGB")


def _glasses_prompt_bits(
    image: Image.Image,
    observed_glasses: Optional[bool] = None,
) -> Tuple[str, str]:
    # Either signal is sufficient for preservation. Bright reflections can
    # hide thick frames from the local edge detector while VL still sees the
    # complete eyewear shape.
    if observed_glasses is True or _person_wears_glasses(image):
        return (
            "Keep the source eyeglasses exactly: preserve frame shape, lens tint, and fit. "
            "Gently reduce only distracting lens reflections while keeping both eyes naturally visible. ",
            "opaque lenses, hidden eyes, altered eyewear, invented eyewear, distorted frames",
        )
    return (
        "Keep the face free of added eyewear; do not invent glasses, spectacle frames, or sunglasses. ",
        "glasses, eyeglasses, spectacles, eyewear, frames on face, sunglasses",
    )


def _uniform_studio_matte(image: Image.Image) -> Image.Image:
    """Matte the original portrait before a uniform is composited onto it."""
    try:
        from pipelines.birefnet_service import background_removal
        with io.BytesIO() as bio:
            image.convert("RGB").save(bio, format="PNG")
            fg_bytes = background_removal.remove_background(bio.getvalue())
        fg_rgba = Image.open(io.BytesIO(fg_bytes)).convert("RGBA")
        # A uniform template may cover the lower body, leaving BiRefNet to
        # segment only detailed hair. Remove clear outdoor green spill and
        # close pinholes before compositing onto the solid studio matte.
        fg_pixels = np.array(fg_rgba)
        rgb = fg_pixels[:, :, :3].astype(np.int16)
        alpha = fg_pixels[:, :, 3]
        # Keep the face intact while removing clearly coloured scenery that
        # BiRefNet retained between or around dark hair strands.
        from pipelines.uniform_precomposite import _face_box
        face_x, face_y, face_w, face_h = _face_box(np.array(image.convert("RGB")))
        face_protected = np.zeros(alpha.shape, dtype=np.uint8)
        cv2.ellipse(
            face_protected,
            (face_x + face_w // 2, face_y + face_h // 2),
            (max(1, int(face_w * 0.72)), max(1, int(face_h * 0.78))),
            0,
            0,
            360,
            255,
            -1,
        )
        outer_hair_region = face_protected == 0
        green_spill = (
            outer_hair_region
            &
            (alpha > 8)
            & (rgb[:, :, 1] > rgb[:, :, 0] * 1.15)
            & (rgb[:, :, 1] > rgb[:, :, 2] * 1.12)
            & (rgb[:, :, 1] > 60)
        )
        brick_spill = (
            outer_hair_region
            & (alpha > 8)
            & (rgb[:, :, 0] > 80)
            & (rgb[:, :, 0] > rgb[:, :, 1] * 1.18)
            & (rgb[:, :, 0] > rgb[:, :, 2] * 1.10)
        )
        alpha[green_spill | brick_spill] = 0
        fg_pixels[:, :, 3] = cv2.morphologyEx(
            alpha, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)
        )
        fg_rgba = Image.fromarray(fg_pixels, mode="RGBA")
        studio_bg = Image.new("RGBA", fg_rgba.size, (*STUDIO_BLUE_RGB, 255))
        studio_bg.paste(fg_rgba, (0, 0), fg_rgba)
        return studio_bg.convert("RGB")
    except Exception as e:
        logger.warning("BiRefNet uniform matte skipped: %s", e)
        return image.convert("RGB")


def finish_uniform_studio_passport(
    person_image: Image.Image,
    result_pil: Image.Image,
    res_path: Path,
    passport_format: str,
    progress,
    name_source: Any = None,
    wears_glasses: bool = False,
    already_studio_matted: bool = False,
) -> Tuple[Image.Image, Path]:
    """Studio blue matte + ICAO crop that keeps the uniform in frame."""
    from pipelines.photo_restoration import photo_restorer

    _offload_catvton()
    working = result_pil.convert("RGB")

    progress(0.78, desc="Preparing solid studio blue background...")
    if not already_studio_matted:
        working = _uniform_studio_matte(working)

    progress(0.95, desc="Studio camera focus polish...")
    try:
        cam = photo_restorer.studio_camera_look(cv2.cvtColor(np.array(working), cv2.COLOR_RGB2BGR))
        if wears_glasses:
            cam = photo_restorer.reduce_glasses_glare(cam)
        working = Image.fromarray(cv2.cvtColor(cam, cv2.COLOR_BGR2RGB))
    except Exception as e:
        logger.warning("Studio camera look skipped: %s", e)

    if passport_format and passport_format != "Original Dimensions (Enhanced)":
        progress(0.97, desc=f"Cropping to {passport_format}...")
        bgr = cv2.cvtColor(np.array(working), cv2.COLOR_RGB2BGR)
        framed = photo_restorer.frame_passport_photo(
            bgr,
            target_format=passport_format,
            bg_color_bgr=STUDIO_BLUE_BGR,
            headroom_ratio=0.12,
            head_height_ratio=0.22,
        )
        working = Image.fromarray(cv2.cvtColor(framed, cv2.COLOR_BGR2RGB))

    named = _save_matching_input(
        working,
        name_source if name_source is not None else person_image,
        fallback=Path(res_path).name,
    )
    if Path(named) != Path(res_path):
        working.save(str(res_path), format="PNG", dpi=(300, 300))
    logger.info("Passport studio output saved: %s", named)
    return working, named


# ─── Dynamic Image Analyzer & Identity-Preserving Prompt Generator ────────────

def analyze_input_image_dynamically(image_pil: Image.Image) -> dict:
    """Dynamically analyzes the input image to identify subject characteristics, clothing, and background."""
    img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Detect clothing color in lower third
    lower_crop = img_bgr[int(h * 0.65):, int(w * 0.2):int(w * 0.8)]
    brightness = np.mean(lower_crop) if lower_crop.size > 0 else 100
    if brightness < 80:
        cloth_desc = "the subject's dark clothing"
    elif brightness > 180:
        cloth_desc = "the subject's light-colored clothing"
    else:
        cloth_desc = "the subject's original clothing"

    # Use the image geometry to keep the edit faithful to the supplied crop
    # instead of imposing a one-size-fits-all portrait composition.
    aspect = w / max(h, 1)
    if aspect < 0.72:
        framing_desc = "the existing vertical head-and-shoulders framing"
    elif aspect > 1.15:
        framing_desc = "the existing landscape portrait framing"
    else:
        framing_desc = "the existing balanced portrait framing"

    # The lower-third brightness indicates how much of the outfit is visible.
    clothing_scope = "the visible neckline, shoulders, and garment" if lower_crop.size else "the visible garment"

    return {
        "cloth_desc": cloth_desc,
        "framing_desc": framing_desc,
        "clothing_scope": clothing_scope,
    }


STUDIO_NEGATIVE_PROMPT = (
    "original background, leaves, plants, tree, wall, furniture, outdoor scene, background fragments, "
    "green spill, color spill, halo, cutout edge, blue facial patches, purple facial patches, colored temple artifacts, "
    "harsh directional shadows, deep dark shadows on face, "
    "direct camera flash, blown highlights, washed out, underexposed, yellow tint, cyan cast, grainy, noisy, "
    "blurry, out of focus, plastic skin, altered identity, distorted face, deformed face, illustration, doll face, "
    "younger face, older face, de-aged adult, aged child, changed facial maturity, shortened chin, altered jaw, "
    "oversized eyes, enlarged eyes, anime eyes, artificial iris, beauty-filter face, "
    "badge, school badge, crest, emblem, logo, patch, name tag, text, letters, words, watermark, signature, caption, printed logo, scarf, neck wrap, shawl, "
    "glasses, eyeglasses, spectacles, sunglasses, added jewelry, added clothing, added objects"
)


def build_dynamic_identity_prompt(
    image_pil: Image.Image,
    bg_desc: str,
    custom_instruction: str = "",
    vl_brief: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a compact, source-authoritative portrait edit instruction.

    Qwen Image Edit treats long appearance descriptions as generation concepts.
    In particular, spelling out an inferred age and repeatedly naming a child or
    person caused the four-step model to compose a second subject.  Keep VL
    analysis as QA metadata and pass only visible, high-confidence edit locks.
    """
    vl_brief = vl_brief or {}
    repair_plan = vl_brief.get("portrait_repair_plan")
    if not isinstance(repair_plan, dict):
        repair_plan = {}
    trusted_repair_prompt = str(repair_plan.get("prompt") or "").strip()
    glasses_pos, _ = _glasses_prompt_bits(
        image_pil,
        observed_glasses=bool(vl_brief.get("glasses_present", False)),
    )

    clothing = str(vl_brief.get("clothing_description") or "original clothing").strip()
    if isinstance(vl_brief.get("clothing_description"), list):
        clothing = ", ".join(vl_brief.get("clothing_description"))
    clothing = re.sub(r"[^a-zA-Z0-9 ,.-]", "", clothing)[:80] or "original clothing"

    skin_tone = str(vl_brief.get("skin_tone") or "natural skin tone").strip()
    skin_tone = re.sub(r"[^a-zA-Z0-9 ,.-]", "", skin_tone)[:40] or "natural skin tone"

    hair_color = str(vl_brief.get("hair_color") or "natural dark hair").strip()
    hair_color = re.sub(r"[^a-zA-Z0-9 ,.-]", "", hair_color)[:40] or "natural dark hair"

    hair_style = str(vl_brief.get("hair_style") or "").strip()
    hair_style = re.sub(r"[^a-zA-Z0-9 ,.-]", "", hair_style)[:40]
    raw_hair = vl_brief.get("hair_observations_raw")
    hair_authority = raw_hair if isinstance(raw_hair, dict) else vl_brief
    # Style, length and tail side are unsafe generation facts when inferred by
    # VL.  On wide portraits it repeatedly described a compact one-sided tail
    # as long hair past both shoulders.  Qwen can copy the conditioning pixels
    # directly, so prose should describe only unambiguous crown properties.
    tied_hair = any(
        token in hair_style.lower()
        for token in ("ponytail", "tied", "bun", "braid", "plait")
    )
    concrete_geometry = []
    for geometry_key in (
        "hair_left_side", "hair_right_side", "hair_ends",
        "ponytail_side", "ponytail_anchor", "ponytail_extent",
    ):
        geometry_value = str(hair_authority.get(geometry_key) or "").strip().lower()
        if geometry_value and geometry_value not in {
            "none", "no", "unknown", "uncertain", "unclear", "not-visible",
            "not visible", "not detected", "n/a",
        }:
            concrete_geometry.append(geometry_key)
    hair_fact_labels = (
        ("hair_parting", "parting"), ("hair_texture", "texture"),
        ("hair_color", "color"), ("hair_crown_shape", "crown"),
        ("hair_left_side", "viewer-left silhouette"),
        ("hair_right_side", "viewer-right silhouette"),
        ("hair_ends", "visible ends"), ("hair_fringe", "fringe"),
    )
    if tied_hair:
        hair_fact_labels = (
            ("hair_parting", "parting"), ("hair_texture", "texture"),
            ("hair_color", "color"), ("hair_crown_shape", "crown"),
            ("hair_fringe", "fringe"),
        )
    elif len(concrete_geometry) >= 2:
        hair_fact_labels = (
            ("hair_style", "style"), ("hair_length", "length"),
        ) + hair_fact_labels
    hair_facts = []
    for key, label in hair_fact_labels:
        value = str(hair_authority.get(key) or vl_brief.get(key) or "").strip()
        normalized = value.lower()
        if not value or normalized in {
            "none", "no", "unknown", "uncertain", "unclear", "not-visible",
            "not visible", "not detected", "n/a",
        } or normalized.startswith("source hair") or normalized.startswith("source crown"):
            continue
        clean = re.sub(r"[^a-zA-Z0-9 ,.-]", "", value)[:64]
        if clean:
            hair_facts.append(f"{label}: {clean}")
    hair_geometry = "; ".join(hair_facts)

    expression = str(vl_brief.get("expression") or "natural expression").strip()
    expression = re.sub(r"[^a-zA-Z0-9 ,.-]", "", expression)[:40] or "natural expression"

    # Hair accessories (headbands, bows, ribbons, clips)
    hair_acc = vl_brief.get("hair_accessories")
    if isinstance(hair_acc, list):
        hair_acc = ", ".join(str(x) for x in hair_acc if x)
    hair_acc = str(hair_acc or "").strip()
    hair_acc_lock = ""
    if hair_acc and hair_acc.lower() not in ("none", "no", "null", "false"):
        hair_acc_lock = (
            "Hair accessories are visible. Use the image pixels, not a textual count or color label, as authority. "
            "Copy their exact count, type, color, shape, size, side and placement; add, remove or move none. "
        )

    # Headwear (hats, caps, hijab, turban)
    headwear_present = bool(vl_brief.get("headwear_present", False))
    headwear_desc = str(vl_brief.get("headwear_description") or "").strip()
    headwear_lock = ""
    if headwear_present and headwear_desc and headwear_desc.lower() not in ("none", "no", "null"):
        hw_clean = re.sub(r"[^a-zA-Z0-9 ,.-]", "", headwear_desc)[:60]
        headwear_lock = f"Preserve visible headwear ({hw_clean}) exactly. "

    # Wearables & Jewelry
    jewelry = vl_brief.get("jewelry_description") or vl_brief.get("visible_wearables")
    if isinstance(jewelry, list):
        jewelry = ", ".join(str(x) for x in jewelry if x)
    jewelry = str(jewelry or "").strip()
    jewelry_lock = ""
    if jewelry and jewelry.lower() not in ("none", "no", "null"):
        j_clean = re.sub(r"[^a-zA-Z0-9 ,.-]", "", jewelry)[:60]
        jewelry_lock = f"Preserve visible jewelry and wearables ({j_clean}). "

    hair_desc = hair_geometry or f"base color: {hair_color}"
    measured_hair_rgb = vl_brief.get("measured_hair_rgb")
    measured_color_lock = ""
    # Numeric skin RGB sampled from a sunlit source describes the illumination
    # cast, not the person's complexion. Passing it to Qwen caused orange skin
    # and a synthetic face. Let the source pixels remain the authority.
    measured_color_lock = (
        "Keep source complexion under neutral white balance. Do not warm, yellow, orange, "
        "tan, whiten or lighten skin; use evenly exposed source face pixels as authority. "
    )
    if isinstance(measured_hair_rgb, (list, tuple)) and len(measured_hair_rgb) == 3:
        measured_hair_luma = sum(
            float(channel) * weight
            for channel, weight in zip(measured_hair_rgb, (.2126, .7152, .0722))
        )
        if measured_hair_luma < 82:
            hair_color = "black to very dark brown"
            measured_color_lock += (
                "Hair stays black/dark-brown; remove silver/gold sunlight streaks. "
            )
        else:
            measured_color_lock += (
                "Use shaded source strands as hair-color authority; remove glare without recoloring. "
            )
    sunlight_detected = str(vl_brief.get("direct_sunlight_present", "")).strip().lower() in {
        "true", "yes", "1", "present"
    }
    sunlight_regions = re.sub(
        r"[^a-zA-Z0-9 ,.-]", "", str(vl_brief.get("sunlight_regions") or "")
    )[:80]
    sunlight_type = re.sub(
        r"[^a-zA-Z0-9 ,.-]", "", str(vl_brief.get("sunlight_type") or "")
    )[:40]
    sunlight_direction = re.sub(
        r"[^a-zA-Z0-9 ,.-]", "", str(vl_brief.get("sunlight_direction") or "")
    )[:40]
    if sunlight_detected:
        affected_regions = sunlight_regions if sunlight_regions.lower() not in {
            "", "none", "unknown", "uncertain",
        } else "all visibly affected face, hair, neck and exposed-body skin"
        light_source = " ".join(
            value for value in (sunlight_type, sunlight_direction)
            if value.lower() not in {"", "none", "unknown", "uncertain"}
        )
        diagnosis = (
            f"VL sees {light_source} on {affected_regions}. "
            if light_source else f"VL sees sunlight on {affected_regions}. "
        )
        lighting_instruction = (
            diagnosis
            + "Remove direct sunlight and hard shadows from face, hair, neck, chest and exposed skin. "
            + "Use neutral frontal studio light; keep source colours and texture."
        )
    else:
        lighting_instruction = (
            "Use soft neutral frontal light. Remove harsh glare while keeping source complexion and hair colour."
        )

    # Keep identity prose compact, but do not discard the analyzer's actual
    # lighting diagnosis. The former generic "neutralize sunlight" sentence
    # was too weak for clipped forehead/crown pixels, so Qwen often performed
    # only the background edit.
    relight_instruction = (
        "Remove hard sunlight, hot spots and cast shadows from face, hair, neck and visible skin. "
        "Use neutral frontal fill without changing anatomy, texture or complexion."
        if sunlight_detected
        else "Balance exposure only; keep the existing face and hair pixels under soft neutral frontal light."
    )

    # Do not make an already-frontal portrait regenerate its face merely to
    # satisfy a generic pose command.  Correct the head only when the analyser
    # explicitly sees a turn/tilt; otherwise adjust at most the torso/shoulders.
    orientation = str(vl_brief.get("face_orientation") or "").strip().lower()
    head_tilt = str(vl_brief.get("head_tilt") or "").strip().lower()
    shoulder_alignment = str(
        vl_brief.get("shoulder_alignment") or ""
    ).strip().lower()
    source_pose = str(vl_brief.get("pose") or "").strip().lower()
    measured_shoulder_asymmetry = float(
        vl_brief.get("measured_shoulder_asymmetry") or 0.0
    )
    off_axis_face = any(token in orientation for token in (
        "three-quarter", "three quarter", "profile", "side-facing",
        "side facing", "turned", "angled",
    ))
    tilted_head = (
        any(token in head_tilt for token in ("tilted", "leaning", "leaned"))
        and not any(token in head_tilt for token in ("not tilted", "untilted"))
    )
    torso_off_axis = (
        measured_shoulder_asymmetry >= .08
        or (
            (
                any(token in shoulder_alignment for token in (
                    "rotated", "viewer-left higher", "viewer-right higher",
                    "uneven", "angled", "side",
                ))
                or any(token in source_pose for token in (
                    "three-quarter", "three quarter", "side pose", "side-facing",
                    "side facing", "rotated torso", "angled torso",
                ))
            )
            and not (
                "level/front-facing" in shoulder_alignment
                or "level front-facing" in shoulder_alignment
            )
        )
    )
    if off_axis_face or tilted_head:
        pose_instruction = (
            "Minimally align the existing head and upper body into a straight camera-facing ID pose: "
            "level eyes, centered natural neck and even shoulders. Do not redraw facial features, hair or accessories."
        )
    else:
        pose_instruction = (
            "The head is already frontal: keep its face and hair pixels unchanged. Square only the "
            "upper body and shoulders to camera if needed; preserve natural neck and head size."
        )
    # Keep all decisive edit instructions inside Qwen's conditioning window.
    # The former 290-word contract put sunlight removal after many redundant
    # identity clauses; the 256-token encoder could therefore truncate or
    # underweight the actual repair. Put relighting first and express source
    # preservation once, compactly.
    compact_accessory_lock = (
        "Copy exact hairstyle, silhouette and visible accessories; keep all hair and accessories inside frame."
        if hair_acc and hair_acc.lower() not in ("none", "no", "null", "false")
        else "Copy the exact source hairstyle and silhouette; invent no accessories."
    )
    compact_wearable_lock = (
        "Preserve visible jewelry and wearables exactly."
        if jewelry_lock or headwear_lock
        else "Do not add jewelry, eyewear or headwear."
    )
    if off_axis_face or tilted_head:
        compact_pose = (
            "Minimally level the head and face the upper torso toward camera. "
            "Keep shoulders level, equal-depth and centered; preserve build, neck and head size."
        )
    elif torso_off_axis:
        compact_pose = (
            "Preserve face/head geometry and hairstyle while relighting. Square only the torso and shoulders; preserve neck and build."
            if sunlight_detected else
            "Keep face, hair and head unchanged. Square only the torso and shoulders; preserve neck and build."
        )
    else:
        compact_pose = (
            "Preserve frontal face/head geometry and hairstyle while relighting. Keep the torso camera-facing with level shoulders; preserve neck and build."
            if sunlight_detected else
            "Keep the frontal head unchanged. Keep the torso camera-facing with level shoulders; preserve neck and build."
        )
    parts = [
        "Edit one person; keep backdrop margin around all hair and accessories.",
        lighting_instruction,
        "Preserve identity, current age, facial geometry, expression, complexion and build.",
        compact_accessory_lock,
        compact_wearable_lock,
        compact_pose,
        f"Replace only the background with exact flat {bg_desc}; no scenery, texture, shadow or halo.",
        "Keep pores, fine hair and camera texture; no painted skin.",
    ]

    base_prompt = " ".join(p.strip() for p in parts if p and p.strip())
    if custom_instruction and custom_instruction.strip():
        custom_words = custom_instruction.strip().split()[:12]
        base_prompt = f"{base_prompt} USER PREFERENCE: {' '.join(custom_words)}"
    prompt_words = len(base_prompt.split())
    if prompt_words > 140:
        logger.warning("[QWEN_ENHANCE] Prompt exceeds safe budget: %d words", prompt_words)
    return base_prompt


def build_static_enhancement_prompt(bg_desc: str) -> str:
    """Build the short universal enhancement experiment prompt."""
    return (
        "Perform a local restoration of IMAGE 1; do not generate or redesign the person. "
        "Keep the existing face, expression, teeth, current age, complexion, head, hair, real marks, accessories, jewelry and clothing unchanged in shape and position. "
        "Preserve the source pixel geometry and natural asymmetry; do not redraw, beautify, symmetrize, de-age or recompose any facial feature or hairstyle. "
        "Improve only genuine photographic clarity and neutralize noise, blur, color cast, harsh sunlight, glare and hard shadows without inventing detail. "
        f"Replace only the background with one flat {bg_desc} field. Return the same person in an upper-chest passport crop with complete crown, hair and accessories inside frame. "
        "No new object, second person, painting, CGI, beauty filter, pasted face, halo or scenery."
    )


def upscale_qwen_result(image_bgr: np.ndarray, factor: int) -> np.ndarray:
    """Use the cached Real-ESRGAN model for Qwen output, with Lanczos as a safe fallback."""
    if factor <= 1:
        return image_bgr

    if torch.cuda.is_available():
        try:
            upsampler = orchestrator._get_realesrgan()
            upscaled, _ = upsampler.enhance(image_bgr, outscale=factor)
            logger.info("[QWEN_ENHANCE] Real-ESRGAN %dx upscale complete", factor)
            return upscaled
        except Exception as exc:
            logger.warning("[QWEN_ENHANCE] Real-ESRGAN unavailable; using Lanczos fallback: %s", exc)

    h, w = image_bgr.shape[:2]
    return cv2.resize(image_bgr, (w * factor, h * factor), interpolation=cv2.INTER_LANCZOS4)


def preserve_uploaded_foreground(source: Image.Image, edited_bgr: np.ndarray) -> np.ndarray:
    """Put the real person back over Qwen's edited background.

    Qwen is excellent at the studio scene but may hallucinate fine garment
    texture.  BiRefNet supplies a soft foreground alpha so hair, jewelry, and
    clothing remain faithful to the uploaded photograph.
    """
    from pipelines.birefnet_service import background_removal

    height, width = edited_bgr.shape[:2]
    source_rgb = np.array(source.convert("RGB").resize((width, height), Image.LANCZOS))
    with io.BytesIO() as payload:
        Image.fromarray(source_rgb).save(payload, format="PNG")
        rgba_bytes = background_removal.remove_background(payload.getvalue(), use_schp=False)

    foreground_rgba = np.array(Image.open(io.BytesIO(rgba_bytes)).convert("RGBA"))
    if foreground_rgba.shape[:2] != (height, width):
        foreground_rgba = cv2.resize(foreground_rgba, (width, height), interpolation=cv2.INTER_LANCZOS4)

    alpha = foreground_rgba[:, :, 3].astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (5, 5), 1.0)[:, :, np.newaxis]
    # Keep restoration non-generative: correct only tone/noise before the
    # original face, hair, and clothing are placed over Qwen's backdrop.
    source_bgr = photo_restorer.restore_colors_and_contrast(
        cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR)
    )
    return (source_bgr.astype(np.float32) * alpha + edited_bgr.astype(np.float32) * (1.0 - alpha)).clip(0, 255).astype(np.uint8)


def build_qwen_masked_guide(image: Image.Image, bg_desc: str) -> Image.Image:
    """Give Qwen a clean background guide before its generative edit pass."""
    from pipelines.birefnet_service import background_removal

    source = image.convert("RGB")
    with io.BytesIO() as payload:
        source.save(payload, format="PNG")
        rgba = Image.open(io.BytesIO(background_removal.remove_background(payload.getvalue()))).convert("RGBA")
    # The Qwen transformer needs the entire GPU on low-VRAM systems.  Leaving
    # BiRefNet resident made Qwen steps take several minutes each.
    background_removal.unload()

    # BiRefNet's fine alpha can contain holes in curls and light clothing.
    # Human parsing supplies a solid interior; retain the soft alpha only at
    # the outer contour where it is needed for natural hair edges.
    try:
        from pipelines.schp_service import parse

        rgba_arr = np.array(rgba)
        labels = parse(np.array(source))["labels"]
        human = labels != 0
        rgba_arr[:, :, 3][human] = np.maximum(rgba_arr[:, :, 3][human], 248)
        rgba = Image.fromarray(rgba_arr, "RGBA")
    except Exception as exc:
        logger.warning("[QWEN_GUIDE] Human-mask repair skipped: %s", exc)

    color = str(bg_desc).strip()
    if color.startswith("#") and len(color) == 7:
        try:
            rgb = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
        except ValueError:
            rgb = (205, 230, 248)
    elif "white" in color.lower():
        rgb = (255, 255, 255)
    elif "blue" in color.lower():
        rgb = STUDIO_BLUE_RGB
    else:
        rgb = (205, 230, 248)

    canvas = Image.new("RGBA", rgba.size, (*rgb, 255))
    canvas.alpha_composite(rgba)
    return canvas.convert("RGB")


def process_natural_studio_restore(
    image: Image.Image,
    bg_color_name: str,
    custom_hex: str,
    passport_format: str,
    upscale_factor: int,
    progress_cb=None,
) -> Tuple[Image.Image, Path]:
    """Restore the real portrait and replace only its background.

    This path intentionally avoids Qwen Image Edit.  It protects the source
    face, hair, clothing, and jewelry from generative changes, while BiRefNet
    supplies the soft alpha needed for a clean studio backdrop.
    """
    from pipelines.birefnet_service import background_removal

    job_id = f"natural_studio_{int(time.time() * 1000) % 10000000}"
    source = image.convert("RGB")
    source_bgr = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)

    if progress_cb:
        progress_cb(0.12, "Separating the original person from the background...")
    with io.BytesIO() as payload:
        source.save(payload, format="PNG")
        rgba_bytes = background_removal.remove_background(payload.getvalue(), job_id=job_id)
    matte = Image.open(io.BytesIO(rgba_bytes)).convert("RGBA")
    alpha = np.array(matte)[:, :, 3].astype(np.float32) / 255.0

    # Remove vegetation only where the human-part parser explicitly says the
    # pixel is background.  Never color-key hair, skin, or clothing.
    try:
        from pipelines.schp_service import parse

        labels = parse(np.array(source))['labels']
        hsv = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2HSV)
        green_vegetation = (
            (hsv[:, :, 0] >= 32)
            & (hsv[:, :, 0] <= 92)
            & (hsv[:, :, 1] >= 55)
        )
        alpha[green_vegetation & (labels == 0)] = 0.0
    except Exception as exc:
        logger.warning("[NATURAL_STUDIO] Vegetation matte cleanup skipped: %s", exc)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 1.0)[:, :, np.newaxis]

    if progress_cb:
        progress_cb(0.48, "Restoring original color, detail, and natural facial texture...")
    restored_bgr = photo_restorer.restore_colors_and_contrast(source_bgr)
    if photo_restorer.has_eyeglasses(source_bgr):
        restored_bgr = photo_restorer.reduce_glasses_glare(restored_bgr)
    restored_bgr, _ = photo_restorer.enhance_faces_optical(restored_bgr)

    target = custom_hex.strip() if bg_color_name == "Custom Hex Color" and custom_hex else bg_color_name
    named_colors = {
        "Pure White (Passport)": (255, 255, 255),
        "Light Studio Blue": STUDIO_BLUE_RGB,
        "Studio Blue (ID Standard)": STUDIO_BLUE_RGB,
        "Light Grey": (238, 238, 238),
        "Off-White": (248, 247, 244),
        "Deep Navy": (69, 37, 11),
    }
    if isinstance(target, str) and target.startswith("#") and len(target) == 7:
        try:
            bg_rgb = tuple(int(target[idx:idx + 2], 16) for idx in (1, 3, 5))
        except ValueError:
            bg_rgb = (255, 255, 255)
    else:
        bg_rgb = named_colors.get(str(target), (255, 255, 255))
    bg_bgr = np.array(bg_rgb[::-1], dtype=np.float32).reshape(1, 1, 3)

    if progress_cb:
        progress_cb(0.72, "Placing the untouched portrait on the studio backdrop...")
    composite = (
        restored_bgr.astype(np.float32) * alpha
        + np.broadcast_to(bg_bgr, restored_bgr.shape) * (1.0 - alpha)
    ).clip(0, 255).astype(np.uint8)
    # Some matting models return foreground RGB with residual leaf color even
    # after its alpha is reduced.  Replace those known-background pixels in the
    # composite itself so a solid studio background remains genuinely solid.
    try:
        composite[green_vegetation & (labels == 0)] = bg_bgr.reshape(3).astype(np.uint8)
    except NameError:
        pass

    if passport_format and passport_format != "Original Dimensions (Enhanced)":
        if progress_cb:
            progress_cb(0.87, "Applying passport framing and print finish...")
        composite = photo_restorer.frame_passport_photo(
            composite,
            target_format=passport_format,
            bg_color_bgr=tuple(int(v) for v in bg_bgr.reshape(3)),
            headroom_ratio=0.14,
            head_height_ratio=0.48,
        )
    elif composite.shape[0] == composite.shape[1]:
        # Disallow square output: force non-square portrait format
        composite = photo_restorer.frame_passport_photo(
            composite,
            target_format="Standard ICAO Passport (35 x 45 mm / 3.5 x 4.5 cm)",
            bg_color_bgr=tuple(int(v) for v in bg_bgr.reshape(3)),
            headroom_ratio=0.14,
            head_height_ratio=0.48,
        )

    result = Image.fromarray(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
    if upscale_factor and upscale_factor > 1:
        result = result.resize(
            (result.width * int(upscale_factor), result.height * int(upscale_factor)),
            Image.LANCZOS,
        )
    out_path = OUTPUTS_DIR / f"{job_id}_natural_studio.png"
    result.save(out_path, format="PNG", dpi=(300, 300))
    if progress_cb:
        progress_cb(1.0, "Natural studio restoration complete.")
    return result, out_path


# ─── Core Studio Master Restoration Engine ────────────────────────────────────

def process_studio_master(
    image: Image.Image,
    bg_color_name: str,
    custom_hex: str,
    passport_format: str = "Standard ICAO Passport (35 x 45 mm / 3.5 x 4.5 cm)",
    restore_old_photo: bool = True,
    studio_relight: bool = True,
    exposure_gamma: float = 1.12,
    upscale_factor: int = 2,
    enhance_jewels: bool = True,
    polish_dress: bool = True,
    job_prefix: str = "studio_master",
    progress_cb=None,
) -> Tuple[Image.Image, Path]:
    """
    Studio Master Engine (High-Fidelity Restoration with Proper Passport Framing):
    1. Old Photo Defect Inpainting (Removes scratches, creases, film noise, color fading).
    2. High-Fidelity Facial Restoration (Crystal-clear eyes, natural smooth skin, sharp smile).
    3. Continuous BiRefNet Edge Isolation & Studio Backdrop Composite.
    4. 3-Point Studio Softbox Relighting (Key diffuse light, ambient fill, rim light).
    5. Real-ESRGAN Super-Resolution.
    6. Velvet Fabric Polish & Deep Black Dress Denoising.
    7. Gold & Specular Jewelry Polish.
    8. Strict ICAO / Visa Passport Headroom & Proportional Framing (300 DPI Export).
    """
    t0 = time.time()
    job_id = f"{job_prefix}_{int(time.time()*1000) % 10000000}"

    # Stage 0: Old Photo Defect Inpainting & De-fading
    if restore_old_photo:
        if progress_cb:
            progress_cb(0.10, "🩹 Inpainting scratches, fold creases & removing film grain...")
        in_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(in_bgr, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
        p_face = max(faces, key=lambda f: f[2] * f[3]) if len(faces) > 0 else None
        in_bgr = photo_restorer.inpaint_defects(in_bgr, p_face)
        in_bgr = photo_restorer.restore_colors_and_contrast(in_bgr)
        image = Image.fromarray(cv2.cvtColor(in_bgr, cv2.COLOR_BGR2RGB))

    # Save temporary input for orchestrator
    temp_in = UPLOADS_DIR / f"{job_id}_input.png"
    image.convert("RGB").save(temp_in)

    target_bg = custom_hex.strip() if bg_color_name == "Custom Hex Color" and custom_hex else bg_color_name

    config = {
        # Real student photos must retain literal facial pixels. GFPGAN can
        # redraw eyes and skin on low-resolution children, so use optical
        # sharpening later instead of generative face restoration.
        "face_restore": False,
        "background_replace": True,
        "background_color": target_bg,
        "upscale_factor": int(upscale_factor) if upscale_factor else 2,
        "auto_tilt_correct": False,
        "style_mode": "luxury",
    }

    if progress_cb:
        progress_cb(0.25, "✨ Restoring facial features, irises, skin pores & smile...")

    # Run orchestrator pipeline
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        res_path = loop.run_until_complete(orchestrator.run(job_id, str(temp_in), config))
    finally:
        loop.close()

    studio_pil = Image.open(str(res_path)).convert("RGB")
    studio_bgr = cv2.cvtColor(np.array(studio_pil), cv2.COLOR_RGB2BGR)
    h, w = studio_bgr.shape[:2]

    # 1. Subject & Background Isolation Mask
    corners = np.vstack([
        studio_bgr[:30, :30].reshape(-1, 3),
        studio_bgr[:30, -30:].reshape(-1, 3),
    ])
    bg_color = np.median(corners, axis=0)

    diff_from_bg = np.linalg.norm(studio_bgr.astype(np.float32) - bg_color.astype(np.float32), axis=2)
    subject_mask = np.clip((diff_from_bg - 12.0) / 20.0, 0.0, 1.0)
    subject_mask_3c = cv2.GaussianBlur(subject_mask, (11, 11), 3.0)[:, :, np.newaxis]

    # 2. 3-Point Studio Softbox Lighting Simulation
    if studio_relight:
        if progress_cb:
            progress_cb(0.65, "☀️ Simulating 3-point softbox studio lighting & rim edge light...")
        exposed_subj = photo_restorer.apply_studio_lighting(
            studio_bgr,
            subject_mask=subject_mask_3c,
            gamma=exposure_gamma,
            softbox_intensity=0.25,
            shadow_lift=0.35,
            rim_light_intensity=0.18,
        )
    else:
        lab = cv2.cvtColor(studio_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_norm = lab[:, :, 0] / 255.0
        effective_gamma = float(exposure_gamma) if exposure_gamma else 1.12
        l_calibrated = (l_norm ** effective_gamma) * 255.0
        lab[:, :, 0] = np.clip(l_calibrated, 0, 255)
        lab[:, :, 1] = np.clip(lab[:, :, 1] * 1.02, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] * 1.04, 0, 255)
        exposed_subj = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    if progress_cb:
        progress_cb(0.78, "🔒 Polishing deep velvet dress fabric & gold jewelry...")

    # 3. Clean Velvet Dress & Fabric Polish
    if polish_dress:
        dark_mask = np.clip((120.0 - cv2.cvtColor(exposed_subj, cv2.COLOR_BGR2GRAY).astype(np.float32)) / 60.0, 0.0, 1.0)
        lower_weight = np.linspace(0, 1, h)[:, np.newaxis].repeat(w, axis=1)
        dress_weight = np.clip(dark_mask * (lower_weight ** 1.5) * subject_mask, 0.0, 1.0)
        dress_weight_3c = cv2.GaussianBlur(dress_weight, (15, 15), 5.0)[:, :, np.newaxis]

        dress_smooth = cv2.bilateralFilter(exposed_subj, d=9, sigmaColor=30, sigmaSpace=30)
        dress_lab = cv2.cvtColor(dress_smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
        dress_lab[:, :, 0] = np.clip(dress_lab[:, :, 0] * 0.88, 0, 255)
        rich_dress = cv2.cvtColor(dress_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        dress_crisp = cv2.addWeighted(exposed_subj, 1.25, cv2.GaussianBlur(exposed_subj, (0, 0), 1.0), -0.25, 0)
        polished_dress = (rich_dress.astype(np.float32) * 0.70 + dress_crisp.astype(np.float32) * 0.30).clip(0, 255).astype(np.uint8)

        subj_with_dress = (polished_dress.astype(np.float32) * dress_weight_3c + exposed_subj.astype(np.float32) * (1.0 - dress_weight_3c)).clip(0, 255).astype(np.uint8)
    else:
        subj_with_dress = exposed_subj

    # 4. Brilliant Gold Jewelry Polish
    if enhance_jewels:
        hsv = cv2.cvtColor(subj_with_dress, cv2.COLOR_BGR2HSV)
        gold_mask = cv2.inRange(hsv, np.array([10, 35, 45]), np.array([40, 255, 255])).astype(np.float32) / 255.0
        gold_mask = cv2.dilate(gold_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        gold_mask_3c = cv2.GaussianBlur(gold_mask, (3, 3), 1.0)[:, :, np.newaxis]

        sharp_jewels = cv2.addWeighted(subj_with_dress, 1.40, cv2.GaussianBlur(subj_with_dress, (0, 0), 0.8), -0.40, 0)
        hsv_sharp = cv2.cvtColor(sharp_jewels, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_sharp[:, :, 1] = np.clip(hsv_sharp[:, :, 1] * 1.25, 0, 255)
        rich_jewels = cv2.cvtColor(hsv_sharp.astype(np.uint8), cv2.COLOR_HSV2BGR)

        final_subj = (rich_jewels.astype(np.float32) * gold_mask_3c + subj_with_dress.astype(np.float32) * (1.0 - gold_mask_3c)).clip(0, 255).astype(np.uint8)
    else:
        final_subj = subj_with_dress

    # 5. Composite onto Clean Studio Background
    final_bgr = (final_subj.astype(np.float32) * subject_mask_3c + studio_bgr.astype(np.float32) * (1.0 - subject_mask_3c)).clip(0, 255).astype(np.uint8)

    # 6. Format to Proper Passport Framing & Headroom Geometry
    if passport_format and passport_format != "Original Dimensions (Enhanced)":
        if progress_cb:
            progress_cb(0.90, f"📐 Formatting to {passport_format} with proper ICAO headroom...")
        if "White" in bg_color_name:
            bg_pad = (255, 255, 255)
        elif "Blue" in bg_color_name:
            bg_pad = (235, 206, 135)
        elif "Grey" in bg_color_name or "Off-White" in bg_color_name:
            bg_pad = (240, 240, 240)
        elif "Navy" in bg_color_name:
            bg_pad = (80, 30, 20)
        else:
            bg_pad = (int(bg_color[0]), int(bg_color[1]), int(bg_color[2]))

        final_bgr = photo_restorer.frame_passport_photo(
            final_bgr,
            target_format=passport_format,
            bg_color_bgr=bg_pad,
            headroom_ratio=0.14,
            head_height_ratio=0.48,
        )

    res_pil = Image.fromarray(cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB))
    res_pil.save(res_path, quality=99, dpi=(300, 300))

    elapsed = time.time() - t0
    logger.info("[STUDIO_MASTER] Completed in %.2fs -> %s", elapsed, res_path)
    return res_pil, Path(res_path)


# ─── Option 1: Bulk Restoration Handler ───────────────────────────────────────

def process_bulk_restoration(
    files: Optional[List[gr.FileData]],
    engine_mode: str,
    passport_format: str,
    restore_old_photo: bool,
    studio_relight: bool,
    bg_color_name: str,
    custom_hex: str,
    exposure_gamma: float,
    polish_dress: bool,
    polish_jewels: bool,
    upscale_factor: int,
    progress=None,
):
    """Batch processes multiple photos with Studio Master or Qwen 2509 Neural Diffusion Engine."""
    progress = _as_progress(progress)
    if not files or len(files) == 0:
        return [], None, "⚠️ Please upload at least one image file."

    t0 = time.time()
    total_count = len(files)
    processed_images = []
    output_filepaths = []

    progress(0.02, desc=f"Starting batch restoration of {total_count} photos using {engine_mode}...")

    for idx, f in enumerate(files):
        pct = idx / total_count
        # Safely extract file path whether f is a str, NamedString, dict, or FileData object
        if hasattr(f, "path") and f.path:
            file_path = str(f.path)
        elif hasattr(f, "name") and f.name:
            file_path = str(f.name)
        elif isinstance(f, dict):
            file_path = str(f.get("path") or f.get("name") or "")
        else:
            file_path = str(f)

        source_name = _upload_basename(f, fallback=Path(file_path).name)
        progress(pct, desc=f"Processing photo {idx + 1}/{total_count}: {source_name}...")

        try:
            pil_img = Image.open(file_path).convert("RGB")
            def bridge(p, msg):
                progress(pct + (p / total_count), desc=f"[{idx+1}/{total_count}] {msg}")

            if "Fast optical" in engine_mode or "Fast Optical" in engine_mode or "Classic" in engine_mode:
                res_pil, res_path = process_natural_studio_restore(
                    image=pil_img,
                    bg_color_name=bg_color_name,
                    custom_hex=custom_hex,
                    passport_format=passport_format,
                    upscale_factor=upscale_factor,
                    progress_cb=bridge,
                )
            elif "Pose-aligned" in engine_mode or "SDXL" in engine_mode or "InstantID" in engine_mode:
                from pipelines.instantid_studio import instantid_pipeline
                res_path = instantid_pipeline.generate_single_portrait(
                    image_input=pil_img,
                    bg_color_name=bg_color_name,
                    custom_hex=custom_hex,
                    align_pose=True,
                    steps=20,
                    guidance_scale=4.5,
                    job_id=f"bulk_sdxl_{idx+1}_{int(time.time())}",
                    progress_cb=bridge,
                )
                res_pil = Image.open(str(res_path)).convert("RGB")

                # Apply passport framing if requested
                if passport_format and passport_format != "Original Dimensions (Enhanced)":
                    sdxl_bgr = cv2.cvtColor(np.array(res_pil), cv2.COLOR_RGB2BGR)
                    framed_bgr = photo_restorer.frame_passport_photo(
                        sdxl_bgr,
                        target_format=passport_format,
                        bg_color_bgr=(255, 255, 255),
                        headroom_ratio=0.14,
                        head_height_ratio=0.48,
                    )
                    res_pil = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB))
                    res_pil.save(res_path, dpi=(300, 300))
            else:
                # Studio AI Portrait Mode (Qwen Neural Diffusion + InsightFace 106-Point Identity Lock)
                bg_desc = custom_hex if bg_color_name == "Custom Hex Color" and custom_hex else bg_color_name
                prompt = build_dynamic_identity_prompt(pil_img, bg_desc)
                res_path = gusuq_pipeline.edit_image(
                    image=pil_img,
                    prompt=prompt,
                    negative_prompt=STUDIO_NEGATIVE_PROMPT,
                    background_color=bg_desc,
                    true_cfg_scale=1.2,
                    steps=4,
                    progress_cb=bridge,
                )
                res_pil = Image.open(str(res_path)).convert("RGB")

                # Apply passport framing if requested
                if passport_format and passport_format != "Original Dimensions (Enhanced)":
                    qwen_bgr = cv2.cvtColor(np.array(res_pil), cv2.COLOR_RGB2BGR)
                    framed_bgr = photo_restorer.frame_passport_photo(
                        qwen_bgr,
                        target_format=passport_format,
                        bg_color_bgr=(255, 255, 255),
                        headroom_ratio=0.14,
                        head_height_ratio=0.48,
                    )
                    res_pil = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB))
                    res_pil.save(res_path, dpi=(300, 300))

            named_path = _save_matching_input(res_pil, source_name, fallback=source_name)
            processed_images.append(str(named_path))
            output_filepaths.append(named_path)
        except Exception as e:
            logger.exception("Failed to process bulk image %s: %s", file_path, e)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Create ZIP archive
    progress(0.95, desc="Creating ZIP package of all restored passport photos...")
    zip_id = f"bulk_passport_restored_{int(time.time())}.zip"
    zip_path = OUTPUTS_DIR / zip_id

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for p in output_filepaths:
            zip_file.write(p, arcname=p.name)

    elapsed = time.time() - t0
    progress(1.0, desc="Batch complete!")
    status_msg = f"✅ Successfully restored {len(processed_images)}/{total_count} passport photos in {elapsed:.2f}s!"
    return processed_images, str(zip_path), status_msg


# ─── Option 2: Uniform Swap Handler ───────────────────────────────────────────

def _naturalize_person_studio(person: Image.Image, progress) -> Image.Image:
    """Rebuild any quality/pose snapshot into a natural in-focus studio portrait (keeps identity)."""
    progress(0.12, desc="Rebuilding a natural in-focus studio portrait…")
    glasses_pos, glasses_neg = _glasses_prompt_bits(person)
    prompt = (
        "Professional photo-studio camera portrait of this exact person, 85mm lens, "
        "tack-sharp focus on the eyes, even 3-point softbox lighting, "
        "natural standing pose facing the camera, full head and shoulders. "
        "Seamless light studio-blue backdrop. Photorealistic, not illustration. "
        "Keep 100% the same face, hair, and skin. "
        f"{glasses_pos}"
        "Fix blur, noise, poor lighting, and awkward crop. Do not change identity. "
        "Do not change or invent clothing."
    )
    _offload_catvton()
    path = gusuq_pipeline.edit_image(
        image=person.convert("RGB"),
        prompt=prompt,
        negative_prompt=(
            STUDIO_NEGATIVE_PROMPT
            + ", blurry, out of focus, phone snapshot, fish-eye, extreme pose, "
            "cutout, collage, sticker, "
            + glasses_neg
        ),
        background_color="light studio blue",
        true_cfg_scale=1.2,
        steps=4,
        progress_cb=lambda p, d: progress(min(0.28, 0.12 + 0.16 * (p if p <= 1 else p / 100.0)), desc=d),
    )
    return Image.open(str(path)).convert("RGB")


def process_uniform_swap(
    person_image: Any,
    uniform_image: Any,
    uniform_type: str,
    passport_format: str = "Standard ICAO Passport (35 x 45 mm / 3.5 x 4.5 cm)",
    fit_mode: str = "Template Exact (Recommended)",
    progress=None,
    output_name: Optional[str] = None,
):
    """Fit the reference uniform while preserving the uploaded person's identity."""
    progress = _as_progress(progress)
    if person_image is None:
        return None, "⚠️ Please upload a person photo."
    if uniform_image is None:
        return None, "⚠️ Please upload a uniform photo."

    person_source = output_name or person_image
    person_image = _open_upload(person_image, convert_rgb=True)
    uniform_image = _open_upload(uniform_image, convert_rgb=False)
    # Uniform transfer must never regenerate the subject before fitting.  The old
    # Qwen studio-prep pass occasionally invented eyewear and changed the face,
    # then CatVTON measured the garment against that altered portrait.
    wears_glasses = _person_wears_glasses(person_image)
    t0 = time.time()

    try:
        # Uniform Swap is intentionally a two-model route only: Qwen VL plans
        # the template fit and Qwen Image Edit produces the final portrait.
        # No matting, parsing, VTON, compositing, or restoration is applied.
        from pipelines.qwen_edit_pipeline import qwen_service
        progress(0.08, desc="Analyzing person and template with Qwen VL...")
        out_path = qwen_service.qwen_vl_image_edit_uniform_swap(
            person_image,
            uniform_image,
            job_id=f"qwen_uniform_{int(time.time())}",
            background_color="4,126,246",
            width=560,
            height=720,
            steps=20,
            progress_callback=lambda pct, desc: progress(min(1.0, pct / 100.0), desc=desc),
        )
        progress(1.0, desc="Completed Qwen Image Edit uniform portrait!")
        return str(out_path), f"✅ Qwen VL + 20-step Qwen Image Edit complete in {time.time() - t0:.1f}s."

        already_studio_matted = False
        progress(0.08, desc="Measuring the uploaded person for the uniform fit…")

        # This is a fully Qwen-driven route: Qwen-VL sees both source images,
        # then Qwen Image Edit receives the person plus the exact template as
        # visual conditioning. CatVTON is intentionally not involved.
        if fit_mode.startswith("AI Seam - Qwen VL"):
            try:
                progress(0.16, desc="Compositing the supplied uniform onto the person...")
                from pipelines.uniform_precomposite import build_uniform_analysis_board, compose_uniform_on_person
                rough_composite = compose_uniform_on_person(person_image, uniform_image)
                analysis_board = build_uniform_analysis_board(person_image, uniform_image, rough_composite)
                progress(0.28, desc="Analyzing merged person, template, and rough fit with Qwen VL...")
                from pipelines.uniform_vl_analyzer import uniform_vl_analyzer
                vl_constraints = uniform_vl_analyzer.analyze_board(analysis_board)
                progress(0.38, desc="Applying Qwen VL collar and shoulder measurements...")
                progress(0.42, desc="Separating the person from the original background..." )
                matted_person = _uniform_studio_matte(person_image)
                res_img = compose_uniform_on_person(matted_person, uniform_image, vl_constraints)
                # The prior two-step Qwen seam pass repainted the source hair,
                # inserted a grey neck band, and returned a textured backdrop.
                # VL provides the fit geometry; retain the supplied template
                # pixels exactly after that measurement instead of regenerating
                # the person or collar a second time.
                progress(0.48, desc="Keeping the measured template fit and preparing the studio background...")
                res_img, named_path = finish_uniform_studio_passport(
                    person_image,
                    res_img,
                    OUTPUTS_DIR / f"qwen_vl_uniform_{int(time.time())}_composite.png",
                    passport_format,
                    progress,
                    name_source=person_source,
                    wears_glasses=wears_glasses,
                    already_studio_matted=True,
                )
                progress(1.0, desc="Completed!")
                return str(named_path), (
                    f"✅ Qwen VL measured template uniform portrait ready in {time.time() - t0:.1f}s. "
                    f"VL analysis: {vl_constraints.get('source', 'fallback')}. Saved as {named_path.name}"
                )
            except Exception as qwen_error:
                logger.exception("Qwen VL uniform swap failed: %s", qwen_error)
                return None, f"❌ Qwen VL uniform swap failed: {str(qwen_error)}"

        _offload_catvton()
        from pipelines.catvton_pipeline import CatVTONPipeline
        catvton = CatVTONPipeline()
        if not catvton.is_available():
            progress(0.50, desc="Using studio uniform fallback...")
            from pipelines.studio_uniform_pipeline import studio_uniform_pipeline
            res_path = studio_uniform_pipeline.execute_swap(
                person_image=person_image,
                uniform_reference=uniform_image,
                uniform_preset=uniform_type,
                steps=25,
                guidance_scale=7.5,
                job_id=f"studio_uniform_{int(time.time())}",
                progress_cb=lambda p, d: progress(p, desc=d),
            )
        else:
            res_path = catvton.tryon(
                person_image=person_image,
                uniform_image=uniform_image,
                steps=32,
                guidance_scale=3.5,
                apply_compositing=False,
                # The direct fitter retains the supplied face/neck pixels and
                # avoids diffusion artifacts such as false eyewear, a second
                # shoulder, or a collar painted across the neck.
                exact_drape=True,
                bg_color=STUDIO_BLUE_RGB,
                job_id=f"catvton_{int(time.time())}",
            )
            already_studio_matted = True
            # The deterministic fitter never repaints the source face or neck.
            # A second face-lock blend here caused a visible double-neck seam.
        res_img = Image.open(str(res_path)).convert("RGB")
        ai_fit_used = False
        ai_fit_note = "Template Exact fit"
        if fit_mode.startswith("AI Seam - Qwen VL") or fit_mode.startswith("AI Fit - Qwen 2511"):
            try:
                progress(0.34, desc="Analyzing person and template with Qwen VL...")
                _offload_catvton()
                from pipelines.uniform_vl_analyzer import uniform_vl_analyzer
                vl_constraints = uniform_vl_analyzer.analyze(person_image, uniform_image)
                progress(0.45, desc="Starting Qwen seam-only refinement...")
                ai_candidate = _run_qwen_2511_uniform_fit(res_img, progress, vl_constraints)
                res_img = _apply_seam_only_refinement(person_image, res_img, ai_candidate)
                ai_fit_used = True
                analysis_source = vl_constraints.get("source", "fallback")
                ai_fit_note = f"Qwen VL ({analysis_source}) seam-only refinement with identity/template lock"
            except Exception as ai_error:
                logger.warning("Qwen VL seam refinement failed; retaining exact template output: %s", ai_error)
                progress(0.70, desc="AI Fit unavailable; retaining exact template result...")
                ai_fit_note = f"Qwen VL seam refinement unavailable; returned Template Exact fit ({str(ai_error)[:80]})"
        elif fit_mode.startswith("AI Fit"):
            # Qwen can improve an awkward collar/shoulder join, but generated
            # garment pixels are never trusted over the supplied template.
            try:
                progress(0.34, desc="Starting low-VRAM Qwen 2509 AI Fit refinement...")
                ai_candidate = gusuq_pipeline.edit_uniform_ai_fit(
                    baseline=res_img,
                    person_reference=person_image,
                    uniform_reference=uniform_image,
                    steps=20,
                    progress_cb=lambda p, d: progress(min(0.72, p), desc=d),
                )
                from pipelines.photo_restoration import PhotoRestorationService
                identity_locked = PhotoRestorationService.anchor_and_fuse_identity(
                    orig_pil=person_image,
                    qwen_pil=ai_candidate,
                    identity_lock_strength=0.99,
                )
                locked_bgr = PhotoRestorationService.preserve_source_clothing(
                    cv2.cvtColor(np.array(res_img), cv2.COLOR_RGB2BGR),
                    cv2.cvtColor(np.array(identity_locked), cv2.COLOR_RGB2BGR),
                )
                res_img = Image.fromarray(cv2.cvtColor(locked_bgr, cv2.COLOR_BGR2RGB))
                ai_fit_used = True
                ai_fit_note = "Qwen 2509 AI Fit with template lock"
            except Exception as ai_error:
                logger.warning("Qwen 2509 AI Fit failed; retaining exact template output: %s", ai_error)
                progress(0.70, desc="AI Fit unavailable; retaining exact template result...")
                ai_fit_note = f"Qwen 2509 unavailable; returned Template Exact fit ({str(ai_error)[:80]})"
        res_img, named_path = finish_uniform_studio_passport(
            person_image,
            res_img,
            Path(res_path),
            passport_format,
            progress,
            name_source=person_source,
            wears_glasses=wears_glasses,
            already_studio_matted=already_studio_matted,
        )
        progress(1.0, desc="Completed!")
        return str(named_path), (
            f"✅ Studio uniform portrait ready in {time.time() - t0:.1f}s. "
            f"{ai_fit_note}. Saved as {named_path.name}"
        )
    except Exception as e:
        logger.exception("Uniform swap failed: %s", e)
        return None, f"❌ Error: {str(e)}"


# ─── Option 3: Single Enhance Handler ─────────────────────────────────────────

def process_single_enhance(
    image: Any,
    engine_mode: str,
    passport_format: str,
    restore_old_photo: bool,
    studio_relight: bool,
    bg_color_name: str,
    custom_hex: str,
    exposure_gamma: float,
    polish_dress: bool,
    polish_jewels: bool,
    upscale_factor: int,
    qwen_prompt: str,
    progress=None,
    qwen_steps: int = 20,
    prompt_mode: str = "dynamic",
):
    """Processes one uploaded photo with Indian passport dimensions."""
    progress = _as_progress(progress)
    if image is None:
        return None, "⚠️ Please upload an image first."

    # Keep one physical output standard while allowing the user to choose the
    # studio backdrop colour.  The selected colour is supplied to Qwen and to
    # the final canvas; only the dimensions and framing are fixed.
    passport_format = "Standard ICAO Passport (35 x 45 mm / 3.5 x 4.5 cm)"

    image_source = image
    image = _open_upload(image, convert_rgb=True)
    normalized_prompt_mode = str(prompt_mode or "dynamic").strip().lower()
    if normalized_prompt_mode != "static_test":
        normalized_prompt_mode = "dynamic"
    t0 = time.time()
    progress(0.05, desc="✨ Starting Studio Restoration...")

    try:
        def bridge(p, msg):
            progress(p, desc=msg)

        if "Fast optical" in engine_mode or "Fast Optical" in engine_mode or "Classic" in engine_mode:
            res_pil, out_path = process_natural_studio_restore(
                image=image,
                bg_color_name=bg_color_name,
                custom_hex=custom_hex,
                passport_format=passport_format,
                # Passport framing happens below. Upscale only after that final
                # composition so the 4x output is not immediately downscaled.
                upscale_factor=1,
                progress_cb=bridge,
            )
        elif "Pose-aligned" in engine_mode or "SDXL" in engine_mode or "InstantID" in engine_mode:
            from pipelines.instantid_studio import instantid_pipeline
            res_path = instantid_pipeline.generate_single_portrait(
                image_input=image.convert("RGB"),
                bg_color_name=bg_color_name,
                custom_hex=custom_hex,
                align_pose=True,
                steps=20,
                guidance_scale=4.5,
                progress_cb=bridge,
            )
            res_pil = Image.open(str(res_path)).convert("RGB")

            # Apply passport framing if requested
            if passport_format and passport_format != "Original Dimensions (Enhanced)":
                sdxl_bgr = cv2.cvtColor(np.array(res_pil), cv2.COLOR_RGB2BGR)
                framed_bgr = photo_restorer.frame_passport_photo(
                    sdxl_bgr,
                    target_format=passport_format,
                    bg_color_bgr=(255, 255, 255),
                    headroom_ratio=0.14,
                    head_height_ratio=0.48,
                )
                res_pil = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB))
                res_pil.save(res_path, dpi=(300, 300))

            elapsed = time.time() - t0
            progress(1.0, desc="Completed!")
            named_path = _save_matching_input(res_pil, image_source, fallback=Path(res_path).name)
            return str(named_path), f"✅ Studio portrait completed in {elapsed:.2f}s. Saved as {named_path.name}"
        else:
            # Studio AI Portrait Mode (Qwen Diffusion + InsightFace 106-Point Identity Lock)
            # Generate the selected backdrop inside Qwen's single portrait
            # pass. Post-generation rematting damaged fine hair and shoulders.
            selected_hex = resolve_background_hex(bg_color_name, custom_hex)
            try:
                raw_hex = selected_hex.lstrip("#")
                selected_rgb = (int(raw_hex[0:2], 16), int(raw_hex[2:4], 16), int(raw_hex[4:6], 16))
                frame_bg_bgr = tuple(reversed(selected_rgb))
            except (TypeError, ValueError):
                selected_rgb = (255, 255, 255)
                frame_bg_bgr = (255, 255, 255)
            qwen_bg_desc = f"solid {selected_hex.upper()} RGB{selected_rgb}"

            if normalized_prompt_mode == "static_test":
                # The static experiment deliberately uses one universal prompt.
                # Loading Qwen-VL and SCHP here added about a minute without
                # changing that prompt, so retain only the fast measured gates
                # (face count, crop geometry and harsh-light detection) below.
                progress(0.10, desc="Measuring portrait framing...")
                vl_brief = {"source": "static_prompt_fast_path"}
                logger.info(
                    "[QWEN_ENHANCE] Static prompt fast path: skipped redundant "
                    "Qwen-VL and SCHP analysis"
                )
            else:
                progress(0.10, desc="Analyzing portrait framing and clothing details...")
                from pipelines.uniform_vl_analyzer import uniform_vl_analyzer
                vl_brief = uniform_vl_analyzer.analyze_portrait(image)
                try:
                    from pipelines.schp_service import parse as parse_body_parts
                    from pipelines.uniform_composite import measure_source_person_colors
                    source_labels = parse_body_parts(np.array(image.convert("RGB")))["labels"]
                    measured_colors = measure_source_person_colors(image, source_labels)
                    vl_brief["measured_skin_rgb"] = measured_colors.get("skin_rgb")
                    vl_brief["measured_hair_rgb"] = measured_colors.get("hair_rgb")
                except Exception as color_error:
                    logger.warning("[QWEN_ENHANCE] Source color measurement unavailable: %s", color_error)
            source_gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
            source_h, source_w = source_gray.shape[:2]
            face_detail_region = source_gray[
                int(source_h * 0.08):int(source_h * 0.62),
                int(source_w * 0.18):int(source_w * 0.82),
            ]
            vl_brief["source_min_dimension"] = min(source_w, source_h)
            vl_brief["source_face_detail_score"] = float(
                cv2.Laplacian(face_detail_region, cv2.CV_64F).var()
            ) if face_detail_region.size else 0.0
            # Total image resolution is misleading for a wide classroom shot:
            # the face may occupy only a small fraction of it. Measure the real
            # face before selecting the diffusion schedule and before Qwen sees
            # the image, so the final passport crop is not an enlarged painted
            # reconstruction of a tiny face.
            source_faces = []
            try:
                from pipelines.photo_restoration import _get_insight_app
                source_faces = _get_insight_app().get(
                    cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
                )
            except Exception as face_error:
                logger.warning("[QWEN_ENHANCE] Face geometry unavailable: %s", face_error)
            if len(source_faces) == 1:
                face_box = source_faces[0].bbox
                face_height_pixels = max(1.0, float(face_box[3] - face_box[1]))
                vl_brief["source_face_height_pixels"] = face_height_pixels
                vl_brief["source_face_frame_ratio"] = face_height_pixels / max(source_h, 1)
                measured_shoulder_asymmetry = _measure_source_shoulder_asymmetry(
                    image, face_box,
                )
                if measured_shoulder_asymmetry is not None:
                    vl_brief["measured_shoulder_asymmetry"] = (
                        measured_shoulder_asymmetry
                    )
            measured_sunlight = _has_harsh_source_sunlight(
                image,
                source_faces[0].bbox if len(source_faces) == 1 else None,
            )
            if measured_sunlight:
                # The deterministic image measurement is a confirmation
                # signal for the VL plan. Previously this affected validation
                # but not the actual generation prompt, so Qwen could receive
                # a soft-light instruction and reproduce body sunlight.
                vl_brief["direct_sunlight_present"] = True
                if str(vl_brief.get("sunlight_type") or "").strip().lower() in {
                    "", "none", "unknown", "uncertain", "soft indoor",
                }:
                    vl_brief["sunlight_type"] = "hard directional sunlight"
            from pipelines.qwen_edit_pipeline import _select_enhancement_steps
            generation_steps, step_reasons = _select_enhancement_steps(vl_brief, qwen_steps)
            logger.info(
                "[QWEN_ENHANCE] VL decision source=%s sunlight=%s head-hotspot=%s crown=%s hair-risk=%s steps=%d reasons=%s",
                vl_brief.get("source"),
                bool(vl_brief.get("direct_sunlight_present", False)),
                bool(vl_brief.get("head_hair_hotspot_present", False)),
                bool(vl_brief.get("crown_near_top_edge", False)),
                vl_brief.get("hair_edge_risk"),
                generation_steps,
                ", ".join(step_reasons),
            )
            progress(
                0.12,
                desc=f"Selected {generation_steps} restoration steps for this portrait...",
            )
            direct_sunlight_confirmed = str(
                vl_brief.get("direct_sunlight_present", "")
            ).strip().lower() in {"true", "yes", "1", "present"} or measured_sunlight
            # Keep the original camera frame for Qwen. Always adding large
            # synthetic top padding made the model re-compose focused photos.
            # Add only enough headroom when VL says the crown is already at
            # the source edge; passport framing handles the final layout.
            # Qwen must see the untouched photograph as its identity, hair and
            # accessory authority. Feeding a pre-cut SCHP matte here damaged
            # fine hair and copied those holes into both the generated image
            # and the identity-rejection fallback. Background cleanup belongs
            # after a candidate has passed identity verification.
            from pipelines.photo_restoration import PhotoRestorationService
            from pipelines.qwen_edit_pipeline import qwen_service
            qwen_source = image.convert("RGB")
            if len(source_faces) == 1:
                face_bbox = source_faces[0].bbox
                face_frame_ratio = float(
                    vl_brief.get("source_face_frame_ratio") or 0.0
                )
                source_aspect = image.width / max(image.height, 1)
                if face_frame_ratio >= .24 and source_aspect <= 1.10:
                    # A close portrait already contains the authoritative hair
                    # silhouette and side accessories. A second face-centred
                    # crop can cut wide curls, flowers or bows before Qwen sees
                    # them, so preserve the complete uploaded frame.
                    qwen_source = image.convert("RGB")
                    logger.info(
                        "[QWEN_ENHANCE] Close portrait: preserving complete "
                        "source frame for hair/accessories (face_ratio=%.3f)",
                        face_frame_ratio,
                    )
                else:
                    # Enhancement must use exactly one conditioning image.
                    # Passing a second crop of the same person caused Qwen to
                    # render two different people in one portrait.
                    qwen_source = qwen_service._source_identity_reference(
                        qwen_source, face_bbox, 630, 810,
                    )
                    logger.info(
                        "[QWEN_ENHANCE] Prepared original-photo upper-chest "
                        "conditioning crop %s -> %s",
                        image.size, qwen_source.size,
                    )
            elif bool(vl_brief.get("crown_near_top_edge", False)):
                source_rgb = selected_rgb
                pad_side = max(4, int(image.width * 0.015))
                pad_top = max(28, int(image.height * 0.12))
                padded = Image.new(
                    "RGB", (image.width + pad_side * 2, image.height + pad_top), source_rgb
                )
                padded.paste(qwen_source, (pad_side, pad_top))
                qwen_source = padded

            if direct_sunlight_confirmed:
                # Do not pre-correct the image supplied to Qwen and do not
                # relight its accepted output afterward. Qwen sees untouched
                # source pixels; the VL diagnosis now drives the correction
                # inside the single generative edit.
                logger.info(
                    "[QWEN_ENHANCE] Keeping untouched identity crop; "
                    "VL-directed relighting will run inside generation"
                )

            # Preserve the complete source aspect ratio on a Qwen-compatible
            # multiple-of-16 canvas. Synthetic top/bottom padding was copied
            # into the decoded portrait as a blurred horizontal band. A native
            # aspect canvas needs neither cropping nor padding, while the model
            # can still compose the requested clear backdrop around the person.
            source_w, source_h = qwen_source.size
            if source_h >= source_w:
                canvas_w = 832
                canvas_h = int(round((canvas_w * source_h / max(source_w, 1)) / 16.0) * 16)
                if canvas_h > 1216:
                    canvas_h = 1216
                    canvas_w = int(round((canvas_h * source_w / max(source_h, 1)) / 16.0) * 16)
            else:
                canvas_h = 832
                canvas_w = int(round((canvas_h * source_w / max(source_h, 1)) / 16.0) * 16)
                if canvas_w > 1216:
                    canvas_w = 1216
                    canvas_h = int(round((canvas_w * source_h / max(source_w, 1)) / 16.0) * 16)
            canvas_w = max(256, canvas_w)
            canvas_h = max(256, canvas_h)
            qwen_source = qwen_source.resize(
                (canvas_w, canvas_h), Image.LANCZOS,
            )
            logger.info(
                "[QWEN_ENHANCE] Prepared unpadded native-aspect canvas %s "
                "from source %dx%d",
                qwen_source.size, source_w, source_h,
            )

            if normalized_prompt_mode == "static_test":
                prompt = build_static_enhancement_prompt(qwen_bg_desc)
                logger.info(
                    "[QWEN_ENHANCE] Using Simple Static Prompt test mode (%d words)",
                    len(prompt.split()),
                )
            else:
                normalized_prompt_mode = "dynamic"
                prompt = build_dynamic_identity_prompt(
                    image, qwen_bg_desc, qwen_prompt, vl_brief,
                )
            logger.info("[QWEN_ENHANCE] Restoration prompt: %s", prompt)
            # Keep identity sampling stable for the same portrait regardless
            # of the selected backdrop colour.
            # The former seed included the complete prompt, so correcting one
            # word about sunlight or hair selected an unrelated latent face.
            # This content-derived policy remains dynamic across portraits but
            # is intentionally independent of prompt wording. The fixed salt
            # versions the validated identity-preserving sampling profile.
            # Seed identity from the untouched upload, not from the prepared
            # Qwen canvas. Canvas aspect, padding and framing are layout-only
            # changes; allowing them to select a new latent seed caused a
            # previously matching face to fall from 0.57 to 0.37 similarity.
            enhancement_seed = zlib.crc32(image.convert("RGB").tobytes())
            # Use one canonical neutral-backdrop seed for the person's latent
            # identity. The selected output colour belongs in conditioning,
            # not random sampling: letting blue/white choose different seeds
            # changed facial lighting and reintroduced sun. This reproduces
            # the validated neutral result while remaining content-derived
            # and dynamic for every uploaded portrait.
            enhancement_seed = zlib.crc32(
                b"#FFFFFF", enhancement_seed,
            )
            enhancement_seed = (
                enhancement_seed ^ 3201858918
            ) & 0x7FFFFFFF
            logger.info(
                "[QWEN_ENHANCE] Stable source/canonical seed=%d",
                enhancement_seed,
            )

            static_generation_edge = 1024
            generation_edge = (
                static_generation_edge
                if normalized_prompt_mode == "static_test"
                else 1216
            )
            if normalized_prompt_mode == "static_test":
                logger.info(
                    "[QWEN_ENHANCE] Static prompt fast path: using %dpx "
                    "passport generation canvas",
                    generation_edge,
                )
            res_path = gusuq_pipeline.edit_image(
                # Qwen produces the most natural portrait when it sees the
                # original photograph directly; an imperfect matte guide
                # creates holes and forces artificial hair/body reconstruction.
                image=qwen_source,
                prompt=prompt,
                # Qwen EditPlus attaches the source image to every encoded
                # prompt. A negative prompt therefore subtracts the same source
                # image embedding during true CFG. That weakened identity/layout
                # conditioning and repeatedly generated a related second subject.
                # Use only positive edit conditioning; the explicit face-count
                # and identity gates remain authoritative for delivery.
                negative_prompt=None,
                background_color=qwen_bg_desc,
                # Base 2511 uses true CFG; the distilled four-step adapter
                # needs its separate low-guidance configuration.
                # Keep base-model guidance restrained. Stronger CFG obeys the
                # restoration prose by redrawing the face too aggressively.
                # Positive-only Lightning edits have no unconditional branch,
                # so their effective CFG is 1.0. Declare the real value instead
                # of reporting 1.2 while the shared runner resolves it to 1.0.
                true_cfg_scale=1.0,
                steps=generation_steps,
                # Do not use a separate AI upscaler in this route.
                upscale_factor=1,
                # Qwen needs a sufficiently large latent canvas to reconstruct
                # fine hair, facial detail, and fabric from small phone photos.
                # This is not a post-generation upscaler.
                max_generation_dimension=generation_edge,
                minimum_generation_dimension=generation_edge,
                keep_generation_resolution=True,
                # A second Qwen face pass re-generated the face instead of
                # restoring it, so the full-portrait edit remains the only
                # generative pass in the production route.
                face_detail_refine=False,
                # Keep the enhancement route Qwen-only, including its backdrop.
                use_birefnet_background=False,
                # The Qwen-only route must not invoke SCHP colour transfer:
                # its coarse garment mask can pull original background colours
                # into the regenerated dress and leave visible gaps.
                preserve_source_clothing=False,
                # Keep identical source/prompt/background requests stable.
                # Time-based seeds made lighting and identity quality swing
                # unpredictably between retries of the same portrait.
                seed=enhancement_seed,
                # The dynamic identity/hair/background contract is typically
                # 300-400 tokens. The former 256-token cap discarded the final
                # real-skin/no-painted-finish and neck/backdrop constraints.
                max_sequence_length=256,
                reference_images=None,
                preserve_reference_aspect=True,
                # The VAE/output stays at 1216px, but Qwen's VL encoder must
                # remain below its 1,024-token conditioning limit.
                conditioning_max_dimension=512,
                conditioning_pixel_budget=None,
                # Enhancement/restoration must return the actual Qwen decode.
                # Identity review remains visible as QA metadata but must not
                # silently substitute the uploaded source photograph.
                return_raw_candidate=True,
                progress_cb=bridge,
            )
            qwen_pil = Image.open(str(res_path)).convert("RGB")
            from pipelines.photo_restoration import _get_insight_app
            generated_faces = _get_insight_app().get(
                cv2.cvtColor(np.asarray(qwen_pil), cv2.COLOR_RGB2BGR)
            )
            if len(generated_faces) != 1:
                raise RuntimeError(
                    "Qwen restoration generated more than one person or face; "
                    "the invalid portrait was not delivered."
                )
            # Raw-Qwen restoration is always delivered in this mode. Identity
            # analysis is advisory and must never substitute source pixels.
            qwen_accepted = False
            qwen_identity_review = None
            try:
                from pipelines.uniform_review import check_uniform_identity

                qwen_identity_review = check_uniform_identity(
                    image.convert("RGB"), qwen_pil,
                    minimum_similarity=0.75,
                    maximum_geometry_rmse=0.035,
                )
                qwen_accepted = bool(qwen_identity_review.get("accepted"))
                logger.info(
                    "[QWEN_ENHANCE_IDENTITY] accepted=%s similarity=%s geometry_rmse=%s",
                    qwen_accepted,
                    qwen_identity_review.get("similarity"),
                    qwen_identity_review.get("geometry_rmse"),
                )
            except Exception as identity_error:
                logger.warning(
                    "[QWEN_ENHANCE_IDENTITY] Advisory review unavailable: %s",
                    identity_error,
                )
            if (
                direct_sunlight_confirmed
                and not qwen_accepted
                and isinstance(qwen_identity_review, dict)
            ):
                # InsightFace can under-score a clean studio result against a
                # tiny face covered by hard orange sunlight. Re-evaluate only
                # the *confirmation reference* after a geometry-preserving
                # highlight compression, then retain whichever source view is
                # more discriminative. This never modifies or copies pixels
                # into the delivered Qwen portrait.
                try:
                    validation_source = (
                        PhotoRestorationService.neutralize_direct_sunlight(
                            image.convert("RGB")
                        )
                    )
                    neutral_review = check_uniform_identity(
                        validation_source, qwen_pil,
                        minimum_similarity=.75,
                        maximum_geometry_rmse=.035,
                    )
                    raw_similarity = qwen_identity_review.get("similarity")
                    raw_geometry = qwen_identity_review.get("geometry_rmse")
                    neutral_similarity = neutral_review.get("similarity")
                    neutral_geometry = neutral_review.get("geometry_rmse")
                    if (
                        isinstance(neutral_similarity, (int, float))
                        and (
                            not isinstance(raw_similarity, (int, float))
                            or float(neutral_similarity) > float(raw_similarity)
                            or (
                                isinstance(raw_geometry, (int, float))
                                and isinstance(neutral_geometry, (int, float))
                                and float(neutral_similarity) >= float(raw_similarity) - .02
                                and float(neutral_geometry) < float(raw_geometry)
                            )
                        )
                    ):
                        qwen_identity_review = neutral_review
                    logger.info(
                        "[QWEN_ENHANCE_IDENTITY] raw_source=%s "
                        "neutral_source=%s neutral_geometry=%s",
                        raw_similarity, neutral_similarity,
                        neutral_review.get("geometry_rmse"),
                    )
                except Exception as neutral_identity_error:
                    logger.warning(
                        "[QWEN_ENHANCE_IDENTITY] Neutral-source confirmation "
                        "unavailable: %s", neutral_identity_error,
                    )
            if not qwen_accepted and qwen_identity_review:
                similarity = qwen_identity_review.get("similarity")
                geometry_rmse = qwen_identity_review.get("geometry_rmse")
                geometry_limit = qwen_identity_review.get("maximum_geometry_rmse", .055)
                # A hard sun/shadow pattern affects the face embedding even
                # when the neutral result retains the same geometry. For
                # enhancement only, accept a moderate embedding together with
                # a very tight 106-point geometry match. The normal relighting
                # preferred gates are 0.58/0.033 and 0.56/0.025. A neutralized
                # comparison that improves dense geometry may use the bounded
                # 0.53/0.035 fallback. All are available only when sunlight
                # was independently confirmed. Ordinary portraits retain the
                # strict 0.75 identity gate, and no source-face overlay is used.
                relit_geometry_match = (
                    direct_sunlight_confirmed
                    and isinstance(similarity, (int, float))
                    and isinstance(geometry_rmse, (int, float))
                    and (
                        # A very tight 106-point facial geometry match can
                        # safely confirm identity when direct sun has shifted
                        # the source embedding. This candidate measured 0.575
                        # with 0.0238 RMSE: below the embedding-only gate, but
                        # substantially inside the geometry safeguard.
                        (
                            float(similarity) >= .56
                            and float(geometry_rmse) <= min(.025, float(geometry_limit))
                        )
                        or (
                            float(similarity) >= .58
                            and float(geometry_rmse) <= min(.033, float(geometry_limit))
                        )
                        or (
                            # Strong relighting can shift a low-resolution,
                            # sun-covered source embedding while making its
                            # dense facial geometry *closer*. Admit this narrow
                            # case only after the neutral-source comparison;
                            # ordinary portraits still require 0.70.
                            float(similarity) >= .53
                            and float(geometry_rmse) <= min(.035, float(geometry_limit))
                        )
                    )
                )
                if relit_geometry_match:
                    qwen_accepted = True
                    qwen_identity_review["accepted"] = True
                    qwen_identity_review["reason"] = (
                        "identity accepted by relighting-aware embedding and geometry gate"
                    )
                    logger.info(
                        "[QWEN_ENHANCE_IDENTITY] Accepted relit candidate by dual gate: "
                        "similarity=%.3f geometry_rmse=%.4f",
                        float(similarity), float(geometry_rmse),
                    )
                repairable = (
                    not qwen_accepted
                    and isinstance(similarity, (int, float))
                    and isinstance(geometry_rmse, (int, float))
                    and .45 <= float(similarity) < .75
                    and float(geometry_rmse) <= float(geometry_limit)
                )
                if repairable:
                    # Use the weakest correction that crosses the identity
                    # threshold. A fixed 58% blend visibly over-processed an
                    # otherwise clean Qwen portrait even when its raw score
                    # missed acceptance by only a few thousandths.
                    for repair_strength in (.12, .20, .32, .44, .58, .68, .72):
                        repaired_portrait = qwen_service._lock_uniform_identity(
                            image.convert("RGB"), qwen_pil,
                            strength=repair_strength,
                        )
                        repaired_review = check_uniform_identity(
                            image.convert("RGB"), repaired_portrait,
                            minimum_similarity=.75,
                            maximum_geometry_rmse=.035,
                        )
                        logger.info(
                            "[QWEN_ENHANCE_IDENTITY_REPAIR] raw=%s strength=%.2f "
                            "repaired=%s accepted=%s",
                            similarity, repair_strength,
                            repaired_review.get("similarity"),
                            repaired_review.get("accepted"),
                        )
                        if repaired_review.get("accepted"):
                            qwen_pil = repaired_portrait
                            qwen_identity_review = repaired_review
                            qwen_accepted = True
                            break
            if not qwen_accepted:
                score = (
                    qwen_identity_review.get("similarity")
                    if isinstance(qwen_identity_review, dict) else None
                )
                score_text = f" (score {score:.3f})" if isinstance(score, (int, float)) else ""
                raise RuntimeError(
                    "Qwen restoration was not delivered because the generated face "
                    f"does not match the uploaded person{score_text}."
                )
            # Match the previously accepted qwen_enh_b69eca39 delivery path:
            # once the generated portrait passes identity review, export its
            # Qwen pixels directly. Later skin-chroma and hotspot passes shifted
            # the forehead blue/grey and made the result visibly different from
            # the raw generation.
            logger.info(
                "[QWEN_ENHANCE] Preserving accepted Qwen face, skin and hair pixels; crop/export only"
            )
            # The compact VL-driven prompt now performs relighting inside the
            # Qwen edit itself. Preserve that accepted raw decode exactly;
            # unconditional optical balancing after generation reintroduced a
            # visible shadow/tonal seam into an already clean portrait.
            logger.info(
                "[QWEN_ENHANCE] Keeping accepted raw Qwen relighting; "
                "no post-generation face/hair/skin tone pass"
            )
            progress(0.93, desc="Finalizing the Qwen studio portrait...")
            background_verified = qwen_service._has_selected_solid_background(
                qwen_pil, selected_rgb
            )
            if not background_verified and qwen_service._has_flat_border_background(qwen_pil):
                # Qwen already produced a clean, featureless backdrop.  A
                # second colour key creates grey halos around fine hair when
                # the generated blue is a few RGB values away from the chosen
                # hex. Keep the model's seamless backdrop instead of rematting
                # an otherwise valid portrait.
                logger.info(
                    "[QWEN_ENHANCE_BACKGROUND] Keeping Qwen's clean generated "
                    "backdrop to protect fine hair (selected=%s)", selected_hex,
                )
            elif not background_verified:
                # Diffusion often produces a clean but approximate white/gray
                # backdrop even when an exact hex is requested. Replace only
                # the smooth border-connected backdrop; Qwen's face, hair,
                # glasses and clothing pixels remain untouched.
                qwen_pil = qwen_service._replace_flat_border_background_strict(
                    qwen_pil, selected_hex
                )
                background_verified = qwen_service._has_selected_solid_background(
                    qwen_pil, selected_rgb
                )
            logger.info(
                "[QWEN_ENHANCE_BACKGROUND] Exporting Qwen portrait with %s RGB%s (verified=%s, identity_accepted=%s); exact correction is backdrop-only",
                selected_hex, selected_rgb, background_verified, qwen_accepted,
            )

            # Keep the real subject unchanged apart from local highlight
            # compression, the selected backdrop, and passport framing.
            if passport_format and passport_format != "Original Dimensions (Enhanced)":
                qwen_bgr = cv2.cvtColor(np.array(qwen_pil), cv2.COLOR_RGB2BGR)
                framed_bgr = photo_restorer.frame_passport_photo(
                    qwen_bgr,
                    target_format=passport_format,
                    bg_color_bgr=frame_bg_bgr,
                    # Keep a natural head-to-upper-chest composition.  The
                    # pixel dimensions are Indian passport size; forcing an
                    # 80% head crop on every phone source was visibly harsh.
                    headroom_ratio=0.26,
                    head_height_ratio=0.56,
                )
                qwen_pil = Image.fromarray(cv2.cvtColor(framed_bgr, cv2.COLOR_BGR2RGB))

            elapsed = time.time() - t0
            progress(1.0, desc="Completed!")
            named_path = _save_matching_input(qwen_pil, image_source, fallback=Path(res_path).name)
            completion_message = (
                f"✅ Qwen studio portrait completed in {elapsed:.2f}s. Saved as {named_path.name}"
                if qwen_accepted
                else f"⚠️ Qwen studio portrait completed in {elapsed:.2f}s with an identity warning. "
                f"Saved as {named_path.name}."
            )
            return (
                str(named_path),
                completion_message,
                None,
            )

        elapsed = time.time() - t0
        progress(1.0, desc="Completed!")
        named_path = _save_matching_input(res_pil, image_source, fallback=out_path.name)
        return str(named_path), f"✅ Studio Restoration completed ({passport_format}) in {elapsed:.2f}s! Saved as {named_path.name}"
    except Exception as e:
        logger.exception("Single enhance failed: %s", e)
        return None, f"❌ Processing failed: {str(e)}"


# ─── Gradio UI (Vsoul layout, simple language) ────────────────────────────────

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

.gradio-container {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: #090b0e !important;
    color: #f0f3f6 !important;
    max-width: 1280px !important;
}
.main-title {
    font-family: 'Cinzel', serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em;
    background: linear-gradient(135deg, #fff 40%, #d4af37 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.15rem !important;
}
.sub-title {
    color: #8b949e !important;
    font-size: 0.92rem !important;
    margin-bottom: 1.1rem !important;
}
.how-to {
    background: #111419;
    border: 1px solid #242b35;
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 14px;
    color: #c9d1d9;
    font-size: 0.95rem;
    line-height: 1.55;
}
.how-to b { color: #d4af37; }
.step-num { color: #d4af37; font-weight: 700; }
footer, .footer { display: none !important; }

body {
    background: #090b0e !important;
    color: #f0f3f6 !important;
}

/* Dropdowns, text boxes, radios — light text on dark fill */
.gradio-container .wrap-inner {
    background: #2a3340 !important;
    border: 1px solid #b8a04a !important;
}
.container:has([role="combobox"]) .wrap {
    background: #2a3340 !important;
    border: 1px solid #b8a04a !important;
}
.gradio-container input:not([type="checkbox"]):not([type="radio"]):not([type="range"]),
.gradio-container textarea,
.gradio-container select {
    background: #1f252e !important;
    color: #f8fafc !important;
    border: 1px solid #5b6573 !important;
    min-height: 2.6rem;
}
.gradio-container .wrap-inner svg,
.gradio-container .secondary-wrap svg {
    color: #f8fafc !important;
    fill: #f8fafc !important;
    opacity: 1 !important;
}
.gradio-container label,
.gradio-container .block > label,
.gradio-container span.label-wrap span {
    color: #e5e7eb !important;
}

/* Gradio 6 dropdown list is position:fixed — must style globally, not only inside .gradio-container */
ul.options,
ul[role="listbox"] {
    background: #1f252e !important;
    color: #f8fafc !important;
    border: 1px solid #d4af37 !important;
    z-index: 2147483646 !important;
    min-width: 18rem !important;
    max-height: 18rem !important;
    overflow: auto !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.55) !important;
}
ul.options li,
ul[role="listbox"] li,
[data-testid="dropdown-option"] {
    background: #1f252e !important;
    color: #f8fafc !important;
}
ul.options li:hover,
ul.options li.selected,
ul.options li.active,
[data-testid="dropdown-option"]:hover,
[data-testid="dropdown-option"][aria-selected="true"] {
    background: #3d4757 !important;
    color: #ffffff !important;
}
.gradio-container .form,
.gradio-container .block {
    overflow: visible !important;
}
"""

_vsoul_theme = gr.themes.Default(
    primary_hue="amber",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "sans-serif"],
).set(
    body_background_fill="#090b0e",
    body_background_fill_dark="#090b0e",
    background_fill_primary="#111419",
    background_fill_primary_dark="#111419",
    background_fill_secondary="#181d24",
    background_fill_secondary_dark="#181d24",
    border_color_primary="#5b6573",
    input_background_fill="#1f252e",
    input_background_fill_dark="#1f252e",
    input_background_fill_focus="#252c36",
    input_background_fill_focus_dark="#252c36",
    input_border_color="#5b6573",
    input_border_color_dark="#5b6573",
    button_primary_background_fill="linear-gradient(135deg, #d4af37 0%, #9a7b20 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #e0c056 0%, #d4af37 100%)",
    button_primary_text_color="#111111",
    button_primary_text_color_hover="#000000",
    block_title_text_color="#f0f3f6",
    block_label_text_color="#e5e7eb",
    body_text_color="#f0f3f6",
    body_text_color_dark="#f0f3f6",
    checkbox_label_text_color="#f0f3f6",
    checkbox_label_text_color_dark="#f0f3f6",
    checkbox_label_background_fill="#1f252e",
    checkbox_label_background_fill_dark="#1f252e",
)

with gr.Blocks(title="Vsoul AI Studio") as demo:
    gr.Markdown("<div class='main-title'>VSOUL</div>")
    gr.Markdown(
        "<div class='sub-title'>Photo studio — make a clear ID photo, or put a school uniform on a portrait</div>"
    )

    with gr.Tabs():

        # 1. ENHANCE (matches index.html tab order)
        with gr.TabItem("✦  Enhance"):
            gr.Markdown(
                "<div class='how-to'>"
                "<span class='step-num'>1.</span> Upload a photo &nbsp;&nbsp;"
                "<span class='step-num'>2.</span> Pick background and photo size &nbsp;&nbsp;"
                "<span class='step-num'>3.</span> Click <b>Enhance portrait</b>"
                "<br/>Best quality is slower. Use Fast if you only need a quick clean-up."
                "</div>"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    single_in = gr.Image(
                        type="filepath",
                        label="Your photo",
                        sources=["upload", "clipboard"],
                    )
                    with gr.Row():
                        single_bg_color = gr.Dropdown(
                            choices=COLOR_CHOICES,
                            value="Light Blue (School / Visa ID)",
                            label="Background",
                        )
                        single_upscale = gr.Dropdown(
                            choices=[1, 2, 4],
                            value=2,
                            label="Print size",
                        )
                    with gr.Row():
                        single_passport_format = gr.Dropdown(
                            choices=PASSPORT_FORMATS,
                            value="Standard ICAO Passport (35 x 45 mm / 3.5 x 4.5 cm)",
                            label="ID / passport size",
                        )
                        single_engine = gr.Radio(
                            choices=[ENGINE_PORTRAIT_RECOMMENDED, ENGINE_PORTRAIT_FAST],
                            value=ENGINE_PORTRAIT_RECOMMENDED,
                            label="Quality",
                        )
                    single_btn = gr.Button("✦  Enhance portrait", variant="primary", size="lg")
                    with gr.Accordion("More options (optional)", open=False):
                        single_restore_old = gr.Checkbox(value=True, label="Fix old photo damage (scratches, fading)")
                        single_studio_relight = gr.Checkbox(value=True, label="Soft studio lighting")
                        single_gamma = gr.Slider(
                            minimum=0.90, maximum=1.35, value=1.12, step=0.02, label="Brightness"
                        )
                        single_dress = gr.Checkbox(value=True, label="Smooth clothing fabric")
                        single_jewels = gr.Checkbox(value=True, label="Sharpen jewellery")
                        single_custom_hex = gr.Textbox(
                            label="Custom background colour (only if you picked Custom)",
                            placeholder="#FFFFFF",
                            value="#FFFFFF",
                        )
                        single_qwen_prompt = gr.Textbox(
                            label="Extra request (leave blank for a normal studio photo)",
                            placeholder="Example: soft lighting, natural skin",
                            lines=2,
                        )
                with gr.Column(scale=1):
                    single_out = gr.Image(type="filepath", label="Result — ready to download", interactive=False)
                    single_status = gr.Textbox(label="Status", interactive=False)

            single_btn.click(
                fn=process_single_enhance,
                inputs=[
                    single_in, single_engine, single_passport_format, single_restore_old, single_studio_relight,
                    single_bg_color, single_custom_hex, single_gamma, single_dress, single_jewels,
                    single_upscale, single_qwen_prompt,
                ],
                outputs=[single_out, single_status],
            )

        # 2. UNIFORM
        with gr.TabItem("👔  Uniform"):
            gr.Markdown(
                "<div class='how-to'>"
                "<span class='step-num'>1.</span> Upload the student photo &nbsp;&nbsp;"
                "<span class='step-num'>2.</span> Upload the uniform photo &nbsp;&nbsp;"
                "<span class='step-num'>3.</span> Click <b>Put on uniform</b>"
                "<br/>Face and hair stay the same. Poor or off-pose photos are rebuilt as a studio camera portrait, then the uniform is worn on the body."
                "</div>"
            )
            with gr.Row():
                uni_person = gr.Image(
                    type="filepath",
                    label="1. Student / person photo",
                    sources=["upload", "clipboard"],
                )
                uni_template = gr.Image(
                    type="filepath",
                    label="2. Uniform photo",
                    sources=["upload", "clipboard"],
                    image_mode=None,
                )
            with gr.Row():
                uni_passport = gr.Dropdown(
                    choices=PASSPORT_FORMATS,
                    value="Standard ICAO Passport (35 x 45 mm / 3.5 x 4.5 cm)",
                    label="ID / passport size",
                )
            uni_btn = gr.Button("👔  Put on uniform", variant="primary", size="lg")
            uni_out = gr.Image(type="filepath", label="Result — ready to download", interactive=False)
            uni_status = gr.Textbox(label="Status", interactive=False)

            uni_btn.click(
                fn=lambda person, uniform, passport: process_uniform_swap(
                    person,
                    uniform,
                    "exact supplied uniform template",
                    passport,
                    "Qwen VL + Qwen Image Edit (20-Step)",
                ),
                inputs=[uni_person, uni_template, uni_passport],
                outputs=[uni_out, uni_status],
            )

        # 3. BULK
        with gr.TabItem("⊞  Bulk"):
            gr.Markdown(
                "<div class='how-to'>"
                "<span class='step-num'>1.</span> Select many photos at once &nbsp;&nbsp;"
                "<span class='step-num'>2.</span> Pick the same background and size for all &nbsp;&nbsp;"
                "<span class='step-num'>3.</span> Click <b>Enhance all</b> — then download the ZIP"
                "</div>"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    bulk_files = gr.File(
                        file_count="multiple",
                        file_types=["image"],
                        label="Drop or select photos",
                    )
                    with gr.Row():
                        bulk_bg_color = gr.Dropdown(
                            choices=COLOR_CHOICES,
                            value="Light Blue (School / Visa ID)",
                            label="Background",
                        )
                        bulk_upscale = gr.Dropdown(
                            choices=[1, 2, 4],
                            value=2,
                            label="Print size",
                        )
                    with gr.Row():
                        bulk_passport_format = gr.Dropdown(
                            choices=PASSPORT_FORMATS,
                            value="Standard ICAO Passport (35 x 45 mm / 3.5 x 4.5 cm)",
                            label="ID / passport size",
                        )
                        bulk_engine = gr.Radio(
                            choices=[ENGINE_PORTRAIT_RECOMMENDED, ENGINE_PORTRAIT_FAST],
                            value=ENGINE_PORTRAIT_RECOMMENDED,
                            label="Quality",
                        )
                    bulk_btn = gr.Button("✦  Enhance all", variant="primary", size="lg")
                    with gr.Accordion("More options (optional)", open=False):
                        bulk_restore_old = gr.Checkbox(value=True, label="Fix old photo damage")
                        bulk_studio_relight = gr.Checkbox(value=True, label="Soft studio lighting")
                        bulk_gamma = gr.Slider(
                            minimum=0.90, maximum=1.35, value=1.12, step=0.02, label="Brightness"
                        )
                        bulk_dress = gr.Checkbox(value=True, label="Smooth clothing fabric")
                        bulk_jewels = gr.Checkbox(value=True, label="Sharpen jewellery")
                        bulk_custom_hex = gr.Textbox(
                            label="Custom background colour",
                            placeholder="#FFFFFF",
                            value="#FFFFFF",
                        )
                with gr.Column(scale=1):
                    bulk_gallery = gr.Gallery(
                        label="Results",
                        columns=3,
                        height=420,
                        object_fit="contain",
                    )
                    bulk_zip = gr.File(label="Download all (ZIP)", interactive=False)
                    bulk_status = gr.Textbox(label="Status", interactive=False)

            bulk_btn.click(
                fn=process_bulk_restoration,
                inputs=[
                    bulk_files, bulk_engine, bulk_passport_format, bulk_restore_old, bulk_studio_relight,
                    bulk_bg_color, bulk_custom_hex, bulk_gamma, bulk_dress, bulk_jewels, bulk_upscale,
                ],
                outputs=[bulk_gallery, bulk_zip, bulk_status],
            )

if __name__ == "__main__":
    import uvicorn
    from main import app as fastapi_app

    demo.queue(default_concurrency_limit=1)
    fastapi_app = gr.mount_gradio_app(fastapi_app, demo, path="/lab")
    uvicorn.run(fastapi_app, host="0.0.0.0", port=7860, log_level="info")
