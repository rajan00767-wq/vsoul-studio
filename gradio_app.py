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
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import torch
import gradio as gr

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")

from pipelines.orchestrator import PipelineOrchestrator
from pipelines.gusuq_pipeline import GUSUQPipeline
from pipelines.photo_restoration import photo_restorer
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


def apply_selected_background_matte(image: Image.Image, target_rgb: Tuple[int, int, int]) -> Image.Image:
    """Use an isolated matte worker to replace only the generated backdrop."""
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
    input_path.read_bytes(), job_id="qwen_background", source_image=None, use_schp=False,
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
    "US / India Visa (2 x 2 inch / 51 x 51 mm)",
    "Standard Studio Portrait (3:4 ratio)",
]

ENGINE_PORTRAIT_RECOMMENDED = "✨ Qwen — slow, best output — Recommended"
ENGINE_PORTRAIT_FAST = "⚡ Fast optical restoration"

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
            "Preserve the exact uniform color, fabric pattern, badge, logo, buttons, collar design, and garment shape. "
            "Create a natural collar-to-neck contact with no double clothing layer, no exposed background fringe, "
            f"no added accessories, and no changed identity. Fit constraints from visual analysis: {constraint_text}."
        ),
        negative_prompt=(
            "changed identity, altered face, different hair, glasses, sunglasses, new jewelry, missing jewelry, altered jewelry, extra collar, "
            "double shirt, recolored uniform, missing badge, missing logo, cropped uniform, distorted shoulders, "
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


def _glasses_prompt_bits(image: Image.Image) -> Tuple[str, str]:
    if _person_wears_glasses(image):
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
    "oversized eyes, enlarged eyes, anime eyes, artificial iris, beauty-filter face, "
    "text, letters, words, watermark, signature, caption, printed logo, invented badge, scarf, neck wrap, shawl, "
    "glasses, eyeglasses, spectacles, sunglasses, added jewelry, added clothing, added objects"
)


def build_dynamic_identity_prompt(
    image_pil: Image.Image,
    bg_desc: str,
    custom_instruction: str = "",
    vl_brief: Optional[Dict[str, Any]] = None,
) -> str:
    """Builds a high-end commercial studio portrait prompt for Qwen-Image-Edit."""
    glasses_pos, _ = _glasses_prompt_bits(image_pil)
    crown_near_edge = bool((vl_brief or {}).get("crown_near_top_edge", False))
    direct_sunlight = bool((vl_brief or {}).get("direct_sunlight_present", False))
    head_hair_hotspot = bool((vl_brief or {}).get("head_hair_hotspot_present", False))
    clothing_description = str((vl_brief or {}).get("clothing_description") or "source garment").strip()
    skin_tone = str((vl_brief or {}).get("skin_tone") or "source natural skin tone").strip()
    hair_color = str((vl_brief or {}).get("hair_color") or "source natural hair color").strip()
    face_orientation = str((vl_brief or {}).get("face_orientation") or "source camera orientation").strip()
    expression = str((vl_brief or {}).get("expression") or "source expression").strip()
    jewelry_description = str((vl_brief or {}).get("jewelry_description") or "visible source jewelry").strip()
    # VL is advisory only. Keep its garment phrase short and neutral so it
    # cannot carry a fabricated scene or additional person into Qwen Edit.
    clothing_description = re.sub(r"[^a-zA-Z0-9 ,.-]", "", clothing_description)[:100] or "source garment"
    skin_tone = re.sub(r"[^a-zA-Z0-9 ,.-]", "", skin_tone)[:80] or "source natural skin tone"
    hair_color = re.sub(r"[^a-zA-Z0-9 ,.-]", "", hair_color)[:80] or "source natural hair color"
    face_orientation = re.sub(r"[^a-zA-Z0-9 ,.-]", "", face_orientation)[:80] or "source camera orientation"
    expression = re.sub(r"[^a-zA-Z0-9 ,.-]", "", expression)[:80] or "source expression"
    jewelry_description = re.sub(r"[^a-zA-Z0-9 ,.-]", "", jewelry_description)[:100] or "visible source jewelry"
    crop_note = (
        "Create exactly one centered, head-and-shoulders portrait of the uploaded person only. "
        "Keep the complete hair crown visible with clear blank headroom above it; never crop the top of the hair. "
        "Remove all other people, hands, objects, and scenery from the frame. "
    )
    clothing_lock = (
        f"VL garment check: visible clothing is {clothing_description}. "
        "Keep that exact source clothing unchanged: color, fabric, pattern, neckline, fit, and details. "
    )
    analysis = analyze_input_image_dynamically(image_pil)
    background_lock = (
        f"Completely remove the original background and replace every non-person pixel with one seamless, evenly lit, solid {bg_desc} studio backdrop. "
        "The final image contains exactly two visual regions: the unchanged uploaded person and the solid backdrop. "
        "Do not retain, blend, ghost, or add any original scenery, tree, leaf, branch, wall, furniture, shadow, color spill, texture, pattern, object, or decoration anywhere around or through the hair. "
        "The backdrop must be perfectly clean and uniform from every canvas edge to the natural hair and clothing boundary. "
    )
    headwear_present = bool((vl_brief or {}).get("headwear_present", False))
    headwear_description = str((vl_brief or {}).get("headwear_description") or "the source headwear").strip()
    actual_headwear_terms = ("cap", "hat", "helmet", "scarf", "hijab", "turban", "head covering")
    headwear_present = headwear_present and any(term in headwear_description.lower() for term in actual_headwear_terms)
    headwear_lock = (
        f"A headwear item is present ({headwear_description}). Preserve it exactly: its shape, color, logo, trim, placement, and edges. "
        "Do not remove, replace, invent, crop, or blend it into the hair. Restore only the visible hair strands outside the headwear. "
        if headwear_present
        else "No headwear is visible. Do not add a cap, hat, helmet, scarf, or any head covering. "
    )
    hair_instruction = (
        f"Visual analysis identifies the source hair as {hair_color}. Keep that exact natural hair color, hairstyle, parting, hairline, curls, clips, and visible hair volume. "
        "Preserve every visible hair clip and its placement. Do not bleach, silver, gloss, extend, restyle, or invent hair. "
        "Dark hair must remain naturally dark; never add white, grey, blue-metallic, or overexposed highlights. "
    )
    jewelry_lock = (
        f"Visual analysis identifies {jewelry_description}. Preserve every visible source jewelry item exactly, including earrings, necklace, chain, pendant, bangle, ring, or ornament. "
        "Do not remove, hide, recolor, reshape, move, blur, merge, duplicate, or replace jewelry. "
        "Retain its real material, fine detail, placement, and natural reflections; do not invent additional jewelry. "
    )
    pose_lock = (
        f"Visual analysis identifies {face_orientation} with {expression}. Treat the uploaded camera geometry as a hard constraint: keep the exact head angle, head tilt, eye direction, "
        "shoulder line, torso orientation, body position, subject scale, and camera viewpoint. "
        "Do not turn the head, change the pose, rotate the body, alter the expression, recenter the person, or create a new camera angle. "
        "If the source is a focused front-facing camera portrait, it must remain the same focused front-facing portrait. "
    )
    lighting_instruction = (
        f"Visual analysis identifies the source complexion as {skin_tone}. Use soft, even, neutral studio lighting. Correct visible harsh sunlight and deep shadows without changing that skin tone, the analyzed hair color, dress color, or subject details. "
    )
    portrait_quality = (
        "Preserve the real face, age, expression, and identity exactly. Do not redraw facial anatomy, eyes, eyebrows, nose, or mouth. "
        "Keep real eye size and facial proportions; never create doll-like or enlarged eyes. "
        "Restore natural high-resolution photographic detail and realistic texture on the existing face, neck, body, hair, and dress without changing their shape, color, or design. "
        "Correct compression noise, blur, and uneven exposure only; retain real skin texture, natural hair strands, and original dress weave. "
        + lighting_instruction
        + hair_instruction +
        jewelry_lock +
        pose_lock +
        "Keep the original dress design, lace straps, fabric weave, color, and pose unchanged. Do not erase, blur, or turn the dress into blank white fabric. "
        "Do not add clothing layers, text, logos, accessories, or objects. "
        "Use soft, neutral, diffuse studio lighting with natural skin tone, gentle shadows, no harsh neck shadow, rim light, haze, bloom, white cast, blown highlights, or cinematic grading. "
    )
    if custom_instruction and custom_instruction.strip():
        return (
            custom_instruction.strip() + " "
            + background_lock
            + portrait_quality
            + f"Preserve the person's exact facial identity, expression, skin tone, hair, and {analysis['framing_desc']}. "
            + clothing_lock + headwear_lock + glasses_pos + crop_note
        )

    prompt = (
        "Create a high-quality professional studio portrait photograph. "
        + background_lock
        + portrait_quality
        + f"Preserve the person's exact facial identity, age, expression, skin tone, eyes, hair, and {analysis['framing_desc']}. "
        + f"Keep {analysis['cloth_desc']} photorealistic. {clothing_lock}{headwear_lock} Do not add glasses, new jewelry, clothing, objects, or background elements. "
        f"{glasses_pos}{crop_note}"
    )
    return prompt


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
                    true_cfg_scale=1.0,
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
        true_cfg_scale=1.0,
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
            # Qwen follows a semantic backdrop name more reliably than a hex
            # string. The exact hex remains the framing/padding color below.
            selected_hex = custom_hex.strip() if custom_hex and custom_hex.strip() else "#FFFFFF"
            try:
                raw_hex = selected_hex.lstrip("#")
                frame_bg_bgr = (int(raw_hex[4:6], 16), int(raw_hex[2:4], 16), int(raw_hex[0:2], 16))
            except (TypeError, ValueError):
                frame_bg_bgr = (255, 255, 255)
            # Keep Qwen's lighting neutral. Naming an exact saturated backdrop
            # here caused it to paint that color into dark hair as rim light.
            # The selected color is applied separately after generation.
            bg_desc = bg_color_name
            progress(0.10, desc="Analyzing portrait framing and clothing details...")
            from pipelines.uniform_vl_analyzer import uniform_vl_analyzer
            vl_brief = uniform_vl_analyzer.analyze_portrait(image)
            logger.info(
                "[QWEN_ENHANCE] VL decision source=%s sunlight=%s head-hotspot=%s crown=%s hair-risk=%s",
                vl_brief.get("source"),
                bool(vl_brief.get("direct_sunlight_present", False)),
                bool(vl_brief.get("head_hair_hotspot_present", False)),
                bool(vl_brief.get("crown_near_top_edge", False)),
                vl_brief.get("hair_edge_risk"),
            )
            # Keep the original camera frame for Qwen. Always adding large
            # synthetic top padding made the model re-compose focused photos.
            # Add only enough headroom when VL says the crown is already at
            # the source edge; passport framing handles the final layout.
            qwen_source = image.convert("RGB")
            if bool(vl_brief.get("crown_near_top_edge", False)):
                source_rgb = tuple(reversed(frame_bg_bgr))
                pad_side = max(4, int(image.width * 0.015))
                pad_top = max(28, int(image.height * 0.12))
                padded = Image.new(
                    "RGB", (image.width + pad_side * 2, image.height + pad_top), source_rgb
                )
                padded.paste(qwen_source, (pad_side, pad_top))
                qwen_source = padded
            prompt = build_dynamic_identity_prompt(image, bg_desc, qwen_prompt, vl_brief)

            _, glasses_negative = _glasses_prompt_bits(image)
            from pipelines.qwen_edit_pipeline import qwen_service
            res_path = gusuq_pipeline.edit_image(
                # Qwen produces the most natural portrait when it sees the
                # original photograph directly; an imperfect matte guide
                # creates holes and forces artificial hair/body reconstruction.
                image=qwen_source,
                prompt=prompt,
                # Avoid naming scene objects in the negative conditioning.
                # Qwen Image Edit can otherwise reproduce those concepts.
                negative_prompt=(
                    "extra person, second face, extra hands, object, decoration, pattern, scenery, "
                    "altered identity, altered hair, altered clothing, cropped hair crown, text, watermark, "
                    "blur, plastic skin, distorted face, enlarged eyes, head turn, changed pose, body rotation, "
                    "different camera angle, changed gaze, changed expression, recentered subject, missing jewelry, "
                    "altered jewelry, duplicated jewelry, " + glasses_negative
                ),
                background_color=bg_desc,
                # 1.15 was too weak for Qwen Image Edit to reliably replace
                # busy phone-photo backgrounds.  This remains conservative
                # enough to retain the person's identity after face anchoring.
                # The selected backdrop is applied by the final matte. Keep
                # edit guidance restrained so it restores the photographed
                # person instead of repainting skin and hair colour.
                true_cfg_scale=1.18,
                # The API can request a four-step Qwen test. The Gradio UI
                # keeps the quality default at twenty steps.
                steps=max(1, int(qwen_steps)),
                # Do not use a separate AI upscaler in this route.
                upscale_factor=1,
                # Qwen needs a sufficiently large latent canvas to reconstruct
                # fine hair, facial detail, and fabric from small phone photos.
                # This is not a post-generation upscaler.
                max_generation_dimension=704,
                minimum_generation_dimension=704,
                keep_generation_resolution=True,
                # A second Qwen face pass re-generated the face instead of
                # restoring it, so the full-portrait edit remains the only
                # generative pass in the production route.
                face_detail_refine=False,
                # Qwen creates the portrait; a final BiRefNet-only soft matte
                # removes any generated source-scene fragments at hair edges.
                # SCHP colour transfer remains disabled in the matte service.
                use_birefnet_background=True,
                # The Qwen-only route must not invoke SCHP colour transfer:
                # its coarse garment mask can pull original background colours
                # into the regenerated dress and leave visible gaps.
                preserve_source_clothing=False,
                # Qwen restores the supplied camera view; the prompt locks the
                # source pose and camera geometry. No source overlay follows.
                seed=int(time.time_ns() % (2 ** 32)),
                progress_cb=bridge,
            )
            qwen_pil = Image.open(str(res_path)).convert("RGB")
            candidate_path = None
            # A face-embedding score can remain high even when diffusion gives
            # a child a different expression, facial texture, or hairstyle.
            # Keep every generated portrait as an explicit review candidate;
            # the normal result remains the real uploaded subject with the
            # requested backdrop and passport framing.
            raw_candidate_path = getattr(qwen_service, "last_qwen_candidate_path", None)
            if raw_candidate_path and Path(raw_candidate_path).is_file():
                candidate_pil = Image.open(str(raw_candidate_path)).convert("RGB")
                candidate_pil = apply_selected_background_matte(candidate_pil, tuple(reversed(frame_bg_bgr)))
                if passport_format and passport_format != "Original Dimensions (Enhanced)":
                    candidate_bgr = cv2.cvtColor(np.array(candidate_pil), cv2.COLOR_RGB2BGR)
                    candidate_pil = Image.fromarray(cv2.cvtColor(
                        photo_restorer.frame_passport_photo(
                            candidate_bgr,
                            target_format=passport_format,
                            bg_color_bgr=frame_bg_bgr,
                            headroom_ratio=0.26,
                            head_height_ratio=0.56,
                        ),
                        cv2.COLOR_BGR2RGB,
                    ))
                candidate_path = _save_matching_input(
                    candidate_pil,
                    OUTPUTS_DIR / f"qwen_candidate_{Path(res_path).stem}.png",
                    fallback="qwen_candidate.png",
                )
            # Do not auto-deliver a regenerated identity. Keep the real source
            # subject, then apply only local, non-generative lighting repair.
            progress(0.91, desc="Balancing harsh sunlight on the portrait...")
            qwen_pil = photo_restorer.neutralize_direct_sunlight(image)
            # The isolated matte worker replaces only the original backdrop
            # with the selected school-ID color, then exits to release GPU memory.
            progress(0.93, desc="Applying the selected studio background...")
            qwen_pil = apply_selected_background_matte(
                qwen_pil, tuple(reversed(frame_bg_bgr))
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
            return (
                str(named_path),
                f"✅ Studio portrait completed in {elapsed:.2f}s. Saved as {named_path.name}",
                str(candidate_path) if candidate_path else None,
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
