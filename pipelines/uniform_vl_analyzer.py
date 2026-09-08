"""Short-lived Qwen-VL analysis for constrained uniform fitting."""

from __future__ import annotations

import gc
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = Path("models/qwen2.5-vl-3b-instruct")


def clean_and_repair_json(text: str) -> Dict[str, Any]:
    """Robustly extract and parse JSON from VL responses, handling code fences, trailing commas, and unclosed braces."""
    if not text or not text.strip():
        raise ValueError("Empty VL response")

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidate = match.group(1) if match else None
    if not candidate:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        candidate = match.group(0) if match else text.strip()

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    cleaned = re.sub(r",\s*([\]}])", r"\1", candidate)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    repaired = cleaned.strip()
    if repaired.count('"') % 2 != 0:
        repaired += '"'
    open_braces = repaired.count('{') - repaired.count('}')
    if open_braces > 0:
        repaired += '}' * open_braces

    parsed = json.loads(repaired)
    if isinstance(parsed, dict):
        return parsed
    raise ValueError(f"Parsed object was not a dictionary: {type(parsed)}")


def flatten_vl_dict(data: Any) -> Dict[str, Any]:
    """Recursively flatten nested dictionaries from VL outputs and format list values as clean strings."""
    if not isinstance(data, dict):
        return {}
    flat: Dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            sub_flat = flatten_vl_dict(v)
            for sub_k, sub_v in sub_flat.items():
                if sub_k not in flat or flat[sub_k] in (None, "none", "unknown", False, []):
                    flat[sub_k] = sub_v
        else:
            flat[k] = v

    # Normalize list values that should be strings for prompt builders
    for str_key in (
        "hair_accessories",
        "visible_wearables",
        "headwear_description",
        "clothing_description",
        "clothing_color_pattern",
        "clothing_collar",
        "clothing_fasteners",
        "hair_style",
        "hair_color",
        "hair_parting",
        "hair_length",
        "hair_texture",
        "hair_accessory_details",
        "skin_tone",
        "expression",
        "face_orientation",
        "jewelry_description",
        "shirt_collar",
        "outer_neckline",
        "shirt_color_and_pattern",
        "shirt_primary_color",
        "shirt_secondary_color",
        "shirt_pattern_type",
        "shirt_pattern_scale",
        "outer_primary_color",
        "outer_pattern_type",
        "button_color",
        "outer_garment_color_and_shape",
    ):
        if str_key in flat and isinstance(flat[str_key], list):
            flat[str_key] = ", ".join(str(item) for item in flat[str_key] if item) or "none"

    return flat


def parse_bbox_pixels(bbox: Any, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    """Parse bounding box into (x1, y1, x2, y2) pixel integers, handling normalized (0-1, 0-1000) and absolute pixels."""
    if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        vals = [float(v) for v in bbox]
    except (ValueError, TypeError):
        return None

    if max(vals) <= 1.05:
        c0, c1, c2, c3 = [v * 1000.0 for v in vals]
        is_normalized = True
    else:
        c0, c1, c2, c3 = vals
        is_normalized = (max(vals) <= 1000.0 and (w > 1100 or h > 1100))

    if is_normalized:
        y1_norm, y2_norm = min(c0, c2), max(c0, c2)
        x1_norm, x2_norm = min(c1, c3), max(c1, c3)
        x1 = int(round(x1_norm * w / 1000.0))
        x2 = int(round(x2_norm * w / 1000.0))
        y1 = int(round(y1_norm * h / 1000.0))
        y2 = int(round(y2_norm * h / 1000.0))
    else:
        pA = (min(c0, c2), max(c0, c2))
        pB = (min(c1, c3), max(c1, c3))
        if pB[1] > w and pB[1] <= h * 1.05:
            x1, x2 = int(round(pA[0])), int(round(pA[1]))
            y1, y2 = int(round(pB[0])), int(round(pB[1]))
        elif pA[1] > w and pA[1] <= h * 1.05:
            x1, x2 = int(round(pB[0])), int(round(pB[1]))
            y1, y2 = int(round(pA[0])), int(round(pA[1]))
        else:
            x1, x2 = int(round(pA[0])), int(round(pA[1]))
            y1, y2 = int(round(pB[0])), int(round(pB[1]))

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _rgb_hex(value: np.ndarray) -> str:
    """Format an RGB colour measurement for a prompt without rounding ambiguity."""
    r, g, b = (int(np.clip(channel, 0, 255)) for channel in value[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


def extract_template_visual_evidence(template: Image.Image) -> Dict[str, str]:
    """Measure palette anchors from the prepared, unpadded uniform template."""
    rgb = np.asarray(template.convert("RGB"))
    if rgb.size == 0:
        return {}
    height, width = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    yy, xx = np.indices((height, width))
    non_canvas = lab[:, :, 0] < 242
    outer = non_canvas & (lab[:, :, 0] < 130) & (yy > height * 0.04)
    shirt_area = (xx > width * 0.12) & (xx < width * 0.88) & (yy > height * 0.02)
    shirt = (
        non_canvas
        & shirt_area
        & (lab[:, :, 0] >= 130)
        & (rgb[:, :, 2].astype(np.int16) >= rgb[:, :, 0].astype(np.int16) + 6)
    )
    evidence: Dict[str, str] = {}
    if np.count_nonzero(shirt) >= 100:
        evidence["measured_shirt_base_color"] = _rgb_hex(np.median(rgb[shirt], axis=0))
    if np.count_nonzero(outer) >= 100:
        evidence["measured_outer_garment_color"] = _rgb_hex(np.median(rgb[outer], axis=0))
    if evidence:
        evidence["template_palette_evidence"] = "; ".join(
            f"{key.removeprefix('measured_').replace('_', ' ')} {value}"
            for key, value in evidence.items()
            if key.startswith("measured_")
        )
    return evidence



class UniformVLAnalyzer:
    """Extract garment-fit and portrait constraints without retaining a VL model in VRAM."""

    _PROMPT = """You inspect two images for a formal school uniform fitting system.
Image 1 is the person portrait.
Image 2 is {second_image}.
Inspect both images carefully and return a JSON object with visible facts only:

Person (Image 1):
- hair_style: concise overall hairstyle (e.g. "curly dark hair")
- hair_parting: visible parting and placement (e.g. "center part", "side part", "none visible")
- hair_length: visible length relative to ears and shoulders (e.g. "shoulder length", "below ears")
- hair_texture: visible strand structure (e.g. "loose curls", "waves", "straight", "braids")
- hair_color: natural hair color (e.g. "dark brown", "black")
- hair_accessories: list any visible headbands, bows, ribbons, clips, or "none" (e.g. "red headband with decorative bow")
- hair_accessory_details: each visible accessory's side, color, material, and shape; use "none" when absent
- visible_wearables: visible earrings, bindi, necklaces, glasses, or "none"

Uniform Blueprint (Image 2):
- garment_components: list of visible clothing pieces (e.g. ["shirt", "vest"] or ["shirt", "tie", "blazer"] or ["polo shirt"])
- shirt_collar: exact shirt collar style (e.g. "mandarin band collar at neck base", "folded pointed collar", "button-down pointed collar", "round collar")
- outer_neckline: exact neckline of the outer garment (e.g. "deep V-neck vest", "crew neck", "blazer lapel", "none")
- shirt_color_and_pattern: shirt color and pattern (e.g. "blue and white micro-checkered", "plain white")
- outer_garment_color_and_shape: outer garment color and style (e.g. "navy blue V-neck vest", "dark blazer", "none")
- sleeve_length: sleeve length (e.g. "short sleeves", "long sleeves", "sleeveless", or "unknown" if cropped)
- button_layout: description of visible buttons on shirt and outer garment (e.g. "front placket with 2 visible white buttons")
- badge_present: boolean (true if any school crest, emblem, patch, logo, or text badge is visible on the uniform; false otherwise)
- badge_location: location of the badge (e.g. "left chest of vest", "right breast", or "none")
- badge_bbox: [ymin, xmin, ymax, xmax] coordinates of the badge on a 0-1000 normalized scale, or null if none

Key garment guidelines:
- Distinguish a standing band collar (mandarin collar) from a turtleneck: a standing band collar sits at the base of the neck with an open neck hole.
- Distinguish a V-neck outer garment (e.g. V-neck vest) from a high collar: if the outer layer plunges or opens in front to expose the shirt, outer_neckline is "V-neck" or "open vest".
Return JSON ONLY with keys: garment_components, shirt_collar, outer_neckline, shirt_color_and_pattern, outer_garment_color_and_shape, sleeve_length, button_layout, badge_present, badge_location, badge_bbox, hair_style, hair_parting, hair_length, hair_texture, hair_color, hair_accessories, hair_accessory_details, visible_wearables.
Do not describe or identify the person."""

    _GARMENT_RETRY_PROMPT = """Analyze this uniform template only. Return one JSON object and nothing else.
Required keys: garment_components, shirt_color_and_pattern, outer_garment_color_and_shape,
shirt_collar, outer_neckline, sleeve_length, button_layout, badge_present, badge_location, badge_bbox.
Describe only visible garment facts. Do not describe a person.
Distinguish a standing band collar from a folded pointed collar. Distinguish a V-neck vest from a high collar.
badge_bbox must be [ymin, xmin, ymax, xmax] normalized to 1000, or null if no badge.
Use concise values and complete the JSON object."""

    _HAIR_RETRY_PROMPT = """Analyze this portrait's visible hair and accessories only. Return one JSON object and nothing else.
Required keys: hair_style, hair_parting, hair_length, hair_texture, hair_color, hair_accessories, hair_accessory_details, visible_wearables.
hair_accessories must list visible headbands, bows, ribbons, clips, or "none".
visible_wearables must list visible earrings, bindi, glasses, necklaces, or "none".
Do not describe or identify the person."""

    _BOARD_PROMPT = """This is one three-panel fitting board. Left panel is PERSON,
middle panel is the exact UNIFORM TEMPLATE, and right panel is a ROUGH FIT.
Inspect only the visual geometry needed to fit the exact template onto the person.
Return JSON only with keys: pose, torso_visibility, shoulders_visible,
neck_visibility, template_type, collar, badge_present, risks,
collar_anchor_ratio, uniform_scale, and fit_confidence.
collar_anchor_ratio is a number from 0.88 to 1.05 measured from the top of the
detected face box to the desired top of the collar. uniform_scale is a number
from 0.88 to 1.10. Do not identify, describe, or alter the person."""

    _PORTRAIT_PROMPT = """Inspect this portrait only to prepare a constrained studio enhancement plan.
Return JSON only with keys: source_subject_count, crown_near_top_edge, hair_edge_risk,
direct_sunlight_present, head_hair_hotspot_present, glasses_present, headwear_present, headwear_description,
hair_style, hair_parting, hair_length, hair_texture, hair_color, hair_accessories, hair_accessory_details,
clothing_description, clothing_color_pattern, clothing_collar, clothing_fasteners, garment_visibility, skin_tone,
face_orientation, expression, and jewelry_description.

Guidelines:
source_subject_count is the number of people visibly present in the source image.
crown_near_top_edge is true only when the top of the hair is close to the image edge.
direct_sunlight_present is true for visible sun hot spots or hard subject shadows. It must be true when head_hair_hotspot_present is true.
head_hair_hotspot_present is true when bright glare is visible on forehead or hair.
hair_accessories must list visible headbands, bows, ribbons, clips, or "none".
hair_accessory_details must identify the visible accessory by side, color, and shape without inventing extras.
headwear_present is true only for hats, caps, turbans, hijabs (distinct from hair accessories).
jewelry_description must list visible earrings, necklaces, chains, pendants, bindis, or "none".
clothing_description must be a short generic description of the visible source garment.
clothing_color_pattern, clothing_collar, and clothing_fasteners must describe only visible features.
skin_tone must describe natural complexion (e.g. "fair", "wheatish", "dusky").
Use concise generic descriptions. Do not identify the person."""

    @staticmethod
    def inpaint_badge(
        image: Image.Image,
        badge_bbox: Optional[List[int]] = None,
    ) -> Image.Image:
        """Inpaint school badge/emblem area with surrounding garment fabric using seamless texture blending."""
        if not badge_bbox or len(badge_bbox) != 4:
            return image
        try:
            rgba = image.convert("RGBA")
            arr = np.array(rgba)
            h, w = arr.shape[:2]

            box = parse_bbox_pixels(badge_bbox, w, h)
            if not box:
                return image
            x1, y1, x2, y2 = box

            # Sample fabric to the left of the badge (or right if near left edge)
            if x1 > 40:
                sample_x1, sample_x2 = max(0, x1 - 60), max(0, x1 - 10)
            else:
                sample_x1, sample_x2 = min(w, x2 + 10), min(w, x2 + 60)
            sample_y1, sample_y2 = max(0, y1), min(h, y2)
            fabric_sample = arr[sample_y1:sample_y2, sample_x1:sample_x2, :3]
            if fabric_sample.size == 0:
                mean_color = np.array([30, 40, 55], dtype=np.float32)
                std_color = np.array([6, 6, 8], dtype=np.float32)
            else:
                mean_color = np.mean(fabric_sample, axis=(0, 1))
                std_color = np.std(fabric_sample, axis=(0, 1))

            mask = np.zeros((h, w), dtype=np.uint8)
            center = ((x1 + x2) // 2, min(h - 1, (y1 + y2) // 2 + 12))
            axes = (max(4, (x2 - x1) // 2 + 12), max(4, (y2 - y1) // 2 + 30))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

            noise = np.random.normal(mean_color, std_color, (h, w, 3)).clip(0, 255).astype(np.uint8)
            mask_f = cv2.GaussianBlur(mask, (11, 11), 0).astype(np.float32) / 255.0
            mask_3d = mask_f[:, :, None]

            rgb = arr[:, :, :3].astype(np.float32)
            blended = rgb * (1.0 - mask_3d) + noise.astype(np.float32) * mask_3d
            arr[:, :, :3] = np.clip(blended, 0, 255).astype(np.uint8)

            logger.info("[UniformVL] Seamlessly inpainted badge in template bbox=[%d, %d, %d, %d]", x1, y1, x2, y2)
            return Image.fromarray(arr)
        except Exception as err:
            logger.warning("[UniformVL] Badge inpainting failed: %s", err)
            return image

    def analyze_portrait(self, image: Image.Image) -> Dict[str, Any]:
        """Run short-lived Qwen VL analysis before Qwen Image Edit portrait work in isolated subprocess."""
        result: Dict[str, Any] = {
            "source_subject_count": 1,
            "crown_near_top_edge": False,
            "hair_edge_risk": "low",
            "direct_sunlight_present": False,
            "head_hair_hotspot_present": False,
            "glasses_present": False,
            "headwear_present": False,
            "headwear_description": "none",
            "hair_style": "natural dark hair",
            "hair_parting": "source hair parting",
            "hair_length": "source hair length",
            "hair_texture": "source hair texture",
            "hair_color": "dark brown",
            "hair_accessories": "none",
            "hair_accessory_details": "none",
            "clothing_description": "source clothing",
            "clothing_color_pattern": "source clothing color and pattern",
            "clothing_collar": "source neckline",
            "clothing_fasteners": "source garment details",
            "garment_visibility": "upper garment visible",
            "skin_tone": "source natural skin tone",
            "face_orientation": "source camera orientation",
            "expression": "source expression",
            "jewelry_description": "visible source jewelry",
            "source": "fallback",
        }
        if not MODEL_DIR.is_dir():
            return result
        cache_dir = Path("scratch/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        image_path = cache_dir / f"portrait_vl_{stamp}.png"
        worker_path = cache_dir / f"portrait_vl_{stamp}.py"
        try:
            image.convert("RGB").save(image_path)
            worker_code = f'''import gc
import json
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

model_dir = Path({str(MODEL_DIR.resolve())!r})
image_path = Path({str(image_path.resolve())!r})
prompt = {self._PORTRAIT_PROMPT!r}
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    str(model_dir), quantization_config=quantization,
    device_map={{"": 0}} if torch.cuda.is_available() else "cpu",
    low_cpu_mem_usage=True, local_files_only=True,
).eval()
portrait = Image.open(image_path).convert("RGB")
messages = [{{"role": "user", "content": [{{"type": "image", "image": portrait}}, {{"type": "text", "text": prompt}}]}}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[portrait], padding=True, return_tensors="pt").to(model.device)
with torch.inference_mode():
    generated = model.generate(**inputs, max_new_tokens=384, do_sample=False)
response = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
print(response)
del generated, inputs, portrait, model, processor
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
'''
            worker_path.write_text(worker_code, encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(worker_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr[-1200:] or "Qwen VL worker failed")
            response = completed.stdout.strip()
            parsed = flatten_vl_dict(clean_and_repair_json(response))
            if isinstance(parsed, dict):
                result.update(parsed)
                # A forehead or hair hotspot is direct-light evidence even if
                # the short VL response forgot to set the broader flag.
                if bool(result.get("head_hair_hotspot_present", False)):
                    result["direct_sunlight_present"] = True
                result["source"] = "qwen2.5-vl"
                logger.info(
                    "[PortraitVL] source=%s hair_acc=%s headwear=%s clothing=%s skin=%s",
                    result["source"],
                    result.get("hair_accessories"),
                    result.get("headwear_description"),
                    str(result.get("clothing_description", "source clothing"))[:80],
                    result.get("skin_tone"),
                )
        except Exception as exc:
            logger.warning("[PortraitVL] Analysis fallback: %s", exc)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            image_path.unlink(missing_ok=True)
            worker_path.unlink(missing_ok=True)
        return result

    @staticmethod
    def fallback(person: Image.Image, template: Image.Image) -> Dict[str, Any]:
        width, height = person.size
        result = {
            "pose": "front portrait",
            "torso_visibility": "limited" if height < width * 1.2 else "visible",
            "shoulders_visible": True,
            "head_to_waist_visible": False,
            "template_type": "product cutout",
            "collar": "unknown",
            "badge_present": False,
            "badge_location": "none",
            "badge_bbox": None,
            "garment_components": ["shirt", "vest"],
            "shirt_color_and_pattern": "checkered shirt",
            "outer_garment_color_and_shape": "navy blue vest",
            "sleeve_length": "unknown",
            "button_layout": "front placket buttons",
            "shirt_collar": "mandarin band collar at neck base",
            "outer_neckline": "V-neck vest",
            "hair_style": "natural dark hair",
            "hair_color": "natural dark hair",
            "hair_accessories": "none",
            "visible_wearables": "none",
            "garment_risks": ["Preserve every visible template layer."],
            "risks": ["Use deterministic face lock and seam-only repair."],
            "source": "geometry fallback",
        }
        result.update(extract_template_visual_evidence(template))
        return result

    def analyze(
        self, person: Image.Image, template: Image.Image, second_image: str = "the required uniform template"
    ) -> Dict[str, Any]:
        """Run uniform and person VL analysis in an isolated subprocess to reclaim all VRAM upon completion."""
        result = self.fallback(person, template)
        if not MODEL_DIR.is_dir():
            return result

        cache_dir = Path("scratch/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        person_path = cache_dir / f"uniform_p_{stamp}.png"
        template_path = cache_dir / f"uniform_t_{stamp}.png"
        worker_path = cache_dir / f"uniform_vl_{stamp}.py"
        try:
            person.convert("RGB").save(person_path)
            template.convert("RGB").save(template_path)
            p_prompt = (
                "Inspect this portrait only. Return JSON with keys: hair_style, hair_parting, hair_length, hair_texture, "
                "hair_color, hair_accessories, hair_accessory_details, visible_wearables, clothing_description, "
                "clothing_color_pattern, clothing_collar, clothing_fasteners, skin_tone, face_orientation, expression.\n"
                "hair_accessories must list visible headbands, bows, ribbons, clips, or 'none'.\n"
                "hair_accessory_details must state side, color, and shape for each visible accessory.\n"
                "Describe hair and clothing only from visible pixels; do not infer or invent missing details.\n"
                "visible_wearables must list visible earrings, bindi, glasses, necklaces, or 'none'.\n"
                "Return JSON only:"
            )
            u_prompt = (
                "Inspect this uniform template image only. Return JSON with keys: garment_components, "
                "shirt_color_and_pattern, shirt_primary_color, shirt_secondary_color, shirt_pattern_type, "
                "shirt_pattern_scale, outer_garment_color_and_shape, outer_primary_color, outer_pattern_type, "
                "shirt_collar, outer_neckline, sleeve_length, button_layout, button_color, badge_present, "
                "badge_location, badge_bbox.\n"
                "Rules:\n"
                "- Inspect literal template pixels only. Do not describe a generic school uniform or infer unseen fabric.\n"
                "- Name each visible layer's exact colors. Classify the shirt pattern as check, stripe, plaid, print, or solid and its scale as micro, fine, medium, or large.\n"
                "- State whether the outer garment is solid, textured, checked, striped, or patterned.\n"
                "- Distinguish a standing band collar from a folded pointed collar.\n"
                "- Distinguish a V-neck outer vest from a high collar.\n"
                "- badge_present: true if an emblem, crest, logo patch, or school badge is visible.\n"
                "- badge_bbox: [ymin, xmin, ymax, xmax] or [xmin, ymin, xmax, ymax] coordinates on this image, or null if no badge.\n"
                "Return JSON only:"
            )

            worker_code = f'''import gc
import json
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

model_dir = Path({str(MODEL_DIR.resolve())!r})
person_path = Path({str(person_path.resolve())!r})
template_path = Path({str(template_path.resolve())!r})
p_prompt = {p_prompt!r}
u_prompt = {u_prompt!r}

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    str(model_dir), quantization_config=quantization,
    device_map={{"": 0}} if torch.cuda.is_available() else "cpu",
    low_cpu_mem_usage=True, local_files_only=True,
).eval()

person_img = Image.open(person_path).convert("RGB")
template_img = Image.open(template_path).convert("RGB")

msg_p = [{{"role": "user", "content": [{{"type": "image", "image": person_img}}, {{"type": "text", "text": p_prompt}}]}}]
txt_p = processor.apply_chat_template(msg_p, tokenize=False, add_generation_prompt=True)
inp_p = processor(text=[txt_p], images=[person_img], padding=True, return_tensors="pt").to(model.device)
with torch.inference_mode():
    gen_p = model.generate(**inp_p, max_new_tokens=256, do_sample=False)
out_p = processor.batch_decode(gen_p[:, inp_p.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

msg_u = [{{"role": "user", "content": [{{"type": "image", "image": template_img}}, {{"type": "text", "text": u_prompt}}]}}]
txt_u = processor.apply_chat_template(msg_u, tokenize=False, add_generation_prompt=True)
inp_u = processor(text=[txt_u], images=[template_img], padding=True, return_tensors="pt").to(model.device)
with torch.inference_mode():
    gen_u = model.generate(**inp_u, max_new_tokens=256, do_sample=False)
out_u = processor.batch_decode(gen_u[:, inp_u.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

combined = {{"person_raw": out_p, "uniform_raw": out_u}}
print("---COMBINED_START---")
print(json.dumps(combined))
print("---COMBINED_END---")

del gen_p, gen_u, inp_p, inp_u, person_img, template_img, model, processor
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
'''
            worker_path.write_text(worker_code, encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(worker_path)],
                capture_output=True,
                text=True,
                timeout=150,
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr[-1200:] or "Uniform VL worker failed")

            response = completed.stdout.strip()
            match = re.search(r"---COMBINED_START---\s*(\{.*?\})\s*---COMBINED_END---", response, flags=re.DOTALL)
            if match:
                combined_dict = json.loads(match.group(1))
                p_dict = flatten_vl_dict(clean_and_repair_json(combined_dict.get("person_raw", "")))
                u_dict = flatten_vl_dict(clean_and_repair_json(combined_dict.get("uniform_raw", "")))
                result.update(p_dict)
                result.update(u_dict)
                result["source"] = "qwen2.5-vl"
            else:
                parsed = flatten_vl_dict(clean_and_repair_json(response))
                if isinstance(parsed, dict):
                    result.update(parsed)
                    result["source"] = "qwen2.5-vl"

            # Keep deterministic palette evidence even if VL uses an
            # incomplete colour name. Both refer to this prepared template.
            result.update(extract_template_visual_evidence(template))
            logger.info(
                "[UniformVL] Analysis succeeded: components=%s shirt=%s pattern=%s/%s outer=%s palette=%s badge=%s bbox=%s hair_acc=%s",
                result.get("garment_components"),
                result.get("shirt_primary_color") or result.get("shirt_color_and_pattern"),
                result.get("shirt_pattern_type"),
                result.get("shirt_pattern_scale"),
                result.get("outer_garment_color_and_shape"),
                result.get("template_palette_evidence"),
                result.get("badge_present"),
                result.get("badge_bbox"),
                result.get("hair_accessories"),
            )
        except Exception as exc:
            logger.warning("[UniformVL] Primary subprocess analysis failed: %s; using geometry fallback", exc)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            person_path.unlink(missing_ok=True)
            template_path.unlink(missing_ok=True)
            worker_path.unlink(missing_ok=True)

        return result

    def analyze_board(self, board: Image.Image) -> Dict[str, Any]:
        """Analyze the combined person/template/rough-fit board in one VL image."""
        result = self.fallback(board, board)
        result.update({"collar_anchor_ratio": 0.96, "uniform_scale": 1.0, "fit_confidence": "limited"})
        if not MODEL_DIR.is_dir():
            return result

        cache_dir = Path("scratch/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        board_path = cache_dir / f"board_vl_{stamp}.png"
        worker_path = cache_dir / f"board_vl_{stamp}.py"
        try:
            board.convert("RGB").save(board_path)
            worker_code = f'''import gc
import json
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

model_dir = Path({str(MODEL_DIR.resolve())!r})
board_path = Path({str(board_path.resolve())!r})
prompt = {self._BOARD_PROMPT!r}

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    str(model_dir), quantization_config=quantization,
    device_map={{"": 0}} if torch.cuda.is_available() else "cpu",
    low_cpu_mem_usage=True, local_files_only=True,
).eval()

board_img = Image.open(board_path).convert("RGB")
messages = [{{"role": "user", "content": [{{"type": "image", "image": board_img}}, {{"type": "text", "text": prompt}}]}}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[board_img], padding=True, return_tensors="pt").to(model.device)
with torch.inference_mode():
    generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
response = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
print(response)

del generated, inputs, board_img, model, processor
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
'''
            worker_path.write_text(worker_code, encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(worker_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if completed.returncode == 0:
                parsed = clean_and_repair_json(completed.stdout.strip())
                if isinstance(parsed, dict):
                    result.update(parsed)
                    result["source"] = "qwen2.5-vl-board"
        except Exception as exc:
            logger.warning("[UniformVL] Board analysis fallback: %s", exc)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            board_path.unlink(missing_ok=True)
            worker_path.unlink(missing_ok=True)

        return result


uniform_vl_analyzer = UniformVLAnalyzer()
