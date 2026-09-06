"""Short-lived Qwen-VL analysis for constrained uniform fitting."""

from __future__ import annotations

import gc
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image

from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = Path("models/qwen2.5-vl-3b-instruct")


class UniformVLAnalyzer:
    """Extract garment-fit constraints without retaining a VL model in VRAM."""

    _PROMPT = """You inspect two images for a school-uniform fitting system.
Image 1 is the person. Image 2 is {second_image}.
Inspect image 1 only for the visible hairstyle, natural hair color, and hair accessories.
Inspect image 2 carefully as the exact garment blueprint. Identify its visible
layers, shirt color and pattern, outer garment color and shape, the shirt collar and
the outer-garment neckline separately, sleeve length, button layout, and school-badge
location. Do not turn the template into a generic school uniform and do not infer
details that are not visible.
Return JSON only with keys: pose, torso_visibility, shoulders_visible,
head_to_waist_visible, template_type, collar, badge_present, garment_components,
shirt_color_and_pattern, outer_garment_color_and_shape, sleeve_length,
button_layout, badge_location, shirt_collar, outer_neckline, hair_style,
hair_color, hair_accessories, and garment_risks.
Use short string or boolean values. Never describe or identify the person."""

    _GARMENT_RETRY_PROMPT = """Analyze this uniform template only. Return one JSON object and nothing else.
Required keys: garment_components, shirt_color_and_pattern, outer_garment_color_and_shape,
shirt_collar, outer_neckline, sleeve_length, button_layout, badge_location.
Describe only visible garment facts. Do not describe a person."""

    _HAIR_RETRY_PROMPT = """Analyze this portrait's visible hair only. Return one JSON object and nothing else.
Required keys: hair_style, hair_color, hair_accessories.
hair_accessories must list only visible clips, bands, bows, jewelry, or headwear and their placement; use 'none' when absent.
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

    _PORTRAIT_PROMPT = """Inspect this portrait only to prepare a constrained image-edit plan.
Return JSON only with keys: source_subject_count, crown_near_top_edge, hair_edge_risk,
    direct_sunlight_present, head_hair_hotspot_present, glasses_present, headwear_present, headwear_description, clothing_description, garment_visibility, skin_tone, hair_color, face_orientation, expression, and jewelry_description.
source_subject_count is the number of people visibly present in the source image. Do not infer people.
crown_near_top_edge is true only when the top of the hair is close to the image edge.
    direct_sunlight_present is true for visible sun hot spots, hard directional subject shadows, or strong outdoor color spill anywhere on the person.
    head_hair_hotspot_present is true when bright sunlight, a white glare, or an overexposed highlight is visible on the forehead, scalp, hair, or hair clips, even if the rest of the person has soft lighting.
clothing_description must be a short generic description of the visible source garment only.
skin_tone must be a short visible colour description of the person's natural complexion. hair_color must be a short visible colour description of the natural hair, excluding sunlight glare and accessories.
face_orientation and expression must describe only the visible camera pose and expression. jewelry_description must list only visible earrings, necklaces, chains, pendants, bindis, or ornaments; use 'none' when absent.
Set headwear_present true for any cap, hat, helmet, headscarf, school cap, or other item worn on the head.
When headwear_present is true, describe only its generic visible type, color, and placement.
Use concise generic descriptions. Do not identify the person."""

    def analyze_portrait(self, image: Image.Image) -> Dict[str, Any]:
        """Run short-lived Qwen VL analysis before Qwen Image Edit portrait work."""
        result: Dict[str, Any] = {
            "source_subject_count": 1,
            "crown_near_top_edge": False,
            "hair_edge_risk": "high",
            "direct_sunlight_present": False,
            "head_hair_hotspot_present": False,
            "glasses_present": False,
            "headwear_present": False,
            "headwear_description": "none",
            "clothing_description": "source clothing",
            "garment_visibility": "upper garment visible",
            "skin_tone": "source natural skin tone",
            "hair_color": "source natural hair color",
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
            # Qwen VL and Qwen Image Edit cannot coexist on this 8 GB GPU.
            # Run visual planning in a child process so Windows releases all
            # of VL's CUDA allocations before the prompt encoder starts.
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
    generated = model.generate(**inputs, max_new_tokens=180, do_sample=False)
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
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr[-1200:] or "Qwen VL worker failed")
            response = completed.stdout.strip()
            match = re.search(r"\{.*\}", response, flags=re.DOTALL)
            parsed = json.loads(match.group(0) if match else response)
            if isinstance(parsed, dict):
                result.update(parsed)
                result["source"] = "qwen2.5-vl"
                logger.info(
                    "[PortraitVL] source=%s sunlight=%s head_hotspot=%s crown_near_top=%s glasses=%s headwear=%s clothing=%s",
                    result["source"],
                    bool(result.get("direct_sunlight_present", False)),
                    bool(result.get("head_hair_hotspot_present", False)),
                    bool(result.get("crown_near_top_edge", False)),
                    bool(result.get("glasses_present", False)),
                    bool(result.get("headwear_present", False)),
                    str(result.get("clothing_description", "source clothing"))[:120],
                )
        except Exception as exc:
            logger.warning("[PortraitVL] Analysis fallback: %s", exc)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            image_path.unlink(missing_ok=True)
            worker_path.unlink(missing_ok=True)
        if result["source"] == "fallback":
            logger.warning(
                "[PortraitVL] Using fallback analysis; sunlight correction decision is unavailable."
            )
        return result

    @staticmethod
    def fallback(person: Image.Image, template: Image.Image) -> Dict[str, Any]:
        width, height = person.size
        return {
            "pose": "front portrait",
            "torso_visibility": "limited" if height < width * 1.2 else "visible",
            "shoulders_visible": True,
            "head_to_waist_visible": False,
            "template_type": "product cutout",
            "collar": "unknown",
            "badge_present": True,
            "garment_components": "shirt and outer uniform garment",
            "shirt_color_and_pattern": "unknown",
            "outer_garment_color_and_shape": "unknown",
            "sleeve_length": "unknown",
            "button_layout": "unknown",
            "badge_location": "unknown",
            "shirt_collar": "unknown",
            "outer_neckline": "unknown",
            "hair_style": "source hairstyle",
            "hair_color": "natural dark hair",
            "hair_accessories": "visible source accessories",
            "garment_risks": ["Preserve every visible template layer."],
            "risks": ["Use deterministic face lock and seam-only repair."],
            "source": "geometry fallback",
        }

    def analyze(
        self, person: Image.Image, template: Image.Image, second_image: str = "the required uniform template"
    ) -> Dict[str, Any]:
        result = self.fallback(person, template)
        if not MODEL_DIR.is_dir():
            return result

        model = None
        try:
            from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            processor = AutoProcessor.from_pretrained(str(MODEL_DIR), local_files_only=True)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(MODEL_DIR),
                quantization_config=quantization,
                device_map={"": 0} if torch.cuda.is_available() else "cpu",
                low_cpu_mem_usage=True,
                local_files_only=True,
            ).eval()
            def parse_response(response: str) -> Dict[str, Any]:
                match = re.search(r"\{.*\}", response, flags=re.DOTALL)
                parsed = json.loads(match.group(0) if match else response)
                if not isinstance(parsed, dict):
                    raise ValueError("Qwen VL response was not a JSON object")
                return parsed

            messages = [{"role": "user", "content": [
                {"type": "image", "image": person.convert("RGB")},
                {"type": "image", "image": template.convert("RGB")},
                {"type": "text", "text": self._PROMPT.format(second_image=second_image)},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                images=[person.convert("RGB"), template.convert("RGB")],
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=160, do_sample=False)
            response = processor.batch_decode(
                generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )[0].strip()
            try:
                parsed = parse_response(response)
            except (json.JSONDecodeError, ValueError):
                logger.warning("[UniformVL] Primary response was not JSON; retrying garment-only analysis: %r", response[:300])
                del generated, inputs
                retry_messages = [{"role": "user", "content": [
                    {"type": "image", "image": template.convert("RGB")},
                    {"type": "text", "text": self._GARMENT_RETRY_PROMPT},
                ]}]
                retry_text = processor.apply_chat_template(retry_messages, tokenize=False, add_generation_prompt=True)
                retry_inputs = processor(
                    text=[retry_text], images=[template.convert("RGB")], padding=True, return_tensors="pt"
                ).to(model.device)
                with torch.inference_mode():
                    retry_generated = model.generate(**retry_inputs, max_new_tokens=120, do_sample=False)
                retry_response = processor.batch_decode(
                    retry_generated[:, retry_inputs.input_ids.shape[1]:], skip_special_tokens=True
                )[0].strip()
                parsed = parse_response(retry_response)
                del retry_generated, retry_inputs
                hair_messages = [{"role": "user", "content": [
                    {"type": "image", "image": person.convert("RGB")},
                    {"type": "text", "text": self._HAIR_RETRY_PROMPT},
                ]}]
                hair_text = processor.apply_chat_template(hair_messages, tokenize=False, add_generation_prompt=True)
                hair_inputs = processor(
                    text=[hair_text], images=[person.convert("RGB")], padding=True, return_tensors="pt"
                ).to(model.device)
                with torch.inference_mode():
                    hair_generated = model.generate(**hair_inputs, max_new_tokens=80, do_sample=False)
                hair_response = processor.batch_decode(
                    hair_generated[:, hair_inputs.input_ids.shape[1]:], skip_special_tokens=True
                )[0].strip()
                try:
                    parsed.update(parse_response(hair_response))
                except (json.JSONDecodeError, ValueError):
                    logger.warning("[UniformVL] Hair-only retry was not JSON: %r", hair_response[:300])
                del hair_generated, hair_inputs
            result.update(parsed)
            result["source"] = "qwen2.5-vl"
        except Exception as exc:
            logger.warning("[UniformVL] Analysis fallback: %s", exc)
        finally:
            if model is not None:
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return result

    def analyze_board(self, board: Image.Image) -> Dict[str, Any]:
        """Analyze the combined person/template/rough-fit board in one VL image."""
        result = self.fallback(board, board)
        result.update({"collar_anchor_ratio": 0.96, "uniform_scale": 1.0, "fit_confidence": "limited"})
        if not MODEL_DIR.is_dir():
            return result

        model = None
        try:
            from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            quantization = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            processor = AutoProcessor.from_pretrained(str(MODEL_DIR), local_files_only=True)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(MODEL_DIR), quantization_config=quantization,
                device_map={"": 0} if torch.cuda.is_available() else "cpu",
                low_cpu_mem_usage=True, local_files_only=True,
            ).eval()
            messages = [{"role": "user", "content": [
                {"type": "image", "image": board.convert("RGB")},
                {"type": "text", "text": self._BOARD_PROMPT},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[board.convert("RGB")], padding=True, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=180, do_sample=False)
            response = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
            match = re.search(r"\{.*\}", response, flags=re.DOTALL)
            parsed = json.loads(match.group(0) if match else response)
            if isinstance(parsed, dict):
                result.update(parsed)
                result["source"] = "qwen2.5-vl-board"
        except Exception as exc:
            logger.warning("[UniformVL] Board analysis fallback: %s", exc)
        finally:
            if model is not None:
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return result


uniform_vl_analyzer = UniformVLAnalyzer()
