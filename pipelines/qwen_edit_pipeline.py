"""
Qwen-Image-Edit Neural Diffusion Pipeline
Production Service powered strictly by the genuine Qwen-Image-Edit-2511 model:
- Decoupled Multimodal Vision-Language Conditioning via Isolated Subprocess Worker
- GGUF Quantized QwenImageTransformer2DModel DiT with 2-Chunk CUDA Streaming
- AutoencoderKLQwenImage 5D VAE on CPU with FlowMatchEulerDiscreteScheduler
- PhotoRestorationService Facial Identity Lock, Optical Restoration & Jewelry Polish
"""

from __future__ import annotations

import io
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
    if (0 < min_dimension < 320) or (0 < face_detail < 35.0):
        risk += 2
        reasons.append("very low source resolution or facial detail")
    elif (0 < min_dimension < 480) or (0 < face_detail < 80.0):
        risk += 1
        reasons.append("limited source resolution or facial detail")

    recommended = 20 if risk >= 4 else 12 if risk >= 2 else 8 if risk >= 1 else 4
    requested_maximum = int(maximum_steps or 20)
    maximum = (
        20 if requested_maximum >= 20 else
        12 if requested_maximum >= 12 else
        8 if requested_maximum >= 8 else 4
    )
    selected = min(recommended, maximum)
    if not reasons:
        reasons.append("clean frontal portrait with balanced lighting")
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
        risk += 2
        reasons.append("hard or direct source lighting")
    elif sunlight or hotspot:
        # A soft-indoor result that also flags a hotspot is contradictory VL
        # evidence. Treat it as moderate relighting work, not a hard case.
        risk += 1
        reasons.append("possible lighting hotspot")

    accessories = vl_plan.get("hair_accessories")
    has_hair_accessories = str(accessories or "").strip().lower() not in {
        "", "none", "[]", "{}", "unknown", "uncertain"
    }
    if has_hair_accessories:
        risk += 2
        reasons.append("hair accessories")

    pose = str(vl_plan.get("pose") or vl_plan.get("face_orientation") or "").lower()
    non_frontal_pose = pose and not any(term in pose for term in ("front", "forward"))
    if non_frontal_pose:
        risk += 2
        reasons.append("non-frontal source pose")

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

    recommended = 20 if risk >= 4 else 12 if risk >= 2 else 8 if risk >= 1 else 4

    # Hair clips, strong source lighting, and a non-frontal pose mean the
    # original pixels are especially valuable. More denoising does not repair
    # those conditions; it makes Qwen rebuild the child's face and hair. Keep
    # such portraits at twelve steps or below, even when the caller allows 20.
    source_preservation_risk = has_hair_accessories or hard_light or bool(non_frontal_pose)
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
        conditioning.save(temp_path)

    models_posix = MODELS_DIR.resolve().as_posix()
    temp_img_paths_posix = [p.resolve().as_posix() for p in temp_img_paths]
    embeds_cache_posix = embeds_cache_file.resolve().as_posix()
    helper_code = f"""
import sys
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from diffusers import QwenImageEditPlusPipeline

MODELS_DIR = Path("{models_posix}")
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained((MODELS_DIR / "tokenizer").as_posix())
processor = AutoProcessor.from_pretrained((MODELS_DIR / "processor").as_posix())

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
    device_map={{"": 0}},
)

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
        if ret.returncode != 0:
            logger.error("[QWEN_ENCODE_WORKER_ERROR] stdout: %s | stderr: %s", ret.stdout, ret.stderr)
            raise RuntimeError(f"Prompt encoding worker failed (code {ret.returncode}): {ret.stderr}")
        loaded_cache = torch.load(str(embeds_cache_file), map_location="cpu")
        results = [(item["embeds"], item["mask"]) for item in loaded_cache["items"]]
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
    2-Chunk streaming forward pass that executes 30 DiT transformer blocks
    per chunk on CUDA without triggering Windows WDDM paging thrashing.
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

    half_len = len(self.transformer_blocks) // 2

    # --- CHUNK 0 (blocks 0..half_len-1) ---
    for b in self.transformer_blocks[:half_len]:
        b.to(dev)

    for block in self.transformer_blocks[:half_len]:
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

    for b in self.transformer_blocks[:half_len]:
        b.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- CHUNK 1 (blocks half_len..end) ---
    for b in self.transformer_blocks[half_len:]:
        b.to(dev)

    for block in self.transformer_blocks[half_len:]:
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

    for b in self.transformer_blocks[half_len:]:
        b.to("cpu")
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
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    return image_latents.to(device=dev, dtype=dtype)


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
        return w, h

    @staticmethod
    def _crop_school_passport_portrait(image: Image.Image, width: int, height: int) -> Image.Image:
        """Frame a generated portrait from crown to upper chest at 35:45 ratio."""
        rgb = np.array(image.convert("RGB"))
        source_h, source_w = rgb.shape[:2]
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = detector.detectMultiScale(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 1.1, 4, minSize=(48, 48)
        )
        if not len(faces):
            return image.resize((width, height), Image.LANCZOS)

        face_x, face_y, face_w, face_h = max(faces, key=lambda box: box[2] * box[3])
        target_ratio = width / height
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
        foreground = np.linalg.norm(lab - backdrop_lab, axis=2) > 18.0
        central_foreground = foreground[:, int(source_w * .04):int(source_w * .96)]
        occupied_rows = np.count_nonzero(central_foreground, axis=1) >= max(3, int(source_w * .02))
        occupied = np.flatnonzero(occupied_rows)
        crown_y = int(occupied[0]) if occupied.size else max(0, int(face_y - face_h * .30))
        estimated_crop_h = min(source_h, max(int(face_h * 2.28), int(source_h * 0.52)))
        minimum_headroom = max(10, int(estimated_crop_h * .055))
        pad_top = max(0, minimum_headroom - crown_y)
        if pad_top:
            backdrop = np.median(np.concatenate((
                rgb[:edge, :edge].reshape(-1, 3),
                rgb[:edge, -edge:].reshape(-1, 3),
            )), axis=0).round().clip(0, 255).astype(np.uint8)
            padded = np.empty((source_h + pad_top, source_w, 3), dtype=np.uint8)
            padded[:] = backdrop
            padded[pad_top:] = rgb
            rgb = padded
            image = Image.fromarray(rgb, "RGB")
            source_h += pad_top
            face_y += pad_top

        # Include the complete hair, chin, shoulders and upper chest. Keeping
        # nearly the whole generated torso made children look unusually short
        # and read like half-body portraits instead of school-ID photographs.
        crop_h = min(source_h, max(int(face_h * 2.28), int(source_h * 0.52)))
        crop_w = min(source_w, max(1, int(crop_h * target_ratio)))
        if crop_w / crop_h > target_ratio:
            crop_w = max(1, int(crop_h * target_ratio))
        else:
            crop_h = max(1, int(crop_w / target_ratio))

        center_x = face_x + face_w // 2
        top = int(face_y - face_h * 0.44)
        top = max(0, min(source_h - crop_h, top))
        left = max(0, min(source_w - crop_w, center_x - crop_w // 2))
        cropped = image.crop((left, top, left + crop_w, top + crop_h))
        return cropped.resize((width, height), Image.LANCZOS)

    @staticmethod
    def _lock_uniform_identity(source: Image.Image, generated: Image.Image) -> Image.Image:
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
        aligned_matched = cv2.cvtColor(
            aligned_lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB,
        ).astype(np.float32)
        # Repair only genuinely borderline candidates. A moderate blend
        # restores facial structure without producing an oval pasted-face
        # boundary; candidates below the raw-score floor are rejected instead.
        alpha = (mask * .36)[:, :, None]
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
    def _lock_uniform_source_head(
        source: Image.Image,
        generated: Image.Image,
        source_labels: np.ndarray,
        background_color: Union[str, Tuple[int, int, int]],
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
        source_hair_label = aligned_labels == 2
        if np.count_nonzero(source_face) < 128 or np.count_nonzero(source_hair_label) < 128:
            return generated.convert("RGB"), {"applied": False, "reason": "head_mask_unavailable"}

        source_lab = cv2.cvtColor(aligned_source, cv2.COLOR_RGB2LAB).astype(np.float32)
        dark_samples = source_lab[source_hair_label & (source_lab[:, :, 0] < 115)]
        if dark_samples.size == 0:
            return generated.convert("RGB"), {"applied": False, "reason": "hair_colour_unavailable"}
        hair_median = np.median(dark_samples, axis=0)
        colour_distance = np.linalg.norm(source_lab - hair_median[None, None, :], axis=2)
        # Outdoor leaves are frequently mislabeled as hair. Genuine dark hair
        # remains close to the dark-strand median even where it has soft sheen.
        dark_hair = source_hair_label & (
            (colour_distance < 62.0) | (source_lab[:, :, 0] < 76.0)
        )

        x1, y1, x2, y2 = np.asarray(generated_faces[0].bbox, dtype=float)
        face_w, face_h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        yy, xx = np.indices((height, width))
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
        head_mask = source_face | dark_hair | light_clip
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
        result[generated_hair & head_zone & ~source_hair_extent] = QwenEditPipeline._parse_bg_color(None, background_color)

        # Match only lightness in the retained source face/hair; chroma stays
        # from the upload so complexion and dark-hair pigment remain authentic.
        result_lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB).astype(np.float32)
        core = source_face & (source_lab[:, :, 0] > 25)
        if np.count_nonzero(core) >= 128:
            shift = float(np.median(result_lab[:, :, 0][core]) - np.median(source_lab[:, :, 0][core]))
            source_lab[:, :, 0] += np.clip(shift, -14.0, 14.0)
        matched_source = cv2.cvtColor(source_lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
        alpha = cv2.GaussianBlur(head_mask.astype(np.uint8) * 255, (7, 7), 0).astype(np.float32) / 255.0
        result = result.astype(np.float32) * (1.0 - alpha[:, :, None]) + matched_source * alpha[:, :, None]
        return Image.fromarray(np.rint(result).clip(0, 255).astype(np.uint8), "RGB"), {
            "applied": True,
            "head_pixels": int(np.count_nonzero(head_mask)),
            "clip_pixels": int(np.count_nonzero(light_clip)),
        }

    def _parse_bg_color(self, bg_color: Union[str, Tuple[int, int, int]]) -> Tuple[int, int, int]:
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
        # Qwen often shades a nominally flat gray backdrop beside dark curls.
        # A wider tolerance removes that neutral smoky fringe. Protect pale
        # flowers separately below so they are not mistaken for the backdrop.
        # Qwen's conditioning backdrop is deliberately flat middle gray. Keep
        # this tolerance narrow so light-brown fringe hair and skin near the
        # forehead cannot be mistaken for backdrop.
        candidates = (distance < 48.0).astype(np.uint8)
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
                candidates[lower_region & clothing_hull.astype(bool)] = 0
            # Protect all parsed subject below the shoulders, plus only reliable
            # face/head classes above them. SCHP occasionally labels a large
            # patch of flat backdrop beside hair as an arm or garment; trusting
            # every upper non-hair label retained exactly the gray halo this
            # correction is meant to remove.
            yy, _ = np.indices(subject.shape)
            lower_subject = subject.astype(bool) & (yy >= int(h * .48))
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
            protected_subject = lower_subject | reliable_upper_subject | hair_core
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
        # A one-pixel optical transition removes the bright matte fringe that
        # Qwen leaves around curls and bows. The protected semantic subject and
        # accessory zone above prevent this narrow antialiasing band from
        # repainting real hair or translucent petals.
        feathered = cv2.GaussianBlur(matte, (3, 3), .6)
        definite_background = cv2.erode(
            matte, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1,
        )
        feathered[definite_background == 255] = 255
        alpha = feathered.astype(np.float32)[:, :, None] / 255.0
        target = np.full_like(rgb, self._parse_bg_color(background_color), dtype=np.uint8)
        old_background = np.median(np.concatenate((
            rgb[:edge, :edge].reshape(-1, 3),
            rgb[:edge, -edge:].reshape(-1, 3),
        )), axis=0).astype(np.float32)
        # Remove the old backdrop contribution before adding the requested one.
        # Ordinary alpha blending mixes gray into blue and produces a pale halo.
        corrected = rgb.astype(np.float32) + alpha * (
            target.astype(np.float32) - old_background[None, None, :]
        )
        corrected[matte == 255] = target[matte == 255]
        return Image.fromarray(corrected.clip(0, 255).astype(np.uint8), "RGB")

    def _replace_background_with_foreground_matte(
        self,
        image: Image.Image,
        background_color: Union[str, Tuple[int, int, int]],
        job_id: Optional[str] = None,
        preserve_foreground_rgb: bool = False,
        strict: bool = False,
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
            subject = (labels != 0).astype(np.uint8)
            if np.count_nonzero(subject) < subject.size * .18:
                raise RuntimeError("semantic subject mask is too small")

            # SCHP supplies the tight person/hair silhouette. Keep surrounding
            # backdrop as definite background; the previous 31-pixel probable-
            # foreground expansion retained a visible gray halo around hair.
            height, width = subject.shape
            near_subject = cv2.dilate(
                subject, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
                iterations=1,
            )
            grab_mask = np.full(subject.shape, cv2.GC_BGD, dtype=np.uint8)
            grab_mask[subject > 0] = cv2.GC_PR_FGD
            non_hair = np.isin(
                labels, (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
            ).astype(np.uint8)
            non_hair_core = cv2.erode(
                non_hair, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )
            grab_mask[non_hair_core > 0] = cv2.GC_FGD
            luma = (
                rgb[:, :, 0].astype(np.float32) * .2126
                + rgb[:, :, 1].astype(np.float32) * .7152
                + rgb[:, :, 2].astype(np.float32) * .0722
            )
            grab_mask[(labels == 2) & (luma < 115.0)] = cv2.GC_FGD
            # Hair accessories are frequently parsed as background. Protect
            # every non-backdrop accessory colour touching the upper hair
            # silhouette, including saturated red bows as well as pale clips.
            # The input here is Qwen's already-flat studio portrait, so corner
            # colour is reliable evidence of backdrop rather than scenery.
            yy, xx = np.indices(subject.shape)
            edge = max(4, min(20, height // 20, width // 20))
            backdrop_rgb = np.median(np.concatenate((
                rgb[:edge, :edge].reshape(-1, 3),
                rgb[:edge, -edge:].reshape(-1, 3),
            )), axis=0)
            corner_samples = np.concatenate((
                rgb[:edge, :edge].reshape(-1, 3),
                rgb[:edge, -edge:].reshape(-1, 3),
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
            # GrabCut can still relabel a small accessory edge after the hard
            # seeds are supplied. Restore only pixels demonstrably unlike the
            # sampled flat backdrop; holes within bows remain background.
            foreground[accessory] = 1
            foreground = cv2.morphologyEx(
                foreground, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )
            foreground_alpha = cv2.GaussianBlur(foreground.astype(np.float32), (3, 3), 0.45)
            foreground_alpha = np.clip(foreground_alpha, 0.0, 1.0)[:, :, None]
            background_alpha = 1.0 - foreground_alpha
            target = np.full_like(rgb, self._parse_bg_color(background_color), dtype=np.uint8)
            # Decontaminate the antialiased edge rather than blending the old
            # studio color into the selected backdrop.
            old_background = backdrop_rgb.astype(np.float32)
            composed = rgb.astype(np.float32) + background_alpha * (
                target.astype(np.float32) - old_background[None, None, :]
            )
            composed[foreground == 0] = target[foreground == 0]
            return Image.fromarray(np.rint(composed).clip(0, 255).astype(np.uint8), "RGB")
        except Exception as exc:
            if strict:
                raise RuntimeError("Background color replacement failed; raw Qwen output is retained.") from exc
            logger.warning("[BACKGROUND_MATTE] Failed; retaining Qwen backdrop: %s", exc)
            return image.convert("RGB")

    @staticmethod
    def _apply_rough_silhouette_guard(
        image: Image.Image, rough_reference: Image.Image
    ) -> Image.Image:
        """Keep Qwen inside the intended passport head/upper-chest silhouette.

        Qwen can preserve identity while adding hands, props, tables, or a
        retail scene. The fixed rough canvas is the authoritative framing, so
        use its semantic person silhouette as a hard geometry guard before
        background validation. This does not replace any pixels inside the
        silhouette or alter the generated face/uniform.
        """
        try:
            from pipelines.schp_service import parse as parse_body_parts

            rgb = np.asarray(image.convert("RGB"))
            reference = np.asarray(
                rough_reference.convert("RGB").resize((rgb.shape[1], rgb.shape[0]), Image.LANCZOS)
            )
            # The rough canvas is deliberately not a silhouette authority.
            # Its template placement can differ from Qwen's restored head and
            # shoulders, so intersecting it with the Qwen subject cuts holes
            # through hair, neck, and sleeves. Use the generated semantic
            # subject mask for all portrait geometry; the rough canvas is used
            # only to recover the requested flat background colour.
            generated_labels = parse_body_parts(rgb)["labels"]
            foreground = (generated_labels != 0).astype(np.uint8)
            foreground = cv2.morphologyEx(
                foreground, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1,
            )
            # Passport framing ends at the upper chest. Remove side hands,
            # bags, and lower-arm pixels that Qwen may attach to the subject;
            # the central shoulders and uniform remain inside this corridor.
            yy, xx = np.indices(foreground.shape)
            side_corridor = (xx >= int(foreground.shape[1] * .25)) & (
                xx <= int(foreground.shape[1] * .75)
            )
            foreground[(yy >= int(foreground.shape[0] * .58)) & (~side_corridor)] = 0
            if np.count_nonzero(foreground) < foreground.size * 0.15:
                return image.convert("RGB")
            backdrop = np.median(np.concatenate((
                reference[:8, :8].reshape(-1, 3),
                reference[:8, -8:].reshape(-1, 3),
            )), axis=0).astype(np.uint8)
            result = rgb.copy()
            # Keep the outside of the generated subject as the plain backdrop.
            # Never inpaint a discrepancy against the rough layout: that was
            # responsible for the blue holes and floating-collar artifacts.
            result[foreground == 0] = backdrop
            return Image.fromarray(result, "RGB")
        except Exception as exc:
            logger.warning("[UNIFORM_SILHOUETTE] Guard unavailable: %s", exc)
            return image.convert("RGB")

    @staticmethod
    def _lock_rough_uniform_pixels(
        image: Image.Image, garment_reference: Image.Image
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
            fitted_reference = generated.copy()
            fitted_mask = np.zeros(generated.shape[:2], dtype=np.uint8)
            fitted_reference[y0:y1, x0:x1] = garment_crop[:crop_h, :crop_w]
            fitted_mask[y0:y1, x0:x1] = mask_crop[:crop_h, :crop_w]
            fitted_mask = cv2.morphologyEx(
                fitted_mask, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1,
            )
            alpha = cv2.GaussianBlur(fitted_mask.astype(np.float32), (0, 0), 1.2)
            alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
            locked = generated.astype(np.float32) * (1.0 - alpha) + fitted_reference.astype(np.float32) * alpha
            return Image.fromarray(np.rint(locked).clip(0, 255).astype(np.uint8), "RGB")
        except Exception as exc:
            logger.warning("[UNIFORM_TEMPLATE_LOCK] Template lock unavailable: %s", exc)
            return image.convert("RGB")

    def _has_selected_solid_background(
        self, image: Image.Image, background_color: Union[str, Tuple[int, int, int]]
    ) -> bool:
        """Check corner backdrop blocks before invoking a segmentation fallback."""
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
        distances = np.linalg.norm(samples - target, axis=1)
        side_width = max(2, int(width * 0.03))
        side_height = max(1, int(height * 0.68))
        side_pixels = np.concatenate((
            rgb[:side_height, :side_width].reshape(-1, 3),
            rgb[:side_height, -side_width:].reshape(-1, 3),
        ))
        # Long hair and bows can legitimately reach a passport crop's side
        # edge. Exclude parsed subject pixels before measuring backdrop hue;
        # otherwise an exact blue background is rejected simply because the
        # child fills the frame.
        try:
            from pipelines.schp_service import parse as parse_body_parts

            subject = parse_body_parts(rgb.astype(np.uint8))["labels"] != 0
            side_subject = np.concatenate((
                subject[:side_height, :side_width].reshape(-1),
                subject[:side_height, -side_width:].reshape(-1),
            ))
            side_samples = side_pixels[~side_subject]
            if side_samples.shape[0] < max(16, side_pixels.shape[0] // 5):
                side_samples = side_pixels
        except Exception:
            side_samples = side_pixels
        side_distances = np.linalg.norm(side_samples - target, axis=1)
        side_match = float(np.mean(side_distances <= 18.0))
        exact_match = bool(
            np.median(distances) <= 8.0
            and np.percentile(distances, 90) <= 18.0
            and side_match >= 0.92
        )
        if exact_match:
            return True
        return False

    @staticmethod
    def _has_generated_uniform(
        image: Image.Image,
        target_outer_rgb: Optional[Tuple[int, int, int]],
        rough_reference: Optional[Image.Image] = None,
    ) -> bool:
        """Reject unchanged source clothing and incomplete garment edits."""
        try:
            from pipelines.schp_service import parse as parse_body_parts

            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            labels = parse_body_parts(rgb)["labels"]
            clothing = np.isin(labels, (5, 6, 7, 9, 10, 12))
            coverage = float(np.mean(clothing))
            if coverage < 0.07:
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
            if target_outer_rgb is None:
                return True
            # Judge the measured outer-template colour only inside the parsed
            # lower clothing. Measuring the full lower frame counts the white
            # shirt, skin, and backdrop against a gray pinafore and rejects a
            # valid multi-layer school uniform.
            yy, _ = np.indices(labels.shape)
            lower_clothing = clothing & (yy >= int(height * .45))
            lower_distance = np.linalg.norm(
                rgb.astype(np.float32) - np.asarray(target_outer_rgb, dtype=np.float32), axis=2
            )
            matching_fraction = float(
                np.mean(lower_distance[lower_clothing] <= 55.0)
            ) if np.any(lower_clothing) else 0.0
            logger.info(
                "[UNIFORM_REVIEW] clothing_coverage=%.3f target_color_fraction=%.3f",
                coverage, matching_fraction,
            )
            if rough_reference is not None:
                reference = np.asarray(
                    rough_reference.convert("RGB").resize((width, height), Image.LANCZOS),
                    dtype=np.uint8,
                )
                ref_labels = parse_body_parts(reference)["labels"]
                ref_clothing = np.isin(ref_labels, (5, 6, 7, 9, 10, 12))
                ref_clothing &= np.indices(ref_labels.shape)[0] >= int(height * .45)
                candidate_clothing = clothing & (
                    np.indices(labels.shape)[0] >= int(height * .45)
                )
                if np.count_nonzero(ref_clothing) >= 256 and np.count_nonzero(candidate_clothing) >= 256:
                    # The final garment lock uses these exact template pixels.
                    # Compare within that semantic garment region rather than
                    # demanding that every shirt, sleeve, and tie pixel match
                    # the outer-pinafore color. This keeps valid multi-layer
                    # uniforms from failing solely because their white layer
                    # is correctly not gray.
                    template_delta = float(np.mean(np.linalg.norm(
                        rgb[ref_clothing].astype(np.float32)
                        - reference[ref_clothing].astype(np.float32),
                        axis=1,
                    )))
                    logger.info(
                        "[UNIFORM_REVIEW] template_garment_delta=%.2f outer_color_fraction=%.3f",
                        template_delta, matching_fraction,
                    )
                    if template_delta > 48.0:
                        logger.warning(
                            "[UNIFORM_REVIEW] Locked garment differs from supplied template: %.2f",
                            template_delta,
                        )
                        return False
                    ref_hsv = cv2.cvtColor(reference, cv2.COLOR_RGB2HSV)
                    candidate_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
                    ref_saturation = float(np.mean(ref_hsv[:, :, 1][ref_clothing] > 40))
                    candidate_saturation = float(
                        np.mean(candidate_hsv[:, :, 1][candidate_clothing] > 40)
                    )
                    # Large newly saturated regions are a reliable signal that
                    # the source clothing was retained (e.g. pink cardigan)
                    # instead of the supplied neutral school uniform.
                    if candidate_saturation - ref_saturation > 0.28:
                        logger.warning(
                            "[UNIFORM_REVIEW] Garment palette diverges from template: "
                            "reference_saturation=%.3f candidate_saturation=%.3f",
                            ref_saturation,
                            candidate_saturation,
                        )
                        return False
            elif matching_fraction < 0.18:
                # This fallback only applies when no template reference is
                # available for direct comparison.
                logger.warning(
                    "[UNIFORM_REVIEW] Template-color coverage too small without a template reference: %.3f",
                    matching_fraction,
                )
                return False
            return True
        except Exception as exc:
            logger.warning("[UNIFORM_REVIEW] Garment verification unavailable: %s", exc)
            return False

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
    def _has_flat_border_background(image: Image.Image) -> bool:
        """Return true only when Qwen generated a uniform studio backdrop."""
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        height, width = rgb.shape[:2]
        edge = max(4, min(24, height // 12, width // 12))
        if height < edge * 2 or width < edge * 2:
            return False
        corners = np.concatenate((
            rgb[:edge, :edge].reshape(-1, 3),
            rgb[:edge, -edge:].reshape(-1, 3),
        ))
        center = np.median(corners, axis=0)
        distances = np.linalg.norm(corners - center, axis=1)
        # A generated seamless backdrop can contain mild sensor/lighting
        # variation. Keep the tolerance tight enough to reject scenery while
        # allowing this normal flat-background noise.
        return bool(np.percentile(distances, 90) <= 28.0)

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
        true_cfg_scale: float = 1.6,
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
            bg_desc = f"with a solid {background_color} studio backdrop"

        if prompt and prompt.strip():
            effective_prompt = prompt.strip()
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
            latents, image_latents = orig_prepare_latents(*args, **kwargs)
            dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
            if latents is not None:
                latents = latents.to(device=dev, dtype=dtype)
            if image_latents is not None:
                image_latents = image_latents.to(device=dev, dtype=dtype)
            return latents, image_latents

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
            num_steps, run_seed, max(1.01, float(true_cfg_scale)) if effective_negative else 1.0,
            use_lightning_lora and LORA_FILE.exists(),
        )
        res = pipe(
            image=conditioning_images if len(conditioning_images) > 1 else input_resized,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            num_inference_steps=num_steps,
            true_cfg_scale=max(1.01, float(true_cfg_scale)) if effective_negative else 1.0,
            height=h,
            width=w,
            generator=torch.Generator(device="cpu").manual_seed(run_seed),
            output_type="latent",
            callback_on_step_end=on_step_end,
        )
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
        raw_diffused = _decode_cpu_latents(vae, output_latents, height=h, width=w)
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
        generation_steps, step_reasons = _select_uniform_steps(vl_plan, steps)
        # Strong classifier-free guidance makes Qwen redraw the person while
        # obeying garment prose. Match the enhancement route's restrained
        # guidance so the upload remains the dominant face/hair reference.
        generation_cfg_scale = (
            1.02 if generation_steps <= 4 else
            1.35 if generation_steps <= 8 else
            1.65 if generation_steps <= 12 else 2.0
        )
        logger.info(
            "[QWEN_ONLY_UNIFORM] Adaptive generation selected %d steps: %s",
            generation_steps, ", ".join(step_reasons),
        )
        if progress_callback:
            progress_callback(11, f"Selected {generation_steps} generation steps for this portrait...")
        from pipelines.uniform_badge_cleanup import remove_template_badges
        from pipelines.uniform_composite import (
            build_person_backdrop_guide, build_rough_composite, describe_fabric_color, describe_hair_correction,
            describe_lighting_correction, garment_on_selected_background, refinement_prompt,
            measure_source_person_colors, passport_garment_conditioning, source_hair_is_dark,
        )
        badge_reference = template_pil.copy()
        template_pil, template_badges_removed = remove_template_badges(template_pil, badge_reference)
        logger.info("[UNIFORM_BADGE] Removed %d template badge(s)", template_badges_removed)
        if vl_plan.get("source") != "qwen2.5-vl":
            logger.warning("[QWEN_ONLY_UNIFORM] VL source is %s; using geometry fallback", vl_plan.get("source"))

        # Keep the visual reference intact. Qwen omits emblems in the output;
        # inaccurate VL badge boxes must not erase template seams or fabric.
        logger.info("[UNIFORM_REFERENCE] original=%s prepared=%s", template_source.size, template_pil.size)

        uniform_seed = zlib.crc32(person_pil.tobytes())
        uniform_seed = zlib.crc32(template_pil.tobytes(), uniform_seed)
        uniform_seed = zlib.crc32(str(self._parse_bg_color(background_color)).encode("ascii"), uniform_seed)
        # A content-stable seed prevents retries of identical inputs from
        # drifting to a completely different child. The salt reproduces the
        # previously validated Qwen-only conditioning profile.
        uniform_seed = (uniform_seed ^ 647635142) & 0x7FFFFFFF
        # Repeated retries of a bad deterministic trajectory otherwise return
        # the same failed garment indefinitely. Keep content anchoring while
        # adding an auditable per-job salt so a new run explores a new edit.
        uniform_seed = zlib.crc32(job_id.encode("utf-8"), uniform_seed) & 0x7FFFFFFF

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
        logger.info("[QWEN_ONLY_UNIFORM] job=%s stable fit seed=%d", job_id, uniform_seed)

        selected_bg_rgb = self._parse_bg_color(background_color)
        # Condition Qwen on the user's selected backdrop from the first edit.
        # A neutral staging canvas made the model invent scenery, which was
        # then impossible to correct reliably without altering the subject.
        qwen_bg_rgb = selected_bg_rgb
        # Keep the opaque cutout normalization step, then reduce it to the
        # passport-sized garment-only conditioning canvas.
        _template_canvas = garment_on_selected_background(template_pil, qwen_bg_rgb)
        template_conditioning = passport_garment_conditioning(
            template_pil, qwen_bg_rgb, (width, height)
        )
        # Pass only the passport upper-chest garment reference. Full-body
        # template canvases give Qwen room to invent a different scene/pose.
        template_guide_path = OUTPUTS_DIR / f"{job_id}_uniform_guide.png"
        template_conditioning.save(template_guide_path, format="PNG")
        logger.info(
            "[UNIFORM_REFERENCE] White-background Qwen uniform guide -> %s",
            template_guide_path,
        )
        transparent_rgb, _ = remove_template_badges(template_source, template_source)
        transparent_template = transparent_rgb.convert("RGBA")
        transparent_template.putalpha(template_source.getchannel("A"))
        from pipelines.photo_restoration import _get_insight_app
        from pipelines.schp_service import parse as parse_body_parts
        source_faces = _get_insight_app().get(
            cv2.cvtColor(np.array(person_pil), cv2.COLOR_RGB2BGR)
        )
        if len(source_faces) != 1:
            raise RuntimeError("Uniform fitting requires exactly one visible face in the uploaded portrait")
        source_labels = parse_body_parts(np.array(person_pil))["labels"]
        measured_person_colors = measure_source_person_colors(person_pil, source_labels)
        # Keep the segmentation-based guide for diagnostics and layout only.
        # It can contain holes around curls, clips and the neck, so using it as
        # Qwen's primary image teaches the model an already damaged person.
        # The untouched upload is the sole identity edit input; Qwen receives
        # the uniform as its only secondary image and the final backdrop is
        # enforced after generation without altering the subject.
        person_guide = build_person_backdrop_guide(
            person_pil, source_labels, qwen_bg_rgb, remove_clothing=True,
        )
        person_guide_path = OUTPUTS_DIR / f"{job_id}_person_guide.png"
        person_guide.save(person_guide_path, format="PNG")
        logger.info("[UNIFORM_REFERENCE] Saved editable person guide -> %s", person_guide_path)
        rough_composite, rough_placement = build_rough_composite(
            person_pil, transparent_template, source_labels, source_faces[0].bbox, qwen_bg_rgb,
        )
        rough_path = OUTPUTS_DIR / f"{job_id}_rough_fit.png"
        rough_composite.save(rough_path, format="PNG")
        logger.info("[UNIFORM_COMPOSITE] Saved non-final Qwen layout guide -> %s", rough_path)
        # Stage 2: fix the rough layout into the same upper-chest passport
        # frame that the final school-ID image must use. This prevents Qwen
        # from seeing lower-body space and inventing hands or props.
        if progress_callback:
            progress_callback(14, "Fixing rough uniform layout for school-ID framing...")
        rough_composite = self._crop_school_passport_portrait(rough_composite, width, height)
        rough_fixed_path = OUTPUTS_DIR / f"{job_id}_rough_fixed.png"
        rough_composite.save(rough_fixed_path, format="PNG")
        logger.info("[UNIFORM_COMPOSITE] Saved fixed passport rough guide -> %s", rough_fixed_path)
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
                "Image 2 visibly includes neckwear; reproduce only that exact template neckwear, with no added jewelry."
            )
        else:
            neckwear_instruction = (
                "Image 2 has no tie, bow, ribbon or other neckwear. Leave the neckwear area bare above the template collar; do not invent any tie or bow."
            )
        portrait_details = observed_details((
            "face_detail", "face_shape", "eye_description", "eyebrow_description",
            "nose_description", "mouth_description", "earring_description", "expression", "skin_tone", "hair_parting",
            "hair_length", "hair_texture", "hair_color", "hair_accessories", "hair_accessory_details",
            "visible_wearables",
        ))
        hair_details = observed_details((
            "hair_style", "hair_parting", "hair_length", "hair_texture",
            "hair_color", "hair_accessories", "hair_accessory_details",
        ))
        # The vision model can confuse flower clips with bows or loose curls
        # with braids. Use only stable non-categorical hair facts in the edit
        # prompt; image 1 remains the shape/accessory authority.
        hair_generation_details = observed_details((
            "hair_parting", "hair_length", "hair_texture", "hair_color",
        ))
        template_generation_details = observed_details((
            "shirt_color_and_pattern", "outer_color_under_neutral_light",
            "fabric_detail",
        ))
        palette_evidence = str(vl_plan.get("template_palette_evidence") or "").strip()
        # VL can confuse the shirt collar with the outer pinafore neckline.
        # Never turn uncertain prose into a geometry command; template pixels
        # remain the sole structural authority for both garment layers.
        collar_constraint = (
            "Copy image 2's shirt collar and outer-garment neckline silhouette exactly. "
            "Keep every straight or square top edge straight or square; create a V-shaped edge only when image 2 visibly has one."
        )
        outer_target_rgb = None
        vl_rgb = vl_plan.get("outer_neutral_rgb") if vl_plan.get("source") == "qwen2.5-vl" else None
        if isinstance(vl_rgb, (list, tuple)) and len(vl_rgb) == 3:
            try:
                candidate = tuple(int(value) for value in vl_rgb)
                if all(0 <= value <= 255 for value in candidate):
                    outer_target_rgb = candidate
            except (TypeError, ValueError):
                pass
        fabric_color_instruction = describe_fabric_color(
            transparent_template,
            vl_color=(vl_plan.get("outer_color_under_neutral_light")
                      if vl_plan.get("source") == "qwen2.5-vl" else None),
            vl_rgb=outer_target_rgb,
        )

        logger.info("[UNIFORM_REFERENCE] Direct two-image edit: original person + uniform template")

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
        measured_hair_rgb = measured_person_colors.get("hair_rgb")
        if measured_hair_rgb:
            measured_hair_luma = sum(
                float(channel) * weight
                for channel, weight in zip(measured_hair_rgb, (.2126, .7152, .0722))
            )
            if measured_hair_luma < 82:
                hair_color_instruction += (
                    " Shaded source strands confirm black to very dark-brown hair; do not make it medium brown, golden, gray, green or blue."
                )
        lighting_instruction = describe_lighting_correction(vl_plan)
        analyzed_skin_tone = (
            str(vl_plan.get("skin_tone") or "").strip()
            if vl_plan.get("source") == "qwen2.5-vl"
            else ""
        )
        unsafe_cast_terms = ("orange", "yellow", "cyan", "blue", "green", "red cast", "color cast", "colour cast")
        if (
            analyzed_skin_tone.lower() in ("", "none", "unknown", "uncertain", "source natural skin tone")
            or any(term in analyzed_skin_tone.lower() for term in unsafe_cast_terms)
        ):
            skin_tone_instruction = (
                "Copy image 2's face/ear/neck complexion and undertone; correct illumination without recoloring skin."
            )
        else:
            skin_tone_instruction = (
                f"VL reads the complexion as {analyzed_skin_tone}; image 2 pixels remain authoritative. Preserve its "
                "undertone; never lighten, darken, whiten, tan, warm, cool or recolor skin."
            )
        measured_skin_rgb = measured_person_colors.get("skin_rgb")
        if measured_skin_rgb:
            skin_tone_instruction += (
                f" Stable source face pixels measure RGB {measured_skin_rgb}; preserve their chroma and undertone while changing only illumination."
            )
        logger.info("[UNIFORM_SKIN] VL=%s instruction=%s", vl_plan.get("skin_tone"), skin_tone_instruction)
        identity_instruction = (
            "Image 2 contains the source person's face, hair and accessories; use those pixels as the only identity authority."
        )
        # Do not narrate facial anatomy back to the diffusion model. Even an
        # accurate VL description encourages Qwen to synthesize a new child
        # matching the prose instead of retaining image 2's identity. Keep the
        # detailed observations in the audit and use the source pixels as the
        # sole authority for face, hair silhouette, accessories and jewelry.
        generated_prompt = (
            f"{refinement_prompt(qwen_bg_rgb, fabric_color_instruction, hair_color_instruction, lighting_instruction, skin_tone_instruction, identity_instruction, primary_is_identity=True)} "
            f"{collar_constraint} "
            "Do not use any text description to infer hair style, clip type, colour, count or position; image 1 pixels are authoritative. "
            f"Stable template fabric facts (must match image 2 pixels): {template_generation_details}. "
            f"{neckwear_instruction}"
        )
        if prompt and prompt.strip():
            generated_prompt += (
                " Additional styling preference (apply only when consistent with the source identity, "
                "hair, garment construction, selected background and no-badge requirements above): "
                + prompt.strip()
            )

        negative_prompt = (
            "multiple people, two people, second child, secondary face, background person, reflection person, duplicate head, adult, man, woman, family portrait, group portrait, badge, crest, emblem, logo, lettering, watermark, "
            "different person, fully regenerated face, redesigned face, face recreation, changed ethnicity appearance, changed ancestry appearance, changed facial proportions, changed mouth opening, altered smile, hidden source teeth, invented teeth, exaggerated eyes, enlarged eyes, doll eyes, beauty filter, airbrushed face, "
            "changed hairstyle, pulled-back hair, ponytail, bun, dyed hair, altered hairline, changed hair length, changed hair parting, "
            "missing source hair accessories, moved source flowers, moved source clips, recolored source accessories, invented accessories, changed hair ties, changed curl pattern, large hair bow, fabric hair bow, ribbon hair bow, oversized hair ribbon, duplicate face, "
            "head tilt, head roll, side gaze, three-quarter pose, rotated torso, slouched shoulders, uneven shoulders, asymmetrical shoulders, smoke, haze, fog, bloom, gray veil, blur, soft focus, low-contrast face, painterly face, "
            "harsh direct sunlight, blown highlights, cyan hair reflections, metallic hair glare, glowing hair edges, orange skin cast, changed complexion, altered skin tone, skin whitening, skin lightening, artificial pale skin, waxy skin, oversharpening halos, plastic fabric, rigid pasted clothing, embossed seams, "
            "distorted neck, hollow neck gap, floating head, disconnected collar, extra collar, wrong collar construction, wrong uniform color, beige background, cream background, "
            "wrong fabric pattern, missing template layers, missing stitched panels, flattened fabric texture, invented trim, invented tie, necklace, neck chain, pendant, locket, neck jewelry, changed earrings, invented earrings, oversized earrings, dangling earrings, "
            "cropped head, long shot, knees, legs, hands, invented lower garment, original background remnants, textured backdrop, collage, pasted garment, rough composite, cutout edges"
        )
        if not template_has_neckwear:
            negative_prompt += ", tie, necktie, school tie, striped tie, bow tie, collar bow, neck ribbon, cravat"
        # Retain the actual instructions and analysis for diagnosing a run,
        # rather than assuming an identity score proves garment/hair fidelity.
        import json
        audit_path = Path("outputs") / f"{job_id}_instructions.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps({
            "analysis": vl_plan, "prompt": generated_prompt,
            "negative_prompt": negative_prompt, "seed": uniform_seed,
            "seed_policy": "content_seed_with_job_retry_salt",
            "steps": generation_steps, "step_policy": "adaptive_4_8_12_20",
            "step_reasons": step_reasons,
            "identity_reference": True, "reference_count": 2,
            "prompt_token_budget": 512, "conditioning_max_dimension": 320,
            "conditioning_pixel_budget": 102400,
            "true_cfg_scale": generation_cfg_scale,
            "template_size": list(template_pil.size), "preserve_reference_aspect": True,
            "vl_observations_used_as_generation_facts": vl_plan.get("source") == "qwen2.5-vl",
            "template_badges_removed": template_badges_removed,
            "fabric_color_instruction": fabric_color_instruction,
            "hair_color_instruction": hair_color_instruction,
            "lighting_instruction": lighting_instruction,
            "skin_tone_instruction": skin_tone_instruction,
            "measured_source_colors": measured_person_colors,
            "identity_instruction": identity_instruction,
            "person_guide_path": str(person_guide_path),
            "primary_conditioning": "untouched uploaded portrait",
            "uniform_guide_path": str(template_guide_path),
            "rough_layout_path": str(rough_path),
            "rough_fixed_layout_path": str(rough_fixed_path),
            "rough_layout_placement": rough_placement,
            "portrait_details": portrait_details,
            "garment_details": garment_details,
            "measured_source_dark_hair": measured_dark_hair,
            "outer_target_rgb": outer_target_rgb,
            "conditioning_lighting": "Qwen edits the original portrait; the rough fit supplies placement only",
            "conditioning_background": "All Qwen layout references use the selected solid backdrop from the first generation",
            "qwen_background_rgb": list(qwen_bg_rgb),
            "selected_background_rgb": list(selected_bg_rgb),
            "delivery_policy": "qwen_uniform_and_identity_then_exact_background_verified",
            "prompt_review": "Image 1 is the original identity, hairstyle and accessory authority; image 2 is the exact uniform authority. The rough composite is used only for final geometry validation.",
        }, ensure_ascii=True, indent=2), encoding="utf-8")
        from pipelines.uniform_review import check_uniform_identity
        if progress_callback:
            progress_callback(90, "Checking that the portrait matches the uploaded person...")
        output_path = self.qwen_edit_enhancer(
            image_input=person_pil,
            reference_images=[template_conditioning],
            prompt=generated_prompt,
            negative_prompt=negative_prompt,
            job_id=job_id,
            background_color=background_color,
            width=width,
            height=height,
            original_filename=f"{job_id}_uniform",
            steps=generation_steps,
            max_sequence_length=512,
            timeout_seconds=None,
            progress_callback=(lambda value, message: progress_callback(15 + int(value * 0.75), message)) if progress_callback else None,
            # The primary Qwen image is the untouched uploaded portrait. The
            # rough composite is deliberately excluded from generative
            # conditioning: it is a geometry-only mask and must never be
            # copied into the delivered portrait as a pasted face, hand, or
            # collar seam.
            preserve_source_clothing=True,
            max_generation_dimension=640,
            minimum_generation_dimension=640,
            keep_generation_resolution=True,
            use_birefnet_background=False,
            true_cfg_scale=generation_cfg_scale,
            seed=uniform_seed,
            face_lock_after_generation=False,
            reject_identity_failure=False,
            # Keep the raw Qwen decode for uniform validation. The processed
            # restoration path can paste a mismatched face over the garment;
            # identity and backdrop are validated in this pipeline instead.
            return_raw_candidate=True,
            conditioning_max_dimension=320,
            conditioning_pixel_budget=320 * 320,
            preserve_reference_aspect=True,
        )
        generated_candidate = Image.open(output_path).convert("RGB")
        if not self._is_regenerated_from_rough(generated_candidate, rough_composite):
            raise RuntimeError(
                "Uniform output was not delivered because Qwen copied the rough layout instead of regenerating a finished portrait."
            )
        # Qwen may render a decorative studio-like scene even when the subject
        # and uniform are correct. The fixed rough canvas owns the passport
        # silhouette, so the silhouette guard below removes all exterior scene
        # pixels and applies the selected flat backdrop before delivery.
        # Validate the same passport upper-chest frame that will be delivered.
        # A hand/prop below the requested crop must not invalidate an otherwise
        # correct school portrait, while anything inside the crop remains gated.
        identity_locked = self._crop_school_passport_portrait(generated_candidate, width, height)
        # Remove a decorative scene before face verification. Qwen can put
        # framed photos or people-shaped props behind an otherwise valid
        # portrait; InsightFace then sees multiple faces and rejects the child
        # before the normal delivery matte has a chance to remove them.
        if not self._has_selected_solid_background(identity_locked, background_color):
            identity_locked = self._replace_smooth_border_background(
                identity_locked, background_color,
            )
            if not self._has_selected_solid_background(identity_locked, background_color):
                identity_locked = self._replace_background_with_foreground_matte(
                    identity_locked,
                    background_color=background_color,
                    job_id=f"{job_id}_identity_background",
                    preserve_foreground_rgb=True,
                    strict=True,
                )
            logger.info("[UNIFORM_BACKGROUND] Applied selected background before identity verification")
        # The raw Qwen candidate is the finished portrait. Do not paste the
        # template back over it: even a geometrically aligned overlay leaves
        # visible collar seams and turns a natural uniform into a cutout.
        # The template stays in the Qwen conditioning and validation paths.
        identity_review = check_uniform_identity(person_pil, identity_locked)
        initial_identity_review = dict(identity_review)
        if not identity_review["accepted"]:
            # Keep this as a localized post-process, not a second generation:
            # align the uploaded face to Qwen's crop and blend it softly only
            # when the biometric gate finds a real identity failure.
            logger.warning("[UNIFORM_IDENTITY_REPAIR] Applying localized face alignment: %s", identity_review)
            identity_locked = self._lock_uniform_identity(person_pil, identity_locked)
            identity_review = check_uniform_identity(person_pil, identity_locked)
            if not identity_review["accepted"]:
                logger.error("[UNIFORM_IDENTITY_REVIEW] Candidate rejected: %s", identity_review)
                score = identity_review.get("similarity")
                score_text = f" (score {score:.3f})" if isinstance(score, (int, float)) else ""
                raise RuntimeError(
                    "Uniform output was not delivered because the generated face does not match the uploaded person"
                    f"{score_text}."
                )
        if not self._has_generated_uniform(
            identity_locked,
            outer_target_rgb,
            rough_reference=None,
        ):
            raise RuntimeError(
                "Uniform output was not delivered because Qwen retained the source clothing or did not reproduce enough of the supplied uniform template."
            )
        # Do not blend source hair after the Qwen pass. In outdoor uploads a
        # partial hair mask carries foliage/light colours into the clean
        # backdrop, producing visible halos and strand glitches. The source
        # remains the Qwen conditioning authority; the delivered portrait
        # keeps its coherent generated hair and neck transition intact.
        logger.info("[UNIFORM_HAIR_RESTORE] Skipped to avoid source-matte hair artifacts")
        # Face embedding similarity is necessary but insufficient for school-ID
        # work: a different hairstyle or a vest substituted for a blazer can
        # still score as the same child. Compare the candidate visually with
        # both uploaded authorities before allowing it to reach final matte or
        # colour finishing.
        from pipelines.uniform_review import review_uniform
        try:
            visual_review = review_uniform(
                person_pil, template_pil, identity_locked, qwen_bg_rgb,
            )
        except Exception as exc:
            raise RuntimeError(
                "Uniform output was not delivered because visual reference verification could not be completed: "
                f"{exc}"
            ) from exc
        # Qwen-VL comparison is useful for a user-visible review note, but it
        # has repeatedly mislabeled curls as braids and collars as different
        # garment types. Do not discard a biometric-verified, template-checked
        # portrait solely because that descriptive model disagrees. Only real
        # candidate defects remain delivery blockers.
        advisory_failures = [
            key for key in ("uniform_mismatch", "hair_changed") if visual_review.get(key)
        ]
        blocking_failures = [
            key for key in ("face_artifacts", "head_cropped") if visual_review.get(key)
        ]
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
            json.dumps({**identity_review, "initial_review": initial_identity_review}, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        # Keep Qwen's relit subject intact. The old dark-pixel palette mask
        # included neck skin, while source face overlays restored outdoor light.
        if progress_callback:
            progress_callback(91, "Cropping to school passport framing...")
        cropped = identity_locked
        cropped, output_badges_removed = remove_template_badges(cropped, badge_reference)
        logger.info("[UNIFORM_BADGE] Removed %d generated badge(s)", output_badges_removed)
        if progress_callback:
            progress_callback(95, "Verifying Qwen-generated background color...")
        # Qwen frequently returns a visually close blue that passes the
        # tolerance check but is not the selected RGB value. Normalize the
        # border-connected backdrop on every result so export pixels match the
        # picker exactly; the semantic subject mask preserves hair and skin.
        cropped = self._replace_smooth_border_background(cropped, background_color)
        if not self._has_selected_solid_background(cropped, background_color):
            flat_backdrop = self._has_flat_border_background(cropped)
            logger.warning(
                "[UNIFORM_BACKGROUND] Qwen backdrop mismatch (flat=%s); normalizing border-connected backdrop",
                flat_backdrop,
            )
            if progress_callback:
                progress_callback(96, "Applying the selected solid background color...")
            # Qwen commonly produces the correct studio hue with mild optical
            # shading. Normalize only border-connected backdrop pixels first;
            # semantic subject protection prevents holes through hair/neck.
            background_matches = self._has_selected_solid_background(cropped, background_color)
            if not background_matches and not flat_backdrop:
                # Reserve segmentation for genuine retained scenery. It is
                # less reliable around hair gaps than connected-color repair.
                cropped = self._replace_background_with_foreground_matte(
                    cropped,
                    background_color=background_color,
                    job_id=f"{job_id}_background",
                    preserve_foreground_rgb=True,
                    strict=True,
                )
                background_matches = self._has_selected_solid_background(cropped, background_color)
            if not background_matches and not flat_backdrop:
                raise RuntimeError("Uniform background correction did not produce the selected color.")
            if not background_matches:
                logger.info(
                    "[UNIFORM_BACKGROUND] Retained protected subject edges on an otherwise flat selected backdrop"
                )
            logger.info("[UNIFORM_BACKGROUND] Applied exact selected background without changing subject RGB")
        else:
            logger.info("[UNIFORM_BACKGROUND] Qwen generated the selected backdrop directly")
        # Keep Qwen's accepted face and head pixels intact. Finishing is limited
        # to measured garment colour and detected dark-hair glare; post-Qwen
        # face contrast/sharpening made eyes, brows and skin look over-processed.
        from pipelines.uniform_finishing import finish_uniform_tones
        finishing_labels = parse_body_parts(np.array(cropped.convert("RGB")))["labels"]
        cropped, finishing = finish_uniform_tones(
            cropped, finishing_labels,
            make_outer_black=False,
            correct_dark_hair=correct_generated_hair_glare,
            restore_head_detail=False,
            outer_target_rgb=outer_target_rgb,
            # Qwen already relights the face as part of the same generative
            # edit. Matching it back to outdoor source luma can undo that work.
            face_target_luma=None,
        )
        logger.info("[UNIFORM_FINISH] %s", finishing)
        # Preserve the raw generation; publish the background-corrected crop.
        output_path = OUTPUTS_DIR / f"{job_id}_uniform.png"
        cropped.save(output_path, format="PNG", dpi=(300, 300))
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
