"""
GUSUQ-WebUI Engine (Gradio Unified Simple UI for Qwen-image with Sequential Offload)
Implements low_vram sequential execution for 8GB GPUs (RTX 5050 Laptop).
"""

import time
import os
import gc
import yaml
from pathlib import Path
from typing import Optional, Union, Callable
from PIL import Image
import torch
import numpy as np

from utils.logger import get_logger

logger = get_logger("gusuq_pipeline")

CONFIG_PATH = Path("config/opt_pol.yaml")
MODELS_DIR = Path("models/qwen_image_edit")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)


class GUSUQPipeline:
    """GUSUQ Unified Qwen-Image Engine with Low-VRAM Sequential Model Swapping."""

    def __init__(self, opt_policy: str = "low_vram", model_variant: str = "2509"):
        self.opt_policy_name = opt_policy
        self.model_variant = model_variant
        self.policy = self._load_policy(opt_policy)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.tokenizer = None
        self.processor = None
        self.text_encoder = None
        self.transformer = None
        self.vae = None
        self.scheduler = None
        self.pipe = None
        self.is_nunchaku = False
        self.is_lightning = False

        logger.info("[GUSUQ] Initialized with policy '%s' on %s", opt_policy, self.device)

    def _load_policy(self, name: str) -> dict:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                policies = yaml.safe_load(f)
                if name in policies:
                    return policies[name]
        return {
            "sequential_offload": True,
            "empty_cache_after_step": True,
            "max_sequence_length": 128,
            "default_inference_steps": 4,
        }

    def _load_text_components(self):
        """Loads tokenizer, processor, and 4-bit vision-language encoder."""
        if self.text_encoder is not None:
            return

        from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
        logger.info("[GUSUQ] Loading 4-bit vision-language text encoder...")

        self.tokenizer = AutoTokenizer.from_pretrained(str(MODELS_DIR / "tokenizer"))
        self.processor = AutoProcessor.from_pretrained(str(MODELS_DIR / "processor"))

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.dtype,
            bnb_4bit_use_double_quant=True,
        )
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(MODELS_DIR / "text_encoder"),
            quantization_config=bnb_config,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            torch_dtype=self.dtype,
        )

    def _load_diffusion_components(self):
        """Loads VAE, Scheduler, and GGUF DiT Transformer."""
        if self.transformer is not None:
            return

        from diffusers import (
            AutoencoderKLQwenImage,
            FlowMatchEulerDiscreteScheduler,
            QwenImageTransformer2DModel,
            GGUFQuantizationConfig,
        )
        logger.info("[GUSUQ] Loading VAE and FlowMatch Euler Scheduler...")
        self.vae = AutoencoderKLQwenImage.from_pretrained(str(MODELS_DIR / "vae"), torch_dtype=self.dtype)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(MODELS_DIR / "scheduler"))

        # Four inference steps require the matching Lightning checkpoint.  The
        # regular 2509 model needs a much longer schedule and otherwise leaves
        # fine regions such as hair and clothing partially denoised.
        if self.model_variant == "2509":
            svdq_candidates = [
                MODELS_DIR / "svdq-fp4_r32-qwen-image-edit-2509.safetensors",
                MODELS_DIR / "svdq-fp4_r32-qwen-image-edit-lightningv1.0-4steps.safetensors",
            ]
        else:
            svdq_candidates = [
                MODELS_DIR / "svdq-fp4_r32-qwen-image-edit-lightningv1.0-4steps.safetensors",
                MODELS_DIR / "svdq-fp4_r32-qwen-image-edit-2509.safetensors",
            ]
        svdq_path = next((p for p in svdq_candidates if p.exists()), None)
        gguf_path = MODELS_DIR / "qwen-image-edit-2511-Q4_K_M.gguf"
        snapshot_cfg = Path("models/models--Qwen--Qwen-Image-Edit-2511/snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9/transformer/config.json")

        if svdq_path is not None:
            try:
                from nunchaku import NunchakuQwenImageTransformer2DModel
                logger.info("[GUSUQ] 🚀 Loading Native Nunchaku SVDQuant FP4 DiT Transformer from %s...", svdq_path.name)
                self.transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
                    str(svdq_path),
                    torch_dtype=self.dtype,
                )
                self.is_nunchaku = True
                self.is_lightning = "lightning" in svdq_path.name.lower()
                logger.info("[GUSUQ] ✅ Nunchaku SVDQuant 2509 Transformer loaded natively into GPU!")
                return
            except Exception as e:
                logger.warning("[GUSUQ] Failed to load Nunchaku transformer: %s. Falling back to GGUF.", e)

        self.is_nunchaku = False
        self.is_lightning = False
        logger.info("[GUSUQ] Loading GGUF DiT Transformer into memory...")
        quant_cfg = GGUFQuantizationConfig(compute_dtype=self.dtype)
        self.transformer = QwenImageTransformer2DModel.from_single_file(
            str(gguf_path),
            config=str(snapshot_cfg),
            quantization_config=quant_cfg,
            torch_dtype=self.dtype,
        )

    def _get_pipe(self):
        """Builds and caches the QwenImageEditPlusPipeline with CPU offload."""
        if self.pipe is not None:
            return self.pipe

        self._load_text_components()
        self._load_diffusion_components()

        from diffusers import QwenImageEditPlusPipeline

        pipe = QwenImageEditPlusPipeline(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            transformer=self.transformer,
            vae=self.vae,
            scheduler=self.scheduler,
            processor=self.processor,
        )

        # Only the pre-baked Nunchaku Lightning checkpoint already contains
        # the four-step adaptation.
        if not getattr(self, "is_lightning", False):
            lora_file = Path("models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
            if lora_file.exists():
                try:
                    pipe.load_lora_weights(str(lora_file.parent), weight_name=lora_file.name, adapter_name="lightning")
                    logger.info("[GUSUQ] Attached 4-Step Lightning LoRA")
                except Exception as e:
                    logger.warning("[GUSUQ] LoRA attach error: %s", e)

        if torch.cuda.is_available():
            pipe.enable_model_cpu_offload(gpu_id=0)
        else:
            pipe.to("cpu")

        self.pipe = pipe
        return self.pipe

    def edit_image(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted, artifacts, bad anatomy, deformed",
        background_color: str = "white",
        true_cfg_scale: float = 1.0,
        steps: int = 4,
        upscale_factor: int = 1,
        use_birefnet_background: bool = True,
        preserve_source_clothing: bool = True,
        max_generation_dimension: int = 576,
        minimum_generation_dimension: int = 0,
        keep_generation_resolution: bool = False,
        face_detail_refine: bool = False,
        seed: int = 42,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> Path:
        """
        Executes genuine decoupled Qwen-Image-Edit Neural Diffusion with studio quality restoration.
        """
        from pipelines.qwen_edit_pipeline import qwen_service

        def adapter_progress(pct: int, msg: str):
            if progress_cb:
                progress_cb(pct / 100.0, msg)

        w, h = image.size
        return qwen_service.qwen_edit_enhancer(
            image_input=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            background_color=background_color,
            use_birefnet_background=use_birefnet_background,
            preserve_source_clothing=preserve_source_clothing,
            max_generation_dimension=max_generation_dimension,
            minimum_generation_dimension=minimum_generation_dimension,
            keep_generation_resolution=keep_generation_resolution,
            seed=seed,
            job_id=f"gusuq_edit_{int(time.time())}",
            width=w,
            height=h,
            steps=steps,
            upscale_factor=max(1, min(int(upscale_factor or 1), 4)),
            true_cfg_scale=true_cfg_scale,
            face_detail_refine=face_detail_refine,
            progress_callback=adapter_progress,
        )

    def edit_uniform_ai_fit(
        self,
        baseline: Image.Image,
        person_reference: Image.Image,
        uniform_reference: Image.Image,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        steps: int = 20,
        seed: int = 42,
    ) -> Image.Image:
        """Run the local Qwen 2509 model for an optional uniform-fit refinement.

        The deterministic baseline remains the source of truth.  Qwen receives
        the baseline, original portrait, and supplied template as separate image
        references; its output is only accepted after face and garment color
        locks are reapplied by the caller.
        """
        from pipelines.qwen_edit_pipeline import _encode_prompt_isolated

        if self.model_variant != "2509":
            raise RuntimeError("AI Fit requires the Qwen Image Edit 2509 model")

        width, height = baseline.size
        width = max(64, (width // 16) * 16)
        height = max(64, (height // 16) * 16)
        baseline_ref = baseline.convert("RGB").resize((width, height), Image.LANCZOS)
        person_ref = person_reference.convert("RGB").resize((width, height), Image.LANCZOS)
        uniform_ref = uniform_reference.convert("RGB")

        prompt = (
            "Image 1 is the exact baseline uniform composite. Image 2 is the original person. "
            "Image 3 is the required uniform reference. Improve only realistic cloth drape, shoulder fit, "
            "collar-to-neck contact, and sleeve alignment in Image 1. Keep Image 2 face, hair, skin, pose, "
            "and eyewear exactly unchanged. Preserve Image 3 uniform color, pattern, badge, logo, buttons, "
            "and garment design exactly. Do not add glasses, ties, jewelry, extra collars, or layered clothing."
        )
        if progress_cb:
            progress_cb(0.35, "Encoding Qwen 2509 person and uniform references...")
        prompt_embeds, prompt_mask = _encode_prompt_isolated(
            [baseline_ref, person_ref, uniform_ref], prompt, max_sequence_length=192, timeout_seconds=150,
        )

        self._load_diffusion_components()
        from transformers import AutoTokenizer, AutoProcessor
        from diffusers import QwenImageEditPlusPipeline

        tokenizer = AutoTokenizer.from_pretrained(str(MODELS_DIR / "tokenizer"))
        processor = AutoProcessor.from_pretrained(str(MODELS_DIR / "processor"))
        pipe = QwenImageEditPlusPipeline(
            tokenizer=tokenizer, text_encoder=None, transformer=self.transformer,
            vae=self.vae, scheduler=self.scheduler, processor=processor,
        )
        if torch.cuda.is_available():
            pipe.enable_model_cpu_offload(gpu_id=0)

        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.dtype)
        if prompt_mask is not None:
            prompt_mask = prompt_mask.to(device=self.device)
        num_steps = max(12, min(int(steps), 24))

        def on_step_end(_pipe, step_index, _timestep, callback_kwargs):
            if progress_cb:
                progress_cb(0.40 + 0.45 * ((step_index + 1) / num_steps), f"Qwen 2509 AI Fit step {step_index + 1}/{num_steps}...")
            return callback_kwargs

        try:
            result = pipe(
                image=[baseline_ref, person_ref, uniform_ref],
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_mask,
                num_inference_steps=num_steps,
                true_cfg_scale=1.0,
                height=height,
                width=width,
                generator=torch.Generator(device="cpu").manual_seed(seed),
                callback_on_step_end=on_step_end,
            ).images[0]
            if not isinstance(result, Image.Image):
                raise RuntimeError("Qwen 2509 returned no image")
            return result.convert("RGB").resize(baseline.size, Image.LANCZOS)
        finally:
            del pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
