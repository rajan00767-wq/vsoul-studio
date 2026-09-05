"""
SDXL Pipeline
Handles Stable Diffusion XL for:
- Text-to-image generation
- Image-to-image (style transfer / refinement)
- Inpainting (background replacement, clothing swap)
"""

import gc
import time
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

MODELS_DIR  = Path("models/sdxl")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

MODEL_ID     = "stabilityai/stable-diffusion-xl-base-1.0"
REFINER_ID   = "stabilityai/stable-diffusion-xl-refiner-1.0"
INPAINT_ID   = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"


class SDXLPipeline:
    def __init__(self):
        self._base     = None
        self._refiner  = None
        self._inpaint  = None
        self._device   = None

    # ── Device detection ──────────────────────────────────────────────────────

    def _get_device(self):
        if self._device:
            return self._device
        try:
            import torch
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        except ImportError:
            self._device = "cpu"
        logger.info("SDXL device: %s", self._device)
        return self._device

    def _dtype(self):
        import torch
        return torch.float16 if self._get_device() in ("cuda", "mps") else torch.float32

    # ── Model loading (lazy) ──────────────────────────────────────────────────

    def _load_base(self):
        if self._base:
            return self._base
        import torch
        from diffusers import StableDiffusionXLPipeline

        model_path = MODELS_DIR / "base"
        src = str(model_path) if model_path.exists() else MODEL_ID
        logger.info("Loading SDXL base from: %s", src)

        self._base = StableDiffusionXLPipeline.from_pretrained(
            src,
            torch_dtype=self._dtype(),
            use_safetensors=True,
            cache_dir=str(MODELS_DIR),
        ).to(self._get_device())

        if self._get_device() == "cuda":
            self._base.enable_xformers_memory_efficient_attention()
        self._base.enable_attention_slicing()
        return self._base

    def _load_inpaint(self):
        if self._inpaint:
            return self._inpaint
        import torch
        from diffusers import StableDiffusionXLInpaintPipeline

        model_path = MODELS_DIR / "inpaint"
        src = str(model_path) if model_path.exists() else INPAINT_ID
        logger.info("Loading SDXL inpaint from: %s", src)

        self._inpaint = StableDiffusionXLInpaintPipeline.from_pretrained(
            src,
            torch_dtype=self._dtype(),
            variant="fp16" if self._get_device() == "cuda" else None,
            use_safetensors=True,
            cache_dir=str(MODELS_DIR),
            low_cpu_mem_usage=True,
        )
        if self._get_device() == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if vram_gb <= 10:
                self._inpaint.enable_model_cpu_offload(gpu_id=0)
                self._inpaint.enable_attention_slicing("max")
            else:
                self._inpaint.to("cuda")
        else:
            self._inpaint.to(self._get_device())
        self._inpaint.vae.enable_slicing()
        self._inpaint.vae.enable_tiling()
        return self._inpaint

    # ── Text-to-image ─────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality, watermark, text",
        width: int = 1024,
        height: int = 1024,
        steps: int = 25,
        guidance: float = 7.5,
        seed: Optional[int] = None,
        job_id: str = "gen",
    ) -> Path:
        import torch
        pipe = self._load_base()

        generator = torch.Generator(device=self._get_device())
        if seed is not None:
            generator.manual_seed(seed)

        logger.info("Generating: %s", prompt[:80])
        t0 = time.time()
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        logger.info("Generated in %.1fs", time.time() - t0)

        out = OUTPUTS_DIR / f"{job_id}_sdxl.png"
        result.images[0].save(str(out))
        self._maybe_free()
        return out

    # ── Image-to-image ────────────────────────────────────────────────────────

    def img2img(
        self,
        input_path: str,
        prompt: str,
        negative_prompt: str = "blurry, low quality, watermark",
        strength: float = 0.6,
        steps: int = 30,
        guidance: float = 7.5,
        seed: Optional[int] = None,
        job_id: str = "i2i",
    ) -> Path:
        import torch
        from diffusers import StableDiffusionXLImg2ImgPipeline
        from PIL import Image

        model_path = MODELS_DIR / "base"
        if not model_path.exists():
            raise RuntimeError(
                "Full-frame SDXL regeneration is disabled for identity safety. "
                "Use masked SDXL inpainting for hair or clothing repair."
            )
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            str(model_path),
            torch_dtype=self._dtype(),
            use_safetensors=True,
            cache_dir=str(MODELS_DIR),
            low_cpu_mem_usage=True,
        )

        if self._get_device() == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if vram_gb <= 10:
                pipe.enable_model_cpu_offload(gpu_id=0)
                pipe.enable_attention_slicing("max")
            else:
                pipe.to("cuda")
        else:
            pipe.to(self._get_device())
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        image = Image.open(input_path).convert("RGB")
        # SDXL needs dimensions divisible by 8
        w = (image.width  // 8) * 8
        h = (image.height // 8) * 8
        image = image.resize((w, h))

        generator = torch.Generator(device=self._get_device())
        if seed is not None:
            generator.manual_seed(seed)

        logger.info("img2img: %s | strength=%.2f", prompt[:60], strength)
        t0 = time.time()
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        logger.info("img2img done in %.1fs", time.time() - t0)

        out = OUTPUTS_DIR / f"{job_id}_sdxl_i2i.png"
        result.images[0].save(str(out))
        self._maybe_free()
        return out

    # ── Inpainting (background / clothing) ────────────────────────────────────

    def inpaint(
        self,
        input_path: str,
        mask_path: str,
        prompt: str,
        negative_prompt: str = "blurry, low quality, deformed, ugly",
        strength: float = 0.99,
        steps: int = 30,
        guidance: float = 8.0,
        seed: Optional[int] = None,
        job_id: str = "inp",
    ) -> Path:
        import torch
        from PIL import Image

        pipe = self._load_inpaint()

        image = Image.open(input_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")

        # Align sizes to 8
        w = (image.width  // 8) * 8
        h = (image.height // 8) * 8
        image = image.resize((w, h))
        mask  = mask.resize((w, h))

        generator = torch.Generator(device=self._get_device())
        if seed is not None:
            generator.manual_seed(seed)

        logger.info("Inpainting: %s", prompt[:60])
        t0 = time.time()
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        logger.info("Inpaint done in %.1fs", time.time() - t0)

        out = OUTPUTS_DIR / f"{job_id}_sdxl_inpaint.png"
        result.images[0].save(str(out))
        self._maybe_free()
        return out

    # ── Uniform swap via inpainting ────────────────────────────────────────────

    def uniform_inpaint(
        self,
        person_path: str,
        shirt_mask_path: str,
        uniform_description: str,
        job_id: str = "unif",
    ) -> Path:
        prompt = (
            f"person wearing {uniform_description}, "
            "photorealistic, natural lighting, high quality, detailed fabric texture"
        )
        negative = "blurry, deformed, cartoon, painting, unrealistic, naked, nude"
        return self.inpaint(
            input_path=person_path,
            mask_path=shirt_mask_path,
            prompt=prompt,
            negative_prompt=negative,
            strength=0.95,
            steps=35,
            guidance=8.5,
            job_id=job_id,
        )

    # ── Memory management ─────────────────────────────────────────────────────

    def _maybe_free(self):
        try:
            import torch
            if self._get_device() == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    def unload(self):
        self._base    = None
        self._refiner = None
        self._inpaint = None
        self._maybe_free()
        logger.info("SDXL models unloaded")

    def is_available(self) -> bool:
        return (
            (MODELS_DIR / "base").exists()
            or (MODELS_DIR / "inpaint").exists()
            or _hf_cache_exists()
        )


def _hf_cache_exists() -> bool:
    import os
    cache = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    return any(cache.glob("*stable-diffusion-xl-base*")) if cache.exists() else False
