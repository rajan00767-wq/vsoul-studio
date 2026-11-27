"""Authoritative Generation Configuration for V11 Uniform Swap.

Centralizes and enforces all pipeline parameters:
- Model path & quantization
- Lightning LoRA & adapter strength
- Scheduler, steps & true CFG
- Spatial resolutions (conditioning, generation, output)
- Background color & seed
- Conditioning strategy & region masks
- Latent identity lock & RGB restoration modes
- Accessory preservation mode
- Caching & deterministic reproducibility hashing
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path("models/qwen_image_edit")
DEFAULT_GGUF_PATH = DEFAULT_MODELS_DIR / "qwen-image-edit-2511-Q4_K_M.gguf"
DEFAULT_CONFIG_PATH = Path("config/qwen_image_edit_2511_transformer.json")
DEFAULT_LORA_FILE = Path("models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
DEFAULT_LORA_FILE_8 = Path("models/loras/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors")


@dataclass
class UniformSwapConfig:
    """Strongly-typed, single authoritative configuration for Uniform Swap."""

    # 1. Model & LoRA Parameters
    model_path: Path = DEFAULT_GGUF_PATH
    quantization: str = "Q4_K_M"
    lora_path: Path = DEFAULT_LORA_FILE
    adapter_strength: float = 1.0
    dtype: str = "bfloat16"

    # 2. Diffusion Schedule & Guidance
    steps: int = 4
    scheduler: str = "euler"
    true_cfg_scale: float = 1.20

    # 3. Spatial Dimensions
    generation_resolution: Tuple[int, int] = (800, 1024)
    conditioning_resolution: int = 512
    output_resolution: Tuple[int, int] = (800, 1024)

    # 4. Environment & Color
    background_color: Tuple[int, int, int] = (255, 255, 255)
    seed: int = 42

    # 5. Prompts
    positive_prompt: Optional[str] = None
    negative_prompt: Optional[str] = None

    # 6. Conditioning Strategy:
    # "original_baseline" (Strategy A): clean photograph on white backdrop via full-person BiRefNet alpha
    # "fitted_seed" (Strategy D): seats rough uniform template into garment area
    # "neutral_torso" (Strategy B): neutral torso grey, strictly outside face/neck/hair/accessories
    conditioning_strategy: str = "original_baseline"

    # 7. Latent Protection & RGB Restoration Modes
    # latent_identity_lock: "off", "soft", "hard"
    latent_identity_lock: str = "off"
    # rgb_identity_restoration: "off", "adaptive", "protected_source", "legacy"
    # In V12.4, defaults to "adaptive" (Protected-Source Garment Compositing)
    rgb_identity_restoration: str = "adaptive"
    compositor_version: str = "v13_3_4"  # "v13_3_4" (continuous skin), "v13_3_3" (anatomical registration) or "v13_3_2" (legacy flat-paste)

    # 8. Accessory & Badge Modes:
    accessory_preservation_mode: str = "semantic_context"
    # badge_policy: "remove_original_keep_target", "no_badge", "preserve_original"
    badge_policy: str = "remove_original_keep_target"

    # Scanned Photo & Framing Controls (Section 4 & 5)
    scanned_photo_mode: str = "auto"  # "auto", "always", "off"
    passport_framing: bool = True     # Upper-chest passport framing crop

    # 9. Diagnostics & Caching
    debug_mode: bool = True
    raw_diffusion_cache_policy: str = "disabled"  # "disabled", "hash_matched", "always"

    # 10. Execution mode: "production", "raw_only", "diagnostic"
    execution_mode: str = "production"
    diagnostic_mode: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.model_path, str):
            self.model_path = Path(self.model_path)
        if isinstance(self.lora_path, str):
            self.lora_path = Path(self.lora_path)
        if isinstance(self.generation_resolution, list):
            self.generation_resolution = tuple(self.generation_resolution)
        if isinstance(self.output_resolution, list):
            self.output_resolution = tuple(self.output_resolution)
        if isinstance(self.background_color, list):
            self.background_color = tuple(self.background_color)
        elif isinstance(self.background_color, str):
            self.background_color = self.parse_bg_color(self.background_color)

    @staticmethod
    def parse_bg_color(val: Union[str, Tuple[int, int, int], List[int]]) -> Tuple[int, int, int]:
        """Parse color into normalized (R, G, B) tuple.

        Supports:
        - Named colors: 'white', 'blue', 'standard_blue', 'red', 'standard_red', 'green', 'standard_green', 'grey', 'gray'
        - Hex codes: '#047EF6', '047EF6', '#DD6A61', '#77BF91', '#FFF', etc.
        - RGB strings: '4,126,246', 'rgb(4, 126, 246)', '255, 255, 255'
        - Sequences: (4, 126, 246), [4, 126, 246]
        """
        if isinstance(val, (tuple, list)):
            if len(val) >= 3:
                return (
                    max(0, min(255, int(val[0]))),
                    max(0, min(255, int(val[1]))),
                    max(0, min(255, int(val[2]))),
                )
            return (255, 255, 255)

        s = str(val).strip().lower()
        if not s:
            return (255, 255, 255)

        # Named colors
        if s == "white":
            return (255, 255, 255)
        if s in ("blue", "standard_blue"):
            return (4, 126, 246)
        if s in ("red", "standard_red"):
            return (221, 106, 97)
        if s in ("green", "standard_green"):
            return (119, 191, 145)
        if s in ("grey", "gray"):
            return (214, 219, 225)

        # Hex formats: #RRGGBB, RRGGBB, #RGB, RGB
        hex_clean = s.lstrip("#").strip()
        if len(hex_clean) == 6 and all(c in "0123456789abcdef" for c in hex_clean):
            try:
                r = int(hex_clean[0:2], 16)
                g = int(hex_clean[2:4], 16)
                b = int(hex_clean[4:6], 16)
                return (r, g, b)
            except Exception:
                pass
        elif len(hex_clean) == 3 and all(c in "0123456789abcdef" for c in hex_clean):
            try:
                r = int(hex_clean[0] * 2, 16)
                g = int(hex_clean[1] * 2, 16)
                b = int(hex_clean[2] * 2, 16)
                return (r, g, b)
            except Exception:
                pass

        # Comma-separated or rgb(...)
        if s.startswith("rgb(") and s.endswith(")"):
            s = s[4:-1]
        if "," in s:
            try:
                parts = [int(p.strip()) for p in s.split(",")][:3]
                if len(parts) == 3:
                    return (
                        max(0, min(255, parts[0])),
                        max(0, min(255, parts[1])),
                        max(0, min(255, parts[2])),
                    )
            except Exception:
                pass

        return (255, 255, 255)

    def validate(self) -> None:
        """Enforce parameter invariants and verify file availability."""
        if self.steps <= 4:
            if not self.lora_path.exists():
                raise FileNotFoundError(
                    f"[CONFIG_ERROR] Lightning 4-step inference requested ({self.steps} steps), "
                    f"but Lightning LoRA file is missing at: {self.lora_path.resolve()}"
                )
        if self.conditioning_resolution not in (512, 768, 1024):
            logger.warning(
                "[CONFIG_WARNING] conditioning_resolution=%d is non-standard; standard benchmarks are 512, 768, 1024",
                self.conditioning_resolution,
            )
        if self.latent_identity_lock not in ("off", "soft", "hard"):
            raise ValueError(f"Invalid latent_identity_lock: {self.latent_identity_lock}; must be 'off', 'soft', or 'hard'")
        if self.rgb_identity_restoration not in ("off", "adaptive", "legacy"):
            raise ValueError(f"Invalid rgb_identity_restoration: {self.rgb_identity_restoration}; must be 'off', 'adaptive', or 'legacy'")
        if self.conditioning_strategy not in ("original_baseline", "fitted_seed", "neutral_torso", "A", "B", "D"):
            raise ValueError(f"Invalid conditioning_strategy: {self.conditioning_strategy}")

    def to_dict(self) -> Dict[str, Any]:
        """Export serializable dictionary representation."""
        d = asdict(self)
        d["model_path"] = str(self.model_path)
        d["lora_path"] = str(self.lora_path)
        return d

    def compute_cache_hash(
        self,
        person_hash: str,
        uniform_hash: str,
        image_1_hash: str,
        image_2_hash: str,
        edit_mask_hash: str,
        hard_mask_hash: str,
        prompt_hash: str,
    ) -> str:
        """Compute an authoritative cryptographic hash identifying this exact run.
        
        Section 11 requirement:
        Include hashes of:
        - Original portrait
        - Uniform reference
        - Exact Qwen Image 1
        - Exact Qwen Image 2
        - Edit mask
        - Hard protection mask
        - Positive & negative prompts
        - Model & LoRA configuration
        - Scheduler
        - Seed
        - Steps
        - CFG
        - Resolution
        """
        hasher = hashlib.sha256()
        components = [
            f"person:{person_hash}",
            f"uniform:{uniform_hash}",
            f"image1:{image_1_hash}",
            f"image2:{image_2_hash}",
            f"edit_mask:{edit_mask_hash}",
            f"hard_mask:{hard_mask_hash}",
            f"prompt:{prompt_hash}",
            f"model:{self.model_path.name}:{self.quantization}",
            f"lora:{self.lora_path.name}:{self.adapter_strength}",
            f"sched:{self.scheduler}",
            f"seed:{self.seed}",
            f"steps:{self.steps}",
            f"cfg:{self.true_cfg_scale:.3f}",
            f"res:{self.generation_resolution[0]}x{self.generation_resolution[1]}",
            f"cond_res:{self.conditioning_resolution}",
            f"strat:{self.conditioning_strategy}",
            f"latent_lock:{self.latent_identity_lock}",
            f"rgb_rest:{self.rgb_identity_restoration}",
        ]
        for c in components:
            hasher.update(c.encode("utf-8"))
        return hasher.hexdigest()

    @classmethod
    def from_job_metadata(
        cls,
        metadata: Dict[str, Any],
        requested_width: Optional[int] = None,
        requested_height: Optional[int] = None,
    ) -> UniformSwapConfig:
        """Construct validated configuration from API / job metadata, cleanly resolving defaults."""
        bg_color_raw = metadata.get("bg_color") or "white"
        bg_rgb = cls.parse_bg_color(str(bg_color_raw))

        # Resolve resolution: respect valid requests or use default 800x1024
        w = int(metadata.get("width") or requested_width or 800)
        h = int(metadata.get("height") or requested_height or 1024)
        if w < 512 or h < 512:
            w, h = 800, 1024

        strategy = str(metadata.get("conditioning_strategy") or metadata.get("strategy") or "original_baseline")
        latent_lock = str(metadata.get("latent_identity_lock") or metadata.get("latent_lock") or "off").lower()
        rgb_rest = str(metadata.get("rgb_identity_restoration") or metadata.get("rgb_restoration") or "adaptive").lower()
        acc_mode = str(metadata.get("accessory_preservation_mode") or "semantic_context").lower()
        badge_pol = str(metadata.get("badge_policy") or "remove_original_keep_target").lower()
        exec_mode = str(metadata.get("execution_mode") or "production").lower()

        steps = int(metadata.get("steps") or 4)
        cfg = float(metadata.get("true_cfg_scale") or metadata.get("cfg_scale") or 1.20)
        seed = int(metadata.get("seed") or 42)
        cond_res = int(metadata.get("conditioning_resolution") or 512)

        cache_policy = str(metadata.get("raw_diffusion_cache_policy") or "disabled")

        cfg_inst = cls(
            steps=steps,
            true_cfg_scale=cfg,
            seed=seed,
            generation_resolution=(w, h),
            conditioning_resolution=cond_res,
            output_resolution=(w, h),
            background_color=bg_rgb,
            conditioning_strategy=strategy,
            latent_identity_lock=latent_lock,
            rgb_identity_restoration=rgb_rest,
            accessory_preservation_mode=acc_mode,
            badge_policy=badge_pol,
            execution_mode=exec_mode,
            raw_diffusion_cache_policy=cache_policy,
        )
        cfg_inst.validate()
        return cfg_inst
