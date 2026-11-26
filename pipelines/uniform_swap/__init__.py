"""VSoul Studio Modular Uniform Swap Architecture (V11).

Provides decoupled, single-responsibility modules:
- config: Authoritative UniformSwapConfig
- semantic_parser: Single-pass semantic parsing with caching
- identity_extractor: Garment-free identity layer extraction & semantic accessories
- conditioning_builder: Qwen Image 1 baseline & Image 2 reference builder
- uniform_generator: Qwen diffusion generation with LoRA verification & raw decode
- neck_collar_resolver: Anatomical collar seating & skin boundary resolution
- compositor: Quality-governed localized compositing & accessory repair
- quality_validator: Biometric ArcFace and landmark RMSE validation
- orchestrator: Authoritative end-to-end pipeline orchestrator
"""

import os
from pathlib import Path

from pipelines.uniform_swap.config import UniformSwapConfig
from pipelines.uniform_swap.semantic_parser import semantic_parser
from pipelines.uniform_swap.identity_extractor import identity_extractor
from pipelines.uniform_swap.conditioning_builder import conditioning_builder
from pipelines.uniform_swap.uniform_generator import uniform_generator
from pipelines.uniform_swap.neck_collar_resolver import neck_collar_resolver
from pipelines.uniform_swap.compositor import uniform_compositor
from pipelines.uniform_swap.quality_validator import quality_validator
from pipelines.uniform_swap.orchestrator import UniformSwapOrchestrator, uniform_swap_orchestrator

# Feature flags for runtime control & regression benchmarking
VSOUL_ENABLE_CLEAN_IDENTITY_EXTRACTION: bool = (
    os.getenv("VSOUL_ENABLE_CLEAN_IDENTITY_EXTRACTION", "1").strip().lower() in ("1", "true", "yes")
)
VSOUL_CONDITIONING_STRATEGY: str = os.getenv("VSOUL_CONDITIONING_STRATEGY", "original_baseline").strip()
VSOUL_ENABLE_SYNTHETIC_NECK_INFILL: bool = (
    os.getenv("VSOUL_ENABLE_SYNTHETIC_NECK_INFILL", "0").strip().lower() in ("1", "true", "yes")
)
VSOUL_ENABLE_COLLAR_WARP: bool = (
    os.getenv("VSOUL_ENABLE_COLLAR_WARP", "0").strip().lower() in ("1", "true", "yes")
)
VSOUL_ENABLE_POST_VL_REVIEW: bool = (
    os.getenv("VSOUL_ENABLE_POST_VL_REVIEW", "1").strip().lower() in ("1", "true", "yes")
)
VSOUL_REUSE_RAW_DIFFUSION: bool = (
    os.getenv("VSOUL_REUSE_RAW_DIFFUSION", "0").strip().lower() in ("1", "true", "yes")
)

__all__ = [
    "UniformSwapConfig",
    "UniformSwapOrchestrator",
    "uniform_swap_orchestrator",
    "semantic_parser",
    "identity_extractor",
    "conditioning_builder",
    "uniform_generator",
    "neck_collar_resolver",
    "uniform_compositor",
    "quality_validator",
    "VSOUL_ENABLE_CLEAN_IDENTITY_EXTRACTION",
    "VSOUL_CONDITIONING_STRATEGY",
    "VSOUL_ENABLE_SYNTHETIC_NECK_INFILL",
    "VSOUL_ENABLE_COLLAR_WARP",
    "VSOUL_ENABLE_POST_VL_REVIEW",
    "VSOUL_REUSE_RAW_DIFFUSION",
]
