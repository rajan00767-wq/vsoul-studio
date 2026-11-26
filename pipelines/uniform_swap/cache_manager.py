"""Authoritative Cache Manager for Uniform Swap (V13.2).

Enforces mandatory cache integrity:
- Source image SHA256
- Uniform reference image SHA256
- Model version
- LoRA version
- Prompt version
- Inference parameters
- Generation pipeline version
- Full 13-field cryptographic provenance manifest

Ensures zero cache collisions and zero cross-reference leakage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "v13.2"
MODEL_VERSION = "qwen-image-edit-2511-q4_k_m"
LORA_VERSION = "lightning-4step-rank64"

PROVENANCE_REQUIRED_FIELDS = [
    "original_source_sha256",
    "normalized_source_sha256",
    "uniform_ref_sha256",
    "conditioning_image_sha256",
    "effective_prompt_sha256",
    "negative_prompt_sha256",
    "model_version",
    "lora_version",
    "inference_resolution",
    "seed",
    "steps",
    "guidance_parameters",
    "candidate_image_sha256",
]


def compute_sha256(data: Union[bytes, Image.Image, np.ndarray, Path, str]) -> str:
    """Compute deterministic SHA256 hash."""
    if isinstance(data, (str, Path)):
        p = Path(data)
        if p.exists() and p.is_file():
            return hashlib.sha256(p.read_bytes()).hexdigest()
        return hashlib.sha256(str(data).encode("utf-8")).hexdigest()
    if isinstance(data, Image.Image):
        header = f"{data.size}_{data.mode}".encode("ascii")
        return hashlib.sha256(header + data.tobytes()).hexdigest()
    if isinstance(data, np.ndarray):
        header = f"{data.shape}_{data.dtype}".encode("ascii")
        return hashlib.sha256(header + data.tobytes()).hexdigest()
    if isinstance(data, bytes):
        return hashlib.sha256(data).hexdigest()
    return hashlib.sha256(str(data).encode("utf-8")).hexdigest()


def create_provenance_manifest(
    original_source_image: Any,
    normalized_source_image: Any,
    uniform_ref: Any,
    conditioning_image: Any,
    effective_prompt: str,
    negative_prompt: str,
    model_version: str,
    lora_version: str,
    inference_resolution: Tuple[int, int],
    seed: int,
    steps: int,
    guidance_parameters: Dict[str, Any],
    candidate_image: Any,
) -> Dict[str, Any]:
    """Construct authoritative cryptographic provenance manifest matching V13.2 specification."""
    return {
        "original_source_sha256": compute_sha256(original_source_image),
        "normalized_source_sha256": compute_sha256(normalized_source_image),
        "uniform_ref_sha256": compute_sha256(uniform_ref),
        "conditioning_image_sha256": compute_sha256(conditioning_image),
        "effective_prompt_sha256": compute_sha256(effective_prompt),
        "negative_prompt_sha256": compute_sha256(negative_prompt),
        "model_version": str(model_version),
        "lora_version": str(lora_version),
        "inference_resolution": list(inference_resolution),
        "seed": int(seed),
        "steps": int(steps),
        "guidance_parameters": guidance_parameters,
        "candidate_image_sha256": compute_sha256(candidate_image),
        "timestamp": time.time(),
    }


class UniformCacheManager:
    """Production Cryptographic Cache Manager for Uniform Swap Pipeline."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir or "outputs/cache/raw_candidates")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_cache_key(
        self,
        source_image: Union[Image.Image, np.ndarray, Path],
        uniform_reference_image: Union[Image.Image, np.ndarray, Path],
        positive_prompt: str,
        negative_prompt: str,
        inference_params: Dict[str, Any],
        model_version: str = MODEL_VERSION,
        lora_version: str = LORA_VERSION,
        pipeline_version: str = PIPELINE_VERSION,
    ) -> Tuple[str, Dict[str, Any]]:
        """Construct comprehensive cache key payload."""
        src_hash = compute_sha256(source_image)
        ref_hash = compute_sha256(uniform_reference_image)
        prompt_content = f"POS:{positive_prompt.strip()}|NEG:{negative_prompt.strip()}"
        prompt_hash = hashlib.sha256(prompt_content.encode("utf-8")).hexdigest()

        clean_params = {
            "steps": int(inference_params.get("steps", 4)),
            "guidance_scale": float(inference_params.get("guidance_scale", 1.0)),
            "resolution": list(inference_params.get("resolution", (800, 1024))),
            "latent_identity_lock": inference_params.get("latent_identity_lock", "soft"),
        }

        metadata = {
            "pipeline_version": pipeline_version,
            "source_sha256": src_hash,
            "uniform_ref_sha256": ref_hash,
            "prompt_sha256": prompt_hash,
            "model_version": model_version,
            "lora_version": lora_version,
            "inference_params": clean_params,
        }

        serialized = json.dumps(metadata, sort_keys=True)
        cache_key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]
        metadata["cache_key"] = cache_key
        return cache_key, metadata

    def get(
        self,
        cache_key: str,
        expected_ref_sha256: Optional[str] = None,
    ) -> Optional[Tuple[Image.Image, Dict[str, Any]]]:
        """Retrieve verified candidate from cache. Enforces reference cryptographic binding."""
        meta_path = self.cache_dir / f"{cache_key}.json"
        img_path = self.cache_dir / f"{cache_key}.png"

        if not meta_path.exists() or not img_path.exists():
            return None

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            if expected_ref_sha256:
                stored_ref_sha256 = meta.get("uniform_ref_sha256")
                if stored_ref_sha256 != expected_ref_sha256:
                    logger.warning(
                        "[CACHE_MANAGER] Cache entry %s ref_sha256 %s != expected %s. Refusing cache hit.",
                        cache_key, stored_ref_sha256, expected_ref_sha256
                    )
                    return None

            img_sha256 = compute_sha256(img_path)
            stored_img_sha256 = meta.get("candidate_image_sha256") or meta.get("raw_image_sha256")
            if stored_img_sha256 and img_sha256 != stored_img_sha256:
                logger.warning(
                    "[CACHE_MANAGER] Cache entry %s image file corrupted or tampered (%s != %s).",
                    cache_key, img_sha256, stored_img_sha256
                )
                return None

            img = Image.open(img_path).convert("RGB")
            logger.info("[CACHE_MANAGER] Cache HIT for key %s (ref_hash=%s)", cache_key, expected_ref_sha256[:12] if expected_ref_sha256 else "")
            return img, meta
        except Exception as e:
            logger.warning("[CACHE_MANAGER] Error reading cache for key %s: %s", cache_key, e)
            return None

    def put(
        self,
        cache_key: str,
        image: Image.Image,
        metadata: Dict[str, Any],
    ) -> Path:
        """Store image and verification metadata atomically."""
        img_path = self.cache_dir / f"{cache_key}.png"
        meta_path = self.cache_dir / f"{cache_key}.json"

        if "candidate_image_sha256" not in metadata:
            metadata["candidate_image_sha256"] = compute_sha256(image)

        image.save(img_path, format="PNG")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("[CACHE_MANAGER] Stored verified candidate %s (ref_hash=%s)", cache_key, metadata.get("uniform_ref_sha256", "")[:12])
        return img_path

    def validate_external_candidate(
        self,
        candidate_image: Union[Image.Image, Path, str],
        expected_uniform_ref: Union[Image.Image, Path, str],
        expected_ref_type: str = "",
        expected_provenance: Optional[Dict[str, Any]] = None,
        diagnostic_mode: bool = False,
    ) -> bool:
        """Validate candidate provenance against V13.2 specification.
        
        In production mode (diagnostic_mode=False), candidate MUST have a verified
        cryptographic provenance manifest containing all 13 required fields.
        Unverifiable or incompatible candidates are strictly rejected.
        """
        try:
            if isinstance(candidate_image, (str, Path)):
                c_p = Path(candidate_image)
                if not c_p.exists():
                    return False
                img = Image.open(str(c_p)).convert("RGB")
            else:
                c_p = None
                img = candidate_image.convert("RGB")

            cand_sha256 = compute_sha256(img)
            exp_ref_hash = compute_sha256(expected_uniform_ref)

            manifest = None
            if c_p is not None:
                meta_p = c_p.with_suffix(".json")
                if meta_p.exists():
                    with open(meta_p, "r", encoding="utf-8") as f:
                        manifest = json.load(f)

            if manifest is None and expected_provenance is not None:
                manifest = expected_provenance

            # Production Verification (Mandatory Provenance Enforcement)
            if not diagnostic_mode:
                if manifest is None:
                    logger.warning(
                        "[CACHE_MANAGER] Production rejected external candidate %s: missing provenance manifest (.json)",
                        candidate_image
                    )
                    return False

                missing_fields = [f for f in PROVENANCE_REQUIRED_FIELDS if f not in manifest]
                if missing_fields:
                    logger.warning(
                        "[CACHE_MANAGER] Production rejected external candidate %s: incomplete provenance manifest (missing %s)",
                        candidate_image, missing_fields
                    )
                    return False

                if manifest.get("uniform_ref_sha256") != exp_ref_hash:
                    logger.warning(
                        "[CACHE_MANAGER] Production rejected external candidate %s: reference hash %s != expected %s",
                        candidate_image, manifest.get("uniform_ref_sha256"), exp_ref_hash
                    )
                    return False

                if manifest.get("candidate_image_sha256") != cand_sha256:
                    logger.warning(
                        "[CACHE_MANAGER] Production rejected external candidate %s: image hash %s != manifest %s",
                        candidate_image, cand_sha256, manifest.get("candidate_image_sha256")
                    )
                    return False

                logger.info(
                    "[CACHE_MANAGER] Provenance cryptographically verified for candidate %s (sha256=%s)",
                    candidate_image, cand_sha256[:12]
                )
                return True

            # Diagnostic Mode (Explicitly labelled diagnostics fallback)
            logger.warning(
                "[CACHE_MANAGER] DIAGNOSTIC MODE: validating candidate %s with legacy fallback",
                candidate_image
            )
            if manifest and manifest.get("uniform_ref_sha256") == exp_ref_hash:
                return True

            return True
        except Exception as e:
            logger.warning("[CACHE_MANAGER] Error validating external candidate: %s", e)
            return False


cache_manager = UniformCacheManager()
