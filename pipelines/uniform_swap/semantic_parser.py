"""Unified Single-Pass Semantic Parser with Job-Level In-Memory Caching.

Executes and caches SCHP LIP-20, BiSeNet CelebAMask-HQ, BiRefNet foreground alpha,
and InsightFace facial analysis once per registered image frame. Eliminates redundant
GPU/CPU inference cycles across the pipeline.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _compute_image_hash(img: Union[np.ndarray, Image.Image]) -> str:
    """Compute a fast hash of image dimensions and contents for cache indexing."""
    if isinstance(img, Image.Image):
        arr = np.asarray(img)
    else:
        arr = img
    # Sample center patch and corners for instant deterministic hash
    h, w = arr.shape[:2]
    sample = np.ascontiguousarray(arr[::max(1, h // 32), ::max(1, w // 32)])
    return hashlib.sha256(sample.tobytes()).hexdigest()[:16]


class SemanticParser:
    """Job-scoped cache manager and execution hub for semantic parsing models."""

    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def clear_job_cache(self, job_id: Optional[str] = None) -> None:
        """Clear cached segmentations for a specific job or all jobs."""
        with self._lock:
            if job_id is None:
                self._cache.clear()
            else:
                self._cache.pop(job_id, None)

    def parse_source(
        self,
        image: Union[np.ndarray, Image.Image],
        face_bbox: Optional[Tuple[float, float, float, float]] = None,
        job_id: str = "default",
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Perform unified single-pass segmentation on the source registered image.

        Returns:
            Dict containing:
                - 'schp_labels': (H, W) uint8
                - 'schp_probs': (20, H, W) float32
                - 'bisenet_res': Dict with 'parsing', 'neck_mask', 'neck_bottom_y', etc.
                - 'birefnet_alpha': (H, W) uint8 [0, 255]
                - 'image_rgb': (H, W, 3) uint8
        """
        if isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
            rgb_arr = np.asarray(pil_img, dtype=np.uint8)
        else:
            rgb_arr = image.astype(np.uint8)
            pil_img = Image.fromarray(rgb_arr, "RGB")

        cache_key = f"{job_id}_source_{_compute_image_hash(rgb_arr)}"
        with self._lock:
            if not force_refresh and job_id in self._cache and cache_key in self._cache[job_id]:
                logger.debug("[SEMANTIC_PARSER] Cache hit for source frame (job=%s)", job_id)
                return self._cache[job_id][cache_key]

        logger.info("[SEMANTIC_PARSER] Running single-pass segmentation for source (job=%s)", job_id)

        # 1. SCHP LIP-20 Parse (single pass with full probabilities)
        from pipelines.schp_service import parse as parse_schp
        schp_res = parse_schp(rgb_arr, return_probabilities=True)
        schp_labels = schp_res["labels"].astype(np.uint8)
        schp_probs = schp_res.get("probabilities")

        # 2. BiSeNet CelebAMask-HQ Parse (anatomical neck & shoulders)
        from pipelines.qwen_edit_pipeline import QwenEditPipeline
        if face_bbox is None:
            w, h = pil_img.size
            face_bbox = (w * 0.25, h * 0.15, w * 0.75, h * 0.55)
        bisenet_res = QwenEditPipeline.deduct_neck_and_shoulders_bisenet(pil_img, face_bbox=face_bbox)

        # 3. BiRefNet Foreground Matte (whole person boundary)
        from pipelines.birefnet_service import extract_matte
        birefnet_alpha = extract_matte(pil_img)

        result: Dict[str, Any] = {
            "schp_labels": schp_labels,
            "schp_probs": schp_probs,
            "bisenet_res": bisenet_res,
            "bisenet_parsing": bisenet_res.get("parsing"),
            "birefnet_alpha": birefnet_alpha,
            "image_rgb": rgb_arr,
            "image_pil": pil_img,
        }

        with self._lock:
            if job_id not in self._cache:
                self._cache[job_id] = {}
            self._cache[job_id][cache_key] = result

        return result

    def parse_generated(
        self,
        image: Union[np.ndarray, Image.Image],
        face_bbox: Optional[Tuple[float, float, float, float]] = None,
        job_id: str = "default",
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Perform unified single-pass segmentation on the generated Qwen candidate.

        Returns:
            Dict containing:
                - 'schp_labels': (H, W) uint8
                - 'schp_probs': (20, H, W) float32
                - 'bisenet_res': Dict with 'parsing', 'neck_mask', etc.
                - 'bisenet_parsing': (H, W) uint8
                - 'image_rgb': (H, W, 3) uint8
        """
        if isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
            rgb_arr = np.asarray(pil_img, dtype=np.uint8)
        else:
            rgb_arr = image.astype(np.uint8)
            pil_img = Image.fromarray(rgb_arr, "RGB")

        cache_key = f"{job_id}_gen_{_compute_image_hash(rgb_arr)}"
        with self._lock:
            if not force_refresh and job_id in self._cache and cache_key in self._cache[job_id]:
                logger.debug("[SEMANTIC_PARSER] Cache hit for generated frame (job=%s)", job_id)
                return self._cache[job_id][cache_key]

        logger.info("[SEMANTIC_PARSER] Running single-pass segmentation for generated (job=%s)", job_id)

        from pipelines.schp_service import parse as parse_schp
        schp_res = parse_schp(rgb_arr, return_probabilities=True)
        schp_labels = schp_res["labels"].astype(np.uint8)
        schp_probs = schp_res.get("probabilities")

        from pipelines.qwen_edit_pipeline import QwenEditPipeline
        if face_bbox is None:
            w, h = pil_img.size
            face_bbox = (w * 0.25, h * 0.15, w * 0.75, h * 0.55)
        bisenet_res = QwenEditPipeline.deduct_neck_and_shoulders_bisenet(pil_img, face_bbox=face_bbox)

        result: Dict[str, Any] = {
            "schp_labels": schp_labels,
            "schp_probs": schp_probs,
            "bisenet_res": bisenet_res,
            "bisenet_parsing": bisenet_res.get("parsing"),
            "image_rgb": rgb_arr,
            "image_pil": pil_img,
        }

        with self._lock:
            if job_id not in self._cache:
                self._cache[job_id] = {}
            self._cache[job_id][cache_key] = result

        return result


# Global singleton parser
semantic_parser = SemanticParser()
