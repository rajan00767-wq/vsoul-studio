"""Local SCHP LIP human-part parsing for BiRefNet matte protection."""

import os
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

from integrations.schp.configuration_schp import SCHPConfig
from integrations.schp.image_processing_schp import SCHPImageProcessor
from integrations.schp.modeling_schp import SCHPForSemanticSegmentation
from utils.logger import get_logger


logger = get_logger(__name__)
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "schp-lip-20"
_LOCK = Lock()
_MODEL = None
_PROCESSOR = None
_PARSE_CACHE: dict[tuple, dict[str, Any]] = {}

# LIP-20: clothing vs background. Dress/cloth must never be treated as background.
CLOTHING_LABELS = {5, 6, 7, 10, 11, 12}  # upper, dress, coat, jumpsuit, scarf/tie, skirt
BACKGROUND_LABEL = 0
SOLID_LABELS = {1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
HAIR_LABEL = 2


def available() -> bool:
    return (MODEL_DIR / "model.safetensors").is_file()


def _load(device: Optional[str] = None):
    global _MODEL, _PROCESSOR
    target_device = device or os.environ.get("SCHP_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if _MODEL is not None:
        if str(next(_MODEL.parameters()).device) != target_device:
            with _LOCK:
                _MODEL = _MODEL.to(target_device)
                logger.info("Moved SCHP LIP-20 to %s", target_device)
        return _MODEL, _PROCESSOR
    with _LOCK:
        if _MODEL is None:
            if not available():
                raise FileNotFoundError(f"SCHP checkpoint missing: {MODEL_DIR}")
            config = SCHPConfig.from_pretrained(str(MODEL_DIR), local_files_only=True)
            _PROCESSOR = SCHPImageProcessor.from_pretrained(
                str(MODEL_DIR), local_files_only=True
            )
            _MODEL = SCHPForSemanticSegmentation.from_pretrained(
                str(MODEL_DIR), config=config, local_files_only=True
            ).eval().to(target_device)
            logger.info("Loaded SCHP LIP-20 on %s", target_device)
    return _MODEL, _PROCESSOR


def parse(
    image_rgb: np.ndarray,
    return_probabilities: bool = False,
    device: Optional[str] = None,
    use_cache: bool = True,
    return_all_classes: bool = False,
) -> dict[str, Any]:
    """Return original-resolution LIP labels and conservative protection masks.

    - FIX 7: Supports use_cache=False to bypass cache during benchmarking and dynamic updates.
    - Selectively resizes only requested hair and clothing probability maps by default
      to prevent caching large 20-channel full-resolution tensors in memory.
    """
    h, w = image_rgb.shape[:2]
    sample_key = (
        h, w,
        image_rgb[0, 0].tobytes(),
        image_rgb[h // 2, w // 2].tobytes(),
        image_rgb[-1, -1].tobytes(),
        int(image_rgb.sum()) % 1000000007,
        return_probabilities,
        return_all_classes,
    )
    if use_cache:
        with _LOCK:
            if sample_key in _PARSE_CACHE:
                return _PARSE_CACHE[sample_key]

    model, processor = _load(device=device)
    target_device = next(model.parameters()).device
    image = Image.fromarray(image_rgb, "RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(target_device) for k, v in inputs.items()}
    with _LOCK, torch.inference_mode():
        outputs = model(**inputs)
    logits = getattr(outputs, "parsing_logits", None)
    if logits is None:
        logits = outputs.logits
    labels = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    labels = cv2.resize(
        labels, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    solid = np.isin(labels, tuple(SOLID_LABELS)).astype(np.uint8)
    hair = (labels == HAIR_LABEL).astype(np.uint8)
    # Eroded cores prevent class-boundary errors from turning into hard matte edges.
    solid_core = cv2.erode(
        solid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1
    ).astype(bool)
    hair_core = cv2.erode(
        hair, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1
    ).astype(bool)
    res: dict[str, Any] = {"labels": labels, "solid_core": solid_core, "hair_core": hair_core}

    if return_probabilities:
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (20, 473, 473)
        # Selectively interpolate only hair and aggregated clothing classes directly
        hair_slice = probs[HAIR_LABEL]
        cloth_slice = np.sum(probs[list(CLOTHING_LABELS)], axis=0)
        res["hair_prob"] = cv2.resize(hair_slice, (w, h), interpolation=cv2.INTER_LINEAR)
        res["clothing_prob"] = cv2.resize(cloth_slice, (w, h), interpolation=cv2.INTER_LINEAR)

        if return_all_classes:
            probs_resized = np.zeros((20, h, w), dtype=np.float32)
            for c in range(20):
                probs_resized[c] = cv2.resize(probs[c], (w, h), interpolation=cv2.INTER_LINEAR)
            res["probabilities"] = probs_resized

    if use_cache:
        with _LOCK:
            if len(_PARSE_CACHE) >= 16:
                _PARSE_CACHE.pop(next(iter(_PARSE_CACHE)))
            _PARSE_CACHE[sample_key] = res
    return res


def benchmark_devices(
    devices: tuple[str, ...] = ("cpu", "cuda:0"),
    num_runs: int = 3,
    image_shape: tuple[int, int] = (1024, 800),
) -> dict[str, dict[str, float]]:
    """Benchmark SCHP inference latency and memory across devices (CPU vs CUDA).

    FIX 7:
    - Bypasses cache by setting use_cache=False to measure real model latency.
    - Resets CUDA peak memory statistics before measuring.
    """
    import time
    results = {}
    dummy_img = np.random.randint(0, 255, (image_shape[0], image_shape[1], 3), dtype=np.uint8)

    for dev in devices:
        if dev.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("[SCHP_BENCHMARK] %s requested but CUDA is unavailable", dev)
            continue
        try:
            model, proc = _load(device=dev)
            if dev.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(dev)
                torch.cuda.synchronize(dev)

            # Warm up without cache
            _ = parse(dummy_img, return_probabilities=True, device=dev, use_cache=False)
            if dev.startswith("cuda"):
                torch.cuda.synchronize(dev)

            latencies = []
            for _ in range(num_runs):
                t0 = time.perf_counter()
                _ = parse(dummy_img, return_probabilities=True, device=dev, use_cache=False)
                if dev.startswith("cuda"):
                    torch.cuda.synchronize(dev)
                latencies.append((time.perf_counter() - t0) * 1000.0)

            mean_latency = float(np.mean(latencies))
            mem_mb = 0.0
            if dev.startswith("cuda"):
                mem_mb = float(torch.cuda.max_memory_allocated(dev) / (1024 * 1024))
            results[dev] = {"mean_latency_ms": round(mean_latency, 2), "peak_memory_mb": round(mem_mb, 2)}
            logger.info("[SCHP_BENCHMARK] Device %s: %.1f ms/image (peak memory: %.1f MB)", dev, mean_latency, mem_mb)
        except Exception as e:
            logger.error("[SCHP_BENCHMARK] Failed on %s: %s", dev, e)
            results[dev] = {"error": str(e)}

    return results

    for dev in devices:
        if dev.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("[SCHP_BENCHMARK] %s requested but CUDA is unavailable", dev)
            continue
        try:
            model, proc = _load(device=dev)
            # Warm up
            _ = parse(dummy_img, device=dev)
            if dev.startswith("cuda"):
                torch.cuda.synchronize()

            latencies = []
            for _ in range(num_runs):
                t0 = time.perf_counter()
                _ = parse(dummy_img, device=dev)
                if dev.startswith("cuda"):
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000.0)

            mean_latency = float(np.mean(latencies))
            mem_mb = 0.0
            if dev.startswith("cuda"):
                mem_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
            results[dev] = {"mean_latency_ms": mean_latency, "peak_memory_mb": mem_mb}
            logger.info("[SCHP_BENCHMARK] Device %s: %.1f ms/image (peak memory: %.1f MB)", dev, mean_latency, mem_mb)
        except Exception as e:
            logger.error("[SCHP_BENCHMARK] Failed on %s: %s", dev, e)
            results[dev] = {"error": str(e)}

    return results


def fuse_with_birefnet(image_rgb: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Protect SCHP interiors while retaining BiRefNet soft boundaries."""
    parsed = parse(image_rgb)
    fused = np.clip(alpha.astype(np.float32), 0.0, 1.0).copy()
    fused[parsed["solid_core"]] = np.maximum(fused[parsed["solid_core"]], 0.985)
    hair = parsed["labels"] == HAIR_LABEL
    fused[hair] = np.maximum(fused[hair], 0.88)
    fused[parsed["hair_core"]] = np.maximum(fused[parsed["hair_core"]], 0.96)
    clothing = np.isin(parsed["labels"], tuple(CLOTHING_LABELS))
    fused[clothing] = np.maximum(fused[clothing], 0.98)
    return fused, parsed["labels"]
