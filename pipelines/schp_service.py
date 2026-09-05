"""Local SCHP LIP human-part parsing for BiRefNet matte protection."""

from pathlib import Path
from threading import Lock

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

# LIP-20: clothing vs background. Dress/cloth must never be treated as background.
CLOTHING_LABELS = {5, 6, 7, 10, 11, 12}  # upper, dress, coat, jumpsuit, scarf/tie, skirt
BACKGROUND_LABEL = 0
SOLID_LABELS = {1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
HAIR_LABEL = 2


def available() -> bool:
    return (MODEL_DIR / "model.safetensors").is_file()


def _load():
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
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
            ).eval().to("cpu")
            logger.info("Loaded SCHP LIP-20 on CPU")
    return _MODEL, _PROCESSOR


def parse(image_rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Return original-resolution LIP labels and conservative protection masks."""
    model, processor = _load()
    image = Image.fromarray(image_rgb, "RGB")
    inputs = processor(images=image, return_tensors="pt")
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
    return {"labels": labels, "solid_core": solid_core, "hair_core": hair_core}


def fuse_with_birefnet(image_rgb: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Protect SCHP interiors while retaining BiRefNet soft boundaries."""
    parsed = parse(image_rgb)
    fused = np.clip(alpha.astype(np.float32), 0.0, 1.0).copy()
    fused[parsed["solid_core"]] = np.maximum(fused[parsed["solid_core"]], 0.985)
    fused[parsed["hair_core"]] = np.maximum(fused[parsed["hair_core"]], 0.94)
    clothing = np.isin(parsed["labels"], tuple(CLOTHING_LABELS))
    fused[clothing] = np.maximum(fused[clothing], 0.98)
    return fused, parsed["labels"]
