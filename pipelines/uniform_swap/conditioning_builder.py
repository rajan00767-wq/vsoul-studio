"""Conditioning Canvas and Reference Builder for Qwen Image Edit (V11).

Prepares exact Qwen inputs adhering to Section 3 and Section 4 requirements:
- Clean original-portrait baseline for Image 1:
  - Preserves 100% of authentic face, eyes, hair, braids, bows, authentic visible neck skin,
    and natural shoulder silhouette.
  - Removes background ONLY with a validated full-person matte (BiRefNet alpha),
    never cutting off the torso with an identity-only alpha.
  - Zero neutral gray torso placeholders ([130, 136, 144]).
  - Zero fitted uniform fabric pasted into Image 1.
  - Zero synthetic neck skin or median infill.
  - Zero additional collar cues or rough composites in baseline.
  - Garment-seeded conditioning ("fitted_seed") is kept strictly as a separate experimental strategy.
- Aspect-ratio-preserving target uniform preparation for Image 2.
- Precise spatial edit mask and hard identity protection mask.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from pipelines.uniform_swap.config import UniformSwapConfig

logger = logging.getLogger(__name__)


class ConditioningBuilder:
    """Prepares Qwen Image 1, Image 2, and spatial edit/protection masks."""

    def __init__(self) -> None:
        pass

    def prepare_uniform_reference(
        self,
        uniform_input: Union[str, Path, Image.Image, np.ndarray],
        target_size: Tuple[int, int] = (800, 1024),
        background_rgb: Tuple[int, int, int] = (255, 255, 255),
    ) -> Tuple[Image.Image, np.ndarray]:
        """Prepare target uniform reference (Qwen Image 2) preserving aspect ratio.
        
        Section 4 requirement:
        - Uploaded target uniform as garment reference.
        - Preserves exact fabric pattern, collar construction, collar color, button arrangement,
          sleeve design, pocket position, ties.
        - Preserves aspect ratio (no distortion).
        - Zero second person or unrelated body geometry.
        """
        if isinstance(uniform_input, (str, Path)):
            u_img = Image.open(str(uniform_input)).convert("RGBA")
        elif isinstance(uniform_input, np.ndarray):
            u_img = Image.fromarray(uniform_input)
            if u_img.mode != "RGBA":
                u_img = u_img.convert("RGBA")
        else:
            u_img = uniform_input.convert("RGBA")

        tw, th = target_size
        uw, uh = u_img.size

        # Scale uniformly to fit within target bounding box
        scale = min(tw / max(1, uw), th / max(1, uh))
        new_w = max(1, int(round(uw * scale)))
        new_h = max(1, int(round(uh * scale)))

        scaled_u = u_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Center on target solid background canvas
        canvas = Image.new("RGB", (tw, th), background_rgb)
        offset_x = (tw - new_w) // 2
        offset_y = (th - new_h) // 2

        if scaled_u.mode == "RGBA":
            # Alpha composite onto solid backdrop
            canvas.paste(scaled_u, (offset_x, offset_y), mask=scaled_u.split()[-1])
        else:
            canvas.paste(scaled_u, (offset_x, offset_y))

        arr = np.asarray(canvas, dtype=np.uint8)
        logger.info(
            "[CONDITIONING_BUILDER] Prepared Qwen Image 2 reference: orig=(%d, %d) -> scaled=(%d, %d) centered on (%d, %d)",
            uw, uh, new_w, new_h, tw, th
        )
        return canvas, arr

    def build_conditioning(
        self,
        immutable_original_rgb: np.ndarray,
        identity_layer: Dict[str, Any],
        config: Optional[UniformSwapConfig] = None,
        rough_uniform_seed: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:
        """Construct Qwen conditioning Image 1 and region masks.

        Returns:
            Dict containing:
                - 'conditioning_rgb': (H, W, 3) uint8 image for Qwen Image 1
                - 'conditioning_pil': PIL Image for Qwen Image 1
                - 'selected_strategy': str
                - 'generation_edit_region': (H, W) bool mask of pixels Qwen may edit
                - 'generation_hard_identity_protection': (H, W) bool mask of pixels Qwen must freeze
                - 'fallback_triggered': bool
        """
        config = config or UniformSwapConfig()
        h, w = immutable_original_rgb.shape[:2]
        bg_rgb = config.background_color
        strategy = config.conditioning_strategy

        val_info = identity_layer.get("validation", {})
        is_valid = val_info.get("is_valid", True)
        fallback_triggered = False
        if not is_valid:
            logger.warning("[CONDITIONING_BUILDER] Identity validation failed; falling back to original baseline")
            strategy = "A"
            fallback_triggered = True

        # Normalize strategy representation: "original_baseline" -> "A", "neutral_torso" -> "B", "fitted_seed" -> "D"
        normalized_strategy = "A" if strategy in ("A", "original_baseline") else ("B" if strategy in ("B", "neutral_torso") else ("D" if strategy in ("D", "fitted_seed") else strategy))

        # Retrieve verified component masks from identity_layer
        orig_face = identity_layer.get("original_face_mask", np.zeros((h, w), dtype=bool))
        orig_ears = identity_layer.get("original_ears_mask", np.zeros((h, w), dtype=bool))
        orig_neck = identity_layer.get("original_neck_mask", np.zeros((h, w), dtype=bool))
        orig_hair = identity_layer.get("original_hair_mask", np.zeros((h, w), dtype=bool))
        orig_acc = identity_layer.get("original_accessory_mask", np.zeros((h, w), dtype=bool))
        orig_badge = identity_layer.get("original_badge_mask", np.zeros((h, w), dtype=bool))
        orig_garment = identity_layer.get("original_garment_mask", np.zeros((h, w), dtype=bool))
        full_person_alpha = identity_layer.get("full_person_alpha")
        neck_cls = identity_layer.get("neck_classification", "uncertain")

        # ── 1. Construct Hard Identity Protection Mask ──
        # Protect authentic face, ears, hair, braids, hair accessories, and authentic visible neck
        core_identity = orig_face | orig_ears | orig_hair | orig_acc
        if neck_cls != "occluded_neck":
            core_identity = core_identity | orig_neck

        # Enforce invariant: Clothing and badges are NEVER protected!
        core_identity = core_identity & ~orig_garment & ~orig_badge

        # Erode slightly to allow natural blending at true external boundaries
        hard_protect = cv2.erode(core_identity.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
        hard_protect = hard_protect & ~orig_garment & ~orig_badge

        # ── 2. Construct Edit Region Mask ──
        # Pixels Qwen is instructed to replace: original garment area + badge
        # Section 5 & 7: Do not punch an unconditioned artificial rectangular throat corridor.
        # Authentic neck skin is protected; the target collar fits naturally along collar boundary.
        edit_region = (orig_garment | orig_badge).copy()

        # Enforce invariant: Hard protected identity is NEVER in edit region!
        edit_region = edit_region & ~hard_protect

        # ── 3. Construct Qwen Image 1 Conditioning Canvas ──
        if strategy in ("original_baseline", "A"):
            # Section 3: Clean original-portrait baseline
            # Preserve original facial features, expression, hair silhouette, braids, bows, ribbons,
            # accessories, authentic visible neck skin, and natural shoulder/torso silhouette.
            # Remove background ONLY with validated full-person matte.
            if full_person_alpha is not None:
                alpha_f = (
                    full_person_alpha.astype(np.float32) / 255.0
                    if full_person_alpha.dtype == np.uint8
                    else full_person_alpha.astype(np.float32)
                )
                if alpha_f.shape != (h, w):
                    alpha_f = cv2.resize(alpha_f, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                alpha_f = np.ones((h, w), dtype=np.float32)

            bg_arr = np.asarray(bg_rgb, dtype=np.float32)[None, None, :]
            orig_f = immutable_original_rgb.astype(np.float32)
            matted = orig_f * alpha_f[:, :, None] + bg_arr * (1.0 - alpha_f[:, :, None])
            conditioning_rgb = np.rint(matted).clip(0, 255).astype(np.uint8)

        elif strategy in ("fitted_seed", "D") and rough_uniform_seed is not None:
            # Experimental Strategy D: seats rough fitted uniform template into clothing area
            seed_arr = np.asarray(rough_uniform_seed.resize((w, h), Image.Resampling.LANCZOS).convert("RGB"))
            conditioning_rgb = immutable_original_rgb.copy()
            conditioning_rgb[edit_region] = seed_arr[edit_region]

        elif strategy in ("neutral_torso", "B"):
            # Experimental Strategy B: neutral torso gray [130, 136, 144] strictly outside face/neck/hair/accessories
            conditioning_rgb = immutable_original_rgb.copy()
            neutral_gray = np.array([130, 136, 144], dtype=np.uint8)
            conditioning_rgb[edit_region] = neutral_gray
            conditioning_rgb[edit_region] = neutral_gray
            fallback_triggered = False

        else:
            # Default fallback to original baseline
            logger.warning("[CONDITIONING_BUILDER] Strategy '%s' unrecognized; falling back to 'original_baseline'", strategy)
            conditioning_rgb = immutable_original_rgb.copy()
            fallback_triggered = True

        conditioning_pil = Image.fromarray(conditioning_rgb, "RGB")
        logger.info(
            "[CONDITIONING_BUILDER] Conditioning built (strategy=%s): edit_pixels=%d, protect_pixels=%d",
            strategy,
            int(np.sum(edit_region)),
            int(np.sum(hard_protect)),
        )

        return {
            "conditioning_rgb": conditioning_rgb,
            "conditioning_pil": conditioning_pil,
            "selected_strategy": normalized_strategy,
            "generation_edit_region": edit_region,
            "generation_hard_identity_protection": hard_protect,
            "fallback_triggered": fallback_triggered,
        }


# Global singleton instance
conditioning_builder = ConditioningBuilder()
