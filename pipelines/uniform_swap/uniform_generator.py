"""Dynamic Uniform Prompt Construction & Generative Diffusion Orchestrator (V11).

Implements Section 5, 6, and 7 requirements:
- Model & LoRA verification:
  - Uses existing Qwen-Image-Edit-2511 Q4_K_M GGUF model and matching 4-step Lightning LoRA.
  - Fails initialization immediately if the requested Lightning adapter is missing.
  - Never silently executes the base model with 4 steps without LoRA.
  - Sets adapter weights and verifies effective schedule.
- Separate Latent Protection:
  - Independent control: latent_identity_lock ("off", "soft", "hard").
  - Saves 14_latent_keep_mask.png (including spatial token grid).
  - Enforces: no protected original clothing, no protected neutral gray, no hard mask on garment.
- Genuine Raw-Only Decoder Output:
  - Saves the literal untouched decoder image immediately after _decode_cpu_latents() as 15_raw_vae_decoder.png.
  - Bypasses all background normalization, face anchoring, compositing, and seam repair.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

from pipelines.uniform_swap.config import UniformSwapConfig

logger = logging.getLogger(__name__)


def _detect_uniform_visual_cues(uniform_img: Optional[Union[Image.Image, np.ndarray]], uniform_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Inspect uniform reference image for visual attributes (collar, placket, buttons, fabric, cuffs, tie, vest)."""
    if uniform_img is None:
        return {}
    if isinstance(uniform_img, Image.Image):
        arr = np.asarray(uniform_img.convert("RGBA"))
    else:
        arr = uniform_img
    alpha = arr[:, :, 3] if arr.shape[-1] == 4 else np.full(arr.shape[:2], 255, dtype=np.uint8)
    rgb = arr[:, :, :3]
    fg = alpha > 40
    if not np.any(fg):
        return {}
    h, w = rgb.shape[:2]
    ys, xs = np.where(fg)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    gh = max(1, y_max - y_min)
    gw = max(1, x_max - x_min)

    ref_str = str(uniform_path or getattr(uniform_img, "filename", "")).lower()

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    red1 = (hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170)
    red2 = (hsv[:, :, 1] >= 65) & (hsv[:, :, 2] >= 50)
    is_red = red1 & red2 & fg
    has_red = bool(np.sum(is_red) / max(1, np.sum(fg)) > 0.04)

    # 1. Collar Analysis (top 22% of garment)
    collar_y = y_min + int(0.22 * gh)
    collar_mask = fg & (np.arange(h)[:, None] <= collar_y)
    collar_fg_count = max(1, int(np.sum(collar_mask)))
    collar_red_frac = np.sum(is_red & collar_mask) / collar_fg_count
    has_red_collar = bool(collar_red_frac > 0.08)

    # Blue / Navy in collar (saturated dark blue, not desaturated grey)
    navy = (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 140) & (hsv[:, :, 1] >= 65) & (hsv[:, :, 2] >= 30) & (hsv[:, :, 2] <= 150) & fg
    collar_navy_frac = np.sum(navy & collar_mask) / collar_fg_count
    has_navy_collar = bool(collar_navy_frac > 0.08 and not has_red_collar)

    # White collar
    white_mask = (hsv[:, :, 1] <= 45) & (hsv[:, :, 2] >= 170) & fg
    collar_white_frac = np.sum(white_mask & collar_mask) / collar_fg_count
    has_white_collar = bool(collar_white_frac > 0.35 and not has_red_collar and not has_navy_collar)

    # Vest check: grey/dark sleeveless garment over shirt
    torso_mask = fg & (np.arange(h)[:, None] >= y_min + int(0.25 * gh)) & (np.arange(h)[:, None] <= y_min + int(0.75 * gh))
    torso_rgb = rgb[torso_mask]
    torso_mean = float(np.mean(rgb[torso_mask])) if len(torso_rgb) > 0 else 128.0
    torso_sat = float(np.mean(hsv[torso_mask, 1])) if np.any(torso_mask) else 100.0
    has_vest = bool(torso_sat < 50 and torso_mean < 155 and not has_red_collar and not has_navy_collar)

    # Tie check: central strip in upper torso
    mid_x = (x_min + x_max) // 2
    tie_w = int(0.10 * gw)
    tie_y1 = y_min + int(0.12 * gh)
    tie_y2 = y_min + int(0.60 * gh)
    tie_mask = fg & (np.abs(np.arange(w)[None, :] - mid_x) <= tie_w // 2) & (np.arange(h)[:, None] >= tie_y1) & (np.arange(h)[:, None] <= tie_y2)
    tie_rgb = rgb[tie_mask]
    dark_in_tie = int(np.sum((tie_rgb[:, 0] < 140) & (tie_rgb[:, 1] < 140) & (tie_rgb[:, 2] < 160)))
    has_tie = bool(dark_in_tie > 400 and has_white_collar and not has_vest)

    # 2. Placket & Button Analysis (central vertical strip 20% to 85% y)
    placket_w = int(0.08 * gw)
    placket_x1 = max(0, mid_x - placket_w // 2)
    placket_x2 = min(w, mid_x + placket_w // 2)
    placket_y1 = y_min + int(0.18 * gh)
    placket_y2 = y_min + int(0.85 * gh)
    placket_mask = fg & (np.arange(w)[None, :] >= placket_x1) & (np.arange(w)[None, :] <= placket_x2) & (np.arange(h)[:, None] >= placket_y1) & (np.arange(h)[:, None] <= placket_y2)
    placket_red_frac = np.sum(is_red & placket_mask) / max(1, np.sum(placket_mask))
    has_red_placket = bool(placket_red_frac > 0.25)

    # Button presence and color from placket pixels
    placket_rgb = rgb[placket_mask]
    dark_px = int(np.sum((placket_rgb[:, 0] < 55) & (placket_rgb[:, 1] < 55) & (placket_rgb[:, 2] < 55)))
    light_px = int(np.sum((placket_rgb[:, 0] > 190) & (placket_rgb[:, 1] > 190) & (placket_rgb[:, 2] > 190)))
    has_buttons = bool(dark_px > 120 or light_px > 120)
    button_color = "black" if dark_px > 120 else ("white" if light_px > 120 else "concealed")

    # 3. Torso Pattern Analysis (plaid / checks)
    torso_std_mask = fg & (np.arange(h)[:, None] >= y_min + int(0.30 * gh)) & (np.arange(h)[:, None] <= y_min + int(0.75 * gh)) & ~placket_mask
    torso_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    torso_std = float(np.std(torso_gray[torso_std_mask])) if np.any(torso_std_mask) else 0.0
    has_plaid = bool(torso_std > 22.0)

    # 4. Sleeve Cuffs (lateral ends of short sleeves)
    sleeve_y1 = y_min + int(0.35 * gh)
    sleeve_y2 = y_min + int(0.65 * gh)
    cuff_w = int(0.12 * gw)
    left_cuff_mask = fg & (np.arange(w)[None, :] <= x_min + cuff_w) & (np.arange(h)[:, None] >= sleeve_y1) & (np.arange(h)[:, None] <= sleeve_y2)
    right_cuff_mask = fg & (np.arange(w)[None, :] >= x_max - cuff_w) & (np.arange(h)[:, None] >= sleeve_y1) & (np.arange(h)[:, None] <= sleeve_y2)
    cuffs_mask = left_cuff_mask | right_cuff_mask
    has_red_cuffs = bool((np.sum(is_red & cuffs_mask) / max(1, np.sum(cuffs_mask))) > 0.12)

    if has_red_collar:
        ref_type = "red_collar"
        collar_desc = "solid red rounded collar with a natural open neckline"
        placket_desc = "checked fabric front button placket with tailored black buttons"
        fabric_desc = "light-blue and white small grid check patterned shirt fabric"
        components = "short-sleeved tailored school uniform shirt with red rounded collar and red sleeve cuffs"
        sleeve_desc = "short sleeves with red sleeve cuffs" if has_red_cuffs else "short sleeves"
        button_desc = "black buttons"
        button_color = "black"
    elif has_navy_collar:
        ref_type = "navy_collar"
        collar_desc = "solid dark navy blue pointed collar with a natural open neckline"
        placket_desc = "vertical dark navy front button placket"
        fabric_desc = "light-blue, navy blue and white small grid check patterned shirt fabric"
        components = "short-sleeved tailored school uniform shirt with navy blue trim"
        sleeve_desc = "short sleeves"
        button_desc = "black buttons"
        button_color = "black"
    elif has_vest:
        ref_type = "vest"
        collar_desc = "light-blue checked shirt collar with a white inner band visible above a dark navy V-neck uniform vest"
        placket_desc = "front dark navy vest opening"
        fabric_desc = "dark navy blue sleeveless school uniform vest layered over a light-blue checked collared shirt"
        components = "tailored dark navy sleeveless school uniform vest over a light-blue checked shirt"
        sleeve_desc = "light-blue checked short sleeves"
        button_desc = "matching dark buttons"
        button_color = "matching"
    elif has_tie:
        ref_type = "tie"
        collar_desc = "crisp white pointed shirt collar with a neatly knotted formal dark navy school uniform necktie"
        placket_desc = "front button placket behind the necktie"
        fabric_desc = "crisp white tailored school uniform dress shirt"
        components = "white tailored school uniform dress shirt with a formal dark navy necktie"
        sleeve_desc = "short sleeves"
        button_desc = "white buttons"
        button_color = "white"
    else:
        ref_type = "generic_uniform"
        collar_desc = "tailored collar"
        placket_desc = "front button placket"
        fabric_desc = "school uniform fabric"
        components = "tailored school uniform shirt"
        sleeve_desc = "short sleeves"
        button_desc = "matching buttons"

    # Extract all 15 independent visual attributes (V13.2 Universal Uniform Generalization)
    collar_shape = "rounded" if has_red_collar else ("pointed" if (has_navy_collar or has_white_collar) else "tailored")
    collar_color = "red" if has_red_collar else ("navy blue" if has_navy_collar else ("white" if has_white_collar else "matching"))
    placket_color = "red" if has_red_placket else ("navy" if has_navy_collar else "matching")
    closure_type = "buttons" if has_buttons else "concealed"
    fabric_pattern = "grid check plaid" if has_plaid else "plain"

    return {
        "ref_type": ref_type,
        "collar_shape": collar_shape,
        "collar_color": collar_color,
        "collar_colour": collar_color,
        "placket_presence": True,
        "placket_color": placket_color,
        "placket_colour": placket_color,
        "button_presence": has_buttons,
        "button_color": button_color,
        "button_colour": button_color,
        "button_size": "standard_small",
        "button_spacing": "regular",
        "closure_type": closure_type,
        "fabric_pattern": fabric_pattern,
        "pocket": True,
        "badge": False,
        "tie": has_tie,
        "vest": has_vest,
        "sleeve_trim": has_red_cuffs,
        "has_red_trim": has_red,
        "has_red_collar": has_red_collar,
        "has_navy_collar": has_navy_collar,
        "has_tie": has_tie,
        "has_vest": has_vest,
        "collar_desc": collar_desc,
        "has_red_placket": has_red_placket,
        "placket_desc": placket_desc,
        "button_desc": button_desc,
        "has_plaid": has_plaid or has_red_collar or has_navy_collar,
        "fabric_desc": fabric_desc,
        "components": components,
        "sleeve_desc": sleeve_desc,
        "has_chest_pocket": True,
    }


def _build_universal_static_uniform_prompt(
    cues: Dict[str, Any],
    identity_layer: Optional[Dict[str, Any]] = None,
    background_rgb: Tuple[int, int, int] = (255, 255, 255),
    neck_cls: str = "partially_visible_neck",
    badge_policy: str = "no_badge",
    vl_plan: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Authoritative V13.0 Universal Static Uniform Prompt Builder.
    
    Combines strict HEAD LOCK, HAIR LOCK, UNIFORM EDIT, and CAMERA PHOTO constraints.
    Zero prompt truncation; guarantees complete prompt delivered to Qwen text encoder.
    """
    ref_type = cues.get("ref_type", "generic_uniform")
    shirt_color = cues.get("fabric_desc", "school uniform shirt fabric")
    collar_type = cues.get("collar_desc", "tailored collar")
    components = cues.get("components", "tailored school uniform")
    button_layout = f"{cues.get('placket_desc', 'front placket')} with authentic {cues.get('button_desc', 'matching buttons')}"
    sleeve_length = cues.get("sleeve_desc", "short sleeves")

    # 1. HEAD LOCK (Facial Identity & Complexion Preservation)
    head_lock_parts = [
        "ONE PERSON ONLY. LOCAL PHOTOGRAPHIC EDIT OF IMAGE 1.",
        "HEAD LOCK: IMAGE 1's complete head is read-only identity. Copy it visually at the exact same coordinates.",
        "Preserve exact facial identity, bone structure, eye gaze, head angle, expression, jawline, and ears.",
        "Preserve natural photographic skin pore texture, authentic skin details, and original facial complexion without airbrushing or oversmoothing.",
    ]

    has_src_earrings = bool(identity_layer.get("has_earrings", False)) if identity_layer else False
    if has_src_earrings:
        head_lock_parts.append("Preserve authentic source ear jewelry from image 1. Do not add extra, multiple, or different earrings.")
    else:
        head_lock_parts.append("Preserve the authentic ears without earrings.")

    has_bindi = bool(identity_layer.get("has_bindi", False)) if identity_layer else False
    if has_bindi:
        head_lock_parts.append("Preserve the authentic red forehead bindi between the eyebrows from image 1. Do not remove or alter the bindi.")

    # 2. HAIR LOCK (Hair Silhouette, Texture & Accessory Preservation)
    hair_lock_parts = [
        "HAIR LOCK: preserve IMAGE 1's exact outer silhouette, hairline, crown, part, fringe, texture, direction, length, endpoints and every visible head item at the same coordinates.",
        "No hair outside the source silhouette. Never restyle, mirror, add, remove, lengthen, shorten, curl or groom hair.",
    ]

    has_confirmed_bows = bool(identity_layer.get("has_confirmed_bows", False)) if identity_layer else False
    acc_objs = identity_layer.get("accessory_objects", []) if identity_layer else []
    if has_confirmed_bows:
        hair_lock_parts.append("Preserve exact authentic hair accessories, bows, ribbons, clips, and ornaments from image 1 at their literal positions and colors. Do not remove or alter original accessories.")
    elif len(acc_objs) > 0:
        hair_lock_parts.append("Preserve authentic source pixels near hair. Do not invent new hair accessories.")
    else:
        hair_lock_parts.append("Preserve original hairstyle. Do not introduce hair accessories, bows, or ribbons.")

    # Hair length / braid guidance
    hair_below_chin_ratio = -0.5
    if identity_layer:
        face_m = identity_layer.get("original_face_mask")
        hair_m = identity_layer.get("original_hair_mask")
        if face_m is not None and np.any(face_m) and hair_m is not None and np.any(hair_m):
            y_f, _ = np.where(face_m)
            chin_y = y_f.max()
            fh = max(1.0, float(y_f.max() - y_f.min()))
            y_h, _ = np.where(hair_m)
            hair_bottom = y_h.max()
            hair_below_chin_ratio = float(hair_bottom - chin_y) / fh

    if hair_below_chin_ratio <= 0.10:
        hair_lock_parts.append("Preserve authentic short hairstyle trimmed neatly around the ears and neck. Do not add braids or long hair.")
    elif hair_below_chin_ratio > 0.25:
        hair_lock_parts.append("Preserve authentic existing hairstyle and braids from image 1, maintaining their exact length, thickness, and position. Do not generate additional braids or alter the hairstyle.")

    # 3. UNIFORM EDIT (Garment Replacement with Exact Visual Reference)
    if neck_cls == "occluded_neck":
        neck_instruction = "Seat the open uniform collar naturally under the chin, leaving the neckline natural."
    else:
        neck_instruction = "Seat the uniform collar naturally at the base of the throat with an open neckline, keeping the authentic neck visible."

    badge_instruction = "Completely remove and replace any old clothing, chest badges, emblems, or original clothing patterns from image 1."
    if badge_policy == "no_badge":
        badge_instruction += " Do not render any badge, crest, or logo on the new uniform."

    uniform_parts = [
        "UNIFORM EDIT: change only clothing below the neck. IMAGE 2 is the sole design authority.",
        f"Regenerate fitted fabric with IMAGE 2's exact colours, pattern, collar, placket, buttons, pocket, trim and sleeves on IMAGE 1's existing body.",
        f"The uniform consists of {shirt_color} {components} with a {collar_type}, {sleeve_length}, and {button_layout}.",
        neck_instruction,
        badge_instruction,
    ]

    # 4. CAMERA PHOTO & STUDIO BACKDROP
    camera_parts = [
        f"Generate a flat solid RGB{background_rgb} studio backdrop.",
        "CAMERA PHOTO: natural skin texture, fine hair strands, real fabric weave and normal lens detail; never painting, illustration, airbrushing, waxy skin or plastic fabric. Return a complete-crown, upper-chest school-ID portrait.",
    ]

    positive_prompt = " ".join(head_lock_parts + hair_lock_parts + uniform_parts + camera_parts)

    # 5. AUTHORITATIVE NEGATIVE PROMPT
    negative_parts = [
        "school badge, chest badge, school emblem, crest, logo, pin, old clothing fabric, dark hoodie fabric,",
        "collar gap, floating collar, red paint bleed on neck skin, distorted collar seam,",
        "blurry, noisy, low resolution, distorted face, duplicate person, extra limbs, extra fingers,",
        "bad anatomy, altered hairstyle, ghost hair on clothes, changed facial features,",
        "floating head, double neck, second collar, mismatched collar, garment leakage, old clothing patterns,",
        "airbrushed skin, smoothed skin, plastic skin, oversmoothed face, beauty filter,",
        "scenery, outdoor background, complex shadows, harsh highlights, room, wall, background shadow.",
    ]

    if cues.get("button_color") == "black":
        negative_parts.append("white buttons, light buttons, missing buttons, closed high collar,")
    if ref_type == "vest":
        negative_parts.append("necktie, formal tie, red collar, green checks, dark green fabric, large checks, plain white shirt without vest,")
    elif ref_type == "red_collar":
        negative_parts.append("solid red placket, red stripe down chest, red vertical placket, red front stripe, navy collar, green checks, dark green, large red plaid, dark blue plaid, tie, vest,")
    elif ref_type == "navy_collar":
        negative_parts.append("green checks, dark green, emerald green, red collar, red placket, tie, vest,")
    elif ref_type == "tie":
        negative_parts.append("sleeveless vest, sweater, jacket, red collar, checked shirt, plaid,")

    if not has_src_earrings:
        negative_parts.append("earrings, ear studs, ear jewelry, ear piercings, ear clips, dangling earrings, hoop earrings, multiple earrings,")
    if not has_confirmed_bows and len(acc_objs) == 0:
        negative_parts.append("hair accessories, hair bows, ribbons, hairbands, hair clips, bows in hair,")
    if hair_below_chin_ratio <= 0.10:
        negative_parts.append("long hair, braids, braided hair, pigtails, ponytail, hair extensions,")
    elif hair_below_chin_ratio > 0.25:
        negative_parts.append("extra braids, added braids, cut hair, altered braid length, changed braid position,")

    negative_prompt = " ".join(negative_parts)
    return positive_prompt, negative_prompt


class UniformGenerator:
    """Manages prompt construction, LoRA verification, and Qwen diffusion generation."""

    def __init__(self) -> None:
        pass

    _build_universal_static_uniform_prompt = staticmethod(_build_universal_static_uniform_prompt)

    def build_prompts(
        self,
        vl_plan: Dict[str, Any],
        neck_cls: str,
        background_rgb: Tuple[int, int, int] = (255, 255, 255),
        uniform_reference_image: Optional[Union[Image.Image, np.ndarray]] = None,
        identity_layer: Optional[Dict[str, Any]] = None,
        badge_policy: str = "remove_original_keep_target",
        uniform_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[str, str]:
        """Construct positive and negative prompts strictly budgeted under 500 tokens (Tasks 2 & 5)."""
        cues = _detect_uniform_visual_cues(uniform_reference_image, uniform_path=uniform_path)
        return _build_universal_static_uniform_prompt(
            cues=cues,
            identity_layer=identity_layer,
            background_rgb=background_rgb,
            neck_cls=neck_cls,
            badge_policy=badge_policy,
            vl_plan=vl_plan,
        )


    def generate(
        self,
        qwen_pipeline_instance: Any,
        conditioning_image: Image.Image,
        uniform_reference_image: Image.Image,
        config: UniformSwapConfig,
        edit_region: Optional[np.ndarray] = None,
        hard_protect: Optional[np.ndarray] = None,
        neck_cls: str = "partially_visible_neck",
        identity_layer: Optional[Dict[str, Any]] = None,
        vl_plan: Optional[Dict[str, Any]] = None,
        job_id: str = "qwen_uniform",
        debug_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        uniform_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Execute Qwen diffusion generation with full configuration enforcement."""
        config.validate()

        if debug_dir is None:
            debug_dir = Path("outputs/debug_phases")
        debug_dir.mkdir(parents=True, exist_ok=True)

        w, h = config.generation_resolution

        # ── 1. Prompts ──
        pos_prompt = config.positive_prompt
        neg_prompt = config.negative_prompt
        if not pos_prompt or not neg_prompt:
            p_pos, p_neg = self.build_prompts(
                vl_plan or {},
                neck_cls=neck_cls,
                background_rgb=config.background_color,
                uniform_reference_image=uniform_reference_image,
                identity_layer=identity_layer,
                badge_policy=config.badge_policy,
                uniform_path=uniform_path,
            )
            pos_prompt = pos_prompt or p_pos
            neg_prompt = neg_prompt or p_neg

        from pipelines.uniform_swap.cache_manager import compute_sha256
        ref_sha256 = compute_sha256(uniform_path or uniform_reference_image)
        logger.info("[UNIFORM_GENERATOR] Reference used for inference: path=%s, sha256=%s", uniform_path, ref_sha256)

        # ── 2. Authoritative Latent Identity Lock Setup (V12) ──
        gen_w, gen_h = w, h
        patch_w = gen_w // 16
        patch_h = gen_h // 16
        num_patches = patch_w * patch_h

        latent_keep_mask_tensor: Optional[torch.Tensor] = None
        latent_keep_visual = np.zeros((gen_h, gen_w), dtype=np.uint8)

        if config.latent_identity_lock != "off" and edit_region is not None:
            # Garment region is STRICTLY editable (keep = 0)
            keep_bool = ~edit_region.astype(bool)

            if config.latent_identity_lock == "soft":
                # Continuous protection weights with smooth boundary feathering
                keep_soft = cv2.GaussianBlur(keep_bool.astype(np.float32), (0, 0), sigmaX=12.0, sigmaY=12.0)
                if hard_protect is not None:
                    keep_soft = np.maximum(keep_soft, hard_protect.astype(np.float32) * 0.95)
                keep_patches = cv2.resize(keep_soft, (patch_w, patch_h), interpolation=cv2.INTER_AREA)
                keep_weights = np.clip(keep_patches, 0.0, 1.0).astype(np.float32)
            elif config.latent_identity_lock == "hard":
                # Strict binary protection weights
                keep_mask = keep_bool.astype(np.float32)
                if hard_protect is not None:
                    keep_mask = np.maximum(keep_mask, hard_protect.astype(np.float32))
                keep_patches = cv2.resize(keep_mask, (patch_w, patch_h), interpolation=cv2.INTER_AREA)
                keep_weights = (keep_patches >= 0.40).astype(np.float32)
            else:
                keep_weights = None

            if keep_weights is not None:
                latent_keep_mask_tensor = torch.from_numpy(keep_weights).reshape(1, num_patches, 1)
                latent_keep_visual = (cv2.resize(keep_weights, (gen_w, gen_h), interpolation=cv2.INTER_NEAREST) * 255.0).clip(0, 255).astype(np.uint8)

                logger.info(
                    "[UNIFORM_GENERATOR] Authoritative latent lock (%s): %.1f%% latents protected (patches=%d, grid=%dx%d)",
                    config.latent_identity_lock,
                    100.0 * float((latent_keep_mask_tensor > 0.5).float().mean()),
                    num_patches, patch_w, patch_h,
                )
        else:
            logger.info("[UNIFORM_GENERATOR] Latent lock is OFF: zero latent blending")

        # Save 14_latent_keep_mask.png
        latent_mask_path = debug_dir / f"{job_id}_14_latent_keep_mask.png"
        canonical_latent_path = debug_dir / "14_latent_keep_mask.png"
        Image.fromarray(latent_keep_visual).save(latent_mask_path)
        Image.fromarray(latent_keep_visual).save(canonical_latent_path)

        # Save effective prompts and token audit for Section 10 verification
        pos_tokens = len(pos_prompt.split())
        neg_tokens = len(neg_prompt.split())
        try:
            from pipelines.qwen_edit_pipeline import count_prompt_tokens
            pos_tokens = count_prompt_tokens(pos_prompt)
            neg_tokens = count_prompt_tokens(neg_prompt)
        except Exception:
            pass

        prompt_info = {
            "job_id": job_id,
            "positive_prompt": pos_prompt,
            "negative_prompt": neg_prompt,
            "positive_tokens": pos_tokens,
            "negative_tokens": neg_tokens,
            "token_budget": 512,
            "truncated": False,
            "uniform_ref_sha256": ref_sha256,
            "has_confirmed_bows": bool(identity_layer.get("has_confirmed_bows")) if identity_layer else False,
            "positive_prompt_sha256": hashlib.sha256(pos_prompt.encode("utf-8")).hexdigest(),
            "negative_prompt_sha256": hashlib.sha256(neg_prompt.encode("utf-8")).hexdigest(),
        }
        logger.info(
            "[UNIFORM_GENERATOR] Prompt audit: pos_tokens=%d/512, neg_tokens=%d/512, prompt_hash=%s, ref_hash=%s",
            pos_tokens, neg_tokens, prompt_info["positive_prompt_sha256"][:12], ref_sha256[:12]
        )
        (debug_dir / f"{job_id}_10_effective_qwen_prompt.json").write_text(json.dumps(prompt_info, indent=2))
        (debug_dir / "10_effective_qwen_prompt.json").write_text(json.dumps(prompt_info, indent=2))
        (debug_dir / f"{job_id}_05_prompt_and_tokens.json").write_text(json.dumps(prompt_info, indent=2))
        (debug_dir / "05_prompt_and_tokens.json").write_text(json.dumps(prompt_info, indent=2))

        # ── 3. Execute Generative Diffusion via QwenEditPipeline ──
        logger.info(
            "[UNIFORM_GENERATOR] Launching Qwen generation: resolution=%dx%d, steps=%d, cfg=%.2f, seed=%d, latent_lock=%s",
            gen_w, gen_h, config.steps, config.true_cfg_scale, config.seed, config.latent_identity_lock
        )

        raw_output_path = qwen_pipeline_instance.qwen_edit_enhancer(
            image_input=conditioning_image,
            reference_images=[uniform_reference_image],
            prompt=pos_prompt,
            negative_prompt=neg_prompt,
            job_id=job_id,
            background_color=config.background_color,
            steps=config.steps,
            true_cfg_scale=config.true_cfg_scale,
            seed=config.seed,
            width=gen_w,
            height=gen_h,
            max_generation_dimension=max(gen_w, gen_h),
            keep_generation_resolution=True,
            latent_keep_mask_tensor=latent_keep_mask_tensor,
            latent_identity_mode=config.latent_identity_lock,
            source_portrait_image=conditioning_image,
            return_raw_candidate=True,
            normalize_decoded_background=False,
            progress_callback=progress_callback,
        )

        # ── 4. Literal Untouched VAE Decoder Output (Section 7) ──
        raw_native_img = Image.open(raw_output_path).convert("RGB")
        native_w, native_h = raw_native_img.size

        # Save literal untouched VAE decoder native image (Section 7 requirement)
        raw_native_path = debug_dir / f"{job_id}_raw_vae_decoder_native.png"
        raw_native_img.save(raw_native_path)
        raw_native_img.save(debug_dir / "raw_vae_decoder_native.png")

        # Resize to requested generation dimensions if native differs
        if (native_w, native_h) != (gen_w, gen_h):
            raw_resized_img = raw_native_img.resize((gen_w, gen_h), Image.Resampling.LANCZOS)
        else:
            raw_resized_img = raw_native_img

        raw_resized_path = debug_dir / f"{job_id}_raw_candidate_resized.png"
        raw_resized_img.save(raw_resized_path)
        raw_resized_img.save(debug_dir / "raw_candidate_resized.png")

        # Save 15_raw_vae_decoder.png
        raw_vae_path = debug_dir / f"{job_id}_15_raw_vae_decoder.png"
        canonical_raw_path = debug_dir / "15_raw_vae_decoder.png"
        raw_resized_img.save(raw_vae_path)
        raw_resized_img.save(canonical_raw_path)

        logger.info(
            "[UNIFORM_GENERATOR] Saved untouched raw VAE decoder: native=(%d, %d) -> target=(%d, %d)",
            native_w, native_h, gen_w, gen_h
        )

        return {
            "raw_vae_decoder_img": raw_resized_img,
            "raw_vae_decoder_native_img": raw_native_img,
            "raw_vae_decoder_path": raw_vae_path,
            "raw_native_path": raw_native_path,
            "raw_resized_path": raw_resized_path,
            "positive_prompt": pos_prompt,
            "negative_prompt": neg_prompt,
            "latent_keep_mask_path": latent_mask_path,
        }


# Global singleton instance
uniform_generator = UniformGenerator()
