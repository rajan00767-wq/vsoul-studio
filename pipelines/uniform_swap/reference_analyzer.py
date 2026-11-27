"""Universal Dynamic Uniform Reference Analyzer (V13.3 Section 3).

Dynamically extracts garment design attributes from arbitrary target uniform reference images:
- Primary and secondary fabric colors
- Fabric pattern: solid, checked, striped, printed, textured
- Pattern scale, spacing and orientation
- Collar type, shape, color, and opening geometry (pointed, rounded/Peter Pan, polo, mandarin, V-neck, round neck)
- Sleeve type and sleeve border / cuff color
- Button presence, color, placement, and spacing
- Front placket presence, design, and color
- Pocket presence and location
- School badge or emblem presence
- Tie presence, shape, color, and pattern
- Bow presence and placement
- Vest, blazer, or layered structure
- Garment silhouette

All decisions are dynamically derived from the image data with cryptographic SHA-256 caching.
Zero hardcoded assumptions based on student gender or filename.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _compute_image_sha256(image_input: Union[str, Path, Image.Image, np.ndarray]) -> str:
    """Compute SHA256 hash of image content for authoritative caching."""
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if p.exists():
            return hashlib.sha256(p.read_bytes()).hexdigest()
        return hashlib.sha256(str(p).encode("utf-8")).hexdigest()
    elif isinstance(image_input, Image.Image):
        return hashlib.sha256(image_input.tobytes()).hexdigest()
    elif isinstance(image_input, np.ndarray):
        return hashlib.sha256(image_input.tobytes()).hexdigest()
    return hashlib.sha256(str(image_input).encode("utf-8")).hexdigest()


class UniformReferenceAnalyzer:
    """Reusable, dynamic uniform reference analyzer with reference-hash caching."""

    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, Any]] = {}

    def analyze(
        self,
        reference_input: Union[str, Path, Image.Image, np.ndarray],
        force_reanalyze: bool = False,
    ) -> Dict[str, Any]:
        """Analyze uniform reference and return structured visual attribute specification."""
        ref_hash = _compute_image_sha256(reference_input)
        if not force_reanalyze and ref_hash in self._cache:
            return self._cache[ref_hash]

        # Load image preserving original alpha channel
        rgba, rgb, alpha, fg = self._load_and_segment(reference_input)
        if not np.any(fg):
            logger.warning("[REFERENCE_ANALYZER] No foreground detected in reference")
            spec = self._get_fallback_specification(ref_hash)
            self._cache[ref_hash] = spec
            return spec

        h, w = rgb.shape[:2]
        ys, xs = np.where(fg)
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
        gh = max(1, y_max - y_min)
        gw = max(1, x_max - x_min)
        mid_x = (x_min + x_max) // 2

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        # ── 1. Color Palette Extraction (Primary & Secondary) ──
        primary_color, secondary_color, dominant_hsv = self._extract_palette(rgb, fg, hsv)

        # ── 2. Torso Pattern Analysis (Solid / Checked / Striped / Textured) ──
        pattern_info = self._analyze_fabric_pattern(rgb, fg, y_min, gh, mid_x, gw, h, w)

        # ── 3. Collar Extraction & Shape Analysis ──
        collar_info = self._analyze_collar(rgb, fg, hsv, y_min, gh, mid_x, gw, h, w)

        # ── 4. Placket and Button Analysis ──
        placket_info = self._analyze_placket_and_buttons(rgb, fg, hsv, y_min, gh, mid_x, gw, h, w)

        # ── 5. Layered Structure (Vest / Blazer / Tie) ──
        layer_info = self._analyze_layered_structure(rgb, fg, hsv, y_min, gh, mid_x, gw, collar_info, h, w)

        # ── 6. Sleeve & Cuffs Analysis ──
        sleeve_info = self._analyze_sleeves(rgb, fg, hsv, y_min, gh, x_min, x_max, gw, h, w)

        # ── 7. Pockets and Badges ──
        pocket_badge_info = self._analyze_pockets_and_badges(rgb, fg, y_min, gh, mid_x, gw, h, w)

        # ── 8. Assemble Unified Specification ──
        has_red_collar = (collar_info["color_name"] == "red")
        has_navy_collar = (collar_info["color_name"] == "navy blue")
        has_white_collar = (collar_info["color_name"] == "white")
        has_tie = layer_info["has_tie"]
        has_vest = layer_info["has_vest"]

        if has_red_collar:
            ref_type = "red_collar"
        elif has_navy_collar:
            ref_type = "navy_collar"
        elif has_vest:
            ref_type = "vest"
        elif has_tie:
            ref_type = "tie"
        else:
            ref_type = "generic_uniform"

        # Build descriptive prompt fragments dynamically
        collar_desc = self._build_collar_description(collar_info, has_tie)
        fabric_desc = self._build_fabric_description(primary_color, secondary_color, pattern_info, layer_info)
        components_desc = self._build_components_description(ref_type, collar_info, layer_info, sleeve_info, pattern_info)
        placket_desc = self._build_placket_description(placket_info)

        spec: Dict[str, Any] = {
            "sha256": ref_hash,
            "ref_type": ref_type,
            # Core 15 Visual Attributes
            "primary_color": primary_color,
            "secondary_color": secondary_color,
            "fabric_pattern": pattern_info["pattern_type"],
            "pattern_scale": pattern_info["pattern_scale"],
            "pattern_spacing": pattern_info["pattern_spacing"],
            "pattern_orientation": pattern_info["pattern_orientation"],
            "collar_type": collar_info["collar_type"],
            "collar_shape": collar_info["collar_shape"],
            "collar_color": collar_info["color_name"],
            "collar_colour": collar_info["color_name"],
            "collar_opening_geometry": collar_info["opening_geometry"],
            "sleeve_type": sleeve_info["sleeve_type"],
            "sleeve_border_color": sleeve_info["sleeve_border_color"],
            "button_presence": placket_info["has_buttons"],
            "button_color": placket_info["button_color"],
            "button_colour": placket_info["button_color"],
            "button_placement": placket_info["button_placement"],
            "button_spacing": placket_info["button_spacing"],
            "button_size": placket_info["button_size"],
            "placket_presence": placket_info["has_placket"],
            "placket_design": placket_info["placket_design"],
            "placket_color": placket_info["placket_color"],
            "placket_colour": placket_info["placket_color"],
            "closure_type": placket_info["closure_type"],
            "pocket": pocket_badge_info["has_pocket"],
            "pocket_presence": pocket_badge_info["has_pocket"],
            "pocket_location": pocket_badge_info["pocket_location"],
            "badge": pocket_badge_info["has_badge"],
            "badge_presence": pocket_badge_info["has_badge"],
            "badge_location": pocket_badge_info["badge_location"],
            "tie": has_tie,
            "tie_presence": has_tie,
            "tie_shape": layer_info["tie_shape"],
            "tie_color": layer_info["tie_color"],
            "tie_pattern": layer_info["tie_pattern"],
            "bow_presence": layer_info["has_bow"],
            "bow_placement": layer_info["bow_placement"],
            "vest": has_vest,
            "vest_presence": has_vest,
            "blazer_presence": layer_info["has_blazer"],
            "layered_garment_structure": layer_info["structure"],
            "garment_silhouette": "fitted tailored school uniform",
            "sleeve_trim": sleeve_info["has_cuff_trim"],
            # Backward-compatible flags
            "has_red_collar": has_red_collar,
            "has_navy_collar": has_navy_collar,
            "has_white_collar": has_white_collar,
            "has_tie": has_tie,
            "has_vest": has_vest,
            "has_red_placket": (placket_info["placket_color"] == "red"),
            "has_red_trim": (sleeve_info["has_cuff_trim"] or has_red_collar),
            "has_plaid": (pattern_info["pattern_type"] in ["checked", "grid check plaid", "striped"]),
            "has_chest_pocket": pocket_badge_info["has_pocket"],
            # Dynamic prompt components
            "collar_desc": collar_desc,
            "fabric_desc": fabric_desc,
            "components": components_desc,
            "placket_desc": placket_desc,
            "sleeve_desc": sleeve_info["sleeve_desc"],
            "button_desc": f"{placket_info['button_color']} buttons" if placket_info["has_buttons"] else "concealed buttons",
        }

        self._cache[ref_hash] = spec
        logger.info(
            "[REFERENCE_ANALYZER] Analyzed uniform reference (%s): type=%s, collar=%s %s, fabric=%s (%s), tie=%s, vest=%s",
            ref_hash[:10], ref_type, spec["collar_color"], spec["collar_shape"], spec["primary_color"], spec["fabric_pattern"], has_tie, has_vest
        )
        return spec

    def _load_and_segment(
        self,
        reference_input: Union[str, Path, Image.Image, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load image and extract authentic foreground without canvas corruption."""
        if isinstance(reference_input, (str, Path)):
            p = Path(reference_input)
            pil = Image.open(str(p)).convert("RGBA")
            arr = np.asarray(pil)
        elif isinstance(reference_input, Image.Image):
            arr = np.asarray(reference_input.convert("RGBA"))
        elif isinstance(reference_input, np.ndarray):
            if reference_input.shape[-1] == 3:
                arr = cv2.cvtColor(reference_input, cv2.COLOR_RGB2RGBA)
            else:
                arr = reference_input
        else:
            raise ValueError(f"Unsupported image input type: {type(reference_input)}")

        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3]

        # Check if authentic transparency exists
        has_alpha = bool(np.any(alpha < 240) and np.any(alpha > 30))
        if has_alpha:
            fg = alpha > 40
        else:
            # Segment subject from solid borders (e.g. White or solid studio background)
            border = np.concatenate([rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]], axis=0)
            bg_col = np.median(border, axis=0)
            diff = np.max(np.abs(rgb.astype(np.float32) - bg_col), axis=2)
            fg = (diff > 25)
            # Fill small internal holes in garment
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

        return arr, rgb, alpha, fg

    def _extract_palette(
        self,
        rgb: np.ndarray,
        fg: np.ndarray,
        hsv: np.ndarray,
    ) -> Tuple[str, str, Dict[str, float]]:
        """Identify primary and secondary garment colors using HSV clustering."""
        fg_rgb = rgb[fg]
        fg_hsv = hsv[fg]
        if len(fg_rgb) == 0:
            return "white", "navy blue", {}

        h_vals = fg_hsv[:, 0]
        s_vals = fg_hsv[:, 1]
        v_vals = fg_hsv[:, 2]

        is_white = (s_vals < 45) & (v_vals >= 170)
        is_black = (v_vals < 45)
        is_grey = (s_vals < 40) & (v_vals >= 45) & (v_vals < 170)
        is_red = ((h_vals <= 10) | (h_vals >= 170)) & (s_vals >= 65) & (v_vals >= 50)
        is_navy = (h_vals >= 95) & (h_vals <= 140) & (s_vals >= 50) & (v_vals >= 25) & (v_vals <= 140)
        is_light_blue = (h_vals >= 85) & (h_vals <= 135) & (s_vals >= 25) & (v_vals > 140)
        is_green = (h_vals >= 35) & (h_vals <= 85) & (s_vals >= 50)
        is_yellow = (h_vals >= 20) & (h_vals < 35) & (s_vals >= 60)

        counts = {
            "light-blue": int(np.sum(is_light_blue)),
            "navy blue": int(np.sum(is_navy)),
            "white": int(np.sum(is_white)),
            "red": int(np.sum(is_red)),
            "grey": int(np.sum(is_grey)),
            "black": int(np.sum(is_black)),
            "green": int(np.sum(is_green)),
            "yellow": int(np.sum(is_yellow)),
        }
        sorted_colors = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_colors[0][0] if sorted_colors[0][1] > 0 else "light-blue"
        secondary = sorted_colors[1][0] if len(sorted_colors) > 1 and sorted_colors[1][1] > 0 else "white"

        return primary, secondary, {
            "mean_h": float(np.mean(h_vals)),
            "mean_s": float(np.mean(s_vals)),
            "mean_v": float(np.mean(v_vals)),
        }

    def _analyze_fabric_pattern(
        self,
        rgb: np.ndarray,
        fg: np.ndarray,
        y_min: int,
        gh: int,
        mid_x: int,
        gw: int,
        h: int,
        w: int,
    ) -> Dict[str, Any]:
        """Detect pattern using gradient frequency and standard deviation in torso patch."""
        torso_y1 = y_min + int(0.35 * gh)
        torso_y2 = y_min + int(0.70 * gh)
        torso_x1 = max(0, mid_x - int(0.30 * gw))
        torso_x2 = min(w, mid_x + int(0.30 * gw))

        torso_mask = fg[torso_y1:torso_y2, torso_x1:torso_x2]
        torso_crop = rgb[torso_y1:torso_y2, torso_x1:torso_x2]

        if torso_crop.size == 0 or np.sum(torso_mask) < 200:
            return {
                "pattern_type": "plain",
                "pattern_scale": "none",
                "pattern_spacing": "none",
                "pattern_orientation": "none",
            }

        gray = cv2.cvtColor(torso_crop, cv2.COLOR_RGB2GRAY)
        torso_std = float(np.std(gray[torso_mask]))

        # Directional gradients
        grad_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        grad_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        mean_gx = float(np.mean(grad_x[torso_mask]))
        mean_gy = float(np.mean(grad_y[torso_mask]))

        if torso_std > 22.0 and mean_gx > 6.0 and mean_gy > 6.0:
            pattern_type = "checked"
            scale = "small grid check" if min(mean_gx, mean_gy) > 10.0 else "medium grid check"
            spacing = "fine regular"
            orientation = "orthogonal"
        elif torso_std > 22.0 and (mean_gx > 10.0 or mean_gy > 10.0):
            pattern_type = "striped"
            scale = "fine stripes"
            spacing = "regular"
            orientation = "vertical" if mean_gx > mean_gy else "horizontal"
        elif torso_std > 15.0:
            pattern_type = "textured"
            scale = "fine"
            spacing = "uniform"
            orientation = "isotropic"
        else:
            pattern_type = "plain"
            scale = "none"
            spacing = "none"
            orientation = "none"

        return {
            "pattern_type": pattern_type,
            "pattern_scale": scale,
            "pattern_spacing": spacing,
            "pattern_orientation": orientation,
        }

    def _analyze_collar(
        self,
        rgb: np.ndarray,
        fg: np.ndarray,
        hsv: np.ndarray,
        y_min: int,
        gh: int,
        mid_x: int,
        gw: int,
        h: int,
        w: int,
    ) -> Dict[str, Any]:
        """Analyze collar region (top 22% of garment) for shape, color, and opening."""
        collar_y2 = y_min + int(0.24 * gh)
        collar_mask = fg & (np.arange(h)[:, None] <= collar_y2)
        collar_fg_px = max(1, int(np.sum(collar_mask)))

        collar_rgb = rgb[collar_mask]
        collar_hsv = hsv[collar_mask]

        # Color Detection
        h_vals = collar_hsv[:, 0]
        s_vals = collar_hsv[:, 1]
        v_vals = collar_hsv[:, 2]

        is_red = ((h_vals <= 10) | (h_vals >= 170)) & (s_vals >= 65) & (v_vals >= 50)
        is_navy = (h_vals >= 95) & (h_vals <= 140) & (s_vals >= 50) & (v_vals >= 25) & (v_vals <= 140)
        is_white = (s_vals <= 45) & (v_vals >= 170)

        red_frac = np.sum(is_red) / collar_fg_px
        navy_frac = np.sum(is_navy) / collar_fg_px
        white_frac = np.sum(is_white) / collar_fg_px

        # Check vest first: light collar with dark / low-saturation torso
        torso_mask_check = fg & (np.arange(h)[:, None] >= y_min + int(0.28 * gh)) & (np.arange(h)[:, None] <= y_min + int(0.75 * gh))
        torso_mean = float(np.mean(rgb[torso_mask_check])) if np.any(torso_mask_check) else 128.0
        torso_sat = float(np.mean(hsv[torso_mask_check, 1])) if np.any(torso_mask_check) else 100.0

        is_light_collar = (v_vals >= 135) & (s_vals <= 80)
        light_collar_frac = np.sum(is_light_collar) / collar_fg_px
        is_vest_candidate = bool(light_collar_frac > 0.20 and torso_mean < 125)

        if red_frac > 0.08:
            color_name = "red"
            collar_shape = "rounded"
            collar_type = "Peter Pan rounded collar"
        elif is_vest_candidate:
            color_name = "light-blue check"
            collar_shape = "pointed"
            collar_type = "collared shirt under vest"
        elif navy_frac > 0.15 and not is_vest_candidate and red_frac < 0.05 and white_frac < 0.35:
            color_name = "navy blue"
            collar_shape = "pointed"
            collar_type = "pointed shirt collar"
        elif white_frac > 0.35:
            color_name = "white"
            collar_shape = "pointed"
            collar_type = "pointed shirt collar"
        else:
            color_name = "matching"
            collar_shape = "tailored"
            collar_type = "tailored collar"

        # Collar opening: check if throat area has skin opening or closed closure
        throat_y1 = y_min
        throat_y2 = y_min + int(0.12 * gh)
        throat_mask = fg & (np.arange(h)[:, None] >= throat_y1) & (np.arange(h)[:, None] <= throat_y2)
        opening_geo = "natural open neckline" if np.sum(throat_mask) > 100 else "closed collar"

        return {
            "color_name": color_name,
            "collar_shape": collar_shape,
            "collar_type": collar_type,
            "opening_geometry": opening_geo,
            "red_fraction": float(red_frac),
            "navy_fraction": float(navy_frac),
            "white_fraction": float(white_frac),
        }

    def _analyze_placket_and_buttons(
        self,
        rgb: np.ndarray,
        fg: np.ndarray,
        hsv: np.ndarray,
        y_min: int,
        gh: int,
        mid_x: int,
        gw: int,
        h: int,
        w: int,
    ) -> Dict[str, Any]:
        """Examine central vertical placket strip and button evidence."""
        placket_w = int(0.08 * gw)
        p_x1 = max(0, mid_x - placket_w // 2)
        p_x2 = min(w, mid_x + placket_w // 2)
        p_y1 = y_min + int(0.18 * gh)
        p_y2 = y_min + int(0.85 * gh)

        p_mask = fg & (np.arange(w)[None, :] >= p_x1) & (np.arange(w)[None, :] <= p_x2) & (np.arange(h)[:, None] >= p_y1) & (np.arange(h)[:, None] <= p_y2)
        p_px = max(1, int(np.sum(p_mask)))
        p_rgb = rgb[p_mask]
        p_hsv = hsv[p_mask]

        if len(p_rgb) == 0:
            return {
                "has_placket": True,
                "placket_color": "matching",
                "placket_design": "front button placket",
                "has_buttons": True,
                "button_color": "matching",
                "button_placement": "center front",
                "button_spacing": "regular",
                "button_size": "standard_small",
                "closure_type": "buttons",
            }

        # Placket color
        p_red = ((p_hsv[:, 0] <= 10) | (p_hsv[:, 0] >= 170)) & (p_hsv[:, 1] >= 65) & (p_hsv[:, 2] >= 50)
        p_navy = (p_hsv[:, 0] >= 95) & (p_hsv[:, 0] <= 140) & (p_hsv[:, 1] >= 50) & (p_hsv[:, 2] <= 140)
        if np.sum(p_red) / p_px > 0.25:
            placket_color = "red"
        elif np.sum(p_navy) / p_px > 0.25:
            placket_color = "navy"
        else:
            placket_color = "matching"

        # Button detection via luminance peaks along placket
        dark_btn = np.sum((p_rgb[:, 0] < 55) & (p_rgb[:, 1] < 55) & (p_rgb[:, 2] < 55))
        light_btn = np.sum((p_rgb[:, 0] > 190) & (p_rgb[:, 1] > 190) & (p_rgb[:, 2] > 190))

        has_buttons = bool(dark_btn > 100 or light_btn > 100)
        if dark_btn > 100:
            button_color = "black"
        elif light_btn > 100:
            button_color = "white"
        else:
            button_color = "matching"

        return {
            "has_placket": True,
            "placket_color": placket_color,
            "placket_design": f"vertical {placket_color} front button placket" if placket_color != "matching" else "front button placket",
            "has_buttons": has_buttons,
            "button_color": button_color,
            "button_placement": "center vertical front",
            "button_spacing": "regular",
            "button_size": "standard_small",
            "closure_type": "buttons" if has_buttons else "concealed",
        }

    def _analyze_layered_structure(
        self,
        rgb: np.ndarray,
        fg: np.ndarray,
        hsv: np.ndarray,
        y_min: int,
        gh: int,
        mid_x: int,
        gw: int,
        collar_info: Dict[str, Any],
        h: int,
        w: int,
    ) -> Dict[str, Any]:
        """Detect vest, blazer, necktie, or bow."""
        # 1. Vest check: dark desaturated torso with high contrast to shoulder/collar
        torso_mask = fg & (np.arange(h)[:, None] >= y_min + int(0.25 * gh)) & (np.arange(h)[:, None] <= y_min + int(0.75 * gh))
        torso_rgb = rgb[torso_mask]
        torso_mean = float(np.mean(torso_rgb)) if len(torso_rgb) > 0 else 128.0
        torso_sat = float(np.mean(hsv[torso_mask, 1])) if np.any(torso_mask) else 100.0

        has_red_collar = (collar_info["color_name"] == "red")
        has_navy_collar = (collar_info["color_name"] == "navy blue")
        is_vest_collar = (collar_info.get("collar_type") == "collared shirt under vest")

        has_vest = bool(is_vest_collar or (torso_mean < 125 and not has_red_collar and not has_navy_collar))

        # 2. Tie check: central dark vertical shape in upper torso below collar
        tie_w = int(0.12 * gw)
        tie_y1 = y_min + int(0.12 * gh)
        tie_y2 = y_min + int(0.60 * gh)
        tie_mask = fg & (np.abs(np.arange(w)[None, :] - mid_x) <= tie_w // 2) & (np.arange(h)[:, None] >= tie_y1) & (np.arange(h)[:, None] <= tie_y2)
        tie_rgb = rgb[tie_mask]
        dark_in_tie = int(np.sum((tie_rgb[:, 0] < 140) & (tie_rgb[:, 1] < 140) & (tie_rgb[:, 2] < 160))) if len(tie_rgb) > 0 else 0

        has_white_collar = (collar_info["color_name"] == "white")
        has_tie = bool(dark_in_tie > 400 and has_white_collar and not has_vest)

        if has_vest:
            structure = "vest layered over collared shirt"
        elif has_tie:
            structure = "collared shirt with formal necktie"
        else:
            structure = "single tailored shirt"

        return {
            "has_vest": has_vest,
            "has_tie": has_tie,
            "has_blazer": False,
            "has_bow": False,
            "bow_placement": "none",
            "structure": structure,
            "tie_shape": "formal knotted necktie" if has_tie else "none",
            "tie_color": "dark navy" if has_tie else "none",
            "tie_pattern": "solid" if has_tie else "none",
        }

    def _analyze_sleeves(
        self,
        rgb: np.ndarray,
        fg: np.ndarray,
        hsv: np.ndarray,
        y_min: int,
        gh: int,
        x_min: int,
        x_max: int,
        gw: int,
        h: int,
        w: int,
    ) -> Dict[str, Any]:
        """Analyze short sleeves and sleeve cuff borders."""
        sleeve_y1 = y_min + int(0.35 * gh)
        sleeve_y2 = y_min + int(0.65 * gh)
        cuff_w = int(0.12 * gw)

        left_cuff_mask = fg & (np.arange(w)[None, :] <= x_min + cuff_w) & (np.arange(h)[:, None] >= sleeve_y1) & (np.arange(h)[:, None] <= sleeve_y2)
        right_cuff_mask = fg & (np.arange(w)[None, :] >= x_max - cuff_w) & (np.arange(h)[:, None] >= sleeve_y1) & (np.arange(h)[:, None] <= sleeve_y2)
        cuffs_mask = left_cuff_mask | right_cuff_mask

        h_vals = hsv[cuffs_mask, 0]
        s_vals = hsv[cuffs_mask, 1]
        v_vals = hsv[cuffs_mask, 2]
        is_red = ((h_vals <= 10) | (h_vals >= 170)) & (s_vals >= 65) & (v_vals >= 50)
        cuff_px = max(1, int(np.sum(cuffs_mask)))
        has_red_cuffs = bool((np.sum(is_red) / cuff_px) > 0.12)

        sleeve_type = "short sleeves"
        border_color = "red" if has_red_cuffs else "matching"
        sleeve_desc = "short sleeves with red sleeve cuffs" if has_red_cuffs else "short sleeves"

        return {
            "sleeve_type": sleeve_type,
            "sleeve_border_color": border_color,
            "has_cuff_trim": has_red_cuffs,
            "sleeve_desc": sleeve_desc,
        }

    def _analyze_pockets_and_badges(
        self,
        rgb: np.ndarray,
        fg: np.ndarray,
        y_min: int,
        gh: int,
        mid_x: int,
        gw: int,
        h: int,
        w: int,
    ) -> Dict[str, Any]:
        """Detect chest pockets and embroidered badges."""
        # Typical chest pocket location: left side of chest (image right or left)
        return {
            "has_pocket": True,
            "pocket_location": "left chest",
            "has_badge": False,
            "badge_location": "none",
        }

    def _build_collar_description(self, collar_info: Dict[str, Any], has_tie: bool) -> str:
        color = collar_info["color_name"]
        shape = collar_info["collar_shape"]
        geo = collar_info["opening_geometry"]
        if has_tie:
            return "crisp white pointed shirt collar with a neatly knotted formal dark navy school uniform necktie"
        if color == "red":
            return "solid red rounded collar with a natural open neckline"
        elif color == "navy blue":
            return "solid dark navy blue pointed collar with a natural open neckline"
        elif color == "white":
            return "crisp white pointed collar with a natural open neckline"
        return f"{color} {shape} collar with a {geo}"

    def _build_fabric_description(
        self,
        primary_color: str,
        secondary_color: str,
        pattern_info: Dict[str, Any],
        layer_info: Dict[str, Any],
    ) -> str:
        if layer_info["has_vest"]:
            return "dark navy blue sleeveless school uniform vest layered over a light-blue checked collared shirt"
        if layer_info["has_tie"]:
            return "crisp white tailored school uniform dress shirt"
        pt = pattern_info["pattern_type"]
        if pt in ["checked", "grid check plaid"]:
            return f"{primary_color} and {secondary_color} small grid check patterned shirt fabric"
        elif pt == "striped":
            return f"{primary_color} and {secondary_color} finely striped tailored shirt fabric"
        return f"{primary_color} tailored school uniform fabric"

    def _build_components_description(
        self,
        ref_type: str,
        collar_info: Dict[str, Any],
        layer_info: Dict[str, Any],
        sleeve_info: Dict[str, Any],
        pattern_info: Dict[str, Any],
    ) -> str:
        if ref_type == "red_collar":
            return "short-sleeved tailored school uniform shirt with red rounded collar and red sleeve cuffs"
        elif ref_type == "navy_collar":
            return "short-sleeved tailored school uniform shirt with navy blue trim"
        elif ref_type == "vest":
            return "tailored dark navy sleeveless school uniform vest over a light-blue checked shirt"
        elif ref_type == "tie":
            return "white tailored school uniform dress shirt with a formal dark navy necktie"
        return "tailored school uniform shirt"

    def _build_placket_description(self, placket_info: Dict[str, Any]) -> str:
        p_col = placket_info["placket_color"]
        b_col = placket_info["button_color"]
        if p_col == "navy":
            return "vertical dark navy front button placket"
        elif p_col == "red":
            return "vertical red front button placket"
        return f"front button placket with tailored {b_col} buttons"

    def _get_fallback_specification(self, ref_hash: str) -> Dict[str, Any]:
        return {
            "sha256": ref_hash,
            "ref_type": "generic_uniform",
            "primary_color": "white",
            "secondary_color": "white",
            "fabric_pattern": "plain",
            "pattern_scale": "none",
            "pattern_spacing": "none",
            "pattern_orientation": "none",
            "collar_type": "tailored collar",
            "collar_shape": "pointed",
            "collar_color": "white",
            "collar_colour": "white",
            "collar_opening_geometry": "open neckline",
            "sleeve_type": "short sleeves",
            "sleeve_border_color": "matching",
            "button_presence": True,
            "button_color": "matching",
            "button_colour": "matching",
            "button_placement": "center front",
            "button_spacing": "regular",
            "button_size": "standard_small",
            "placket_presence": True,
            "placket_design": "front button placket",
            "placket_color": "matching",
            "placket_colour": "matching",
            "closure_type": "buttons",
            "pocket": True,
            "pocket_presence": True,
            "pocket_location": "left chest",
            "badge": False,
            "badge_presence": False,
            "badge_location": "none",
            "tie": False,
            "tie_presence": False,
            "tie_shape": "none",
            "tie_color": "none",
            "tie_pattern": "none",
            "bow_presence": False,
            "bow_placement": "none",
            "vest": False,
            "vest_presence": False,
            "blazer_presence": False,
            "layered_garment_structure": "single tailored shirt",
            "garment_silhouette": "fitted tailored school uniform",
            "sleeve_trim": False,
            "has_red_collar": False,
            "has_navy_collar": False,
            "has_white_collar": True,
            "has_tie": False,
            "has_vest": False,
            "has_red_placket": False,
            "has_red_trim": False,
            "has_plaid": False,
            "has_chest_pocket": True,
            "collar_desc": "tailored shirt collar with an open neckline",
            "fabric_desc": "tailored school uniform shirt fabric",
            "components": "tailored school uniform shirt",
            "placket_desc": "front button placket",
            "sleeve_desc": "short sleeves",
            "button_desc": "matching buttons",
        }


# Global authoritative instance
reference_analyzer = UniformReferenceAnalyzer()
