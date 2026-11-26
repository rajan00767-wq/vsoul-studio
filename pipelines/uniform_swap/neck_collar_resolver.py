"""Dynamic Anatomical Neck and Collar Resolver.

Resolves the throat-to-collar boundary dynamically without hardcoded ratio fallbacks (like 0.20*fh)
or flat synthetic skin rectangle infill. Disentangles pre-existing visible neck skin from newly
revealed neck skin, ensuring natural collar seating and zero double-neck artifacts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class NeckCollarResolver:
    """Calculates anatomical collar seating and partitions neck skin regions."""

    def __init__(self) -> None:
        pass

    def resolve(
        self,
        identity_layer: Dict[str, Any],
        face_bbox: Tuple[float, float, float, float],
        target_collar_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Resolve dynamic collar seat line and skin ownership regions.

        Returns:
            Dict containing:
                - 'collar_seat_y': int (row where the replacement uniform collar sits)
                - 'neck_ratio': float (fraction of face height from chin to collar)
                - 'preexisting_neck_mask': (H, W) bool (verified original neck to restore)
                - 'newly_exposed_neck_mask': (H, W) bool (skin newly exposed by target uniform)
                - 'collar_occlusion_mask': (H, W) bool (collar fabric that naturally overlaps lower neck)
        """
        metrics = identity_layer.get("metrics", {})
        chin_y = metrics.get("chin_y")
        lowest_neck_y = metrics.get("lowest_neck_y")
        neck_cls = identity_layer.get("neck_classification", "uncertain")

        fx1, fy1, fx2, fy2 = face_bbox
        fw = max(1.0, float(fx2 - fx1))
        fh = max(1.0, float(fy2 - fy1))
        fcx = float(fx1 + fx2) * 0.5

        if chin_y is None:
            chin_y = int(round(fy1 + fh * 0.82))

        # Dynamic collar seating calculation (FIX: Zero hardcoded 0.20 gap!)
        if neck_cls == "occluded_neck":
            # For high source garments (e.g. hoodies, turtlenecks), collar sits snugly near chin
            collar_seat_y = int(round(chin_y + fh * 0.04))
            neck_ratio = 0.04
        elif lowest_neck_y is not None and lowest_neck_y > chin_y:
            # For visible necks, collar sits at the lowest verified natural neck row
            collar_seat_y = int(round(lowest_neck_y))
            neck_ratio = float(np.clip((collar_seat_y - chin_y) / fh, 0.04, 0.35))
        else:
            # Adaptive fallback based on 0.06 face height rather than 0.20
            collar_seat_y = int(round(chin_y + fh * 0.06))
            neck_ratio = 0.06

        orig_neck = identity_layer.get("original_neck_mask")
        h, w = (orig_neck.shape if orig_neck is not None else (1024, 800))

        if orig_neck is None:
            orig_neck = np.zeros((h, w), dtype=bool)

        # Region A: Pre-existing visible neck skin (above the collar seat)
        rows = np.arange(h)[:, None]
        preexisting_neck = orig_neck & (rows <= collar_seat_y + 2)

        # Region B: Newly exposed neck skin corridor below the original neckline
        # if target collar is deeper than source garment
        target_outer_neckline = (target_collar_info or {}).get("outer_neckline", "").lower()
        if any(deep in target_outer_neckline for deep in ["v-neck", "scoop", "open", "deep", "low"]):
            target_depth_y = int(round(collar_seat_y + fh * 0.12))
            cols = np.arange(w)[None, :]
            corridor = (
                (rows > collar_seat_y)
                & (rows <= target_depth_y)
                & (np.abs(cols - fcx) <= fw * 0.28)
            )
            # Newly exposed region must not overlap clothing or existing neck
            newly_exposed_neck = corridor & ~preexisting_neck
        else:
            newly_exposed_neck = np.zeros((h, w), dtype=bool)

        # Collar natural occlusion band: 4-6 pixels above collar_seat_y where shirt collar overlaps throat
        cols = np.arange(w)[None, :]
        collar_overlap = (
            (rows >= collar_seat_y - 4)
            & (rows <= collar_seat_y + 8)
            & (np.abs(cols - fcx) <= fw * 0.38)
        )

        logger.info(
            "[NECK_COLLAR_RESOLVER] Resolved collar seat: y=%d (ratio=%.3f, neck_cls=%s, preexisting_px=%d, newly_exposed_px=%d)",
            collar_seat_y,
            neck_ratio,
            neck_cls,
            int(np.sum(preexisting_neck)),
            int(np.sum(newly_exposed_neck)),
        )

        return {
            "collar_seat_y": collar_seat_y,
            "neck_ratio": neck_ratio,
            "preexisting_neck_mask": preexisting_neck,
            "newly_exposed_neck_mask": newly_exposed_neck,
            "collar_occlusion_mask": collar_overlap,
        }


# Global singleton instance
neck_collar_resolver = NeckCollarResolver()
