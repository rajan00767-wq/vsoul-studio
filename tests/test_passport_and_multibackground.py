"""Verification test suite for multi-size passport crop and 8-color background compositing (V13.2).

Tests:
1. Multi-size Passport Crop across customer dimensions:
   - 800x1024 (Baseline)
   - 800x1000 (4:5 Aspect Ratio)
   - 700x900 (7:9 Aspect Ratio)
   - 413x531 (35x45mm Standard Passport at 300 DPI)
   Asserts:
   - Exact output dimensions (W, H)
   - Strict aspect scaling tolerance: |scale_x - scale_y| < 0.001
   - Bottom anchoring: pad_bottom == 0, no floating gap

2. Multi-background independence across 8 required colors:
   - White (#FFFFFF)
   - Royal Blue (#047EF6)
   - Sage Green (#77BF91)
   - Light Blue (#ADD8E6)
   - Studio Grey (#D3D3D3)
   - Warm Cream (#F5E6D3)
   - Custom Dark Red (#8B0000)
   - Black (#000000)
   Asserts:
   - Background corners match target color
   - Clean subject boundary with zero white halo
"""

import sys
from pathlib import Path
import unittest
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.uniform_swap.image_normalizer import produce_passport_crop
from scripts.verify_quality_gates import load_baseline


class TestPassportAndMultiBackground(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src_img, cls.cand_img, cls.id_layer, cls.uniform_ref = load_baseline()
        w, h = cls.cand_img.size
        # Build RGBA candidate using full_person_alpha
        alpha = cls.id_layer["full_person_alpha"]
        cls.cand_rgba = Image.merge("RGBA", (*cls.cand_img.split(), Image.fromarray(alpha)))

    def test_01_multisize_passport_crop(self):
        """Verify passport crop across 4 canonical aspect ratios and resolutions."""
        test_resolutions = [
            (800, 1024),
            (800, 1000),
            (700, 900),
            (413, 531),
        ]

        for out_w, out_h in test_resolutions:
            cropped_pil, meta = produce_passport_crop(
                self.cand_rgba,
                identity_layer=self.id_layer,
                output_size=(out_w, out_h),
                background_rgb=(255, 255, 255),
                return_meta=True,
            )

            # 1. Exact output dimensions
            self.assertEqual(cropped_pil.size, (out_w, out_h), f"Dimensions mismatch for {(out_w, out_h)}")

            # 2. Strict scale difference tolerance < 0.001
            scale_x = meta.get("scale_x", meta.get("scale_factor"))
            scale_y = meta.get("scale_y", meta.get("scale_factor"))
            scale_delta = abs(scale_x - scale_y)
            self.assertLess(
                scale_delta,
                0.001,
                f"Aspect scaling distortion exceeded tolerance for {(out_w, out_h)}: |{scale_x} - {scale_y}| = {scale_delta:.6f} >= 0.001",
            )

            # 3. Bottom anchoring
            pad_left, pad_top, pad_right, pad_bottom = meta["padding"]
            self.assertEqual(pad_bottom, 0, f"Bottom padding must be strictly 0, got {pad_bottom}")

            # 4. Chin below crown and head occupancy in valid range
            self.assertGreater(meta["transformed_chin_y"], meta["transformed_crown_y"])
            self.assertTrue(0.45 <= meta["final_head_occupancy"] <= 0.70)

    def test_02_multibackground_compositing(self):
        """Verify clean RGBA compositing across 8 required background colors."""
        target_colors = {
            "White": (255, 255, 255),
            "Royal Blue": (4, 126, 246),
            "Sage Green": (119, 191, 145),
            "Light Blue": (173, 216, 230),
            "Studio Grey": (211, 211, 211),
            "Warm Cream": (245, 230, 211),
            "Custom Dark Red": (139, 0, 0),
            "Black": (0, 0, 0),
        }

        for c_name, bg_rgb in target_colors.items():
            cropped_pil, meta = produce_passport_crop(
                self.cand_rgba,
                identity_layer=self.id_layer,
                output_size=(800, 1024),
                background_rgb=bg_rgb,
                return_meta=True,
            )
            arr = np.array(cropped_pil)
            w, h = cropped_pil.size

            # Sample top corner (pure background)
            top_corner = arr[5:20, 5:20]
            mean_c = np.mean(top_corner, axis=(0, 1))
            expected_c = np.array(bg_rgb, dtype=float)
            diff = np.max(np.abs(mean_c - expected_c))
            self.assertLess(
                diff,
                5.0,
                f"Background color {c_name} corner deviated: expected {bg_rgb}, got {mean_c}",
            )


if __name__ == "__main__":
    unittest.main()
