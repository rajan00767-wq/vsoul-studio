import unittest
import numpy as np
from PIL import Image
from pipelines.uniform_composite import (
    build_rough_composite, describe_fabric_color, describe_hair_correction,
    describe_lighting_correction, refinement_prompt,
    ensure_garment_cutout, source_hair_is_dark,
)


class CompositeTests(unittest.TestCase):
    def setUp(self):
        self.person = Image.new("RGB", (100, 140), (180, 130, 90))
        self.garment = Image.new("RGBA", (100, 80), (10, 20, 40, 0))
        self.garment.paste((10, 20, 40, 255), (10, 10, 90, 80))
        self.labels = np.zeros((140, 100), dtype=np.uint8)
        self.labels[20:70, 30:70] = 13
        self.labels[80:, 5:95] = 5

    def test_preserves_head_and_places_garment(self):
        result, meta = build_rough_composite(self.person, self.garment, self.labels,
                                            (30, 20, 70, 70), (4, 126, 246))
        self.assertEqual(result.getpixel((0, 0)), (4, 126, 246))
        self.assertEqual(result.getpixel((50, 40)), self.person.getpixel((50, 40)))
        self.assertEqual(result.getpixel((50, 100)), (10, 20, 40))
        self.assertTrue(meta["not_final"])

    def test_derives_cutout_from_solid_product_background(self):
        garment = Image.new("RGB", (100, 80), (255, 255, 255))
        garment.paste((80, 82, 86), (10, 10, 90, 80))
        cutout = ensure_garment_cutout(garment)
        self.assertEqual(cutout.getpixel((0, 0))[3], 0)
        self.assertGreater(cutout.getpixel((50, 40))[3], 240)
        result, meta = build_rough_composite(self.person, garment, self.labels,
                                             (30, 20, 70, 70), (4, 126, 246))
        self.assertEqual(result.size, self.person.size)
        self.assertTrue(meta["not_final"])

    def test_prompt_has_fabric_and_identity_roles(self):
        prompt = refinement_prompt(
            (4, 126, 246), "TEST BLACK FABRIC", "TEST HAIR CORRECTION", None,
            "TEST MEASURED SKIN TONE", "TEST IDENTITY CROP",
        )
        for phrase in ("EDIT IMAGE 1", "IMAGES 1 AND 2 ARE THE SAME PERSON", "IMAGE 3 IS UNIFORM ONLY",
                       "pattern scale", "weave", "Remove badges", "#047EF6",
                       "TEST BLACK FABRIC", "TEST HAIR CORRECTION", "exact face"):
            self.assertIn(phrase, prompt)
        self.assertIn("TEST MEASURED SKIN TONE", prompt)
        self.assertIn("TEST IDENTITY CROP", prompt)
        self.assertIn("IMAGES 1 AND 2 ARE THE SAME PERSON", prompt)
        self.assertIn("IMAGE 3 IS UNIFORM ONLY", prompt)
        for phrase in ("EDIT IMAGE 1", "#047EF6", "No original scenery", "gradient or texture"):
            self.assertIn(phrase, prompt)
        self.assertIn("FINAL BACKGROUND", prompt)
        for phrase in ("frontal gaze", "level shoulders"):
            self.assertIn(phrase, prompt)
        for phrase in ("Remove all necklaces", "earrings", "ethnicity"):
            self.assertIn(phrase, prompt)
        for phrase in ("chains, pendants", "override VL"):
            self.assertIn(phrase, prompt)
        self.assertIn(
            "skin color and undertone directly from image 1",
            refinement_prompt((4, 126, 246)),
        )
        self.assertNotIn("Preserve jewelry", prompt)
        self.assertLessEqual(len(prompt.split()), 225)

    def test_vl_decides_outer_color_without_fixed_preset(self):
        garment = Image.new("RGBA", (20, 20), (26, 36, 52, 255))
        description = describe_fabric_color(garment, "smoky gray", [62, 61, 59])
        self.assertIn("smoky gray", description)
        self.assertIn("(62, 61, 59)", description)
        self.assertIn("image 3 pixels remain authoritative", description)

    def test_missing_vl_color_uses_reference_without_named_color(self):
        garment = Image.new("RGBA", (20, 20), (26, 36, 52, 255))
        description = describe_fabric_color(garment)
        self.assertIn("VL color is uncertain", description)
        self.assertIn("without a named-color preset", description)

    def test_black_hair_analysis_targets_only_glare(self):
        description = describe_hair_correction("black")
        self.assertIn("black/dark-brown hair", description)
        self.assertIn("false colored glare", description)
        self.assertIn("flowers and clips", description)

    def test_gray_hair_is_not_darkened(self):
        description = describe_hair_correction("gray")
        self.assertIn("do not darken genuinely light, gray or colored hair", description)

    def test_vl_sunlight_becomes_identity_safe_relighting_instruction(self):
        description = describe_lighting_correction({
            "direct_sunlight_present": True,
            "head_hair_hotspot_present": True,
            "sunlight_type": "hard direct sun",
            "sunlight_direction": "upper left",
            "sunlight_regions": "crown, roots, forehead",
            "sunlight_evidence": "silver crown glare and a hard forehead boundary",
        })
        for phrase in ("neutral indoor studio light", "natural detail"):
            self.assertIn(phrase, description)
        for phrase in ("upper left", "reconstruct", "balance the forehead", "reinterpret"):
            self.assertNotIn(phrase, description)

    def test_dark_hair_measurement_uses_shaded_strands(self):
        pixels = np.full((20, 20, 3), (205, 195, 180), dtype=np.uint8)
        pixels[:10] = (32, 28, 26)
        labels = np.full((20, 20), 2, dtype=np.uint8)
        self.assertTrue(source_hair_is_dark(Image.fromarray(pixels), labels))


if __name__ == "__main__":
    unittest.main()
