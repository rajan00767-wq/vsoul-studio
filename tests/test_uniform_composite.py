import unittest
import numpy as np
from PIL import Image
from pipelines.uniform_composite import (
    build_person_backdrop_guide, build_person_identity_guide, build_rough_composite, describe_fabric_color, describe_hair_correction,
    describe_lighting_correction, refinement_prompt,
    ensure_garment_cutout, garment_on_selected_background, opaque_garment_cutout,
    measure_source_person_colors, source_hair_is_dark,
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

    def test_person_guide_keeps_subject_and_removes_scenery(self):
        result = build_person_backdrop_guide(self.person, self.labels, (4, 126, 246))
        self.assertEqual(result.getpixel((0, 0)), (4, 126, 246))
        self.assertEqual(result.getpixel((50, 40)), self.person.getpixel((50, 40)))
        self.assertEqual(result.getpixel((50, 110)), self.person.getpixel((50, 110)))

    def test_uniform_edit_canvas_keeps_identity_and_blanks_source_clothing(self):
        result = build_person_backdrop_guide(
            self.person, self.labels, (4, 126, 246), remove_clothing=True,
        )
        self.assertEqual(result.getpixel((50, 40)), self.person.getpixel((50, 40)))
        self.assertEqual(result.getpixel((50, 110)), (4, 126, 246))

    def test_identity_guide_is_an_untouched_rectangular_source_crop(self):
        result = build_person_identity_guide(
            self.person, self.labels, (30, 20, 70, 70),
        )
        source = np.asarray(self.person)
        crop = np.asarray(result)
        self.assertGreaterEqual(result.width, 40)
        self.assertGreaterEqual(result.height, 50)
        self.assertTrue(any(
            np.array_equal(crop, source[top:top + result.height, left:left + result.width])
            for top in range(source.shape[0] - result.height + 1)
            for left in range(source.shape[1] - result.width + 1)
        ))

    def test_selected_background_does_not_bleed_through_white_fabric(self):
        garment = Image.new("RGB", (100, 80), (255, 255, 255))
        garment.paste((245, 245, 245), (10, 10, 90, 70))
        garment.paste((80, 82, 86), (25, 20, 75, 70))
        result = garment_on_selected_background(garment, (4, 126, 246))
        self.assertEqual(result.getpixel((0, 0)), (4, 126, 246))
        self.assertEqual(result.getpixel((15, 15)), (245, 245, 245))
        self.assertEqual(result.getpixel((50, 40)), (80, 82, 86))

    def test_rough_cutout_has_opaque_white_fabric(self):
        garment = Image.new("RGB", (100, 80), (255, 255, 255))
        garment.paste((245, 245, 245), (10, 10, 90, 70))
        garment.paste((80, 82, 86), (25, 20, 75, 70))
        cutout = opaque_garment_cutout(garment)
        self.assertEqual(cutout.getpixel((15, 15)), (245, 245, 245, 255))

    def test_source_person_colors_use_face_and_shaded_hair_pixels(self):
        portrait = np.full((140, 100, 3), (250, 250, 250), dtype=np.uint8)
        portrait[20:70, 30:70] = (171, 116, 82)
        portrait[5:35, 20:80] = (30, 24, 21)
        portrait[5:10, 20:80] = (210, 190, 160)  # glare must not define hair pigment
        labels = np.zeros((140, 100), dtype=np.uint8)
        labels[20:70, 30:70] = 13
        labels[5:35, 20:80] = 2
        colors = measure_source_person_colors(Image.fromarray(portrait), labels)
        self.assertEqual(colors["skin_rgb"], (171, 116, 82))
        self.assertEqual(colors["hair_rgb"], (30, 24, 21))

    def test_prompt_has_fabric_and_identity_roles(self):
        prompt = refinement_prompt(
            (4, 126, 246), "TEST BLACK FABRIC", "TEST HAIR CORRECTION", None,
            "TEST MEASURED SKIN TONE", "TEST IDENTITY CROP",
        )
        for phrase in ("REFINE IMAGE 1'S ROUGH UNIFORM LAYOUT", "IMAGE 2 IS THE ONLY IDENTITY SOURCE", "IMAGE 3 IS THE GARMENT AUTHORITY",
                       "pattern scale", "weave", "Remove badges", "#047EF6",
                       "TEST BLACK FABRIC", "TEST HAIR CORRECTION", "exact face"):
            self.assertIn(phrase, prompt)
        self.assertIn("Photographic restoration, not person recreation", prompt)
        self.assertIn("do not regenerate or redesign the face", prompt)
        self.assertIn("RGB(4, 126, 246)", prompt)
        self.assertIn("no gradient, texture or scenery", prompt)
        self.assertIn("TEST MEASURED SKIN TONE", prompt)
        self.assertIn("TEST IDENTITY CROP", prompt)
        self.assertIn("IMAGE 2 IS THE ONLY IDENTITY SOURCE", prompt)
        self.assertIn("IMAGE 3 IS THE GARMENT AUTHORITY", prompt)
        for phrase in ("IMAGE 2", "#047EF6", "texture or scenery"):
            self.assertIn(phrase, prompt)
        self.assertIn("FINAL BACKGROUND", prompt)
        for phrase in (
            "Camera-facing: upright head",
            "level eyes",
            "square even shoulders",
            "a photorealistic portrait, not a pasted composite",
            "shoulders and upper chest",
            "application will crop to passport size",
        ):
            self.assertIn(phrase, prompt)
        for phrase in ("Remove all necklaces", "earrings", "ethnicity"):
            self.assertIn(phrase, prompt)
        for phrase in ("chains, pendants", "Preserve exact face"):
            self.assertIn(phrase, prompt)
        self.assertIn(
            "skin color and undertone directly from image 2",
            refinement_prompt((4, 126, 246)),
        )
        self.assertNotIn("Preserve jewelry", prompt)
        self.assertLessEqual(len(prompt.split()), 305)
        self.assertIn("IMAGE 2 IS THE SOFT-EDGED ORIGINAL IDENTITY/HAIR GUIDE", prompt)
        self.assertIn("hairline, hairstyle", prompt)

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
        self.assertIn("image 2's black/dark-brown hair", description)
        self.assertIn("false colored glare", description)
        self.assertIn("only accessories that are visibly present", description)

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
