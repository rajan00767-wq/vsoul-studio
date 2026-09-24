import unittest
import numpy as np
from PIL import Image
from pipelines.uniform_composite import (
    build_rough_composite, describe_fabric_color, describe_hair_correction, refinement_prompt,
)


class CompositeTests(unittest.TestCase):
    def setUp(self):
        self.person = Image.new("RGB", (100, 140), (180, 130, 90))
        self.garment = Image.new("RGBA", (100, 80), (10, 20, 40, 0))
        self.garment.paste((10, 20, 40, 255), (10, 10, 90, 80))
        self.labels = np.zeros((140, 100), dtype=np.uint8)
        self.labels[80:, 5:95] = 5

    def test_preserves_head_and_places_garment(self):
        result, meta = build_rough_composite(self.person, self.garment, self.labels,
                                            (30, 20, 70, 70), (4, 126, 246))
        np.testing.assert_array_equal(np.array(result)[:70], np.array(self.person)[:70])
        self.assertEqual(result.getpixel((50, 100)), (10, 20, 40))
        self.assertTrue(meta["not_final"])

    def test_requires_cutout(self):
        with self.assertRaisesRegex(ValueError, "transparent"):
            build_rough_composite(self.person, self.garment.convert("RGB"), self.labels,
                                  (30, 20, 70, 70), (4, 126, 246))

    def test_prompt_has_fabric_and_identity_roles(self):
        prompt = refinement_prompt((4, 126, 246), "TEST BLACK FABRIC", "TEST HAIR CORRECTION")
        for phrase in ("only identity reference", "authoritative uniform reference",
                       "pattern-line color", "weave", "spacing", "Remove badges", "(4, 126, 246)",
                       "TEST BLACK FABRIC", "TEST HAIR CORRECTION", "exact facial geometry"):
            self.assertIn(phrase, prompt)

    def test_dark_cool_pixels_are_described_as_near_black(self):
        garment = Image.new("RGBA", (20, 20), (26, 36, 52, 255))
        description = describe_fabric_color(garment)
        self.assertIn("near-black", description)
        self.assertIn("not navy blue", description)

    def test_black_hair_analysis_targets_only_glare(self):
        description = describe_hair_correction("black")
        self.assertIn("outdoor glare", description)
        self.assertIn("adjacent dark hair", description)

    def test_gray_hair_is_not_darkened(self):
        description = describe_hair_correction("gray")
        self.assertIn("do not darken genuinely light, gray or colored hair", description)


if __name__ == "__main__":
    unittest.main()
