import unittest
import numpy as np
from PIL import Image
from pipelines.uniform_composite import (
    _measure_garment_neck_opening,
    build_neutral_body_fit_guide, build_person_backdrop_guide, build_person_identity_guide, build_rough_composite, describe_fabric_color, describe_hair_correction,
    describe_lighting_correction, refinement_prompt,
    ensure_garment_cutout, garment_on_selected_background, opaque_garment_cutout,
    measure_source_person_colors, source_hair_is_dark,
    hide_product_rear_collar,
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
        # Preserve the real neck immediately below the jaw, but let the
        # supplied collar take over before a fixed neck extension can run
        # through the uniform.
        neck_pixel = result.getpixel((50, 71))
        source_neck = self.person.getpixel((50, 71))
        self.assertLess(max(abs(a - b) for a, b in zip(neck_pixel, source_neck)), 40)
        # Preserve the upload's measured jaw-to-collar distance, then feather
        # into the supplied collar rather than imposing a fixed neck length.
        self.assertLess(
            max(abs(a - b) for a, b in zip(result.getpixel((50, 74)), source_neck)),
            40,
        )
        self.assertAlmostEqual(meta["source_neck_gap_ratio"], .20, places=2)
        self.assertNotEqual(result.getpixel((50, 90)), source_neck)
        garment_pixel = result.getpixel((50, 100))
        self.assertLess(max(abs(a - b) for a, b in zip(garment_pixel, (10, 20, 40))), 3)
        self.assertTrue(meta["not_final"])

    def test_rough_fit_has_continuous_source_neck_into_uniform(self):
        # Simulate a parser that detects the face but misses the neck entirely.
        # The rough guide must still be one connected portrait, not a floating
        # head above a separately pasted garment.
        result, _ = build_rough_composite(
            self.person,
            self.garment,
            self.labels,
            (30, 20, 70, 70),
            (4, 126, 246),
        )
        centre = result.width // 2
        for y in range(68, 81):
            self.assertNotEqual(result.getpixel((centre, y)), (4, 126, 246))

    def test_rough_fit_does_not_extend_skin_scan_through_collar_opening(self):
        pixels = np.full((180, 120, 3), (225, 225, 225), dtype=np.uint8)
        skin = (170, 108, 78)
        pixels[20:80, 40:80] = skin
        pixels[80:125, 53:67] = skin  # false skin-coloured shirt opening
        pixels[86:, 8:112] = (60, 64, 72)
        labels = np.zeros((180, 120), dtype=np.uint8)
        labels[20:80, 40:80] = 13
        labels[86:, 8:112] = 5
        garment = Image.new("RGBA", (120, 100), (12, 28, 50, 0))
        garment.paste((12, 28, 50, 255), (8, 5, 112, 100))

        result, meta = build_rough_composite(
            Image.fromarray(pixels), garment, labels,
            (40, 20, 80, 80), (4, 126, 246),
        )

        collar_y = meta["garment_box"][1]
        max_extension = int(round((80 - 20) * .045)) + 2
        self.assertNotEqual(result.getpixel((60, collar_y + max_extension + 3)), skin)

    def test_hides_unworn_rear_collar_without_removing_collar_leaves(self):
        garment = np.zeros((180, 140, 4), dtype=np.uint8)
        garment[20:170, 10:130] = (70, 72, 76, 255)
        garment[24:38, 45:95] = (230, 25, 35, 255)  # rear collar band
        garment[28:85, 28:52] = (230, 25, 35, 255)  # left collar leaf
        garment[28:85, 88:112] = (230, 25, 35, 255)  # right collar leaf

        cleaned, removed = hide_product_rear_collar(Image.fromarray(garment, "RGBA"))

        self.assertGreater(removed, 0)
        self.assertEqual(cleaned.getpixel((70, 30))[3], 0)
        self.assertEqual(cleaned.getpixel((70, 36))[3], 0)
        self.assertEqual(cleaned.getpixel((38, 65))[3], 255)
        self.assertEqual(cleaned.getpixel((102, 65))[3], 255)

    def test_colored_front_collar_at_garment_top_is_not_stripped(self):
        garment = np.zeros((180, 140, 4), dtype=np.uint8)
        # Red folded collar forming the top edge of the shirt
        garment[20:50, 40:100] = (220, 25, 35, 255)
        # Shirt body below
        garment[50:170, 10:130] = (70, 72, 76, 255)
        cleaned, removed = hide_product_rear_collar(Image.fromarray(garment, "RGBA"))
        self.assertEqual(removed, 0)
        self.assertEqual(cleaned.getpixel((70, 30))[3], 255)

    def test_rough_fit_preserves_face_edge_when_parser_misses_it(self):
        pixels = np.full((140, 100, 3), (225, 225, 225), dtype=np.uint8)
        pixels[20:70, 30:70] = (166, 104, 76)
        pixels[44:48, 28:32] = (142, 84, 64)  # ear/temple at face edge
        pixels[80:, 5:95] = (70, 72, 76)
        labels = np.zeros((140, 100), dtype=np.uint8)
        labels[20:70, 32:70] = 13  # parser misses the left face edge
        labels[80:, 5:95] = 5

        result, _ = build_rough_composite(
            Image.fromarray(pixels), self.garment, labels,
            (30, 20, 70, 70), (4, 126, 246),
        )

        self.assertEqual(result.getpixel((29, 45)), (142, 84, 64))
        # A geometric face oval used to copy this pale source-background pixel
        # as a visible cheek halo. Only the real skin-coloured edge is kept.
        self.assertEqual(result.getpixel((28, 40)), (4, 126, 246))

    def test_rough_fit_removes_lower_arm_fragments_from_passport_guide(self):
        pixels = np.full((180, 120, 3), (230, 230, 230), dtype=np.uint8)
        pixels[20:80, 40:80] = (166, 104, 76)
        pixels[86:180, 10:110] = (70, 72, 76)
        pixels[160:180, :10] = (166, 104, 76)
        pixels[160:180, 110:] = (166, 104, 76)
        labels = np.zeros((180, 120), dtype=np.uint8)
        labels[20:80, 40:80] = 13
        labels[86:180, 10:110] = 5
        labels[160:180, :10] = 14
        labels[160:180, 110:] = 15
        garment = Image.new("RGBA", (120, 100), (12, 28, 50, 0))
        garment.paste((12, 28, 50, 255), (8, 5, 112, 100))

        result, _ = build_rough_composite(
            Image.fromarray(pixels), garment, labels,
            (40, 20, 80, 80), (4, 126, 246),
        )

        self.assertEqual(result.getpixel((5, 170)), (4, 126, 246))
        self.assertEqual(result.getpixel((115, 170)), (4, 126, 246))

    def test_folded_collar_adds_only_missing_natural_neck_gap(self):
        labels = self.labels.copy()
        labels[70:, :] = 0
        labels[73:, 5:95] = 5
        # Give the fixture a real shirt under the chin: the neck measurement
        # reads pixel colour, so a flat skin-coloured torso would be reported as
        # an unreasonably long bare neck.
        person = self.person.copy()
        person.paste((40, 60, 110), (0, 74, 100, 140))
        _, measured = build_rough_composite(
            person, self.garment, labels,
            (30, 20, 70, 70), (4, 126, 246),
        )
        _, adjusted = build_rough_composite(
            person, self.garment, labels,
            (30, 20, 70, 70), (4, 126, 246),
            minimum_neck_gap_ratio=.10,
        )
        # The seat follows this portrait's own visible neck, not a fixed value.
        self.assertAlmostEqual(measured["neck_seat_ratio"], .07, places=2)
        self.assertAlmostEqual(measured["source_neck_gap_ratio"], .07, places=2)
        self.assertAlmostEqual(adjusted["measured_source_neck_gap_ratio"], .06, places=2)
        self.assertAlmostEqual(adjusted["source_neck_gap_ratio"], .10, places=2)
        # A caller asking for a longer neck may only move the collar downward;
        # it must never rise back into the jaw.
        self.assertGreaterEqual(
            adjusted["garment_box"][1], measured["garment_box"][1],
        )

    def test_rough_fit_keeps_lens_reflection_and_nearby_missed_hair(self):
        pixels = np.full((140, 100, 3), (210, 190, 170), dtype=np.uint8)
        pixels[:20] = (90, 150, 90)
        pixels[8:32, 24:76] = (28, 24, 22)
        pixels[16:27, 18:24] = (30, 25, 23)  # hair omitted by parser
        pixels[34:38, 34:40] = (235, 245, 250)  # reflected light in clear lens
        pixels[80:] = (120, 80, 50)
        labels = np.zeros((140, 100), dtype=np.uint8)
        labels[8:32, 24:76] = 2
        labels[20:70, 30:70] = 13
        labels[34:38, 34:40] = 0  # parser hole through the lens
        labels[80:, 5:95] = 5
        result, _ = build_rough_composite(
            Image.fromarray(pixels), self.garment, labels,
            (30, 20, 70, 70), (4, 126, 246),
        )
        self.assertEqual(result.getpixel((36, 35)), tuple(pixels[35, 36]))
        self.assertEqual(result.getpixel((21, 20)), tuple(pixels[20, 21]))

    def test_rough_fit_rejects_disconnected_hair_colored_scenery_near_crown(self):
        pixels = np.full((140, 100, 3), (210, 190, 170), dtype=np.uint8)
        pixels[8:32, 30:70] = (28, 24, 22)
        pixels[18:45, 14:17] = (29, 24, 22)  # nearby dark scenery sliver
        pixels[20:70, 30:70] = (166, 104, 76)
        pixels[80:] = (120, 80, 50)
        labels = np.zeros((140, 100), dtype=np.uint8)
        labels[8:32, 30:70] = 2
        labels[20:70, 30:70] = 13
        labels[80:, 5:95] = 5

        result, _ = build_rough_composite(
            Image.fromarray(pixels), self.garment, labels,
            (30, 20, 70, 70), (4, 126, 246),
        )

        self.assertEqual(result.getpixel((15, 30)), (4, 126, 246))

    def test_rough_fit_does_not_paint_a_connector_to_disconnected_dark_patch(self):
        pixels = np.full((180, 140, 3), (220, 220, 220), dtype=np.uint8)
        pixels[18:72, 42:98] = (30, 25, 22)
        pixels[35:105, 48:92] = (176, 116, 82)
        pixels[101:124, 96:117] = (31, 25, 22)
        pixels[120:] = (120, 80, 50)
        labels = np.zeros((180, 140), dtype=np.uint8)
        labels[18:72, 42:98] = 2
        labels[35:105, 48:92] = 13
        labels[120:, 8:132] = 5
        garment = Image.new("RGBA", (140, 80), (12, 28, 50, 0))
        garment.paste((12, 28, 50, 255), (8, 5, 132, 80))
        result, meta = build_rough_composite(
            Image.fromarray(pixels), garment, labels,
            (48, 35, 92, 105), (4, 126, 246),
        )
        self.assertEqual(meta["detected_ponytail_side"], "unknown")
        self.assertEqual(result.getpixel((106, 112)), (4, 126, 246))

    def test_rough_fit_detects_right_ponytail_occluded_behind_neck(self):
        pixels = np.full((220, 160, 3), (225, 225, 225), dtype=np.uint8)
        pixels[16:82, 48:112] = (27, 23, 21)
        pixels[34:120, 54:106] = (170, 108, 78)
        # The tail reappears on viewer-right after a neck-sized occlusion.
        pixels[116:154, 108:142] = (29, 24, 22)
        pixels[152:] = (115, 78, 52)
        labels = np.zeros((220, 160), dtype=np.uint8)
        labels[16:82, 48:112] = 2
        labels[34:120, 54:106] = 13
        labels[152:, 8:152] = 5
        garment = Image.new("RGBA", (160, 100), (12, 28, 50, 0))
        garment.paste((12, 28, 50, 255), (8, 5, 152, 100))
        result, meta = build_rough_composite(
            Image.fromarray(pixels), garment, labels,
            (54, 34, 106, 120), (4, 126, 246),
        )
        self.assertEqual(meta["detected_ponytail_side"], "viewer-right")
        self.assertEqual(result.getpixel((124, 138)), tuple(pixels[138, 124]))

    def test_rough_fit_detects_two_side_ponytails_without_static_style_assumption(self):
        pixels = np.full((220, 160, 3), (225, 225, 225), dtype=np.uint8)
        pixels[16:82, 48:112] = (27, 23, 21)
        pixels[34:120, 54:106] = (170, 108, 78)
        # Two independently visible tied sections beside and below both ears.
        pixels[112:154, 18:48] = (29, 24, 22)
        pixels[112:154, 112:142] = (29, 24, 22)
        pixels[152:] = (115, 78, 52)
        labels = np.zeros((220, 160), dtype=np.uint8)
        labels[16:82, 48:112] = 2
        labels[34:120, 54:106] = 13
        labels[152:, 8:152] = 5
        garment = Image.new("RGBA", (160, 100), (12, 28, 50, 0))
        garment.paste((12, 28, 50, 255), (8, 5, 152, 100))

        _, meta = build_rough_composite(
            Image.fromarray(pixels), garment, labels,
            (54, 34, 106, 120), (4, 126, 246), expect_tied_hair=True,
        )

        self.assertEqual(meta["detected_ponytail_side"], "bilateral")
        self.assertEqual(meta["detected_tied_section_count"], 2)
        self.assertEqual(
            meta["detected_tied_section_sides"],
            "viewer-left and viewer-right",
        )
        self.assertNotEqual(meta["measured_hair_extent"], "unknown")
        self.assertIsNotNone(meta["measured_hair_end_ratio"])
        self.assertIsNotNone(meta["measured_left_hair_end_ratio"])
        self.assertIsNotNone(meta["measured_right_hair_end_ratio"])

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

    def test_close_up_face_preserves_uploaded_body_width(self):
        result, meta = build_rough_composite(
            self.person,
            self.garment,
            self.labels,
            (20, 20, 80, 70),
            (4, 126, 246),
        )
        self.assertEqual(result.width, self.person.width)
        self.assertAlmostEqual(meta["source_body_to_face_width"], 1.50, places=2)
        self.assertAlmostEqual(
            meta["garment_to_face_width"],
            meta["source_body_to_face_width"],
            places=2,
        )
        # The rear collar meets the lower jaw/neck transition without rising
        # into the cheeks or leaving an empty wedge for Qwen to reproduce. The
        # seat is this portrait's measured neck column: the fixture is flat
        # skin below the chin, so the measurement reaches its 20%-face-height
        # cap instead of the old fixed chin-level placement.
        self.assertEqual(meta["garment_box"], [5, 80, 95, 140])
        self.assertAlmostEqual(meta["neck_seat_ratio"], .20, places=2)
        self.assertEqual(meta["source_canvas_size"], [100, 140])
        self.assertEqual(meta["expanded_canvas_size"], list(result.size))

    def test_body_fit_uses_centered_upper_chest_not_wide_lower_sleeves(self):
        labels = np.zeros((180, 180), dtype=np.uint8)
        labels[20:80, 70:110] = 13
        labels[82:125, 45:135] = 5
        labels[125:180, 5:175] = 5
        garment = Image.new("RGBA", (180, 120), (20, 30, 50, 0))
        garment.paste((20, 30, 50, 255), (5, 5, 175, 120))
        person = Image.new("RGB", (180, 180), (170, 110, 80))

        _, meta = build_rough_composite(
            person, garment, labels, (70, 20, 110, 80), (4, 126, 246),
        )

        left, _, right, _ = meta["garment_box"]
        self.assertLess(right - left, 120)
        self.assertAlmostEqual((left + right) / 2.0, 90.0, delta=2.0)
        self.assertLess(meta["source_body_to_face_width"], 3.0)

    def test_person_guide_keeps_subject_and_removes_scenery(self):
        result = build_person_backdrop_guide(self.person, self.labels, (4, 126, 246))
        self.assertEqual(result.getpixel((0, 0)), (4, 126, 246))
        self.assertEqual(result.getpixel((50, 40)), self.person.getpixel((50, 40)))
        self.assertEqual(result.getpixel((50, 110)), self.person.getpixel((50, 110)))

    def test_primary_prompt_preserves_the_preconditioned_flat_backdrop(self):
        prompt = refinement_prompt((4, 126, 246), primary_is_identity=True)
        self.assertIn("HIGHEST PRIORITY BACKGROUND EDIT", prompt)
        self.assertIn("Do not copy the original photo background", prompt)
        self.assertIn("FINAL CHECK: the background must contain only", prompt)
        self.assertIn("#047EF6, RGB(4, 126, 246)", prompt)

    def test_single_composite_prompt_forbids_secondary_people(self):
        prompt = refinement_prompt((4, 126, 246), single_composite=True)
        self.assertIn("COMPLETE CANVAS", prompt)
        self.assertIn("CURRENT CENTERED SINGLE-SUBJECT COMPOSITION", prompt)
        self.assertIn("head must sit anatomically on the neck", prompt)
        self.assertIn("shoulders broad enough to support the head", prompt)
        self.assertNotIn("image 2", prompt.lower())
        self.assertNotIn("image 3", prompt.lower())

    def test_uniform_edit_canvas_keeps_identity_and_blanks_source_clothing(self):
        result = build_person_backdrop_guide(
            self.person, self.labels, (4, 126, 246), remove_clothing=True,
        )
        self.assertEqual(result.getpixel((50, 40)), self.person.getpixel((50, 40)))
        self.assertEqual(result.getpixel((50, 110)), (4, 126, 246))

    def test_identity_reference_removes_source_arms_as_well_as_clothing(self):
        labels = self.labels.copy()
        labels[85:130, 2:12] = 14
        labels[85:130, 88:98] = 15
        result = build_person_backdrop_guide(
            self.person, labels, (4, 126, 246), remove_clothing=True,
        )
        self.assertEqual(result.getpixel((6, 105)), (4, 126, 246))
        self.assertEqual(result.getpixel((94, 105)), (4, 126, 246))

    def test_neutral_body_fit_guide_keeps_person_shape_without_garment_color(self):
        person = self.person.copy()
        person.paste((220, 20, 40), (5, 80, 95, 140))
        result, meta = build_neutral_body_fit_guide(
            person, self.labels, (30, 20, 70, 70), (255, 255, 255),
        )
        self.assertEqual(result.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(result.getpixel((50, 40)), person.getpixel((50, 40)))
        garment_pixel = result.getpixel((50, 110))
        self.assertLess(max(garment_pixel) - min(garment_pixel), 3)
        self.assertNotEqual(garment_pixel, (220, 20, 40))
        self.assertEqual(meta["mode"], "neutral_source_body_fit_for_qwen")
        self.assertEqual(meta["garment_box"], [5, 80, 96, 140])

    def test_person_guide_recovers_skin_neck_missed_by_parser(self):
        person = np.full((140, 100, 3), (4, 126, 246), np.uint8)
        person[20:70, 30:70] = (170, 108, 78)
        person[70:95, 42:58] = (150, 88, 62)
        labels = np.zeros((140, 100), dtype=np.uint8)
        labels[20:70, 30:70] = 13
        result = build_person_backdrop_guide(
            Image.fromarray(person), labels, (4, 126, 246), face_box=(30, 20, 70, 70),
        )
        self.assertNotEqual(result.getpixel((50, 82)), (4, 126, 246))

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
        # Source clothing starts at row 80 in this fixture and must never be
        # visible in the Qwen identity reference.
        self.assertLessEqual(result.height, 80)

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
        # Feathering must stay inside the silhouette; otherwise the white
        # catalog background becomes a visible shoulder/sleeve halo.
        self.assertEqual(cutout.getpixel((9, 40))[3], 0)

    def test_rough_cutout_removes_catalog_color_from_feathered_edge(self):
        pixels = np.full((40, 40, 4), (245, 245, 245, 0), dtype=np.uint8)
        pixels[8:32, 8:32] = (245, 245, 245, 96)
        pixels[10:30, 10:30] = (32, 48, 84, 255)
        cutout = opaque_garment_cutout(Image.fromarray(pixels, "RGBA"))
        edge = cutout.getpixel((9, 20))
        self.assertGreater(edge[3], 0)
        self.assertLess(edge[0], 80)
        self.assertLess(edge[1], 100)
        self.assertGreater(edge[2], 60)

    def test_rough_cutout_has_smooth_inner_alpha_without_outer_halo(self):
        pixels = np.full((48, 48, 4), (250, 250, 250, 0), dtype=np.uint8)
        pixels[8:40, 8:40] = (32, 48, 84, 255)
        cutout = opaque_garment_cutout(Image.fromarray(pixels, "RGBA"))
        alpha = np.asarray(cutout.getchannel("A"))
        self.assertEqual(int(alpha[7, 24]), 0)
        self.assertGreater(int(alpha[8, 24]), 0)
        self.assertLess(int(alpha[8, 24]), 255)
        self.assertGreater(int(alpha[9, 24]), int(alpha[8, 24]))
        self.assertEqual(int(alpha[14, 24]), 255)

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
        self.assertIn("image 1's black/dark-brown hair", description)
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
        for phrase in ("neutral even frontal light", "natural detail"):
            self.assertIn(phrase, description)
        for phrase in ("upper left", "reconstruct", "balance the forehead", "reinterpret"):
            self.assertNotIn(phrase, description)

    def test_dark_hair_measurement_uses_shaded_strands(self):
        pixels = np.full((20, 20, 3), (205, 195, 180), dtype=np.uint8)
        pixels[:10] = (32, 28, 26)
        labels = np.full((20, 20), 2, dtype=np.uint8)
        self.assertTrue(source_hair_is_dark(Image.fromarray(pixels), labels))


    def edge_clipped_labels(self):
        """Shoulders that taper in frame, then run off both edges lower down."""
        labels = np.zeros((140, 100), dtype=np.uint8)
        labels[20:70, 30:70] = 13
        # A few in-frame shoulder rows: the real body is about 60px wide.
        labels[72:80, 20:80] = 5
        # Most of the measured band reaches both image edges, so an unguarded
        # 85th percentile is pinned at the full canvas width instead.
        labels[80:, 0:100] = 5
        return labels

    def test_body_width_ignores_rows_that_run_off_the_canvas(self):
        _, meta = build_rough_composite(
            self.person, self.garment, self.edge_clipped_labels(),
            (30, 20, 70, 70), (4, 126, 246),
        )
        # Counting the clipped rows reports the full 100px canvas and scales
        # the garment edge to edge, erasing the shoulder slope entirely.
        self.assertLess(meta["source_body_width_px"], 100.0)
        self.assertLessEqual(meta["source_body_width_px"], 62.0)
        box = meta["garment_box"]
        self.assertGreater(box[0], 0)
        self.assertLess(box[2], self.person.width)

    def test_shoulders_cropped_on_one_side_keep_their_real_width(self):
        """A clipped row still proves the body is at least that wide.

        Discarding every edge-touching row collapsed the measurement onto the
        narrow neck/upper-chest rows: measured 838px against a true 943px body,
        so the fitted shirt sat ~180px inside the real shoulder line.
        """
        labels = np.zeros((140, 100), dtype=np.uint8)
        labels[20:70, 30:70] = 13
        # Upper chest well inside the frame. Deliberately wider than 1.35 face
        # widths so the separate too-narrow safety net stays quiet and only the
        # clipped-row floor can supply the real shoulder width.
        labels[72:88, 21:79] = 5
        # Shoulders that overhang the left edge only.
        labels[88:, 0:92] = 5
        _, meta = build_rough_composite(
            self.person, self.garment, labels, (30, 20, 70, 70), (4, 126, 246),
        )
        self.assertGreaterEqual(
            meta["source_body_width_px"], 90.0,
            "clipped shoulder rows were discarded, shirt fitted too narrow",
        )
        self.assertLess(meta["source_body_width_px"], 100.0)

    def test_fully_clipped_body_still_places_a_garment(self):
        # Every clothing row touches both edges: a tightly cropped upload.
        labels = np.zeros((140, 100), dtype=np.uint8)
        labels[20:70, 30:70] = 13
        labels[80:, :] = 5
        _, meta = build_rough_composite(
            self.person, self.garment, labels, (30, 20, 70, 70), (4, 126, 246),
        )
        self.assertGreater(meta["source_body_width_px"], 0.0)
        self.assertTrue(meta["not_final"])

    def test_closely_cropped_upload_extends_torso_to_canvas_bottom(self):
        # Source upload where clothing ends well above the canvas bottom (e.g. row 110 in 140px canvas).
        # In a passport frame, the uniform torso must extend down to the bottom of the canvas.
        labels = np.zeros((140, 100), dtype=np.uint8)
        labels[20:70, 30:70] = 13
        labels[80:110, 20:80] = 5  # clothing stops at row 109
        _, meta = build_rough_composite(
            self.person, self.garment, labels, (30, 20, 70, 70), (4, 126, 246),
        )
        box = meta["garment_box"]
        self.assertEqual(box[3], 140)
        self.assertGreater(meta["source_body_to_face_width"], 1.35)

    def test_neck_survives_when_the_parser_has_no_neck_class(self):
        """LIP-20 has no neck label, so neck skin arrives as label 0.

        Label 13 (Face) stops at the jaw and everything below it comes back as
        0 (Background), which build_rough_composite paints with the backdrop.
        Only the colour scan recovers it. When that scan stopped short of the
        collar the guide showed Qwen a floating head over a white wedge, and
        Qwen rebuilt the neck and neckline to match.
        """
        person = Image.new("RGB", (100, 140), (30, 30, 30))
        # Face, then a real skin-coloured neck the parser calls background.
        person.paste((176, 112, 79), (35, 20, 65, 95))
        garment = Image.new("RGBA", (100, 80), (10, 20, 40, 0))
        garment.paste((10, 20, 40, 255), (10, 10, 90, 80))
        labels = np.zeros((140, 100), dtype=np.uint8)
        labels[20:70, 35:65] = 13
        labels[95:, 5:95] = 5
        result, _ = build_rough_composite(
            person, garment, labels, (35, 20, 65, 70), (255, 255, 255),
        )
        pixels = np.asarray(result.convert("RGB")).astype(int)
        # Walk the centre column from the jaw down to the first garment pixel.
        # Every row in between must be skin, never the flat backdrop: that wedge
        # is what makes the head read as floating above the collar.
        garment = (10, 20, 40)
        jaw_to_collar = []
        for y in range(70, pixels.shape[0]):
            centre = pixels[y, 45:55].mean(0)
            if max(abs(centre - np.asarray(garment))) <= 12:
                break
            jaw_to_collar.append((y, centre))
        self.assertTrue(jaw_to_collar, "collar started at the jaw, no neck at all")
        backdrop_rows = [
            y for y, centre in jaw_to_collar if abs(centre.mean() - 255) <= 6
        ]
        self.assertEqual(
            backdrop_rows, [],
            f"backdrop wedge left between jaw and collar at rows {backdrop_rows}",
        )

if __name__ == "__main__":
    unittest.main()
