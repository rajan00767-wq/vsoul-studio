import unittest
import cv2
import numpy as np
from PIL import Image
from pipelines.uniform_badge_cleanup import (
    remove_generated_backdrop_badge,
    remove_template_badges,
)


class UniformBadgeTests(unittest.TestCase):
    def test_badge_removed_without_touching_person_or_backdrop(self):
        rgb = np.full((400, 300, 3), (4, 126, 246), np.uint8)
        rgb[180:, 40:260] = (20, 30, 45)
        cv2.circle(rgb, (210, 285), 18, (0, 175, 240), -1)
        result, count = remove_template_badges(Image.fromarray(rgb), Image.fromarray(rgb))
        self.assertEqual(count, 1)
        out = np.array(result)
        np.testing.assert_array_equal(out[:180], rgb[:180])
        np.testing.assert_array_equal(out[:, :35], rgb[:, :35])
        self.assertLess(out[285, 210, 2], 100)

    def test_round_badge_with_light_logo_is_removed(self):
        rgb = np.full((720, 560, 3), (235, 235, 235), np.uint8)
        rgb[300:, ::24] = (35, 45, 90)
        cv2.circle(rgb, (380, 625), 32, (0, 155, 225), -1)
        cv2.circle(rgb, (380, 625), 18, (220, 225, 225), 5)
        cv2.line(rgb, (368, 625), (392, 625), (220, 225, 225), 6)
        result, count = remove_template_badges(
            Image.fromarray(rgb), Image.fromarray(rgb), repair_mode="texture",
        )
        self.assertEqual(count, 1)
        out = np.asarray(result)
        self.assertGreater(
            np.mean(np.abs(out[585:665, 340:420].astype(float) - rgb[585:665, 340:420].astype(float))),
            1.0,
        )

    def test_plain_uniform_is_unchanged(self):
        rgb = np.full((400, 300, 3), (20, 30, 45), np.uint8)
        result, count = remove_template_badges(Image.fromarray(rgb), Image.fromarray(rgb))
        self.assertEqual(count, 0)
        np.testing.assert_array_equal(np.array(result), rgb)

    def test_brown_badge_on_bright_plaid_is_removed(self):
        rgb = np.full((400, 300, 3), (235, 235, 235), np.uint8)
        rgb[180:, ::24] = (30, 45, 90)
        rgb[180:, 180:224] = (112, 55, 38)
        rgb[265:275, 195:209] = (225, 190, 90)
        result, count = remove_template_badges(Image.fromarray(rgb), Image.fromarray(rgb))
        self.assertGreaterEqual(count, 1)
        out = np.array(result)
        self.assertGreater(
            np.mean(np.abs(out[220:320, 170:235].astype(float) - rgb[220:320, 170:235].astype(float))),
            1.0,
        )

    def test_rough_texture_repair_does_not_damage_sleeve_cuff(self):
        rgb = np.full((420, 320, 3), (235, 235, 235), np.uint8)
        rgb[150:, ::24] = (30, 45, 90)
        rgb[330:365, 5:65] = (220, 20, 30)  # saturated sleeve cuff near edge
        rgb[230:285, 205:252] = (92, 42, 32)
        rgb[246:260, 216:242] = (225, 190, 90)
        cuff = rgb[325:370, :70].copy()
        result, count = remove_template_badges(
            Image.fromarray(rgb), Image.fromarray(rgb), repair_mode="texture",
        )
        self.assertEqual(count, 1)
        out = np.asarray(result)
        np.testing.assert_array_equal(out[325:370, :70], cuff)
        self.assertGreater(
            np.mean(np.abs(out[220:295, 195:262].astype(float) - rgb[220:295, 195:262].astype(float))),
            1.0,
        )

    def test_dark_badge_is_removed_from_middle_of_full_shirt_template(self):
        rgb = np.full((500, 400, 3), (235, 235, 235), np.uint8)
        rgb[180:, ::24] = (30, 45, 90)
        rgb[205:275, 265:325] = (92, 42, 32)
        rgb[230:245, 280:310] = (225, 190, 90)
        result, count = remove_template_badges(Image.fromarray(rgb), Image.fromarray(rgb))
        self.assertGreaterEqual(count, 1)
        out = np.asarray(result)
        self.assertGreater(
            np.mean(np.abs(out[195:285, 250:340].astype(float) - rgb[195:285, 250:340].astype(float))),
            1.0,
        )

    def test_lower_right_badge_uses_surrounding_uniform_texture(self):
        rgb = np.full((600, 440, 3), (238, 243, 250), np.uint8)
        rgb[180:, 36:420] = (22, 32, 48)
        # Real product templates often crop a crest close to the lower-right
        # boundary of the vest rather than the middle of the image.
        cv2.circle(rgb, (370, 565), 24, (0, 155, 225), -1)
        cv2.circle(rgb, (370, 565), 11, (225, 235, 240), 4)
        result, count = remove_template_badges(
            Image.fromarray(rgb), Image.fromarray(rgb), repair_mode="texture",
        )
        self.assertEqual(count, 1)
        out = np.asarray(result)
        repaired = out[545:585, 350:390].mean(axis=(0, 1))
        surrounding = rgb[545:585, 300:340].mean(axis=(0, 1))
        self.assertLess(float(np.linalg.norm(repaired - surrounding)), 25.0)

    def test_vl_badge_bbox_never_samples_white_product_background(self):
        rgb = np.full((500, 400, 3), (248, 248, 248), np.uint8)
        # Person-free product photo: plaid shirt on a white catalogue canvas.
        rgb[130:490, 45:355] = (40, 50, 88)
        rgb[130:490:22, 45:355] = (230, 232, 238)
        rgb[130:490, 45:355:26] = (225, 228, 235)
        rgb[255:325, 258:322] = (86, 38, 31)
        rgb[275:290, 272:307] = (226, 190, 92)
        result, count = remove_template_badges(
            Image.fromarray(rgb),
            Image.fromarray(rgb),
            repair_mode="texture",
            reference_bbox=(258, 255, 322, 325),
            reference_bbox_size=(400, 500),
        )
        self.assertEqual(count, 1)
        repaired = np.asarray(result)[260:320, 263:317]
        white_fraction = np.mean(np.linalg.norm(
            repaired.astype(np.float32) - np.array((248, 248, 248), np.float32),
            axis=2,
        ) < 18.0)
        self.assertLess(white_fraction, .08)

    def test_generated_backdrop_colour_block_is_removed_inside_uniform(self):
        background = (4, 126, 246)
        rgb = np.full((500, 360, 3), background, np.uint8)
        clothing = np.zeros((500, 360), dtype=bool)
        clothing[210:, 40:320] = True
        rgb[clothing] = (24, 34, 50)
        rgb[430:490, 245:305] = background
        template = np.zeros((260, 320, 4), np.uint8)
        template[20:250, 20:300, :3] = (24, 34, 50)
        template[20:250, 20:300, 3] = 255
        result, count = remove_generated_backdrop_badge(
            Image.fromarray(rgb), Image.fromarray(template, mode="RGBA"),
            background, clothing,
        )
        self.assertEqual(count, 1)
        repaired = np.asarray(result)[445:475, 260:290].mean(axis=(0, 1))
        self.assertLess(float(np.linalg.norm(repaired - np.array((24, 34, 50)))), 30.0)

    def test_face_colours_are_never_treated_as_chest_badges(self):
        template = np.full((400, 300, 3), (235, 235, 235), np.uint8)
        template[180:, ::24] = (30, 45, 90)
        template[270:315, 195:235] = (112, 55, 38)
        image = template.copy()
        image[200:225, 130:170] = (112, 55, 38)
        face_patch = image[195:230, 125:175].copy()
        result, count = remove_template_badges(Image.fromarray(image), Image.fromarray(template))
        self.assertGreaterEqual(count, 1)
        np.testing.assert_array_equal(np.asarray(result)[195:230, 125:175], face_patch)

    def test_saturated_collar_patch_is_not_treated_as_badge(self):
        template = np.full((400, 300, 3), (235, 235, 235), np.uint8)
        template[180:, ::24] = (30, 45, 90)
        template[310:350, 195:235] = (112, 55, 38)
        image = template.copy()
        image[210:250, 205:235] = (220, 20, 30)
        collar_patch = image[205:255, 200:240].copy()
        result, count = remove_template_badges(Image.fromarray(image), Image.fromarray(template))
        self.assertGreaterEqual(count, 1)
        np.testing.assert_array_equal(np.asarray(result)[205:255, 200:240], collar_patch)
