import unittest
import cv2
import numpy as np
from PIL import Image
from pipelines.uniform_badge_cleanup import remove_template_badges


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

    def test_plain_uniform_is_unchanged(self):
        rgb = np.full((400, 300, 3), (20, 30, 45), np.uint8)
        result, count = remove_template_badges(Image.fromarray(rgb), Image.fromarray(rgb))
        self.assertEqual(count, 0)
        np.testing.assert_array_equal(np.array(result), rgb)
