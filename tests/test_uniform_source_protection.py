import unittest

import numpy as np
from PIL import Image

from pipelines.uniform_source_protection import compose_parts


class SourceProtectionTests(unittest.TestCase):
    def setUp(self):
        self.source = Image.new("RGBA", (100, 100), (160, 100, 70, 0))
        alpha = np.zeros((100, 100), dtype=np.uint8)
        alpha[10:70, 20:80] = 255
        self.source.putalpha(Image.fromarray(alpha))
        self.candidate = Image.new("RGBA", (100, 100), (20, 40, 90, 255))
        self.head = np.zeros((100, 100), dtype=np.uint8)
        self.head[10:60, 20:80] = 13
        self.cloth = np.zeros_like(self.head)
        self.cloth[65:, 10:90] = 5
        self.points = np.array([[35, 30], [65, 30], [50, 40], [40, 50], [60, 50]], dtype=float)

    def compose(self, **changes):
        args = dict(source=self.source, candidate=self.candidate,
                    source_labels=self.head, candidate_labels=self.cloth,
                    source_points=self.points, candidate_points=self.points,
                    source_box=np.array([20, 10, 80, 60]), background=(4, 126, 246))
        args.update(changes)
        return compose_parts(**args)

    def test_original_face_and_selected_background(self):
        result, info = self.compose()
        self.assertEqual(result.getpixel((50, 35)), (160, 100, 70))
        self.assertEqual(result.getpixel((0, 0)), (4, 126, 246))
        self.assertEqual(result.getpixel((50, 90)), (20, 40, 90))
        self.assertTrue(info["source_lighting_preserved"])

    def test_missing_clothes_rejected(self):
        with self.assertRaisesRegex(ValueError, "uniform mask"):
            self.compose(candidate_labels=np.zeros_like(self.cloth))

    def test_mismatched_masks_rejected(self):
        with self.assertRaisesRegex(ValueError, "masks must match"):
            self.compose(source_labels=np.zeros((10, 10)))

    def test_invalid_landmarks_rejected(self):
        with self.assertRaisesRegex(ValueError, "landmarks"):
            self.compose(candidate_points=np.full((5, 2), np.nan))

    def test_badly_scaled_candidate_rejected(self):
        with self.assertRaisesRegex(ValueError, "pose differs"):
            self.compose(candidate_points=self.points * 4)


if __name__ == "__main__":
    unittest.main()
