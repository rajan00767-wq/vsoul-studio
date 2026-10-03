import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from pipelines.qwen_edit_pipeline import QwenEditPipeline


class UniformCropTests(unittest.TestCase):
    def test_crown_touching_source_edge_gets_passport_headroom(self):
        image = Image.new("RGB", (496, 640), (128, 128, 128))
        draw = ImageDraw.Draw(image)
        draw.ellipse((145, -25, 350, 330), fill=(25, 20, 18))
        draw.rectangle((90, 300, 410, 640), fill=(70, 75, 82))

        class Detector:
            def detectMultiScale(self, *args, **kwargs):
                return np.asarray([[172, 36, 150, 150]])

        with patch("pipelines.qwen_edit_pipeline.cv2.CascadeClassifier", return_value=Detector()):
            result = QwenEditPipeline._crop_school_passport_portrait(image, 560, 720)

        self.assertEqual(result.size, (560, 720))
        rgb = np.asarray(result)
        foreground = np.linalg.norm(rgb.astype(np.float32) - 128.0, axis=2) > 20.0
        occupied = np.flatnonzero(np.count_nonzero(foreground, axis=1) > 10)
        self.assertGreaterEqual(int(occupied[0]), 24)
        self.assertLessEqual(int(occupied[0]), 60)

    def test_full_portrait_is_cropped_to_upper_chest(self):
        image = Image.new("RGB", (496, 640), (128, 128, 128))
        draw = ImageDraw.Draw(image)
        draw.ellipse((150, 25, 345, 270), fill=(45, 32, 28))
        draw.rectangle((95, 255, 400, 440), fill=(75, 80, 88))
        draw.rectangle((95, 440, 400, 640), fill=(210, 30, 30))

        class Detector:
            def detectMultiScale(self, *args, **kwargs):
                return np.asarray([[174, 82, 148, 148]])

        with patch("pipelines.qwen_edit_pipeline.cv2.CascadeClassifier", return_value=Detector()):
            result = QwenEditPipeline._crop_school_passport_portrait(image, 560, 720)

        rgb = np.asarray(result)
        lower_red = (rgb[int(result.height * .86):, 0] > 170) & (rgb[int(result.height * .86):, 1] < 80)
        self.assertLess(float(np.mean(lower_red)), 0.02)


if __name__ == "__main__":
    unittest.main()
