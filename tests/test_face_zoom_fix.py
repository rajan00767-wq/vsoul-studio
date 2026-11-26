import unittest
import numpy as np
from PIL import Image, ImageDraw
import cv2

from pipelines.qwen_edit_pipeline import QwenEditPipeline


class TestFaceZoomFix(unittest.TestCase):
    def test_delivery_crop_face_fraction_is_natural_and_not_zoomed(self):
        # Synthetic portrait with height 1136, width 880, face_h 390
        # Subject occupies crown to bottom row
        image = Image.new("RGB", (880, 1136), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        # Face box: x1=290, y1=220, x2=590, y2=610 (face_h=390, face_w=300)
        draw.ellipse((290, 220, 590, 610), fill=(48, 34, 28))
        # Torso and shoulders extending to canvas bottom
        draw.rectangle((50, 580, 830, 1136), fill=(72, 78, 86))

        result = QwenEditPipeline._crop_school_passport_portrait(
            image,
            560,
            720,
            tight_school_id=True,
            delivery_upper_chest=True,
            face_bbox=(290, 220, 590, 610),
            background_color=(255, 255, 255),
        )

        self.assertEqual(result.size, (560, 720))
        rgb = np.asarray(result)
        face = (
            (rgb[:, :, 0] >= 35) & (rgb[:, :, 0] <= 65)
            & (rgb[:, :, 1] >= 20) & (rgb[:, :, 1] <= 48)
            & (rgb[:, :, 2] >= 16) & (rgb[:, :, 2] <= 42)
        )
        rows = np.flatnonzero(np.count_nonzero(face, axis=1) > 5)
        face_h_fraction = float(rows[-1] - rows[0] + 1) / result.height
        # Upper chest passport framing: natural scale (~38% to 42%), not a loose half-body or zoomed face
        self.assertGreaterEqual(face_h_fraction, 0.36)
        self.assertLessEqual(face_h_fraction, 0.44)

        # Upper chest / clothing must occupy at least 25% of canvas height below face
        torso_rows = result.height - rows[-1]
        self.assertGreater(torso_rows, int(result.height * 0.25))

        # Modest headroom must be present above face
        self.assertGreaterEqual(rows[0], 25)

    def test_delivery_crop_tight_raw_canvas_does_not_overclamp_shoulders(self):
        # A decode where uniform touches bottom row:
        # Image size 928x1216, face_h 514
        image = Image.new("RGB", (928, 1216), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.ellipse((277, 300, 647, 815), fill=(48, 34, 28))
        draw.rectangle((60, 750, 868, 1216), fill=(72, 78, 86))

        result = QwenEditPipeline._crop_school_passport_portrait(
            image,
            560,
            720,
            tight_school_id=True,
            delivery_upper_chest=True,
            face_bbox=(277, 300, 647, 815),
            background_color=(255, 255, 255),
        )

        self.assertEqual(result.size, (560, 720))
        rgb = np.asarray(result)
        # Verify clothing reaches the bottom row
        clothing_bottom = (
            (rgb[-1, :, 0] >= 60) & (rgb[-1, :, 0] <= 85)
            & (rgb[-1, :, 1] >= 65) & (rgb[-1, :, 1] <= 90)
            & (rgb[-1, :, 2] >= 75) & (rgb[-1, :, 2] <= 100)
        )
        self.assertGreater(np.count_nonzero(clothing_bottom), 100)

        # Verify face is not zoomed past 43.5%
        face = (
            (rgb[:, :, 0] >= 35) & (rgb[:, :, 0] <= 65)
            & (rgb[:, :, 1] >= 20) & (rgb[:, :, 1] <= 48)
            & (rgb[:, :, 2] >= 16) & (rgb[:, :, 2] <= 42)
        )
        rows = np.flatnonzero(np.count_nonzero(face, axis=1) > 5)
        face_h_fraction = float(rows[-1] - rows[0] + 1) / result.height
        self.assertLess(face_h_fraction, 0.43)


if __name__ == "__main__":
    unittest.main()
