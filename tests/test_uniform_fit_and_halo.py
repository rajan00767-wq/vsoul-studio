import unittest
import numpy as np
from PIL import Image
import cv2

from pipelines.uniform_composite import build_rough_composite
from pipelines.qwen_edit_pipeline import QwenEditPipeline


class TestUniformFitAndHalo(unittest.TestCase):
    def setUp(self):
        self.pipe = QwenEditPipeline()

    def test_shoulder_span_bounded_to_natural_proportions(self):
        """Test that shoulder span measurement never produces oversized boxy torsos."""
        # Create an upload where lower rows flare out wide (underarms/sleeves)
        person = Image.new("RGB", (200, 260), (180, 130, 90))
        garment = Image.new("RGBA", (200, 160), (0, 0, 0, 0))
        garment.paste((20, 30, 60, 255), (15, 10, 185, 150))
        labels = np.zeros((260, 200), dtype=np.uint8)
        # Face: y 30..110, x 70..130 -> face_h = 80, face_w = 60, y2 = 110
        labels[30:110, 70:130] = 13
        # True shoulders at y 118..142: width 114 (1.90x face_w)
        labels[118:142, 43:157] = 5
        # Flared lower torso / sleeves at y 155..260: width 190 (3.16x face_w)
        labels[155:260, 5:195] = 5

        _, meta = build_rough_composite(
            person, garment, labels, (70, 30, 130, 110), (255, 255, 255),
        )

        # Body-to-face ratio must reflect the shoulders (1.80..2.15), not the 3.16x lower flare
        ratio = meta["source_body_to_face_width"]
        self.assertLessEqual(ratio, 2.15)
        self.assertGreaterEqual(ratio, 1.70)
        # Garment box width must not touch edges
        box_w = meta["garment_box"][2] - meta["garment_box"][0]
        self.assertLess(box_w, 150)

    def test_strict_background_removes_crown_and_shoulder_halo(self):
        """Test that boundary halos around hair crown and shoulders are eliminated."""
        h, w = 140, 120
        img_arr = np.full((h, w, 3), 255, dtype=np.uint8) # pure white bg
        # Dark hair
        img_arr[15:45, 45:75] = [25, 25, 30]
        # Crown halo: off-white/greenish fringe at boundary y=12..14
        img_arr[12:15, 45:75] = [225, 235, 225]
        # Face
        img_arr[35:65, 48:72] = [210, 155, 125]
        # White shirt with slight gray tone
        img_arr[65:130, 25:95] = [238, 240, 242]
        # Shoulder halo: near-white / gray fringe at x=20..24 and x=96..100
        img_arr[65:130, 20:25] = [218, 222, 225]
        img_arr[65:130, 95:100] = [218, 222, 225]

        img = Image.fromarray(img_arr, "RGB")
        mock_labels = np.zeros((h, w), dtype=np.uint8)
        mock_labels[15:45, 45:75] = 2 # Hair
        mock_labels[35:65, 48:72] = 13 # Face
        mock_labels[65:130, 25:95] = 5 # Shirt

        import unittest.mock
        with unittest.mock.patch("pipelines.schp_service.parse", return_value={"labels": mock_labels}):
            cleaned = self.pipe._replace_flat_border_background_strict(img, (255, 255, 255), force=True)

        res_arr = np.array(cleaned)
        # 1. Crown halo at y=13 must be absorbed into pure white background
        self.assertEqual(res_arr[13, 60].tolist(), [255, 255, 255])
        # 2. Shoulder halo at x=22 must be absorbed into pure white background
        self.assertEqual(res_arr[80, 22].tolist(), [255, 255, 255])
        self.assertEqual(res_arr[80, 98].tolist(), [255, 255, 255])
        # 3. Inside the shirt at x=60 must retain authentic fabric tone (not overwritten)
        self.assertEqual(res_arr[80, 60].tolist(), [238, 240, 242])

    def test_has_selected_solid_background_rejects_greenish_backdrop(self):
        """Test that a shaded/greenish backdrop touching the person is NOT absorbed as an accessory."""
        h, w = 160, 120
        # Background is greenish sage [184, 207, 161]
        img_arr = np.full((h, w, 3), [184, 207, 161], dtype=np.uint8)
        # Person in center
        img_arr[30:70, 45:75] = [120, 80, 60] # Face
        img_arr[70:160, 25:95] = [30, 40, 80] # Clothing

        img = Image.fromarray(img_arr, "RGB")
        mock_labels = np.zeros((h, w), dtype=np.uint8)
        mock_labels[30:70, 45:75] = 13 # Face
        mock_labels[70:160, 25:95] = 5  # Clothing

        import unittest.mock
        with unittest.mock.patch("pipelines.schp_service.parse", return_value={"labels": mock_labels}):
            passed = self.pipe._has_selected_solid_background(img, (255, 255, 255))

        # Must correctly fail verification so fallback replacement runs!
        self.assertFalse(passed)

    def test_template_fit_scale_never_exceeds_natural_proportions(self):
        """Test that uniform template fitting never blows up by 1.6x, keeping natural collar and chest."""
        # Person upload with shoulders width 114
        person = Image.new("RGB", (200, 260), (180, 130, 90))
        # Template shirt width 300, height 400
        garment = Image.new("RGBA", (300, 400), (0, 0, 0, 0))
        garment.paste((20, 30, 60, 255), (20, 20, 280, 380))
        labels = np.zeros((260, 200), dtype=np.uint8)
        labels[30:110, 70:130] = 13 # Face (face_w=60, face_h=80)
        labels[118:142, 43:157] = 5  # Shoulders (width 114)
        labels[142:260, 35:165] = 5  # Torso

        composite, meta = build_rough_composite(
            person, garment, labels, (70, 30, 130, 110), (255, 255, 255),
        )

        # Garment to face width ratio must be in tailored proportion (<= 2.25)
        self.assertLessEqual(meta["garment_to_face_width"], 2.25)
        # Garment box width must not exceed natural body proportion
        gbox = meta["garment_box"]
        gbox_w = gbox[2] - gbox[0]
        self.assertLessEqual(gbox_w, 140)


if __name__ == "__main__":
    unittest.main()
